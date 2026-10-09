// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "windows_heap_summary.h"

#ifdef _WIN32

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <vector>

#include <windows.h>
#include <intrin.h>

#include "core/ipc/allocation_callsite_summary.h"
#include "core/ipc/heap_size_summary.h"

namespace Core::Ipc {
namespace {

using MallocFunction = void*(__cdecl*)(std::size_t);
using FreeFunction = void(__cdecl*)(void*);
MallocFunction original_malloc{};
FreeFunction original_free{};
AllocationCallsiteSummary malloc_summary;
SampledAllocationTracker<262144> live_malloc_samples;
std::atomic_flag malloc_summary_lock = ATOMIC_FLAG_INIT;
std::atomic_bool malloc_trace_enabled{};
std::atomic_size_t malloc_sample_overflow{};

void* __cdecl TracedMalloc(const std::size_t size) {
    void* const result = original_malloc(size);
    static thread_local std::uint32_t sample_counter{};
    if (malloc_trace_enabled.load(std::memory_order_relaxed) && size <= 160 &&
        (++sample_counter & 63) == 0 &&
        !malloc_summary_lock.test_and_set(std::memory_order_acquire)) {
        void* frames[3]{};
        const USHORT frame_count = RtlCaptureStackBackTrace(1, 3, frames, nullptr);
        const auto caller = reinterpret_cast<std::uintptr_t>(frames[frame_count > 1 ? 1 : 0]);
        malloc_summary.Insert(caller, size);
        if (!live_malloc_samples.Add(result, caller, size)) {
            malloc_sample_overflow.fetch_add(1, std::memory_order_relaxed);
        }
        malloc_summary_lock.clear(std::memory_order_release);
    }
    return result;
}

void __cdecl TracedFree(void* pointer) {
    if (malloc_trace_enabled.load(std::memory_order_relaxed) &&
        !malloc_summary_lock.test_and_set(std::memory_order_acquire)) {
        live_malloc_samples.Remove(pointer);
        malloc_summary_lock.clear(std::memory_order_release);
    }
    original_free(pointer);
}

bool InstallMallocTrace() {
    auto* const module = GetModuleHandleW(nullptr);
    auto* const dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(module);
    auto* const nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(reinterpret_cast<const std::byte*>(module) +
                                                               dos->e_lfanew);
    const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    auto* descriptor = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(reinterpret_cast<std::byte*>(module) +
                                                                  directory.VirtualAddress);
    for (; descriptor->Name != 0; ++descriptor) {
        const auto* dll_name = reinterpret_cast<const char*>(reinterpret_cast<const std::byte*>(module) +
                                                             descriptor->Name);
        if (std::strstr(dll_name, "api-ms-win-crt-heap") == nullptr) {
            continue;
        }
        auto* original_thunk = reinterpret_cast<IMAGE_THUNK_DATA*>(
            reinterpret_cast<std::byte*>(module) + descriptor->OriginalFirstThunk);
        auto* thunk = reinterpret_cast<IMAGE_THUNK_DATA*>(reinterpret_cast<std::byte*>(module) +
                                                          descriptor->FirstThunk);
        for (; original_thunk->u1.AddressOfData != 0; ++original_thunk, ++thunk) {
            if (IMAGE_SNAP_BY_ORDINAL(original_thunk->u1.Ordinal)) {
                continue;
            }
            const auto* import = reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(
                reinterpret_cast<const std::byte*>(module) + original_thunk->u1.AddressOfData);
            const char* const import_name = reinterpret_cast<const char*>(import->Name);
            const bool is_malloc = std::strcmp(import_name, "malloc") == 0;
            const bool is_free = std::strcmp(import_name, "free") == 0;
            if (!is_malloc && !is_free) {
                continue;
            }
            DWORD previous_protection{};
            if (!VirtualProtect(&thunk->u1.Function, sizeof(thunk->u1.Function), PAGE_READWRITE,
                                &previous_protection)) {
                return false;
            }
            if (is_malloc) {
                original_malloc = reinterpret_cast<MallocFunction>(thunk->u1.Function);
                thunk->u1.Function = reinterpret_cast<ULONGLONG>(&TracedMalloc);
            } else {
                original_free = reinterpret_cast<FreeFunction>(thunk->u1.Function);
                thunk->u1.Function = reinterpret_cast<ULONGLONG>(&TracedFree);
            }
            DWORD ignored{};
            VirtualProtect(&thunk->u1.Function, sizeof(thunk->u1.Function), previous_protection,
                           &ignored);
        }
    }
    const bool installed = original_malloc != nullptr && original_free != nullptr;
    malloc_trace_enabled.store(installed, std::memory_order_release);
    return installed;
}

void DumpMallocTrace() {
    malloc_trace_enabled.store(false, std::memory_order_release);
    while (malloc_summary_lock.test_and_set(std::memory_order_acquire)) {
        YieldProcessor();
    }
    const auto top = malloc_summary.TopByTotalBytes();
    const auto live_top = live_malloc_samples.Summarize().TopByTotalBytes();
    malloc_summary_lock.clear(std::memory_order_release);
    std::cerr << ";MALLOC_CALLSITE_SUMMARY callsites=" << top.size()
              << " overflow_count=" << malloc_summary.OverflowCount()
              << " overflow_bytes=" << malloc_summary.OverflowBytes() << '\n';
    for (std::size_t index = 0; index < std::min<std::size_t>(top.size(), 64); ++index) {
        const auto& stat = top[index];
        HMODULE caller_module{};
        char module_path[MAX_PATH]{};
        std::uintptr_t module_base{};
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<const char*>(stat.caller), &caller_module)) {
            module_base = reinterpret_cast<std::uintptr_t>(caller_module);
            GetModuleFileNameA(caller_module, module_path, MAX_PATH);
        }
        std::cerr << ";MALLOC_CALLSITE rank=" << index + 1 << " size=" << stat.size
                  << " count=" << stat.count << " bytes=" << stat.total_bytes << " caller=0x"
                  << std::hex << stat.caller << " module_base=0x" << module_base << " rva=0x"
                  << (module_base == 0 ? 0 : stat.caller - module_base) << std::dec
                  << " module=" << module_path << '\n';
    }
    std::cerr << ";LIVE_MALLOC_SAMPLE_SUMMARY callsites=" << live_top.size()
              << " tracker_overflow=" << malloc_sample_overflow.load(std::memory_order_relaxed)
              << '\n';
    for (std::size_t index = 0; index < std::min<std::size_t>(live_top.size(), 64); ++index) {
        const auto& stat = live_top[index];
        HMODULE caller_module{};
        char module_path[MAX_PATH]{};
        std::uintptr_t module_base{};
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<const char*>(stat.caller), &caller_module)) {
            module_base = reinterpret_cast<std::uintptr_t>(caller_module);
            GetModuleFileNameA(caller_module, module_path, MAX_PATH);
        }
        std::cerr << ";LIVE_MALLOC_SAMPLE rank=" << index + 1 << " size=" << stat.size
                  << " count=" << stat.count << " bytes=" << stat.total_bytes << " caller=0x"
                  << std::hex << stat.caller << " module_base=0x" << module_base << " rva=0x"
                  << (module_base == 0 ? 0 : stat.caller - module_base) << std::dec
                  << " module=" << module_path << '\n';
    }
}

void DumpSample(const std::size_t size, const char* position, const std::uintptr_t address) {
    if (address == 0) {
        return;
    }
    std::array<std::uintptr_t, 8> words{};
    SIZE_T copied{};
    if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), words.data(),
                          std::min(size, sizeof(words)), &copied)) {
        std::cerr << ";HEAP_SAMPLE size=" << size << " position=" << position << " address=0x"
                  << std::hex << address << " words=";
        for (std::size_t word = 0; word < copied / sizeof(std::uintptr_t); ++word) {
            std::cerr << (word == 0 ? "" : ",") << "0x" << words[word];
        }
        std::cerr << std::dec << '\n';
    }
}

void DumpWindowsHeapSummary() {
    DumpMallocTrace();
    const DWORD heap_count = GetProcessHeaps(0, nullptr);
    std::vector<HANDLE> heaps(heap_count);
    const DWORD populated = GetProcessHeaps(heap_count, heaps.data());
    heaps.resize(std::min(heap_count, populated));

    HeapSizeSummary summary;
    std::size_t allocation_count{};
    std::size_t allocation_bytes{};
    std::size_t failed_heaps{};
    std::size_t walked_entries{};
    bool truncated{};
    const ULONGLONG deadline = GetTickCount64() + 2000;
    for (const HANDLE heap : heaps) {
        if (!HeapLock(heap)) {
            ++failed_heaps;
            continue;
        }
        PROCESS_HEAP_ENTRY entry{};
        while (HeapWalk(heap, &entry)) {
            ++walked_entries;
            if ((entry.wFlags & PROCESS_HEAP_ENTRY_BUSY) != 0) {
                summary.Insert(entry.cbData, reinterpret_cast<std::uintptr_t>(entry.lpData));
                ++allocation_count;
                allocation_bytes += entry.cbData;
            }
            if ((walked_entries & 1023) == 0 && GetTickCount64() >= deadline) {
                truncated = true;
                break;
            }
        }
        HeapUnlock(heap);
        if (truncated) {
            break;
        }
    }

    std::cerr << ";HEAP_SIZE_SUMMARY allocations=" << allocation_count
              << " bytes=" << allocation_bytes << " heaps=" << heaps.size()
              << " failed_heaps=" << failed_heaps << " walked_entries=" << walked_entries
              << " truncated=" << truncated << " overflow_count="
              << summary.OverflowCount() << " overflow_bytes=" << summary.OverflowBytes() << '\n';
    const auto top = summary.TopByTotalBytes();
    for (std::size_t index = 0; index < std::min<std::size_t>(top.size(), 32); ++index) {
        const auto& stat = top[index];
        std::cerr << ";HEAP_SIZE rank=" << index + 1 << " size=" << stat.size
                  << " count=" << stat.count << " bytes=" << stat.total_bytes << " sample=0x"
                  << std::hex << stat.sample_address << " last_sample=0x"
                  << stat.last_sample_address << std::dec << '\n';
        if (stat.sample_address != 0 && (stat.size == 146 || index < 5)) {
            DumpSample(stat.size, "first", stat.sample_address);
            DumpSample(stat.size, "last", stat.last_sample_address);
        }
    }
    std::cerr.flush();
}

} // namespace

void MaybeStartWindowsHeapSummary(std::jthread& thread) {
    const char* value = std::getenv("SHADPS4_DIAGNOSTIC_HEAP_SECONDS");
    if (!value) {
        return;
    }
    char* end{};
    const unsigned long seconds = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || seconds == 0 || seconds > 3600) {
        return;
    }
    const char* malloc_trace = std::getenv("SHADPS4_DIAGNOSTIC_MALLOC");
    if (malloc_trace && std::strcmp(malloc_trace, "1") == 0 && !InstallMallocTrace()) {
        std::cerr << ";MALLOC_CALLSITE_TRACE install_failed=1\n";
    }
    thread = std::jthread([seconds](std::stop_token token) {
        std::mutex mutex;
        std::condition_variable_any wake;
        std::unique_lock lock{mutex};
        wake.wait_for(lock, token, std::chrono::seconds{seconds}, [] { return false; });
        if (!token.stop_requested()) {
            lock.unlock();
            DumpWindowsHeapSummary();
        }
    });
}

} // namespace Core::Ipc

#else

namespace Core::Ipc {

void MaybeStartWindowsHeapSummary(std::jthread&) {}

} // namespace Core::Ipc

#endif
