// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstddef>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <windows.h>

#include "core/windows_protection_snapshot.h"

namespace {

class SplitFileMapping {
public:
    ~SplitFileMapping() {
        for (void* view : views) {
            EXPECT_TRUE(UnmapViewOfFile2(process, view, MEM_PRESERVE_PLACEHOLDER));
        }
        for (void* placeholder : placeholders) {
            EXPECT_TRUE(VirtualFreeEx(process, placeholder, 0, MEM_RELEASE));
        }
        if (mapping != nullptr) {
            EXPECT_TRUE(CloseHandle(mapping));
        }
    }

    bool Initialize() {
        SYSTEM_INFO system_info{};
        GetSystemInfo(&system_info);
        segment_size = system_info.dwAllocationGranularity;
        const u64 total_size = segment_size * 3;

        mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                     static_cast<DWORD>(total_size), nullptr);
        if (mapping == nullptr) {
            return false;
        }

        base = static_cast<std::byte*>(VirtualAlloc2(process, nullptr, total_size,
                                                     MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                                     PAGE_NOACCESS, nullptr, 0));
        if (base == nullptr) {
            return false;
        }
        placeholders.push_back(base);

        void* view = MapViewOfFile3(mapping, process, base, 0, total_size, MEM_REPLACE_PLACEHOLDER,
                                    PAGE_READWRITE, nullptr, 0);
        if (view != base) {
            return false;
        }
        views.push_back(view);
        return true;
    }

    bool SplitAndRemap() {
        if (!UnmapViewOfFile2(process, base, MEM_PRESERVE_PLACEHOLDER)) {
            return false;
        }
        views.clear();

        if (!VirtualFreeEx(process, base, segment_size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            return false;
        }
        placeholders = {base, base + segment_size};

        void* first = MapViewOfFile3(mapping, process, base, 0, segment_size,
                                     MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0);
        if (first != base) {
            return false;
        }
        views.push_back(first);

        void* remainder =
            MapViewOfFile3(mapping, process, base + segment_size, segment_size, segment_size * 2,
                           MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0);
        if (remainder != base + segment_size) {
            return false;
        }
        views.push_back(remainder);
        return true;
    }

    [[nodiscard]] VAddr Address() const {
        return reinterpret_cast<VAddr>(base);
    }

    [[nodiscard]] u64 SegmentSize() const {
        return segment_size;
    }

private:
    HANDLE process = GetCurrentProcess();
    HANDLE mapping{};
    std::byte* base{};
    u64 segment_size{};
    std::vector<void*> views;
    std::vector<void*> placeholders;
};

DWORD ProtectionAt(VAddr address) {
    MEMORY_BASIC_INFORMATION info{};
    EXPECT_NE(VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)), 0U);
    return info.Protect;
}

} // namespace

TEST(WindowsProtectionSnapshot, RestoresProtectionOverridesAfterMappedViewSplit) {
    SplitFileMapping mapping;
    ASSERT_TRUE(mapping.Initialize());

    const VAddr protected_address = mapping.Address() + mapping.SegmentSize() * 2;
    DWORD old_protection{};
    ASSERT_TRUE(VirtualProtect(reinterpret_cast<void*>(protected_address), mapping.SegmentSize(),
                               PAGE_READONLY, &old_protection));

    const auto protections = Core::CaptureWindowsProtectionOverrides(
        GetCurrentProcess(), mapping.Address(), mapping.SegmentSize() * 3, PAGE_READWRITE);
    ASSERT_TRUE(protections.has_value());
    ASSERT_EQ(protections->size(), 1U);
    EXPECT_EQ(protections->front().address, protected_address);
    EXPECT_EQ(protections->front().size, mapping.SegmentSize());
    EXPECT_EQ(protections->front().protection, PAGE_READONLY);

    ASSERT_TRUE(mapping.SplitAndRemap());
    ASSERT_EQ(ProtectionAt(protected_address), PAGE_READWRITE);

    ASSERT_TRUE(Core::RestoreWindowsProtectionOverrides(GetCurrentProcess(), *protections));
    EXPECT_EQ(ProtectionAt(protected_address), PAGE_READONLY);
}

TEST(WindowsProtectionSnapshot, RestoresOverrideAcrossNewViewBoundaries) {
    SplitFileMapping mapping;
    ASSERT_TRUE(mapping.Initialize());

    DWORD old_protection{};
    ASSERT_TRUE(VirtualProtect(reinterpret_cast<void*>(mapping.Address()),
                               mapping.SegmentSize() * 3, PAGE_READONLY, &old_protection));

    const auto protections = Core::CaptureWindowsProtectionOverrides(
        GetCurrentProcess(), mapping.Address(), mapping.SegmentSize() * 3, PAGE_READWRITE);
    ASSERT_TRUE(protections.has_value());
    ASSERT_EQ(protections->size(), 1U);
    EXPECT_EQ(protections->front().address, mapping.Address());
    EXPECT_EQ(protections->front().size, mapping.SegmentSize() * 3);

    ASSERT_TRUE(mapping.SplitAndRemap());
    ASSERT_EQ(ProtectionAt(mapping.Address()), PAGE_READWRITE);
    ASSERT_EQ(ProtectionAt(mapping.Address() + mapping.SegmentSize() * 2), PAGE_READWRITE);

    ASSERT_TRUE(Core::RestoreWindowsProtectionOverrides(GetCurrentProcess(), *protections));
    EXPECT_EQ(ProtectionAt(mapping.Address()), PAGE_READONLY);
    EXPECT_EQ(ProtectionAt(mapping.Address() + mapping.SegmentSize() * 2), PAGE_READONLY);
}
