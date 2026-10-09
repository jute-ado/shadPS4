// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "ngs2_diagnostics.h"

#include <algorithm>
#include <cstring>
#include <limits>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libraries::Ngs2::Diagnostic {
namespace {
constexpr u64 MaxAddress = std::numeric_limits<u64>::max();

void Add(Record& record, std::string_view name, u64 value, bool is_signed = false) {
    record.fields[record.count++] = {name, value, is_signed};
}

Status Read(const Reader& reader, u64 base, u64 offset, std::span<u8> output) {
    if (offset > MaxAddress - base || output.size() > MaxAddress - (base + offset)) {
        return Status::Overflow;
    }
    return reader(base + offset, output) ? Status::Observed : Status::Unreadable;
}

template <typename T>
std::span<u8> Bytes(T& object) {
    return {reinterpret_cast<u8*>(&object), sizeof(T)};
}

// These are fixed scalar prefixes of the existing PS4 ABI, not a general payload dump.
// Production static_asserts tie their offsets/sizes to the public declarations.
struct Payload {
    u32 id;
    u16 minimum_size;
    u16 offset;
    std::array<std::string_view, 7> names;
    u32 count;
};
constexpr std::array Payloads{
    Payload{0x10000000,
            40,
            8,
            {"waveform_type", "channels", "sample_rate", "config", "frame_offset", "frame_margin",
             "flags"},
            7},
    Payload{0x10000001, 32, 16, {"flags", "blocks"}, 2},
    Payload{0x20000000, 16, 8, {"channels", "flags"}, 2},
    Payload{0x30000000, 16, 8, {"channels", "flags"}, 2},
    Payload{0x30000005, 16, 8, {"output_id"}, 1},
    Payload{1, 24, 8, {"matrix_id", "levels"}, 2},
    Payload{2, 16, 8, {"port", "level_bits"}, 2},
    Payload{3, 16, 8, {"port", "matrix_id"}, 2},
    Payload{4, 16, 8, {"port", "delay_samples"}, 2},
    Payload{5, 24, 8, {"port", "dest_input"}, 2},
    Payload{6, 12, 8, {"event_id"}, 1},
};

Status ReadPayload(const Reader& reader, u64 address, u16 size, u32 id, Record& record) {
    const auto it = std::ranges::find(Payloads, id, &Payload::id);
    if (it == Payloads.end()) {
        return Status::UnknownId;
    }
    if (size < it->minimum_size) {
        return Status::ShortAbi;
    }
    std::array<u32, 7> values{};
    const auto status = Read(reader, address, it->offset, Bytes(values).first(it->count * 4));
    if (status != Status::Observed) {
        return status;
    }
    for (u32 i = 0; i < it->count; ++i) {
        const bool signed_matrix = id == 3 && i == 1;
        Add(record, it->names[i],
            signed_matrix ? static_cast<u64>(static_cast<s32>(values[i])) : values[i],
            signed_matrix);
    }
    return Status::Observed;
}
} // namespace

bool ReadMetadata(u64 address, std::span<u8> output,
                  std::span<const AddressRange> reservations) noexcept {
    if (output.empty() || output.size() > 64 || output.size() > MaxAddress - address) {
        return false;
    }
    const bool reserved = std::ranges::any_of(reservations, [&](const AddressRange& range) {
        return address >= range.base && address - range.base <= range.size &&
               output.size() <= range.size - (address - range.base);
    });
    if (!reserved) {
        return false;
    }
#ifdef _WIN32
    // Reservation membership alone does not mean mapped/readable. Check the host pages without
    // touching guard/no-access memory; ReadProcessMemory still handles a concurrent unmap safely.
    const u64 end = address + output.size();
    for (u64 cursor = address; cursor < end;) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(reinterpret_cast<const void*>(cursor), &info, sizeof(info)) ||
            info.State != MEM_COMMIT || (info.Protect & (PAGE_GUARD | PAGE_NOACCESS))) {
            return false;
        }
        const DWORD access = info.Protect & 0xff;
        if (access != PAGE_READONLY && access != PAGE_READWRITE && access != PAGE_WRITECOPY &&
            access != PAGE_EXECUTE_READ && access != PAGE_EXECUTE_READWRITE &&
            access != PAGE_EXECUTE_WRITECOPY) {
            return false;
        }
        const auto base = reinterpret_cast<u64>(info.BaseAddress);
        if (info.RegionSize > MaxAddress - base || base + info.RegionSize <= cursor) {
            return false;
        }
        cursor = std::min(end, base + info.RegionSize);
    }
    SIZE_T copied{};
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address),
                             output.data(), output.size(), &copied) &&
           copied == output.size();
#else
    return false;
#endif
}

bool Observer::Claim(bool enabled, Kind kind) noexcept {
    if (!enabled) {
        return false;
    }
    const auto index = static_cast<size_t>(kind);
    auto& counter = requests[index];
    auto value = counter.load(std::memory_order_relaxed);
    while (value < RequestCaps[index]) {
        if (counter.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

void Observer::System(bool enabled, u64 option, bool allocator, const Reader& reader,
                      const Sink& sink) noexcept {
    if (!Claim(enabled, Kind::System)) {
        return;
    }
    try {
        Record record{Kind::System};
        Add(record, "allocator", allocator);
        u64 size{};
        record.status = option == 0 ? Status::Null : Read(reader, option, 0, Bytes(size));
        if (record.status == Status::Observed) {
            Add(record, "option_size", size);
            std::array<u32, 4> fields{};
            record.status = size < 64 ? Status::ShortAbi : Read(reader, option, 24, Bytes(fields));
            if (record.status == Status::Observed) {
                constexpr std::array names{"flags", "max_grain_samples", "grain_samples",
                                           "sample_rate"};
                for (u32 i = 0; i < fields.size(); ++i) {
                    Add(record, names[i], fields[i]);
                }
            }
        }
        sink(record);
    } catch (...) {
        // Optional observations must not change the API result or unwind through guest code.
    }
}

void Observer::Rack(bool enabled, u32 rack_id, u64 option, bool allocator, const Reader& reader,
                    const Sink& sink) noexcept {
    if (!Claim(enabled, Kind::Rack)) {
        return;
    }
    try {
        Record record{Kind::Rack};
        Add(record, "rack_id", rack_id);
        Add(record, "allocator", allocator);
        u64 size{};
        record.status = option == 0 ? Status::Null : Read(reader, option, 0, Bytes(size));
        if (record.status == Status::Observed) {
            Add(record, "option_size", size);
            std::array<u32, 6> fields{};
            record.status = size < 128 ? Status::ShortAbi : Read(reader, option, 24, Bytes(fields));
            if (record.status == Status::Observed) {
                constexpr std::array names{"flags",        "max_grain_samples",
                                           "max_voices",   "max_input_delays",
                                           "max_matrices", "max_ports"};
                for (u32 i = 0; i < fields.size(); ++i) {
                    Add(record, names[i], fields[i]);
                }
            }
        }
        sink(record);
    } catch (...) {
    }
}

void Observer::Control(bool enabled, u64 list, const Reader& reader, const Sink& sink) noexcept {
    if (!Claim(enabled, Kind::Control)) {
        return;
    }
    try {
        std::array<u64, MaxControlHeaders> visited{};
        for (u32 index = 0; index < MaxControlHeaders; ++index) {
            Record record{Kind::Control, Status::Observed, index};
            struct Header {
                u16 size;
                s16 next;
                u32 id;
            } header{};
            static_assert(sizeof(Header) == 8);
            record.status = list == 0 ? Status::Null : Read(reader, list, 0, Bytes(header));
            if (record.status != Status::Observed) {
                sink(record);
                return;
            }
            visited[index] = list;
            Add(record, "id", header.id);
            Add(record, "size", header.size);
            Add(record, "next", static_cast<u64>(header.next), true);
            record.status = header.size < 8
                                ? Status::ShortAbi
                                : ReadPayload(reader, list, header.size, header.id, record);
            if (record.status != Status::Observed && record.status != Status::UnknownId) {
                sink(record);
                return;
            }
            if (header.next == 0) {
                sink(record);
                return;
            }
            const auto distance =
                static_cast<u64>(header.next < 0 ? -static_cast<s32>(header.next) : header.next);
            if ((header.next < 0 && distance > list) ||
                (header.next > 0 && distance > MaxAddress - list)) {
                record.status = Status::Overflow;
            } else {
                const u64 next = header.next < 0 ? list - distance : list + distance;
                if (std::find(visited.begin(), visited.begin() + index + 1, next) !=
                    visited.begin() + index + 1) {
                    record.status = Status::Cycle;
                } else if (index + 1 == MaxControlHeaders) {
                    record.status = Status::Limit;
                } else {
                    list = next;
                    sink(record);
                    continue;
                }
            }
            sink(record);
            return;
        }
    } catch (...) {
    }
}

void Observer::Render(bool enabled, u64 buffers, u32 count, const Reader& reader,
                      const Sink& sink) noexcept {
    if (!Claim(enabled, Kind::Render)) {
        return;
    }
    try {
        Record request{Kind::Render};
        Add(request, "buffer_count", count);
        request.status = count != 0 && buffers == 0 ? Status::Null
                         : count > MaxRenderBuffers ? Status::Limit
                                                    : Status::Observed;
        sink(request);
        if (count == 0 || buffers == 0) {
            return;
        }
        for (u32 i = 0; i < std::min(count, MaxRenderBuffers); ++i) {
            Record record{Kind::Render, Status::Observed, i};
            struct Fields {
                u64 size;
                u32 format;
                u32 channels;
            } fields{};
            static_assert(sizeof(Fields) == 16);
            record.status = Read(reader, buffers, i * 24 + 8, Bytes(fields));
            if (record.status == Status::Observed) {
                Add(record, "buffer_size", fields.size);
                Add(record, "waveform_type", fields.format);
                Add(record, "channels", fields.channels);
            }
            sink(record);
            if (record.status != Status::Observed) {
                return;
            }
        }
    } catch (...) {
    }
}

void Observer::State(bool enabled, u64 state_size, const Sink& sink) noexcept {
    if (!Claim(enabled, Kind::State)) {
        return;
    }
    try {
        Record record{Kind::State};
        Add(record, "state_size", state_size);
        sink(record);
    } catch (...) {
    }
}

std::string_view Name(Kind kind) noexcept {
    constexpr std::array names{"system", "rack", "control", "render", "state"};
    return names[static_cast<size_t>(kind)];
}

std::string_view Name(Status status) noexcept {
    constexpr std::array names{"observed",   "null",  "unreadable", "short_abi",
                               "unknown_id", "cycle", "overflow",   "limit"};
    return names[static_cast<size_t>(status)];
}
} // namespace Libraries::Ngs2::Diagnostic
