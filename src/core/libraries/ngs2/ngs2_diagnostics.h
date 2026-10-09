// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <span>
#include <string_view>
#include "common/types.h"

namespace Libraries::Ngs2::Diagnostic {

enum class Kind : u8 { System, Rack, Control, Render, State, Count };
enum class Status : u8 { Observed, Null, Unreadable, ShortAbi, UnknownId, Cycle, Overflow, Limit };

struct Field {
    std::string_view name;
    u64 value{};
    bool is_signed{};
};

// Only explicitly selected scalar metadata may enter a record. Never add handles, addresses,
// strings read from guest memory, raw payloads, sample bytes, or pointed-to waveform blocks.
struct Record {
    Kind kind{};
    Status status{Status::Observed};
    u32 index{};
    std::array<Field, 12> fields{};
    u32 count{};
};

struct AddressRange {
    u64 base{};
    u64 size{};
};

using Reader = std::function<bool(u64, std::span<u8>)>;
using Sink = std::function<void(const Record&)>;

// A checked host-readable snapshot, not proof of guest ABI validity or guest CPU permission.
// Windows only; other platforms deliberately report unreadable for this optional diagnostic.
bool ReadMetadata(u64 address, std::span<u8> output,
                  std::span<const AddressRange> reservations) noexcept;

class Observer {
public:
    static constexpr std::array<u32, static_cast<size_t>(Kind::Count)> RequestCaps{8, 24, 16, 8, 8};
    static constexpr u32 MaxControlHeaders = 8;
    static constexpr u32 MaxRenderBuffers = 4;

    void System(bool enabled, u64 option, bool allocator, const Reader&, const Sink&) noexcept;
    void Rack(bool enabled, u32 rack_id, u64 option, bool allocator, const Reader&,
              const Sink&) noexcept;
    void Control(bool enabled, u64 list, const Reader&, const Sink&) noexcept;
    void Render(bool enabled, u64 buffers, u32 count, const Reader&, const Sink&) noexcept;
    void State(bool enabled, u64 state_size, const Sink&) noexcept;

private:
    bool Claim(bool enabled, Kind kind) noexcept;
    std::array<std::atomic<u32>, static_cast<size_t>(Kind::Count)> requests{};
};

std::string_view Name(Kind kind) noexcept;
std::string_view Name(Status status) noexcept;

} // namespace Libraries::Ngs2::Diagnostic
