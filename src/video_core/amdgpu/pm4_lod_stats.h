// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <optional>
#include <span>

#include "common/types.h"

namespace AmdGpu {

// IT_GET_LOD_STATS carries a three-dword body:
//   [0] report size, [1] low address, [2] high address/control.
// The high eight bits of the GPU address share the low byte of the control dword.
inline constexpr size_t LodStatsPayloadDwords = 3;
inline constexpr u64 LodStatsGpuAddressLimit = 1ULL << 40;
inline constexpr u32 LodStatsAddressLowMask = 0xffffffc0u;
inline constexpr u32 LodStatsAddressHighMask = 0x000000ffu;

struct LodStatsPacket {
    VAddr address{};
    u32 size{};
    u32 control{};
};

// Decode the three information dwords carried by IT_GET_LOD_STATS. This is deliberately a
// syntax/range gate only: it does not dereference the report address or claim the report complete.
[[nodiscard]] constexpr std::optional<LodStatsPacket> DecodeLodStatsPacket(
    std::span<const u32> payload) noexcept {
    if (payload.size() != LodStatsPayloadDwords) {
        return std::nullopt;
    }

    const u32 size = payload[0];
    const u32 control = payload[2];
    const u32 address_high = control & LodStatsAddressHighMask;

    const u64 address = (static_cast<u64>(address_high) << 32) |
                        static_cast<u64>(payload[1] & LodStatsAddressLowMask);

    // The Liverpool report address is 64-byte aligned and uses a 40-bit GPU address. Keep the
    // range check overflow-safe so a malformed report cannot wrap into an apparently valid range.
    if (address == 0 || size == 0 || address >= LodStatsGpuAddressLimit ||
        static_cast<u64>(size) > LodStatsGpuAddressLimit - address) {
        return std::nullopt;
    }

    return LodStatsPacket{.address = static_cast<VAddr>(address), .size = size, .control = control};
}

} // namespace AmdGpu
