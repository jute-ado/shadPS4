// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <span>
#include "common/types.h"

namespace AmdGpu {

struct LodStatsReport {
    VAddr address;
    u32 size;
    u32 control;
};

[[nodiscard]] inline LodStatsReport DecodeLodStatsReport(std::span<const u32, 4> payload) {
    // The ME packet stores a 40-bit byte address as base_lo/base_hi.  The low
    // six bits of base_lo are reserved, and the fourth body dword is the
    // report-control word.
    const u32 control = payload[3];
    const VAddr address = static_cast<VAddr>(payload[1] & 0xffffffc0u) |
                          (static_cast<VAddr>(payload[2]) << 32u);
    return {.address = address, .size = payload[0], .control = control};
}

inline void InitializeLodStatsReport(std::span<std::byte> output) {
    std::ranges::fill(output, std::byte{});
    if (output.size() >= sizeof(u32)) {
        constexpr u32 ReportComplete = 1;
        std::memcpy(output.data(), &ReportComplete, sizeof(ReportComplete));
    }
}

} // namespace AmdGpu
