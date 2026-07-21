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

[[nodiscard]] inline LodStatsReport DecodeLodStatsReport(std::span<const u32, 3> payload) {
    const u32 control = payload[2];
    const VAddr address = (payload[1] & 0xffffffc0u) | (u64(control & 0xffu) << 32u);
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
