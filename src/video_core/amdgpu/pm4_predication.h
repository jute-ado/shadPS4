// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>

#include "common/types.h"

namespace AmdGpu {

enum class PredicationOperation : u32 {
    Clear = 0,
    Zpass = 1,
    PrimCount = 2,
    Reserved3 = 3,
    Reserved4 = 4,
    Reserved5 = 5,
    Reserved6 = 6,
    Reserved7 = 7,
};

struct LiverpoolSetPredication {
    u64 address{};
    bool draw_if_visible{};
    bool wait_for_results{};
    PredicationOperation operation{};
    bool continue_predicate{};
};

[[nodiscard]] constexpr std::optional<LiverpoolSetPredication>
DecodeLiverpoolSetPredication(std::span<const u32> payload) {
    if (payload.size() < 2) {
        return std::nullopt;
    }

    const u32 control_and_high = payload[1];
    return LiverpoolSetPredication{
        .address = static_cast<u64>(control_and_high & 0xffu) << 32 |
                   static_cast<u64>(payload[0] & 0xfffffff0u),
        .draw_if_visible = ((control_and_high >> 8) & 1u) != 0,
        // Liverpool hint 0 waits for the final result. Hint 1 samples once and must fail open
        // when the result is not yet available.
        .wait_for_results = ((control_and_high >> 12) & 1u) == 0,
        .operation = static_cast<PredicationOperation>((control_and_high >> 16) & 7u),
        .continue_predicate = ((control_and_high >> 31) & 1u) != 0,
    };
}

} // namespace AmdGpu
