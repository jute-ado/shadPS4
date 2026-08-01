// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>
#include <vector>

#include "common/types.h"
#include "common/unique_function.h"

namespace AmdGpu {

enum class FlipEopPosition {
    Preceding,
    Following,
};

[[nodiscard]] constexpr FlipEopPosition DecodeFlipEopPosition(u32 nop_count) noexcept {
    return nop_count == 0x33 ? FlipEopPosition::Following : FlipEopPosition::Preceding;
}

class EopFlipTracker {
public:
    void BeginEop() {
        last_eop_completed = false;
    }

    void CompleteEop() {
        last_eop_completed = true;
        for (auto& flip : pending_flips) {
            flip();
        }
        pending_flips.clear();
    }

    [[nodiscard]] bool QueueFlip(FlipEopPosition position,
                                 Common::UniqueFunction<void>&& callback) {
        if (position == FlipEopPosition::Following) {
            pending_flips.emplace_back(std::move(callback));
            return true;
        }
        if (!last_eop_completed) {
            return false;
        }
        callback();
        return true;
    }

private:
    bool last_eop_completed{};
    std::vector<Common::UniqueFunction<void>> pending_flips;
};

} // namespace AmdGpu
