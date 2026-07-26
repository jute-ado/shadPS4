// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace AmdGpu {

class WaitYieldTracker {
public:
    explicit WaitYieldTracker(u64 report_threshold_) : next_report_threshold{report_threshold_} {}

    [[nodiscard]] bool ObserveYield() noexcept {
        ++yield_count;
        if (yield_count != next_report_threshold) {
            return false;
        }
        next_report_threshold *= 2;
        return true;
    }

    [[nodiscard]] u64 YieldCount() const noexcept {
        return yield_count;
    }

private:
    u64 next_report_threshold;
    u64 yield_count{};
};

} // namespace AmdGpu
