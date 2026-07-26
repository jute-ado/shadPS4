// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace AmdGpu {

class WaitYieldTracker {
public:
    explicit WaitYieldTracker(u64 report_threshold_) : report_threshold{report_threshold_} {}

    [[nodiscard]] bool ObserveYield() noexcept {
        ++yield_count;
        return yield_count == report_threshold;
    }

private:
    u64 report_threshold;
    u64 yield_count{};
};

} // namespace AmdGpu
