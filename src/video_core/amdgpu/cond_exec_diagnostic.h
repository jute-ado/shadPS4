// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/types.h"

namespace AmdGpu {

enum class CondExecSampleKind {
    Zero,
    One,
    OtherNonZero,
};

struct CondExecDiagnosticObservation {
    bool skip{};
    bool should_report{};
    bool registered{};
    bool gpu_dirty{};
    CondExecSampleKind sample_kind{};
    u64 occurrence{};
};

class CondExecDiagnosticTracker {
public:
    explicit CondExecDiagnosticTracker(u64 report_limit_) : report_limit{report_limit_} {}

    CondExecDiagnosticObservation Observe(u32 value, bool registered, bool gpu_dirty) noexcept {
        const bool skip = value == 0;
        const u64 occurrence = total.fetch_add(1, std::memory_order_relaxed) + 1;
        if (skip) {
            skipped.fetch_add(1, std::memory_order_relaxed);
        }
        const auto sample_kind = value == 0   ? CondExecSampleKind::Zero
                                 : value == 1 ? CondExecSampleKind::One
                                              : CondExecSampleKind::OtherNonZero;
        return {
            .skip = skip,
            .should_report = occurrence <= report_limit,
            .registered = registered,
            .gpu_dirty = gpu_dirty,
            .sample_kind = sample_kind,
            .occurrence = occurrence,
        };
    }

    [[nodiscard]] u64 Total() const noexcept {
        return total.load(std::memory_order_relaxed);
    }

    [[nodiscard]] u64 Skipped() const noexcept {
        return skipped.load(std::memory_order_relaxed);
    }

    [[nodiscard]] u64 Executed() const noexcept {
        return Total() - Skipped();
    }

private:
    u64 report_limit;
    std::atomic<u64> total{};
    std::atomic<u64> skipped{};
};

} // namespace AmdGpu
