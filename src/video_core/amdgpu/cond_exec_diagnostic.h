// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/types.h"

namespace AmdGpu {

struct CondExecDiagnosticObservation {
    bool skip{};
    bool should_report{};
    u64 occurrence{};
};

class CondExecDiagnosticTracker {
public:
    explicit CondExecDiagnosticTracker(u64 report_limit_) : report_limit{report_limit_} {}

    CondExecDiagnosticObservation Observe(bool predicate) noexcept {
        const bool skip = !predicate;
        const u64 occurrence = total.fetch_add(1, std::memory_order_relaxed) + 1;
        if (skip) {
            skipped.fetch_add(1, std::memory_order_relaxed);
        }
        return {
            .skip = skip,
            .should_report = occurrence <= report_limit,
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
