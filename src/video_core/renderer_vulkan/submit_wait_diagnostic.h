// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <string_view>

#include "common/types.h"

namespace Vulkan {

inline constexpr std::string_view SubmitWaitDiagnosticEnvironment =
    "SHADPS4_SUBMIT_WAIT_DIAGNOSTIC";

// Submit/wait tracing is intentionally launch-time and strict. The normal environment cannot
// enable a high-volume diagnostic accidentally, and no Vulkan commands are changed by tracing.
[[nodiscard]] constexpr bool ParseSubmitWaitDiagnostic(std::string_view value) noexcept {
    return value == "1";
}

[[nodiscard]] inline bool SubmitWaitDiagnosticEnabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv(SubmitWaitDiagnosticEnvironment.data());
        return value != nullptr && ParseSubmitWaitDiagnostic(value);
    }();
    return enabled;
}

inline std::atomic<u64> submit_wait_next_submit_id{1};
inline std::atomic<u64> submit_wait_next_wait_id{1};

[[nodiscard]] inline u64 NextSubmitDiagnosticId() noexcept {
    return submit_wait_next_submit_id.fetch_add(1, std::memory_order_relaxed);
}

[[nodiscard]] inline u64 NextWaitDiagnosticId() noexcept {
    return submit_wait_next_wait_id.fetch_add(1, std::memory_order_relaxed);
}

} // namespace Vulkan
