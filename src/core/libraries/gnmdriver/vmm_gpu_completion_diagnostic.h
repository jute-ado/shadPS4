// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>

namespace Libraries::GnmDriver {

inline constexpr std::string_view VmmGpuCompletionWaitEnvironment =
    "SHADPS4_VMM_WAIT_GPU_COMPLETION";

/// Parses the launch-time diagnostic switch without accepting near-miss values.
[[nodiscard]] constexpr bool IsVmmGpuCompletionWaitEnabled(const char* value) noexcept {
    return value != nullptr && std::string_view{value} == "1";
}

/// Acquires address-space admission using the retired fence sequence only when explicitly enabled.
/// A missing GPU owner falls back to ordinary admission so early VMM calls cannot wait forever for
/// a boundary or completion worker that does not exist yet.
template <typename Gate, typename Gpu>
[[nodiscard]] auto AcquireVmmGpuAddressSpaceGuard(Gate& gate, Gpu* gpu, bool enabled) {
    if (enabled && gpu != nullptr) {
        auto admission = gate.WaitForBoundaryAndEnter();
        gpu->WaitGpuCompletion();
        return admission;
    }
    return gate.Enter();
}

} // namespace Libraries::GnmDriver
