// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>

namespace Libraries::GnmDriver {

[[nodiscard]] constexpr bool IsGfxBoundaryWaitDiagnosticEnabled(const char* value) noexcept {
    return value != nullptr && std::string_view{value} == "1";
}

// Diagnostic only: this waits for the prior global SubmitDone parser boundary, not a GPU fence
// or a graphics-only queue boundary. An earlier ASC submission needing this next GFX submission
// can therefore stall indefinitely. Use only with an external experiment timeout.
template <typename Gate>
[[nodiscard]] auto EnterGraphicsSubmissionDiagnostic(Gate& gate, bool enabled) {
    if (enabled) {
        return gate.WaitForBoundaryAndEnter();
    }
    return gate.Enter();
}

} // namespace Libraries::GnmDriver
