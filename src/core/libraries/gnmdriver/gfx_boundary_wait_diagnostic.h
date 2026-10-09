// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Libraries::GnmDriver {

// Diagnostic only: this waits for the prior global SubmitDone parser boundary, not a GPU fence
// or a graphics-only queue boundary. It intentionally releases admission while waiting so an
// earlier ASC submission can make parser progress; address-space revocation must use the stronger
// SubmissionGate::WaitForBoundaryAndEnter() operation, which retains admission across the wait.
template <typename Gate>
[[nodiscard]] auto EnterGraphicsSubmissionDiagnostic(Gate& gate, bool enabled) {
    if (enabled) {
        if constexpr (requires { gate.WaitForBoundaryThenEnter(); }) {
            return gate.WaitForBoundaryThenEnter();
        }
        return gate.WaitForBoundaryAndEnter();
    }
    return gate.Enter();
}

} // namespace Libraries::GnmDriver
