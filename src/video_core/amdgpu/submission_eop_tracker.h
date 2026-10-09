// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace AmdGpu {

class SubmissionEopTracker {
public:
    void ObserveInterruptingEop() noexcept {
        observed_interrupting_eop = true;
    }

    [[nodiscard]] bool CompleteBoundary() noexcept {
        const bool requires_progress_interrupt = !observed_interrupting_eop;
        observed_interrupting_eop = false;
        return requires_progress_interrupt;
    }

private:
    bool observed_interrupting_eop{};
};

} // namespace AmdGpu
