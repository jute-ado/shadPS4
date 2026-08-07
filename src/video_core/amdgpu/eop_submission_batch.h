// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

namespace AmdGpu {

class EopSubmissionBatch {
public:
    [[nodiscard]] bool HasPending() const noexcept {
        return eop_pending;
    }

    void MarkEopPending() noexcept {
        eop_pending = true;
    }

    template <typename Prepare, typename Submit>
    void SubmitBoundary(Prepare&& prepare, Submit&& submit) {
        FlushIfPending(std::forward<Prepare>(prepare), std::forward<Submit>(submit));
    }

    template <typename Prepare, typename Submit>
    void FlushIfPending(Prepare&& prepare, Submit&& submit) {
        if (!std::exchange(eop_pending, false)) {
            return;
        }
        std::forward<Prepare>(prepare)();
        std::forward<Submit>(submit)();
    }

private:
    bool eop_pending{};
};

} // namespace AmdGpu
