// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>

#include "common/polyfill_thread.h"
#include "core/libraries/gnmdriver/submission_gate.h"

namespace Libraries::GnmDriver {

// Connected by the VMM provider: only destructive backing retirement requests this admission.
template <typename CaptureTicket, typename IssuedSequence, typename FinishGpu>
[[nodiscard]] std::optional<SubmissionGate::Guard> AcquireSubmissionRetirement(
    SubmissionGate& gate, CaptureTicket&& capture_ticket, IssuedSequence&& issued_sequence,
    FinishGpu&& finish_gpu, std::stop_token stop = {}) {
    while (!stop.stop_requested()) {
        const auto ticket = [&] {
            auto issuance = gate.Enter();
            return capture_ticket();
        }();
        // Never retain submission admission here: an old WAIT_REG_MEM may need a fresh
        // allocation and a newer ASC/GFX submission to make progress.
        if (!ticket.Wait(stop)) {
            return std::nullopt;
        }
        auto admission = gate.TryEnterRetirement();
        if (!admission) {
            if (!gate.WaitForBoundary(stop)) {
                return std::nullopt;
            }
            continue;
        }
        if (issued_sequence() != ticket.sequence) {
            continue;
        }
        if (stop.stop_requested()) {
            return std::nullopt;
        }
        // All issued commands have retired and later issuance is now excluded. The caller has
        // released allocator locks; Finish also publishes owner-drained EOP host completions.
        finish_gpu();
        if (stop.stop_requested()) {
            return std::nullopt;
        }
        return std::move(*admission);
    }
    return std::nullopt;
}

} // namespace Libraries::GnmDriver
