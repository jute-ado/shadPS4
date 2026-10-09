// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

#include "core/libraries/gnmdriver/submission_gate.h"

namespace Libraries::GnmDriver {

// Keep a flip's side effects and its command-buffer copy in one admission critical section.  The
// callbacks deliberately run in this order: malformed input is rejected without taking the gate,
// then flip registration is serialized with the subsequent enqueue.  In particular, the enqueue
// callback must call an implementation that assumes admission is already held; calling the public
// submit entry point here would try to acquire the non-recursive gate a second time.
template <typename Validate, typename Register, typename Enqueue>
auto RunSubmissionTransaction(SubmissionGate& gate, Validate&& validate,
                              Register&& register_submission, Enqueue&& enqueue) {
    const auto validation_result = std::invoke(std::forward<Validate>(validate));
    if (validation_result != 0) {
        return validation_result;
    }

    auto submission = gate.Enter();
    const auto registration_result = std::invoke(std::forward<Register>(register_submission));
    if (registration_result != 0) {
        return registration_result;
    }

    return std::invoke(std::forward<Enqueue>(enqueue));
}

} // namespace Libraries::GnmDriver
