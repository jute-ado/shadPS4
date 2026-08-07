// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

#include "core/libraries/gnmdriver/submission_gate.h"

namespace Libraries::GnmDriver {

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
