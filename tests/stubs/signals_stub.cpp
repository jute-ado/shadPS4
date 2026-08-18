// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/signals.h"

namespace Core {

SignalDispatch::SignalDispatch() = default;
SignalDispatch::~SignalDispatch() = default;

void SignalDispatch::RemoveHandlers() {
    access_violation_handlers.clear();
    illegal_instruction_handlers.clear();
}

bool SignalDispatch::DispatchAccessViolation(void* context, void* fault_address) const {
    for (const auto& entry : access_violation_handlers) {
        if (entry.handler(context, fault_address)) {
            return true;
        }
    }
    return false;
}

bool SignalDispatch::DispatchIllegalInstruction(void* context) const {
    for (const auto& entry : illegal_instruction_handlers) {
        if (entry.handler(context)) {
            return true;
        }
    }
    return false;
}

} // namespace Core
