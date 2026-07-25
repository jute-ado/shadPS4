// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/kernel/threads/pthread.h"

namespace Libraries::Kernel {

[[nodiscard]] constexpr bool IsValidPthreadScope(PthreadAttrFlags scope) noexcept {
    return scope == PthreadAttrFlags::ScopeProcess || scope == PthreadAttrFlags::ScopeSystem;
}

} // namespace Libraries::Kernel
