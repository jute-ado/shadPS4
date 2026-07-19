// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::Kernel {

/// Applies select(2) timeout semantics when no host socket can perform the wait.
/// Returns false when the timeval components are invalid.
bool WaitForSelectTimeout(s64 seconds, s64 microseconds);

} // namespace Libraries::Kernel
