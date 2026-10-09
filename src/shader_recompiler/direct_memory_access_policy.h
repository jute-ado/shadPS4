// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Shader {

[[nodiscard]] constexpr bool ShouldUseDirectMemoryAccess(bool enabled, bool is_compute, u64 = 0) {
    return enabled && is_compute;
}

} // namespace Shader
