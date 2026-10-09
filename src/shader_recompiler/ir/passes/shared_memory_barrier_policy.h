// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Shader::Optimization {

enum class SharedMemoryAccess {
    Load,
    Write,
};

[[nodiscard]] constexpr bool NeedsSharedMemoryBarrier(SharedMemoryAccess previous,
                                                       SharedMemoryAccess current) {
    return previous != current;
}

[[nodiscard]] constexpr bool NeedsSharedMemoryBarrierEmulation(u32 threadgroup_size,
                                                                bool driver_requires_barriers) {
    return driver_requires_barriers && threadgroup_size > 0;
}

} // namespace Shader::Optimization
