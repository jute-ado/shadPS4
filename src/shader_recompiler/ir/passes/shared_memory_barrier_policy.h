// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Shader::Optimization {

constexpr bool NeedsSharedMemoryBarriers(bool is_compute, u32 shared_memory_size,
                                         u32 threadgroup_size, bool host_needs_barriers) {
    return is_compute && shared_memory_size != 0 && threadgroup_size != 0 && host_needs_barriers;
}

} // namespace Shader::Optimization
