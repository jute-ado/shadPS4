// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>

#include "common/types.h"

namespace VideoCore {

constexpr bool IsCacheableFaultRange(VAddr start, VAddr end, VAddr address_space_size) {
    return start != 0 && start < address_space_size && end > start && end <= address_space_size &&
           end - start <= std::numeric_limits<u32>::max();
}

} // namespace VideoCore
