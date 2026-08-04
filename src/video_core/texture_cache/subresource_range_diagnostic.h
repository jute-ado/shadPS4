// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

struct SubresourceDimensions {
    u32 levels;
    u32 layers;
};

struct SubresourceRangeBounds {
    u32 base_level;
    u32 base_layer;
    u32 levels;
    u32 layers;
};

[[nodiscard]] constexpr bool IsSubresourceRangeInBounds(
    const SubresourceDimensions resources, const SubresourceRangeBounds range) {
    if (resources.levels == 0 || resources.layers == 0 || range.levels == 0 ||
        range.layers == 0) {
        return false;
    }
    if (range.base_level >= resources.levels || range.base_layer >= resources.layers) {
        return false;
    }
    return range.levels <= resources.levels - range.base_level &&
           range.layers <= resources.layers - range.base_layer;
}

} // namespace VideoCore
