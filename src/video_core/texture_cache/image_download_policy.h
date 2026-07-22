// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

[[nodiscard]] constexpr bool ShouldQueueImageDownload(bool readback_enabled, bool is_tiled,
                                                       u32 width) noexcept {
    return readback_enabled && (!is_tiled || width <= 8);
}

} // namespace VideoCore
