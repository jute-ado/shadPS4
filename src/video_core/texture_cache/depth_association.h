// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

// Preserve the current redirect behavior while the format invariant is under test.
constexpr bool ShouldUseAssociatedDepthForView(vk::Format) {
    return true;
}

} // namespace VideoCore
