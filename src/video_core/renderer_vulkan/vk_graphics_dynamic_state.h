// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

template <typename DynamicStates>
void AppendAttachmentFeedbackLoopDynamicState(DynamicStates& states, bool supported) {
    if (supported) {
        states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
    }
}

} // namespace Vulkan
