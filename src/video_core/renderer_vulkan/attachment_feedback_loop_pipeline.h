// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Vulkan {

struct AttachmentFeedbackLoopPipelinePlan {
    bool declare_dynamic_state{};
    bool declare_static_color_feedback{};
};

[[nodiscard]] constexpr AttachmentFeedbackLoopPipelinePlan PlanAttachmentFeedbackLoopPipeline(
    bool dynamic_state_supported) noexcept {
    return {
        .declare_dynamic_state = dynamic_state_supported,
        .declare_static_color_feedback = false,
    };
}

} // namespace Vulkan
