// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

// A timeout means work is still pending. Every other result must reach the caller's
// normal error handling; retrying a terminal failure can otherwise hide it forever.
template <typename Operation>
[[nodiscard]] vk::Result RetryOnTimeout(Operation&& operation) {
    vk::Result result;
    do {
        result = operation();
    } while (result == vk::Result::eTimeout);
    return result;
}

} // namespace Vulkan
