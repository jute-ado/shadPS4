// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>
#include <vulkan/vulkan_core.h>

namespace Vulkan {

[[nodiscard]] constexpr bool ParseAsyncGraphicsDisableOptimization(
    std::string_view value) noexcept {
    return value == "1";
}

[[nodiscard]] constexpr VkPipelineCreateFlags AsyncGraphicsDiagnosticFlags(
    bool asynchronous, bool disable_optimization) noexcept {
    return asynchronous && disable_optimization
               ? VkPipelineCreateFlags{VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT}
               : VkPipelineCreateFlags{};
}

} // namespace Vulkan
