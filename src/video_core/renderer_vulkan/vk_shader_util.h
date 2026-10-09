// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

inline constexpr size_t MaxCachedShaderDwords = 1U << 20;

/**
 * Performs a bounded structural and semantic preflight for the SPIR-V dialect emitted by Sirit.
 * Cache modules outside that known dialect fail closed instead of reaching the Vulkan driver.
 */
[[nodiscard]] bool IsValidCachedSpirv(std::span<const u32> code) noexcept;

/**
 * @brief Creates a vulkan shader module from GLSL by converting it to SPIR-V using glslang.
 * @param code The string containing GLSL code.
 * @param stage The pipeline stage the shader will be used in.
 * @param device The vulkan device handle.
 */
vk::ShaderModule Compile(std::string_view code, vk::ShaderStageFlagBits stage, vk::Device device,
                         std::vector<std::string> defines = {});

/**
 * @brief Creates a vulkan shader module from SPIR-V bytecode.
 * @param code The SPIR-V bytecode data.
 * @param device The vulkan device handle
 */
vk::ShaderModule CompileSPV(std::span<const u32> code, vk::Device device);

/** Creates a module for untrusted cached SPIR-V without asserting on validation/device failure. */
[[nodiscard]] std::optional<vk::ShaderModule> TryCompileSPV(std::span<const u32> code,
                                                            vk::Device device) noexcept;

} // namespace Vulkan
