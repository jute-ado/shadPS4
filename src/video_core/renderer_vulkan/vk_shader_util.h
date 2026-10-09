// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

inline constexpr size_t MaxCachedShaderDwords = 1U << 20;

[[nodiscard]] inline bool IsValidCachedSpirv(std::span<const u32> code) noexcept {
    constexpr u32 SpirvMagic = 0x07230203U;
    constexpr u32 MinimumVersion = 0x00010000U;
    constexpr u32 MaximumVersion = 0x00010600U;
    if (code.size() <= 5 || code.size() > MaxCachedShaderDwords || code[0] != SpirvMagic ||
        code[1] < MinimumVersion || code[1] > MaximumVersion || (code[1] & 0xFFU) != 0 ||
        code[3] == 0 || code[3] > MaxCachedShaderDwords || code[4] != 0) {
        return false;
    }
    for (size_t offset = 5; offset < code.size();) {
        const size_t word_count = code[offset] >> 16;
        if (word_count == 0 || word_count > code.size() - offset) {
            return false;
        }
        offset += word_count;
    }
    return true;
}

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
