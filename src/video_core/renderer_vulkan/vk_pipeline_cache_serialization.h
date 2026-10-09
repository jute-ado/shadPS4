// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Profile;
}

namespace Vulkan {

inline constexpr u32 PipelineKeyVersion = 6u;
inline constexpr u32 ShaderProfileVersion = 1u;

void SerializeShaderProfile(Serialization::Archive& ar, const Shader::Profile& profile);

/** Decodes a complete canonical profile record without publishing partial state. */
[[nodiscard]] bool DeserializeShaderProfile(Serialization::Archive& ar, Shader::Profile& profile);

[[nodiscard]] bool ValidateGraphicsPipelineKey(const GraphicsPipelineKey& key) noexcept;

/**
 * Decodes the complete graphics portion of a cached pipeline-key record.
 * Outputs are published only after the key, bounded support data and exact end-of-stream pass.
 */
[[nodiscard]] bool DeserializeGraphicsPipelineCacheRecord(
    Serialization::Archive& ar, GraphicsPipelineKey& key,
    GraphicsPipeline::SerializationSupport& support);

} // namespace Vulkan
