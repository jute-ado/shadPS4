// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"

namespace Serialization {
struct Archive;
}

namespace Vulkan {

inline constexpr u32 PipelineKeyVersion = 6u;

[[nodiscard]] bool ValidateGraphicsPipelineKey(const GraphicsPipelineKey& key) noexcept;

/**
 * Decodes the complete graphics portion of a cached pipeline-key record.
 * Outputs are published only after the key, bounded support data and exact end-of-stream pass.
 */
[[nodiscard]] bool DeserializeGraphicsPipelineCacheRecord(
    Serialization::Archive& ar, GraphicsPipelineKey& key,
    GraphicsPipeline::SerializationSupport& support);

} // namespace Vulkan
