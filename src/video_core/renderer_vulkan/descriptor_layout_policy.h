// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <optional>
#include <span>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

enum class DescriptorLayoutFailureReason {
    MalformedBinding,
    UnsupportedType,
    PerStageType,
    PipelineType,
    PerStageResources,
    FragmentOutputs,
    PoolType,
    PoolSets,
};

struct DescriptorLayoutFailure {
    DescriptorLayoutFailureReason reason;
    u64 required{};
    u64 limit{};
};

// Validates the ordinary, single-set graphics layouts emitted by this renderer, not UAB
// layouts or every possible Vulkan descriptor type. Immutable samplers have the same cost.
[[nodiscard]] inline std::optional<DescriptorLayoutFailure> ValidateGraphicsDescriptorLayout(
    std::span<const vk::DescriptorSetLayoutBinding> bindings,
    const vk::PhysicalDeviceLimits& limits, std::span<const vk::DescriptorPoolSize> pool_sizes,
    u32 pool_max_sets, u32 batch_size, bool push_descriptors, u32 framebuffer_colors,
    u32 fragment_output_locations) {
    constexpr std::array types{vk::DescriptorType::eUniformBuffer,
                               vk::DescriptorType::eStorageBuffer,
                               vk::DescriptorType::eSampledImage, vk::DescriptorType::eStorageImage,
                               vk::DescriptorType::eSampler};
    constexpr std::array stages{
        vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eTessellationControl,
        vk::ShaderStageFlagBits::eTessellationEvaluation, vk::ShaderStageFlagBits::eGeometry,
        vk::ShaderStageFlagBits::eFragment};
    constexpr vk::ShaderStageFlags supported_stages =
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eTessellationControl |
        vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eGeometry |
        vk::ShaderStageFlagBits::eFragment;
    const std::array stage_limits{
        limits.maxPerStageDescriptorUniformBuffers, limits.maxPerStageDescriptorStorageBuffers,
        limits.maxPerStageDescriptorSampledImages, limits.maxPerStageDescriptorStorageImages,
        limits.maxPerStageDescriptorSamplers};
    const std::array pipeline_limits{
        limits.maxDescriptorSetUniformBuffers, limits.maxDescriptorSetStorageBuffers,
        limits.maxDescriptorSetSampledImages, limits.maxDescriptorSetStorageImages,
        limits.maxDescriptorSetSamplers};
    using Counts = std::array<u64, types.size()>;
    Counts pipeline{};
    std::array<Counts, stages.size()> per_stage{};
    Counts pool{};
    if (limits.maxBoundDescriptorSets < 1U) {
        return DescriptorLayoutFailure{DescriptorLayoutFailureReason::PipelineType, 1U,
                                       limits.maxBoundDescriptorSets};
    }
    for (size_t i = 0; i < bindings.size(); ++i) {
        const auto& binding = bindings[i];
        // Our renderer emits dense, increasing binding numbers. Also catches duplicates before
        // passing a malformed layout to Vulkan; descriptor arrays consume one binding each.
        if (binding.binding != i || binding.descriptorCount == 0U || !binding.stageFlags ||
            (binding.stageFlags & ~supported_stages)) {
            return DescriptorLayoutFailure{DescriptorLayoutFailureReason::MalformedBinding};
        }
        size_t type{};
        while (type < types.size() && types[type] != binding.descriptorType) {
            ++type;
        }
        if (type == types.size()) {
            // No implicit accounting for combined samplers, texel buffers or future descriptor
            // types. The current layout builder emits only these five ordinary descriptor types.
            return DescriptorLayoutFailure{DescriptorLayoutFailureReason::UnsupportedType};
        }
        pipeline[type] += binding.descriptorCount;
        for (size_t stage = 0; stage < stages.size(); ++stage) {
            if (binding.stageFlags & stages[stage]) {
                per_stage[stage][type] += binding.descriptorCount;
            }
        }
    }
    for (size_t type = 0; type < types.size(); ++type) {
        if (pipeline[type] > pipeline_limits[type]) {
            return DescriptorLayoutFailure{DescriptorLayoutFailureReason::PipelineType,
                                           pipeline[type], pipeline_limits[type]};
        }
        for (const auto& stage : per_stage) {
            if (stage[type] > stage_limits[type]) {
                return DescriptorLayoutFailure{DescriptorLayoutFailureReason::PerStageType,
                                               stage[type], stage_limits[type]};
            }
        }
    }
    for (size_t stage = 0; stage < stages.size(); ++stage) {
        const auto& counts = per_stage[stage];
        // Standalone samplers have their own limits, but are not maxPerStageResources.
        const u64 resources =
            counts[0] + counts[1] + counts[2] + counts[3] +
            (stages[stage] == vk::ShaderStageFlagBits::eFragment ? framebuffer_colors : 0U);
        if (resources > limits.maxPerStageResources) {
            return DescriptorLayoutFailure{DescriptorLayoutFailureReason::PerStageResources,
                                           resources, limits.maxPerStageResources};
        }
    }
    const auto& fragment = per_stage.back();
    const u64 outputs = fragment[1] + fragment[3] + fragment_output_locations;
    if (outputs > limits.maxFragmentCombinedOutputResources) {
        return DescriptorLayoutFailure{DescriptorLayoutFailureReason::FragmentOutputs, outputs,
                                       limits.maxFragmentCombinedOutputResources};
    }
    if (!push_descriptors) {
        if (batch_size == 0U || batch_size > pool_max_sets) {
            return DescriptorLayoutFailure{DescriptorLayoutFailureReason::PoolSets, batch_size,
                                           pool_max_sets};
        }
        for (const auto& entry : pool_sizes) {
            for (size_t type = 0; type < types.size(); ++type) {
                if (entry.type == types[type]) {
                    pool[type] += entry.descriptorCount;
                }
            }
        }
        for (size_t type = 0; type < types.size(); ++type) {
            // Use division: neither multiplying a full descriptor count by a caller's batch
            // size nor summing array counts in U32 may wrap and admit an impossible allocation.
            if (pipeline[type] > pool[type] / batch_size) {
                return DescriptorLayoutFailure{DescriptorLayoutFailureReason::PoolType,
                                               pipeline[type], pool[type] / batch_size};
            }
        }
    }
    return std::nullopt;
}

} // namespace Vulkan
