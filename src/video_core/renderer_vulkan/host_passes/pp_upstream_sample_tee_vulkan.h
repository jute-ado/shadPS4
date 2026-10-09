// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/host_passes/pp_upstream_sample_tee.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan::HostPasses {

struct PpUpstreamSampleTeeVulkanBarrier {
    vk::PipelineStageFlags2 src_stages{};
    vk::AccessFlags2 src_access{};
    vk::PipelineStageFlags2 dst_stages{};
    vk::AccessFlags2 dst_access{};
    u32 offset{};
    u32 bytes{};
    bool valid{};
};

struct PpUpstreamSampleTeeVulkanPushStages {
    vk::ShaderStageFlags stages{};
    bool valid{};
};

namespace Detail {

[[nodiscard]] constexpr bool HasPpUpstreamSampleTeeStage(
    PpUpstreamSampleTeeStage value, PpUpstreamSampleTeeStage bit) noexcept {
    return (static_cast<u32>(value) & static_cast<u32>(bit)) != 0;
}

[[nodiscard]] constexpr bool HasPpUpstreamSampleTeeAccess(
    PpUpstreamSampleTeeAccess value, PpUpstreamSampleTeeAccess bit) noexcept {
    return (static_cast<u32>(value) & static_cast<u32>(bit)) != 0;
}

[[nodiscard]] constexpr bool IsKnownPpUpstreamSampleTeeStages(
    PpUpstreamSampleTeeStage value) noexcept {
    constexpr u32 Known = static_cast<u32>(PpUpstreamSampleTeeStage::AllCommands) |
                          static_cast<u32>(PpUpstreamSampleTeeStage::Host) |
                          static_cast<u32>(PpUpstreamSampleTeeStage::Transfer) |
                          static_cast<u32>(PpUpstreamSampleTeeStage::FragmentShader);
    return (static_cast<u32>(value) & ~Known) == 0;
}

[[nodiscard]] constexpr bool IsKnownPpUpstreamSampleTeeAccess(
    PpUpstreamSampleTeeAccess value) noexcept {
    constexpr u32 Known = static_cast<u32>(PpUpstreamSampleTeeAccess::MemoryRead) |
                          static_cast<u32>(PpUpstreamSampleTeeAccess::MemoryWrite) |
                          static_cast<u32>(PpUpstreamSampleTeeAccess::HostRead) |
                          static_cast<u32>(PpUpstreamSampleTeeAccess::TransferWrite) |
                          static_cast<u32>(PpUpstreamSampleTeeAccess::ShaderRead) |
                          static_cast<u32>(PpUpstreamSampleTeeAccess::ShaderWrite);
    return (static_cast<u32>(value) & ~Known) == 0;
}

[[nodiscard]] constexpr vk::PipelineStageFlags2 MapPpUpstreamSampleTeeStages(
    PpUpstreamSampleTeeStage value) noexcept {
    vk::PipelineStageFlags2 result{};
    if (HasPpUpstreamSampleTeeStage(value, PpUpstreamSampleTeeStage::AllCommands)) {
        result |= vk::PipelineStageFlagBits2::eAllCommands;
    }
    if (HasPpUpstreamSampleTeeStage(value, PpUpstreamSampleTeeStage::Host)) {
        result |= vk::PipelineStageFlagBits2::eHost;
    }
    if (HasPpUpstreamSampleTeeStage(value, PpUpstreamSampleTeeStage::Transfer)) {
        result |= vk::PipelineStageFlagBits2::eTransfer;
    }
    if (HasPpUpstreamSampleTeeStage(value, PpUpstreamSampleTeeStage::FragmentShader)) {
        result |= vk::PipelineStageFlagBits2::eFragmentShader;
    }
    return result;
}

[[nodiscard]] constexpr vk::AccessFlags2 MapPpUpstreamSampleTeeAccess(
    PpUpstreamSampleTeeAccess value) noexcept {
    vk::AccessFlags2 result{};
    if (HasPpUpstreamSampleTeeAccess(value, PpUpstreamSampleTeeAccess::MemoryRead)) {
        result |= vk::AccessFlagBits2::eMemoryRead;
    }
    if (HasPpUpstreamSampleTeeAccess(value, PpUpstreamSampleTeeAccess::MemoryWrite)) {
        result |= vk::AccessFlagBits2::eMemoryWrite;
    }
    if (HasPpUpstreamSampleTeeAccess(value, PpUpstreamSampleTeeAccess::HostRead)) {
        result |= vk::AccessFlagBits2::eHostRead;
    }
    if (HasPpUpstreamSampleTeeAccess(value, PpUpstreamSampleTeeAccess::TransferWrite)) {
        result |= vk::AccessFlagBits2::eTransferWrite;
    }
    if (HasPpUpstreamSampleTeeAccess(value, PpUpstreamSampleTeeAccess::ShaderRead)) {
        result |= vk::AccessFlagBits2::eShaderRead;
    }
    if (HasPpUpstreamSampleTeeAccess(value, PpUpstreamSampleTeeAccess::ShaderWrite)) {
        result |= vk::AccessFlagBits2::eShaderWrite;
    }
    return result;
}

} // namespace Detail

[[nodiscard]] constexpr PpUpstreamSampleTeeVulkanBarrier
MapPpUpstreamSampleTeeBarrierToVulkan(const PpUpstreamSampleTeeBarrierPlan& plan) noexcept {
    if (plan.src_stages == PpUpstreamSampleTeeStage::None &&
        plan.src_access == PpUpstreamSampleTeeAccess::None &&
        plan.dst_stages == PpUpstreamSampleTeeStage::None &&
        plan.dst_access == PpUpstreamSampleTeeAccess::None) {
        return {};
    }
    if (!Detail::IsKnownPpUpstreamSampleTeeStages(plan.src_stages) ||
        !Detail::IsKnownPpUpstreamSampleTeeAccess(plan.src_access) ||
        !Detail::IsKnownPpUpstreamSampleTeeStages(plan.dst_stages) ||
        !Detail::IsKnownPpUpstreamSampleTeeAccess(plan.dst_access)) {
        return {};
    }
    return {
        .src_stages = Detail::MapPpUpstreamSampleTeeStages(plan.src_stages),
        .src_access = Detail::MapPpUpstreamSampleTeeAccess(plan.src_access),
        .dst_stages = Detail::MapPpUpstreamSampleTeeStages(plan.dst_stages),
        .dst_access = Detail::MapPpUpstreamSampleTeeAccess(plan.dst_access),
        .offset = plan.offset,
        .bytes = plan.bytes,
        .valid = true,
    };
}

[[nodiscard]] constexpr PpUpstreamSampleTeeVulkanPushStages
MapPpUpstreamSampleTeePushStagesToVulkan(PpUpstreamSampleTeeStage stages) noexcept {
    if (stages != PpUpstreamSampleTeeStage::FragmentShader) {
        return {};
    }
    return {
        .stages = vk::ShaderStageFlagBits::eFragment,
        .valid = true,
    };
}

} // namespace Vulkan::HostPasses
