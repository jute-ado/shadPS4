// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bit>
#include <cmath>
#include <limits>

#include "common/types.h"
#include "video_core/renderer_vulkan/final_guest_surface_content.h"

namespace Vulkan::HostPasses {

enum class PpDiagnosticMode : u8 {
    None,
    ComputedShadow,
    SampledInput,
};

} // namespace Vulkan::HostPasses

namespace Vulkan {

[[nodiscard]] constexpr HostPasses::PpDiagnosticMode PpDiagnosticModeForStage(
    FinalGuestSurfaceStage stage) noexcept {
    switch (stage) {
    case FinalGuestSurfaceStage::PpInputShadow:
        return HostPasses::PpDiagnosticMode::ComputedShadow;
    case FinalGuestSurfaceStage::PpSampledInput:
        return HostPasses::PpDiagnosticMode::SampledInput;
    default:
        return HostPasses::PpDiagnosticMode::None;
    }
}

struct PpSampledInputSourceViewDescriptor {
    u32 resolved_base_mip{};
    u32 resolved_mip_count{};
    u32 resolved_base_layer{};
    u32 resolved_layer_count{};
    FinalGuestSurfaceFormat requested_format{FinalGuestSurfaceFormat::Unsupported};
    bool force_alpha_one{};
};

struct PpSampledInputSourceViewPlan {
    u32 base_mip{};
    u32 mip_count{};
    u32 base_layer{};
    u32 layer_count{};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
    bool force_alpha_one{};
};

[[nodiscard]] constexpr PpSampledInputSourceViewPlan PlanPpSampledInputSourceView(
    PpSampledInputSourceViewDescriptor descriptor) noexcept {
    if (descriptor.resolved_mip_count != 1 || descriptor.resolved_layer_count != 1 ||
        descriptor.requested_format == FinalGuestSurfaceFormat::Unsupported) {
        return {.status = FinalGuestSurfaceStatus::Unsupported};
    }
    return {
        .base_mip = descriptor.resolved_base_mip,
        .mip_count = descriptor.resolved_mip_count,
        .base_layer = descriptor.resolved_base_layer,
        .layer_count = descriptor.resolved_layer_count,
        .format = descriptor.requested_format,
        .force_alpha_one = descriptor.force_alpha_one,
    };
}

struct FinalGuestSurfaceSampledInputDescriptor {
    bool fsr_enabled{};
    u32 input_width{};
    u32 input_height{};
    u32 output_width{};
    u32 output_height{};
    FinalGuestSurfaceFormat source_format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceFormat output_format{FinalGuestSurfaceFormat::Unsupported};
    u32 resolved_base_mip{};
    u32 resolved_mip_count{};
    u32 resolved_base_layer{};
    u32 resolved_layer_count{};
    u64 source_image_uid{};
    u64 source_backing_generation{};
    u32 gamma_bits{};
    bool source_view_srgb{};
    bool settings_snapshot_matches_push{};
    bool pp_hdr{};
    bool frame_hdr{};
};

struct FinalGuestSurfaceSampledInputMetadata {
    u32 source_width{};
    u32 source_height{};
    u32 output_width{};
    u32 output_height{};
    FinalGuestSurfaceFormat source_format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceFormat output_format{FinalGuestSurfaceFormat::Unsupported};
    u32 resolved_base_mip{};
    u32 resolved_mip_count{};
    u32 resolved_base_layer{};
    u32 resolved_layer_count{};
    u64 source_image_uid{};
    u64 source_backing_generation{};
    u32 gamma_bits{};
    u64 config_generation{};
    bool source_view_srgb{};
    bool settings_snapshot_matches_push{};
    bool fsr_bypassed{};
    bool valid{};

    bool operator==(const FinalGuestSurfaceSampledInputMetadata&) const = default;
};

class FinalGuestSurfaceSampledInputConfigTracker {
public:
    [[nodiscard]] FinalGuestSurfaceSampledInputMetadata Observe(
        const FinalGuestSurfaceSampledInputDescriptor& descriptor) noexcept {
        const bool bypassed =
            !descriptor.fsr_enabled || (descriptor.input_width >= descriptor.output_width &&
                                        descriptor.input_height >= descriptor.output_height);
        const bool format_supported =
            (descriptor.source_format == FinalGuestSurfaceFormat::Rgba8 ||
             descriptor.source_format == FinalGuestSurfaceFormat::Bgra8) &&
            (descriptor.output_format == FinalGuestSurfaceFormat::Rgba8 ||
             descriptor.output_format == FinalGuestSurfaceFormat::Bgra8);
        const float gamma = std::bit_cast<float>(descriptor.gamma_bits);
        const bool valid =
            bypassed && format_supported && descriptor.input_width != 0 &&
            descriptor.input_height != 0 && descriptor.output_width != 0 &&
            descriptor.output_height != 0 && descriptor.resolved_mip_count == 1 &&
            descriptor.resolved_layer_count == 1 && descriptor.source_image_uid != 0 &&
            descriptor.source_backing_generation != 0 && descriptor.source_view_srgb &&
            descriptor.settings_snapshot_matches_push &&
            descriptor.pp_hdr == descriptor.frame_hdr && !descriptor.frame_hdr &&
            std::isfinite(gamma) && gamma >= 0.1f && gamma <= 2.0f;
        if (!valid) {
            return {};
        }
        const LogicalDescriptor logical{
            .input_width = descriptor.input_width,
            .input_height = descriptor.input_height,
            .output_width = descriptor.output_width,
            .output_height = descriptor.output_height,
            .source_format = descriptor.source_format,
            .output_format = descriptor.output_format,
            .resolved_base_mip = descriptor.resolved_base_mip,
            .resolved_mip_count = descriptor.resolved_mip_count,
            .resolved_base_layer = descriptor.resolved_base_layer,
            .resolved_layer_count = descriptor.resolved_layer_count,
            .gamma_bits = descriptor.gamma_bits,
            .source_view_srgb = descriptor.source_view_srgb,
        };
        if (!has_descriptor || logical != last_descriptor) {
            if (generation == std::numeric_limits<u64>::max()) {
                return {};
            }
            last_descriptor = logical;
            has_descriptor = true;
            ++generation;
        }
        return {
            .source_width = descriptor.input_width,
            .source_height = descriptor.input_height,
            .output_width = descriptor.output_width,
            .output_height = descriptor.output_height,
            .source_format = descriptor.source_format,
            .output_format = descriptor.output_format,
            .resolved_base_mip = descriptor.resolved_base_mip,
            .resolved_mip_count = descriptor.resolved_mip_count,
            .resolved_base_layer = descriptor.resolved_base_layer,
            .resolved_layer_count = descriptor.resolved_layer_count,
            .source_image_uid = descriptor.source_image_uid,
            .source_backing_generation = descriptor.source_backing_generation,
            .gamma_bits = descriptor.gamma_bits,
            .config_generation = generation,
            .source_view_srgb = descriptor.source_view_srgb,
            .settings_snapshot_matches_push = descriptor.settings_snapshot_matches_push,
            .fsr_bypassed = true,
            .valid = true,
        };
    }

private:
    struct LogicalDescriptor {
        u32 input_width{};
        u32 input_height{};
        u32 output_width{};
        u32 output_height{};
        FinalGuestSurfaceFormat source_format{FinalGuestSurfaceFormat::Unsupported};
        FinalGuestSurfaceFormat output_format{FinalGuestSurfaceFormat::Unsupported};
        u32 resolved_base_mip{};
        u32 resolved_mip_count{};
        u32 resolved_base_layer{};
        u32 resolved_layer_count{};
        u32 gamma_bits{};
        bool source_view_srgb{};

        bool operator==(const LogicalDescriptor&) const = default;
    };

    LogicalDescriptor last_descriptor{};
    u64 generation{};
    bool has_descriptor{};
};

[[nodiscard]] constexpr bool ShouldAssignPpSampledInputFrame(bool enabled, bool stamp_valid,
                                                             bool metadata_valid) noexcept {
    return enabled && stamp_valid && metadata_valid;
}

struct FinalGuestSurfaceSampledInputPayload {
    u64 sequence{};
    u64 process_time_us{};
    u64 token{};
    FinalGuestSurfaceSampledInputMetadata metadata{};
};

struct FinalGuestSurfaceSampledInputTakeResult {
    FinalGuestSurfaceSampledInputPayload payload{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    bool emit{};
};

class FinalGuestSurfaceSampledInputFrameState {
public:
    [[nodiscard]] FinalGuestSurfaceStatus AssignIfValid(
        bool eligible, FinalGuestSurfaceSampledInputPayload next) noexcept {
        if (!eligible) {
            return FinalGuestSurfaceStatus::AlreadyConsumed;
        }
        if (pending || poisoned) {
            ClearPending();
            poisoned = true;
            return FinalGuestSurfaceStatus::GapLoss;
        }
        if (next.sequence == 0 || next.process_time_us == 0 || next.token == 0 ||
            !next.metadata.valid) {
            poisoned = true;
            return FinalGuestSurfaceStatus::InvalidationLoss;
        }
        payload = next;
        pending = true;
        return FinalGuestSurfaceStatus::Complete;
    }

    [[nodiscard]] FinalGuestSurfaceSampledInputTakeResult TakeForPresent(bool reused) noexcept {
        if (poisoned) {
            Clear();
            return {.status = FinalGuestSurfaceStatus::GapLoss};
        }
        if (!pending) {
            return {};
        }
        if (reused) {
            ClearPending();
            return {.status = FinalGuestSurfaceStatus::GapLoss};
        }
        const auto result = FinalGuestSurfaceSampledInputTakeResult{
            .payload = payload,
            .status = FinalGuestSurfaceStatus::Complete,
            .emit = true,
        };
        ClearPending();
        return result;
    }

    void Clear() noexcept {
        ClearPending();
        poisoned = false;
    }

private:
    void ClearPending() noexcept {
        payload = {};
        pending = false;
    }

    FinalGuestSurfaceSampledInputPayload payload{};
    bool pending{};
    bool poisoned{};
};

struct PpSampledInputTransferPlan {
    u32 color_write_to_transfer_barriers{};
    u32 copy_regions{};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    bool copy{};
    bool callback_payload_is_scalar_only{};
    bool cpu_wait{};
    bool finish{};
    bool callback_retains_frame{};
    bool callback_retains_image{};
    bool callback_retains_vk_image{};
};

[[nodiscard]] constexpr PpSampledInputTransferPlan PlanPpSampledInputTransfer(
    bool enabled, bool reused, bool frame_valid, bool metadata_valid) noexcept {
    if (!enabled || reused || !frame_valid || !metadata_valid) {
        return {};
    }
    return {
        .color_write_to_transfer_barriers = 1,
        .copy_regions = 1,
        .format = FinalGuestSurfaceFormat::Rgba16Float,
        .copy = true,
        .callback_payload_is_scalar_only = true,
    };
}

} // namespace Vulkan

namespace Vulkan::HostPasses {

struct PpSampledInputPlan {
    u32 pipeline_count{};
    u32 shader_count{};
    u32 float_image_count{};
    u32 copy_count{};
    u32 fragment_output_count{};
    u32 color_attachment_count{};
    u32 draw_count{};
    u32 texture_sample_count{};
    FinalGuestSurfaceFormat diagnostic_format{FinalGuestSurfaceFormat::Unsupported};
    bool normal_output_is_computed_color{};
    bool diagnostic_output_is_raw_linear_sample{};
    bool cpu_wait{};
    bool finish{};
};

[[nodiscard]] constexpr PpSampledInputPlan PlanPpSampledInput(bool enabled,
                                                              u32 frame_count) noexcept {
    if (!enabled) {
        return {};
    }
    return {
        .pipeline_count = 1,
        .shader_count = 1,
        .float_image_count = frame_count,
        .copy_count = 1,
        .fragment_output_count = 2,
        .color_attachment_count = 2,
        .draw_count = 1,
        .texture_sample_count = 1,
        .diagnostic_format = FinalGuestSurfaceFormat::Rgba16Float,
        .normal_output_is_computed_color = true,
        .diagnostic_output_is_raw_linear_sample = true,
    };
}

struct PpSampledInputInvocationPlan {
    std::array<FinalGuestSurfaceFormat, 2> attachment_formats{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
    u32 color_attachment_count{1};
    u32 draw_count{1};
    bool raw_sample_output{};
};

[[nodiscard]] constexpr PpSampledInputInvocationPlan PlanPpSampledInputInvocation(
    bool enabled, bool available, FinalGuestSurfaceFormat normal_format,
    FinalGuestSurfaceFormat sampled_format) noexcept {
    PpSampledInputInvocationPlan plan{.attachment_formats = {normal_format}};
    if (!enabled) {
        return plan;
    }
    if (!available) {
        plan.status = FinalGuestSurfaceStatus::InvalidationLoss;
        return plan;
    }
    if ((normal_format != FinalGuestSurfaceFormat::Rgba8 &&
         normal_format != FinalGuestSurfaceFormat::Bgra8) ||
        sampled_format != FinalGuestSurfaceFormat::Rgba16Float) {
        plan.status = FinalGuestSurfaceStatus::Unsupported;
        return plan;
    }
    plan.attachment_formats[1] = sampled_format;
    plan.color_attachment_count = 2;
    plan.raw_sample_output = true;
    return plan;
}

} // namespace Vulkan::HostPasses
