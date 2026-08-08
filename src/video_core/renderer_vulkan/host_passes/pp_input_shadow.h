// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>

#include "common/types.h"
#include "video_core/renderer_vulkan/final_guest_surface_content.h"

namespace Vulkan {

enum class FinalGuestSurfaceDeferredScheduler : u8 {
    None,
    Present,
};

struct FinalGuestSurfacePpInputPresentHandoffPlan {
    FinalGuestSurfaceDeferredScheduler scheduler{FinalGuestSurfaceDeferredScheduler::None};
    u32 content_callback_order{};
    u32 calibration_callback_order{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
    bool copy{};
    bool defer_on_draw_scheduler{};
    bool callback_payload_is_scalar_only{};
};

[[nodiscard]] constexpr FinalGuestSurfacePpInputPresentHandoffPlan PlanPpInputShadowPresentHandoff(
    bool enabled, bool reused, bool frame_valid, bool slot_available) noexcept {
    if (!enabled || reused || !frame_valid) {
        return {};
    }
    return {
        .scheduler = FinalGuestSurfaceDeferredScheduler::Present,
        .content_callback_order = 1,
        .calibration_callback_order = 2,
        .status =
            slot_available ? FinalGuestSurfaceStatus::Complete : FinalGuestSurfaceStatus::BusyLoss,
        .copy = slot_available,
        .callback_payload_is_scalar_only = true,
    };
}

struct FinalGuestSurfacePpInputDescriptor {
    bool fsr_enabled{};
    u32 input_width{};
    u32 input_height{};
    u32 output_width{};
    u32 output_height{};
    FinalGuestSurfaceFormat source_format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceFormat output_format{FinalGuestSurfaceFormat::Unsupported};
    u32 gamma_bits{};
    bool pp_hdr{};
    bool frame_hdr{};

    bool operator==(const FinalGuestSurfacePpInputDescriptor&) const = default;
};

struct FinalGuestSurfacePpInputMetadata {
    u32 source_width{};
    u32 source_height{};
    u32 output_width{};
    u32 output_height{};
    FinalGuestSurfaceFormat source_format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceFormat output_format{FinalGuestSurfaceFormat::Unsupported};
    u32 gamma_bits{};
    u64 config_generation{};
    bool hdr{};
    bool fsr_bypassed{};
    bool valid{};

    bool operator==(const FinalGuestSurfacePpInputMetadata&) const = default;
};

class FinalGuestSurfacePpInputConfigTracker {
public:
    [[nodiscard]] FinalGuestSurfacePpInputMetadata Observe(
        FinalGuestSurfacePpInputDescriptor descriptor) noexcept {
        const bool bypassed =
            !descriptor.fsr_enabled || (descriptor.input_width >= descriptor.output_width &&
                                        descriptor.input_height >= descriptor.output_height);
        const bool source_supported =
            descriptor.source_format == FinalGuestSurfaceFormat::Rgba8 ||
            descriptor.source_format == FinalGuestSurfaceFormat::Bgra8 ||
            descriptor.source_format == FinalGuestSurfaceFormat::A2R10G10B10 ||
            descriptor.source_format == FinalGuestSurfaceFormat::A2B10G10R10 ||
            descriptor.source_format == FinalGuestSurfaceFormat::Rgba16;
        const bool output_supported = descriptor.output_format == FinalGuestSurfaceFormat::Rgba8 ||
                                      descriptor.output_format == FinalGuestSurfaceFormat::Bgra8;
        const bool gamma_valid = (descriptor.gamma_bits & 0x8000'0000u) == 0 &&
                                 (descriptor.gamma_bits & 0x7f80'0000u) != 0x7f80'0000u &&
                                 (descriptor.gamma_bits & 0x7fff'ffffu) != 0;
        const bool extent_valid = descriptor.input_width != 0 && descriptor.input_height != 0 &&
                                  descriptor.output_width != 0 && descriptor.output_height != 0;
        if (!bypassed || !source_supported || !output_supported || !gamma_valid || !extent_valid ||
            descriptor.pp_hdr != descriptor.frame_hdr || descriptor.frame_hdr) {
            return {};
        }
        if (!has_descriptor || descriptor != last_descriptor) {
            if (generation == std::numeric_limits<u64>::max()) {
                return {};
            }
            last_descriptor = descriptor;
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
            .gamma_bits = descriptor.gamma_bits,
            .config_generation = generation,
            .hdr = descriptor.frame_hdr,
            .fsr_bypassed = true,
            .valid = true,
        };
    }

private:
    FinalGuestSurfacePpInputDescriptor last_descriptor{};
    u64 generation{};
    bool has_descriptor{};
};

struct FinalGuestSurfacePpInputPayload {
    u64 sequence{};
    u64 process_time_us{};
    u64 token{};
    FinalGuestSurfacePpInputMetadata metadata{};
};

struct FinalGuestSurfacePpInputTakeResult {
    FinalGuestSurfacePpInputPayload payload{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    bool emit{};
};

class FinalGuestSurfacePpInputFrameState {
public:
    [[nodiscard]] FinalGuestSurfaceStatus Assign(FinalGuestSurfacePpInputPayload next) noexcept {
        if (pending || poisoned) {
            payload = {};
            pending = false;
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

    [[nodiscard]] FinalGuestSurfacePpInputTakeResult TakeForPresent(bool reused) noexcept {
        if (poisoned) {
            Clear();
            return {.status = FinalGuestSurfaceStatus::GapLoss};
        }
        if (!pending) {
            return {};
        }
        if (reused) {
            payload = {};
            pending = false;
            return {.status = FinalGuestSurfaceStatus::GapLoss};
        }
        const auto result = FinalGuestSurfacePpInputTakeResult{
            .payload = payload,
            .status = FinalGuestSurfaceStatus::Complete,
            .emit = true,
        };
        payload = {};
        pending = false;
        return result;
    }

    void Clear() noexcept {
        payload = {};
        pending = false;
        poisoned = false;
    }

private:
    FinalGuestSurfacePpInputPayload payload{};
    bool pending{};
    bool poisoned{};
};

struct FinalGuestSurfacePpInputTransferPlan {
    u32 color_write_to_transfer_barriers{};
    u32 copy_regions{};
    bool copy{};
    bool cpu_wait{};
    bool finish{};
    bool callback_retains_frame{};
    bool callback_retains_image{};
    bool callback_retains_vk_image{};
};

[[nodiscard]] constexpr FinalGuestSurfacePpInputTransferPlan PlanPpInputShadowTransfer(
    bool enabled, bool reused, bool frame_valid, bool metadata_valid) noexcept {
    if (!enabled || reused || !frame_valid || !metadata_valid) {
        return {};
    }
    return {
        .color_write_to_transfer_barriers = 1,
        .copy_regions = 1,
        .copy = true,
    };
}

} // namespace Vulkan

namespace Vulkan::HostPasses {

enum class PpPipelineSelection : u8 {
    Normal,
    ShadowDualOutput,
};

struct PostProcessingInvocationPlan {
    std::array<FinalGuestSurfaceFormat, 2> attachment_formats{};
    PpPipelineSelection pipeline{PpPipelineSelection::Normal};
    u32 fragment_output_count{1};
    u32 color_attachment_count{1};
    u32 draw_count{1};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
    bool draw_normal_output{true};
    bool draw_shadow_output{};
    bool identical_computed_color{};
};

[[nodiscard]] constexpr PostProcessingInvocationPlan PlanPostProcessingInvocation(
    bool shadow_enabled, bool shadow_available, FinalGuestSurfaceFormat normal_format,
    FinalGuestSurfaceFormat shadow_format) noexcept {
    PostProcessingInvocationPlan plan{.attachment_formats = {normal_format}};
    if (!shadow_enabled) {
        return plan;
    }
    if (!shadow_available) {
        plan.status = FinalGuestSurfaceStatus::InvalidationLoss;
        return plan;
    }
    if (normal_format == FinalGuestSurfaceFormat::Unsupported || normal_format != shadow_format) {
        plan.status = FinalGuestSurfaceStatus::Unsupported;
        return plan;
    }
    plan.attachment_formats[1] = shadow_format;
    plan.pipeline = PpPipelineSelection::ShadowDualOutput;
    plan.fragment_output_count = 2;
    plan.color_attachment_count = 2;
    plan.draw_shadow_output = true;
    plan.identical_computed_color = true;
    return plan;
}

struct PpInputShadowPlan {
    u32 shadow_pipeline_count{};
    u32 shadow_shader_count{};
    u32 shadow_image_count{};
    u32 shadow_copy_count{};
    u32 shadow_fragment_output_count{};
    u32 shadow_color_attachment_count{};
    u32 draw_count{};
    bool identical_computed_color{};
    bool same_format_required{};
    bool requires_cpu_wait{};
};

[[nodiscard]] constexpr PpInputShadowPlan PlanPpInputShadow(bool enabled,
                                                            u32 frame_count) noexcept {
    if (!enabled) {
        return {};
    }
    return {
        .shadow_pipeline_count = 1,
        .shadow_shader_count = 1,
        .shadow_image_count = frame_count,
        .shadow_copy_count = 1,
        .shadow_fragment_output_count = 2,
        .shadow_color_attachment_count = 2,
        .draw_count = 1,
        .identical_computed_color = true,
        .same_format_required = true,
    };
}

} // namespace Vulkan::HostPasses
