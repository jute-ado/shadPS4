// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bit>
#include <cmath>
#include <limits>

#include "common/types.h"
#include "video_core/renderer_vulkan/final_guest_surface_content.h"
#include "video_core/renderer_vulkan/host_passes/pp_source_backing.h"

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
    case FinalGuestSurfaceStage::PpSourceReconstruction:
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
    u32 bound_base_mip{};
    u32 bound_mip_count{};
    u32 bound_base_layer{};
    u32 bound_layer_count{};
};

struct PpSampledInputSourceViewAssessment {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
    bool same_view_as_baseline{true};
    bool diagnostic_changes_bound_view{};
    bool resolved_range_mismatch{};
};

[[nodiscard]] constexpr PpSampledInputSourceViewAssessment AssessPpSampledInputSourceView(
    PpSampledInputSourceViewDescriptor descriptor) noexcept {
    const bool valid = descriptor.resolved_mip_count == 1 && descriptor.resolved_layer_count == 1 &&
                       descriptor.bound_mip_count == 1 && descriptor.bound_layer_count == 1;
    if (!valid) {
        return {.status = FinalGuestSurfaceStatus::InvalidationLoss};
    }
    const bool mismatch = descriptor.resolved_base_mip != descriptor.bound_base_mip ||
                          descriptor.resolved_mip_count != descriptor.bound_mip_count ||
                          descriptor.resolved_base_layer != descriptor.bound_base_layer ||
                          descriptor.resolved_layer_count != descriptor.bound_layer_count;
    return {
        .status = mismatch ? FinalGuestSurfaceStatus::InvalidationLoss
                           : FinalGuestSurfaceStatus::Complete,
        .resolved_range_mismatch = mismatch,
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
    u32 bound_base_mip{};
    u32 bound_mip_count{};
    u32 bound_base_layer{};
    u32 bound_layer_count{};
    u64 source_image_uid{};
    u64 source_backing_generation{};
    u32 gamma_bits{};
    bool source_view_srgb{};
    bool bound_view_observed{};
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
    u32 bound_base_mip{};
    u32 bound_mip_count{};
    u32 bound_base_layer{};
    u32 bound_layer_count{};
    u64 source_image_uid{};
    u64 source_backing_generation{};
    u32 gamma_bits{};
    u64 config_generation{};
    bool source_view_srgb{};
    bool resolved_range_mismatch{};
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
        const auto view_assessment = AssessPpSampledInputSourceView({
            .resolved_base_mip = descriptor.resolved_base_mip,
            .resolved_mip_count = descriptor.resolved_mip_count,
            .resolved_base_layer = descriptor.resolved_base_layer,
            .resolved_layer_count = descriptor.resolved_layer_count,
            .bound_base_mip = descriptor.bound_base_mip,
            .bound_mip_count = descriptor.bound_mip_count,
            .bound_base_layer = descriptor.bound_base_layer,
            .bound_layer_count = descriptor.bound_layer_count,
        });
        const bool valid =
            bypassed && format_supported && descriptor.input_width != 0 &&
            descriptor.input_height != 0 && descriptor.output_width != 0 &&
            descriptor.output_height != 0 && descriptor.resolved_mip_count == 1 &&
            descriptor.resolved_layer_count == 1 && descriptor.source_image_uid != 0 &&
            descriptor.source_backing_generation != 0 && descriptor.source_view_srgb &&
            (!descriptor.bound_view_observed ||
             view_assessment.status == FinalGuestSurfaceStatus::Complete) &&
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
            .bound_base_mip = descriptor.bound_base_mip,
            .bound_mip_count = descriptor.bound_mip_count,
            .bound_base_layer = descriptor.bound_base_layer,
            .bound_layer_count = descriptor.bound_layer_count,
            .gamma_bits = descriptor.gamma_bits,
            .source_view_srgb = descriptor.source_view_srgb,
            .resolved_range_mismatch = view_assessment.resolved_range_mismatch,
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
            .bound_base_mip = descriptor.bound_base_mip,
            .bound_mip_count = descriptor.bound_mip_count,
            .bound_base_layer = descriptor.bound_base_layer,
            .bound_layer_count = descriptor.bound_layer_count,
            .source_image_uid = descriptor.source_image_uid,
            .source_backing_generation = descriptor.source_backing_generation,
            .gamma_bits = descriptor.gamma_bits,
            .config_generation = generation,
            .source_view_srgb = descriptor.source_view_srgb,
            .resolved_range_mismatch = view_assessment.resolved_range_mismatch,
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
        u32 bound_base_mip{};
        u32 bound_mip_count{};
        u32 bound_base_layer{};
        u32 bound_layer_count{};
        u32 gamma_bits{};
        bool source_view_srgb{};
        bool resolved_range_mismatch{};

        bool operator==(const LogicalDescriptor&) const = default;
    };

    LogicalDescriptor last_descriptor{};
    u64 generation{};
    bool has_descriptor{};
};

struct PpSampledInputObservationDescriptor {
    bool in_window{};
    bool stamp_valid{};
    bool metadata_valid{};
};

struct PpSampledInputObservationPlan {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    bool emit{};
};

[[nodiscard]] constexpr PpSampledInputObservationPlan PlanPpSampledInputObservation(
    PpSampledInputObservationDescriptor descriptor) noexcept {
    if (!descriptor.in_window) {
        return {};
    }
    if (!descriptor.stamp_valid || !descriptor.metadata_valid) {
        return {.status = FinalGuestSurfaceStatus::InvalidationLoss, .emit = true};
    }
    return {.status = FinalGuestSurfaceStatus::Complete, .emit = true};
}

[[nodiscard]] constexpr FinalGuestSurfaceTilePlan ApplyPpSampledInputObservationStatus(
    FinalGuestSurfaceTilePlan plan, FinalGuestSurfaceStatus observation_status) noexcept {
    if (observation_status == FinalGuestSurfaceStatus::Complete) {
        return plan;
    }
    plan.loss = {};
    if (observation_status == FinalGuestSurfaceStatus::GapLoss) {
        plan.status = observation_status;
        plan.loss.gap = 1;
    } else {
        plan.status = FinalGuestSurfaceStatus::InvalidationLoss;
        plan.loss.invalidation = 1;
    }
    return plan;
}

[[nodiscard]] constexpr bool ShouldAssignPpSampledInputFrame(bool enabled, bool stamp_valid,
                                                             bool metadata_valid) noexcept {
    return enabled && stamp_valid && metadata_valid;
}

struct FinalGuestSurfaceSampledInputPayload {
    u64 sequence{};
    u64 process_time_us{};
    u64 token{};
    FinalGuestSurfaceSampledInputMetadata metadata{};
    PpSourceBackingFootprintPlan source_backing{};
    bool source_backing_captured{};
};

struct FinalGuestSurfaceSampledInputTakeResult {
    FinalGuestSurfaceSampledInputPayload payload{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    bool emit{};
};

class FinalGuestSurfaceSampledInputFrameState {
public:
    [[nodiscard]] FinalGuestSurfaceStatus Assign(
        PpSampledInputObservationPlan observation,
        FinalGuestSurfaceSampledInputPayload next) noexcept {
        if (!observation.emit) {
            return FinalGuestSurfaceStatus::AlreadyConsumed;
        }
        if (pending || poisoned) {
            ClearPending();
            poisoned = true;
            return FinalGuestSurfaceStatus::GapLoss;
        }
        if (next.sequence == 0 || next.token == 0 ||
            (observation.status == FinalGuestSurfaceStatus::Complete &&
             (next.process_time_us == 0 || !next.metadata.valid))) {
            poisoned = true;
            return FinalGuestSurfaceStatus::InvalidationLoss;
        }
        payload = next;
        pending_status = observation.status;
        pending = true;
        return pending_status;
    }

    [[nodiscard]] FinalGuestSurfaceStatus AssignIfValid(
        bool eligible, FinalGuestSurfaceSampledInputPayload next) noexcept {
        return Assign({.status = next.metadata.valid ? FinalGuestSurfaceStatus::Complete
                                                     : FinalGuestSurfaceStatus::InvalidationLoss,
                       .emit = eligible},
                      next);
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
            const auto result = FinalGuestSurfaceSampledInputTakeResult{
                .payload = payload,
                .status = FinalGuestSurfaceStatus::GapLoss,
                .emit = true,
            };
            ClearPending();
            return result;
        }
        const auto result = FinalGuestSurfaceSampledInputTakeResult{
            .payload = payload,
            .status = pending_status,
            .emit = true,
        };
        ClearPending();
        return result;
    }

    [[nodiscard]] FinalGuestSurfaceStatus MarkPendingLoss(FinalGuestSurfaceStatus status) noexcept {
        if (!pending) {
            return FinalGuestSurfaceStatus::AlreadyConsumed;
        }
        pending_status = status == FinalGuestSurfaceStatus::GapLoss
                             ? status
                             : FinalGuestSurfaceStatus::InvalidationLoss;
        return pending_status;
    }

    void Clear() noexcept {
        ClearPending();
        poisoned = false;
    }

private:
    void ClearPending() noexcept {
        payload = {};
        pending_status = FinalGuestSurfaceStatus::AlreadyConsumed;
        pending = false;
    }

    FinalGuestSurfaceSampledInputPayload payload{};
    FinalGuestSurfaceStatus pending_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    bool pending{};
    bool poisoned{};
};

struct PpSampledInputTransferPlan {
    u32 color_write_to_transfer_barriers{};
    u32 copy_regions{};
    bool copy{};
    bool paired_output_and_raw{};
    bool paired_source_backing_snapshot{};
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
        .color_write_to_transfer_barriers = 2,
        .copy_regions = 3,
        .copy = true,
        .paired_output_and_raw = true,
        .paired_source_backing_snapshot = true,
        .callback_payload_is_scalar_only = true,
    };
}

[[nodiscard]] constexpr bool IsPpSampledInputTransferContractValid(
    const PpSampledInputTransferPlan& plan) noexcept {
    if (!plan.copy) {
        return plan.color_write_to_transfer_barriers == 0 && plan.copy_regions == 0 &&
               !plan.paired_output_and_raw && !plan.paired_source_backing_snapshot &&
               !plan.callback_payload_is_scalar_only && !plan.cpu_wait && !plan.finish &&
               !plan.callback_retains_frame && !plan.callback_retains_image &&
               !plan.callback_retains_vk_image;
    }
    return plan.color_write_to_transfer_barriers == 2 && plan.copy_regions == 3 &&
           plan.paired_output_and_raw && plan.paired_source_backing_snapshot &&
           plan.callback_payload_is_scalar_only && !plan.cpu_wait && !plan.finish &&
           !plan.callback_retains_frame && !plan.callback_retains_image &&
           !plan.callback_retains_vk_image;
}

struct PpSampledInputPairedCaptureDescriptor {
    bool enabled{};
    u32 width{};
    u32 height{};
    FinalGuestSurfaceFormat output_format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceFormat sampled_format{FinalGuestSurfaceFormat::Unsupported};
    u64 slot_bytes{};
    u64 alignment{};
};

struct PpSampledInputPairedCapturePlan {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceComparison output_comparison{FinalGuestSurfaceComparison::ExactVisible};
    u64 output_offset{};
    u64 output_bytes{};
    u64 sampled_offset{};
    u64 sampled_bytes{};
    u64 total_bytes{};
    u32 slot_count{};
    u32 copy_region_count{};
    bool output_is_authoritative{};
    bool cpu_gamma_reconstruction_is_authoritative{};
    bool callback_payload_is_scalar_only{};
    bool cpu_wait{};
    bool finish{};
};

[[nodiscard]] constexpr PpSampledInputPairedCapturePlan PlanPpSampledInputPairedCapture(
    PpSampledInputPairedCaptureDescriptor descriptor) noexcept {
    if (!descriptor.enabled) {
        return {};
    }
    if (descriptor.width == 0 || descriptor.height == 0 || descriptor.alignment == 0 ||
        (descriptor.output_format != FinalGuestSurfaceFormat::Rgba8 &&
         descriptor.output_format != FinalGuestSurfaceFormat::Bgra8) ||
        descriptor.sampled_format != FinalGuestSurfaceFormat::Rgba16Float) {
        return {.status = FinalGuestSurfaceStatus::Unsupported};
    }
    constexpr u64 OutputBytesPerPixel = 4;
    constexpr u64 SampledBytesPerPixel = 8;
    const u64 pixels = static_cast<u64>(descriptor.width) * descriptor.height;
    if (pixels > std::numeric_limits<u64>::max() / SampledBytesPerPixel) {
        return {.status = FinalGuestSurfaceStatus::CapacityLoss};
    }
    const u64 output_bytes = pixels * OutputBytesPerPixel;
    const u64 sampled_bytes = pixels * SampledBytesPerPixel;
    const u64 remainder = output_bytes % descriptor.alignment;
    const u64 padding = remainder == 0 ? 0 : descriptor.alignment - remainder;
    if (output_bytes > std::numeric_limits<u64>::max() - padding ||
        output_bytes + padding > std::numeric_limits<u64>::max() - sampled_bytes) {
        return {.status = FinalGuestSurfaceStatus::CapacityLoss};
    }
    const u64 sampled_offset = output_bytes + padding;
    const u64 total_bytes = sampled_offset + sampled_bytes;
    if (total_bytes > descriptor.slot_bytes) {
        return {.status = FinalGuestSurfaceStatus::CapacityLoss};
    }
    return {
        .status = FinalGuestSurfaceStatus::Complete,
        .output_comparison = FinalGuestSurfaceComparison::LocalizedVisualReturn,
        .output_bytes = output_bytes,
        .sampled_offset = sampled_offset,
        .sampled_bytes = sampled_bytes,
        .total_bytes = total_bytes,
        .slot_count = 1,
        .copy_region_count = 2,
        .output_is_authoritative = true,
        .callback_payload_is_scalar_only = true,
    };
}

[[nodiscard]] constexpr FinalGuestSurfaceTilePlan MakePpSampledInputPairedTilePlan(
    FinalGuestSurfaceTilePlan output, const PpSampledInputPairedCapturePlan& pair) noexcept {
    if (output.status != FinalGuestSurfaceStatus::Complete ||
        pair.status != FinalGuestSurfaceStatus::Complete ||
        output.sample_bytes != pair.output_bytes ||
        pair.sampled_offset > std::numeric_limits<u32>::max() ||
        pair.sampled_bytes > std::numeric_limits<u32>::max() ||
        pair.total_bytes > std::numeric_limits<u32>::max() || output.surface_width == 0 ||
        pair.sampled_bytes / 8 != static_cast<u64>(output.surface_width) * output.surface_height) {
        output.status = FinalGuestSurfaceStatus::CapacityLoss;
        output.loss.byte_capacity = 1;
        return output;
    }
    output.sample_bytes = static_cast<u32>(pair.total_bytes);
    output.copy_region_count = pair.copy_region_count;
    output.paired_sampled_offset = static_cast<u32>(pair.sampled_offset);
    output.paired_sampled_bytes = static_cast<u32>(pair.sampled_bytes);
    output.paired_sampled_row_bytes = output.surface_width * 8;
    output.paired_sampled_format = FinalGuestSurfaceFormat::Rgba16Float;
    return output;
}

enum class PpSampledInputBoundary : u8 {
    Ambiguous,
    OutputClean,
    AtOrBeforeSample,
    AfterSample,
};

struct PpSampledInputPairClassification {
    bool output_visual_return{};
    bool raw_sample_return{};
    bool raw_sample_stable{};
    bool complete{};
};

[[nodiscard]] constexpr PpSampledInputBoundary ClassifyPpSampledInputPair(
    PpSampledInputPairClassification classification) noexcept {
    if (!classification.complete) {
        return PpSampledInputBoundary::Ambiguous;
    }
    if (!classification.output_visual_return) {
        return PpSampledInputBoundary::OutputClean;
    }
    if (classification.raw_sample_return) {
        return PpSampledInputBoundary::AtOrBeforeSample;
    }
    if (classification.raw_sample_stable) {
        return PpSampledInputBoundary::AfterSample;
    }
    return PpSampledInputBoundary::Ambiguous;
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
