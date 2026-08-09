// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <limits>
#include <string>

#include "common/types.h"
#include "video_core/renderer_vulkan/final_guest_surface_content.h"

namespace Vulkan::HostPasses {

enum class PpSourceSnapshotPoint : u8 {
    AfterVisiblePp,
    FlipPublication,
};

struct PpSourceReconstructionExecutionPlan {
    bool enabled{};
    bool prepare_resources_before_visible_pp{};
    bool capture_snapshot_before_visible_pp{};
    bool capture_snapshot_after_visible_pp{};
    bool run_reconstruction_after_visible_pp{};
    bool visible_pp_draw_remains_original_source{};
};

[[nodiscard]] constexpr PpSourceReconstructionExecutionPlan PlanPpSourceReconstructionExecution(
    FinalGuestSurfaceStage stage, bool in_window, bool frame_is_new, bool metadata_valid) noexcept {
    const bool post_sample = stage == FinalGuestSurfaceStage::PpSourceReconstruction;
    const bool publication = stage == FinalGuestSurfaceStage::PpSourcePublicationReconstruction;
    if ((!post_sample && !publication) || !in_window || !frame_is_new || !metadata_valid) {
        return {};
    }
    return {
        .enabled = true,
        .prepare_resources_before_visible_pp = publication,
        .capture_snapshot_before_visible_pp = publication,
        .capture_snapshot_after_visible_pp = post_sample,
        .run_reconstruction_after_visible_pp = true,
        .visible_pp_draw_remains_original_source = true,
    };
}

struct PpSourceReconstructionDescriptor {
    bool enabled{};
    bool in_window{};
    bool frame_is_new{};
    bool visible_pp_draw_encoded{};
    bool sampled_metadata_valid{};
    bool fsr_bypassed{};
    bool source_view_matches_baseline{};
    bool source_view_srgb{};
    bool source_snapshot_available{};
    bool source_snapshot_view_available{};
    bool reconstruction_output_available{};
    PpSourceSnapshotPoint snapshot_point{PpSourceSnapshotPoint::AfterVisiblePp};
    u32 source_width{};
    u32 source_height{};
    u32 output_width{};
    u32 output_height{};
    u32 bound_base_mip{};
    u32 bound_mip_count{};
    u32 bound_base_layer{};
    u32 bound_layer_count{};
    FinalGuestSurfaceFormat source_format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceFormat output_format{FinalGuestSurfaceFormat::Unsupported};
    u64 existing_readback_bytes{};
    u64 slot_bytes{};
    u64 alignment{};
};

struct PpSourceReconstructionPlan {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceFormat output_format{FinalGuestSurfaceFormat::Unsupported};
    u32 resource_image_count{};
    u32 source_snapshot_bytes{};
    u32 reconstruction_output_bytes{};
    u32 reconstruction_readback_offset{};
    u32 final_readback_bytes{};
    u32 source_copy_count{};
    u32 reconstruction_draw_count{};
    u32 total_pp_draw_count{};
    u32 present_copy_count{};
    u32 total_present_plane_count{};
    u32 source_image_barrier_count{};
    u32 snapshot_image_barrier_count{};
    u32 reconstruction_output_barrier_count{};
    bool visible_draw_precedes_source_snapshot{};
    bool guest_rendering_ended_before_snapshot{};
    bool source_snapshot_precedes_visible_draw{};
    bool source_snapshot_precedes_reconstruction_draw{};
    bool snapshot_is_immutable_during_reconstruction{};
    bool same_pp_pipeline{};
    bool same_pp_sampler{};
    bool same_pp_settings{};
    bool same_source_view_format_and_swizzle{};
    bool source_transition_occurs_after_visible_sample{};
    bool reconstruction_restored_for_present_transfer{};
    bool inserts_barrier_before_visible_sample{};
    bool snapshot_visibility_transition_can_perturb_visible_pp{};
    bool cpu_wait{};
    bool finish{};
    bool callback_retains_frame{};
    bool callback_retains_source_image{};
    bool callback_retains_reconstruction_image{};
    bool callback_payload_is_scalar_only{};
};

[[nodiscard]] constexpr u64 AlignPpSourceReconstructionOffset(u64 value, u64 alignment) noexcept {
    if (alignment == 0 || value > std::numeric_limits<u64>::max() - (alignment - 1)) {
        return std::numeric_limits<u64>::max();
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

[[nodiscard]] constexpr PpSourceReconstructionPlan PlanPpSourceReconstruction(
    const PpSourceReconstructionDescriptor& descriptor) noexcept {
    if (!descriptor.enabled || !descriptor.in_window || !descriptor.frame_is_new) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status) {
        return PpSourceReconstructionPlan{.status = status};
    };
    if (!descriptor.visible_pp_draw_encoded || !descriptor.sampled_metadata_valid ||
        !descriptor.fsr_bypassed || !descriptor.source_view_matches_baseline ||
        !descriptor.source_snapshot_available || !descriptor.source_snapshot_view_available ||
        !descriptor.reconstruction_output_available || descriptor.source_width == 0 ||
        descriptor.source_height == 0 || descriptor.output_width == 0 ||
        descriptor.output_height == 0) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss);
    }
    const bool source_format_supported =
        descriptor.source_format == FinalGuestSurfaceFormat::Rgba8 ||
        descriptor.source_format == FinalGuestSurfaceFormat::Bgra8;
    const bool output_format_supported =
        descriptor.output_format == FinalGuestSurfaceFormat::Rgba8 ||
        descriptor.output_format == FinalGuestSurfaceFormat::Bgra8;
    if (!source_format_supported || !output_format_supported || !descriptor.source_view_srgb) {
        return reject(FinalGuestSurfaceStatus::Unsupported);
    }
    if (descriptor.bound_base_mip != 0 || descriptor.bound_mip_count != 1 ||
        descriptor.bound_base_layer != 0 || descriptor.bound_layer_count != 1) {
        return reject(FinalGuestSurfaceStatus::Unsupported);
    }
    if (descriptor.alignment == 0) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss);
    }
    const u64 source_bytes =
        static_cast<u64>(descriptor.source_width) * descriptor.source_height * 4;
    const u64 output_bytes =
        static_cast<u64>(descriptor.output_width) * descriptor.output_height * 4;
    const u64 output_offset =
        AlignPpSourceReconstructionOffset(descriptor.existing_readback_bytes, descriptor.alignment);
    if (source_bytes > std::numeric_limits<u32>::max() ||
        output_bytes > std::numeric_limits<u32>::max() ||
        output_offset > std::numeric_limits<u32>::max() || output_offset > descriptor.slot_bytes ||
        output_bytes > descriptor.slot_bytes - output_offset ||
        output_offset + output_bytes > std::numeric_limits<u32>::max()) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss);
    }
    const bool at_flip_publication =
        descriptor.snapshot_point == PpSourceSnapshotPoint::FlipPublication;
    return {
        .status = FinalGuestSurfaceStatus::Complete,
        .output_format = descriptor.output_format,
        .resource_image_count = 2,
        .source_snapshot_bytes = static_cast<u32>(source_bytes),
        .reconstruction_output_bytes = static_cast<u32>(output_bytes),
        .reconstruction_readback_offset = static_cast<u32>(output_offset),
        .final_readback_bytes = static_cast<u32>(output_offset + output_bytes),
        .source_copy_count = 1,
        .reconstruction_draw_count = 1,
        .total_pp_draw_count = 2,
        .present_copy_count = 1,
        .total_present_plane_count = 4,
        .source_image_barrier_count = 2,
        .snapshot_image_barrier_count = 2,
        .reconstruction_output_barrier_count = 2,
        .visible_draw_precedes_source_snapshot = !at_flip_publication,
        .guest_rendering_ended_before_snapshot = at_flip_publication,
        .source_snapshot_precedes_visible_draw = at_flip_publication,
        .source_snapshot_precedes_reconstruction_draw = true,
        .snapshot_is_immutable_during_reconstruction = true,
        .same_pp_pipeline = true,
        .same_pp_sampler = true,
        .same_pp_settings = true,
        .same_source_view_format_and_swizzle = true,
        .source_transition_occurs_after_visible_sample = !at_flip_publication,
        .reconstruction_restored_for_present_transfer = true,
        .inserts_barrier_before_visible_sample = at_flip_publication,
        .snapshot_visibility_transition_can_perturb_visible_pp = at_flip_publication,
        .callback_payload_is_scalar_only = true,
    };
}

[[nodiscard]] constexpr FinalGuestSurfaceTilePlan AttachPpSourceReconstructionPlane(
    FinalGuestSurfaceTilePlan plan, const PpSourceReconstructionPlan& reconstruction) noexcept {
    if (plan.status != FinalGuestSurfaceStatus::Complete ||
        reconstruction.status != FinalGuestSurfaceStatus::Complete ||
        reconstruction.reconstruction_readback_offset < plan.sample_bytes ||
        reconstruction.final_readback_bytes < reconstruction.reconstruction_readback_offset ||
        reconstruction.reconstruction_output_bytes !=
            static_cast<u64>(plan.row_bytes) * plan.surface_height ||
        reconstruction.final_readback_bytes !=
            static_cast<u64>(reconstruction.reconstruction_readback_offset) +
                reconstruction.reconstruction_output_bytes) {
        plan.status = FinalGuestSurfaceStatus::InvalidationLoss;
        plan.loss.invalidation = 1;
        return plan;
    }
    plan.paired_reconstruction_offset = reconstruction.reconstruction_readback_offset;
    plan.paired_reconstruction_bytes = reconstruction.reconstruction_output_bytes;
    plan.paired_reconstruction_row_bytes = plan.row_bytes;
    const bool reconstruction_stage =
        plan.stage == FinalGuestSurfaceStage::PpSourceReconstruction ||
        plan.stage == FinalGuestSurfaceStage::PpSourcePublicationReconstruction;
    plan.paired_reconstruction_format = reconstruction_stage && plan.bytes_per_pixel == 4
                                            ? reconstruction.output_format
                                            : FinalGuestSurfaceFormat::Unsupported;
    if (plan.paired_reconstruction_format == FinalGuestSurfaceFormat::Unsupported) {
        plan.status = FinalGuestSurfaceStatus::Unsupported;
        plan.loss.unsupported_format = 1;
        return plan;
    }
    plan.sample_bytes = reconstruction.final_readback_bytes;
    ++plan.copy_region_count;
    return plan;
}

enum class PpSourceReconstructionClass : u8 {
    Unassessed,
    ReproducedFromSnapshot,
    NotReproducedFromSnapshot,
};

[[nodiscard]] constexpr PpSourceReconstructionClass ClassifyPpSourceReconstruction(
    bool authoritative_output_return, bool reconstructed_output_return) noexcept {
    if (!authoritative_output_return) {
        return PpSourceReconstructionClass::Unassessed;
    }
    return reconstructed_output_return ? PpSourceReconstructionClass::ReproducedFromSnapshot
                                       : PpSourceReconstructionClass::NotReproducedFromSnapshot;
}

struct PpSourceReconstructionReport {
    u32 request_ordinal{};
    u64 a_sequence{};
    u64 b_sequence{};
    u64 c_sequence{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> reproduced_ordinals{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> not_reproduced_ordinals{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> ambiguous_ordinals{};
    u32 reproduced_count{};
    u32 not_reproduced_count{};
    u32 ambiguous_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
};

[[nodiscard]] inline std::string FormatPpSourceReconstructionOrdinals(
    const std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals>& values, u32 count) {
    std::string result;
    for (u32 index = 0; index < count && index < values.size(); ++index) {
        if (!result.empty()) {
            result.push_back(',');
        }
        result += std::to_string(values[index]);
    }
    return result;
}

[[nodiscard]] inline std::string FormatPpSourceReconstructionReport(
    const PpSourceReconstructionReport& report) {
    return "PPSR q=" + std::to_string(report.request_ordinal) +
           " abc=" + std::to_string(report.a_sequence) + '/' + std::to_string(report.b_sequence) +
           '/' + std::to_string(report.c_sequence) + " yes=" +
           FormatPpSourceReconstructionOrdinals(report.reproduced_ordinals,
                                                report.reproduced_count) +
           " no=" +
           FormatPpSourceReconstructionOrdinals(report.not_reproduced_ordinals,
                                                report.not_reproduced_count) +
           " amb=" +
           FormatPpSourceReconstructionOrdinals(report.ambiguous_ordinals, report.ambiguous_count) +
           " st=" + std::to_string(static_cast<u32>(report.status));
}

} // namespace Vulkan::HostPasses
