// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "video_core/renderer_vulkan/host_passes/pp_upstream_sample_tee.h"

namespace Vulkan::HostPasses {

struct PpUpstreamFeedbackTeeAuthorizedTargetPrePostReceipt {
    FinalGuestSurfaceReadbackSlotPool::Token outer_slot{};
    u64 download_offset{};
    u64 download_bytes{};
    u64 rendering_serial{};
    u32 selected_candidate_index{};
    u32 recorded_plane_mask{};
};

struct PpUpstreamFeedbackTeeAuthorizedTargetPrePostDescriptor {
    PpUpstreamFeedbackPrePostTakeEvaluation take{};
    PpUpstreamSampleTeeDeferredReadbackDescriptor tee_attempt{};
    PpUpstreamFeedbackTeeAuthorizedTargetPrePostReceipt receipt{};
    bool download_invalidation_already_planned{};
    u32 input_count{};
    PpUpstreamSampleRouteSet sample_routes{};
};

struct PpUpstreamFeedbackTeeAuthorizedTargetPrePostPlan {
    FinalGuestSurfaceReadbackSlotPool::Token expected_outer_slot{};
    u64 expected_rendering_serial{};
    u64 download_offset{};
    u64 download_bytes{};
    u32 selected_candidate_index{};
    u32 region_index{};
    u32 logical_ordinal{};
    u32 region_offset{};
    u32 region_bytes{};
    u32 source_plane_bytes{};
    u32 bytes_per_pixel{};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    vk::Format storage_format{vk::Format::eUndefined};
    bool raw_compare{};
    u32 input_count{};
    PpUpstreamSampleRouteSet sample_routes{};
    PpUpstreamSampleTeeReadbackHeader expected_tee_header{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    u32 additional_download_invalidation_count{};
    u32 additional_gpu_copy_count{};
    u32 additional_command_count{};
    u32 additional_barrier_count{};
    u32 additional_slot_count{};
    u32 additional_release_count{};
    bool provisionally_compact{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
    bool retains_private_state{};
};

[[nodiscard]] inline bool IsCanonicalPpUpstreamFeedbackTeeAuthorizedBasePlan(
    const PpUpstreamFeedbackPrePostPlan& plan, u32 selected_candidate_index,
    const PpUpstreamSampleTeeWindowPlan& window, u32& matched_region_index) noexcept {
    const bool semantic = plan.format == FinalGuestSurfaceFormat::Rgba8 ||
                          plan.format == FinalGuestSurfaceFormat::Bgra8;
    const bool supported_raw = plan.bytes_per_pixel == 1 || plan.bytes_per_pixel == 2 ||
                               plan.bytes_per_pixel == 4 || plan.bytes_per_pixel == 8 ||
                               plan.bytes_per_pixel == 16;
    if (plan.status != FinalGuestSurfaceStatus::Complete || plan.loss.Any() || !plan.copy ||
        plan.region_count == 0 || plan.region_count > plan.MaxRegions ||
        plan.candidate_capacity == 0 || plan.candidate_capacity > plan.MaxCandidates ||
        selected_candidate_index >= plan.candidate_capacity || plan.plane_bytes == 0 ||
        plan.post_plane_offset != plan.plane_bytes ||
        plan.candidate_stride < plan.post_plane_offset + plan.plane_bytes ||
        static_cast<u64>(plan.candidate_stride) * plan.candidate_capacity != plan.total_bytes ||
        (plan.raw_compare ? plan.format != FinalGuestSurfaceFormat::Unsupported || !supported_raw
                          : !semantic || plan.bytes_per_pixel != 4) ||
        window.width == 0 || window.height == 0 || window.pixel_count == 0 ||
        static_cast<u64>(window.width) * window.height != window.pixel_count) {
        return false;
    }

    u32 matches{};
    u32 previous_end{};
    for (u32 index = 0; index < plan.region_count; ++index) {
        const auto& region = plan.regions[index];
        const u64 pixels = static_cast<u64>(region.width) * region.height;
        const u64 bytes = pixels * plan.bytes_per_pixel;
        if (region.width == 0 || region.height == 0 || region.byte_size == 0 ||
            bytes != region.byte_size || (index == 0 && region.buffer_offset != 0) ||
            region.buffer_offset < previous_end || region.buffer_offset > plan.plane_bytes ||
            region.byte_size > plan.plane_bytes - region.buffer_offset) {
            return false;
        }
        previous_end = region.buffer_offset + region.byte_size;
        const bool exact = region.logical_ordinal == window.logical_ordinal &&
                           region.x == window.x && region.y == window.y &&
                           region.width == window.width && region.height == window.height &&
                           region.byte_size == window.pixel_count * plan.bytes_per_pixel;
        if (exact) {
            matched_region_index = index;
            ++matches;
        }
    }
    return matches == 1;
}

[[nodiscard]] inline bool IsCanonicalPpUpstreamFeedbackTeeAuthorizedRoutes(
    const PpUpstreamSampleRouteSet& routes, u32 input_count) noexcept {
    if (input_count == 0 || input_count > PpTerminalScopeSampledInputContentPlan::MaxInputs ||
        routes.status != FinalGuestSurfaceStatus::Complete || routes.loss.Any() ||
        routes.selected_input_index >= input_count || routes.route_count == 0 ||
        routes.route_count > routes.MaxRoutes || routes.retains_image || routes.retains_view ||
        routes.retains_sampler || routes.logs_private_words) {
        return false;
    }
    for (u32 index = 0; index < routes.route_count; ++index) {
        if (!routes.routes[index].valid ||
            routes.routes[index].sampler_count > routes.routes[index].MaxSamplers) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline PpUpstreamFeedbackTeeAuthorizedTargetPrePostPlan
PlanPpUpstreamFeedbackTeeAuthorizedTargetPrePost(
    const PpUpstreamFeedbackTeeAuthorizedTargetPrePostDescriptor& descriptor) noexcept {
    const auto reject = [] {
        return PpUpstreamFeedbackTeeAuthorizedTargetPrePostPlan{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
        };
    };
    const auto& take = descriptor.take;
    const auto& attempt = descriptor.tee_attempt;
    const auto& receipt = descriptor.receipt;
    const auto& plan = take.take.plan;
    const bool missing_selection_exact =
        take.raw_status == FinalGuestSurfaceStatus::Complete && !take.raw_loss.Any() &&
        take.recorded_plane_mask == 0x3u &&
        take.first_failure == PpUpstreamFeedbackPrePostTakeFirstFailure::MissingSelection &&
        take.armed && !take.selected && take.has_slot && take.take.plan == plan &&
        take.take.candidate_index == 0 && take.take.recorded_plane_mask == 0x3u &&
        take.take.status == FinalGuestSurfaceStatus::GapLoss && take.take.loss.gap == 1 &&
        FinalGuestSurfaceLossMask(take.take.loss, 0) == (1u << 13) && !take.take.copy &&
        !take.take.cpu_wait && !take.take.finish && !take.take.retains_image &&
        !take.take.retains_vk_image;
    const auto tee_plan = PlanPpUpstreamSampleTeeDeferredReadback(attempt);
    const bool tee_exact =
        attempt.enabled && attempt.attempted && !attempt.release_only_outer_slot &&
        attempt.outer_slot && attempt.capture_status == FinalGuestSurfaceStatus::Complete &&
        !attempt.capture_loss.Any() && attempt.capture_has_copy && attempt.controller.ready &&
        attempt.controller.status == FinalGuestSurfaceStatus::Complete &&
        !attempt.controller.loss.Any() && attempt.controller.serial != 0 &&
        attempt.commands.serial == attempt.controller.serial &&
        attempt.selected_candidate_index == attempt.controller.candidate_index &&
        attempt.selected_candidate_index == attempt.record.candidate_index &&
        attempt.capture.header.candidate_index == attempt.selected_candidate_index &&
        IsCanonicalPpUpstreamSampleTeeCandidateRecord(attempt.record,
                                                      attempt.selected_candidate_index) &&
        IsCanonicalPpUpstreamSampleTeeDeferredReadbackPlan(tee_plan);
    const bool receipt_exact =
        receipt.outer_slot && receipt.outer_slot == attempt.outer_slot &&
        receipt.rendering_serial != 0 && receipt.rendering_serial == attempt.controller.serial &&
        receipt.selected_candidate_index == attempt.controller.candidate_index &&
        receipt.recorded_plane_mask == 0x3u && receipt.download_bytes >= plan.total_bytes &&
        receipt.download_bytes != 0 &&
        receipt.download_offset <= std::numeric_limits<u64>::max() - receipt.download_bytes &&
        receipt.download_offset ==
            static_cast<u64>(receipt.outer_slot.slot) * receipt.download_bytes;
    u32 region_index{};
    if (!missing_selection_exact || !tee_exact || !receipt_exact ||
        !IsCanonicalPpUpstreamFeedbackTeeAuthorizedBasePlan(
            plan, receipt.selected_candidate_index, attempt.record.window, region_index) ||
        !IsCanonicalPpUpstreamFeedbackTeeAuthorizedRoutes(descriptor.sample_routes,
                                                          descriptor.input_count)) {
        return reject();
    }
    const auto& region = plan.regions[region_index];
    return {
        .expected_outer_slot = receipt.outer_slot,
        .expected_rendering_serial = receipt.rendering_serial,
        .download_offset = receipt.download_offset,
        .download_bytes = receipt.download_bytes,
        .selected_candidate_index = receipt.selected_candidate_index,
        .region_index = region_index,
        .logical_ordinal = region.logical_ordinal,
        .region_offset = region.buffer_offset,
        .region_bytes = region.byte_size,
        .source_plane_bytes = plan.plane_bytes,
        .bytes_per_pixel = plan.bytes_per_pixel,
        .format = plan.format,
        .storage_format = plan.storage_format,
        .raw_compare = plan.raw_compare,
        .input_count = descriptor.input_count,
        .sample_routes = descriptor.sample_routes,
        .expected_tee_header = tee_plan.expected_header,
        .status = FinalGuestSurfaceStatus::Complete,
        .additional_download_invalidation_count =
            descriptor.download_invalidation_already_planned ? 0u : 1u,
        .provisionally_compact = true,
    };
}

struct PpUpstreamFeedbackTeeAuthorizedTargetPrePostDownloadObservation {
    FinalGuestSurfaceReadbackSlotPool::Token outer_slot{};
    u64 offset{};
    u64 bytes{};
    bool attempted{};
    bool succeeded{};
};

struct PpUpstreamFeedbackTeeAuthorizedTargetPrePostResult {
    PpTerminalScopeContentHistoryLayout layout{};
    std::vector<std::byte> bytes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    vk::Format storage_format{vk::Format::eUndefined};
    u32 additional_gpu_copy_count{};
    u32 additional_command_count{};
    u32 additional_barrier_count{};
    u32 additional_slot_count{};
    u32 additional_release_count{};
    bool publish{};
    bool retains_private_state{};
};

[[nodiscard]] inline PpUpstreamFeedbackTeeAuthorizedTargetPrePostResult
AuthorizePpUpstreamFeedbackTeeAuthorizedTargetPrePost(
    const PpUpstreamFeedbackTeeAuthorizedTargetPrePostPlan& plan,
    PpUpstreamFeedbackPrePostCompactResult&& provisional,
    const PpUpstreamFeedbackTeeAuthorizedTargetPrePostDownloadObservation& download,
    const PpUpstreamSampleTeeDeferredReadbackDescriptor& attempt,
    const PpUpstreamSampleTeeDeferredObservation& finalized) {
    std::vector<std::byte> provisional_bytes{};
    provisional_bytes.swap(provisional.bytes);
    const auto reject = [&] {
        std::vector<std::byte>{}.swap(provisional_bytes);
        return PpUpstreamFeedbackTeeAuthorizedTargetPrePostResult{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
        };
    };
    const auto tee_plan = PlanPpUpstreamSampleTeeDeferredReadback(attempt);
    const bool plan_exact =
        plan.provisionally_compact && plan.status == FinalGuestSurfaceStatus::Complete &&
        !plan.loss.Any() && plan.expected_outer_slot && plan.expected_rendering_serial != 0 &&
        plan.region_bytes != 0 && plan.source_plane_bytes != 0 &&
        plan.region_offset <= plan.source_plane_bytes &&
        plan.region_bytes <= plan.source_plane_bytes - plan.region_offset;
    const bool download_exact = download.attempted && download.succeeded &&
                                download.outer_slot == plan.expected_outer_slot &&
                                download.offset == plan.download_offset &&
                                download.bytes == plan.download_bytes;
    const bool attempt_exact =
        attempt.enabled && attempt.attempted && attempt.outer_slot == plan.expected_outer_slot &&
        attempt.selected_candidate_index == plan.selected_candidate_index &&
        attempt.controller.candidate_index == plan.selected_candidate_index &&
        attempt.controller.serial == plan.expected_rendering_serial &&
        attempt.commands.serial == plan.expected_rendering_serial &&
        IsCanonicalPpUpstreamSampleTeeDeferredReadbackPlan(tee_plan) &&
        tee_plan.expected_header == plan.expected_tee_header;
    const auto& lifecycle = finalized.lifecycle;
    const auto& publication = finalized.publication;
    const auto& owned = finalized.owned;
    const bool finalized_exact =
        finalized.observe_owned && finalized.record == attempt.record && publication.publish &&
        publication.status == FinalGuestSurfaceStatus::Complete && !publication.loss.Any() &&
        publication.shader == attempt.record.shader && publication.window == attempt.record.window &&
        lifecycle.status == FinalGuestSurfaceStatus::Complete && !lifecycle.loss.Any() &&
        lifecycle.expected_header == plan.expected_tee_header && lifecycle.finalized &&
        !lifecycle.lifecycle_invalid && lifecycle.expects_owned_copy &&
        lifecycle.mapped_invalidation_succeeded && lifecycle.bounded_record_copy_succeeded &&
        lifecycle.bounded_copy_completed_before_outer_slot_release &&
        lifecycle.expected_outer_slot_release_attempts == 1 &&
        lifecycle.observed_outer_slot_release_attempts == 1 &&
        lifecycle.slot_release_succeeded && owned.status == FinalGuestSurfaceStatus::Complete &&
        !owned.loss.Any() && owned.mapped_invalidation_succeeded &&
        owned.bounded_record_copy_succeeded && !owned.retains_mapped_span &&
        owned.header == plan.expected_tee_header && owned.records.size() == tee_plan.record_count;
    const bool writers_exact = finalized_exact &&
                               std::ranges::all_of(owned.records, [](const auto& record) {
                                   return record.writers == 1;
                               });
    const u64 expected_source_bytes = static_cast<u64>(plan.source_plane_bytes) * 2u;
    if (!plan_exact || !download_exact || !attempt_exact || !finalized_exact || !writers_exact ||
        provisional.status != FinalGuestSurfaceStatus::Complete || provisional.loss.Any() ||
        provisional_bytes.size() != expected_source_bytes) {
        return reject();
    }

    PpUpstreamFeedbackTeeAuthorizedTargetPrePostResult result{
        .bytes = std::vector<std::byte>(static_cast<u64>(plan.region_bytes) * 2u),
        .status = FinalGuestSurfaceStatus::Complete,
        .storage_format = plan.storage_format,
        .publish = true,
    };
    const auto source = std::span<const std::byte>{provisional_bytes};
    std::ranges::copy(source.subspan(plan.region_offset, plan.region_bytes), result.bytes.begin());
    std::ranges::copy(
        source.subspan(static_cast<u64>(plan.source_plane_bytes) + plan.region_offset,
                       plan.region_bytes),
        result.bytes.begin() + plan.region_bytes);
    result.layout = {
        .region_count = 1,
        .plane_bytes = plan.region_bytes,
        .second_plane_offset = plan.region_bytes,
        .total_bytes = plan.region_bytes * 2u,
        .input_count = plan.input_count,
        .input_unavailable_mask = (1u << plan.input_count) - 1u,
        .plane_mask = 0x3u,
        .regions = {{{.logical_ordinal = plan.logical_ordinal,
                      .buffer_offset = 0,
                      .byte_size = plan.region_bytes}}},
        .bytes_per_pixel = plan.bytes_per_pixel,
        .format = plan.format,
        .raw_compare = plan.raw_compare,
        .upstream_sample_routes = plan.sample_routes,
    };
    std::vector<std::byte>{}.swap(provisional_bytes);
    return result;
}

[[nodiscard]] inline PpUpstreamFeedbackTeeTargetSameFrameFact
ReducePpUpstreamFeedbackTeeTargetSameFrame(
    u64 sequence, const PpUpstreamFeedbackTeeAuthorizedTargetPrePostResult& authorized) noexcept {
    const auto reject = [sequence](bool available, FinalGuestSurfaceStatus status,
                                   PpUpstreamFeedbackTeeTargetSameFrameLoss loss,
                                   bool overflowed = false) {
        return PpUpstreamFeedbackTeeTargetSameFrameFact{
            .sequence = sequence,
            .status = status,
            .loss = loss,
            .available = available,
            .overflowed = overflowed,
        };
    };
    const auto invalid = [&, sequence](bool available) {
        return reject(available, FinalGuestSurfaceStatus::InvalidationLoss,
                      {.invalidation = 1});
    };

    // Authority is established only by the explicit authorization result.  A
    // layout that merely resembles it cannot make this observation available.
    if (sequence == 0 || !authorized.publish ||
        authorized.status != FinalGuestSurfaceStatus::Complete || authorized.loss.Any() ||
        authorized.bytes.empty() ||
        authorized.additional_gpu_copy_count != 0 || authorized.additional_command_count != 0 ||
        authorized.additional_barrier_count != 0 || authorized.additional_slot_count != 0 ||
        authorized.additional_release_count != 0 || authorized.retains_private_state) {
        return invalid(false);
    }

    const auto& layout = authorized.layout;
    const u64 expected_total = static_cast<u64>(layout.plane_bytes) * 2u;
    if (expected_total > std::numeric_limits<u32>::max()) {
        return reject(true, FinalGuestSurfaceStatus::CapacityLoss, {.byte_capacity = 1}, true);
    }
    if (!IsPpTerminalScopeContentHistoryEncodingCanonical(layout)) {
        return reject(true, FinalGuestSurfaceStatus::Unsupported, {.unsupported_format = 1});
    }
    const bool input_count_valid =
        layout.input_count != 0 &&
        layout.input_count <= PpTerminalScopeSampledInputContentPlan::MaxInputs;
    const u32 expected_unavailable_mask =
        input_count_valid ? (1u << layout.input_count) - 1u : 0u;
    const bool input_planes_empty = std::ranges::all_of(layout.input_planes, [](const auto& plane) {
        return plane == PpTerminalScopeSampledInputContentHistoryPlane{};
    });
    const bool unused_regions_empty =
        std::ranges::all_of(layout.regions.begin() + std::min<u32>(layout.region_count,
                                                                   layout.regions.size()),
                            layout.regions.end(), [](const auto& region) {
                                return region == PpTerminalScopeContentHistoryRegion{};
                            });
    if (layout.region_count != 1 || layout.region_count > layout.regions.size() ||
        layout.plane_bytes == 0 || layout.second_plane_offset != layout.plane_bytes ||
        layout.total_bytes != expected_total || authorized.bytes.size() != expected_total ||
        layout.consumer_plane_offset != 0 || layout.output_plane_offset != 0 ||
        layout.pre_first_plane_offset != 0 || layout.predecessor_pre_plane_offset != 0 ||
        layout.predecessor_post_plane_offset != 0 || layout.plane_mask != 0x3u ||
        !input_count_valid || layout.input_capture_mask != 0 || layout.input_alias_mask != 0 ||
        layout.input_unavailable_mask != expected_unavailable_mask || !input_planes_empty ||
        !unused_regions_empty ||
        !IsCanonicalPpUpstreamFeedbackTeeAuthorizedRoutes(layout.upstream_sample_routes,
                                                          layout.input_count)) {
        return invalid(true);
    }
    const auto& region = layout.regions[0];
    if (region.logical_ordinal == 0 || region.buffer_offset != 0 || region.byte_size == 0 ||
        region.byte_size != layout.plane_bytes ||
        region.byte_size % layout.bytes_per_pixel != 0) {
        return invalid(true);
    }

    const u64 pixels = region.byte_size / layout.bytes_per_pixel;
    if (pixels == 0 || pixels > std::numeric_limits<u32>::max()) {
        return reject(true, FinalGuestSurfaceStatus::CapacityLoss, {.byte_capacity = 1}, true);
    }
    u32 equal_pixels{};
    u32 changed_pixels{};
    const u64 pre_base = region.buffer_offset;
    const u64 post_base = static_cast<u64>(layout.second_plane_offset) + region.buffer_offset;
    for (u64 pixel = 0; pixel < pixels; ++pixel) {
        bool equal = true;
        const u64 pixel_offset = pixel * layout.bytes_per_pixel;
        for (u32 component = 0; component < layout.bytes_per_pixel; ++component) {
            if (!layout.raw_compare && component == 3) {
                continue;
            }
            if (authorized.bytes[pre_base + pixel_offset + component] !=
                authorized.bytes[post_base + pixel_offset + component]) {
                equal = false;
            }
        }
        equal ? ++equal_pixels : ++changed_pixels;
    }
    if (static_cast<u64>(equal_pixels) + changed_pixels != pixels) {
        return reject(true, FinalGuestSurfaceStatus::CapacityLoss, {.byte_capacity = 1}, true);
    }
    return {
        .sequence = sequence,
        .covered_pixel_count = static_cast<u32>(pixels),
        .equal_pixel_count = equal_pixels,
        .changed_pixel_count = changed_pixels,
        .status = FinalGuestSurfaceStatus::Complete,
        .available = true,
    };
}

struct PpUpstreamFeedbackFragmentExportTargetRelation {
    u64 sequence{};
    u32 covered_pixel_count{};
    u32 linear_exact_pixel_count{};
    u32 linear_close_pixel_count{};
    u32 srgb_exact_pixel_count{};
    u32 srgb_close_pixel_count{};
    u32 truncate_exact_pixel_count{};
    u32 target_lower_than_half_pixel_count{};
    u32 target_higher_than_half_pixel_count{};
    u32 neither_close_pixel_count{};
    u32 storage_encoding{};
    u32 storage_format_code{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool available{};
    bool retains_values{};
    bool affects_renderer_behavior{};
};

static_assert(std::is_trivially_copyable_v<PpUpstreamFeedbackFragmentExportTargetRelation>);
static_assert(sizeof(PpUpstreamFeedbackFragmentExportTargetRelation) <= 128);

[[nodiscard]] constexpr u32 ClassifyPpUpstreamFeedbackFragmentExportTargetStorageEncoding(
    vk::Format format) noexcept {
    switch (format) {
    case vk::Format::eR8Unorm:
        return 9;
    case vk::Format::eR8G8B8A8Unorm:
        return 1;
    case vk::Format::eR8G8B8A8Srgb:
        return 2;
    case vk::Format::eB8G8R8A8Unorm:
        return 3;
    case vk::Format::eB8G8R8A8Srgb:
        return 4;
    case vk::Format::eA2R10G10B10UnormPack32:
        return 5;
    case vk::Format::eA2B10G10R10UnormPack32:
        return 6;
    case vk::Format::eR16G16B16A16Unorm:
        return 7;
    case vk::Format::eR16G16B16A16Sfloat:
        return 8;
    default:
        return 0;
    }
}

[[nodiscard]] inline PpUpstreamFeedbackFragmentExportTargetRelation
ReducePpUpstreamFeedbackFragmentExportTargetRelation(
    u64 sequence, bool capture_fragment_export, u32 selected_candidate_index,
    const PpUpstreamFeedbackTeeAuthorizedTargetPrePostResult& authorized,
    const PpUpstreamSampleTeeDeferredObservation& finalized) noexcept {
    const u32 storage_encoding =
        ClassifyPpUpstreamFeedbackFragmentExportTargetStorageEncoding(
            authorized.storage_format);
    const u32 storage_format_code = static_cast<u32>(authorized.storage_format);
    const auto reject = [sequence, storage_encoding, storage_format_code](
                            FinalGuestSurfaceStatus status,
                            u32 FinalGuestSurfaceLoss::*member) {
        PpUpstreamFeedbackFragmentExportTargetRelation result{
            .sequence = sequence,
            .storage_encoding = storage_encoding,
            .storage_format_code = storage_format_code,
            .status = status,
        };
        result.loss.*member = 1;
        return result;
    };
    if (sequence == 0 || !capture_fragment_export || !authorized.publish ||
        authorized.status != FinalGuestSurfaceStatus::Complete || authorized.loss.Any() ||
        authorized.additional_gpu_copy_count != 0 || authorized.additional_command_count != 0 ||
        authorized.additional_barrier_count != 0 || authorized.additional_slot_count != 0 ||
        authorized.additional_release_count != 0 || authorized.retains_private_state) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }

    const auto& layout = authorized.layout;
    const bool semantic_format = layout.format == FinalGuestSurfaceFormat::Rgba8 ||
                                 layout.format == FinalGuestSurfaceFormat::Bgra8;
    const bool raw_r8_unorm = layout.format == FinalGuestSurfaceFormat::Unsupported &&
                              layout.raw_compare && layout.bytes_per_pixel == 1 &&
                              authorized.storage_format == vk::Format::eR8Unorm;
    const u64 expected_target_bytes = static_cast<u64>(layout.plane_bytes) * 2u;
    if ((!semantic_format && !raw_r8_unorm) ||
        (semantic_format && (layout.raw_compare || layout.bytes_per_pixel != 4)) ||
        layout.region_count != 1 || layout.plane_mask != 0x3u || layout.plane_bytes == 0 ||
        layout.second_plane_offset != layout.plane_bytes ||
        layout.total_bytes != expected_target_bytes ||
        authorized.bytes.size() != expected_target_bytes) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::unsupported_format);
    }
    const auto& region = layout.regions[0];
    if (region.logical_ordinal == 0 || region.buffer_offset != 0 || region.byte_size == 0 ||
        region.byte_size != layout.plane_bytes ||
        region.byte_size % layout.bytes_per_pixel != 0) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }

    const auto& lifecycle = finalized.lifecycle;
    const auto& publication = finalized.publication;
    const auto& owned = finalized.owned;
    const bool finalized_exact =
        selected_candidate_index < 4 && finalized.observe_owned && publication.publish &&
        publication.status == FinalGuestSurfaceStatus::Complete && !publication.loss.Any() &&
        lifecycle.status == FinalGuestSurfaceStatus::Complete && !lifecycle.loss.Any() &&
        lifecycle.finalized && !lifecycle.lifecycle_invalid &&
        lifecycle.mapped_invalidation_succeeded && lifecycle.bounded_record_copy_succeeded &&
        lifecycle.bounded_copy_completed_before_outer_slot_release &&
        lifecycle.expected_outer_slot_release_attempts == 1 &&
        lifecycle.observed_outer_slot_release_attempts == 1 && lifecycle.slot_release_succeeded &&
        finalized.record.candidate_index == selected_candidate_index &&
        IsCanonicalPpUpstreamSampleTeeCandidateRecord(finalized.record,
                                                       selected_candidate_index) &&
        owned.status == FinalGuestSurfaceStatus::Complete && !owned.loss.Any() &&
        owned.mapped_invalidation_succeeded && owned.bounded_record_copy_succeeded &&
        !owned.retains_mapped_span && owned.header == lifecycle.expected_header &&
        owned.records.size() == finalized.record.window.pixel_count &&
        publication.shader == finalized.record.shader &&
        publication.window == finalized.record.window;
    const u64 pixel_count = region.byte_size / layout.bytes_per_pixel;
    if (!finalized_exact || pixel_count == 0 || pixel_count > std::numeric_limits<u32>::max() ||
        owned.records.size() != pixel_count) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    for (const auto& record : owned.records) {
        if (record.writers != 1) {
            return reject(record.writers == 0 ? FinalGuestSurfaceStatus::GapLoss
                                               : FinalGuestSurfaceStatus::InvalidationLoss,
                          record.writers == 0 ? &FinalGuestSurfaceLoss::gap
                                               : &FinalGuestSurfaceLoss::invalidation);
        }
        for (const u32 value : record.value) {
            if (!std::isfinite(std::bit_cast<float>(value))) {
                return reject(FinalGuestSurfaceStatus::Unsupported,
                              &FinalGuestSurfaceLoss::unsupported_format);
            }
        }
    }

    const auto quantize = [](float value, bool srgb) {
        double encoded = std::clamp(static_cast<double>(value), 0.0, 1.0);
        if (srgb) {
            encoded = encoded <= 0.0031308
                          ? encoded * 12.92
                          : 1.055 * std::pow(encoded, 1.0 / 2.4) - 0.055;
        }
        return static_cast<u8>(std::clamp(std::floor(encoded * 255.0 + 0.5), 0.0, 255.0));
    };
    PpUpstreamFeedbackFragmentExportTargetRelation result{
        .sequence = sequence,
        .covered_pixel_count = static_cast<u32>(pixel_count),
        .storage_encoding = storage_encoding,
        .storage_format_code = static_cast<u32>(authorized.storage_format),
        .status = FinalGuestSurfaceStatus::Complete,
        .available = true,
    };
    const u64 post_base = layout.second_plane_offset;
    for (u64 pixel = 0; pixel < pixel_count; ++pixel) {
        bool linear_exact = true;
        bool linear_close = true;
        bool srgb_exact = true;
        bool srgb_close = true;
        bool truncate_exact = raw_r8_unorm;
        bool target_lower_than_half = false;
        bool target_higher_than_half = false;
        const u32 compared_components = raw_r8_unorm ? 1u : 3u;
        for (u32 target_component = 0; target_component < compared_components;
             ++target_component) {
            const u32 export_component =
                raw_r8_unorm
                    ? 0u
                    : layout.format == FinalGuestSurfaceFormat::Bgra8
                          ? 2u - target_component
                          : target_component;
            const float value = std::bit_cast<float>(owned.records[pixel].value[export_component]);
            const u8 actual = std::to_integer<u8>(
                authorized.bytes[post_base + pixel * layout.bytes_per_pixel + target_component]);
            const u8 expected_linear = quantize(value, false);
            const u8 expected_srgb = quantize(value, true);
            const u8 expected_truncate = static_cast<u8>(std::clamp(
                std::floor(std::clamp(static_cast<double>(value), 0.0, 1.0) * 255.0),
                0.0, 255.0));
            const u32 linear_delta = actual > expected_linear ? actual - expected_linear
                                                               : expected_linear - actual;
            const u32 srgb_delta = actual > expected_srgb ? actual - expected_srgb
                                                           : expected_srgb - actual;
            linear_exact &= linear_delta == 0;
            linear_close &= linear_delta <= 1;
            srgb_exact &= srgb_delta == 0;
            srgb_close &= srgb_delta <= 1;
            truncate_exact &= actual == expected_truncate;
            target_lower_than_half |= actual < expected_linear;
            target_higher_than_half |= actual > expected_linear;
        }
        result.linear_exact_pixel_count += linear_exact;
        result.linear_close_pixel_count += linear_close;
        result.srgb_exact_pixel_count += srgb_exact;
        result.srgb_close_pixel_count += srgb_close;
        result.truncate_exact_pixel_count += truncate_exact;
        result.target_lower_than_half_pixel_count += target_lower_than_half;
        result.target_higher_than_half_pixel_count += target_higher_than_half;
        result.neither_close_pixel_count += !linear_close && !srgb_close;
    }
    return result;
}

[[nodiscard]] inline std::string FormatPpUpstreamFeedbackFragmentExportTargetRelation(
    const PpUpstreamFeedbackFragmentExportTargetRelation& relation) {
    return "FGSCET s=" + std::to_string(relation.sequence) +
           " v=" + std::to_string(relation.available) +
           " cov=" + std::to_string(relation.covered_pixel_count) +
           " lx=" + std::to_string(relation.linear_exact_pixel_count) +
           " lc=" + std::to_string(relation.linear_close_pixel_count) +
           " sx=" + std::to_string(relation.srgb_exact_pixel_count) +
           " sc=" + std::to_string(relation.srgb_close_pixel_count) +
           " tr=" + std::to_string(relation.truncate_exact_pixel_count) +
           " dn=" + std::to_string(relation.target_lower_than_half_pixel_count) +
           " up=" + std::to_string(relation.target_higher_than_half_pixel_count) +
           " nx=" + std::to_string(relation.neither_close_pixel_count) +
           " ec=" + std::to_string(relation.storage_encoding) +
           " ef=" + std::to_string(relation.storage_format_code) +
           " st=" + std::to_string(static_cast<u32>(relation.status)) +
           " lm=" + std::to_string(FinalGuestSurfaceLossMask(relation.loss, 0));
}

} // namespace Vulkan::HostPasses
