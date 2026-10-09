// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "shader_recompiler/pp_upstream_sample_tee.h"
#include "video_core/renderer_vulkan/host_passes/pp_terminal_scope_content.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_refresh_result.h"
#include "video_core/texture_cache/sampler_observation.h"

namespace Vulkan::HostPasses {

struct PpUpstreamFeedbackPrePostDescriptor {
    bool enabled{};
    u32 logical_width{};
    u32 logical_height{};
    u32 source_width{};
    u32 source_height{};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    vk::Format storage_format{vk::Format::eUndefined};
    u32 raw_bytes_per_texel{};
    u32 samples{};
    u32 base_mip{};
    u32 mip_count{};
    u32 base_layer{};
    u32 layer_count{};
    bool color{};
    bool depth{};
    bool type_2d{};
    bool uniform_state{};
    FinalGuestSurfaceWatchOrdinals selector{};
    u32 candidate_capacity{};
    u32 buffer_alignment{};
    u32 max_regions{};
    u32 max_bytes{};
};

struct PpUpstreamFeedbackPrePostRegion {
    u32 logical_ordinal{};
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};
    u32 buffer_offset{};
    u32 byte_size{};

    bool operator==(const PpUpstreamFeedbackPrePostRegion&) const = default;
};

struct PpUpstreamFeedbackPrePostPlan {
    static constexpr u32 MaxCandidates = 4;
    static constexpr u32 MaxRegions = FinalGuestSurfaceWatchOrdinals::MaxOrdinals;

    std::array<PpUpstreamFeedbackPrePostRegion, MaxRegions> regions{};
    u32 region_count{};
    u32 plane_bytes{};
    u32 post_plane_offset{};
    u32 candidate_stride{};
    u32 candidate_capacity{};
    u32 total_bytes{};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    vk::Format storage_format{vk::Format::eUndefined};
    u32 bytes_per_pixel{4};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool raw_compare{};
    bool depth{};
    bool copy{};

    bool operator==(const PpUpstreamFeedbackPrePostPlan&) const = default;
};

struct PpUpstreamDepthFirstWriterPreObservation {
    bool recorded{};
    bool depth_stencil_exact{};
    bool same_scope{};
    bool scope_depth_history_valid{};
    bool first_depth_writer_valid{};
    u32 depth_writer_count{};
    bool depth_writer_count_overflow{};
};

struct PpUpstreamDepthFirstWriterPostObservation {
    bool recorded{};
    u64 current_mutation_epoch{};
    u64 current_backing_generation{};
    u64 current_rendering_serial{};
    VideoCore::ImageDepthWriterShape target_shape{};
    bool target_depth_write{};
    bool observed_valid{};
    u64 observed_mutation_epoch{};
    u64 observed_backing_generation{};
    u64 observed_rendering_serial{};
    VideoCore::ImageDepthWriterShape observed_shape{};
    bool observed_depth_write{};
    VideoCore::ImageDepthWriterScopeState scope{};
};

struct PpUpstreamDepthFirstWriterClassification {
    bool post_exact{};
    bool target_first_depth_writer{};
    bool target_scope_load{};
    bool target_scope_clear{};
};

[[nodiscard]] constexpr PpUpstreamDepthFirstWriterClassification
ClassifyPpUpstreamDepthFirstWriter(
    const PpUpstreamDepthFirstWriterPreObservation& pre,
    const PpUpstreamDepthFirstWriterPostObservation& post) noexcept {
    const bool post_exact = post.recorded && post.target_depth_write && post.observed_valid &&
                            post.observed_depth_write && post.current_mutation_epoch != 0 &&
                            post.current_backing_generation != 0 &&
                            post.current_rendering_serial != 0 &&
                            post.observed_mutation_epoch == post.current_mutation_epoch &&
                            post.observed_backing_generation == post.current_backing_generation &&
                            post.observed_rendering_serial == post.current_rendering_serial &&
                            post.target_shape.Valid() && post.observed_shape == post.target_shape;
    const bool pre_proves_no_prior_depth_writer =
        pre.recorded && pre.depth_stencil_exact &&
        (!pre.same_scope ||
         (pre.scope_depth_history_valid && !pre.first_depth_writer_valid &&
          pre.depth_writer_count == 0 &&
          !pre.depth_writer_count_overflow));
    const bool scope_exact =
        post.scope.valid && post.scope.rendering_serial == post.current_rendering_serial &&
        post.scope.backing_generation == post.current_backing_generation &&
        post.scope.latest_mutation_epoch == post.current_mutation_epoch &&
        post.scope.first_valid && !post.scope.writer_count_overflow &&
        post.scope.writer_count == 1 && post.scope.first_shape == post.target_shape;
    const bool target_first_depth_writer =
        post_exact && pre_proves_no_prior_depth_writer && scope_exact;
    return {
        .post_exact = post_exact,
        .target_first_depth_writer = target_first_depth_writer,
        .target_scope_load = target_first_depth_writer && !post.scope.depth_clear,
        .target_scope_clear = target_first_depth_writer && post.scope.depth_clear,
    };
}

struct PpUpstreamFeedbackPrePostCopy {
    u32 logical_ordinal{};
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};
    u32 buffer_offset{};
    u32 byte_size{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy{};
};

[[nodiscard]] inline PpUpstreamFeedbackPrePostPlan PlanPpUpstreamFeedbackPrePost(
    const PpUpstreamFeedbackPrePostDescriptor& descriptor) noexcept {
    PpUpstreamFeedbackPrePostPlan result{};
    if (!descriptor.enabled) {
        return result;
    }
    const auto reject = [](FinalGuestSurfaceStatus status, u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamFeedbackPrePostPlan rejected{};
        rejected.status = status;
        rejected.loss.*member = 1;
        return rejected;
    };
    if (descriptor.logical_width == 0 || descriptor.logical_height == 0 ||
        descriptor.source_width == 0 || descriptor.source_height == 0 ||
        descriptor.buffer_alignment == 0 || descriptor.candidate_capacity == 0) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    const bool semantic_format = descriptor.format == FinalGuestSurfaceFormat::Rgba8 ||
                                 descriptor.format == FinalGuestSurfaceFormat::Bgra8;
    const bool supported_raw_bytes = descriptor.raw_bytes_per_texel == 1 ||
                                     descriptor.raw_bytes_per_texel == 2 ||
                                     descriptor.raw_bytes_per_texel == 4 ||
                                     descriptor.raw_bytes_per_texel == 8 ||
                                     descriptor.raw_bytes_per_texel == 16;
    const bool raw_compare = descriptor.format == FinalGuestSurfaceFormat::Unsupported &&
                             supported_raw_bytes;
    const u32 bytes_per_pixel = raw_compare ? descriptor.raw_bytes_per_texel : 4;
    if ((!semantic_format && !raw_compare) ||
        (semantic_format && descriptor.raw_bytes_per_texel != 0) ||
        descriptor.buffer_alignment % 4 != 0 ||
        descriptor.buffer_alignment % bytes_per_pixel != 0 || descriptor.samples != 1 ||
        descriptor.base_mip != 0 || descriptor.mip_count != 1 || descriptor.base_layer != 0 ||
        descriptor.layer_count != 1 || descriptor.color == descriptor.depth ||
        (descriptor.depth && !raw_compare) || !descriptor.type_2d ||
        !descriptor.uniform_state) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::unsupported_format);
    }
    if (static_cast<u64>(descriptor.source_width) * descriptor.logical_height !=
        static_cast<u64>(descriptor.source_height) * descriptor.logical_width) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::logical_mapping);
    }

    const u32 window_width =
        std::min(descriptor.logical_width, FinalGuestSurfaceTilePlan::WindowExtent);
    const u32 window_height =
        std::min(descriptor.logical_height, FinalGuestSurfaceTilePlan::WindowExtent);
    const u32 columns =
        descriptor.logical_width <= window_width
            ? 1
            : (descriptor.logical_width - window_width) / FinalGuestSurfaceTilePlan::WindowStride +
                  1;
    const u32 rows = descriptor.logical_height <= window_height
                         ? 1
                         : (descriptor.logical_height - window_height) /
                                   FinalGuestSurfaceTilePlan::WindowStride +
                               1;
    const u64 window_count = static_cast<u64>(columns) * rows;
    const auto selector = ValidateFinalGuestSurfaceWatchOrdinals(
        descriptor.selector,
        window_count <= std::numeric_limits<u32>::max() ? static_cast<u32>(window_count) : 0u);
    if (selector.status != FinalGuestSurfaceStatus::Complete || selector.count == 0) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::ordinal_capacity);
    }
    const u64 requested_regions =
        static_cast<u64>(descriptor.candidate_capacity) * 2 * selector.count;
    if (descriptor.candidate_capacity > PpUpstreamFeedbackPrePostPlan::MaxCandidates ||
        descriptor.max_regions == 0 || requested_regions > descriptor.max_regions) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss,
                      descriptor.candidate_capacity > PpUpstreamFeedbackPrePostPlan::MaxCandidates
                          ? &FinalGuestSurfaceLoss::byte_capacity
                          : &FinalGuestSurfaceLoss::tile_capacity);
    }

    result.region_count = selector.count;
    result.candidate_capacity = descriptor.candidate_capacity;
    result.format = descriptor.format;
    result.storage_format = descriptor.storage_format;
    result.bytes_per_pixel = bytes_per_pixel;
    result.raw_compare = raw_compare;
    result.depth = descriptor.depth;
    u64 next_offset{};
    const auto maps_exactly = [](u32 coordinate, u32 source_extent, u32 logical_extent) {
        return static_cast<u64>(coordinate) * source_extent % logical_extent == 0;
    };
    for (u32 index = 0; index < selector.count; ++index) {
        const u32 zero_based = selector.ordinals[index] - 1;
        const u32 column = zero_based % columns;
        const u32 row = zero_based / columns;
        const u32 logical_x = column * FinalGuestSurfaceTilePlan::WindowStride;
        const u32 logical_y = row * FinalGuestSurfaceTilePlan::WindowStride;
        const u32 logical_end_x = logical_x + window_width;
        const u32 logical_end_y = logical_y + window_height;
        if (!maps_exactly(logical_x, descriptor.source_width, descriptor.logical_width) ||
            !maps_exactly(logical_end_x, descriptor.source_width, descriptor.logical_width) ||
            !maps_exactly(logical_y, descriptor.source_height, descriptor.logical_height) ||
            !maps_exactly(logical_end_y, descriptor.source_height, descriptor.logical_height)) {
            return reject(FinalGuestSurfaceStatus::Unsupported,
                          &FinalGuestSurfaceLoss::logical_mapping);
        }
        next_offset = AlignPpSourceBackingOffset(next_offset, descriptor.buffer_alignment);
        if (next_offset == std::numeric_limits<u64>::max()) {
            return reject(FinalGuestSurfaceStatus::CapacityLoss,
                          &FinalGuestSurfaceLoss::byte_capacity);
        }
        const u32 x = static_cast<u32>(static_cast<u64>(logical_x) * descriptor.source_width /
                                       descriptor.logical_width);
        const u32 y = static_cast<u32>(static_cast<u64>(logical_y) * descriptor.source_height /
                                       descriptor.logical_height);
        const u32 end_x = static_cast<u32>(static_cast<u64>(logical_end_x) *
                                           descriptor.source_width / descriptor.logical_width);
        const u32 end_y = static_cast<u32>(static_cast<u64>(logical_end_y) *
                                           descriptor.source_height / descriptor.logical_height);
        const u64 pixel_count = static_cast<u64>(end_x - x) * (end_y - y);
        if (pixel_count > std::numeric_limits<u64>::max() / bytes_per_pixel) {
            return reject(FinalGuestSurfaceStatus::CapacityLoss,
                          &FinalGuestSurfaceLoss::byte_capacity);
        }
        const u64 bytes = pixel_count * bytes_per_pixel;
        if (next_offset > std::numeric_limits<u32>::max() ||
            bytes > std::numeric_limits<u32>::max() - next_offset) {
            return reject(FinalGuestSurfaceStatus::CapacityLoss,
                          &FinalGuestSurfaceLoss::byte_capacity);
        }
        result.regions[index] = {
            .logical_ordinal = selector.ordinals[index],
            .x = x,
            .y = y,
            .width = end_x - x,
            .height = end_y - y,
            .buffer_offset = static_cast<u32>(next_offset),
            .byte_size = static_cast<u32>(bytes),
        };
        next_offset += bytes;
    }
    const u64 plane_bytes = AlignPpSourceBackingOffset(next_offset, descriptor.buffer_alignment);
    const u64 post_offset = plane_bytes;
    const u64 candidate_stride =
        AlignPpSourceBackingOffset(post_offset + plane_bytes, descriptor.buffer_alignment);
    const u64 total_bytes = candidate_stride * descriptor.candidate_capacity;
    if (plane_bytes == std::numeric_limits<u64>::max() ||
        candidate_stride == std::numeric_limits<u64>::max() || total_bytes > descriptor.max_bytes ||
        total_bytes > std::numeric_limits<u32>::max()) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss, &FinalGuestSurfaceLoss::byte_capacity);
    }
    result.plane_bytes = static_cast<u32>(plane_bytes);
    result.post_plane_offset = static_cast<u32>(post_offset);
    result.candidate_stride = static_cast<u32>(candidate_stride);
    result.total_bytes = static_cast<u32>(total_bytes);
    result.status = FinalGuestSurfaceStatus::Complete;
    result.copy = true;
    return result;
}

[[nodiscard]] constexpr PpUpstreamFeedbackPrePostCopy ResolvePpUpstreamFeedbackPrePostCopy(
    const PpUpstreamFeedbackPrePostPlan& plan, u32 candidate_index, bool post,
    u32 region_index) noexcept {
    if (plan.status != FinalGuestSurfaceStatus::Complete || plan.loss.Any() || !plan.copy ||
        candidate_index >= plan.candidate_capacity || region_index >= plan.region_count) {
        return {
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
        };
    }
    const auto& region = plan.regions[region_index];
    return {
        .logical_ordinal = region.logical_ordinal,
        .x = region.x,
        .y = region.y,
        .width = region.width,
        .height = region.height,
        .buffer_offset = candidate_index * plan.candidate_stride +
                         (post ? plan.post_plane_offset : 0) + region.buffer_offset,
        .byte_size = region.byte_size,
        .status = FinalGuestSurfaceStatus::Complete,
        .copy = true,
    };
}

struct PpUpstreamFeedbackPrePostCompactResult {
    std::vector<std::byte> bytes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
};

[[nodiscard]] inline PpUpstreamFeedbackPrePostCompactResult CompactPpUpstreamFeedbackPrePost(
    const PpUpstreamFeedbackPrePostPlan& plan, u32 candidate_index,
    std::span<const std::byte> slot) {
    if (plan.status != FinalGuestSurfaceStatus::Complete || plan.loss.Any() || !plan.copy ||
        candidate_index >= plan.candidate_capacity) {
        return {.status = FinalGuestSurfaceStatus::InvalidationLoss, .loss = {.invalidation = 1}};
    }
    const u64 source_base = static_cast<u64>(candidate_index) * plan.candidate_stride;
    const u64 source_end = source_base + plan.post_plane_offset + plan.plane_bytes;
    if (plan.plane_bytes == 0 || plan.post_plane_offset < plan.plane_bytes ||
        source_end > slot.size()) {
        return {.status = FinalGuestSurfaceStatus::InvalidationLoss, .loss = {.invalidation = 1}};
    }
    PpUpstreamFeedbackPrePostCompactResult result{
        .bytes = std::vector<std::byte>(plan.plane_bytes * 2u),
        .status = FinalGuestSurfaceStatus::Complete,
    };
    const auto pre = slot.subspan(source_base, plan.plane_bytes);
    const auto post = slot.subspan(source_base + plan.post_plane_offset, plan.plane_bytes);
    std::ranges::copy(pre, result.bytes.begin());
    std::ranges::copy(post, result.bytes.begin() + plan.plane_bytes);
    return result;
}

struct PpUpstreamFeedbackPrePostRegistryAction {
    u32 candidate_index{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool capture{};
    bool restart{};
};

[[nodiscard]] constexpr bool IsExactPpDepthPredecessorRawPrefixRestart(
    const PpUpstreamFeedbackPrePostRegistryAction& action) noexcept {
    return action.capture && action.restart &&
           action.status == FinalGuestSurfaceStatus::Complete && !action.loss.Any() &&
           action.candidate_index < PpUpstreamFeedbackPrePostPlan::MaxCandidates;
}

struct PpUpstreamFeedbackPrePostRegistryResolution {
    u32 candidate_index{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool matched{};
};

class PpUpstreamFeedbackPrePostRegistry {
public:
    explicit constexpr PpUpstreamFeedbackPrePostRegistry(PpTerminalScopeDrawSelector selector_,
                                                         u32 capacity_) noexcept
        : configured_selector{selector_}, selector{selector_}, capacity{capacity_} {}

    [[nodiscard]] constexpr PpUpstreamFeedbackPrePostRegistryAction Preview(
        VideoCore::ImageColorScopePrivateLink output,
        const PpTerminalScopeDrawSelector& draw) noexcept {
        if (!MatchesPpTerminalScopeDraw(selector, draw)) {
            return {};
        }
        if (!output.Valid() || capacity == 0 || capacity > entries.size()) {
            return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        for (u32 index = 0; index < capacity; ++index) {
            auto& entry = entries[index];
            if (entry.active && entry.output == output) {
                if (rolling_completed_same_link && entry.before && entry.after) {
                    entry.before = true;
                    entry.after = false;
                    return {.candidate_index = index,
                            .status = FinalGuestSurfaceStatus::Complete,
                            .capture = true,
                            .restart = true};
                }
                return {.candidate_index = index,
                        .status = FinalGuestSurfaceStatus::GapLoss,
                        .loss = {.gap = 1}};
            }
        }
        for (u32 index = 0; index < capacity; ++index) {
            auto& entry = entries[index];
            if (!entry.active) {
                entry = {.output = output, .active = true, .before = true};
                return {.candidate_index = index,
                        .status = FinalGuestSurfaceStatus::Complete,
                        .capture = true};
            }
        }
        capacity_loss = true;
        return {.status = FinalGuestSurfaceStatus::CapacityLoss, .loss = {.tile_capacity = 1}};
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackPrePostRegistryAction Complete(
        VideoCore::ImageColorScopePrivateLink output,
        const PpTerminalScopeDrawSelector& draw) noexcept {
        if (!MatchesPpTerminalScopeDraw(selector, draw)) {
            return {};
        }
        for (u32 index = 0; index < capacity && index < entries.size(); ++index) {
            auto& entry = entries[index];
            if (entry.active && entry.output == output) {
                if (!entry.before || entry.after) {
                    return {.candidate_index = index,
                            .status = FinalGuestSurfaceStatus::GapLoss,
                            .loss = {.gap = 1}};
                }
                entry.after = true;
                return {.candidate_index = index,
                        .status = FinalGuestSurfaceStatus::Complete,
                        .capture = true};
            }
        }
        return {.status = FinalGuestSurfaceStatus::GapLoss, .loss = {.gap = 1}};
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackPrePostRegistryResolution Resolve(
        VideoCore::ImageColorScopePrivateLink output) const noexcept {
        if (!output.Valid()) {
            return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        if (capacity_loss) {
            return {.status = FinalGuestSurfaceStatus::CapacityLoss, .loss = {.tile_capacity = 1}};
        }
        for (u32 index = 0; index < capacity && index < entries.size(); ++index) {
            const auto& entry = entries[index];
            if (entry.active && entry.output == output) {
                if (!entry.before || !entry.after) {
                    return {.candidate_index = index,
                            .status = FinalGuestSurfaceStatus::GapLoss,
                            .loss = {.gap = 1}};
                }
                return {.candidate_index = index,
                        .status = FinalGuestSurfaceStatus::Complete,
                        .matched = true};
            }
        }
        return {};
    }

    constexpr void Reset() noexcept {
        selector = configured_selector;
        entries = {};
        capacity_loss = false;
        rolling_completed_same_link = false;
    }

    constexpr void ResetForRolling(PpTerminalScopeDrawSelector selector_) noexcept {
        selector = selector_;
        entries = {};
        capacity_loss = false;
        rolling_completed_same_link = true;
    }

private:
    struct Entry {
        VideoCore::ImageColorScopePrivateLink output{};
        bool active{};
        bool before{};
        bool after{};
    };

    PpTerminalScopeDrawSelector configured_selector{};
    PpTerminalScopeDrawSelector selector{};
    u32 capacity{};
    std::array<Entry, PpUpstreamFeedbackPrePostPlan::MaxCandidates> entries{};
    bool capacity_loss{};
    bool rolling_completed_same_link{};
};

struct PpUpstreamFeedbackPrePostTakeResult {
    PpUpstreamFeedbackPrePostPlan plan{};
    u32 candidate_index{};
    u32 recorded_plane_mask{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy{};
    bool cpu_wait{};
    bool finish{};
    bool retains_image{};
    bool retains_vk_image{};
};

enum class PpUpstreamFeedbackPrePostTakeFirstFailure : u8 {
    Idle = 0,
    NotArmed = 1,
    PreexistingFrameLoss = 2,
    MissingSelection = 3,
    CaptureCardinality = 4,
    MissingSlot = 5,
    InvalidTransition = 6,
    Exact = 7,
};

struct PpUpstreamFeedbackPrePostTakeEvaluation {
    PpUpstreamFeedbackPrePostTakeResult take{};
    FinalGuestSurfaceStatus raw_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss raw_loss{};
    u32 recorded_plane_mask{};
    PpUpstreamFeedbackPrePostTakeFirstFailure first_failure{
        PpUpstreamFeedbackPrePostTakeFirstFailure::Idle};
    bool armed{};
    bool selected{};
    bool has_slot{};
};

[[nodiscard]] constexpr bool IsCanonicalPpUpstreamFeedbackPrePostFrameState(
    FinalGuestSurfaceStatus status, const FinalGuestSurfaceLoss& loss) noexcept {
    const u32 mask = FinalGuestSurfaceLossMask(loss, 0);
    switch (status) {
    case FinalGuestSurfaceStatus::Complete:
        return mask == 0;
    case FinalGuestSurfaceStatus::Unsupported:
        return mask != 0 &&
               (mask & ~((1u << 8) - 1u | (1u << 10))) == 0;
    case FinalGuestSurfaceStatus::CapacityLoss:
        return mask != 0 &&
               (mask & ~((1u << 8) | (1u << 9) | (1u << 10))) == 0;
    case FinalGuestSurfaceStatus::BusyLoss:
        return mask == (1u << 11);
    case FinalGuestSurfaceStatus::InvalidationLoss:
        return mask == (1u << 12);
    case FinalGuestSurfaceStatus::GapLoss:
        return mask == (1u << 13);
    default:
        return false;
    }
}

[[nodiscard]] constexpr PpUpstreamFeedbackPrePostTakeEvaluation
EvaluatePpUpstreamFeedbackPrePostTake(PpUpstreamFeedbackPrePostTakeResult raw, bool armed,
                                      bool selected, bool has_slot) noexcept {
    using FirstFailure = PpUpstreamFeedbackPrePostTakeFirstFailure;
    PpUpstreamFeedbackPrePostTakeEvaluation evaluation{
        .take = raw,
        .raw_status = raw.status,
        .raw_loss = raw.loss,
        .recorded_plane_mask = raw.recorded_plane_mask,
        .first_failure = FirstFailure::Idle,
        .armed = armed,
        .selected = selected,
        .has_slot = has_slot,
    };

    if (!armed) {
        evaluation.take.status = FinalGuestSurfaceStatus::InvalidationLoss;
        evaluation.take.loss = {.invalidation = 1};
    } else if (evaluation.take.status == FinalGuestSurfaceStatus::Complete &&
               (!selected || evaluation.take.recorded_plane_mask != 0x3u)) {
        evaluation.take.status = FinalGuestSurfaceStatus::GapLoss;
        evaluation.take.loss = {.gap = 1};
    } else if (evaluation.take.status == FinalGuestSurfaceStatus::Complete && !has_slot) {
        evaluation.take.status = FinalGuestSurfaceStatus::BusyLoss;
        evaluation.take.loss = {.busy = 1};
    }
    evaluation.take.copy = evaluation.take.status == FinalGuestSurfaceStatus::Complete &&
                           !evaluation.take.loss.Any() && selected && has_slot &&
                           evaluation.take.plan.copy;

    if (!armed) {
        evaluation.first_failure = FirstFailure::NotArmed;
    } else if (!IsCanonicalPpUpstreamFeedbackPrePostFrameState(raw.status, raw.loss)) {
        evaluation.first_failure = FirstFailure::InvalidTransition;
    } else if (raw.status != FinalGuestSurfaceStatus::Complete) {
        evaluation.first_failure = FirstFailure::PreexistingFrameLoss;
    } else if (!selected) {
        evaluation.first_failure = FirstFailure::MissingSelection;
    } else if ((raw.recorded_plane_mask & ~0x3u) != 0) {
        evaluation.first_failure = FirstFailure::InvalidTransition;
    } else if (raw.recorded_plane_mask != 0x3u) {
        evaluation.first_failure = FirstFailure::CaptureCardinality;
    } else if (!has_slot) {
        evaluation.first_failure = FirstFailure::MissingSlot;
    } else if (raw.plan.status != FinalGuestSurfaceStatus::Complete || raw.plan.loss.Any() ||
               !raw.plan.copy || raw.plan.candidate_capacity == 0 ||
               raw.plan.candidate_capacity > PpUpstreamFeedbackPrePostPlan::MaxCandidates ||
               raw.candidate_index >= raw.plan.candidate_capacity) {
        evaluation.first_failure = FirstFailure::InvalidTransition;
    } else {
        evaluation.first_failure = FirstFailure::Exact;
    }
    return evaluation;
}

struct PpUpstreamFeedbackPrePostTakeReport {
    u64 sequence{};
    FinalGuestSurfaceStatus raw_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss raw_loss{};
    u32 recorded_plane_mask{};
    PpUpstreamFeedbackPrePostTakeFirstFailure first_failure{
        PpUpstreamFeedbackPrePostTakeFirstFailure::Idle};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    u32 additional_ownership_transfer_count{};
    bool armed{};
    bool selected{};
    bool has_slot{};
    bool retains_private_state{};
    bool retains_image{};
    bool retains_vk_image{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
};

[[nodiscard]] constexpr PpUpstreamFeedbackPrePostTakeReport
MakePpUpstreamFeedbackPrePostTakeReport(
    u64 sequence, const PpUpstreamFeedbackPrePostTakeEvaluation& evaluation) noexcept {
    return {
        .sequence = sequence,
        .raw_status = evaluation.raw_status,
        .raw_loss = evaluation.raw_loss,
        .recorded_plane_mask = evaluation.recorded_plane_mask,
        .first_failure = evaluation.first_failure,
        .armed = evaluation.armed,
        .selected = evaluation.selected,
        .has_slot = evaluation.has_slot,
    };
}

[[nodiscard]] inline std::string FormatPpUpstreamFeedbackPrePostTakeReport(
    const PpUpstreamFeedbackPrePostTakeReport& report) {
    return "FGSCTUF s=" + std::to_string(report.sequence) +
           " rs=" + std::to_string(static_cast<u32>(report.raw_status)) +
           " rlm=" + std::to_string(FinalGuestSurfaceLossMask(report.raw_loss, 0)) +
           " a=" + std::to_string(report.armed ? 1 : 0) +
           " se=" + std::to_string(report.selected ? 1 : 0) +
           " hs=" + std::to_string(report.has_slot ? 1 : 0) +
           " pm=" + std::to_string(report.recorded_plane_mask) +
           " ff=" + std::to_string(static_cast<u32>(report.first_failure));
}

class PpUpstreamFeedbackPrePostFrame {
public:
    explicit constexpr PpUpstreamFeedbackPrePostFrame(PpTerminalScopeDrawSelector selector,
                                                      u32 capacity_) noexcept
        : registry{selector, capacity_}, capacity{capacity_} {}

    [[nodiscard]] constexpr bool Arm() noexcept {
        registry.Reset();
        return ArmAfterReset(true);
    }

    [[nodiscard]] constexpr bool Arm(PpTerminalScopeDrawSelector selector) noexcept {
        registry.ResetForRolling(selector);
        return ArmAfterReset(MatchesPpTerminalScopeDraw(selector, selector));
    }

    [[nodiscard]] constexpr bool Configure(
        const PpUpstreamFeedbackPrePostPlan& candidate_plan) noexcept {
        if (!armed || candidate_plan.status != FinalGuestSurfaceStatus::Complete ||
            candidate_plan.loss.Any() || !candidate_plan.copy ||
            candidate_plan.candidate_capacity != capacity) {
            Fail(FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1});
            return false;
        }
        if (plan.copy && plan != candidate_plan) {
            Fail(FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1});
            return false;
        }
        plan = candidate_plan;
        return true;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackPrePostRegistryAction Preview(
        VideoCore::ImageColorScopePrivateLink output,
        const PpTerminalScopeDrawSelector& draw) noexcept {
        if (!armed || status != FinalGuestSurfaceStatus::Complete || !plan.copy) {
            return {.status = status, .loss = loss};
        }
        const auto action = registry.Preview(output, draw);
        if (action.restart && action.candidate_index < copied_masks.size()) {
            copied_masks[action.candidate_index] = 0;
            if (selected && selected_index == action.candidate_index) {
                selected = false;
            }
        }
        Apply(action.status, action.loss);
        return action;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackPrePostRegistryAction Complete(
        VideoCore::ImageColorScopePrivateLink output,
        const PpTerminalScopeDrawSelector& draw) noexcept {
        if (!armed || status != FinalGuestSurfaceStatus::Complete || !plan.copy) {
            return {.status = status, .loss = loss};
        }
        const auto action = registry.Complete(output, draw);
        Apply(action.status, action.loss);
        return action;
    }

    constexpr void MarkCopied(u32 candidate_index, bool post) noexcept {
        if (!armed || candidate_index >= capacity || candidate_index >= copied_masks.size()) {
            Fail(FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1});
            return;
        }
        copied_masks[candidate_index] |= post ? 0x2u : 0x1u;
    }

    [[nodiscard]] constexpr u32 RecordedPlaneMask(u32 candidate_index) const noexcept {
        if (!armed || candidate_index >= capacity || candidate_index >= copied_masks.size()) {
            return 0;
        }
        return copied_masks[candidate_index];
    }

    constexpr void Reject(FinalGuestSurfaceStatus failed_status,
                          FinalGuestSurfaceLoss failed_loss) noexcept {
        if (failed_status == FinalGuestSurfaceStatus::Complete || !failed_loss.Any()) {
            failed_status = FinalGuestSurfaceStatus::InvalidationLoss;
            failed_loss = {.invalidation = 1};
        }
        Fail(failed_status, failed_loss);
    }

    [[nodiscard]] constexpr bool Select(VideoCore::ImageColorScopePrivateLink output) noexcept {
        if (!armed || status != FinalGuestSurfaceStatus::Complete || !plan.copy) {
            return false;
        }
        const auto resolution = registry.Resolve(output);
        if (!resolution.matched) {
            const auto failed_status = resolution.status == FinalGuestSurfaceStatus::AlreadyConsumed
                                           ? FinalGuestSurfaceStatus::GapLoss
                                           : resolution.status;
            auto failed_loss = resolution.loss;
            if (!failed_loss.Any()) {
                failed_loss.gap = 1;
            }
            Fail(failed_status, failed_loss);
            return false;
        }
        if (resolution.candidate_index >= capacity ||
            copied_masks[resolution.candidate_index] != 0x3u) {
            Fail(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
            return false;
        }
        selected_index = resolution.candidate_index;
        selected = true;
        return true;
    }

    [[nodiscard]] constexpr bool SelectCandidate(u32 candidate_index) noexcept {
        if (!armed || status != FinalGuestSurfaceStatus::Complete || !plan.copy ||
            candidate_index >= capacity || candidate_index >= copied_masks.size() ||
            copied_masks[candidate_index] != 0x3u) {
            Fail(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
            return false;
        }
        selected_index = candidate_index;
        selected = true;
        return true;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackPrePostTakeEvaluation TakeEvaluation(
        bool has_slot) noexcept {
        const PpUpstreamFeedbackPrePostTakeResult raw{
            .plan = plan,
            .candidate_index = selected_index,
            .recorded_plane_mask =
                selected_index < copied_masks.size() ? copied_masks[selected_index] : 0,
            .status = status,
            .loss = loss,
        };
        auto evaluation =
            EvaluatePpUpstreamFeedbackPrePostTake(raw, armed, selected, has_slot);
        armed = false;
        selected = false;
        return evaluation;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackPrePostTakeResult Take(bool has_slot) noexcept {
        return TakeEvaluation(has_slot).take;
    }

private:
    [[nodiscard]] constexpr bool ArmAfterReset(bool selector_valid) noexcept {
        plan = {};
        copied_masks = {};
        selected_index = 0;
        selected = false;
        armed = selector_valid && capacity != 0 && capacity <= copied_masks.size();
        status =
            armed ? FinalGuestSurfaceStatus::Complete : FinalGuestSurfaceStatus::InvalidationLoss;
        loss = armed ? FinalGuestSurfaceLoss{} : FinalGuestSurfaceLoss{.invalidation = 1};
        return armed;
    }

    constexpr void Apply(FinalGuestSurfaceStatus candidate_status,
                         FinalGuestSurfaceLoss candidate_loss) noexcept {
        if (candidate_status == FinalGuestSurfaceStatus::Complete ||
            candidate_status == FinalGuestSurfaceStatus::AlreadyConsumed) {
            return;
        }
        if (!candidate_loss.Any()) {
            candidate_loss.invalidation = 1;
        }
        Fail(candidate_status, candidate_loss);
    }

    constexpr void Fail(FinalGuestSurfaceStatus failed_status,
                        FinalGuestSurfaceLoss failed_loss) noexcept {
        status = failed_status;
        loss = failed_loss;
        selected = false;
    }

    PpUpstreamFeedbackPrePostRegistry registry;
    u32 capacity{};
    PpUpstreamFeedbackPrePostPlan plan{};
    std::array<u32, PpUpstreamFeedbackPrePostPlan::MaxCandidates> copied_masks{};
    u32 selected_index{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool armed{};
    bool selected{};
};

enum class PpUpstreamFeedbackSelectionAuthorityDisposition : u8 {
    Idle = 0,
    NoOpportunity = 1,
    InputUnavailable = 2,
    RegistryUnmatched = 3,
    FrameSelectRejected = 4,
    Selected = 5,
    InvalidTransition = 6,
};

enum class PpUpstreamFeedbackSelectionAuthorityMode : u8 {
    Exact = 0,
    Disabled = 1,
    DepthPrePost = 2,
    PredecessorHint = 3,
};

enum class PpUpstreamFeedbackProducerAdmissionDisposition : u8 {
    Idle = 0,
    SelectorMismatch = 1,
    InvalidOutput = 2,
    SampledInputStatus = 3,
    SampledInputLoss = 4,
    SampledInputCountMismatch = 5,
    Capacity = 6,
    Stored = 7,
    InvalidTransition = 8,
};

enum class PpUpstreamFeedbackProducerCandidateDisposition : u8 {
    Idle = 0,
    NotSingleCandidate = 1,
    UnknownProducer = 2,
    InvalidColorScope = 3,
    Exact = 4,
    InvalidTransition = 5,
};

enum class PpUpstreamFeedbackProducerDepthConsumerDisposition : u8 {
    Idle = 0,
    NotSingleCandidate = 1,
    MissingWriter = 2,
    StaleWriter = 3,
    ResourceMismatch = 4,
    BackingMismatch = 5,
    MutationMismatch = 6,
    SubresourceMismatch = 7,
    UnsupportedState = 8,
    AlreadyReadable = 9,
    MissingBarrier = 10,
    PartialBarrier = 11,
    LiveStateMismatch = 12,
    Exact = 13,
    InvalidTransition = 14,
};

struct PpUpstreamFeedbackSampledImageTransitFact {
    u64 selected_subresource_mask{};
    u64 prior_depth_stencil_write_mask{};
    u64 already_readable_mask{};
    u64 covering_barrier_mask{};
    u64 source_depth_stencil_layout_cover_mask{};
    u64 source_depth_stencil_access_cover_mask{};
    u64 source_depth_stencil_stage_cover_mask{};
    u64 destination_layout_cover_mask{};
    u64 destination_access_cover_mask{};
    u64 destination_stage_cover_mask{};
    u64 post_readable_mask{};
    u32 selected_subresource_count{};
    u32 prior_depth_stencil_write_count{};
    u32 already_readable_count{};
    u32 covering_barrier_count{};
    u32 source_depth_stencil_layout_cover_count{};
    u32 source_depth_stencil_access_cover_count{};
    u32 source_depth_stencil_stage_cover_count{};
    u32 destination_layout_cover_count{};
    u32 destination_access_cover_count{};
    u32 destination_stage_cover_count{};
    u32 post_readable_count{};
    u32 barrier_count{};
    bool writer_valid{};
    bool writer_depth_write{};
    bool resource_exact{};
    bool backing_exact{};
    bool mutation_exact{};
    bool depth_aspect_exact{};
    bool selected_cardinality_exact{};
    bool selected_subresource_exact{};
    bool destination_shader_read{};
    bool destination_supported{};
    bool uniform_before{};
    bool uniform_after{};
    bool transit_invoked{};
    bool submitted{};
    bool malformed{};
    bool overflowed{};

    bool operator==(const PpUpstreamFeedbackSampledImageTransitFact&) const = default;
};

[[nodiscard]] constexpr PpUpstreamFeedbackSampledImageTransitFact
MakePpUpstreamFeedbackSampledImageTransitFact(
    const VideoCore::ImageTransitBarrierReceipt& receipt,
    const VideoCore::ImageColorScopeTrackedView& bound_view,
    const VideoCore::ImageDepthStencilProducerObservation& bound_writer) noexcept {
    const VideoCore::SubresourceRange selected{
        .base = {.level = bound_view.base_mip, .layer = bound_view.base_layer},
        .extent = {.levels = bound_view.mip_count, .layers = bound_view.layer_count},
    };
    const u64 selected_count = static_cast<u64>(bound_view.mip_count) * bound_view.layer_count;
    const u64 dense_mask =
        selected_count == VideoCore::ImageTransitBarrierReceiptMaxSubresources
            ? std::numeric_limits<u64>::max()
            : selected_count < VideoCore::ImageTransitBarrierReceiptMaxSubresources
                  ? (selected_count == 0 ? 0 : (u64{1} << selected_count) - 1)
                  : 0;
    constexpr u32 DepthAspect = static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT);
    return {
        .selected_subresource_mask = receipt.selected_subresource_mask,
        .prior_depth_stencil_write_mask = receipt.prior_depth_stencil_write_mask,
        .already_readable_mask = receipt.already_readable_mask,
        .covering_barrier_mask = receipt.covering_barrier_mask,
        .source_depth_stencil_layout_cover_mask =
            receipt.source_depth_stencil_layout_cover_mask,
        .source_depth_stencil_access_cover_mask =
            receipt.source_depth_stencil_access_cover_mask,
        .source_depth_stencil_stage_cover_mask = receipt.source_depth_stencil_stage_cover_mask,
        .destination_layout_cover_mask = receipt.destination_layout_cover_mask,
        .destination_access_cover_mask = receipt.destination_access_cover_mask,
        .destination_stage_cover_mask = receipt.destination_stage_cover_mask,
        .post_readable_mask = receipt.post_readable_mask,
        .selected_subresource_count = receipt.selected_subresource_count,
        .prior_depth_stencil_write_count = receipt.prior_depth_stencil_write_count,
        .already_readable_count = receipt.already_readable_count,
        .covering_barrier_count = receipt.covering_barrier_count,
        .source_depth_stencil_layout_cover_count =
            receipt.source_depth_stencil_layout_cover_count,
        .source_depth_stencil_access_cover_count =
            receipt.source_depth_stencil_access_cover_count,
        .source_depth_stencil_stage_cover_count =
            receipt.source_depth_stencil_stage_cover_count,
        .destination_layout_cover_count = receipt.destination_layout_cover_count,
        .destination_access_cover_count = receipt.destination_access_cover_count,
        .destination_stage_cover_count = receipt.destination_stage_cover_count,
        .post_readable_count = receipt.post_readable_count,
        .barrier_count = receipt.barrier_count,
        .writer_valid = receipt.transition_writer.valid,
        .writer_depth_write = receipt.transition_writer.depth_write && bound_writer.depth_write,
        .resource_exact = bound_view.valid && bound_writer.valid &&
                          receipt.transition_writer.link == bound_view.link &&
                          bound_writer.link == bound_view.link,
        .backing_exact = bound_view.backing_generation != 0 &&
                         receipt.transition_writer.backing_generation ==
                             bound_view.backing_generation &&
                         bound_writer.backing_generation == bound_view.backing_generation,
        .mutation_exact = bound_view.mutation_epoch != 0 &&
                          receipt.transition_writer.mutation_epoch == bound_view.mutation_epoch &&
                          bound_writer.mutation_epoch == bound_view.mutation_epoch,
        .depth_aspect_exact = bound_view.aspect_mask == DepthAspect &&
                              (receipt.request_aspect_mask & DepthAspect) != 0 &&
                              (receipt.transition_writer.attachment_aspect_mask & DepthAspect) !=
                                  0 &&
                              (bound_writer.attachment_aspect_mask & DepthAspect) != 0,
        .selected_cardinality_exact =
            selected_count != 0 &&
            selected_count <= VideoCore::ImageTransitBarrierReceiptMaxSubresources &&
            receipt.selected_subresource_count == selected_count &&
            receipt.selected_subresource_mask == dense_mask,
        .selected_subresource_exact =
            selected_count != 0 &&
            selected_count <= VideoCore::ImageTransitBarrierReceiptMaxSubresources &&
            receipt.selected_subresource_count == selected_count &&
            receipt.selected_subresource_mask == dense_mask && selected == receipt.request_range &&
            receipt.transition_writer.range == receipt.request_range &&
            bound_writer.range == receipt.request_range,
        .destination_shader_read = receipt.destination_shader_read,
        .destination_supported = receipt.destination_supported,
        .uniform_before = receipt.uniform_before,
        .uniform_after = receipt.uniform_after,
        .transit_invoked = receipt.transit_invoked,
        .submitted = receipt.submitted,
        .malformed = receipt.malformed,
        .overflowed = receipt.overflowed,
    };
}

struct PpUpstreamFeedbackSampledImageTransitObservation {
    PpUpstreamFeedbackSampledImageTransitFact fact{};
    u32 binding_count{};
    bool duplicate_drift{};
    bool overflowed{};
};

class PpUpstreamFeedbackSampledImageTransitAccumulator {
public:
    static constexpr u32 MaxBindingCount = 255;

    [[nodiscard]] constexpr bool ObserveFirst(
        const PpUpstreamFeedbackSampledImageTransitFact& fact) noexcept {
        if (value.binding_count != 0) {
            value.duplicate_drift = true;
            value.overflowed |= fact.overflowed;
            Increment();
            return false;
        }
        value.fact = fact;
        value.binding_count = 1;
        value.overflowed = fact.overflowed;
        return true;
    }

    [[nodiscard]] constexpr bool ObserveDuplicate(
        const PpUpstreamFeedbackSampledImageTransitFact& fact) noexcept {
        if (value.binding_count == 0) {
            value.fact = fact;
            value.binding_count = 1;
            value.duplicate_drift = true;
            value.overflowed = fact.overflowed;
            return false;
        }
        Increment();
        const auto& first = value.fact;
        const bool same_source =
            fact.selected_subresource_mask == first.selected_subresource_mask &&
            fact.selected_subresource_count == first.selected_subresource_count &&
            fact.writer_valid == first.writer_valid &&
            fact.writer_depth_write == first.writer_depth_write &&
            fact.resource_exact == first.resource_exact &&
            fact.backing_exact == first.backing_exact &&
            fact.mutation_exact == first.mutation_exact &&
            fact.depth_aspect_exact == first.depth_aspect_exact &&
            fact.selected_cardinality_exact == first.selected_cardinality_exact &&
            fact.selected_subresource_exact == first.selected_subresource_exact &&
            fact.destination_shader_read == first.destination_shader_read &&
            fact.destination_supported == first.destination_supported;
        const bool canonical_noop =
            fact.transit_invoked && !fact.submitted && !fact.malformed && !fact.overflowed &&
            fact.destination_supported && fact.barrier_count == 0 &&
            fact.prior_depth_stencil_write_mask == 0 &&
            fact.already_readable_mask == fact.selected_subresource_mask &&
            fact.post_readable_mask == fact.selected_subresource_mask &&
            fact.covering_barrier_mask == 0 &&
            fact.source_depth_stencil_layout_cover_mask == 0 &&
            fact.source_depth_stencil_access_cover_mask == 0 &&
            fact.source_depth_stencil_stage_cover_mask == 0 &&
            fact.destination_layout_cover_mask == 0 &&
            fact.destination_access_cover_mask == 0 &&
            fact.destination_stage_cover_mask == 0;
        if (!same_source || !canonical_noop) {
            value.duplicate_drift = true;
        }
        value.overflowed |= fact.overflowed;
        return same_source && canonical_noop;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackSampledImageTransitObservation Peek() const noexcept {
        return value;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackSampledImageTransitObservation Take() noexcept {
        const auto taken = value;
        value = {};
        return taken;
    }

private:
    constexpr void Increment() noexcept {
        if (value.binding_count == MaxBindingCount) {
            value.overflowed = true;
        } else {
            ++value.binding_count;
        }
    }

    PpUpstreamFeedbackSampledImageTransitObservation value{};
};

struct PpUpstreamFeedbackProducerDepthConsumerInput {
    VideoCore::ImageColorScopeTrackedView selected_view{};
    VideoCore::ImageDepthStencilProducerObservation current_writer{};
    PpUpstreamFeedbackSampledImageTransitObservation transit{};
    VideoCore::ImageReadStateObservation live_state{};
};

struct PpUpstreamFeedbackProducerDepthConsumerDescriptor {
    u32 expected_input_count{};
    u32 candidate_count{};
    VideoCore::ImageColorScopeTrackedView selected_view{};
    VideoCore::ImageDepthStencilProducerObservation current_writer{};
    PpUpstreamFeedbackSampledImageTransitObservation transit{};
    VideoCore::ImageReadStateObservation live_state{};
};

struct PpUpstreamFeedbackProducerDepthConsumerSnapshot {
    u32 selected_subresource_count{};
    u32 prior_depth_stencil_write_count{};
    u32 already_readable_count{};
    u32 barrier_count{};
    u32 covering_barrier_count{};
    u32 source_depth_stencil_layout_cover_count{};
    u32 source_depth_stencil_access_cover_count{};
    u32 source_depth_stencil_stage_cover_count{};
    u32 destination_layout_cover_count{};
    u32 destination_access_cover_count{};
    u32 destination_stage_cover_count{};
    u32 post_readable_count{};
    u32 binding_count{};
    u32 live_layout_readable_count{};
    u32 live_access_readable_count{};
    u32 live_stage_readable_count{};
    u32 live_readable_count{};
    u32 additional_command_count{};
    u32 additional_barrier_count{};
    u32 additional_barrier_query_count{};
    u32 additional_transit_call_count{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_wait_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    u32 additional_ownership_transfer_count{};
    bool writer_valid{};
    bool writer_current{};
    bool resource_exact{};
    bool backing_exact{};
    bool mutation_exact{};
    bool depth_aspect_exact{};
    bool selected_subresource_exact{};
    bool synchronization_covers_selected{};
    bool source_depth_stencil_write{};
    bool destination_shader_read{};
    bool destination_layout_exact{};
    bool post_readable{};
    bool barrier_needed{};
    bool barrier_submitted{};
    bool uniform_before{};
    bool uniform_after{};
    bool overflowed{};
    bool retains_barrier_vector{};
    bool retains_vk_barrier{};
    bool retains_private_state{};
    bool retains_image{};
    bool retains_vk_image{};
    bool affects_capture_behavior{};
    bool affects_registry_behavior{};
    bool affects_renderer_behavior{};
    PpUpstreamFeedbackProducerDepthConsumerDisposition disposition{
        PpUpstreamFeedbackProducerDepthConsumerDisposition::Idle};

    bool operator==(const PpUpstreamFeedbackProducerDepthConsumerSnapshot&) const = default;
};

[[nodiscard]] constexpr PpUpstreamFeedbackProducerDepthConsumerSnapshot
ClassifyPpUpstreamFeedbackProducerDepthConsumerState(
    const PpUpstreamFeedbackProducerDepthConsumerDescriptor& descriptor) noexcept {
    using Disposition = PpUpstreamFeedbackProducerDepthConsumerDisposition;
    const auto& transit = descriptor.transit;
    const auto& fact = transit.fact;
    const auto& live = descriptor.live_state;
    PpUpstreamFeedbackProducerDepthConsumerSnapshot result{
        .selected_subresource_count = fact.selected_subresource_count,
        .prior_depth_stencil_write_count = fact.prior_depth_stencil_write_count,
        .already_readable_count = fact.already_readable_count,
        .barrier_count = fact.barrier_count,
        .covering_barrier_count = fact.covering_barrier_count,
        .source_depth_stencil_layout_cover_count =
            fact.source_depth_stencil_layout_cover_count,
        .source_depth_stencil_access_cover_count =
            fact.source_depth_stencil_access_cover_count,
        .source_depth_stencil_stage_cover_count = fact.source_depth_stencil_stage_cover_count,
        .destination_layout_cover_count = fact.destination_layout_cover_count,
        .destination_access_cover_count = fact.destination_access_cover_count,
        .destination_stage_cover_count = fact.destination_stage_cover_count,
        .post_readable_count = fact.post_readable_count,
        .binding_count = transit.binding_count,
        .live_layout_readable_count = live.layout_readable_count,
        .live_access_readable_count = live.access_readable_count,
        .live_stage_readable_count = live.stage_readable_count,
        .live_readable_count = live.readable_count,
        .writer_valid = fact.writer_valid,
        .writer_current = descriptor.current_writer.valid,
        .barrier_needed = fact.prior_depth_stencil_write_mask != 0,
        .barrier_submitted = fact.submitted,
        .uniform_before = fact.uniform_before,
        .uniform_after = fact.uniform_after,
        .overflowed = fact.overflowed || transit.overflowed || live.overflowed,
    };

    const bool fact_empty = fact == PpUpstreamFeedbackSampledImageTransitFact{};
    const bool live_empty = !live.observed && !live.destination_supported && !live.uniform &&
                            !live.malformed && !live.overflowed &&
                            live.selected_subresource_mask == 0 &&
                            live.layout_readable_mask == 0 && live.access_readable_mask == 0 &&
                            live.stage_readable_mask == 0 && live.readable_mask == 0 &&
                            live.selected_subresource_count == 0 &&
                            live.layout_readable_count == 0 && live.access_readable_count == 0 &&
                            live.stage_readable_count == 0 && live.readable_count == 0;
    const bool no_sidecar = transit.binding_count == 0 && fact_empty && live_empty &&
                            !descriptor.selected_view.valid && !descriptor.current_writer.valid;
    if (descriptor.expected_input_count == 0 && descriptor.candidate_count == 0) {
        if (!no_sidecar) {
            result.disposition = Disposition::InvalidTransition;
        }
        return result;
    }
    if (descriptor.expected_input_count != 1 || descriptor.candidate_count != 1) {
        result.disposition = Disposition::NotSingleCandidate;
        return result;
    }
    if (no_sidecar) {
        return result;
    }
    if (transit.binding_count == 0 || fact_empty) {
        result.disposition = Disposition::InvalidTransition;
        return result;
    }
    if (!fact.writer_valid) {
        result.disposition = Disposition::MissingWriter;
        return result;
    }
    if (!descriptor.current_writer.valid) {
        result.disposition = Disposition::StaleWriter;
        return result;
    }
    if (!fact.selected_cardinality_exact) {
        result.disposition = Disposition::InvalidTransition;
        return result;
    }

    result.resource_exact = fact.resource_exact && descriptor.selected_view.valid &&
                            descriptor.current_writer.link == descriptor.selected_view.link;
    result.backing_exact = fact.backing_exact &&
                           descriptor.selected_view.backing_generation != 0 &&
                           descriptor.current_writer.backing_generation ==
                               descriptor.selected_view.backing_generation;
    result.mutation_exact = fact.mutation_exact &&
                            descriptor.selected_view.mutation_epoch != 0 &&
                            descriptor.current_writer.mutation_epoch ==
                                descriptor.selected_view.mutation_epoch;
    constexpr u32 DepthAspect = static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT);
    result.depth_aspect_exact = fact.depth_aspect_exact &&
                                descriptor.selected_view.aspect_mask == DepthAspect &&
                                (descriptor.current_writer.attachment_aspect_mask & DepthAspect) !=
                                    0;
    const VideoCore::SubresourceRange selected_range{
        .base = {.level = descriptor.selected_view.base_mip,
                 .layer = descriptor.selected_view.base_layer},
        .extent = {.levels = descriptor.selected_view.mip_count,
                   .layers = descriptor.selected_view.layer_count},
    };
    const u64 selected_count =
        static_cast<u64>(selected_range.extent.levels) * selected_range.extent.layers;
    result.selected_subresource_exact =
        fact.selected_subresource_exact && selected_count == fact.selected_subresource_count &&
        descriptor.current_writer.range == selected_range;
    if (!result.resource_exact) {
        result.disposition = Disposition::ResourceMismatch;
        return result;
    }
    if (!result.backing_exact) {
        result.disposition = Disposition::BackingMismatch;
        return result;
    }
    if (!result.mutation_exact) {
        result.disposition = Disposition::MutationMismatch;
        return result;
    }
    if (!result.depth_aspect_exact || !result.selected_subresource_exact) {
        result.disposition = Disposition::SubresourceMismatch;
        return result;
    }

    const auto count = [](u64 mask) { return static_cast<u32>(std::popcount(mask)); };
    const u64 selected = fact.selected_subresource_mask;
    const u64 prior = fact.prior_depth_stencil_write_mask;
    const u64 already = fact.already_readable_mask;
    const u64 covering = fact.covering_barrier_mask;
    const u64 dense_selected =
        fact.selected_subresource_count == VideoCore::ImageTransitBarrierReceiptMaxSubresources
            ? std::numeric_limits<u64>::max()
            : fact.selected_subresource_count == 0
                  ? 0
                  : (u64{1} << fact.selected_subresource_count) - 1;
    const bool mask_counts_exact =
        fact.selected_subresource_count <=
            VideoCore::ImageTransitBarrierReceiptMaxSubresources &&
        selected == dense_selected && fact.selected_subresource_count == count(selected) &&
        fact.prior_depth_stencil_write_count == count(prior) &&
        fact.already_readable_count == count(already) &&
        fact.covering_barrier_count == count(covering) &&
        fact.source_depth_stencil_layout_cover_count ==
            count(fact.source_depth_stencil_layout_cover_mask) &&
        fact.source_depth_stencil_access_cover_count ==
            count(fact.source_depth_stencil_access_cover_mask) &&
        fact.source_depth_stencil_stage_cover_count ==
            count(fact.source_depth_stencil_stage_cover_mask) &&
        fact.destination_layout_cover_count == count(fact.destination_layout_cover_mask) &&
        fact.destination_access_cover_count == count(fact.destination_access_cover_mask) &&
        fact.destination_stage_cover_count == count(fact.destination_stage_cover_mask) &&
        fact.post_readable_count == count(fact.post_readable_mask);
    const bool masks_canonical =
        selected != 0 && (prior & already) == 0 && (prior | already) == selected &&
        (covering & ~selected) == 0 && (covering & already) == 0 &&
        (fact.source_depth_stencil_layout_cover_mask & ~covering) == 0 &&
        (fact.source_depth_stencil_access_cover_mask & ~covering) == 0 &&
        (fact.source_depth_stencil_stage_cover_mask & ~covering) == 0 &&
        (fact.destination_layout_cover_mask & ~covering) == 0 &&
        (fact.destination_access_cover_mask & ~covering) == 0 &&
        (fact.destination_stage_cover_mask & ~covering) == 0 &&
        (fact.post_readable_mask & ~selected) == 0;
    const bool barrier_shape_exact =
        fact.submitted == (fact.barrier_count != 0) &&
        (fact.barrier_count == 0 || covering != 0);
    if (!fact.transit_invoked || fact.malformed || fact.overflowed ||
        transit.duplicate_drift || transit.overflowed || !mask_counts_exact ||
        !masks_canonical || !barrier_shape_exact) {
        result.disposition = Disposition::InvalidTransition;
        return result;
    }

    result.destination_shader_read = fact.destination_shader_read;
    result.destination_layout_exact = fact.destination_supported;
    if (!fact.destination_supported || !fact.destination_shader_read ||
        !fact.writer_depth_write || !descriptor.current_writer.depth_write) {
        result.disposition = Disposition::UnsupportedState;
        return result;
    }

    result.source_depth_stencil_write =
        prior != 0 && fact.source_depth_stencil_access_cover_mask == prior;
    result.post_readable = fact.post_readable_mask == selected;
    const bool covered_barriers_supported =
        fact.source_depth_stencil_layout_cover_mask == covering &&
        fact.source_depth_stencil_access_cover_mask == covering &&
        fact.source_depth_stencil_stage_cover_mask == covering &&
        fact.destination_layout_cover_mask == covering &&
        fact.destination_access_cover_mask == covering &&
        fact.destination_stage_cover_mask == covering;
    if (!covered_barriers_supported) {
        result.disposition = Disposition::UnsupportedState;
        return result;
    }
    const bool write_coverage =
        covering == prior && fact.source_depth_stencil_layout_cover_mask == prior &&
        fact.source_depth_stencil_access_cover_mask == prior &&
        fact.source_depth_stencil_stage_cover_mask == prior &&
        fact.destination_layout_cover_mask == prior &&
        fact.destination_access_cover_mask == prior &&
        fact.destination_stage_cover_mask == prior;
    result.synchronization_covers_selected = write_coverage && result.post_readable;
    const bool already_readable =
        prior == 0 && already == selected && fact.barrier_count == 0 && result.post_readable;
    if (prior != 0 && fact.barrier_count == 0) {
        result.disposition = Disposition::MissingBarrier;
        return result;
    }
    if (!already_readable && (!result.synchronization_covers_selected || !fact.submitted)) {
        result.disposition = Disposition::PartialBarrier;
        return result;
    }
    const bool live_counts_exact =
        live.selected_subresource_count == count(live.selected_subresource_mask) &&
        live.layout_readable_count == count(live.layout_readable_mask) &&
        live.access_readable_count == count(live.access_readable_mask) &&
        live.stage_readable_count == count(live.stage_readable_mask) &&
        live.readable_count == count(live.readable_mask);
    const bool live_masks_exact =
        live.selected_subresource_mask == selected &&
        (live.layout_readable_mask & ~selected) == 0 &&
        (live.access_readable_mask & ~selected) == 0 &&
        (live.stage_readable_mask & ~selected) == 0 &&
        live.readable_mask == (live.layout_readable_mask & live.access_readable_mask &
                               live.stage_readable_mask);
    if (!live.observed || live.malformed || live.overflowed || !live_counts_exact ||
        !live_masks_exact) {
        result.disposition = Disposition::InvalidTransition;
        return result;
    }
    if (!live.destination_supported) {
        result.disposition = Disposition::UnsupportedState;
        return result;
    }
    if (live.readable_mask != selected) {
        result.disposition = Disposition::LiveStateMismatch;
        return result;
    }
    result.synchronization_covers_selected = true;
    result.disposition = already_readable ? Disposition::AlreadyReadable : Disposition::Exact;
    return result;
}

using PpUpstreamFeedbackImageSubresourceLookupDisposition =
    VideoCore::ImageSubresourceLookupDisposition;

[[nodiscard]] constexpr VideoCore::ImageSubresourceLookupRelation
ClassifyPpUpstreamFeedbackImageSubresourceLookupRelation(
    const bool actual_accepted, const bool canonical_accepted,
    const bool actual_canonical_equal) noexcept {
    return VideoCore::ClassifyImageSubresourceLookupRelation(
        actual_accepted, canonical_accepted, actual_canonical_equal);
}

[[nodiscard]] constexpr PpUpstreamFeedbackImageSubresourceLookupDisposition
ClassifyPpUpstreamFeedbackImageSubresourceLookup(
    const VideoCore::ImageSubresourceLookupFact& fact) noexcept {
    return VideoCore::ClassifyImageSubresourceLookup(fact);
}

struct PpUpstreamFeedbackImageSubresourceLookupBindingDescriptor {
    VideoCore::ImageSubresourceLookupFact initial_lookup{};
    VideoCore::ImageSubresourceLookupFact rebound_lookup{};
    VideoCore::ImageSubresourceLookupFact selected_origin{};
    bool rebind_requested{};
};

[[nodiscard]] constexpr VideoCore::ImageSubresourceLookupFact
MakePpUpstreamFeedbackImageSubresourceLookupBindingFact(
    const PpUpstreamFeedbackImageSubresourceLookupBindingDescriptor& descriptor) noexcept {
    using Disposition = PpUpstreamFeedbackImageSubresourceLookupDisposition;
    using Path = VideoCore::ImageSubresourceLookupPath;
    auto initial = descriptor.initial_lookup;
    initial.disposition = ClassifyPpUpstreamFeedbackImageSubresourceLookup(initial);
    if (initial.disposition == Disposition::AmbiguousExact ||
        initial.disposition == Disposition::InvalidTransition) {
        return initial;
    }
    if (!descriptor.rebind_requested && descriptor.rebound_lookup.observed) {
        initial.disposition = Disposition::InvalidTransition;
        return initial;
    }
    auto lookup = descriptor.rebind_requested ? descriptor.rebound_lookup : initial;
    lookup.disposition = ClassifyPpUpstreamFeedbackImageSubresourceLookup(lookup);
    if (descriptor.rebind_requested && !descriptor.rebound_lookup.observed) {
        lookup.disposition = Disposition::InvalidTransition;
        return lookup;
    }
    if (lookup.disposition == Disposition::AmbiguousExact ||
        lookup.disposition == Disposition::InvalidTransition) {
        return lookup;
    }
    auto origin = descriptor.selected_origin;
    origin.disposition = ClassifyPpUpstreamFeedbackImageSubresourceLookup(origin);
    if (lookup.path == Path::Perfect) {
        if (!origin.observed) {
            return lookup;
        }
        if (!origin.provenance_current || origin.disposition == Disposition::Idle ||
            origin.disposition == Disposition::AmbiguousExact ||
            origin.disposition == Disposition::InvalidTransition) {
            lookup.disposition = Disposition::InvalidTransition;
            return lookup;
        }
        return origin;
    }
    if (origin.observed) {
        lookup.disposition = Disposition::InvalidTransition;
    }
    return lookup;
}

struct PpUpstreamFeedbackImageSubresourceLookupObservation {
    VideoCore::ImageSubresourceLookupFact fact{};
    u32 lookup_count{};
    bool duplicate_drift{};
    bool overflowed{};

    bool operator==(const PpUpstreamFeedbackImageSubresourceLookupObservation&) const = default;
};

class PpUpstreamFeedbackImageSubresourceLookupAccumulator {
public:
    static constexpr u32 MaxLookupCount = 255;

    [[nodiscard]] constexpr bool ObserveFirst(
        VideoCore::ImageSubresourceLookupFact fact) noexcept {
        fact.disposition = ClassifyPpUpstreamFeedbackImageSubresourceLookup(fact);
        if (value.lookup_count != 0) {
            Increment();
            FailClosed();
            return false;
        }
        value = {.fact = fact, .lookup_count = 1, .overflowed = fact.overflowed};
        return fact.disposition !=
               PpUpstreamFeedbackImageSubresourceLookupDisposition::InvalidTransition;
    }

    [[nodiscard]] constexpr bool ObserveDuplicate(
        VideoCore::ImageSubresourceLookupFact fact) noexcept {
        fact.disposition = ClassifyPpUpstreamFeedbackImageSubresourceLookup(fact);
        if (value.lookup_count == 0) {
            value = {.fact = fact, .lookup_count = 1, .duplicate_drift = true,
                     .overflowed = fact.overflowed};
            FailClosed();
            return false;
        }
        if (value.lookup_count == MaxLookupCount) {
            value.overflowed = true;
            FailClosed();
            return false;
        }
        ++value.lookup_count;
        if (fact != value.fact || fact.overflowed) {
            FailClosed();
            return false;
        }
        return true;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackImageSubresourceLookupObservation Peek()
        const noexcept {
        return value;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackImageSubresourceLookupObservation Take() noexcept {
        const auto result = value;
        value = {};
        return result;
    }

private:
    constexpr void Increment() noexcept {
        if (value.lookup_count == MaxLookupCount) {
            value.overflowed = true;
        } else {
            ++value.lookup_count;
        }
    }

    constexpr void FailClosed() noexcept {
        value.duplicate_drift = true;
        value.fact.disposition =
            PpUpstreamFeedbackImageSubresourceLookupDisposition::InvalidTransition;
    }

    PpUpstreamFeedbackImageSubresourceLookupObservation value{};
};

struct PpUpstreamFeedbackImageSubresourceLookupSnapshot {
    VideoCore::ImageSubresourceLookupPath path{VideoCore::ImageSubresourceLookupPath::Idle};
    VideoCore::ImageSubresourceLookupRelation relation{
        VideoCore::ImageSubresourceLookupRelation::None};
    u32 perfect_candidate_count{};
    u32 overlap_candidate_count{};
    u32 lookup_count{};
    u32 additional_image_lookup_count{};
    u32 additional_mip_lookup_count{};
    u32 additional_slice_lookup_count{};
    u32 additional_gpu_copy_count{};
    u32 additional_command_count{};
    u32 additional_barrier_count{};
    u32 additional_slot_count{};
    u32 additional_wait_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    u32 additional_ownership_transfer_count{};
    bool actual_accepted{};
    bool canonical_accepted{};
    bool actual_canonical_equal{};
    bool nonzero_mip{};
    bool nonzero_layer{};
    bool logical_single_layer{};
    bool parent_stride_aligned{};
    bool bounded_by_parent{};
    bool parent_gpu_authoritative{};
    bool created_standalone{};
    bool provenance_current{};
    bool observed{};
    bool candidate_drift{};
    bool overflowed{};
    bool retains_private_state{};
    bool retains_image{};
    bool retains_subresource{};
    bool affects_capture_behavior{};
    bool affects_registry_behavior{};
    bool affects_renderer_behavior{};
    PpUpstreamFeedbackImageSubresourceLookupDisposition disposition{
        PpUpstreamFeedbackImageSubresourceLookupDisposition::Idle};

    bool operator==(const PpUpstreamFeedbackImageSubresourceLookupSnapshot&) const = default;
};

[[nodiscard]] constexpr PpUpstreamFeedbackImageSubresourceLookupSnapshot
MakePpUpstreamFeedbackImageSubresourceLookupSnapshot(
    const PpUpstreamFeedbackImageSubresourceLookupObservation& observation) noexcept {
    const auto& fact = observation.fact;
    const auto disposition =
        ClassifyPpUpstreamFeedbackImageSubresourceLookup(fact);
    return {
        .path = fact.path,
        .relation = fact.relation,
        .perfect_candidate_count = fact.perfect_candidate_count,
        .overlap_candidate_count = fact.overlap_candidate_count,
        .lookup_count = observation.lookup_count,
        .actual_accepted = fact.actual_accepted,
        .canonical_accepted = fact.canonical_accepted,
        .actual_canonical_equal = fact.actual_canonical_equal,
        .nonzero_mip = fact.nonzero_mip,
        .nonzero_layer = fact.nonzero_layer,
        .logical_single_layer = fact.logical_single_layer,
        .parent_stride_aligned = fact.parent_stride_aligned,
        .bounded_by_parent = fact.bounded_by_parent,
        .parent_gpu_authoritative = fact.parent_gpu_authoritative,
        .created_standalone = fact.created_standalone,
        .provenance_current = fact.provenance_current,
        .observed = fact.observed,
        .candidate_drift = fact.candidate_drift,
        .overflowed = fact.overflowed || observation.overflowed,
        .disposition = observation.duplicate_drift || fact.overflowed || observation.overflowed
                           ? PpUpstreamFeedbackImageSubresourceLookupDisposition::InvalidTransition
                           : disposition,
    };
}

enum class PpUpstreamFeedbackSamplerAdjustmentRelation : u8 {
    Idle = 0,
    InactiveExact = 1,
    AppliedExact = 2,
    NotAppliedExact = 3,
    DiagnosticAssociationLoss = 4,
    EffectiveAssociationMismatch = 5,
    SharedIntentConflict = 6,
    SharedOutcomeConflict = 7,
    InvalidTransition = 8,
};

struct PpUpstreamFeedbackSamplerAdjustmentDescriptor {
    u32 full_image_resource_index{};
    u32 legacy_associated_image{};
    bool disable_aniso{};
    bool image_single_level{};
    bool shared_intent_conflict{};
    bool shared_outcome_conflict{};
    bool pair_intent_conflict{};
};

[[nodiscard]] constexpr PpUpstreamFeedbackSamplerAdjustmentRelation
ClassifyPpUpstreamFeedbackSamplerAdjustment(
    const PpUpstreamFeedbackSamplerAdjustmentDescriptor& descriptor) noexcept {
    using Relation = PpUpstreamFeedbackSamplerAdjustmentRelation;
    const bool intent_conflict =
        descriptor.shared_intent_conflict || descriptor.pair_intent_conflict;
    if (intent_conflict && descriptor.shared_outcome_conflict) {
        return Relation::InvalidTransition;
    }
    if (intent_conflict) {
        return Relation::SharedIntentConflict;
    }
    if (descriptor.shared_outcome_conflict) {
        return Relation::SharedOutcomeConflict;
    }
    if (descriptor.full_image_resource_index != descriptor.legacy_associated_image) {
        return descriptor.disable_aniso ? Relation::EffectiveAssociationMismatch
                                        : Relation::DiagnosticAssociationLoss;
    }
    if (!descriptor.disable_aniso) {
        return Relation::InactiveExact;
    }
    return descriptor.image_single_level ? Relation::AppliedExact
                                         : Relation::NotAppliedExact;
}

enum class PpUpstreamFeedbackSamplerDescriptorRelation : u8 {
    Idle = 0,
    Exact = 1,
    HandleMismatch = 2,
    ImageViewMismatch = 3,
    LayoutMismatch = 4,
    WriteMismatch = 5,
    PointerMismatch = 6,
    MissingObservation = 7,
};

enum class PpUpstreamFeedbackSamplerBindingDisposition : u8 {
    Idle = 0,
    Exact = 1,
    MissingObservation = 2,
    MissingAssociation = 3,
    AdjustmentMismatch = 4,
    RealizationMismatch = 5,
    CacheMismatch = 6,
    DescriptorMismatch = 7,
    InvalidTransition = 8,
};

struct PpUpstreamFeedbackSamplerBindingDescriptor {
    u32 logical_stage{};
    u32 sampler_resource_index{};
    u32 image_resource_index{};
    u32 pair_operation_use_mask{};
    const AmdGpu::Sampler* effective_sampler{};
    bool association_present{};
    bool adjustment_intended{};
    PpUpstreamFeedbackSamplerAdjustmentRelation adjustment_outcome{
        PpUpstreamFeedbackSamplerAdjustmentRelation::Idle};
    bool inline_sampler{};
    bool shared_sampler{};
    VideoCore::SamplerRealizationFact creation{};
    VideoCore::SamplerCacheFact cache{};
    vk::Sampler returned_sampler{};
    const vk::DescriptorImageInfo* descriptor_info{};
    const vk::WriteDescriptorSet* write{};
    u32 expected_binding{};
    bool observed{};
};

struct PpUpstreamFeedbackSamplerBindingFact {
    VideoCore::SamplerRealizationFact creation{};
    VideoCore::SamplerCacheFact cache{};
    PpUpstreamFeedbackSamplerAdjustmentRelation adjustment_outcome{
        PpUpstreamFeedbackSamplerAdjustmentRelation::Idle};
    PpUpstreamFeedbackSamplerDescriptorRelation descriptor_relation{
        PpUpstreamFeedbackSamplerDescriptorRelation::Idle};
    bool association_present{};
    bool adjustment_intended{};
    bool inline_sampler{};
    bool shared_sampler{};
    bool observed{};
    PpUpstreamFeedbackSamplerBindingDisposition disposition{
        PpUpstreamFeedbackSamplerBindingDisposition::Idle};

    auto operator<=>(const PpUpstreamFeedbackSamplerBindingFact&) const = default;
};

[[nodiscard]] constexpr PpUpstreamFeedbackSamplerDescriptorRelation
ClassifyPpUpstreamFeedbackSamplerDescriptor(
    const PpUpstreamFeedbackSamplerBindingDescriptor& descriptor) noexcept {
    using Relation = PpUpstreamFeedbackSamplerDescriptorRelation;
    if (!descriptor.observed || !descriptor.returned_sampler || descriptor.descriptor_info == nullptr ||
        descriptor.write == nullptr) {
        return Relation::MissingObservation;
    }
    if (!descriptor.descriptor_info->sampler ||
        descriptor.descriptor_info->sampler != descriptor.returned_sampler) {
        return Relation::HandleMismatch;
    }
    if (descriptor.descriptor_info->imageView) {
        return Relation::ImageViewMismatch;
    }
    if (descriptor.descriptor_info->imageLayout != vk::ImageLayout::eGeneral) {
        return Relation::LayoutMismatch;
    }
    if (descriptor.write->descriptorType != vk::DescriptorType::eSampler ||
        descriptor.write->dstBinding != descriptor.expected_binding ||
        descriptor.write->dstArrayElement != 0 || descriptor.write->descriptorCount != 1) {
        return Relation::WriteMismatch;
    }
    if (descriptor.write->pImageInfo != descriptor.descriptor_info) {
        return Relation::PointerMismatch;
    }
    return Relation::Exact;
}

[[nodiscard]] constexpr PpUpstreamFeedbackSamplerBindingDisposition
ClassifyPpUpstreamFeedbackSamplerBindingFact(
    const PpUpstreamFeedbackSamplerBindingFact& fact) noexcept {
    using Adjustment = PpUpstreamFeedbackSamplerAdjustmentRelation;
    using Disposition = PpUpstreamFeedbackSamplerBindingDisposition;
    if (fact == PpUpstreamFeedbackSamplerBindingFact{}) {
        return Disposition::Idle;
    }
    if (!fact.observed) {
        return Disposition::MissingObservation;
    }
    if (!fact.association_present) {
        return Disposition::MissingAssociation;
    }
    const bool exact_adjustment =
        fact.adjustment_outcome == Adjustment::InactiveExact ||
        fact.adjustment_outcome == Adjustment::AppliedExact ||
        fact.adjustment_outcome == Adjustment::NotAppliedExact ||
        fact.adjustment_outcome == Adjustment::DiagnosticAssociationLoss;
    const bool intended_consistent =
        fact.adjustment_intended
            ? fact.adjustment_outcome == Adjustment::AppliedExact ||
                  fact.adjustment_outcome == Adjustment::NotAppliedExact
            : fact.adjustment_outcome == Adjustment::InactiveExact ||
                  fact.adjustment_outcome == Adjustment::DiagnosticAssociationLoss;
    if (!exact_adjustment || !intended_consistent) {
        return fact.adjustment_outcome == Adjustment::InvalidTransition
                   ? Disposition::InvalidTransition
                   : Disposition::AdjustmentMismatch;
    }
    if (fact.creation.disposition != VideoCore::SamplerRealizationDisposition::Exact ||
        fact.creation.compare_relation == VideoCore::SamplerCompareRelation::DescriptorExact) {
        return Disposition::RealizationMismatch;
    }
    if (fact.cache.disposition != VideoCore::SamplerCacheDisposition::Exact) {
        return Disposition::CacheMismatch;
    }
    if (fact.descriptor_relation != PpUpstreamFeedbackSamplerDescriptorRelation::Exact) {
        return Disposition::DescriptorMismatch;
    }
    return Disposition::Exact;
}

[[nodiscard]] inline PpUpstreamFeedbackSamplerBindingFact
MakePpUpstreamFeedbackSamplerBindingFact(
    const PpUpstreamFeedbackSamplerBindingDescriptor& descriptor) noexcept {
    auto creation = descriptor.creation;
    if (creation.compare_relation == VideoCore::SamplerCompareRelation::DescriptorExact) {
        if (descriptor.effective_sampler != nullptr) {
            creation = VideoCore::ResolveSamplerRealizationForPair(
                creation, *descriptor.effective_sampler, descriptor.pair_operation_use_mask);
        } else {
            creation.disposition = VideoCore::SamplerRealizationDisposition::InvalidTransition;
        }
    }
    auto cache = descriptor.cache;
    cache.disposition = VideoCore::ClassifySamplerCacheFact(cache);
    if (cache.creation != descriptor.creation) {
        cache.disposition = VideoCore::SamplerCacheDisposition::InvalidTransition;
    }
    PpUpstreamFeedbackSamplerBindingFact fact{
        .creation = creation,
        .cache = cache,
        .adjustment_outcome = descriptor.adjustment_outcome,
        .descriptor_relation = ClassifyPpUpstreamFeedbackSamplerDescriptor(descriptor),
        .association_present = descriptor.association_present,
        .adjustment_intended = descriptor.adjustment_intended,
        .inline_sampler = descriptor.inline_sampler,
        .shared_sampler = descriptor.shared_sampler,
        .observed = descriptor.observed,
    };
    fact.disposition = ClassifyPpUpstreamFeedbackSamplerBindingFact(fact);
    return fact;
}

struct PpUpstreamFeedbackSamplerPairKey {
    u32 logical_stage{};
    u32 sampler_resource_index{};
    u32 image_resource_index{};

    bool operator==(const PpUpstreamFeedbackSamplerPairKey&) const = default;
};

struct PpUpstreamFeedbackSamplerPairObservation {
    PpUpstreamFeedbackSamplerPairKey key{};
    PpUpstreamFeedbackSamplerBindingFact fact{};
};

enum class PpUpstreamFeedbackSamplerSelectionDisposition : u8 {
    Idle = 0,
    Exact = 1,
    NoAuthority = 2,
    NoAssociatedSampler = 3,
    InvalidTransition = 4,
};

struct PpUpstreamFeedbackSamplerSelectionSnapshot {
    PpUpstreamFeedbackSamplerBindingFact fact{};
    u32 binding_count{};
    bool overflowed{};
    PpUpstreamFeedbackSamplerSelectionDisposition disposition{
        PpUpstreamFeedbackSamplerSelectionDisposition::Idle};
};

struct PpUpstreamFeedbackSamplerSelectionDescriptor {
    bool single_physical_image_authoritative{};
    std::span<const PpUpstreamFeedbackSamplerPairObservation> bindings{};
    std::span<const PpUpstreamFeedbackSamplerPairKey> selected_pairs{};
};

struct PpUpstreamFeedbackSamplerSelection {
    PpUpstreamFeedbackSamplerBindingFact fact{};
    PpUpstreamFeedbackSamplerSelectionSnapshot snapshot{};
    u32 binding_count{};
    bool overflowed{};
    PpUpstreamFeedbackSamplerSelectionDisposition selection_disposition{
        PpUpstreamFeedbackSamplerSelectionDisposition::Idle};
};

[[nodiscard]] inline PpUpstreamFeedbackSamplerSelection
SelectPpUpstreamFeedbackSamplerBindingsForSoleImage(
    const PpUpstreamFeedbackSamplerSelectionDescriptor& descriptor) noexcept {
    using BindingDisposition = PpUpstreamFeedbackSamplerBindingDisposition;
    using SelectionDisposition = PpUpstreamFeedbackSamplerSelectionDisposition;
    PpUpstreamFeedbackSamplerSelection result{};
    if (!descriptor.single_physical_image_authoritative) {
        result.selection_disposition = SelectionDisposition::NoAuthority;
        result.snapshot.disposition = result.selection_disposition;
        return result;
    }
    if (descriptor.selected_pairs.empty()) {
        result.selection_disposition = SelectionDisposition::NoAssociatedSampler;
        result.snapshot.disposition = result.selection_disposition;
        return result;
    }

    PpUpstreamFeedbackSamplerPairKey first_key{};
    bool have_first{};
    bool heterogeneous{};
    bool drift{};
    for (const auto& binding : descriptor.bindings) {
        const bool selected = std::ranges::any_of(
            descriptor.selected_pairs,
            [&](const auto& key) { return key == binding.key; });
        if (!selected) {
            continue;
        }
        if (result.binding_count == 255) {
            result.overflowed = true;
            drift = true;
            continue;
        }
        ++result.binding_count;
        if (!have_first) {
            have_first = true;
            first_key = binding.key;
            result.fact = binding.fact;
        } else {
            heterogeneous |= binding.key != first_key;
            drift |= binding.fact != result.fact;
        }
    }
    if (!have_first) {
        result.selection_disposition = SelectionDisposition::NoAssociatedSampler;
    } else if (heterogeneous || drift || result.overflowed ||
               result.fact.disposition != BindingDisposition::Exact) {
        result.fact.disposition = BindingDisposition::InvalidTransition;
        result.selection_disposition = SelectionDisposition::InvalidTransition;
    } else {
        result.selection_disposition = SelectionDisposition::Exact;
    }
    result.snapshot = {
        .fact = result.fact,
        .binding_count = result.binding_count,
        .overflowed = result.overflowed,
        .disposition = result.selection_disposition,
    };
    return result;
}

struct PpUpstreamFeedbackSamplerBindingObservation {
    PpUpstreamFeedbackSamplerBindingFact fact{};
    u32 binding_count{};
    bool duplicate_drift{};
    bool overflowed{};

    bool operator==(const PpUpstreamFeedbackSamplerBindingObservation&) const = default;
};

class PpUpstreamFeedbackSamplerBindingAccumulator {
public:
    static constexpr u32 MaxBindingCount = 255;

    [[nodiscard]] constexpr bool ObserveFirst(
        PpUpstreamFeedbackSamplerBindingFact fact) noexcept {
        fact.disposition = ClassifyPpUpstreamFeedbackSamplerBindingFact(fact);
        if (value.binding_count != 0) {
            Increment();
            FailClosed();
            return false;
        }
        value = {.fact = fact, .binding_count = 1};
        return fact.disposition != PpUpstreamFeedbackSamplerBindingDisposition::InvalidTransition;
    }

    [[nodiscard]] constexpr bool ObserveDuplicate(
        PpUpstreamFeedbackSamplerBindingFact fact) noexcept {
        fact.disposition = ClassifyPpUpstreamFeedbackSamplerBindingFact(fact);
        if (value.binding_count == 0) {
            value = {.fact = fact, .binding_count = 1, .duplicate_drift = true};
            FailClosed();
            return false;
        }
        if (value.binding_count == MaxBindingCount) {
            value.overflowed = true;
            FailClosed();
            return false;
        }
        ++value.binding_count;
        if (fact != value.fact) {
            FailClosed();
            return false;
        }
        return true;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackSamplerBindingObservation Peek() const noexcept {
        return value;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackSamplerBindingObservation Take() noexcept {
        const auto result = value;
        value = {};
        return result;
    }

private:
    constexpr void Increment() noexcept {
        if (value.binding_count == MaxBindingCount) {
            value.overflowed = true;
        } else {
            ++value.binding_count;
        }
    }

    constexpr void FailClosed() noexcept {
        value.duplicate_drift = true;
        value.fact.disposition = PpUpstreamFeedbackSamplerBindingDisposition::InvalidTransition;
    }

    PpUpstreamFeedbackSamplerBindingObservation value{};
};

struct PpUpstreamFeedbackSamplerBindingSnapshot {
    VideoCore::SamplerRealizationFact creation{};
    VideoCore::SamplerCacheFact cache{};
    PpUpstreamFeedbackSamplerAdjustmentRelation adjustment_outcome{
        PpUpstreamFeedbackSamplerAdjustmentRelation::Idle};
    PpUpstreamFeedbackSamplerDescriptorRelation descriptor_relation{
        PpUpstreamFeedbackSamplerDescriptorRelation::Idle};
    u32 binding_count{};
    u32 additional_sampler_lookup_count{};
    u32 additional_sampler_create_count{};
    u32 additional_descriptor_write_count{};
    u32 additional_gpu_copy_count{};
    u32 additional_command_count{};
    u32 additional_barrier_count{};
    u32 additional_slot_count{};
    u32 additional_wait_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    u32 additional_ownership_transfer_count{};
    bool association_present{};
    bool adjustment_intended{};
    bool inline_sampler{};
    bool shared_sampler{};
    bool overflowed{};
    bool retains_private_state{};
    bool retains_sampler{};
    bool retains_descriptor{};
    bool retains_pair_key{};
    bool affects_capture_behavior{};
    bool affects_registry_behavior{};
    bool affects_renderer_behavior{};
    PpUpstreamFeedbackSamplerBindingDisposition disposition{
        PpUpstreamFeedbackSamplerBindingDisposition::Idle};

    bool operator==(const PpUpstreamFeedbackSamplerBindingSnapshot&) const = default;
};

[[nodiscard]] constexpr PpUpstreamFeedbackSamplerBindingSnapshot
MakePpUpstreamFeedbackSamplerBindingSnapshot(
    const PpUpstreamFeedbackSamplerBindingObservation& observation) noexcept {
    const auto& fact = observation.fact;
    return {
        .creation = fact.creation,
        .cache = fact.cache,
        .adjustment_outcome = fact.adjustment_outcome,
        .descriptor_relation = fact.descriptor_relation,
        .binding_count = observation.binding_count,
        .association_present = fact.association_present,
        .adjustment_intended = fact.adjustment_intended,
        .inline_sampler = fact.inline_sampler,
        .shared_sampler = fact.shared_sampler,
        .overflowed = observation.overflowed,
        .disposition = observation.duplicate_drift || observation.overflowed
                           ? PpUpstreamFeedbackSamplerBindingDisposition::InvalidTransition
                           : fact.disposition,
    };
}

enum class PpUpstreamFeedbackSampledViewBindingPath : u8 {
    Idle = 0,
    Direct = 1,
    DirectAfterRebind = 2,
    EligibleNoAssociation = 3,
    AssociatedDepth = 4,
    AssociatedDepthAfterRebindPreserved = 5,
    AssociatedDepthAfterRebindLost = 6,
    Invalid = 7,
};

enum class PpUpstreamFeedbackSampledViewBindingDisposition : u8 {
    Idle = 0,
    MissingObservation = 1,
    InvalidOperationUse = 2,
    MixedDepthUse = 3,
    AssociationLost = 4,
    ViewCreationMismatch = 5,
    DescriptorMismatch = 6,
    RefreshMissing = 7,
    Exact = 8,
    InvalidTransition = 9,
};

enum class PpUpstreamFeedbackSampledViewBindingRelation : u32 {
    AssociationPreserved = 1u << 0,
    CreationFormat = 1u << 1,
    CreationAspect = 1u << 2,
    CreationType = 1u << 3,
    CreationRange = 1u << 4,
    CreationMapping = 1u << 5,
    CreationUsage = 1u << 6,
    CreationMinLod = 1u << 7,
    CreationImage = 1u << 8,
    DescriptorView = 1u << 9,
    DescriptorSampler = 1u << 10,
    DescriptorLayout = 1u << 11,
    DescriptorType = 1u << 12,
    DescriptorBinding = 1u << 13,
    DescriptorCount = 1u << 14,
    DescriptorArrayElement = 1u << 15,
    DescriptorSpan = 1u << 16,
};

inline constexpr u32 PpUpstreamFeedbackSampledViewBindingAllRelations =
    (1u << 17) - 1;

struct PpUpstreamFeedbackSampledViewBindingSelection {
    VideoCore::ImageId initial_direct{};
    VideoCore::ImageId initial_associated{};
    VideoCore::ImageId initial_selected{};
    VideoCore::ImageId rebound_direct{};
    VideoCore::ImageId final_selected{};
    bool rebind_requested{};
};

struct PpUpstreamFeedbackSampledViewBindingDescriptor {
    u32 operation_use_mask{};
    bool resource_depth{};
    PpUpstreamFeedbackSampledViewBindingSelection selection{};
    VideoCore::ImageViewCreationObservation creation{};
    VideoCore::ImageRefreshObservation refresh{};
    bool refresh_forwarded{};
    vk::ImageView final_view{};
    bool view_owner_exact{};
    vk::ImageLayout final_layout{vk::ImageLayout::eUndefined};
    std::span<const vk::DescriptorImageInfo> descriptor_infos{};
    u32 selected_descriptor_index{};
    const vk::WriteDescriptorSet* write{};
    vk::DescriptorType expected_descriptor_type{vk::DescriptorType::eSampledImage};
    u32 expected_binding{};
    u32 expected_descriptor_count{};
    u32 expected_array_element{};
    bool observed{};
};

struct PpUpstreamFeedbackSampledViewBindingFact {
    PpUpstreamFeedbackSampledViewBindingPath path{
        PpUpstreamFeedbackSampledViewBindingPath::Idle};
    u32 operation_use_mask{};
    bool resource_depth{};
    VideoCore::ImageViewFormatClass requested_format_class{
        VideoCore::ImageViewFormatClass::Unknown};
    VideoCore::ImageViewFormatClass backing_format_class{
        VideoCore::ImageViewFormatClass::Unknown};
    VideoCore::ImageViewFormatClass actual_format_class{
        VideoCore::ImageViewFormatClass::Unknown};
    VideoCore::ImageViewAspectClass actual_aspect_class{VideoCore::ImageViewAspectClass::None};
    VideoCore::ImageViewMappingClass mapping_class{VideoCore::ImageViewMappingClass::Identity};
    u32 relation_mask{};
    u32 refresh_outcome_mask{};
    u32 gpu_modified_before_mask{};
    VideoCore::ImageSubresourceLookupFact subresource_lookup{};
    bool refresh_forwarded{};
    bool observed{};
    bool malformed{};
    bool overflowed{};
    PpUpstreamFeedbackSampledViewBindingDisposition disposition{
        PpUpstreamFeedbackSampledViewBindingDisposition::Idle};

    bool operator==(const PpUpstreamFeedbackSampledViewBindingFact&) const = default;
};

[[nodiscard]] constexpr PpUpstreamFeedbackSampledViewBindingPath
ClassifyPpUpstreamFeedbackSampledViewBindingPath(
    const PpUpstreamFeedbackSampledViewBindingSelection& selection,
    VideoCore::ImageViewFormatClass requested_format) noexcept {
    using Path = PpUpstreamFeedbackSampledViewBindingPath;
    const bool direct_valid = static_cast<bool>(selection.initial_direct);
    const bool associated_valid = static_cast<bool>(selection.initial_associated);
    const bool selected_valid = static_cast<bool>(selection.initial_selected);
    const bool rebound_valid = static_cast<bool>(selection.rebound_direct);
    const bool final_valid = static_cast<bool>(selection.final_selected);
    const bool association_eligible =
        requested_format == VideoCore::ImageViewFormatClass::StencilCompatible;
    if (!direct_valid && !associated_valid && !selected_valid && !rebound_valid && !final_valid &&
        !selection.rebind_requested) {
        return Path::Idle;
    }
    if (!direct_valid || !selected_valid || !final_valid) {
        return Path::Invalid;
    }
    if (associated_valid) {
        if (!association_eligible || selection.initial_selected != selection.initial_associated) {
            return Path::Invalid;
        }
        if (!selection.rebind_requested) {
            return !rebound_valid && selection.final_selected == selection.initial_associated
                       ? Path::AssociatedDepth
                       : Path::Invalid;
        }
        if (!rebound_valid) {
            return Path::Invalid;
        }
        if (selection.final_selected == selection.initial_associated) {
            return Path::AssociatedDepthAfterRebindPreserved;
        }
        if (selection.final_selected == selection.rebound_direct) {
            return Path::AssociatedDepthAfterRebindLost;
        }
        return Path::Invalid;
    }
    if (selection.initial_selected != selection.initial_direct) {
        return Path::Invalid;
    }
    if (!selection.rebind_requested) {
        if (rebound_valid || selection.final_selected != selection.initial_direct) {
            return Path::Invalid;
        }
        return association_eligible ? Path::EligibleNoAssociation : Path::Direct;
    }
    if (!rebound_valid || selection.final_selected != selection.rebound_direct) {
        return Path::Invalid;
    }
    return association_eligible ? Path::EligibleNoAssociation : Path::DirectAfterRebind;
}

[[nodiscard]] constexpr PpUpstreamFeedbackSampledViewBindingDisposition
ClassifyPpUpstreamFeedbackSampledViewBindingFact(
    const PpUpstreamFeedbackSampledViewBindingFact& fact) noexcept {
    using Disposition = PpUpstreamFeedbackSampledViewBindingDisposition;
    using Path = PpUpstreamFeedbackSampledViewBindingPath;
    constexpr u32 ValidOperationMask =
        static_cast<u32>(Shader::ImageResourceOperationUse::RegularSample) |
        static_cast<u32>(Shader::ImageResourceOperationUse::DepthCompare) |
        static_cast<u32>(Shader::ImageResourceOperationUse::Gather) |
        static_cast<u32>(Shader::ImageResourceOperationUse::Fetch) |
        static_cast<u32>(Shader::ImageResourceOperationUse::Query) |
        static_cast<u32>(Shader::ImageResourceOperationUse::Storage);
    const bool empty = fact == PpUpstreamFeedbackSampledViewBindingFact{};
    if (empty) {
        return Disposition::Idle;
    }
    if (!fact.observed) {
        return Disposition::MissingObservation;
    }
    if (fact.operation_use_mask == 0 || (fact.operation_use_mask & ~ValidOperationMask) != 0) {
        return Disposition::InvalidOperationUse;
    }
    if (fact.path == Path::Idle || fact.path == Path::Invalid || fact.malformed ||
        fact.overflowed) {
        return Disposition::InvalidTransition;
    }
    const u32 regular = static_cast<u32>(Shader::ImageResourceOperationUse::RegularSample);
    const u32 depth = static_cast<u32>(Shader::ImageResourceOperationUse::DepthCompare);
    if ((fact.operation_use_mask & regular) != 0 && (fact.operation_use_mask & depth) != 0) {
        return Disposition::MixedDepthUse;
    }
    if (fact.path == Path::AssociatedDepthAfterRebindLost) {
        return Disposition::AssociationLost;
    }
    constexpr u32 CreationRelations =
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::CreationFormat) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::CreationAspect) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::CreationType) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::CreationRange) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::CreationMapping) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::CreationUsage) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::CreationMinLod) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::CreationImage);
    constexpr u32 DescriptorRelations =
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::DescriptorView) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::DescriptorSampler) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::DescriptorLayout) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::DescriptorType) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::DescriptorBinding) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::DescriptorCount) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::DescriptorArrayElement) |
        static_cast<u32>(PpUpstreamFeedbackSampledViewBindingRelation::DescriptorSpan);
    if ((fact.relation_mask & CreationRelations) != CreationRelations) {
        return Disposition::ViewCreationMismatch;
    }
    if ((fact.relation_mask & DescriptorRelations) != DescriptorRelations) {
        return Disposition::DescriptorMismatch;
    }
    if (!fact.refresh_forwarded || fact.refresh_outcome_mask == 0 ||
        fact.gpu_modified_before_mask == 0) {
        return Disposition::RefreshMissing;
    }
    return Disposition::Exact;
}

[[nodiscard]] inline PpUpstreamFeedbackSampledViewBindingFact
MakePpUpstreamFeedbackSampledViewBindingFact(
    const PpUpstreamFeedbackSampledViewBindingDescriptor& descriptor) noexcept {
    using Relation = PpUpstreamFeedbackSampledViewBindingRelation;
    PpUpstreamFeedbackSampledViewBindingFact fact{
        .path = ClassifyPpUpstreamFeedbackSampledViewBindingPath(
            descriptor.selection, descriptor.creation.requested_format_class),
        .operation_use_mask = descriptor.operation_use_mask,
        .resource_depth = descriptor.resource_depth,
        .requested_format_class = descriptor.creation.requested_format_class,
        .backing_format_class = descriptor.creation.backing_format_class,
        .actual_format_class = descriptor.creation.actual_format_class,
        .actual_aspect_class = descriptor.creation.actual_aspect_class,
        .mapping_class = descriptor.creation.mapping_class,
        .refresh_outcome_mask =
            1u << static_cast<u32>(descriptor.refresh.result),
        .gpu_modified_before_mask = 1u << (descriptor.refresh.gpu_modified_before ? 1u : 0u),
        .refresh_forwarded = descriptor.refresh_forwarded,
        .observed = descriptor.observed,
    };
    if (fact.path != PpUpstreamFeedbackSampledViewBindingPath::AssociatedDepthAfterRebindLost &&
        fact.path != PpUpstreamFeedbackSampledViewBindingPath::Invalid &&
        fact.path != PpUpstreamFeedbackSampledViewBindingPath::Idle) {
        fact.relation_mask |= static_cast<u32>(Relation::AssociationPreserved);
    }
    if (descriptor.creation.format_exact) {
        fact.relation_mask |= static_cast<u32>(Relation::CreationFormat);
    }
    if (descriptor.creation.aspect_exact) {
        fact.relation_mask |= static_cast<u32>(Relation::CreationAspect);
    }
    if (descriptor.creation.view_type_exact) {
        fact.relation_mask |= static_cast<u32>(Relation::CreationType);
    }
    if (descriptor.creation.subresource_exact) {
        fact.relation_mask |= static_cast<u32>(Relation::CreationRange);
    }
    if (descriptor.creation.mapping_exact) {
        fact.relation_mask |= static_cast<u32>(Relation::CreationMapping);
    }
    if (descriptor.creation.usage_exact) {
        fact.relation_mask |= static_cast<u32>(Relation::CreationUsage);
    }
    if (descriptor.creation.min_lod_exact && descriptor.creation.chain_exact) {
        fact.relation_mask |= static_cast<u32>(Relation::CreationMinLod);
    }
    if (descriptor.creation.image_exact && descriptor.creation.created &&
        descriptor.view_owner_exact) {
        fact.relation_mask |= static_cast<u32>(Relation::CreationImage);
    }
    if (descriptor.selected_descriptor_index >= descriptor.descriptor_infos.size()) {
        fact.malformed = true;
        fact.disposition = ClassifyPpUpstreamFeedbackSampledViewBindingFact(fact);
        return fact;
    }
    const auto& selected = descriptor.descriptor_infos[descriptor.selected_descriptor_index];
    if (selected.imageView == descriptor.final_view) {
        fact.relation_mask |= static_cast<u32>(Relation::DescriptorView);
    }
    if (!selected.sampler) {
        fact.relation_mask |= static_cast<u32>(Relation::DescriptorSampler);
    }
    if (selected.imageLayout == descriptor.final_layout) {
        fact.relation_mask |= static_cast<u32>(Relation::DescriptorLayout);
    }
    if (descriptor.write) {
        if (descriptor.write->descriptorType == descriptor.expected_descriptor_type) {
            fact.relation_mask |= static_cast<u32>(Relation::DescriptorType);
        }
        if (descriptor.write->dstBinding == descriptor.expected_binding) {
            fact.relation_mask |= static_cast<u32>(Relation::DescriptorBinding);
        }
        if (descriptor.write->descriptorCount == descriptor.expected_descriptor_count) {
            fact.relation_mask |= static_cast<u32>(Relation::DescriptorCount);
        }
        if (descriptor.write->dstArrayElement == descriptor.expected_array_element) {
            fact.relation_mask |= static_cast<u32>(Relation::DescriptorArrayElement);
        }
        if (descriptor.write->pImageInfo == descriptor.descriptor_infos.data() &&
            descriptor.descriptor_infos.size() == descriptor.expected_descriptor_count) {
            fact.relation_mask |= static_cast<u32>(Relation::DescriptorSpan);
        }
    }
    fact.disposition = ClassifyPpUpstreamFeedbackSampledViewBindingFact(fact);
    return fact;
}

struct PpUpstreamFeedbackSampledViewBindingObservation {
    PpUpstreamFeedbackSampledViewBindingFact fact{};
    PpUpstreamFeedbackImageSubresourceLookupObservation subresource_lookup{};
    PpUpstreamFeedbackSamplerBindingObservation sampler_binding{};
    u32 binding_count{};
    bool duplicate_drift{};
    bool overflowed{};

    bool operator==(const PpUpstreamFeedbackSampledViewBindingObservation&) const = default;
};

class PpUpstreamFeedbackSampledViewBindingAccumulator {
public:
    static constexpr u32 MaxBindingCount = 255;

    [[nodiscard]] constexpr bool ObserveFirst(
        PpUpstreamFeedbackSampledViewBindingFact fact) noexcept {
        bool nested_exact = true;
        if (!combining_lookup && lookup_accumulator.Peek().lookup_count != 0) {
            static_cast<void>(lookup_accumulator.ObserveDuplicate({}));
            value.subresource_lookup = lookup_accumulator.Peek();
            value.fact.subresource_lookup = value.subresource_lookup.fact;
            nested_exact = false;
        }
        if (!combining_sampler && sampler_accumulator.Peek().binding_count != 0) {
            static_cast<void>(sampler_accumulator.ObserveDuplicate({}));
            value.sampler_binding = sampler_accumulator.Peek();
            nested_exact = false;
        }
        if (value.binding_count != 0) {
            value.duplicate_drift = true;
            value.fact.disposition =
                PpUpstreamFeedbackSampledViewBindingDisposition::InvalidTransition;
            Increment();
            return false;
        }
        fact.disposition = ClassifyPpUpstreamFeedbackSampledViewBindingFact(fact);
        value = {.fact = fact, .binding_count = 1, .overflowed = fact.overflowed};
        return nested_exact;
    }

    [[nodiscard]] constexpr bool ObserveFirst(
        PpUpstreamFeedbackSampledViewBindingFact fact,
        VideoCore::ImageSubresourceLookupFact subresource_lookup) noexcept {
        subresource_lookup.disposition =
            ClassifyPpUpstreamFeedbackImageSubresourceLookup(subresource_lookup);
        fact.subresource_lookup = subresource_lookup;
        combining_lookup = true;
        const bool view_exact = ObserveFirst(fact);
        combining_lookup = false;
        const bool lookup_exact = lookup_accumulator.ObserveFirst(subresource_lookup);
        value.subresource_lookup = lookup_accumulator.Peek();
        value.fact.subresource_lookup = value.subresource_lookup.fact;
        return view_exact && lookup_exact;
    }

    [[nodiscard]] constexpr bool ObserveDuplicate(
        PpUpstreamFeedbackSampledViewBindingFact fact) noexcept {
        bool nested_exact = true;
        if (!combining_lookup && lookup_accumulator.Peek().lookup_count != 0) {
            static_cast<void>(lookup_accumulator.ObserveDuplicate({}));
            value.subresource_lookup = lookup_accumulator.Peek();
            value.fact.subresource_lookup = value.subresource_lookup.fact;
            nested_exact = false;
        }
        if (!combining_sampler && sampler_accumulator.Peek().binding_count != 0) {
            static_cast<void>(sampler_accumulator.ObserveDuplicate({}));
            value.sampler_binding = sampler_accumulator.Peek();
            nested_exact = false;
        }
        if (value.binding_count == 0) {
            value = {.fact = fact, .binding_count = 1, .duplicate_drift = true,
                     .overflowed = fact.overflowed};
            value.fact.disposition =
                PpUpstreamFeedbackSampledViewBindingDisposition::InvalidTransition;
            return false;
        }
        if (value.binding_count == MaxBindingCount) {
            value.overflowed = true;
            value.fact.overflowed = true;
            value.fact.disposition =
                PpUpstreamFeedbackSampledViewBindingDisposition::InvalidTransition;
            return false;
        }
        ++value.binding_count;
        const auto& first = value.fact;
        const bool same = fact.path == first.path && fact.resource_depth == first.resource_depth &&
                          fact.requested_format_class == first.requested_format_class &&
                          fact.backing_format_class == first.backing_format_class &&
                          fact.actual_format_class == first.actual_format_class &&
                          fact.actual_aspect_class == first.actual_aspect_class &&
                          fact.mapping_class == first.mapping_class &&
                          fact.relation_mask == first.relation_mask &&
                          fact.subresource_lookup == first.subresource_lookup &&
                          fact.refresh_forwarded == first.refresh_forwarded &&
                          fact.observed == first.observed && fact.malformed == first.malformed &&
                          !fact.overflowed;
        if (!same) {
            value.duplicate_drift = true;
            value.fact.disposition =
                PpUpstreamFeedbackSampledViewBindingDisposition::InvalidTransition;
            return false;
        }
        value.fact.operation_use_mask |= fact.operation_use_mask;
        value.fact.refresh_outcome_mask |= fact.refresh_outcome_mask;
        value.fact.gpu_modified_before_mask |= fact.gpu_modified_before_mask;
        value.fact.disposition = ClassifyPpUpstreamFeedbackSampledViewBindingFact(value.fact);
        return nested_exact;
    }

    [[nodiscard]] constexpr bool ObserveDuplicate(
        PpUpstreamFeedbackSampledViewBindingFact fact,
        VideoCore::ImageSubresourceLookupFact subresource_lookup) noexcept {
        subresource_lookup.disposition =
            ClassifyPpUpstreamFeedbackImageSubresourceLookup(subresource_lookup);
        fact.subresource_lookup = subresource_lookup;
        combining_lookup = true;
        const bool view_exact = ObserveDuplicate(fact);
        combining_lookup = false;
        const bool lookup_exact = lookup_accumulator.ObserveDuplicate(subresource_lookup);
        value.subresource_lookup = lookup_accumulator.Peek();
        value.fact.subresource_lookup = value.subresource_lookup.fact;
        return view_exact && lookup_exact;
    }

    [[nodiscard]] constexpr bool ObserveFirst(
        PpUpstreamFeedbackSampledViewBindingFact fact,
        VideoCore::ImageSubresourceLookupFact subresource_lookup,
        PpUpstreamFeedbackSamplerBindingFact sampler_binding) noexcept {
        subresource_lookup.disposition =
            ClassifyPpUpstreamFeedbackImageSubresourceLookup(subresource_lookup);
        fact.subresource_lookup = subresource_lookup;
        combining_lookup = true;
        combining_sampler = true;
        const bool view_exact = ObserveFirst(fact);
        combining_lookup = false;
        combining_sampler = false;
        const bool lookup_exact = lookup_accumulator.ObserveFirst(subresource_lookup);
        const bool sampler_exact = sampler_accumulator.ObserveFirst(sampler_binding);
        value.subresource_lookup = lookup_accumulator.Peek();
        value.fact.subresource_lookup = value.subresource_lookup.fact;
        value.sampler_binding = sampler_accumulator.Peek();
        return view_exact && lookup_exact && sampler_exact;
    }

    [[nodiscard]] constexpr bool ObserveDuplicate(
        PpUpstreamFeedbackSampledViewBindingFact fact,
        VideoCore::ImageSubresourceLookupFact subresource_lookup,
        PpUpstreamFeedbackSamplerBindingFact sampler_binding) noexcept {
        subresource_lookup.disposition =
            ClassifyPpUpstreamFeedbackImageSubresourceLookup(subresource_lookup);
        fact.subresource_lookup = subresource_lookup;
        combining_lookup = true;
        combining_sampler = true;
        const bool view_exact = ObserveDuplicate(fact);
        combining_lookup = false;
        combining_sampler = false;
        const bool lookup_exact = lookup_accumulator.ObserveDuplicate(subresource_lookup);
        const bool sampler_exact = sampler_accumulator.ObserveDuplicate(sampler_binding);
        value.subresource_lookup = lookup_accumulator.Peek();
        value.fact.subresource_lookup = value.subresource_lookup.fact;
        value.sampler_binding = sampler_accumulator.Peek();
        return view_exact && lookup_exact && sampler_exact;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackSampledViewBindingObservation Peek() const noexcept {
        return value;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackSampledViewBindingObservation Take() noexcept {
        value.subresource_lookup = lookup_accumulator.Take();
        value.fact.subresource_lookup = value.subresource_lookup.fact;
        value.sampler_binding = sampler_accumulator.Take();
        const auto result = value;
        value = {};
        combining_lookup = false;
        combining_sampler = false;
        return result;
    }

private:
    constexpr void Increment() noexcept {
        if (value.binding_count == MaxBindingCount) {
            value.overflowed = true;
            value.fact.overflowed = true;
        } else {
            ++value.binding_count;
        }
    }

    PpUpstreamFeedbackSampledViewBindingObservation value{};
    PpUpstreamFeedbackImageSubresourceLookupAccumulator lookup_accumulator{};
    PpUpstreamFeedbackSamplerBindingAccumulator sampler_accumulator{};
    bool combining_lookup{};
    bool combining_sampler{};
};

struct PpUpstreamFeedbackSampledViewBindingSnapshot {
    PpUpstreamFeedbackSampledViewBindingPath path{
        PpUpstreamFeedbackSampledViewBindingPath::Idle};
    u32 operation_use_mask{};
    bool resource_depth{};
    VideoCore::ImageViewFormatClass requested_format_class{
        VideoCore::ImageViewFormatClass::Unknown};
    VideoCore::ImageViewFormatClass backing_format_class{
        VideoCore::ImageViewFormatClass::Unknown};
    VideoCore::ImageViewFormatClass actual_format_class{
        VideoCore::ImageViewFormatClass::Unknown};
    VideoCore::ImageViewAspectClass actual_aspect_class{VideoCore::ImageViewAspectClass::None};
    VideoCore::ImageViewMappingClass mapping_class{VideoCore::ImageViewMappingClass::Identity};
    u32 relation_mask{};
    u32 refresh_outcome_mask{};
    u32 gpu_modified_before_mask{};
    u32 binding_count{};
    PpUpstreamFeedbackImageSubresourceLookupSnapshot subresource_lookup{};
    PpUpstreamFeedbackSamplerBindingSnapshot sampler_binding{};
    u32 additional_image_lookup_count{};
    u32 additional_association_lookup_count{};
    u32 additional_refresh_count{};
    u32 additional_view_create_count{};
    u32 additional_descriptor_write_count{};
    u32 additional_command_count{};
    u32 additional_barrier_count{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_wait_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    u32 additional_ownership_transfer_count{};
    bool overflowed{};
    bool retains_private_state{};
    bool retains_image{};
    bool retains_vk_image{};
    bool retains_descriptor{};
    bool repairs_association{};
    bool affects_capture_behavior{};
    bool affects_registry_behavior{};
    bool affects_renderer_behavior{};
    PpUpstreamFeedbackSampledViewBindingDisposition disposition{
        PpUpstreamFeedbackSampledViewBindingDisposition::Idle};

    bool operator==(const PpUpstreamFeedbackSampledViewBindingSnapshot&) const = default;
};

[[nodiscard]] constexpr PpUpstreamFeedbackSampledViewBindingSnapshot
MakePpUpstreamFeedbackSampledViewBindingSnapshot(
    const PpUpstreamFeedbackSampledViewBindingObservation& observation) noexcept {
    const auto& fact = observation.fact;
    return {
        .path = fact.path,
        .operation_use_mask = fact.operation_use_mask,
        .resource_depth = fact.resource_depth,
        .requested_format_class = fact.requested_format_class,
        .backing_format_class = fact.backing_format_class,
        .actual_format_class = fact.actual_format_class,
        .actual_aspect_class = fact.actual_aspect_class,
        .mapping_class = fact.mapping_class,
        .relation_mask = fact.relation_mask,
        .refresh_outcome_mask = fact.refresh_outcome_mask,
        .gpu_modified_before_mask = fact.gpu_modified_before_mask,
        .binding_count = observation.binding_count,
        .subresource_lookup =
            MakePpUpstreamFeedbackImageSubresourceLookupSnapshot(
                observation.subresource_lookup),
        .sampler_binding =
            MakePpUpstreamFeedbackSamplerBindingSnapshot(observation.sampler_binding),
        .overflowed = fact.overflowed || observation.overflowed,
        .disposition = observation.duplicate_drift || fact.overflowed || observation.overflowed
                           ? PpUpstreamFeedbackSampledViewBindingDisposition::InvalidTransition
                           : fact.disposition,
    };
}

struct PpUpstreamFeedbackProducerCandidateSnapshot {
    u32 candidate_count{};
    VideoCore::ImageProducerClass producer{VideoCore::ImageProducerClass::Unknown};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    u32 additional_ownership_transfer_count{};
    bool fresh{};
    bool scope_valid{};
    bool scope_overflow{};
    bool retains_private_state{};
    bool retains_image{};
    bool retains_vk_image{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
    PpUpstreamFeedbackProducerCandidateDisposition disposition{
        PpUpstreamFeedbackProducerCandidateDisposition::Idle};
};

struct PpUpstreamFeedbackProducerAdmissionSnapshot {
    u32 observe_attempts{};
    u32 stored{};
    u32 input_count{};
    u32 expected_input_count{};
    FinalGuestSurfaceStatus input_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss input_loss{};
    FinalGuestSurfaceStatus observation_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss observation_loss{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    u32 additional_ownership_transfer_count{};
    bool observation_stored{};
    bool selector_matched{};
    bool output_valid{};
    bool overflowed{};
    bool retains_private_state{};
    bool retains_image{};
    bool retains_vk_image{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
    PpUpstreamFeedbackProducerCandidateSnapshot candidate{};
    PpUpstreamFeedbackProducerDepthConsumerSnapshot depth_consumer{};
    PpUpstreamFeedbackSampledViewBindingSnapshot view_binding{};
    PpUpstreamFeedbackProducerAdmissionDisposition disposition{
        PpUpstreamFeedbackProducerAdmissionDisposition::Idle};
};

struct PpUpstreamFeedbackSelectionAuthoritySnapshot {
    u32 producer_observe_attempts{};
    u32 producer_stored{};
    u32 predecessor_opportunities{};
    u32 resolve_attempts{};
    u32 resolve_matches{};
    u32 select_attempts{};
    u32 select_successes{};
    FinalGuestSurfaceStatus resolve_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss resolve_loss{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    u32 additional_ownership_transfer_count{};
    bool input_available{};
    bool overflowed{};
    bool retains_private_state{};
    bool retains_image{};
    bool retains_vk_image{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
    PpUpstreamFeedbackProducerAdmissionSnapshot producer_admission{};
    PpUpstreamFeedbackSelectionAuthorityDisposition disposition{
        PpUpstreamFeedbackSelectionAuthorityDisposition::Idle};
};

struct PpUpstreamFeedbackSelectionPredecessorAttempt {
    PpTerminalScopeUpstreamInputRegistryResolution resolution{};
    bool input_available{};
    bool resolve_attempted{};
    bool select_attempted{};
    bool select_succeeded{};
};

struct PpUpstreamFeedbackSelectionAuthorityTake {
    PpUpstreamFeedbackPrePostTakeEvaluation evaluation{};
    PpUpstreamFeedbackSelectionAuthoritySnapshot authority{};
};

class PpUpstreamFeedbackSelectionAuthorityFrame {
public:
    static constexpr u32 MaxCount = 255;

    [[nodiscard]] constexpr bool Arm(
        PpUpstreamFeedbackPrePostFrame& frame,
        PpUpstreamFeedbackSelectionAuthorityMode mode) noexcept {
        if (mode == PpUpstreamFeedbackSelectionAuthorityMode::Disabled) {
            Reset(false);
            return false;
        }
        const bool armed = frame.Arm();
        Reset(mode == PpUpstreamFeedbackSelectionAuthorityMode::Exact && armed);
        return armed;
    }

    [[nodiscard]] constexpr bool Arm(
        PpUpstreamFeedbackPrePostFrame& frame, PpTerminalScopeDrawSelector selector,
        PpUpstreamFeedbackSelectionAuthorityMode mode) noexcept {
        if (mode == PpUpstreamFeedbackSelectionAuthorityMode::Disabled) {
            Reset(false);
            return false;
        }
        const bool armed = frame.Arm(selector);
        Reset(mode == PpUpstreamFeedbackSelectionAuthorityMode::Exact && armed);
        return armed;
    }

    [[nodiscard]] constexpr PpTerminalScopeUpstreamInputRegistryObservation ObserveProducer(
        PpTerminalScopeUpstreamInputRegistry& registry,
        VideoCore::ImageColorScopePrivateLink output, const PpTerminalScopeDrawSelector& draw,
        const PpTerminalScopeSampledInputs& inputs,
        const PpUpstreamFeedbackProducerDepthConsumerInput& depth_consumer = {},
        const PpUpstreamFeedbackSampledViewBindingObservation& view_binding = {}) noexcept {
        const bool selector_matched = registry.MatchesSelector(draw);
        const u32 expected_input_count = registry.ExpectedSampledInputCount();
        const bool output_valid = output.Valid();
        const auto observation = registry.Observe(output, draw, inputs);
        if (active) {
            Increment(snapshot.producer_observe_attempts);
            if (observation.stored) {
                Increment(snapshot.producer_stored);
            }
            ObserveProducerAdmission(selector_matched, output_valid, inputs.count,
                                     expected_input_count, inputs.status, inputs.loss,
                                     observation, inputs, depth_consumer, view_binding);
        }
        return observation;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackSelectionPredecessorAttempt ObservePredecessor(
        PpTerminalScopeUpstreamInputRegistry& registry, PpUpstreamFeedbackPrePostFrame& frame,
        std::span<const VideoCore::ImageColorScopePrivateLink> sampled_links, u32 selected,
        bool pre_post_enabled) noexcept {
        PpUpstreamFeedbackSelectionPredecessorAttempt attempt{};
        if (!active) {
            return attempt;
        }

        if (snapshot.predecessor_opportunities != 0) {
            invalid_transition = true;
        }
        Increment(snapshot.predecessor_opportunities);
        attempt.input_available = selected < sampled_links.size();
        snapshot.input_available |= attempt.input_available;
        if (!attempt.input_available) {
            return attempt;
        }

        Increment(snapshot.resolve_attempts);
        attempt.resolve_attempted = true;
        attempt.resolution = registry.Resolve(sampled_links[selected]);
        snapshot.resolve_status = attempt.resolution.status;
        snapshot.resolve_loss = attempt.resolution.loss;
        if (attempt.resolution.matched) {
            Increment(snapshot.resolve_matches);
        }
        if (!pre_post_enabled) {
            invalid_transition = true;
            return attempt;
        }
        if (!attempt.resolution.matched) {
            return attempt;
        }

        Increment(snapshot.select_attempts);
        attempt.select_attempted = true;
        attempt.select_succeeded = frame.Select(sampled_links[selected]);
        if (attempt.select_succeeded) {
            Increment(snapshot.select_successes);
        }
        return attempt;
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackSelectionAuthorityTake Take(
        PpUpstreamFeedbackPrePostFrame& frame, bool has_slot) noexcept {
        const auto evaluation = frame.TakeEvaluation(has_slot);
        auto taken = snapshot;
        if (active) {
            taken.disposition = Classify(taken, evaluation);
        }
        Reset(false);
        return {.evaluation = evaluation, .authority = taken};
    }

private:
    constexpr void Reset(bool active_) noexcept {
        snapshot = {};
        active = active_;
        invalid_transition = false;
        producer_admission_invalid_transition = false;
        producer_candidate_invalid_transition = false;
        producer_depth_consumer_invalid_transition = false;
        producer_view_binding_invalid_transition = false;
    }

    constexpr void Increment(u32& value) noexcept {
        if (value == MaxCount) {
            snapshot.overflowed = true;
            return;
        }
        ++value;
    }

    constexpr void IncrementProducerAdmission(u32& value) noexcept {
        if (value == MaxCount) {
            snapshot.producer_admission.overflowed = true;
            return;
        }
        ++value;
    }

    [[nodiscard]] static constexpr PpUpstreamFeedbackProducerAdmissionDisposition
    ClassifyProducerAdmissionCall(
        bool selector_matched, bool output_valid, u32 input_count,
        u32 expected_input_count, FinalGuestSurfaceStatus input_status,
        const FinalGuestSurfaceLoss& input_loss,
        const PpTerminalScopeUpstreamInputRegistryObservation& observation) noexcept {
        using Disposition = PpUpstreamFeedbackProducerAdmissionDisposition;
        const bool default_rejection =
            observation.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            !observation.loss.Any() && !observation.stored;
        const bool invalidation_rejection =
            observation.status == FinalGuestSurfaceStatus::InvalidationLoss &&
            observation.loss == FinalGuestSurfaceLoss{.invalidation = 1} &&
            !observation.stored;
        if (!selector_matched) {
            return default_rejection ? Disposition::SelectorMismatch
                                     : Disposition::InvalidTransition;
        }
        if (!output_valid) {
            return invalidation_rejection ? Disposition::InvalidOutput
                                          : Disposition::InvalidTransition;
        }
        if (input_status != FinalGuestSurfaceStatus::Complete) {
            return invalidation_rejection ? Disposition::SampledInputStatus
                                          : Disposition::InvalidTransition;
        }
        if (input_loss.Any()) {
            return invalidation_rejection ? Disposition::SampledInputLoss
                                          : Disposition::InvalidTransition;
        }
        if (input_count != expected_input_count) {
            return invalidation_rejection ? Disposition::SampledInputCountMismatch
                                          : Disposition::InvalidTransition;
        }
        if (observation.status == FinalGuestSurfaceStatus::CapacityLoss &&
            observation.loss == FinalGuestSurfaceLoss{.tile_capacity = 1} &&
            !observation.stored) {
            return Disposition::Capacity;
        }
        if (observation.status == FinalGuestSurfaceStatus::Complete &&
            !observation.loss.Any() && observation.stored) {
            return Disposition::Stored;
        }
        return Disposition::InvalidTransition;
    }

    [[nodiscard]] static constexpr bool HasDefaultProducerDrawEvidence(
        const PpTerminalScopePredecessor& candidate) noexcept {
        return candidate.draw_count == 0 &&
               candidate.last_draw == VideoCore::ImageColorScopeDrawKind::Unknown &&
               !candidate.indexed && candidate.element_count == 0 &&
               candidate.instance_count == 0 && candidate.sampled_images == 0 &&
               candidate.storage_writes == 0 && !candidate.clear_at_begin;
    }

    [[nodiscard]] static constexpr bool IsCanonicalProducerCandidate(
        const PpTerminalScopePredecessor& candidate) noexcept {
        const bool default_draw = HasDefaultProducerDrawEvidence(candidate);
        switch (candidate.producer) {
        case VideoCore::ImageProducerClass::Unknown:
            return !candidate.scope_valid && !candidate.scope_overflow && default_draw &&
                   candidate.status == FinalGuestSurfaceStatus::InvalidationLoss &&
                   candidate.loss == FinalGuestSurfaceLoss{.invalidation = 1};
        case VideoCore::ImageProducerClass::StorageImage:
        case VideoCore::ImageProducerClass::Transfer:
        case VideoCore::ImageProducerClass::CpuUpload:
            return !candidate.scope_valid && !candidate.scope_overflow && default_draw &&
                   candidate.status == FinalGuestSurfaceStatus::Complete &&
                   !candidate.loss.Any();
        case VideoCore::ImageProducerClass::ColorAttachment:
            if (candidate.scope_overflow) {
                return default_draw &&
                       candidate.status == FinalGuestSurfaceStatus::CapacityLoss &&
                       candidate.loss == FinalGuestSurfaceLoss{.tile_capacity = 1};
            }
            if (!candidate.scope_valid) {
                return default_draw &&
                       candidate.status == FinalGuestSurfaceStatus::InvalidationLoss &&
                       candidate.loss == FinalGuestSurfaceLoss{.invalidation = 1};
            }
            if (candidate.status == FinalGuestSurfaceStatus::GapLoss) {
                return default_draw && candidate.loss == FinalGuestSurfaceLoss{.gap = 1};
            }
            return candidate.draw_count != 0 &&
                   candidate.last_draw != VideoCore::ImageColorScopeDrawKind::Unknown &&
                   candidate.status == FinalGuestSurfaceStatus::Complete &&
                   !candidate.loss.Any();
        }
        return false;
    }

    [[nodiscard]] static constexpr PpUpstreamFeedbackProducerCandidateSnapshot
    ClassifyProducerCandidateCall(
        u32 expected_input_count, const PpTerminalScopeSampledInputs& inputs) noexcept {
        using Disposition = PpUpstreamFeedbackProducerCandidateDisposition;
        PpUpstreamFeedbackProducerCandidateSnapshot result{
            .candidate_count = inputs.count,
        };
        if (inputs.count > PpTerminalScopeSampledInputs::MaxInputs) {
            result.disposition = Disposition::InvalidTransition;
            return result;
        }
        if (expected_input_count != 1 || inputs.count != 1) {
            result.disposition = Disposition::NotSingleCandidate;
            return result;
        }

        const auto& input = inputs.inputs[0];
        const auto& candidate = input.producer;
        result.producer = candidate.producer;
        result.status = candidate.status;
        result.loss = candidate.loss;
        result.fresh = candidate.fresh;
        result.scope_valid = candidate.scope_valid;
        result.scope_overflow = candidate.scope_overflow;
        const u32 expected_alias_count = input.aliases_output ? 1u : 0u;
        if (inputs.alias_count != expected_alias_count ||
            inputs.status != candidate.status || inputs.loss != candidate.loss ||
            !IsCanonicalProducerCandidate(candidate)) {
            result.disposition = Disposition::InvalidTransition;
            return result;
        }
        if (candidate.producer == VideoCore::ImageProducerClass::Unknown) {
            result.disposition = Disposition::UnknownProducer;
        } else if (candidate.producer == VideoCore::ImageProducerClass::ColorAttachment &&
                   !candidate.scope_overflow && !candidate.scope_valid) {
            result.disposition = Disposition::InvalidColorScope;
        } else {
            result.disposition = Disposition::Exact;
        }
        return result;
    }

    [[nodiscard]] static constexpr bool SameProducerCandidateTuple(
        const PpUpstreamFeedbackProducerCandidateSnapshot& lhs,
        const PpUpstreamFeedbackProducerCandidateSnapshot& rhs) noexcept {
        return lhs.candidate_count == rhs.candidate_count && lhs.producer == rhs.producer &&
               lhs.status == rhs.status && lhs.loss == rhs.loss && lhs.fresh == rhs.fresh &&
               lhs.scope_valid == rhs.scope_valid &&
               lhs.scope_overflow == rhs.scope_overflow && lhs.disposition == rhs.disposition;
    }

    constexpr void ObserveProducerAdmission(
        bool selector_matched, bool output_valid, u32 input_count,
        u32 expected_input_count, FinalGuestSurfaceStatus input_status,
        FinalGuestSurfaceLoss input_loss,
        const PpTerminalScopeUpstreamInputRegistryObservation& observation,
        const PpTerminalScopeSampledInputs& inputs,
        const PpUpstreamFeedbackProducerDepthConsumerInput& depth_consumer,
        const PpUpstreamFeedbackSampledViewBindingObservation& view_binding) noexcept {
        using Disposition = PpUpstreamFeedbackProducerAdmissionDisposition;
        auto& value = snapshot.producer_admission;
        const bool first = value.observe_attempts == 0;
        IncrementProducerAdmission(value.observe_attempts);
        if (observation.stored) {
            IncrementProducerAdmission(value.stored);
        }
        const auto candidate = ClassifyProducerCandidateCall(expected_input_count, inputs);
        if (first) {
            value.candidate = candidate;
        } else if (!SameProducerCandidateTuple(value.candidate, candidate)) {
            producer_candidate_invalid_transition = true;
        }
        if (candidate.disposition ==
                PpUpstreamFeedbackProducerCandidateDisposition::InvalidTransition ||
            value.overflowed) {
            producer_candidate_invalid_transition = true;
        }
        if (producer_candidate_invalid_transition) {
            value.candidate.disposition =
                PpUpstreamFeedbackProducerCandidateDisposition::InvalidTransition;
        }
        const auto depth = ClassifyPpUpstreamFeedbackProducerDepthConsumerState({
            .expected_input_count = expected_input_count,
            .candidate_count = inputs.count,
            .selected_view = depth_consumer.selected_view,
            .current_writer = depth_consumer.current_writer,
            .transit = depth_consumer.transit,
            .live_state = depth_consumer.live_state,
        });
        if (first) {
            value.depth_consumer = depth;
        } else if (value.depth_consumer != depth) {
            producer_depth_consumer_invalid_transition = true;
        }
        if (depth.disposition ==
                PpUpstreamFeedbackProducerDepthConsumerDisposition::InvalidTransition ||
            value.overflowed) {
            producer_depth_consumer_invalid_transition = true;
        }
        if (producer_depth_consumer_invalid_transition) {
            value.depth_consumer.disposition =
                PpUpstreamFeedbackProducerDepthConsumerDisposition::InvalidTransition;
        }
        const auto binding = MakePpUpstreamFeedbackSampledViewBindingSnapshot(view_binding);
        if (first) {
            value.view_binding = binding;
        } else if (value.view_binding != binding) {
            if (value.view_binding.subresource_lookup != binding.subresource_lookup) {
                value.view_binding.subresource_lookup.disposition =
                    PpUpstreamFeedbackImageSubresourceLookupDisposition::InvalidTransition;
            }
            if (value.view_binding.sampler_binding != binding.sampler_binding) {
                value.view_binding.sampler_binding.disposition =
                    PpUpstreamFeedbackSamplerBindingDisposition::InvalidTransition;
            }
            producer_view_binding_invalid_transition = true;
        }
        if (binding.subresource_lookup.disposition ==
                PpUpstreamFeedbackImageSubresourceLookupDisposition::InvalidTransition ||
            value.overflowed) {
            value.view_binding.subresource_lookup.disposition =
                PpUpstreamFeedbackImageSubresourceLookupDisposition::InvalidTransition;
        }
        if (binding.sampler_binding.disposition ==
                PpUpstreamFeedbackSamplerBindingDisposition::InvalidTransition ||
            value.overflowed) {
            value.view_binding.sampler_binding.disposition =
                PpUpstreamFeedbackSamplerBindingDisposition::InvalidTransition;
            producer_view_binding_invalid_transition = true;
        }
        if (binding.disposition ==
                PpUpstreamFeedbackSampledViewBindingDisposition::InvalidTransition ||
            value.overflowed) {
            producer_view_binding_invalid_transition = true;
        }
        if (producer_view_binding_invalid_transition) {
            value.view_binding.disposition =
                PpUpstreamFeedbackSampledViewBindingDisposition::InvalidTransition;
        }
        const auto disposition = ClassifyProducerAdmissionCall(
            selector_matched, output_valid, input_count, expected_input_count,
            input_status, input_loss, observation);
        if (first) {
            value.input_count = input_count;
            value.expected_input_count = expected_input_count;
            value.input_status = input_status;
            value.input_loss = input_loss;
            value.observation_status = observation.status;
            value.observation_loss = observation.loss;
            value.observation_stored = observation.stored;
            value.selector_matched = selector_matched;
            value.output_valid = output_valid;
            value.disposition = disposition;
        } else if (value.input_count != input_count ||
                   value.expected_input_count != expected_input_count ||
                   value.input_status != input_status || value.input_loss != input_loss ||
                   value.observation_status != observation.status ||
                   value.observation_loss != observation.loss ||
                   value.observation_stored != observation.stored ||
                   value.selector_matched != selector_matched ||
                   value.output_valid != output_valid || value.disposition != disposition) {
            producer_admission_invalid_transition = true;
        }
        if (disposition == Disposition::InvalidTransition || value.overflowed) {
            producer_admission_invalid_transition = true;
        }
        if (producer_admission_invalid_transition) {
            value.disposition = Disposition::InvalidTransition;
        }
    }

    [[nodiscard]] constexpr PpUpstreamFeedbackSelectionAuthorityDisposition Classify(
        const PpUpstreamFeedbackSelectionAuthoritySnapshot& value,
        const PpUpstreamFeedbackPrePostTakeEvaluation& evaluation) const noexcept {
        using Disposition = PpUpstreamFeedbackSelectionAuthorityDisposition;
        using FirstFailure = PpUpstreamFeedbackPrePostTakeFirstFailure;
        const bool counts_valid =
            value.producer_stored <= value.producer_observe_attempts &&
            value.resolve_matches <= value.resolve_attempts &&
            value.select_attempts <= value.resolve_matches &&
            value.select_successes <= value.select_attempts;
        if (invalid_transition || value.overflowed || !counts_valid) {
            return Disposition::InvalidTransition;
        }
        if (value.predecessor_opportunities == 0) {
            return Disposition::NoOpportunity;
        }
        if (value.predecessor_opportunities != 1) {
            return Disposition::InvalidTransition;
        }
        if (!value.input_available) {
            return value.resolve_attempts == 0 && value.resolve_matches == 0 &&
                           value.select_attempts == 0 && value.select_successes == 0 &&
                           evaluation.first_failure == FirstFailure::MissingSelection
                       ? Disposition::InputUnavailable
                       : Disposition::InvalidTransition;
        }
        if (value.resolve_attempts != 1) {
            return Disposition::InvalidTransition;
        }
        if (value.resolve_matches == 0) {
            return value.select_attempts == 0 && value.select_successes == 0 &&
                           evaluation.first_failure == FirstFailure::MissingSelection
                       ? Disposition::RegistryUnmatched
                       : Disposition::InvalidTransition;
        }
        if (value.resolve_matches != 1 || value.producer_stored == 0 ||
            value.resolve_status != FinalGuestSurfaceStatus::Complete ||
            value.resolve_loss.Any() || value.select_attempts != 1) {
            return Disposition::InvalidTransition;
        }
        if (value.select_successes == 0) {
            return evaluation.first_failure == FirstFailure::PreexistingFrameLoss
                       ? Disposition::FrameSelectRejected
                       : Disposition::InvalidTransition;
        }
        if (value.select_successes == 1 && evaluation.first_failure == FirstFailure::Exact) {
            return Disposition::Selected;
        }
        return Disposition::InvalidTransition;
    }

    PpUpstreamFeedbackSelectionAuthoritySnapshot snapshot{};
    bool active{};
    bool invalid_transition{};
    bool producer_admission_invalid_transition{};
    bool producer_candidate_invalid_transition{};
    bool producer_depth_consumer_invalid_transition{};
    bool producer_view_binding_invalid_transition{};
};

[[nodiscard]] constexpr bool ArmPpUpstreamFeedbackSelectionAuthority(
    PpUpstreamFeedbackSelectionAuthorityFrame& authority,
    PpUpstreamFeedbackPrePostFrame& frame,
    PpUpstreamFeedbackSelectionAuthorityMode mode =
        PpUpstreamFeedbackSelectionAuthorityMode::Exact) noexcept {
    return authority.Arm(frame, mode);
}

[[nodiscard]] constexpr bool ArmPpUpstreamFeedbackSelectionAuthority(
    PpUpstreamFeedbackSelectionAuthorityFrame& authority,
    PpUpstreamFeedbackPrePostFrame& frame, PpTerminalScopeDrawSelector selector,
    PpUpstreamFeedbackSelectionAuthorityMode mode) noexcept {
    return authority.Arm(frame, selector, mode);
}

[[nodiscard]] constexpr PpTerminalScopeUpstreamInputRegistryObservation
ObservePpUpstreamFeedbackSelectionProducer(
    PpUpstreamFeedbackSelectionAuthorityFrame& authority,
    PpTerminalScopeUpstreamInputRegistry& registry, VideoCore::ImageColorScopePrivateLink output,
    const PpTerminalScopeDrawSelector& draw,
    const PpTerminalScopeSampledInputs& inputs,
    const PpUpstreamFeedbackProducerDepthConsumerInput& depth_consumer = {},
    const PpUpstreamFeedbackSampledViewBindingObservation& view_binding = {}) noexcept {
    return authority.ObserveProducer(registry, output, draw, inputs, depth_consumer, view_binding);
}

[[nodiscard]] constexpr PpUpstreamFeedbackSelectionPredecessorAttempt
ObservePpUpstreamFeedbackSelectionPredecessorAttempt(
    PpUpstreamFeedbackSelectionAuthorityFrame& authority,
    PpTerminalScopeUpstreamInputRegistry& registry, PpUpstreamFeedbackPrePostFrame& frame,
    std::span<const VideoCore::ImageColorScopePrivateLink> sampled_links, u32 selected,
    bool pre_post_enabled) noexcept {
    return authority.ObservePredecessor(registry, frame, sampled_links, selected,
                                        pre_post_enabled);
}

[[nodiscard]] constexpr PpUpstreamFeedbackSelectionAuthorityTake
TakePpUpstreamFeedbackSelectionAuthority(
    PpUpstreamFeedbackSelectionAuthorityFrame& authority,
    PpUpstreamFeedbackPrePostFrame& frame, bool has_slot) noexcept {
    return authority.Take(frame, has_slot);
}

struct PpUpstreamFeedbackSelectionAuthorityReport {
    u64 sequence{};
    u32 producer_observe_attempts{};
    u32 producer_stored{};
    u32 predecessor_opportunities{};
    u32 resolve_attempts{};
    u32 resolve_matches{};
    u32 select_attempts{};
    u32 select_successes{};
    FinalGuestSurfaceStatus resolve_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss resolve_loss{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    u32 additional_ownership_transfer_count{};
    bool input_available{};
    bool overflowed{};
    bool retains_private_state{};
    bool retains_image{};
    bool retains_vk_image{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
    PpUpstreamFeedbackSelectionAuthorityDisposition disposition{
        PpUpstreamFeedbackSelectionAuthorityDisposition::Idle};
};

[[nodiscard]] constexpr PpUpstreamFeedbackSelectionAuthorityReport
MakePpUpstreamFeedbackSelectionAuthorityReport(
    u64 sequence, const PpUpstreamFeedbackSelectionAuthoritySnapshot& snapshot) noexcept {
    return {
        .sequence = sequence,
        .producer_observe_attempts = snapshot.producer_observe_attempts,
        .producer_stored = snapshot.producer_stored,
        .predecessor_opportunities = snapshot.predecessor_opportunities,
        .resolve_attempts = snapshot.resolve_attempts,
        .resolve_matches = snapshot.resolve_matches,
        .select_attempts = snapshot.select_attempts,
        .select_successes = snapshot.select_successes,
        .resolve_status = snapshot.resolve_status,
        .resolve_loss = snapshot.resolve_loss,
        .additional_gpu_copy_count = snapshot.additional_gpu_copy_count,
        .additional_slot_count = snapshot.additional_slot_count,
        .additional_invalidation_count = snapshot.additional_invalidation_count,
        .additional_release_count = snapshot.additional_release_count,
        .additional_ownership_transfer_count = snapshot.additional_ownership_transfer_count,
        .input_available = snapshot.input_available,
        .overflowed = snapshot.overflowed,
        .retains_private_state = snapshot.retains_private_state,
        .retains_image = snapshot.retains_image,
        .retains_vk_image = snapshot.retains_vk_image,
        .affects_capture_behavior = snapshot.affects_capture_behavior,
        .affects_renderer_behavior = snapshot.affects_renderer_behavior,
        .disposition = snapshot.disposition,
    };
}

[[nodiscard]] inline std::string FormatPpUpstreamFeedbackSelectionAuthorityReport(
    const PpUpstreamFeedbackSelectionAuthorityReport& report) {
    return "FGSCTUA s=" + std::to_string(report.sequence) +
           " oa=" + std::to_string(report.producer_observe_attempts) +
           " os=" + std::to_string(report.producer_stored) +
           " po=" + std::to_string(report.predecessor_opportunities) +
           " ia=" + std::to_string(report.input_available ? 1 : 0) +
           " ra=" + std::to_string(report.resolve_attempts) +
           " rm=" + std::to_string(report.resolve_matches) +
           " rs=" + std::to_string(static_cast<u32>(report.resolve_status)) +
           " rlm=" + std::to_string(FinalGuestSurfaceLossMask(report.resolve_loss, 0)) +
           " sa=" + std::to_string(report.select_attempts) +
           " ss=" + std::to_string(report.select_successes) +
           " ov=" + std::to_string(report.overflowed ? 1 : 0) +
           " d=" + std::to_string(static_cast<u32>(report.disposition));
}

struct PpUpstreamFeedbackProducerAdmissionReport {
    u64 sequence{};
    u32 observe_attempts{};
    u32 stored{};
    u32 input_count{};
    u32 expected_input_count{};
    FinalGuestSurfaceStatus input_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss input_loss{};
    FinalGuestSurfaceStatus observation_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss observation_loss{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    u32 additional_ownership_transfer_count{};
    bool observation_stored{};
    bool selector_matched{};
    bool output_valid{};
    bool overflowed{};
    bool retains_private_state{};
    bool retains_image{};
    bool retains_vk_image{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
    PpUpstreamFeedbackProducerCandidateSnapshot candidate{};
    PpUpstreamFeedbackProducerDepthConsumerSnapshot depth_consumer{};
    PpUpstreamFeedbackSampledViewBindingSnapshot view_binding{};
    PpUpstreamFeedbackProducerAdmissionDisposition disposition{
        PpUpstreamFeedbackProducerAdmissionDisposition::Idle};
};

[[nodiscard]] constexpr PpUpstreamFeedbackProducerAdmissionReport
MakePpUpstreamFeedbackProducerAdmissionReport(
    u64 sequence, const PpUpstreamFeedbackProducerAdmissionSnapshot& snapshot) noexcept {
    return {
        .sequence = sequence,
        .observe_attempts = snapshot.observe_attempts,
        .stored = snapshot.stored,
        .input_count = snapshot.input_count,
        .expected_input_count = snapshot.expected_input_count,
        .input_status = snapshot.input_status,
        .input_loss = snapshot.input_loss,
        .observation_status = snapshot.observation_status,
        .observation_loss = snapshot.observation_loss,
        .additional_gpu_copy_count = snapshot.additional_gpu_copy_count,
        .additional_slot_count = snapshot.additional_slot_count,
        .additional_invalidation_count = snapshot.additional_invalidation_count,
        .additional_release_count = snapshot.additional_release_count,
        .additional_ownership_transfer_count = snapshot.additional_ownership_transfer_count,
        .observation_stored = snapshot.observation_stored,
        .selector_matched = snapshot.selector_matched,
        .output_valid = snapshot.output_valid,
        .overflowed = snapshot.overflowed,
        .retains_private_state = snapshot.retains_private_state,
        .retains_image = snapshot.retains_image,
        .retains_vk_image = snapshot.retains_vk_image,
        .affects_capture_behavior = snapshot.affects_capture_behavior,
        .affects_renderer_behavior = snapshot.affects_renderer_behavior,
        .candidate = snapshot.candidate,
        .depth_consumer = snapshot.depth_consumer,
        .view_binding = snapshot.view_binding,
        .disposition = snapshot.disposition,
    };
}

[[nodiscard]] inline std::string FormatPpUpstreamFeedbackProducerAdmissionReport(
    const PpUpstreamFeedbackProducerAdmissionReport& report) {
    const auto sampler_compare = [](VideoCore::SamplerCompareRelation relation) -> u32 {
        using Relation = VideoCore::SamplerCompareRelation;
        if (relation == Relation::Idle) {
            return 0;
        }
        if (relation == Relation::RegularExact || relation == Relation::DepthCompareExact) {
            return 1;
        }
        if (relation == Relation::MixedUse || relation == Relation::MissingOperationUse ||
            relation == Relation::Invalid || relation == Relation::DescriptorExact) {
            return 3;
        }
        return 2;
    };
    return "FGSCTUO s=" + std::to_string(report.sequence) +
           " oa=" + std::to_string(report.observe_attempts) +
           " os=" + std::to_string(report.stored) +
           " sm=" + std::to_string(report.selector_matched ? 1 : 0) +
           " vo=" + std::to_string(report.output_valid ? 1 : 0) +
           " ic=" + std::to_string(report.input_count) +
           " ec=" + std::to_string(report.expected_input_count) +
           " is=" + std::to_string(static_cast<u32>(report.input_status)) +
           " ilm=" + std::to_string(FinalGuestSurfaceLossMask(report.input_loss, 0)) +
           " rs=" + std::to_string(static_cast<u32>(report.observation_status)) +
           " rlm=" + std::to_string(FinalGuestSurfaceLossMask(report.observation_loss, 0)) +
           " rst=" + std::to_string(report.observation_stored ? 1 : 0) +
           " ov=" + std::to_string(report.overflowed ? 1 : 0) +
           " d=" + std::to_string(static_cast<u32>(report.disposition)) +
           " cc=" + std::to_string(report.candidate.candidate_count) +
           " pc=" + std::to_string(static_cast<u32>(report.candidate.producer)) +
           " pf=" + std::to_string(report.candidate.fresh ? 1 : 0) +
           " pv=" + std::to_string(report.candidate.scope_valid ? 1 : 0) +
           " px=" + std::to_string(report.candidate.scope_overflow ? 1 : 0) +
           " ps=" + std::to_string(static_cast<u32>(report.candidate.status)) +
           " plm=" + std::to_string(FinalGuestSurfaceLossMask(report.candidate.loss, 0)) +
           " pd=" + std::to_string(static_cast<u32>(report.candidate.disposition)) +
           " dc=" + std::to_string(report.depth_consumer.selected_subresource_count) +
           " wv=" + std::to_string(report.depth_consumer.writer_valid ? 1 : 0) +
           " wc=" + std::to_string(report.depth_consumer.writer_current ? 1 : 0) +
           " re=" + std::to_string(report.depth_consumer.resource_exact ? 1 : 0) +
           " be=" + std::to_string(report.depth_consumer.backing_exact ? 1 : 0) +
           " me=" + std::to_string(report.depth_consumer.mutation_exact ? 1 : 0) +
           " ae=" + std::to_string(report.depth_consumer.depth_aspect_exact ? 1 : 0) +
           " se=" + std::to_string(report.depth_consumer.selected_subresource_exact ? 1 : 0) +
           " sc=" +
           std::to_string(report.depth_consumer.synchronization_covers_selected ? 1 : 0) +
           " pw=" + std::to_string(report.depth_consumer.prior_depth_stencil_write_count) +
           " ar=" + std::to_string(report.depth_consumer.already_readable_count) +
           " bc=" + std::to_string(report.depth_consumer.barrier_count) +
           " cv=" + std::to_string(report.depth_consumer.covering_barrier_count) +
           " sl=" +
           std::to_string(report.depth_consumer.source_depth_stencil_layout_cover_count) +
           " sa=" +
           std::to_string(report.depth_consumer.source_depth_stencil_access_cover_count) +
           " ss=" +
           std::to_string(report.depth_consumer.source_depth_stencil_stage_cover_count) +
           " dl=" + std::to_string(report.depth_consumer.destination_layout_cover_count) +
           " da=" + std::to_string(report.depth_consumer.destination_access_cover_count) +
           " ds=" + std::to_string(report.depth_consumer.destination_stage_cover_count) +
           " pr=" + std::to_string(report.depth_consumer.post_readable_count) +
           " bd=" + std::to_string(report.depth_consumer.binding_count) +
           " ll=" + std::to_string(report.depth_consumer.live_layout_readable_count) +
           " la=" + std::to_string(report.depth_consumer.live_access_readable_count) +
           " ls=" + std::to_string(report.depth_consumer.live_stage_readable_count) +
           " lr=" + std::to_string(report.depth_consumer.live_readable_count) +
           " bn=" + std::to_string(report.depth_consumer.barrier_needed ? 1 : 0) +
           " bs=" + std::to_string(report.depth_consumer.barrier_submitted ? 1 : 0) +
           " ub=" + std::to_string(report.depth_consumer.uniform_before ? 1 : 0) +
           " ua=" + std::to_string(report.depth_consumer.uniform_after ? 1 : 0) +
           " of=" + std::to_string(report.depth_consumer.overflowed ? 1 : 0) +
           " dd=" + std::to_string(static_cast<u32>(report.depth_consumer.disposition)) +
           " bp=" + std::to_string(static_cast<u32>(report.view_binding.path)) +
           " ou=" + std::to_string(report.view_binding.operation_use_mask) +
           " rd=" + std::to_string(report.view_binding.resource_depth ? 1 : 0) +
           " rf=" +
           std::to_string(static_cast<u32>(report.view_binding.requested_format_class)) +
           " bf=" + std::to_string(static_cast<u32>(report.view_binding.backing_format_class)) +
           " vf=" + std::to_string(static_cast<u32>(report.view_binding.actual_format_class)) +
           " va=" + std::to_string(static_cast<u32>(report.view_binding.actual_aspect_class)) +
           " vm=" + std::to_string(static_cast<u32>(report.view_binding.mapping_class)) +
           " wx=" + std::to_string(report.view_binding.relation_mask) +
           " ur=" + std::to_string(report.view_binding.refresh_outcome_mask) +
           " ug=" + std::to_string(report.view_binding.gpu_modified_before_mask) +
           " vb=" + std::to_string(report.view_binding.binding_count) +
           " vx=" + std::to_string(report.view_binding.overflowed ? 1 : 0) +
           " vd=" + std::to_string(static_cast<u32>(report.view_binding.disposition)) +
           " lp=" +
           std::to_string(static_cast<u32>(report.view_binding.subresource_lookup.path)) +
           " lq=" +
           std::to_string(static_cast<u32>(report.view_binding.subresource_lookup.relation)) +
           " pe=" +
           std::to_string(report.view_binding.subresource_lookup.perfect_candidate_count) +
           " oe=" +
           std::to_string(report.view_binding.subresource_lookup.overlap_candidate_count) +
           " aa=" +
           std::to_string(report.view_binding.subresource_lookup.actual_accepted ? 1 : 0) +
           " ca=" +
           std::to_string(report.view_binding.subresource_lookup.canonical_accepted ? 1 : 0) +
           " eq=" +
           std::to_string(report.view_binding.subresource_lookup.actual_canonical_equal ? 1 : 0) +
           " nm=" +
           std::to_string(report.view_binding.subresource_lookup.nonzero_mip ? 1 : 0) +
           " nl=" +
           std::to_string(report.view_binding.subresource_lookup.nonzero_layer ? 1 : 0) +
           " lg=" +
           std::to_string(report.view_binding.subresource_lookup.logical_single_layer ? 1 : 0) +
           " pa=" +
           std::to_string(report.view_binding.subresource_lookup.parent_stride_aligned ? 1 : 0) +
           " pb=" +
           std::to_string(report.view_binding.subresource_lookup.bounded_by_parent ? 1 : 0) +
           " pg=" +
           std::to_string(report.view_binding.subresource_lookup.parent_gpu_authoritative ? 1 : 0) +
           " cs=" +
           std::to_string(report.view_binding.subresource_lookup.created_standalone ? 1 : 0) +
           " qc=" +
           std::to_string(report.view_binding.subresource_lookup.provenance_current ? 1 : 0) +
           " ob=" +
           std::to_string(report.view_binding.subresource_lookup.observed ? 1 : 0) +
           " cd=" +
           std::to_string(report.view_binding.subresource_lookup.candidate_drift ? 1 : 0) +
           " lc=" + std::to_string(report.view_binding.subresource_lookup.lookup_count) +
           " lx=" +
           std::to_string(report.view_binding.subresource_lookup.overflowed ? 1 : 0) +
           " ld=" +
           std::to_string(static_cast<u32>(report.view_binding.subresource_lookup.disposition)) +
           " sr=" +
           std::to_string(static_cast<u32>(report.view_binding.sampler_binding.creation.disposition)) +
           " sl=" +
           std::to_string(static_cast<u32>(report.view_binding.sampler_binding.creation.lod_relation)) +
           " sc=" +
           std::to_string(sampler_compare(
               report.view_binding.sampler_binding.creation.compare_relation)) +
           " sp=" +
           std::to_string(static_cast<u32>(
               report.view_binding.sampler_binding.creation.reduction_relation)) +
           " sa=" +
           std::to_string(static_cast<u32>(report.view_binding.sampler_binding.adjustment_outcome)) +
           " so=" +
           std::to_string(static_cast<u32>(report.view_binding.sampler_binding.creation.pnext_order)) +
           " si=" +
           std::to_string(report.view_binding.sampler_binding.inline_sampler ? 1 : 0) +
           " sh=" +
           std::to_string(static_cast<u32>(report.view_binding.sampler_binding.cache.disposition)) +
           " sd=" +
           std::to_string(static_cast<u32>(report.view_binding.sampler_binding.descriptor_relation)) +
           " sb=" + std::to_string(report.view_binding.sampler_binding.binding_count) +
           " sx=" + std::to_string(report.view_binding.sampler_binding.overflowed ? 1 : 0) +
           " sf=" +
           std::to_string(static_cast<u32>(report.view_binding.sampler_binding.disposition));
}

struct PpDepthPredecessorHint {
    PpTerminalScopeDrawSelector draw{};
    u64 fragment_pgm_hash{};
    bool valid{};

    [[nodiscard]] constexpr bool Valid() const noexcept {
        return valid && fragment_pgm_hash != 0 && MatchesPpTerminalScopeDraw(draw, draw);
    }

    bool operator==(const PpDepthPredecessorHint&) const = default;
};

[[nodiscard]] constexpr bool ShouldUseConfiguredDepthCandidateFallback(
    bool depth_pre_post_mode, bool depth_predecessor_hint_mode) noexcept {
    return depth_pre_post_mode && !depth_predecessor_hint_mode;
}

[[nodiscard]] constexpr bool IsPpDepthPredecessorMultiHopHintMode(
    bool depth_predecessor_hint_mode, u32 configured_hops) noexcept {
    return depth_predecessor_hint_mode && (configured_hops == 2 || configured_hops == 3);
}

[[nodiscard]] constexpr bool IsPpDepthPredecessorCrossScopePairHintMode(
    bool depth_predecessor_hint_mode, u32 configured_hops) noexcept {
    return depth_predecessor_hint_mode && configured_hops == 4;
}

[[nodiscard]] constexpr u32 ResolvePpDepthPredecessorMultiHopLifecycleHops(
    bool depth_predecessor_hint_mode, u32 configured_hops) noexcept {
    return IsPpDepthPredecessorMultiHopHintMode(depth_predecessor_hint_mode, configured_hops)
               ? configured_hops
               : 2;
}

[[nodiscard]] constexpr u32 ResolvePpDepthPredecessorCrossScopeBootstrapLifecycleHops(
    bool depth_predecessor_hint_mode, u32 configured_hops) noexcept {
    return IsPpDepthPredecessorCrossScopePairHintMode(depth_predecessor_hint_mode,
                                                      configured_hops)
               ? 3
               : 0;
}

enum class PpDepthPredecessorRuntimePath : u32 {
    Disabled,
    OneHop,
    MultiHop,
    CrossScopeBootstrap,
    CrossScopeArmed,
};

struct PpDepthPredecessorRuntimeFramePlanInput {
    bool hint_mode{};
    u32 configured_hops{};
    bool cross_scope_pair_ready{};
    bool prospective_capture{};
};

struct PpDepthPredecessorRuntimeFramePlan {
    PpDepthPredecessorRuntimePath path{PpDepthPredecessorRuntimePath::Disabled};
    u32 structural_hops{};
    bool begin_one_hop{};
    bool begin_multi_hop{};
    bool begin_cross_scope{};
    bool feedback_eligible{};
    bool arm_feedback{};
    bool acquire_slot{};
    bool training_only{};
    bool publish_feedback{};
    bool allow_configured_fallback{};
};

[[nodiscard]] constexpr PpDepthPredecessorRuntimeFramePlan
PlanPpDepthPredecessorRuntimeFrame(
    const PpDepthPredecessorRuntimeFramePlanInput& input) noexcept {
    PpDepthPredecessorRuntimeFramePlan result{};
    if (!input.hint_mode) {
        return result;
    }
    if (input.configured_hops == 1) {
        result.path = PpDepthPredecessorRuntimePath::OneHop;
        result.structural_hops = 1;
    } else if (IsPpDepthPredecessorMultiHopHintMode(true, input.configured_hops)) {
        result.path = PpDepthPredecessorRuntimePath::MultiHop;
        result.structural_hops = input.configured_hops;
    } else if (IsPpDepthPredecessorCrossScopePairHintMode(true,
                                                          input.configured_hops)) {
        result.path = input.cross_scope_pair_ready
                          ? PpDepthPredecessorRuntimePath::CrossScopeArmed
                          : PpDepthPredecessorRuntimePath::CrossScopeBootstrap;
        result.structural_hops = input.cross_scope_pair_ready ? 0 : 3;
    } else {
        return result;
    }
    if (!input.prospective_capture) {
        return result;
    }
    result.begin_one_hop = result.path == PpDepthPredecessorRuntimePath::OneHop;
    result.begin_multi_hop = result.path == PpDepthPredecessorRuntimePath::MultiHop ||
                             result.path ==
                                 PpDepthPredecessorRuntimePath::CrossScopeBootstrap;
    result.begin_cross_scope =
        result.path == PpDepthPredecessorRuntimePath::CrossScopeArmed;
    result.feedback_eligible = true;
    result.arm_feedback = true;
    result.acquire_slot = true;
    result.training_only =
        result.path == PpDepthPredecessorRuntimePath::CrossScopeBootstrap;
    result.publish_feedback = !result.training_only;
    return result;
}

struct PpDepthPredecessorWriterStamp {
    PpDepthPredecessorHint hint{};
    VideoCore::ImageColorScopePrivateLink link{};
    VideoCore::SubresourceRange range{};
    u64 backing_generation{};
    u64 mutation_epoch{};
    bool valid{};

    [[nodiscard]] constexpr bool Valid() const noexcept;

    bool operator==(const PpDepthPredecessorWriterStamp&) const = default;
};

struct PpDepthPredecessorCandidateStamp {
    bool valid{};
    u64 generation{};
    u32 candidate_index{};
    u32 recorded_plane_mask{};
    PpDepthPredecessorWriterStamp writer{};

    bool operator==(const PpDepthPredecessorCandidateStamp&) const = default;
};

struct PpDepthPredecessorWriterIntent {
    PpDepthPredecessorHint hint{};
    VideoCore::ImageColorScopePrivateLink link{};
    VideoCore::SubresourceRange range{};
    u64 backing_generation{};
    u64 mutation_epoch{};
    bool valid{};

    [[nodiscard]] constexpr bool Valid() const noexcept;

    bool operator==(const PpDepthPredecessorWriterIntent&) const = default;
};

struct PpDepthPredecessorWriterObservation {
    VideoCore::ImageColorScopeDrawKind kind{VideoCore::ImageColorScopeDrawKind::Unknown};
    VideoCore::ImageColorScopePrivateLink link{};
    VideoCore::SubresourceRange range{};
    u64 mutation_epoch{};
    u64 backing_generation{};
    u64 fragment_pgm_hash{};
    u64 upstream_candidate_generation{};
    u32 element_count{};
    u32 instance_count{};
    u32 sampled_images{};
    u32 storage_writes{};
    u32 upstream_candidate_index{};
    bool depth_aspect{};
    bool indexed{};
    bool depth_write{};
    bool upstream_candidate_valid{};
    bool valid{};
};

struct PpDepthPredecessorObservationInput {
    PpDepthPredecessorWriterObservation observed{};
    VideoCore::ImageColorScopePrivateLink current_link{};
    VideoCore::SubresourceRange current_range{};
    u64 current_backing_generation{};
    u64 current_mutation_epoch{};
    u64 expected_candidate_generation{};
    u32 recorded_plane_mask{};
};

struct PpDepthPredecessorObservationResult {
    PpDepthPredecessorHint hint{};
    PpDepthPredecessorWriterStamp writer{};
    PpDepthPredecessorCandidateStamp candidate{};
};

[[nodiscard]] constexpr bool IsCanonicalPpDepthPredecessorRange(
    const VideoCore::SubresourceRange& range) noexcept {
    return range.base.level == 0 && range.base.layer == 0 && range.extent.levels == 1 &&
           range.extent.layers == 1;
}

[[nodiscard]] constexpr bool PpDepthPredecessorWriterStamp::Valid() const noexcept {
    return valid && hint.Valid() && link.Valid() &&
           IsCanonicalPpDepthPredecessorRange(range) && backing_generation != 0 &&
           mutation_epoch != 0;
}

[[nodiscard]] constexpr bool PpDepthPredecessorWriterIntent::Valid() const noexcept {
    return valid && hint.Valid() && link.Valid() &&
           IsCanonicalPpDepthPredecessorRange(range) && backing_generation != 0 &&
           mutation_epoch != 0;
}

[[nodiscard]] constexpr bool IsValidPpDepthPredecessorDraw(
    const PpTerminalScopeDrawSelector& draw) noexcept {
    return draw.kind != VideoCore::ImageColorScopeDrawKind::Unknown && draw.instance_count != 0 &&
           draw.sampled_images != 0 &&
           (draw.kind != VideoCore::ImageColorScopeDrawKind::Direct || draw.element_count != 0);
}

[[nodiscard]] constexpr PpDepthPredecessorObservationResult
BuildPpDepthPredecessorObservation(const PpDepthPredecessorObservationInput& input) noexcept {
    const auto& observed = input.observed;
    const PpTerminalScopeDrawSelector draw{
        .kind = observed.kind,
        .indexed = observed.indexed,
        .element_count = observed.element_count,
        .instance_count = observed.instance_count,
        .sampled_images = observed.sampled_images,
        .storage_writes = observed.storage_writes,
    };
    const bool exact =
        observed.valid && observed.link.Valid() && input.current_link.Valid() &&
        observed.link == input.current_link && IsCanonicalPpDepthPredecessorRange(observed.range) &&
        IsCanonicalPpDepthPredecessorRange(input.current_range) &&
        observed.range == input.current_range && observed.backing_generation != 0 &&
        input.current_backing_generation != 0 &&
        observed.backing_generation == input.current_backing_generation &&
        observed.mutation_epoch != 0 && input.current_mutation_epoch != 0 &&
        observed.mutation_epoch == input.current_mutation_epoch &&
        observed.depth_aspect && observed.depth_write && IsValidPpDepthPredecessorDraw(draw) &&
        observed.fragment_pgm_hash != 0;
    if (!exact) {
        return {};
    }

    PpDepthPredecessorObservationResult result{
        .hint = {
            .draw = draw,
            .fragment_pgm_hash = observed.fragment_pgm_hash,
            .valid = true,
        },
    };
    result.writer = {
        .hint = result.hint,
        .link = observed.link,
        .range = observed.range,
        .backing_generation = observed.backing_generation,
        .mutation_epoch = observed.mutation_epoch,
        .valid = true,
    };
    if (observed.upstream_candidate_valid && input.expected_candidate_generation != 0 &&
        observed.upstream_candidate_generation == input.expected_candidate_generation &&
        observed.upstream_candidate_index < PpUpstreamFeedbackPrePostPlan::MaxCandidates &&
        input.recorded_plane_mask == 0x3u) {
        result.candidate = {
            .valid = true,
            .generation = observed.upstream_candidate_generation,
            .candidate_index = observed.upstream_candidate_index,
            .recorded_plane_mask = input.recorded_plane_mask,
            .writer = result.writer,
        };
    }
    return result;
}

enum class PpDepthPredecessorTwoHopPhase : u32 {
    LearnP1,
    LearnP2,
    ArmedP2,
    GenerationWrap,
};

enum class PpDepthPredecessorTwoHopDecision : u32 {
    Idle,
    LearnedP1,
    Committed,
    Selected,
    MissingAnchor,
    MissingP1,
    MissingP1Post,
    DuplicateP1,
    DuplicateP1Post,
    DuplicateAnchor,
    P1Mismatch,
    P1StampMismatch,
    P2Mismatch,
    CandidateMismatch,
    IdentityCollision,
    GenerationWrap,
};

enum class PpDepthPredecessorTwoHopIdentityCollision : u32 {
    None,
    P2EqualsP1,
    P2EqualsConfiguredAnchor,
};

enum class PpDepthPredecessorTwoHopChainKind : u32 {
    Distinct,
    SameWriterImmediatePrior,
    DistinctPrefixBeforeSameWriter,
};

enum class PpDepthPredecessorTwoHopOccurrenceRole : u32 {
    None,
    RawP2,
    P1Bridge,
    RawPrefix,
    P1Tail,
};

enum class PpDepthPredecessorPrefixJoinDiscriminator : u32 {
    NotAttempted,
    MissingCurrentWriter,
    CurrentObservationMismatch,
    PrefixHintMismatch,
    PrefixLinkMismatch,
    PrefixRangeMismatch,
    PrefixBackingMismatch,
    PrefixMutationMismatch,
    CandidateMismatch,
    Exact,
};

enum class PpDepthPredecessorTwoHopPostDispatch : u32 {
    Ignore,
    RawP2,
    P1Bridge,
    RawPrefix,
    P1Tail,
    Reject,
};

[[nodiscard]] constexpr bool TryIncrementPpDepthPredecessorOccurrenceCount(
    u32& count) noexcept {
    if (count == std::numeric_limits<u32>::max()) {
        return false;
    }
    ++count;
    return true;
}

[[nodiscard]] constexpr bool
TryIncrementPpDepthPredecessorCrossScopeMode3RollingCount(u32& count) noexcept {
    if (count == std::numeric_limits<u32>::max()) {
        return false;
    }
    ++count;
    return true;
}

struct PpDepthPredecessorTwoHopPostDispatchInput {
    PpDepthPredecessorTwoHopOccurrenceRole occurrence_role{
        PpDepthPredecessorTwoHopOccurrenceRole::None};
    bool matches_raw{};
    bool matches_bridge{};
    bool raw_open{};
    bool raw_open_exact{};
    bool bridge_open{};
    bool bridge_open_exact{};
};

[[nodiscard]] constexpr PpDepthPredecessorTwoHopPostDispatch
ClassifyPpDepthPredecessorTwoHopPostDispatch(
    const PpDepthPredecessorTwoHopPostDispatchInput& input) noexcept {
    if (input.matches_raw && input.matches_bridge) {
        return PpDepthPredecessorTwoHopPostDispatch::Reject;
    }
    if (input.matches_raw) {
        if (!input.raw_open || !input.raw_open_exact || input.bridge_open) {
            return PpDepthPredecessorTwoHopPostDispatch::Reject;
        }
        switch (input.occurrence_role) {
        case PpDepthPredecessorTwoHopOccurrenceRole::None:
        case PpDepthPredecessorTwoHopOccurrenceRole::RawP2:
            return PpDepthPredecessorTwoHopPostDispatch::RawP2;
        case PpDepthPredecessorTwoHopOccurrenceRole::RawPrefix:
            return PpDepthPredecessorTwoHopPostDispatch::RawPrefix;
        default:
            return PpDepthPredecessorTwoHopPostDispatch::Reject;
        }
    }
    if (input.matches_bridge) {
        if (!input.bridge_open || !input.bridge_open_exact || input.raw_open) {
            return PpDepthPredecessorTwoHopPostDispatch::Reject;
        }
        switch (input.occurrence_role) {
        case PpDepthPredecessorTwoHopOccurrenceRole::None:
        case PpDepthPredecessorTwoHopOccurrenceRole::P1Bridge:
            return PpDepthPredecessorTwoHopPostDispatch::P1Bridge;
        case PpDepthPredecessorTwoHopOccurrenceRole::P1Tail:
            return PpDepthPredecessorTwoHopPostDispatch::P1Tail;
        default:
            return PpDepthPredecessorTwoHopPostDispatch::Reject;
        }
    }
    return input.raw_open || input.bridge_open
               ? PpDepthPredecessorTwoHopPostDispatch::Reject
               : PpDepthPredecessorTwoHopPostDispatch::Ignore;
}

struct PpDepthPredecessorTwoHopResult {
    PpDepthPredecessorTwoHopPhase phase{PpDepthPredecessorTwoHopPhase::LearnP1};
    PpDepthPredecessorTwoHopDecision decision{PpDepthPredecessorTwoHopDecision::Idle};
    PpDepthPredecessorTwoHopIdentityCollision identity_collision{
        PpDepthPredecessorTwoHopIdentityCollision::None};
    PpDepthPredecessorTwoHopChainKind chain_kind{
        PpDepthPredecessorTwoHopChainKind::Distinct};
    PpDepthPredecessorTwoHopOccurrenceRole occurrence_role{
        PpDepthPredecessorTwoHopOccurrenceRole::None};
    PpDepthPredecessorPrefixJoinDiscriminator prefix_join_discriminator{
        PpDepthPredecessorPrefixJoinDiscriminator::NotAttempted};
    u64 generation{};
    u32 candidate_index{};
    u32 collision_p1_ordinal{};
    u32 expected_occurrences{};
    u32 observed_occurrences{};
    u32 raw_occurrences{};
    u32 bridge_occurrences{};
    bool metadata_only{};
    bool raw_armed{};
    bool anchor_exact{};
    bool p1_intent{};
    bool p2_exact{};
    bool p1_post_exact{};
    bool prefix_join_exact{};
    bool candidate_saved{};
    bool committed{};
    bool select{};

    bool operator==(const PpDepthPredecessorTwoHopResult&) const = default;
};

struct PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness {
    PpDepthPredecessorWriterStamp root_prefix{};
    PpDepthPredecessorCandidateStamp root_candidate{};
    PpDepthPredecessorWriterStamp latest_prefix{};
    PpDepthPredecessorCandidateStamp latest_candidate{};
    u32 candidate_capacity{};
    u32 restart_attempt_count{};
    u32 restart_completion_count{};
    bool restart_open{};
    bool count_overflow{};
    bool chain_exact{};
    bool root_valid{};
    bool root_to_latest_chain_exact{};
    bool valid{};

    bool operator==(
        const PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness&) const = default;
};

enum class PpDepthPredecessorCrossScopeMode3SelectedProofReason : u32 {
    NotAttempted,
    PhaseMismatch,
    DecisionMismatch,
    ChainMismatch,
    OccurrenceRoleMismatch,
    GenerationContinuityMismatch,
    ExpectedContinuityMismatch,
    ObservedCardinalityMismatch,
    RawCardinalityMismatch,
    BridgeCardinalityMismatch,
    IdentityCollision,
    ProofFlagsMismatch,
    CandidateMismatch,
    SelectMismatch,
    Exact,
};

struct PpDepthPredecessorCrossScopeMode3SelectedProof {
    bool attempted{};
    PpDepthPredecessorCrossScopeMode3SelectedProofReason reason{
        PpDepthPredecessorCrossScopeMode3SelectedProofReason::NotAttempted};
    u32 expected_occurrences{};
    u32 observed_occurrences{};
    u32 raw_occurrences{};
    u32 bridge_occurrences{};
    bool phase_exact{};
    bool decision_exact{};
    bool chain_exact{};
    bool occurrence_role_exact{};
    bool generation_continuity_exact{};
    bool expected_continuity_exact{};
    bool observed_cardinality_exact{};
    bool raw_cardinality_exact{};
    bool bridge_cardinality_exact{};
    bool collision_exact{};
    bool proof_flags_exact{};
    bool candidate_exact{};
    bool select_exact{};
    bool exact{};

    bool operator==(
        const PpDepthPredecessorCrossScopeMode3SelectedProof&) const = default;
};

[[nodiscard]] constexpr
PpDepthPredecessorCrossScopeMode3SelectedProofReason
SanitizePpDepthPredecessorCrossScopeMode3SelectedProofReason(
    PpDepthPredecessorCrossScopeMode3SelectedProofReason reason) noexcept {
    using Reason =
        PpDepthPredecessorCrossScopeMode3SelectedProofReason;
    switch (reason) {
    case Reason::NotAttempted:
    case Reason::PhaseMismatch:
    case Reason::DecisionMismatch:
    case Reason::ChainMismatch:
    case Reason::OccurrenceRoleMismatch:
    case Reason::GenerationContinuityMismatch:
    case Reason::ExpectedContinuityMismatch:
    case Reason::ObservedCardinalityMismatch:
    case Reason::RawCardinalityMismatch:
    case Reason::BridgeCardinalityMismatch:
    case Reason::IdentityCollision:
    case Reason::ProofFlagsMismatch:
    case Reason::CandidateMismatch:
    case Reason::SelectMismatch:
    case Reason::Exact:
        return reason;
    }
    return Reason::NotAttempted;
}

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeMode3SelectedProof
ClassifyPpDepthPredecessorCrossScopeMode3SelectedProof(
    const PpDepthPredecessorTwoHopResult& begin,
    const PpDepthPredecessorTwoHopResult& selected) noexcept {
    using Reason =
        PpDepthPredecessorCrossScopeMode3SelectedProofReason;
    const bool phase_exact =
        selected.phase == PpDepthPredecessorTwoHopPhase::ArmedP2;
    const bool decision_exact =
        selected.decision == PpDepthPredecessorTwoHopDecision::Selected;
    const bool chain_exact =
        selected.chain_kind ==
        PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter;
    const bool occurrence_role_exact =
        selected.occurrence_role ==
        PpDepthPredecessorTwoHopOccurrenceRole::P1Tail;
    const bool generation_continuity_exact =
        begin.generation != 0 && selected.generation == begin.generation;
    const bool expected_continuity_exact =
        begin.expected_occurrences >= 2 &&
        begin.expected_occurrences != std::numeric_limits<u32>::max() &&
        selected.expected_occurrences == begin.expected_occurrences;
    const bool observed_cardinality_exact =
        selected.observed_occurrences == selected.expected_occurrences;
    const bool raw_cardinality_exact = selected.raw_occurrences == 1;
    const bool bridge_cardinality_exact =
        selected.bridge_occurrences == selected.expected_occurrences;
    const bool collision_exact =
        selected.identity_collision ==
        PpDepthPredecessorTwoHopIdentityCollision::None;
    const bool proof_flags_exact =
        selected.anchor_exact && selected.p1_intent && selected.p2_exact &&
        selected.p1_post_exact && selected.prefix_join_exact &&
        selected.raw_armed && !selected.metadata_only &&
        selected.prefix_join_discriminator ==
            PpDepthPredecessorPrefixJoinDiscriminator::Exact;
    const bool candidate_exact = selected.candidate_saved;
    const bool select_exact = selected.select;
    const bool exact =
        phase_exact && decision_exact && chain_exact &&
        occurrence_role_exact && generation_continuity_exact &&
        expected_continuity_exact && observed_cardinality_exact &&
        raw_cardinality_exact && bridge_cardinality_exact &&
        collision_exact && proof_flags_exact && candidate_exact &&
        select_exact;

    Reason reason = Reason::Exact;
    if (!phase_exact) {
        reason = Reason::PhaseMismatch;
    } else if (!decision_exact) {
        reason = Reason::DecisionMismatch;
    } else if (!chain_exact) {
        reason = Reason::ChainMismatch;
    } else if (!occurrence_role_exact) {
        reason = Reason::OccurrenceRoleMismatch;
    } else if (!generation_continuity_exact) {
        reason = Reason::GenerationContinuityMismatch;
    } else if (!expected_continuity_exact) {
        reason = Reason::ExpectedContinuityMismatch;
    } else if (!observed_cardinality_exact) {
        reason = Reason::ObservedCardinalityMismatch;
    } else if (!raw_cardinality_exact) {
        reason = Reason::RawCardinalityMismatch;
    } else if (!bridge_cardinality_exact) {
        reason = Reason::BridgeCardinalityMismatch;
    } else if (!collision_exact) {
        reason = Reason::IdentityCollision;
    } else if (!proof_flags_exact) {
        reason = Reason::ProofFlagsMismatch;
    } else if (!candidate_exact) {
        reason = Reason::CandidateMismatch;
    } else if (!select_exact) {
        reason = Reason::SelectMismatch;
    }
    return {
        .attempted = true,
        .reason = reason,
        .expected_occurrences = selected.expected_occurrences,
        .observed_occurrences = selected.observed_occurrences,
        .raw_occurrences = selected.raw_occurrences,
        .bridge_occurrences = selected.bridge_occurrences,
        .phase_exact = phase_exact,
        .decision_exact = decision_exact,
        .chain_exact = chain_exact,
        .occurrence_role_exact = occurrence_role_exact,
        .generation_continuity_exact = generation_continuity_exact,
        .expected_continuity_exact = expected_continuity_exact,
        .observed_cardinality_exact = observed_cardinality_exact,
        .raw_cardinality_exact = raw_cardinality_exact,
        .bridge_cardinality_exact = bridge_cardinality_exact,
        .collision_exact = collision_exact,
        .proof_flags_exact = proof_flags_exact,
        .candidate_exact = candidate_exact,
        .select_exact = select_exact,
        .exact = exact,
    };
}

struct PpDepthPredecessorCrossScopeBootstrapSnapshot {
    PpDepthPredecessorHint p0_raw_prefix{};
    PpDepthPredecessorHint p1_tail{};
    PpDepthPredecessorHint configured_anchor{};
    bool valid{};

    bool operator==(const PpDepthPredecessorCrossScopeBootstrapSnapshot&) const = default;
};

enum class PpDepthPredecessorCrossScopeBootstrapReadiness : u32 {
    NotRequested,
    PairNotReady,
    BeginShapeMismatch,
    GenerationOrCountMismatch,
    HintMismatch,
    Exact,
};

struct PpDepthPredecessorCrossScopeBootstrapReadinessInput {
    bool requested{};
    bool pair_ready{};
    bool begin_shape_exact{};
    bool generation_and_count_exact{};
    bool hints_valid{};
    bool hints_distinct{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeBootstrapReadiness
ClassifyPpDepthPredecessorCrossScopeBootstrapReadiness(
    const PpDepthPredecessorCrossScopeBootstrapReadinessInput& input) noexcept {
    if (!input.requested) {
        return PpDepthPredecessorCrossScopeBootstrapReadiness::NotRequested;
    }
    if (!input.pair_ready) {
        return PpDepthPredecessorCrossScopeBootstrapReadiness::PairNotReady;
    }
    if (!input.begin_shape_exact) {
        return PpDepthPredecessorCrossScopeBootstrapReadiness::BeginShapeMismatch;
    }
    if (!input.generation_and_count_exact) {
        return PpDepthPredecessorCrossScopeBootstrapReadiness::GenerationOrCountMismatch;
    }
    if (!input.hints_valid || !input.hints_distinct) {
        return PpDepthPredecessorCrossScopeBootstrapReadiness::HintMismatch;
    }
    return PpDepthPredecessorCrossScopeBootstrapReadiness::Exact;
}

struct PpDepthPredecessorCrossScopeBootstrapClassification {
    PpDepthPredecessorCrossScopeBootstrapSnapshot snapshot{};
    PpDepthPredecessorCrossScopeBootstrapReadiness readiness{
        PpDepthPredecessorCrossScopeBootstrapReadiness::NotRequested};
};

struct PpDepthPredecessorCrossScopeBootstrapRuntimeEvidence {
    PpDepthPredecessorTwoHopPhase begin_phase{PpDepthPredecessorTwoHopPhase::LearnP1};
    PpDepthPredecessorTwoHopDecision begin_decision{PpDepthPredecessorTwoHopDecision::Idle};
    PpDepthPredecessorTwoHopChainKind begin_chain_kind{
        PpDepthPredecessorTwoHopChainKind::Distinct};
    PpDepthPredecessorTwoHopOccurrenceRole begin_occurrence_role{
        PpDepthPredecessorTwoHopOccurrenceRole::None};
    bool begin_raw_armed{};
    PpDepthPredecessorCrossScopeBootstrapReadiness readiness{
        PpDepthPredecessorCrossScopeBootstrapReadiness::NotRequested};
    PpDepthPredecessorTwoHopDecision end_decision{PpDepthPredecessorTwoHopDecision::Idle};
    PpDepthPredecessorCrossScopeMode3SelectedProof selected_proof{};

    bool operator==(const PpDepthPredecessorCrossScopeBootstrapRuntimeEvidence&) const = default;
};

enum class PpDepthPredecessorCrossScopeBootstrapRuntimeEvidenceEvent : u32 {
    Reset,
    Begin,
    End,
};

struct PpDepthPredecessorCrossScopeBootstrapRuntimeEvidenceObservation {
    PpDepthPredecessorCrossScopeBootstrapRuntimeEvidenceEvent event{
        PpDepthPredecessorCrossScopeBootstrapRuntimeEvidenceEvent::Reset};
    PpDepthPredecessorTwoHopResult begin{};
    PpDepthPredecessorCrossScopeBootstrapReadiness readiness{
        PpDepthPredecessorCrossScopeBootstrapReadiness::NotRequested};
    PpDepthPredecessorTwoHopDecision end_decision{PpDepthPredecessorTwoHopDecision::Idle};
    PpDepthPredecessorCrossScopeMode3SelectedProof selected_proof{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeBootstrapRuntimeEvidence
AccumulatePpDepthPredecessorCrossScopeBootstrapRuntimeEvidence(
    const PpDepthPredecessorCrossScopeBootstrapRuntimeEvidence& current,
    const PpDepthPredecessorCrossScopeBootstrapRuntimeEvidenceObservation& observation) noexcept {
    switch (observation.event) {
    case PpDepthPredecessorCrossScopeBootstrapRuntimeEvidenceEvent::Reset:
        return {};
    case PpDepthPredecessorCrossScopeBootstrapRuntimeEvidenceEvent::Begin:
        return {
            .begin_phase = observation.begin.phase,
            .begin_decision = observation.begin.decision,
            .begin_chain_kind = observation.begin.chain_kind,
            .begin_occurrence_role = observation.begin.occurrence_role,
            .begin_raw_armed = observation.begin.raw_armed,
            .readiness = observation.readiness,
        };
    case PpDepthPredecessorCrossScopeBootstrapRuntimeEvidenceEvent::End:
        if (current.readiness ==
            PpDepthPredecessorCrossScopeBootstrapReadiness::NotRequested) {
            return current;
        }
        auto completed = current;
        completed.end_decision = observation.end_decision;
        completed.selected_proof =
            current.readiness ==
                    PpDepthPredecessorCrossScopeBootstrapReadiness::Exact &&
                observation.selected_proof.attempted
            ? observation.selected_proof
            : PpDepthPredecessorCrossScopeMode3SelectedProof{};
        return completed;
    }
    return {};
}

class PpDepthPredecessorTwoHopLifecycle {
public:
    explicit constexpr PpDepthPredecessorTwoHopLifecycle(u32 candidate_capacity,
                                                         u64 initial_generation = 0,
                                                         u32 hint_hops = 2) noexcept
        : capacity{candidate_capacity}, generation{initial_generation}, hint_hops{hint_hops} {}

    [[nodiscard]] constexpr PpDepthPredecessorTwoHopResult BeginFrame() noexcept {
        if (frame_open) {
            Fail(PpDepthPredecessorTwoHopDecision::MissingAnchor);
            return MakeResult();
        }
        ResetFrame();
        frame_open = true;
        if (pair_ready) {
            if (generation == std::numeric_limits<u64>::max() || capacity == 0 ||
                capacity > PpUpstreamFeedbackPrePostPlan::MaxCandidates) {
                phase = PpDepthPredecessorTwoHopPhase::GenerationWrap;
                Fail(PpDepthPredecessorTwoHopDecision::GenerationWrap);
                frame_open = false;
                return MakeResult();
            }
            ++generation;
            phase = PpDepthPredecessorTwoHopPhase::ArmedP2;
            if (frame_chain_kind ==
                PpDepthPredecessorTwoHopChainKind::SameWriterImmediatePrior) {
                occurrence_role = PpDepthPredecessorTwoHopOccurrenceRole::RawP2;
            } else if (frame_chain_kind ==
                       PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter) {
                occurrence_role = PpDepthPredecessorTwoHopOccurrenceRole::RawPrefix;
            }
        } else if (p1_hint.Valid()) {
            phase = PpDepthPredecessorTwoHopPhase::LearnP2;
        } else {
            phase = PpDepthPredecessorTwoHopPhase::LearnP1;
        }
        return MakeResult();
    }

    [[nodiscard]] constexpr bool MatchesP1(const PpTerminalScopeDrawSelector& draw,
                                           u64 fragment_pgm_hash) const noexcept {
        const bool armed_role_matches =
            phase != PpDepthPredecessorTwoHopPhase::ArmedP2 ||
            (frame_chain_kind ==
                 PpDepthPredecessorTwoHopChainKind::SameWriterImmediatePrior &&
             occurrence_role == PpDepthPredecessorTwoHopOccurrenceRole::P1Bridge) ||
            (frame_chain_kind ==
                 PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter &&
             occurrence_role == PpDepthPredecessorTwoHopOccurrenceRole::P1Tail) ||
            frame_chain_kind == PpDepthPredecessorTwoHopChainKind::Distinct;
        return frame_open && armed_role_matches && p1_hint.Valid() &&
               PpDepthPredecessorHint{.draw = draw,
                                      .fragment_pgm_hash = fragment_pgm_hash,
                                      .valid = true} == p1_hint;
    }

    [[nodiscard]] constexpr bool MatchesRawP2(const PpTerminalScopeDrawSelector& draw,
                                              u64 fragment_pgm_hash) const noexcept {
        if (!frame_open || phase != PpDepthPredecessorTwoHopPhase::ArmedP2) {
            return false;
        }
        const auto* expected_hint = &p2_hint;
        if (frame_chain_kind ==
            PpDepthPredecessorTwoHopChainKind::SameWriterImmediatePrior) {
            if (occurrence_role != PpDepthPredecessorTwoHopOccurrenceRole::RawP2) {
                return false;
            }
            expected_hint = &p1_hint;
        } else if (frame_chain_kind ==
                       PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter &&
                   occurrence_role != PpDepthPredecessorTwoHopOccurrenceRole::RawPrefix) {
            return false;
        }
        return expected_hint->Valid() &&
               PpDepthPredecessorHint{.draw = draw,
                                      .fragment_pgm_hash = fragment_pgm_hash,
                                       .valid = true} == *expected_hint;
    }

    [[nodiscard]] constexpr bool MatchesRawPrefixRestart(
        const PpTerminalScopeDrawSelector& draw, u64 fragment_pgm_hash) const noexcept {
        return CanBeginRawPrefixRestart() &&
               PpDepthPredecessorHint{.draw = draw,
                                      .fragment_pgm_hash = fragment_pgm_hash,
                                      .valid = true} == p2_hint;
    }

    [[nodiscard]] constexpr const PpDepthPredecessorHint* PeekP1Hint() const noexcept {
        return p1_hint.Valid() ? &p1_hint : nullptr;
    }

    [[nodiscard]] constexpr const PpDepthPredecessorHint* PeekArmedRawP2Hint() const noexcept {
        if (!frame_open || phase != PpDepthPredecessorTwoHopPhase::ArmedP2) {
            return nullptr;
        }
        if (frame_chain_kind ==
            PpDepthPredecessorTwoHopChainKind::SameWriterImmediatePrior) {
            return p1_hint.Valid() ? &p1_hint : nullptr;
        }
        return p2_hint.Valid() ? &p2_hint : nullptr;
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopeBootstrapSnapshot
    PeekCrossScopeBootstrapSnapshot(
        const PpDepthPredecessorTwoHopResult& begin) const noexcept {
        return ClassifyCrossScopeBootstrapSnapshot(begin).snapshot;
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopeBootstrapClassification
    ClassifyCrossScopeBootstrapSnapshot(
        const PpDepthPredecessorTwoHopResult& begin, bool requested = true) const noexcept {
        const bool pair_ready_exact = hint_hops == 3 && frame_open && pair_ready;
        const bool begin_shape_exact =
            phase == PpDepthPredecessorTwoHopPhase::ArmedP2 &&
            frame_chain_kind ==
                PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter &&
            occurrence_role == PpDepthPredecessorTwoHopOccurrenceRole::RawPrefix &&
            begin.phase == PpDepthPredecessorTwoHopPhase::ArmedP2 &&
            begin.decision == PpDepthPredecessorTwoHopDecision::Idle &&
            begin.identity_collision == PpDepthPredecessorTwoHopIdentityCollision::None &&
            begin.chain_kind ==
                PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter &&
            begin.occurrence_role == PpDepthPredecessorTwoHopOccurrenceRole::RawPrefix &&
            begin.prefix_join_discriminator ==
                PpDepthPredecessorPrefixJoinDiscriminator::NotAttempted &&
            !begin.metadata_only && begin.raw_armed && !begin.anchor_exact &&
            !begin.p1_intent && !begin.p2_exact && !begin.p1_post_exact &&
            !begin.prefix_join_exact && !begin.candidate_saved && !begin.committed &&
            !begin.select;
        const bool generation_and_count_exact =
            begin.generation != 0 && begin.generation == generation &&
            begin.expected_occurrences >= 2 &&
            begin.expected_occurrences != std::numeric_limits<u32>::max() &&
            begin.expected_occurrences == frame_expected_occurrences &&
            begin.expected_occurrences == committed_expected_occurrences &&
            begin.observed_occurrences == 0 && begin.raw_occurrences == 0 &&
            begin.bridge_occurrences == 0;
        const bool hints_valid = p2_hint.Valid() && p1_hint.Valid() && anchor_hint.Valid();
        const bool hints_distinct = p2_hint != p1_hint && p2_hint != anchor_hint &&
                                    p1_hint != anchor_hint;
        const auto readiness = ClassifyPpDepthPredecessorCrossScopeBootstrapReadiness({
            .requested = requested,
            .pair_ready = pair_ready_exact,
            .begin_shape_exact = begin_shape_exact,
            .generation_and_count_exact = generation_and_count_exact,
            .hints_valid = hints_valid,
            .hints_distinct = hints_distinct,
        });
        if (readiness != PpDepthPredecessorCrossScopeBootstrapReadiness::Exact) {
            return {.readiness = readiness};
        }
        return {
            .snapshot = {
                .p0_raw_prefix = p2_hint,
                .p1_tail = p1_hint,
                .configured_anchor = anchor_hint,
                .valid = true,
            },
            .readiness = readiness,
        };
    }

    [[nodiscard]] constexpr
    PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness
    SnapshotCrossScopeSelectedLatestPrivateWitness(
        const PpDepthPredecessorTwoHopResult& result) const noexcept {
        if (result != MakeResult() || hint_hops != 3 || !frame_open ||
            phase != PpDepthPredecessorTwoHopPhase::ArmedP2) {
            return {};
        }
        if (raw_occurrences == 0) {
            return {};
        }

        PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness witness{
            .root_prefix = root_prefix,
            .root_candidate = root_candidate,
            .restart_attempt_count = raw_prefix_restart_attempt_count,
            .restart_completion_count = raw_prefix_restart_completion_count,
            .restart_open = raw_prefix_restart_open,
            .count_overflow = raw_prefix_restart_count_overflow,
            .chain_exact =
                frame_chain_kind ==
                PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter,
            .root_valid = root_prefix_valid,
        };
        const bool counts_exact =
            raw_occurrences != 0 && raw_occurrences != std::numeric_limits<u32>::max() &&
            raw_prefix_restart_attempt_count == raw_occurrences - 1 &&
            raw_prefix_restart_completion_count == raw_occurrences - 1 &&
            !raw_prefix_restart_open && !raw_prefix_restart_count_overflow;
        const bool selected_exact =
            witness.chain_exact && counts_exact && selected &&
            last_decision == PpDepthPredecessorTwoHopDecision::Selected &&
            occurrence_role == PpDepthPredecessorTwoHopOccurrenceRole::P1Tail &&
            pending_prefix.Valid() && pending_candidate.valid &&
            pending_candidate.writer == pending_prefix &&
            pending_candidate.generation == generation && capacity != 0 &&
            capacity <= PpUpstreamFeedbackPrePostPlan::MaxCandidates &&
            pending_candidate.candidate_index == selected_index &&
            pending_candidate.candidate_index < capacity &&
            pending_candidate.recorded_plane_mask == 0x3u && result.select &&
            result.candidate_saved && result.candidate_index == selected_index;
        if (!selected_exact) {
            return witness;
        }
        witness.latest_prefix = pending_prefix;
        witness.latest_candidate = pending_candidate;
        witness.candidate_capacity = capacity;
        witness.root_to_latest_chain_exact =
            root_prefix_valid && raw_occurrences > 1 &&
            root_candidate.writer == root_prefix &&
            root_candidate.generation == pending_candidate.generation &&
            root_candidate.candidate_index == pending_candidate.candidate_index &&
            root_candidate.recorded_plane_mask == 0x3u &&
            root_prefix.hint == pending_prefix.hint &&
            root_prefix.link == pending_prefix.link &&
            root_prefix.range == pending_prefix.range &&
            root_prefix.backing_generation == pending_prefix.backing_generation &&
            root_prefix.mutation_epoch < pending_prefix.mutation_epoch;
        witness.valid = true;
        return witness;
    }

    [[nodiscard]] constexpr PpDepthPredecessorTwoHopResult BeginRawPrefixRestart() noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (!CanBeginRawPrefixRestart()) {
            Fail(PpDepthPredecessorTwoHopDecision::DuplicateP1);
            return MakeResult();
        }

        if (!TryIncrementPpDepthPredecessorCrossScopeMode3RollingCount(
                raw_prefix_restart_attempt_count)) {
            raw_prefix_restart_count_overflow = true;
        }
        rolling_prefix_prior = pending_prefix;
        rolling_prefix_candidate_index = pending_candidate.candidate_index;
        pending_p2 = {};
        pending_prefix = {};
        pending_candidate = {};
        selected_index = 0;
        p2_exact = false;
        prefix_join_exact = false;
        candidate_saved = false;
        occurrence_role = PpDepthPredecessorTwoHopOccurrenceRole::RawPrefix;
        raw_prefix_restart_open = true;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorTwoHopResult ObserveRawP2Complete(
        const PpDepthPredecessorObservationResult& observation) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        const bool same_writer_armed =
            frame_chain_kind == PpDepthPredecessorTwoHopChainKind::SameWriterImmediatePrior;
        const bool prefix_armed =
            frame_chain_kind ==
            PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter;
        if (!frame_open || phase != PpDepthPredecessorTwoHopPhase::ArmedP2 ||
            (!same_writer_armed && !prefix_armed)) {
            Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
            return MakeResult();
        }
        const auto expected_role = prefix_armed
                                       ? PpDepthPredecessorTwoHopOccurrenceRole::RawPrefix
                                       : PpDepthPredecessorTwoHopOccurrenceRole::RawP2;
        const bool rolling_prefix = prefix_armed && raw_prefix_restart_open;
        if (occurrence_role != expected_role ||
            (same_writer_armed && observed_occurrences >= frame_expected_occurrences) ||
            (prefix_armed && raw_occurrences != 0 && !rolling_prefix)) {
            Fail(PpDepthPredecessorTwoHopDecision::DuplicateP1);
            return MakeResult();
        }
        if (p1_seen) {
            if (!p1_post_seen || !p1_post_exact) {
                Fail(PpDepthPredecessorTwoHopDecision::MissingP1Post);
                return MakeResult();
            }
            ResetP1Occurrence();
        }

        const auto& writer = observation.writer;
        const auto& candidate = observation.candidate;
        const auto& expected_hint = prefix_armed ? p2_hint : p1_hint;
        if (rolling_prefix &&
            (!observation.hint.Valid() || !writer.Valid() || observation.hint != writer.hint ||
             !rolling_prefix_prior.Valid() || writer.hint != rolling_prefix_prior.hint ||
             writer.link != rolling_prefix_prior.link ||
             writer.range != rolling_prefix_prior.range ||
             writer.backing_generation != rolling_prefix_prior.backing_generation ||
             writer.mutation_epoch <= rolling_prefix_prior.mutation_epoch)) {
            Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
            return MakeResult();
        }
        if (rolling_prefix &&
            (!candidate.valid || candidate.generation == 0 ||
             candidate.generation != generation ||
             candidate.candidate_index != rolling_prefix_candidate_index ||
             candidate.candidate_index >= capacity || candidate.recorded_plane_mask != 0x3u ||
             candidate.writer != writer)) {
            Fail(PpDepthPredecessorTwoHopDecision::CandidateMismatch);
            return MakeResult();
        }
        if (!observation.hint.Valid() || !writer.Valid() || observation.hint != writer.hint ||
            writer.hint != expected_hint || !candidate.valid || candidate.generation == 0 ||
            candidate.generation != generation || candidate.candidate_index >= capacity ||
            candidate.recorded_plane_mask != 0x3u || candidate.writer != writer) {
            Fail(PpDepthPredecessorTwoHopDecision::CandidateMismatch);
            return MakeResult();
        }

        pending_p2 = writer;
        if (prefix_armed) {
            pending_prefix = writer;
        }
        pending_candidate = candidate;
        p2_exact = true;
        candidate_saved = false;
        if (!TryIncrementPpDepthPredecessorOccurrenceCount(raw_occurrences)) {
            Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
            return MakeResult();
        }
        if (prefix_armed && !rolling_prefix && raw_occurrences == 1) {
            root_prefix = writer;
            root_candidate = candidate;
            root_prefix_valid = true;
        }
        if (rolling_prefix &&
            !TryIncrementPpDepthPredecessorCrossScopeMode3RollingCount(
                raw_prefix_restart_completion_count)) {
            raw_prefix_restart_count_overflow = true;
        }
        if (same_writer_armed) {
            ++observed_occurrences;
        }
        occurrence_role = prefix_armed ? PpDepthPredecessorTwoHopOccurrenceRole::P1Tail
                                       : PpDepthPredecessorTwoHopOccurrenceRole::P1Bridge;
        raw_prefix_restart_open = false;
        rolling_prefix_prior = {};
        rolling_prefix_candidate_index = 0;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorTwoHopResult ObserveP1Pre(
        const PpDepthPredecessorWriterIntent& intent,
        const PpDepthPredecessorObservationResult& observation) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (raw_prefix_restart_open) {
            Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
            return MakeResult();
        }
        if (!frame_open ||
            (phase != PpDepthPredecessorTwoHopPhase::LearnP2 &&
             phase != PpDepthPredecessorTwoHopPhase::ArmedP2)) {
            Fail(PpDepthPredecessorTwoHopDecision::MissingP1);
            return MakeResult();
        }
        if (anchor_seen) {
            Fail(PpDepthPredecessorTwoHopDecision::DuplicateP1);
            return MakeResult();
        }
        const bool same_writer_armed =
            phase == PpDepthPredecessorTwoHopPhase::ArmedP2 &&
            frame_chain_kind ==
                PpDepthPredecessorTwoHopChainKind::SameWriterImmediatePrior;
        const bool prefix_armed =
            phase == PpDepthPredecessorTwoHopPhase::ArmedP2 &&
            frame_chain_kind ==
                PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter;
        if ((same_writer_armed &&
             occurrence_role != PpDepthPredecessorTwoHopOccurrenceRole::P1Bridge) ||
            (prefix_armed &&
             occurrence_role != PpDepthPredecessorTwoHopOccurrenceRole::P1Tail)) {
            Fail(PpDepthPredecessorTwoHopDecision::MissingP1);
            return MakeResult();
        }
        if (prefix_armed && observed_occurrences >= frame_expected_occurrences) {
            Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
            return MakeResult();
        }
        PpDepthPredecessorWriterStamp immediate_prior_p1{};
        if (p1_seen) {
            if (!p1_post_seen || !p1_post_exact) {
                Fail(prefix_armed ? PpDepthPredecessorTwoHopDecision::MissingP1Post
                                  : PpDepthPredecessorTwoHopDecision::DuplicateP1);
                return MakeResult();
            }
            immediate_prior_p1 = p1_post;
            ResetP1Occurrence(prefix_armed);
        }
        p1_seen = true;
        if (!intent.Valid() || intent.hint != p1_hint) {
            Fail(PpDepthPredecessorTwoHopDecision::P1Mismatch);
            return MakeResult();
        }
        p1_intent_exact = true;
        const auto& observed_p2 = observation.writer;
        const auto& candidate = observation.candidate;
        const bool first_prefix_tail = prefix_armed && observed_occurrences == 0;
        if (first_prefix_tail) {
            const bool exact_candidate =
                candidate.valid && pending_candidate.valid &&
                candidate.generation == pending_candidate.generation &&
                candidate.candidate_index == pending_candidate.candidate_index &&
                candidate.recorded_plane_mask == pending_candidate.recorded_plane_mask &&
                candidate.writer == pending_candidate.writer &&
                candidate.generation == generation && candidate.candidate_index < capacity &&
                candidate.recorded_plane_mask == 0x3u;
            if (!observed_p2.valid) {
                prefix_join_discriminator =
                    PpDepthPredecessorPrefixJoinDiscriminator::MissingCurrentWriter;
            } else if (!observation.hint.Valid() || observation.hint != observed_p2.hint) {
                prefix_join_discriminator =
                    PpDepthPredecessorPrefixJoinDiscriminator::CurrentObservationMismatch;
            } else if (!pending_prefix.Valid() || observed_p2.hint != pending_prefix.hint) {
                prefix_join_discriminator =
                    PpDepthPredecessorPrefixJoinDiscriminator::PrefixHintMismatch;
            } else if (observed_p2.link != pending_prefix.link) {
                prefix_join_discriminator =
                    PpDepthPredecessorPrefixJoinDiscriminator::PrefixLinkMismatch;
            } else if (observed_p2.range != pending_prefix.range) {
                prefix_join_discriminator =
                    PpDepthPredecessorPrefixJoinDiscriminator::PrefixRangeMismatch;
            } else if (observed_p2.backing_generation != pending_prefix.backing_generation) {
                prefix_join_discriminator =
                    PpDepthPredecessorPrefixJoinDiscriminator::PrefixBackingMismatch;
            } else if (observed_p2.mutation_epoch != pending_prefix.mutation_epoch) {
                prefix_join_discriminator =
                    PpDepthPredecessorPrefixJoinDiscriminator::PrefixMutationMismatch;
            } else if (!observed_p2.Valid() || observed_p2.link != intent.link ||
                       observed_p2.range != intent.range ||
                       observed_p2.backing_generation != intent.backing_generation ||
                       observed_p2.mutation_epoch != intent.mutation_epoch) {
                prefix_join_discriminator =
                    PpDepthPredecessorPrefixJoinDiscriminator::CurrentObservationMismatch;
            } else {
                prefix_join_discriminator =
                    exact_candidate ? PpDepthPredecessorPrefixJoinDiscriminator::Exact
                                    : PpDepthPredecessorPrefixJoinDiscriminator::CandidateMismatch;
            }
        }
        if (!observation.hint.Valid() || !observed_p2.Valid() ||
            observation.hint != observed_p2.hint || observed_p2.link != intent.link ||
            observed_p2.range != intent.range ||
            observed_p2.backing_generation != intent.backing_generation ||
            observed_p2.mutation_epoch != intent.mutation_epoch) {
            Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
            return MakeResult();
        }
        if (p1_pre_ordinal != std::numeric_limits<u32>::max()) {
            ++p1_pre_ordinal;
        }

        if (same_writer_armed) {
            const bool exact_raw = pending_p2.Valid() && observed_p2 == pending_p2 &&
                                   candidate.valid && pending_candidate.valid &&
                                   candidate.generation == pending_candidate.generation &&
                                   candidate.candidate_index == pending_candidate.candidate_index &&
                                   candidate.recorded_plane_mask ==
                                       pending_candidate.recorded_plane_mask &&
                                   candidate.writer == pending_candidate.writer;
            if (!exact_raw || candidate.generation != generation ||
                candidate.candidate_index >= capacity ||
                candidate.recorded_plane_mask != 0x3u) {
                Fail(PpDepthPredecessorTwoHopDecision::CandidateMismatch);
                return MakeResult();
            }
            p2_exact = true;
            p1_intent = intent;
            candidate_saved = true;
            return MakeResult();
        }

        if (prefix_armed) {
            if (first_prefix_tail) {
                if (!pending_prefix.Valid() || observed_p2 != pending_prefix) {
                    Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
                    return MakeResult();
                }
                const bool exact_candidate =
                    candidate.valid && pending_candidate.valid &&
                    candidate.generation == pending_candidate.generation &&
                    candidate.candidate_index == pending_candidate.candidate_index &&
                    candidate.recorded_plane_mask == pending_candidate.recorded_plane_mask &&
                    candidate.writer == pending_candidate.writer;
                if (!exact_candidate || candidate.generation != generation ||
                    candidate.candidate_index >= capacity ||
                    candidate.recorded_plane_mask != 0x3u) {
                    Fail(PpDepthPredecessorTwoHopDecision::CandidateMismatch);
                    return MakeResult();
                }
                prefix_join_exact = true;
            } else {
                if (!immediate_prior_p1.Valid() || observed_p2 != immediate_prior_p1) {
                    Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
                    return MakeResult();
                }
                if (candidate.valid) {
                    Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
                    return MakeResult();
                }
            }
            p2_exact = true;
            p1_intent = intent;
            candidate_saved = true;
            return MakeResult();
        }

        if (observed_p2.hint == intent.hint) {
            const auto self_chain_kind =
                hint_hops == 3
                    ? PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter
                    : PpDepthPredecessorTwoHopChainKind::SameWriterImmediatePrior;
            const bool can_begin_same_writer_chain =
                phase == PpDepthPredecessorTwoHopPhase::LearnP2 && p1_pre_ordinal == 2 &&
                immediate_prior_p1.Valid() && (hint_hops != 3 || pending_prefix.Valid());
            const bool continues_same_writer_chain =
                phase == PpDepthPredecessorTwoHopPhase::LearnP2 &&
                frame_chain_kind == self_chain_kind && p1_pre_ordinal > 2 &&
                immediate_prior_p1.Valid();
            if (!can_begin_same_writer_chain && !continues_same_writer_chain) {
                identity_collision = PpDepthPredecessorTwoHopIdentityCollision::P2EqualsP1;
                collision_p1_ordinal = p1_pre_ordinal;
                Fail(PpDepthPredecessorTwoHopDecision::IdentityCollision);
                return MakeResult();
            }
            if (observed_p2 != immediate_prior_p1) {
                Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
                return MakeResult();
            }
            frame_chain_kind = self_chain_kind;
        } else if (hint_hops == 3 && phase == PpDepthPredecessorTwoHopPhase::LearnP2) {
            if (p1_pre_ordinal != 1 || frame_chain_kind !=
                                             PpDepthPredecessorTwoHopChainKind::Distinct) {
                Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
                return MakeResult();
            }
            pending_prefix = observed_p2;
        } else if (frame_chain_kind ==
                   PpDepthPredecessorTwoHopChainKind::SameWriterImmediatePrior) {
            Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
            return MakeResult();
        }
        if (anchor_hint.Valid() && observed_p2.hint == anchor_hint) {
            identity_collision =
                PpDepthPredecessorTwoHopIdentityCollision::P2EqualsConfiguredAnchor;
            collision_p1_ordinal = p1_pre_ordinal;
            Fail(PpDepthPredecessorTwoHopDecision::IdentityCollision);
            return MakeResult();
        }
        if (phase == PpDepthPredecessorTwoHopPhase::ArmedP2 &&
            observed_p2.hint != p2_hint) {
            Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
            return MakeResult();
        }
        p2_exact = true;
        p1_intent = intent;
        pending_p2 = observed_p2;
        if (phase == PpDepthPredecessorTwoHopPhase::ArmedP2) {
            if (!candidate.valid || candidate.generation == 0 ||
                candidate.generation != generation || candidate.candidate_index >= capacity ||
                candidate.recorded_plane_mask != 0x3u || candidate.writer != observed_p2) {
                Fail(PpDepthPredecessorTwoHopDecision::CandidateMismatch);
                return MakeResult();
            }
            pending_candidate = candidate;
            candidate_saved = true;
        }
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorTwoHopResult ObserveP1Post(
        const PpDepthPredecessorWriterStamp& observed_p1) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (!frame_open || !p1_seen || !p1_intent_exact) {
            Fail(PpDepthPredecessorTwoHopDecision::MissingP1);
            return MakeResult();
        }
        if (p1_post_seen) {
            Fail(PpDepthPredecessorTwoHopDecision::DuplicateP1Post);
            return MakeResult();
        }
        p1_post_seen = true;
        const bool exact = observed_p1.Valid() && observed_p1.hint == p1_intent.hint &&
                           observed_p1.link == p1_intent.link &&
                           observed_p1.range == p1_intent.range &&
                           observed_p1.backing_generation == p1_intent.backing_generation &&
                           p1_intent.mutation_epoch != std::numeric_limits<u64>::max() &&
                           observed_p1.mutation_epoch == p1_intent.mutation_epoch + 1;
        if (!exact) {
            Fail(PpDepthPredecessorTwoHopDecision::P1StampMismatch);
            return MakeResult();
        }
        p1_post = observed_p1;
        p1_post_exact = true;
        if (phase == PpDepthPredecessorTwoHopPhase::LearnP2) {
            if (observed_occurrences != std::numeric_limits<u32>::max()) {
                ++observed_occurrences;
            }
        } else if (phase == PpDepthPredecessorTwoHopPhase::ArmedP2 &&
                   frame_chain_kind ==
                       PpDepthPredecessorTwoHopChainKind::SameWriterImmediatePrior) {
            if (occurrence_role != PpDepthPredecessorTwoHopOccurrenceRole::P1Bridge ||
                observed_occurrences >= frame_expected_occurrences) {
                Fail(PpDepthPredecessorTwoHopDecision::DuplicateP1Post);
                return MakeResult();
            }
            ++bridge_occurrences;
            ++observed_occurrences;
            occurrence_role = PpDepthPredecessorTwoHopOccurrenceRole::RawP2;
        } else if (phase == PpDepthPredecessorTwoHopPhase::ArmedP2 &&
                   frame_chain_kind ==
                       PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter) {
            if (occurrence_role != PpDepthPredecessorTwoHopOccurrenceRole::P1Tail ||
                !prefix_join_exact || !candidate_saved ||
                observed_occurrences >= frame_expected_occurrences) {
                Fail(PpDepthPredecessorTwoHopDecision::DuplicateP1Post);
                return MakeResult();
            }
            ++bridge_occurrences;
            ++observed_occurrences;
        }
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorTwoHopResult ObserveAnchor(
        const PpDepthPredecessorHint& configured_anchor,
        const PpDepthPredecessorWriterStamp& current_p1) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (raw_prefix_restart_open) {
            Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
            return MakeResult();
        }
        if (!frame_open) {
            Fail(PpDepthPredecessorTwoHopDecision::MissingAnchor);
            return MakeResult();
        }
        if (anchor_seen) {
            Fail(PpDepthPredecessorTwoHopDecision::DuplicateAnchor);
            return MakeResult();
        }
        anchor_seen = true;
        if (!configured_anchor.Valid() ||
            (current_p1.Valid() && configured_anchor == current_p1.hint)) {
            Fail(PpDepthPredecessorTwoHopDecision::IdentityCollision);
            return MakeResult();
        }
        if (!current_p1.Valid()) {
            Fail(PpDepthPredecessorTwoHopDecision::P1StampMismatch);
            return MakeResult();
        }
        if (anchor_hint.Valid() && configured_anchor != anchor_hint) {
            Fail(PpDepthPredecessorTwoHopDecision::P1Mismatch);
            return MakeResult();
        }
        anchor_exact = true;
        if (phase == PpDepthPredecessorTwoHopPhase::LearnP1) {
            pending_anchor_hint = configured_anchor;
            pending_p1_hint = current_p1.hint;
            last_decision = PpDepthPredecessorTwoHopDecision::LearnedP1;
            return MakeResult();
        }
        if (!p1_seen) {
            Fail(PpDepthPredecessorTwoHopDecision::MissingP1);
            return MakeResult();
        }
        if (!p1_post_seen || !p1_post_exact) {
            Fail(PpDepthPredecessorTwoHopDecision::MissingP1Post);
            return MakeResult();
        }
        if (current_p1.hint != p1_hint) {
            Fail(PpDepthPredecessorTwoHopDecision::P1Mismatch);
            return MakeResult();
        }
        if (current_p1 != p1_post) {
            Fail(PpDepthPredecessorTwoHopDecision::P1StampMismatch);
            return MakeResult();
        }
        if (!pending_p2.Valid()) {
            Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
            return MakeResult();
        }
        pending_p1_hint = current_p1.hint;
        if (phase == PpDepthPredecessorTwoHopPhase::LearnP2) {
            if (hint_hops == 3) {
                if (frame_chain_kind !=
                        PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter ||
                    !pending_prefix.Valid() || observed_occurrences < 2 ||
                    observed_occurrences != p1_pre_ordinal) {
                    Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
                    return MakeResult();
                }
                pending_p2 = pending_prefix;
                frame_expected_occurrences = observed_occurrences;
            } else if (frame_chain_kind ==
                       PpDepthPredecessorTwoHopChainKind::SameWriterImmediatePrior) {
                if (observed_occurrences < 2 || (observed_occurrences & 1u) != 0 ||
                    observed_occurrences != p1_pre_ordinal) {
                    Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
                    return MakeResult();
                }
                frame_expected_occurrences = observed_occurrences;
            }
            committed = true;
            last_decision = PpDepthPredecessorTwoHopDecision::Committed;
            return MakeResult();
        }
        if (frame_chain_kind ==
                PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter &&
            (occurrence_role != PpDepthPredecessorTwoHopOccurrenceRole::P1Tail ||
             frame_expected_occurrences < 2 ||
             observed_occurrences != frame_expected_occurrences || raw_occurrences == 0 ||
             bridge_occurrences != frame_expected_occurrences || !prefix_join_exact)) {
            Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
            return MakeResult();
        }
        if (frame_chain_kind ==
                PpDepthPredecessorTwoHopChainKind::SameWriterImmediatePrior &&
            (occurrence_role != PpDepthPredecessorTwoHopOccurrenceRole::RawP2 ||
             frame_expected_occurrences < 2 ||
             (frame_expected_occurrences & 1u) != 0 ||
             observed_occurrences != frame_expected_occurrences ||
             raw_occurrences != bridge_occurrences ||
             raw_occurrences + bridge_occurrences != observed_occurrences)) {
            Fail(PpDepthPredecessorTwoHopDecision::CandidateMismatch);
            return MakeResult();
        }
        if (!candidate_saved) {
            Fail(PpDepthPredecessorTwoHopDecision::CandidateMismatch);
            return MakeResult();
        }
        selected = true;
        selected_index = pending_candidate.candidate_index;
        last_decision = PpDepthPredecessorTwoHopDecision::Selected;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorTwoHopResult EndFrame() noexcept {
        if (!frame_open) {
            return MakeResult();
        }
        if (raw_prefix_restart_open && !HasFailure()) {
            Fail(PpDepthPredecessorTwoHopDecision::P2Mismatch);
        } else if (!anchor_seen && last_decision == PpDepthPredecessorTwoHopDecision::Idle) {
            Fail(PpDepthPredecessorTwoHopDecision::MissingAnchor);
        }
        switch (last_decision) {
        case PpDepthPredecessorTwoHopDecision::LearnedP1:
            anchor_hint = pending_anchor_hint;
            p1_hint = pending_p1_hint;
            p2_hint = {};
            pair_ready = false;
            committed_chain_kind = PpDepthPredecessorTwoHopChainKind::Distinct;
            committed_expected_occurrences = 0;
            break;
        case PpDepthPredecessorTwoHopDecision::Committed:
        case PpDepthPredecessorTwoHopDecision::Selected:
            p1_hint = pending_p1_hint;
            p2_hint = frame_chain_kind ==
                              PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter
                          ? pending_prefix.hint
                          : pending_p2.hint;
            pair_ready = true;
            committed_chain_kind = frame_chain_kind;
            committed_expected_occurrences = frame_expected_occurrences;
            break;
        case PpDepthPredecessorTwoHopDecision::MissingP1:
        case PpDepthPredecessorTwoHopDecision::P1Mismatch:
        case PpDepthPredecessorTwoHopDecision::P1StampMismatch:
            anchor_hint = {};
            p1_hint = {};
            p2_hint = {};
            pair_ready = false;
            committed_chain_kind = PpDepthPredecessorTwoHopChainKind::Distinct;
            committed_expected_occurrences = 0;
            break;
        case PpDepthPredecessorTwoHopDecision::P2Mismatch:
            p2_hint = {};
            pair_ready = false;
            committed_chain_kind = PpDepthPredecessorTwoHopChainKind::Distinct;
            committed_expected_occurrences = 0;
            break;
        case PpDepthPredecessorTwoHopDecision::MissingAnchor:
        case PpDepthPredecessorTwoHopDecision::MissingP1Post:
        case PpDepthPredecessorTwoHopDecision::DuplicateP1:
        case PpDepthPredecessorTwoHopDecision::DuplicateP1Post:
        case PpDepthPredecessorTwoHopDecision::DuplicateAnchor:
        case PpDepthPredecessorTwoHopDecision::IdentityCollision:
            if (frame_chain_kind ==
                PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter) {
                p2_hint = {};
                pair_ready = false;
                committed_chain_kind = PpDepthPredecessorTwoHopChainKind::Distinct;
                committed_expected_occurrences = 0;
            }
            break;
        default:
            break;
        }
        const auto result = MakeResult();
        frame_open = false;
        selected = false;
        return result;
    }

private:
    [[nodiscard]] constexpr bool CanBeginRawPrefixRestart() const noexcept {
        return !HasFailure() && frame_open && phase == PpDepthPredecessorTwoHopPhase::ArmedP2 &&
               frame_chain_kind ==
                   PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter &&
               occurrence_role == PpDepthPredecessorTwoHopOccurrenceRole::P1Tail &&
               raw_occurrences != 0 && observed_occurrences == 0 && bridge_occurrences == 0 &&
               !p1_seen && !anchor_seen && !prefix_join_exact && !candidate_saved &&
               pending_prefix.Valid() && pending_candidate.valid;
    }

    [[nodiscard]] constexpr bool HasFailure() const noexcept {
        switch (last_decision) {
        case PpDepthPredecessorTwoHopDecision::Idle:
        case PpDepthPredecessorTwoHopDecision::LearnedP1:
        case PpDepthPredecessorTwoHopDecision::Committed:
        case PpDepthPredecessorTwoHopDecision::Selected:
            return false;
        default:
            return true;
        }
    }

    constexpr void ResetP1Occurrence(bool preserve_prefix_candidate = false) noexcept {
        p1_intent = {};
        p1_post = {};
        if (!preserve_prefix_candidate) {
            pending_p2 = {};
            pending_candidate = {};
            selected_index = 0;
            prefix_join_exact = false;
            candidate_saved = false;
        }
        p1_seen = false;
        p1_post_seen = false;
        p1_intent_exact = false;
        p2_exact = false;
        p1_post_exact = false;
        committed = false;
        selected = false;
    }

    constexpr void ResetFrame() noexcept {
        phase = PpDepthPredecessorTwoHopPhase::LearnP1;
        last_decision = PpDepthPredecessorTwoHopDecision::Idle;
        frame_chain_kind = pair_ready ? committed_chain_kind
                                      : PpDepthPredecessorTwoHopChainKind::Distinct;
        occurrence_role = PpDepthPredecessorTwoHopOccurrenceRole::None;
        frame_expected_occurrences = pair_ready ? committed_expected_occurrences : 0;
        observed_occurrences = 0;
        raw_occurrences = 0;
        bridge_occurrences = 0;
        raw_prefix_restart_attempt_count = 0;
        raw_prefix_restart_completion_count = 0;
        pending_anchor_hint = {};
        pending_p1_hint = {};
        p1_intent = {};
        pending_p2 = {};
        pending_prefix = {};
        rolling_prefix_prior = {};
        root_prefix = {};
        p1_post = {};
        pending_candidate = {};
        root_candidate = {};
        selected_index = 0;
        rolling_prefix_candidate_index = 0;
        p1_pre_ordinal = 0;
        collision_p1_ordinal = 0;
        identity_collision = PpDepthPredecessorTwoHopIdentityCollision::None;
        prefix_join_discriminator =
            PpDepthPredecessorPrefixJoinDiscriminator::NotAttempted;
        anchor_seen = false;
        p1_seen = false;
        p1_post_seen = false;
        anchor_exact = false;
        p1_intent_exact = false;
        p2_exact = false;
        p1_post_exact = false;
        prefix_join_exact = false;
        candidate_saved = false;
        raw_prefix_restart_open = false;
        raw_prefix_restart_count_overflow = false;
        root_prefix_valid = false;
        committed = false;
        selected = false;
    }

    constexpr void Fail(PpDepthPredecessorTwoHopDecision decision) noexcept {
        last_decision = decision;
        root_prefix = {};
        root_candidate = {};
        root_prefix_valid = false;
        committed = false;
        selected = false;
    }

    [[nodiscard]] constexpr PpDepthPredecessorTwoHopResult MakeResult() const noexcept {
        return {
            .phase = phase,
            .decision = last_decision,
            .identity_collision = identity_collision,
            .chain_kind = frame_chain_kind,
            .occurrence_role = occurrence_role,
            .prefix_join_discriminator = prefix_join_discriminator,
            .generation = phase == PpDepthPredecessorTwoHopPhase::ArmedP2 ? generation : 0,
            .candidate_index = selected || candidate_saved ? pending_candidate.candidate_index : 0,
            .collision_p1_ordinal = collision_p1_ordinal,
            .expected_occurrences = frame_expected_occurrences,
            .observed_occurrences = observed_occurrences,
            .raw_occurrences = raw_occurrences,
            .bridge_occurrences = bridge_occurrences,
            .metadata_only = phase == PpDepthPredecessorTwoHopPhase::LearnP1 ||
                             phase == PpDepthPredecessorTwoHopPhase::LearnP2,
            .raw_armed = phase == PpDepthPredecessorTwoHopPhase::ArmedP2 && frame_open,
            .anchor_exact = anchor_exact,
            .p1_intent = p1_intent_exact,
            .p2_exact = p2_exact,
            .p1_post_exact = p1_post_exact,
            .prefix_join_exact = prefix_join_exact,
            .candidate_saved = candidate_saved,
            .committed = committed,
            .select = selected,
        };
    }

    u32 capacity{};
    u64 generation{};
    u32 hint_hops{2};
    PpDepthPredecessorTwoHopPhase phase{PpDepthPredecessorTwoHopPhase::LearnP1};
    PpDepthPredecessorTwoHopDecision last_decision{PpDepthPredecessorTwoHopDecision::Idle};
    PpDepthPredecessorHint anchor_hint{};
    PpDepthPredecessorHint p1_hint{};
    PpDepthPredecessorHint p2_hint{};
    PpDepthPredecessorTwoHopChainKind committed_chain_kind{
        PpDepthPredecessorTwoHopChainKind::Distinct};
    PpDepthPredecessorTwoHopChainKind frame_chain_kind{
        PpDepthPredecessorTwoHopChainKind::Distinct};
    PpDepthPredecessorTwoHopOccurrenceRole occurrence_role{
        PpDepthPredecessorTwoHopOccurrenceRole::None};
    PpDepthPredecessorHint pending_anchor_hint{};
    PpDepthPredecessorHint pending_p1_hint{};
    PpDepthPredecessorWriterIntent p1_intent{};
    PpDepthPredecessorWriterStamp pending_p2{};
    PpDepthPredecessorWriterStamp pending_prefix{};
    PpDepthPredecessorWriterStamp rolling_prefix_prior{};
    PpDepthPredecessorWriterStamp root_prefix{};
    PpDepthPredecessorWriterStamp p1_post{};
    PpDepthPredecessorCandidateStamp pending_candidate{};
    PpDepthPredecessorCandidateStamp root_candidate{};
    u32 selected_index{};
    u32 rolling_prefix_candidate_index{};
    u32 p1_pre_ordinal{};
    u32 collision_p1_ordinal{};
    u32 committed_expected_occurrences{};
    u32 frame_expected_occurrences{};
    u32 observed_occurrences{};
    u32 raw_occurrences{};
    u32 bridge_occurrences{};
    u32 raw_prefix_restart_attempt_count{};
    u32 raw_prefix_restart_completion_count{};
    PpDepthPredecessorTwoHopIdentityCollision identity_collision{
        PpDepthPredecessorTwoHopIdentityCollision::None};
    PpDepthPredecessorPrefixJoinDiscriminator prefix_join_discriminator{
        PpDepthPredecessorPrefixJoinDiscriminator::NotAttempted};
    bool pair_ready{};
    bool frame_open{};
    bool anchor_seen{};
    bool p1_seen{};
    bool p1_post_seen{};
    bool anchor_exact{};
    bool p1_intent_exact{};
    bool p2_exact{};
    bool p1_post_exact{};
    bool prefix_join_exact{};
    bool candidate_saved{};
    bool raw_prefix_restart_open{};
    bool raw_prefix_restart_count_overflow{};
    bool root_prefix_valid{};
    bool committed{};
    bool selected{};
};

struct PpDepthPredecessorEndpointStamp {
    PpDepthPredecessorWriterStamp writer{};
    u64 guest_rendering_serial{};
    u64 host_rendering_instance_id{};
    bool producer_marked{};
    // Host-scope provenance only. A later real guest writer can be encoded after a diagnostic
    // resume and remains a valid producer endpoint.
    bool after_diagnostic_resume{};
    bool valid{};

    [[nodiscard]] constexpr bool Valid() const noexcept {
        return valid && writer.Valid() && guest_rendering_serial != 0 &&
               host_rendering_instance_id != 0 && producer_marked;
    }

    bool operator==(const PpDepthPredecessorEndpointStamp&) const = default;
};

struct PpDepthPredecessorCrossScopeLearningObservation {
    PpDepthPredecessorEndpointStamp source_current{};
    PpDepthPredecessorWriterIntent target_intent{};
    PpUpstreamDepthFirstWriterPreObservation predecessor{};
    u64 target_guest_rendering_serial{};
    u64 target_host_rendering_instance_id{};
    bool source_outside_target_guest_scope{};
    bool target_rendering_active{};
    bool after_begin_rendering{};
    bool target_regular_origin{};
    bool target_after_diagnostic_resume{};
    bool target_depth_load{};
    bool target_depth_clear{};
    bool target_stencil_clear{};
    bool depth_aspect{};
    bool canonical_attachment{};
};

struct PpDepthPredecessorCrossScopeLearningClassification {
    bool predecessor_exact{};
    bool join_exact{};
    bool regular_cross_scope_exact{};
    bool load_exact{};
    bool exact{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeLearningClassification
ClassifyPpDepthPredecessorCrossScopeLearningBoundary(
    const PpDepthPredecessorCrossScopeLearningObservation& observation) noexcept {
    const auto& source = observation.source_current;
    const auto& target = observation.target_intent;
    const bool predecessor_exact = observation.predecessor.recorded &&
                                   observation.predecessor.depth_stencil_exact &&
                                   !observation.predecessor.same_scope;
    const bool join_exact = source.Valid() && target.Valid() &&
                            source.writer.link == target.link &&
                            source.writer.range == target.range &&
                            source.writer.backing_generation == target.backing_generation &&
                            source.writer.mutation_epoch == target.mutation_epoch &&
                            IsCanonicalPpDepthPredecessorRange(source.writer.range) &&
                            IsCanonicalPpDepthPredecessorRange(target.range);
    const bool serial_exact = source.guest_rendering_serial != 0 &&
                              source.guest_rendering_serial !=
                                  std::numeric_limits<u64>::max() &&
                              observation.target_guest_rendering_serial ==
                                  source.guest_rendering_serial + 1;
    const bool host_exact = source.host_rendering_instance_id != 0 &&
                            observation.target_host_rendering_instance_id != 0 &&
                            observation.target_host_rendering_instance_id !=
                                source.host_rendering_instance_id;
    const bool regular_cross_scope_exact =
        serial_exact && host_exact && observation.source_outside_target_guest_scope &&
        observation.target_rendering_active && observation.after_begin_rendering &&
        observation.target_regular_origin && !observation.target_after_diagnostic_resume;
    const bool load_exact = observation.target_depth_load &&
                            !observation.target_depth_clear &&
                            !observation.target_stencil_clear && observation.depth_aspect &&
                            observation.canonical_attachment;
    return {
        .predecessor_exact = predecessor_exact,
        .join_exact = join_exact,
        .regular_cross_scope_exact = regular_cross_scope_exact,
        .load_exact = load_exact,
        .exact = predecessor_exact && join_exact && regular_cross_scope_exact && load_exact,
    };
}

struct PpDepthPredecessorCrossScopeBoundaryObservation {
    PpDepthPredecessorEndpointStamp source_post{};
    PpDepthPredecessorWriterStamp target_pre_current_writer{};
    PpDepthPredecessorWriterIntent target_intent{};
    PpDepthPredecessorCandidateStamp candidate{};
    u64 resume_guest_rendering_serial{};
    u64 resume_host_rendering_instance_id{};
    u64 target_guest_rendering_serial{};
    u64 target_host_rendering_instance_id{};
    bool resume_diagnostic_origin{};
    bool resume_preserved_guest_serial{};
    bool resume_rendering_active{};
    bool source_outside_target_guest_scope{};
    bool after_begin_rendering{};
    bool target_rendering_active{};
    bool target_regular_origin{};
    bool target_after_diagnostic_resume{};
    bool target_depth_load{};
    bool target_depth_clear{};
    bool target_stencil_clear{};
    bool depth_aspect{};
    bool canonical_attachment{};
};

struct PpDepthPredecessorCrossScopeBoundaryClassification {
    bool endpoint_exact{};
    bool candidate_exact{};
    bool resume_exact{};
    bool scope_transition_exact{};
    bool load_exact{};
    bool exact{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeBoundaryClassification
ClassifyPpDepthPredecessorCrossScopeBoundary(
    const PpDepthPredecessorCrossScopeBoundaryObservation& observation,
    u64 expected_generation, u32 candidate_capacity) noexcept {
    const auto& source = observation.source_post;
    const auto& current = observation.target_pre_current_writer;
    const auto& target = observation.target_intent;
    const auto& candidate = observation.candidate;
    const bool endpoint_exact =
        source.Valid() && current.Valid() && target.Valid() && current == source.writer &&
        current.link == target.link && current.range == target.range &&
        current.backing_generation == target.backing_generation &&
        current.mutation_epoch == target.mutation_epoch &&
        IsCanonicalPpDepthPredecessorRange(current.range) &&
        IsCanonicalPpDepthPredecessorRange(target.range);
    const bool candidate_exact =
        candidate.valid && expected_generation != 0 && candidate.generation == expected_generation &&
        candidate_capacity != 0 &&
        candidate_capacity <= PpUpstreamFeedbackPrePostPlan::MaxCandidates &&
        candidate.candidate_index < candidate_capacity && candidate.recorded_plane_mask == 0x1u &&
        candidate.writer == source.writer;
    const bool resume_exact =
        source.guest_rendering_serial != 0 && source.host_rendering_instance_id != 0 &&
        observation.resume_guest_rendering_serial == source.guest_rendering_serial &&
        observation.resume_host_rendering_instance_id != 0 &&
        observation.resume_host_rendering_instance_id != source.host_rendering_instance_id &&
        observation.resume_diagnostic_origin && observation.resume_preserved_guest_serial &&
        observation.resume_rendering_active;
    const bool scope_transition_exact =
        source.guest_rendering_serial != std::numeric_limits<u64>::max() &&
        observation.target_guest_rendering_serial == source.guest_rendering_serial + 1 &&
        observation.target_host_rendering_instance_id != 0 &&
        observation.target_host_rendering_instance_id != source.host_rendering_instance_id &&
        observation.target_host_rendering_instance_id !=
            observation.resume_host_rendering_instance_id &&
        observation.source_outside_target_guest_scope && observation.after_begin_rendering &&
        observation.target_rendering_active && observation.target_regular_origin &&
        !observation.target_after_diagnostic_resume;
    const bool load_exact = observation.target_depth_load &&
                            !observation.target_depth_clear &&
                            !observation.target_stencil_clear && observation.depth_aspect &&
                            observation.canonical_attachment;
    return {
        .endpoint_exact = endpoint_exact,
        .candidate_exact = candidate_exact,
        .resume_exact = resume_exact,
        .scope_transition_exact = scope_transition_exact,
        .load_exact = load_exact,
        .exact = endpoint_exact && candidate_exact && resume_exact && scope_transition_exact &&
                 load_exact,
    };
}

enum class PpDepthPredecessorCrossScopePairPhase : u32 {
    Learning,
    Armed,
    GenerationWrap,
};

enum class PpDepthPredecessorCrossScopePairRole : u32 {
    None,
    SourcePost,
    TargetPre,
    Tail,
};

enum class PpDepthPredecessorCrossScopePairDecision : u32 {
    Idle,
    Committed,
    Selected,
    Mode3ProofMismatch,
    SourceMismatch,
    CandidateMismatch,
    BoundaryMismatch,
    TargetMismatch,
    TargetFirstWriterMismatch,
    MissingSourcePost,
    DuplicateSourcePost,
    MissingTargetPre,
    DuplicateTargetPre,
    MissingTailPost,
    DuplicateTailPost,
    TailMismatch,
    EarlyAnchor,
    AnchorMismatch,
    DuplicateAnchor,
    MissingAnchor,
    GenerationWrap,
    MissingTargetRepeatPost,
    DuplicateTargetRepeatPre,
    DuplicateTargetRepeatPost,
    TargetRepeatMismatch,
};

struct PpDepthPredecessorCrossScopeResumeWitness;
struct PpDepthPredecessorCrossScopeSourceRestartPreObservation;
struct PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot;
struct PpDepthPredecessorCrossScopeLearningLogicalCandidateSummary;
struct PpDepthPredecessorCrossScopeLearningTransitionCandidateSummary;
struct PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence;
struct PpDepthPredecessorCrossScopeMode3RootBindingEvidence;
enum class PpDepthPredecessorCrossScopeMode3SelectedLatestBindingReason : u32 {
    NotAttempted,
    Exact,
    SelectedCoreMismatch,
    RawRollingMismatch,
    RestartCountMismatch,
    LatestWitnessMismatch,
    RuntimeSelectionMismatch,
    GenerationMismatch,
    CandidateMismatch,
    TargetPostMismatch,
    LogicalSummaryMismatch,
    TransitionSummaryMismatch,
    LearningStateMismatch,
};
enum class PpDepthPredecessorCrossScopeMode3RootBindingReason : u32;

struct PpDepthPredecessorCrossScopePairResult {
    PpDepthPredecessorCrossScopePairPhase phase{
        PpDepthPredecessorCrossScopePairPhase::Learning};
    PpDepthPredecessorCrossScopePairRole role{PpDepthPredecessorCrossScopePairRole::None};
    PpDepthPredecessorCrossScopePairDecision decision{
        PpDepthPredecessorCrossScopePairDecision::Idle};
    u64 generation{};
    u32 candidate_index{};
    u32 expected_tail_occurrences{};
    u32 observed_tail_occurrences{};
    u32 source_post_count{};
    u32 source_restart_count{};
    u32 target_pre_count{};
    u32 expected_target_repeat_count{};
    u32 target_repeat_attempt_count{};
    u32 target_repeat_completion_count{};
    bool boundary_exact{};
    bool target_first_writer_exact{};
    bool tail_exact{};
    bool candidate_saved{};
    bool source_restart_open{};
    bool target_repeat_pre_open{};
    bool committed{};
    bool select{};

    bool operator==(const PpDepthPredecessorCrossScopePairResult&) const = default;
};

struct PpDepthPredecessorCrossScopeArmedP0RepeatProbeSlotStamp {
    u64 generation{};
    u32 candidate_index{};
    bool present{};

    bool operator==(
        const PpDepthPredecessorCrossScopeArmedP0RepeatProbeSlotStamp&) const = default;
};

struct PpDepthPredecessorCrossScopeTargetRepeatPreObservation {
    PpDepthPredecessorEndpointStamp stored_latest{};
    PpDepthPredecessorEndpointStamp current_endpoint{};
    PpDepthPredecessorWriterIntent intent{};
    PpDepthPredecessorCandidateStamp candidate{};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeSlotStamp slot{};
    FinalGuestSurfaceReadbackSlotPool::Token retained_slot_token{};
    u64 live_guest_rendering_serial{};
    u64 live_host_rendering_instance_id{};
    bool live_rendering_active{};
    bool live_regular_origin{};
    bool live_diagnostic_origin{};
    bool configured_effective_depth_write{};
    bool canonical_draw{};
    bool cross_open_collision{};
};

struct PpDepthPredecessorCrossScopeTargetRepeatPostObservation {
    PpDepthPredecessorEndpointStamp post_endpoint{};
    PpDepthPredecessorCandidateStamp candidate{};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeSlotStamp slot{};
    FinalGuestSurfaceReadbackSlotPool::Token retained_slot_token{};
    bool configured_effective_depth_write{};
    bool regular_origin{};
    bool diagnostic_origin{};
    PpUpstreamDepthFirstWriterClassification first_writer{};
};

struct PpDepthPredecessorCrossScopeTargetRepeatRetainedState {
    PpDepthPredecessorCandidateStamp candidate{};
    PpDepthPredecessorEndpointStamp latest_p0_post{};
    FinalGuestSurfaceReadbackSlotPool::Token retained_slot_token{};
    u32 expected_count{};
    u32 attempt_count{};
    u32 completion_count{};
    bool pre_open{};

    bool operator==(
        const PpDepthPredecessorCrossScopeTargetRepeatRetainedState&) const = default;
};

enum class PpDepthPredecessorCrossScopeFirstTailPreP0DriftReason : u32 {
    NotAttempted,
    LifecycleNotReady,
    StoredInvalid,
    CurrentInvalid,
    StoredNotTarget,
    CurrentNotTarget,
    IdentityMismatch,
    ScopeMismatch,
    IntentMismatch,
    NoDrift,
    SuccessorP0Drift,
    AdvancedP0Drift,
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeFirstTailPreP0DriftReason
SanitizePpDepthPredecessorCrossScopeFirstTailPreP0DriftReason(
    PpDepthPredecessorCrossScopeFirstTailPreP0DriftReason reason) noexcept {
    using Reason = PpDepthPredecessorCrossScopeFirstTailPreP0DriftReason;
    switch (reason) {
    case Reason::NotAttempted:
    case Reason::LifecycleNotReady:
    case Reason::StoredInvalid:
    case Reason::CurrentInvalid:
    case Reason::StoredNotTarget:
    case Reason::CurrentNotTarget:
    case Reason::IdentityMismatch:
    case Reason::ScopeMismatch:
    case Reason::IntentMismatch:
    case Reason::NoDrift:
    case Reason::SuccessorP0Drift:
    case Reason::AdvancedP0Drift:
        return reason;
    }
    return Reason::NotAttempted;
}

struct PpDepthPredecessorCrossScopeFirstTailPreP0DriftObservation {
    PpDepthPredecessorEndpointStamp stored_p0_post{};
    PpDepthPredecessorEndpointStamp current_endpoint{};
    PpDepthPredecessorWriterIntent tail_intent{};
    PpDepthPredecessorHint expected_p0_hint{};
    PpDepthPredecessorHint expected_p1_hint{};
    u64 live_guest_rendering_serial{};
    u64 live_host_rendering_instance_id{};
    bool live_rendering_active{};
    bool live_regular_origin{};
    bool live_diagnostic_origin{};
    bool lifecycle_ready{};
    bool first_tail_pre{};
};

struct PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidence {
    PpDepthPredecessorCrossScopeFirstTailPreP0DriftReason reason{
        PpDepthPredecessorCrossScopeFirstTailPreP0DriftReason::NotAttempted};
    bool attempted{};
    bool lifecycle_ready{};
    bool stored_valid{};
    bool current_valid{};
    bool stored_target_hint_exact{};
    bool current_target_hint_exact{};
    bool identity_exact{};
    bool stored_current_guest_equal{};
    bool stored_current_host_equal{};
    bool current_live_guest_equal{};
    bool current_live_host_equal{};
    bool live_active{};
    bool live_regular{};
    bool live_non_diagnostic{};
    bool stored_endpoint_non_diagnostic{};
    bool current_endpoint_non_diagnostic{};
    bool stored_current_guest_successor{};
    bool stored_current_host_changed{};
    bool current_live_guest_successor{};
    bool current_live_host_changed{};
    bool scope_exact{};
    bool intent_exact{};
    bool mutation_equal{};
    bool mutation_successor{};
    bool mutation_newer{};
    bool all_except_prior_exact{};
    bool stale_target_exact{};

    bool operator==(
        const PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidence&) const = default;
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidence
ClassifyPpDepthPredecessorCrossScopeFirstTailPreP0Drift(
    const PpDepthPredecessorCrossScopeFirstTailPreP0DriftObservation& observation) noexcept {
    using Reason = PpDepthPredecessorCrossScopeFirstTailPreP0DriftReason;
    const auto& stored = observation.stored_p0_post;
    const auto& current = observation.current_endpoint;
    const auto& intent = observation.tail_intent;
    const bool stored_valid = stored.Valid();
    const bool current_valid = current.Valid();
    const bool stored_target_hint_exact =
        stored_valid && observation.expected_p0_hint.Valid() &&
        stored.writer.hint == observation.expected_p0_hint;
    const bool current_target_hint_exact =
        current_valid && observation.expected_p0_hint.Valid() &&
        current.writer.hint == observation.expected_p0_hint;
    const bool identity_exact =
        stored_valid && current_valid && stored.writer.link == current.writer.link &&
        stored.writer.range == current.writer.range &&
        stored.writer.backing_generation == current.writer.backing_generation;
    const bool stored_current_guest_equal =
        stored.guest_rendering_serial == current.guest_rendering_serial;
    const bool stored_current_host_equal =
        stored.host_rendering_instance_id == current.host_rendering_instance_id;
    const bool current_live_guest_equal =
        observation.live_guest_rendering_serial == current.guest_rendering_serial;
    const bool current_live_host_equal =
        observation.live_host_rendering_instance_id ==
        current.host_rendering_instance_id;
    const bool live_active = observation.live_rendering_active;
    const bool live_regular = observation.live_regular_origin;
    const bool live_non_diagnostic = !observation.live_diagnostic_origin;
    const bool stored_endpoint_non_diagnostic = !stored.after_diagnostic_resume;
    const bool current_endpoint_non_diagnostic = !current.after_diagnostic_resume;
    const bool stored_current_guest_successor =
        stored.guest_rendering_serial != 0 &&
        current.guest_rendering_serial != 0 &&
        stored.guest_rendering_serial != std::numeric_limits<u64>::max() &&
        current.guest_rendering_serial == stored.guest_rendering_serial + 1;
    const bool stored_current_host_changed =
        stored.host_rendering_instance_id != 0 &&
        current.host_rendering_instance_id != 0 &&
        stored.host_rendering_instance_id != current.host_rendering_instance_id;
    const bool current_live_guest_successor =
        current.guest_rendering_serial != 0 &&
        observation.live_guest_rendering_serial != 0 &&
        current.guest_rendering_serial != std::numeric_limits<u64>::max() &&
        observation.live_guest_rendering_serial ==
            current.guest_rendering_serial + 1;
    const bool current_live_host_changed =
        current.host_rendering_instance_id != 0 &&
        observation.live_host_rendering_instance_id != 0 &&
        current.host_rendering_instance_id !=
            observation.live_host_rendering_instance_id;
    const bool scope_exact =
        stored_current_guest_equal && stored_current_host_equal &&
        current_live_guest_equal && current_live_host_equal && live_active &&
        live_regular && live_non_diagnostic && stored_endpoint_non_diagnostic &&
        current_endpoint_non_diagnostic;
    const bool intent_exact =
        intent.Valid() && observation.expected_p1_hint.Valid() &&
        intent.hint == observation.expected_p1_hint && current_valid &&
        intent.link == current.writer.link && intent.range == current.writer.range &&
        intent.backing_generation == current.writer.backing_generation &&
        intent.mutation_epoch == current.writer.mutation_epoch;
    const bool mutation_equal = stored_valid && current_valid &&
                                stored.writer.mutation_epoch ==
                                    current.writer.mutation_epoch;
    const bool mutation_successor =
        stored_valid && current_valid &&
        stored.writer.mutation_epoch != std::numeric_limits<u64>::max() &&
        current.writer.mutation_epoch == stored.writer.mutation_epoch + 1;
    const bool mutation_newer =
        stored_valid && current_valid &&
        stored.writer.mutation_epoch < current.writer.mutation_epoch;
    const bool all_except_prior_exact =
        observation.first_tail_pre && observation.lifecycle_ready && stored_valid &&
        current_valid && stored_target_hint_exact && current_target_hint_exact &&
        identity_exact && scope_exact && intent_exact;
    const bool stale_target_exact = all_except_prior_exact && mutation_newer;
    const auto reason =
        !observation.first_tail_pre
            ? Reason::NotAttempted
        : !observation.lifecycle_ready ? Reason::LifecycleNotReady
        : !stored_valid               ? Reason::StoredInvalid
        : !current_valid              ? Reason::CurrentInvalid
        : !stored_target_hint_exact   ? Reason::StoredNotTarget
        : !current_target_hint_exact  ? Reason::CurrentNotTarget
        : !identity_exact             ? Reason::IdentityMismatch
        : !scope_exact                ? Reason::ScopeMismatch
        : !intent_exact               ? Reason::IntentMismatch
        : mutation_successor          ? Reason::SuccessorP0Drift
        : mutation_newer              ? Reason::AdvancedP0Drift
                                      : Reason::NoDrift;
    return {
        .reason = reason,
        .attempted = observation.first_tail_pre,
        .lifecycle_ready = observation.lifecycle_ready,
        .stored_valid = stored_valid,
        .current_valid = current_valid,
        .stored_target_hint_exact = stored_target_hint_exact,
        .current_target_hint_exact = current_target_hint_exact,
        .identity_exact = identity_exact,
        .stored_current_guest_equal = stored_current_guest_equal,
        .stored_current_host_equal = stored_current_host_equal,
        .current_live_guest_equal = current_live_guest_equal,
        .current_live_host_equal = current_live_host_equal,
        .live_active = live_active,
        .live_regular = live_regular,
        .live_non_diagnostic = live_non_diagnostic,
        .stored_endpoint_non_diagnostic = stored_endpoint_non_diagnostic,
        .current_endpoint_non_diagnostic = current_endpoint_non_diagnostic,
        .stored_current_guest_successor = stored_current_guest_successor,
        .stored_current_host_changed = stored_current_host_changed,
        .current_live_guest_successor = current_live_guest_successor,
        .current_live_host_changed = current_live_host_changed,
        .scope_exact = scope_exact,
        .intent_exact = intent_exact,
        .mutation_equal = mutation_equal,
        .mutation_successor = mutation_successor,
        .mutation_newer = mutation_newer,
        .all_except_prior_exact = all_except_prior_exact,
        .stale_target_exact = stale_target_exact,
    };
}

struct PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceSummary {
    u32 attempt_count{};
    bool count_overflow{};
    bool unique_valid{};
    PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidence unique_evidence{};

    bool operator==(
        const PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceSummary&) const = default;
};

enum class PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceEvent : u32 {
    NonAttempt,
    Attempt,
    Reset,
};

struct PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceUpdate {
    PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceEvent event{
        PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceEvent::NonAttempt};
    PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidence evidence{};
};

[[nodiscard]] constexpr
PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceSummary
AccumulatePpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidence(
    PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceSummary summary,
    const PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceUpdate& update) noexcept {
    using Event = PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceEvent;
    switch (update.event) {
    case Event::NonAttempt:
        return summary;
    case Event::Reset:
        return {};
    case Event::Attempt:
        break;
    default:
        return {};
    }
    if (summary.count_overflow || summary.attempt_count ==
                                      std::numeric_limits<u32>::max()) {
        summary.count_overflow = true;
        summary.unique_valid = false;
        summary.unique_evidence = {};
        return summary;
    }
    ++summary.attempt_count;
    if (summary.attempt_count == 1) {
        summary.unique_valid = true;
        summary.unique_evidence = update.evidence;
        summary.unique_evidence.reason =
            SanitizePpDepthPredecessorCrossScopeFirstTailPreP0DriftReason(
                summary.unique_evidence.reason);
    } else {
        summary.unique_valid = false;
        summary.unique_evidence = {};
    }
    return summary;
}

enum class PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason : u32 {
    NotAttempted,
    Exact,
    RootMismatch,
    LifecycleMismatch,
    BudgetMismatch,
    PreMismatch,
    PostMismatch,
    Incomplete,
    TailMismatch,
    Multiplicity,
    CountOverflow,
};

[[nodiscard]] constexpr
PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason
SanitizePpDepthPredecessorCrossScopeArmedP0RepeatProbeReason(
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason reason) noexcept {
    using Reason = PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason;
    switch (reason) {
    case Reason::NotAttempted:
    case Reason::Exact:
    case Reason::RootMismatch:
    case Reason::LifecycleMismatch:
    case Reason::BudgetMismatch:
    case Reason::PreMismatch:
    case Reason::PostMismatch:
    case Reason::Incomplete:
    case Reason::TailMismatch:
    case Reason::Multiplicity:
    case Reason::CountOverflow:
        return reason;
    }
    return Reason::NotAttempted;
}

struct PpDepthPredecessorCrossScopeArmedP0RepeatProbeSelectedLatestEvidence {
    bool attempted{};
    PpDepthPredecessorCrossScopeMode3SelectedLatestBindingReason reason{};
    u32 raw_occurrences{};
    u32 restart_attempt_count{};
    u32 restart_completion_count{};
    bool raw_rolling_bounded{};
    bool selected_core_without_raw_exact{};
    bool rolling_counts_exact{};
    bool latest_witness_exact{};
    bool runtime_selection_exact{};
    bool generation_exact{};
    bool candidate_index_exact{};
    bool candidate_mask_exact{};
    bool actual_selected_latest_exact{};
    bool latest_p0_post_exact{};
    bool intent_p0_post_exact{};
    bool logical_summary_exact{};
    bool transition_summary_exact{};
    bool lifecycle_learning_exact{};
    bool exact{};
};

struct PpDepthPredecessorCrossScopeArmedP0RepeatProbeRootBindingEvidence {
    bool attempted{};
    PpDepthPredecessorCrossScopeMode3RootBindingReason reason{};
    bool root_witness_exact{};
    bool cross_state_present{};
    bool cross_root_exact{};
    bool chain_exact{};
    bool latest_selection_exact{};
    bool lifecycle_exact{};
    bool exact{};
};

struct PpDepthPredecessorCrossScopeArmedP0RepeatProbeConfiguration {
    PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness selected_latest{};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeSelectedLatestEvidence
        selected_latest_evidence{};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeRootBindingEvidence
        root_binding_evidence{};
    PpDepthPredecessorHint expected_p0_hint{};
};

struct PpDepthPredecessorCrossScopeArmedP0RepeatTargetPostObservation {
    PpDepthPredecessorCrossScopePairResult lifecycle{};
    PpDepthPredecessorEndpointStamp target_post{};
    PpDepthPredecessorCandidateStamp candidate{};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeSlotStamp slot{};
    bool tail_pre_open{};
    bool anchor_seen{};
    bool repeat_pre_open{};
    bool collision{};
};

struct PpDepthPredecessorCrossScopeArmedP0RepeatPreObservation {
    PpDepthPredecessorCrossScopePairResult lifecycle{};
    PpDepthPredecessorEndpointStamp stored_probe_latest{};
    PpDepthPredecessorEndpointStamp current_endpoint{};
    PpDepthPredecessorWriterIntent intent{};
    PpDepthPredecessorCandidateStamp candidate{};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeSlotStamp slot{};
    u64 live_guest_rendering_serial{};
    u64 live_host_rendering_instance_id{};
    bool live_rendering_active{};
    bool live_regular_origin{};
    bool live_diagnostic_origin{};
    bool configured_effective_depth_write{};
    bool canonical_draw{};
    bool tail_pre_open{};
    bool anchor_seen{};
    bool repeat_pre_open{};
    bool collision{};
};

struct PpDepthPredecessorCrossScopeArmedP0RepeatPostObservation {
    PpDepthPredecessorCrossScopePairResult lifecycle{};
    PpDepthPredecessorEndpointStamp post_endpoint{};
    PpDepthPredecessorCandidateStamp candidate{};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeSlotStamp slot{};
    bool configured_effective_depth_write{};
    bool regular_origin{};
    bool diagnostic_origin{};
    PpUpstreamDepthFirstWriterClassification first_writer{};
};

struct PpDepthPredecessorCrossScopeArmedP0RepeatTailPreObservation {
    PpDepthPredecessorCrossScopePairResult lifecycle{};
    PpDepthPredecessorEndpointStamp current_endpoint{};
    PpDepthPredecessorWriterIntent tail_intent{};
    PpDepthPredecessorHint expected_p1_hint{};
    PpDepthPredecessorCandidateStamp candidate{};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeSlotStamp slot{};
    bool first_tail_pre{};
    bool anchor_seen{};
};

struct PpDepthPredecessorCrossScopeArmedP0RepeatProbeResult {
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason reason{
        PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::NotAttempted};
    PpDepthPredecessorEndpointStamp latest_endpoint{};
    u32 expected_repeat_count{};
    u32 attempt_count{};
    u32 completion_count{};
    bool attempted{};
    bool count_overflow{};
    bool root_exact{};
    bool lifecycle_exact{};
    bool stored_target_post_after_diagnostic{};
    bool budget_exact{};
    bool stored_latest_exact{};
    bool current_writer_exact{};
    bool intent_exact{};
    bool pre_identity_exact{};
    bool pre_scope_exact{};
    bool candidate_exact{};
    bool open_state_exact{};
    bool pre_open{};
    bool pre_exact{};
    bool post_correlation_exact{};
    bool post_writer_successor{};
    bool post_scope_exact{};
    bool first_writer_exact{};
    bool candidate_preserved{};
    bool slot_preserved{};
    bool post_exact{};
    bool latest_current_exact{};
    bool tail_intent_exact{};
    bool counts_exact{};
    bool exact{};

    bool operator==(
        const PpDepthPredecessorCrossScopeArmedP0RepeatProbeResult&) const = default;
};

class PpDepthPredecessorCrossScopeArmedP0RepeatProbe {
public:
    [[nodiscard]] constexpr PpDepthPredecessorCrossScopeArmedP0RepeatProbeResult
    Configure(
        const PpDepthPredecessorCrossScopeArmedP0RepeatProbeConfiguration& config) noexcept {
        Reset();
        const auto& binding = config.selected_latest_evidence;
        if (binding.raw_occurrences <= 1 ||
            binding.raw_occurrences == std::numeric_limits<u32>::max()) {
            Fail(PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::BudgetMismatch);
            return result;
        }
        result.expected_repeat_count = binding.raw_occurrences - 1;
        const auto& witness = config.selected_latest;
        const auto& root = config.root_binding_evidence;
        const bool counts_exact =
            witness.restart_attempt_count == result.expected_repeat_count &&
            witness.restart_completion_count == result.expected_repeat_count &&
            binding.restart_attempt_count == result.expected_repeat_count &&
            binding.restart_completion_count == result.expected_repeat_count;
        const bool witness_exact =
            witness.valid && witness.root_valid && witness.chain_exact &&
            witness.root_to_latest_chain_exact && !witness.restart_open &&
            !witness.count_overflow && witness.candidate_capacity != 0 &&
            witness.candidate_capacity <= PpUpstreamFeedbackPrePostPlan::MaxCandidates &&
            witness.root_prefix.Valid() && witness.latest_prefix.Valid() &&
            config.expected_p0_hint.valid &&
            witness.root_prefix.hint == config.expected_p0_hint &&
            witness.latest_prefix.hint == config.expected_p0_hint &&
            CandidateMatchesWitness(witness.root_candidate, witness.root_prefix,
                                    witness.candidate_capacity) &&
            CandidateMatchesWitness(witness.latest_candidate, witness.latest_prefix,
                                    witness.candidate_capacity) &&
            witness.root_candidate.generation == witness.latest_candidate.generation &&
            witness.root_candidate.candidate_index ==
                witness.latest_candidate.candidate_index && counts_exact;
        const bool binding_exact =
            binding.attempted &&
            binding.reason ==
                PpDepthPredecessorCrossScopeMode3SelectedLatestBindingReason::
                    TargetPostMismatch &&
            binding.raw_rolling_bounded &&
            binding.selected_core_without_raw_exact && binding.rolling_counts_exact &&
            binding.latest_witness_exact && binding.runtime_selection_exact &&
            binding.generation_exact && binding.candidate_index_exact &&
            binding.candidate_mask_exact && binding.actual_selected_latest_exact &&
            !binding.latest_p0_post_exact && binding.intent_p0_post_exact &&
            binding.logical_summary_exact && binding.transition_summary_exact &&
            binding.lifecycle_learning_exact && !binding.exact;
        const bool root_exact =
            root.attempted &&
            static_cast<u32>(root.reason) == 1 &&
            root.root_witness_exact && root.cross_state_present &&
            root.cross_root_exact && root.chain_exact &&
            root.latest_selection_exact && root.lifecycle_exact && root.exact;
        result.root_exact = witness_exact && binding_exact && root_exact;
        result.budget_exact = counts_exact;
        if (!result.root_exact) {
            Fail(PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::RootMismatch);
            return result;
        }
        configured = true;
        expected_p0_hint = config.expected_p0_hint;
        return result;
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopeArmedP0RepeatProbeResult
    ObserveTargetPost(
        const PpDepthPredecessorCrossScopeArmedP0RepeatTargetPostObservation& observation) noexcept {
        if (failed || !configured) {
            return result;
        }
        const bool lifecycle_exact =
            LifecycleExact(observation.lifecycle) &&
            observation.lifecycle.candidate_index == observation.candidate.candidate_index &&
            observation.target_post.Valid() &&
            observation.target_post.writer.hint == expected_p0_hint &&
            CandidateExact(observation.candidate, observation.lifecycle) &&
            SlotExact(observation.slot, observation.lifecycle) &&
            observation.slot.candidate_index == observation.candidate.candidate_index &&
            !observation.tail_pre_open && !observation.anchor_seen &&
            !observation.repeat_pre_open && !observation.collision;
        result.lifecycle_exact = lifecycle_exact;
        result.stored_target_post_after_diagnostic =
            observation.target_post.after_diagnostic_resume;
        if (!lifecycle_exact) {
            Fail(PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::LifecycleMismatch);
            return result;
        }
        seeded = true;
        latest_endpoint = observation.target_post;
        saved_candidate = observation.candidate;
        saved_slot = observation.slot;
        result.latest_endpoint = latest_endpoint;
        return result;
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopeArmedP0RepeatProbeResult
    ObservePre(
        const PpDepthPredecessorCrossScopeArmedP0RepeatPreObservation& observation) noexcept {
        if (failed || !configured || !seeded) {
            if (!failed) {
                Fail(PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::PreMismatch);
            }
            return result;
        }
        if (pre_open || observation.repeat_pre_open) {
            Fail(PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::PreMismatch);
            return result;
        }
        if (result.attempt_count == std::numeric_limits<u32>::max()) {
            result.count_overflow = true;
            Fail(PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::CountOverflow);
            return result;
        }
        ++result.attempt_count;
        result.attempted = true;
        if (result.attempt_count > result.expected_repeat_count) {
            result.budget_exact = false;
            Fail(PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::BudgetMismatch);
            return result;
        }
        result.lifecycle_exact = LifecycleExact(observation.lifecycle);
        result.stored_latest_exact = observation.stored_probe_latest == latest_endpoint;
        result.current_writer_exact = observation.current_endpoint == latest_endpoint;
        result.intent_exact =
            observation.intent.Valid() && observation.intent.hint == expected_p0_hint &&
            observation.current_endpoint.writer.Valid() &&
            observation.current_endpoint.writer.hint == observation.intent.hint &&
            observation.current_endpoint.writer.link == observation.intent.link &&
            observation.current_endpoint.writer.range == observation.intent.range &&
            observation.current_endpoint.writer.backing_generation ==
                observation.intent.backing_generation &&
            observation.current_endpoint.writer.mutation_epoch ==
                observation.intent.mutation_epoch;
        result.pre_identity_exact =
            observation.stored_probe_latest.writer.Valid() &&
            observation.current_endpoint.writer.Valid() && observation.intent.Valid() &&
            observation.stored_probe_latest.writer ==
                observation.current_endpoint.writer && result.intent_exact &&
            observation.configured_effective_depth_write && observation.canonical_draw;
        result.pre_scope_exact =
            observation.current_endpoint.guest_rendering_serial != 0 &&
            observation.current_endpoint.guest_rendering_serial !=
                std::numeric_limits<u64>::max() &&
            observation.live_guest_rendering_serial ==
                observation.current_endpoint.guest_rendering_serial + 1 &&
            observation.current_endpoint.host_rendering_instance_id != 0 &&
            observation.live_host_rendering_instance_id != 0 &&
            observation.live_host_rendering_instance_id !=
                observation.current_endpoint.host_rendering_instance_id &&
            observation.live_rendering_active && observation.live_regular_origin &&
            !observation.live_diagnostic_origin;
        result.candidate_exact =
            CandidateSame(observation.candidate, saved_candidate) &&
            observation.candidate.recorded_plane_mask == 0x3u &&
            observation.slot == saved_slot;
        result.open_state_exact =
            !pre_open && !observation.tail_pre_open && !observation.anchor_seen &&
            !observation.repeat_pre_open && !observation.collision;
        result.pre_exact =
            result.lifecycle_exact && result.stored_latest_exact &&
            result.current_writer_exact && result.intent_exact &&
            result.pre_identity_exact && result.pre_scope_exact &&
            result.candidate_exact && result.open_state_exact;
        if (!result.pre_exact) {
            Fail(PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::PreMismatch);
            return result;
        }
        pre_open = true;
        result.pre_open = true;
        pre_intent = observation.intent;
        pre_live_guest = observation.live_guest_rendering_serial;
        pre_live_host = observation.live_host_rendering_instance_id;
        return result;
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopeArmedP0RepeatProbeResult
    ObservePost(
        const PpDepthPredecessorCrossScopeArmedP0RepeatPostObservation& observation) noexcept {
        if (failed || !configured || !seeded || !pre_open) {
            if (!failed) {
                Fail(PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::PostMismatch);
            }
            return result;
        }
        result.lifecycle_exact = LifecycleExact(observation.lifecycle);
        result.post_correlation_exact =
            observation.post_endpoint.Valid() && pre_intent.Valid() &&
            observation.post_endpoint.writer.hint == pre_intent.hint &&
            observation.post_endpoint.writer.link == pre_intent.link &&
            observation.post_endpoint.writer.range == pre_intent.range &&
            observation.post_endpoint.writer.backing_generation ==
                pre_intent.backing_generation;
        result.post_writer_successor =
            pre_intent.mutation_epoch != std::numeric_limits<u64>::max() &&
            observation.post_endpoint.writer.mutation_epoch ==
                pre_intent.mutation_epoch + 1;
        result.post_scope_exact =
            observation.post_endpoint.Valid() &&
            observation.post_endpoint.guest_rendering_serial == pre_live_guest &&
            observation.post_endpoint.host_rendering_instance_id == pre_live_host &&
            !observation.post_endpoint.after_diagnostic_resume &&
            observation.configured_effective_depth_write && observation.regular_origin &&
            !observation.diagnostic_origin;
        result.first_writer_exact =
            observation.first_writer.post_exact &&
            observation.first_writer.target_first_depth_writer &&
            observation.first_writer.target_scope_load &&
            !observation.first_writer.target_scope_clear;
        result.candidate_preserved =
            CandidateSame(observation.candidate, saved_candidate) &&
            observation.candidate.recorded_plane_mask == 0x3u;
        result.slot_preserved = observation.slot == saved_slot;
        result.post_exact =
            result.lifecycle_exact && result.post_correlation_exact &&
            result.post_writer_successor && result.post_scope_exact &&
            result.first_writer_exact && result.candidate_preserved &&
            result.slot_preserved;
        pre_open = false;
        result.pre_open = false;
        if (!result.post_exact) {
            Fail(PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::PostMismatch);
            return result;
        }
        if (result.completion_count == std::numeric_limits<u32>::max()) {
            result.count_overflow = true;
            Fail(PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::CountOverflow);
            return result;
        }
        ++result.completion_count;
        latest_endpoint = observation.post_endpoint;
        result.latest_endpoint = latest_endpoint;
        return result;
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopeArmedP0RepeatProbeResult
    SnapshotTailPre(
        const PpDepthPredecessorCrossScopeArmedP0RepeatTailPreObservation& observation) const noexcept {
        auto snapshot = result;
        snapshot.attempted = observation.first_tail_pre;
        snapshot.lifecycle_exact = LifecycleExact(observation.lifecycle) &&
                                   observation.first_tail_pre &&
                                   !observation.anchor_seen;
        snapshot.latest_current_exact =
            seeded && observation.current_endpoint == latest_endpoint;
        snapshot.tail_intent_exact =
            observation.tail_intent.Valid() && observation.expected_p1_hint.valid &&
            observation.tail_intent.hint == observation.expected_p1_hint &&
            observation.current_endpoint.writer.Valid() &&
            observation.current_endpoint.writer.link == observation.tail_intent.link &&
            observation.current_endpoint.writer.range == observation.tail_intent.range &&
            observation.current_endpoint.writer.backing_generation ==
                observation.tail_intent.backing_generation &&
            observation.current_endpoint.writer.mutation_epoch ==
                observation.tail_intent.mutation_epoch;
        snapshot.candidate_preserved =
            CandidateSame(observation.candidate, saved_candidate) &&
            observation.candidate.recorded_plane_mask == 0x3u;
        snapshot.slot_preserved = observation.slot == saved_slot;
        snapshot.counts_exact =
            !snapshot.count_overflow && !pre_open &&
            snapshot.attempt_count == snapshot.expected_repeat_count &&
            snapshot.completion_count == snapshot.expected_repeat_count;
        snapshot.exact =
            !failed && snapshot.lifecycle_exact && snapshot.root_exact &&
            snapshot.budget_exact && snapshot.counts_exact &&
            snapshot.latest_current_exact && snapshot.tail_intent_exact &&
            snapshot.candidate_preserved && snapshot.slot_preserved;
        if (failed) {
            snapshot.reason = result.reason;
        } else if (pre_open || !snapshot.counts_exact) {
            snapshot.reason =
                PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::Incomplete;
        } else if (!snapshot.lifecycle_exact || !snapshot.latest_current_exact ||
                   !snapshot.tail_intent_exact || !snapshot.candidate_preserved ||
                   !snapshot.slot_preserved) {
            snapshot.reason =
                PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::TailMismatch;
        } else {
            snapshot.reason =
                PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason::Exact;
        }
        return snapshot;
    }

    constexpr void Reset() noexcept {
        *this = {};
    }

    [[nodiscard]] constexpr const PpDepthPredecessorCrossScopeArmedP0RepeatProbeResult&
    Result() const noexcept {
        return result;
    }

private:
    [[nodiscard]] static constexpr bool CandidateMatchesWitness(
        const PpDepthPredecessorCandidateStamp& candidate,
        const PpDepthPredecessorWriterStamp& writer, u32 capacity) noexcept {
        return candidate.valid && candidate.generation != 0 &&
               candidate.candidate_index < capacity &&
               candidate.candidate_index < PpUpstreamFeedbackPrePostPlan::MaxCandidates &&
               candidate.recorded_plane_mask == 0x3u && candidate.writer == writer;
    }

    [[nodiscard]] static constexpr bool CandidateSame(
        const PpDepthPredecessorCandidateStamp& left,
        const PpDepthPredecessorCandidateStamp& right) noexcept {
        return left.valid && right.valid && left == right;
    }

    [[nodiscard]] static constexpr bool CandidateExact(
        const PpDepthPredecessorCandidateStamp& candidate,
        const PpDepthPredecessorCrossScopePairResult& lifecycle) noexcept {
        return candidate.valid && candidate.generation == lifecycle.generation &&
               candidate.candidate_index == lifecycle.candidate_index &&
               candidate.recorded_plane_mask == 0x3u;
    }

    [[nodiscard]] static constexpr bool SlotExact(
        const PpDepthPredecessorCrossScopeArmedP0RepeatProbeSlotStamp& slot,
        const PpDepthPredecessorCrossScopePairResult& lifecycle) noexcept {
        return slot.present && slot.generation == lifecycle.generation &&
               slot.candidate_index == lifecycle.candidate_index;
    }

    [[nodiscard]] static constexpr bool LifecycleExact(
        const PpDepthPredecessorCrossScopePairResult& lifecycle) noexcept {
        return lifecycle.phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
               lifecycle.role == PpDepthPredecessorCrossScopePairRole::Tail &&
               lifecycle.decision == PpDepthPredecessorCrossScopePairDecision::Idle &&
               lifecycle.generation != 0 && lifecycle.target_pre_count == 1 &&
               lifecycle.observed_tail_occurrences == 0 && lifecycle.boundary_exact &&
               lifecycle.target_first_writer_exact && lifecycle.candidate_saved &&
               !lifecycle.committed && !lifecycle.select;
    }

    constexpr void Fail(
        PpDepthPredecessorCrossScopeArmedP0RepeatProbeReason reason) noexcept {
        result.reason = reason;
        result.exact = false;
        result.pre_open = false;
        pre_open = false;
        failed = true;
    }

    PpDepthPredecessorCrossScopeArmedP0RepeatProbeResult result{};
    PpDepthPredecessorHint expected_p0_hint{};
    PpDepthPredecessorEndpointStamp latest_endpoint{};
    PpDepthPredecessorCandidateStamp saved_candidate{};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeSlotStamp saved_slot{};
    PpDepthPredecessorWriterIntent pre_intent{};
    u64 pre_live_guest{};
    u64 pre_live_host{};
    bool configured{};
    bool seeded{};
    bool pre_open{};
    bool failed{};
};

struct PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceSummary {
    u32 attempt_count{};
    bool count_overflow{};
    bool unique_valid{};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeResult unique_evidence{};

    bool operator==(
        const PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceSummary&) const = default;
};

struct PpDepthPredecessorCrossScopeArmedP0RepeatProbeRuntimeFrameState {
    bool correlation_open{};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceSummary evidence{};

    bool operator==(
        const PpDepthPredecessorCrossScopeArmedP0RepeatProbeRuntimeFrameState&) const = default;
};

[[nodiscard]] constexpr
PpDepthPredecessorCrossScopeArmedP0RepeatProbeRuntimeFrameState
ResetPpDepthPredecessorCrossScopeArmedP0RepeatProbeRuntimeFrame(
    const PpDepthPredecessorCrossScopeArmedP0RepeatProbeRuntimeFrameState&) noexcept {
    return {};
}

enum class PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceEvent : u32 {
    NonAttempt,
    Attempt,
    Reset,
};

struct PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceUpdate {
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceEvent event{
        PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceEvent::NonAttempt};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeResult evidence{};
};

[[nodiscard]] constexpr
PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceSummary
AccumulatePpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidence(
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceSummary summary,
    const PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceUpdate& update) noexcept {
    using Event =
        PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceEvent;
    switch (update.event) {
    case Event::NonAttempt:
        return summary;
    case Event::Reset:
        return {};
    case Event::Attempt:
        break;
    default:
        return {};
    }
    if (summary.attempt_count == std::numeric_limits<u32>::max()) {
        summary.count_overflow = true;
        summary.unique_valid = false;
        summary.unique_evidence = {};
        return summary;
    }
    ++summary.attempt_count;
    if (summary.attempt_count == 1 && !summary.count_overflow) {
        summary.unique_valid = true;
        summary.unique_evidence = update.evidence;
    } else {
        summary.unique_valid = false;
        summary.unique_evidence = {};
    }
    return summary;
}

class PpDepthPredecessorCrossScopePairLifecycle {
public:
    explicit constexpr PpDepthPredecessorCrossScopePairLifecycle(
        u32 candidate_capacity, u64 initial_generation = 0) noexcept
        : capacity{candidate_capacity}, generation{initial_generation} {}

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult BeginLearningFrame(
        const PpDepthPredecessorTwoHopResult& mode3_begin,
        const PpDepthPredecessorHint& p0, const PpDepthPredecessorHint& p1,
        const PpDepthPredecessorHint& anchor) noexcept {
        if (frame_open) {
            Fail(PpDepthPredecessorCrossScopePairDecision::Mode3ProofMismatch);
            return MakeResult();
        }
        ResetFrame(PpDepthPredecessorCrossScopePairPhase::Learning,
                   PpDepthPredecessorCrossScopePairRole::TargetPre);
        frame_open = true;
        const bool hints_exact = p0.Valid() && p1.Valid() && anchor.Valid() && p0 != p1 &&
                                 p0 != anchor && p1 != anchor;
        if (!hints_exact || !IsExactMode3Begin(mode3_begin)) {
            Fail(PpDepthPredecessorCrossScopePairDecision::Mode3ProofMismatch);
            return MakeResult();
        }
        staged_mode3_begin = mode3_begin;
        staged_p0_hint = p0;
        staged_p1_hint = p1;
        staged_anchor_hint = anchor;
        expected_tail_occurrences = mode3_begin.expected_occurrences;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult BeginFrame() noexcept {
        if (frame_open) {
            Fail(PpDepthPredecessorCrossScopePairDecision::MissingAnchor);
            return MakeResult();
        }
        ResetFrame(pair_ready ? PpDepthPredecessorCrossScopePairPhase::Armed
                              : PpDepthPredecessorCrossScopePairPhase::Learning,
                   pair_ready ? PpDepthPredecessorCrossScopePairRole::SourcePost
                              : PpDepthPredecessorCrossScopePairRole::None);
        if (!pair_ready) {
            frame_open = true;
            Fail(PpDepthPredecessorCrossScopePairDecision::Mode3ProofMismatch);
            return MakeResult();
        }
        if (generation == std::numeric_limits<u64>::max() || capacity == 0 ||
            capacity > PpUpstreamFeedbackPrePostPlan::MaxCandidates) {
            phase = PpDepthPredecessorCrossScopePairPhase::GenerationWrap;
            role = PpDepthPredecessorCrossScopePairRole::None;
            last_decision = PpDepthPredecessorCrossScopePairDecision::GenerationWrap;
            frame_open = false;
            return MakeResult();
        }
        ++generation;
        frame_open = true;
        expected_tail_occurrences = committed_expected_tail_occurrences;
        expected_target_repeat_count = committed_expected_target_repeat_count;
        return MakeResult();
    }

    [[nodiscard]] constexpr const PpDepthPredecessorHint* PeekArmedSourceHint() const noexcept {
        return pair_ready && source_hint.Valid() ? &source_hint : nullptr;
    }

    [[nodiscard]] constexpr bool MatchesSourcePost(
        const PpTerminalScopeDrawSelector& draw, u64 fragment_pgm_hash) const noexcept {
        return frame_open && phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
               role == PpDepthPredecessorCrossScopePairRole::SourcePost &&
               MatchesHint(source_hint, draw, fragment_pgm_hash);
    }

    [[nodiscard]] constexpr bool MatchesArmedSource(
        const PpTerminalScopeDrawSelector& draw, u64 fragment_pgm_hash) const noexcept {
        return frame_open && phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
               MatchesHint(source_hint, draw, fragment_pgm_hash);
    }

    [[nodiscard]] constexpr bool CanBeginSourcePostRestart() const noexcept {
        return !HasFailure() && frame_open &&
               phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
               role == PpDepthPredecessorCrossScopePairRole::TargetPre &&
               source_post_count == 1 && source_post.Valid() && source_candidate.valid &&
               !source_restart_open && !target_pre_attempted && !target_pre_seen &&
               !target_pre_complete && !target_post_seen && !tail_pre_open && !anchor_seen &&
               !candidate_saved && !selected;
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult BeginSourcePostRestart(
        const PpDepthPredecessorCrossScopeSourceRestartPreObservation& observation) noexcept;

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult
    ObserveSourcePostRestartComplete(
        const PpDepthPredecessorEndpointStamp& endpoint,
        const PpDepthPredecessorCandidateStamp& candidate,
        const PpDepthPredecessorCrossScopeResumeWitness& new_resume) noexcept;

    [[nodiscard]] constexpr bool MatchesTargetPre(
        const PpTerminalScopeDrawSelector& draw, u64 fragment_pgm_hash) const noexcept {
        return frame_open && phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
               role == PpDepthPredecessorCrossScopePairRole::TargetPre &&
               MatchesHint(p0_hint, draw, fragment_pgm_hash);
    }

    [[nodiscard]] constexpr bool MatchesTail(
        const PpTerminalScopeDrawSelector& draw, u64 fragment_pgm_hash) const noexcept {
        return frame_open && phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
               role == PpDepthPredecessorCrossScopePairRole::Tail &&
               MatchesHint(p1_hint, draw, fragment_pgm_hash);
    }

    [[nodiscard]] constexpr bool MatchesTargetRepeat(
        const PpTerminalScopeDrawSelector& draw, u64 fragment_pgm_hash) const noexcept {
        return !HasFailure() && frame_open &&
               phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
               role == PpDepthPredecessorCrossScopePairRole::Tail &&
               expected_target_repeat_count != 0 && !tail_pre_open && !anchor_seen &&
               MatchesHint(p0_hint, draw, fragment_pgm_hash);
    }

    [[nodiscard]] constexpr const PpDepthPredecessorHint* PeekArmedTailHint() const noexcept {
        return pair_ready && p1_hint.Valid() ? &p1_hint : nullptr;
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult
    ObserveLearningTargetPre(
        const PpDepthPredecessorCrossScopeLearningObservation& observation) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (!frame_open || phase != PpDepthPredecessorCrossScopePairPhase::Learning ||
            role != PpDepthPredecessorCrossScopePairRole::TargetPre) {
            Fail(PpDepthPredecessorCrossScopePairDecision::MissingTargetPre);
            return MakeResult();
        }
        if (target_pre_seen) {
            Fail(PpDepthPredecessorCrossScopePairDecision::DuplicateTargetPre);
            return MakeResult();
        }
        const auto classified =
            ClassifyPpDepthPredecessorCrossScopeLearningBoundary(observation);
        if (!classified.exact) {
            Fail(PpDepthPredecessorCrossScopePairDecision::BoundaryMismatch);
            return MakeResult();
        }
        if (observation.target_intent.hint != staged_p0_hint) {
            Fail(PpDepthPredecessorCrossScopePairDecision::TargetMismatch);
            return MakeResult();
        }
        const auto& source = observation.source_current;
        if (source.writer.hint == staged_p0_hint || source.writer.hint == staged_p1_hint ||
            source.writer.hint == staged_anchor_hint) {
            Fail(PpDepthPredecessorCrossScopePairDecision::SourceMismatch);
            return MakeResult();
        }
        staged_source = source;
        target_intent = observation.target_intent;
        target_pre_seen = true;
        target_pre_count = 1;
        boundary_exact = true;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult
    ObserveSourcePostComplete(const PpDepthPredecessorEndpointStamp& endpoint,
                              const PpDepthPredecessorCandidateStamp& candidate) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (!frame_open || phase != PpDepthPredecessorCrossScopePairPhase::Armed) {
            Fail(PpDepthPredecessorCrossScopePairDecision::MissingSourcePost);
            return MakeResult();
        }
        if (role != PpDepthPredecessorCrossScopePairRole::SourcePost || source_post_count != 0) {
            Fail(source_post_count != 0
                     ? PpDepthPredecessorCrossScopePairDecision::DuplicateSourcePost
                     : PpDepthPredecessorCrossScopePairDecision::MissingSourcePost);
            return MakeResult();
        }
        const bool endpoint_exact =
            endpoint.Valid() && endpoint.writer.hint == source_hint &&
            endpoint.writer.link == learned_source.writer.link &&
            endpoint.writer.range == learned_source.writer.range &&
            endpoint.writer.backing_generation == learned_source.writer.backing_generation;
        if (!endpoint_exact) {
            Fail(PpDepthPredecessorCrossScopePairDecision::SourceMismatch);
            return MakeResult();
        }
        if (!candidate.valid || candidate.generation != generation ||
            candidate.candidate_index >= capacity ||
            candidate.candidate_index >= PpUpstreamFeedbackPrePostPlan::MaxCandidates ||
            candidate.recorded_plane_mask != 0x1u || candidate.writer != endpoint.writer) {
            Fail(PpDepthPredecessorCrossScopePairDecision::CandidateMismatch);
            return MakeResult();
        }
        source_post = endpoint;
        source_candidate = candidate;
        source_post_count = 1;
        role = PpDepthPredecessorCrossScopePairRole::TargetPre;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult ObserveTargetPre(
        const PpDepthPredecessorCrossScopeBoundaryObservation& observation) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (!frame_open || phase != PpDepthPredecessorCrossScopePairPhase::Armed) {
            Fail(PpDepthPredecessorCrossScopePairDecision::MissingTargetPre);
            return MakeResult();
        }
        if (role == PpDepthPredecessorCrossScopePairRole::SourcePost) {
            Fail(PpDepthPredecessorCrossScopePairDecision::MissingSourcePost);
            return MakeResult();
        }
        if (target_pre_seen) {
            Fail(PpDepthPredecessorCrossScopePairDecision::DuplicateTargetPre);
            return MakeResult();
        }
        if (role != PpDepthPredecessorCrossScopePairRole::TargetPre || source_post_count != 1) {
            Fail(PpDepthPredecessorCrossScopePairDecision::MissingTargetPre);
            return MakeResult();
        }
        target_pre_attempted = true;
        const auto classified =
            ClassifyPpDepthPredecessorCrossScopeBoundary(observation, generation, capacity);
        if (!classified.candidate_exact) {
            Fail(PpDepthPredecessorCrossScopePairDecision::CandidateMismatch);
            return MakeResult();
        }
        if (!SameCandidate(observation.candidate, source_candidate)) {
            Fail(PpDepthPredecessorCrossScopePairDecision::CandidateMismatch);
            return MakeResult();
        }
        if (!classified.exact || observation.source_post != source_post) {
            Fail(PpDepthPredecessorCrossScopePairDecision::BoundaryMismatch);
            return MakeResult();
        }
        if (observation.target_intent.hint != p0_hint) {
            Fail(PpDepthPredecessorCrossScopePairDecision::TargetMismatch);
            return MakeResult();
        }
        target_intent = observation.target_intent;
        target_pre_seen = true;
        target_pre_count = 1;
        boundary_exact = true;
        return MakeResult();
    }

    [[nodiscard]] constexpr bool MatchesStoredSourcePost(
        const PpDepthPredecessorEndpointStamp& observed) const noexcept {
        return source_post_count == 1 && observed.Valid() && source_post.Valid() &&
               observed == source_post;
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult
    ObserveTargetPreComplete(const PpDepthPredecessorCandidateStamp& candidate) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (!frame_open || phase != PpDepthPredecessorCrossScopePairPhase::Armed ||
            role != PpDepthPredecessorCrossScopePairRole::TargetPre || !target_pre_seen) {
            Fail(PpDepthPredecessorCrossScopePairDecision::MissingTargetPre);
            return MakeResult();
        }
        if (target_pre_complete) {
            Fail(PpDepthPredecessorCrossScopePairDecision::DuplicateTargetPre);
            return MakeResult();
        }
        const bool exact = candidate.valid && source_candidate.valid &&
                           candidate.generation == generation &&
                           candidate.generation == source_candidate.generation &&
                           candidate.candidate_index == source_candidate.candidate_index &&
                           candidate.candidate_index < capacity &&
                           candidate.recorded_plane_mask == 0x3u &&
                           candidate.writer == source_candidate.writer;
        if (!exact) {
            Fail(PpDepthPredecessorCrossScopePairDecision::CandidateMismatch);
            return MakeResult();
        }
        completed_candidate = candidate;
        candidate_saved = true;
        target_pre_complete = true;
        role = PpDepthPredecessorCrossScopePairRole::Tail;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult ObserveTargetPost(
        const PpDepthPredecessorWriterStamp& target_post,
        const PpUpstreamDepthFirstWriterClassification& first_writer) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        const bool learning = phase == PpDepthPredecessorCrossScopePairPhase::Learning;
        const bool armed = phase == PpDepthPredecessorCrossScopePairPhase::Armed;
        if (!frame_open || !target_pre_seen || (!learning && !armed) ||
            (armed && (!target_pre_complete || role != PpDepthPredecessorCrossScopePairRole::Tail))) {
            Fail(PpDepthPredecessorCrossScopePairDecision::MissingTargetPre);
            return MakeResult();
        }
        if (target_post_seen) {
            Fail(PpDepthPredecessorCrossScopePairDecision::TargetFirstWriterMismatch);
            return MakeResult();
        }
        target_post_seen = true;
        const bool stamp_exact =
            target_post.Valid() && target_intent.Valid() &&
            target_post.hint == target_intent.hint && target_post.link == target_intent.link &&
            target_post.range == target_intent.range &&
            target_post.backing_generation == target_intent.backing_generation &&
            target_intent.mutation_epoch != std::numeric_limits<u64>::max() &&
            target_post.mutation_epoch == target_intent.mutation_epoch + 1;
        const bool first_exact = first_writer.post_exact &&
                                 first_writer.target_first_depth_writer &&
                                 first_writer.target_scope_load &&
                                 !first_writer.target_scope_clear;
        if (!stamp_exact || !first_exact) {
            Fail(PpDepthPredecessorCrossScopePairDecision::TargetFirstWriterMismatch);
            return MakeResult();
        }
        p0_post = target_post;
        target_first_writer_exact = true;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult ObserveTargetPost(
        const PpDepthPredecessorEndpointStamp& target_post,
        const PpUpstreamDepthFirstWriterClassification& first_writer,
        FinalGuestSurfaceReadbackSlotPool::Token retained_slot_token) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (phase != PpDepthPredecessorCrossScopePairPhase::Armed ||
            !target_post.Valid() || !retained_slot_token) {
            Fail(PpDepthPredecessorCrossScopePairDecision::TargetFirstWriterMismatch);
            return MakeResult();
        }
        const auto result = ObserveTargetPost(target_post.writer, first_writer);
        if (!HasFailure()) {
            p0_post_endpoint = target_post;
            target_repeat_retained_slot = retained_slot_token;
        }
        return result;
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopeTargetRepeatRetainedState
    SnapshotTargetRepeatRetainedState() const noexcept {
        return {
            .candidate = completed_candidate,
            .latest_p0_post = p0_post_endpoint,
            .retained_slot_token = target_repeat_retained_slot,
            .expected_count = expected_target_repeat_count,
            .attempt_count = target_repeat_attempt_count,
            .completion_count = target_repeat_completion_count,
            .pre_open = target_repeat_pre_open,
        };
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult
    ObserveTargetRepeatPreMetadata(
        const PpDepthPredecessorCrossScopeTargetRepeatPreObservation& observation) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (!frame_open || phase != PpDepthPredecessorCrossScopePairPhase::Armed ||
            role != PpDepthPredecessorCrossScopePairRole::Tail ||
            !target_first_writer_exact || !candidate_saved || tail_pre_open || anchor_seen) {
            FailTargetRepeat(PpDepthPredecessorCrossScopePairDecision::TargetRepeatMismatch);
            return MakeResult();
        }
        if (target_repeat_pre_open) {
            FailTargetRepeat(
                PpDepthPredecessorCrossScopePairDecision::DuplicateTargetRepeatPre);
            return MakeResult();
        }
        if (target_repeat_attempt_count == std::numeric_limits<u32>::max()) {
            FailTargetRepeat(PpDepthPredecessorCrossScopePairDecision::TargetRepeatMismatch);
            return MakeResult();
        }
        ++target_repeat_attempt_count;
        const bool budget_exact = expected_target_repeat_count != 0 &&
                                  target_repeat_attempt_count <=
                                      expected_target_repeat_count &&
                                  target_repeat_completion_count + 1 ==
                                      target_repeat_attempt_count;
        const bool scope_exact =
            observation.current_endpoint.writer.mutation_epoch !=
                std::numeric_limits<u64>::max() &&
            observation.current_endpoint.guest_rendering_serial !=
                std::numeric_limits<u64>::max() &&
            observation.live_guest_rendering_serial ==
                observation.current_endpoint.guest_rendering_serial + 1 &&
            observation.live_host_rendering_instance_id != 0 &&
            observation.live_host_rendering_instance_id !=
                observation.current_endpoint.host_rendering_instance_id &&
            observation.live_rendering_active && observation.live_regular_origin &&
            !observation.live_diagnostic_origin;
        const bool intent_exact =
            observation.intent.Valid() && observation.intent.hint == p0_hint &&
            observation.intent.link == observation.current_endpoint.writer.link &&
            observation.intent.range == observation.current_endpoint.writer.range &&
            observation.intent.backing_generation ==
                observation.current_endpoint.writer.backing_generation &&
            observation.intent.mutation_epoch ==
                observation.current_endpoint.writer.mutation_epoch;
        const bool slot_exact =
            observation.slot.present && observation.slot.generation == generation &&
            observation.slot.candidate_index == completed_candidate.candidate_index &&
            observation.retained_slot_token && target_repeat_retained_slot &&
            observation.retained_slot_token == target_repeat_retained_slot;
        const bool exact =
            budget_exact && p0_post_endpoint.Valid() &&
            observation.stored_latest == p0_post_endpoint &&
            observation.current_endpoint == p0_post_endpoint && intent_exact &&
            SameCandidate(observation.candidate, completed_candidate) && slot_exact &&
            scope_exact && observation.configured_effective_depth_write &&
            observation.canonical_draw && !observation.cross_open_collision;
        if (!exact) {
            FailTargetRepeat(PpDepthPredecessorCrossScopePairDecision::TargetRepeatMismatch);
            return MakeResult();
        }
        target_repeat_intent = observation.intent;
        target_repeat_live_guest = observation.live_guest_rendering_serial;
        target_repeat_live_host = observation.live_host_rendering_instance_id;
        target_repeat_pre_open = true;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult
    ObserveTargetRepeatPostMetadata(
        const PpDepthPredecessorCrossScopeTargetRepeatPostObservation& observation) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (!target_repeat_pre_open) {
            FailTargetRepeat(target_repeat_completion_count != 0
                                 ? PpDepthPredecessorCrossScopePairDecision::
                                       DuplicateTargetRepeatPost
                                 : PpDepthPredecessorCrossScopePairDecision::
                                       MissingTargetRepeatPost);
            return MakeResult();
        }
        const bool first_exact = observation.first_writer.post_exact &&
                                 observation.first_writer.target_first_depth_writer &&
                                 observation.first_writer.target_scope_load &&
                                 !observation.first_writer.target_scope_clear;
        const bool endpoint_exact =
            observation.post_endpoint.Valid() && target_repeat_intent.Valid() &&
            observation.post_endpoint.writer.hint == target_repeat_intent.hint &&
            observation.post_endpoint.writer.link == target_repeat_intent.link &&
            observation.post_endpoint.writer.range == target_repeat_intent.range &&
            observation.post_endpoint.writer.backing_generation ==
                target_repeat_intent.backing_generation &&
            target_repeat_intent.mutation_epoch != std::numeric_limits<u64>::max() &&
            observation.post_endpoint.writer.mutation_epoch ==
                target_repeat_intent.mutation_epoch + 1 &&
            observation.post_endpoint.guest_rendering_serial ==
                target_repeat_live_guest &&
            observation.post_endpoint.host_rendering_instance_id ==
                target_repeat_live_host &&
            !observation.post_endpoint.after_diagnostic_resume;
        const bool slot_exact =
            observation.slot.present && observation.slot.generation == generation &&
            observation.slot.candidate_index == completed_candidate.candidate_index &&
            observation.retained_slot_token && target_repeat_retained_slot &&
            observation.retained_slot_token == target_repeat_retained_slot;
        const bool exact =
            frame_open && phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
            role == PpDepthPredecessorCrossScopePairRole::Tail && endpoint_exact &&
            SameCandidate(observation.candidate, completed_candidate) && slot_exact &&
            observation.configured_effective_depth_write && observation.regular_origin &&
            !observation.diagnostic_origin && first_exact &&
            target_repeat_completion_count < expected_target_repeat_count;
        if (!exact) {
            target_repeat_pre_open = false;
            FailTargetRepeat(PpDepthPredecessorCrossScopePairDecision::TargetRepeatMismatch);
            return MakeResult();
        }
        p0_post_endpoint = observation.post_endpoint;
        p0_post = observation.post_endpoint.writer;
        ++target_repeat_completion_count;
        target_repeat_pre_open = false;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult
    ObserveLearningChainComplete(const PpDepthPredecessorTwoHopResult& selected) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (!frame_open || phase != PpDepthPredecessorCrossScopePairPhase::Learning ||
            !target_pre_seen || !target_post_seen || !boundary_exact ||
            !target_first_writer_exact || !IsExactMode3Selected(selected)) {
            Fail(PpDepthPredecessorCrossScopePairDecision::Mode3ProofMismatch);
            return MakeResult();
        }
        staged_commit = true;
        last_decision = PpDepthPredecessorCrossScopePairDecision::Committed;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult
    ObserveLearningRootBoundChainComplete(
        const PpDepthPredecessorTwoHopResult& selected,
        const PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness&
            selected_latest,
        const PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot&
            runtime_selection,
        const PpDepthPredecessorCrossScopeLearningLogicalCandidateSummary&
            logical_candidates,
        const PpDepthPredecessorCrossScopeLearningTransitionCandidateSummary&
            transition_candidates) noexcept;

    [[nodiscard]] constexpr
    PpDepthPredecessorCrossScopeMode3SelectedProof
    ClassifyMode3SelectedProof(
        const PpDepthPredecessorTwoHopResult& selected) const noexcept {
        return ClassifyPpDepthPredecessorCrossScopeMode3SelectedProof(
            staged_mode3_begin, selected);
    }

    [[nodiscard]] constexpr
    PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence
    ClassifyMode3SelectedLatestBinding(
        const PpDepthPredecessorTwoHopResult& selected,
        const PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness&
            selected_latest,
        const PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot&
            runtime_selection,
        const PpDepthPredecessorCrossScopeLearningLogicalCandidateSummary&
            logical_candidates,
        const PpDepthPredecessorCrossScopeLearningTransitionCandidateSummary&
            transition_candidates) const noexcept;

    [[nodiscard]] constexpr
    PpDepthPredecessorCrossScopeMode3RootBindingEvidence
    ClassifyMode3RootBinding(
        const PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness&
            selected_latest,
        const PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence&
            selected_latest_evidence) const noexcept;

    [[nodiscard]] constexpr
    PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidence
    SnapshotFirstTailPreP0Drift(
        const PpDepthPredecessorEndpointStamp& stored_target_post,
        const PpDepthPredecessorEndpointStamp& current_endpoint,
        const PpDepthPredecessorWriterIntent& intent,
        u64 live_guest_rendering_serial, u64 live_host_rendering_instance_id,
        bool live_rendering_active, bool live_regular_origin,
        bool live_diagnostic_origin) const noexcept {
        const bool lifecycle_ready =
            !HasFailure() && frame_open &&
            phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
            role == PpDepthPredecessorCrossScopePairRole::Tail &&
            target_pre_complete && target_post_seen && target_first_writer_exact &&
            candidate_saved && stored_target_post.Valid() &&
            stored_target_post.writer == p0_post && !tail_pre_open && !anchor_seen;
        const bool first_tail_pre = lifecycle_ready && observed_tail_occurrences == 0;
        return ClassifyPpDepthPredecessorCrossScopeFirstTailPreP0Drift({
            .stored_p0_post = stored_target_post,
            .current_endpoint = current_endpoint,
            .tail_intent = intent,
            .expected_p0_hint = p0_hint,
            .expected_p1_hint = p1_hint,
            .live_guest_rendering_serial = live_guest_rendering_serial,
            .live_host_rendering_instance_id = live_host_rendering_instance_id,
            .live_rendering_active = live_rendering_active,
            .live_regular_origin = live_regular_origin,
            .live_diagnostic_origin = live_diagnostic_origin,
            .lifecycle_ready = lifecycle_ready,
            .first_tail_pre = first_tail_pre,
        });
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult ObserveTailPre(
        const PpDepthPredecessorWriterIntent& intent,
        const PpDepthPredecessorWriterStamp& current_writer,
        FinalGuestSurfaceReadbackSlotPool::Token retained_slot_token = {}) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        const bool repeat_budget_exact =
            target_repeat_attempt_count == expected_target_repeat_count &&
            target_repeat_completion_count == expected_target_repeat_count &&
            !target_repeat_pre_open;
        const bool retained_slot_exact =
            expected_target_repeat_count == 0 ||
            (retained_slot_token && target_repeat_retained_slot &&
             retained_slot_token == target_repeat_retained_slot);
        if (!repeat_budget_exact) {
            FailTargetRepeat(
                PpDepthPredecessorCrossScopePairDecision::TargetRepeatMismatch);
            return MakeResult();
        }
        if (!retained_slot_exact) {
            FailTargetRepeat(PpDepthPredecessorCrossScopePairDecision::TailMismatch);
            return MakeResult();
        }
        if (!frame_open || phase != PpDepthPredecessorCrossScopePairPhase::Armed ||
            role != PpDepthPredecessorCrossScopePairRole::Tail ||
            !target_first_writer_exact || !candidate_saved) {
            Fail(PpDepthPredecessorCrossScopePairDecision::TailMismatch);
            return MakeResult();
        }
        if (anchor_seen || observed_tail_occurrences >= expected_tail_occurrences) {
            Fail(PpDepthPredecessorCrossScopePairDecision::TailMismatch);
            return MakeResult();
        }
        if (tail_pre_open) {
            Fail(PpDepthPredecessorCrossScopePairDecision::MissingTailPost);
            return MakeResult();
        }
        const auto& expected_prior = observed_tail_occurrences == 0 ? p0_post : tail_post;
        const bool exact = intent.Valid() && intent.hint == p1_hint && current_writer.Valid() &&
                           current_writer == expected_prior &&
                           current_writer.link == intent.link &&
                           current_writer.range == intent.range &&
                           current_writer.backing_generation == intent.backing_generation &&
                           current_writer.mutation_epoch == intent.mutation_epoch;
        if (!exact) {
            Fail(PpDepthPredecessorCrossScopePairDecision::TailMismatch);
            return MakeResult();
        }
        tail_intent = intent;
        tail_pre_open = true;
        tail_exact = true;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult ObserveTailPost(
        const PpDepthPredecessorWriterStamp& post) noexcept {
        if (HasFailure()) {
            return MakeResult();
        }
        if (!frame_open || phase != PpDepthPredecessorCrossScopePairPhase::Armed ||
            role != PpDepthPredecessorCrossScopePairRole::Tail) {
            Fail(PpDepthPredecessorCrossScopePairDecision::TailMismatch);
            return MakeResult();
        }
        if (!tail_pre_open) {
            Fail(observed_tail_occurrences != 0
                     ? PpDepthPredecessorCrossScopePairDecision::DuplicateTailPost
                     : PpDepthPredecessorCrossScopePairDecision::MissingTailPost);
            return MakeResult();
        }
        const bool exact =
            post.Valid() && tail_intent.Valid() && post.hint == tail_intent.hint &&
            post.link == tail_intent.link && post.range == tail_intent.range &&
            post.backing_generation == tail_intent.backing_generation &&
            tail_intent.mutation_epoch != std::numeric_limits<u64>::max() &&
            post.mutation_epoch == tail_intent.mutation_epoch + 1;
        if (!exact || observed_tail_occurrences == std::numeric_limits<u32>::max()) {
            Fail(PpDepthPredecessorCrossScopePairDecision::TailMismatch);
            return MakeResult();
        }
        tail_post = post;
        tail_pre_open = false;
        ++observed_tail_occurrences;
        tail_exact = true;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult ObserveAnchor(
        const PpDepthPredecessorHint& configured_anchor,
        const PpDepthPredecessorWriterStamp& current_writer) noexcept {
        if (anchor_seen) {
            Fail(PpDepthPredecessorCrossScopePairDecision::DuplicateAnchor);
            return MakeResult();
        }
        anchor_seen = true;
        if (HasFailure()) {
            return MakeResult();
        }
        if (!frame_open || phase != PpDepthPredecessorCrossScopePairPhase::Armed ||
            role != PpDepthPredecessorCrossScopePairRole::Tail || !candidate_saved ||
            !target_first_writer_exact || tail_pre_open || target_repeat_pre_open ||
            target_repeat_attempt_count != expected_target_repeat_count ||
            target_repeat_completion_count != expected_target_repeat_count ||
            observed_tail_occurrences != expected_tail_occurrences) {
            Fail(PpDepthPredecessorCrossScopePairDecision::EarlyAnchor);
            return MakeResult();
        }
        if (!configured_anchor.Valid() || configured_anchor != anchor_hint ||
            !current_writer.Valid() || current_writer != tail_post) {
            Fail(PpDepthPredecessorCrossScopePairDecision::AnchorMismatch);
            return MakeResult();
        }
        if (!completed_candidate.valid || completed_candidate.generation != generation ||
            completed_candidate.candidate_index >= capacity ||
            completed_candidate.recorded_plane_mask != 0x3u ||
            completed_candidate.writer != source_post.writer) {
            Fail(PpDepthPredecessorCrossScopePairDecision::CandidateMismatch);
            return MakeResult();
        }
        selected = true;
        selected_index = completed_candidate.candidate_index;
        last_decision = PpDepthPredecessorCrossScopePairDecision::Selected;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult EndFrame() noexcept {
        if (!frame_open) {
            return MakeResult();
        }
        if (source_restart_open) {
            if (!HasFailure()) {
                Fail(PpDepthPredecessorCrossScopePairDecision::MissingSourcePost);
            }
            source_restart_open = false;
            ClearActiveSource();
            ClearSourceRestartProof();
        }
        if (target_repeat_pre_open) {
            if (!HasFailure()) {
                FailTargetRepeat(
                    PpDepthPredecessorCrossScopePairDecision::MissingTargetRepeatPost);
            }
            target_repeat_pre_open = false;
        } else if (phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
                   !HasFailure() &&
                   (target_repeat_attempt_count != expected_target_repeat_count ||
                    target_repeat_completion_count != expected_target_repeat_count)) {
            FailTargetRepeat(PpDepthPredecessorCrossScopePairDecision::TargetRepeatMismatch);
        }
        if (!anchor_seen && phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
            !HasFailure()) {
            Fail(PpDepthPredecessorCrossScopePairDecision::MissingAnchor);
        }
        if (phase == PpDepthPredecessorCrossScopePairPhase::Learning) {
            if (last_decision == PpDepthPredecessorCrossScopePairDecision::Committed &&
                staged_commit) {
                source_hint = staged_source.writer.hint;
                learned_source = staged_source;
                p0_hint = staged_p0_hint;
                p1_hint = staged_p1_hint;
                anchor_hint = staged_anchor_hint;
                committed_expected_tail_occurrences = expected_tail_occurrences;
                committed_expected_target_repeat_count =
                    staged_expected_target_repeat_count;
                pair_ready = true;
            } else {
                source_hint = {};
                learned_source = {};
                p0_hint = {};
                p1_hint = {};
                anchor_hint = {};
                committed_expected_tail_occurrences = 0;
                committed_expected_target_repeat_count = 0;
                pair_ready = false;
            }
        }
        const auto result = MakeResult();
        frame_open = false;
        role = PpDepthPredecessorCrossScopePairRole::None;
        ClearCandidate();
        selected = false;
        return result;
    }

private:
    [[nodiscard]] constexpr bool IsExactMode3Begin(
        const PpDepthPredecessorTwoHopResult& result) const noexcept {
        return result.phase == PpDepthPredecessorTwoHopPhase::ArmedP2 &&
               result.decision == PpDepthPredecessorTwoHopDecision::Idle &&
               result.chain_kind ==
                   PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter &&
               result.occurrence_role == PpDepthPredecessorTwoHopOccurrenceRole::RawPrefix &&
               result.generation != 0 && result.expected_occurrences >= 2 &&
               result.expected_occurrences != std::numeric_limits<u32>::max() &&
                result.identity_collision ==
                    PpDepthPredecessorTwoHopIdentityCollision::None &&
                result.raw_armed && !result.metadata_only &&
                result.prefix_join_discriminator ==
                    PpDepthPredecessorPrefixJoinDiscriminator::NotAttempted;
    }

    [[nodiscard]] constexpr bool IsExactMode3Selected(
        const PpDepthPredecessorTwoHopResult& result) const noexcept {
        return ClassifyMode3SelectedProof(result).exact;
    }

    [[nodiscard]] static constexpr bool MatchesHint(
        const PpDepthPredecessorHint& hint, const PpTerminalScopeDrawSelector& draw,
        u64 fragment_pgm_hash) noexcept {
        return hint.Valid() &&
               PpDepthPredecessorHint{.draw = draw,
                                      .fragment_pgm_hash = fragment_pgm_hash,
                                      .valid = true} == hint;
    }

    [[nodiscard]] static constexpr bool SameCandidate(
        const PpDepthPredecessorCandidateStamp& left,
        const PpDepthPredecessorCandidateStamp& right) noexcept {
        return left.valid == right.valid && left.generation == right.generation &&
               left.candidate_index == right.candidate_index &&
               left.recorded_plane_mask == right.recorded_plane_mask &&
               left.writer == right.writer;
    }

    [[nodiscard]] constexpr bool HasFailure() const noexcept {
        return last_decision != PpDepthPredecessorCrossScopePairDecision::Idle &&
               last_decision != PpDepthPredecessorCrossScopePairDecision::Committed &&
               last_decision != PpDepthPredecessorCrossScopePairDecision::Selected;
    }

    constexpr void Fail(PpDepthPredecessorCrossScopePairDecision decision) noexcept {
        last_decision = decision;
        tail_exact = false;
        staged_commit = false;
        selected = false;
        ClearCandidate();
    }

    constexpr void FailTargetRepeat(
        PpDepthPredecessorCrossScopePairDecision decision) noexcept {
        last_decision = decision;
        tail_exact = false;
        staged_commit = false;
        selected = false;
        target_repeat_pre_open = false;
    }

    constexpr void ClearCandidate() noexcept {
        source_candidate = {};
        completed_candidate = {};
        selected_index = 0;
        candidate_saved = false;
    }

    constexpr void ClearActiveSource() noexcept {
        source_post = {};
        source_candidate = {};
        source_post_count = 0;
    }

    constexpr void ClearSourceRestartProof() noexcept {
        restart_prior_source = {};
        restart_prior_candidate = {};
        restart_prior_candidate_capacity = 0;
        restart_prior_resume_guest_rendering_serial = 0;
        restart_prior_resume_host_rendering_instance_id = 0;
        restart_prior_resume_rendering_active = false;
        restart_prior_resume_diagnostic_origin = false;
        restart_prior_resume_preserved_guest_serial = false;
        restart_prior_resume_valid = false;
    }

    constexpr void ResetFrame(PpDepthPredecessorCrossScopePairPhase next_phase,
                              PpDepthPredecessorCrossScopePairRole next_role) noexcept {
        phase = next_phase;
        role = next_role;
        last_decision = PpDepthPredecessorCrossScopePairDecision::Idle;
        staged_mode3_begin = {};
        staged_p0_hint = {};
        staged_p1_hint = {};
        staged_anchor_hint = {};
        staged_source = {};
        staged_expected_target_repeat_count = 0;
        source_post = {};
        target_intent = {};
        p0_post = {};
        p0_post_endpoint = {};
        tail_intent = {};
        tail_post = {};
        expected_tail_occurrences = pair_ready ? committed_expected_tail_occurrences : 0;
        expected_target_repeat_count =
            pair_ready ? committed_expected_target_repeat_count : 0;
        observed_tail_occurrences = 0;
        source_post_count = 0;
        source_restart_count = 0;
        target_pre_count = 0;
        target_repeat_attempt_count = 0;
        target_repeat_completion_count = 0;
        boundary_exact = false;
        target_first_writer_exact = false;
        tail_exact = false;
        target_pre_seen = false;
        target_pre_attempted = false;
        target_pre_complete = false;
        target_post_seen = false;
        tail_pre_open = false;
        target_repeat_pre_open = false;
        anchor_seen = false;
        staged_commit = false;
        selected = false;
        source_restart_open = false;
        target_repeat_retained_slot = {};
        target_repeat_intent = {};
        target_repeat_live_guest = 0;
        target_repeat_live_host = 0;
        ClearSourceRestartProof();
        ClearCandidate();
    }

    [[nodiscard]] constexpr PpDepthPredecessorCrossScopePairResult MakeResult() const noexcept {
        return {
            .phase = phase,
            .role = role,
            .decision = last_decision,
            .generation = phase == PpDepthPredecessorCrossScopePairPhase::Armed ? generation : 0,
            .candidate_index = candidate_saved || selected ? completed_candidate.candidate_index : 0,
            .expected_tail_occurrences = expected_tail_occurrences,
            .observed_tail_occurrences = observed_tail_occurrences,
            .source_post_count = source_post_count,
            .source_restart_count = source_restart_count,
            .target_pre_count = target_pre_count,
            .expected_target_repeat_count = expected_target_repeat_count,
            .target_repeat_attempt_count = target_repeat_attempt_count,
            .target_repeat_completion_count = target_repeat_completion_count,
            .boundary_exact = boundary_exact,
            .target_first_writer_exact = target_first_writer_exact,
            .tail_exact = tail_exact,
            .candidate_saved = candidate_saved,
            .source_restart_open = source_restart_open,
            .target_repeat_pre_open = target_repeat_pre_open,
            .committed = staged_commit,
            .select = selected,
        };
    }

    u32 capacity{};
    u64 generation{};
    PpDepthPredecessorCrossScopePairPhase phase{
        PpDepthPredecessorCrossScopePairPhase::Learning};
    PpDepthPredecessorCrossScopePairRole role{PpDepthPredecessorCrossScopePairRole::None};
    PpDepthPredecessorCrossScopePairDecision last_decision{
        PpDepthPredecessorCrossScopePairDecision::Idle};
    PpDepthPredecessorHint source_hint{};
    PpDepthPredecessorHint p0_hint{};
    PpDepthPredecessorHint p1_hint{};
    PpDepthPredecessorHint anchor_hint{};
    PpDepthPredecessorEndpointStamp learned_source{};
    u32 committed_expected_tail_occurrences{};
    u32 committed_expected_target_repeat_count{};
    u32 staged_expected_target_repeat_count{};
    PpDepthPredecessorTwoHopResult staged_mode3_begin{};
    PpDepthPredecessorHint staged_p0_hint{};
    PpDepthPredecessorHint staged_p1_hint{};
    PpDepthPredecessorHint staged_anchor_hint{};
    PpDepthPredecessorEndpointStamp staged_source{};
    PpDepthPredecessorEndpointStamp source_post{};
    PpDepthPredecessorWriterIntent target_intent{};
    PpDepthPredecessorWriterStamp p0_post{};
    PpDepthPredecessorEndpointStamp p0_post_endpoint{};
    PpDepthPredecessorWriterIntent tail_intent{};
    PpDepthPredecessorWriterStamp tail_post{};
    PpDepthPredecessorCandidateStamp source_candidate{};
    PpDepthPredecessorCandidateStamp completed_candidate{};
    u32 selected_index{};
    u32 expected_tail_occurrences{};
    u32 observed_tail_occurrences{};
    u32 source_post_count{};
    u32 source_restart_count{};
    u32 target_pre_count{};
    u32 expected_target_repeat_count{};
    u32 target_repeat_attempt_count{};
    u32 target_repeat_completion_count{};
    bool pair_ready{};
    bool frame_open{};
    bool boundary_exact{};
    bool target_first_writer_exact{};
    bool tail_exact{};
    bool target_pre_seen{};
    bool target_pre_attempted{};
    bool target_pre_complete{};
    bool target_post_seen{};
    bool tail_pre_open{};
    bool target_repeat_pre_open{};
    bool anchor_seen{};
    bool candidate_saved{};
    bool staged_commit{};
    bool selected{};
    bool source_restart_open{};
    FinalGuestSurfaceReadbackSlotPool::Token target_repeat_retained_slot{};
    PpDepthPredecessorWriterIntent target_repeat_intent{};
    u64 target_repeat_live_guest{};
    u64 target_repeat_live_host{};
    PpDepthPredecessorEndpointStamp restart_prior_source{};
    PpDepthPredecessorCandidateStamp restart_prior_candidate{};
    u32 restart_prior_candidate_capacity{};
    u64 restart_prior_resume_guest_rendering_serial{};
    u64 restart_prior_resume_host_rendering_instance_id{};
    bool restart_prior_resume_rendering_active{};
    bool restart_prior_resume_diagnostic_origin{};
    bool restart_prior_resume_preserved_guest_serial{};
    bool restart_prior_resume_valid{};
};

enum class PpDepthPredecessorCrossScopeRuntimeDispatch : u32 {
    Ignore,
    LearningTargetPre,
    SourcePreReserve,
    SourcePreRestart,
    SourcePostPlane0,
    SourcePostRestartPlane0,
    TargetPrePlane1,
    TargetPostFirstWriter,
    TargetRepeatPreMetadata,
    TargetRepeatPostMetadata,
    TailPreMetadata,
    TailPostMetadata,
    Anchor,
    Reject,
};

struct PpDepthPredecessorCrossScopeRuntimeDispatchInput {
    PpDepthPredecessorCrossScopePairPhase phase{
        PpDepthPredecessorCrossScopePairPhase::Learning};
    PpDepthPredecessorCrossScopePairRole role{
        PpDepthPredecessorCrossScopePairRole::None};
    bool pre_draw{};
    bool configured_anchor{};
    bool matches_source_post{};
    bool matches_target_pre{};
    bool matches_target_repeat{};
    bool matches_tail{};
    bool source_reservation_open{};
    bool target_post_open{};
    bool target_repeat_post_open{};
    bool tail_post_open{};
    bool cross_open_collision{};
    bool source_restart_allowed{};
    bool source_restart_open{};
    bool target_pre_attempted{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeRuntimeDispatch
ClassifyPpDepthPredecessorCrossScopeRuntimeDispatch(
    const PpDepthPredecessorCrossScopeRuntimeDispatchInput& input) noexcept {
    using Dispatch = PpDepthPredecessorCrossScopeRuntimeDispatch;
    if (input.cross_open_collision) {
        return Dispatch::Reject;
    }
    if (input.configured_anchor) {
        return input.pre_draw &&
                       input.phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
                       input.role == PpDepthPredecessorCrossScopePairRole::Tail &&
                       !input.source_reservation_open && !input.target_post_open &&
                       !input.target_repeat_post_open && !input.tail_post_open &&
                       !input.matches_target_repeat
                   ? Dispatch::Anchor
                   : Dispatch::Reject;
    }

    const u32 match_count = static_cast<u32>(input.matches_source_post) +
                            static_cast<u32>(input.matches_target_pre) +
                            static_cast<u32>(input.matches_target_repeat) +
                            static_cast<u32>(input.matches_tail);
    if (match_count > 1) {
        return Dispatch::Reject;
    }
    if (input.source_reservation_open) {
        if (input.pre_draw || !input.matches_source_post ||
            input.phase != PpDepthPredecessorCrossScopePairPhase::Armed ||
            input.role != PpDepthPredecessorCrossScopePairRole::SourcePost ||
            input.target_post_open || input.target_repeat_post_open ||
            input.tail_post_open) {
            return Dispatch::Reject;
        }
        return input.source_restart_open ? Dispatch::SourcePostRestartPlane0
                                         : Dispatch::SourcePostPlane0;
    }
    if (input.target_post_open) {
        const bool role_exact =
            (input.phase == PpDepthPredecessorCrossScopePairPhase::Learning &&
             input.role == PpDepthPredecessorCrossScopePairRole::TargetPre) ||
            (input.phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
             input.role == PpDepthPredecessorCrossScopePairRole::Tail);
        return !input.pre_draw && input.matches_target_pre && role_exact &&
                       !input.target_repeat_post_open && !input.tail_post_open
                   ? Dispatch::TargetPostFirstWriter
                   : Dispatch::Reject;
    }
    if (input.target_repeat_post_open) {
        return !input.pre_draw && input.matches_target_repeat &&
                       input.phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
                       input.role == PpDepthPredecessorCrossScopePairRole::Tail &&
                       !input.source_reservation_open && !input.target_post_open &&
                       !input.tail_post_open
                   ? Dispatch::TargetRepeatPostMetadata
                   : Dispatch::Reject;
    }
    if (input.tail_post_open) {
        return !input.pre_draw && input.matches_tail &&
                       input.phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
                       input.role == PpDepthPredecessorCrossScopePairRole::Tail &&
                       !input.source_reservation_open && !input.target_post_open &&
                       !input.target_repeat_post_open
                   ? Dispatch::TailPostMetadata
                   : Dispatch::Reject;
    }
    if (input.matches_source_post) {
        if (!input.pre_draw ||
            input.phase != PpDepthPredecessorCrossScopePairPhase::Armed) {
            return Dispatch::Reject;
        }
        if (input.role == PpDepthPredecessorCrossScopePairRole::SourcePost) {
            return Dispatch::SourcePreReserve;
        }
        return input.role == PpDepthPredecessorCrossScopePairRole::TargetPre &&
                       input.source_restart_allowed && !input.target_pre_attempted
                   ? Dispatch::SourcePreRestart
                   : Dispatch::Reject;
    }
    if (input.matches_target_pre) {
        if (!input.pre_draw) {
            return Dispatch::Reject;
        }
        if (input.phase == PpDepthPredecessorCrossScopePairPhase::Learning &&
            input.role == PpDepthPredecessorCrossScopePairRole::TargetPre) {
            return Dispatch::LearningTargetPre;
        }
        return input.phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
                       input.role == PpDepthPredecessorCrossScopePairRole::TargetPre
                   ? Dispatch::TargetPrePlane1
                   : Dispatch::Reject;
    }
    if (input.matches_target_repeat) {
        return input.pre_draw &&
                       input.phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
                       input.role == PpDepthPredecessorCrossScopePairRole::Tail &&
                       !input.source_reservation_open && !input.target_post_open &&
                       !input.target_repeat_post_open && !input.tail_post_open
                   ? Dispatch::TargetRepeatPreMetadata
                   : Dispatch::Reject;
    }
    if (input.matches_tail) {
        return input.pre_draw &&
                       input.phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
                       input.role == PpDepthPredecessorCrossScopePairRole::Tail
                   ? Dispatch::TailPreMetadata
                   : Dispatch::Reject;
    }
    return Dispatch::Ignore;
}

enum class PpDepthPredecessorCrossScopeEndpointPlane : u32 {
    None,
    SourcePostPlane0,
    TargetPrePlane1,
};

struct PpDepthPredecessorCrossScopeRuntimeAction {
    PpDepthPredecessorCrossScopeEndpointPlane endpoint_plane{
        PpDepthPredecessorCrossScopeEndpointPlane::None};
    u32 plane_index{};
    bool frame_preview{};
    bool frame_complete{};
    bool direct_record{};
    bool record_plane{};
    bool reject_restart{};
    bool require_after_producer_mark{};
    bool require_before_draw{};
    bool classify_first_writer{};
    bool metadata_only{};
    bool tag_depth_candidate{};
    bool select_candidate{};
    bool reuse_retained_slot{};
    bool first_copy{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeRuntimeAction
PlanPpDepthPredecessorCrossScopeRuntimeAction(
    PpDepthPredecessorCrossScopeRuntimeDispatch dispatch,
    bool lifecycle_select) noexcept {
    using Dispatch = PpDepthPredecessorCrossScopeRuntimeDispatch;
    using Plane = PpDepthPredecessorCrossScopeEndpointPlane;
    switch (dispatch) {
    case Dispatch::LearningTargetPre:
        return {.require_before_draw = true, .metadata_only = true};
    case Dispatch::SourcePreReserve:
        return {.frame_preview = true, .reject_restart = true,
                .require_before_draw = true};
    case Dispatch::SourcePreRestart:
        return {.frame_preview = true,
                .require_before_draw = true,
                .reuse_retained_slot = true};
    case Dispatch::SourcePostPlane0:
        return {.endpoint_plane = Plane::SourcePostPlane0,
                .plane_index = 0,
                .frame_complete = true,
                .record_plane = true,
                .require_after_producer_mark = true,
                .first_copy = true};
    case Dispatch::SourcePostRestartPlane0:
        return {.endpoint_plane = Plane::SourcePostPlane0,
                .plane_index = 0,
                .frame_complete = true,
                .record_plane = true,
                .require_after_producer_mark = true,
                .reuse_retained_slot = true,
                .first_copy = false};
    case Dispatch::TargetPrePlane1:
        return {.endpoint_plane = Plane::TargetPrePlane1,
                .plane_index = 1,
                .direct_record = true,
                .record_plane = true,
                .require_before_draw = true};
    case Dispatch::TargetPostFirstWriter:
        return {.classify_first_writer = true, .metadata_only = true};
    case Dispatch::TargetRepeatPreMetadata:
        return {.require_before_draw = true, .metadata_only = true};
    case Dispatch::TargetRepeatPostMetadata:
        return {.classify_first_writer = true, .metadata_only = true};
    case Dispatch::TailPreMetadata:
        return {.require_before_draw = true, .metadata_only = true};
    case Dispatch::TailPostMetadata:
        return {.metadata_only = true};
    case Dispatch::Anchor:
        return {.metadata_only = true, .select_candidate = lifecycle_select};
    default:
        return {};
    }
}

[[nodiscard]] constexpr bool IsExactPpDepthPredecessorCrossScopeSourceReservation(
    const PpUpstreamFeedbackPrePostRegistryAction& action) noexcept {
    return action.capture && !action.restart &&
           action.status == FinalGuestSurfaceStatus::Complete && !action.loss.Any() &&
           action.candidate_index < PpUpstreamFeedbackPrePostPlan::MaxCandidates;
}

[[nodiscard]] constexpr bool IsExactPpDepthPredecessorCrossScopeSourceCompletion(
    const PpUpstreamFeedbackPrePostRegistryAction& action,
    u32 reserved_candidate_index) noexcept {
    return IsExactPpDepthPredecessorCrossScopeSourceReservation(action) &&
           reserved_candidate_index < PpUpstreamFeedbackPrePostPlan::MaxCandidates &&
           action.candidate_index == reserved_candidate_index;
}

[[nodiscard]] constexpr bool IsExactPpDepthPredecessorCrossScopeSourceRestart(
    const PpUpstreamFeedbackPrePostRegistryAction& action,
    u32 reserved_candidate_index, u32 recorded_plane_mask_before_preview,
    u32 recorded_plane_mask_after_preview, bool retained_slot_present) noexcept {
    return IsExactPpDepthPredecessorRawPrefixRestart(action) &&
           reserved_candidate_index < PpUpstreamFeedbackPrePostPlan::MaxCandidates &&
           action.candidate_index == reserved_candidate_index &&
           recorded_plane_mask_before_preview == 0x1u &&
           recorded_plane_mask_after_preview == 0 && retained_slot_present;
}

struct PpDepthPredecessorCrossScopeSourcePlaneCopyPlan {
    u32 plane_index{};
    bool reuse_retained_slot{};
    bool first_copy{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeSourcePlaneCopyPlan
PlanPpDepthPredecessorCrossScopeSourcePlaneCopy(bool restart) noexcept {
    return {
        .plane_index = 0,
        .reuse_retained_slot = restart,
        .first_copy = !restart,
    };
}

struct PpDepthPredecessorCrossScopeRenderingCheckpoint {
    u64 guest_rendering_serial{};
    u64 host_rendering_instance_id{};
    bool rendering_active{};
    bool regular_origin{};
    bool diagnostic_origin{};

    bool operator==(const PpDepthPredecessorCrossScopeRenderingCheckpoint&) const = default;
};

struct PpDepthPredecessorCrossScopeRuntimeBoundaryCheckpoints {
    PpDepthPredecessorCrossScopeRenderingCheckpoint source_pre_transit{};
    PpDepthPredecessorCrossScopeRenderingCheckpoint pre_external_begin{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeRenderingCheckpoint
SelectPpDepthPredecessorCrossScopeLearningSourceCheckpoint(
    const PpDepthPredecessorCrossScopeRuntimeBoundaryCheckpoints& checkpoints) noexcept {
    return checkpoints.source_pre_transit;
}

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeRenderingCheckpoint
SelectPpDepthPredecessorCrossScopeArmedResumeCheckpoint(
    const PpDepthPredecessorCrossScopeRuntimeBoundaryCheckpoints& checkpoints) noexcept {
    return checkpoints.pre_external_begin;
}

enum class PpDepthPredecessorCrossScopeSourceLineageKind : u32 {
    None,
    SameHost,
    DiagnosticDescendant,
};

struct PpDepthPredecessorCrossScopeSourceLineageObservation {
    u64 source_guest_rendering_serial{};
    u64 source_host_rendering_instance_id{};
    bool source_after_diagnostic_resume{};
    PpDepthPredecessorCrossScopeRenderingCheckpoint checkpoint{};
};

struct PpDepthPredecessorCrossScopeSourceLineageClassification {
    PpDepthPredecessorCrossScopeSourceLineageKind kind{
        PpDepthPredecessorCrossScopeSourceLineageKind::None};
    bool source_identity_nonzero{};
    bool checkpoint_identity_nonzero{};
    bool hosts_below_max{};
    bool guest_exact{};
    bool origin_one_hot{};
    bool same_host{};
    bool source_origin_exact{};
    bool strictly_newer_host{};
    bool diagnostic_descendant_exact{};
    bool exact{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeSourceLineageClassification
ClassifyPpDepthPredecessorCrossScopeSourceLineage(
    const PpDepthPredecessorCrossScopeSourceLineageObservation& observation) noexcept {
    const auto& checkpoint = observation.checkpoint;
    const bool source_identity_nonzero =
        observation.source_guest_rendering_serial != 0 &&
        observation.source_host_rendering_instance_id != 0;
    const bool checkpoint_identity_nonzero =
        checkpoint.guest_rendering_serial != 0 &&
        checkpoint.host_rendering_instance_id != 0;
    const bool hosts_below_max =
        observation.source_host_rendering_instance_id !=
            std::numeric_limits<u64>::max() &&
        checkpoint.host_rendering_instance_id !=
            std::numeric_limits<u64>::max();
    const bool guest_exact =
        checkpoint.guest_rendering_serial ==
        observation.source_guest_rendering_serial;
    const bool origin_one_hot =
        checkpoint.regular_origin != checkpoint.diagnostic_origin;
    const bool same_host = checkpoint.host_rendering_instance_id ==
                           observation.source_host_rendering_instance_id;
    const bool source_origin_exact =
        origin_one_hot && checkpoint.diagnostic_origin ==
                              observation.source_after_diagnostic_resume;
    const bool strictly_newer_host =
        checkpoint.host_rendering_instance_id >
        observation.source_host_rendering_instance_id;
    const bool common_exact = source_identity_nonzero &&
                              checkpoint_identity_nonzero &&
                              hosts_below_max && guest_exact && origin_one_hot;
    const bool same_host_exact =
        common_exact && same_host && source_origin_exact;
    const bool diagnostic_descendant_exact =
        common_exact && strictly_newer_host && checkpoint.diagnostic_origin;
    const auto kind = same_host_exact
                          ? PpDepthPredecessorCrossScopeSourceLineageKind::SameHost
                      : diagnostic_descendant_exact
                          ? PpDepthPredecessorCrossScopeSourceLineageKind::
                                DiagnosticDescendant
                          : PpDepthPredecessorCrossScopeSourceLineageKind::None;
    return {
        .kind = kind,
        .source_identity_nonzero = source_identity_nonzero,
        .checkpoint_identity_nonzero = checkpoint_identity_nonzero,
        .hosts_below_max = hosts_below_max,
        .guest_exact = guest_exact,
        .origin_one_hot = origin_one_hot,
        .same_host = same_host,
        .source_origin_exact = source_origin_exact,
        .strictly_newer_host = strictly_newer_host,
        .diagnostic_descendant_exact = diagnostic_descendant_exact,
        .exact = kind != PpDepthPredecessorCrossScopeSourceLineageKind::None,
    };
}

struct PpDepthPredecessorCrossScopeLearningTargetPreRuntimeObservation {
    PpDepthPredecessorEndpointStamp source_current{};
    PpDepthPredecessorWriterIntent target_intent{};
    PpUpstreamDepthFirstWriterPreObservation predecessor{};
    VideoCore::ImageColorScopePrivateLink raw_current_link{};
    VideoCore::SubresourceRange raw_current_range{};
    u64 raw_current_backing_generation{};
    u64 raw_current_mutation_epoch{};
    bool raw_boundary_active{};
    bool raw_boundary_tracked{};
    u64 boundary_before_guest_rendering_serial{};
    u64 boundary_before_host_rendering_instance_id{};
    u64 boundary_after_guest_rendering_serial{};
    u64 boundary_after_host_rendering_instance_id{};
    bool boundary_before_rendering_active{};
    bool boundary_before_regular_origin{};
    bool boundary_before_diagnostic_origin{};
    bool boundary_after_rendering_active{};
    bool boundary_after_regular_origin{};
    bool boundary_after_diagnostic_origin{};
    u64 live_target_guest_rendering_serial{};
    u64 live_target_host_rendering_instance_id{};
    bool live_target_rendering_active{};
    bool live_target_regular_origin{};
    bool live_target_diagnostic_origin{};
    bool source_outside_target_guest_scope{};
    bool after_begin_rendering{};
    bool target_depth_load{};
    bool target_depth_clear{};
    bool target_stencil_clear{};
    bool depth_aspect{};
    bool canonical_attachment{};
};

struct PpDepthPredecessorCrossScopeLearningJoinDiscriminator {
    bool source_structural_valid{};
    bool source_after_diagnostic_resume{};
    bool target_intent_valid{};
    bool source_target_link_exact{};
    bool source_target_range_exact{};
    bool source_target_backing_exact{};
    bool source_target_mutation_exact{};
    bool raw_current_matches_source{};
};

struct PpDepthPredecessorCrossScopeLearningTargetPreRuntimeBoundary {
    PpDepthPredecessorCrossScopeLearningObservation observation{};
    PpDepthPredecessorCrossScopeLearningJoinDiscriminator join_discriminator{};
    bool raw_boundary_exact{};
    bool boundary_before_guest_exact{};
    bool boundary_before_host_exact{};
    bool boundary_before_rendering_active{};
    bool boundary_before_regular_origin_exact{};
    bool boundary_before_source_origin_exact{};
    bool boundary_before_diagnostic_descendant_exact{};
    bool boundary_before_source_lineage_exact{};
    bool raw_join_exact{};
    bool live_target_exact{};
    bool exact{};
};

struct PpDepthPredecessorCrossScopeLearningEvidence {
    bool raw_boundary_exact{};
    bool boundary_before_guest_exact{};
    bool boundary_before_host_exact{};
    bool boundary_before_rendering_active{};
    bool boundary_before_regular_origin_exact{};
    bool boundary_before_source_origin_exact{};
    bool boundary_before_diagnostic_descendant_exact{};
    bool boundary_before_source_lineage_exact{};
    bool raw_join_exact{};
    bool live_target_exact{};
    bool predecessor_exact{};
    bool join_exact{};
    bool regular_cross_scope_exact{};
    bool load_exact{};
    PpDepthPredecessorCrossScopeLearningJoinDiscriminator join_discriminator{};
};

struct PpDepthPredecessorCrossScopeLearningTransitionCandidateSummary {
    u32 count{};
    bool count_overflow{};
    bool unique_valid{};
    bool boundary_before_source_origin_exact{};
    bool boundary_before_diagnostic_descendant_exact{};
    bool boundary_before_source_lineage_exact{};
    PpDepthPredecessorCrossScopeLearningJoinDiscriminator join_discriminator{};
};

[[nodiscard]] constexpr bool IsPpDepthPredecessorCrossScopeLearningTransitionCandidate(
    const PpDepthPredecessorCrossScopeLearningEvidence& evidence) noexcept {
    return evidence.live_target_exact && evidence.predecessor_exact &&
           evidence.regular_cross_scope_exact && evidence.load_exact;
}

[[nodiscard]] constexpr
PpDepthPredecessorCrossScopeLearningTransitionCandidateSummary
AccumulatePpDepthPredecessorCrossScopeLearningTransitionCandidate(
    const PpDepthPredecessorCrossScopeLearningTransitionCandidateSummary& previous,
    const PpDepthPredecessorCrossScopeLearningEvidence& evidence) noexcept {
    if (!IsPpDepthPredecessorCrossScopeLearningTransitionCandidate(evidence)) {
        return previous;
    }

    auto result = previous;
    if (result.count == std::numeric_limits<u32>::max()) {
        result.count_overflow = true;
    } else {
        ++result.count;
    }
    if (previous.count == 0 && !previous.count_overflow) {
        result.unique_valid = true;
        result.boundary_before_source_origin_exact =
            evidence.boundary_before_source_origin_exact;
        result.boundary_before_diagnostic_descendant_exact =
            evidence.boundary_before_diagnostic_descendant_exact;
        result.boundary_before_source_lineage_exact =
            evidence.boundary_before_source_lineage_exact;
        result.join_discriminator = evidence.join_discriminator;
    } else {
        result.unique_valid = false;
        result.boundary_before_source_origin_exact = false;
        result.boundary_before_diagnostic_descendant_exact = false;
        result.boundary_before_source_lineage_exact = false;
        result.join_discriminator = {};
    }
    return result;
}

struct PpDepthPredecessorCrossScopeLearningLogicalCandidateSummary {
    u32 count{};
    bool count_overflow{};
    bool unique_valid{};
    bool raw_boundary_exact{};
    bool boundary_before_guest_exact{};
    bool boundary_before_host_exact{};
    bool boundary_before_rendering_active{};
    bool boundary_before_regular_origin_exact{};
    bool boundary_before_source_origin_exact{};
    bool boundary_before_diagnostic_descendant_exact{};
    bool boundary_before_source_lineage_exact{};
};

[[nodiscard]] constexpr bool
IsPpDepthPredecessorCrossScopeLearningLogicalCandidate(
    const PpDepthPredecessorCrossScopeLearningEvidence& evidence) noexcept {
    return evidence.raw_join_exact && evidence.live_target_exact &&
           evidence.predecessor_exact && evidence.join_exact &&
           evidence.regular_cross_scope_exact && evidence.load_exact;
}

[[nodiscard]] constexpr
PpDepthPredecessorCrossScopeLearningLogicalCandidateSummary
AccumulatePpDepthPredecessorCrossScopeLearningLogicalCandidate(
    const PpDepthPredecessorCrossScopeLearningLogicalCandidateSummary& previous,
    const PpDepthPredecessorCrossScopeLearningEvidence& evidence) noexcept {
    if (!IsPpDepthPredecessorCrossScopeLearningLogicalCandidate(evidence)) {
        return previous;
    }

    auto result = previous;
    if (result.count == std::numeric_limits<u32>::max()) {
        result.count_overflow = true;
    } else {
        ++result.count;
    }
    if (previous.count == 0 && !previous.count_overflow) {
        result.unique_valid = true;
        result.raw_boundary_exact = evidence.raw_boundary_exact;
        result.boundary_before_guest_exact =
            evidence.boundary_before_guest_exact;
        result.boundary_before_host_exact =
            evidence.boundary_before_host_exact;
        result.boundary_before_rendering_active =
            evidence.boundary_before_rendering_active;
        result.boundary_before_regular_origin_exact =
            evidence.boundary_before_regular_origin_exact;
        result.boundary_before_source_origin_exact =
            evidence.boundary_before_source_origin_exact;
        result.boundary_before_diagnostic_descendant_exact =
            evidence.boundary_before_diagnostic_descendant_exact;
        result.boundary_before_source_lineage_exact =
            evidence.boundary_before_source_lineage_exact;
    } else {
        result.unique_valid = false;
        result.raw_boundary_exact = false;
        result.boundary_before_guest_exact = false;
        result.boundary_before_host_exact = false;
        result.boundary_before_rendering_active = false;
        result.boundary_before_regular_origin_exact = false;
        result.boundary_before_source_origin_exact = false;
        result.boundary_before_diagnostic_descendant_exact = false;
        result.boundary_before_source_lineage_exact = false;
    }
    return result;
}

enum class PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshotEvent : u32 {
    Reset,
    NonSelection,
    Selection,
    FrameInvalidated,
};

struct PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot {
    PpDepthPredecessorCandidateStamp selected_candidate{};
    u64 frame_generation{};
    u64 expected_generation{};
    u32 frame_candidate_index{};
    u32 live_recorded_plane_mask{};
    u32 restart_count{};
    u32 attempt_count{};
    PpDepthPredecessorTwoHopChainKind chain_kind{
        PpDepthPredecessorTwoHopChainKind::Distinct};
    bool generation_exact{};
    bool candidate_index_exact{};
    bool candidate_mask_exact{};
    bool restart_count_bounded{};
    bool chain_exact{};
    bool select_exact{};
    bool slot_exact{};
    bool frame_exact{};
    bool final_binding_exact{};
    bool valid{};

    bool operator==(
        const PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot&) const = default;
};

struct PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshotObservation {
    PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshotEvent event{
        PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshotEvent::Reset};
    PpDepthPredecessorCandidateStamp selected_candidate{};
    u64 frame_generation{};
    u64 expected_generation{};
    u32 frame_candidate_index{};
    u32 live_recorded_plane_mask{};
    u32 restart_count{};
    PpDepthPredecessorTwoHopChainKind chain_kind{
        PpDepthPredecessorTwoHopChainKind::Distinct};
    bool select_attempted{};
    bool select_succeeded{};
    bool slot_present{};
    bool frame_complete{};
    bool frame_loss_free{};
    bool final_binding_point{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot
AccumulatePpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot(
    const PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot& current,
    const PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshotObservation&
        observation) noexcept {
    using Event =
        PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshotEvent;
    switch (observation.event) {
    case Event::Reset:
    case Event::FrameInvalidated:
        return {};
    case Event::NonSelection:
        return current;
    case Event::Selection:
        break;
    default:
        return {};
    }

    const u32 next_attempt_count =
        current.attempt_count == std::numeric_limits<u32>::max()
            ? std::numeric_limits<u32>::max()
            : current.attempt_count + 1;
    if (current.attempt_count != 0) {
        return {.attempt_count = next_attempt_count};
    }

    const bool candidate_valid =
        observation.selected_candidate.valid &&
        observation.selected_candidate.writer.Valid();
    const bool generation_exact =
        candidate_valid && observation.frame_generation != 0 &&
        observation.expected_generation != 0 &&
        observation.selected_candidate.generation ==
            observation.frame_generation &&
        observation.frame_generation == observation.expected_generation;
    const bool candidate_index_exact =
        candidate_valid && observation.selected_candidate.candidate_index ==
                               observation.frame_candidate_index;
    const bool candidate_mask_exact =
        observation.selected_candidate.recorded_plane_mask == 0x3u &&
        observation.live_recorded_plane_mask == 0x3u;
    const bool restart_count_bounded =
        observation.restart_count != 0 &&
        observation.restart_count != std::numeric_limits<u32>::max();
    const bool chain_exact =
        observation.chain_kind ==
        PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter;
    const bool select_exact =
        observation.select_attempted && observation.select_succeeded;
    const bool slot_exact = observation.slot_present;
    const bool frame_exact =
        observation.frame_complete && observation.frame_loss_free;
    const bool final_binding_exact = observation.final_binding_point;
    const bool valid =
        generation_exact && candidate_index_exact && candidate_mask_exact &&
        restart_count_bounded && chain_exact && select_exact && slot_exact &&
        frame_exact && final_binding_exact;
    return {
        .selected_candidate = valid ? observation.selected_candidate
                                    : PpDepthPredecessorCandidateStamp{},
        .frame_generation = observation.frame_generation,
        .expected_generation = observation.expected_generation,
        .frame_candidate_index = observation.frame_candidate_index,
        .live_recorded_plane_mask = observation.live_recorded_plane_mask,
        .restart_count = observation.restart_count,
        .attempt_count = next_attempt_count,
        .chain_kind = observation.chain_kind,
        .generation_exact = generation_exact,
        .candidate_index_exact = candidate_index_exact,
        .candidate_mask_exact = candidate_mask_exact,
        .restart_count_bounded = restart_count_bounded,
        .chain_exact = chain_exact,
        .select_exact = select_exact,
        .slot_exact = slot_exact,
        .frame_exact = frame_exact,
        .final_binding_exact = final_binding_exact,
        .valid = valid,
    };
}

[[nodiscard]] constexpr
PpDepthPredecessorCrossScopeMode3SelectedLatestBindingReason
SanitizePpDepthPredecessorCrossScopeMode3SelectedLatestBindingReason(
    PpDepthPredecessorCrossScopeMode3SelectedLatestBindingReason reason) noexcept {
    using Reason =
        PpDepthPredecessorCrossScopeMode3SelectedLatestBindingReason;
    switch (reason) {
    case Reason::NotAttempted:
    case Reason::Exact:
    case Reason::SelectedCoreMismatch:
    case Reason::RawRollingMismatch:
    case Reason::RestartCountMismatch:
    case Reason::LatestWitnessMismatch:
    case Reason::RuntimeSelectionMismatch:
    case Reason::GenerationMismatch:
    case Reason::CandidateMismatch:
    case Reason::TargetPostMismatch:
    case Reason::LogicalSummaryMismatch:
    case Reason::TransitionSummaryMismatch:
    case Reason::LearningStateMismatch:
        return reason;
    }
    return Reason::NotAttempted;
}

struct PpDepthPredecessorCrossScopeMode3SelectedLatestBindingObservation {
    PpDepthPredecessorTwoHopResult begin{};
    PpDepthPredecessorTwoHopResult selected{};
    PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness selected_latest{};
    PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot runtime_selection{};
    PpDepthPredecessorHint learning_p0_hint{};
    PpDepthPredecessorWriterIntent learning_target_intent{};
    PpDepthPredecessorWriterStamp learning_p0_post{};
    PpDepthPredecessorCrossScopeLearningLogicalCandidateSummary logical_candidates{};
    PpDepthPredecessorCrossScopeLearningTransitionCandidateSummary transition_candidates{};
    u64 expected_generation{};
    u32 candidate_capacity{};
    PpDepthPredecessorCrossScopePairPhase lifecycle_phase{
        PpDepthPredecessorCrossScopePairPhase::Learning};
    PpDepthPredecessorCrossScopePairRole lifecycle_role{
        PpDepthPredecessorCrossScopePairRole::None};
    PpDepthPredecessorCrossScopePairDecision lifecycle_decision{
        PpDepthPredecessorCrossScopePairDecision::Idle};
    bool lifecycle_frame_open{};
    bool lifecycle_boundary_exact{};
    bool lifecycle_target_first_writer_exact{};
    bool lifecycle_target_pre_seen{};
    bool lifecycle_target_post_seen{};
};

struct PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence {
    bool attempted{};
    PpDepthPredecessorCrossScopeMode3SelectedLatestBindingReason reason{
        PpDepthPredecessorCrossScopeMode3SelectedLatestBindingReason::NotAttempted};
    u32 raw_occurrences{};
    u32 restart_attempt_count{};
    u32 restart_completion_count{};
    bool raw_rolling_bounded{};
    bool selected_core_without_raw_exact{};
    bool rolling_counts_exact{};
    bool latest_witness_exact{};
    bool runtime_selection_exact{};
    bool generation_exact{};
    bool candidate_index_exact{};
    bool candidate_mask_exact{};
    bool actual_selected_latest_exact{};
    bool latest_p0_post_exact{};
    bool intent_p0_post_exact{};
    bool logical_summary_exact{};
    bool transition_summary_exact{};
    bool lifecycle_learning_exact{};
    bool exact{};

    bool operator==(
        const PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence&) const = default;
};

[[nodiscard]] constexpr
PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence
ClassifyPpDepthPredecessorCrossScopeMode3SelectedLatestBinding(
    const PpDepthPredecessorCrossScopeMode3SelectedLatestBindingObservation&
        observation) noexcept {
    using Reason =
        PpDepthPredecessorCrossScopeMode3SelectedLatestBindingReason;
    const auto& begin = observation.begin;
    const auto& selected = observation.selected;
    const auto& latest = observation.selected_latest;
    const auto& runtime = observation.runtime_selection;

    const bool begin_exact =
        begin.phase == PpDepthPredecessorTwoHopPhase::ArmedP2 &&
        begin.decision == PpDepthPredecessorTwoHopDecision::Idle &&
        begin.chain_kind ==
            PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter &&
        begin.occurrence_role ==
            PpDepthPredecessorTwoHopOccurrenceRole::RawPrefix &&
        begin.generation != 0 && begin.expected_occurrences >= 2 &&
        begin.expected_occurrences != std::numeric_limits<u32>::max() &&
        begin.identity_collision ==
            PpDepthPredecessorTwoHopIdentityCollision::None &&
        begin.raw_armed && !begin.metadata_only;
    const bool selected_core_without_raw_exact =
        begin_exact && selected.phase == PpDepthPredecessorTwoHopPhase::ArmedP2 &&
        selected.decision == PpDepthPredecessorTwoHopDecision::Selected &&
        selected.chain_kind ==
            PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter &&
        selected.occurrence_role ==
            PpDepthPredecessorTwoHopOccurrenceRole::P1Tail &&
        selected.generation == begin.generation &&
        selected.expected_occurrences == begin.expected_occurrences &&
        selected.observed_occurrences == selected.expected_occurrences &&
        selected.bridge_occurrences == selected.expected_occurrences &&
        selected.identity_collision ==
            PpDepthPredecessorTwoHopIdentityCollision::None &&
        selected.anchor_exact && selected.p1_intent && selected.p2_exact &&
        selected.p1_post_exact && selected.prefix_join_exact &&
        selected.prefix_join_discriminator ==
            PpDepthPredecessorPrefixJoinDiscriminator::Exact &&
        selected.raw_armed && !selected.metadata_only &&
        selected.candidate_saved && selected.select;
    const bool raw_rolling_bounded =
        selected.raw_occurrences > 1 &&
        selected.raw_occurrences != std::numeric_limits<u32>::max();
    const u32 expected_restarts =
        raw_rolling_bounded ? selected.raw_occurrences - 1 : 0;
    const bool rolling_counts_exact =
        raw_rolling_bounded && !latest.restart_open && !latest.count_overflow &&
        latest.restart_attempt_count == expected_restarts &&
        latest.restart_completion_count == expected_restarts;

    const bool capacity_exact =
        observation.candidate_capacity != 0 &&
        observation.candidate_capacity <=
            PpUpstreamFeedbackPrePostPlan::MaxCandidates &&
        latest.candidate_capacity == observation.candidate_capacity;
    const bool latest_witness_structural_exact =
        latest.valid && latest.chain_exact && !latest.restart_open &&
        !latest.count_overflow && capacity_exact && latest.latest_prefix.Valid() &&
        latest.latest_candidate.valid && latest.latest_candidate.writer.Valid();
    const bool latest_candidate_writer_exact =
        latest.latest_candidate.writer == latest.latest_prefix;
    const bool latest_witness_exact =
        latest_witness_structural_exact && latest_candidate_writer_exact;

    const bool runtime_selection_exact =
        runtime.valid && runtime.attempt_count == 1 && runtime.generation_exact &&
        runtime.candidate_index_exact && runtime.candidate_mask_exact &&
        runtime.restart_count_bounded && runtime.chain_exact && runtime.select_exact &&
        runtime.slot_exact && runtime.frame_exact && runtime.final_binding_exact &&
        runtime.restart_count == expected_restarts &&
        runtime.chain_kind ==
            PpDepthPredecessorTwoHopChainKind::DistinctPrefixBeforeSameWriter;
    const bool generation_exact =
        observation.expected_generation != 0 &&
        begin.generation == observation.expected_generation &&
        selected.generation == observation.expected_generation &&
        runtime.frame_generation == observation.expected_generation &&
        runtime.expected_generation == observation.expected_generation;
    const bool candidate_generation_exact =
        latest.latest_candidate.generation == observation.expected_generation &&
        runtime.selected_candidate.generation == observation.expected_generation;
    const bool candidate_index_exact =
        capacity_exact && selected.candidate_index < observation.candidate_capacity &&
        latest.latest_candidate.candidate_index == selected.candidate_index &&
        runtime.frame_candidate_index == selected.candidate_index &&
        runtime.selected_candidate.candidate_index == selected.candidate_index;
    const bool candidate_mask_exact =
        latest.latest_candidate.recorded_plane_mask == 0x3u &&
        runtime.selected_candidate.recorded_plane_mask == 0x3u &&
        runtime.live_recorded_plane_mask == 0x3u;
    const bool actual_selected_latest_exact =
        runtime.selected_candidate == latest.latest_candidate;

    const bool latest_p0_post_exact =
        observation.learning_p0_post.Valid() &&
        observation.learning_p0_post == latest.latest_prefix;
    const auto& intent = observation.learning_target_intent;
    const auto& post = observation.learning_p0_post;
    const bool intent_p0_post_exact =
        observation.learning_p0_hint.Valid() && intent.Valid() && post.Valid() &&
        intent.hint == observation.learning_p0_hint &&
        post.hint == observation.learning_p0_hint && intent.link == post.link &&
        intent.range == post.range &&
        intent.backing_generation == post.backing_generation &&
        intent.mutation_epoch != std::numeric_limits<u64>::max() &&
        post.mutation_epoch == intent.mutation_epoch + 1;
    const bool logical_summary_exact =
        observation.logical_candidates.count == 1 &&
        !observation.logical_candidates.count_overflow &&
        observation.logical_candidates.unique_valid;
    const bool transition_summary_exact =
        observation.transition_candidates.count == 1 &&
        !observation.transition_candidates.count_overflow &&
        observation.transition_candidates.unique_valid;
    const bool lifecycle_learning_exact =
        observation.lifecycle_phase ==
            PpDepthPredecessorCrossScopePairPhase::Learning &&
        observation.lifecycle_role ==
            PpDepthPredecessorCrossScopePairRole::TargetPre &&
        observation.lifecycle_decision ==
            PpDepthPredecessorCrossScopePairDecision::Idle &&
        observation.lifecycle_frame_open && observation.lifecycle_boundary_exact &&
        observation.lifecycle_target_first_writer_exact &&
        observation.lifecycle_target_pre_seen &&
        observation.lifecycle_target_post_seen;

    const bool candidate_exact =
        candidate_generation_exact && candidate_index_exact && candidate_mask_exact &&
        latest_candidate_writer_exact && actual_selected_latest_exact;
    const bool exact =
        selected_core_without_raw_exact && raw_rolling_bounded &&
        rolling_counts_exact && latest_witness_exact && runtime_selection_exact &&
        generation_exact && candidate_exact && latest_p0_post_exact &&
        intent_p0_post_exact && logical_summary_exact && transition_summary_exact &&
        lifecycle_learning_exact;

    Reason reason = Reason::Exact;
    if (!selected_core_without_raw_exact) {
        reason = Reason::SelectedCoreMismatch;
    } else if (!raw_rolling_bounded) {
        reason = Reason::RawRollingMismatch;
    } else if (!rolling_counts_exact) {
        reason = Reason::RestartCountMismatch;
    } else if (!latest_witness_structural_exact) {
        reason = Reason::LatestWitnessMismatch;
    } else if (!runtime_selection_exact) {
        reason = Reason::RuntimeSelectionMismatch;
    } else if (!generation_exact) {
        reason = Reason::GenerationMismatch;
    } else if (!candidate_exact) {
        reason = Reason::CandidateMismatch;
    } else if (!latest_p0_post_exact || !intent_p0_post_exact) {
        reason = Reason::TargetPostMismatch;
    } else if (!logical_summary_exact) {
        reason = Reason::LogicalSummaryMismatch;
    } else if (!transition_summary_exact) {
        reason = Reason::TransitionSummaryMismatch;
    } else if (!lifecycle_learning_exact) {
        reason = Reason::LearningStateMismatch;
    }

    return {
        .attempted = true,
        .reason = reason,
        .raw_occurrences = selected.raw_occurrences,
        .restart_attempt_count = latest.restart_attempt_count,
        .restart_completion_count = latest.restart_completion_count,
        .raw_rolling_bounded = raw_rolling_bounded,
        .selected_core_without_raw_exact = selected_core_without_raw_exact,
        .rolling_counts_exact = rolling_counts_exact,
        .latest_witness_exact = latest_witness_exact,
        .runtime_selection_exact = runtime_selection_exact,
        .generation_exact = generation_exact,
        .candidate_index_exact = candidate_index_exact,
        .candidate_mask_exact = candidate_mask_exact,
        .actual_selected_latest_exact = actual_selected_latest_exact,
        .latest_p0_post_exact = latest_p0_post_exact,
        .intent_p0_post_exact = intent_p0_post_exact,
        .logical_summary_exact = logical_summary_exact,
        .transition_summary_exact = transition_summary_exact,
        .lifecycle_learning_exact = lifecycle_learning_exact,
        .exact = exact,
    };
}

enum class PpDepthPredecessorCrossScopeMode3RootBindingReason : u32 {
    NotAttempted,
    Exact,
    RootWitnessMismatch,
    CrossStateAbsent,
    CrossRootMismatch,
    ChainMismatch,
    LatestSelectionMismatch,
    LifecycleMismatch,
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeMode3RootBindingReason
SanitizePpDepthPredecessorCrossScopeMode3RootBindingReason(
    PpDepthPredecessorCrossScopeMode3RootBindingReason reason) noexcept {
    using Reason = PpDepthPredecessorCrossScopeMode3RootBindingReason;
    switch (reason) {
    case Reason::NotAttempted:
    case Reason::Exact:
    case Reason::RootWitnessMismatch:
    case Reason::CrossStateAbsent:
    case Reason::CrossRootMismatch:
    case Reason::ChainMismatch:
    case Reason::LatestSelectionMismatch:
    case Reason::LifecycleMismatch:
        return reason;
    }
    return Reason::NotAttempted;
}

struct PpDepthPredecessorCrossScopeMode3RootBindingObservation {
    PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness selected_latest{};
    PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence
        selected_latest_evidence{};
    PpDepthPredecessorEndpointStamp stored_source{};
    PpDepthPredecessorWriterIntent target_intent{};
    PpDepthPredecessorWriterStamp p0_post{};
    PpDepthPredecessorCrossScopePairPhase lifecycle_phase{
        PpDepthPredecessorCrossScopePairPhase::Learning};
    PpDepthPredecessorCrossScopePairRole lifecycle_role{
        PpDepthPredecessorCrossScopePairRole::None};
    PpDepthPredecessorCrossScopePairDecision lifecycle_decision{
        PpDepthPredecessorCrossScopePairDecision::Idle};
    bool lifecycle_frame_open{};
    bool lifecycle_boundary_exact{};
    bool lifecycle_target_first_writer_exact{};
    bool lifecycle_target_pre_seen{};
    bool lifecycle_target_post_seen{};
};

struct PpDepthPredecessorCrossScopeMode3RootBindingEvidence {
    bool attempted{};
    PpDepthPredecessorCrossScopeMode3RootBindingReason reason{
        PpDepthPredecessorCrossScopeMode3RootBindingReason::NotAttempted};
    bool root_witness_exact{};
    bool cross_state_present{};
    bool cross_root_exact{};
    bool chain_exact{};
    bool latest_selection_exact{};
    bool lifecycle_exact{};
    bool exact{};

    bool operator==(
        const PpDepthPredecessorCrossScopeMode3RootBindingEvidence&) const = default;
};

[[nodiscard]] constexpr
PpDepthPredecessorCrossScopeArmedP0RepeatProbeConfiguration
MakePpDepthPredecessorCrossScopeArmedP0RepeatProbeConfiguration(
    const PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness& witness,
    const PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence& selected,
    const PpDepthPredecessorCrossScopeMode3RootBindingEvidence& root) noexcept {
    return {
        .selected_latest = witness,
        .selected_latest_evidence = {
            .attempted = selected.attempted,
            .reason = selected.reason,
            .raw_occurrences = selected.raw_occurrences,
            .restart_attempt_count = selected.restart_attempt_count,
            .restart_completion_count = selected.restart_completion_count,
            .raw_rolling_bounded = selected.raw_rolling_bounded,
            .selected_core_without_raw_exact =
                selected.selected_core_without_raw_exact,
            .rolling_counts_exact = selected.rolling_counts_exact,
            .latest_witness_exact = selected.latest_witness_exact,
            .runtime_selection_exact = selected.runtime_selection_exact,
            .generation_exact = selected.generation_exact,
            .candidate_index_exact = selected.candidate_index_exact,
            .candidate_mask_exact = selected.candidate_mask_exact,
            .actual_selected_latest_exact =
                selected.actual_selected_latest_exact,
            .latest_p0_post_exact = selected.latest_p0_post_exact,
            .intent_p0_post_exact = selected.intent_p0_post_exact,
            .logical_summary_exact = selected.logical_summary_exact,
            .transition_summary_exact = selected.transition_summary_exact,
            .lifecycle_learning_exact = selected.lifecycle_learning_exact,
            .exact = selected.exact,
        },
        .root_binding_evidence = {
            .attempted = root.attempted,
            .reason = root.reason,
            .root_witness_exact = root.root_witness_exact,
            .cross_state_present = root.cross_state_present,
            .cross_root_exact = root.cross_root_exact,
            .chain_exact = root.chain_exact,
            .latest_selection_exact = root.latest_selection_exact,
            .lifecycle_exact = root.lifecycle_exact,
            .exact = root.exact,
        },
        .expected_p0_hint = witness.root_prefix.hint,
    };
}

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeMode3RootBindingEvidence
ClassifyPpDepthPredecessorCrossScopeMode3RootBinding(
    const PpDepthPredecessorCrossScopeMode3RootBindingObservation& observation) noexcept {
    using Reason = PpDepthPredecessorCrossScopeMode3RootBindingReason;
    const auto& latest = observation.selected_latest;
    const auto& selected_latest = observation.selected_latest_evidence;
    const bool attempted =
        selected_latest.raw_occurrences > 1 &&
        selected_latest.raw_occurrences != std::numeric_limits<u32>::max();
    if (!attempted) {
        return {};
    }

    const bool root_witness_exact =
        latest.root_valid && latest.root_prefix.Valid() &&
        latest.root_candidate.valid && latest.root_candidate.writer.Valid() &&
        latest.root_candidate.writer == latest.root_prefix &&
        latest.candidate_capacity != 0 &&
        latest.candidate_capacity <= PpUpstreamFeedbackPrePostPlan::MaxCandidates &&
        latest.root_candidate.generation != 0 &&
        latest.root_candidate.candidate_index < latest.candidate_capacity &&
        latest.root_candidate.recorded_plane_mask == 0x3u;
    const bool cross_state_present =
        observation.stored_source.Valid() && observation.target_intent.Valid() &&
        observation.p0_post.Valid();
    const auto& root = latest.root_prefix;
    const auto& intent = observation.target_intent;
    const auto& source = observation.stored_source.writer;
    const bool cross_root_exact =
        root_witness_exact && cross_state_present && observation.p0_post == root &&
        source.link == intent.link && source.range == intent.range &&
        source.backing_generation == intent.backing_generation &&
        source.mutation_epoch == intent.mutation_epoch &&
        intent.hint == root.hint && intent.link == root.link &&
        intent.range == root.range &&
        intent.backing_generation == root.backing_generation &&
        intent.mutation_epoch != std::numeric_limits<u64>::max() &&
        root.mutation_epoch == intent.mutation_epoch + 1;
    const u32 expected_restarts = selected_latest.raw_occurrences - 1;
    const bool chain_exact =
        root_witness_exact && latest.valid && latest.chain_exact &&
        latest.root_to_latest_chain_exact && !latest.restart_open &&
        !latest.count_overflow &&
        latest.restart_attempt_count == expected_restarts &&
        latest.restart_completion_count == expected_restarts &&
        latest.latest_prefix.Valid() && latest.latest_candidate.valid &&
        latest.latest_candidate.writer == latest.latest_prefix &&
        latest.latest_candidate.generation == latest.root_candidate.generation &&
        latest.latest_candidate.candidate_index ==
            latest.root_candidate.candidate_index &&
        latest.latest_candidate.recorded_plane_mask == 0x3u &&
        root.hint == latest.latest_prefix.hint &&
        root.link == latest.latest_prefix.link &&
        root.range == latest.latest_prefix.range &&
        root.backing_generation == latest.latest_prefix.backing_generation &&
        root.mutation_epoch < latest.latest_prefix.mutation_epoch;
    const bool latest_selection_exact =
        selected_latest.attempted && selected_latest.raw_rolling_bounded &&
        selected_latest.restart_attempt_count == expected_restarts &&
        selected_latest.restart_completion_count == expected_restarts &&
        selected_latest.restart_attempt_count == latest.restart_attempt_count &&
        selected_latest.restart_completion_count ==
            latest.restart_completion_count &&
        selected_latest.rolling_counts_exact &&
        selected_latest.latest_witness_exact &&
        selected_latest.runtime_selection_exact &&
        selected_latest.generation_exact &&
        selected_latest.candidate_index_exact &&
        selected_latest.candidate_mask_exact &&
        selected_latest.actual_selected_latest_exact &&
        selected_latest.reason ==
            PpDepthPredecessorCrossScopeMode3SelectedLatestBindingReason::
                TargetPostMismatch &&
        !selected_latest.latest_p0_post_exact &&
        selected_latest.intent_p0_post_exact &&
        selected_latest.logical_summary_exact &&
        selected_latest.transition_summary_exact &&
        selected_latest.lifecycle_learning_exact && !selected_latest.exact;
    const bool lifecycle_exact =
        observation.lifecycle_phase ==
            PpDepthPredecessorCrossScopePairPhase::Learning &&
        observation.lifecycle_role ==
            PpDepthPredecessorCrossScopePairRole::TargetPre &&
        observation.lifecycle_decision ==
            PpDepthPredecessorCrossScopePairDecision::Idle &&
        observation.lifecycle_frame_open &&
        observation.lifecycle_boundary_exact &&
        observation.lifecycle_target_first_writer_exact &&
        observation.lifecycle_target_pre_seen &&
        observation.lifecycle_target_post_seen;
    const bool exact = root_witness_exact && cross_state_present &&
                       cross_root_exact && chain_exact &&
                       latest_selection_exact && lifecycle_exact;

    Reason reason = Reason::Exact;
    if (!root_witness_exact) {
        reason = Reason::RootWitnessMismatch;
    } else if (!cross_state_present) {
        reason = Reason::CrossStateAbsent;
    } else if (!cross_root_exact) {
        reason = Reason::CrossRootMismatch;
    } else if (!chain_exact) {
        reason = Reason::ChainMismatch;
    } else if (!latest_selection_exact) {
        reason = Reason::LatestSelectionMismatch;
    } else if (!lifecycle_exact) {
        reason = Reason::LifecycleMismatch;
    }
    return {
        .attempted = true,
        .reason = reason,
        .root_witness_exact = root_witness_exact,
        .cross_state_present = cross_state_present,
        .cross_root_exact = cross_root_exact,
        .chain_exact = chain_exact,
        .latest_selection_exact = latest_selection_exact,
        .lifecycle_exact = lifecycle_exact,
        .exact = exact,
    };
}

constexpr PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence
PpDepthPredecessorCrossScopePairLifecycle::ClassifyMode3SelectedLatestBinding(
    const PpDepthPredecessorTwoHopResult& selected,
    const PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness&
        selected_latest,
    const PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot&
        runtime_selection,
    const PpDepthPredecessorCrossScopeLearningLogicalCandidateSummary&
        logical_candidates,
    const PpDepthPredecessorCrossScopeLearningTransitionCandidateSummary&
        transition_candidates) const noexcept {
    return ClassifyPpDepthPredecessorCrossScopeMode3SelectedLatestBinding({
        .begin = staged_mode3_begin,
        .selected = selected,
        .selected_latest = selected_latest,
        .runtime_selection = runtime_selection,
        .learning_p0_hint = staged_p0_hint,
        .learning_target_intent = target_intent,
        .learning_p0_post = p0_post,
        .logical_candidates = logical_candidates,
        .transition_candidates = transition_candidates,
        .expected_generation = selected.generation,
        .candidate_capacity = capacity,
        .lifecycle_phase = phase,
        .lifecycle_role = role,
        .lifecycle_decision = last_decision,
        .lifecycle_frame_open = frame_open,
        .lifecycle_boundary_exact = boundary_exact,
        .lifecycle_target_first_writer_exact = target_first_writer_exact,
        .lifecycle_target_pre_seen = target_pre_seen,
        .lifecycle_target_post_seen = target_post_seen,
    });
}

constexpr PpDepthPredecessorCrossScopeMode3RootBindingEvidence
PpDepthPredecessorCrossScopePairLifecycle::ClassifyMode3RootBinding(
    const PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness&
        selected_latest,
    const PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence&
        selected_latest_evidence) const noexcept {
    return ClassifyPpDepthPredecessorCrossScopeMode3RootBinding({
        .selected_latest = selected_latest,
        .selected_latest_evidence = selected_latest_evidence,
        .stored_source = staged_source,
        .target_intent = target_intent,
        .p0_post = p0_post,
        .lifecycle_phase = phase,
        .lifecycle_role = role,
        .lifecycle_decision = last_decision,
        .lifecycle_frame_open = frame_open,
        .lifecycle_boundary_exact = boundary_exact,
        .lifecycle_target_first_writer_exact = target_first_writer_exact,
        .lifecycle_target_pre_seen = target_pre_seen,
        .lifecycle_target_post_seen = target_post_seen,
    });
}

constexpr PpDepthPredecessorCrossScopePairResult
PpDepthPredecessorCrossScopePairLifecycle::ObserveLearningRootBoundChainComplete(
    const PpDepthPredecessorTwoHopResult& selected,
    const PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness&
        selected_latest,
    const PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot&
        runtime_selection,
    const PpDepthPredecessorCrossScopeLearningLogicalCandidateSummary&
        logical_candidates,
    const PpDepthPredecessorCrossScopeLearningTransitionCandidateSummary&
        transition_candidates) noexcept {
    if (HasFailure()) {
        return MakeResult();
    }
    const auto selected_latest_evidence = ClassifyMode3SelectedLatestBinding(
        selected, selected_latest, runtime_selection, logical_candidates,
        transition_candidates);
    const auto root_binding =
        ClassifyMode3RootBinding(selected_latest, selected_latest_evidence);
    if (!selected_latest_evidence.raw_rolling_bounded || !root_binding.exact) {
        Fail(PpDepthPredecessorCrossScopePairDecision::Mode3ProofMismatch);
        return MakeResult();
    }
    staged_expected_target_repeat_count =
        selected_latest_evidence.raw_occurrences - 1;
    staged_commit = true;
    last_decision = PpDepthPredecessorCrossScopePairDecision::Committed;
    return MakeResult();
}

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeLearningTargetPreRuntimeBoundary
BuildPpDepthPredecessorCrossScopeLearningTargetPreBoundary(
    const PpDepthPredecessorCrossScopeLearningTargetPreRuntimeObservation& raw) noexcept {
    const auto& source = raw.source_current;
    const auto& target = raw.target_intent;
    const PpDepthPredecessorCrossScopeLearningJoinDiscriminator join_discriminator{
        .source_structural_valid =
            source.valid && source.writer.Valid() && source.guest_rendering_serial != 0 &&
            source.host_rendering_instance_id != 0 && source.producer_marked,
        .source_after_diagnostic_resume = source.after_diagnostic_resume,
        .target_intent_valid = target.Valid(),
        .source_target_link_exact = source.writer.link == target.link,
        .source_target_range_exact = source.writer.range == target.range,
        .source_target_backing_exact =
            source.writer.backing_generation == target.backing_generation,
        .source_target_mutation_exact = source.writer.mutation_epoch == target.mutation_epoch,
        .raw_current_matches_source =
            raw.raw_current_link.Valid() && raw.raw_current_link == source.writer.link &&
            raw.raw_current_range == source.writer.range &&
            raw.raw_current_backing_generation != 0 &&
            raw.raw_current_backing_generation == source.writer.backing_generation &&
            raw.raw_current_mutation_epoch != 0 &&
            raw.raw_current_mutation_epoch == source.writer.mutation_epoch,
    };
    const bool join_exact =
        join_discriminator.source_structural_valid &&
        join_discriminator.target_intent_valid &&
        join_discriminator.source_target_link_exact &&
        join_discriminator.source_target_range_exact &&
        join_discriminator.source_target_backing_exact &&
        join_discriminator.source_target_mutation_exact;
    const bool raw_join_exact = join_exact && join_discriminator.raw_current_matches_source;
    const auto source_lineage =
        ClassifyPpDepthPredecessorCrossScopeSourceLineage({
            .source_guest_rendering_serial = source.guest_rendering_serial,
            .source_host_rendering_instance_id =
                source.host_rendering_instance_id,
            .source_after_diagnostic_resume = source.after_diagnostic_resume,
            .checkpoint = {
                .guest_rendering_serial =
                    raw.boundary_before_guest_rendering_serial,
                .host_rendering_instance_id =
                    raw.boundary_before_host_rendering_instance_id,
                .rendering_active = raw.boundary_before_rendering_active,
                .regular_origin = raw.boundary_before_regular_origin,
                .diagnostic_origin = raw.boundary_before_diagnostic_origin,
            },
        });
    const bool boundary_before_guest_exact = source_lineage.guest_exact;
    const bool boundary_before_host_exact = source_lineage.same_host;
    const bool boundary_before_rendering_active =
        raw.boundary_before_rendering_active;
    const bool boundary_before_regular_origin_exact =
        raw.boundary_before_regular_origin &&
        !raw.boundary_before_diagnostic_origin;
    const bool boundary_before_source_origin_exact =
        source_lineage.source_origin_exact;
    const bool boundary_before_diagnostic_descendant_exact =
        source_lineage.diagnostic_descendant_exact;
    const bool boundary_before_source_lineage_exact = source_lineage.exact;
    const bool after_exact =
        raw.boundary_after_guest_rendering_serial != 0 &&
        raw.boundary_after_host_rendering_instance_id != 0 &&
        raw.boundary_after_rendering_active && raw.boundary_after_regular_origin &&
        !raw.boundary_after_diagnostic_origin;
    const bool live_target_exact =
        raw.live_target_guest_rendering_serial ==
            raw.boundary_after_guest_rendering_serial &&
        raw.live_target_host_rendering_instance_id ==
            raw.boundary_after_host_rendering_instance_id &&
        raw.live_target_rendering_active == raw.boundary_after_rendering_active &&
        raw.live_target_regular_origin == raw.boundary_after_regular_origin &&
        raw.live_target_diagnostic_origin == raw.boundary_after_diagnostic_origin &&
        raw.live_target_guest_rendering_serial != 0 &&
        raw.live_target_host_rendering_instance_id != 0 &&
        raw.live_target_rendering_active && raw.live_target_regular_origin &&
        !raw.live_target_diagnostic_origin;
    const bool raw_boundary_exact =
        raw.raw_boundary_active && raw.raw_boundary_tracked &&
        boundary_before_source_lineage_exact && after_exact && live_target_exact;
    const PpDepthPredecessorCrossScopeLearningObservation observation{
        .source_current = source,
        .target_intent = target,
        .predecessor = raw.predecessor,
        .target_guest_rendering_serial = raw.live_target_guest_rendering_serial,
        .target_host_rendering_instance_id = raw.live_target_host_rendering_instance_id,
        .source_outside_target_guest_scope = raw.source_outside_target_guest_scope,
        .target_rendering_active = raw.live_target_rendering_active,
        .after_begin_rendering = raw.after_begin_rendering,
        .target_regular_origin = raw.live_target_regular_origin,
        .target_after_diagnostic_resume = raw.live_target_diagnostic_origin,
        .target_depth_load = raw.target_depth_load,
        .target_depth_clear = raw.target_depth_clear,
        .target_stencil_clear = raw.target_stencil_clear,
        .depth_aspect = raw.depth_aspect,
        .canonical_attachment = raw.canonical_attachment,
    };
    const bool classified_exact =
        ClassifyPpDepthPredecessorCrossScopeLearningBoundary(observation).exact;
    return {
        .observation = observation,
        .join_discriminator = join_discriminator,
        .raw_boundary_exact = raw_boundary_exact,
        .boundary_before_guest_exact = boundary_before_guest_exact,
        .boundary_before_host_exact = boundary_before_host_exact,
        .boundary_before_rendering_active = boundary_before_rendering_active,
        .boundary_before_regular_origin_exact =
            boundary_before_regular_origin_exact,
        .boundary_before_source_origin_exact =
            boundary_before_source_origin_exact,
        .boundary_before_diagnostic_descendant_exact =
            boundary_before_diagnostic_descendant_exact,
        .boundary_before_source_lineage_exact =
            boundary_before_source_lineage_exact,
        .raw_join_exact = raw_join_exact,
        .live_target_exact = live_target_exact,
        .exact = raw_boundary_exact && raw_join_exact && classified_exact,
    };
}

struct PpDepthPredecessorCrossScopeLearningTargetPreAttemptPlan {
    bool observe_cross_scope{};
    bool reject_shared_feedback{};
    bool continue_nested_multi_hop{};
};

[[nodiscard]] constexpr
PpDepthPredecessorCrossScopeLearningTargetPreAttemptPlan
PlanPpDepthPredecessorCrossScopeLearningTargetPreAttempt(
    const PpDepthPredecessorCrossScopeLearningTargetPreRuntimeBoundary& boundary) noexcept {
    return {
        .observe_cross_scope = boundary.exact,
        .reject_shared_feedback = false,
        .continue_nested_multi_hop = true,
    };
}

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeLearningEvidence
SnapshotPpDepthPredecessorCrossScopeLearningEvidence(
    const PpDepthPredecessorCrossScopeLearningTargetPreRuntimeBoundary& boundary) noexcept {
    const auto classified =
        ClassifyPpDepthPredecessorCrossScopeLearningBoundary(boundary.observation);
    return {
        .raw_boundary_exact = boundary.raw_boundary_exact,
        .boundary_before_guest_exact = boundary.boundary_before_guest_exact,
        .boundary_before_host_exact = boundary.boundary_before_host_exact,
        .boundary_before_rendering_active =
            boundary.boundary_before_rendering_active,
        .boundary_before_regular_origin_exact =
            boundary.boundary_before_regular_origin_exact,
        .boundary_before_source_origin_exact =
            boundary.boundary_before_source_origin_exact,
        .boundary_before_diagnostic_descendant_exact =
            boundary.boundary_before_diagnostic_descendant_exact,
        .boundary_before_source_lineage_exact =
            boundary.boundary_before_source_lineage_exact,
        .raw_join_exact = boundary.raw_join_exact,
        .live_target_exact = boundary.live_target_exact,
        .predecessor_exact = classified.predecessor_exact,
        .join_exact = classified.join_exact,
        .regular_cross_scope_exact = classified.regular_cross_scope_exact,
        .load_exact = classified.load_exact,
        .join_discriminator = boundary.join_discriminator,
    };
}

struct PpDepthPredecessorCrossScopeSourcePostRuntimeObservation {
    PpDepthPredecessorEndpointStamp endpoint{};
    PpDepthPredecessorCandidateStamp candidate{};
    u64 expected_generation{};
    u32 candidate_capacity{};
    u64 resume_guest_rendering_serial{};
    u64 resume_host_rendering_instance_id{};
    bool after_producer_mark{};
    bool source_plane_copied{};
    bool resume_rendering_active{};
    bool resume_diagnostic_origin{};
};

struct PpDepthPredecessorCrossScopeResumeWitness {
    PpDepthPredecessorEndpointStamp source_post{};
    PpDepthPredecessorCandidateStamp source_candidate{};
    u32 candidate_capacity{};
    u64 resume_guest_rendering_serial{};
    u64 resume_host_rendering_instance_id{};
    bool resume_rendering_active{};
    bool resume_diagnostic_origin{};
    bool resume_preserved_guest_serial{};
    bool valid{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeResumeWitness
BuildPpDepthPredecessorCrossScopeResumeWitness(
    const PpDepthPredecessorCrossScopeSourcePostRuntimeObservation& raw) noexcept {
    const bool candidate_exact =
        raw.candidate.valid && raw.expected_generation != 0 &&
        raw.candidate.generation == raw.expected_generation &&
        raw.candidate_capacity != 0 &&
        raw.candidate_capacity <= PpUpstreamFeedbackPrePostPlan::MaxCandidates &&
        raw.candidate.candidate_index < raw.candidate_capacity &&
        raw.candidate.recorded_plane_mask == 0x1u &&
        raw.candidate.writer == raw.endpoint.writer;
    const bool resume_preserved =
        raw.endpoint.guest_rendering_serial != 0 &&
        raw.resume_guest_rendering_serial == raw.endpoint.guest_rendering_serial;
    const bool resume_exact =
        resume_preserved && raw.resume_host_rendering_instance_id != 0 &&
        raw.resume_host_rendering_instance_id !=
            raw.endpoint.host_rendering_instance_id &&
        raw.resume_rendering_active && raw.resume_diagnostic_origin;
    const bool valid = raw.after_producer_mark && raw.source_plane_copied &&
                       raw.endpoint.Valid() && candidate_exact && resume_exact;
    if (!valid) {
        return {};
    }
    return {
        .source_post = raw.endpoint,
        .source_candidate = raw.candidate,
        .candidate_capacity = raw.candidate_capacity,
        .resume_guest_rendering_serial = raw.resume_guest_rendering_serial,
        .resume_host_rendering_instance_id = raw.resume_host_rendering_instance_id,
        .resume_rendering_active = raw.resume_rendering_active,
        .resume_diagnostic_origin = raw.resume_diagnostic_origin,
        .resume_preserved_guest_serial = resume_preserved,
        .valid = true,
    };
}

[[nodiscard]] constexpr u64 NextPpDepthPredecessorCrossScopeHostInstanceId(
    u64 current) noexcept {
    if (current == 0) {
        return 0;
    }
    return current == std::numeric_limits<u64>::max() ? 1 : current + 1;
}

[[nodiscard]] constexpr bool TryIncrementPpDepthPredecessorCrossScopeSourceRestartCount(
    u32& count) noexcept {
    return TryIncrementPpDepthPredecessorOccurrenceCount(count);
}

struct PpDepthPredecessorCrossScopeSourceRestartPreObservation {
    PpDepthPredecessorEndpointStamp stored_source_post{};
    PpDepthPredecessorCandidateStamp stored_source_candidate{};
    PpDepthPredecessorCrossScopeResumeWitness prior_resume{};
    PpDepthPredecessorWriterStamp current_source_writer{};
    PpDepthPredecessorCrossScopeRenderingCheckpoint repeated_source_checkpoint{};
    VideoCore::ImageColorScopePrivateLink frame_output{};
    PpUpstreamFeedbackPrePostRegistryAction frame_preview{};
    u64 expected_generation{};
    u32 candidate_capacity{};
    u32 recorded_plane_mask_before_preview{};
    u32 recorded_plane_mask_after_preview{};
    bool retained_slot_present{};
    bool target_pre_attempted{};
};

struct PpDepthPredecessorCrossScopeSourceRestartPreClassification {
    bool stored_witness_exact{};
    bool current_writer_exact{};
    bool mutation_successor_available{};
    bool rendering_lineage_exact{};
    bool frame_restart_exact{};
    bool rolling_window_exact{};
    bool exact{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeSourceRestartPreClassification
ClassifyPpDepthPredecessorCrossScopeSourceRestartPre(
    const PpDepthPredecessorCrossScopeSourceRestartPreObservation& observation) noexcept {
    const auto& source = observation.stored_source_post;
    const auto& candidate = observation.stored_source_candidate;
    const auto& resume = observation.prior_resume;
    const auto& checkpoint = observation.repeated_source_checkpoint;
    const bool capacity_exact =
        observation.candidate_capacity != 0 &&
        observation.candidate_capacity <= PpUpstreamFeedbackPrePostPlan::MaxCandidates;
    const bool stored_witness_exact =
        source.Valid() && candidate.valid && resume.valid && capacity_exact &&
        observation.expected_generation != 0 &&
        candidate.generation == observation.expected_generation &&
        candidate.candidate_index < observation.candidate_capacity &&
        candidate.recorded_plane_mask == 0x1u && candidate.writer == source.writer &&
        resume.source_post == source && resume.source_candidate == candidate &&
        resume.candidate_capacity == observation.candidate_capacity &&
        resume.resume_guest_rendering_serial == source.guest_rendering_serial &&
        resume.resume_host_rendering_instance_id != 0 &&
        resume.resume_host_rendering_instance_id != source.host_rendering_instance_id &&
        resume.resume_rendering_active && resume.resume_diagnostic_origin &&
        resume.resume_preserved_guest_serial;
    const bool current_writer_exact =
        observation.current_source_writer.Valid() &&
        observation.current_source_writer == source.writer;
    const bool mutation_successor_available =
        source.writer.mutation_epoch != 0 &&
        source.writer.mutation_epoch != std::numeric_limits<u64>::max();
    const bool rendering_lineage_exact =
        resume.resume_guest_rendering_serial != 0 &&
        resume.resume_host_rendering_instance_id != 0 && checkpoint.rendering_active &&
        !checkpoint.regular_origin && checkpoint.diagnostic_origin &&
        checkpoint.guest_rendering_serial == resume.resume_guest_rendering_serial &&
        checkpoint.host_rendering_instance_id == resume.resume_host_rendering_instance_id;
    const bool frame_restart_exact =
        observation.frame_output.Valid() && observation.frame_output == source.writer.link &&
        IsExactPpDepthPredecessorCrossScopeSourceRestart(
            observation.frame_preview, candidate.candidate_index,
            observation.recorded_plane_mask_before_preview,
            observation.recorded_plane_mask_after_preview,
            observation.retained_slot_present);
    const bool rolling_window_exact = !observation.target_pre_attempted;
    return {
        .stored_witness_exact = stored_witness_exact,
        .current_writer_exact = current_writer_exact,
        .mutation_successor_available = mutation_successor_available,
        .rendering_lineage_exact = rendering_lineage_exact,
        .frame_restart_exact = frame_restart_exact,
        .rolling_window_exact = rolling_window_exact,
        .exact = stored_witness_exact && current_writer_exact &&
                 mutation_successor_available && rendering_lineage_exact &&
                 frame_restart_exact && rolling_window_exact,
    };
}

struct PpDepthPredecessorCrossScopeSourceRestartPostObservation {
    PpDepthPredecessorCrossScopeResumeWitness prior_resume{};
    PpDepthPredecessorEndpointStamp endpoint{};
    PpDepthPredecessorCandidateStamp candidate{};
    PpDepthPredecessorCrossScopeResumeWitness new_resume{};
    u64 expected_generation{};
    u32 candidate_capacity{};
    u32 expected_candidate_index{};
};

struct PpDepthPredecessorCrossScopeSourceRestartPostClassification {
    bool endpoint_identity_exact{};
    bool mutation_successor_exact{};
    bool endpoint_lineage_exact{};
    bool candidate_exact{};
    bool resume_successor_exact{};
    bool exact{};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeSourceRestartPostClassification
ClassifyPpDepthPredecessorCrossScopeSourceRestartPost(
    const PpDepthPredecessorCrossScopeSourceRestartPostObservation& observation) noexcept {
    const auto& prior = observation.prior_resume;
    const auto& old_source = prior.source_post;
    const auto& endpoint = observation.endpoint;
    const auto& candidate = observation.candidate;
    const auto& resume = observation.new_resume;
    const bool capacity_exact =
        observation.candidate_capacity != 0 &&
        observation.candidate_capacity <= PpUpstreamFeedbackPrePostPlan::MaxCandidates;
    const bool endpoint_identity_exact =
        prior.valid && old_source.Valid() && endpoint.Valid() &&
        endpoint.writer.hint == old_source.writer.hint &&
        endpoint.writer.link == old_source.writer.link &&
        endpoint.writer.range == old_source.writer.range &&
        endpoint.writer.backing_generation == old_source.writer.backing_generation;
    const bool mutation_successor_exact =
        old_source.writer.mutation_epoch != 0 &&
        old_source.writer.mutation_epoch != std::numeric_limits<u64>::max() &&
        endpoint.writer.mutation_epoch == old_source.writer.mutation_epoch + 1;
    const bool endpoint_lineage_exact =
        prior.resume_guest_rendering_serial != 0 &&
        prior.resume_host_rendering_instance_id != 0 &&
        prior.resume_rendering_active && prior.resume_diagnostic_origin &&
        prior.resume_preserved_guest_serial && endpoint.producer_marked &&
        endpoint.after_diagnostic_resume &&
        endpoint.guest_rendering_serial == prior.resume_guest_rendering_serial &&
        endpoint.host_rendering_instance_id == prior.resume_host_rendering_instance_id;
    const bool candidate_exact =
        capacity_exact && observation.expected_generation != 0 && candidate.valid &&
        candidate.generation == observation.expected_generation &&
        candidate.candidate_index == observation.expected_candidate_index &&
        candidate.candidate_index < observation.candidate_capacity &&
        candidate.recorded_plane_mask == 0x1u && candidate.writer == endpoint.writer &&
        prior.source_candidate.valid &&
        prior.source_candidate.generation == observation.expected_generation &&
        prior.source_candidate.candidate_index == observation.expected_candidate_index &&
        prior.candidate_capacity == observation.candidate_capacity;
    const u64 expected_resume_host =
        NextPpDepthPredecessorCrossScopeHostInstanceId(endpoint.host_rendering_instance_id);
    const bool resume_successor_exact =
        expected_resume_host != 0 && resume.valid && resume.source_post == endpoint &&
        resume.source_candidate == candidate &&
        resume.candidate_capacity == observation.candidate_capacity &&
        resume.resume_guest_rendering_serial == endpoint.guest_rendering_serial &&
        resume.resume_host_rendering_instance_id == expected_resume_host &&
        resume.resume_rendering_active && resume.resume_diagnostic_origin &&
        resume.resume_preserved_guest_serial;
    return {
        .endpoint_identity_exact = endpoint_identity_exact,
        .mutation_successor_exact = mutation_successor_exact,
        .endpoint_lineage_exact = endpoint_lineage_exact,
        .candidate_exact = candidate_exact,
        .resume_successor_exact = resume_successor_exact,
        .exact = endpoint_identity_exact && mutation_successor_exact &&
                 endpoint_lineage_exact && candidate_exact && resume_successor_exact,
    };
}

constexpr PpDepthPredecessorCrossScopePairResult
PpDepthPredecessorCrossScopePairLifecycle::BeginSourcePostRestart(
    const PpDepthPredecessorCrossScopeSourceRestartPreObservation& observation) noexcept {
    const bool lifecycle_ready = CanBeginSourcePostRestart();
    const auto active_source = source_post;
    const auto active_candidate = source_candidate;
    ClearActiveSource();
    ClearCandidate();
    ClearSourceRestartProof();
    source_restart_open = false;
    role = PpDepthPredecessorCrossScopePairRole::SourcePost;

    const auto classified =
        ClassifyPpDepthPredecessorCrossScopeSourceRestartPre(observation);
    const bool lifecycle_binding_exact =
        observation.stored_source_post == active_source &&
        observation.stored_source_candidate == active_candidate;
    if (!lifecycle_ready || !lifecycle_binding_exact || !classified.exact) {
        Fail(!classified.frame_restart_exact
                 ? PpDepthPredecessorCrossScopePairDecision::CandidateMismatch
                 : PpDepthPredecessorCrossScopePairDecision::SourceMismatch);
        return MakeResult();
    }

    restart_prior_source = observation.prior_resume.source_post;
    restart_prior_candidate = observation.prior_resume.source_candidate;
    restart_prior_candidate_capacity = observation.prior_resume.candidate_capacity;
    restart_prior_resume_guest_rendering_serial =
        observation.prior_resume.resume_guest_rendering_serial;
    restart_prior_resume_host_rendering_instance_id =
        observation.prior_resume.resume_host_rendering_instance_id;
    restart_prior_resume_rendering_active =
        observation.prior_resume.resume_rendering_active;
    restart_prior_resume_diagnostic_origin =
        observation.prior_resume.resume_diagnostic_origin;
    restart_prior_resume_preserved_guest_serial =
        observation.prior_resume.resume_preserved_guest_serial;
    restart_prior_resume_valid = observation.prior_resume.valid;
    source_restart_open = true;
    return MakeResult();
}

constexpr PpDepthPredecessorCrossScopePairResult
PpDepthPredecessorCrossScopePairLifecycle::ObserveSourcePostRestartComplete(
    const PpDepthPredecessorEndpointStamp& endpoint,
    const PpDepthPredecessorCandidateStamp& candidate,
    const PpDepthPredecessorCrossScopeResumeWitness& new_resume) noexcept {
    const bool restart_ready = !HasFailure() && frame_open &&
                               phase == PpDepthPredecessorCrossScopePairPhase::Armed &&
                               role == PpDepthPredecessorCrossScopePairRole::SourcePost &&
                               source_restart_open && restart_prior_resume_valid;
    const PpDepthPredecessorCrossScopeResumeWitness prior_resume{
        .source_post = restart_prior_source,
        .source_candidate = restart_prior_candidate,
        .candidate_capacity = restart_prior_candidate_capacity,
        .resume_guest_rendering_serial = restart_prior_resume_guest_rendering_serial,
        .resume_host_rendering_instance_id = restart_prior_resume_host_rendering_instance_id,
        .resume_rendering_active = restart_prior_resume_rendering_active,
        .resume_diagnostic_origin = restart_prior_resume_diagnostic_origin,
        .resume_preserved_guest_serial = restart_prior_resume_preserved_guest_serial,
        .valid = restart_prior_resume_valid,
    };
    ClearActiveSource();
    ClearCandidate();
    source_restart_open = false;
    const auto classified = ClassifyPpDepthPredecessorCrossScopeSourceRestartPost({
        .prior_resume = prior_resume,
        .endpoint = endpoint,
        .candidate = candidate,
        .new_resume = new_resume,
        .expected_generation = generation,
        .candidate_capacity = capacity,
        .expected_candidate_index = prior_resume.source_candidate.candidate_index,
    });
    ClearSourceRestartProof();
    if (!restart_ready || !classified.exact ||
        !TryIncrementPpDepthPredecessorCrossScopeSourceRestartCount(
            source_restart_count)) {
        Fail(!classified.candidate_exact
                 ? PpDepthPredecessorCrossScopePairDecision::CandidateMismatch
                 : PpDepthPredecessorCrossScopePairDecision::SourceMismatch);
        return MakeResult();
    }
    source_post = endpoint;
    source_candidate = candidate;
    source_post_count = 1;
    role = PpDepthPredecessorCrossScopePairRole::TargetPre;
    return MakeResult();
}

struct PpDepthPredecessorCrossScopeTargetPreRuntimeObservation {
    PpDepthPredecessorCrossScopeResumeWitness source_resume{};
    PpDepthPredecessorWriterStamp target_pre_current_writer{};
    PpDepthPredecessorWriterIntent target_intent{};
    PpDepthPredecessorCandidateStamp candidate{};
    VideoCore::ImageColorScopePrivateLink raw_current_link{};
    VideoCore::SubresourceRange raw_current_range{};
    u64 raw_current_backing_generation{};
    u64 raw_current_mutation_epoch{};
    bool raw_boundary_active{};
    bool raw_boundary_tracked{};
    u64 boundary_before_guest_rendering_serial{};
    u64 boundary_before_host_rendering_instance_id{};
    u64 boundary_after_guest_rendering_serial{};
    u64 boundary_after_host_rendering_instance_id{};
    bool boundary_before_rendering_active{};
    bool boundary_before_diagnostic_origin{};
    bool boundary_after_rendering_active{};
    bool boundary_after_regular_origin{};
    bool boundary_after_diagnostic_origin{};
    u64 live_target_guest_rendering_serial{};
    u64 live_target_host_rendering_instance_id{};
    bool live_target_rendering_active{};
    bool live_target_regular_origin{};
    bool live_target_diagnostic_origin{};
    bool source_outside_target_guest_scope{};
    bool after_begin_rendering{};
    bool target_depth_load{};
    bool target_depth_clear{};
    bool target_stencil_clear{};
    bool depth_aspect{};
    bool canonical_attachment{};
};

struct PpDepthPredecessorCrossScopeTargetPreRuntimeBoundary {
    PpDepthPredecessorCrossScopeBoundaryObservation observation{};
    PpDepthPredecessorCrossScopeBoundaryClassification classification{};
    bool raw_resume_exact{};
    bool raw_target_exact{};
    bool raw_join_exact{};
    bool exact{};
};

enum class PpDepthPredecessorCrossScopeTargetPreAttemptDisposition : u32 {
    Reject,
    IgnoreNoTransition,
    AcceptTransition,
};

struct PpDepthPredecessorCrossScopeTargetPreAttemptClassification {
    PpDepthPredecessorCrossScopeTargetPreRuntimeBoundary boundary{};
    PpDepthPredecessorCrossScopeTargetPreAttemptDisposition disposition{
        PpDepthPredecessorCrossScopeTargetPreAttemptDisposition::Reject};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeTargetPreRuntimeBoundary
BuildPpDepthPredecessorCrossScopeTargetPreBoundary(
    const PpDepthPredecessorCrossScopeTargetPreRuntimeObservation& raw) noexcept {
    const auto& witness = raw.source_resume;
    const auto& source = witness.source_post;
    const auto& current = raw.target_pre_current_writer;
    const auto& target = raw.target_intent;
    const bool active_resume_exact =
        witness.valid && raw.raw_boundary_active && raw.raw_boundary_tracked &&
        raw.boundary_before_guest_rendering_serial ==
            witness.resume_guest_rendering_serial &&
        raw.boundary_before_host_rendering_instance_id ==
            witness.resume_host_rendering_instance_id &&
        raw.boundary_before_rendering_active == witness.resume_rendering_active &&
        raw.boundary_before_diagnostic_origin == witness.resume_diagnostic_origin &&
        raw.boundary_before_rendering_active && raw.boundary_before_diagnostic_origin;
    const bool live_target_exact =
        raw.live_target_guest_rendering_serial ==
            raw.boundary_after_guest_rendering_serial &&
        raw.live_target_host_rendering_instance_id ==
            raw.boundary_after_host_rendering_instance_id &&
        raw.live_target_rendering_active == raw.boundary_after_rendering_active &&
        raw.live_target_regular_origin == raw.boundary_after_regular_origin &&
        raw.live_target_diagnostic_origin == raw.boundary_after_diagnostic_origin;
    const bool raw_target_exact =
        raw.raw_boundary_active && raw.raw_boundary_tracked && live_target_exact &&
        raw.live_target_guest_rendering_serial != 0 &&
        raw.live_target_host_rendering_instance_id != 0 &&
        raw.live_target_rendering_active && raw.live_target_regular_origin &&
        !raw.live_target_diagnostic_origin && raw.boundary_after_rendering_active &&
        raw.boundary_after_regular_origin && !raw.boundary_after_diagnostic_origin;
    const bool raw_join_exact =
        source.Valid() && current.Valid() && target.Valid() &&
        raw.candidate == witness.source_candidate &&
        raw.raw_current_link.Valid() && raw.raw_current_link == source.writer.link &&
        raw.raw_current_link == current.link && raw.raw_current_link == target.link &&
        raw.raw_current_range == source.writer.range &&
        raw.raw_current_range == current.range && raw.raw_current_range == target.range &&
        raw.raw_current_backing_generation != 0 &&
        raw.raw_current_backing_generation == source.writer.backing_generation &&
        raw.raw_current_backing_generation == current.backing_generation &&
        raw.raw_current_backing_generation == target.backing_generation &&
        raw.raw_current_mutation_epoch != 0 &&
        raw.raw_current_mutation_epoch == source.writer.mutation_epoch &&
        raw.raw_current_mutation_epoch == current.mutation_epoch &&
        raw.raw_current_mutation_epoch == target.mutation_epoch;
    const PpDepthPredecessorCrossScopeBoundaryObservation observation{
        .source_post = source,
        .target_pre_current_writer = current,
        .target_intent = target,
        .candidate = raw.candidate,
        .resume_guest_rendering_serial = witness.resume_guest_rendering_serial,
        .resume_host_rendering_instance_id = witness.resume_host_rendering_instance_id,
        .target_guest_rendering_serial = raw.live_target_guest_rendering_serial,
        .target_host_rendering_instance_id = raw.live_target_host_rendering_instance_id,
        .resume_diagnostic_origin = witness.resume_diagnostic_origin,
        .resume_preserved_guest_serial = witness.resume_preserved_guest_serial,
        .resume_rendering_active = witness.resume_rendering_active,
        .source_outside_target_guest_scope = raw.source_outside_target_guest_scope,
        .after_begin_rendering = raw.after_begin_rendering,
        .target_rendering_active = raw.live_target_rendering_active,
        .target_regular_origin = raw.live_target_regular_origin,
        .target_after_diagnostic_resume = raw.live_target_diagnostic_origin,
        .target_depth_load = raw.target_depth_load,
        .target_depth_clear = raw.target_depth_clear,
        .target_stencil_clear = raw.target_stencil_clear,
        .depth_aspect = raw.depth_aspect,
        .canonical_attachment = raw.canonical_attachment,
    };
    const auto classification =
        ClassifyPpDepthPredecessorCrossScopeBoundary(
            observation, witness.source_candidate.generation,
            witness.candidate_capacity);
    const bool ended_resume_exact =
        witness.valid && witness.resume_rendering_active &&
        witness.resume_diagnostic_origin && witness.resume_preserved_guest_serial &&
        raw.raw_boundary_active && raw.raw_boundary_tracked &&
        raw.boundary_before_guest_rendering_serial ==
            witness.resume_guest_rendering_serial &&
        raw.boundary_before_host_rendering_instance_id ==
            witness.resume_host_rendering_instance_id &&
        !raw.boundary_before_rendering_active &&
        raw.boundary_before_diagnostic_origin == witness.resume_diagnostic_origin &&
        raw.boundary_before_diagnostic_origin && raw_target_exact && raw_join_exact &&
        classification.exact;
    const bool raw_resume_exact = active_resume_exact || ended_resume_exact;
    return {
        .observation = observation,
        .classification = classification,
        .raw_resume_exact = raw_resume_exact,
        .raw_target_exact = raw_target_exact,
        .raw_join_exact = raw_join_exact,
        .exact = raw_resume_exact && raw_target_exact && raw_join_exact &&
                 classification.exact,
    };
}

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeTargetPreAttemptClassification
ClassifyPpDepthPredecessorCrossScopeTargetPreAttempt(
    const PpDepthPredecessorCrossScopeTargetPreRuntimeObservation& raw,
    bool lifecycle_stored_source_exact) noexcept {
    const auto boundary = BuildPpDepthPredecessorCrossScopeTargetPreBoundary(raw);
    const auto& witness = raw.source_resume;
    const bool same_state_diagnostic_no_transition =
        lifecycle_stored_source_exact && boundary.raw_join_exact &&
        boundary.classification.endpoint_exact &&
        boundary.classification.candidate_exact &&
        boundary.classification.resume_exact && witness.valid &&
        witness.resume_rendering_active && witness.resume_diagnostic_origin &&
        witness.resume_preserved_guest_serial && raw.raw_boundary_active &&
        raw.raw_boundary_tracked && raw.boundary_before_guest_rendering_serial ==
                                       witness.resume_guest_rendering_serial &&
        raw.boundary_before_host_rendering_instance_id ==
            witness.resume_host_rendering_instance_id &&
        raw.boundary_before_rendering_active &&
        raw.boundary_before_diagnostic_origin &&
        raw.boundary_after_guest_rendering_serial ==
            raw.boundary_before_guest_rendering_serial &&
        raw.boundary_after_host_rendering_instance_id ==
            raw.boundary_before_host_rendering_instance_id &&
        raw.boundary_after_rendering_active ==
            raw.boundary_before_rendering_active &&
        !raw.boundary_after_regular_origin &&
        raw.boundary_after_diagnostic_origin ==
            raw.boundary_before_diagnostic_origin &&
        raw.live_target_guest_rendering_serial ==
            raw.boundary_after_guest_rendering_serial &&
        raw.live_target_host_rendering_instance_id ==
            raw.boundary_after_host_rendering_instance_id &&
        raw.live_target_rendering_active ==
            raw.boundary_after_rendering_active &&
        raw.live_target_regular_origin == raw.boundary_after_regular_origin &&
        raw.live_target_diagnostic_origin ==
            raw.boundary_after_diagnostic_origin &&
        raw.live_target_rendering_active && !raw.live_target_regular_origin &&
        raw.live_target_diagnostic_origin &&
        !raw.source_outside_target_guest_scope && raw.after_begin_rendering;
    const auto disposition =
        same_state_diagnostic_no_transition
            ? PpDepthPredecessorCrossScopeTargetPreAttemptDisposition::
                  IgnoreNoTransition
        : lifecycle_stored_source_exact && boundary.exact
            ? PpDepthPredecessorCrossScopeTargetPreAttemptDisposition::
                  AcceptTransition
            : PpDepthPredecessorCrossScopeTargetPreAttemptDisposition::Reject;
    return {
        .boundary = boundary,
        .disposition = disposition,
    };
}

struct PpDepthPredecessorCrossScopeArmedTargetPreEvidence {
    bool witness_valid{};
    bool raw_boundary_active{};
    bool raw_boundary_tracked{};
    bool witness_resume_active{};
    bool witness_resume_diagnostic_origin{};
    bool checkpoint_resume_active{};
    bool checkpoint_resume_diagnostic_origin{};
    bool resume_guest_exact{};
    bool resume_host_exact{};
    bool resume_activity_exact{};
    bool resume_origin_exact{};
    bool resume_metadata_without_activity_exact{};
    bool target_guest_plus_one_exact{};
    bool target_host_nonzero{};
    bool target_host_diff_source{};
    bool target_host_diff_resume{};
    bool boundary_live_guest_equal{};
    bool boundary_live_host_equal{};
    bool boundary_live_active_equal{};
    bool boundary_live_origin_equal{};
    bool target_after_active{};
    bool live_active{};
    bool target_regular{};
    bool target_not_diagnostic{};
    bool source_outside_target_scope{};
    bool after_begin{};
    bool raw_resume_exact{};
    bool raw_target_exact{};
    bool raw_join_exact{};
    bool builder_exact{};
    bool classifier_endpoint_exact{};
    bool classifier_candidate_exact{};
    bool classifier_resume_exact{};
    bool classifier_scope_transition_exact{};
    bool classifier_load_exact{};
    bool classifier_exact{};
    bool lifecycle_stored_source_exact{};

    bool operator==(const PpDepthPredecessorCrossScopeArmedTargetPreEvidence&) const = default;
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeArmedTargetPreEvidence
SnapshotPpDepthPredecessorCrossScopeArmedTargetPreEvidence(
    const PpDepthPredecessorCrossScopeTargetPreRuntimeObservation& raw,
    bool lifecycle_stored_source_exact) noexcept {
    const auto& witness = raw.source_resume;
    const bool resume_guest_exact =
        raw.boundary_before_guest_rendering_serial == witness.resume_guest_rendering_serial;
    const bool resume_host_exact =
        raw.boundary_before_host_rendering_instance_id == witness.resume_host_rendering_instance_id;
    const bool resume_activity_exact =
        raw.boundary_before_rendering_active == witness.resume_rendering_active;
    const bool resume_origin_exact =
        raw.boundary_before_diagnostic_origin == witness.resume_diagnostic_origin;
    const bool resume_metadata_without_activity_exact =
        witness.valid && raw.raw_boundary_active && raw.raw_boundary_tracked &&
        resume_guest_exact && resume_host_exact && resume_origin_exact &&
        raw.boundary_before_diagnostic_origin;
    const bool target_guest_plus_one_exact =
        witness.source_post.guest_rendering_serial != std::numeric_limits<u64>::max() &&
        raw.live_target_guest_rendering_serial == witness.source_post.guest_rendering_serial + 1;
    const bool boundary_live_origin_equal =
        raw.live_target_regular_origin == raw.boundary_after_regular_origin &&
        raw.live_target_diagnostic_origin == raw.boundary_after_diagnostic_origin;
    const auto boundary = BuildPpDepthPredecessorCrossScopeTargetPreBoundary(raw);
    return {
        .witness_valid = witness.valid,
        .raw_boundary_active = raw.raw_boundary_active,
        .raw_boundary_tracked = raw.raw_boundary_tracked,
        .witness_resume_active = witness.resume_rendering_active,
        .witness_resume_diagnostic_origin = witness.resume_diagnostic_origin,
        .checkpoint_resume_active = raw.boundary_before_rendering_active,
        .checkpoint_resume_diagnostic_origin = raw.boundary_before_diagnostic_origin,
        .resume_guest_exact = resume_guest_exact,
        .resume_host_exact = resume_host_exact,
        .resume_activity_exact = resume_activity_exact,
        .resume_origin_exact = resume_origin_exact,
        .resume_metadata_without_activity_exact = resume_metadata_without_activity_exact,
        .target_guest_plus_one_exact = target_guest_plus_one_exact,
        .target_host_nonzero = raw.live_target_host_rendering_instance_id != 0,
        .target_host_diff_source = raw.live_target_host_rendering_instance_id !=
                                   witness.source_post.host_rendering_instance_id,
        .target_host_diff_resume =
            raw.live_target_host_rendering_instance_id != witness.resume_host_rendering_instance_id,
        .boundary_live_guest_equal =
            raw.live_target_guest_rendering_serial == raw.boundary_after_guest_rendering_serial,
        .boundary_live_host_equal = raw.live_target_host_rendering_instance_id ==
                                    raw.boundary_after_host_rendering_instance_id,
        .boundary_live_active_equal =
            raw.live_target_rendering_active == raw.boundary_after_rendering_active,
        .boundary_live_origin_equal = boundary_live_origin_equal,
        .target_after_active = raw.boundary_after_rendering_active,
        .live_active = raw.live_target_rendering_active,
        .target_regular = raw.live_target_regular_origin,
        .target_not_diagnostic = !raw.live_target_diagnostic_origin,
        .source_outside_target_scope = raw.source_outside_target_guest_scope,
        .after_begin = raw.after_begin_rendering,
        .raw_resume_exact = boundary.raw_resume_exact,
        .raw_target_exact = boundary.raw_target_exact,
        .raw_join_exact = boundary.raw_join_exact,
        .builder_exact = boundary.exact,
        .classifier_endpoint_exact = boundary.classification.endpoint_exact,
        .classifier_candidate_exact = boundary.classification.candidate_exact,
        .classifier_resume_exact = boundary.classification.resume_exact,
        .classifier_scope_transition_exact = boundary.classification.scope_transition_exact,
        .classifier_load_exact = boundary.classification.load_exact,
        .classifier_exact = boundary.classification.exact,
        .lifecycle_stored_source_exact = lifecycle_stored_source_exact,
    };
}

enum class PpDepthPredecessorCrossScopeArmedTargetPreEvidenceEvent : u32 {
    Reset,
    NonAttempt,
    Attempt,
};

struct PpDepthPredecessorCrossScopeArmedTargetPreEvidenceObservation {
    PpDepthPredecessorCrossScopeArmedTargetPreEvidenceEvent event{
        PpDepthPredecessorCrossScopeArmedTargetPreEvidenceEvent::NonAttempt};
    PpDepthPredecessorCrossScopeArmedTargetPreEvidence evidence{};
};

struct PpDepthPredecessorCrossScopeArmedTargetPreEvidenceSummary {
    u32 attempt_count{};
    bool count_overflow{};
    bool unique_valid{};
    PpDepthPredecessorCrossScopeArmedTargetPreEvidence unique_evidence{};

    bool operator==(
        const PpDepthPredecessorCrossScopeArmedTargetPreEvidenceSummary&) const = default;
};

[[nodiscard]] constexpr
PpDepthPredecessorCrossScopeArmedTargetPreEvidenceSummary
AccumulatePpDepthPredecessorCrossScopeArmedTargetPreEvidence(
    const PpDepthPredecessorCrossScopeArmedTargetPreEvidenceSummary& current,
    const PpDepthPredecessorCrossScopeArmedTargetPreEvidenceObservation&
        observation) noexcept {
    using Event =
        PpDepthPredecessorCrossScopeArmedTargetPreEvidenceEvent;
    switch (observation.event) {
    case Event::Reset:
        return {};
    case Event::NonAttempt:
        return current;
    case Event::Attempt:
        break;
    default:
        return {};
    }

    auto result = current;
    if (result.attempt_count == std::numeric_limits<u32>::max()) {
        result.count_overflow = true;
    } else {
        ++result.attempt_count;
    }
    if (current.attempt_count == 0 && !current.count_overflow) {
        result.unique_valid = true;
        result.unique_evidence = observation.evidence;
    } else {
        result.unique_valid = false;
        result.unique_evidence = {};
    }
    return result;
}

enum class PpDepthPredecessorCrossScopeSourceRestartReason : u32 {
    NotAttempted = 0,
    Exact = 1,
    RollingClosed = 2,
    StoredWitnessMismatch = 3,
    CurrentWriterMismatch = 4,
    RenderingLineageMismatch = 5,
    FrameRestartMismatch = 6,
    EndpointMismatch = 7,
    CandidateMismatch = 8,
    ResumeMismatch = 9,
    CountOverflow = 10,
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeSourceRestartReason
SanitizePpDepthPredecessorCrossScopeSourceRestartReason(
    PpDepthPredecessorCrossScopeSourceRestartReason reason) noexcept {
    return static_cast<u32>(reason) <=
                   static_cast<u32>(
                       PpDepthPredecessorCrossScopeSourceRestartReason::CountOverflow)
               ? reason
               : PpDepthPredecessorCrossScopeSourceRestartReason::NotAttempted;
}

struct PpDepthPredecessorCrossScopeSourceRestartAttemptEvidence {
    PpDepthPredecessorCrossScopeSourceRestartReason reason{
        PpDepthPredecessorCrossScopeSourceRestartReason::NotAttempted};
    bool completed{};
    bool pre_exact{};
    bool endpoint_exact{};
    bool candidate_exact{};
    bool resume_exact{};
    bool retained_slot_exact{};
    bool copy_mode_exact{};
    bool exact{};

    bool operator==(
        const PpDepthPredecessorCrossScopeSourceRestartAttemptEvidence&) const = default;
};

enum class PpDepthPredecessorCrossScopeSourceRestartEvidenceEvent : u32 {
    Reset,
    NonAttempt,
    Attempt,
    Completion,
    Failure,
};

struct PpDepthPredecessorCrossScopeSourceRestartEvidenceObservation {
    PpDepthPredecessorCrossScopeSourceRestartEvidenceEvent event{
        PpDepthPredecessorCrossScopeSourceRestartEvidenceEvent::NonAttempt};
    PpDepthPredecessorCrossScopeSourceRestartAttemptEvidence evidence{};
};

struct PpDepthPredecessorCrossScopeSourceRestartEvidence {
    PpDepthPredecessorCrossScopeSourceRestartReason reason{
        PpDepthPredecessorCrossScopeSourceRestartReason::NotAttempted};
    u32 attempt_count{};
    u32 completion_count{};
    bool count_overflow{};
    bool latest_valid{};
    bool pre_exact{};
    bool endpoint_exact{};
    bool candidate_exact{};
    bool resume_exact{};
    bool retained_slot_exact{};
    bool copy_mode_exact{};
    bool exact{};

    bool operator==(
        const PpDepthPredecessorCrossScopeSourceRestartEvidence&) const = default;
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeSourceRestartEvidence
AccumulatePpDepthPredecessorCrossScopeSourceRestartEvidence(
    const PpDepthPredecessorCrossScopeSourceRestartEvidence& current,
    const PpDepthPredecessorCrossScopeSourceRestartEvidenceObservation& observation) noexcept {
    using Event = PpDepthPredecessorCrossScopeSourceRestartEvidenceEvent;
    if (observation.event == Event::Reset) {
        return {};
    }
    if (observation.event == Event::NonAttempt) {
        return current;
    }
    if (observation.event != Event::Attempt && observation.event != Event::Completion &&
        observation.event != Event::Failure) {
        return {};
    }

    auto next = current;
    bool overflow{};
    if (observation.event == Event::Attempt) {
        overflow |= !TryIncrementPpDepthPredecessorCrossScopeSourceRestartCount(
            next.attempt_count);
    }
    const bool completes =
        observation.event == Event::Completion ||
        (observation.event == Event::Attempt && observation.evidence.completed);
    if (completes) {
        overflow |= !TryIncrementPpDepthPredecessorCrossScopeSourceRestartCount(
            next.completion_count);
    }
    next.count_overflow |= overflow;
    next.reason = overflow
                      ? PpDepthPredecessorCrossScopeSourceRestartReason::CountOverflow
                      : SanitizePpDepthPredecessorCrossScopeSourceRestartReason(
                            observation.evidence.reason);
    const bool retain = observation.event != Event::Failure && completes &&
                        !next.count_overflow && observation.evidence.exact &&
                        next.reason ==
                            PpDepthPredecessorCrossScopeSourceRestartReason::Exact;
    next.latest_valid = retain;
    next.pre_exact = retain && observation.evidence.pre_exact;
    next.endpoint_exact = retain && observation.evidence.endpoint_exact;
    next.candidate_exact = retain && observation.evidence.candidate_exact;
    next.resume_exact = retain && observation.evidence.resume_exact;
    next.retained_slot_exact = retain && observation.evidence.retained_slot_exact;
    next.copy_mode_exact = retain && observation.evidence.copy_mode_exact;
    next.exact = retain && observation.evidence.exact;
    return next;
}

enum class PpDepthPredecessorRuntimeFrameDisposition : u32 {
    RejectAndDiscard,
    Publish,
    DiscardTraining,
    Ignore,
};

struct PpDepthPredecessorRuntimeFrameDispositionInput {
    PpDepthPredecessorRuntimePath path{PpDepthPredecessorRuntimePath::Disabled};
    bool feedback_armed{};
    bool slot_present{};
    u32 recorded_plane_mask{};
    bool candidate_saved{};
    bool lifecycle_selected{};
    bool frame_failed{};
};

struct PpDepthPredecessorRuntimeFrameDispositionResult {
    PpDepthPredecessorRuntimeFrameDisposition disposition{
        PpDepthPredecessorRuntimeFrameDisposition::Ignore};
    bool reject_before_take{};
    bool take_feedback{};
    bool retain_slot_through_take{};
    bool schedule_deferred_cleanup{};
    bool publish_feedback{};
    bool clear_candidate{};
    bool allow_configured_fallback{};
};

[[nodiscard]] constexpr PpDepthPredecessorRuntimeFrameDispositionResult
ClassifyPpDepthPredecessorRuntimeFrameDisposition(
    const PpDepthPredecessorRuntimeFrameDispositionInput& input) noexcept {
    const bool mode4_bootstrap =
        input.path == PpDepthPredecessorRuntimePath::CrossScopeBootstrap;
    const bool mode4_armed =
        input.path == PpDepthPredecessorRuntimePath::CrossScopeArmed;
    if (!mode4_bootstrap && !mode4_armed) {
        return {};
    }
    const bool take_feedback = input.feedback_armed;
    const bool deferred_slot = input.feedback_armed && input.slot_present;
    if (mode4_bootstrap) {
        return {
            .disposition = PpDepthPredecessorRuntimeFrameDisposition::DiscardTraining,
            .reject_before_take = input.feedback_armed,
            .take_feedback = take_feedback,
            .retain_slot_through_take = deferred_slot,
            .schedule_deferred_cleanup = deferred_slot,
            .publish_feedback = false,
            .clear_candidate = true,
        };
    }
    const bool publish = input.feedback_armed && input.slot_present &&
                         input.recorded_plane_mask == 0x3u && input.candidate_saved &&
                         input.lifecycle_selected && !input.frame_failed;
    if (publish) {
        return {
            .disposition = PpDepthPredecessorRuntimeFrameDisposition::Publish,
            .take_feedback = true,
            .retain_slot_through_take = true,
            .schedule_deferred_cleanup = true,
            .publish_feedback = true,
        };
    }
    return {
        .disposition = PpDepthPredecessorRuntimeFrameDisposition::RejectAndDiscard,
        .reject_before_take = input.feedback_armed,
        .take_feedback = take_feedback,
        .retain_slot_through_take = deferred_slot,
        .schedule_deferred_cleanup = deferred_slot,
        .publish_feedback = false,
        .clear_candidate = true,
    };
}

struct PpDepthPredecessorCrossScopeRuntimeOutcome {
    PpDepthPredecessorCrossScopePairResult result{};
    PpDepthPredecessorCrossScopePairPhase phase{
        PpDepthPredecessorCrossScopePairPhase::Learning};
    PpDepthPredecessorCrossScopePairRole role{
        PpDepthPredecessorCrossScopePairRole::None};
    PpDepthPredecessorCrossScopePairDecision decision{
        PpDepthPredecessorCrossScopePairDecision::Idle};
    PpDepthPredecessorCrossScopeBootstrapRuntimeEvidence bootstrap{};
    u32 expected_tail_occurrences{};
    u32 observed_tail_occurrences{};
    u32 source_post_count{};
    u32 target_pre_count{};
    bool boundary_exact{};
    PpDepthPredecessorCrossScopeLearningEvidence learning{};
    PpDepthPredecessorCrossScopeLearningLogicalCandidateSummary
        learning_logical_candidates{};
    PpDepthPredecessorCrossScopeLearningTransitionCandidateSummary
        learning_transition_candidates{};
    PpDepthPredecessorCrossScopeArmedTargetPreEvidenceSummary
        armed_target_pre_evidence{};
    PpDepthPredecessorCrossScopeSourceRestartEvidence source_restart_evidence{};
    PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence
        selected_latest_evidence{};
    PpDepthPredecessorCrossScopeMode3RootBindingEvidence
        root_binding_evidence{};
    PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceSummary
        first_tail_pre_drift_evidence{};
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceSummary
        armed_p0_repeat_probe_evidence{};
    bool target_first_writer_exact{};
    bool tail_exact{};
    bool candidate_saved{};
    bool committed{};
    bool select{};
    bool source_plane_recorded{};
    bool target_plane_recorded{};
    PpDepthPredecessorRuntimeFrameDisposition frame_disposition{
        PpDepthPredecessorRuntimeFrameDisposition::Ignore};
};

[[nodiscard]] constexpr PpDepthPredecessorCrossScopeRuntimeOutcome
MakePpDepthPredecessorCrossScopeRuntimeOutcome(
    const PpDepthPredecessorCrossScopePairResult& result,
    PpDepthPredecessorRuntimeFrameDisposition frame_disposition,
    bool source_plane_recorded, bool target_plane_recorded,
    PpDepthPredecessorCrossScopeLearningEvidence learning = {},
    PpDepthPredecessorCrossScopeLearningLogicalCandidateSummary
        learning_logical_candidates = {},
    PpDepthPredecessorCrossScopeLearningTransitionCandidateSummary
        learning_transition_candidates = {},
    PpDepthPredecessorCrossScopeBootstrapRuntimeEvidence bootstrap = {},
    PpDepthPredecessorCrossScopeArmedTargetPreEvidenceSummary
        armed_target_pre_evidence = {},
    PpDepthPredecessorCrossScopeSourceRestartEvidence
        source_restart_evidence = {},
    PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence
        selected_latest_evidence = {},
    PpDepthPredecessorCrossScopeMode3RootBindingEvidence
        root_binding_evidence = {},
    PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceSummary
        first_tail_pre_drift_evidence = {},
    PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceSummary
        armed_p0_repeat_probe_evidence = {}) noexcept {
    return {
        .result = result,
        .phase = result.phase,
        .role = result.role,
        .decision = result.decision,
        .bootstrap = bootstrap,
        .expected_tail_occurrences = result.expected_tail_occurrences,
        .observed_tail_occurrences = result.observed_tail_occurrences,
        .source_post_count = result.source_post_count,
        .target_pre_count = result.target_pre_count,
        .boundary_exact = result.boundary_exact,
        .learning = learning,
        .learning_logical_candidates = learning_logical_candidates,
        .learning_transition_candidates = learning_transition_candidates,
        .armed_target_pre_evidence = armed_target_pre_evidence,
        .source_restart_evidence = source_restart_evidence,
        .selected_latest_evidence = selected_latest_evidence,
        .root_binding_evidence = root_binding_evidence,
        .first_tail_pre_drift_evidence = first_tail_pre_drift_evidence,
        .armed_p0_repeat_probe_evidence = armed_p0_repeat_probe_evidence,
        .target_first_writer_exact = result.target_first_writer_exact,
        .tail_exact = result.tail_exact,
        .candidate_saved = result.candidate_saved,
        .committed = result.committed,
        .select = result.select,
        .source_plane_recorded = source_plane_recorded,
        .target_plane_recorded = target_plane_recorded,
        .frame_disposition = frame_disposition,
    };
}

[[nodiscard]] inline std::string FormatPpDepthPredecessorCrossScopeRuntimeOutcome(
    const PpDepthPredecessorCrossScopeRuntimeOutcome& outcome) {
    const auto flag = [](bool value) { return value ? '1' : '0'; };
    std::string text{"FGSCTUH4 ph="};
    text += std::to_string(static_cast<u32>(outcome.phase));
    text += " ro=";
    text += std::to_string(static_cast<u32>(outcome.role));
    text += " d=";
    const auto decision =
        outcome.result.decision != PpDepthPredecessorCrossScopePairDecision::Idle
            ? outcome.result.decision
            : outcome.decision;
    text += std::to_string(static_cast<u32>(decision));
    text += " m3p=";
    text += std::to_string(static_cast<u32>(outcome.bootstrap.begin_phase));
    text += " m3d=";
    text += std::to_string(static_cast<u32>(outcome.bootstrap.begin_decision));
    text += " m3k=";
    text += std::to_string(static_cast<u32>(outcome.bootstrap.begin_chain_kind));
    text += " m3r=";
    text += std::to_string(static_cast<u32>(outcome.bootstrap.begin_occurrence_role));
    text += " m3a=";
    text += flag(outcome.bootstrap.begin_raw_armed);
    text += " bs=";
    text += std::to_string(static_cast<u32>(outcome.bootstrap.readiness));
    text += " m3e=";
    text += std::to_string(static_cast<u32>(outcome.bootstrap.end_decision));
    const auto& selected_proof = outcome.bootstrap.selected_proof;
    text += " msq=";
    text += std::to_string(static_cast<u32>(
        SanitizePpDepthPredecessorCrossScopeMode3SelectedProofReason(
            selected_proof.reason)));
    text += " msa=";
    text += flag(selected_proof.attempted);
    text += " msp=";
    text += flag(selected_proof.phase_exact);
    text += " msd=";
    text += flag(selected_proof.decision_exact);
    text += " msk=";
    text += flag(selected_proof.chain_exact);
    text += " msr=";
    text += flag(selected_proof.occurrence_role_exact);
    text += " msg=";
    text += flag(selected_proof.generation_continuity_exact);
    text += " mse=";
    text += flag(selected_proof.expected_continuity_exact);
    text += " mso=";
    text += flag(selected_proof.observed_cardinality_exact);
    text += " msw=";
    text += flag(selected_proof.raw_cardinality_exact);
    text += " msb=";
    text += flag(selected_proof.bridge_cardinality_exact);
    text += " msi=";
    text += flag(selected_proof.collision_exact);
    text += " msf=";
    text += flag(selected_proof.proof_flags_exact);
    text += " msc=";
    text += flag(selected_proof.candidate_exact);
    text += " mss=";
    text += flag(selected_proof.select_exact);
    text += " msx=";
    text += flag(selected_proof.exact);
    text += " msn=";
    text += std::to_string(selected_proof.expected_occurrences);
    text += "/";
    text += std::to_string(selected_proof.observed_occurrences);
    text += "/";
    text += std::to_string(selected_proof.raw_occurrences);
    text += "/";
    text += std::to_string(selected_proof.bridge_occurrences);
    const auto& selected_latest = outcome.selected_latest_evidence;
    text += " mlq=";
    text += std::to_string(static_cast<u32>(
        SanitizePpDepthPredecessorCrossScopeMode3SelectedLatestBindingReason(
            selected_latest.reason)));
    text += " mlu=";
    text += flag(selected_latest.attempted);
    const bool selected_latest_counts_bounded =
        selected_latest.raw_occurrences != std::numeric_limits<u32>::max() &&
        selected_latest.restart_attempt_count != std::numeric_limits<u32>::max() &&
        selected_latest.restart_completion_count != std::numeric_limits<u32>::max();
    text += " mln=";
    text += std::to_string(selected_latest_counts_bounded
                               ? selected_latest.raw_occurrences
                               : 0);
    text += "/";
    text += std::to_string(selected_latest_counts_bounded
                               ? selected_latest.restart_attempt_count
                               : 0);
    text += "/";
    text += std::to_string(selected_latest_counts_bounded
                               ? selected_latest.restart_completion_count
                               : 0);
    text += " mlr=";
    text += flag(selected_latest.raw_rolling_bounded);
    text += " mlc=";
    text += flag(selected_latest.selected_core_without_raw_exact);
    text += " mlk=";
    text += flag(selected_latest.rolling_counts_exact);
    text += " mlw=";
    text += flag(selected_latest.latest_witness_exact);
    text += " mls=";
    text += flag(selected_latest.runtime_selection_exact);
    text += " mlg=";
    text += flag(selected_latest.generation_exact);
    text += " mli=";
    text += flag(selected_latest.candidate_index_exact);
    text += " mlm=";
    text += flag(selected_latest.candidate_mask_exact);
    text += " mla=";
    text += flag(selected_latest.actual_selected_latest_exact);
    text += " mlp=";
    text += flag(selected_latest.latest_p0_post_exact);
    text += " mlt=";
    text += flag(selected_latest.intent_p0_post_exact);
    text += " mll=";
    text += flag(selected_latest.logical_summary_exact);
    text += " mlz=";
    text += flag(selected_latest.transition_summary_exact);
    text += " mlh=";
    text += flag(selected_latest.lifecycle_learning_exact);
    text += " mlx=";
    text += flag(selected_latest.exact);
    const auto& root_binding = outcome.root_binding_evidence;
    text += " mrq=";
    text += std::to_string(static_cast<u32>(
        SanitizePpDepthPredecessorCrossScopeMode3RootBindingReason(
            root_binding.reason)));
    text += " mra=";
    text += flag(root_binding.attempted);
    text += " mrw=";
    text += flag(root_binding.root_witness_exact);
    text += " mrp=";
    text += flag(root_binding.cross_state_present);
    text += " mrr=";
    text += flag(root_binding.cross_root_exact);
    text += " mrk=";
    text += flag(root_binding.chain_exact);
    text += " mrs=";
    text += flag(root_binding.latest_selection_exact);
    text += " mrl=";
    text += flag(root_binding.lifecycle_exact);
    text += " mrx=";
    text += flag(root_binding.exact);
    text += " tc=";
    text += std::to_string(outcome.expected_tail_occurrences);
    text += '/';
    text += std::to_string(outcome.observed_tail_occurrences);
    text += " ep=";
    text += std::to_string(outcome.source_post_count);
    text += '/';
    text += std::to_string(outcome.target_pre_count);
    text += " bx=";
    text += flag(outcome.boundary_exact);
    text += " rb=";
    text += flag(outcome.learning.raw_boundary_exact);
    text += " bg=";
    text += flag(outcome.learning.boundary_before_guest_exact);
    text += " bh=";
    text += flag(outcome.learning.boundary_before_host_exact);
    text += " ba=";
    text += flag(outcome.learning.boundary_before_rendering_active);
    text += " br=";
    text += flag(outcome.learning.boundary_before_regular_origin_exact);
    text += " bo=";
    text += flag(outcome.learning.boundary_before_source_origin_exact);
    text += " bd=";
    text += flag(
        outcome.learning.boundary_before_diagnostic_descendant_exact);
    text += " bl=";
    text += flag(outcome.learning.boundary_before_source_lineage_exact);
    text += " lc=";
    text += std::to_string(outcome.learning_logical_candidates.count);
    text += " lo=";
    text += flag(outcome.learning_logical_candidates.count_overflow);
    text += " lu=";
    text += flag(outcome.learning_logical_candidates.unique_valid);
    text += " lrb=";
    text += flag(outcome.learning_logical_candidates.raw_boundary_exact);
    text += " lbg=";
    text += flag(
        outcome.learning_logical_candidates.boundary_before_guest_exact);
    text += " lbh=";
    text += flag(
        outcome.learning_logical_candidates.boundary_before_host_exact);
    text += " lba=";
    text += flag(
        outcome.learning_logical_candidates.boundary_before_rendering_active);
    text += " lbr=";
    text += flag(outcome.learning_logical_candidates
                     .boundary_before_regular_origin_exact);
    text += " lbo=";
    text += flag(outcome.learning_logical_candidates
                     .boundary_before_source_origin_exact);
    text += " lbd=";
    text += flag(outcome.learning_logical_candidates
                     .boundary_before_diagnostic_descendant_exact);
    text += " lbl=";
    text += flag(outcome.learning_logical_candidates
                     .boundary_before_source_lineage_exact);
    text += " xc=";
    text += std::to_string(outcome.learning_transition_candidates.count);
    text += " xo=";
    text += flag(outcome.learning_transition_candidates.count_overflow);
    text += " xu=";
    text += flag(outcome.learning_transition_candidates.unique_valid);
    text += " xsv=";
    text += flag(outcome.learning_transition_candidates.join_discriminator
                     .source_structural_valid);
    text += " xad=";
    text += flag(outcome.learning_transition_candidates.join_discriminator
                     .source_after_diagnostic_resume);
    text += " xtv=";
    text += flag(outcome.learning_transition_candidates.join_discriminator
                     .target_intent_valid);
    text += " xle=";
    text += flag(outcome.learning_transition_candidates.join_discriminator
                     .source_target_link_exact);
    text += " xre=";
    text += flag(outcome.learning_transition_candidates.join_discriminator
                     .source_target_range_exact);
    text += " xbe=";
    text += flag(outcome.learning_transition_candidates.join_discriminator
                     .source_target_backing_exact);
    text += " xme=";
    text += flag(outcome.learning_transition_candidates.join_discriminator
                     .source_target_mutation_exact);
    text += " xci=";
    text += flag(outcome.learning_transition_candidates.join_discriminator
                     .raw_current_matches_source);
    text += " xbo=";
    text += flag(outcome.learning_transition_candidates
                     .boundary_before_source_origin_exact);
    text += " xbd=";
    text += flag(outcome.learning_transition_candidates
                     .boundary_before_diagnostic_descendant_exact);
    text += " xbl=";
    text += flag(outcome.learning_transition_candidates
                     .boundary_before_source_lineage_exact);
    const auto& armed = outcome.armed_target_pre_evidence;
    const auto& armed_unique = armed.unique_evidence;
    text += " ac=";
    text += std::to_string(armed.attempt_count);
    text += " ao=";
    text += flag(armed.count_overflow);
    text += " au=";
    text += flag(armed.unique_valid);
    text += " awv=";
    text += flag(armed_unique.witness_valid);
    text += " ara=";
    text += flag(armed_unique.raw_boundary_active);
    text += " art=";
    text += flag(armed_unique.raw_boundary_tracked);
    text += " awa=";
    text += flag(armed_unique.witness_resume_active);
    text += " awd=";
    text += flag(armed_unique.witness_resume_diagnostic_origin);
    text += " aca=";
    text += flag(armed_unique.checkpoint_resume_active);
    text += " acd=";
    text += flag(armed_unique.checkpoint_resume_diagnostic_origin);
    text += " age=";
    text += flag(armed_unique.resume_guest_exact);
    text += " ahe=";
    text += flag(armed_unique.resume_host_exact);
    text += " aae=";
    text += flag(armed_unique.resume_activity_exact);
    text += " aoe=";
    text += flag(armed_unique.resume_origin_exact);
    text += " amx=";
    text += flag(armed_unique.resume_metadata_without_activity_exact);
    text += " agp=";
    text += flag(armed_unique.target_guest_plus_one_exact);
    text += " ahn=";
    text += flag(armed_unique.target_host_nonzero);
    text += " ahs=";
    text += flag(armed_unique.target_host_diff_source);
    text += " ahr=";
    text += flag(armed_unique.target_host_diff_resume);
    text += " abg=";
    text += flag(armed_unique.boundary_live_guest_equal);
    text += " abh=";
    text += flag(armed_unique.boundary_live_host_equal);
    text += " aba=";
    text += flag(armed_unique.boundary_live_active_equal);
    text += " abo=";
    text += flag(armed_unique.boundary_live_origin_equal);
    text += " ata=";
    text += flag(armed_unique.target_after_active);
    text += " ala=";
    text += flag(armed_unique.live_active);
    text += " atr=";
    text += flag(armed_unique.target_regular);
    text += " ant=";
    text += flag(armed_unique.target_not_diagnostic);
    text += " aso=";
    text += flag(armed_unique.source_outside_target_scope);
    text += " aab=";
    text += flag(armed_unique.after_begin);
    text += " arx=";
    text += flag(armed_unique.raw_resume_exact);
    text += " atx=";
    text += flag(armed_unique.raw_target_exact);
    text += " ajx=";
    text += flag(armed_unique.raw_join_exact);
    text += " abx=";
    text += flag(armed_unique.builder_exact);
    text += " aep=";
    text += flag(armed_unique.classifier_endpoint_exact);
    text += " acc=";
    text += flag(armed_unique.classifier_candidate_exact);
    text += " acr=";
    text += flag(armed_unique.classifier_resume_exact);
    text += " ast=";
    text += flag(armed_unique.classifier_scope_transition_exact);
    text += " ald=";
    text += flag(armed_unique.classifier_load_exact);
    text += " acx=";
    text += flag(armed_unique.classifier_exact);
    text += " als=";
    text += flag(armed_unique.lifecycle_stored_source_exact);
    const auto& restart = outcome.source_restart_evidence;
    text += " srq=";
    text += std::to_string(static_cast<u32>(
        SanitizePpDepthPredecessorCrossScopeSourceRestartReason(restart.reason)));
    text += " srn=";
    text += std::to_string(restart.attempt_count);
    text += "/";
    text += std::to_string(restart.completion_count);
    text += " sro=";
    text += flag(restart.count_overflow);
    text += " srv=";
    text += flag(restart.latest_valid);
    text += " srp=";
    text += flag(restart.pre_exact);
    text += " sre=";
    text += flag(restart.endpoint_exact);
    text += " src=";
    text += flag(restart.candidate_exact);
    text += " srr=";
    text += flag(restart.resume_exact);
    text += " srs=";
    text += flag(restart.retained_slot_exact);
    text += " srf=";
    text += flag(restart.copy_mode_exact);
    text += " srx=";
    text += flag(restart.exact);
    const auto& tail_drift_summary = outcome.first_tail_pre_drift_evidence;
    const bool tail_drift_summary_bounded =
        tail_drift_summary.attempt_count != std::numeric_limits<u32>::max() &&
        (!tail_drift_summary.count_overflow ||
         !tail_drift_summary.unique_valid) &&
        (!tail_drift_summary.unique_valid ||
         tail_drift_summary.attempt_count == 1);
    const auto bounded_tail_drift_summary =
        tail_drift_summary_bounded
            ? tail_drift_summary
            : PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceSummary{};
    const auto tail_drift = bounded_tail_drift_summary.unique_valid
                                ? bounded_tail_drift_summary.unique_evidence
                                : PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidence{};
    text += " tdq=";
    text += std::to_string(static_cast<u32>(
        SanitizePpDepthPredecessorCrossScopeFirstTailPreP0DriftReason(
            tail_drift.reason)));
    text += " tdn=";
    text += std::to_string(bounded_tail_drift_summary.attempt_count);
    text += " tdo=";
    text += flag(bounded_tail_drift_summary.count_overflow);
    text += " tdu=";
    text += flag(bounded_tail_drift_summary.unique_valid);
    text += " tdm=";
    text += flag(tail_drift.attempted);
    text += " tdl=";
    text += flag(tail_drift.lifecycle_ready);
    text += " tds=";
    text += flag(tail_drift.stored_valid);
    text += " tdc=";
    text += flag(tail_drift.current_valid);
    text += " tdp=";
    text += flag(tail_drift.stored_target_hint_exact);
    text += " tdh=";
    text += flag(tail_drift.current_target_hint_exact);
    text += " tdi=";
    text += flag(tail_drift.identity_exact);
    text += " tdsg=";
    text += flag(tail_drift.stored_current_guest_equal);
    text += " tdsh=";
    text += flag(tail_drift.stored_current_host_equal);
    text += " tdlg=";
    text += flag(tail_drift.current_live_guest_equal);
    text += " tdlh=";
    text += flag(tail_drift.current_live_host_equal);
    text += " tdlv=";
    text += flag(tail_drift.live_active);
    text += " tdlr=";
    text += flag(tail_drift.live_regular);
    text += " tdln=";
    text += flag(tail_drift.live_non_diagnostic);
    text += " tdsn=";
    text += flag(tail_drift.stored_endpoint_non_diagnostic);
    text += " tdcn=";
    text += flag(tail_drift.current_endpoint_non_diagnostic);
    text += " tdsu=";
    text += flag(tail_drift.stored_current_guest_successor);
    text += " tdhc=";
    text += flag(tail_drift.stored_current_host_changed);
    text += " tdlu=";
    text += flag(tail_drift.current_live_guest_successor);
    text += " tdlc=";
    text += flag(tail_drift.current_live_host_changed);
    text += " tdg=";
    text += flag(tail_drift.scope_exact);
    text += " tdt=";
    text += flag(tail_drift.intent_exact);
    text += " tde=";
    text += flag(tail_drift.mutation_equal);
    text += " tdr=";
    text += flag(tail_drift.mutation_successor);
    text += " tdw=";
    text += flag(tail_drift.mutation_newer);
    text += " tda=";
    text += flag(tail_drift.all_except_prior_exact);
    text += " tdx=";
    text += flag(tail_drift.stale_target_exact);
    const auto& repeat_summary = outcome.armed_p0_repeat_probe_evidence;
    const bool repeat_summary_bounded =
        repeat_summary.attempt_count <= 1 &&
        (!repeat_summary.count_overflow || !repeat_summary.unique_valid) &&
        (!repeat_summary.unique_valid || repeat_summary.attempt_count == 1) &&
        (!repeat_summary.unique_valid ||
         (repeat_summary.unique_evidence.expected_repeat_count != 0 &&
          repeat_summary.unique_evidence.expected_repeat_count !=
              std::numeric_limits<u32>::max() &&
          repeat_summary.unique_evidence.attempt_count <=
              repeat_summary.unique_evidence.expected_repeat_count &&
          repeat_summary.unique_evidence.completion_count <=
              repeat_summary.unique_evidence.expected_repeat_count));
    const auto bounded_repeat_summary =
        repeat_summary_bounded
            ? repeat_summary
            : PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceSummary{};
    const auto repeat = bounded_repeat_summary.unique_valid
                            ? bounded_repeat_summary.unique_evidence
                            : PpDepthPredecessorCrossScopeArmedP0RepeatProbeResult{};
    text += " prq=";
    text += std::to_string(static_cast<u32>(
        SanitizePpDepthPredecessorCrossScopeArmedP0RepeatProbeReason(
            repeat.reason)));
    text += " prn=";
    text += std::to_string(repeat.expected_repeat_count);
    text += "/";
    text += std::to_string(repeat.attempt_count);
    text += "/";
    text += std::to_string(repeat.completion_count);
    text += " pra=";
    text += flag(repeat.attempted);
    text += " pro=";
    text += flag(bounded_repeat_summary.count_overflow);
    text += " pru=";
    text += flag(bounded_repeat_summary.unique_valid);
    text += " prr=";
    text += flag(repeat.root_exact);
    text += " prl=";
    text += flag(repeat.lifecycle_exact);
    text += " prds=";
    text += flag(repeat.stored_target_post_after_diagnostic);
    text += " prb=";
    text += flag(repeat.budget_exact);
    text += " prs=";
    text += flag(repeat.stored_latest_exact);
    text += " prw=";
    text += flag(repeat.current_writer_exact);
    text += " pri=";
    text += flag(repeat.intent_exact);
    text += " pry=";
    text += flag(repeat.pre_identity_exact);
    text += " prg=";
    text += flag(repeat.pre_scope_exact);
    text += " prc=";
    text += flag(repeat.candidate_exact);
    text += " prx=";
    text += flag(repeat.open_state_exact);
    text += " prk=";
    text += flag(repeat.post_correlation_exact);
    text += " prj=";
    text += flag(repeat.post_writer_successor);
    text += " prp=";
    text += flag(repeat.post_scope_exact);
    text += " prf=";
    text += flag(repeat.first_writer_exact);
    text += " prv=";
    text += flag(repeat.candidate_preserved);
    text += " prh=";
    text += flag(repeat.slot_preserved);
    text += " pre=";
    text += flag(repeat.latest_current_exact);
    text += " prt=";
    text += flag(repeat.tail_intent_exact);
    text += " prd=";
    text += flag(repeat.counts_exact);
    text += " prz=";
    text += flag(repeat.exact);
    text += " rj=";
    text += flag(outcome.learning.raw_join_exact);
    text += " lt=";
    text += flag(outcome.learning.live_target_exact);
    text += " lp=";
    text += flag(outcome.learning.predecessor_exact);
    text += " lj=";
    text += flag(outcome.learning.join_exact);
    text += " lr=";
    text += flag(outcome.learning.regular_cross_scope_exact);
    text += " ll=";
    text += flag(outcome.learning.load_exact);
    text += " fw=";
    text += flag(outcome.target_first_writer_exact);
    text += " tx=";
    text += flag(outcome.tail_exact);
    text += " cs=";
    text += flag(outcome.candidate_saved);
    text += " cm=";
    text += flag(outcome.committed);
    text += " sel=";
    text += flag(outcome.select);
    text += " sp=";
    text += flag(outcome.source_plane_recorded);
    text += " tp=";
    text += flag(outcome.target_plane_recorded);
    text += " fd=";
    text += std::to_string(static_cast<u32>(outcome.frame_disposition));
    return text;
}

enum class PpDepthPredecessorHintDecision : u32 {
    Idle,
    Warmup,
    Armed,
    Selected,
    HintMismatch,
    CandidateMismatch,
    MissingAnchor,
    DuplicateAnchor,
    GenerationWrap,
};

struct PpDepthPredecessorHintResult {
    PpDepthPredecessorHintDecision decision{PpDepthPredecessorHintDecision::Idle};
    u64 generation{};
    u32 candidate_index{};
    bool armed{};
    bool select{};
};

class PpDepthPredecessorHintLifecycle {
public:
    explicit constexpr PpDepthPredecessorHintLifecycle(u32 candidate_capacity,
                                                       u64 initial_generation = 0) noexcept
        : capacity{candidate_capacity}, generation{initial_generation} {}

    [[nodiscard]] constexpr bool LearnNext(const PpTerminalScopeDrawSelector& draw,
                                           u64 fragment_pgm_hash) noexcept {
        const PpDepthPredecessorHint candidate{
            .draw = draw,
            .fragment_pgm_hash = fragment_pgm_hash,
            .valid = true,
        };
        if (!candidate.Valid() || capacity == 0 ||
            capacity > PpUpstreamFeedbackPrePostPlan::MaxCandidates) {
            next_hint = {};
            return false;
        }
        next_hint = candidate;
        return true;
    }

    [[nodiscard]] constexpr PpDepthPredecessorHintResult BeginFrame() noexcept {
        if (frame_open) {
            next_hint = {};
            armed_hint = {};
            selected = false;
            last_decision = PpDepthPredecessorHintDecision::MissingAnchor;
            return MakeResult();
        }
        frame_open = true;
        anchor_seen = false;
        selected = false;
        selected_index = 0;
        armed_hint = {};
        last_decision = PpDepthPredecessorHintDecision::Warmup;

        const auto promoted = next_hint;
        next_hint = {};
        if (!promoted.Valid()) {
            return MakeResult();
        }
        if (generation == std::numeric_limits<u64>::max()) {
            frame_open = false;
            last_decision = PpDepthPredecessorHintDecision::GenerationWrap;
            return MakeResult();
        }
        ++generation;
        armed_hint = promoted;
        last_decision = PpDepthPredecessorHintDecision::Armed;
        return MakeResult();
    }

    [[nodiscard]] constexpr bool MatchesArmed(const PpTerminalScopeDrawSelector& draw,
                                              u64 fragment_pgm_hash) const noexcept {
        const PpDepthPredecessorHint observed{
            .draw = draw,
            .fragment_pgm_hash = fragment_pgm_hash,
            .valid = true,
        };
        return frame_open && armed_hint.Valid() && observed.Valid() && observed == armed_hint;
    }

    [[nodiscard]] constexpr const PpDepthPredecessorHint* PeekArmedHint() const noexcept {
        return frame_open && armed_hint.Valid() ? &armed_hint : nullptr;
    }

    [[nodiscard]] constexpr PpDepthPredecessorHintResult ObserveAnchor(
        const PpTerminalScopeDrawSelector& current_draw, u64 current_fragment_pgm_hash,
        PpDepthPredecessorCandidateStamp stamp) noexcept {
        if (!frame_open) {
            selected = false;
            last_decision = PpDepthPredecessorHintDecision::MissingAnchor;
            return MakeResult();
        }
        if (anchor_seen) {
            next_hint = {};
            selected = false;
            last_decision = PpDepthPredecessorHintDecision::DuplicateAnchor;
            return MakeResult();
        }
        anchor_seen = true;

        const bool learned = LearnNext(current_draw, current_fragment_pgm_hash);
        if (!armed_hint.Valid()) {
            last_decision = learned ? PpDepthPredecessorHintDecision::Warmup
                                    : PpDepthPredecessorHintDecision::HintMismatch;
            return MakeResult();
        }
        if (!learned || next_hint != armed_hint) {
            last_decision = PpDepthPredecessorHintDecision::HintMismatch;
            return MakeResult();
        }
        if (!stamp.valid || stamp.generation == 0 || stamp.generation != generation ||
            stamp.candidate_index >= capacity || stamp.recorded_plane_mask != 0x3u) {
            last_decision = PpDepthPredecessorHintDecision::CandidateMismatch;
            return MakeResult();
        }
        selected_index = stamp.candidate_index;
        selected = true;
        last_decision = PpDepthPredecessorHintDecision::Selected;
        return MakeResult();
    }

    [[nodiscard]] constexpr PpDepthPredecessorHintResult EndFrame() noexcept {
        if (!frame_open) {
            return MakeResult();
        }
        if (!anchor_seen) {
            next_hint = {};
            selected = false;
            last_decision = PpDepthPredecessorHintDecision::MissingAnchor;
        }
        const auto result = MakeResult();
        frame_open = false;
        armed_hint = {};
        selected = false;
        return result;
    }

private:
    [[nodiscard]] constexpr PpDepthPredecessorHintResult MakeResult() const noexcept {
        return {
            .decision = last_decision,
            .generation = armed_hint.Valid() ? generation : 0,
            .candidate_index = selected ? selected_index : 0,
            .armed = armed_hint.Valid(),
            .select = selected,
        };
    }

    u32 capacity{};
    u64 generation{};
    PpDepthPredecessorHint armed_hint{};
    PpDepthPredecessorHint next_hint{};
    PpDepthPredecessorHintDecision last_decision{PpDepthPredecessorHintDecision::Idle};
    u32 selected_index{};
    bool frame_open{};
    bool anchor_seen{};
    bool selected{};
};

} // namespace Vulkan::HostPasses
