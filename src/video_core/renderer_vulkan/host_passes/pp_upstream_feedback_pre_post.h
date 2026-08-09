// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <limits>

#include "video_core/renderer_vulkan/host_passes/pp_terminal_scope_content.h"

namespace Vulkan::HostPasses {

struct PpUpstreamFeedbackPrePostDescriptor {
    bool enabled{};
    u32 logical_width{};
    u32 logical_height{};
    u32 source_width{};
    u32 source_height{};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    u32 samples{};
    u32 base_mip{};
    u32 mip_count{};
    u32 base_layer{};
    u32 layer_count{};
    bool color{};
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
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy{};
};

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
    const auto reject = [](FinalGuestSurfaceStatus status,
                           u32 FinalGuestSurfaceLoss::* member) {
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
    if ((descriptor.format != FinalGuestSurfaceFormat::Rgba8 &&
         descriptor.format != FinalGuestSurfaceFormat::Bgra8) ||
        descriptor.samples != 1 || descriptor.base_mip != 0 || descriptor.mip_count != 1 ||
        descriptor.base_layer != 0 || descriptor.layer_count != 1 || !descriptor.color ||
        !descriptor.type_2d || !descriptor.uniform_state) {
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
            : (descriptor.logical_width - window_width) /
                      FinalGuestSurfaceTilePlan::WindowStride +
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
                      descriptor.candidate_capacity >
                              PpUpstreamFeedbackPrePostPlan::MaxCandidates
                          ? &FinalGuestSurfaceLoss::byte_capacity
                          : &FinalGuestSurfaceLoss::tile_capacity);
    }

    result.region_count = selector.count;
    result.candidate_capacity = descriptor.candidate_capacity;
    result.format = descriptor.format;
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
        const u64 bytes = static_cast<u64>(end_x - x) * (end_y - y) * 4;
        if (next_offset + bytes > std::numeric_limits<u32>::max()) {
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
        candidate_stride == std::numeric_limits<u64>::max() ||
        total_bytes > descriptor.max_bytes || total_bytes > std::numeric_limits<u32>::max()) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss,
                      &FinalGuestSurfaceLoss::byte_capacity);
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

struct PpUpstreamFeedbackPrePostRegistryAction {
    u32 candidate_index{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool capture{};
};

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
        : selector{selector_}, capacity{capacity_} {}

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
        return {.status = FinalGuestSurfaceStatus::CapacityLoss,
                .loss = {.tile_capacity = 1}};
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
            return {.status = FinalGuestSurfaceStatus::CapacityLoss,
                    .loss = {.tile_capacity = 1}};
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
        entries = {};
        capacity_loss = false;
    }

private:
    struct Entry {
        VideoCore::ImageColorScopePrivateLink output{};
        bool active{};
        bool before{};
        bool after{};
    };

    PpTerminalScopeDrawSelector selector{};
    u32 capacity{};
    std::array<Entry, PpUpstreamFeedbackPrePostPlan::MaxCandidates> entries{};
    bool capacity_loss{};
};

} // namespace Vulkan::HostPasses
