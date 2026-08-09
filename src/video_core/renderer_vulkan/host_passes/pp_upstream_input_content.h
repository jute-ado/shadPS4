// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/host_passes/pp_terminal_scope_content.h"

namespace Vulkan::HostPasses {

inline constexpr u32 PpUpstreamInputSnapshotBytes = 4u << 20;

struct PpUpstreamInputContentView {
    u32 width{};
    u32 height{};
    u32 bytes_per_texel{};
    u32 base_mip{};
    u32 mip_count{};
    u32 base_layer{};
    u32 layer_count{1};
    bool color{};
    bool type_2d{};
    bool uniform_state{};
    bool view_conflict{};
    bool aliases_output{};
};

struct PpUpstreamInputContentDescriptor {
    bool enabled{};
    u32 logical_width{};
    u32 logical_height{};
    u32 base_offset{};
    FinalGuestSurfaceWatchOrdinals selector{};
    std::span<const PpUpstreamInputContentView> views{};
    u32 buffer_alignment{};
    u32 max_regions{};
    u32 max_bytes{};
};

struct PpUpstreamInputContentRegion {
    u32 logical_ordinal{};
    u32 mip_level{};
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};
    u32 buffer_offset{};
    u32 byte_size{};
};

struct PpUpstreamInputContentPlane {
    static constexpr u32 MaxMips = 16;
    static constexpr u32 MaxRegions = FinalGuestSurfaceWatchOrdinals::MaxOrdinals * MaxMips;

    std::array<PpUpstreamInputContentRegion, MaxRegions> regions{};
    u32 region_count{};
    u32 plane_offset{};
    u32 plane_bytes{};
    u32 bytes_per_texel{};
    u32 base_mip{};
    u32 mip_count{};
    u32 base_layer{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
};

struct PpUpstreamInputContentPlan {
    static constexpr u32 MaxInputs = PpTerminalScopeSampledInputs::MaxInputs;

    std::array<PpUpstreamInputContentPlane, MaxInputs> inputs{};
    u32 input_count{};
    u32 capture_mask{};
    u32 unavailable_mask{};
    u32 alias_mask{};
    u32 copy_region_count{};
    u32 total_bytes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy{};
    bool cpu_wait{};
    bool finish{};
    bool retains_image{};
    bool retains_vk_image{};
};

[[nodiscard]] constexpr bool IsPpUpstreamInputBytesPerTexelSupported(u32 value) noexcept {
    return value == 1 || value == 2 || value == 4 || value == 8 || value == 16;
}

[[nodiscard]] inline PpUpstreamInputContentPlan PlanPpUpstreamInputContent(
    const PpUpstreamInputContentDescriptor& descriptor) noexcept {
    PpUpstreamInputContentPlan result{};
    if (!descriptor.enabled) {
        return result;
    }
    const auto reject = [](FinalGuestSurfaceStatus status, u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamInputContentPlan rejected{};
        rejected.status = status;
        rejected.loss.*member = 1;
        return rejected;
    };
    if (descriptor.logical_width == 0 || descriptor.logical_height == 0 ||
        descriptor.views.empty() || descriptor.views.size() > result.MaxInputs ||
        descriptor.buffer_alignment == 0) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
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

    result.input_count = static_cast<u32>(descriptor.views.size());
    u64 next_offset{descriptor.base_offset};
    for (u32 input_index = 0; input_index < result.input_count; ++input_index) {
        const auto& view = descriptor.views[input_index];
        auto& input = result.inputs[input_index];
        input.bytes_per_texel = view.bytes_per_texel;
        input.base_mip = view.base_mip;
        input.mip_count = view.mip_count;
        input.base_layer = view.base_layer;
        if (view.aliases_output) {
            result.alias_mask |= 1u << input_index;
            input.status = FinalGuestSurfaceStatus::Complete;
            continue;
        }
        const auto unavailable = [&](FinalGuestSurfaceStatus status,
                                     u32 FinalGuestSurfaceLoss::* member) {
            input.status = status;
            input.loss.*member = 1;
            result.unavailable_mask |= 1u << input_index;
        };
        if (view.width == 0 || view.height == 0) {
            unavailable(FinalGuestSurfaceStatus::InvalidationLoss,
                        &FinalGuestSurfaceLoss::invalid_extent);
            continue;
        }
        if (!view.type_2d) {
            unavailable(FinalGuestSurfaceStatus::Unsupported,
                        &FinalGuestSurfaceLoss::unsupported_type);
            continue;
        }
        if (!view.color) {
            unavailable(FinalGuestSurfaceStatus::Unsupported,
                        &FinalGuestSurfaceLoss::unsupported_aspect);
            continue;
        }
        if (!IsPpUpstreamInputBytesPerTexelSupported(view.bytes_per_texel)) {
            unavailable(FinalGuestSurfaceStatus::Unsupported,
                        &FinalGuestSurfaceLoss::unsupported_format);
            continue;
        }
        if (view.mip_count == 0 || view.mip_count > input.MaxMips) {
            unavailable(FinalGuestSurfaceStatus::Unsupported,
                        &FinalGuestSurfaceLoss::unsupported_mip);
            continue;
        }
        if (view.layer_count != 1) {
            unavailable(FinalGuestSurfaceStatus::Unsupported,
                        &FinalGuestSurfaceLoss::unsupported_layer);
            continue;
        }
        if (!view.uniform_state || view.view_conflict) {
            unavailable(FinalGuestSurfaceStatus::InvalidationLoss,
                        &FinalGuestSurfaceLoss::invalidation);
            continue;
        }
        const u64 input_regions = static_cast<u64>(selector.count) * view.mip_count;
        if (input_regions > input.MaxRegions ||
            static_cast<u64>(result.copy_region_count) + input_regions > descriptor.max_regions) {
            return reject(FinalGuestSurfaceStatus::CapacityLoss,
                          &FinalGuestSurfaceLoss::tile_capacity);
        }
        next_offset = AlignPpSourceBackingOffset(next_offset, descriptor.buffer_alignment);
        if (next_offset == std::numeric_limits<u64>::max() ||
            next_offset > std::numeric_limits<u32>::max()) {
            return reject(FinalGuestSurfaceStatus::CapacityLoss,
                          &FinalGuestSurfaceLoss::byte_capacity);
        }
        input.plane_offset = static_cast<u32>(next_offset);
        u64 local_offset{};
        for (u32 selector_index = 0; selector_index < selector.count; ++selector_index) {
            const u32 zero_based = selector.ordinals[selector_index] - 1;
            const u32 logical_x = (zero_based % columns) * FinalGuestSurfaceTilePlan::WindowStride;
            const u32 logical_y = (zero_based / columns) * FinalGuestSurfaceTilePlan::WindowStride;
            const u32 logical_end_x = logical_x + window_width;
            const u32 logical_end_y = logical_y + window_height;
            for (u32 mip = 0; mip < view.mip_count; ++mip) {
                const u32 width = std::max(1u, view.width >> mip);
                const u32 height = std::max(1u, view.height >> mip);
                const auto map_floor = [](u32 value, u32 extent, u32 logical) {
                    return static_cast<u32>(static_cast<u64>(value) * extent / logical);
                };
                const auto map_ceil = [](u32 value, u32 extent, u32 logical) {
                    return static_cast<u32>((static_cast<u64>(value) * extent + logical - 1) /
                                            logical);
                };
                const u32 x =
                    std::min(map_floor(logical_x, width, descriptor.logical_width), width - 1);
                const u32 y =
                    std::min(map_floor(logical_y, height, descriptor.logical_height), height - 1);
                const u32 end_x = std::clamp(
                    map_ceil(logical_end_x, width, descriptor.logical_width), x + 1, width);
                const u32 end_y = std::clamp(
                    map_ceil(logical_end_y, height, descriptor.logical_height), y + 1, height);
                local_offset =
                    AlignPpSourceBackingOffset(local_offset, descriptor.buffer_alignment);
                const u64 byte_size =
                    static_cast<u64>(end_x - x) * (end_y - y) * view.bytes_per_texel;
                if (local_offset == std::numeric_limits<u64>::max() || byte_size == 0 ||
                    local_offset + byte_size > std::numeric_limits<u32>::max()) {
                    return reject(FinalGuestSurfaceStatus::CapacityLoss,
                                  &FinalGuestSurfaceLoss::byte_capacity);
                }
                input.regions[input.region_count++] = {
                    .logical_ordinal = selector.ordinals[selector_index],
                    .mip_level = view.base_mip + mip,
                    .x = x,
                    .y = y,
                    .width = end_x - x,
                    .height = end_y - y,
                    .buffer_offset = static_cast<u32>(local_offset),
                    .byte_size = static_cast<u32>(byte_size),
                };
                local_offset += byte_size;
            }
        }
        if (next_offset + local_offset > descriptor.max_bytes ||
            next_offset + local_offset > std::numeric_limits<u32>::max()) {
            return reject(FinalGuestSurfaceStatus::CapacityLoss,
                          &FinalGuestSurfaceLoss::byte_capacity);
        }
        input.plane_bytes = static_cast<u32>(local_offset);
        input.status = FinalGuestSurfaceStatus::Complete;
        result.capture_mask |= 1u << input_index;
        result.copy_region_count += input.region_count;
        next_offset += local_offset;
    }
    if (result.capture_mask == 0) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::unsupported_format);
    }
    result.total_bytes = static_cast<u32>(next_offset);
    result.status = FinalGuestSurfaceStatus::Complete;
    result.copy = true;
    return result;
}

struct PpUpstreamInputContentCompactResult {
    std::vector<std::byte> bytes{};
    std::array<u32, PpUpstreamInputContentPlan::MaxInputs> plane_offsets{};
    u32 capture_mask{};
    u32 unavailable_mask{};
    u32 alias_mask{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
};

[[nodiscard]] inline PpUpstreamInputContentCompactResult CompactPpUpstreamInputContent(
    const PpUpstreamInputContentPlan& plan, std::span<const std::byte> slot) {
    PpUpstreamInputContentCompactResult result{
        .capture_mask = plan.capture_mask,
        .unavailable_mask = plan.unavailable_mask,
        .alias_mask = plan.alias_mask,
        .status = plan.status,
        .loss = plan.loss,
    };
    if (!plan.copy || plan.status != FinalGuestSurfaceStatus::Complete || plan.loss.Any() ||
        plan.total_bytes > slot.size()) {
        result.status = FinalGuestSurfaceStatus::InvalidationLoss;
        result.loss = {.invalidation = 1};
        return result;
    }
    u64 compact_size{};
    for (u32 input_index = 0; input_index < plan.input_count; ++input_index) {
        if ((plan.capture_mask & (1u << input_index)) == 0) {
            continue;
        }
        const auto& input = plan.inputs[input_index];
        if (input.plane_bytes == 0 ||
            static_cast<u64>(input.plane_offset) + input.plane_bytes > slot.size() ||
            compact_size + input.plane_bytes > std::numeric_limits<u32>::max()) {
            result.status = FinalGuestSurfaceStatus::InvalidationLoss;
            result.loss = {.invalidation = 1};
            result.bytes.clear();
            return result;
        }
        result.plane_offsets[input_index] = static_cast<u32>(compact_size);
        compact_size += input.plane_bytes;
    }
    result.bytes.resize(compact_size);
    for (u32 input_index = 0; input_index < plan.input_count; ++input_index) {
        if ((plan.capture_mask & (1u << input_index)) == 0) {
            continue;
        }
        const auto& input = plan.inputs[input_index];
        std::ranges::copy(slot.subspan(input.plane_offset, input.plane_bytes),
                          result.bytes.begin() + result.plane_offsets[input_index]);
    }
    return result;
}

enum class PpUpstreamInputRawClass : u8 {
    Unavailable,
    Aba,
    Stable,
    Ambiguous,
};

[[nodiscard]] inline PpUpstreamInputRawClass ClassifyPpUpstreamInputRawTriplet(
    std::span<const std::byte> a, std::span<const std::byte> b,
    std::span<const std::byte> c) noexcept {
    if (a.empty() || a.size() != b.size() || a.size() != c.size()) {
        return PpUpstreamInputRawClass::Unavailable;
    }
    const bool ab = std::ranges::equal(a, b);
    const bool bc = std::ranges::equal(b, c);
    const bool ac = std::ranges::equal(a, c);
    if (ac && !ab) {
        return PpUpstreamInputRawClass::Aba;
    }
    if (ab && bc) {
        return PpUpstreamInputRawClass::Stable;
    }
    return PpUpstreamInputRawClass::Ambiguous;
}

struct PpUpstreamInputCaptureAction {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool capture{};
    bool publish{};
};

struct PpUpstreamInputPublication {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy{};
};

[[nodiscard]] constexpr PpUpstreamInputPublication ReconcilePpUpstreamInputPublication(
    bool enabled, bool input_ready, u32 candidate_index, u32 expected_candidate_index,
    FinalGuestSurfaceStatus status, FinalGuestSurfaceLoss loss, bool copy) noexcept {
    if (!enabled || status != FinalGuestSurfaceStatus::Complete || loss.Any() || !copy) {
        return {.status = status, .loss = loss, .copy = copy};
    }
    if (!input_ready || candidate_index != expected_candidate_index) {
        return {.status = FinalGuestSurfaceStatus::GapLoss, .loss = {.gap = 1}};
    }
    return {.status = status, .loss = loss, .copy = true};
}

class PpUpstreamInputCaptureGate {
public:
    explicit constexpr PpUpstreamInputCaptureGate(u32 candidate_index_) noexcept
        : candidate_index{candidate_index_} {}

    [[nodiscard]] constexpr PpUpstreamInputCaptureAction Preview(u32 observed) noexcept {
        if (observed != candidate_index) {
            return {};
        }
        previewed = true;
        return {.status = FinalGuestSurfaceStatus::Complete, .capture = true};
    }

    [[nodiscard]] constexpr PpUpstreamInputCaptureAction Complete(u32 observed,
                                                                  bool copy) noexcept {
        if (observed != candidate_index || !previewed) {
            return {.status = FinalGuestSurfaceStatus::GapLoss, .loss = {.gap = 1}};
        }
        previewed = false;
        if (!copy) {
            return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        return {.status = FinalGuestSurfaceStatus::Complete, .publish = true};
    }

    constexpr void Reset() noexcept {
        previewed = false;
    }

private:
    u32 candidate_index{};
    bool previewed{};
};

template <typename State, typename SubresourceStates>
void RestorePpUpstreamInputUniformTracker(State& current_state,
                                          SubresourceStates& subresource_states,
                                          const State& original_state) {
    current_state = original_state;
    subresource_states.clear();
}

} // namespace Vulkan::HostPasses
