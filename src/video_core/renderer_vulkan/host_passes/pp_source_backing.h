// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <string>

#include "common/types.h"
#include "video_core/renderer_vulkan/final_guest_surface_content.h"

namespace Vulkan {

struct PpSourceBackingRegion {
    u32 logical_ordinal{};
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};
    u32 buffer_offset{};
    u32 byte_size{};

    bool operator==(const PpSourceBackingRegion&) const = default;
};

struct PpSourceBackingFootprintDescriptor {
    bool enabled{};
    bool in_window{};
    bool pp_draw_encoded{};
    bool fsr_bypassed{};
    u32 source_width{};
    u32 source_height{};
    u32 logical_width{};
    u32 logical_height{};
    FinalGuestSurfaceFormat source_format{FinalGuestSurfaceFormat::Unsupported};
    u32 samples{};
    u32 resolved_base_mip{};
    u32 resolved_mip_count{};
    u32 resolved_base_layer{};
    u32 resolved_layer_count{};
    u32 bound_base_mip{};
    u32 bound_mip_count{};
    u32 bound_base_layer{};
    u32 bound_layer_count{};
    bool logical_full_fit{};
    bool logical_top_left{};
    bool logical_no_y_flip{};
    u32 buffer_alignment{};
    u32 max_regions{};
    u32 max_bytes{};
    FinalGuestSurfaceWatchOrdinals selector{};
};

struct PpSourceBackingFootprintPlan {
    static constexpr u32 MaxRegions = FinalGuestSurfaceWatchOrdinals::MaxOrdinals;

    std::array<PpSourceBackingRegion, MaxRegions> regions{};
    u32 region_count{};
    u32 copy_region_count{};
    u32 image_barrier_count{};
    u32 buffer_bytes{};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool allocate_snapshot{};
    bool pp_draw_precedes_copy{};
    bool restores_shader_read{};
    bool callback_payload_is_scalar_only{};
    bool cpu_wait{};
    bool finish{};
    bool callback_retains_frame{};
    bool callback_retains_image{};
    bool callback_retains_vk_image{};
};

[[nodiscard]] constexpr u64 AlignPpSourceBackingOffset(u64 value, u64 alignment) noexcept {
    if (alignment == 0) {
        return std::numeric_limits<u64>::max();
    }
    const u64 remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    const u64 padding = alignment - remainder;
    if (value > std::numeric_limits<u64>::max() - padding) {
        return std::numeric_limits<u64>::max();
    }
    return value + padding;
}

[[nodiscard]] inline PpSourceBackingFootprintPlan PlanPpSourceBackingFootprints(
    const PpSourceBackingFootprintDescriptor& descriptor) noexcept {
    PpSourceBackingFootprintPlan plan{};
    if (!descriptor.enabled || !descriptor.in_window) {
        return plan;
    }
    const auto reject = [&](FinalGuestSurfaceStatus status,
                            u32 FinalGuestSurfaceLoss::* loss_member) {
        PpSourceBackingFootprintPlan rejected{};
        rejected.status = status;
        rejected.loss.*loss_member = 1;
        return rejected;
    };
    if (!descriptor.pp_draw_encoded) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (!descriptor.fsr_bypassed || descriptor.samples != 1 ||
        (descriptor.source_format != FinalGuestSurfaceFormat::Rgba8 &&
         descriptor.source_format != FinalGuestSurfaceFormat::Bgra8)) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      descriptor.samples != 1 ? &FinalGuestSurfaceLoss::unsupported_samples
                                              : &FinalGuestSurfaceLoss::unsupported_format);
    }
    if (descriptor.source_width == 0 || descriptor.source_height == 0 ||
        descriptor.logical_width == 0 || descriptor.logical_height == 0) {
        return reject(FinalGuestSurfaceStatus::Unsupported, &FinalGuestSurfaceLoss::invalid_extent);
    }
    if (descriptor.resolved_mip_count != 1 || descriptor.resolved_layer_count != 1 ||
        descriptor.bound_mip_count != 1 || descriptor.bound_layer_count != 1 ||
        descriptor.resolved_base_mip != descriptor.bound_base_mip ||
        descriptor.resolved_base_layer != descriptor.bound_base_layer) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (descriptor.bound_base_mip != 0 || descriptor.bound_base_layer != 0) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      descriptor.bound_base_mip != 0 ? &FinalGuestSurfaceLoss::unsupported_mip
                                                     : &FinalGuestSurfaceLoss::unsupported_layer);
    }
    if (!descriptor.logical_full_fit || !descriptor.logical_top_left ||
        !descriptor.logical_no_y_flip ||
        static_cast<u64>(descriptor.source_width) * descriptor.logical_height !=
            static_cast<u64>(descriptor.source_height) * descriptor.logical_width) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::logical_mapping);
    }
    const auto selector = ValidateFinalGuestSurfaceWatchOrdinals(descriptor.selector, [&] {
        const u32 window_width =
            std::min(descriptor.logical_width, FinalGuestSurfaceTilePlan::WindowExtent);
        const u32 window_height =
            std::min(descriptor.logical_height, FinalGuestSurfaceTilePlan::WindowExtent);
        const u32 columns = descriptor.logical_width <= window_width
                                ? 1
                                : (descriptor.logical_width - window_width) /
                                          FinalGuestSurfaceTilePlan::WindowStride +
                                      1;
        const u32 rows = descriptor.logical_height <= window_height
                             ? 1
                             : (descriptor.logical_height - window_height) /
                                       FinalGuestSurfaceTilePlan::WindowStride +
                                   1;
        const u64 count = static_cast<u64>(columns) * rows;
        return count <= std::numeric_limits<u32>::max() ? static_cast<u32>(count) : 0u;
    }());
    if (selector.status != FinalGuestSurfaceStatus::Complete || selector.count == 0) {
        return reject(selector.status == FinalGuestSurfaceStatus::CapacityLoss
                          ? FinalGuestSurfaceStatus::CapacityLoss
                          : FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::ordinal_capacity);
    }
    if (descriptor.buffer_alignment == 0 || descriptor.max_regions == 0 ||
        selector.count > descriptor.max_regions || selector.count > plan.regions.size()) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss, &FinalGuestSurfaceLoss::tile_capacity);
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
    u64 next_offset{};
    for (u32 index = 0; index < selector.count; ++index) {
        const u32 zero_based = selector.ordinals[index] - 1;
        const u32 column = zero_based % columns;
        const u32 row = zero_based / columns;
        const u32 logical_x = column * FinalGuestSurfaceTilePlan::WindowStride;
        const u32 logical_y = row * FinalGuestSurfaceTilePlan::WindowStride;
        const u32 logical_end_x = logical_x + window_width;
        const u32 logical_end_y = logical_y + window_height;
        const auto maps_exactly = [](u32 coordinate, u32 source_extent,
                                     u32 logical_extent) noexcept {
            return static_cast<u64>(coordinate) * source_extent % logical_extent == 0;
        };
        if (!maps_exactly(logical_x, descriptor.source_width, descriptor.logical_width) ||
            !maps_exactly(logical_end_x, descriptor.source_width, descriptor.logical_width) ||
            !maps_exactly(logical_y, descriptor.source_height, descriptor.logical_height) ||
            !maps_exactly(logical_end_y, descriptor.source_height, descriptor.logical_height)) {
            return reject(FinalGuestSurfaceStatus::Unsupported,
                          &FinalGuestSurfaceLoss::logical_mapping);
        }
        const u32 x = static_cast<u32>(static_cast<u64>(logical_x) * descriptor.source_width /
                                       descriptor.logical_width);
        const u32 y = static_cast<u32>(static_cast<u64>(logical_y) * descriptor.source_height /
                                       descriptor.logical_height);
        const u32 end_x = static_cast<u32>(static_cast<u64>(logical_end_x) *
                                           descriptor.source_width / descriptor.logical_width);
        const u32 end_y = static_cast<u32>(static_cast<u64>(logical_end_y) *
                                           descriptor.source_height / descriptor.logical_height);
        const u64 byte_size = static_cast<u64>(end_x - x) * (end_y - y) * 4;
        next_offset = AlignPpSourceBackingOffset(next_offset, descriptor.buffer_alignment);
        if (next_offset == std::numeric_limits<u64>::max() ||
            byte_size > std::numeric_limits<u32>::max() ||
            next_offset > std::numeric_limits<u32>::max() ||
            next_offset + byte_size > descriptor.max_bytes ||
            next_offset + byte_size > std::numeric_limits<u32>::max()) {
            return reject(FinalGuestSurfaceStatus::CapacityLoss,
                          &FinalGuestSurfaceLoss::byte_capacity);
        }
        plan.regions[index] = {
            .logical_ordinal = selector.ordinals[index],
            .x = x,
            .y = y,
            .width = end_x - x,
            .height = end_y - y,
            .buffer_offset = static_cast<u32>(next_offset),
            .byte_size = static_cast<u32>(byte_size),
        };
        next_offset += byte_size;
    }
    plan.region_count = selector.count;
    plan.copy_region_count = selector.count;
    plan.image_barrier_count = 2;
    plan.buffer_bytes = static_cast<u32>(next_offset);
    plan.format = descriptor.source_format;
    plan.status = FinalGuestSurfaceStatus::Complete;
    plan.allocate_snapshot = true;
    plan.pp_draw_precedes_copy = true;
    plan.restores_shader_read = true;
    plan.callback_payload_is_scalar_only = true;
    return plan;
}

[[nodiscard]] inline FinalGuestSurfaceTilePlan MakePpSampledInputSourceBackingTilePlan(
    FinalGuestSurfaceTilePlan plan, const PpSourceBackingFootprintPlan& backing, u64 slot_bytes,
    u64 alignment) noexcept {
    if (plan.status != FinalGuestSurfaceStatus::Complete ||
        backing.status != FinalGuestSurfaceStatus::Complete || backing.region_count == 0 ||
        backing.region_count > plan.paired_backing_regions.size() || alignment == 0) {
        plan.status = FinalGuestSurfaceStatus::InvalidationLoss;
        plan.loss.invalidation = 1;
        return plan;
    }
    const u64 backing_offset = AlignPpSourceBackingOffset(plan.sample_bytes, alignment);
    if (backing_offset == std::numeric_limits<u64>::max() ||
        backing_offset > std::numeric_limits<u32>::max() ||
        backing.buffer_bytes > std::numeric_limits<u32>::max() ||
        backing_offset + backing.buffer_bytes > slot_bytes ||
        backing_offset + backing.buffer_bytes > std::numeric_limits<u32>::max()) {
        plan.status = FinalGuestSurfaceStatus::CapacityLoss;
        plan.loss.byte_capacity = 1;
        return plan;
    }
    plan.paired_backing_offset = static_cast<u32>(backing_offset);
    plan.paired_backing_bytes = backing.buffer_bytes;
    plan.paired_backing_region_count = backing.region_count;
    plan.paired_backing_format = backing.format;
    for (u32 index = 0; index < backing.region_count; ++index) {
        plan.paired_backing_regions[index] = {
            .logical_ordinal = backing.regions[index].logical_ordinal,
            .buffer_offset = backing.regions[index].buffer_offset,
            .byte_size = backing.regions[index].byte_size,
            .width = backing.regions[index].width,
            .height = backing.regions[index].height,
        };
    }
    plan.sample_bytes = static_cast<u32>(backing_offset + backing.buffer_bytes);
    ++plan.copy_region_count;
    return plan;
}

struct PpSourceBackingTripletClassification {
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> aba_ordinals{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> stable_ordinals{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> ambiguous_ordinals{};
    u32 aba_count{};
    u32 stable_count{};
    u32 ambiguous_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
};

[[nodiscard]] inline PpSourceBackingTripletClassification ClassifyPpSourceBackingTriplet(
    const PpSourceBackingFootprintPlan& plan, std::span<const std::byte> a,
    std::span<const std::byte> b, std::span<const std::byte> c) noexcept {
    PpSourceBackingTripletClassification result{};
    if (plan.status != FinalGuestSurfaceStatus::Complete ||
        (plan.format != FinalGuestSurfaceFormat::Rgba8 &&
         plan.format != FinalGuestSurfaceFormat::Bgra8) ||
        a.size() < plan.buffer_bytes || b.size() < plan.buffer_bytes ||
        c.size() < plan.buffer_bytes) {
        result.status = FinalGuestSurfaceStatus::InvalidationLoss;
        return result;
    }
    for (u32 index = 0; index < plan.region_count; ++index) {
        const auto& region = plan.regions[index];
        const size_t end = static_cast<size_t>(region.buffer_offset) + region.byte_size;
        if (end > plan.buffer_bytes || region.byte_size % 4 != 0) {
            result = {};
            result.status = FinalGuestSurfaceStatus::InvalidationLoss;
            return result;
        }
        bool equal_ab = true;
        bool equal_ac = true;
        for (size_t offset = region.buffer_offset; offset < end; ++offset) {
            if ((offset - region.buffer_offset) % 4 == 3) {
                continue;
            }
            equal_ab &= a[offset] == b[offset];
            equal_ac &= a[offset] == c[offset];
        }
        if (equal_ab && equal_ac) {
            result.stable_ordinals[result.stable_count++] = region.logical_ordinal;
        } else if (equal_ac) {
            result.aba_ordinals[result.aba_count++] = region.logical_ordinal;
        } else {
            result.ambiguous_ordinals[result.ambiguous_count++] = region.logical_ordinal;
        }
    }
    return result;
}

struct PpSourceBackingObservation {
    u64 logical_generation{};
    u64 expected_image_uid{};
    u64 expected_backing_generation{};
    u64 captured_image_uid{};
    u64 captured_backing_generation{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
};

[[nodiscard]] constexpr PpSourceBackingObservation ValidatePpSourceBackingObservation(
    PpSourceBackingObservation observation) noexcept {
    observation.status =
        observation.logical_generation != 0 && observation.expected_image_uid != 0 &&
                observation.expected_backing_generation != 0 &&
                observation.expected_image_uid == observation.captured_image_uid &&
                observation.expected_backing_generation == observation.captured_backing_generation
            ? FinalGuestSurfaceStatus::Complete
            : FinalGuestSurfaceStatus::InvalidationLoss;
    return observation;
}

[[nodiscard]] constexpr bool PpSourceBackingTripletTransportCompatible(
    const PpSourceBackingObservation& a, const PpSourceBackingObservation& b,
    const PpSourceBackingObservation& c) noexcept {
    return a.status == FinalGuestSurfaceStatus::Complete &&
           b.status == FinalGuestSurfaceStatus::Complete &&
           c.status == FinalGuestSurfaceStatus::Complete &&
           a.logical_generation == b.logical_generation &&
           a.logical_generation == c.logical_generation;
}

struct PpSourceBackingTripletReport {
    u32 request_ordinal{};
    u64 a_sequence{};
    u64 b_sequence{};
    u64 c_sequence{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> aba_ordinals{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> stable_ordinals{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> ambiguous_ordinals{};
    u32 aba_count{};
    u32 stable_count{};
    u32 ambiguous_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
};

[[nodiscard]] inline std::string FormatPpSourceBackingOrdinals(
    const std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals>& ordinals, u32 count) {
    std::string result;
    for (u32 index = 0; index < count && index < ordinals.size(); ++index) {
        if (!result.empty()) {
            result.push_back(',');
        }
        result += std::to_string(ordinals[index]);
    }
    return result;
}

[[nodiscard]] inline std::string FormatPpSourceBackingTripletReport(
    const PpSourceBackingTripletReport& report) {
    return "PPSB q=" + std::to_string(report.request_ordinal) +
           " abc=" + std::to_string(report.a_sequence) + "/" + std::to_string(report.b_sequence) +
           "/" + std::to_string(report.c_sequence) +
           " aba=" + FormatPpSourceBackingOrdinals(report.aba_ordinals, report.aba_count) +
           " stable=" + FormatPpSourceBackingOrdinals(report.stable_ordinals, report.stable_count) +
           " amb=" +
           FormatPpSourceBackingOrdinals(report.ambiguous_ordinals, report.ambiguous_count) +
           " st=" + std::to_string(static_cast<u32>(report.status));
}

} // namespace Vulkan
