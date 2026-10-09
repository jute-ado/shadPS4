// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <string>
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
    bool allow_all_unavailable{};
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
    bool unavailable_only{};
    bool cpu_wait{};
    bool finish{};
    bool retains_image{};
    bool retains_vk_image{};
};

[[nodiscard]] constexpr bool IsPpUpstreamInputBytesPerTexelSupported(u32 value) noexcept {
    return value == 1 || value == 2 || value == 4 || value == 8 || value == 16;
}

[[nodiscard]] constexpr bool IsPpUpstreamInputContentPlanRecordable(
    const PpUpstreamInputContentPlan& plan) noexcept {
    if (plan.status != FinalGuestSurfaceStatus::Complete || plan.loss.Any() ||
        plan.input_count == 0 || plan.input_count > plan.MaxInputs) {
        return false;
    }
    const u32 valid_mask = (1u << plan.input_count) - 1;
    const u32 classified_mask = plan.capture_mask | plan.unavailable_mask | plan.alias_mask;
    if (classified_mask != valid_mask || (plan.capture_mask & plan.unavailable_mask) != 0 ||
        (plan.capture_mask & plan.alias_mask) != 0 ||
        (plan.unavailable_mask & plan.alias_mask) != 0) {
        return false;
    }
    if (!plan.unavailable_only) {
        return plan.copy && plan.capture_mask != 0;
    }
    if (plan.copy || plan.capture_mask != 0 || plan.unavailable_mask != valid_mask ||
        plan.alias_mask != 0 || plan.copy_region_count != 0) {
        return false;
    }
    for (u32 input_index = 0; input_index < plan.input_count; ++input_index) {
        const auto& input = plan.inputs[input_index];
        const bool terminal_unavailable =
            input.status == FinalGuestSurfaceStatus::Unsupported ||
            input.status == FinalGuestSurfaceStatus::InvalidationLoss;
        if (!terminal_unavailable || !input.loss.Any() ||
            input.region_count != 0 || input.plane_bytes != 0) {
            return false;
        }
    }
    return true;
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
        const u32 valid_mask = (1u << result.input_count) - 1;
        if (descriptor.allow_all_unavailable && result.unavailable_mask == valid_mask &&
            result.alias_mask == 0) {
            if (descriptor.base_offset > descriptor.max_bytes) {
                return reject(FinalGuestSurfaceStatus::CapacityLoss,
                              &FinalGuestSurfaceLoss::byte_capacity);
            }
            result.total_bytes = descriptor.base_offset;
            result.status = FinalGuestSurfaceStatus::Complete;
            result.unavailable_only = true;
            return result;
        }
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
    if (!IsPpUpstreamInputContentPlanRecordable(plan) || plan.total_bytes > slot.size()) {
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

enum class PpUpstreamInputPublicationDisposition : u8 {
    Disabled = 0,
    BaseNotPublishable = 1,
    InputNotReady = 2,
    CandidateMismatch = 3,
    Published = 4,
};

struct PpUpstreamInputPublicationEvaluation {
    PpUpstreamInputPublication publication{};
    PpUpstreamInputPublicationDisposition disposition{
        PpUpstreamInputPublicationDisposition::Disabled};
    bool input_enabled{};
    bool input_ready{};
    bool candidate_equal{};
};

[[nodiscard]] constexpr PpUpstreamInputPublicationEvaluation
EvaluatePpUpstreamInputPublication(bool enabled, bool input_ready, u32 candidate_index,
                                   u32 expected_candidate_index,
                                   FinalGuestSurfaceStatus status, FinalGuestSurfaceLoss loss,
                                   bool copy) noexcept {
    const PpUpstreamInputPublication base{.status = status, .loss = loss, .copy = copy};
    const bool candidate_equal = candidate_index == expected_candidate_index;
    if (!enabled) {
        return {.publication = base,
                .disposition = PpUpstreamInputPublicationDisposition::Disabled,
                .input_enabled = false,
                .input_ready = input_ready,
                .candidate_equal = candidate_equal};
    }
    if (status != FinalGuestSurfaceStatus::Complete || loss.Any() || !copy) {
        return {.publication = base,
                .disposition = PpUpstreamInputPublicationDisposition::BaseNotPublishable,
                .input_enabled = true,
                .input_ready = input_ready,
                .candidate_equal = candidate_equal};
    }
    if (!input_ready) {
        return {.publication = {.status = FinalGuestSurfaceStatus::GapLoss,
                                .loss = {.gap = 1}},
                .disposition = PpUpstreamInputPublicationDisposition::InputNotReady,
                .input_enabled = true,
                .input_ready = false,
                .candidate_equal = candidate_equal};
    }
    if (!candidate_equal) {
        return {.publication = {.status = FinalGuestSurfaceStatus::GapLoss,
                                .loss = {.gap = 1}},
                .disposition = PpUpstreamInputPublicationDisposition::CandidateMismatch,
                .input_enabled = true,
                .input_ready = true,
                .candidate_equal = false};
    }
    return {.publication = base,
            .disposition = PpUpstreamInputPublicationDisposition::Published,
            .input_enabled = true,
            .input_ready = true,
            .candidate_equal = true};
}

[[nodiscard]] constexpr PpUpstreamInputPublication ReconcilePpUpstreamInputPublication(
    bool enabled, bool input_ready, u32 candidate_index, u32 expected_candidate_index,
    FinalGuestSurfaceStatus status, FinalGuestSurfaceLoss loss, bool copy) noexcept {
    return EvaluatePpUpstreamInputPublication(enabled, input_ready, candidate_index,
                                              expected_candidate_index, status, loss, copy)
        .publication;
}

enum class PpUpstreamInputPublicationFrameFirstFailure : u8 {
    Idle = 0,
    CaptureCardinality = 1,
    PrePublication = 2,
    InputDisabled = 3,
    InputNotReady = 4,
    CandidateMismatch = 5,
    LaterFinalLoss = 6,
    InvalidFinalTransition = 7,
    Exact = 8,
};

struct PpUpstreamInputPublicationFrameDescriptor {
    u64 sequence{};
    u32 candidate_index{};
    u32 recorded_plane_mask{};
    u32 region_count{};
    PpUpstreamInputPublication take{};
    PpUpstreamInputPublicationEvaluation evaluation{};
    PpUpstreamInputPublication final{};
};

struct PpUpstreamInputPublicationFrameReport {
    u64 sequence{};
    u32 candidate_index{};
    u32 recorded_plane_mask{};
    u32 region_count{};
    PpUpstreamInputPublication take{};
    PpUpstreamInputPublication publication{};
    PpUpstreamInputPublication final{};
    PpUpstreamInputPublicationDisposition disposition{
        PpUpstreamInputPublicationDisposition::Disabled};
    PpUpstreamInputPublicationFrameFirstFailure first_failure{
        PpUpstreamInputPublicationFrameFirstFailure::Idle};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    u32 additional_ownership_transfer_count{};
    bool input_enabled{};
    bool input_ready{};
    bool candidate_equal{};
    bool retains_private_state{};
    bool retains_image{};
    bool retains_vk_image{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
};

[[nodiscard]] constexpr PpUpstreamInputPublicationFrameReport
MakePpUpstreamInputPublicationFrameReport(
    const PpUpstreamInputPublicationFrameDescriptor& descriptor) noexcept {
    using Disposition = PpUpstreamInputPublicationDisposition;
    using FirstFailure = PpUpstreamInputPublicationFrameFirstFailure;
    const auto& evaluation = descriptor.evaluation;
    const bool idle =
        descriptor.sequence == 0 && descriptor.candidate_index == 0 &&
        descriptor.recorded_plane_mask == 0 && descriptor.region_count == 0 &&
        descriptor.take.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
        !descriptor.take.loss.Any() && !descriptor.take.copy &&
        evaluation.publication.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
        !evaluation.publication.loss.Any() && !evaluation.publication.copy &&
        evaluation.disposition == Disposition::Disabled && !evaluation.input_enabled &&
        !evaluation.input_ready && !evaluation.candidate_equal &&
        descriptor.final.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
        !descriptor.final.loss.Any() && !descriptor.final.copy;

    FirstFailure first_failure{FirstFailure::Idle};
    if (!idle) {
        if (descriptor.recorded_plane_mask != 0x3u) {
            first_failure = FirstFailure::CaptureCardinality;
        } else if (descriptor.region_count == 0) {
            first_failure = FirstFailure::InvalidFinalTransition;
        } else if (descriptor.take.status != FinalGuestSurfaceStatus::Complete ||
                   descriptor.take.loss.Any() || !descriptor.take.copy) {
            first_failure = FirstFailure::PrePublication;
        } else if (evaluation.disposition == Disposition::Disabled) {
            first_failure = FirstFailure::InputDisabled;
        } else if (evaluation.disposition == Disposition::InputNotReady) {
            first_failure = FirstFailure::InputNotReady;
        } else if (evaluation.disposition == Disposition::CandidateMismatch) {
            first_failure = FirstFailure::CandidateMismatch;
        } else if (evaluation.disposition != Disposition::Published ||
                   evaluation.publication.status != FinalGuestSurfaceStatus::Complete ||
                   evaluation.publication.loss.Any() || !evaluation.publication.copy) {
            first_failure = FirstFailure::InvalidFinalTransition;
        } else if (descriptor.final.status == FinalGuestSurfaceStatus::GapLoss ||
                   descriptor.final.loss.gap != 0) {
            first_failure = FirstFailure::InvalidFinalTransition;
        } else if (descriptor.final.status == FinalGuestSurfaceStatus::InvalidationLoss &&
                   descriptor.final.loss == FinalGuestSurfaceLoss{.invalidation = 1} &&
                   !descriptor.final.copy) {
            first_failure = FirstFailure::LaterFinalLoss;
        } else if (descriptor.final.status == FinalGuestSurfaceStatus::Complete &&
                   !descriptor.final.loss.Any() && descriptor.final.copy) {
            first_failure = FirstFailure::Exact;
        } else {
            first_failure = FirstFailure::InvalidFinalTransition;
        }
    }

    return {
        .sequence = descriptor.sequence,
        .candidate_index = descriptor.candidate_index,
        .recorded_plane_mask = descriptor.recorded_plane_mask,
        .region_count = descriptor.region_count,
        .take = descriptor.take,
        .publication = evaluation.publication,
        .final = descriptor.final,
        .disposition = evaluation.disposition,
        .first_failure = first_failure,
        .input_enabled = evaluation.input_enabled,
        .input_ready = evaluation.input_ready,
        .candidate_equal = evaluation.candidate_equal,
    };
}

[[nodiscard]] inline std::string FormatPpUpstreamInputPublicationFrameReport(
    const PpUpstreamInputPublicationFrameReport& report) {
    return "FGSCTU s=" + std::to_string(report.sequence) +
           " st=" + std::to_string(static_cast<u32>(report.final.status)) +
           " lm=" + std::to_string(FinalGuestSurfaceLossMask(report.final.loss, 0)) +
           " ci=" + std::to_string(report.candidate_index) +
           " pm=" + std::to_string(report.recorded_plane_mask) +
           " r=" + std::to_string(report.region_count) +
           " ts=" + std::to_string(static_cast<u32>(report.take.status)) +
           " tlm=" + std::to_string(FinalGuestSurfaceLossMask(report.take.loss, 0)) +
           " tc=" + std::to_string(report.take.copy ? 1 : 0) +
           " ps=" + std::to_string(static_cast<u32>(report.publication.status)) +
           " plm=" + std::to_string(FinalGuestSurfaceLossMask(report.publication.loss, 0)) +
           " pc=" + std::to_string(report.publication.copy ? 1 : 0) +
           " ie=" + std::to_string(report.input_enabled ? 1 : 0) +
           " ir=" + std::to_string(report.input_ready ? 1 : 0) +
           " ce=" + std::to_string(report.candidate_equal ? 1 : 0) +
           " pd=" + std::to_string(static_cast<u32>(report.disposition)) +
           " fc=" + std::to_string(report.final.copy ? 1 : 0) +
           " ff=" + std::to_string(static_cast<u32>(report.first_failure));
}

[[nodiscard]] constexpr PpUpstreamInputPublication ReconcilePpUpstreamInputSlotRelease(
    bool had_slot, bool released, FinalGuestSurfaceStatus status, FinalGuestSurfaceLoss loss,
    bool copy) noexcept {
    if (!had_slot || released) {
        return {.status = status, .loss = loss, .copy = copy};
    }
    return {.status = FinalGuestSurfaceStatus::InvalidationLoss, .loss = {.invalidation = 1}};
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
