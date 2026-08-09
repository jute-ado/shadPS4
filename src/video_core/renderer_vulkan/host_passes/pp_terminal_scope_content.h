// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <string>

#include "video_core/renderer_vulkan/host_passes/pp_source_backing.h"
#include "video_core/texture_cache/image_color_scope_producer.h"

namespace Vulkan {

inline constexpr u32 PpTerminalScopeSnapshotBytes = 1u << 20;

struct PpTerminalScopeDrawSelector {
    VideoCore::ImageColorScopeDrawKind kind{VideoCore::ImageColorScopeDrawKind::Unknown};
    bool indexed{};
    u32 element_count{};
    u32 instance_count{};
    u32 sampled_images{};
    u32 storage_writes{};

    bool operator==(const PpTerminalScopeDrawSelector&) const = default;
};

[[nodiscard]] constexpr bool MatchesPpTerminalScopeDraw(
    const PpTerminalScopeDrawSelector& expected,
    const PpTerminalScopeDrawSelector& observed) noexcept {
    return expected == observed && expected.kind != VideoCore::ImageColorScopeDrawKind::Unknown;
}

struct PpTerminalScopeContentConfig {
    bool enabled{};
    PpTerminalScopeDrawSelector first{};
    PpTerminalScopeDrawSelector second{};
};

enum class PpTerminalScopeContentAction : u8 {
    None,
    CaptureFirst,
    CaptureSecond,
    ShapeLoss,
};

struct PpTerminalScopeContentTakeResult {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    u32 draw_count{};
    bool cpu_wait{};
    bool finish{};
    bool retains_image{};
    bool retains_vk_image{};
};

class PpTerminalScopeContentGate {
public:
    explicit constexpr PpTerminalScopeContentGate(PpTerminalScopeContentConfig config_) noexcept
        : config{config_} {}

    [[nodiscard]] constexpr bool Arm(u64 target_token, u64 generation) noexcept {
        if (!config.enabled || target_token == 0 || generation == 0 ||
            config.first.kind == VideoCore::ImageColorScopeDrawKind::Unknown ||
            config.second.kind == VideoCore::ImageColorScopeDrawKind::Unknown) {
            Reset();
            return false;
        }
        token = target_token;
        armed_generation = generation;
        scope_serial = 0;
        phase = 0;
        return true;
    }

    [[nodiscard]] constexpr PpTerminalScopeContentAction ObserveDraw(
        u64 target_token, u64 observed_scope_serial,
        const PpTerminalScopeDrawSelector& draw) noexcept {
        if (target_token == 0 || target_token != token || observed_scope_serial == 0) {
            return PpTerminalScopeContentAction::None;
        }
        if (phase == 0) {
            if (!MatchesPpTerminalScopeDraw(config.first, draw)) {
                return PpTerminalScopeContentAction::None;
            }
            scope_serial = observed_scope_serial;
            phase = 1;
            return PpTerminalScopeContentAction::CaptureFirst;
        }
        if (phase == 1) {
            if (scope_serial != observed_scope_serial ||
                !MatchesPpTerminalScopeDraw(config.second, draw)) {
                scope_serial = 0;
                phase = 0;
                return PpTerminalScopeContentAction::ShapeLoss;
            }
            phase = 2;
            return PpTerminalScopeContentAction::CaptureSecond;
        }
        return PpTerminalScopeContentAction::None;
    }

    [[nodiscard]] constexpr PpTerminalScopeContentTakeResult Take(u64 target_token,
                                                                  u64 generation) noexcept {
        PpTerminalScopeContentTakeResult result{};
        if (target_token != token || generation != armed_generation) {
            result.status = FinalGuestSurfaceStatus::InvalidationLoss;
            result.loss.invalidation = 1;
        } else if (phase != 2) {
            result.status = FinalGuestSurfaceStatus::GapLoss;
            result.loss.gap = 1;
            result.draw_count = phase;
        } else {
            result.status = FinalGuestSurfaceStatus::Complete;
            result.draw_count = 2;
        }
        scope_serial = 0;
        phase = 0;
        return result;
    }

private:
    constexpr void Reset() noexcept {
        token = 0;
        armed_generation = 0;
        scope_serial = 0;
        phase = 0;
    }

    PpTerminalScopeContentConfig config{};
    u64 token{};
    u64 armed_generation{};
    u64 scope_serial{};
    u32 phase{};
};

struct PpTerminalScopeContentDescriptor {
    bool enabled{};
    bool armed{};
    u32 target_width{};
    u32 target_height{};
    u32 final_source_width{};
    u32 final_source_height{};
    u32 logical_width{};
    u32 logical_height{};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    u32 samples{};
    FinalGuestSurfaceWatchOrdinals selector{};
    u32 buffer_alignment{};
    u32 max_regions{};
    u32 max_bytes{};
};

struct PpTerminalScopeContentPlan {
    std::array<PpSourceBackingRegion, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> regions{};
    u32 region_count{};
    u32 copy_region_count{};
    u32 plane_bytes{};
    u32 first_plane_offset{};
    u32 second_plane_offset{};
    u32 total_bytes{};
    u32 image_barriers_per_draw{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy{};
    bool ends_rendering{};
    bool resumes_rendering_with_load{};
    bool preserves_rendering_serial{};
    bool callback_payload_is_scalar_only{};
    bool cpu_wait{};
    bool finish{};
};

[[nodiscard]] inline PpTerminalScopeContentPlan PlanPpTerminalScopeContent(
    const PpTerminalScopeContentDescriptor& descriptor) noexcept {
    if (!descriptor.enabled || !descriptor.armed) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status, u32 FinalGuestSurfaceLoss::* member) {
        PpTerminalScopeContentPlan plan{};
        plan.status = status;
        plan.loss.*member = 1;
        return plan;
    };
    if (descriptor.target_width != descriptor.final_source_width ||
        descriptor.target_height != descriptor.final_source_height) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::logical_mapping);
    }
    const auto footprint = PlanPpSourceBackingFootprints({
        .enabled = true,
        .in_window = true,
        .pp_draw_encoded = true,
        .fsr_bypassed = true,
        .source_width = descriptor.target_width,
        .source_height = descriptor.target_height,
        .logical_width = descriptor.logical_width,
        .logical_height = descriptor.logical_height,
        .source_format = descriptor.format,
        .samples = descriptor.samples,
        .resolved_base_mip = 0,
        .resolved_mip_count = 1,
        .resolved_base_layer = 0,
        .resolved_layer_count = 1,
        .bound_base_mip = 0,
        .bound_mip_count = 1,
        .bound_base_layer = 0,
        .bound_layer_count = 1,
        .logical_full_fit = true,
        .logical_top_left = true,
        .logical_no_y_flip = true,
        .buffer_alignment = descriptor.buffer_alignment,
        .max_regions = descriptor.max_regions,
        .max_bytes = descriptor.max_bytes,
        .selector = descriptor.selector,
    });
    if (footprint.status != FinalGuestSurfaceStatus::Complete) {
        PpTerminalScopeContentPlan plan{};
        plan.status = footprint.status;
        plan.loss = footprint.loss;
        return plan;
    }
    if (footprint.region_count > descriptor.max_regions / 2) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss, &FinalGuestSurfaceLoss::tile_capacity);
    }
    const u64 second_offset =
        AlignPpSourceBackingOffset(footprint.buffer_bytes, descriptor.buffer_alignment);
    const u64 total_bytes = second_offset + footprint.buffer_bytes;
    if (second_offset == std::numeric_limits<u64>::max() || total_bytes > descriptor.max_bytes ||
        total_bytes > std::numeric_limits<u32>::max()) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss, &FinalGuestSurfaceLoss::byte_capacity);
    }
    PpTerminalScopeContentPlan plan{
        .region_count = footprint.region_count,
        .copy_region_count = footprint.region_count * 2,
        .plane_bytes = footprint.buffer_bytes,
        .first_plane_offset = 0,
        .second_plane_offset = static_cast<u32>(second_offset),
        .total_bytes = static_cast<u32>(total_bytes),
        .image_barriers_per_draw = 2,
        .status = FinalGuestSurfaceStatus::Complete,
        .copy = true,
        .ends_rendering = true,
        .resumes_rendering_with_load = true,
        .preserves_rendering_serial = true,
        .callback_payload_is_scalar_only = true,
    };
    for (u32 index = 0; index < footprint.region_count; ++index) {
        plan.regions[index] = footprint.regions[index];
    }
    return plan;
}

struct PpTerminalScopeContentReport {
    u64 sequence{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    u32 draw_count{};
    u32 region_count{};
    u32 first_aba{};
    u32 first_stable{};
    u32 second_aba{};
    u32 second_stable{};
};

[[nodiscard]] inline std::string FormatPpTerminalScopeContentReport(
    const PpTerminalScopeContentReport& report) {
    return "FGSCTS s=" + std::to_string(report.sequence) +
           " st=" + std::to_string(static_cast<u32>(report.status)) +
           " d=" + std::to_string(report.draw_count) + " r=" + std::to_string(report.region_count) +
           " a0=" + std::to_string(report.first_aba) +
           " s0=" + std::to_string(report.first_stable) +
           " a1=" + std::to_string(report.second_aba) +
           " s1=" + std::to_string(report.second_stable) +
           " lm=" + std::to_string(report.loss.Any() ? 1 : 0);
}

} // namespace Vulkan
