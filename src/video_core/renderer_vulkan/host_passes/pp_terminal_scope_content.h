// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <charconv>
#include <deque>
#include <limits>
#include <span>
#include <string>
#include <vector>

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
    PpTerminalScopeDrawSelector consumer{};
};

struct PpTerminalScopeRuntimeConfig {
    PpTerminalScopeContentConfig content{};
    FinalGuestSurfaceCaptureWindow window{};
    FinalGuestSurfaceWatchOrdinals watch_ordinals{};
    u32 expected_calibrations{};
};

[[nodiscard]] inline std::optional<PpTerminalScopeDrawSelector> ParsePpTerminalScopeDrawSelector(
    std::string_view value) noexcept {
    std::array<std::string_view, 6> fields{};
    for (u32 index = 0; index < fields.size(); ++index) {
        const size_t separator = value.find(',');
        if (separator == std::string_view::npos) {
            if (index != fields.size() - 1) {
                return std::nullopt;
            }
            fields[index] = value;
            value = {};
        } else {
            fields[index] = value.substr(0, separator);
            value.remove_prefix(separator + 1);
        }
        if (fields[index].empty()) {
            return std::nullopt;
        }
    }
    if (!value.empty()) {
        return std::nullopt;
    }
    PpTerminalScopeDrawSelector selector{};
    if (fields[0] == "direct") {
        selector.kind = VideoCore::ImageColorScopeDrawKind::Direct;
    } else if (fields[0] == "indirect") {
        selector.kind = VideoCore::ImageColorScopeDrawKind::Indirect;
    } else {
        return std::nullopt;
    }
    if (fields[1] == "indexed") {
        selector.indexed = true;
    } else if (fields[1] == "nonindexed") {
        selector.indexed = false;
    } else {
        return std::nullopt;
    }
    const auto parse_u32 = [](std::string_view field, u32& output) {
        const auto result = std::from_chars(field.data(), field.data() + field.size(), output);
        return result.ec == std::errc{} && result.ptr == field.data() + field.size();
    };
    if (!parse_u32(fields[2], selector.element_count) ||
        !parse_u32(fields[3], selector.instance_count) ||
        !parse_u32(fields[4], selector.sampled_images) ||
        !parse_u32(fields[5], selector.storage_writes) || selector.instance_count == 0 ||
        selector.sampled_images == 0 ||
        (selector.kind == VideoCore::ImageColorScopeDrawKind::Direct &&
         selector.element_count == 0)) {
        return std::nullopt;
    }
    return selector;
}

template <typename ReadValue>
[[nodiscard]] std::optional<PpTerminalScopeRuntimeConfig> ResolvePpTerminalScopeRuntimeConfig(
    ReadValue&& read_value) {
    const auto enabled = read_value("SHADPS4_PP_TERMINAL_SCOPE_CONTENT");
    if (!enabled || *enabled != "1") {
        return std::nullopt;
    }
    const auto final_config = ResolveFinalGuestSurfaceContentConfig(read_value);
    if (!final_config ||
        final_config->stage != FinalGuestSurfaceStage::PpSourcePublicationReconstruction ||
        !final_config->calibrated_triplets || final_config->expected_calibrations == 0 ||
        final_config->watch_ordinals.status != FinalGuestSurfaceStatus::Complete ||
        final_config->watch_ordinals.loss != 0 || final_config->watch_ordinals.count == 0) {
        return std::nullopt;
    }
    const auto first_value = read_value("SHADPS4_PP_TERMINAL_SCOPE_FIRST");
    const auto second_value = read_value("SHADPS4_PP_TERMINAL_SCOPE_SECOND");
    const auto consumer_value = read_value("SHADPS4_PP_TERMINAL_SCOPE_CONSUMER");
    if (!first_value || !second_value || !consumer_value) {
        return std::nullopt;
    }
    const auto first = ParsePpTerminalScopeDrawSelector(*first_value);
    const auto second = ParsePpTerminalScopeDrawSelector(*second_value);
    const auto consumer = ParsePpTerminalScopeDrawSelector(*consumer_value);
    if (!first || !second || !consumer) {
        return std::nullopt;
    }
    return PpTerminalScopeRuntimeConfig{
        .content = {.enabled = true, .first = *first, .second = *second, .consumer = *consumer},
        .window = final_config->window,
        .watch_ordinals = final_config->watch_ordinals,
        .expected_calibrations = final_config->expected_calibrations,
    };
}

enum class PpTerminalScopeContentAction : u8 {
    None,
    CaptureFirst,
    CaptureSecond,
    ShapeLoss,
};

enum PpTerminalScopeConsumerAction : u8 {
    None,
    CaptureConsumer,
};

struct PpTerminalScopeRenderingSplitPlan {
    u64 serial{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool end_rendering{};
    bool resume_rendering{};
    bool force_load{};
    bool preserve_serial{};
};

struct PpTerminalScopeFlipDecision {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool use_existing_capture{};
    bool synthesize_loss{};
    bool arm_next{};
};

[[nodiscard]] constexpr PpTerminalScopeFlipDecision PlanPpTerminalScopeFlipDecision(
    bool has_existing_capture, bool target_valid, bool target_capacity, bool in_window) noexcept {
    if (has_existing_capture) {
        return {
            .use_existing_capture = true,
            .arm_next = target_valid && target_capacity,
        };
    }
    if (!target_valid) {
        return {
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
            .synthesize_loss = in_window,
        };
    }
    if (!target_capacity) {
        return {
            .status = FinalGuestSurfaceStatus::CapacityLoss,
            .loss = {.tile_capacity = 1},
            .synthesize_loss = in_window,
        };
    }
    return {
        .status = FinalGuestSurfaceStatus::GapLoss,
        .loss = {.gap = 1},
        .synthesize_loss = in_window,
        .arm_next = true,
    };
}

[[nodiscard]] constexpr PpTerminalScopeRenderingSplitPlan PlanPpTerminalScopeRenderingSplit(
    bool is_rendering, u64 current_serial, u64 expected_serial) noexcept {
    if (!is_rendering || current_serial == 0 || expected_serial == 0) {
        return {
            .status = FinalGuestSurfaceStatus::GapLoss,
            .loss = {.gap = 1},
        };
    }
    if (current_serial != expected_serial) {
        return {
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
        };
    }
    return {
        .serial = current_serial,
        .status = FinalGuestSurfaceStatus::Complete,
        .end_rendering = true,
        .resume_rendering = true,
        .force_load = true,
        .preserve_serial = true,
    };
}

struct PpTerminalScopeContentTakeResult {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    u32 draw_count{};
    u32 consumer_observations{};
    u32 consumer_phase_mask{};
    u32 consumer_shape_matches{};
    bool consumer_frozen{};
    bool cpu_wait{};
    bool finish{};
    bool retains_image{};
    bool retains_vk_image{};
};

struct PpTerminalScopeContentActionResult {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
};

[[nodiscard]] constexpr PpTerminalScopeContentActionResult ApplyPpTerminalScopeContentAction(
    FinalGuestSurfaceStatus current_status, FinalGuestSurfaceLoss current_loss,
    PpTerminalScopeContentAction action) noexcept {
    if (action == PpTerminalScopeContentAction::CaptureFirst) {
        return {.status = FinalGuestSurfaceStatus::Complete};
    }
    if (action == PpTerminalScopeContentAction::ShapeLoss) {
        return {
            .status = FinalGuestSurfaceStatus::GapLoss,
            .loss = {.gap = 1},
        };
    }
    return {.status = current_status, .loss = current_loss};
}

struct PpTerminalScopePlaneSlotDecision {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
    FinalGuestSurfaceLoss loss{};
    bool acquire{};
    bool reuse{};
};

[[nodiscard]] constexpr PpTerminalScopePlaneSlotDecision PlanPpTerminalScopePlaneSlot(
    u32 plane, bool has_slot) noexcept {
    if (plane > 2 || (plane == 1 && !has_slot)) {
        return {
            .status = FinalGuestSurfaceStatus::GapLoss,
            .loss = {.gap = 1},
        };
    }
    return {
        .acquire = (plane == 0 || plane == 2) && !has_slot,
        .reuse = has_slot,
    };
}

class PpTerminalScopeContentGate {
public:
    explicit constexpr PpTerminalScopeContentGate(PpTerminalScopeContentConfig config_) noexcept
        : config{config_} {}

    [[nodiscard]] constexpr bool Arm(u64 target_token, u64 generation) noexcept {
        if (!config.enabled || target_token == 0 || generation == 0 ||
            config.first.kind == VideoCore::ImageColorScopeDrawKind::Unknown ||
            config.second.kind == VideoCore::ImageColorScopeDrawKind::Unknown ||
            config.consumer.kind == VideoCore::ImageColorScopeDrawKind::Unknown) {
            Reset();
            return false;
        }
        token = target_token;
        armed_generation = generation;
        scope_serial = 0;
        phase = 0;
        frozen = false;
        consumer_observations = 0;
        consumer_phase_mask = 0;
        consumer_shape_matches = 0;
        return true;
    }

    [[nodiscard]] constexpr PpTerminalScopeContentAction ObserveDraw(
        u64 target_token, u64 observed_scope_serial,
        const PpTerminalScopeDrawSelector& draw) noexcept {
        if (target_token == 0 || target_token != token || observed_scope_serial == 0) {
            return PpTerminalScopeContentAction::None;
        }
        if (frozen) {
            return PpTerminalScopeContentAction::None;
        }
        if (scope_serial != observed_scope_serial) {
            scope_serial = observed_scope_serial;
            phase = 0;
        }
        if (phase == 0) {
            if (!MatchesPpTerminalScopeDraw(config.first, draw)) {
                phase = 3;
                return PpTerminalScopeContentAction::ShapeLoss;
            }
            phase = 1;
            return PpTerminalScopeContentAction::CaptureFirst;
        }
        if (phase == 1) {
            if (!MatchesPpTerminalScopeDraw(config.second, draw)) {
                phase = 3;
                return PpTerminalScopeContentAction::ShapeLoss;
            }
            phase = 2;
            return PpTerminalScopeContentAction::CaptureSecond;
        }
        if (phase == 2) {
            phase = 3;
            return PpTerminalScopeContentAction::ShapeLoss;
        }
        return PpTerminalScopeContentAction::None;
    }

    [[nodiscard]] constexpr PpTerminalScopeConsumerAction ObserveConsumer(
        u64 target_token, const PpTerminalScopeDrawSelector& consumer) noexcept {
        if (target_token == 0 || target_token != token || frozen) {
            return PpTerminalScopeConsumerAction::None;
        }
        if (consumer_observations != std::numeric_limits<u32>::max()) {
            ++consumer_observations;
        }
        consumer_phase_mask |= 1u << std::min(phase, 3u);
        const bool shape_matches = MatchesPpTerminalScopeDraw(config.consumer, consumer);
        if (shape_matches && consumer_shape_matches != std::numeric_limits<u32>::max()) {
            ++consumer_shape_matches;
        }
        if (phase != 2 || !shape_matches) {
            if (phase != 3 || !shape_matches) {
                return PpTerminalScopeConsumerAction::None;
            }
        }
        frozen = true;
        return PpTerminalScopeConsumerAction::CaptureConsumer;
    }

    [[nodiscard]] constexpr PpTerminalScopeContentTakeResult Take(u64 target_token,
                                                                  u64 generation) noexcept {
        PpTerminalScopeContentTakeResult result{};
        result.consumer_observations = consumer_observations;
        result.consumer_phase_mask = consumer_phase_mask;
        result.consumer_shape_matches = consumer_shape_matches;
        result.consumer_frozen = frozen;
        if (target_token != token || generation != armed_generation) {
            result.status = FinalGuestSurfaceStatus::InvalidationLoss;
            result.loss.invalidation = 1;
        } else if (!frozen) {
            result.status = FinalGuestSurfaceStatus::GapLoss;
            result.loss.gap = 1;
            result.draw_count = phase;
        } else {
            result.status = FinalGuestSurfaceStatus::Complete;
            result.draw_count = phase;
        }
        scope_serial = 0;
        phase = 0;
        frozen = false;
        consumer_observations = 0;
        consumer_phase_mask = 0;
        consumer_shape_matches = 0;
        return result;
    }

private:
    constexpr void Reset() noexcept {
        token = 0;
        armed_generation = 0;
        scope_serial = 0;
        phase = 0;
        frozen = false;
        consumer_observations = 0;
        consumer_phase_mask = 0;
        consumer_shape_matches = 0;
    }

    PpTerminalScopeContentConfig config{};
    u64 token{};
    u64 armed_generation{};
    u64 scope_serial{};
    u32 phase{};
    bool frozen{};
    u32 consumer_observations{};
    u32 consumer_phase_mask{};
    u32 consumer_shape_matches{};
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
    u32 consumer_plane_offset{};
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
    if (footprint.region_count > descriptor.max_regions / 3) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss, &FinalGuestSurfaceLoss::tile_capacity);
    }
    const u64 second_offset =
        AlignPpSourceBackingOffset(footprint.buffer_bytes, descriptor.buffer_alignment);
    const u64 consumer_offset = AlignPpSourceBackingOffset(second_offset + footprint.buffer_bytes,
                                                           descriptor.buffer_alignment);
    const u64 total_bytes = consumer_offset + footprint.buffer_bytes;
    if (second_offset == std::numeric_limits<u64>::max() || total_bytes > descriptor.max_bytes ||
        total_bytes > std::numeric_limits<u32>::max()) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss, &FinalGuestSurfaceLoss::byte_capacity);
    }
    PpTerminalScopeContentPlan plan{
        .region_count = footprint.region_count,
        .copy_region_count = footprint.region_count * 3,
        .plane_bytes = footprint.buffer_bytes,
        .first_plane_offset = 0,
        .second_plane_offset = static_cast<u32>(second_offset),
        .consumer_plane_offset = static_cast<u32>(consumer_offset),
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
    u32 plane_mask{};
    u32 consumer_observations{};
    u32 consumer_phase_mask{};
    u32 consumer_shape_matches{};
    bool consumer_frozen{};
    u32 first_aba{};
    u32 first_stable{};
    u32 second_aba{};
    u32 second_stable{};
};

[[nodiscard]] constexpr PpTerminalScopeContentReport MakePpTerminalScopeContentReport(
    u64 sequence, FinalGuestSurfaceStatus status, FinalGuestSurfaceLoss loss, u32 draw_count,
    u32 region_count, u32 consumer_observations = 0, u32 consumer_phase_mask = 0,
    u32 consumer_shape_matches = 0, bool consumer_frozen = false, u32 plane_mask = 0) noexcept {
    return {
        .sequence = sequence,
        .status = status,
        .loss = loss,
        .draw_count = draw_count,
        .region_count = region_count,
        .plane_mask = plane_mask,
        .consumer_observations = consumer_observations,
        .consumer_phase_mask = consumer_phase_mask,
        .consumer_shape_matches = consumer_shape_matches,
        .consumer_frozen = consumer_frozen,
    };
}

struct PpTerminalScopeContentHistoryRegion {
    u32 logical_ordinal{};
    u32 buffer_offset{};
    u32 byte_size{};

    auto operator<=>(const PpTerminalScopeContentHistoryRegion&) const = default;
};

struct PpTerminalScopeContentHistoryLayout {
    u32 region_count{};
    u32 plane_bytes{};
    u32 second_plane_offset{};
    u32 consumer_plane_offset{};
    u32 total_bytes{};
    u32 plane_mask{};
    std::array<PpTerminalScopeContentHistoryRegion, FinalGuestSurfaceWatchOrdinals::MaxOrdinals>
        regions{};
    u32 bytes_per_pixel{4};

    auto operator<=>(const PpTerminalScopeContentHistoryLayout&) const = default;
};

struct PpTerminalScopeCalibratedReport {
    u32 request_ordinal{};
    std::array<u64, 3> sequences{};
    std::vector<u32> first_aba_ordinals{};
    std::vector<u32> first_stable_ordinals{};
    std::vector<u32> first_ambiguous_ordinals{};
    std::vector<u32> second_aba_ordinals{};
    std::vector<u32> second_stable_ordinals{};
    std::vector<u32> second_ambiguous_ordinals{};
    std::vector<u32> consumer_aba_ordinals{};
    std::vector<u32> consumer_stable_ordinals{};
    std::vector<u32> consumer_ambiguous_ordinals{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
};

struct PpTerminalScopeCalibratedCoverage {
    u32 calibrations{};
    u32 outside{};
    u32 eligible{};
    u32 emitted{};
    u32 complete{};
    u32 loss{};
    bool ready{};
};

class PpTerminalScopeContentReducer {
public:
    static constexpr u32 MaxCalibrations = FinalGuestSurfaceMaxScreenshotRequests;

    explicit PpTerminalScopeContentReducer(FinalGuestSurfaceCaptureWindow window_,
                                           u32 history_capacity_ = 32)
        : window{window_}, history_capacity{std::clamp(history_capacity_, 1u, 32u)} {}

    void ObserveContent(u64 sequence, const PpTerminalScopeContentHistoryLayout& layout,
                        std::span<const std::byte> bytes, FinalGuestSurfaceStatus status,
                        FinalGuestSurfaceLoss loss) {
        if (!window.Contains(sequence)) {
            return;
        }
        if (history.size() == history_capacity) {
            history.pop_front();
        }
        history.push_back({
            .sequence = sequence,
            .layout = layout,
            .bytes = std::vector<std::byte>{bytes.begin(), bytes.end()},
            .status = status,
            .loss = loss,
        });
        last_content_sequence = std::max(last_content_sequence, sequence);
        Reconcile();
    }

    void ObserveCalibration(FinalGuestSurfaceCalibratedStamp stamp) {
        if (stamp.request_ordinal == 0 || stamp.request_ordinal > MaxCalibrations ||
            calibrations[stamp.request_ordinal].has_value()) {
            ++coverage_loss;
            return;
        }
        calibrations[stamp.request_ordinal] = stamp;
        ++calibration_count;
        Reconcile();
    }

    [[nodiscard]] std::vector<PpTerminalScopeCalibratedReport> TakeReports() {
        auto result = std::move(reports);
        reports.clear();
        return result;
    }

    [[nodiscard]] PpTerminalScopeCalibratedCoverage GetCoverage(
        u32 expected_calibrations) const noexcept {
        const u32 bounded_expected = std::min(expected_calibrations, MaxCalibrations);
        const u32 missing =
            bounded_expected > calibration_count ? bounded_expected - calibration_count : 0;
        const u32 loss =
            coverage_loss + missing +
            (expected_calibrations > MaxCalibrations ? expected_calibrations - MaxCalibrations : 0);
        return {
            .calibrations = calibration_count,
            .outside = outside_count,
            .eligible = eligible_count,
            .emitted = emitted_count,
            .complete = complete_count,
            .loss = loss,
            .ready = missing == 0 && loss == 0 && emitted_count == eligible_count,
        };
    }

private:
    struct Observation {
        u64 sequence{};
        PpTerminalScopeContentHistoryLayout layout{};
        std::vector<std::byte> bytes{};
        FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss loss{};
    };

    [[nodiscard]] const Observation* Find(u64 sequence) const noexcept {
        const auto found = std::ranges::find(history, sequence, &Observation::sequence);
        return found == history.end() ? nullptr : &*found;
    }

    static void AddLoss(FinalGuestSurfaceLoss& target, const FinalGuestSurfaceLoss& source) {
        target.unsupported_type += source.unsupported_type;
        target.unsupported_samples += source.unsupported_samples;
        target.unsupported_mip += source.unsupported_mip;
        target.unsupported_layer += source.unsupported_layer;
        target.unsupported_aspect += source.unsupported_aspect;
        target.unsupported_format += source.unsupported_format;
        target.invalid_extent += source.invalid_extent;
        target.logical_mapping += source.logical_mapping;
        target.tile_capacity += source.tile_capacity;
        target.byte_capacity += source.byte_capacity;
        target.ordinal_capacity += source.ordinal_capacity;
        target.busy += source.busy;
        target.invalidation += source.invalidation;
        target.gap += source.gap;
        target.history += source.history;
        target.tile_detail += source.tile_detail;
    }

    [[nodiscard]] static bool EqualVisibleRegion(const Observation& left, const Observation& right,
                                                 u32 plane, u32 region_index) noexcept {
        if (left.layout != right.layout || region_index >= left.layout.region_count || plane > 2 ||
            left.layout.bytes_per_pixel != 4) {
            return false;
        }
        const auto& region = left.layout.regions[region_index];
        const auto plane_offset = [plane](const PpTerminalScopeContentHistoryLayout& layout) {
            return plane == 0
                       ? 0u
                       : (plane == 1 ? layout.second_plane_offset : layout.consumer_plane_offset);
        };
        const u64 left_offset = region.buffer_offset + plane_offset(left.layout);
        const u64 right_offset = region.buffer_offset + plane_offset(right.layout);
        if (region.byte_size == 0 || region.byte_size % left.layout.bytes_per_pixel != 0 ||
            left_offset + region.byte_size > left.bytes.size() ||
            right_offset + region.byte_size > right.bytes.size()) {
            return false;
        }
        for (u32 offset = 0; offset < region.byte_size; ++offset) {
            if (offset % left.layout.bytes_per_pixel == 3) {
                continue;
            }
            if (left.bytes[left_offset + offset] != right.bytes[right_offset + offset]) {
                return false;
            }
        }
        return true;
    }

    static void ClassifyPlane(const Observation& a, const Observation& b, const Observation& c,
                              u32 plane, std::vector<u32>& aba, std::vector<u32>& stable,
                              std::vector<u32>& ambiguous) {
        for (u32 index = 0; index < a.layout.region_count; ++index) {
            const bool ab = EqualVisibleRegion(a, b, plane, index);
            const bool bc = EqualVisibleRegion(b, c, plane, index);
            const bool ac = EqualVisibleRegion(a, c, plane, index);
            const u32 ordinal = a.layout.regions[index].logical_ordinal;
            if (ac && !ab) {
                aba.push_back(ordinal);
            } else if (ab && bc) {
                stable.push_back(ordinal);
            } else {
                ambiguous.push_back(ordinal);
            }
        }
    }

    [[nodiscard]] PpTerminalScopeCalibratedReport Evaluate(u32 request, const Observation* a,
                                                           const Observation* b,
                                                           const Observation* c) const {
        PpTerminalScopeCalibratedReport report{
            .request_ordinal = request,
            .sequences = {calibrations[request - 2]->sequence, calibrations[request - 1]->sequence,
                          calibrations[request]->sequence},
            .status = FinalGuestSurfaceStatus::Complete,
        };
        if (!a || !b || !c) {
            report.status = FinalGuestSurfaceStatus::GapLoss;
            report.loss.history = 1;
            return report;
        }
        AddLoss(report.loss, a->loss);
        AddLoss(report.loss, b->loss);
        AddLoss(report.loss, c->loss);
        if (a->status != FinalGuestSurfaceStatus::Complete ||
            b->status != FinalGuestSurfaceStatus::Complete ||
            c->status != FinalGuestSurfaceStatus::Complete || report.loss.Any()) {
            report.status = FinalGuestSurfaceStatus::GapLoss;
            if (!report.loss.Any()) {
                report.loss.gap = 1;
            }
            return report;
        }
        if (a->layout != b->layout || a->layout != c->layout ||
            a->bytes.size() != a->layout.total_bytes || b->bytes.size() != b->layout.total_bytes ||
            c->bytes.size() != c->layout.total_bytes) {
            report.status = FinalGuestSurfaceStatus::InvalidationLoss;
            report.loss.invalidation = 1;
            return report;
        }
        if ((a->layout.plane_mask & (1u << 0)) != 0) {
            ClassifyPlane(*a, *b, *c, 0, report.first_aba_ordinals, report.first_stable_ordinals,
                          report.first_ambiguous_ordinals);
        }
        if ((a->layout.plane_mask & (1u << 1)) != 0) {
            ClassifyPlane(*a, *b, *c, 1, report.second_aba_ordinals, report.second_stable_ordinals,
                          report.second_ambiguous_ordinals);
        }
        if ((a->layout.plane_mask & (1u << 2)) != 0) {
            ClassifyPlane(*a, *b, *c, 2, report.consumer_aba_ordinals,
                          report.consumer_stable_ordinals, report.consumer_ambiguous_ordinals);
        }
        return report;
    }

    void Reconcile() {
        for (u32 request = 3; request <= MaxCalibrations; ++request) {
            if (classified[request] || !calibrations[request] || !calibrations[request - 1] ||
                !calibrations[request - 2]) {
                continue;
            }
            const auto& a_stamp = *calibrations[request - 2];
            const auto& b_stamp = *calibrations[request - 1];
            const auto& c_stamp = *calibrations[request];
            if (!a_stamp.valid || !b_stamp.valid || !c_stamp.valid ||
                !window.Contains(a_stamp.sequence) || !window.Contains(b_stamp.sequence) ||
                !window.Contains(c_stamp.sequence)) {
                classified[request] = true;
                ++outside_count;
                continue;
            }
            if (!eligible_request[request]) {
                eligible_request[request] = true;
                ++eligible_count;
            }
            const auto* a = Find(a_stamp.sequence);
            const auto* b = Find(b_stamp.sequence);
            const auto* c = Find(c_stamp.sequence);
            if ((!a || !b || !c) && last_content_sequence < c_stamp.sequence) {
                continue;
            }
            auto report = Evaluate(request, a, b, c);
            ++emitted_count;
            complete_count +=
                report.status == FinalGuestSurfaceStatus::Complete && !report.loss.Any();
            coverage_loss +=
                report.status != FinalGuestSurfaceStatus::Complete || report.loss.Any();
            reports.push_back(std::move(report));
            classified[request] = true;
        }
    }

    FinalGuestSurfaceCaptureWindow window{};
    u32 history_capacity{};
    std::deque<Observation> history{};
    std::array<std::optional<FinalGuestSurfaceCalibratedStamp>, MaxCalibrations + 1> calibrations{};
    std::array<bool, MaxCalibrations + 1> classified{};
    std::array<bool, MaxCalibrations + 1> eligible_request{};
    std::vector<PpTerminalScopeCalibratedReport> reports{};
    u64 last_content_sequence{};
    u32 calibration_count{};
    u32 outside_count{};
    u32 eligible_count{};
    u32 emitted_count{};
    u32 complete_count{};
    u32 coverage_loss{};
};

[[nodiscard]] inline std::string FormatPpTerminalScopeOrdinalList(std::span<const u32> ordinals) {
    if (ordinals.empty()) {
        return "-";
    }
    std::string result{};
    for (const u32 ordinal : ordinals) {
        if (!result.empty()) {
            result += ',';
        }
        result += std::to_string(ordinal);
    }
    return result;
}

[[nodiscard]] inline std::string FormatPpTerminalScopeCalibratedReport(
    const PpTerminalScopeCalibratedReport& report) {
    return "FGSCTST q=" + std::to_string(report.request_ordinal) +
           " abc=" + std::to_string(report.sequences[0]) + '/' +
           std::to_string(report.sequences[1]) + '/' + std::to_string(report.sequences[2]) +
           " a0=" + FormatPpTerminalScopeOrdinalList(report.first_aba_ordinals) +
           " s0=" + FormatPpTerminalScopeOrdinalList(report.first_stable_ordinals) +
           " x0=" + FormatPpTerminalScopeOrdinalList(report.first_ambiguous_ordinals) +
           " a1=" + FormatPpTerminalScopeOrdinalList(report.second_aba_ordinals) +
           " s1=" + FormatPpTerminalScopeOrdinalList(report.second_stable_ordinals) +
           " x1=" + FormatPpTerminalScopeOrdinalList(report.second_ambiguous_ordinals) +
           " a2=" + FormatPpTerminalScopeOrdinalList(report.consumer_aba_ordinals) +
           " s2=" + FormatPpTerminalScopeOrdinalList(report.consumer_stable_ordinals) +
           " x2=" + FormatPpTerminalScopeOrdinalList(report.consumer_ambiguous_ordinals) +
           " st=" + std::to_string(static_cast<u32>(report.status)) +
           " lm=" + std::to_string(report.loss.Any() ? 1 : 0);
}

[[nodiscard]] inline std::string FormatPpTerminalScopeCalibratedCoverage(
    const PpTerminalScopeCalibratedCoverage& coverage) {
    return "FGSCTSTC c=" + std::to_string(coverage.calibrations) +
           " o=" + std::to_string(coverage.outside) + " e=" + std::to_string(coverage.eligible) +
           '/' + std::to_string(coverage.emitted) + '/' + std::to_string(coverage.complete) +
           " lm=" + std::to_string(coverage.loss);
}

[[nodiscard]] inline std::string FormatPpTerminalScopeContentReport(
    const PpTerminalScopeContentReport& report) {
    return "FGSCTS s=" + std::to_string(report.sequence) +
           " st=" + std::to_string(static_cast<u32>(report.status)) +
           " d=" + std::to_string(report.draw_count) + " r=" + std::to_string(report.region_count) +
           " pm=" + std::to_string(report.plane_mask) +
           " co=" + std::to_string(report.consumer_observations) +
           " cp=" + std::to_string(report.consumer_phase_mask) +
           " cm=" + std::to_string(report.consumer_shape_matches) +
           " cf=" + std::to_string(report.consumer_frozen ? 1 : 0) +
           " a0=" + std::to_string(report.first_aba) +
           " s0=" + std::to_string(report.first_stable) +
           " a1=" + std::to_string(report.second_aba) +
           " s1=" + std::to_string(report.second_stable) +
           " lm=" + std::to_string(report.loss.Any() ? 1 : 0);
}

} // namespace Vulkan
