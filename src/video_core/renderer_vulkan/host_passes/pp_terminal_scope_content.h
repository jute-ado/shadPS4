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
    bool join_final_backing{};
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
    const auto final_backing_join_value = read_value("SHADPS4_PP_TERMINAL_FINAL_BACKING_JOIN");
    if (!first_value || !second_value || !consumer_value) {
        return std::nullopt;
    }
    if (final_backing_join_value && *final_backing_join_value != "1") {
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
        .join_final_backing = final_backing_join_value.has_value(),
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

struct PpTerminalScopeDiscoveryDecision {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool allocate{};
    bool arm{};
    bool capture_current_draw{};
};

struct PpTerminalScopeDiscoveryObservation {
    bool exact_candidate{};
    bool tracked{};
    bool mapping_valid{};
    bool target_valid{};
    bool capacity{};
    bool allocated{};
    bool restarted{};
};

struct PpTerminalScopeDiscoveryCoverage {
    u32 candidates{};
    u32 tracked{};
    u32 allocated{};
    u32 restarted{};
    u32 mapping_rejected{};
    u32 target_rejected{};
    u32 capacity_rejected{};
    u32 root_accepted{};
    u32 root_missing{};
    u32 root_hazard{};
    u32 root_structure{};
    u32 root_plan{};
    u32 root_plan_loss_mask{};
    u32 root_view_valid{};
    u32 root_image_invalid{};
    u32 root_view_missing{};
    u32 root_view_ambiguous{};
    u32 root_view_mip{};
    u32 root_view_layer{};
    u32 root_view_invalid{};

    constexpr void Observe(PpTerminalScopeDiscoveryObservation observation) noexcept {
        if (!observation.exact_candidate) {
            return;
        }
        const auto increment = [](u32& value) {
            if (value != std::numeric_limits<u32>::max()) {
                ++value;
            }
        };
        increment(candidates);
        if (observation.tracked) {
            increment(tracked);
            if (observation.restarted) {
                increment(restarted);
            }
            return;
        }
        if (!observation.mapping_valid) {
            increment(mapping_rejected);
        } else if (!observation.target_valid) {
            increment(target_rejected);
        } else if (!observation.capacity) {
            increment(capacity_rejected);
        } else if (observation.allocated) {
            increment(allocated);
        }
    }

    template <typename Descriptor>
    constexpr void ObserveRootInput(const Descriptor& descriptor, bool structure_valid,
                                    FinalGuestSurfaceStatus plan_status,
                                    const FinalGuestSurfaceLoss& plan_loss) noexcept {
        if (!descriptor.capture_first) {
            return;
        }
        const auto increment = [](u32& value) {
            if (value != std::numeric_limits<u32>::max()) {
                ++value;
            }
        };
        if (!descriptor.single_sampled_input || !descriptor.sampled_input_valid) {
            increment(root_missing);
        } else if (descriptor.sampled_input_alias || !descriptor.read_only) {
            increment(root_hazard);
        } else if (!structure_valid) {
            increment(root_structure);
        } else if (plan_status != FinalGuestSurfaceStatus::Complete || plan_loss.Any()) {
            increment(root_plan);
            root_plan_loss_mask |= FinalGuestSurfaceLossMask(plan_loss, 0);
        } else {
            increment(root_accepted);
        }
    }

    constexpr void ObserveRootView(bool image_valid, bool view_present, bool view_ambiguous,
                                   FinalGuestSurfaceStatus view_status,
                                   const FinalGuestSurfaceLoss& view_loss) noexcept {
        const auto increment = [](u32& value) {
            if (value != std::numeric_limits<u32>::max()) {
                ++value;
            }
        };
        if (!image_valid) {
            increment(root_image_invalid);
        } else if (view_ambiguous) {
            increment(root_view_ambiguous);
        } else if (!view_present) {
            increment(root_view_missing);
        } else if (view_status == FinalGuestSurfaceStatus::Complete && !view_loss.Any()) {
            increment(root_view_valid);
        } else if (view_loss.unsupported_mip) {
            increment(root_view_mip);
        } else if (view_loss.unsupported_layer) {
            increment(root_view_layer);
        } else {
            increment(root_view_invalid);
        }
    }
};

class PpTerminalScopeDiscoveryCoverageEmissionGate {
public:
    [[nodiscard]] constexpr bool Observe(const FinalGuestSurfaceCaptureWindow& window,
                                         u64 sequence) noexcept {
        if (emitted || !window.IsFinal(sequence)) {
            return false;
        }
        emitted = true;
        return true;
    }

    [[nodiscard]] constexpr bool Finalize() noexcept {
        if (emitted) {
            return false;
        }
        emitted = true;
        return true;
    }

private:
    bool emitted{};
};

struct PpTerminalScopePrivateLineageExtension {
    VideoCore::ImageColorScopePrivateLink input{};
    u64 input_generation{};
    VideoCore::ImageColorScopePrivateLink output{};
    u64 output_generation{};
    bool single_input{};
    bool single_output{};
};

struct PpTerminalScopePrivateLineageResult {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    u32 hops{};
    bool matched{};
    bool retains_pointer{};
    bool retains_image{};
    bool retains_vk_image{};
};

class PpTerminalScopePrivateLineage {
public:
    static constexpr u32 MaxDepth = VideoCore::MaxImageColorScopeAncestryDepth;

    [[nodiscard]] constexpr bool Start(VideoCore::ImageColorScopePrivateLink root,
                                       u64 generation) noexcept {
        Reset();
        if (!root.Valid() || generation == 0) {
            Fail(FinalGuestSurfaceStatus::InvalidationLoss, &FinalGuestSurfaceLoss::invalidation);
            return false;
        }
        nodes[0] = {.link = root, .generation = generation};
        depth = 1;
        status = FinalGuestSurfaceStatus::Complete;
        return true;
    }

    [[nodiscard]] constexpr PpTerminalScopePrivateLineageResult Extend(
        const PpTerminalScopePrivateLineageExtension& extension) noexcept {
        if (status != FinalGuestSurfaceStatus::Complete || depth == 0) {
            return Result(false);
        }
        if (!extension.single_input || !extension.single_output) {
            Fail(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
            return Result(false);
        }
        const auto& tail = nodes[depth - 1];
        if (!extension.input.Valid() || extension.input != tail.link ||
            extension.input_generation == 0 || extension.input_generation != tail.generation ||
            !extension.output.Valid() || extension.output_generation == 0) {
            Fail(FinalGuestSurfaceStatus::InvalidationLoss, &FinalGuestSurfaceLoss::invalidation);
            return Result(false);
        }
        for (u32 index = 0; index < depth; ++index) {
            if (nodes[index].link == extension.output) {
                Fail(FinalGuestSurfaceStatus::InvalidationLoss,
                     &FinalGuestSurfaceLoss::invalidation);
                return Result(false);
            }
        }
        if (depth == MaxDepth) {
            Fail(FinalGuestSurfaceStatus::CapacityLoss, &FinalGuestSurfaceLoss::tile_capacity);
            return Result(false);
        }
        nodes[depth++] = {.link = extension.output, .generation = extension.output_generation};
        return Result(true);
    }

    [[nodiscard]] constexpr PpTerminalScopePrivateLineageResult Resolve(
        VideoCore::ImageColorScopePrivateLink output, u64 generation) const noexcept {
        if (status != FinalGuestSurfaceStatus::Complete || depth == 0 || !output.Valid() ||
            generation == 0 || nodes[depth - 1].link != output ||
            nodes[depth - 1].generation != generation) {
            PpTerminalScopePrivateLineageResult result = Result(false);
            if (result.status == FinalGuestSurfaceStatus::Complete) {
                result.status = FinalGuestSurfaceStatus::InvalidationLoss;
                result.loss = {.invalidation = 1};
            }
            return result;
        }
        return Result(true);
    }

    [[nodiscard]] constexpr VideoCore::ImageColorScopePrivateLink Root() const noexcept {
        return depth == 0 ? VideoCore::ImageColorScopePrivateLink{} : nodes[0].link;
    }

    [[nodiscard]] constexpr u64 RootGeneration() const noexcept {
        return depth == 0 ? 0 : nodes[0].generation;
    }

    [[nodiscard]] constexpr VideoCore::ImageColorScopePrivateLink Tail() const noexcept {
        return depth == 0 ? VideoCore::ImageColorScopePrivateLink{} : nodes[depth - 1].link;
    }

    [[nodiscard]] constexpr u32 Hops() const noexcept {
        return depth == 0 ? 0 : depth - 1;
    }

    [[nodiscard]] constexpr FinalGuestSurfaceStatus Status() const noexcept {
        return status;
    }

    [[nodiscard]] constexpr FinalGuestSurfaceLoss Loss() const noexcept {
        return loss;
    }

    constexpr void Reset() noexcept {
        nodes = {};
        depth = 0;
        status = FinalGuestSurfaceStatus::AlreadyConsumed;
        loss = {};
    }

private:
    struct Node {
        VideoCore::ImageColorScopePrivateLink link{};
        u64 generation{};
    };

    [[nodiscard]] constexpr PpTerminalScopePrivateLineageResult Result(
        bool matched) const noexcept {
        return {
            .status = status,
            .loss = loss,
            .hops = Hops(),
            .matched = matched,
        };
    }

    constexpr void Fail(FinalGuestSurfaceStatus next_status,
                        u32 FinalGuestSurfaceLoss::* member) noexcept {
        status = next_status;
        loss = {};
        loss.*member = 1;
    }

    std::array<Node, MaxDepth> nodes{};
    u32 depth{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
};

struct PpTerminalScopePrivateLineageReport {
    u32 hops{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
};

[[nodiscard]] constexpr PpTerminalScopeDiscoveryDecision PlanPpTerminalScopeDiscoveryDecision(
    bool enabled, bool already_tracked, bool first_selector_matches, bool mapping_valid,
    bool target_valid, bool target_capacity) noexcept {
    if (!enabled || already_tracked || !first_selector_matches) {
        return {};
    }
    if (!mapping_valid || !target_valid) {
        return {
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
        };
    }
    if (!target_capacity) {
        return {
            .status = FinalGuestSurfaceStatus::CapacityLoss,
            .loss = {.tile_capacity = 1},
        };
    }
    return {
        .status = FinalGuestSurfaceStatus::Complete,
        .allocate = true,
        .arm = true,
        .capture_current_draw = true,
    };
}

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

struct PpTerminalScopeRootInputCaptureDescriptor {
    bool capture_first{};
    bool single_sampled_input{};
    bool sampled_input_valid{};
    bool sampled_input_alias{};
    bool read_only{};
    bool compatible_layout{};
};

struct PpTerminalScopeRootInputCaptureDecision {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    u32 plane{};
    bool capture{};
    bool post_draw{};
    bool cpu_wait{};
    bool finish{};
};

[[nodiscard]] constexpr PpTerminalScopeRootInputCaptureDecision PlanPpTerminalScopeRootInputCapture(
    const PpTerminalScopeRootInputCaptureDescriptor& descriptor) noexcept {
    if (!descriptor.capture_first) {
        return {};
    }
    if (!descriptor.single_sampled_input || !descriptor.sampled_input_valid) {
        return {.status = FinalGuestSurfaceStatus::GapLoss, .loss = {.gap = 1}};
    }
    if (descriptor.sampled_input_alias || !descriptor.read_only) {
        return {
            .status = FinalGuestSurfaceStatus::Unsupported,
            .loss = {.unsupported_type = 1},
        };
    }
    if (!descriptor.compatible_layout) {
        return {
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
        };
    }
    return {
        .status = FinalGuestSurfaceStatus::Complete,
        .plane = 4,
        .capture = true,
        .post_draw = true,
    };
}

struct PpTerminalScopeRootViewCaptureDescriptor {
    u32 image_width{};
    u32 image_height{};
    u32 image_levels{};
    u32 image_layers{};
    u32 view_base_mip{};
    u32 view_mip_count{};
    u32 view_base_layer{};
    u32 view_layer_count{};
};

struct PpTerminalScopeRootViewCaptureDecision {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    u32 source_width{};
    u32 source_height{};
    u32 mip{};
    u32 layer{};
    u32 state_index{};
    bool capture_partial_subresource{};
    bool restore_exact_tracker_state{};
    bool cpu_wait{};
    bool finish{};
};

[[nodiscard]] constexpr PpTerminalScopeRootViewCaptureDecision PlanPpTerminalScopeRootViewCapture(
    const PpTerminalScopeRootViewCaptureDescriptor& descriptor) noexcept {
    if (descriptor.image_width == 0 || descriptor.image_height == 0 ||
        descriptor.image_levels == 0 || descriptor.image_layers == 0 ||
        descriptor.view_base_mip >= descriptor.image_levels ||
        descriptor.view_base_layer >= descriptor.image_layers ||
        descriptor.view_mip_count > descriptor.image_levels - descriptor.view_base_mip ||
        descriptor.view_layer_count > descriptor.image_layers - descriptor.view_base_layer) {
        return {
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
        };
    }
    if (descriptor.view_mip_count != 1 || descriptor.view_layer_count != 1) {
        FinalGuestSurfaceLoss loss{};
        if (descriptor.view_mip_count != 1) {
            loss.unsupported_mip = 1;
        }
        if (descriptor.view_layer_count != 1) {
            loss.unsupported_layer = 1;
        }
        return {.status = FinalGuestSurfaceStatus::Unsupported, .loss = loss};
    }
    const u64 state_index = static_cast<u64>(descriptor.view_base_mip) * descriptor.image_layers +
                            descriptor.view_base_layer;
    if (state_index > std::numeric_limits<u32>::max()) {
        return {
            .status = FinalGuestSurfaceStatus::CapacityLoss,
            .loss = {.tile_capacity = 1},
        };
    }
    return {
        .status = FinalGuestSurfaceStatus::Complete,
        .source_width = std::max(1u, descriptor.image_width >> descriptor.view_base_mip),
        .source_height = std::max(1u, descriptor.image_height >> descriptor.view_base_mip),
        .mip = descriptor.view_base_mip,
        .layer = descriptor.view_base_layer,
        .state_index = static_cast<u32>(state_index),
        .capture_partial_subresource = true,
        .restore_exact_tracker_state = true,
    };
}

[[nodiscard]] constexpr PpTerminalScopePlaneSlotDecision PlanPpTerminalScopePlaneSlot(
    u32 plane, bool has_slot) noexcept {
    if (plane > 4 || (plane == 4 && has_slot) ||
        ((plane == 0 || plane == 1 || plane == 3) && !has_slot)) {
        return {
            .status = FinalGuestSurfaceStatus::GapLoss,
            .loss = {.gap = 1},
        };
    }
    return {
        .acquire = (plane == 2 || plane == 4) && !has_slot,
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

    [[nodiscard]] constexpr bool CanRestartAtFirst(
        u64 target_token, u64 observed_scope_serial,
        const PpTerminalScopeDrawSelector& draw) const noexcept {
        return target_token != 0 && target_token == token && observed_scope_serial != 0 &&
               observed_scope_serial == scope_serial && phase == 3 && !frozen &&
               MatchesPpTerminalScopeDraw(config.first, draw);
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
    std::array<PpSourceBackingRegion, FinalGuestSurfaceWatchOrdinals::MaxOrdinals>
        root_input_regions{};
    u32 region_count{};
    u32 root_input_region_count{};
    u32 copy_region_count{};
    u32 target_width{};
    u32 target_height{};
    u32 logical_width{};
    u32 logical_height{};
    u32 plane_bytes{};
    u32 root_input_plane_bytes{};
    u32 first_plane_offset{};
    u32 second_plane_offset{};
    u32 consumer_plane_offset{};
    u32 output_plane_offset{};
    u32 root_input_plane_offset{};
    u32 total_bytes{};
    u32 image_barriers_per_draw{};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceFormat root_input_format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceWatchOrdinals selector{};
    u32 buffer_alignment{};
    u32 max_regions{};
    u32 max_bytes{};
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
    if (footprint.region_count > descriptor.max_regions / 5) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss, &FinalGuestSurfaceLoss::tile_capacity);
    }
    const u64 second_offset =
        AlignPpSourceBackingOffset(footprint.buffer_bytes, descriptor.buffer_alignment);
    const u64 consumer_offset = AlignPpSourceBackingOffset(second_offset + footprint.buffer_bytes,
                                                           descriptor.buffer_alignment);
    const u64 output_offset = AlignPpSourceBackingOffset(consumer_offset + footprint.buffer_bytes,
                                                         descriptor.buffer_alignment);
    const u64 root_input_offset = AlignPpSourceBackingOffset(output_offset + footprint.buffer_bytes,
                                                             descriptor.buffer_alignment);
    const u64 total_bytes = root_input_offset == std::numeric_limits<u64>::max()
                                ? root_input_offset
                                : root_input_offset + footprint.buffer_bytes;
    if (second_offset == std::numeric_limits<u64>::max() ||
        consumer_offset == std::numeric_limits<u64>::max() ||
        output_offset == std::numeric_limits<u64>::max() ||
        root_input_offset == std::numeric_limits<u64>::max() ||
        total_bytes > descriptor.max_bytes || total_bytes > std::numeric_limits<u32>::max()) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss, &FinalGuestSurfaceLoss::byte_capacity);
    }
    PpTerminalScopeContentPlan plan{
        .region_count = footprint.region_count,
        .root_input_region_count = footprint.region_count,
        .copy_region_count = footprint.region_count * 5,
        .target_width = descriptor.target_width,
        .target_height = descriptor.target_height,
        .logical_width = descriptor.logical_width,
        .logical_height = descriptor.logical_height,
        .plane_bytes = footprint.buffer_bytes,
        .root_input_plane_bytes = footprint.buffer_bytes,
        .first_plane_offset = 0,
        .second_plane_offset = static_cast<u32>(second_offset),
        .consumer_plane_offset = static_cast<u32>(consumer_offset),
        .output_plane_offset = static_cast<u32>(output_offset),
        .root_input_plane_offset = static_cast<u32>(root_input_offset),
        .total_bytes = static_cast<u32>(total_bytes),
        .image_barriers_per_draw = 2,
        .format = footprint.format,
        .root_input_format = footprint.format,
        .selector = descriptor.selector,
        .buffer_alignment = descriptor.buffer_alignment,
        .max_regions = descriptor.max_regions,
        .max_bytes = descriptor.max_bytes,
        .status = FinalGuestSurfaceStatus::Complete,
        .copy = true,
        .ends_rendering = true,
        .resumes_rendering_with_load = true,
        .preserves_rendering_serial = true,
        .callback_payload_is_scalar_only = true,
    };
    for (u32 index = 0; index < footprint.region_count; ++index) {
        plan.regions[index] = footprint.regions[index];
        plan.root_input_regions[index] = footprint.regions[index];
    }
    return plan;
}

struct PpTerminalScopeRootInputPlanDescriptor {
    u32 source_width{};
    u32 source_height{};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    u32 samples{};
};

[[nodiscard]] inline PpTerminalScopeContentPlan AttachPpTerminalScopeRootInputPlan(
    PpTerminalScopeContentPlan plan,
    const PpTerminalScopeRootInputPlanDescriptor& descriptor) noexcept {
    if (plan.status != FinalGuestSurfaceStatus::Complete || !plan.copy) {
        return plan;
    }
    const auto reject = [&](FinalGuestSurfaceStatus status, const FinalGuestSurfaceLoss& loss) {
        plan.status = status;
        plan.loss = loss;
        plan.copy = false;
        return plan;
    };
    const auto root = PlanPpSourceBackingFootprints({
        .enabled = true,
        .in_window = true,
        .pp_draw_encoded = true,
        .fsr_bypassed = true,
        .source_width = descriptor.source_width,
        .source_height = descriptor.source_height,
        .logical_width = plan.logical_width,
        .logical_height = plan.logical_height,
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
        .buffer_alignment = plan.buffer_alignment,
        .max_regions = plan.max_regions,
        .max_bytes = plan.max_bytes,
        .selector = plan.selector,
    });
    if (root.status != FinalGuestSurfaceStatus::Complete) {
        return reject(root.status, root.loss);
    }
    if (static_cast<u64>(plan.region_count) * 4 + root.region_count > plan.max_regions) {
        FinalGuestSurfaceLoss loss{};
        loss.tile_capacity = 1;
        return reject(FinalGuestSurfaceStatus::CapacityLoss, loss);
    }
    const u64 root_offset = AlignPpSourceBackingOffset(
        static_cast<u64>(plan.output_plane_offset) + plan.plane_bytes, plan.buffer_alignment);
    const u64 total_bytes = root_offset == std::numeric_limits<u64>::max()
                                ? root_offset
                                : root_offset + root.buffer_bytes;
    if (root_offset == std::numeric_limits<u64>::max() || total_bytes > plan.max_bytes ||
        total_bytes > std::numeric_limits<u32>::max()) {
        FinalGuestSurfaceLoss loss{};
        loss.byte_capacity = 1;
        return reject(FinalGuestSurfaceStatus::CapacityLoss, loss);
    }
    plan.root_input_regions = {};
    for (u32 index = 0; index < root.region_count; ++index) {
        plan.root_input_regions[index] = root.regions[index];
    }
    plan.root_input_region_count = root.region_count;
    plan.copy_region_count = plan.region_count * 4 + root.region_count;
    plan.root_input_plane_offset = static_cast<u32>(root_offset);
    plan.root_input_plane_bytes = root.buffer_bytes;
    plan.total_bytes = static_cast<u32>(total_bytes);
    plan.root_input_format = root.format;
    return plan;
}

struct PpTerminalScopeFinalizeOrder {
    bool drain_draw_callbacks_before_terminal{};
    bool finish_present_before_terminal{};
    bool drain_present_callbacks_before_terminal{};
    bool finalize_terminal{true};
    bool drain_draw_callbacks_after_terminal{};
};

[[nodiscard]] constexpr PpTerminalScopeFinalizeOrder PlanPpTerminalScopeFinalizeOrder(
    bool join_final_backing, bool final_surface_uses_draw_scheduler) noexcept {
    return {
        .drain_draw_callbacks_before_terminal =
            join_final_backing && final_surface_uses_draw_scheduler,
        .finish_present_before_terminal = join_final_backing && !final_surface_uses_draw_scheduler,
        .drain_present_callbacks_before_terminal =
            join_final_backing && !final_surface_uses_draw_scheduler,
        .finalize_terminal = true,
        .drain_draw_callbacks_after_terminal =
            !join_final_backing && final_surface_uses_draw_scheduler,
    };
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
    u32 lineage_hops{};
    FinalGuestSurfaceStatus lineage_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss lineage_loss{};
};

struct PpTerminalScopeLineageHandoffDecision {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool capture_consumer{};
    bool capture_output{};
    bool publish_flip_alias{};
};

[[nodiscard]] constexpr PpTerminalScopeLineageHandoffDecision PlanPpTerminalScopeLineageHandoff(
    bool exact_consumer, FinalGuestSurfaceStatus lineage_status, FinalGuestSurfaceLoss lineage_loss,
    u32 producer_plane_mask, bool single_output) noexcept {
    if (!exact_consumer) {
        return {};
    }
    if (lineage_status != FinalGuestSurfaceStatus::Complete || lineage_loss.Any()) {
        return {.status = lineage_status, .loss = lineage_loss};
    }
    if ((producer_plane_mask & 19u) != 19u || !single_output) {
        return {.status = FinalGuestSurfaceStatus::GapLoss, .loss = {.gap = 1}};
    }
    return {
        .status = FinalGuestSurfaceStatus::Complete,
        .capture_consumer = true,
        .capture_output = true,
        .publish_flip_alias = true,
    };
}

[[nodiscard]] constexpr bool ShouldArmPpTerminalScopeFallbackAfterFlip(
    bool consumed_lineage) noexcept {
    return !consumed_lineage;
}

[[nodiscard]] constexpr PpTerminalScopeContentReport MakePpTerminalScopeContentReport(
    u64 sequence, FinalGuestSurfaceStatus status, FinalGuestSurfaceLoss loss, u32 draw_count,
    u32 region_count, u32 consumer_observations = 0, u32 consumer_phase_mask = 0,
    u32 consumer_shape_matches = 0, bool consumer_frozen = false, u32 plane_mask = 0,
    u32 lineage_hops = 0,
    FinalGuestSurfaceStatus lineage_status = FinalGuestSurfaceStatus::AlreadyConsumed,
    FinalGuestSurfaceLoss lineage_loss = {}) noexcept {
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
        .lineage_hops = lineage_hops,
        .lineage_status = lineage_status,
        .lineage_loss = lineage_loss,
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
    u32 root_input_region_count{};
    u32 plane_bytes{};
    u32 root_input_plane_bytes{};
    u32 second_plane_offset{};
    u32 consumer_plane_offset{};
    u32 output_plane_offset{};
    u32 root_input_plane_offset{};
    u32 total_bytes{};
    u32 plane_mask{};
    std::array<PpTerminalScopeContentHistoryRegion, FinalGuestSurfaceWatchOrdinals::MaxOrdinals>
        regions{};
    std::array<PpTerminalScopeContentHistoryRegion, FinalGuestSurfaceWatchOrdinals::MaxOrdinals>
        root_input_regions{};
    u32 bytes_per_pixel{4};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceFormat root_input_format{FinalGuestSurfaceFormat::Unsupported};

    auto operator<=>(const PpTerminalScopeContentHistoryLayout&) const = default;
};

struct PpTerminalScopeFinalBackingLayout {
    u32 region_count{};
    u32 total_bytes{};
    std::array<PpTerminalScopeContentHistoryRegion, FinalGuestSurfaceWatchOrdinals::MaxOrdinals>
        regions{};
    u32 bytes_per_pixel{4};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};

    auto operator<=>(const PpTerminalScopeFinalBackingLayout&) const = default;
};

struct PpTerminalScopeFinalBackingObservationPlan {
    PpTerminalScopeFinalBackingLayout layout{};
    u32 source_offset{};
    u32 source_bytes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool observe{};
    bool gpu_copy{};
    bool cpu_wait{};
    bool finish{};
};

[[nodiscard]] inline PpTerminalScopeFinalBackingObservationPlan
PlanPpTerminalScopeFinalBackingObservation(bool enabled, FinalGuestSurfaceStage stage,
                                           const FinalGuestSurfaceTilePlan& plan,
                                           bool bytes_available) noexcept {
    if (!enabled) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status, u32 FinalGuestSurfaceLoss::* member) {
        PpTerminalScopeFinalBackingObservationPlan result{};
        result.status = status;
        result.loss.*member = 1;
        return result;
    };
    if (stage != FinalGuestSurfaceStage::PpSourcePublicationReconstruction) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::unsupported_type);
    }
    if (plan.status != FinalGuestSurfaceStatus::Complete || !bytes_available ||
        (plan.paired_backing_format != FinalGuestSurfaceFormat::Rgba8 &&
         plan.paired_backing_format != FinalGuestSurfaceFormat::Bgra8) ||
        plan.paired_backing_bytes == 0 || plan.paired_backing_region_count == 0 ||
        plan.paired_backing_region_count > FinalGuestSurfaceWatchOrdinals::MaxOrdinals) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    PpTerminalScopeFinalBackingObservationPlan result{
        .layout =
            {
                .region_count = plan.paired_backing_region_count,
                .total_bytes = plan.paired_backing_bytes,
                .format = plan.paired_backing_format,
            },
        .source_offset = plan.paired_backing_offset,
        .source_bytes = plan.paired_backing_bytes,
        .status = FinalGuestSurfaceStatus::Complete,
        .observe = true,
    };
    for (u32 index = 0; index < plan.paired_backing_region_count; ++index) {
        const auto& region = plan.paired_backing_regions[index];
        if (region.logical_ordinal == 0 || region.byte_size == 0 || region.byte_size % 4 != 0 ||
            static_cast<u64>(region.buffer_offset) + region.byte_size > plan.paired_backing_bytes) {
            return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                          &FinalGuestSurfaceLoss::invalidation);
        }
        for (u32 previous = 0; previous < index; ++previous) {
            if (plan.paired_backing_regions[previous].logical_ordinal == region.logical_ordinal) {
                return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                              &FinalGuestSurfaceLoss::invalidation);
            }
        }
        result.layout.regions[index] = {
            .logical_ordinal = region.logical_ordinal,
            .buffer_offset = region.buffer_offset,
            .byte_size = region.byte_size,
        };
    }
    return result;
}

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
    std::vector<u32> output_aba_ordinals{};
    std::vector<u32> output_stable_ordinals{};
    std::vector<u32> output_ambiguous_ordinals{};
    std::vector<u32> root_input_aba_ordinals{};
    std::vector<u32> root_input_stable_ordinals{};
    std::vector<u32> root_input_ambiguous_ordinals{};
    std::vector<u32> first_localized_visual_return_ordinals{};
    std::vector<u32> second_localized_visual_return_ordinals{};
    std::vector<u32> consumer_localized_visual_return_ordinals{};
    std::vector<u32> output_localized_visual_return_ordinals{};
    std::vector<u32> root_input_localized_visual_return_ordinals{};
    std::vector<u32> output_final_backing_equal_ordinals{};
    std::vector<u32> output_final_backing_different_ordinals{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
};

[[nodiscard]] inline std::optional<bool> IsPpTerminalScopeLocalizedVisualReturn(
    FinalGuestSurfaceFormat format, std::span<const std::byte> baseline,
    std::span<const std::byte> departure, std::span<const std::byte> returned) noexcept {
    if ((format != FinalGuestSurfaceFormat::Rgba8 && format != FinalGuestSurfaceFormat::Bgra8) ||
        baseline.empty() || baseline.size() != departure.size() ||
        baseline.size() != returned.size() || baseline.size() % 4 != 0) {
        return std::nullopt;
    }
    const auto changed_pixels = [](std::span<const std::byte> left,
                                   std::span<const std::byte> right) {
        u32 changed{};
        for (size_t offset = 0; offset < left.size(); offset += 4) {
            u32 difference{};
            for (u32 channel = 0; channel < 3; ++channel) {
                const u8 first = std::to_integer<u8>(left[offset + channel]);
                const u8 second = std::to_integer<u8>(right[offset + channel]);
                difference += first > second ? first - second : second - first;
            }
            changed += difference >= 48;
        }
        return changed;
    };
    const u64 pixels = baseline.size() / 4;
    const u64 baseline_to_departure = changed_pixels(baseline, departure);
    const u64 departure_to_returned = changed_pixels(departure, returned);
    const u64 baseline_to_returned = changed_pixels(baseline, returned);
    return baseline_to_departure * 4 >= pixels && departure_to_returned * 4 >= pixels &&
           baseline_to_returned * 100 <= pixels;
}

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
                                           u32 history_capacity_ = 32,
                                           bool join_final_backing_ = false)
        : window{window_}, history_capacity{std::clamp(history_capacity_, 1u, 32u)},
          join_final_backing{join_final_backing_} {}

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

    void ObserveFinalBacking(u64 sequence, const PpTerminalScopeFinalBackingLayout& layout,
                             std::span<const std::byte> bytes, FinalGuestSurfaceStatus status,
                             FinalGuestSurfaceLoss loss) {
        if (!join_final_backing || !window.Contains(sequence)) {
            return;
        }
        if (const auto existing = std::ranges::find(final_backing_history, sequence,
                                                    &FinalBackingObservation::sequence);
            existing != final_backing_history.end()) {
            existing->status = FinalGuestSurfaceStatus::InvalidationLoss;
            existing->loss = {.invalidation = 1};
            existing->bytes.clear();
            Reconcile();
            return;
        }
        if (final_backing_history.size() == history_capacity) {
            final_backing_history.pop_front();
        }
        final_backing_history.push_back({
            .sequence = sequence,
            .layout = layout,
            .bytes = std::vector<std::byte>{bytes.begin(), bytes.end()},
            .status = status,
            .loss = loss,
        });
        last_final_backing_sequence = std::max(last_final_backing_sequence, sequence);
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

    struct FinalBackingObservation {
        u64 sequence{};
        PpTerminalScopeFinalBackingLayout layout{};
        std::vector<std::byte> bytes{};
        FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss loss{};
    };

    [[nodiscard]] const Observation* Find(u64 sequence) const noexcept {
        const auto found = std::ranges::find(history, sequence, &Observation::sequence);
        return found == history.end() ? nullptr : &*found;
    }

    [[nodiscard]] const FinalBackingObservation* FindFinalBacking(u64 sequence) const noexcept {
        const auto found =
            std::ranges::find(final_backing_history, sequence, &FinalBackingObservation::sequence);
        return found == final_backing_history.end() ? nullptr : &*found;
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

    [[nodiscard]] static u32 PlaneRegionCount(const PpTerminalScopeContentHistoryLayout& layout,
                                              u32 plane) noexcept {
        return plane == 4 ? layout.root_input_region_count : layout.region_count;
    }

    [[nodiscard]] static const PpTerminalScopeContentHistoryRegion* PlaneRegion(
        const PpTerminalScopeContentHistoryLayout& layout, u32 plane, u32 index) noexcept {
        if (index >= PlaneRegionCount(layout, plane)) {
            return nullptr;
        }
        return plane == 4 ? &layout.root_input_regions[index] : &layout.regions[index];
    }

    [[nodiscard]] static FinalGuestSurfaceFormat PlaneFormat(
        const PpTerminalScopeContentHistoryLayout& layout, u32 plane) noexcept {
        return plane == 4 ? layout.root_input_format : layout.format;
    }

    [[nodiscard]] static bool EqualVisibleRegion(const Observation& left, const Observation& right,
                                                 u32 plane, u32 region_index) noexcept {
        const auto* region = PlaneRegion(left.layout, plane, region_index);
        if (left.layout != right.layout || !region || plane > 4 ||
            left.layout.bytes_per_pixel != 4) {
            return false;
        }
        const auto plane_offset = [plane](const PpTerminalScopeContentHistoryLayout& layout) {
            return plane == 0   ? 0u
                   : plane == 1 ? layout.second_plane_offset
                   : plane == 2 ? layout.consumer_plane_offset
                   : plane == 3 ? layout.output_plane_offset
                                : layout.root_input_plane_offset;
        };
        const u64 left_offset = region->buffer_offset + plane_offset(left.layout);
        const u64 right_offset = region->buffer_offset + plane_offset(right.layout);
        if (region->byte_size == 0 || region->byte_size % left.layout.bytes_per_pixel != 0 ||
            left_offset + region->byte_size > left.bytes.size() ||
            right_offset + region->byte_size > right.bytes.size()) {
            return false;
        }
        for (u32 offset = 0; offset < region->byte_size; ++offset) {
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
        for (u32 index = 0; index < PlaneRegionCount(a.layout, plane); ++index) {
            const bool ab = EqualVisibleRegion(a, b, plane, index);
            const bool bc = EqualVisibleRegion(b, c, plane, index);
            const bool ac = EqualVisibleRegion(a, c, plane, index);
            const u32 ordinal = PlaneRegion(a.layout, plane, index)->logical_ordinal;
            if (ac && !ab) {
                aba.push_back(ordinal);
            } else if (ab && bc) {
                stable.push_back(ordinal);
            } else {
                ambiguous.push_back(ordinal);
            }
        }
    }

    [[nodiscard]] static bool ClassifyLocalizedPlane(const Observation& a, const Observation& b,
                                                     const Observation& c, u32 plane,
                                                     std::vector<u32>& localized_visual_return) {
        if (a.layout != b.layout || a.layout != c.layout || plane > 4) {
            return false;
        }
        const auto plane_offset = [plane](const PpTerminalScopeContentHistoryLayout& layout) {
            return plane == 0   ? 0u
                   : plane == 1 ? layout.second_plane_offset
                   : plane == 2 ? layout.consumer_plane_offset
                   : plane == 3 ? layout.output_plane_offset
                                : layout.root_input_plane_offset;
        };
        const u64 a_plane_offset = plane_offset(a.layout);
        const u64 b_plane_offset = plane_offset(b.layout);
        const u64 c_plane_offset = plane_offset(c.layout);
        if (a_plane_offset > a.bytes.size() || b_plane_offset > b.bytes.size() ||
            c_plane_offset > c.bytes.size()) {
            return false;
        }
        for (u32 index = 0; index < PlaneRegionCount(a.layout, plane); ++index) {
            const auto& region = *PlaneRegion(a.layout, plane, index);
            if (region.logical_ordinal == 0 || region.byte_size == 0 ||
                a_plane_offset + region.buffer_offset + region.byte_size > a.bytes.size() ||
                b_plane_offset + region.buffer_offset + region.byte_size > b.bytes.size() ||
                c_plane_offset + region.buffer_offset + region.byte_size > c.bytes.size()) {
                return false;
            }
            const auto result = IsPpTerminalScopeLocalizedVisualReturn(
                PlaneFormat(a.layout, plane),
                std::span{a.bytes}.subspan(a_plane_offset + region.buffer_offset, region.byte_size),
                std::span{b.bytes}.subspan(b_plane_offset + region.buffer_offset, region.byte_size),
                std::span{c.bytes}.subspan(c_plane_offset + region.buffer_offset,
                                           region.byte_size));
            if (!result) {
                return false;
            }
            if (*result) {
                localized_visual_return.push_back(region.logical_ordinal);
            }
        }
        return true;
    }

    [[nodiscard]] static std::optional<bool> EqualOutputToFinalBacking(
        const Observation& output, const FinalBackingObservation& backing,
        u32 region_index) noexcept {
        if (region_index >= output.layout.region_count || output.layout.bytes_per_pixel != 4 ||
            backing.layout.bytes_per_pixel != 4 ||
            output.layout.format == FinalGuestSurfaceFormat::Unsupported ||
            output.layout.format != backing.layout.format ||
            backing.bytes.size() != backing.layout.total_bytes) {
            return std::nullopt;
        }
        const auto& output_region = output.layout.regions[region_index];
        const auto backing_region = std::ranges::find(
            backing.layout.regions.begin(),
            backing.layout.regions.begin() + backing.layout.region_count,
            output_region.logical_ordinal, &PpTerminalScopeContentHistoryRegion::logical_ordinal);
        if (backing_region == backing.layout.regions.begin() + backing.layout.region_count ||
            output_region.byte_size == 0 || output_region.byte_size != backing_region->byte_size ||
            output_region.byte_size % output.layout.bytes_per_pixel != 0) {
            return std::nullopt;
        }
        const u64 output_offset =
            static_cast<u64>(output.layout.output_plane_offset) + output_region.buffer_offset;
        const u64 backing_offset = backing_region->buffer_offset;
        if (output_offset + output_region.byte_size > output.bytes.size() ||
            backing_offset + backing_region->byte_size > backing.bytes.size()) {
            return std::nullopt;
        }
        for (u32 offset = 0; offset < output_region.byte_size; ++offset) {
            if (offset % output.layout.bytes_per_pixel == 3) {
                continue;
            }
            if (output.bytes[output_offset + offset] != backing.bytes[backing_offset + offset]) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static bool ClassifyFinalBackingJoin(const Observation& a, const Observation& b,
                                                       const Observation& c,
                                                       const FinalBackingObservation& backing_a,
                                                       const FinalBackingObservation& backing_b,
                                                       const FinalBackingObservation& backing_c,
                                                       std::vector<u32>& equal,
                                                       std::vector<u32>& different) {
        for (u32 index = 0; index < a.layout.region_count; ++index) {
            const auto a_equal = EqualOutputToFinalBacking(a, backing_a, index);
            const auto b_equal = EqualOutputToFinalBacking(b, backing_b, index);
            const auto c_equal = EqualOutputToFinalBacking(c, backing_c, index);
            if (!a_equal || !b_equal || !c_equal) {
                return false;
            }
            const u32 ordinal = a.layout.regions[index].logical_ordinal;
            if (*a_equal && *b_equal && *c_equal) {
                equal.push_back(ordinal);
            } else {
                different.push_back(ordinal);
            }
        }
        return true;
    }

    [[nodiscard]] PpTerminalScopeCalibratedReport Evaluate(
        u32 request, const Observation* a, const Observation* b, const Observation* c,
        const FinalBackingObservation* backing_a, const FinalBackingObservation* backing_b,
        const FinalBackingObservation* backing_c) const {
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
        if (join_final_backing && (!backing_a || !backing_b || !backing_c)) {
            report.status = FinalGuestSurfaceStatus::GapLoss;
            report.loss.history = 1;
            return report;
        }
        AddLoss(report.loss, a->loss);
        AddLoss(report.loss, b->loss);
        AddLoss(report.loss, c->loss);
        if (join_final_backing) {
            AddLoss(report.loss, backing_a->loss);
            AddLoss(report.loss, backing_b->loss);
            AddLoss(report.loss, backing_c->loss);
        }
        if (a->status != FinalGuestSurfaceStatus::Complete ||
            b->status != FinalGuestSurfaceStatus::Complete ||
            c->status != FinalGuestSurfaceStatus::Complete ||
            (join_final_backing && (backing_a->status != FinalGuestSurfaceStatus::Complete ||
                                    backing_b->status != FinalGuestSurfaceStatus::Complete ||
                                    backing_c->status != FinalGuestSurfaceStatus::Complete)) ||
            report.loss.Any()) {
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
            if (!ClassifyLocalizedPlane(*a, *b, *c, 0,
                                        report.first_localized_visual_return_ordinals)) {
                report.status = FinalGuestSurfaceStatus::InvalidationLoss;
                report.loss.invalidation = 1;
                return report;
            }
        }
        if ((a->layout.plane_mask & (1u << 1)) != 0) {
            ClassifyPlane(*a, *b, *c, 1, report.second_aba_ordinals, report.second_stable_ordinals,
                          report.second_ambiguous_ordinals);
            if (!ClassifyLocalizedPlane(*a, *b, *c, 1,
                                        report.second_localized_visual_return_ordinals)) {
                report.status = FinalGuestSurfaceStatus::InvalidationLoss;
                report.loss.invalidation = 1;
                return report;
            }
        }
        if ((a->layout.plane_mask & (1u << 2)) != 0) {
            ClassifyPlane(*a, *b, *c, 2, report.consumer_aba_ordinals,
                          report.consumer_stable_ordinals, report.consumer_ambiguous_ordinals);
            if (!ClassifyLocalizedPlane(*a, *b, *c, 2,
                                        report.consumer_localized_visual_return_ordinals)) {
                report.status = FinalGuestSurfaceStatus::InvalidationLoss;
                report.loss.invalidation = 1;
                return report;
            }
        }
        if ((a->layout.plane_mask & (1u << 3)) != 0) {
            ClassifyPlane(*a, *b, *c, 3, report.output_aba_ordinals, report.output_stable_ordinals,
                          report.output_ambiguous_ordinals);
            if (!ClassifyLocalizedPlane(*a, *b, *c, 3,
                                        report.output_localized_visual_return_ordinals)) {
                report.status = FinalGuestSurfaceStatus::InvalidationLoss;
                report.loss.invalidation = 1;
                return report;
            }
        }
        if ((a->layout.plane_mask & (1u << 4)) != 0) {
            ClassifyPlane(*a, *b, *c, 4, report.root_input_aba_ordinals,
                          report.root_input_stable_ordinals, report.root_input_ambiguous_ordinals);
            if (!ClassifyLocalizedPlane(*a, *b, *c, 4,
                                        report.root_input_localized_visual_return_ordinals)) {
                report.status = FinalGuestSurfaceStatus::InvalidationLoss;
                report.loss.invalidation = 1;
                return report;
            }
        }
        if (join_final_backing &&
            ((a->layout.plane_mask & (1u << 3)) == 0 ||
             !ClassifyFinalBackingJoin(*a, *b, *c, *backing_a, *backing_b, *backing_c,
                                       report.output_final_backing_equal_ordinals,
                                       report.output_final_backing_different_ordinals))) {
            report.status = FinalGuestSurfaceStatus::InvalidationLoss;
            report.loss.invalidation = 1;
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
            const auto* backing_a =
                join_final_backing ? FindFinalBacking(a_stamp.sequence) : nullptr;
            const auto* backing_b =
                join_final_backing ? FindFinalBacking(b_stamp.sequence) : nullptr;
            const auto* backing_c =
                join_final_backing ? FindFinalBacking(c_stamp.sequence) : nullptr;
            if (join_final_backing && (!backing_a || !backing_b || !backing_c) &&
                last_final_backing_sequence < c_stamp.sequence) {
                continue;
            }
            auto report = Evaluate(request, a, b, c, backing_a, backing_b, backing_c);
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
    std::deque<FinalBackingObservation> final_backing_history{};
    std::array<std::optional<FinalGuestSurfaceCalibratedStamp>, MaxCalibrations + 1> calibrations{};
    std::array<bool, MaxCalibrations + 1> classified{};
    std::array<bool, MaxCalibrations + 1> eligible_request{};
    std::vector<PpTerminalScopeCalibratedReport> reports{};
    u64 last_content_sequence{};
    u64 last_final_backing_sequence{};
    u32 calibration_count{};
    u32 outside_count{};
    u32 eligible_count{};
    u32 emitted_count{};
    u32 complete_count{};
    u32 coverage_loss{};
    bool join_final_backing{};
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
           " a3=" + FormatPpTerminalScopeOrdinalList(report.output_aba_ordinals) +
           " s3=" + FormatPpTerminalScopeOrdinalList(report.output_stable_ordinals) +
           " x3=" + FormatPpTerminalScopeOrdinalList(report.output_ambiguous_ordinals) + " y0=" +
           FormatPpTerminalScopeOrdinalList(report.first_localized_visual_return_ordinals) +
           " y1=" +
           FormatPpTerminalScopeOrdinalList(report.second_localized_visual_return_ordinals) +
           " y2=" +
           FormatPpTerminalScopeOrdinalList(report.consumer_localized_visual_return_ordinals) +
           " y3=" +
           FormatPpTerminalScopeOrdinalList(report.output_localized_visual_return_ordinals) +
           " a4=" + FormatPpTerminalScopeOrdinalList(report.root_input_aba_ordinals) +
           " s4=" + FormatPpTerminalScopeOrdinalList(report.root_input_stable_ordinals) +
           " x4=" + FormatPpTerminalScopeOrdinalList(report.root_input_ambiguous_ordinals) +
           " y4=" +
           FormatPpTerminalScopeOrdinalList(report.root_input_localized_visual_return_ordinals) +
           " e3=" + FormatPpTerminalScopeOrdinalList(report.output_final_backing_equal_ordinals) +
           " d3=" +
           FormatPpTerminalScopeOrdinalList(report.output_final_backing_different_ordinals) +
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
           " lh=" + std::to_string(report.lineage_hops) +
           " ls=" + std::to_string(static_cast<u32>(report.lineage_status)) +
           " ll=" + std::to_string(report.lineage_loss.Any() ? 1 : 0) +
           " lm=" + std::to_string(report.loss.Any() ? 1 : 0);
}

[[nodiscard]] inline std::string FormatPpTerminalScopeDiscoveryCoverage(
    const PpTerminalScopeDiscoveryCoverage& coverage) {
    return "FGSCTSD c=" + std::to_string(coverage.candidates) +
           " t=" + std::to_string(coverage.tracked) + " a=" + std::to_string(coverage.allocated) +
           " r=" + std::to_string(coverage.restarted) +
           " m=" + std::to_string(coverage.mapping_rejected) +
           " v=" + std::to_string(coverage.target_rejected) +
           " z=" + std::to_string(coverage.capacity_rejected) +
           " ri=" + std::to_string(coverage.root_accepted) + "/" +
           std::to_string(coverage.root_missing) + "/" + std::to_string(coverage.root_hazard) +
           "/" + std::to_string(coverage.root_structure) + "/" +
           std::to_string(coverage.root_plan) + "/" + std::to_string(coverage.root_plan_loss_mask) +
           " rv=" + std::to_string(coverage.root_view_valid) + "/" +
           std::to_string(coverage.root_image_invalid) + "/" +
           std::to_string(coverage.root_view_missing) + "/" +
           std::to_string(coverage.root_view_ambiguous) + "/" +
           std::to_string(coverage.root_view_mip) + "/" + std::to_string(coverage.root_view_layer) +
           "/" + std::to_string(coverage.root_view_invalid);
}

[[nodiscard]] inline std::string FormatPpTerminalScopePrivateLineageReport(
    const PpTerminalScopePrivateLineageReport& report) {
    return "lh=" + std::to_string(report.hops) +
           " ls=" + std::to_string(static_cast<u32>(report.status)) +
           " ll=" + std::to_string(report.loss.Any() ? 1 : 0);
}

} // namespace Vulkan
