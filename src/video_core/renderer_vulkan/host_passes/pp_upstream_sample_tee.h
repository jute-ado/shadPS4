// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstring>
#include <deque>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/pp_upstream_sample_tee.h"
#include "video_core/renderer_vulkan/final_guest_surface_content.h"
#include "video_core/renderer_vulkan/host_passes/pp_upstream_feedback_pre_post.h"
#include "video_core/renderer_vulkan/host_passes/pp_upstream_input_content.h"
#include "video_core/renderer_vulkan/host_passes/pp_upstream_sample_route.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_color_scope_producer.h"

namespace Vulkan::HostPasses {

struct PpUpstreamSampleTeePipelinePolicy {
    bool instrument_profile{};
    u64 fragment_pgm_hash{};
    bool capture_fragment_export{};
    bool allow_persistent_artifacts{true};
    bool allow_shader_patches{true};
};

[[nodiscard]] constexpr PpUpstreamSampleTeePipelinePolicy PlanPpUpstreamSampleTeePipelinePolicy(
    bool strict_tee, u64 fragment_pgm_hash, bool capture_fragment_export = false) noexcept {
    return {
        .instrument_profile = strict_tee,
        .fragment_pgm_hash = strict_tee ? fragment_pgm_hash : 0,
        .capture_fragment_export = strict_tee && capture_fragment_export,
        .allow_persistent_artifacts = !strict_tee,
        .allow_shader_patches = !strict_tee,
    };
}

struct PpUpstreamSampleTeeBindingDescriptor {
    bool enabled{};
    Shader::LogicalStage stage{};
    Shader::BufferType buffer_type{};
    bool is_written{};
    u64 descriptor_bytes{};
    bool buffer_present{};
    u64 buffer_bytes{};
    u64 mapped_bytes{};
};

struct PpUpstreamSampleTeeBindingPlan {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool bind{};
};

[[nodiscard]] inline PpUpstreamSampleTeeBindingPlan PlanPpUpstreamSampleTeeBinding(
    const PpUpstreamSampleTeeBindingDescriptor& descriptor) noexcept {
    if (!descriptor.enabled) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status, u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeBindingPlan result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    if (descriptor.stage != Shader::LogicalStage::Fragment ||
        descriptor.buffer_type != Shader::BufferType::PpUpstreamSampleTee ||
        !descriptor.is_written) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::unsupported_type);
    }
    if (!descriptor.buffer_present ||
        descriptor.descriptor_bytes != Shader::PpUpstreamSampleTeeBufferBytes ||
        descriptor.buffer_bytes != Shader::PpUpstreamSampleTeeBufferBytes ||
        descriptor.mapped_bytes != Shader::PpUpstreamSampleTeeBufferBytes) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    return {
        .status = FinalGuestSurfaceStatus::Complete,
        .bind = true,
    };
}

struct PpUpstreamSampleTeeShaderPlan {
    static constexpr u32 FragmentLogicalStage = 0;

    u32 resource_index{};
    u32 eligible_result_count{};
    u32 additional_image_operations{};
    u32 tee_storage_operations{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool arm{};
    bool retains_private_route{};

    bool operator==(const PpUpstreamSampleTeeShaderPlan&) const = default;
};

struct PpUpstreamSampleTeeShaderDescriptor {
    bool enabled{};
    PpUpstreamSampleRouteSet routes{};
    std::span<const u32> eligible_results_per_resource{};
};

[[nodiscard]] inline PpUpstreamSampleTeeShaderPlan PlanPpUpstreamSampleTeeShader(
    const PpUpstreamSampleTeeShaderDescriptor& descriptor) noexcept {
    if (!descriptor.enabled) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status, u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeShaderPlan result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    if (descriptor.routes.status != FinalGuestSurfaceStatus::Complete ||
        descriptor.routes.loss.Any() || descriptor.routes.route_count != 1) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    const auto& route = descriptor.routes.routes[0];
    if (!route.valid ||
        route.logical_stage != PpUpstreamSampleTeeShaderPlan::FragmentLogicalStage ||
        route.array_element != 0) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::unsupported_type);
    }
    if (route.image_resource_index >= descriptor.eligible_results_per_resource.size()) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    const u32 eligible_count = descriptor.eligible_results_per_resource[route.image_resource_index];
    if (eligible_count == 0) {
        return reject(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
    }
    if (eligible_count != 1) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    return {
        .resource_index = route.image_resource_index,
        .eligible_result_count = eligible_count,
        .additional_image_operations = 0,
        .tee_storage_operations = 1,
        .status = FinalGuestSurfaceStatus::Complete,
        .arm = true,
    };
}

struct PpUpstreamSampleTeeEmissionPlan {
    u32 returned_value_id{};
    u32 image_operations_after{};
    u32 storage_operations{};
};

struct PpUpstreamSampleTeeEmissionDescriptor {
    bool armed{};
    bool resource_matches{};
    u32 existing_value_id{};
    u32 image_operations_before{};
};

[[nodiscard]] constexpr PpUpstreamSampleTeeEmissionPlan PlanPpUpstreamSampleTeeEmission(
    const PpUpstreamSampleTeeEmissionDescriptor& descriptor) noexcept {
    return {
        .returned_value_id = descriptor.existing_value_id,
        .image_operations_after = descriptor.image_operations_before,
        .storage_operations = descriptor.armed && descriptor.resource_matches ? 1u : 0u,
    };
}

struct PpUpstreamSampleTeeWindowPlan {
    u32 logical_ordinal{};
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};
    u32 pixel_count{};
    u32 total_bytes{};
    u32 image_copy_operations{};
    u32 texture_operations{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool capture{};
    bool logs_coordinates{};
    bool logs_values{};

    bool operator==(const PpUpstreamSampleTeeWindowPlan&) const = default;
};

struct PpUpstreamSampleTeeWindowDescriptor {
    bool enabled{};
    u32 target_width{};
    u32 target_height{};
    u32 logical_width{};
    u32 logical_height{};
    u32 logical_ordinal{};
    u32 max_width{64};
    u32 max_height{64};
    u32 max_bytes{128u << 10};
};

[[nodiscard]] inline PpUpstreamSampleTeeWindowPlan PlanPpUpstreamSampleTeeWindow(
    const PpUpstreamSampleTeeWindowDescriptor& descriptor) noexcept {
    if (!descriptor.enabled) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status, u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeWindowPlan result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    if (descriptor.logical_ordinal == 0) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    const auto surface = PlanFinalGuestSurfaceTiles(
        {
            .width = descriptor.target_width,
            .height = descriptor.target_height,
            .depth = 1,
            .mip_levels = 1,
            .array_layers = 1,
            .samples = 1,
            .type = FinalGuestSurfaceImageType::Color2D,
            .aspect = FinalGuestSurfaceAspect::Color,
            .format = FinalGuestSurfaceFormat::Rgba8,
            .logical_width = descriptor.logical_width,
            .logical_height = descriptor.logical_height,
            .logical_full_fit = true,
            .logical_top_left = true,
            .logical_no_y_flip = true,
        },
        {.max_tiles = FinalGuestSurfaceTilePlan::MaxTiles,
         .max_bytes = std::numeric_limits<u32>::max()});
    if (surface.status != FinalGuestSurfaceStatus::Complete || surface.loss.Any()) {
        PpUpstreamSampleTeeWindowPlan result{.status = surface.status, .loss = surface.loss};
        return result;
    }
    if (descriptor.logical_ordinal > surface.tile_count) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    const auto tile = surface.TileAt(descriptor.logical_ordinal - 1);
    if (tile.width == 0 || tile.height == 0) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (tile.width > descriptor.max_width || tile.height > descriptor.max_height) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss, &FinalGuestSurfaceLoss::tile_capacity);
    }
    constexpr u64 BytesPerPixelRecord =
        Shader::PpUpstreamSampleTeeRecordDwords * sizeof(u32);
    const u64 pixel_count = static_cast<u64>(tile.width) * tile.height;
    const u64 total_bytes = pixel_count * BytesPerPixelRecord;
    if (pixel_count > std::numeric_limits<u32>::max() || total_bytes > descriptor.max_bytes ||
        total_bytes > std::numeric_limits<u32>::max()) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss, &FinalGuestSurfaceLoss::byte_capacity);
    }
    return {
        .logical_ordinal = descriptor.logical_ordinal,
        .x = tile.x,
        .y = tile.y,
        .width = tile.width,
        .height = tile.height,
        .pixel_count = static_cast<u32>(pixel_count),
        .total_bytes = static_cast<u32>(total_bytes),
        .status = FinalGuestSurfaceStatus::Complete,
        .capture = true,
    };
}

struct PpUpstreamSampleTeePixelRecord {
    std::array<u32, 4> value{};
    u32 cube_metadata{};
    u32 dependency_metadata{};
    u32 writers{};

    bool operator==(const PpUpstreamSampleTeePixelRecord&) const = default;
};

struct PpUpstreamSampleTeeCandidateRecord {
    u32 candidate_index{};
    PpUpstreamSampleTeeShaderPlan shader{};
    PpUpstreamSampleTeeWindowPlan window{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool ready{};

    bool operator==(const PpUpstreamSampleTeeCandidateRecord&) const = default;
};

[[nodiscard]] inline bool IsCanonicalPpUpstreamSampleTeeCandidateRecord(
    const PpUpstreamSampleTeeCandidateRecord& record, u32 selected_candidate_index) noexcept {
    const u64 expected_pixels = static_cast<u64>(record.window.width) * record.window.height;
    const u64 expected_bytes = expected_pixels * sizeof(PpUpstreamSampleTeePixelRecord);
    return record.ready && record.status == FinalGuestSurfaceStatus::Complete &&
           !record.loss.Any() && record.candidate_index == selected_candidate_index &&
           record.candidate_index < 4 &&
           record.shader.status == FinalGuestSurfaceStatus::Complete && !record.shader.loss.Any() &&
           record.shader.arm && record.shader.resource_index < Shader::NUM_IMAGES &&
           record.shader.eligible_result_count == 1 &&
           record.shader.additional_image_operations == 0 &&
           record.shader.tee_storage_operations == 1 && !record.shader.retains_private_route &&
           record.window.status == FinalGuestSurfaceStatus::Complete && !record.window.loss.Any() &&
           record.window.capture && record.window.logical_ordinal != 0 &&
           record.window.width != 0 && record.window.height != 0 &&
           record.window.width <= Shader::PpUpstreamSampleTeeMaxDimension &&
           record.window.height <= Shader::PpUpstreamSampleTeeMaxDimension &&
           expected_pixels == record.window.pixel_count &&
           expected_bytes == record.window.total_bytes &&
           expected_bytes <= Shader::PpUpstreamSampleTeeSlotStrideBytes -
                                 Shader::PpUpstreamSampleTeeHeaderDwords * sizeof(u32) &&
           record.window.image_copy_operations == 0 && record.window.texture_operations == 0 &&
           !record.window.logs_coordinates && !record.window.logs_values;
}

struct PpUpstreamSampleTeeCandidateDescriptor {
    bool enabled{};
    u32 configured_candidate_index{};
    u32 observed_candidate_index{};
    PpUpstreamSampleTeeShaderPlan shader{};
    PpUpstreamSampleTeeWindowPlan window{};
};

[[nodiscard]] inline PpUpstreamSampleTeeCandidateRecord RecordPpUpstreamSampleTeeCandidate(
    PpUpstreamSampleTeeCandidateRecord current,
    const PpUpstreamSampleTeeCandidateDescriptor& descriptor) noexcept {
    if (!descriptor.enabled) {
        return {};
    }
    if (descriptor.observed_candidate_index != descriptor.configured_candidate_index) {
        return current;
    }
    const bool valid = descriptor.shader.status == FinalGuestSurfaceStatus::Complete &&
                       !descriptor.shader.loss.Any() && descriptor.shader.arm &&
                       descriptor.window.status == FinalGuestSurfaceStatus::Complete &&
                       !descriptor.window.loss.Any() && descriptor.window.capture;
    if (!valid) {
        PpUpstreamSampleTeeCandidateRecord rejected{
            .candidate_index = descriptor.observed_candidate_index,
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
        };
        rejected.loss.invalidation = 1;
        return rejected;
    }
    if (current.ready) {
        PpUpstreamSampleTeeCandidateRecord rejected{
            .candidate_index = descriptor.observed_candidate_index,
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
        };
        rejected.loss.invalidation = 1;
        return rejected;
    }
    return {
        .candidate_index = descriptor.observed_candidate_index,
        .shader = descriptor.shader,
        .window = descriptor.window,
        .status = FinalGuestSurfaceStatus::Complete,
        .ready = true,
    };
}

struct PpUpstreamSampleTeeCandidatePublication {
    PpUpstreamSampleTeeShaderPlan shader{};
    PpUpstreamSampleTeeWindowPlan window{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool publish{};
};

struct PpUpstreamSampleTeeCandidateReconciliation {
    bool enabled{};
    PpUpstreamSampleTeeCandidateRecord record{};
    u32 selected_candidate_index{};
};

[[nodiscard]] inline PpUpstreamSampleTeeCandidatePublication ReconcilePpUpstreamSampleTeeCandidate(
    const PpUpstreamSampleTeeCandidateReconciliation& reconciliation) noexcept {
    if (!reconciliation.enabled) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status, u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeCandidatePublication result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    const auto& record = reconciliation.record;
    if (!record.ready) {
        if (record.status != FinalGuestSurfaceStatus::AlreadyConsumed && record.loss.Any()) {
            return {
                .status = record.status,
                .loss = record.loss,
            };
        }
        if (record.status != FinalGuestSurfaceStatus::AlreadyConsumed || record.loss.Any()) {
            return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                          &FinalGuestSurfaceLoss::invalidation);
        }
        return reject(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
    }
    if (!IsCanonicalPpUpstreamSampleTeeCandidateRecord(record,
                                                       reconciliation.selected_candidate_index)) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    return {
        .shader = record.shader,
        .window = record.window,
        .status = FinalGuestSurfaceStatus::Complete,
        .publish = true,
    };
}

struct PpUpstreamSampleTeeReadbackHeader {
    u32 magic{};
    u32 generation{};
    u32 candidate_index{};
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};
    u32 pixel_count{};

    bool operator==(const PpUpstreamSampleTeeReadbackHeader&) const = default;
};
static_assert(sizeof(PpUpstreamSampleTeePixelRecord) ==
              Shader::PpUpstreamSampleTeeRecordDwords * sizeof(u32));
static_assert(sizeof(PpUpstreamSampleTeeReadbackHeader) ==
              Shader::PpUpstreamSampleTeeHeaderDwords * sizeof(u32));
static_assert(std::is_standard_layout_v<PpUpstreamSampleTeePixelRecord>);
static_assert(std::is_trivially_copyable_v<PpUpstreamSampleTeePixelRecord>);
static_assert(offsetof(PpUpstreamSampleTeePixelRecord, cube_metadata) == 4 * sizeof(u32));
static_assert(offsetof(PpUpstreamSampleTeePixelRecord, dependency_metadata) == 5 * sizeof(u32));
static_assert(offsetof(PpUpstreamSampleTeePixelRecord, writers) == 6 * sizeof(u32));
static_assert(std::is_standard_layout_v<PpUpstreamSampleTeeReadbackHeader>);
static_assert(std::is_trivially_copyable_v<PpUpstreamSampleTeeReadbackHeader>);
static_assert(offsetof(PpUpstreamSampleTeeReadbackHeader, pixel_count) == 7 * sizeof(u32));
static_assert(sizeof(PpUpstreamSampleTeeReadbackHeader) <
              Shader::PpUpstreamSampleTeeSlotStrideBytes);
static_assert(Shader::PpUpstreamSampleTeeSlotCount == FinalGuestSurfaceReadbackSlotPool::MaxSlots);

struct PpUpstreamSampleTeeCaptureDescriptor {
    bool enabled{};
    bool slot_present{};
    u32 slot{};
    u32 generation{};
    u32 candidate_index{};
    PpUpstreamSampleTeeShaderPlan shader{};
    PpUpstreamSampleTeeWindowPlan window{};
};

struct PpUpstreamSampleTeeCapturePlan {
    PpUpstreamSampleTeeReadbackHeader header{};
    u32 slot_offset{};
    u32 slot_bytes{};
    u32 clear_offset{};
    u32 clear_bytes{};
    u32 base_dword{};
    u32 control{};
    u32 push_offset{};
    u32 push_bytes{};
    u32 image_operations{};
    u32 texture_operations{};
    u32 scheduler_waits{};
    u32 scheduler_finishes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool initialize{};
    bool arm{};
    bool require_post_barrier{};
    bool logs_private_state{};

    bool operator==(const PpUpstreamSampleTeeCapturePlan&) const = default;
};

[[nodiscard]] inline PpUpstreamSampleTeeCapturePlan PlanPpUpstreamSampleTeeCapture(
    const PpUpstreamSampleTeeCaptureDescriptor& descriptor) noexcept {
    if (!descriptor.enabled) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status, u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeCapturePlan result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    if (!descriptor.slot_present) {
        return reject(FinalGuestSurfaceStatus::BusyLoss, &FinalGuestSurfaceLoss::busy);
    }
    if (descriptor.slot >= Shader::PpUpstreamSampleTeeSlotCount || descriptor.generation == 0 ||
        descriptor.candidate_index >= 4) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (descriptor.shader.status != FinalGuestSurfaceStatus::Complete ||
        descriptor.shader.loss.Any() || !descriptor.shader.arm ||
        descriptor.shader.resource_index >= Shader::NUM_IMAGES ||
        descriptor.shader.eligible_result_count != 1 ||
        descriptor.shader.additional_image_operations != 0 ||
        descriptor.shader.tee_storage_operations != 1 || descriptor.shader.retains_private_route) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    const auto& window = descriptor.window;
    const u64 expected_pixels = static_cast<u64>(window.width) * window.height;
    const u64 expected_bytes = expected_pixels * sizeof(PpUpstreamSampleTeePixelRecord);
    constexpr u32 HeaderBytes = sizeof(PpUpstreamSampleTeeReadbackHeader);
    if (window.status != FinalGuestSurfaceStatus::Complete || window.loss.Any() ||
        !window.capture || window.logical_ordinal == 0 || window.width == 0 || window.height == 0 ||
        window.width > Shader::PpUpstreamSampleTeeMaxDimension ||
        window.height > Shader::PpUpstreamSampleTeeMaxDimension ||
        expected_pixels != window.pixel_count || expected_bytes != window.total_bytes ||
        expected_bytes > Shader::PpUpstreamSampleTeeSlotStrideBytes - HeaderBytes ||
        window.image_copy_operations != 0 || window.texture_operations != 0 ||
        window.logs_coordinates || window.logs_values) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }

    const u32 slot_offset = descriptor.slot * Shader::PpUpstreamSampleTeeSlotStrideBytes;
    const u32 base_dword = slot_offset / sizeof(u32);
    if (!Shader::IsValidPpUpstreamSampleTeeBaseDword(base_dword)) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    return {
        .header = {.magic = Shader::PpUpstreamSampleTeeHeaderMagic,
                   .generation = descriptor.generation,
                   .candidate_index = descriptor.candidate_index,
                   .x = window.x,
                   .y = window.y,
                   .width = window.width,
                   .height = window.height,
                   .pixel_count = window.pixel_count},
        .slot_offset = slot_offset,
        .slot_bytes = Shader::PpUpstreamSampleTeeSlotStrideBytes,
        .clear_offset = slot_offset + HeaderBytes,
        .clear_bytes = Shader::PpUpstreamSampleTeeSlotStrideBytes - HeaderBytes,
        .base_dword = base_dword,
        .control = Shader::PpUpstreamSampleTeeArmMask | descriptor.shader.resource_index |
                   (descriptor.candidate_index << Shader::PpUpstreamSampleTeeCandidateShift),
        .push_offset = static_cast<u32>(offsetof(Shader::PushData, pp_upstream_sample_tee_control)),
        .push_bytes = sizeof(u32) * 2,
        .status = FinalGuestSurfaceStatus::Complete,
        .initialize = true,
        .arm = true,
        .require_post_barrier = true,
    };
}

enum class PpUpstreamSampleTeeStage : u32 {
    None = 0,
    AllCommands = 1u << 0,
    Host = 1u << 1,
    Transfer = 1u << 2,
    FragmentShader = 1u << 3,
    AllGraphics = 1u << 4,
};

[[nodiscard]] constexpr PpUpstreamSampleTeeStage operator|(PpUpstreamSampleTeeStage lhs,
                                                           PpUpstreamSampleTeeStage rhs) noexcept {
    return static_cast<PpUpstreamSampleTeeStage>(static_cast<u32>(lhs) | static_cast<u32>(rhs));
}

enum class PpUpstreamSampleTeeAccess : u32 {
    None = 0,
    MemoryRead = 1u << 0,
    MemoryWrite = 1u << 1,
    HostRead = 1u << 2,
    TransferWrite = 1u << 3,
    ShaderRead = 1u << 4,
    ShaderWrite = 1u << 5,
};

[[nodiscard]] constexpr PpUpstreamSampleTeeAccess operator|(
    PpUpstreamSampleTeeAccess lhs, PpUpstreamSampleTeeAccess rhs) noexcept {
    return static_cast<PpUpstreamSampleTeeAccess>(static_cast<u32>(lhs) | static_cast<u32>(rhs));
}

struct PpUpstreamSampleTeeBarrierPlan {
    PpUpstreamSampleTeeStage src_stages{PpUpstreamSampleTeeStage::None};
    PpUpstreamSampleTeeAccess src_access{PpUpstreamSampleTeeAccess::None};
    PpUpstreamSampleTeeStage dst_stages{PpUpstreamSampleTeeStage::None};
    PpUpstreamSampleTeeAccess dst_access{PpUpstreamSampleTeeAccess::None};
    u32 offset{};
    u32 bytes{};

    bool operator==(const PpUpstreamSampleTeeBarrierPlan&) const = default;
};

struct PpUpstreamSampleTeeDepthCopyAspectRestorePlan {
    u32 copy_aspect{};
    u32 transitioned_aspect{};
    u32 restore_aspect{};
    bool exact{};
};

[[nodiscard]] constexpr PpUpstreamSampleTeeDepthCopyAspectRestorePlan
PlanPpUpstreamSampleTeeDepthCopyAspectRestore(u32 actual_image_aspect,
                                              u32 copy_aspect) noexcept {
    if (actual_image_aspect == 0 || copy_aspect == 0 ||
        (actual_image_aspect & copy_aspect) != copy_aspect) {
        return {};
    }
    return {
        .copy_aspect = copy_aspect,
        .transitioned_aspect = actual_image_aspect,
        .restore_aspect = actual_image_aspect,
        .exact = true,
    };
}

enum class PpUpstreamSampleTeeCommandOp : u8 {
    EndPreRendering,
    BarrierToTransfer,
    ClearRecords,
    UpdateHeader,
    BarrierToFragment,
    ResumePreRendering,
    LateFragmentPush,
    GuestDraw,
    EndPostRendering,
    BarrierToHost,
    ResumePostRendering,
    DeferReadback,
};

struct PpUpstreamSampleTeeOutputDescriptor {
    bool enabled{};
    u32 attachment_span{};
    u32 active_attachment_mask{};
};

struct PpUpstreamSampleTeeOutputPlan {
    u32 output_candidate_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
};

[[nodiscard]] inline PpUpstreamSampleTeeOutputPlan PlanPpUpstreamSampleTeeOutputs(
    const PpUpstreamSampleTeeOutputDescriptor& descriptor) noexcept {
    if (!descriptor.enabled) {
        return {};
    }
    constexpr u32 MaxAttachments = 8;
    if (descriptor.attachment_span > MaxAttachments) {
        return {
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
        };
    }
    const u32 valid_mask = descriptor.attachment_span == MaxAttachments
                               ? (1u << MaxAttachments) - 1u
                           : descriptor.attachment_span == 0
                               ? 0u
                               : (1u << descriptor.attachment_span) - 1u;
    if ((descriptor.active_attachment_mask & ~valid_mask) != 0) {
        return {
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
        };
    }
    return {
        .output_candidate_count =
            static_cast<u32>(std::popcount(descriptor.active_attachment_mask)),
        .status = FinalGuestSurfaceStatus::Complete,
    };
}

struct PpUpstreamSampleTeeControllerDescriptor {
    bool enabled{};
    bool candidate_selected{};
    bool direct_draw{};
    bool fragment_stage_present{};
    u32 observed_candidate_index{};
    u32 configured_candidate_index{};
    PpUpstreamSampleTeeOutputPlan outputs{};
};

struct PpUpstreamSampleTeeControllerPlan {
    u32 candidate_index{};
    u32 output_candidate_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool plan_commands{};
    bool selection_pending{};
};

[[nodiscard]] inline PpUpstreamSampleTeeControllerPlan PlanPpUpstreamSampleTeeController(
    const PpUpstreamSampleTeeControllerDescriptor& descriptor) noexcept {
    if (!descriptor.enabled || !descriptor.candidate_selected) {
        return {};
    }
    const auto reject = [&](FinalGuestSurfaceStatus status, u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeControllerPlan result{
            .candidate_index = descriptor.observed_candidate_index,
            .status = status,
            .selection_pending = true,
        };
        result.loss.*member = 1;
        return result;
    };
    constexpr u32 MaxCandidateIndex =
        Shader::PpUpstreamSampleTeeCandidateMask >> Shader::PpUpstreamSampleTeeCandidateShift;
    if (descriptor.observed_candidate_index > MaxCandidateIndex ||
        descriptor.configured_candidate_index > MaxCandidateIndex ||
        descriptor.observed_candidate_index != descriptor.configured_candidate_index) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (!descriptor.direct_draw) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::unsupported_type);
    }
    if (!descriptor.fragment_stage_present) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (descriptor.outputs.status != FinalGuestSurfaceStatus::Complete ||
        descriptor.outputs.loss.Any()) {
        PpUpstreamSampleTeeControllerPlan result{
            .candidate_index = descriptor.observed_candidate_index,
            .status = descriptor.outputs.status,
            .loss = descriptor.outputs.loss,
            .selection_pending = true,
        };
        if (result.status == FinalGuestSurfaceStatus::Complete || !result.loss.Any()) {
            result.status = FinalGuestSurfaceStatus::InvalidationLoss;
            result.loss = {.invalidation = 1};
        }
        return result;
    }
    if (descriptor.outputs.output_candidate_count == 0) {
        return reject(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
    }
    if (descriptor.outputs.output_candidate_count != 1) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    return {
        .candidate_index = descriptor.observed_candidate_index,
        .output_candidate_count = descriptor.outputs.output_candidate_count,
        .status = FinalGuestSurfaceStatus::Complete,
        .plan_commands = true,
        .selection_pending = true,
    };
}

struct PpUpstreamSampleTeeControllerCompletionDescriptor {
    PpUpstreamSampleTeeControllerPlan controller{};
    bool upstream_copy_recorded{};
    bool slot_initialized{};
    bool rendering_resumed{};
    bool late_fragment_push_recorded{};
};

struct PpUpstreamSampleTeeControllerCompletionPlan {
    u32 candidate_index{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool complete_selection{};
    bool selection_succeeded{};
    bool record_candidate{};
};

[[nodiscard]] inline PpUpstreamSampleTeeControllerCompletionPlan
PlanPpUpstreamSampleTeeControllerCompletion(
    const PpUpstreamSampleTeeControllerCompletionDescriptor& descriptor) noexcept {
    const auto& controller = descriptor.controller;
    if (!controller.selection_pending) {
        return {};
    }
    if (controller.status != FinalGuestSurfaceStatus::Complete || controller.loss.Any() ||
        !controller.plan_commands) {
        auto status = controller.status;
        auto loss = controller.loss;
        if (status == FinalGuestSurfaceStatus::Complete || !loss.Any()) {
            status = FinalGuestSurfaceStatus::InvalidationLoss;
            loss = {.invalidation = 1};
        }
        return {
            .candidate_index = controller.candidate_index,
            .status = status,
            .loss = loss,
            .complete_selection = true,
        };
    }
    const bool succeeded = descriptor.upstream_copy_recorded && descriptor.slot_initialized &&
                           descriptor.rendering_resumed &&
                           descriptor.late_fragment_push_recorded;
    if (!succeeded) {
        return {
            .candidate_index = controller.candidate_index,
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
            .complete_selection = true,
        };
    }
    return {
        .candidate_index = controller.candidate_index,
        .status = FinalGuestSurfaceStatus::Complete,
        .complete_selection = true,
        .selection_succeeded = true,
        .record_candidate = true,
    };
}

enum class PpUpstreamSampleTeeFailureScope : u8 {
    Unspecified,
    TeeOnly,
    SharedCapture,
};

enum class PpUpstreamSampleTeeBaseCaptureState : u8 {
    Unspecified,
    Pending,
    Recorded,
    Rejected,
};

struct PpUpstreamSampleTeeFailureFanoutDescriptor {
    bool enabled{};
    bool tee_attempted{};
    bool input_selection_pending{};
    PpUpstreamSampleTeeFailureScope scope{};
    PpUpstreamSampleTeeBaseCaptureState base_state{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
};

struct PpUpstreamSampleTeeFailureFanoutPlan {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool record_tee_failure{};
    bool reject_base_capture{};
    bool continue_base_capture_without_tee{};
    bool complete_input_selection{};
    bool input_selection_succeeded{};
    bool preserve_consumer_bookkeeping{true};
    bool preserve_producer_bookkeeping{true};
};

[[nodiscard]] inline PpUpstreamSampleTeeFailureFanoutPlan PlanPpUpstreamSampleTeeFailureFanout(
    const PpUpstreamSampleTeeFailureFanoutDescriptor& descriptor) noexcept {
    PpUpstreamSampleTeeFailureFanoutPlan plan{};
    if (!descriptor.enabled || !descriptor.tee_attempted) {
        return plan;
    }
    plan.status = descriptor.status;
    plan.loss = descriptor.loss;
    const auto terminal_failure = [](FinalGuestSurfaceStatus status) {
        switch (status) {
        case FinalGuestSurfaceStatus::Unsupported:
        case FinalGuestSurfaceStatus::CapacityLoss:
        case FinalGuestSurfaceStatus::BusyLoss:
        case FinalGuestSurfaceStatus::InvalidationLoss:
        case FinalGuestSurfaceStatus::GapLoss:
            return true;
        default:
            return false;
        }
    };
    if (!terminal_failure(plan.status) || !plan.loss.Any()) {
        plan.status = FinalGuestSurfaceStatus::InvalidationLoss;
        plan.loss = {.invalidation = 1};
    }
    plan.record_tee_failure = true;
    const auto invalid_transition = [&] {
        plan.status = FinalGuestSurfaceStatus::InvalidationLoss;
        plan.loss = {.invalidation = 1};
        plan.reject_base_capture =
            descriptor.base_state != PpUpstreamSampleTeeBaseCaptureState::Rejected;
        plan.complete_input_selection = descriptor.input_selection_pending;
    };
    switch (descriptor.scope) {
    case PpUpstreamSampleTeeFailureScope::TeeOnly:
        switch (descriptor.base_state) {
        case PpUpstreamSampleTeeBaseCaptureState::Pending:
            plan.continue_base_capture_without_tee = true;
            break;
        case PpUpstreamSampleTeeBaseCaptureState::Recorded:
            plan.complete_input_selection = descriptor.input_selection_pending;
            plan.input_selection_succeeded = descriptor.input_selection_pending;
            break;
        case PpUpstreamSampleTeeBaseCaptureState::Rejected:
            plan.complete_input_selection = descriptor.input_selection_pending;
            break;
        default:
            invalid_transition();
            break;
        }
        break;
    case PpUpstreamSampleTeeFailureScope::SharedCapture:
        switch (descriptor.base_state) {
        case PpUpstreamSampleTeeBaseCaptureState::Pending:
        case PpUpstreamSampleTeeBaseCaptureState::Recorded:
            plan.reject_base_capture = true;
            plan.complete_input_selection = descriptor.input_selection_pending;
            break;
        case PpUpstreamSampleTeeBaseCaptureState::Rejected:
            plan.complete_input_selection = descriptor.input_selection_pending;
            break;
        default:
            invalid_transition();
            break;
        }
        break;
    default:
        invalid_transition();
        break;
    }
    return plan;
}

struct PpUpstreamFeedbackPlaneRecordResult {
    bool recorded{};
    PpUpstreamSampleTeeFailureScope failure_scope{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
};

struct PpUpstreamFeedbackPlaneRecordDispositionDescriptor {
    bool tee_enabled{};
    bool tee_attempted{};
    bool input_selection_pending{};
    PpUpstreamSampleTeeBaseCaptureState base_state{};
    PpUpstreamFeedbackPlaneRecordResult record{};
};

struct PpUpstreamFeedbackPlaneRecordDisposition {
    PpUpstreamSampleTeeFailureFanoutPlan fanout{};
    bool mark_copied{};
    bool retry_without_tee{};
};

[[nodiscard]] inline PpUpstreamFeedbackPlaneRecordDisposition
PlanPpUpstreamFeedbackPlaneRecordDisposition(
    const PpUpstreamFeedbackPlaneRecordDispositionDescriptor& descriptor) noexcept {
    const auto& record = descriptor.record;
    const bool canonical_success =
        record.recorded &&
        record.failure_scope == PpUpstreamSampleTeeFailureScope::Unspecified &&
        record.status == FinalGuestSurfaceStatus::Complete && !record.loss.Any();
    if (canonical_success) {
        return {.mark_copied = true};
    }
    const auto terminal_failure = [](FinalGuestSurfaceStatus status) {
        switch (status) {
        case FinalGuestSurfaceStatus::Unsupported:
        case FinalGuestSurfaceStatus::CapacityLoss:
        case FinalGuestSurfaceStatus::BusyLoss:
        case FinalGuestSurfaceStatus::InvalidationLoss:
        case FinalGuestSurfaceStatus::GapLoss:
            return true;
        default:
            return false;
        }
    };
    const bool canonical_failure =
        !record.recorded &&
        (record.failure_scope == PpUpstreamSampleTeeFailureScope::TeeOnly ||
         record.failure_scope == PpUpstreamSampleTeeFailureScope::SharedCapture) &&
        terminal_failure(record.status) && record.loss.Any();
    const bool valid_tee_only = record.failure_scope != PpUpstreamSampleTeeFailureScope::TeeOnly ||
                                (descriptor.tee_enabled && descriptor.tee_attempted);
    const auto scope = canonical_failure && valid_tee_only
                           ? record.failure_scope
                           : PpUpstreamSampleTeeFailureScope::SharedCapture;
    const auto status = canonical_failure && valid_tee_only
                            ? record.status
                            : FinalGuestSurfaceStatus::InvalidationLoss;
    const auto loss = canonical_failure && valid_tee_only
                          ? record.loss
                          : FinalGuestSurfaceLoss{.invalidation = 1};
    auto fanout = PlanPpUpstreamSampleTeeFailureFanout({
        .enabled = true,
        .tee_attempted = true,
        .input_selection_pending = descriptor.input_selection_pending,
        .scope = scope,
        .base_state = descriptor.base_state,
        .status = status,
        .loss = loss,
    });
    return {
        .fanout = fanout,
        .retry_without_tee = fanout.continue_base_capture_without_tee,
    };
}

struct PpUpstreamSampleTeeCommandDescriptor {
    bool enabled{};
    PpUpstreamSampleTeeBindingPlan binding{};
    PpUpstreamSampleTeeCapturePlan capture{};
    u64 non_coherent_atom_size{};
    u32 output_candidate_count{};
    bool direct_draw{};
    bool rendering_active{};
    u64 observed_rendering_serial{};
    u64 expected_rendering_serial{};
};

struct PpUpstreamSampleTeeCommandPlan {
    static constexpr u32 MaxOperations = 12;

    std::array<PpUpstreamSampleTeeCommandOp, MaxOperations> operations{};
    u32 operation_count{};
    PpUpstreamSampleTeeBarrierPlan reuse_barrier{};
    PpUpstreamSampleTeeBarrierPlan fragment_barrier{};
    PpUpstreamSampleTeeBarrierPlan host_barrier{};
    PpUpstreamSampleTeeStage partial_push_stages{PpUpstreamSampleTeeStage::None};
    u32 partial_push_offset{};
    u32 partial_push_bytes{};
    u64 serial{};
    u32 scheduler_waits{};
    u32 scheduler_finishes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool requires_allocation{};
    bool bind_buffer{};
    bool initialize{};
    bool arm{};
    bool issue_draw{};
    bool require_post_barrier{};
    bool defer_readback{};
    bool force_load{};
    bool require_preserved_serial{};
    bool logs_private_state{};

    bool operator==(const PpUpstreamSampleTeeCommandPlan&) const = default;
};

[[nodiscard]] inline PpUpstreamSampleTeeCommandPlan PlanPpUpstreamSampleTeeCommands(
    const PpUpstreamSampleTeeCommandDescriptor& descriptor) noexcept {
    if (!descriptor.enabled) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status, u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeCommandPlan result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    if (descriptor.binding.status != FinalGuestSurfaceStatus::Complete ||
        descriptor.binding.loss.Any() || !descriptor.binding.bind) {
        if (descriptor.binding.status != FinalGuestSurfaceStatus::Complete &&
            descriptor.binding.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
            descriptor.binding.loss.Any()) {
            return {.status = descriptor.binding.status, .loss = descriptor.binding.loss};
        }
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    const auto& capture = descriptor.capture;
    if (capture.status != FinalGuestSurfaceStatus::Complete || capture.loss.Any() ||
        !capture.initialize || !capture.arm || !capture.require_post_barrier) {
        if (capture.status != FinalGuestSurfaceStatus::Complete &&
            capture.status != FinalGuestSurfaceStatus::AlreadyConsumed && capture.loss.Any()) {
            return {.status = capture.status, .loss = capture.loss};
        }
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (!descriptor.direct_draw) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      &FinalGuestSurfaceLoss::unsupported_type);
    }
    if (descriptor.output_candidate_count == 0) {
        return reject(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
    }
    if (descriptor.output_candidate_count != 1) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (!descriptor.rendering_active) {
        return reject(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
    }
    if (descriptor.observed_rendering_serial == 0 || descriptor.expected_rendering_serial == 0) {
        return reject(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
    }
    if (descriptor.observed_rendering_serial != descriptor.expected_rendering_serial ||
        descriptor.non_coherent_atom_size == 0 ||
        capture.slot_offset % descriptor.non_coherent_atom_size != 0 ||
        capture.slot_bytes % descriptor.non_coherent_atom_size != 0) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }

    constexpr auto ReuseStages =
        PpUpstreamSampleTeeStage::AllCommands | PpUpstreamSampleTeeStage::Host;
    constexpr auto ReuseAccess = PpUpstreamSampleTeeAccess::MemoryRead |
                                 PpUpstreamSampleTeeAccess::MemoryWrite |
                                 PpUpstreamSampleTeeAccess::HostRead;
    constexpr auto FragmentAccess =
        PpUpstreamSampleTeeAccess::ShaderRead | PpUpstreamSampleTeeAccess::ShaderWrite;
    constexpr auto PostStages =
        PpUpstreamSampleTeeStage::Transfer | PpUpstreamSampleTeeStage::FragmentShader;
    constexpr auto PostAccess =
        PpUpstreamSampleTeeAccess::TransferWrite | PpUpstreamSampleTeeAccess::ShaderWrite;
    return {
        .operations = {PpUpstreamSampleTeeCommandOp::EndPreRendering,
                       PpUpstreamSampleTeeCommandOp::BarrierToTransfer,
                       PpUpstreamSampleTeeCommandOp::ClearRecords,
                       PpUpstreamSampleTeeCommandOp::UpdateHeader,
                       PpUpstreamSampleTeeCommandOp::BarrierToFragment,
                       PpUpstreamSampleTeeCommandOp::ResumePreRendering,
                       PpUpstreamSampleTeeCommandOp::LateFragmentPush,
                       PpUpstreamSampleTeeCommandOp::GuestDraw,
                       PpUpstreamSampleTeeCommandOp::EndPostRendering,
                       PpUpstreamSampleTeeCommandOp::BarrierToHost,
                       PpUpstreamSampleTeeCommandOp::ResumePostRendering,
                       PpUpstreamSampleTeeCommandOp::DeferReadback},
        .operation_count = PpUpstreamSampleTeeCommandPlan::MaxOperations,
        .reuse_barrier = {.src_stages = ReuseStages,
                          .src_access = ReuseAccess,
                          .dst_stages = PpUpstreamSampleTeeStage::Transfer,
                          .dst_access = PpUpstreamSampleTeeAccess::TransferWrite,
                          .offset = capture.slot_offset,
                          .bytes = capture.slot_bytes},
        .fragment_barrier = {.src_stages = PpUpstreamSampleTeeStage::Transfer,
                             .src_access = PpUpstreamSampleTeeAccess::TransferWrite,
                             .dst_stages = PpUpstreamSampleTeeStage::FragmentShader,
                             .dst_access = FragmentAccess,
                             .offset = capture.slot_offset,
                             .bytes = capture.slot_bytes},
        .host_barrier = {.src_stages = PostStages,
                         .src_access = PostAccess,
                         .dst_stages = PpUpstreamSampleTeeStage::Host,
                         .dst_access = PpUpstreamSampleTeeAccess::HostRead,
                         .offset = capture.slot_offset,
                         .bytes = capture.slot_bytes},
        .partial_push_stages = PpUpstreamSampleTeeStage::AllGraphics,
        .partial_push_offset = capture.push_offset,
        .partial_push_bytes = capture.push_bytes,
        .serial = descriptor.expected_rendering_serial,
        .status = FinalGuestSurfaceStatus::Complete,
        .requires_allocation = true,
        .bind_buffer = true,
        .initialize = true,
        .arm = true,
        .issue_draw = true,
        .require_post_barrier = true,
        .defer_readback = true,
        .force_load = true,
        .require_preserved_serial = true,
    };
}

enum class PpUpstreamSampleTeeControllerEvent : u8 {
    Initialized,
    Armed,
    DrawExecuted,
    PostBarrierComplete,
};

struct PpUpstreamSampleTeeControllerObservation {
    PpUpstreamSampleTeeControllerEvent event{};
    u32 observed_candidate_index{};
    u64 observed_serial{};
    bool succeeded{};
    bool direct_draw_executed{};
    bool host_barrier_recorded{};
    bool post_rendering_resumed{};
};

struct PpUpstreamSampleTeeControllerState {
    u32 candidate_index{};
    u64 serial{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool active{};
    bool initialized{};
    bool armed{};
    bool draw_executed{};
    bool post_barrier_complete{};
    bool ready{};

    bool operator==(const PpUpstreamSampleTeeControllerState&) const = default;
};

[[nodiscard]] inline PpUpstreamSampleTeeControllerState
BeginPpUpstreamSampleTeeControllerLifecycle(const PpUpstreamSampleTeeControllerPlan& controller,
                                            const PpUpstreamSampleTeeCommandPlan& commands) noexcept {
    if (controller.status == FinalGuestSurfaceStatus::AlreadyConsumed && !controller.loss.Any() &&
        !controller.plan_commands) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status, const FinalGuestSurfaceLoss& loss) {
        if (status != FinalGuestSurfaceStatus::Complete && loss.Any()) {
            return PpUpstreamSampleTeeControllerState{.status = status, .loss = loss};
        }
        return PpUpstreamSampleTeeControllerState{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
        };
    };
    if (controller.status != FinalGuestSurfaceStatus::Complete || controller.loss.Any() ||
        !controller.plan_commands || controller.output_candidate_count != 1) {
        return reject(controller.status, controller.loss);
    }
    if (commands.status != FinalGuestSurfaceStatus::Complete || commands.loss.Any() ||
        commands.serial == 0 || !commands.initialize || !commands.arm || !commands.issue_draw ||
        !commands.require_post_barrier || !commands.require_preserved_serial) {
        return reject(commands.status, commands.loss);
    }
    return {
        .candidate_index = controller.candidate_index,
        .serial = commands.serial,
        .status = FinalGuestSurfaceStatus::Complete,
        .active = true,
    };
}

[[nodiscard]] inline PpUpstreamSampleTeeControllerState
ObservePpUpstreamSampleTeeControllerLifecycle(
    PpUpstreamSampleTeeControllerState state,
    const PpUpstreamSampleTeeControllerObservation& observation) noexcept {
    const auto reject = [](PpUpstreamSampleTeeControllerState result) {
        result.status = FinalGuestSurfaceStatus::InvalidationLoss;
        result.loss = {.invalidation = 1};
        result.active = false;
        result.ready = false;
        return result;
    };
    if (!state.active || state.ready || state.status != FinalGuestSurfaceStatus::Complete ||
        state.loss.Any() || observation.observed_candidate_index != state.candidate_index ||
        !observation.succeeded || observation.observed_serial == 0 ||
        observation.observed_serial != state.serial) {
        return reject(state);
    }
    const auto expected = !state.initialized
                              ? PpUpstreamSampleTeeControllerEvent::Initialized
                          : !state.armed
                              ? PpUpstreamSampleTeeControllerEvent::Armed
                          : !state.draw_executed
                              ? PpUpstreamSampleTeeControllerEvent::DrawExecuted
                              : PpUpstreamSampleTeeControllerEvent::PostBarrierComplete;
    if (observation.event != expected) {
        return reject(state);
    }
    const bool invalid_evidence = [&] {
        switch (observation.event) {
        case PpUpstreamSampleTeeControllerEvent::Initialized:
        case PpUpstreamSampleTeeControllerEvent::Armed:
            return observation.direct_draw_executed || observation.host_barrier_recorded ||
                   observation.post_rendering_resumed;
        case PpUpstreamSampleTeeControllerEvent::DrawExecuted:
            return !observation.direct_draw_executed || observation.host_barrier_recorded ||
                   observation.post_rendering_resumed;
        case PpUpstreamSampleTeeControllerEvent::PostBarrierComplete:
            return observation.direct_draw_executed || !observation.host_barrier_recorded ||
                   !observation.post_rendering_resumed;
        }
        return true;
    }();
    if (invalid_evidence) {
        return reject(state);
    }
    switch (observation.event) {
    case PpUpstreamSampleTeeControllerEvent::Initialized:
        state.initialized = true;
        break;
    case PpUpstreamSampleTeeControllerEvent::Armed:
        state.armed = true;
        break;
    case PpUpstreamSampleTeeControllerEvent::DrawExecuted:
        state.draw_executed = true;
        break;
    case PpUpstreamSampleTeeControllerEvent::PostBarrierComplete:
        state.post_barrier_complete = true;
        state.ready = true;
        state.active = false;
        break;
    }
    return state;
}

struct PpUpstreamSampleTeePostDrawDescriptor {
    bool enabled{};
    PpUpstreamSampleTeeControllerState controller{};
    PpUpstreamSampleTeeCandidateRecord candidate{};
    PpUpstreamSampleTeeCapturePlan capture{};
    PpUpstreamSampleTeeCommandPlan commands{};
    u32 observed_candidate_index{};
    u64 observed_serial{};
    bool direct_draw_executed{};
};

struct PpUpstreamSampleTeePostDrawPlan {
    PpUpstreamSampleTeeControllerState controller{};
    PpUpstreamSampleTeeBarrierPlan host_barrier{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool issue_host_barrier{};
    bool resume_rendering{};
    bool observe_consumer{true};
    bool continue_producer_bookkeeping{true};
};

[[nodiscard]] inline PpUpstreamSampleTeePostDrawPlan PlanPpUpstreamSampleTeePostDraw(
    const PpUpstreamSampleTeePostDrawDescriptor& descriptor) noexcept {
    if (!descriptor.enabled) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status, FinalGuestSurfaceLoss loss) {
        if (status == FinalGuestSurfaceStatus::Complete || !loss.Any()) {
            status = FinalGuestSurfaceStatus::InvalidationLoss;
            loss = {.invalidation = 1};
        }
        return PpUpstreamSampleTeePostDrawPlan{
            .status = status,
            .loss = loss,
        };
    };
    const auto& controller = descriptor.controller;
    const auto& candidate = descriptor.candidate;
    const auto& capture = descriptor.capture;
    const auto& commands = descriptor.commands;
    const auto canonical_host = PpUpstreamSampleTeeBarrierPlan{
        .src_stages = PpUpstreamSampleTeeStage::Transfer | PpUpstreamSampleTeeStage::FragmentShader,
        .src_access =
            PpUpstreamSampleTeeAccess::TransferWrite | PpUpstreamSampleTeeAccess::ShaderWrite,
        .dst_stages = PpUpstreamSampleTeeStage::Host,
        .dst_access = PpUpstreamSampleTeeAccess::HostRead,
        .offset = capture.slot_offset,
        .bytes = capture.slot_bytes,
    };
    const bool exact_host = commands.host_barrier.src_stages == canonical_host.src_stages &&
                            commands.host_barrier.src_access == canonical_host.src_access &&
                            commands.host_barrier.dst_stages == canonical_host.dst_stages &&
                            commands.host_barrier.dst_access == canonical_host.dst_access &&
                            commands.host_barrier.offset == canonical_host.offset &&
                            commands.host_barrier.bytes == canonical_host.bytes;
    if (!controller.active || controller.ready ||
        controller.status != FinalGuestSurfaceStatus::Complete || controller.loss.Any() ||
        !controller.initialized || !controller.armed || controller.draw_executed ||
        controller.post_barrier_complete || !candidate.ready ||
        candidate.status != FinalGuestSurfaceStatus::Complete || candidate.loss.Any() ||
        candidate.candidate_index != descriptor.observed_candidate_index ||
        controller.candidate_index != descriptor.observed_candidate_index ||
        capture.status != FinalGuestSurfaceStatus::Complete || capture.loss.Any() ||
        !capture.initialize || !capture.arm || !capture.require_post_barrier ||
        capture.header.generation == 0 ||
        capture.header.candidate_index != descriptor.observed_candidate_index ||
        commands.status != FinalGuestSurfaceStatus::Complete || commands.loss.Any() ||
        !commands.issue_draw || !commands.require_post_barrier || !commands.force_load ||
        !commands.require_preserved_serial || commands.serial != descriptor.observed_serial ||
        !descriptor.direct_draw_executed || !exact_host) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1});
    }
    auto observed = ObservePpUpstreamSampleTeeControllerLifecycle(
        controller, {.event = PpUpstreamSampleTeeControllerEvent::DrawExecuted,
                     .observed_candidate_index = descriptor.observed_candidate_index,
                     .observed_serial = descriptor.observed_serial,
                     .succeeded = true,
                     .direct_draw_executed = true});
    if (!observed.active || observed.status != FinalGuestSurfaceStatus::Complete ||
        observed.loss.Any() || !observed.draw_executed) {
        return reject(observed.status, observed.loss);
    }
    return {
        .controller = observed,
        .host_barrier = commands.host_barrier,
        .status = FinalGuestSurfaceStatus::Complete,
        .issue_host_barrier = true,
        .resume_rendering = true,
    };
}

struct PpUpstreamSampleTeePostDrawCompletionDescriptor {
    PpUpstreamSampleTeePostDrawPlan plan{};
    u32 observed_candidate_index{};
    u64 observed_serial{};
    bool host_barrier_recorded{};
    bool rendering_resumed{};
};

struct PpUpstreamSampleTeePostDrawCompletion {
    PpUpstreamSampleTeeControllerState controller{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool ready{};
    bool observe_consumer{true};
    bool continue_producer_bookkeeping{true};
};

[[nodiscard]] inline PpUpstreamSampleTeePostDrawCompletion CompletePpUpstreamSampleTeePostDraw(
    const PpUpstreamSampleTeePostDrawCompletionDescriptor& descriptor) noexcept {
    const auto reject = [](FinalGuestSurfaceStatus status, FinalGuestSurfaceLoss loss) {
        if (status == FinalGuestSurfaceStatus::Complete || !loss.Any()) {
            status = FinalGuestSurfaceStatus::InvalidationLoss;
            loss = {.invalidation = 1};
        }
        return PpUpstreamSampleTeePostDrawCompletion{
            .status = status,
            .loss = loss,
        };
    };
    if (descriptor.plan.status != FinalGuestSurfaceStatus::Complete || descriptor.plan.loss.Any() ||
        !descriptor.plan.issue_host_barrier || !descriptor.plan.resume_rendering ||
        !descriptor.host_barrier_recorded || !descriptor.rendering_resumed) {
        return reject(descriptor.plan.status, descriptor.plan.loss);
    }
    auto observed = ObservePpUpstreamSampleTeeControllerLifecycle(
        descriptor.plan.controller,
        {.event = PpUpstreamSampleTeeControllerEvent::PostBarrierComplete,
         .observed_candidate_index = descriptor.observed_candidate_index,
         .observed_serial = descriptor.observed_serial,
         .succeeded = true,
         .host_barrier_recorded = true,
         .post_rendering_resumed = true});
    if (!observed.ready || observed.active ||
        observed.status != FinalGuestSurfaceStatus::Complete || observed.loss.Any()) {
        return reject(observed.status, observed.loss);
    }
    return {
        .controller = observed,
        .status = FinalGuestSurfaceStatus::Complete,
        .ready = true,
    };
}

enum class PpUpstreamFeedbackCubeBindingDisposition : u8 {
    Idle = 0,
    NoAuthority = 1,
    MissingBinding = 2,
    Exact = 3,
    InvalidTransition = 4,
};

struct PpUpstreamFeedbackCubeLiveObservation {
    VideoCore::ImageId physical_image{};
    Shader::PpUpstreamSampleTeeCubeBindingFact binding{};
    PpUpstreamFeedbackSampledViewBindingFact view{};

    bool operator==(const PpUpstreamFeedbackCubeLiveObservation&) const = default;
};

[[nodiscard]] constexpr bool ShouldRecordPpUpstreamFeedbackCubeLiveObservation(
    const Shader::PpUpstreamSampleTeeCubeBindingFact& binding) noexcept {
    return binding.observed;
}

struct PpUpstreamFeedbackCubeBindingSummary {
    Shader::PpUpstreamSampleTeeCubeShape shape{
        Shader::PpUpstreamSampleTeeCubeShape::NonCube};
    u32 cube_count{};
    u32 base_layer{};
    u32 view_layer_count{};
    bool full_cube_range{};
    PpUpstreamFeedbackCubeBindingDisposition disposition{
        PpUpstreamFeedbackCubeBindingDisposition::Idle};

    bool operator==(const PpUpstreamFeedbackCubeBindingSummary&) const = default;
};

struct PpUpstreamFeedbackCubeBindingSnapshot {
    PpUpstreamFeedbackCubeBindingSummary fact{};
    u32 selected_candidate_index{};
    u8 observation_count{};
    bool overflowed{};

    bool operator==(const PpUpstreamFeedbackCubeBindingSnapshot&) const = default;
};

static_assert(std::is_trivially_copyable_v<PpUpstreamFeedbackCubeBindingSnapshot>);
static_assert(sizeof(PpUpstreamFeedbackCubeBindingSnapshot) <= 96);

struct PpUpstreamSampleTeeSourceObservation {
    PpUpstreamSampleRoute route{};
    VideoCore::ImageColorScopeTrackedView binding{};
    u32 resolved_view_format{};
    u32 resolved_view_aspect{};
    VideoCore::ImageProducerClass producer{VideoCore::ImageProducerClass::Unknown};
    PpUpstreamFeedbackCubeBindingSnapshot cube_binding{};
    bool valid{};

    bool operator==(const PpUpstreamSampleTeeSourceObservation&) const = default;
};

[[nodiscard]] constexpr bool IsCanonicalPpUpstreamSampleTeeSourceObservation(
    const PpUpstreamSampleTeeSourceObservation& source,
    const PpUpstreamSampleTeeCandidateRecord& candidate) noexcept {
    const auto& route = source.route;
    const auto& binding = source.binding;
    const auto& view = route.view;
    return source.valid && route.valid &&
           route.logical_stage == PpUpstreamSampleTeeShaderPlan::FragmentLogicalStage &&
           route.image_resource_index == candidate.shader.resource_index &&
           route.array_element == 0 && route.sampler_count <= route.MaxSamplers && binding.valid &&
           binding.link.Valid() && binding.backing_generation != 0 && binding.mutation_epoch != 0 &&
           !binding.view_conflict && source.resolved_view_format != 0 &&
           source.resolved_view_aspect != 0 && view.type == binding.type &&
           view.format == binding.format && view.base_mip == binding.base_mip &&
           view.mip_count == binding.mip_count && view.base_layer == binding.base_layer &&
           view.layer_count == binding.layer_count && view.mapping == binding.mapping &&
           view.min_lod == binding.min_lod && view.storage == binding.storage &&
           static_cast<u32>(source.producer) <=
               static_cast<u32>(VideoCore::ImageProducerClass::CpuUpload);
}

struct PpUpstreamSampleTeeTransferredDepthSourceToken {
    VideoCore::ImageColorScopePrivateLink link{};
    u64 backing_generation{};
    u64 mutation_epoch{};

    [[nodiscard]] constexpr bool Valid() const noexcept {
        return link.Valid() && backing_generation != 0 && mutation_epoch != 0;
    }

    bool operator==(const PpUpstreamSampleTeeTransferredDepthSourceToken&) const = default;
};

[[nodiscard]] constexpr bool IsCurrentPpUpstreamSampleTeeTransferredDepthSourceToken(
    PpUpstreamSampleTeeTransferredDepthSourceToken token,
    VideoCore::ImageColorScopePrivateLink resolved_link, u64 resolved_backing_generation,
    u64 resolved_mutation_epoch, bool source_still_live) noexcept {
    return source_still_live && token.Valid() && resolved_link == token.link &&
           resolved_backing_generation == token.backing_generation &&
           resolved_mutation_epoch == token.mutation_epoch;
}

struct PpUpstreamSampleTeeTransferredDepthSourceDescriptor {
    bool enabled{};
    VideoCore::ImageTransferKind outer_kind{VideoCore::ImageTransferKind::None};
    VideoCore::ImageTransferOrigin outer_origin{VideoCore::ImageTransferOrigin::Unknown};
    VideoCore::ImageTransferKind source_kind{VideoCore::ImageTransferKind::None};
    VideoCore::ImageTransferOrigin source_origin{VideoCore::ImageTransferOrigin::Unknown};
    PpUpstreamSampleTeeTransferredDepthSourceToken source_token{};
    u32 logical_width{};
    u32 logical_height{};
    u32 source_width{};
    u32 source_height{};
    u32 bytes_per_texel{};
    u32 base_offset{};
    FinalGuestSurfaceWatchOrdinals selector{};
    u32 buffer_alignment{};
    u32 max_regions{};
    u32 max_bytes{};
    u32 base_mip{};
    u32 mip_count{};
    u32 base_layer{};
    u32 layer_count{};
    u32 source_samples{};
    u32 old_samples{};
    u32 new_samples{};
    bool range_exact{};
    bool request_matches_encoded{};
    bool full_xy{};
    bool source_transfer_exact{};
    bool source_is_depth{};
    bool depth_format_supported{};
    bool source_still_live{};
    bool source_identity_exact{};
    bool source_backing_exact{};
    bool source_mutation_exact{};
    bool shared_slot_present{};
    bool barrier_supported{};
    bool deferred_shared_release{};
};

struct PpUpstreamSampleTeeTransferredDepthSourcePlan {
    PpUpstreamInputContentPlane plane{};
    PpUpstreamSampleTeeTransferredDepthSourceToken source_token{};
    u32 total_bytes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy{};
    bool uses_shared_slot{};
    bool require_pre_draw_barrier{};
    bool restore_source_state{};
    bool deferred_shared_release{};
    bool logs_private_state{};
};

[[nodiscard]] inline bool IsPpUpstreamSampleTeeTransferredDepthSourcePlanRecordable(
    const PpUpstreamSampleTeeTransferredDepthSourcePlan& plan) noexcept {
    const auto& plane = plan.plane;
    if (plan.status != FinalGuestSurfaceStatus::Complete || plan.loss.Any() || !plan.copy ||
        !plan.source_token.Valid() || !plan.uses_shared_slot || !plan.require_pre_draw_barrier ||
        !plan.restore_source_state || !plan.deferred_shared_release || plan.logs_private_state ||
        plane.status != FinalGuestSurfaceStatus::Complete || plane.loss.Any() ||
        plane.region_count == 0 || plane.region_count > FinalGuestSurfaceWatchOrdinals::MaxOrdinals ||
        plane.plane_bytes == 0 || plane.bytes_per_texel != 4 || plane.base_mip != 0 ||
        plane.mip_count != 1 || plane.base_layer != 0 ||
        static_cast<u64>(plane.plane_offset) + plane.plane_bytes != plan.total_bytes ||
        plan.total_bytes > Shader::PpUpstreamSampleTeeSlotStrideBytes ||
        static_cast<u64>(plane.plane_offset) + plane.plane_bytes >
            Shader::PpUpstreamSampleTeeSlotStrideBytes) {
        return false;
    }
    u64 covered{};
    for (u32 index = 0; index < plane.region_count; ++index) {
        const auto& region = plane.regions[index];
        if (region.logical_ordinal == 0 || region.mip_level != 0 || region.width == 0 ||
            region.height == 0 || region.byte_size == 0 ||
            region.byte_size % plane.bytes_per_texel != 0 ||
            region.buffer_offset != covered ||
            static_cast<u64>(region.buffer_offset) + region.byte_size > plane.plane_bytes) {
            return false;
        }
        covered += region.byte_size;
    }
    return covered == plane.plane_bytes;
}

[[nodiscard]] inline PpUpstreamSampleTeeBarrierPlan
PlanPpUpstreamSampleTeeTransferredDepthSourcePreCopyBarrier(
    const PpUpstreamSampleTeeTransferredDepthSourcePlan& plan, u32 slot_offset) noexcept {
    if (!IsPpUpstreamSampleTeeTransferredDepthSourcePlanRecordable(plan) ||
        plan.plane.plane_offset > std::numeric_limits<u32>::max() - slot_offset) {
        return {};
    }
    return {
        .src_stages = PpUpstreamSampleTeeStage::Transfer,
        .src_access = PpUpstreamSampleTeeAccess::TransferWrite,
        .dst_stages = PpUpstreamSampleTeeStage::Transfer,
        .dst_access = PpUpstreamSampleTeeAccess::TransferWrite,
        .offset = slot_offset + plan.plane.plane_offset,
        .bytes = plan.plane.plane_bytes,
    };
}

[[nodiscard]] inline PpUpstreamSampleTeeTransferredDepthSourcePlan
PlanPpUpstreamSampleTeeTransferredDepthSource(
    const PpUpstreamSampleTeeTransferredDepthSourceDescriptor& descriptor) noexcept {
    if (!descriptor.enabled) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status,
                           u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeTransferredDepthSourcePlan result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    if (descriptor.outer_kind != VideoCore::ImageTransferKind::CopyImage ||
        descriptor.outer_origin != VideoCore::ImageTransferOrigin::ComputeCopy ||
        descriptor.source_kind != VideoCore::ImageTransferKind::CopyImage ||
        descriptor.source_origin != VideoCore::ImageTransferOrigin::DepthOverlap ||
        !descriptor.source_is_depth || !descriptor.depth_format_supported ||
        descriptor.base_mip != 0 || descriptor.mip_count != 1 || descriptor.base_layer != 0 ||
        descriptor.layer_count != 1 || descriptor.source_samples != 1 ||
        descriptor.old_samples != 1 || descriptor.new_samples != 1) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      descriptor.source_is_depth && descriptor.depth_format_supported
                          ? &FinalGuestSurfaceLoss::unsupported_type
                          : &FinalGuestSurfaceLoss::unsupported_format);
    }
    if (!descriptor.source_still_live) {
        return reject(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
    }
    if (!descriptor.source_token.Valid() || !descriptor.range_exact ||
        !descriptor.request_matches_encoded || !descriptor.full_xy ||
        !descriptor.source_transfer_exact || !descriptor.source_identity_exact ||
        !descriptor.source_backing_exact || !descriptor.source_mutation_exact ||
        !descriptor.barrier_supported || !descriptor.deferred_shared_release) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (!descriptor.shared_slot_present) {
        return reject(FinalGuestSurfaceStatus::BusyLoss, &FinalGuestSurfaceLoss::busy);
    }
    const std::array view{PpUpstreamInputContentView{
        .width = descriptor.source_width,
        .height = descriptor.source_height,
        .bytes_per_texel = descriptor.bytes_per_texel,
        .base_mip = descriptor.base_mip,
        .mip_count = descriptor.mip_count,
        .base_layer = descriptor.base_layer,
        .layer_count = descriptor.layer_count,
        .color = true,
        .type_2d = true,
        .uniform_state = true,
    }};
    const auto mapped = PlanPpUpstreamInputContent({
        .enabled = true,
        .logical_width = descriptor.logical_width,
        .logical_height = descriptor.logical_height,
        .base_offset = descriptor.base_offset,
        .selector = descriptor.selector,
        .views = view,
        .buffer_alignment = descriptor.buffer_alignment,
        .max_regions = descriptor.max_regions,
        .max_bytes = descriptor.max_bytes,
    });
    if (mapped.status != FinalGuestSurfaceStatus::Complete || mapped.loss.Any() || !mapped.copy ||
        mapped.input_count != 1 || mapped.capture_mask != 1 || mapped.unavailable_mask != 0 ||
        mapped.alias_mask != 0) {
        return {.status = mapped.status, .loss = mapped.loss};
    }
    PpUpstreamSampleTeeTransferredDepthSourcePlan result{
        .plane = mapped.inputs[0],
        .source_token = descriptor.source_token,
        .total_bytes = mapped.total_bytes,
        .status = FinalGuestSurfaceStatus::Complete,
        .copy = true,
        .uses_shared_slot = true,
        .require_pre_draw_barrier = true,
        .restore_source_state = true,
        .deferred_shared_release = true,
    };
    if (!IsPpUpstreamSampleTeeTransferredDepthSourcePlanRecordable(result)) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    return result;
}

struct PpUpstreamSampleTeeSelectedDepthToken {
    VideoCore::ImageColorScopePrivateLink link{};
    u64 backing_generation{};
    u64 mutation_epoch{};

    [[nodiscard]] constexpr bool Valid() const noexcept {
        return link.Valid() && backing_generation != 0 && mutation_epoch != 0;
    }

    bool operator==(const PpUpstreamSampleTeeSelectedDepthToken&) const = default;
};

[[nodiscard]] constexpr bool IsCurrentPpUpstreamSampleTeeSelectedDepthToken(
    PpUpstreamSampleTeeSelectedDepthToken token,
    VideoCore::ImageColorScopePrivateLink resolved_link, u64 resolved_backing_generation,
    u64 resolved_mutation_epoch, bool source_still_live) noexcept {
    return source_still_live && token.Valid() && resolved_link == token.link &&
           resolved_backing_generation == token.backing_generation &&
           resolved_mutation_epoch == token.mutation_epoch;
}

struct PpUpstreamSampleTeeSelectedDepthDescriptor {
    bool enabled{};
    PpUpstreamSampleRouteSet routes{};
    PpUpstreamSampleTeeSourceObservation selected_source{};
    PpUpstreamSampleTeeSelectedDepthToken selected_token{};
    u32 selected_input_index{};
    u32 selected_candidate_index{};
    PpUpstreamSampleTeeCandidateRecord candidate{};
    u32 logical_width{};
    u32 logical_height{};
    u32 tee_occupied_bytes{};
    PpUpstreamSampleTeeTransferredDepthSourcePlan transferred_depth_source{};
    u32 base_offset{};
    FinalGuestSurfaceWatchOrdinals selector{};
    u32 buffer_alignment{};
    u32 max_regions{};
    u32 max_bytes{};
    bool depth_format_supported{};
    bool actual_image_depth_aspect{};
    bool source_still_live{};
    bool source_identity_exact{};
    bool source_backing_exact{};
    bool source_mutation_exact{};
    bool shared_slot_present{};
    bool barrier_supported{};
    bool capture_pre_draw{};
    bool deferred_shared_release{};
};

struct PpUpstreamSampleTeeSelectedDepthPlane {
    static constexpr u32 MaxRegions = FinalGuestSurfaceWatchOrdinals::MaxOrdinals;

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

    PpUpstreamSampleTeeSelectedDepthPlane() = default;

    PpUpstreamSampleTeeSelectedDepthPlane(const PpUpstreamInputContentPlane& source) noexcept {
        *this = source;
    }

    PpUpstreamSampleTeeSelectedDepthPlane& operator=(
        const PpUpstreamInputContentPlane& source) noexcept {
        regions = {};
        region_count = source.region_count;
        for (u32 index = 0; index < std::min(region_count, MaxRegions); ++index) {
            regions[index] = source.regions[index];
        }
        plane_offset = source.plane_offset;
        plane_bytes = source.plane_bytes;
        bytes_per_texel = source.bytes_per_texel;
        base_mip = source.base_mip;
        mip_count = source.mip_count;
        base_layer = source.base_layer;
        status = source.status;
        loss = source.loss;
        return *this;
    }
};

struct PpUpstreamSampleTeeSelectedDepthPlan {
    PpUpstreamSampleTeeSelectedDepthPlane plane{};
    PpUpstreamSampleTeeSelectedDepthToken selected_token{};
    u32 total_bytes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy{};
    bool uses_shared_slot{};
    bool require_pre_copy_barrier{};
    bool restore_selected_state{};
    bool capture_pre_draw{};
    bool deferred_shared_release{};
    bool observation_only{};
    bool affects_renderer_behavior{};
    bool logs_private_state{};
};

[[nodiscard]] inline bool IsPpUpstreamSampleTeeSelectedDepthPlanRecordable(
    const PpUpstreamSampleTeeSelectedDepthPlan& plan) noexcept {
    const auto& plane = plan.plane;
    if (plan.status != FinalGuestSurfaceStatus::Complete || plan.loss.Any() || !plan.copy ||
        !plan.selected_token.Valid() || !plan.uses_shared_slot ||
        !plan.require_pre_copy_barrier || !plan.restore_selected_state ||
        !plan.capture_pre_draw || !plan.deferred_shared_release || !plan.observation_only ||
        plan.affects_renderer_behavior || plan.logs_private_state ||
        plane.status != FinalGuestSurfaceStatus::Complete || plane.loss.Any() ||
        plane.region_count == 0 ||
        plane.region_count > FinalGuestSurfaceWatchOrdinals::MaxOrdinals ||
        plane.plane_bytes == 0 || plane.bytes_per_texel != 4 || plane.base_mip != 0 ||
        plane.mip_count != 1 || plane.base_layer != 0 ||
        static_cast<u64>(plane.plane_offset) + plane.plane_bytes != plan.total_bytes ||
        plan.total_bytes > Shader::PpUpstreamSampleTeeSlotStrideBytes) {
        return false;
    }
    u64 covered{};
    for (u32 index = 0; index < plane.region_count; ++index) {
        const auto& region = plane.regions[index];
        if (region.logical_ordinal == 0 || region.mip_level != 0 || region.width == 0 ||
            region.height == 0 || region.byte_size == 0 ||
            region.byte_size % plane.bytes_per_texel != 0 || region.buffer_offset != covered ||
            static_cast<u64>(region.buffer_offset) + region.byte_size > plane.plane_bytes) {
            return false;
        }
        covered += region.byte_size;
    }
    return covered == plane.plane_bytes;
}

[[nodiscard]] inline PpUpstreamSampleTeeBarrierPlan
PlanPpUpstreamSampleTeeSelectedDepthPreCopyBarrier(
    const PpUpstreamSampleTeeSelectedDepthPlan& plan, u32 slot_offset) noexcept {
    if (!IsPpUpstreamSampleTeeSelectedDepthPlanRecordable(plan) ||
        plan.plane.plane_offset > std::numeric_limits<u32>::max() - slot_offset) {
        return {};
    }
    return {
        .src_stages = PpUpstreamSampleTeeStage::Transfer,
        .src_access = PpUpstreamSampleTeeAccess::TransferWrite,
        .dst_stages = PpUpstreamSampleTeeStage::Transfer,
        .dst_access = PpUpstreamSampleTeeAccess::TransferWrite,
        .offset = slot_offset + plan.plane.plane_offset,
        .bytes = plan.plane.plane_bytes,
    };
}

[[nodiscard]] inline PpUpstreamSampleTeeSelectedDepthPlan
PlanPpUpstreamSampleTeeSelectedDepth(
    const PpUpstreamSampleTeeSelectedDepthDescriptor& descriptor) noexcept {
    if (!descriptor.enabled) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status,
                           u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeSelectedDepthPlan result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    const auto& routes = descriptor.routes;
    if (routes.route_count == 0) {
        return reject(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
    }
    if (routes.status != FinalGuestSurfaceStatus::Complete || routes.loss.Any()) {
        return {.status = routes.status, .loss = routes.loss};
    }
    constexpr u32 DepthAspect = 2;
    const auto& binding = descriptor.selected_source.binding;
    if (descriptor.selected_source.resolved_view_aspect != DepthAspect ||
        binding.aspect_mask != DepthAspect || binding.color || !binding.type_2d ||
        binding.base_mip != 0 || binding.mip_count != 1 || binding.base_layer != 0 ||
        binding.layer_count != 1 || binding.samples != 1 || binding.bytes_per_texel != 4 ||
        !descriptor.depth_format_supported || !descriptor.actual_image_depth_aspect) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      !descriptor.depth_format_supported || binding.bytes_per_texel != 4
                          ? &FinalGuestSurfaceLoss::unsupported_format
                          : &FinalGuestSurfaceLoss::unsupported_type);
    }
    if (routes.selected_input_index != descriptor.selected_input_index ||
        routes.route_count != 1 || routes.retains_image || routes.retains_view ||
        routes.retains_sampler || routes.logs_private_words ||
        routes.routes[0] != descriptor.selected_source.route ||
        !IsCanonicalPpUpstreamSampleTeeCandidateRecord(
            descriptor.candidate, descriptor.selected_candidate_index) ||
        !IsCanonicalPpUpstreamSampleTeeSourceObservation(descriptor.selected_source,
                                                         descriptor.candidate)) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (!descriptor.source_still_live) {
        return reject(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
    }
    if (!descriptor.selected_token.Valid() || binding.link != descriptor.selected_token.link ||
        binding.backing_generation != descriptor.selected_token.backing_generation ||
        binding.mutation_epoch != descriptor.selected_token.mutation_epoch ||
        !binding.uniform_state || !descriptor.source_identity_exact ||
        !descriptor.source_backing_exact || !descriptor.source_mutation_exact ||
        !descriptor.barrier_supported || !descriptor.capture_pre_draw ||
        !descriptor.deferred_shared_release || descriptor.tee_occupied_bytes == 0) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (!descriptor.shared_slot_present) {
        return reject(FinalGuestSurfaceStatus::BusyLoss, &FinalGuestSurfaceLoss::busy);
    }
    const auto& transfer = descriptor.transferred_depth_source;
    const bool has_transfer = transfer.status != FinalGuestSurfaceStatus::AlreadyConsumed ||
                              transfer.loss.Any() || transfer.copy || transfer.source_token.Valid() ||
                              transfer.plane.region_count != 0 || transfer.total_bytes != 0;
    const u32 occupied_prefix = has_transfer ? transfer.total_bytes : descriptor.tee_occupied_bytes;
    if ((has_transfer &&
         (!IsPpUpstreamSampleTeeTransferredDepthSourcePlanRecordable(transfer) ||
          transfer.plane.plane_offset < descriptor.tee_occupied_bytes)) ||
        descriptor.base_offset != occupied_prefix) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (descriptor.max_bytes > Shader::PpUpstreamSampleTeeSlotStrideBytes ||
        occupied_prefix >= descriptor.max_bytes) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss,
                      &FinalGuestSurfaceLoss::byte_capacity);
    }
    const std::array view{PpUpstreamInputContentView{
        .width = binding.width,
        .height = binding.height,
        .bytes_per_texel = binding.bytes_per_texel,
        .base_mip = binding.base_mip,
        .mip_count = binding.mip_count,
        .base_layer = binding.base_layer,
        .layer_count = binding.layer_count,
        .color = true,
        .type_2d = true,
        .uniform_state = true,
    }};
    const auto mapped = PlanPpUpstreamInputContent({
        .enabled = true,
        .logical_width = descriptor.logical_width,
        .logical_height = descriptor.logical_height,
        .base_offset = descriptor.base_offset,
        .selector = descriptor.selector,
        .views = view,
        .buffer_alignment = descriptor.buffer_alignment,
        .max_regions = descriptor.max_regions,
        .max_bytes = descriptor.max_bytes,
    });
    if (mapped.status != FinalGuestSurfaceStatus::Complete || mapped.loss.Any() || !mapped.copy ||
        mapped.input_count != 1 || mapped.capture_mask != 1 || mapped.unavailable_mask != 0 ||
        mapped.alias_mask != 0) {
        return {.status = mapped.status, .loss = mapped.loss};
    }
    PpUpstreamSampleTeeSelectedDepthPlan result{
        .plane = mapped.inputs[0],
        .selected_token = descriptor.selected_token,
        .total_bytes = mapped.total_bytes,
        .status = FinalGuestSurfaceStatus::Complete,
        .copy = true,
        .uses_shared_slot = true,
        .require_pre_copy_barrier = true,
        .restore_selected_state = true,
        .capture_pre_draw = true,
        .deferred_shared_release = true,
        .observation_only = true,
    };
    if (!IsPpUpstreamSampleTeeSelectedDepthPlanRecordable(result)) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    return result;
}

inline constexpr u32 PpUpstreamSampleTeeProducerPreDepthTailBytes = 128u << 10;
inline constexpr u32 PpUpstreamSampleTeeProducerPreDepthPrefixBytes =
    PpUpstreamInputSnapshotBytes - PpUpstreamSampleTeeProducerPreDepthTailBytes;
inline constexpr u32 PpUpstreamSampleTeePredecessorWriterPreDepthTailBytes = 128u << 10;
inline constexpr u32 PpUpstreamSampleTeeDownloadPrefixBytes =
    PpUpstreamInputSnapshotBytes - PpUpstreamSampleTeeProducerPreDepthTailBytes -
    PpUpstreamSampleTeePredecessorWriterPreDepthTailBytes;
inline constexpr u32 PpUpstreamSampleTeePredecessorWriterPreDepthTailOffset =
    PpUpstreamSampleTeeDownloadPrefixBytes;
inline constexpr u32 PpUpstreamSampleTeeProducerPreDepthTailOffset =
    PpUpstreamSampleTeeProducerPreDepthPrefixBytes;

struct PpUpstreamSampleTeeProducerPreDepthHint {
    VideoCore::ImageColorScopePrivateLink link{};
    VideoCore::SubresourceRange range{};
    u64 backing_generation{};
    PpDepthPredecessorHint writer{};
    bool valid{};
    bool private_only{};
    bool logs_private_state{};

    bool operator==(const PpUpstreamSampleTeeProducerPreDepthHint&) const = default;
};

using PpUpstreamSampleTeePredecessorWriterPreDepthHint =
    PpUpstreamSampleTeeProducerPreDepthHint;

struct PpUpstreamSampleTeePredecessorWriterPreDepthFrameDescriptor {
    PpUpstreamSampleTeePredecessorWriterPreDepthHint prior_hint{};
    PpUpstreamSampleTeePredecessorWriterPreDepthHint observed_load_hint{};
    u32 exact_load_count{};
    bool predecessor_pre_attempted{};
    bool predecessor_pre_open{};
    bool predecessor_post_exact{};
    bool predecessor_bound_to_load{};
    bool predecessor_collision{};
};

struct PpUpstreamSampleTeePredecessorWriterPreDepthFrameResult {
    PpUpstreamSampleTeePredecessorWriterPreDepthHint next_hint{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool warmup_completed{};
    bool retains_private_state{};
};

[[nodiscard]] inline PpUpstreamSampleTeePredecessorWriterPreDepthFrameResult
FinishPpUpstreamSampleTeePredecessorWriterPreDepthFrame(
    const PpUpstreamSampleTeePredecessorWriterPreDepthFrameDescriptor& descriptor) noexcept {
    const bool warmup = !descriptor.prior_hint.valid;
    const bool exact_hint = descriptor.observed_load_hint.valid &&
                            (warmup || descriptor.observed_load_hint == descriptor.prior_hint);
    const bool exact_attempt =
        warmup || (descriptor.predecessor_pre_attempted && !descriptor.predecessor_pre_open &&
                   descriptor.predecessor_post_exact && descriptor.predecessor_bound_to_load &&
                   !descriptor.predecessor_collision);
    if (descriptor.exact_load_count != 1 || !exact_hint || !exact_attempt) {
        return {.status = FinalGuestSurfaceStatus::GapLoss, .loss = {.gap = 1}};
    }
    return {.next_hint = descriptor.observed_load_hint,
            .status = FinalGuestSurfaceStatus::Complete,
            .warmup_completed = warmup};
}

struct PpUpstreamSampleTeePredecessorWriterPreDepthHookDescriptor {
    bool enabled{};
    bool sample_tee_enabled{};
    bool capture_upstream_depth_pre_post{};
    bool predecessor_hint_armed{};
    bool pre_draw{};
    PpUpstreamSampleTeePredecessorWriterPreDepthHint learned_hint{};
    PpDepthPredecessorWriterIntent observed_writer{};
    bool effective_depth_write{};
    bool actual_depth_aspect{};
    bool canonical_range{};
    bool source_still_live{};
    bool source_identity_exact{};
    bool source_backing_exact{};
    bool source_mutation_exact{};
    bool current_target_armed{};
    bool current_target_observed{};
};

struct PpUpstreamSampleTeePredecessorWriterPreDepthHookPlan {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool capture_pre{};
    bool acquire_existing_outer_slot_early{};
    bool requires_current_target_arm{};
    bool requires_current_target_observation{};
    bool owns_second_token{};
    bool observation_only{};
    bool affects_renderer_behavior{};
};

[[nodiscard]] inline PpUpstreamSampleTeePredecessorWriterPreDepthHookPlan
PlanPpUpstreamSampleTeePredecessorWriterPreDepthHook(
    const PpUpstreamSampleTeePredecessorWriterPreDepthHookDescriptor& descriptor) noexcept {
    const auto& learned = descriptor.learned_hint;
    const auto& observed = descriptor.observed_writer;
    const bool exact = descriptor.enabled && descriptor.sample_tee_enabled &&
                       !descriptor.capture_upstream_depth_pre_post &&
                       descriptor.predecessor_hint_armed && descriptor.pre_draw && learned.valid &&
                       learned.private_only && !learned.logs_private_state && observed.Valid() &&
                       observed.link == learned.link && observed.range == learned.range &&
                       observed.backing_generation == learned.backing_generation &&
                       observed.hint == learned.writer && descriptor.effective_depth_write &&
                       descriptor.actual_depth_aspect && descriptor.canonical_range &&
                       descriptor.source_still_live && descriptor.source_identity_exact &&
                       descriptor.source_backing_exact && descriptor.source_mutation_exact;
    if (!exact) {
        return {};
    }
    return {.status = FinalGuestSurfaceStatus::Complete,
            .capture_pre = true,
            .acquire_existing_outer_slot_early = true,
            .observation_only = true};
}

struct PpUpstreamSampleTeeProducerPreDepthScopeEntryRuntimeEligibilityDescriptor {
    bool armed{};
    PpUpstreamSampleTeeProducerPreDepthHint learned_hint{};
    PpDepthPredecessorWriterIntent observed_writer{};
    u64 current_backing_generation{};
    u64 current_mutation_epoch{};
    bool source_still_live{};
};

[[nodiscard]] inline bool IsPpUpstreamSampleTeeProducerPreDepthScopeEntryRuntimeEligible(
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryRuntimeEligibilityDescriptor&
        descriptor) noexcept {
    const auto& learned = descriptor.learned_hint;
    const auto& observed = descriptor.observed_writer;
    return descriptor.armed && learned.valid && learned.private_only &&
           !learned.logs_private_state && learned.link.Valid() && learned.writer.Valid() &&
           learned.backing_generation != 0 && observed.Valid() &&
           observed.link == learned.link && observed.range == learned.range &&
           observed.backing_generation == learned.backing_generation &&
           observed.hint == learned.writer && descriptor.current_backing_generation != 0 &&
           descriptor.current_mutation_epoch != 0 &&
           observed.backing_generation == descriptor.current_backing_generation &&
           observed.mutation_epoch == descriptor.current_mutation_epoch &&
           descriptor.source_still_live;
}

struct PpUpstreamSampleTeeProducerPreDepthLearnDescriptor {
    bool enabled{};
    bool sample_tee_enabled{};
    PpUpstreamSampleTeeSelectedDepthToken selected_token{};
    PpDepthPredecessorEndpointStamp selected_endpoint{};
    bool selected_source_still_live{};
    bool selected_identity_exact{};
    bool selected_backing_exact{};
    bool selected_mutation_exact{};
    bool selected_route_exact{};
    bool selected_view_exact{};
    bool selected_depth_aspect{};
    bool selected_depth_write{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthHint
LearnPpUpstreamSampleTeeProducerPreDepthHint(
    const PpUpstreamSampleTeeProducerPreDepthLearnDescriptor& descriptor) noexcept {
    const auto& endpoint = descriptor.selected_endpoint;
    const auto& writer = endpoint.writer;
    const bool exact_draw = writer.hint.Valid();
    const bool exact_token = descriptor.selected_token.Valid() && endpoint.Valid() &&
                             descriptor.selected_token.link == writer.link &&
                             descriptor.selected_token.backing_generation ==
                                 writer.backing_generation &&
                             descriptor.selected_token.mutation_epoch == writer.mutation_epoch;
    if (!descriptor.enabled || !descriptor.sample_tee_enabled || !exact_token || !exact_draw ||
        !descriptor.selected_source_still_live || !descriptor.selected_identity_exact ||
        !descriptor.selected_backing_exact || !descriptor.selected_mutation_exact ||
        !descriptor.selected_route_exact || !descriptor.selected_view_exact ||
        !descriptor.selected_depth_aspect || !descriptor.selected_depth_write) {
        return {};
    }
    return {.link = writer.link,
            .range = writer.range,
            .backing_generation = writer.backing_generation,
            .writer = writer.hint,
            .valid = true,
            .private_only = true};
}

struct PpUpstreamSampleTeeProducerPreDepthHookDescriptor {
    bool enabled{};
    bool sample_tee_enabled{};
    bool capture_upstream_depth_pre_post{};
    bool upstream_feedback_armed{};
    bool pre_draw{};
    PpUpstreamSampleTeeProducerPreDepthHint learned_hint{};
    PpDepthPredecessorWriterIntent observed_writer{};
    bool effective_depth_write{};
    bool actual_depth_aspect{};
    bool canonical_range{};
    bool source_still_live{};
    bool source_identity_exact{};
    bool source_backing_exact{};
    bool source_mutation_exact{};
};

struct PpUpstreamSampleTeeProducerPreDepthHookPlan {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool warmup_required{};
    bool capture_pre{};
    bool observe_metadata_post{};
    bool capture_post_content{};
    bool touch_depth_predecessor_runtime{};
    bool observation_only{};
    bool affects_renderer_behavior{};

    bool operator==(const PpUpstreamSampleTeeProducerPreDepthHookPlan&) const = default;
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthHookPlan
PlanPpUpstreamSampleTeeProducerPreDepthHook(
    const PpUpstreamSampleTeeProducerPreDepthHookDescriptor& descriptor) noexcept {
    if (!descriptor.enabled || !descriptor.sample_tee_enabled ||
        descriptor.capture_upstream_depth_pre_post || !descriptor.upstream_feedback_armed ||
        !descriptor.pre_draw) {
        return {};
    }
    if (!descriptor.learned_hint.valid) {
        return {.status = FinalGuestSurfaceStatus::AlreadyConsumed,
                .warmup_required = true,
                .observation_only = true};
    }
    const auto& observed = descriptor.observed_writer;
    const bool exact = descriptor.learned_hint.private_only &&
                       !descriptor.learned_hint.logs_private_state && observed.Valid() &&
                       observed.link == descriptor.learned_hint.link &&
                       observed.range == descriptor.learned_hint.range &&
                       observed.backing_generation == descriptor.learned_hint.backing_generation &&
                       observed.hint == descriptor.learned_hint.writer &&
                       descriptor.effective_depth_write && descriptor.actual_depth_aspect &&
                       descriptor.canonical_range && descriptor.source_still_live &&
                       descriptor.source_identity_exact && descriptor.source_backing_exact &&
                       descriptor.source_mutation_exact;
    if (!exact) {
        return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                .loss = {.invalidation = 1},
                .observation_only = true};
    }
    return {.status = FinalGuestSurfaceStatus::Complete,
            .capture_pre = true,
            .observe_metadata_post = true,
            .observation_only = true};
}

enum class PpUpstreamSampleTeeProducerPreDepthAdmissionReason : u8 {
    Idle,
    TargetNotSeen,
    ContextOrHintInvalid,
    ResourceNotSeen,
    WriterMismatch,
    SemanticMismatch,
    CurrentnessMismatch,
    Admitted,
    Ambiguous,
};

enum class PpUpstreamSampleTeeProducerPreDepthLearnEvidenceReason : u8 {
    Idle,
    Rejected,
    Learned,
    Ambiguous,
};

struct PpUpstreamSampleTeeProducerPreDepthLearnEvidenceDescriptor {
    bool context_exact{};
    bool token_endpoint_join_exact{};
    bool writer_shape_and_hash_exact{};
    bool live_identity_backing_mutation_exact{};
    bool route_and_view_exact{};
    bool depth_aspect_and_write_exact{};
    bool accepted{};

    bool operator==(
        const PpUpstreamSampleTeeProducerPreDepthLearnEvidenceDescriptor&) const = default;
};

struct PpUpstreamSampleTeeProducerPreDepthPreDrawEvidenceDescriptor {
    bool outer_context_exact{};
    bool learned_hint_private_and_valid{};
    bool observed_writer_valid{};
    bool resource_identity_exact{};
    bool full_writer_shape_and_hash_exact{};
    bool semantic_write_aspect_range_exact{};
    bool live_identity_backing_mutation_exact{};
    bool admitted{};

    bool operator==(
        const PpUpstreamSampleTeeProducerPreDepthPreDrawEvidenceDescriptor&) const = default;
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthLearnEvidenceDescriptor
AdaptPpUpstreamSampleTeeProducerPreDepthLearnEvidence(
    const PpUpstreamSampleTeeProducerPreDepthLearnDescriptor& descriptor,
    const PpUpstreamSampleTeeProducerPreDepthHint& returned) noexcept {
    const auto& endpoint = descriptor.selected_endpoint;
    const auto& writer = endpoint.writer;
    const bool writer_exact = writer.hint.Valid();
    const bool token_endpoint = descriptor.selected_token.Valid() && endpoint.Valid() &&
                                writer.Valid() &&
                                descriptor.selected_token.link == writer.link &&
                                descriptor.selected_token.backing_generation ==
                                    writer.backing_generation &&
                                descriptor.selected_token.mutation_epoch ==
                                    writer.mutation_epoch;
    const bool returned_exact = returned.valid && returned.private_only &&
                                !returned.logs_private_state && returned.link == writer.link &&
                                returned.range == writer.range &&
                                returned.backing_generation == writer.backing_generation &&
                                returned.writer == writer.hint;
    return {
        .context_exact = descriptor.enabled && descriptor.sample_tee_enabled,
        .token_endpoint_join_exact = token_endpoint,
        .writer_shape_and_hash_exact = writer_exact,
        .live_identity_backing_mutation_exact =
            descriptor.selected_source_still_live && descriptor.selected_identity_exact &&
            descriptor.selected_backing_exact && descriptor.selected_mutation_exact,
        .route_and_view_exact =
            descriptor.selected_route_exact && descriptor.selected_view_exact,
        .depth_aspect_and_write_exact =
            descriptor.selected_depth_aspect && descriptor.selected_depth_write,
        .accepted = returned_exact,
    };
}

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthPreDrawEvidenceDescriptor
AdaptPpUpstreamSampleTeeProducerPreDepthPreDrawEvidence(
    const PpUpstreamSampleTeeProducerPreDepthHookDescriptor& descriptor,
    const PpUpstreamSampleTeeProducerPreDepthHookPlan& plan) noexcept {
    const auto& learned = descriptor.learned_hint;
    const auto& observed = descriptor.observed_writer;
    return {
        .outer_context_exact =
            descriptor.enabled && descriptor.sample_tee_enabled &&
            !descriptor.capture_upstream_depth_pre_post && descriptor.upstream_feedback_armed &&
            descriptor.pre_draw,
        .learned_hint_private_and_valid =
            learned.valid && learned.private_only && !learned.logs_private_state,
        .observed_writer_valid = observed.Valid(),
        .resource_identity_exact = observed.link == learned.link &&
                                   observed.backing_generation == learned.backing_generation,
        .full_writer_shape_and_hash_exact = observed.hint == learned.writer,
        .semantic_write_aspect_range_exact =
            observed.range == learned.range && descriptor.effective_depth_write &&
            descriptor.actual_depth_aspect && descriptor.canonical_range,
        .live_identity_backing_mutation_exact =
            descriptor.source_still_live && descriptor.source_identity_exact &&
            descriptor.source_backing_exact && descriptor.source_mutation_exact,
        .admitted = plan.capture_pre,
    };
}

struct PpUpstreamSampleTeeProducerPreDepthLearnEvidence {
    u32 attempts{};
    u32 context_exact{};
    u32 token_endpoint_join_exact{};
    u32 writer_shape_and_hash_exact{};
    u32 live_identity_backing_mutation_exact{};
    u32 route_and_view_exact{};
    u32 depth_aspect_and_write_exact{};
    u32 accepted{};
    PpUpstreamSampleTeeProducerPreDepthLearnEvidenceReason reason{
        PpUpstreamSampleTeeProducerPreDepthLearnEvidenceReason::Idle};

    bool operator==(const PpUpstreamSampleTeeProducerPreDepthLearnEvidence&) const = default;
};

struct PpUpstreamSampleTeeProducerPreDepthAdmissionEvidence {
    u32 observations{};
    u32 outer_context_exact{};
    u32 learned_hint_private_and_valid{};
    u32 observed_writer_valid{};
    u32 resource_identity_exact{};
    u32 full_writer_shape_and_hash_exact{};
    u32 semantic_write_aspect_range_exact{};
    u32 live_identity_backing_mutation_exact{};
    u32 admitted{};
    PpUpstreamSampleTeeProducerPreDepthAdmissionReason reason{
        PpUpstreamSampleTeeProducerPreDepthAdmissionReason::Idle};

    bool operator==(
        const PpUpstreamSampleTeeProducerPreDepthAdmissionEvidence&) const = default;
};

struct PpUpstreamSampleTeeProducerPreDepthGateEvidenceTakeDescriptor {
    u64 sequence{};
    bool prior_hint_valid{};
};

struct PpUpstreamSampleTeeProducerPreDepthGateEvidenceReport {
    u64 sequence{};
    PpUpstreamSampleTeeProducerPreDepthAdmissionEvidence producer{};
    PpUpstreamSampleTeeProducerPreDepthLearnEvidence learn{};
    bool overflow{};
    bool diagnostic_saturated{};
    bool ambiguous{};
    bool retains_private_state{};
    bool affects_capture_behavior{};
    u32 additional_gpu_copy_count{};

    bool operator==(
        const PpUpstreamSampleTeeProducerPreDepthGateEvidenceReport&) const = default;
};

class PpUpstreamSampleTeeProducerPreDepthGateEvidence {
public:
    static constexpr u32 MaxEvents = 255;

    void ObserveLearn(
        const PpUpstreamSampleTeeProducerPreDepthLearnEvidenceDescriptor& descriptor) noexcept {
        Increment(learn.attempts, learn_overflow);
        IncrementIf(learn.context_exact, descriptor.context_exact, learn_overflow);
        IncrementIf(learn.token_endpoint_join_exact, descriptor.token_endpoint_join_exact,
                    learn_overflow);
        IncrementIf(learn.writer_shape_and_hash_exact, descriptor.writer_shape_and_hash_exact,
                    learn_overflow);
        IncrementIf(learn.live_identity_backing_mutation_exact,
                    descriptor.live_identity_backing_mutation_exact, learn_overflow);
        IncrementIf(learn.route_and_view_exact, descriptor.route_and_view_exact,
                    learn_overflow);
        IncrementIf(learn.depth_aspect_and_write_exact,
                    descriptor.depth_aspect_and_write_exact, learn_overflow);
        IncrementIf(learn.accepted, descriptor.accepted, learn_overflow);
        const bool exact = descriptor.context_exact && descriptor.token_endpoint_join_exact &&
                           descriptor.writer_shape_and_hash_exact &&
                           descriptor.live_identity_backing_mutation_exact &&
                           descriptor.route_and_view_exact &&
                           descriptor.depth_aspect_and_write_exact;
        learn_contradictory |= descriptor.accepted != exact;
    }

    void ObserveProducerPreDraw(
        const PpUpstreamSampleTeeProducerPreDepthPreDrawEvidenceDescriptor& descriptor) noexcept {
        Increment(producer.observations, producer_diagnostic_saturated);
        IncrementIf(producer.outer_context_exact, descriptor.outer_context_exact,
                    producer_diagnostic_saturated);
        IncrementIf(producer.learned_hint_private_and_valid,
                    descriptor.learned_hint_private_and_valid, producer_diagnostic_saturated);
        IncrementIf(producer.observed_writer_valid, descriptor.observed_writer_valid,
                    producer_diagnostic_saturated);
        IncrementIf(producer.resource_identity_exact, descriptor.resource_identity_exact,
                    producer_diagnostic_saturated);
        IncrementIf(producer.full_writer_shape_and_hash_exact,
                    descriptor.full_writer_shape_and_hash_exact, producer_diagnostic_saturated);
        IncrementIf(producer.semantic_write_aspect_range_exact,
                    descriptor.semantic_write_aspect_range_exact,
                    producer_diagnostic_saturated);
        IncrementIf(producer.live_identity_backing_mutation_exact,
                    descriptor.live_identity_backing_mutation_exact,
                    producer_diagnostic_saturated);
        IncrementIf(producer.admitted, descriptor.admitted, producer_diagnostic_saturated);
        producer_terminal_overflow |= producer.admitted > 1;
        const bool comparable = descriptor.outer_context_exact &&
                                descriptor.learned_hint_private_and_valid &&
                                descriptor.observed_writer_valid;
        IncrementIf(comparable_observations, comparable, producer_diagnostic_saturated);
        IncrementIf(resource_matches, comparable && descriptor.resource_identity_exact,
                    producer_diagnostic_saturated);
        IncrementIf(writer_matches, comparable && descriptor.resource_identity_exact &&
                                           descriptor.full_writer_shape_and_hash_exact,
                    producer_diagnostic_saturated);
        IncrementIf(semantic_matches,
                    comparable && descriptor.resource_identity_exact &&
                        descriptor.full_writer_shape_and_hash_exact &&
                        descriptor.semantic_write_aspect_range_exact,
                    producer_diagnostic_saturated);
        IncrementIf(current_matches,
                    comparable && descriptor.resource_identity_exact &&
                        descriptor.full_writer_shape_and_hash_exact &&
                        descriptor.semantic_write_aspect_range_exact &&
                        descriptor.live_identity_backing_mutation_exact,
                    producer_diagnostic_saturated);
        const bool exact = comparable && descriptor.resource_identity_exact &&
                           descriptor.full_writer_shape_and_hash_exact &&
                           descriptor.semantic_write_aspect_range_exact &&
                           descriptor.live_identity_backing_mutation_exact;
        producer_contradictory |= descriptor.admitted != exact;
    }

    [[nodiscard]] PpUpstreamSampleTeeProducerPreDepthGateEvidenceReport Take(
        const PpUpstreamSampleTeeProducerPreDepthGateEvidenceTakeDescriptor& descriptor) noexcept {
        const bool producer_ambiguous = producer_terminal_overflow || producer_contradictory ||
                                        producer.admitted > 1 ||
                                        (!descriptor.prior_hint_valid &&
                                         producer.observations != 0);
        if (producer_ambiguous) {
            producer.reason = PpUpstreamSampleTeeProducerPreDepthAdmissionReason::Ambiguous;
        } else if (producer.admitted == 1) {
            producer.reason = PpUpstreamSampleTeeProducerPreDepthAdmissionReason::Admitted;
        } else if (producer.observations == 0) {
            producer.reason = descriptor.prior_hint_valid
                                  ? PpUpstreamSampleTeeProducerPreDepthAdmissionReason::
                                        TargetNotSeen
                                  : PpUpstreamSampleTeeProducerPreDepthAdmissionReason::Idle;
        } else if (comparable_observations == 0) {
            producer.reason =
                PpUpstreamSampleTeeProducerPreDepthAdmissionReason::ContextOrHintInvalid;
        } else if (resource_matches == 0) {
            producer.reason =
                PpUpstreamSampleTeeProducerPreDepthAdmissionReason::ResourceNotSeen;
        } else if (writer_matches == 0) {
            producer.reason =
                PpUpstreamSampleTeeProducerPreDepthAdmissionReason::WriterMismatch;
        } else if (semantic_matches == 0) {
            producer.reason =
                PpUpstreamSampleTeeProducerPreDepthAdmissionReason::SemanticMismatch;
        } else if (current_matches == 0) {
            producer.reason =
                PpUpstreamSampleTeeProducerPreDepthAdmissionReason::CurrentnessMismatch;
        } else {
            producer.reason = PpUpstreamSampleTeeProducerPreDepthAdmissionReason::Ambiguous;
        }

        const bool learn_ambiguous = learn_overflow || learn_contradictory || learn.attempts > 1;
        if (learn_ambiguous) {
            learn.reason = PpUpstreamSampleTeeProducerPreDepthLearnEvidenceReason::Ambiguous;
        } else if (learn.attempts == 0) {
            learn.reason = PpUpstreamSampleTeeProducerPreDepthLearnEvidenceReason::Idle;
        } else if (learn.accepted == 1) {
            learn.reason = PpUpstreamSampleTeeProducerPreDepthLearnEvidenceReason::Learned;
        } else {
            learn.reason = PpUpstreamSampleTeeProducerPreDepthLearnEvidenceReason::Rejected;
        }
        PpUpstreamSampleTeeProducerPreDepthGateEvidenceReport report{
            .sequence = descriptor.sequence,
            .producer = producer,
            .learn = learn,
            .overflow = producer_terminal_overflow || learn_overflow,
            .diagnostic_saturated = producer_diagnostic_saturated,
            .ambiguous = producer.reason ==
                             PpUpstreamSampleTeeProducerPreDepthAdmissionReason::Ambiguous ||
                         learn.reason ==
                             PpUpstreamSampleTeeProducerPreDepthLearnEvidenceReason::Ambiguous,
        };
        *this = {};
        return report;
    }

private:
    static void Increment(u32& value, bool& overflow) noexcept {
        if (value == MaxEvents) {
            overflow = true;
        } else {
            ++value;
        }
    }

    static void IncrementIf(u32& value, bool condition, bool& overflow) noexcept {
        if (condition) {
            Increment(value, overflow);
        }
    }

    PpUpstreamSampleTeeProducerPreDepthAdmissionEvidence producer{};
    PpUpstreamSampleTeeProducerPreDepthLearnEvidence learn{};
    u32 comparable_observations{};
    u32 resource_matches{};
    u32 writer_matches{};
    u32 semantic_matches{};
    u32 current_matches{};
    bool producer_diagnostic_saturated{};
    bool producer_terminal_overflow{};
    bool learn_overflow{};
    bool producer_contradictory{};
    bool learn_contradictory{};
};

[[nodiscard]] inline std::string FormatPpUpstreamSampleTeeProducerPreDepthGateEvidence(
    const PpUpstreamSampleTeeProducerPreDepthGateEvidenceReport& report) {
    const auto& producer = report.producer;
    const auto& learn = report.learn;
    return "FGSCTPDG s=" + std::to_string(report.sequence) +
           " pr=" + std::to_string(static_cast<u32>(producer.reason)) +
           " lr=" + std::to_string(static_cast<u32>(learn.reason)) +
           " pa=" + std::to_string(producer.observations) + "/" +
           std::to_string(producer.outer_context_exact) + "/" +
           std::to_string(producer.learned_hint_private_and_valid) + "/" +
           std::to_string(producer.observed_writer_valid) + "/" +
           std::to_string(producer.resource_identity_exact) + "/" +
           std::to_string(producer.full_writer_shape_and_hash_exact) + "/" +
           std::to_string(producer.semantic_write_aspect_range_exact) + "/" +
           std::to_string(producer.live_identity_backing_mutation_exact) + "/" +
           std::to_string(producer.admitted) +
           " la=" + std::to_string(learn.attempts) + "/" +
           std::to_string(learn.context_exact) + "/" +
           std::to_string(learn.token_endpoint_join_exact) + "/" +
           std::to_string(learn.writer_shape_and_hash_exact) + "/" +
           std::to_string(learn.live_identity_backing_mutation_exact) + "/" +
           std::to_string(learn.route_and_view_exact) + "/" +
           std::to_string(learn.depth_aspect_and_write_exact) + "/" +
           std::to_string(learn.accepted) + " ds=" +
           std::to_string(report.diagnostic_saturated) +
           " ov=" + std::to_string(report.overflow);
}

enum class PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence : u8 {
    Unavailable,
    First,
    Middle,
    Latest,
    None,
    Multiple,
};

inline constexpr u32 PpUpstreamSampleTeeProducerPreDepthMaxOccurrences = 3;

struct PpUpstreamSampleTeeProducerPreDepthOccurrenceClassificationDescriptor {
    std::span<const PpDepthPredecessorEndpointStamp> occurrences{};
    PpDepthPredecessorEndpointStamp selected_endpoint{};
};

struct PpUpstreamSampleTeeProducerPreDepthOccurrenceClassification {
    PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence selected_occurrence{
        PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence::Unavailable};
    u32 match_count{};
    u32 match_index{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthOccurrenceClassification
ClassifyPpUpstreamSampleTeeProducerPreDepthSelectedOccurrence(
    const PpUpstreamSampleTeeProducerPreDepthOccurrenceClassificationDescriptor&
        descriptor) noexcept {
    if (!descriptor.selected_endpoint.Valid()) {
        return {};
    }
    u32 matches{};
    u32 match_index{};
    for (u32 index = 0; index < descriptor.occurrences.size(); ++index) {
        const auto& occurrence = descriptor.occurrences[index];
        if (occurrence.Valid() && occurrence == descriptor.selected_endpoint) {
            match_index = index;
            ++matches;
        }
    }
    if (matches == 0) {
        return {.selected_occurrence =
                    PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence::None};
    }
    if (matches > 1) {
        return {.selected_occurrence =
                    PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence::Multiple,
                .match_count = matches};
    }
    const auto selected = match_index == 0
                              ? PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence::First
                          : match_index == 1
                              ? PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence::Middle
                              : PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence::Latest;
    return {.selected_occurrence = selected, .match_count = 1, .match_index = match_index};
}

struct PpUpstreamSampleTeeProducerPreDepthOccurrencePreObservation {
    PpUpstreamSampleTeeProducerPreDepthHookDescriptor hook_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthHookPlan hook_plan{};
    u64 guest_rendering_serial{};
    u64 host_rendering_instance_id{};
    bool immediately_before_draw{};
    PpDepthPredecessorEndpointStamp private_immediate_predecessor_endpoint{};
    PpUpstreamDepthFirstWriterPreObservation first_writer_pre{};
};

struct PpUpstreamSampleTeeProducerPreDepthOccurrencePostObservation {
    PpDepthPredecessorEndpointStamp endpoint{};
    u64 live_guest_rendering_serial{};
    u64 live_host_rendering_instance_id{};
    bool immediately_after_mark{};
};

struct PpUpstreamSampleTeeProducerPreDepthOccurrenceSelectedObservation {
    PpUpstreamSampleTeeProducerPreDepthLearnDescriptor learn_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthHint learned_hint{};
};

struct PpUpstreamSampleTeeProducerPreDepthOccurrenceTakeDescriptor {
    u64 sequence{};
};

struct PpUpstreamSampleTeeProducerPreDepthOccurrenceReport {
    u64 sequence{};
    u32 total_exact_occurrences{};
    u32 selected_observation_count{};
    PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence selected_occurrence{
        PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence::Unavailable};
    u32 selected_match_count{};
    u32 occurrences_after_consumer{};
    u32 selected_matches_after_consumer{};
    u32 intervening_mutation_drift{};
    u32 resource_drift{};
    u32 currentness_drift{};
    u32 malformed_plan_drift{};
    u32 semantic_drift{};
    u32 endpoint_drift{};
    u32 ordering_drift{};
    u32 unrelated_writer_misses{};
    u32 diagnostic_rejections{};
    u32 terminal_contradictions{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool overflow{};
    bool diagnostic_saturated{};
    bool selected_endpoint_frozen{};
    bool metadata_only{true};
    bool retains_private_state{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_release_count{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
    u32 paired_predecessor_count{};
    u32 pair_predecessor_exact_occurrence_count{};
    u32 pair_first_writer_exact_occurrence_count{};
    u32 selected_predecessor_match_count{};
    u32 learned_predecessor_valid_count{};
    enum class SelectedPredecessorOrdinalStatus : u8 {
        Unavailable,
        Latest,
        NotLatest,
        Invalid,
        Multiple,
    } selected_predecessor_ordinal_status{SelectedPredecessorOrdinalStatus::Unavailable};

    bool operator==(
        const PpUpstreamSampleTeeProducerPreDepthOccurrenceReport&) const = default;
};

using PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus =
    PpUpstreamSampleTeeProducerPreDepthOccurrenceReport::SelectedPredecessorOrdinalStatus;

struct PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorPair {
    PpDepthPredecessorEndpointStamp predecessor_endpoint{};
    PpDepthPredecessorWriterIntent target_pre_intent{};
    PpDepthPredecessorEndpointStamp target_post_endpoint{};
    PpUpstreamSampleTeePredecessorWriterPreDepthHint learned_hint{};
    u32 selected_target_scope_writer_count{};
    u32 selected_target_occurrence_index{};
    u32 completed_occurrence_count{};
    u32 selected_pairable_occurrence_index{};
    u32 completed_pairable_occurrence_count{};
    u32 expected_predecessor_pre_scope_writer_count{};
    bool valid{};
    bool private_only{};
    bool logs_private_state{};
};

struct PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm {
    PpUpstreamSampleTeePredecessorWriterPreDepthHint predecessor_hint{};
    PpUpstreamSampleTeeProducerPreDepthHint target_hint{};
    u32 predecessor_scope_writer_count{};
    u32 target_scope_writer_count{};
    u32 target_occurrence_index{};
    u32 completed_occurrence_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool valid{};
    bool private_only{};
    bool logs_private_state{};
    bool retains_private_state{};
    bool selected_producer_occurrence{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};

    bool operator==(
        const PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm&) const = default;
};

struct PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArmDescriptor {
    bool enabled{};
    bool sample_tee_enabled{};
    PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorPair prior_selected_pair{};
    u32 inactive_flip_count{};
};

[[nodiscard]] inline bool IsExactPpUpstreamSampleTeeProducerPreDepthSelectedPair(
    const PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorPair& pair) noexcept {
    const auto& predecessor = pair.predecessor_endpoint.writer;
    const auto& target_pre = pair.target_pre_intent;
    const auto& target_post = pair.target_post_endpoint.writer;
    const auto& learned = pair.learned_hint;
    return pair.valid && pair.private_only && !pair.logs_private_state &&
           pair.predecessor_endpoint.Valid() && predecessor.hint.Valid() &&
           predecessor.hint.fragment_pgm_hash != 0 && target_pre.Valid() &&
           pair.target_post_endpoint.Valid() && target_post.hint.Valid() && learned.valid &&
           learned.private_only && !learned.logs_private_state && learned.writer.Valid() &&
           learned.writer == predecessor.hint && learned.link == predecessor.link &&
           learned.range == predecessor.range &&
           learned.backing_generation == predecessor.backing_generation &&
           target_pre.hint == target_post.hint && predecessor.link == target_pre.link &&
           predecessor.range == target_pre.range &&
           predecessor.backing_generation == target_pre.backing_generation &&
           target_pre.link == target_post.link && target_pre.range == target_post.range &&
           target_pre.backing_generation == target_post.backing_generation &&
           predecessor.mutation_epoch == target_pre.mutation_epoch &&
           target_pre.mutation_epoch != std::numeric_limits<u64>::max() &&
           target_post.mutation_epoch == target_pre.mutation_epoch + 1 &&
           pair.selected_target_scope_writer_count > 1 &&
           pair.expected_predecessor_pre_scope_writer_count ==
               pair.selected_target_scope_writer_count - 1 &&
           pair.completed_occurrence_count != 0 &&
           pair.selected_target_occurrence_index < pair.completed_occurrence_count &&
           pair.completed_pairable_occurrence_count != 0 &&
           pair.selected_pairable_occurrence_index <
               pair.completed_pairable_occurrence_count &&
           pair.selected_pairable_occurrence_index + 1 ==
               pair.completed_pairable_occurrence_count &&
           pair.completed_pairable_occurrence_count <= pair.completed_occurrence_count &&
           pair.selected_pairable_occurrence_index <=
               pair.selected_target_occurrence_index;
}

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm
PlanPpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm(
    const PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArmDescriptor& descriptor) noexcept {
    if (!descriptor.enabled || !descriptor.sample_tee_enabled) {
        return {};
    }
    if (descriptor.inactive_flip_count > 1 ||
        !IsExactPpUpstreamSampleTeeProducerPreDepthSelectedPair(
            descriptor.prior_selected_pair)) {
        return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                .loss = {.invalidation = 1}};
    }
    const auto& pair = descriptor.prior_selected_pair;
    return {
        .predecessor_hint = pair.learned_hint,
        .target_hint = {.link = pair.target_pre_intent.link,
                        .range = pair.target_pre_intent.range,
                        .backing_generation = pair.target_pre_intent.backing_generation,
                        .writer = pair.target_pre_intent.hint,
                        .valid = true,
                        .private_only = true},
        .predecessor_scope_writer_count = pair.expected_predecessor_pre_scope_writer_count,
        .target_scope_writer_count = pair.selected_target_scope_writer_count,
        .status = FinalGuestSurfaceStatus::Complete,
        .valid = true,
        .private_only = true,
        .retains_private_state = true,
    };
}

struct PpUpstreamSampleTeeProducerPreDepthSelectedProducerOrdinalArmDescriptor {
    PpUpstreamSampleTeeProducerPreDepthHint learned_hint{};
    PpDepthPredecessorWriterIntent target_pre{};
    PpDepthPredecessorEndpointStamp target_post{};
    u32 target_occurrence_index{};
    u32 completed_occurrence_count{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm
PlanPpUpstreamSampleTeeProducerPreDepthSelectedProducerOrdinalArm(
    const PpUpstreamSampleTeeProducerPreDepthSelectedProducerOrdinalArmDescriptor&
        descriptor) noexcept {
    const auto& learned = descriptor.learned_hint;
    const auto& pre = descriptor.target_pre;
    const auto& post = descriptor.target_post;
    const bool exact = learned.valid && learned.private_only && !learned.logs_private_state &&
                       learned.link.Valid() && learned.writer.Valid() && pre.Valid() &&
                       post.Valid() && descriptor.completed_occurrence_count != 0 &&
                       descriptor.completed_occurrence_count <=
                           PpUpstreamSampleTeeProducerPreDepthMaxOccurrences &&
                       descriptor.target_occurrence_index <
                           descriptor.completed_occurrence_count &&
                       descriptor.target_occurrence_index + 1 ==
                           descriptor.completed_occurrence_count &&
                       pre.link == learned.link && pre.range == learned.range &&
                       pre.backing_generation == learned.backing_generation &&
                       pre.hint == learned.writer && post.writer.link == pre.link &&
                       post.writer.range == pre.range &&
                       post.writer.backing_generation == pre.backing_generation &&
                       post.writer.hint == pre.hint &&
                       pre.mutation_epoch != std::numeric_limits<u64>::max() &&
                       post.writer.mutation_epoch == pre.mutation_epoch + 1;
    if (!exact) {
        return {};
    }
    return {.target_hint = learned,
            .target_occurrence_index = descriptor.target_occurrence_index,
            .completed_occurrence_count = descriptor.completed_occurrence_count,
            .status = FinalGuestSurfaceStatus::Complete,
            .valid = true,
            .private_only = true,
            .retains_private_state = true,
            .selected_producer_occurrence = true};
}

struct PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan {
    u32 scope_writer_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool capture_pre{};
    bool run_collision_check{};
    bool arm_generic_predecessor_observation{};
    bool arm_generic_target_observation{};
    bool wrong_ordinal_ignored{};
    bool unrelated_writer_ignored{};
    bool context_ignored{};
    bool terminal_contradiction{};
};

struct PpUpstreamSampleTeeProducerPreDepthOrdinalCaptureDescriptor {
    PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm arm{};
    PpUpstreamSampleTeeProducerPreDepthHookDescriptor hook_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthHookPlan hook_plan{};
    PpUpstreamDepthFirstWriterPreObservation first_writer_pre{};
};

enum class PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingStage : u8 {
    SnapshotPredecessorAndFirstWriter,
    GenericHook,
    GenericEvidence,
    OccurrencePre,
    PredecessorOrdinalGateAndCopy,
    TargetOrdinalGateAndCopy,
    Draw,
    Mark,
    OccurrencePost,
};

struct PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingDescriptor {
    std::array<PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingStage, 9> stages{};
    u64 pre_host_rendering_instance_id{};
    u64 post_host_rendering_instance_id{};
    bool predecessor_ordinal_copy{};
    bool target_ordinal_copy{};
};

struct PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingPlan {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    u32 existing_ordinal_copy_count{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_release_count{};
    bool exact{};
    bool snapshots_before_diagnostics{};
    bool occurrence_pre_before_predecessor_ordinal_gate{};
    bool predecessor_ordinal_gate_before_target_ordinal_gate{};
    bool occurrence_post_after_mark{};
    bool at_most_one_ordinal_copy{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingPlan
PlanPpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrdering(
    const PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingDescriptor&
        descriptor) noexcept {
    using Stage = PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingStage;
    constexpr std::array CanonicalStages{
        Stage::SnapshotPredecessorAndFirstWriter,
        Stage::GenericHook,
        Stage::GenericEvidence,
        Stage::OccurrencePre,
        Stage::PredecessorOrdinalGateAndCopy,
        Stage::TargetOrdinalGateAndCopy,
        Stage::Draw,
        Stage::Mark,
        Stage::OccurrencePost,
    };
    const u32 copy_count = static_cast<u32>(descriptor.predecessor_ordinal_copy) +
                           static_cast<u32>(descriptor.target_ordinal_copy);
    const bool at_most_one_copy = copy_count <= 1;
    const bool host_contiguous =
        descriptor.pre_host_rendering_instance_id != 0 &&
        descriptor.post_host_rendering_instance_id != 0 &&
        (descriptor.post_host_rendering_instance_id ==
             descriptor.pre_host_rendering_instance_id ||
         (descriptor.pre_host_rendering_instance_id != std::numeric_limits<u64>::max() &&
          descriptor.post_host_rendering_instance_id ==
              descriptor.pre_host_rendering_instance_id + 1));
    const bool exact = descriptor.stages == CanonicalStages && at_most_one_copy && host_contiguous;
    return {
        .status = exact ? FinalGuestSurfaceStatus::Complete
                        : FinalGuestSurfaceStatus::InvalidationLoss,
        .loss = exact ? FinalGuestSurfaceLoss{} : FinalGuestSurfaceLoss{.invalidation = 1},
        .existing_ordinal_copy_count = copy_count,
        .exact = exact,
        .snapshots_before_diagnostics = descriptor.stages == CanonicalStages,
        .occurrence_pre_before_predecessor_ordinal_gate =
            descriptor.stages == CanonicalStages,
        .predecessor_ordinal_gate_before_target_ordinal_gate =
            descriptor.stages == CanonicalStages,
        .occurrence_post_after_mark = descriptor.stages == CanonicalStages,
        .at_most_one_ordinal_copy = at_most_one_copy,
    };
}

struct PpUpstreamSampleTeePredecessorWriterPreDepthOrdinalCaptureDescriptor {
    PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm arm{};
    PpUpstreamSampleTeePredecessorWriterPreDepthHookDescriptor hook_descriptor{};
    PpUpstreamSampleTeePredecessorWriterPreDepthHookPlan hook_plan{};
    PpUpstreamDepthFirstWriterPreObservation first_writer_pre{};
};

[[nodiscard]] inline bool IsExactPpUpstreamSampleTeeOrdinalArmHint(
    const PpUpstreamSampleTeeProducerPreDepthHint& hint,
    const PpDepthPredecessorWriterIntent& writer) noexcept {
    return hint.valid && hint.private_only && !hint.logs_private_state && hint.writer.Valid() &&
           writer.Valid() && writer.hint == hint.writer && writer.link == hint.link &&
           writer.range == hint.range && writer.backing_generation == hint.backing_generation;
}

[[nodiscard]] inline bool IsValidPpUpstreamSampleTeeOrdinalHistory(
    const PpUpstreamDepthFirstWriterPreObservation& history) noexcept {
    return history.recorded && history.depth_stencil_exact && history.same_scope &&
           history.scope_depth_history_valid && history.first_depth_writer_valid &&
           history.depth_writer_count != 0 && !history.depth_writer_count_overflow;
}

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan
PlanPpUpstreamSampleTeeProducerPreDepthOrdinalCapture(
    const PpUpstreamSampleTeeProducerPreDepthOrdinalCaptureDescriptor& descriptor) noexcept {
    const auto reject = [](bool contradiction = false) {
        return PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
            .terminal_contradiction = contradiction};
    };
    const auto& arm = descriptor.arm;
    const auto& hook = descriptor.hook_descriptor;
    const bool context = hook.enabled && hook.sample_tee_enabled &&
                         !hook.capture_upstream_depth_pre_post && hook.pre_draw;
    if (!context) {
        return {.context_ignored = true};
    }
    if (!arm.valid || !arm.private_only || arm.logs_private_state ||
        arm.status != FinalGuestSurfaceStatus::Complete || arm.loss.Any() ||
        arm.target_scope_writer_count == 0) {
        return reject();
    }
    if (descriptor.first_writer_pre.depth_writer_count == 0 &&
        descriptor.first_writer_pre.first_depth_writer_valid) {
        return reject();
    }
    if (descriptor.first_writer_pre.depth_writer_count != arm.target_scope_writer_count) {
        return {.scope_writer_count = descriptor.first_writer_pre.depth_writer_count,
                .wrong_ordinal_ignored = true};
    }
    if (!IsValidPpUpstreamSampleTeeOrdinalHistory(descriptor.first_writer_pre)) {
        return reject();
    }
    const bool writer_identity_exact = hook.observed_writer.hint == arm.target_hint.writer &&
                                       hook.observed_writer.link == arm.target_hint.link &&
                                       hook.observed_writer.range == arm.target_hint.range &&
                                       hook.observed_writer.backing_generation ==
                                           arm.target_hint.backing_generation;
    if (!writer_identity_exact) {
        return {.scope_writer_count = descriptor.first_writer_pre.depth_writer_count,
                .unrelated_writer_ignored = true};
    }
    if (!hook.observed_writer.Valid() ||
        !IsExactPpUpstreamSampleTeeOrdinalArmHint(arm.target_hint,
                                                  hook.observed_writer)) {
        return reject();
    }
    if (!hook.effective_depth_write || !hook.actual_depth_aspect || !hook.canonical_range ||
        !hook.source_still_live || !hook.source_identity_exact ||
        !hook.source_backing_exact || !hook.source_mutation_exact) {
        return reject();
    }
    const auto canonical = PlanPpUpstreamSampleTeeProducerPreDepthHook(hook);
    if (descriptor.hook_plan != canonical) {
        return reject(true);
    }
    return {.scope_writer_count = descriptor.first_writer_pre.depth_writer_count,
            .status = FinalGuestSurfaceStatus::Complete,
            .capture_pre = true,
            .run_collision_check = true,
             .arm_generic_target_observation = !descriptor.hook_plan.capture_pre};
}

struct PpUpstreamSampleTeeProducerPreDepthSelectedProducerOccurrenceCaptureDescriptor {
    PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm arm{};
    PpUpstreamSampleTeeProducerPreDepthHookDescriptor hook_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthHookPlan hook_plan{};
    u32 observed_occurrence_index{};
};

[[nodiscard]] inline bool IsPpUpstreamSampleTeeProducerPreDepthSelectedProducerOccurrenceMatch(
    const PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm& arm,
    const PpUpstreamSampleTeeProducerPreDepthHookDescriptor& hook) noexcept {
    return arm.valid && arm.private_only && !arm.logs_private_state &&
           arm.retains_private_state && arm.selected_producer_occurrence &&
           arm.status == FinalGuestSurfaceStatus::Complete && !arm.loss.Any() &&
           !arm.predecessor_hint.valid && arm.predecessor_scope_writer_count == 0 &&
           arm.target_scope_writer_count == 0 && arm.target_hint.valid &&
           arm.target_hint.private_only && !arm.target_hint.logs_private_state &&
           arm.target_hint.writer.Valid() && arm.completed_occurrence_count != 0 &&
           arm.completed_occurrence_count <=
               PpUpstreamSampleTeeProducerPreDepthMaxOccurrences &&
           arm.target_occurrence_index < arm.completed_occurrence_count &&
           arm.target_occurrence_index + 1 == arm.completed_occurrence_count &&
           hook.enabled && hook.sample_tee_enabled && !hook.capture_upstream_depth_pre_post &&
           hook.pre_draw && hook.observed_writer.Valid() &&
           IsExactPpUpstreamSampleTeeOrdinalArmHint(arm.target_hint, hook.observed_writer);
}

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan
PlanPpUpstreamSampleTeeProducerPreDepthSelectedProducerOccurrenceCapture(
    const PpUpstreamSampleTeeProducerPreDepthSelectedProducerOccurrenceCaptureDescriptor&
        descriptor) noexcept {
    const auto reject = [](bool contradiction = false) {
        return PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
            .terminal_contradiction = contradiction};
    };
    const auto& arm = descriptor.arm;
    const auto& hook = descriptor.hook_descriptor;
    const bool context = hook.enabled && hook.sample_tee_enabled &&
                         !hook.capture_upstream_depth_pre_post && hook.pre_draw;
    if (!context) {
        return {.context_ignored = true};
    }
    if (!arm.valid || !arm.private_only || arm.logs_private_state ||
        !arm.retains_private_state || !arm.selected_producer_occurrence ||
        arm.status != FinalGuestSurfaceStatus::Complete || arm.loss.Any() ||
        arm.predecessor_hint.valid || arm.predecessor_scope_writer_count != 0 ||
        arm.target_scope_writer_count != 0 || arm.completed_occurrence_count == 0 ||
        arm.completed_occurrence_count > PpUpstreamSampleTeeProducerPreDepthMaxOccurrences ||
        arm.target_occurrence_index >= arm.completed_occurrence_count ||
        arm.target_occurrence_index + 1 != arm.completed_occurrence_count) {
        return reject();
    }
    if (!IsPpUpstreamSampleTeeProducerPreDepthSelectedProducerOccurrenceMatch(arm, hook)) {
        return {.unrelated_writer_ignored = true};
    }
    if (descriptor.observed_occurrence_index != arm.target_occurrence_index) {
        return {.scope_writer_count = descriptor.observed_occurrence_index,
                .wrong_ordinal_ignored = true};
    }
    if (!hook.effective_depth_write || !hook.actual_depth_aspect || !hook.canonical_range ||
        !hook.source_still_live || !hook.source_identity_exact ||
        !hook.source_backing_exact || !hook.source_mutation_exact) {
        return reject();
    }
    const auto canonical = PlanPpUpstreamSampleTeeProducerPreDepthHook(hook);
    if (descriptor.hook_plan != canonical || !canonical.capture_pre) {
        return reject(true);
    }
    return {.scope_writer_count = descriptor.observed_occurrence_index,
            .status = FinalGuestSurfaceStatus::Complete,
            .capture_pre = true,
            .run_collision_check = true};
}

[[nodiscard]] inline bool
IsCompletePpUpstreamSampleTeeProducerPreDepthSelectedProducerOccurrenceFrame(
    const PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm& arm,
    u32 observed_occurrence_count) noexcept {
    return arm.valid && arm.private_only && !arm.logs_private_state &&
           arm.retains_private_state && arm.selected_producer_occurrence &&
           arm.status == FinalGuestSurfaceStatus::Complete && !arm.loss.Any() &&
           arm.completed_occurrence_count != 0 &&
           arm.completed_occurrence_count <=
               PpUpstreamSampleTeeProducerPreDepthMaxOccurrences &&
           arm.target_occurrence_index + 1 == arm.completed_occurrence_count &&
           observed_occurrence_count == arm.completed_occurrence_count;
}

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan
PlanPpUpstreamSampleTeePredecessorWriterPreDepthOrdinalCapture(
    const PpUpstreamSampleTeePredecessorWriterPreDepthOrdinalCaptureDescriptor&
        descriptor) noexcept {
    const auto reject = [](bool contradiction = false) {
        return PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
            .terminal_contradiction = contradiction};
    };
    const auto& arm = descriptor.arm;
    const auto& hook = descriptor.hook_descriptor;
    const bool context = hook.enabled && hook.sample_tee_enabled &&
                         !hook.capture_upstream_depth_pre_post && hook.pre_draw;
    if (!context) {
        return {.context_ignored = true};
    }
    if (!arm.valid || !arm.private_only || arm.logs_private_state ||
        arm.status != FinalGuestSurfaceStatus::Complete || arm.loss.Any() ||
        arm.predecessor_scope_writer_count == 0) {
        return reject();
    }
    if (descriptor.first_writer_pre.depth_writer_count == 0 &&
        descriptor.first_writer_pre.first_depth_writer_valid) {
        return reject();
    }
    if (descriptor.first_writer_pre.depth_writer_count !=
        arm.predecessor_scope_writer_count) {
        return {.scope_writer_count = descriptor.first_writer_pre.depth_writer_count,
                .wrong_ordinal_ignored = true};
    }
    if (!IsValidPpUpstreamSampleTeeOrdinalHistory(descriptor.first_writer_pre)) {
        return reject();
    }
    const bool writer_identity_exact =
        hook.observed_writer.hint == arm.predecessor_hint.writer &&
        hook.observed_writer.link == arm.predecessor_hint.link &&
        hook.observed_writer.range == arm.predecessor_hint.range &&
        hook.observed_writer.backing_generation ==
            arm.predecessor_hint.backing_generation;
    if (!writer_identity_exact) {
        return {.scope_writer_count = descriptor.first_writer_pre.depth_writer_count,
                .unrelated_writer_ignored = true};
    }
    if (!hook.observed_writer.Valid() ||
        !IsExactPpUpstreamSampleTeeOrdinalArmHint(arm.predecessor_hint,
                                                  hook.observed_writer)) {
        return reject();
    }
    if (!hook.effective_depth_write || !hook.actual_depth_aspect || !hook.canonical_range ||
        !hook.source_still_live || !hook.source_identity_exact ||
        !hook.source_backing_exact || !hook.source_mutation_exact) {
        return reject();
    }
    const auto canonical = PlanPpUpstreamSampleTeePredecessorWriterPreDepthHook(hook);
    const bool canonical_plan =
        descriptor.hook_plan.status == canonical.status &&
        descriptor.hook_plan.loss == canonical.loss &&
        descriptor.hook_plan.capture_pre == canonical.capture_pre &&
        descriptor.hook_plan.acquire_existing_outer_slot_early ==
            canonical.acquire_existing_outer_slot_early &&
        descriptor.hook_plan.requires_current_target_arm ==
            canonical.requires_current_target_arm &&
        descriptor.hook_plan.requires_current_target_observation ==
            canonical.requires_current_target_observation &&
        descriptor.hook_plan.owns_second_token == canonical.owns_second_token &&
        descriptor.hook_plan.observation_only == canonical.observation_only &&
        descriptor.hook_plan.affects_renderer_behavior == canonical.affects_renderer_behavior;
    if (!canonical_plan) {
        return reject(true);
    }
    return {.scope_writer_count = descriptor.first_writer_pre.depth_writer_count,
            .status = FinalGuestSurfaceStatus::Complete,
            .capture_pre = true,
            .run_collision_check = true,
            .arm_generic_predecessor_observation = !descriptor.hook_plan.capture_pre};
}

enum class PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason : u8 {
    Idle,
    ExactRouteAbsent,
    ExactArmedGenericNonAdmission,
    MissingOrDuplicate,
    Invalid,
};

enum class PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureFirstFailure : u8 {
    Idle,
    GenericEnabled,
    GenericSampleTee,
    GenericCaptureMode,
    GenericPreDraw,
    GenericRouteArmed,
    CrossEnabled,
    CrossSampleTee,
    CrossCaptureMode,
    CrossRouteArmed,
    CrossPreDraw,
    CrossWriterHint,
    CrossWriterLink,
    CrossWriterRange,
    CrossWriterBacking,
    CrossWriterMutation,
    CrossWriterValidity,
    CrossEffectiveDepthWrite,
    CrossDepthAspect,
    CrossCanonicalRange,
    CrossSourceLive,
    CrossSourceIdentity,
    CrossSourceBacking,
    CrossSourceMutation,
    CurrentTargetArmed,
    CurrentTargetObserved,
    GenericPlanContradiction,
    GenericPlanAdmitted,
    OrdinalPlanMismatch,
    OrdinalNotAdmitted,
    OrderingPlanMismatch,
    OrderingNotExact,
    WrongCopyRole,
    PredecessorPostMissing,
    HostNotContiguous,
    Exact,
};

struct PpUpstreamSampleTeeProducerPreDepthRouteAbsentPredecessorOrdinalObservation {
    PpUpstreamSampleTeeProducerPreDepthHookDescriptor generic_hook_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthHookPlan generic_hook_plan{};
    PpUpstreamSampleTeePredecessorWriterPreDepthOrdinalCaptureDescriptor
        ordinal_capture_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan ordinal_capture_plan{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingDescriptor ordering_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingPlan ordering_plan{};
};

struct PpUpstreamSampleTeeProducerPreDepthRouteAbsentTargetOrdinalObservation {
    PpUpstreamSampleTeeProducerPreDepthHookDescriptor generic_hook_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthHookPlan generic_hook_plan{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalCaptureDescriptor ordinal_capture_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan ordinal_capture_plan{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingDescriptor ordering_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingPlan ordering_plan{};
    u64 predecessor_post_host_rendering_instance_id{};
};

struct PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureTakeDescriptor {
    u64 sequence{};
};

struct PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReport {
    u64 sequence{};
    PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason reason{
        PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason::Idle};
    u32 predecessor_observation_count{};
    u32 target_observation_count{};
    u32 generic_route_unarmed_count{};
    u32 generic_route_armed_count{};
    u32 generic_rejection_count{};
    u32 generic_warmup_count{};
    u32 generic_invalidation_rejection_count{};
    u32 ordinal_admission_count{};
    u32 ordering_exact_count{};
    PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureFirstFailure
        predecessor_first_failure{};
    PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureFirstFailure
        target_first_failure{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool diagnostic_saturated{};
    bool overflow{};
    bool metadata_only{true};
    bool retains_private_state{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
};

class PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureEvidence {
public:
    static constexpr u32 MaxEvents = 255;

    void ObservePredecessor(
        const PpUpstreamSampleTeeProducerPreDepthRouteAbsentPredecessorOrdinalObservation&
            observation) noexcept {
        Increment(predecessor_observation_count);
        if (predecessor_observation_count != 1) {
            RecordGap();
            return;
        }
        using Failure =
            PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureFirstFailure;
        const auto fail = [&](Failure failure) {
            predecessor_first_failure = failure;
            RecordInvalidation();
        };
        const auto& generic = observation.generic_hook_descriptor;
        const auto& ordinal = observation.ordinal_capture_descriptor;
        const auto expected_generic = PlanPpUpstreamSampleTeeProducerPreDepthHook(generic);
        const auto expected_ordinal =
            PlanPpUpstreamSampleTeePredecessorWriterPreDepthOrdinalCapture(ordinal);
        const auto expected_order =
            PlanPpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrdering(
                observation.ordering_descriptor);
        const auto& ordinal_hook = ordinal.hook_descriptor;
        if (!generic.enabled) {
            fail(Failure::GenericEnabled);
            return;
        }
        if (!generic.sample_tee_enabled) {
            fail(Failure::GenericSampleTee);
            return;
        }
        if (generic.capture_upstream_depth_pre_post) {
            fail(Failure::GenericCaptureMode);
            return;
        }
        if (!generic.pre_draw) {
            fail(Failure::GenericPreDraw);
            return;
        }
        if (generic.enabled != ordinal_hook.enabled) {
            fail(Failure::CrossEnabled);
            return;
        }
        if (generic.sample_tee_enabled != ordinal_hook.sample_tee_enabled) {
            fail(Failure::CrossSampleTee);
            return;
        }
        if (generic.capture_upstream_depth_pre_post !=
            ordinal_hook.capture_upstream_depth_pre_post) {
            fail(Failure::CrossCaptureMode);
            return;
        }
        if (generic.upstream_feedback_armed != ordinal_hook.current_target_armed) {
            fail(Failure::CrossRouteArmed);
            return;
        }
        if (generic.pre_draw != ordinal_hook.pre_draw) {
            fail(Failure::CrossPreDraw);
            return;
        }
        if (generic.observed_writer.hint != ordinal_hook.observed_writer.hint) {
            fail(Failure::CrossWriterHint);
            return;
        }
        if (generic.observed_writer.link != ordinal_hook.observed_writer.link) {
            fail(Failure::CrossWriterLink);
            return;
        }
        if (generic.observed_writer.range != ordinal_hook.observed_writer.range) {
            fail(Failure::CrossWriterRange);
            return;
        }
        if (generic.observed_writer.backing_generation !=
            ordinal_hook.observed_writer.backing_generation) {
            fail(Failure::CrossWriterBacking);
            return;
        }
        if (generic.observed_writer.mutation_epoch !=
            ordinal_hook.observed_writer.mutation_epoch) {
            fail(Failure::CrossWriterMutation);
            return;
        }
        if (generic.observed_writer.valid != ordinal_hook.observed_writer.valid) {
            fail(Failure::CrossWriterValidity);
            return;
        }
        if (generic.effective_depth_write != ordinal_hook.effective_depth_write) {
            fail(Failure::CrossEffectiveDepthWrite);
            return;
        }
        if (generic.actual_depth_aspect != ordinal_hook.actual_depth_aspect) {
            fail(Failure::CrossDepthAspect);
            return;
        }
        if (generic.canonical_range != ordinal_hook.canonical_range) {
            fail(Failure::CrossCanonicalRange);
            return;
        }
        if (generic.source_still_live != ordinal_hook.source_still_live) {
            fail(Failure::CrossSourceLive);
            return;
        }
        if (generic.source_identity_exact != ordinal_hook.source_identity_exact) {
            fail(Failure::CrossSourceIdentity);
            return;
        }
        if (generic.source_backing_exact != ordinal_hook.source_backing_exact) {
            fail(Failure::CrossSourceBacking);
            return;
        }
        if (generic.source_mutation_exact != ordinal_hook.source_mutation_exact) {
            fail(Failure::CrossSourceMutation);
            return;
        }
        if (ordinal_hook.current_target_observed) {
            fail(Failure::CurrentTargetObserved);
            return;
        }
        if (generic.upstream_feedback_armed) {
            Increment(generic_route_armed_count);
        } else {
            Increment(generic_route_unarmed_count);
        }
        if (observation.generic_hook_plan != expected_generic) {
            fail(Failure::GenericPlanContradiction);
            return;
        }
        if (generic.upstream_feedback_armed) {
            if (expected_generic.capture_pre) {
                fail(Failure::GenericPlanAdmitted);
                return;
            }
            if (expected_generic.warmup_required) {
                Increment(generic_warmup_count);
            } else if (expected_generic.status == FinalGuestSurfaceStatus::InvalidationLoss &&
                       expected_generic.loss == FinalGuestSurfaceLoss{.invalidation = 1} &&
                       expected_generic.observation_only) {
                Increment(generic_invalidation_rejection_count);
            } else {
                fail(Failure::GenericPlanContradiction);
                return;
            }
        } else if (expected_generic != PpUpstreamSampleTeeProducerPreDepthHookPlan{}) {
            fail(Failure::GenericPlanContradiction);
            return;
        }
        Increment(generic_rejection_count);
        if (!IsMatchingOrdinalCapturePlan(observation.ordinal_capture_plan,
                                          expected_ordinal)) {
            fail(Failure::OrdinalPlanMismatch);
            return;
        }
        if (!IsAdmittedOrdinalCapturePlan(expected_ordinal, true)) {
            fail(Failure::OrdinalNotAdmitted);
            return;
        }
        Increment(ordinal_admission_count);
        if (!IsExactOrderingPlan(observation.ordering_plan, expected_order)) {
            fail(Failure::OrderingPlanMismatch);
            return;
        }
        if (!expected_order.exact) {
            fail(Failure::OrderingNotExact);
            return;
        }
        if (!observation.ordering_descriptor.predecessor_ordinal_copy ||
            observation.ordering_descriptor.target_ordinal_copy) {
            fail(Failure::WrongCopyRole);
            return;
        }
        Increment(ordering_exact_count);
        predecessor_first_failure = Failure::Exact;
    }

    void ObserveTarget(
        const PpUpstreamSampleTeeProducerPreDepthRouteAbsentTargetOrdinalObservation&
            observation) noexcept {
        Increment(target_observation_count);
        if (target_observation_count != 1) {
            RecordGap();
            return;
        }
        if (predecessor_observation_count == 0) {
            RecordGap();
            return;
        }
        using Failure =
            PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureFirstFailure;
        const auto fail = [&](Failure failure) {
            target_first_failure = failure;
            RecordInvalidation();
        };
        const auto& generic = observation.generic_hook_descriptor;
        const auto& ordinal = observation.ordinal_capture_descriptor;
        const auto expected_generic = PlanPpUpstreamSampleTeeProducerPreDepthHook(generic);
        const auto expected_ordinal =
            PlanPpUpstreamSampleTeeProducerPreDepthOrdinalCapture(ordinal);
        const auto expected_order =
            PlanPpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrdering(
                observation.ordering_descriptor);
        const auto& ordinal_hook = ordinal.hook_descriptor;
        if (!generic.enabled) {
            fail(Failure::GenericEnabled);
            return;
        }
        if (!generic.sample_tee_enabled) {
            fail(Failure::GenericSampleTee);
            return;
        }
        if (generic.capture_upstream_depth_pre_post) {
            fail(Failure::GenericCaptureMode);
            return;
        }
        if (!generic.pre_draw) {
            fail(Failure::GenericPreDraw);
            return;
        }
        if (generic.enabled != ordinal_hook.enabled) {
            fail(Failure::CrossEnabled);
            return;
        }
        if (generic.sample_tee_enabled != ordinal_hook.sample_tee_enabled) {
            fail(Failure::CrossSampleTee);
            return;
        }
        if (generic.capture_upstream_depth_pre_post !=
            ordinal_hook.capture_upstream_depth_pre_post) {
            fail(Failure::CrossCaptureMode);
            return;
        }
        if (generic.upstream_feedback_armed != ordinal_hook.upstream_feedback_armed) {
            fail(Failure::CrossRouteArmed);
            return;
        }
        if (generic.pre_draw != ordinal_hook.pre_draw) {
            fail(Failure::CrossPreDraw);
            return;
        }
        if (generic.observed_writer.hint != ordinal_hook.observed_writer.hint) {
            fail(Failure::CrossWriterHint);
            return;
        }
        if (generic.observed_writer.link != ordinal_hook.observed_writer.link) {
            fail(Failure::CrossWriterLink);
            return;
        }
        if (generic.observed_writer.range != ordinal_hook.observed_writer.range) {
            fail(Failure::CrossWriterRange);
            return;
        }
        if (generic.observed_writer.backing_generation !=
            ordinal_hook.observed_writer.backing_generation) {
            fail(Failure::CrossWriterBacking);
            return;
        }
        if (generic.observed_writer.mutation_epoch !=
            ordinal_hook.observed_writer.mutation_epoch) {
            fail(Failure::CrossWriterMutation);
            return;
        }
        if (generic.observed_writer.valid != ordinal_hook.observed_writer.valid) {
            fail(Failure::CrossWriterValidity);
            return;
        }
        if (generic.effective_depth_write != ordinal_hook.effective_depth_write) {
            fail(Failure::CrossEffectiveDepthWrite);
            return;
        }
        if (generic.actual_depth_aspect != ordinal_hook.actual_depth_aspect) {
            fail(Failure::CrossDepthAspect);
            return;
        }
        if (generic.canonical_range != ordinal_hook.canonical_range) {
            fail(Failure::CrossCanonicalRange);
            return;
        }
        if (generic.source_still_live != ordinal_hook.source_still_live) {
            fail(Failure::CrossSourceLive);
            return;
        }
        if (generic.source_identity_exact != ordinal_hook.source_identity_exact) {
            fail(Failure::CrossSourceIdentity);
            return;
        }
        if (generic.source_backing_exact != ordinal_hook.source_backing_exact) {
            fail(Failure::CrossSourceBacking);
            return;
        }
        if (generic.source_mutation_exact != ordinal_hook.source_mutation_exact) {
            fail(Failure::CrossSourceMutation);
            return;
        }
        if (generic.upstream_feedback_armed) {
            Increment(generic_route_armed_count);
        } else {
            Increment(generic_route_unarmed_count);
        }
        if (observation.generic_hook_plan != expected_generic) {
            fail(Failure::GenericPlanContradiction);
            return;
        }
        if (generic.upstream_feedback_armed) {
            if (expected_generic.capture_pre) {
                fail(Failure::GenericPlanAdmitted);
                return;
            }
            if (expected_generic.warmup_required) {
                Increment(generic_warmup_count);
            } else if (expected_generic.status == FinalGuestSurfaceStatus::InvalidationLoss &&
                       expected_generic.loss == FinalGuestSurfaceLoss{.invalidation = 1} &&
                       expected_generic.observation_only) {
                Increment(generic_invalidation_rejection_count);
            } else {
                fail(Failure::GenericPlanContradiction);
                return;
            }
        } else if (expected_generic != PpUpstreamSampleTeeProducerPreDepthHookPlan{}) {
            fail(Failure::GenericPlanContradiction);
            return;
        }
        Increment(generic_rejection_count);
        if (!IsMatchingOrdinalCapturePlan(observation.ordinal_capture_plan,
                                          expected_ordinal)) {
            fail(Failure::OrdinalPlanMismatch);
            return;
        }
        if (!IsAdmittedOrdinalCapturePlan(expected_ordinal, false)) {
            fail(Failure::OrdinalNotAdmitted);
            return;
        }
        Increment(ordinal_admission_count);
        if (!IsExactOrderingPlan(observation.ordering_plan, expected_order)) {
            fail(Failure::OrderingPlanMismatch);
            return;
        }
        if (!expected_order.exact) {
            fail(Failure::OrderingNotExact);
            return;
        }
        if (observation.ordering_descriptor.predecessor_ordinal_copy ||
            !observation.ordering_descriptor.target_ordinal_copy) {
            fail(Failure::WrongCopyRole);
            return;
        }
        Increment(ordering_exact_count);
        if (observation.predecessor_post_host_rendering_instance_id == 0) {
            fail(Failure::PredecessorPostMissing);
            return;
        }
        if (observation.predecessor_post_host_rendering_instance_id !=
            observation.ordering_descriptor.pre_host_rendering_instance_id) {
            fail(Failure::HostNotContiguous);
            return;
        }
        target_first_failure = Failure::Exact;
    }

    [[nodiscard]] PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReport Take(
        const PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureTakeDescriptor&
            descriptor) noexcept {
        const bool idle = predecessor_observation_count == 0 && target_observation_count == 0 &&
                          status == FinalGuestSurfaceStatus::AlreadyConsumed;
        auto final_status = status;
        auto final_loss = loss;
        auto reason = PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason::Idle;
        if (!idle) {
            if (final_status == FinalGuestSurfaceStatus::AlreadyConsumed) {
                using Failure =
                    PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureFirstFailure;
                const bool exact_common =
                    predecessor_observation_count == 1 && target_observation_count == 1 &&
                    generic_rejection_count == 2 && ordinal_admission_count == 2 &&
                    ordering_exact_count == 2 &&
                    predecessor_first_failure == Failure::Exact &&
                    target_first_failure == Failure::Exact;
                const bool exact_unarmed =
                    exact_common && generic_route_unarmed_count == 2 &&
                    generic_route_armed_count == 0 && generic_warmup_count == 0 &&
                    generic_invalidation_rejection_count == 0;
                const bool exact_armed_warmup =
                    exact_common && generic_route_unarmed_count == 0 &&
                    generic_route_armed_count == 2 && generic_warmup_count == 2 &&
                    generic_invalidation_rejection_count == 0;
                const bool exact_armed_rejection =
                    exact_common && generic_route_unarmed_count == 0 &&
                    generic_route_armed_count == 2 && generic_warmup_count == 0 &&
                    generic_invalidation_rejection_count == 2;
                if (exact_unarmed) {
                    final_status = FinalGuestSurfaceStatus::Complete;
                    reason = PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason::
                        ExactRouteAbsent;
                } else if (exact_armed_warmup || exact_armed_rejection) {
                    final_status = FinalGuestSurfaceStatus::Complete;
                    reason = PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason::
                        ExactArmedGenericNonAdmission;
                } else if (exact_common) {
                    final_status = FinalGuestSurfaceStatus::InvalidationLoss;
                    final_loss = {.invalidation = 1};
                    reason = PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason::
                        Invalid;
                } else {
                    final_status = FinalGuestSurfaceStatus::GapLoss;
                    final_loss = {.gap = 1};
                    reason = PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason::
                        MissingOrDuplicate;
                }
            } else {
                reason = final_status == FinalGuestSurfaceStatus::GapLoss
                             ? PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason::
                                   MissingOrDuplicate
                             : PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason::
                                   Invalid;
            }
        }
        const PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReport report{
            .sequence = descriptor.sequence,
            .reason = reason,
            .predecessor_observation_count = predecessor_observation_count,
            .target_observation_count = target_observation_count,
            .generic_route_unarmed_count = generic_route_unarmed_count,
            .generic_route_armed_count = generic_route_armed_count,
            .generic_rejection_count = generic_rejection_count,
            .generic_warmup_count = generic_warmup_count,
            .generic_invalidation_rejection_count =
                generic_invalidation_rejection_count,
            .ordinal_admission_count = ordinal_admission_count,
            .ordering_exact_count = ordering_exact_count,
            .predecessor_first_failure = predecessor_first_failure,
            .target_first_failure = target_first_failure,
            .status = final_status,
            .loss = final_loss,
            .diagnostic_saturated = diagnostic_saturated,
            .overflow = overflow,
        };
        *this = {};
        return report;
    }

    [[nodiscard]] u32 RetainedPrivateStateCount() const noexcept {
        return 0;
    }

private:
    static bool IsMatchingOrdinalCapturePlan(
        const PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan& actual,
        const PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan& expected) noexcept {
        return actual.scope_writer_count == expected.scope_writer_count &&
               actual.status == expected.status && actual.loss == expected.loss &&
               actual.capture_pre == expected.capture_pre &&
               actual.run_collision_check == expected.run_collision_check &&
               actual.arm_generic_predecessor_observation ==
                   expected.arm_generic_predecessor_observation &&
               actual.arm_generic_target_observation ==
                   expected.arm_generic_target_observation &&
               actual.wrong_ordinal_ignored == expected.wrong_ordinal_ignored &&
               actual.unrelated_writer_ignored == expected.unrelated_writer_ignored &&
               actual.context_ignored == expected.context_ignored &&
               actual.terminal_contradiction == expected.terminal_contradiction;
    }

    static bool IsAdmittedOrdinalCapturePlan(
        const PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan& plan,
        bool /* predecessor */) noexcept {
        return plan.status == FinalGuestSurfaceStatus::Complete && !plan.loss.Any() &&
               plan.capture_pre && plan.run_collision_check;
    }

    static bool IsExactOrdinalCapturePlan(
        const PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan& actual,
        const PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan& expected,
        bool predecessor) noexcept {
        return IsMatchingOrdinalCapturePlan(actual, expected) &&
               IsAdmittedOrdinalCapturePlan(actual, predecessor);
    }

    static bool IsExactOrderingPlan(
        const PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingPlan& actual,
        const PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingPlan& expected) noexcept {
        return actual.status == expected.status && actual.loss == expected.loss &&
               actual.existing_ordinal_copy_count == expected.existing_ordinal_copy_count &&
               actual.additional_gpu_copy_count == expected.additional_gpu_copy_count &&
               actual.additional_slot_count == expected.additional_slot_count &&
               actual.additional_release_count == expected.additional_release_count &&
               actual.exact == expected.exact &&
               actual.snapshots_before_diagnostics == expected.snapshots_before_diagnostics &&
               actual.occurrence_pre_before_predecessor_ordinal_gate ==
                   expected.occurrence_pre_before_predecessor_ordinal_gate &&
               actual.predecessor_ordinal_gate_before_target_ordinal_gate ==
                   expected.predecessor_ordinal_gate_before_target_ordinal_gate &&
               actual.occurrence_post_after_mark == expected.occurrence_post_after_mark &&
               actual.at_most_one_ordinal_copy == expected.at_most_one_ordinal_copy &&
               actual.affects_capture_behavior == expected.affects_capture_behavior &&
               actual.affects_renderer_behavior == expected.affects_renderer_behavior;
    }

    void Increment(u32& value) noexcept {
        if (value == MaxEvents) {
            diagnostic_saturated = true;
        } else {
            ++value;
        }
    }

    void RecordGap() noexcept {
        if (status == FinalGuestSurfaceStatus::AlreadyConsumed) {
            status = FinalGuestSurfaceStatus::GapLoss;
            loss = {.gap = 1};
        }
    }

    void RecordInvalidation() noexcept {
        if (status != FinalGuestSurfaceStatus::GapLoss) {
            status = FinalGuestSurfaceStatus::InvalidationLoss;
            loss = {.invalidation = 1};
        }
    }

    u32 predecessor_observation_count{};
    u32 target_observation_count{};
    u32 generic_route_unarmed_count{};
    u32 generic_route_armed_count{};
    u32 generic_rejection_count{};
    u32 generic_warmup_count{};
    u32 generic_invalidation_rejection_count{};
    u32 ordinal_admission_count{};
    u32 ordering_exact_count{};
    PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureFirstFailure
        predecessor_first_failure{};
    PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureFirstFailure
        target_first_failure{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool diagnostic_saturated{};
    bool overflow{};
};

[[nodiscard]] inline std::string
FormatPpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReport(
    const PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReport& report) {
    return "FGSCTPDRA s=" + std::to_string(report.sequence) +
           " r=" + std::to_string(static_cast<u32>(report.reason)) +
           " p=" + std::to_string(report.predecessor_observation_count) +
           " t=" + std::to_string(report.target_observation_count) +
           " gu=" + std::to_string(report.generic_route_unarmed_count) +
           " ga=" + std::to_string(report.generic_route_armed_count) +
           " gr=" + std::to_string(report.generic_rejection_count) +
           " gw=" + std::to_string(report.generic_warmup_count) +
           " gi=" + std::to_string(report.generic_invalidation_rejection_count) +
           " oa=" + std::to_string(report.ordinal_admission_count) +
           " oe=" + std::to_string(report.ordering_exact_count) +
           " pf=" + std::to_string(static_cast<u32>(report.predecessor_first_failure)) +
           " tf=" + std::to_string(static_cast<u32>(report.target_first_failure)) +
           " st=" + std::to_string(static_cast<u32>(report.status)) +
           " lm=" + std::to_string(FinalGuestSurfaceLossMask(report.loss, 0)) +
           " sat=" + std::to_string(report.diagnostic_saturated) +
           " ov=" + std::to_string(report.overflow);
}

struct PpUpstreamSampleTeeProducerPreDepthOccurrencePairTake {
    PpUpstreamSampleTeeProducerPreDepthOccurrenceReport report{};
    PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorPair private_pair{};
    PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm
        private_selected_producer_arm{};
};

class PpUpstreamSampleTeeProducerPreDepthOccurrenceEvidence {
public:
    static constexpr u32 MaxOccurrences = PpUpstreamSampleTeeProducerPreDepthMaxOccurrences;
    static constexpr u32 MaxEvents = 255;
    static constexpr u32 MaxPrivateState = MaxOccurrences + 4;

    void ObservePre(
        const PpUpstreamSampleTeeProducerPreDepthOccurrencePreObservation& observation) noexcept {
        const auto& descriptor = observation.hook_descriptor;
        const auto& plan = observation.hook_plan;
        const auto& learned = descriptor.learned_hint;
        const auto& observed = descriptor.observed_writer;
        const auto expected_plan = PlanPpUpstreamSampleTeeProducerPreDepthHook(descriptor);
        if (plan != expected_plan) {
            Increment(terminal_contradictions);
            Increment(malformed_plan_drift);
            RecordInvalidation();
            ClearPrivate();
            return;
        }
        const bool outer_context = descriptor.enabled && descriptor.sample_tee_enabled &&
                                   !descriptor.capture_upstream_depth_pre_post &&
                                   descriptor.upstream_feedback_armed && descriptor.pre_draw;
        if (!outer_context || !learned.valid) {
            Increment(diagnostic_rejections);
            return;
        }
        if (!learned.writer.Valid() || !observed.hint.Valid()) {
            Increment(diagnostic_rejections);
            Increment(malformed_plan_drift);
            return;
        }
        if (observed.hint != learned.writer) {
            Increment(diagnostic_rejections);
            Increment(unrelated_writer_misses);
            return;
        }
        const bool resource_exact = observed.link == learned.link &&
                                    observed.range == learned.range &&
                                    observed.backing_generation == learned.backing_generation;
        if (!resource_exact || !descriptor.source_identity_exact ||
            !descriptor.source_backing_exact) {
            Increment(diagnostic_rejections);
            Increment(resource_drift);
            return;
        }
        const bool learned_well_formed =
            learned.private_only && !learned.logs_private_state && learned.link.Valid() &&
            IsCanonicalPpDepthPredecessorRange(learned.range) &&
            learned.backing_generation != 0;
        if (!learned_well_formed || !observed.Valid()) {
            Increment(diagnostic_rejections);
            Increment(malformed_plan_drift);
            return;
        }
        if (!descriptor.effective_depth_write || !descriptor.actual_depth_aspect ||
            !descriptor.canonical_range) {
            Increment(diagnostic_rejections);
            Increment(semantic_drift);
            return;
        }
        if (!descriptor.source_still_live || !descriptor.source_mutation_exact) {
            Increment(diagnostic_rejections);
            Increment(currentness_drift);
            return;
        }
        if (observation.guest_rendering_serial == 0 ||
            observation.host_rendering_instance_id == 0 ||
            !observation.immediately_before_draw) {
            Increment(ordering_drift);
            RecordInvalidation();
            return;
        }
        if (open.valid) {
            RecordGap();
            return;
        }
        if (total_exact_occurrences == MaxOccurrences) {
            overflow = true;
            RecordCapacity();
            ClearPrivate();
            return;
        }
        if (continuity.Valid() &&
            observed.mutation_epoch != continuity.writer.mutation_epoch) {
            Increment(intervening_mutation_drift);
            RecordInvalidation();
        }
        const auto& predecessor = observation.private_immediate_predecessor_endpoint;
        const auto& first_writer = observation.first_writer_pre;
        const bool predecessor_exact =
            predecessor.Valid() && predecessor.writer.hint.Valid() &&
            predecessor.writer.link == observed.link &&
            predecessor.writer.range == observed.range &&
            predecessor.writer.backing_generation == observed.backing_generation &&
            predecessor.writer.mutation_epoch == observed.mutation_epoch &&
            predecessor.guest_rendering_serial == observation.guest_rendering_serial &&
            predecessor.host_rendering_instance_id <= observation.host_rendering_instance_id;
        const bool first_writer_exact =
            first_writer.recorded && first_writer.depth_stencil_exact && first_writer.same_scope &&
            first_writer.scope_depth_history_valid && first_writer.first_depth_writer_valid &&
            first_writer.depth_writer_count != 0 &&
            first_writer.depth_writer_count != std::numeric_limits<u32>::max() &&
            !first_writer.depth_writer_count_overflow;
        open = {.pre = observed,
                .guest_rendering_serial = observation.guest_rendering_serial,
                .host_rendering_instance_id = observation.host_rendering_instance_id,
                .predecessor = predecessor,
                .first_writer = first_writer,
                .predecessor_exact = predecessor_exact,
                .first_writer_exact = first_writer_exact,
                .pair_candidate_exact = predecessor_exact && first_writer_exact,
                .valid = true};
    }

    void ObservePost(
        const PpUpstreamSampleTeeProducerPreDepthOccurrencePostObservation&
            observation) noexcept {
        if (!open.valid) {
            RecordGap();
            return;
        }
        const auto& endpoint = observation.endpoint;
        const auto& writer = endpoint.writer;
        const bool resource_exact = writer.link == open.pre.link &&
                                    writer.range == open.pre.range &&
                                    writer.backing_generation == open.pre.backing_generation;
        if (!resource_exact) {
            Increment(resource_drift);
            RecordInvalidation();
            open = {};
            return;
        }
        const bool endpoint_exact = endpoint.valid && writer.valid && writer.hint.Valid() &&
                                    writer.hint == open.pre.hint && writer.link.Valid() &&
                                    IsCanonicalPpDepthPredecessorRange(writer.range) &&
                                    writer.backing_generation != 0 &&
                                    writer.mutation_epoch != 0 && endpoint.producer_marked;
        if (!endpoint_exact) {
            Increment(endpoint_drift);
            RecordInvalidation();
            open = {};
            return;
        }
        const bool mutation_exact =
            open.pre.mutation_epoch != std::numeric_limits<u64>::max() &&
            writer.mutation_epoch == open.pre.mutation_epoch + 1;
        if (!mutation_exact) {
            Increment(intervening_mutation_drift);
            RecordInvalidation();
            open = {};
            return;
        }
        const bool host_contiguous =
            endpoint.host_rendering_instance_id == open.host_rendering_instance_id ||
            (open.host_rendering_instance_id != std::numeric_limits<u64>::max() &&
             endpoint.host_rendering_instance_id == open.host_rendering_instance_id + 1);
        const bool ordering_exact =
            endpoint.guest_rendering_serial == open.guest_rendering_serial &&
            endpoint.host_rendering_instance_id != 0 && host_contiguous &&
            observation.live_guest_rendering_serial == endpoint.guest_rendering_serial &&
            observation.live_host_rendering_instance_id ==
                endpoint.host_rendering_instance_id &&
            observation.immediately_after_mark;
        if (!ordering_exact) {
            Increment(ordering_drift);
            RecordInvalidation();
            open = {};
            return;
        }
        ++total_exact_occurrences;
        continuity = endpoint;
        if (selected_endpoint_frozen) {
            Increment(occurrences_after_consumer);
            IncrementIf(selected_matches_after_consumer, endpoint == selected_endpoint);
        } else {
            IncrementIf(pair_predecessor_exact_occurrence_count, open.predecessor_exact);
            IncrementIf(pair_first_writer_exact_occurrence_count, open.first_writer_exact);
            auto& pair = predecessor_pairs[occurrence_count];
            pair = {.predecessor = open.predecessor,
                    .target_pre = open.pre,
                    .target_post = endpoint,
                    .scope_writer_count = open.first_writer.depth_writer_count,
                    .writer_history_exact = open.first_writer_exact,
                    .valid = open.pair_candidate_exact};
            if (pair.valid && occurrence_count != 0) {
                const auto& previous = predecessor_pairs[occurrence_count - 1];
                const bool mutation_contiguous =
                    previous.target_post.Valid() &&
                    pair.target_pre.mutation_epoch == previous.target_post.writer.mutation_epoch;
                const bool count_contiguous =
                    previous.scope_writer_count != std::numeric_limits<u32>::max() &&
                    pair.scope_writer_count == previous.scope_writer_count + 1;
                if (mutation_contiguous && (!previous.valid || !count_contiguous)) {
                    pair.valid = false;
                    predecessor_pair_continuity_exact = false;
                }
            }
            if (pair.valid) {
                ++paired_predecessor_count;
            }
            occurrences[occurrence_count++] = endpoint;
        }
        open = {};
    }

    void ObserveSelected(
        const PpUpstreamSampleTeeProducerPreDepthOccurrenceSelectedObservation&
            observation) noexcept {
        if (selected_observations != 0) {
            Increment(selected_observations);
            RecordGap();
            ClearPrivate();
            return;
        }
        Increment(selected_observations);
        const auto adapted = AdaptPpUpstreamSampleTeeProducerPreDepthLearnEvidence(
            observation.learn_descriptor, observation.learned_hint);
        const bool exact = adapted.context_exact && adapted.token_endpoint_join_exact &&
                           adapted.writer_shape_and_hash_exact &&
                           adapted.live_identity_backing_mutation_exact &&
                           adapted.route_and_view_exact &&
                           adapted.depth_aspect_and_write_exact && adapted.accepted;
        if (!exact) {
            pair_ordinal_status =
                PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::Invalid;
            RecordInvalidation();
            ClearPrivate();
            return;
        }
        const auto classification =
            ClassifyPpUpstreamSampleTeeProducerPreDepthSelectedOccurrence(
                {.occurrences = std::span{occurrences.data(), occurrence_count},
                 .selected_endpoint = observation.learn_descriptor.selected_endpoint});
        selected_occurrence = classification.selected_occurrence;
        selected_match_count = classification.match_count;
        selected_endpoint_frozen = true;
        if (selected_match_count > 1 ||
            selected_occurrence ==
                PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence::Multiple) {
            pair_ordinal_status =
                PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::Multiple;
            RecordInvalidation();
            ClearPrivate();
            return;
        }
        selected_endpoint = observation.learn_descriptor.selected_endpoint;
        if (selected_match_count == 1) {
            selected_pair_match_count = 1;
            const auto selected_index = classification.match_index;
            const auto& selected_target = predecessor_pairs[selected_index];
            frozen_selected_producer_arm =
                PlanPpUpstreamSampleTeeProducerPreDepthSelectedProducerOrdinalArm({
                    .learned_hint = observation.learned_hint,
                    .target_pre = selected_target.target_pre,
                    .target_post = selected_target.target_post,
                    .target_occurrence_index = selected_index,
                    .completed_occurrence_count = occurrence_count,
                });
            bool later_pairable_occurrence = false;
            for (u32 index = selected_index + 1; index < occurrence_count; ++index) {
                later_pairable_occurrence |= predecessor_pairs[index].valid;
            }
            if (later_pairable_occurrence) {
                pair_ordinal_status =
                    PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::
                        NotLatest;
            } else if (predecessor_pairs[selected_index].valid &&
                       predecessor_pair_continuity_exact) {
                u32 selected_pairable_index{};
                for (u32 index = 0; index < selected_index; ++index) {
                    selected_pairable_index += predecessor_pairs[index].valid;
                }
                frozen_predecessor_pair = predecessor_pairs[selected_index];
                frozen_pair_occurrence_index = selected_index;
                frozen_pair_completed_count = total_exact_occurrences;
                frozen_pairable_occurrence_index = selected_pairable_index;
                frozen_pairable_completed_count = paired_predecessor_count;
                pair_ordinal_status =
                    PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::Latest;
            } else {
                pair_ordinal_status =
                    PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::Invalid;
            }
        }
        occurrences = {};
        predecessor_pairs = {};
        occurrence_count = 0;
    }

    [[nodiscard]] PpUpstreamSampleTeeProducerPreDepthOccurrenceReport Take(
        const PpUpstreamSampleTeeProducerPreDepthOccurrenceTakeDescriptor& descriptor) noexcept {
        return TakeWithSelectedPredecessorPair(descriptor).report;
    }

    [[nodiscard]] PpUpstreamSampleTeeProducerPreDepthOccurrencePairTake
    TakeWithSelectedPredecessorPair(
        const PpUpstreamSampleTeeProducerPreDepthOccurrenceTakeDescriptor& descriptor) noexcept {
        if (status == FinalGuestSurfaceStatus::AlreadyConsumed) {
            if (open.valid || occurrences_after_consumer != 0) {
                RecordGap();
            } else if (selected_endpoint_frozen && selected_match_count != 1) {
                RecordInvalidation();
            } else if (total_exact_occurrences != 0 && !selected_endpoint_frozen) {
                RecordGap();
            } else if (total_exact_occurrences != 0 && selected_endpoint_frozen &&
                       selected_match_count == 1) {
                status = FinalGuestSurfaceStatus::Complete;
            }
        }
        auto final_pair_status = pair_ordinal_status;
        const bool missing_selected = total_exact_occurrences != 0 && selected_observations == 0;
        if (overflow || open.valid || occurrences_after_consumer != 0 ||
            status == FinalGuestSurfaceStatus::InvalidationLoss) {
            final_pair_status =
                PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::Invalid;
        } else if (selected_observations > 1) {
            final_pair_status =
                PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::Multiple;
        } else if (missing_selected) {
            final_pair_status =
                PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::Unavailable;
        } else if (status == FinalGuestSurfaceStatus::GapLoss &&
                   selected_observations != 0) {
            final_pair_status =
                PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::Invalid;
        }
        const bool publish_pair =
            status == FinalGuestSurfaceStatus::Complete &&
            final_pair_status ==
                PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::Latest &&
            frozen_predecessor_pair.valid && frozen_predecessor_pair.scope_writer_count != 0;
        PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorPair private_pair{};
        if (publish_pair) {
            const auto& pair = frozen_predecessor_pair;
            private_pair = {
                .predecessor_endpoint = pair.predecessor,
                .target_pre_intent = pair.target_pre,
                .target_post_endpoint = pair.target_post,
                .learned_hint = {.link = pair.predecessor.writer.link,
                                 .range = pair.predecessor.writer.range,
                                 .backing_generation = pair.predecessor.writer.backing_generation,
                                 .writer = pair.predecessor.writer.hint,
                                 .valid = true,
                                 .private_only = true},
                .selected_target_scope_writer_count = pair.scope_writer_count,
                .selected_target_occurrence_index = frozen_pair_occurrence_index,
                .completed_occurrence_count = frozen_pair_completed_count,
                .selected_pairable_occurrence_index =
                    frozen_pairable_occurrence_index,
                .completed_pairable_occurrence_count =
                    frozen_pairable_completed_count,
                .expected_predecessor_pre_scope_writer_count = pair.scope_writer_count - 1,
                .valid = true,
                .private_only = true,
            };
        }
        const auto private_selected_producer_arm =
            status == FinalGuestSurfaceStatus::Complete && selected_match_count == 1 &&
                    occurrences_after_consumer == 0 && selected_matches_after_consumer == 0 &&
                    frozen_selected_producer_arm.valid &&
                    frozen_selected_producer_arm.target_occurrence_index + 1 ==
                        total_exact_occurrences
                ? frozen_selected_producer_arm
                : PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm{};
        PpUpstreamSampleTeeProducerPreDepthOccurrenceReport report{
            .sequence = descriptor.sequence,
            .total_exact_occurrences = total_exact_occurrences,
            .selected_observation_count = selected_observations,
            .selected_occurrence = selected_occurrence,
            .selected_match_count = selected_match_count,
            .occurrences_after_consumer = occurrences_after_consumer,
            .selected_matches_after_consumer = selected_matches_after_consumer,
            .intervening_mutation_drift = intervening_mutation_drift,
            .resource_drift = resource_drift,
            .currentness_drift = currentness_drift,
            .malformed_plan_drift = malformed_plan_drift,
            .semantic_drift = semantic_drift,
            .endpoint_drift = endpoint_drift,
            .ordering_drift = ordering_drift,
            .unrelated_writer_misses = unrelated_writer_misses,
            .diagnostic_rejections = diagnostic_rejections,
            .terminal_contradictions = terminal_contradictions,
            .status = status,
            .loss = loss,
            .overflow = overflow,
            .diagnostic_saturated = diagnostic_saturated,
            .selected_endpoint_frozen = selected_endpoint_frozen,
            .paired_predecessor_count = paired_predecessor_count,
            .pair_predecessor_exact_occurrence_count =
                pair_predecessor_exact_occurrence_count,
            .pair_first_writer_exact_occurrence_count =
                pair_first_writer_exact_occurrence_count,
            .selected_predecessor_match_count = selected_pair_match_count,
            .learned_predecessor_valid_count = static_cast<u32>(publish_pair),
            .selected_predecessor_ordinal_status = final_pair_status,
        };
        *this = {};
        return {.report = report,
                .private_pair = private_pair,
                .private_selected_producer_arm = private_selected_producer_arm};
    }

    [[nodiscard]] u32 RetainedPrivateOccurrenceEndpointCount() const noexcept {
        return occurrence_count;
    }

    [[nodiscard]] u32 RetainedPrivateStateCount() const noexcept {
        return occurrence_count + static_cast<u32>(open.valid) +
               static_cast<u32>(continuity.Valid()) +
               static_cast<u32>(selected_endpoint.Valid()) +
               static_cast<u32>(frozen_selected_producer_arm.valid);
    }

    [[nodiscard]] PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence
    FrozenSelectedOccurrence() const noexcept {
        return selected_endpoint_frozen
                   ? selected_occurrence
                   : PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence::Unavailable;
    }

    [[nodiscard]] bool AwaitingPost() const noexcept {
        return open.valid;
    }

private:
    struct OpenOccurrence {
        PpDepthPredecessorWriterIntent pre{};
        u64 guest_rendering_serial{};
        u64 host_rendering_instance_id{};
        PpDepthPredecessorEndpointStamp predecessor{};
        PpUpstreamDepthFirstWriterPreObservation first_writer{};
        bool predecessor_exact{};
        bool first_writer_exact{};
        bool pair_candidate_exact{};
        bool valid{};
    };

    struct PredecessorPair {
        PpDepthPredecessorEndpointStamp predecessor{};
        PpDepthPredecessorWriterIntent target_pre{};
        PpDepthPredecessorEndpointStamp target_post{};
        u32 scope_writer_count{};
        bool writer_history_exact{};
        bool valid{};
    };

    static void IncrementBounded(u32& value, bool& did_overflow) noexcept {
        if (value == MaxEvents) {
            did_overflow = true;
        } else {
            ++value;
        }
    }

    void Increment(u32& value) noexcept {
        IncrementBounded(value, diagnostic_saturated);
    }

    void IncrementIf(u32& value, bool condition) noexcept {
        if (condition) {
            Increment(value);
        }
    }

    void RecordGap() noexcept {
        if (status == FinalGuestSurfaceStatus::AlreadyConsumed) {
            status = FinalGuestSurfaceStatus::GapLoss;
            loss = {.gap = 1};
        }
    }

    void RecordInvalidation() noexcept {
        if (status != FinalGuestSurfaceStatus::CapacityLoss) {
            status = FinalGuestSurfaceStatus::InvalidationLoss;
            loss = {.invalidation = 1};
        }
    }

    void RecordCapacity() noexcept {
        status = FinalGuestSurfaceStatus::CapacityLoss;
        loss = {.tile_capacity = 1};
    }

    void ClearPrivate() noexcept {
        occurrences = {};
        predecessor_pairs = {};
        occurrence_count = 0;
        open = {};
        continuity = {};
        selected_endpoint = {};
        frozen_predecessor_pair = {};
        frozen_selected_producer_arm = {};
    }

    std::array<PpDepthPredecessorEndpointStamp, MaxOccurrences> occurrences{};
    std::array<PredecessorPair, MaxOccurrences> predecessor_pairs{};
    u32 occurrence_count{};
    OpenOccurrence open{};
    PpDepthPredecessorEndpointStamp continuity{};
    PpDepthPredecessorEndpointStamp selected_endpoint{};
    PredecessorPair frozen_predecessor_pair{};
    PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm
        frozen_selected_producer_arm{};
    u32 total_exact_occurrences{};
    u32 selected_observations{};
    u32 occurrences_after_consumer{};
    u32 selected_matches_after_consumer{};
    u32 intervening_mutation_drift{};
    u32 resource_drift{};
    u32 currentness_drift{};
    u32 malformed_plan_drift{};
    u32 semantic_drift{};
    u32 endpoint_drift{};
    u32 ordering_drift{};
    u32 unrelated_writer_misses{};
    u32 diagnostic_rejections{};
    u32 terminal_contradictions{};
    u32 selected_match_count{};
    u32 selected_pair_match_count{};
    u32 paired_predecessor_count{};
    u32 pair_predecessor_exact_occurrence_count{};
    u32 pair_first_writer_exact_occurrence_count{};
    u32 frozen_pair_occurrence_index{};
    u32 frozen_pair_completed_count{};
    u32 frozen_pairable_occurrence_index{};
    u32 frozen_pairable_completed_count{};
    PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence selected_occurrence{
        PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence::Unavailable};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool overflow{};
    bool diagnostic_saturated{};
    bool selected_endpoint_frozen{};
    bool predecessor_pair_continuity_exact{true};
    PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus pair_ordinal_status{
        PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::Unavailable};
};

[[nodiscard]] inline std::string FormatPpUpstreamSampleTeeProducerPreDepthOccurrenceReport(
    const PpUpstreamSampleTeeProducerPreDepthOccurrenceReport& report) {
    return "FGSCTPDO s=" + std::to_string(report.sequence) +
           " n=" + std::to_string(report.total_exact_occurrences) +
           " sel=" + std::to_string(static_cast<u32>(report.selected_occurrence)) +
           " sm=" + std::to_string(report.selected_match_count) +
           " ac=" + std::to_string(report.occurrences_after_consumer) +
           " am=" + std::to_string(report.selected_matches_after_consumer) +
           " im=" + std::to_string(report.intervening_mutation_drift) +
           " rd=" + std::to_string(report.resource_drift) +
           " cur=" + std::to_string(report.currentness_drift) +
           " mal=" + std::to_string(report.malformed_plan_drift) +
           " sem=" + std::to_string(report.semantic_drift) +
           " end=" + std::to_string(report.endpoint_drift) +
           " ord=" + std::to_string(report.ordering_drift) +
           " wm=" + std::to_string(report.unrelated_writer_misses) +
           " rej=" + std::to_string(report.diagnostic_rejections) +
           " sat=" + std::to_string(report.diagnostic_saturated) +
           " tc=" + std::to_string(report.terminal_contradictions) +
           " st=" + std::to_string(static_cast<u32>(report.status)) +
           " ov=" + std::to_string(report.overflow) +
           " ls=" + std::to_string(report.loss.gap) + "/" +
           std::to_string(report.loss.invalidation) + "/" +
           std::to_string(report.loss.tile_capacity) +
           " pc=" + std::to_string(report.paired_predecessor_count) +
           " pm=" + std::to_string(report.selected_predecessor_match_count) +
           " lv=" + std::to_string(report.learned_predecessor_valid_count) +
           " os=" +
           std::to_string(static_cast<u32>(report.selected_predecessor_ordinal_status)) +
           " pex=" + std::to_string(report.pair_predecessor_exact_occurrence_count) +
           " fwx=" + std::to_string(report.pair_first_writer_exact_occurrence_count);
}

struct PpUpstreamSampleTeeProducerPreDepthPostDescriptor {
    PpDepthPredecessorWriterIntent pre{};
    PpDepthPredecessorEndpointStamp post{};
    u64 expected_guest_rendering_serial{};
    u64 pre_host_rendering_instance_id{};
    u64 live_post_host_rendering_instance_id{};
    bool diagnostic_resume{};
    bool effective_depth_write{};
};

struct PpUpstreamSampleTeeProducerPreDepthPost {
    PpDepthPredecessorEndpointStamp output{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool exact{};
    bool mutation_successor_exact{};
    bool diagnostic_resume_host_successor_exact{};
    bool copy_post_content{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthPost
CompletePpUpstreamSampleTeeProducerPreDepthPost(
    const PpUpstreamSampleTeeProducerPreDepthPostDescriptor& descriptor) noexcept {
    const auto& pre = descriptor.pre;
    const auto& post = descriptor.post;
    const bool mutation_exact = pre.mutation_epoch != std::numeric_limits<u64>::max() &&
                                post.writer.mutation_epoch == pre.mutation_epoch + 1;
    const bool host_exact = descriptor.diagnostic_resume && post.after_diagnostic_resume &&
                            descriptor.pre_host_rendering_instance_id !=
                                std::numeric_limits<u64>::max() &&
                            descriptor.live_post_host_rendering_instance_id ==
                                descriptor.pre_host_rendering_instance_id + 1 &&
                            post.host_rendering_instance_id ==
                                descriptor.live_post_host_rendering_instance_id;
    const bool exact = pre.Valid() && post.Valid() && mutation_exact && host_exact &&
                       post.writer.hint == pre.hint && post.writer.link == pre.link &&
                       post.writer.range == pre.range &&
                       post.writer.backing_generation == pre.backing_generation &&
                       post.guest_rendering_serial == descriptor.expected_guest_rendering_serial &&
                       descriptor.effective_depth_write;
    if (!exact) {
        return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                .loss = {.invalidation = 1},
                .mutation_successor_exact = mutation_exact,
                .diagnostic_resume_host_successor_exact = host_exact};
    }
    return {.output = post,
            .status = FinalGuestSurfaceStatus::Complete,
            .exact = true,
            .mutation_successor_exact = true,
            .diagnostic_resume_host_successor_exact = true};
}

[[nodiscard]] inline bool IsPpUpstreamSampleTeeProducerPreDepthSemanticPlaneExact(
    const PpUpstreamSampleTeeSelectedDepthPlane& lhs,
    const PpUpstreamSampleTeeSelectedDepthPlane& rhs) noexcept {
    if (lhs.region_count != rhs.region_count || lhs.plane_bytes != rhs.plane_bytes ||
        lhs.bytes_per_texel != rhs.bytes_per_texel || lhs.base_mip != rhs.base_mip ||
        lhs.mip_count != rhs.mip_count || lhs.base_layer != rhs.base_layer ||
        lhs.status != FinalGuestSurfaceStatus::Complete ||
        rhs.status != FinalGuestSurfaceStatus::Complete || lhs.loss.Any() || rhs.loss.Any()) {
        return false;
    }
    for (u32 index = 0; index < lhs.region_count; ++index) {
        const auto& a = lhs.regions[index];
        const auto& b = rhs.regions[index];
        if (a.logical_ordinal != b.logical_ordinal || a.mip_level != b.mip_level ||
            a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height ||
            a.buffer_offset != b.buffer_offset || a.byte_size != b.byte_size) {
            return false;
        }
    }
    return true;
}

struct PpUpstreamSampleTeeProducerPreDepthBindDescriptor {
    PpUpstreamSampleTeeProducerPreDepthPost post{};
    PpUpstreamSampleTeeSelectedDepthToken selected_token{};
    PpDepthPredecessorEndpointStamp live_selected_endpoint{};
    PpUpstreamSampleTeeSelectedDepthPlane producer_pre_plane{};
    PpUpstreamSampleTeeSelectedDepthPlane selected_depth_plane{};
    bool selected_source_still_live{};
};

struct PpUpstreamSampleTeeProducerPreDepthBinding {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool exact{};
    bool output_endpoint_exact{};
    bool semantic_layout_exact{};
    bool publish_producer_pre{};
    bool retains_private_hint{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthBinding
BindPpUpstreamSampleTeeProducerPreDepthToSelected(
    const PpUpstreamSampleTeeProducerPreDepthBindDescriptor& descriptor) noexcept {
    const auto& endpoint = descriptor.live_selected_endpoint;
    const bool output_exact = descriptor.post.exact && endpoint == descriptor.post.output &&
                              descriptor.selected_source_still_live &&
                              descriptor.selected_token.Valid() && endpoint.Valid() &&
                              descriptor.selected_token.link == endpoint.writer.link &&
                              descriptor.selected_token.backing_generation ==
                                  endpoint.writer.backing_generation &&
                              descriptor.selected_token.mutation_epoch ==
                                  endpoint.writer.mutation_epoch;
    const bool layout_exact = IsPpUpstreamSampleTeeProducerPreDepthSemanticPlaneExact(
        descriptor.producer_pre_plane, descriptor.selected_depth_plane);
    if (!output_exact || !layout_exact) {
        return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                .loss = {.invalidation = 1},
                .output_endpoint_exact = output_exact,
                .semantic_layout_exact = layout_exact};
    }
    return {.status = FinalGuestSurfaceStatus::Complete,
            .exact = true,
            .output_endpoint_exact = true,
            .semantic_layout_exact = true,
            .publish_producer_pre = true};
}

enum class PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp : u8 {
    None,
    Clear,
    Load,
};

struct PpUpstreamSampleTeeProducerPreDepthScopeEntryState {
    PpDepthPredecessorEndpointStamp predecessor_endpoint{};
    PpUpstreamSampleTeeProducerPreDepthHint private_target_hint{};
    u64 guest_rendering_serial{};
    u64 host_rendering_instance_id{};
    u32 private_depth_clear_bits{};
    VideoCore::ImageDepthWriterShape predecessor_shape{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp load_op{
        PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::None};
    bool accepted{};
    bool valid{};
    bool has_private_depth_clear_bits{};
    bool raw_depth_clear_encoding_exact{};
    bool predecessor_shape_exact{};
    bool retains_private_predecessor_endpoint{};
    bool retains_private_target_hint{};
    bool metadata_only{};
    u32 additional_gpu_copy_count{};
    bool affects_renderer_behavior{};
    bool logs_private_state{};

    bool operator==(const PpUpstreamSampleTeeProducerPreDepthScopeEntryState&) const = default;
};

struct PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation
    : PpUpstreamSampleTeeProducerPreDepthScopeEntryState {
    PpUpstreamSampleTeeProducerPreDepthScopeEntryState preserved_entry{};
    bool collision{};

    bool operator==(
        const PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation&) const = default;
};

struct PpUpstreamSampleTeeProducerPreDepthScopeEntryBeginDescriptor {
    bool enabled{};
    PpDepthPredecessorEndpointStamp predecessor_endpoint{};
    u64 guest_rendering_serial{};
    u64 host_rendering_instance_id{};
    bool attachment_has_depth{};
    bool depth_clear{};
    bool stencil_clear{};
    u32 private_depth_clear_bits{};
    bool raw_depth_clear_encoding_exact{};
    bool rendering_active{};
    bool regular_origin{};
    bool diagnostic_origin{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation
ObservePpUpstreamSampleTeeProducerPreDepthScopeEntryBegin(
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation& previous,
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryBeginDescriptor& descriptor) noexcept {
    if (descriptor.diagnostic_origin && !descriptor.regular_origin) {
        return previous;
    }
    if (previous.valid) {
        PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation collision{};
        collision.preserved_entry = previous;
        collision.status = FinalGuestSurfaceStatus::GapLoss;
        collision.loss.gap = 1;
        collision.collision = true;
        return collision;
    }
    const bool exact = descriptor.enabled && descriptor.regular_origin &&
                       !descriptor.diagnostic_origin && descriptor.attachment_has_depth &&
                       !descriptor.stencil_clear && descriptor.rendering_active &&
                       descriptor.guest_rendering_serial != 0 &&
                       descriptor.host_rendering_instance_id != 0 &&
                       descriptor.predecessor_endpoint.Valid();
    if (!exact) {
        return {};
    }
    PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation result{};
    result.predecessor_endpoint = descriptor.predecessor_endpoint;
    result.guest_rendering_serial = descriptor.guest_rendering_serial;
    result.host_rendering_instance_id = descriptor.host_rendering_instance_id;
    result.status = FinalGuestSurfaceStatus::Complete;
    result.load_op = descriptor.depth_clear
                         ? PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Clear
                         : PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Load;
    result.accepted = true;
    result.valid = true;
    result.has_private_depth_clear_bits = descriptor.depth_clear;
    result.raw_depth_clear_encoding_exact =
        descriptor.depth_clear && descriptor.raw_depth_clear_encoding_exact;
    result.private_depth_clear_bits =
        descriptor.depth_clear ? descriptor.private_depth_clear_bits : 0;
    result.retains_private_predecessor_endpoint = true;
    result.metadata_only = true;
    return result;
}

struct PpUpstreamSampleTeeProducerPreDepthScopeEntryResourceBeginDescriptor {
    bool enabled{};
    bool sample_tee_enabled{};
    // This observation intentionally precedes current-frame target arming.
    bool upstream_feedback_armed{};
    PpUpstreamSampleTeeProducerPreDepthHint target_hint{};
    VideoCore::ImageColorScopePrivateLink resolved_attachment_link{};
    VideoCore::SubresourceRange resolved_attachment_range{};
    u64 resolved_backing_generation{};
    u64 resolved_mutation_epoch{};
    PpDepthPredecessorEndpointStamp current_predecessor_endpoint{};
    VideoCore::ImageDepthWriterShape predecessor_shape{};
    bool predecessor_shape_exact{};
    bool source_still_live{};
    u64 guest_rendering_serial{};
    u64 host_rendering_instance_id{};
    bool attachment_has_depth{};
    bool depth_clear{};
    bool stencil_clear{};
    bool rendering_active{};
    bool regular_origin{};
    bool diagnostic_origin{};
};

struct PpUpstreamSampleTeeProducerPreDepthScopeLoadWriterShapeDescriptor {
    VideoCore::ImageDepthStencilProducerObservation producer{};
    PpDepthPredecessorEndpointStamp endpoint{};
};

struct PpUpstreamSampleTeeProducerPreDepthScopeLoadWriterShapeResult {
    VideoCore::ImageDepthWriterShape shape{};
    bool exact{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthScopeLoadWriterShapeResult
MakePpUpstreamSampleTeeProducerPreDepthScopeLoadWriterShape(
    const PpUpstreamSampleTeeProducerPreDepthScopeLoadWriterShapeDescriptor& descriptor) noexcept {
    const auto& producer = descriptor.producer;
    const auto& endpoint = descriptor.endpoint;
    const bool exact = producer.valid && producer.depth_write && endpoint.valid &&
                       endpoint.producer_marked && endpoint.writer.Valid() &&
                       producer.link == endpoint.writer.link &&
                       producer.range == endpoint.writer.range &&
                       producer.backing_generation == endpoint.writer.backing_generation &&
                       producer.mutation_epoch == endpoint.writer.mutation_epoch &&
                       producer.rendering_serial == endpoint.guest_rendering_serial &&
                       producer.host_rendering_instance_id ==
                           endpoint.host_rendering_instance_id &&
                       producer.kind == endpoint.writer.hint.draw.kind &&
                       producer.indexed == endpoint.writer.hint.draw.indexed &&
                       producer.element_count == endpoint.writer.hint.draw.element_count &&
                       producer.instance_count == endpoint.writer.hint.draw.instance_count &&
                       producer.sampled_images == endpoint.writer.hint.draw.sampled_images &&
                       producer.storage_writes == endpoint.writer.hint.draw.storage_writes &&
                       producer.fragment_pgm_hash == endpoint.writer.hint.fragment_pgm_hash;
    if (!exact) {
        return {};
    }
    return {
        .shape = {.kind = producer.kind,
                  .fragment_pgm_hash = producer.fragment_pgm_hash,
                  .element_count = producer.element_count,
                  .instance_count = producer.instance_count,
                  .sampled_images = producer.sampled_images,
                  .storage_writes = producer.storage_writes,
                  .guest_primitive_type = producer.guest_primitive_type,
                  .effective_primitive_type = producer.effective_primitive_type,
                  .cull_mode = producer.cull_mode,
                  .front_face = producer.front_face,
                  .fragment_mrt_mask = producer.fragment_mrt_mask,
                  .depth_compare_op = producer.depth_compare_op,
                  .indexed = producer.indexed,
                  .quad_list_converted = producer.quad_list_converted,
                  .fragment_writes_depth = producer.fragment_writes_depth,
                  .fragment_has_discard = producer.fragment_has_discard,
                  .depth_test_enabled = producer.depth_test_enabled,
                  .depth_bounds_test_enabled = producer.depth_bounds_test_enabled,
                  .depth_bias_enabled = producer.depth_bias_enabled,
                  .stencil_test_enabled = producer.stencil_test_enabled},
        .exact = true,
    };
}

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation
ObservePpUpstreamSampleTeeProducerPreDepthScopeEntryResourceBegin(
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation& previous,
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryResourceBeginDescriptor&
        descriptor) noexcept {
    if (descriptor.diagnostic_origin && !descriptor.regular_origin) {
        return previous;
    }
    if (previous.valid &&
        previous.guest_rendering_serial == descriptor.guest_rendering_serial &&
        previous.host_rendering_instance_id == descriptor.host_rendering_instance_id) {
        return previous;
    }
    if (previous.valid || previous.collision) {
        PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation collision{};
        collision.preserved_entry = previous.valid ? previous : previous.preserved_entry;
        collision.status = FinalGuestSurfaceStatus::GapLoss;
        collision.loss.gap = 1;
        collision.collision = true;
        return collision;
    }
    const auto& hint = descriptor.target_hint;
    const auto& endpoint = descriptor.current_predecessor_endpoint;
    const bool hint_valid =
        hint.valid && hint.private_only && !hint.logs_private_state && hint.link.Valid() &&
        IsCanonicalPpDepthPredecessorRange(hint.range) && hint.backing_generation != 0 &&
        hint.writer.Valid();
    const bool resource_exact =
        hint_valid && descriptor.resolved_attachment_link == hint.link &&
        descriptor.resolved_attachment_range == hint.range &&
        descriptor.resolved_backing_generation == hint.backing_generation &&
        descriptor.resolved_mutation_epoch != 0 && endpoint.Valid() &&
        endpoint.writer.link == descriptor.resolved_attachment_link &&
        endpoint.writer.range == descriptor.resolved_attachment_range &&
        endpoint.writer.backing_generation == descriptor.resolved_backing_generation &&
        endpoint.writer.mutation_epoch == descriptor.resolved_mutation_epoch;
    const bool exact =
        descriptor.enabled && descriptor.sample_tee_enabled && resource_exact &&
        descriptor.source_still_live && descriptor.guest_rendering_serial != 0 &&
        descriptor.host_rendering_instance_id != 0 && descriptor.attachment_has_depth &&
        !descriptor.depth_clear && !descriptor.stencil_clear && descriptor.rendering_active &&
        descriptor.regular_origin && !descriptor.diagnostic_origin;
    if (!exact) {
        return {};
    }
    auto result = ObservePpUpstreamSampleTeeProducerPreDepthScopeEntryBegin(
        {}, {.enabled = true,
             .predecessor_endpoint = endpoint,
             .guest_rendering_serial = descriptor.guest_rendering_serial,
             .host_rendering_instance_id = descriptor.host_rendering_instance_id,
             .attachment_has_depth = true,
             .rendering_active = true,
             .regular_origin = true});
    if (result.valid) {
        result.private_target_hint = hint;
        result.retains_private_target_hint = true;
        result.predecessor_shape = descriptor.predecessor_shape;
        result.predecessor_shape_exact = descriptor.predecessor_shape_exact &&
                                         descriptor.predecessor_shape.Valid();
    }
    return result;
}

[[nodiscard]] inline bool IsPpUpstreamSampleTeeProducerPreDepthScopeEntryResourceBindable(
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation& observation) noexcept {
    return observation.valid && observation.accepted && !observation.collision &&
           observation.status == FinalGuestSurfaceStatus::Complete &&
           !observation.loss.Any() && observation.load_op ==
                                          PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Load;
}

struct PpUpstreamSampleTeeProducerPreDepthScopeEntryResourceFrameDescriptor {
    PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation entry{};
    bool target_attempted{};
    bool upstream_feedback_armed{};
    bool outer_slot_present{};
};

struct PpUpstreamSampleTeeProducerPreDepthScopeEntryResourceFrameResult {
    PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation next_entry{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool retains_private_state{};
    bool requires_outer_slot{};
    u32 additional_release_count{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthScopeEntryResourceFrameResult
FinishPpUpstreamSampleTeeProducerPreDepthScopeEntryResourceFrame(
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryResourceFrameDescriptor&
        descriptor) noexcept {
    const bool collision = descriptor.entry.collision || descriptor.entry.loss.gap != 0;
    const bool exact = IsPpUpstreamSampleTeeProducerPreDepthScopeEntryResourceBindable(
                           descriptor.entry) &&
                       descriptor.target_attempted;
    if (collision || !exact) {
        return {.status = FinalGuestSurfaceStatus::GapLoss, .loss = {.gap = 1}};
    }
    return {.status = FinalGuestSurfaceStatus::Complete};
}

enum class PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch : u8 {
    Idle,
    None,
    Unique,
    Multiple,
};

enum class PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge : u8 {
    Idle,
    Unavailable,
    PriorFrame,
    CurrentFrame,
    InvalidTransition,
};

struct PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAgeDescriptor {
    u64 frame_boundary_guest_rendering_serial{};
    u64 target_guest_rendering_serial{};
    u64 predecessor_guest_rendering_serial{};
    bool exact{};
};

[[nodiscard]] constexpr PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge
ClassifyPpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge(
    const PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAgeDescriptor& descriptor) noexcept {
    if (!descriptor.exact || descriptor.frame_boundary_guest_rendering_serial == 0 ||
        descriptor.target_guest_rendering_serial == 0 ||
        descriptor.predecessor_guest_rendering_serial == 0) {
        return PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge::Unavailable;
    }
    if (descriptor.target_guest_rendering_serial <=
            descriptor.frame_boundary_guest_rendering_serial ||
        descriptor.predecessor_guest_rendering_serial >=
            descriptor.target_guest_rendering_serial) {
        return PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge::InvalidTransition;
    }
    return descriptor.predecessor_guest_rendering_serial <=
                   descriptor.frame_boundary_guest_rendering_serial
               ? PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge::PriorFrame
               : PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge::CurrentFrame;
}

enum class PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage : u8 {
    CanonicalCandidate,
    LearnedFullHint,
    TargetGuest,
    HostOrder,
    PredecessorValid,
    PredecessorGuestBefore,
    PredecessorHostBefore,
    TargetValid,
    ImageId,
    Uid,
    Range,
    Backing,
    Mutation,
    Count,
};

enum class PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure : u8 {
    Idle,
    FirstWriterNotRecorded,
    FirstWriterDepthInexact,
    FirstWriterHistoryInvalid,
    FirstWriterAlreadyPresent,
    FirstWriterCountNonzero,
    FirstWriterCountOverflow,
    TargetOperandMismatch,
    NoCanonicalCandidate,
    LearnedHintMismatch,
    TargetGuestMismatch,
    HostOrderMismatch,
    PredecessorInvalid,
    PredecessorGuestOrder,
    PredecessorHostOrder,
    TargetInvalid,
    ImageIdMismatch,
    UidMismatch,
    RangeMismatch,
    BackingMismatch,
    MutationMismatch,
    Exact,
};

struct PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelCount {
    u32 survivors{};
    u32 targets_with_any{};
};

struct PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateObservation {
    PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation entry{};
    bool accepted{};
    bool rejected{};
    bool diagnostic_resume{};
    bool terminal_contradiction{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateObservation
AdaptPpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidence(
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryResourceBeginDescriptor& descriptor,
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation& terminal) noexcept {
    if (descriptor.diagnostic_origin && !descriptor.regular_origin) {
        return {.diagnostic_resume = true};
    }
    const auto expected =
        ObservePpUpstreamSampleTeeProducerPreDepthScopeEntryResourceBegin({}, descriptor);
    const bool expected_accepted = expected.valid && expected.accepted;
    const bool terminal_accepted = terminal.valid && terminal.accepted;
    const bool contradiction = expected_accepted != terminal_accepted ||
                               (expected_accepted && terminal != expected);
    return {.entry = terminal,
            .accepted = expected_accepted && !contradiction,
            .rejected = !expected_accepted && !contradiction,
            .terminal_contradiction = contradiction};
}

struct PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetDescriptor {
    u64 target_guest_rendering_serial{};
    u64 target_pre_draw_host_rendering_instance_id{};
    PpUpstreamSampleTeeProducerPreDepthHookDescriptor hook_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthHookPlan hook_plan{};
    PpDepthPredecessorWriterIntent live_target_intent{};
    PpUpstreamDepthFirstWriterPreObservation first_writer_pre{};
};

struct PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidenceTakeDescriptor {
    u64 sequence{};
};

constexpr u32 PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain = 8;

struct PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidenceReport {
    u64 sequence{};
    u32 distinct_regular_load_entries{};
    u32 idempotent_regular_load_entries{};
    u32 diagnostic_resume_entries{};
    u32 rejected_regular_load_entries{};
    u32 regular_terminal_contradictions{};
    u32 target_observation_count{};
    u32 exact_target_hook_count{};
    u32 rejected_target_hooks{};
    u32 target_terminal_contradictions{};
    u32 unique_target_observations{};
    u32 no_match_target_observations{};
    u32 multiple_match_target_observations{};
    u32 duplicate_target_observations{};
    u32 contradictory_target_observations{};
    u32 target_guest_match_count{};
    u32 first_writer_exact_targets{};
    std::array<PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelCount,
               static_cast<u32>(
                   PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::Count)>
        funnel{};
    PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure frozen_first_failure{
        PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::Idle};
    PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch target_match{
        PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch::Idle};
    PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge predecessor_carry_age{
        PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge::Idle};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    VideoCore::ImageDepthWriterShape predecessor_shape{};
    std::array<VideoCore::ImageDepthWriterShape,
               PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain>
        predecessor_chain{};
    u32 predecessor_chain_count{};
    bool overflow{};
    bool counter_saturated{};
    bool predecessor_shape_exact{};
    bool predecessor_chain_exact{};
    bool predecessor_chain_ambiguous{};
    bool predecessor_chain_truncated{};
    bool retains_private_state{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_release_count{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
};

struct PpUpstreamSampleTeeProducerPreDepthScopeWriterPrivateTraceArm {
    std::array<PpDepthPredecessorHint,
               PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain>
        writer_hints{};
    std::array<VideoCore::ImageDepthWriterShape,
               PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain>
        writer_shapes{};
    u32 writer_count{};
    bool exact{};
    bool retains_image_identity{};
    bool retains_backing_identity{};
    bool retains_mutation_identity{};
    u32 additional_gpu_copy_count{};
    bool affects_renderer_behavior{};
};

struct PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidenceTakeResult {
    PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidenceReport report{};
    PpUpstreamSampleTeeProducerPreDepthScopeWriterPrivateTraceArm private_trace_arm{};
};

struct PpUpstreamSampleTeeScopeWriterTraceFrameDescriptor {
    PpUpstreamSampleTeeProducerPreDepthScopeWriterPrivateTraceArm arm{};
    u32 cycle_index{};
};

struct PpUpstreamSampleTeeScopeWriterTraceFramePlan {
    PpDepthPredecessorHint writer_hint{};
    VideoCore::ImageDepthWriterShape writer_shape{};
    u32 writer_index{};
    u32 writer_key_occurrence{};
    u32 next_cycle_index{};
    bool exact{};
    bool retains_private_identity{};
    bool affects_renderer_behavior{};
};

[[nodiscard]] constexpr bool MatchesPpUpstreamSampleTeeScopeWriterTraceKey(
    const VideoCore::ImageDepthWriterShape& expected,
    const VideoCore::ImageDepthWriterShape& observed) noexcept {
    return expected.Valid() && observed.Valid() && expected.kind == observed.kind &&
           expected.indexed == observed.indexed &&
           expected.fragment_pgm_hash == observed.fragment_pgm_hash &&
           expected.element_count == observed.element_count;
}

[[nodiscard]] inline PpUpstreamSampleTeeScopeWriterTraceFramePlan
PlanPpUpstreamSampleTeeScopeWriterTraceFrame(
    const PpUpstreamSampleTeeScopeWriterTraceFrameDescriptor& descriptor) noexcept {
    const auto& arm = descriptor.arm;
    if (!arm.exact || arm.writer_count == 0 || arm.writer_count > arm.writer_hints.size() ||
        arm.retains_image_identity || arm.retains_backing_identity ||
        arm.retains_mutation_identity || arm.additional_gpu_copy_count != 0 ||
        arm.affects_renderer_behavior) {
        return {};
    }
    for (u32 index = 0; index < arm.writer_count; ++index) {
        if (!arm.writer_hints[index].Valid() || !arm.writer_shapes[index].Valid()) {
            return {};
        }
    }
    const u32 writer_index = descriptor.cycle_index % arm.writer_count;
    u32 writer_key_occurrence{};
    for (u32 prior = 0; prior < writer_index; ++prior) {
        writer_key_occurrence += static_cast<u32>(
            MatchesPpUpstreamSampleTeeScopeWriterTraceKey(
                arm.writer_shapes[prior], arm.writer_shapes[writer_index]));
    }
    return {
        .writer_hint = arm.writer_hints[writer_index],
        .writer_shape = arm.writer_shapes[writer_index],
        .writer_index = writer_index,
        .writer_key_occurrence = writer_key_occurrence,
        .next_cycle_index = writer_index + 1 == arm.writer_count ? 0u : writer_index + 1,
        .exact = true,
    };
}

struct PpUpstreamSampleTeeScopeWriterTraceAdvanceDescriptor {
    PpUpstreamSampleTeeScopeWriterTraceFramePlan current{};
    PpUpstreamSampleTeeProducerPreDepthScopeWriterPrivateTraceArm incoming_arm{};
    u32 cycle_index{};
    bool current_captured{};
};

struct PpUpstreamSampleTeeScopeWriterTraceAdvanceResult {
    PpUpstreamSampleTeeScopeWriterTraceFramePlan frame{};
    u32 next_cycle_index{};
    bool preserved_current{};
};

[[nodiscard]] inline PpUpstreamSampleTeeScopeWriterTraceAdvanceResult
AdvancePpUpstreamSampleTeeScopeWriterTraceFrame(
    const PpUpstreamSampleTeeScopeWriterTraceAdvanceDescriptor& descriptor) noexcept {
    if (descriptor.current.exact && !descriptor.current_captured) {
        return {
            .frame = descriptor.current,
            .next_cycle_index = descriptor.cycle_index,
            .preserved_current = true,
        };
    }
    const auto incoming = PlanPpUpstreamSampleTeeScopeWriterTraceFrame({
        .arm = descriptor.incoming_arm,
        .cycle_index = descriptor.cycle_index,
    });
    if (incoming.exact) {
        return {
            .frame = incoming,
            .next_cycle_index = incoming.next_cycle_index,
        };
    }
    return {.next_cycle_index = descriptor.cycle_index};
}

enum class PpUpstreamSampleTeeScopeWriterTraceMatch : u8 {
    Invalid,
    Ignore,
    Capture,
    Duplicate,
};

struct PpUpstreamSampleTeeScopeWriterTraceMatchDescriptor {
    PpUpstreamSampleTeeScopeWriterTraceFramePlan frame{};
    PpDepthPredecessorHint observed_hint{};
    VideoCore::ImageDepthWriterShape observed_shape{};
    u32 observed_key_occurrence{};
    bool already_captured{};
};

[[nodiscard]] constexpr PpUpstreamSampleTeeScopeWriterTraceMatch
ClassifyPpUpstreamSampleTeeScopeWriterTraceMatch(
    const PpUpstreamSampleTeeScopeWriterTraceMatchDescriptor& descriptor) noexcept {
    if (!descriptor.frame.exact || !descriptor.frame.writer_hint.Valid() ||
        !descriptor.frame.writer_shape.Valid() || !descriptor.observed_hint.Valid() ||
        !descriptor.observed_shape.Valid()) {
        return PpUpstreamSampleTeeScopeWriterTraceMatch::Invalid;
    }
    if (!MatchesPpUpstreamSampleTeeScopeWriterTraceKey(
            descriptor.frame.writer_shape, descriptor.observed_shape)) {
        return PpUpstreamSampleTeeScopeWriterTraceMatch::Ignore;
    }
    if (descriptor.observed_key_occurrence != descriptor.frame.writer_key_occurrence) {
        return PpUpstreamSampleTeeScopeWriterTraceMatch::Ignore;
    }
    return descriptor.already_captured ? PpUpstreamSampleTeeScopeWriterTraceMatch::Duplicate
                                       : PpUpstreamSampleTeeScopeWriterTraceMatch::Capture;
}

struct PpUpstreamSampleTeeScopeWriterTraceCaptureDescriptor {
    PpUpstreamSampleTeeScopeWriterTraceFramePlan frame{};
    PpDepthPredecessorHint observed_hint{};
    VideoCore::ImageDepthWriterShape observed_shape{};
    u32 observed_key_occurrence{};
    bool sample_tee_enabled{};
    bool upstream_feedback_armed{};
    bool pre_draw{};
    bool effective_depth_write{};
    bool actual_depth_aspect{};
    bool canonical_range{};
    bool source_current{};
    bool already_captured{};
};

struct PpUpstreamSampleTeeScopeWriterTraceCapturePlan {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool capture_pre{};
    bool capture_post{};
    bool observation_only{};
    bool duplicate{};
    bool affects_renderer_behavior{};
};

[[nodiscard]] inline PpUpstreamSampleTeeScopeWriterTraceCapturePlan
PlanPpUpstreamSampleTeeScopeWriterTraceCapture(
    const PpUpstreamSampleTeeScopeWriterTraceCaptureDescriptor& descriptor) noexcept {
    const auto match = ClassifyPpUpstreamSampleTeeScopeWriterTraceMatch({
        .frame = descriptor.frame,
        .observed_hint = descriptor.observed_hint,
        .observed_shape = descriptor.observed_shape,
        .observed_key_occurrence = descriptor.observed_key_occurrence,
        .already_captured = descriptor.already_captured,
    });
    if (match == PpUpstreamSampleTeeScopeWriterTraceMatch::Ignore) {
        return {};
    }
    if (match == PpUpstreamSampleTeeScopeWriterTraceMatch::Capture &&
        !descriptor.upstream_feedback_armed) {
        return {};
    }
    if (match == PpUpstreamSampleTeeScopeWriterTraceMatch::Duplicate) {
        return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                .loss = {.invalidation = 1},
                .duplicate = true};
    }
    if (match != PpUpstreamSampleTeeScopeWriterTraceMatch::Capture ||
        !descriptor.sample_tee_enabled || !descriptor.upstream_feedback_armed ||
        !descriptor.pre_draw ||
        !descriptor.effective_depth_write || !descriptor.actual_depth_aspect ||
        !descriptor.canonical_range || !descriptor.source_current) {
        return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                .loss = {.invalidation = 1}};
    }
    return {.status = FinalGuestSurfaceStatus::Complete,
            .capture_pre = true,
            .capture_post = true,
            .observation_only = true};
}

struct PpUpstreamSampleTeeScopeWriterTraceTailDescriptor {
    PpUpstreamSampleTeeScopeWriterTraceFramePlan frame{};
    PpUpstreamInputContentPlane plane{};
    u32 tail_offset{};
    u32 tail_bytes{};
    u32 alignment{};
};

struct PpUpstreamSampleTeeScopeWriterTraceTailPlan {
    PpUpstreamInputContentPlane pre_plane{};
    PpUpstreamInputContentPlane post_plane{};
    u32 writer_index{};
    u32 total_bytes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy_pre{};
    bool copy_post{};
    u32 additional_slot_count{};
    bool affects_renderer_behavior{};
};

[[nodiscard]] inline PpUpstreamSampleTeeScopeWriterTraceTailPlan
PlanPpUpstreamSampleTeeScopeWriterTraceTail(
    const PpUpstreamSampleTeeScopeWriterTraceTailDescriptor& descriptor) noexcept {
    const auto reject = [](FinalGuestSurfaceStatus status,
                           u32 FinalGuestSurfaceLoss::*member) {
        PpUpstreamSampleTeeScopeWriterTraceTailPlan result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    const auto& plane = descriptor.plane;
    if (!descriptor.frame.exact || descriptor.alignment == 0 ||
        plane.status != FinalGuestSurfaceStatus::Complete || plane.loss.Any() ||
        plane.region_count == 0 || plane.region_count > plane.regions.size() ||
        plane.plane_bytes == 0 || plane.bytes_per_texel != 4 || plane.mip_count != 1) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    u64 covered{};
    for (u32 index = 0; index < plane.region_count; ++index) {
        const auto& region = plane.regions[index];
        if (region.logical_ordinal == 0 || region.width == 0 || region.height == 0 ||
            region.byte_size == 0 || region.byte_size % plane.bytes_per_texel != 0 ||
            region.buffer_offset != covered) {
            return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                          &FinalGuestSurfaceLoss::invalidation);
        }
        covered += region.byte_size;
    }
    if (covered != plane.plane_bytes) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    const auto align_up = [](u64 value, u64 alignment) {
        const u64 remainder = value % alignment;
        return remainder == 0 ? value : value + alignment - remainder;
    };
    const u64 pre_offset = descriptor.tail_offset;
    const u64 post_offset = align_up(pre_offset + plane.plane_bytes, descriptor.alignment);
    const u64 end = post_offset + plane.plane_bytes;
    const u64 tail_end = static_cast<u64>(descriptor.tail_offset) + descriptor.tail_bytes;
    if (post_offset > std::numeric_limits<u32>::max() || end > tail_end ||
        end > std::numeric_limits<u32>::max()) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss, &FinalGuestSurfaceLoss::history);
    }
    auto pre = plane;
    auto post = plane;
    pre.plane_offset = static_cast<u32>(pre_offset);
    post.plane_offset = static_cast<u32>(post_offset);
    return {
        .pre_plane = pre,
        .post_plane = post,
        .writer_index = descriptor.frame.writer_index,
        .total_bytes = static_cast<u32>(end - pre_offset),
        .status = FinalGuestSurfaceStatus::Complete,
        .copy_pre = true,
        .copy_post = true,
    };
}

struct PpUpstreamSampleTeeScopeWriterTraceOwned {
    std::vector<std::byte> bytes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool mapped_invalidation_succeeded{};
    bool bounded_copy_succeeded{};
    bool shared_release_deferred{};
    bool retains_mapped_span{};
};

[[nodiscard]] inline PpUpstreamSampleTeeScopeWriterTraceOwned
CopyPpUpstreamSampleTeeScopeWriterTrace(
    const PpUpstreamSampleTeeScopeWriterTraceTailPlan& plan,
    bool mapped_invalidation_succeeded, std::span<const std::byte> download_slot) {
    const auto reject = [mapped_invalidation_succeeded] {
        return PpUpstreamSampleTeeScopeWriterTraceOwned{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
            .mapped_invalidation_succeeded = mapped_invalidation_succeeded,
            .shared_release_deferred = true,
        };
    };
    if (plan.status != FinalGuestSurfaceStatus::Complete || plan.loss.Any() ||
        !plan.copy_pre || !plan.copy_post || !mapped_invalidation_succeeded ||
        plan.pre_plane.plane_bytes == 0 ||
        plan.pre_plane.plane_bytes != plan.post_plane.plane_bytes ||
        plan.pre_plane.plane_offset > download_slot.size() ||
        plan.post_plane.plane_offset > download_slot.size() ||
        plan.pre_plane.plane_bytes > download_slot.size() - plan.pre_plane.plane_offset ||
        plan.post_plane.plane_bytes > download_slot.size() - plan.post_plane.plane_offset) {
        return reject();
    }
    PpUpstreamSampleTeeScopeWriterTraceOwned result{
        .status = FinalGuestSurfaceStatus::Complete,
        .mapped_invalidation_succeeded = true,
        .bounded_copy_succeeded = true,
        .shared_release_deferred = true,
    };
    result.bytes.resize(static_cast<size_t>(plan.pre_plane.plane_bytes) * 2);
    std::ranges::copy(download_slot.subspan(plan.pre_plane.plane_offset,
                                            plan.pre_plane.plane_bytes),
                      result.bytes.begin());
    std::ranges::copy(download_slot.subspan(plan.post_plane.plane_offset,
                                            plan.post_plane.plane_bytes),
                      result.bytes.begin() + plan.pre_plane.plane_bytes);
    return result;
}

struct PpUpstreamSampleTeeScopeWriterTraceReductionDescriptor {
    u64 sequence{};
    PpUpstreamSampleTeeScopeWriterTraceFramePlan frame{};
    PpUpstreamSampleTeeScopeWriterTraceTailPlan tail{};
    PpUpstreamSampleTeeScopeWriterTraceOwned owned{};
    bool tee_authoritative{};
};

struct PpUpstreamSampleTeeScopeWriterTraceReport {
    u64 sequence{};
    VideoCore::ImageDepthWriterShape writer_shape{};
    u32 writer_index{};
    u32 equal_pixels{};
    u32 changed_pixels{};
    u32 covered_pixels{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool retains_bytes{};
};

[[nodiscard]] inline PpUpstreamSampleTeeScopeWriterTraceReport
ReducePpUpstreamSampleTeeScopeWriterTrace(
    const PpUpstreamSampleTeeScopeWriterTraceReductionDescriptor& descriptor) noexcept {
    const auto reject = [&] {
        return PpUpstreamSampleTeeScopeWriterTraceReport{
            .sequence = descriptor.sequence,
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
        };
    };
    const auto& frame = descriptor.frame;
    const auto& tail = descriptor.tail;
    const auto& owned = descriptor.owned;
    if (descriptor.sequence == 0 || !descriptor.tee_authoritative || !frame.exact ||
        !frame.writer_shape.Valid() || tail.status != FinalGuestSurfaceStatus::Complete ||
        tail.loss.Any() || tail.writer_index != frame.writer_index || !tail.copy_pre ||
        !tail.copy_post || tail.pre_plane.status != FinalGuestSurfaceStatus::Complete ||
        tail.post_plane.status != FinalGuestSurfaceStatus::Complete ||
        tail.pre_plane.loss.Any() || tail.post_plane.loss.Any() ||
        tail.pre_plane.region_count == 0 ||
        tail.pre_plane.region_count != tail.post_plane.region_count ||
        tail.pre_plane.plane_bytes == 0 ||
        tail.pre_plane.plane_bytes != tail.post_plane.plane_bytes ||
        tail.pre_plane.bytes_per_texel != 4 || tail.post_plane.bytes_per_texel != 4 ||
        owned.status != FinalGuestSurfaceStatus::Complete || owned.loss.Any() ||
        !owned.mapped_invalidation_succeeded || !owned.bounded_copy_succeeded ||
        !owned.shared_release_deferred || owned.retains_mapped_span ||
        owned.bytes.size() != static_cast<size_t>(tail.pre_plane.plane_bytes) * 2) {
        return reject();
    }
    PpUpstreamSampleTeeScopeWriterTraceReport result{
        .sequence = descriptor.sequence,
        .writer_shape = frame.writer_shape,
        .writer_index = frame.writer_index,
        .status = FinalGuestSurfaceStatus::Complete,
    };
    const auto pre = std::span<const std::byte>{owned.bytes}.first(tail.pre_plane.plane_bytes);
    const auto post = std::span<const std::byte>{owned.bytes}.subspan(
        tail.pre_plane.plane_bytes, tail.post_plane.plane_bytes);
    u64 expected_offset{};
    for (u32 index = 0; index < tail.pre_plane.region_count; ++index) {
        const auto& a = tail.pre_plane.regions[index];
        const auto& b = tail.post_plane.regions[index];
        const bool same_region = a.logical_ordinal == b.logical_ordinal &&
                                 a.mip_level == b.mip_level && a.x == b.x && a.y == b.y &&
                                 a.width == b.width && a.height == b.height &&
                                 a.buffer_offset == b.buffer_offset &&
                                 a.byte_size == b.byte_size;
        if (!same_region || a.logical_ordinal == 0 || a.buffer_offset != expected_offset ||
            a.byte_size == 0 || a.byte_size % 4 != 0 ||
            static_cast<u64>(a.buffer_offset) + a.byte_size > pre.size()) {
            return reject();
        }
        for (u32 offset = 0; offset < a.byte_size; offset += 4) {
            const auto before = pre.subspan(a.buffer_offset + offset, 4);
            const auto after = post.subspan(a.buffer_offset + offset, 4);
            if (std::ranges::equal(before, after)) {
                ++result.equal_pixels;
            } else {
                ++result.changed_pixels;
            }
            ++result.covered_pixels;
        }
        expected_offset += a.byte_size;
    }
    if (expected_offset != pre.size() ||
        result.equal_pixels + result.changed_pixels != result.covered_pixels) {
        return reject();
    }
    return result;
}

[[nodiscard]] inline std::string FormatPpUpstreamSampleTeeScopeWriterTrace(
    const PpUpstreamSampleTeeScopeWriterTraceReport& report) {
    const auto& shape = report.writer_shape;
    return "FGSCTPDW s=" + std::to_string(report.sequence) +
           " wi=" + std::to_string(report.writer_index) +
           " k=" + std::to_string(static_cast<u32>(shape.kind)) +
           " e=" + std::to_string(shape.element_count) +
           " n=" + std::to_string(shape.instance_count) +
           " sm=" + std::to_string(shape.sampled_images) +
           " h=" + std::to_string(shape.fragment_pgm_hash) +
           " eq=" + std::to_string(report.equal_pixels) +
           " ch=" + std::to_string(report.changed_pixels) +
           " cov=" + std::to_string(report.covered_pixels) +
           " st=" + std::to_string(static_cast<u32>(report.status)) +
           " lm=" + std::to_string(FinalGuestSurfaceLossMask(report.loss, 0));
}

struct PpUpstreamSampleTeeScopeWriterTraceGateReport {
    u64 sequence{};
    bool frame_exact{};
    bool incoming_arm_exact{};
    u32 incoming_writer_count{};
    bool incoming_hints_valid{};
    bool incoming_shapes_valid{};
    u32 live_pre_draws{};
    u32 key_matches{};
    u32 armed_key_matches{};
    u32 capture_attempts{};
    u32 pre_complete{};
    u32 post_complete{};
    bool tee_attempted{};
};

[[nodiscard]] inline std::string FormatPpUpstreamSampleTeeScopeWriterTraceGate(
    const PpUpstreamSampleTeeScopeWriterTraceGateReport& report) {
    return "FGSCTPDWG s=" + std::to_string(report.sequence) +
           " fx=" + std::to_string(report.frame_exact ? 1 : 0) +
           " ix=" + std::to_string(report.incoming_arm_exact ? 1 : 0) +
           " iw=" + std::to_string(report.incoming_writer_count) +
           " ih=" + std::to_string(report.incoming_hints_valid ? 1 : 0) +
           " is=" + std::to_string(report.incoming_shapes_valid ? 1 : 0) +
           " lp=" + std::to_string(report.live_pre_draws) +
           " km=" + std::to_string(report.key_matches) +
           " am=" + std::to_string(report.armed_key_matches) +
           " ca=" + std::to_string(report.capture_attempts) +
           " pc=" + std::to_string(report.pre_complete) +
           " qc=" + std::to_string(report.post_complete) +
           " ta=" + std::to_string(report.tee_attempted ? 1 : 0);
}

class PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidence {
public:
    static constexpr u32 MaxCandidates = 32;
    static constexpr u32 MaxPublicCount = std::numeric_limits<u8>::max();

    void BeginFrame(u64 guest_rendering_serial) noexcept {
        if (guest_rendering_serial == 0) {
            frame_boundary_conflict = true;
            active = true;
            return;
        }
        if (frame_boundary_guest_rendering_serial == 0) {
            frame_boundary_guest_rendering_serial = guest_rendering_serial;
            return;
        }
        if (frame_boundary_guest_rendering_serial != guest_rendering_serial) {
            frame_boundary_conflict = true;
            active = true;
        }
    }

    void ObserveRegularLoad(
        const PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateObservation&
            observation) noexcept {
        active = true;
        if (observation.diagnostic_resume) {
            Increment(diagnostic_resume_entries);
            return;
        }
        if (observation.terminal_contradiction) {
            Increment(regular_terminal_contradictions);
            return;
        }
        if (observation.rejected) {
            Increment(rejected_regular_load_entries);
            return;
        }
        if (!observation.accepted || !observation.entry.valid) {
            Increment(regular_terminal_contradictions);
            return;
        }
        const auto duplicate = std::find_if(
            candidates.begin(), candidates.begin() + candidate_count,
            [&](const auto& candidate) {
                return candidate.guest_rendering_serial ==
                           observation.entry.guest_rendering_serial &&
                       candidate.host_rendering_instance_id ==
                           observation.entry.host_rendering_instance_id;
            });
        if (duplicate != candidates.begin() + candidate_count) {
            if (*duplicate != observation.entry) {
                Increment(regular_terminal_contradictions);
                return;
            }
            Increment(idempotent_regular_load_entries);
            return;
        }
        if (candidate_count == MaxCandidates) {
            overflow = true;
            return;
        }
        candidates[candidate_count++] = observation.entry;
        distinct_regular_load_entries = candidate_count;
    }

    void ObserveTarget(
        const PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetDescriptor& descriptor) noexcept {
        active = true;
        Increment(target_observation_count);
        const bool duplicate = has_previous_target &&
                               previous_target_guest ==
                                   descriptor.target_guest_rendering_serial &&
                               previous_target_host ==
                                   descriptor.target_pre_draw_host_rendering_instance_id;
        if (duplicate) {
            Increment(duplicate_target_observations);
        }
        previous_target_guest = descriptor.target_guest_rendering_serial;
        previous_target_host = descriptor.target_pre_draw_host_rendering_instance_id;
        has_previous_target = true;

        const auto expected =
            PlanPpUpstreamSampleTeeProducerPreDepthHook(descriptor.hook_descriptor);
        const bool terminal_contradiction = expected != descriptor.hook_plan;
        if (terminal_contradiction) {
            Increment(target_terminal_contradictions);
            Increment(contradictory_target_observations);
            return;
        }
        const bool exact_hook = descriptor.hook_plan.status ==
                                    FinalGuestSurfaceStatus::Complete &&
                                !descriptor.hook_plan.loss.Any() &&
                                descriptor.hook_plan.capture_pre &&
                                descriptor.hook_plan.observe_metadata_post &&
                                descriptor.hook_plan.observation_only &&
                                !descriptor.hook_plan.capture_post_content &&
                                !descriptor.hook_plan.touch_depth_predecessor_runtime &&
                                !descriptor.hook_plan.affects_renderer_behavior;
        if (!exact_hook) {
            Increment(rejected_target_hooks);
            return;
        }
        Increment(exact_target_hook_count);
        const auto& observed_target = descriptor.hook_descriptor.observed_writer;
        const auto& target = descriptor.live_target_intent;
        const auto& learned = descriptor.hook_descriptor.learned_hint;
        const auto& pre = descriptor.first_writer_pre;
        const auto first_writer_failure = ClassifyFirstWriterFailure(pre);
        const bool first_writer_exact =
            first_writer_failure ==
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::Exact;
        if (first_writer_exact) {
            Increment(first_writer_exact_targets);
        }
        const bool target_operand_exact =
            target.hint == observed_target.hint && target.link == observed_target.link &&
            target.range == observed_target.range &&
            target.backing_generation == observed_target.backing_generation &&
            target.mutation_epoch == observed_target.mutation_epoch;
        if (!target_operand_exact) {
            Increment(target_terminal_contradictions);
            Increment(contradictory_target_observations);
            Increment(no_match_target_observations);
            FreezeTarget(PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch::None, 0,
                         PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::
                             TargetOperandMismatch);
            return;
        }

        std::array<bool, MaxCandidates> surviving{};
        u32 survivor_count = candidate_count;
        std::fill(surviving.begin(), surviving.begin() + candidate_count, true);
        PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure funnel_failure{
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::Exact};
        const auto apply_stage = [&](PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage stage,
                                     auto&& predicate,
                                     PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure
                                         failed) {
            u32 next_survivors{};
            for (u32 index = 0; index < candidate_count; ++index) {
                if (!surviving[index] || !predicate(candidates[index])) {
                    surviving[index] = false;
                    continue;
                }
                ++next_survivors;
                Increment(funnel[static_cast<u32>(stage)].survivors);
            }
            survivor_count = next_survivors;
            if (next_survivors != 0) {
                Increment(funnel[static_cast<u32>(stage)].targets_with_any);
            } else if (funnel_failure ==
                       PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::Exact) {
                funnel_failure = failed;
            }
        };
        apply_stage(
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::CanonicalCandidate,
            [](const auto& candidate) {
                return candidate.valid && candidate.accepted && candidate.metadata_only &&
                       candidate.load_op ==
                           PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Load;
            },
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::NoCanonicalCandidate);
        apply_stage(
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::LearnedFullHint,
            [&](const auto& candidate) {
                return candidate.retains_private_target_hint &&
                       candidate.private_target_hint == learned;
            },
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::LearnedHintMismatch);
        apply_stage(
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::TargetGuest,
            [&](const auto& candidate) {
                return candidate.guest_rendering_serial ==
                       descriptor.target_guest_rendering_serial;
            },
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::TargetGuestMismatch);
        apply_stage(
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::HostOrder,
            [&](const auto& candidate) {
                return candidate.host_rendering_instance_id <=
                       descriptor.target_pre_draw_host_rendering_instance_id;
            },
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::HostOrderMismatch);
        apply_stage(
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::PredecessorValid,
            [](const auto& candidate) {
                const auto& predecessor = candidate.predecessor_endpoint;
                return predecessor.valid && predecessor.writer.valid &&
                       predecessor.writer.hint.Valid() &&
                       predecessor.guest_rendering_serial != 0 &&
                       predecessor.host_rendering_instance_id != 0 &&
                       predecessor.producer_marked;
            },
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::PredecessorInvalid);
        apply_stage(
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::
                PredecessorGuestBefore,
            [](const auto& candidate) {
                return candidate.predecessor_endpoint.guest_rendering_serial <
                       candidate.guest_rendering_serial;
            },
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::
                PredecessorGuestOrder);
        apply_stage(
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::
                PredecessorHostBefore,
            [](const auto& candidate) {
                return candidate.predecessor_endpoint.host_rendering_instance_id <
                       candidate.host_rendering_instance_id;
            },
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::
                PredecessorHostOrder);
        apply_stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::TargetValid,
                    [&](const auto&) { return target.Valid(); },
                    PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::TargetInvalid);
        apply_stage(
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::ImageId,
            [&](const auto& candidate) {
                return candidate.predecessor_endpoint.writer.link.id == target.link.id;
            },
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::ImageIdMismatch);
        apply_stage(
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::Uid,
            [&](const auto& candidate) {
                return candidate.predecessor_endpoint.writer.link.uid == target.link.uid;
            },
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::UidMismatch);
        apply_stage(
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::Range,
            [&](const auto& candidate) {
                return candidate.predecessor_endpoint.writer.range == target.range;
            },
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::RangeMismatch);
        apply_stage(
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::Backing,
            [&](const auto& candidate) {
                return candidate.predecessor_endpoint.writer.backing_generation ==
                       target.backing_generation;
            },
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::BackingMismatch);
        apply_stage(
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::Mutation,
            [&](const auto& candidate) {
                return candidate.predecessor_endpoint.writer.mutation_epoch ==
                       target.mutation_epoch;
            },
            PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::MutationMismatch);
        const u32 matches = first_writer_exact ? survivor_count : 0;
        const auto classification =
            matches == 0
                ? PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch::None
            : matches == 1
                ? PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch::Unique
                : PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch::Multiple;
        VideoCore::ImageDepthWriterShape matched_shape{};
        bool matched_shape_exact{};
        u32 matched_index = candidate_count;
        if (classification ==
            PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch::Unique) {
            for (u32 index = 0; index < candidate_count; ++index) {
                if (!surviving[index]) {
                    continue;
                }
                matched_shape = candidates[index].predecessor_shape;
                matched_shape_exact = candidates[index].predecessor_shape_exact &&
                                      matched_shape.Valid();
                matched_index = index;
                break;
            }
        }
        std::array<VideoCore::ImageDepthWriterShape,
                   PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain>
            predecessor_chain{};
        u32 predecessor_chain_count{};
        bool predecessor_chain_exact{};
        bool predecessor_chain_ambiguous{};
        bool predecessor_chain_truncated{};
        std::array<PpDepthPredecessorHint,
                   PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain>
            private_writer_hints{};
        std::array<VideoCore::ImageDepthWriterShape,
                   PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain>
            private_writer_shapes{};
        u32 private_writer_count{};
        bool private_writer_trace_exact{};
        bool private_writer_trace_truncated{};
        auto carry_age =
            PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge::Unavailable;
        if (matched_shape_exact && matched_index != candidate_count) {
            carry_age = ClassifyPpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge({
                .frame_boundary_guest_rendering_serial =
                    frame_boundary_guest_rendering_serial,
                .target_guest_rendering_serial = candidates[matched_index].guest_rendering_serial,
                .predecessor_guest_rendering_serial =
                    candidates[matched_index].predecessor_endpoint.guest_rendering_serial,
                .exact = true,
            });
            auto cursor = candidates[matched_index].predecessor_endpoint;
            auto cursor_shape = matched_shape;
            u32 traversed{};
            while (traversed++ < candidate_count) {
                if (private_writer_count == private_writer_hints.size()) {
                    private_writer_trace_truncated = true;
                } else {
                    private_writer_hints[private_writer_count] = cursor.writer.hint;
                    private_writer_shapes[private_writer_count] = cursor_shape;
                    ++private_writer_count;
                }
                const bool new_shape = predecessor_chain_count == 0 ||
                                       predecessor_chain[predecessor_chain_count - 1] !=
                                           cursor_shape;
                if (new_shape) {
                    if (predecessor_chain_count == predecessor_chain.size()) {
                        predecessor_chain_exact = true;
                        predecessor_chain_truncated = true;
                        break;
                    }
                    predecessor_chain[predecessor_chain_count++] = cursor_shape;
                }
                u32 next_index = candidate_count;
                u64 latest_host{};
                for (u32 index = 0; index < candidate_count; ++index) {
                    const auto& candidate = candidates[index];
                    const bool eligible = candidate.valid && candidate.accepted &&
                                          candidate.predecessor_shape_exact &&
                                          candidate.predecessor_shape.Valid() &&
                                          candidate.retains_private_target_hint &&
                                          candidate.private_target_hint == learned &&
                                          candidate.guest_rendering_serial ==
                                              cursor.guest_rendering_serial &&
                                          candidate.host_rendering_instance_id <=
                                              cursor.host_rendering_instance_id;
                    if (!eligible) {
                        continue;
                    }
                    if (candidate.host_rendering_instance_id > latest_host) {
                        latest_host = candidate.host_rendering_instance_id;
                        next_index = index;
                        predecessor_chain_ambiguous = false;
                    } else if (candidate.host_rendering_instance_id == latest_host &&
                               next_index != candidate_count &&
                               candidate != candidates[next_index]) {
                        predecessor_chain_ambiguous = true;
                    }
                }
                if (predecessor_chain_ambiguous) {
                    predecessor_chain = {};
                    predecessor_chain_count = 0;
                    break;
                }
                if (next_index == candidate_count) {
                    predecessor_chain_exact = true;
                    private_writer_trace_exact = true;
                    break;
                }
                cursor = candidates[next_index].predecessor_endpoint;
                cursor_shape = candidates[next_index].predecessor_shape;
            }
            if (!predecessor_chain_exact && !predecessor_chain_ambiguous &&
                traversed > candidate_count) {
                predecessor_chain = {};
                predecessor_chain_count = 0;
                predecessor_chain_ambiguous = true;
            }
            if (predecessor_chain_ambiguous || private_writer_trace_truncated ||
                !private_writer_trace_exact) {
                private_writer_hints = {};
                private_writer_shapes = {};
                private_writer_count = 0;
                private_writer_trace_exact = false;
            }
        }
        if (classification ==
            PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch::None) {
            Increment(no_match_target_observations);
        } else if (classification ==
                   PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch::Unique) {
            Increment(unique_target_observations);
        } else {
            Increment(multiple_match_target_observations);
        }
        FreezeTarget(classification, matches,
                     first_writer_exact ? funnel_failure : first_writer_failure,
                     matched_shape, matched_shape_exact, predecessor_chain,
                     predecessor_chain_count, predecessor_chain_exact,
                     predecessor_chain_ambiguous, predecessor_chain_truncated, carry_age,
                     private_writer_hints, private_writer_shapes, private_writer_count,
                     private_writer_trace_exact);
    }

    [[nodiscard]] u32 RetainedPrivateStateCount() const noexcept {
        return candidate_count + static_cast<u32>(has_previous_target);
    }

    [[nodiscard]]
    PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidenceTakeResult
    TakeWithPrivateTraceArm(
        const PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidenceTakeDescriptor&
            descriptor) noexcept {
        PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidenceReport report{
            .sequence = descriptor.sequence,
            .distinct_regular_load_entries = distinct_regular_load_entries,
            .idempotent_regular_load_entries = idempotent_regular_load_entries,
            .diagnostic_resume_entries = diagnostic_resume_entries,
            .rejected_regular_load_entries = rejected_regular_load_entries,
            .regular_terminal_contradictions = regular_terminal_contradictions,
            .target_observation_count = target_observation_count,
            .exact_target_hook_count = exact_target_hook_count,
            .rejected_target_hooks = rejected_target_hooks,
            .target_terminal_contradictions = target_terminal_contradictions,
            .unique_target_observations = unique_target_observations,
            .no_match_target_observations = no_match_target_observations,
            .multiple_match_target_observations = multiple_match_target_observations,
            .duplicate_target_observations = duplicate_target_observations,
            .contradictory_target_observations = contradictory_target_observations,
            .target_guest_match_count = frozen_target_match_count,
            .first_writer_exact_targets = first_writer_exact_targets,
            .funnel = funnel,
            .frozen_first_failure = frozen_first_failure,
            .target_match = frozen_target_match,
            .predecessor_carry_age =
                frame_boundary_conflict
                    ? PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge::InvalidTransition
                    : frozen_predecessor_carry_age,
            .status = !active ? FinalGuestSurfaceStatus::AlreadyConsumed
                    : overflow ? FinalGuestSurfaceStatus::CapacityLoss
                    : frame_boundary_conflict || regular_terminal_contradictions != 0 ||
                              target_terminal_contradictions != 0
                        ? FinalGuestSurfaceStatus::InvalidationLoss
                        : FinalGuestSurfaceStatus::Complete,
            .loss = {.invalidation = !overflow &&
                                             (frame_boundary_conflict ||
                                              regular_terminal_contradictions != 0 ||
                                              target_terminal_contradictions != 0)
                                         ? 1u
                                         : 0u,
                     .history = overflow ? 1u : 0u},
            .predecessor_shape = frozen_predecessor_shape,
            .predecessor_chain = frozen_predecessor_chain,
            .predecessor_chain_count = frozen_predecessor_chain_count,
            .overflow = overflow,
            .counter_saturated = counter_saturated,
            .predecessor_shape_exact = frozen_predecessor_shape_exact,
            .predecessor_chain_exact = frozen_predecessor_chain_exact,
            .predecessor_chain_ambiguous = frozen_predecessor_chain_ambiguous,
            .predecessor_chain_truncated = frozen_predecessor_chain_truncated,
        };
        PpUpstreamSampleTeeProducerPreDepthScopeWriterPrivateTraceArm private_trace_arm{};
        if (report.status == FinalGuestSurfaceStatus::Complete && !report.loss.Any() &&
            report.target_match ==
                PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch::Unique &&
            report.predecessor_carry_age ==
                PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge::CurrentFrame &&
            frozen_private_writer_trace_exact && frozen_private_writer_count != 0 &&
            !report.predecessor_chain_ambiguous && !report.predecessor_chain_truncated) {
            private_trace_arm.writer_hints = frozen_private_writer_hints;
            private_trace_arm.writer_shapes = frozen_private_writer_shapes;
            private_trace_arm.writer_count = frozen_private_writer_count;
            private_trace_arm.exact = true;
        }
        *this = {};
        return {.report = report, .private_trace_arm = private_trace_arm};
    }

    [[nodiscard]] PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidenceReport Take(
        const PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidenceTakeDescriptor&
            descriptor) noexcept {
        return TakeWithPrivateTraceArm(descriptor).report;
    }

private:
    [[nodiscard]] static PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure
    ClassifyFirstWriterFailure(
        const PpUpstreamDepthFirstWriterPreObservation& pre) noexcept {
        if (!pre.recorded) {
            return PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::
                FirstWriterNotRecorded;
        }
        if (!pre.depth_stencil_exact) {
            return PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::
                FirstWriterDepthInexact;
        }
        if (!pre.same_scope) {
            return PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::Exact;
        }
        if (!pre.scope_depth_history_valid) {
            return PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::
                FirstWriterHistoryInvalid;
        }
        if (pre.first_depth_writer_valid) {
            return PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::
                FirstWriterAlreadyPresent;
        }
        if (pre.depth_writer_count != 0) {
            return PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::
                FirstWriterCountNonzero;
        }
        if (pre.depth_writer_count_overflow) {
            return PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::
                FirstWriterCountOverflow;
        }
        return PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::Exact;
    }

    void Increment(u32& value) noexcept {
        if (value != MaxPublicCount) {
            ++value;
        } else {
            counter_saturated = true;
        }
    }

    void FreezeTarget(PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch match,
                      u32 count,
                      PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure failure,
                      VideoCore::ImageDepthWriterShape predecessor_shape = {},
                      bool predecessor_shape_exact = false,
                      std::array<VideoCore::ImageDepthWriterShape,
                                 PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain>
                          predecessor_chain = {},
                      u32 predecessor_chain_count = 0,
                      bool predecessor_chain_exact = false,
                      bool predecessor_chain_ambiguous = false,
                      bool predecessor_chain_truncated = false,
                      PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge carry_age =
                          PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge::Unavailable,
                      std::array<PpDepthPredecessorHint,
                                 PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain>
                          private_writer_hints = {},
                      std::array<VideoCore::ImageDepthWriterShape,
                                 PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain>
                          private_writer_shapes = {},
                      u32 private_writer_count = 0,
                      bool private_writer_trace_exact = false) noexcept {
        if (!target_frozen) {
            frozen_target_match = match;
            frozen_target_match_count = count;
            frozen_first_failure = failure;
            frozen_predecessor_shape = predecessor_shape;
            frozen_predecessor_shape_exact = predecessor_shape_exact;
            frozen_predecessor_chain = predecessor_chain;
            frozen_predecessor_chain_count = predecessor_chain_count;
            frozen_predecessor_chain_exact = predecessor_chain_exact;
            frozen_predecessor_chain_ambiguous = predecessor_chain_ambiguous;
            frozen_predecessor_chain_truncated = predecessor_chain_truncated;
            frozen_predecessor_carry_age = carry_age;
            frozen_private_writer_hints = private_writer_hints;
            frozen_private_writer_shapes = private_writer_shapes;
            frozen_private_writer_count = private_writer_count;
            frozen_private_writer_trace_exact = private_writer_trace_exact;
            target_frozen = true;
        }
    }

    std::array<PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation, MaxCandidates>
        candidates{};
    u32 candidate_count{};
    u32 distinct_regular_load_entries{};
    u32 idempotent_regular_load_entries{};
    u32 diagnostic_resume_entries{};
    u32 rejected_regular_load_entries{};
    u32 regular_terminal_contradictions{};
    u32 target_observation_count{};
    u32 exact_target_hook_count{};
    u32 rejected_target_hooks{};
    u32 target_terminal_contradictions{};
    u32 unique_target_observations{};
    u32 no_match_target_observations{};
    u32 multiple_match_target_observations{};
    u32 duplicate_target_observations{};
    u32 contradictory_target_observations{};
    u32 frozen_target_match_count{};
    u32 first_writer_exact_targets{};
    std::array<PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelCount,
               static_cast<u32>(
                   PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::Count)>
        funnel{};
    u64 previous_target_guest{};
    u64 previous_target_host{};
    u64 frame_boundary_guest_rendering_serial{};
    PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch frozen_target_match{
        PpUpstreamSampleTeeProducerPreDepthScopeLoadTargetMatch::Idle};
    PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure frozen_first_failure{
        PpUpstreamSampleTeeProducerPreDepthScopeLoadFirstFailure::Idle};
    bool active{};
    bool frame_boundary_conflict{};
    bool overflow{};
    bool counter_saturated{};
    bool has_previous_target{};
    bool target_frozen{};
    VideoCore::ImageDepthWriterShape frozen_predecessor_shape{};
    bool frozen_predecessor_shape_exact{};
    std::array<VideoCore::ImageDepthWriterShape,
               PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain>
        frozen_predecessor_chain{};
    u32 frozen_predecessor_chain_count{};
    bool frozen_predecessor_chain_exact{};
    bool frozen_predecessor_chain_ambiguous{};
    bool frozen_predecessor_chain_truncated{};
    PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge frozen_predecessor_carry_age{
        PpUpstreamSampleTeeProducerPreDepthScopeLoadCarryAge::Unavailable};
    std::array<PpDepthPredecessorHint,
               PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain>
        frozen_private_writer_hints{};
    std::array<VideoCore::ImageDepthWriterShape,
               PpUpstreamSampleTeeProducerPreDepthScopeLoadMaxPredecessorChain>
        frozen_private_writer_shapes{};
    u32 frozen_private_writer_count{};
    bool frozen_private_writer_trace_exact{};
};

[[nodiscard]] inline std::string
FormatPpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidence(
    const PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidenceReport& report) {
    const auto stage = [&](PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage value) {
        const auto& count = report.funnel[static_cast<u32>(value)];
        return std::to_string(count.survivors) + "/" +
               std::to_string(count.targets_with_any);
    };
    auto line = "FGSCTPDSL s=" + std::to_string(report.sequence) +
           " d=" + std::to_string(report.distinct_regular_load_entries) +
           " i=" + std::to_string(report.idempotent_regular_load_entries) +
           " r=" + std::to_string(report.diagnostic_resume_entries) +
           " rr=" + std::to_string(report.rejected_regular_load_entries) +
           " rc=" + std::to_string(report.regular_terminal_contradictions) +
           " to=" + std::to_string(report.target_observation_count) +
           " te=" + std::to_string(report.exact_target_hook_count) +
           " tr=" + std::to_string(report.rejected_target_hooks) +
           " tc=" + std::to_string(report.target_terminal_contradictions) +
           " tn=" + std::to_string(report.no_match_target_observations) +
           " tu=" + std::to_string(report.unique_target_observations) +
           " tm=" + std::to_string(report.multiple_match_target_observations) +
           " td=" + std::to_string(report.duplicate_target_observations) +
           " mc=" + std::to_string(report.target_guest_match_count) +
           " cls=" + std::to_string(static_cast<u32>(report.target_match)) +
           " st=" + std::to_string(static_cast<u32>(report.status)) +
           " ov=" + std::to_string(report.overflow) +
           " ls=" + std::to_string(report.loss.history) + "/" +
           std::to_string(report.loss.invalidation) +
           " fw=" + std::to_string(report.first_writer_exact_targets) +
           " ca=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::
                     CanonicalCandidate) +
           " hi=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::LearnedFullHint) +
           " gu=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::TargetGuest) +
           " ho=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::HostOrder) +
           " pv=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::
                     PredecessorValid) +
           " pg=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::
                     PredecessorGuestBefore) +
           " ph=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::
                     PredecessorHostBefore) +
           " tv=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::TargetValid) +
           " id=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::ImageId) +
           " ui=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::Uid) +
           " rg=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::Range) +
           " bk=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::Backing) +
           " mu=" +
           stage(PpUpstreamSampleTeeProducerPreDepthScopeLoadFunnelStage::Mutation) +
           " ff=" + std::to_string(static_cast<u32>(report.frozen_first_failure)) +
           " psk=" + std::to_string(report.predecessor_shape_exact) +
           " pk=" + std::to_string(static_cast<u32>(report.predecessor_shape.kind)) +
           " pi=" + std::to_string(report.predecessor_shape.indexed) +
           " pe=" + std::to_string(report.predecessor_shape.element_count) +
           " pn=" + std::to_string(report.predecessor_shape.instance_count) +
           " ps=" + std::to_string(report.predecessor_shape.sampled_images) +
           " pw=" + std::to_string(report.predecessor_shape.storage_writes) +
           " pfg=" + std::to_string(report.predecessor_shape.fragment_pgm_hash) +
           " pt=" + std::to_string(report.predecessor_shape.guest_primitive_type) +
           " pte=" + std::to_string(report.predecessor_shape.effective_primitive_type) +
           " pq=" + std::to_string(report.predecessor_shape.quad_list_converted) +
           " pcm=" + std::to_string(report.predecessor_shape.cull_mode) +
           " pff=" + std::to_string(report.predecessor_shape.front_face) +
           " pmm=" + std::to_string(report.predecessor_shape.fragment_mrt_mask) +
           " pdc=" + std::to_string(report.predecessor_shape.depth_compare_op) +
           " pfd=" + std::to_string(report.predecessor_shape.fragment_writes_depth) +
           " pdi=" + std::to_string(report.predecessor_shape.fragment_has_discard) +
           " pdt=" + std::to_string(report.predecessor_shape.depth_test_enabled) +
           " pdb=" + std::to_string(report.predecessor_shape.depth_bounds_test_enabled) +
           " pds=" + std::to_string(report.predecessor_shape.depth_bias_enabled) +
           " pst=" + std::to_string(report.predecessor_shape.stencil_test_enabled) +
           " pac=" + std::to_string(report.predecessor_chain_count) +
           " pax=" + std::to_string(report.predecessor_chain_exact) +
           " paa=" + std::to_string(report.predecessor_chain_ambiguous) +
           " pat=" + std::to_string(report.predecessor_chain_truncated);
    line += " pca=" + std::to_string(static_cast<u32>(report.predecessor_carry_age));
    for (u32 index = 0; index < report.predecessor_chain.size(); ++index) {
        const auto& shape = report.predecessor_chain[index];
        const auto prefix = " p" + std::to_string(index);
        line += prefix + "k=" + std::to_string(static_cast<u32>(shape.kind)) +
                prefix + "i=" + std::to_string(shape.indexed) +
                prefix + "e=" + std::to_string(shape.element_count) +
                prefix + "n=" + std::to_string(shape.instance_count) +
                prefix + "s=" + std::to_string(shape.sampled_images) +
                prefix + "w=" + std::to_string(shape.storage_writes) +
                prefix + "h=" + std::to_string(shape.fragment_pgm_hash) +
                prefix + "t=" + std::to_string(shape.guest_primitive_type) +
                prefix + "te=" + std::to_string(shape.effective_primitive_type) +
                prefix + "q=" + std::to_string(shape.quad_list_converted) +
                prefix + "c=" + std::to_string(shape.cull_mode) +
                prefix + "f=" + std::to_string(shape.front_face) +
                prefix + "mm=" + std::to_string(shape.fragment_mrt_mask) +
                prefix + "dc=" + std::to_string(shape.depth_compare_op) +
                prefix + "fd=" + std::to_string(shape.fragment_writes_depth) +
                prefix + "di=" + std::to_string(shape.fragment_has_discard) +
                prefix + "dt=" + std::to_string(shape.depth_test_enabled) +
                prefix + "db=" + std::to_string(shape.depth_bounds_test_enabled) +
                prefix + "ds=" + std::to_string(shape.depth_bias_enabled) +
                prefix + "st=" + std::to_string(shape.stencil_test_enabled);
    }
    line += " sat=" + std::to_string(report.counter_saturated);
    return line;
}

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation
ResetPpUpstreamSampleTeeProducerPreDepthScopeEntry(
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation&) noexcept {
    return {};
}

struct PpUpstreamSampleTeeProducerPreDepthCollisionDescriptor {
    bool open{};
    PpDepthPredecessorWriterIntent open_writer{};
    PpDepthPredecessorWriterIntent incoming_writer{};
    bool prior_attempted{};
    bool prior_bound{};
};

struct PpUpstreamSampleTeeProducerPreDepthCollision {
    PpDepthPredecessorWriterIntent preserved_open_writer{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool collision{};
    bool accept_incoming{};
    bool overwrite_open{};
    bool clear_prior_binding{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthCollision
ObservePpUpstreamSampleTeeProducerPreDepthCollision(
    const PpUpstreamSampleTeeProducerPreDepthCollisionDescriptor& descriptor) noexcept {
    if (!descriptor.open && !descriptor.prior_attempted) {
        return {.preserved_open_writer = descriptor.incoming_writer,
                .status = FinalGuestSurfaceStatus::Complete,
                .accept_incoming = descriptor.incoming_writer.Valid()};
    }
    return {.preserved_open_writer = descriptor.open_writer,
            .status = FinalGuestSurfaceStatus::GapLoss,
            .loss = {.gap = 1},
            .collision = true,
            .clear_prior_binding = true};
}

struct PpUpstreamSampleTeeProducerPreDepthFrameDescriptor {
    PpUpstreamSampleTeeProducerPreDepthHint prior_hint{};
    PpUpstreamSampleTeeProducerPreDepthHint observed_selected_hint{};
    u32 selected_consumer_count{};
    u32 exact_selected_consumer_count{};
    bool producer_pre_attempted{};
    bool producer_pre_open{};
    bool producer_pre_bound{};
    bool selected_route_exact{};
    bool selected_view_exact{};
    bool selected_link_and_uid_exact{};
    bool selected_backing_exact{};
};

struct PpUpstreamSampleTeeProducerPreDepthFrameResult {
    PpUpstreamSampleTeeProducerPreDepthHint next_hint{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool warmup_completed{};
    bool sidecar_open{};
    bool retains_private_sidecar{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthFrameResult
FinishPpUpstreamSampleTeeProducerPreDepthFrame(
    const PpUpstreamSampleTeeProducerPreDepthFrameDescriptor& descriptor) noexcept {
    const bool drift = !descriptor.selected_route_exact || !descriptor.selected_view_exact ||
                       !descriptor.selected_link_and_uid_exact ||
                       !descriptor.selected_backing_exact;
    if (drift) {
        return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                .loss = {.invalidation = 1}};
    }
    const bool exact_consumer = descriptor.selected_consumer_count == 1 &&
                                descriptor.exact_selected_consumer_count == 1 &&
                                descriptor.observed_selected_hint.valid;
    const bool producer_exact = !descriptor.prior_hint.valid ||
                                (descriptor.producer_pre_attempted &&
                                 !descriptor.producer_pre_open && descriptor.producer_pre_bound);
    if (!exact_consumer || !producer_exact) {
        return {.status = FinalGuestSurfaceStatus::GapLoss, .loss = {.gap = 1}};
    }
    return {.next_hint = descriptor.observed_selected_hint,
            .status = FinalGuestSurfaceStatus::Complete,
            .warmup_completed = !descriptor.prior_hint.valid};
}

enum class PpUpstreamSampleTeeProducerPreDepthBufferKind : u8 {
    None,
    SharedDownload,
};

struct PpUpstreamSampleTeeProducerPreDepthDownloadTailDescriptor {
    bool enabled{};
    bool slot_present{};
    u32 slot{};
    u64 slot_stride{};
    u32 reserved_tail_bytes{};
    u32 buffer_alignment{};
    u32 base_total_bytes{};
    u32 input_total_bytes{};
    PpUpstreamInputContentPlane plane{};
    bool source_2d{};
    u32 source_samples{};
    bool depth_format_supported{};
    bool barrier_supported{};
    bool deferred_shared_release{};
};

struct PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan {
    PpUpstreamSampleTeeSelectedDepthPlane plane{};
    u64 slot_stride{};
    u32 slot{};
    u32 prefix_max_bytes{};
    u32 tail_offset{};
    u32 tail_bytes{};
    u32 buffer_alignment{};
    u32 base_total_bytes{};
    u32 input_total_bytes{};
    PpUpstreamSampleTeeProducerPreDepthBufferKind buffer_kind{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy{};
    bool uses_existing_outer_slot{};
    bool uses_tee_buffer{};
    bool owns_second_token{};
    bool base_prefix_non_overlapping{};
    bool input_prefix_non_overlapping{};
    bool barrier_supported{};
    bool deferred_shared_release{};
    bool acquired_before_current_target_arm{};
    bool observation_only{};
    bool affects_renderer_behavior{};

    bool operator==(
        const PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan& other) const noexcept {
        if (plane.region_count != other.plane.region_count ||
            plane.plane_offset != other.plane.plane_offset ||
            plane.plane_bytes != other.plane.plane_bytes ||
            plane.bytes_per_texel != other.plane.bytes_per_texel ||
            plane.base_mip != other.plane.base_mip || plane.mip_count != other.plane.mip_count ||
            plane.base_layer != other.plane.base_layer || plane.status != other.plane.status ||
            plane.loss != other.plane.loss) {
            return false;
        }
        for (u32 index = 0; index < plane.region_count; ++index) {
            const auto& a = plane.regions[index];
            const auto& b = other.plane.regions[index];
            if (a.logical_ordinal != b.logical_ordinal || a.mip_level != b.mip_level ||
                a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height ||
                a.buffer_offset != b.buffer_offset || a.byte_size != b.byte_size) {
                return false;
            }
        }
        return slot_stride == other.slot_stride && slot == other.slot &&
               prefix_max_bytes == other.prefix_max_bytes && tail_offset == other.tail_offset &&
               tail_bytes == other.tail_bytes && buffer_alignment == other.buffer_alignment &&
               base_total_bytes == other.base_total_bytes &&
               input_total_bytes == other.input_total_bytes && buffer_kind == other.buffer_kind &&
               status == other.status && loss == other.loss && copy == other.copy &&
               uses_existing_outer_slot == other.uses_existing_outer_slot &&
               uses_tee_buffer == other.uses_tee_buffer &&
               owns_second_token == other.owns_second_token &&
               base_prefix_non_overlapping == other.base_prefix_non_overlapping &&
               input_prefix_non_overlapping == other.input_prefix_non_overlapping &&
               barrier_supported == other.barrier_supported &&
               deferred_shared_release == other.deferred_shared_release &&
               acquired_before_current_target_arm == other.acquired_before_current_target_arm &&
               observation_only == other.observation_only &&
               affects_renderer_behavior == other.affects_renderer_behavior;
    }
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan
ValidatePpUpstreamSampleTeeProducerPreDepthDownloadTail(
    const PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan& plan) noexcept {
    const auto reject = [](FinalGuestSurfaceStatus status,
                           u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    if (plan.slot_stride != PpUpstreamInputSnapshotBytes ||
        plan.base_total_bytes > PpUpstreamSampleTeeDownloadPrefixBytes ||
        plan.input_total_bytes > PpUpstreamSampleTeeDownloadPrefixBytes ||
        plan.plane.plane_offset > std::numeric_limits<u32>::max() - plan.plane.plane_bytes ||
        static_cast<u64>(plan.plane.plane_offset) + plan.plane.plane_bytes > plan.slot_stride) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss,
                      &FinalGuestSurfaceLoss::byte_capacity);
    }
    if (plan.plane.region_count > FinalGuestSurfaceWatchOrdinals::MaxOrdinals) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss,
                      &FinalGuestSurfaceLoss::tile_capacity);
    }
    if (plan.prefix_max_bytes != PpUpstreamSampleTeeDownloadPrefixBytes ||
        plan.tail_offset != PpUpstreamSampleTeeProducerPreDepthPrefixBytes ||
        plan.tail_bytes != PpUpstreamSampleTeeProducerPreDepthTailBytes ||
        plan.buffer_alignment == 0 || !std::has_single_bit(plan.buffer_alignment) ||
        plan.plane.plane_offset % plan.buffer_alignment != 0 ||
        plan.plane.plane_offset < plan.tail_offset) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (plan.status != FinalGuestSurfaceStatus::Complete || plan.loss.Any() || !plan.copy ||
        plan.buffer_kind != PpUpstreamSampleTeeProducerPreDepthBufferKind::SharedDownload ||
        !plan.uses_existing_outer_slot || plan.uses_tee_buffer || plan.owns_second_token ||
        !plan.base_prefix_non_overlapping || !plan.input_prefix_non_overlapping ||
        !plan.barrier_supported || !plan.deferred_shared_release || !plan.observation_only ||
        plan.affects_renderer_behavior || plan.plane.status != FinalGuestSurfaceStatus::Complete ||
        plan.plane.loss.Any() || plan.plane.region_count == 0 || plan.plane.plane_bytes == 0 ||
        plan.plane.bytes_per_texel != 4 || plan.plane.base_mip != 0 ||
        plan.plane.mip_count != 1 || plan.plane.base_layer != 0) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    u64 covered{};
    for (u32 index = 0; index < plan.plane.region_count; ++index) {
        const auto& region = plan.plane.regions[index];
        if (region.logical_ordinal == 0 || region.mip_level != 0 || region.width == 0 ||
            region.height == 0 || region.byte_size == 0 ||
            region.byte_size % plan.plane.bytes_per_texel != 0 ||
            region.buffer_offset != covered || covered + region.byte_size > plan.plane.plane_bytes) {
            return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                          &FinalGuestSurfaceLoss::invalidation);
        }
        covered += region.byte_size;
    }
    if (covered != plan.plane.plane_bytes) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    return plan;
}

[[nodiscard]] inline bool IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(
    const PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan& plan) noexcept {
    const auto checked = ValidatePpUpstreamSampleTeeProducerPreDepthDownloadTail(plan);
    return checked.status == FinalGuestSurfaceStatus::Complete && !checked.loss.Any();
}

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan
PlanPpUpstreamSampleTeeProducerPreDepthDownloadTail(
    const PpUpstreamSampleTeeProducerPreDepthDownloadTailDescriptor& descriptor) noexcept {
    const auto reject = [](FinalGuestSurfaceStatus status,
                           u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    if (!descriptor.enabled) {
        return {};
    }
    if (!descriptor.slot_present) {
        return reject(FinalGuestSurfaceStatus::BusyLoss, &FinalGuestSurfaceLoss::busy);
    }
    if (!descriptor.source_2d || descriptor.source_samples != 1 ||
        !descriptor.depth_format_supported) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      !descriptor.depth_format_supported ? &FinalGuestSurfaceLoss::unsupported_format
                                                         : &FinalGuestSurfaceLoss::unsupported_type);
    }
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan result{
        .plane = descriptor.plane,
        .slot_stride = descriptor.slot_stride,
        .slot = descriptor.slot,
        .prefix_max_bytes = PpUpstreamSampleTeeDownloadPrefixBytes,
        .tail_offset = PpUpstreamSampleTeeProducerPreDepthPrefixBytes,
        .tail_bytes = descriptor.reserved_tail_bytes,
        .buffer_alignment = descriptor.buffer_alignment,
        .base_total_bytes = descriptor.base_total_bytes,
        .input_total_bytes = descriptor.input_total_bytes,
        .buffer_kind = PpUpstreamSampleTeeProducerPreDepthBufferKind::SharedDownload,
        .status = FinalGuestSurfaceStatus::Complete,
        .copy = true,
        .uses_existing_outer_slot = true,
        .base_prefix_non_overlapping =
            descriptor.base_total_bytes <= PpUpstreamSampleTeeDownloadPrefixBytes,
        .input_prefix_non_overlapping =
            descriptor.input_total_bytes <= PpUpstreamSampleTeeDownloadPrefixBytes,
        .barrier_supported = descriptor.barrier_supported,
        .deferred_shared_release = descriptor.deferred_shared_release,
        .observation_only = true};
    return ValidatePpUpstreamSampleTeeProducerPreDepthDownloadTail(result);
}

using PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailDescriptor =
    PpUpstreamSampleTeeProducerPreDepthDownloadTailDescriptor;
using PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan =
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan;

[[nodiscard]] inline PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan
ValidatePpUpstreamSampleTeePredecessorWriterPreDepthDownloadTail(
    const PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan& plan) noexcept {
    const auto reject = [](FinalGuestSurfaceStatus status,
                           u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    if (plan.slot_stride != PpUpstreamInputSnapshotBytes ||
        plan.base_total_bytes > PpUpstreamSampleTeeDownloadPrefixBytes ||
        plan.input_total_bytes > PpUpstreamSampleTeeDownloadPrefixBytes ||
        plan.plane.plane_offset > std::numeric_limits<u32>::max() - plan.plane.plane_bytes ||
        static_cast<u64>(plan.plane.plane_offset) + plan.plane.plane_bytes > plan.slot_stride ||
        static_cast<u64>(plan.plane.plane_offset) + plan.plane.plane_bytes >
            PpUpstreamSampleTeeProducerPreDepthTailOffset) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss,
                      &FinalGuestSurfaceLoss::byte_capacity);
    }
    if (plan.plane.region_count > FinalGuestSurfaceWatchOrdinals::MaxOrdinals) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss,
                      &FinalGuestSurfaceLoss::tile_capacity);
    }
    if (plan.prefix_max_bytes != PpUpstreamSampleTeeDownloadPrefixBytes ||
        plan.tail_offset != PpUpstreamSampleTeePredecessorWriterPreDepthTailOffset ||
        plan.tail_bytes != PpUpstreamSampleTeePredecessorWriterPreDepthTailBytes ||
        plan.buffer_alignment == 0 || !std::has_single_bit(plan.buffer_alignment) ||
        plan.plane.plane_offset % plan.buffer_alignment != 0 ||
        plan.plane.plane_offset < plan.tail_offset ||
        plan.status != FinalGuestSurfaceStatus::Complete || plan.loss.Any() || !plan.copy ||
        plan.buffer_kind != PpUpstreamSampleTeeProducerPreDepthBufferKind::SharedDownload ||
        !plan.uses_existing_outer_slot || plan.uses_tee_buffer || plan.owns_second_token ||
        !plan.base_prefix_non_overlapping || !plan.input_prefix_non_overlapping ||
        !plan.barrier_supported || !plan.deferred_shared_release ||
        !plan.acquired_before_current_target_arm || !plan.observation_only ||
        plan.affects_renderer_behavior || plan.plane.status != FinalGuestSurfaceStatus::Complete ||
        plan.plane.loss.Any() || plan.plane.region_count == 0 || plan.plane.plane_bytes == 0 ||
        plan.plane.bytes_per_texel != 4 || plan.plane.base_mip != 0 ||
        plan.plane.mip_count != 1 || plan.plane.base_layer != 0) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    u64 covered{};
    for (u32 index = 0; index < plan.plane.region_count; ++index) {
        const auto& region = plan.plane.regions[index];
        if (region.logical_ordinal == 0 || region.mip_level != 0 || region.width == 0 ||
            region.height == 0 || region.byte_size == 0 ||
            region.byte_size % plan.plane.bytes_per_texel != 0 ||
            region.buffer_offset != covered || covered + region.byte_size > plan.plane.plane_bytes) {
            return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                          &FinalGuestSurfaceLoss::invalidation);
        }
        covered += region.byte_size;
    }
    if (covered != plan.plane.plane_bytes) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    return plan;
}

[[nodiscard]] inline bool
IsPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailRecordable(
    const PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan& plan) noexcept {
    const auto checked =
        ValidatePpUpstreamSampleTeePredecessorWriterPreDepthDownloadTail(plan);
    return checked.status == FinalGuestSurfaceStatus::Complete && !checked.loss.Any();
}

[[nodiscard]] inline PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan
PlanPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTail(
    const PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailDescriptor& descriptor) noexcept {
    const auto reject = [](FinalGuestSurfaceStatus status,
                           u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    if (!descriptor.enabled) {
        return {};
    }
    if (!descriptor.slot_present) {
        return reject(FinalGuestSurfaceStatus::BusyLoss, &FinalGuestSurfaceLoss::busy);
    }
    if (!descriptor.source_2d || descriptor.source_samples != 1 ||
        !descriptor.depth_format_supported) {
        return reject(FinalGuestSurfaceStatus::Unsupported,
                      !descriptor.depth_format_supported ? &FinalGuestSurfaceLoss::unsupported_format
                                                         : &FinalGuestSurfaceLoss::unsupported_type);
    }
    PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan result{
        .plane = descriptor.plane,
        .slot_stride = descriptor.slot_stride,
        .slot = descriptor.slot,
        .prefix_max_bytes = PpUpstreamSampleTeeDownloadPrefixBytes,
        .tail_offset = PpUpstreamSampleTeePredecessorWriterPreDepthTailOffset,
        .tail_bytes = descriptor.reserved_tail_bytes,
        .buffer_alignment = descriptor.buffer_alignment,
        .base_total_bytes = descriptor.base_total_bytes,
        .input_total_bytes = descriptor.input_total_bytes,
        .buffer_kind = PpUpstreamSampleTeeProducerPreDepthBufferKind::SharedDownload,
        .status = FinalGuestSurfaceStatus::Complete,
        .copy = true,
        .uses_existing_outer_slot = true,
        .base_prefix_non_overlapping =
            descriptor.base_total_bytes <= PpUpstreamSampleTeeDownloadPrefixBytes,
        .input_prefix_non_overlapping =
            descriptor.input_total_bytes <= PpUpstreamSampleTeeDownloadPrefixBytes,
        .barrier_supported = descriptor.barrier_supported,
        .deferred_shared_release = descriptor.deferred_shared_release,
        .acquired_before_current_target_arm = true,
        .observation_only = true};
    return ValidatePpUpstreamSampleTeePredecessorWriterPreDepthDownloadTail(result);
}

struct PpUpstreamSampleTeeDepthDownloadTails {
    PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan predecessor_writer_pre{};
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan producer_pre{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool non_overlapping{};
    bool same_outer_slot{};
    bool owns_second_token{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_release_count{};

    bool operator==(const PpUpstreamSampleTeeDepthDownloadTails&) const = default;
};

[[nodiscard]] inline PpUpstreamSampleTeeDepthDownloadTails
ValidatePpUpstreamSampleTeeDepthDownloadTails(
    const PpUpstreamSampleTeeDepthDownloadTails& tails) noexcept {
    const u64 predecessor_end = static_cast<u64>(tails.predecessor_writer_pre.plane.plane_offset) +
                                tails.predecessor_writer_pre.plane.plane_bytes;
    const bool exact =
        IsPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailRecordable(
            tails.predecessor_writer_pre) &&
        IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(tails.producer_pre) &&
        predecessor_end <= tails.producer_pre.plane.plane_offset &&
        tails.predecessor_writer_pre.slot == tails.producer_pre.slot &&
        tails.predecessor_writer_pre.slot_stride == tails.producer_pre.slot_stride &&
        tails.status == FinalGuestSurfaceStatus::Complete && !tails.loss.Any() &&
        tails.non_overlapping && tails.same_outer_slot && !tails.owns_second_token &&
        tails.additional_gpu_copy_count == 1 && tails.additional_slot_count == 0 &&
        tails.additional_release_count == 0;
    if (!exact) {
        return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                .loss = {.invalidation = 1}};
    }
    return tails;
}

struct PpUpstreamSampleTeeDepthDownloadTailsDescriptor {
    PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan predecessor_writer_pre{};
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan producer_pre{};
};

[[nodiscard]] inline PpUpstreamSampleTeeDepthDownloadTails
PlanPpUpstreamSampleTeeDepthDownloadTails(
    const PpUpstreamSampleTeeDepthDownloadTailsDescriptor& descriptor) noexcept {
    PpUpstreamSampleTeeDepthDownloadTails result{
        .predecessor_writer_pre = descriptor.predecessor_writer_pre,
        .producer_pre = descriptor.producer_pre,
        .status = FinalGuestSurfaceStatus::Complete,
        .non_overlapping = true,
        .same_outer_slot = true,
        .additional_gpu_copy_count = 1};
    return ValidatePpUpstreamSampleTeeDepthDownloadTails(result);
}

struct PpUpstreamSampleTeeDepthDownloadTailsOrderingDescriptor {
    PpUpstreamSampleTeeDepthDownloadTails tails{};
    u64 absolute_slot_offset{};
    bool predecessor_copy_before_writer{};
    bool producer_copy_before_target_writer{};
    bool base_copy_pending{};
};

struct PpUpstreamSampleTeeDepthDownloadTailsOrdering {
    PpUpstreamSampleTeeBarrierPlan predecessor_pre_copy{};
    PpUpstreamSampleTeeBarrierPlan producer_pre_copy{};
    PpUpstreamSampleTeeBarrierPlan base_prefix{};
    PpUpstreamSampleTeeBarrierPlan host_read{};
    bool valid{};
    bool producer_tail_first_use_independent{};
    bool predecessor_tail_ordered_before_producer_tail{};
    bool predecessor_before_its_writer{};
    bool producer_before_target_writer{};
    bool base_prefix_first_use_independent{};
};

[[nodiscard]] inline PpUpstreamSampleTeeDepthDownloadTailsOrdering
PlanPpUpstreamSampleTeeDepthDownloadTailsOrdering(
    const PpUpstreamSampleTeeDepthDownloadTailsOrderingDescriptor& descriptor) noexcept {
    const auto checked = ValidatePpUpstreamSampleTeeDepthDownloadTails(descriptor.tails);
    if (checked.status != FinalGuestSurfaceStatus::Complete ||
        descriptor.absolute_slot_offset > std::numeric_limits<u32>::max() ||
        !descriptor.predecessor_copy_before_writer ||
        !descriptor.producer_copy_before_target_writer) {
        return {};
    }
    constexpr auto SrcStages = PpUpstreamSampleTeeStage::AllCommands |
                               PpUpstreamSampleTeeStage::Host;
    constexpr auto SrcAccess = PpUpstreamSampleTeeAccess::MemoryWrite |
                               PpUpstreamSampleTeeAccess::HostRead;
    const auto copy_barrier = [&](const auto& tail) {
        return PpUpstreamSampleTeeBarrierPlan{
            .src_stages = SrcStages,
            .src_access = SrcAccess,
            .dst_stages = PpUpstreamSampleTeeStage::Transfer,
            .dst_access = PpUpstreamSampleTeeAccess::TransferWrite,
            .offset = static_cast<u32>(descriptor.absolute_slot_offset) +
                      tail.plane.plane_offset,
            .bytes = tail.plane.plane_bytes};
    };
    return {
        .predecessor_pre_copy = copy_barrier(checked.predecessor_writer_pre),
        .producer_pre_copy = copy_barrier(checked.producer_pre),
        .base_prefix = {.src_stages = SrcStages,
                        .src_access = SrcAccess,
                        .dst_stages = PpUpstreamSampleTeeStage::Transfer,
                        .dst_access = PpUpstreamSampleTeeAccess::TransferWrite,
                        .offset = static_cast<u32>(descriptor.absolute_slot_offset),
                        .bytes = PpUpstreamSampleTeeDownloadPrefixBytes},
        .host_read = {.src_stages = PpUpstreamSampleTeeStage::Transfer,
                      .src_access = PpUpstreamSampleTeeAccess::TransferWrite,
                      .dst_stages = PpUpstreamSampleTeeStage::Host,
                      .dst_access = PpUpstreamSampleTeeAccess::HostRead,
                      .offset = static_cast<u32>(descriptor.absolute_slot_offset) +
                                PpUpstreamSampleTeePredecessorWriterPreDepthTailOffset,
                      .bytes = PpUpstreamSampleTeePredecessorWriterPreDepthTailBytes +
                               PpUpstreamSampleTeeProducerPreDepthTailBytes},
        .valid = true,
        .producer_tail_first_use_independent = true,
        .predecessor_tail_ordered_before_producer_tail = true,
        .predecessor_before_its_writer = true,
        .producer_before_target_writer = true,
        .base_prefix_first_use_independent = descriptor.base_copy_pending};
}

struct PpUpstreamSampleTeeProducerPreDepthScopeEntryPlanDescriptor {
    PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation entry{};
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan producer_pre_plan{};
    PpDepthPredecessorWriterIntent target_intent{};
    PpUpstreamDepthFirstWriterPreObservation first_writer_pre{};
    u64 pre_draw_host_rendering_instance_id{};
    u32 render_width{};
    u32 render_height{};
    u32 render_layers{};
    bool watched_regions_inside_render_area{};
};

struct PpUpstreamSampleTeeProducerPreDepthScopeEntryPlan {
    PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation entry{};
    PpDepthPredecessorWriterIntent target_intent{};
    PpUpstreamDepthFirstWriterPreObservation first_writer_pre{};
    PpUpstreamSampleTeeSelectedDepthPlane producer_pre_plane{};
    u64 pre_draw_host_rendering_instance_id{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool record_metadata{};
    bool metadata_only{};
    bool copy_content{};
    u32 additional_gpu_copy_count{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthScopeEntryPlan
PlanPpUpstreamSampleTeeProducerPreDepthScopeEntryProvenance(
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryPlanDescriptor& descriptor) noexcept {
    const auto reject = [] {
        return PpUpstreamSampleTeeProducerPreDepthScopeEntryPlan{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1}};
    };
    const auto& entry = descriptor.entry;
    const auto& predecessor = entry.predecessor_endpoint;
    const auto& target = descriptor.target_intent;
    const auto& pre = descriptor.first_writer_pre;
    const u64 pre_draw_host = descriptor.pre_draw_host_rendering_instance_id != 0
                                  ? descriptor.pre_draw_host_rendering_instance_id
                                  : entry.host_rendering_instance_id;
    const bool predecessor_before_entry = predecessor.Valid() &&
                                          predecessor.guest_rendering_serial <
                                              entry.guest_rendering_serial &&
                                          predecessor.host_rendering_instance_id <
                                              entry.host_rendering_instance_id;
    const bool predecessor_join = predecessor_before_entry && target.Valid() &&
                                  predecessor.writer.link == target.link &&
                                  predecessor.writer.range == target.range &&
                                  predecessor.writer.backing_generation ==
                                      target.backing_generation &&
                                  predecessor.writer.mutation_epoch == target.mutation_epoch;
    const bool retained_target_join =
        !entry.retains_private_target_hint ||
        (entry.private_target_hint.valid && entry.private_target_hint.private_only &&
         !entry.private_target_hint.logs_private_state &&
         entry.private_target_hint.writer == target.hint &&
         entry.private_target_hint.link == target.link &&
         entry.private_target_hint.range == target.range &&
         entry.private_target_hint.backing_generation == target.backing_generation);
    const bool no_prior_writer_in_scope =
        pre.recorded && pre.depth_stencil_exact &&
        (!pre.same_scope ||
         (pre.scope_depth_history_valid && !pre.first_depth_writer_valid &&
          pre.depth_writer_count == 0 && !pre.depth_writer_count_overflow));
    const bool exact_entry = entry.valid && entry.accepted && entry.metadata_only &&
                             entry.status == FinalGuestSurfaceStatus::Complete &&
                             !entry.loss.Any() &&
                             entry.load_op !=
                                 PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::None &&
                             entry.retains_private_predecessor_endpoint &&
                             (entry.load_op ==
                                      PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Clear
                                  ? entry.has_private_depth_clear_bits
                                  : !entry.has_private_depth_clear_bits);
    auto canonical_producer_pre_plan = descriptor.producer_pre_plan;
    if (canonical_producer_pre_plan.prefix_max_bytes ==
            PpUpstreamSampleTeeProducerPreDepthPrefixBytes &&
        canonical_producer_pre_plan.base_total_bytes == 0 &&
        canonical_producer_pre_plan.input_total_bytes == 0) {
        canonical_producer_pre_plan.prefix_max_bytes = PpUpstreamSampleTeeDownloadPrefixBytes;
    }
    if (!exact_entry || pre_draw_host < entry.host_rendering_instance_id ||
        !predecessor_join || !retained_target_join || !no_prior_writer_in_scope ||
        !IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(
            canonical_producer_pre_plan) ||
        descriptor.render_width == 0 || descriptor.render_height == 0 ||
        descriptor.render_layers != 1 || !descriptor.watched_regions_inside_render_area) {
        return reject();
    }
    return {.entry = entry,
            .target_intent = target,
            .first_writer_pre = pre,
            .producer_pre_plane = descriptor.producer_pre_plan.plane,
            .pre_draw_host_rendering_instance_id = pre_draw_host,
            .status = FinalGuestSurfaceStatus::Complete,
            .record_metadata = true,
            .metadata_only = true};
}

struct PpUpstreamSampleTeeProducerPreDepthScopeEntryOwned {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp load_op{
        PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::None};
    PpDepthPredecessorEndpointStamp private_predecessor_endpoint{};
    u32 private_depth_clear_bits{};
    bool target_first_depth_writer{};
    bool scope_clear{};
    bool scope_load{};
    bool prior_endpoint_owns_pre_content{};
    bool has_private_depth_clear_bits{};
    bool raw_depth_clear_encoding_exact{};
    bool metadata_only{};
    bool exact{};
    bool retains_private_predecessor_endpoint{};
    u32 additional_gpu_copy_count{};

    bool operator==(
        const PpUpstreamSampleTeeProducerPreDepthScopeEntryOwned&) const = default;
};

struct PpUpstreamSampleTeePredecessorWriterPreDepthLearnDescriptor {
    bool enabled{};
    bool sample_tee_enabled{};
    PpUpstreamSampleTeeProducerPreDepthScopeEntryOwned bound_scope{};
    u32 bound_scope_count{};
    PpUpstreamSampleTeeSelectedDepthToken selected_token{};
    PpDepthPredecessorEndpointStamp selected_endpoint{};
    VideoCore::SubresourceRange selected_resolved_range{};
    bool selected_source_still_live{};
    bool selected_identity_exact{};
    bool selected_backing_exact{};
    bool selected_mutation_exact{};
    bool selected_route_exact{};
    bool selected_view_exact{};
    bool selected_depth_aspect{};
    bool selected_depth_write{};
};

[[nodiscard]] inline PpUpstreamSampleTeePredecessorWriterPreDepthHint
LearnPpUpstreamSampleTeePredecessorWriterPreDepthHint(
    const PpUpstreamSampleTeePredecessorWriterPreDepthLearnDescriptor& descriptor) noexcept {
    const auto& scope = descriptor.bound_scope;
    const auto& predecessor = scope.private_predecessor_endpoint;
    const auto& selected = descriptor.selected_endpoint;
    const auto& token = descriptor.selected_token;
    const bool scope_exact = descriptor.bound_scope_count == 1 &&
                             scope.status == FinalGuestSurfaceStatus::Complete &&
                             !scope.loss.Any() &&
                             scope.load_op ==
                                 PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Load &&
                             scope.target_first_depth_writer && scope.scope_load &&
                             scope.prior_endpoint_owns_pre_content && scope.metadata_only &&
                             scope.exact && scope.retains_private_predecessor_endpoint &&
                             predecessor.Valid() && predecessor.writer.hint.Valid() &&
                             predecessor.writer.hint.fragment_pgm_hash != 0 &&
                             predecessor.writer.mutation_epoch != 0;
    const bool selected_exact = token.Valid() && selected.Valid() &&
                                token.link == selected.writer.link &&
                                token.backing_generation == selected.writer.backing_generation &&
                                token.mutation_epoch == selected.writer.mutation_epoch &&
                                selected.writer.range == descriptor.selected_resolved_range;
    const bool same_loaded_source = selected.writer.link == predecessor.writer.link &&
                                    selected.writer.range == predecessor.writer.range &&
                                    selected.writer.backing_generation ==
                                        predecessor.writer.backing_generation &&
                                    predecessor.writer.mutation_epoch !=
                                        std::numeric_limits<u64>::max() &&
                                    selected.writer.mutation_epoch ==
                                        predecessor.writer.mutation_epoch + 1;
    if (!descriptor.enabled || !descriptor.sample_tee_enabled || !scope_exact ||
        !selected_exact || !same_loaded_source || !descriptor.selected_source_still_live ||
        !descriptor.selected_identity_exact || !descriptor.selected_backing_exact ||
        !descriptor.selected_mutation_exact || !descriptor.selected_route_exact ||
        !descriptor.selected_view_exact || !descriptor.selected_depth_aspect ||
        !descriptor.selected_depth_write) {
        return {};
    }
    return {.link = predecessor.writer.link,
            .range = predecessor.writer.range,
            .backing_generation = predecessor.writer.backing_generation,
            .writer = predecessor.writer.hint,
            .valid = true,
            .private_only = true};
}

struct PpUpstreamSampleTeePredecessorWriterPreDepthBindDescriptor {
    PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan predecessor_pre_plan{};
    PpUpstreamSampleTeeProducerPreDepthPost predecessor_post{};
    PpUpstreamSampleTeeProducerPreDepthScopeEntryOwned later_load{};
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan current_producer_pre_plan{};
    PpUpstreamSampleTeeProducerPreDepthBinding selected_binding{};
    u32 pre_observation_count{};
    u32 post_observation_count{};
    u32 load_observation_count{};
    u32 selected_binding_count{};
    bool later_route_exact{};
    bool later_view_exact{};
    bool later_selected_token_current{};
};

struct PpUpstreamSampleTeePredecessorWriterPreDepthBinding {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool exact{};
    bool publish_predecessor_pre{};
    bool post_endpoint_equals_load_predecessor{};
    bool current_producer_pre_is_predecessor_post_proxy{};
    bool semantic_layout_exact{};
    bool copy_predecessor_post_content{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_release_count{};
    bool retains_private_endpoint{};
    bool affects_renderer_behavior{};
};

[[nodiscard]] inline bool IsPpUpstreamSampleTeeDepthPlaneSemanticLayoutExact(
    const PpUpstreamSampleTeeSelectedDepthPlane& a,
    const PpUpstreamSampleTeeSelectedDepthPlane& b) noexcept {
    if (a.region_count != b.region_count || a.plane_bytes != b.plane_bytes ||
        a.bytes_per_texel != b.bytes_per_texel || a.base_mip != b.base_mip ||
        a.mip_count != b.mip_count || a.base_layer != b.base_layer) {
        return false;
    }
    for (u32 index = 0; index < a.region_count; ++index) {
        const auto& first = a.regions[index];
        const auto& second = b.regions[index];
        if (first.logical_ordinal != second.logical_ordinal ||
            first.mip_level != second.mip_level || first.x != second.x ||
            first.y != second.y || first.width != second.width ||
            first.height != second.height || first.buffer_offset != second.buffer_offset ||
            first.byte_size != second.byte_size) {
            return false;
        }
    }
    return true;
}

enum class PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindingMode : u8 {
    None,
    FreshSelectedPair,
    CapturePhaseAwaitingFreshPair,
    DirectSelectedSameFrame,
};

struct PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindingMode mode{
        PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindingMode::None};
    bool exact{};
    bool publish_predecessor_pre{};
    bool publish_target_pre{};
    bool predecessor_post_equals_fresh_predecessor{};
    bool target_pre_post_equals_fresh_target{};
    bool fresh_selected_pair_latest_exact{};
    bool direct_selected_same_frame_exact{};
    bool predecessor_post_equals_target_immediate_predecessor{};
    bool target_post_equals_direct_selected{};
    bool generic_nonadmission_exact{};
    bool ordinal_pair_exact{};
    bool semantic_layout_exact{};
    bool consumes_scope_entry{};
    bool affects_pds{};
    u32 existing_gpu_copy_count{};
    u32 existing_token_count{};
    u32 existing_download_invalidation_count{};
    u32 existing_release_count{};
    u32 additional_gpu_copy_count{};
    u32 additional_slot_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};
    bool capture_phase_exact{};
    bool awaiting_fresh_selected_pair{};
    bool legacy_target_pre_unchanged{};
    bool pds_unchanged{};
    bool retains_private_endpoints{};

    bool operator==(
        const PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding&) const = default;
};

struct PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindDescriptor {
    PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm prior_arm{};
    PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorPair fresh_selected_pair{};
    u32 fresh_selected_pair_count{};
    PpUpstreamSampleTeeDepthDownloadTails combined_tails{};
    PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan predecessor_pre_plan{};
    PpDepthPredecessorWriterIntent predecessor_pre_intent{};
    PpUpstreamSampleTeeProducerPreDepthPost predecessor_post{};
    u32 predecessor_scope_writer_count{};
    u32 predecessor_pre_observation_count{};
    u32 predecessor_post_observation_count{};
    bool predecessor_open{};
    bool predecessor_collision{};
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan target_pre_plan{};
    PpDepthPredecessorWriterIntent target_pre_intent{};
    PpUpstreamSampleTeeProducerPreDepthPost target_post{};
    u32 target_scope_writer_count{};
    u32 target_pre_observation_count{};
    u32 target_post_observation_count{};
    bool target_open{};
    bool target_collision{};
    PpUpstreamSampleTeeProducerPreDepthBinding selected_binding{};
    u32 selected_binding_count{};
    bool later_route_exact{};
    bool later_view_exact{};
    bool later_selected_token_current{};
};

[[nodiscard]] inline bool IsExactPpUpstreamSampleTeeOrdinalArmForPair(
    const PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm& arm,
    const PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorPair& pair) noexcept {
    return arm.valid && arm.private_only && !arm.logs_private_state &&
           arm.status == FinalGuestSurfaceStatus::Complete && !arm.loss.Any() &&
           arm.predecessor_hint.valid && arm.predecessor_hint.private_only &&
           !arm.predecessor_hint.logs_private_state && arm.target_hint.valid &&
           arm.target_hint.private_only && !arm.target_hint.logs_private_state &&
           arm.predecessor_scope_writer_count ==
               pair.expected_predecessor_pre_scope_writer_count &&
           arm.target_scope_writer_count == pair.selected_target_scope_writer_count &&
           arm.predecessor_hint.link == pair.learned_hint.link &&
           arm.predecessor_hint.range == pair.learned_hint.range &&
           arm.predecessor_hint.backing_generation == pair.learned_hint.backing_generation &&
           arm.predecessor_hint.writer == pair.learned_hint.writer &&
           arm.target_hint.link == pair.target_pre_intent.link &&
           arm.target_hint.range == pair.target_pre_intent.range &&
           arm.target_hint.backing_generation == pair.target_pre_intent.backing_generation &&
           arm.target_hint.writer == pair.target_pre_intent.hint;
}

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding
BindPpUpstreamSampleTeeProducerPreDepthOrdinalContentToFreshSelectedPair(
    const PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindDescriptor& descriptor) noexcept {
    const auto reject = [](FinalGuestSurfaceStatus status,
                           u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding result{
            .status = status,
            .legacy_target_pre_unchanged = true,
            .pds_unchanged = true};
        result.loss.*member = 1;
        return result;
    };
    const auto& tails = descriptor.combined_tails;
    const auto exceeds_slot = [](const auto& tail) {
        return tail.plane.plane_offset > std::numeric_limits<u32>::max() -
                                             tail.plane.plane_bytes ||
               static_cast<u64>(tail.plane.plane_offset) + tail.plane.plane_bytes >
                   tail.slot_stride;
    };
    if (exceeds_slot(tails.predecessor_writer_pre) || exceeds_slot(tails.producer_pre)) {
        return reject(FinalGuestSurfaceStatus::CapacityLoss,
                      &FinalGuestSurfaceLoss::tile_capacity);
    }
    const auto propagate_tail_loss = [&](const auto& plan)
        -> std::optional<PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding> {
        if (plan.status != FinalGuestSurfaceStatus::Complete &&
            plan.status != FinalGuestSurfaceStatus::AlreadyConsumed && plan.loss.Any()) {
            PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding result{
                .status = plan.status,
                .loss = plan.loss,
                .legacy_target_pre_unchanged = true,
                .pds_unchanged = true};
            return result;
        }
        return std::nullopt;
    };
    if (const auto loss = propagate_tail_loss(descriptor.predecessor_pre_plan)) {
        return *loss;
    }
    if (const auto loss = propagate_tail_loss(descriptor.target_pre_plan)) {
        return *loss;
    }
    const bool capture_phase = descriptor.fresh_selected_pair_count == 0 &&
                               descriptor.selected_binding_count == 0;
    if (capture_phase) {
        if (descriptor.predecessor_pre_observation_count != 1 ||
            descriptor.predecessor_post_observation_count != 1 ||
            descriptor.target_pre_observation_count != 1 ||
            descriptor.target_post_observation_count != 1 || descriptor.predecessor_open ||
            descriptor.target_open || descriptor.predecessor_collision ||
            descriptor.target_collision) {
            return reject(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
        }
        const auto& arm = descriptor.prior_arm;
        const auto& predecessor_post = descriptor.predecessor_post.output;
        const auto& target_post = descriptor.target_post.output;
        const bool unused_selected_state =
            !descriptor.fresh_selected_pair.valid &&
            !descriptor.fresh_selected_pair.predecessor_endpoint.Valid() &&
            !descriptor.fresh_selected_pair.target_pre_intent.Valid() &&
            !descriptor.fresh_selected_pair.target_post_endpoint.Valid() &&
            descriptor.selected_binding.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            !descriptor.selected_binding.loss.Any() && !descriptor.selected_binding.exact &&
            !descriptor.selected_binding.publish_producer_pre && !descriptor.later_route_exact &&
            !descriptor.later_view_exact && !descriptor.later_selected_token_current;
        const bool tails_exact =
            ValidatePpUpstreamSampleTeeDepthDownloadTails(tails).status ==
                FinalGuestSurfaceStatus::Complete &&
            IsPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailRecordable(
                descriptor.predecessor_pre_plan) &&
            IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(
                descriptor.target_pre_plan) &&
            tails.predecessor_writer_pre == descriptor.predecessor_pre_plan &&
            tails.producer_pre == descriptor.target_pre_plan &&
            IsPpUpstreamSampleTeeDepthPlaneSemanticLayoutExact(
                tails.predecessor_writer_pre.plane, tails.producer_pre.plane);
        const bool predecessor_exact =
            IsExactPpUpstreamSampleTeeOrdinalArmHint(arm.predecessor_hint,
                                                     descriptor.predecessor_pre_intent) &&
            descriptor.predecessor_post.status == FinalGuestSurfaceStatus::Complete &&
            !descriptor.predecessor_post.loss.Any() && descriptor.predecessor_post.exact &&
            predecessor_post.Valid() &&
            predecessor_post.writer.hint == descriptor.predecessor_pre_intent.hint &&
            predecessor_post.writer.link == descriptor.predecessor_pre_intent.link &&
            predecessor_post.writer.range == descriptor.predecessor_pre_intent.range &&
            predecessor_post.writer.backing_generation ==
                descriptor.predecessor_pre_intent.backing_generation &&
            descriptor.predecessor_pre_intent.mutation_epoch !=
                std::numeric_limits<u64>::max() &&
            predecessor_post.writer.mutation_epoch ==
                descriptor.predecessor_pre_intent.mutation_epoch + 1;
        const bool target_exact =
            IsExactPpUpstreamSampleTeeOrdinalArmHint(arm.target_hint,
                                                     descriptor.target_pre_intent) &&
            descriptor.target_post.status == FinalGuestSurfaceStatus::Complete &&
            !descriptor.target_post.loss.Any() && descriptor.target_post.exact &&
            target_post.Valid() && target_post.writer.hint == descriptor.target_pre_intent.hint &&
            target_post.writer.link == descriptor.target_pre_intent.link &&
            target_post.writer.range == descriptor.target_pre_intent.range &&
            target_post.writer.backing_generation ==
                descriptor.target_pre_intent.backing_generation &&
            descriptor.target_pre_intent.mutation_epoch != std::numeric_limits<u64>::max() &&
            target_post.writer.mutation_epoch == descriptor.target_pre_intent.mutation_epoch + 1;
        const bool endpoints_ordered =
            predecessor_post.guest_rendering_serial != 0 &&
            predecessor_post.guest_rendering_serial == target_post.guest_rendering_serial &&
            predecessor_post.host_rendering_instance_id != std::numeric_limits<u64>::max() &&
            target_post.host_rendering_instance_id ==
                predecessor_post.host_rendering_instance_id + 1;
        const bool arm_exact = arm.valid && arm.private_only && !arm.logs_private_state &&
                               arm.status == FinalGuestSurfaceStatus::Complete &&
                               !arm.loss.Any() &&
                               arm.predecessor_scope_writer_count != 0 &&
                               arm.target_scope_writer_count ==
                                   arm.predecessor_scope_writer_count + 1 &&
                               descriptor.predecessor_scope_writer_count ==
                                   arm.predecessor_scope_writer_count &&
                               descriptor.target_scope_writer_count ==
                                   arm.target_scope_writer_count;
        if (!unused_selected_state || !tails_exact || !predecessor_exact || !target_exact ||
            !endpoints_ordered || !arm_exact) {
            return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                          &FinalGuestSurfaceLoss::invalidation);
        }
        return {.status = FinalGuestSurfaceStatus::AlreadyConsumed,
                .mode = PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindingMode::
                    CapturePhaseAwaitingFreshPair,
                .capture_phase_exact = true,
                .awaiting_fresh_selected_pair = true,
                .legacy_target_pre_unchanged = true,
                .pds_unchanged = true};
    }
    if (descriptor.fresh_selected_pair_count != 1 ||
        descriptor.predecessor_pre_observation_count != 1 ||
        descriptor.predecessor_post_observation_count != 1 ||
        descriptor.target_pre_observation_count != 1 ||
        descriptor.target_post_observation_count != 1 ||
        descriptor.selected_binding_count != 1 || descriptor.predecessor_open ||
        descriptor.target_open || descriptor.predecessor_collision ||
        descriptor.target_collision) {
        return reject(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
    }
    const auto& pair = descriptor.fresh_selected_pair;
    const auto& predecessor = pair.predecessor_endpoint;
    const auto& target = pair.target_post_endpoint;
    const auto& selected = descriptor.selected_binding;
    const bool tails_exact =
        ValidatePpUpstreamSampleTeeDepthDownloadTails(tails).status ==
            FinalGuestSurfaceStatus::Complete &&
        IsPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailRecordable(
            descriptor.predecessor_pre_plan) &&
        IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(
            descriptor.target_pre_plan) &&
        tails.predecessor_writer_pre.slot == descriptor.predecessor_pre_plan.slot &&
        tails.predecessor_writer_pre.slot_stride ==
            descriptor.predecessor_pre_plan.slot_stride &&
        tails.predecessor_writer_pre.plane.plane_offset ==
            descriptor.predecessor_pre_plan.plane.plane_offset &&
        tails.producer_pre.slot == descriptor.target_pre_plan.slot &&
        tails.producer_pre.slot_stride == descriptor.target_pre_plan.slot_stride &&
        tails.producer_pre.plane.plane_offset ==
            descriptor.target_pre_plan.plane.plane_offset;
    const bool layout_exact = IsPpUpstreamSampleTeeDepthPlaneSemanticLayoutExact(
        tails.predecessor_writer_pre.plane, tails.producer_pre.plane);
    const bool predecessor_pre_exact =
        descriptor.predecessor_pre_intent.Valid() && predecessor.Valid() &&
        descriptor.predecessor_pre_intent.hint == predecessor.writer.hint &&
        descriptor.predecessor_pre_intent.link == predecessor.writer.link &&
        descriptor.predecessor_pre_intent.range == predecessor.writer.range &&
        descriptor.predecessor_pre_intent.backing_generation ==
            predecessor.writer.backing_generation &&
        descriptor.predecessor_pre_intent.mutation_epoch !=
            std::numeric_limits<u64>::max() &&
        predecessor.writer.mutation_epoch ==
            descriptor.predecessor_pre_intent.mutation_epoch + 1;
    const bool predecessor_post_exact =
        descriptor.predecessor_post.status == FinalGuestSurfaceStatus::Complete &&
        !descriptor.predecessor_post.loss.Any() && descriptor.predecessor_post.exact &&
        descriptor.predecessor_post.output == predecessor;
    const bool target_pre_exact = descriptor.target_pre_intent.Valid() &&
                                  descriptor.target_pre_intent == pair.target_pre_intent;
    const bool target_post_exact =
        descriptor.target_post.status == FinalGuestSurfaceStatus::Complete &&
        !descriptor.target_post.loss.Any() && descriptor.target_post.exact &&
        descriptor.target_post.output == target;
    const bool selected_exact = selected.status == FinalGuestSurfaceStatus::Complete &&
                                !selected.loss.Any() && selected.exact &&
                                selected.output_endpoint_exact && selected.semantic_layout_exact &&
                                selected.publish_producer_pre;
    const bool exact =
        IsExactPpUpstreamSampleTeeProducerPreDepthSelectedPair(pair) &&
        IsExactPpUpstreamSampleTeeOrdinalArmForPair(descriptor.prior_arm, pair) && tails_exact &&
        layout_exact && predecessor_pre_exact && predecessor_post_exact && target_pre_exact &&
        target_post_exact &&
        descriptor.predecessor_scope_writer_count ==
            pair.expected_predecessor_pre_scope_writer_count &&
        descriptor.target_scope_writer_count == pair.selected_target_scope_writer_count &&
        selected_exact && descriptor.later_route_exact && descriptor.later_view_exact &&
        descriptor.later_selected_token_current;
    if (!exact) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    return {.status = FinalGuestSurfaceStatus::Complete,
            .mode = PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindingMode::
                FreshSelectedPair,
            .exact = true,
            .publish_predecessor_pre = true,
            .publish_target_pre = true,
            .predecessor_post_equals_fresh_predecessor = true,
            .target_pre_post_equals_fresh_target = true,
            .fresh_selected_pair_latest_exact = true,
            .ordinal_pair_exact = true,
            .semantic_layout_exact = true,
            .existing_gpu_copy_count = 2,
            .existing_token_count = 1,
            .existing_download_invalidation_count = 1,
            .existing_release_count = 1,
            .legacy_target_pre_unchanged = true,
            .pds_unchanged = true};
}

struct PpUpstreamSampleTeeProducerPreDepthOrdinalDirectSelectedBindDescriptor {
    u64 sequence{};
    PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm prior_arm{};
    u32 fresh_selected_pair_count{};
    PpUpstreamSampleTeeDepthDownloadTails combined_tails{};
    PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan predecessor_pre_plan{};
    PpDepthPredecessorWriterIntent predecessor_pre_intent{};
    PpUpstreamSampleTeeProducerPreDepthPost predecessor_post{};
    u32 predecessor_scope_writer_count{};
    u32 predecessor_pre_observation_count{};
    u32 predecessor_post_observation_count{};
    PpUpstreamSampleTeePredecessorWriterPreDepthOrdinalCaptureDescriptor
        predecessor_ordinal_capture_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan
        predecessor_ordinal_capture_plan{};
    u32 predecessor_ordinal_capture_count{};
    bool predecessor_open{};
    bool predecessor_collision{};
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan target_pre_plan{};
    PpDepthPredecessorWriterIntent target_pre_intent{};
    PpDepthPredecessorEndpointStamp target_pre_diagnostic_immediate_predecessor{};
    u32 target_pre_diagnostic_immediate_predecessor_count{};
    PpUpstreamSampleTeeProducerPreDepthPost target_post{};
    u32 target_scope_writer_count{};
    u32 target_pre_observation_count{};
    u32 target_post_observation_count{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalCaptureDescriptor
        target_ordinal_capture_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan target_ordinal_capture_plan{};
    u32 target_ordinal_capture_count{};
    bool target_open{};
    bool target_collision{};
    PpUpstreamSampleTeeProducerPreDepthBindDescriptor selected_binding_descriptor{};
    PpUpstreamSampleTeeProducerPreDepthBinding selected_binding{};
    u32 selected_binding_count{};
    PpUpstreamSampleTeeProducerPreDepthOccurrenceReport occurrence_report{};
    PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReport
        route_absent_ordinal_capture{};
    bool later_route_exact{};
    bool later_view_exact{};
    bool later_selected_token_current{};
};

enum class PpUpstreamSampleTeeProducerPreDepthOrdinalDirectSelectedFirstFailure : u8 {
    Idle = 0,
    PredecessorTailLoss = 1,
    TargetTailLoss = 2,
    CountsOrCollision = 3,
    Topology = 4,
    TailsOrLayout = 5,
    Arm = 6,
    PredecessorPost = 7,
    TargetPost = 8,
    PostContinuity = 9,
    ImmediatePredecessor = 10,
    PredecessorOrdinal = 11,
    TargetOrdinal = 12,
    SelectedBinding = 13,
    OccurrenceEvidence = 14,
    GenericNonAdmissionEvidence = 15,
    LaterExactness = 16,
    Exact = 17,
};

struct PpUpstreamSampleTeeProducerPreDepthOrdinalDirectSelectedEvaluation {
    PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding binding{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalDirectSelectedFirstFailure
        first_failure{};
};

[[nodiscard]] inline bool IsCanonicalPpUpstreamSampleTeeProducerPreDepthOrdinalArm(
    const PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm& arm) noexcept {
    return arm.status == FinalGuestSurfaceStatus::Complete && !arm.loss.Any() && arm.valid &&
           arm.private_only && !arm.logs_private_state && arm.retains_private_state &&
           arm.predecessor_hint.valid && arm.predecessor_hint.private_only &&
           !arm.predecessor_hint.logs_private_state && arm.predecessor_hint.writer.Valid() &&
           arm.target_hint.valid && arm.target_hint.private_only &&
           !arm.target_hint.logs_private_state && arm.target_hint.writer.Valid() &&
           arm.predecessor_scope_writer_count != 0 &&
           arm.predecessor_scope_writer_count != std::numeric_limits<u32>::max() &&
           arm.target_scope_writer_count == arm.predecessor_scope_writer_count + 1 &&
           arm.additional_gpu_copy_count == 0 && arm.additional_slot_count == 0 &&
           arm.additional_invalidation_count == 0 && arm.additional_release_count == 0 &&
           !arm.affects_capture_behavior && !arm.affects_renderer_behavior;
}

[[nodiscard]] inline bool SamePpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan(
    const PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan& lhs,
    const PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan& rhs) noexcept {
    return lhs.scope_writer_count == rhs.scope_writer_count && lhs.status == rhs.status &&
           lhs.loss == rhs.loss && lhs.capture_pre == rhs.capture_pre &&
           lhs.run_collision_check == rhs.run_collision_check &&
           lhs.arm_generic_predecessor_observation ==
               rhs.arm_generic_predecessor_observation &&
           lhs.arm_generic_target_observation == rhs.arm_generic_target_observation &&
           lhs.wrong_ordinal_ignored == rhs.wrong_ordinal_ignored &&
           lhs.unrelated_writer_ignored == rhs.unrelated_writer_ignored &&
           lhs.context_ignored == rhs.context_ignored &&
           lhs.terminal_contradiction == rhs.terminal_contradiction;
}

[[nodiscard]] inline bool SamePpUpstreamSampleTeeProducerPreDepthPost(
    const PpUpstreamSampleTeeProducerPreDepthPost& lhs,
    const PpUpstreamSampleTeeProducerPreDepthPost& rhs) noexcept {
    return lhs.output == rhs.output && lhs.status == rhs.status && lhs.loss == rhs.loss &&
           lhs.exact == rhs.exact && lhs.mutation_successor_exact == rhs.mutation_successor_exact &&
           lhs.diagnostic_resume_host_successor_exact ==
               rhs.diagnostic_resume_host_successor_exact &&
           lhs.copy_post_content == rhs.copy_post_content;
}

[[nodiscard]] inline bool SamePpUpstreamSampleTeeProducerPreDepthBinding(
    const PpUpstreamSampleTeeProducerPreDepthBinding& lhs,
    const PpUpstreamSampleTeeProducerPreDepthBinding& rhs) noexcept {
    return lhs.status == rhs.status && lhs.loss == rhs.loss && lhs.exact == rhs.exact &&
           lhs.output_endpoint_exact == rhs.output_endpoint_exact &&
           lhs.semantic_layout_exact == rhs.semantic_layout_exact &&
           lhs.publish_producer_pre == rhs.publish_producer_pre &&
           lhs.retains_private_hint == rhs.retains_private_hint;
}

[[nodiscard]] inline bool IsExactPpUpstreamSampleTeeDirectSelectedOccurrenceReport(
    const PpUpstreamSampleTeeProducerPreDepthOccurrenceReport& report,
    u64 sequence) noexcept {
    return sequence != 0 && report.sequence == sequence && report.total_exact_occurrences == 0 &&
           report.selected_observation_count == 1 &&
           report.selected_occurrence ==
               PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence::None &&
           report.selected_match_count == 0 && report.occurrences_after_consumer == 0 &&
           report.selected_matches_after_consumer == 0 &&
           report.intervening_mutation_drift == 0 && report.resource_drift == 0 &&
           report.currentness_drift == 0 && report.malformed_plan_drift == 0 &&
           report.semantic_drift == 0 && report.endpoint_drift == 0 &&
           report.ordering_drift == 0 && report.terminal_contradictions == 0 &&
           report.unrelated_writer_misses <=
               PpUpstreamSampleTeeProducerPreDepthOccurrenceEvidence::MaxEvents &&
           report.diagnostic_rejections <=
               PpUpstreamSampleTeeProducerPreDepthOccurrenceEvidence::MaxEvents &&
           report.status == FinalGuestSurfaceStatus::InvalidationLoss &&
           report.loss == FinalGuestSurfaceLoss{.invalidation = 1} && !report.overflow &&
           report.selected_endpoint_frozen && report.metadata_only &&
           !report.retains_private_state && report.additional_gpu_copy_count == 0 &&
           report.additional_slot_count == 0 && report.additional_release_count == 0 &&
           !report.affects_capture_behavior && !report.affects_renderer_behavior &&
           report.paired_predecessor_count == 0 &&
           report.selected_predecessor_match_count == 0 &&
           report.learned_predecessor_valid_count == 0 &&
           report.selected_predecessor_ordinal_status ==
               PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::Invalid;
}

[[nodiscard]] inline bool IsExactPpUpstreamSampleTeeDirectSelectedGenericNonAdmission(
    const PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReport& report,
    u64 sequence) noexcept {
    using Reason = PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason;
    using Failure = PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureFirstFailure;
    return sequence != 0 && report.sequence == sequence &&
           report.reason == Reason::ExactArmedGenericNonAdmission &&
           report.predecessor_observation_count == 1 && report.target_observation_count == 1 &&
           report.generic_route_unarmed_count == 0 &&
           report.generic_route_armed_count == 2 && report.generic_rejection_count == 2 &&
           ((report.generic_warmup_count == 2 &&
             report.generic_invalidation_rejection_count == 0) ||
            (report.generic_warmup_count == 0 &&
             report.generic_invalidation_rejection_count == 2)) &&
           report.ordinal_admission_count == 2 && report.ordering_exact_count == 2 &&
           report.predecessor_first_failure == Failure::Exact &&
           report.target_first_failure == Failure::Exact &&
           report.status == FinalGuestSurfaceStatus::Complete && !report.loss.Any() &&
           !report.diagnostic_saturated && !report.overflow && report.metadata_only &&
           !report.retains_private_state && report.additional_gpu_copy_count == 0 &&
           report.additional_slot_count == 0 && report.additional_invalidation_count == 0 &&
           report.additional_release_count == 0 && !report.affects_capture_behavior &&
           !report.affects_renderer_behavior;
}

[[nodiscard]] inline bool IsExactPpUpstreamSampleTeeDirectSelectedBinding(
    const PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding& binding) noexcept {
    using Mode = PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindingMode;
    return binding.status == FinalGuestSurfaceStatus::Complete && !binding.loss.Any() &&
           binding.mode == Mode::DirectSelectedSameFrame && binding.exact &&
           binding.publish_predecessor_pre && binding.publish_target_pre &&
           !binding.predecessor_post_equals_fresh_predecessor &&
           !binding.target_pre_post_equals_fresh_target &&
           !binding.fresh_selected_pair_latest_exact && binding.direct_selected_same_frame_exact &&
           binding.predecessor_post_equals_target_immediate_predecessor &&
           binding.target_post_equals_direct_selected && binding.generic_nonadmission_exact &&
           binding.ordinal_pair_exact && binding.semantic_layout_exact &&
           !binding.consumes_scope_entry && !binding.affects_pds &&
           binding.existing_gpu_copy_count == 2 && binding.existing_token_count == 1 &&
           binding.existing_download_invalidation_count == 1 &&
           binding.existing_release_count == 1 && binding.additional_gpu_copy_count == 0 &&
           binding.additional_slot_count == 0 && binding.additional_invalidation_count == 0 &&
           binding.additional_release_count == 0 && !binding.affects_capture_behavior &&
           !binding.affects_renderer_behavior && !binding.capture_phase_exact &&
           !binding.awaiting_fresh_selected_pair && binding.legacy_target_pre_unchanged &&
           binding.pds_unchanged && !binding.retains_private_endpoints;
}

[[nodiscard]] inline bool IsExactPpUpstreamSampleTeeFreshSelectedPairBinding(
    const PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding& binding) noexcept {
    using Mode = PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindingMode;
    return binding.status == FinalGuestSurfaceStatus::Complete && !binding.loss.Any() &&
           binding.mode == Mode::FreshSelectedPair && binding.exact &&
           binding.publish_predecessor_pre && binding.publish_target_pre &&
           binding.predecessor_post_equals_fresh_predecessor &&
           binding.target_pre_post_equals_fresh_target &&
           binding.fresh_selected_pair_latest_exact && !binding.direct_selected_same_frame_exact &&
           !binding.predecessor_post_equals_target_immediate_predecessor &&
           !binding.target_post_equals_direct_selected && !binding.generic_nonadmission_exact &&
           binding.ordinal_pair_exact && binding.semantic_layout_exact &&
           !binding.consumes_scope_entry && !binding.affects_pds &&
           binding.existing_gpu_copy_count == 2 && binding.existing_token_count == 1 &&
           binding.existing_download_invalidation_count == 1 &&
           binding.existing_release_count == 1 && binding.additional_gpu_copy_count == 0 &&
           binding.additional_slot_count == 0 && binding.additional_invalidation_count == 0 &&
           binding.additional_release_count == 0 && !binding.affects_capture_behavior &&
           !binding.affects_renderer_behavior && !binding.capture_phase_exact &&
           !binding.awaiting_fresh_selected_pair && binding.legacy_target_pre_unchanged &&
           binding.pds_unchanged && !binding.retains_private_endpoints;
}

[[nodiscard]] inline
PpUpstreamSampleTeeProducerPreDepthOrdinalDirectSelectedEvaluation
EvaluatePpUpstreamSampleTeeProducerPreDepthOrdinalContentToDirectSelected(
    const PpUpstreamSampleTeeProducerPreDepthOrdinalDirectSelectedBindDescriptor&
        descriptor) noexcept {
    using Evaluation =
        PpUpstreamSampleTeeProducerPreDepthOrdinalDirectSelectedEvaluation;
    using FirstFailure =
        PpUpstreamSampleTeeProducerPreDepthOrdinalDirectSelectedFirstFailure;
    const auto reject = [](FinalGuestSurfaceStatus status,
                           u32 FinalGuestSurfaceLoss::* member,
                           FirstFailure first_failure) {
        Evaluation result{
            .binding = {.status = status,
                        .legacy_target_pre_unchanged = true,
                        .pds_unchanged = true},
            .first_failure = first_failure,
        };
        result.binding.loss.*member = 1;
        return result;
    };
    const auto reject_invalidation = [&](FirstFailure first_failure) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation, first_failure);
    };
    const auto propagate_tail_loss = [&](const auto& plan,
                                         FirstFailure first_failure)
        -> std::optional<Evaluation> {
        if (plan.status != FinalGuestSurfaceStatus::Complete &&
            plan.status != FinalGuestSurfaceStatus::AlreadyConsumed && plan.loss.Any()) {
            return Evaluation{
                .binding = {.status = plan.status,
                            .loss = plan.loss,
                            .legacy_target_pre_unchanged = true,
                            .pds_unchanged = true},
                .first_failure = first_failure,
            };
        }
        return std::nullopt;
    };
    if (const auto loss = propagate_tail_loss(
            descriptor.predecessor_pre_plan,
            FirstFailure::PredecessorTailLoss)) {
        return *loss;
    }
    if (const auto loss = propagate_tail_loss(descriptor.target_pre_plan,
                                              FirstFailure::TargetTailLoss)) {
        return *loss;
    }
    if (descriptor.predecessor_pre_observation_count != 1 ||
        descriptor.predecessor_post_observation_count != 1 ||
        descriptor.predecessor_ordinal_capture_count != 1 ||
        descriptor.target_pre_diagnostic_immediate_predecessor_count != 1 ||
        descriptor.target_pre_observation_count != 1 ||
        descriptor.target_post_observation_count != 1 ||
        descriptor.target_ordinal_capture_count != 1 || descriptor.selected_binding_count != 1 ||
        descriptor.predecessor_open || descriptor.predecessor_collision ||
        descriptor.target_open || descriptor.target_collision) {
        return reject(FinalGuestSurfaceStatus::GapLoss,
                      &FinalGuestSurfaceLoss::gap,
                      FirstFailure::CountsOrCollision);
    }
    if (descriptor.sequence == 0 || descriptor.fresh_selected_pair_count != 0) {
        return reject_invalidation(FirstFailure::Topology);
    }

    const auto& arm = descriptor.prior_arm;
    const auto& predecessor_endpoint = descriptor.predecessor_post.output;
    const auto& target_endpoint = descriptor.target_post.output;
    const auto canonical_predecessor_ordinal =
        PlanPpUpstreamSampleTeePredecessorWriterPreDepthOrdinalCapture(
            descriptor.predecessor_ordinal_capture_descriptor);
    const auto canonical_target_ordinal =
        PlanPpUpstreamSampleTeeProducerPreDepthOrdinalCapture(
            descriptor.target_ordinal_capture_descriptor);
    const auto canonical_selected = BindPpUpstreamSampleTeeProducerPreDepthToSelected(
        descriptor.selected_binding_descriptor);
    const bool tails_exact =
        ValidatePpUpstreamSampleTeeDepthDownloadTails(descriptor.combined_tails).status ==
            FinalGuestSurfaceStatus::Complete &&
        IsPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailRecordable(
            descriptor.predecessor_pre_plan) &&
        IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(descriptor.target_pre_plan) &&
        descriptor.combined_tails.predecessor_writer_pre == descriptor.predecessor_pre_plan &&
        descriptor.combined_tails.producer_pre == descriptor.target_pre_plan &&
        IsPpUpstreamSampleTeeDepthPlaneSemanticLayoutExact(
            descriptor.combined_tails.predecessor_writer_pre.plane,
            descriptor.combined_tails.producer_pre.plane);
    if (!tails_exact) {
        return reject_invalidation(FirstFailure::TailsOrLayout);
    }
    const bool arm_exact =
        IsCanonicalPpUpstreamSampleTeeProducerPreDepthOrdinalArm(arm) &&
        descriptor.predecessor_scope_writer_count == arm.predecessor_scope_writer_count &&
        descriptor.target_scope_writer_count == arm.target_scope_writer_count &&
        IsExactPpUpstreamSampleTeeOrdinalArmHint(arm.predecessor_hint,
                                                 descriptor.predecessor_pre_intent) &&
        IsExactPpUpstreamSampleTeeOrdinalArmHint(arm.target_hint,
                                                 descriptor.target_pre_intent);
    if (!arm_exact) {
        return reject_invalidation(FirstFailure::Arm);
    }
    const auto post_exact = [](const PpUpstreamSampleTeeProducerPreDepthPost& post,
                               const PpDepthPredecessorWriterIntent& pre) {
        return post.status == FinalGuestSurfaceStatus::Complete && !post.loss.Any() &&
               post.exact && post.mutation_successor_exact &&
               post.diagnostic_resume_host_successor_exact && !post.copy_post_content &&
               pre.Valid() && post.output.Valid() && post.output.after_diagnostic_resume &&
               post.output.writer.hint == pre.hint && post.output.writer.link == pre.link &&
               post.output.writer.range == pre.range &&
               post.output.writer.backing_generation == pre.backing_generation &&
               pre.mutation_epoch != std::numeric_limits<u64>::max() &&
               post.output.writer.mutation_epoch == pre.mutation_epoch + 1;
    };
    if (!post_exact(descriptor.predecessor_post,
                    descriptor.predecessor_pre_intent)) {
        return reject_invalidation(FirstFailure::PredecessorPost);
    }
    if (!post_exact(descriptor.target_post, descriptor.target_pre_intent)) {
        return reject_invalidation(FirstFailure::TargetPost);
    }
    const bool post_continuity_exact =
        predecessor_endpoint.guest_rendering_serial == target_endpoint.guest_rendering_serial &&
        predecessor_endpoint.host_rendering_instance_id != std::numeric_limits<u64>::max() &&
        target_endpoint.host_rendering_instance_id ==
            predecessor_endpoint.host_rendering_instance_id + 1 &&
        predecessor_endpoint.writer.link == descriptor.target_pre_intent.link &&
        predecessor_endpoint.writer.range == descriptor.target_pre_intent.range &&
        predecessor_endpoint.writer.backing_generation ==
            descriptor.target_pre_intent.backing_generation &&
        predecessor_endpoint.writer.mutation_epoch ==
            descriptor.target_pre_intent.mutation_epoch;
    if (!post_continuity_exact) {
        return reject_invalidation(FirstFailure::PostContinuity);
    }
    if (descriptor.target_pre_diagnostic_immediate_predecessor !=
        predecessor_endpoint) {
        return reject_invalidation(FirstFailure::ImmediatePredecessor);
    }
    const bool predecessor_ordinal_exact =
        descriptor.predecessor_ordinal_capture_descriptor.arm == arm &&
        descriptor.predecessor_ordinal_capture_descriptor.hook_descriptor.observed_writer ==
            descriptor.predecessor_pre_intent &&
        SamePpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan(
            descriptor.predecessor_ordinal_capture_plan, canonical_predecessor_ordinal) &&
        descriptor.predecessor_ordinal_capture_plan.status == FinalGuestSurfaceStatus::Complete &&
        !descriptor.predecessor_ordinal_capture_plan.loss.Any() &&
        descriptor.predecessor_ordinal_capture_plan.scope_writer_count ==
            descriptor.predecessor_scope_writer_count &&
        descriptor.predecessor_ordinal_capture_plan.capture_pre &&
        descriptor.predecessor_ordinal_capture_plan.run_collision_check &&
        descriptor.predecessor_ordinal_capture_plan.arm_generic_predecessor_observation &&
        !descriptor.predecessor_ordinal_capture_plan.arm_generic_target_observation;
    if (!predecessor_ordinal_exact) {
        return reject_invalidation(FirstFailure::PredecessorOrdinal);
    }
    const bool target_ordinal_exact =
        descriptor.target_ordinal_capture_descriptor.arm == arm &&
        descriptor.target_ordinal_capture_descriptor.hook_descriptor.observed_writer ==
            descriptor.target_pre_intent &&
        SamePpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan(
            descriptor.target_ordinal_capture_plan, canonical_target_ordinal) &&
        descriptor.target_ordinal_capture_plan.status == FinalGuestSurfaceStatus::Complete &&
        !descriptor.target_ordinal_capture_plan.loss.Any() &&
        descriptor.target_ordinal_capture_plan.scope_writer_count ==
            descriptor.target_scope_writer_count &&
        descriptor.target_ordinal_capture_plan.capture_pre &&
        descriptor.target_ordinal_capture_plan.run_collision_check &&
        !descriptor.target_ordinal_capture_plan.arm_generic_predecessor_observation &&
        !descriptor.target_ordinal_capture_plan.arm_generic_target_observation;
    if (!target_ordinal_exact) {
        return reject_invalidation(FirstFailure::TargetOrdinal);
    }
    const bool selected_exact =
        SamePpUpstreamSampleTeeProducerPreDepthPost(
            descriptor.selected_binding_descriptor.post, descriptor.target_post) &&
        descriptor.selected_binding_descriptor.live_selected_endpoint == target_endpoint &&
        IsPpUpstreamSampleTeeDepthPlaneSemanticLayoutExact(
            descriptor.selected_binding_descriptor.producer_pre_plane,
            descriptor.target_pre_plan.plane) &&
        SamePpUpstreamSampleTeeProducerPreDepthBinding(descriptor.selected_binding,
                                                       canonical_selected) &&
        canonical_selected.status == FinalGuestSurfaceStatus::Complete &&
        !canonical_selected.loss.Any() && canonical_selected.exact &&
        canonical_selected.output_endpoint_exact && canonical_selected.semantic_layout_exact &&
        canonical_selected.publish_producer_pre && !canonical_selected.retains_private_hint;
    if (!selected_exact) {
        return reject_invalidation(FirstFailure::SelectedBinding);
    }
    if (!IsExactPpUpstreamSampleTeeDirectSelectedOccurrenceReport(
            descriptor.occurrence_report, descriptor.sequence)) {
        return reject_invalidation(FirstFailure::OccurrenceEvidence);
    }
    if (!IsExactPpUpstreamSampleTeeDirectSelectedGenericNonAdmission(
            descriptor.route_absent_ordinal_capture, descriptor.sequence)) {
        return reject_invalidation(FirstFailure::GenericNonAdmissionEvidence);
    }
    if (!descriptor.later_route_exact || !descriptor.later_view_exact ||
        !descriptor.later_selected_token_current) {
        return reject_invalidation(FirstFailure::LaterExactness);
    }
    return {
        .binding =
            {
                .status = FinalGuestSurfaceStatus::Complete,
                .mode = PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindingMode::
                    DirectSelectedSameFrame,
                .exact = true,
                .publish_predecessor_pre = true,
                .publish_target_pre = true,
                .direct_selected_same_frame_exact = true,
                .predecessor_post_equals_target_immediate_predecessor = true,
                .target_post_equals_direct_selected = true,
                .generic_nonadmission_exact = true,
                .ordinal_pair_exact = true,
                .semantic_layout_exact = true,
                .existing_gpu_copy_count = 2,
                .existing_token_count = 1,
                .existing_download_invalidation_count = 1,
                .existing_release_count = 1,
                .legacy_target_pre_unchanged = true,
                .pds_unchanged = true,
            },
        .first_failure = FirstFailure::Exact,
    };
}

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding
BindPpUpstreamSampleTeeProducerPreDepthOrdinalContentToDirectSelected(
    const PpUpstreamSampleTeeProducerPreDepthOrdinalDirectSelectedBindDescriptor&
        descriptor) noexcept {
    return EvaluatePpUpstreamSampleTeeProducerPreDepthOrdinalContentToDirectSelected(
               descriptor)
        .binding;
}

enum class PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition : u8 {
    Idle,
    Published,
    RetryScheduled,
    Discarded,
};

struct PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameReport {
    u64 sequence{};
    u32 arm_count{};
    u32 fresh_selected_pair_count{};
    u32 predecessor_capture_count{};
    u32 predecessor_post_count{};
    u32 target_capture_count{};
    u32 target_post_count{};
    u32 binding_count{};
    u32 retry_scheduled_count{};
    u32 discarded_tail_count{};
    u32 inactive_frame_count{};
    u32 inactive_flip_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalDirectSelectedFirstFailure
        direct_selected_first_failure{};
    bool retains_private_state{};
};

struct PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDescriptor {
    bool enabled{true};
    u64 sequence{};
    PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm prior_arm{};
    u32 inactive_flip_count{};
    PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorPair fresh_selected_pair{};
    u32 fresh_selected_pair_count{};
    bool frame_active{};
    bool frame_inactive{};
    u32 predecessor_capture_count{};
    u32 predecessor_post_count{};
    u32 target_capture_count{};
    u32 target_post_count{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding ordinal_binding{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalDirectSelectedFirstFailure
        direct_selected_first_failure{};
    bool predecessor_collision{};
    bool target_collision{};
    PpUpstreamSampleTeeProducerPreDepthOccurrenceReport occurrence_report{};
    PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReport
        route_absent_ordinal_capture{};
};

struct PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameResult {
    PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm next_arm{};
    u32 next_inactive_flip_count{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameReport report{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition disposition{
        PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::Idle};
    bool preserved_for_one_inactive_flip{};
    bool preserved_for_one_capture_retry{};
    bool warmup_completed{};
    bool discard_current_ordinal_tails{};
    bool defer_existing_outer_release{};
    bool publish_current_ordinal_tails{};
    bool retains_private_state{};
    u32 retained_private_state_count{};
    bool retains_tail_bytes{};
    bool retains_endpoints{};
    bool retains_outer_token{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameResult
FinishPpUpstreamSampleTeeProducerPreDepthOrdinalContentFrame(
    const PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDescriptor& descriptor) noexcept {
    auto finish = [&](FinalGuestSurfaceStatus status, FinalGuestSurfaceLoss loss,
                      PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm next = {},
                      u32 next_age = 0, bool preserved = false, bool warmup = false,
                      PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition
                          disposition = PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::Idle,
                      bool capture_retry = false, bool discard_tails = false,
                      bool defer_release = false, bool publish_tails = false) {
        const bool retained = next.valid;
        const bool bound = descriptor.ordinal_binding.exact &&
                           descriptor.ordinal_binding.publish_predecessor_pre &&
                           descriptor.ordinal_binding.publish_target_pre;
        return PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameResult{
            .next_arm = next,
            .next_inactive_flip_count = next_age,
            .report = {.sequence = descriptor.sequence,
                       .arm_count = descriptor.prior_arm.valid ? 1u : 0u,
                       .fresh_selected_pair_count = descriptor.fresh_selected_pair_count,
                       .predecessor_capture_count = descriptor.predecessor_capture_count,
                       .predecessor_post_count = descriptor.predecessor_post_count,
                       .target_capture_count = descriptor.target_capture_count,
                       .target_post_count = descriptor.target_post_count,
                       .binding_count = bound ? 1u : 0u,
                       .retry_scheduled_count = capture_retry ? 1u : 0u,
                       .discarded_tail_count = discard_tails ? 2u : 0u,
                       .inactive_frame_count = descriptor.frame_inactive ? 1u : 0u,
                        .inactive_flip_count = next_age,
                        .status = status,
                        .loss = loss,
                        .direct_selected_first_failure =
                            descriptor.direct_selected_first_failure},
            .status = status,
            .loss = loss,
            .disposition = disposition,
            .preserved_for_one_inactive_flip = preserved,
            .preserved_for_one_capture_retry = capture_retry,
            .warmup_completed = warmup,
            .discard_current_ordinal_tails = discard_tails,
            .defer_existing_outer_release = defer_release,
            .publish_current_ordinal_tails = publish_tails,
            .retains_private_state = retained,
            .retained_private_state_count = retained ? 1u : 0u};
    };
    if (!descriptor.enabled) {
        return finish(FinalGuestSurfaceStatus::AlreadyConsumed, {});
    }
    const bool has_activity = descriptor.predecessor_capture_count != 0 ||
                              descriptor.predecessor_post_count != 0 ||
                              descriptor.target_capture_count != 0 ||
                              descriptor.target_post_count != 0 ||
                              descriptor.ordinal_binding.exact ||
                              descriptor.predecessor_collision || descriptor.target_collision;
    if (descriptor.frame_inactive &&
        (descriptor.frame_active || descriptor.fresh_selected_pair_count != 0 || has_activity)) {
        return finish(FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1});
    }
    if (descriptor.frame_inactive) {
        if (descriptor.prior_arm.valid && descriptor.inactive_flip_count == 0) {
            return finish(FinalGuestSurfaceStatus::Complete, {}, descriptor.prior_arm, 1, true);
        }
        return finish(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
    }
    const bool fresh_exact = descriptor.fresh_selected_pair_count == 1 &&
                             IsExactPpUpstreamSampleTeeProducerPreDepthSelectedPair(
                                 descriptor.fresh_selected_pair);
    PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm fresh_arm{};
    if (fresh_exact) {
        fresh_arm = PlanPpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm({
            .enabled = true,
            .sample_tee_enabled = true,
            .prior_selected_pair = descriptor.fresh_selected_pair});
    }
    const bool exact_capture_counts = descriptor.predecessor_capture_count == 1 &&
                                      descriptor.predecessor_post_count == 1 &&
                                      descriptor.target_capture_count == 1 &&
                                      descriptor.target_post_count == 1 &&
                                      !descriptor.predecessor_collision &&
                                      !descriptor.target_collision;
    const bool capture_phase_binding =
        descriptor.ordinal_binding.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
        !descriptor.ordinal_binding.loss.Any() && descriptor.ordinal_binding.capture_phase_exact &&
        descriptor.ordinal_binding.awaiting_fresh_selected_pair &&
        (descriptor.ordinal_binding.mode ==
             PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindingMode::
                 CapturePhaseAwaitingFreshPair ||
         descriptor.ordinal_binding.mode ==
             PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindingMode::None) &&
        !descriptor.ordinal_binding.exact &&
        !descriptor.ordinal_binding.publish_predecessor_pre &&
        !descriptor.ordinal_binding.publish_target_pre &&
        !descriptor.ordinal_binding.predecessor_post_equals_fresh_predecessor &&
        !descriptor.ordinal_binding.target_pre_post_equals_fresh_target &&
        !descriptor.ordinal_binding.fresh_selected_pair_latest_exact &&
        !descriptor.ordinal_binding.direct_selected_same_frame_exact &&
        !descriptor.ordinal_binding.predecessor_post_equals_target_immediate_predecessor &&
        !descriptor.ordinal_binding.target_post_equals_direct_selected &&
        !descriptor.ordinal_binding.generic_nonadmission_exact &&
        descriptor.ordinal_binding.legacy_target_pre_unchanged &&
        descriptor.ordinal_binding.pds_unchanged &&
        !descriptor.ordinal_binding.retains_private_endpoints;
    const bool direct_claim =
        descriptor.ordinal_binding.mode ==
            PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindingMode::
                DirectSelectedSameFrame ||
        descriptor.ordinal_binding.direct_selected_same_frame_exact ||
        descriptor.ordinal_binding.predecessor_post_equals_target_immediate_predecessor ||
        descriptor.ordinal_binding.target_post_equals_direct_selected ||
        descriptor.ordinal_binding.generic_nonadmission_exact;
    if (descriptor.frame_active && descriptor.fresh_selected_pair_count == 0 && direct_claim) {
        const bool evidence_sequence_exact =
            descriptor.occurrence_report.sequence != 0 &&
            descriptor.occurrence_report.sequence ==
                descriptor.route_absent_ordinal_capture.sequence;
        const bool direct_exact =
            exact_capture_counts && descriptor.inactive_flip_count <= 1 &&
            IsCanonicalPpUpstreamSampleTeeProducerPreDepthOrdinalArm(descriptor.prior_arm) &&
            IsExactPpUpstreamSampleTeeDirectSelectedBinding(descriptor.ordinal_binding) &&
            evidence_sequence_exact &&
            IsExactPpUpstreamSampleTeeDirectSelectedOccurrenceReport(
                descriptor.occurrence_report, descriptor.occurrence_report.sequence) &&
            IsExactPpUpstreamSampleTeeDirectSelectedGenericNonAdmission(
                descriptor.route_absent_ordinal_capture,
                descriptor.route_absent_ordinal_capture.sequence);
        if (direct_exact) {
            return finish(
                FinalGuestSurfaceStatus::Complete, {}, descriptor.prior_arm, 0, false, false,
                PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::Published,
                false, false, false, true);
        }
        const bool gap = !exact_capture_counts || descriptor.inactive_flip_count > 1;
        return finish(
            gap ? FinalGuestSurfaceStatus::GapLoss
                : FinalGuestSurfaceStatus::InvalidationLoss,
            gap ? FinalGuestSurfaceLoss{.gap = 1}
                : FinalGuestSurfaceLoss{.invalidation = 1},
            {}, 0, false, false,
            PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::Discarded,
            false, exact_capture_counts, exact_capture_counts);
    }
    if (descriptor.frame_active && descriptor.fresh_selected_pair_count == 0 &&
        descriptor.ordinal_binding.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
        descriptor.ordinal_binding.loss.Any()) {
        return finish(descriptor.ordinal_binding.status, descriptor.ordinal_binding.loss, {}, 0,
                      false, false,
                      PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::Discarded,
                      false, exact_capture_counts, exact_capture_counts);
    }
    if (descriptor.frame_active && descriptor.fresh_selected_pair_count == 0 &&
        capture_phase_binding) {
        const auto& report = descriptor.occurrence_report;
        const auto& route_absent = descriptor.route_absent_ordinal_capture;
        const bool common_clean = report.selected_observation_count == 0 &&
            report.selected_occurrence ==
                PpUpstreamSampleTeeProducerPreDepthSelectedOccurrence::Unavailable &&
            report.selected_match_count == 0 && report.occurrences_after_consumer == 0 &&
            report.selected_matches_after_consumer == 0 &&
            report.intervening_mutation_drift == 0 && report.resource_drift == 0 &&
            report.currentness_drift == 0 && report.malformed_plan_drift == 0 &&
            report.semantic_drift == 0 && report.endpoint_drift == 0 &&
            report.ordering_drift == 0 && report.terminal_contradictions == 0 &&
            !report.overflow && !report.selected_endpoint_frozen &&
            report.selected_predecessor_match_count == 0 &&
            report.learned_predecessor_valid_count == 0 &&
            report.selected_predecessor_ordinal_status ==
                PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorOrdinalStatus::Unavailable;
        const bool clean_missing_selected =
            common_clean && report.total_exact_occurrences == 2 &&
            report.paired_predecessor_count == 2 &&
            report.status == FinalGuestSurfaceStatus::GapLoss &&
            report.loss == FinalGuestSurfaceLoss{.gap = 1};
        const bool exact_nonadmission_common =
            route_absent.predecessor_observation_count == 1 &&
            route_absent.target_observation_count == 1 &&
            route_absent.generic_rejection_count == 2 &&
            route_absent.ordinal_admission_count == 2 &&
            route_absent.ordering_exact_count == 2 &&
            route_absent.predecessor_first_failure ==
                PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureFirstFailure::Exact &&
            route_absent.target_first_failure ==
                PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureFirstFailure::Exact &&
            route_absent.status == FinalGuestSurfaceStatus::Complete &&
            !route_absent.loss.Any() && !route_absent.diagnostic_saturated &&
            !route_absent.overflow &&
            route_absent.metadata_only && !route_absent.retains_private_state &&
            route_absent.additional_gpu_copy_count == 0 &&
            route_absent.additional_slot_count == 0 &&
            route_absent.additional_invalidation_count == 0 &&
            route_absent.additional_release_count == 0 &&
            !route_absent.affects_capture_behavior &&
            !route_absent.affects_renderer_behavior;
        const bool exact_route_absent =
            exact_nonadmission_common &&
            route_absent.reason ==
                PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason::
                    ExactRouteAbsent &&
            route_absent.generic_route_unarmed_count == 2 &&
            route_absent.generic_route_armed_count == 0 &&
            route_absent.generic_warmup_count == 0 &&
            route_absent.generic_invalidation_rejection_count == 0;
        const bool exact_armed_nonadmission =
            exact_nonadmission_common &&
            route_absent.reason ==
                PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReason::
                    ExactArmedGenericNonAdmission &&
            route_absent.generic_route_unarmed_count == 0 &&
            route_absent.generic_route_armed_count == 2 &&
            ((route_absent.generic_warmup_count == 2 &&
              route_absent.generic_invalidation_rejection_count == 0) ||
             (route_absent.generic_warmup_count == 0 &&
              route_absent.generic_invalidation_rejection_count == 2));
        const bool clean_route_absent =
            common_clean && report.total_exact_occurrences == 0 &&
            report.paired_predecessor_count == 0 &&
            report.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            !report.loss.Any() && (exact_route_absent || exact_armed_nonadmission);
        const bool clean_capture_phase = clean_missing_selected || clean_route_absent;
        if (!exact_capture_counts || !descriptor.prior_arm.valid || !clean_capture_phase) {
            const bool occurrence_specific =
                report.status != FinalGuestSurfaceStatus::AlreadyConsumed && report.loss.Any();
            const bool route_specific = route_absent.status !=
                                             FinalGuestSurfaceStatus::AlreadyConsumed &&
                                         route_absent.status !=
                                             FinalGuestSurfaceStatus::Complete &&
                                         route_absent.loss.Any();
            const auto status = occurrence_specific
                                    ? report.status
                                : route_specific
                                    ? route_absent.status
                                    : FinalGuestSurfaceStatus::GapLoss;
            const auto loss = occurrence_specific
                                  ? report.loss
                              : route_specific
                                  ? route_absent.loss
                                  : FinalGuestSurfaceLoss{.gap = 1};
            return finish(
                status, loss, {}, 0, false, false,
                PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::Discarded,
                false, exact_capture_counts, exact_capture_counts);
        }
        if (descriptor.inactive_flip_count == 0) {
            return finish(
                FinalGuestSurfaceStatus::Complete, {}, descriptor.prior_arm, 1, false, false,
                PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::RetryScheduled,
                true, true, true);
        }
        return finish(
            FinalGuestSurfaceStatus::GapLoss, {.gap = 1}, {}, 0, false, false,
            PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::Discarded, false,
            true, true);
    }
    if (!descriptor.prior_arm.valid && !descriptor.frame_active && !has_activity) {
        if (fresh_exact && fresh_arm.valid) {
            return finish(FinalGuestSurfaceStatus::Complete, {}, fresh_arm, 0, false, true);
        }
        return finish(FinalGuestSurfaceStatus::AlreadyConsumed, {});
    }
    if (!descriptor.frame_active || descriptor.fresh_selected_pair_count != 1 ||
        !fresh_exact || descriptor.predecessor_capture_count != 1 ||
        descriptor.predecessor_post_count != 1 || descriptor.target_capture_count != 1 ||
        descriptor.target_post_count != 1 || descriptor.predecessor_collision ||
        descriptor.target_collision) {
        const bool discard = descriptor.frame_active && exact_capture_counts;
        return finish(
            FinalGuestSurfaceStatus::GapLoss, {.gap = 1}, {}, 0, false, false,
            discard ? PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::Discarded
                    : PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::Idle,
            false, discard, discard);
    }
    const bool legacy_fresh_binding =
        descriptor.ordinal_binding.status == FinalGuestSurfaceStatus::Complete &&
        !descriptor.ordinal_binding.loss.Any() && descriptor.ordinal_binding.exact &&
        descriptor.ordinal_binding.mode ==
            PpUpstreamSampleTeeProducerPreDepthOrdinalContentBindingMode::None &&
        descriptor.ordinal_binding.publish_predecessor_pre &&
        descriptor.ordinal_binding.publish_target_pre &&
        descriptor.ordinal_binding.predecessor_post_equals_fresh_predecessor &&
        descriptor.ordinal_binding.target_pre_post_equals_fresh_target &&
        descriptor.ordinal_binding.fresh_selected_pair_latest_exact &&
        !descriptor.ordinal_binding.direct_selected_same_frame_exact &&
        !descriptor.ordinal_binding.predecessor_post_equals_target_immediate_predecessor &&
        !descriptor.ordinal_binding.target_post_equals_direct_selected &&
        !descriptor.ordinal_binding.generic_nonadmission_exact &&
        descriptor.ordinal_binding.ordinal_pair_exact &&
        descriptor.ordinal_binding.semantic_layout_exact &&
        !descriptor.ordinal_binding.retains_private_endpoints;
    const bool binding_exact =
        IsExactPpUpstreamSampleTeeFreshSelectedPairBinding(descriptor.ordinal_binding) ||
        legacy_fresh_binding;
    if (!binding_exact ||
        !IsExactPpUpstreamSampleTeeOrdinalArmForPair(descriptor.prior_arm,
                                                     descriptor.fresh_selected_pair)) {
        return finish(
            FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1}, fresh_arm, 0, false,
            false, PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::Discarded,
            false, true, true);
    }
    return finish(FinalGuestSurfaceStatus::Complete, {}, fresh_arm, 0, false, false,
                  PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::Published,
                  false, false, false, true);
}

[[nodiscard]] inline std::string
FormatPpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameReport(
    const PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameReport& report) {
    return "FGSCTPDK s=" + std::to_string(report.sequence) +
           " arm=" + std::to_string(report.arm_count) +
           " fresh=" + std::to_string(report.fresh_selected_pair_count) +
           " p=" + std::to_string(report.predecessor_capture_count) + "/" +
           std::to_string(report.predecessor_post_count) +
           " w=" + std::to_string(report.target_capture_count) + "/" +
           std::to_string(report.target_post_count) +
           " bound=" + std::to_string(report.binding_count) +
           " retry=" + std::to_string(report.retry_scheduled_count) +
           " discard=" + std::to_string(report.discarded_tail_count) +
           " idle=" + std::to_string(report.inactive_frame_count) +
           " age=" + std::to_string(report.inactive_flip_count) +
           " st=" + std::to_string(static_cast<u32>(report.status)) +
           " loss=" + std::to_string(report.loss.invalidation) + "/" +
           std::to_string(report.loss.gap) + "/" +
           std::to_string(report.loss.tile_capacity) +
           " df=" + std::to_string(
               static_cast<u32>(report.direct_selected_first_failure));
}

struct PpUpstreamSampleTeeProducerPreDepthOrdinalCaptureRetryDiscardDescriptor {
    PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameResult frame{};
    FinalGuestSurfaceReadbackSlotPool::Token outer_slot{};
    PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan
        predecessor_writer_pre_depth{};
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan producer_pre_depth{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding ordinal_content_binding{};
    PpUpstreamSampleTeeDepthDownloadTails ordinal_depth_download_tails{};
};

struct PpUpstreamSampleTeeProducerPreDepthOrdinalCaptureRetryDiscard {
    FinalGuestSurfaceReadbackSlotPool::Token outer_slot{};
    PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan
        predecessor_writer_pre_depth{};
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan producer_pre_depth{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding ordinal_content_binding{};
    PpUpstreamSampleTeeDepthDownloadTails ordinal_depth_download_tails{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool release_only_outer_slot{};
    bool defer_existing_outer_release{};
    u32 discarded_tail_count{};
    bool retains_tail_bytes{};
    bool retains_endpoints{};
    bool retains_outer_token_after_release{};
    bool pds_unchanged{true};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthOrdinalCaptureRetryDiscard
DiscardPpUpstreamSampleTeeProducerPreDepthOrdinalCaptureForRetry(
    const PpUpstreamSampleTeeProducerPreDepthOrdinalCaptureRetryDiscardDescriptor&
        descriptor) noexcept {
    const bool has_token = static_cast<bool>(descriptor.outer_slot);
    const auto sanitized = [&](FinalGuestSurfaceStatus status, FinalGuestSurfaceLoss loss,
                               u32 discarded = 0) {
        return PpUpstreamSampleTeeProducerPreDepthOrdinalCaptureRetryDiscard{
            .outer_slot = descriptor.outer_slot,
            .status = status,
            .loss = loss,
            .release_only_outer_slot = has_token,
            .defer_existing_outer_release = has_token,
            .discarded_tail_count = discarded,
        };
    };
    const auto& frame = descriptor.frame;
    const bool retry_frame =
        frame.disposition ==
            PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::RetryScheduled &&
        frame.status == FinalGuestSurfaceStatus::Complete && !frame.loss.Any() &&
        frame.next_arm.valid && frame.next_inactive_flip_count == 1 &&
        frame.preserved_for_one_capture_retry && frame.discard_current_ordinal_tails &&
        frame.defer_existing_outer_release && !frame.publish_current_ordinal_tails &&
        frame.retains_private_state && frame.retained_private_state_count == 1 &&
        !frame.retains_tail_bytes && !frame.retains_endpoints && !frame.retains_outer_token;
    if (!retry_frame) {
        return sanitized(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
    }
    if (!has_token) {
        return sanitized(FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1});
    }
    const bool predecessor_present =
        descriptor.predecessor_writer_pre_depth !=
        PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan{};
    const bool producer_present =
        descriptor.producer_pre_depth != PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan{};
    if (!predecessor_present || !producer_present) {
        return sanitized(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
    }
    const auto& binding = descriptor.ordinal_content_binding;
    const auto checked_tails = ValidatePpUpstreamSampleTeeDepthDownloadTails(
        descriptor.ordinal_depth_download_tails);
    const bool exact =
        IsPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailRecordable(
            descriptor.predecessor_writer_pre_depth) &&
        IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(
            descriptor.producer_pre_depth) &&
        descriptor.predecessor_writer_pre_depth.slot == descriptor.outer_slot.slot &&
        descriptor.producer_pre_depth.slot == descriptor.outer_slot.slot &&
        checked_tails.status == FinalGuestSurfaceStatus::Complete && !checked_tails.loss.Any() &&
        checked_tails.predecessor_writer_pre == descriptor.predecessor_writer_pre_depth &&
        checked_tails.producer_pre == descriptor.producer_pre_depth &&
        binding.status == FinalGuestSurfaceStatus::AlreadyConsumed && !binding.loss.Any() &&
        binding.capture_phase_exact && binding.awaiting_fresh_selected_pair && !binding.exact &&
        !binding.publish_predecessor_pre && !binding.publish_target_pre &&
        binding.legacy_target_pre_unchanged && binding.pds_unchanged;
    if (!exact) {
        return sanitized(FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1});
    }
    return sanitized(FinalGuestSurfaceStatus::Complete, {}, 2);
}

[[nodiscard]] inline PpUpstreamSampleTeePredecessorWriterPreDepthBinding
BindPpUpstreamSampleTeePredecessorWriterPreDepthToLaterLoad(
    const PpUpstreamSampleTeePredecessorWriterPreDepthBindDescriptor& descriptor) noexcept {
    const auto reject = [] {
        return PpUpstreamSampleTeePredecessorWriterPreDepthBinding{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1}};
    };
    const auto& load = descriptor.later_load;
    const auto& selected = descriptor.selected_binding;
    const bool endpoint_exact = descriptor.predecessor_post.exact &&
                                descriptor.predecessor_post.output.Valid() &&
                                load.private_predecessor_endpoint.Valid() &&
                                descriptor.predecessor_post.output ==
                                    load.private_predecessor_endpoint;
    const bool load_exact = load.status == FinalGuestSurfaceStatus::Complete &&
                            !load.loss.Any() &&
                            load.load_op ==
                                PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Load &&
                            load.target_first_depth_writer && load.scope_load &&
                            load.prior_endpoint_owns_pre_content && load.metadata_only &&
                            load.exact && load.retains_private_predecessor_endpoint;
    const bool layout_exact = IsPpUpstreamSampleTeeDepthPlaneSemanticLayoutExact(
        descriptor.predecessor_pre_plan.plane,
        descriptor.current_producer_pre_plan.plane);
    const bool selected_exact = selected.status == FinalGuestSurfaceStatus::Complete &&
                                !selected.loss.Any() && selected.exact &&
                                selected.output_endpoint_exact && selected.semantic_layout_exact &&
                                selected.publish_producer_pre;
    const bool exact =
        descriptor.pre_observation_count == 1 && descriptor.post_observation_count == 1 &&
        descriptor.load_observation_count == 1 && descriptor.selected_binding_count == 1 &&
        IsPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailRecordable(
            descriptor.predecessor_pre_plan) &&
        IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(
            descriptor.current_producer_pre_plan) &&
        endpoint_exact && load_exact && layout_exact && selected_exact &&
        descriptor.later_route_exact && descriptor.later_view_exact &&
        descriptor.later_selected_token_current;
    if (!exact) {
        return reject();
    }
    return {.status = FinalGuestSurfaceStatus::Complete,
            .exact = true,
            .publish_predecessor_pre = true,
            .post_endpoint_equals_load_predecessor = true,
            .current_producer_pre_is_predecessor_post_proxy = true,
            .semantic_layout_exact = true,
            .additional_gpu_copy_count = 1};
}

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthScopeEntryOwned
ResetPpUpstreamSampleTeeProducerPreDepthScopeEntryOwned(
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryOwned&) noexcept {
    return {};
}

struct PpUpstreamSampleTeeProducerPreDepthScopeEntryBindDescriptor {
    PpUpstreamSampleTeeProducerPreDepthScopeEntryPlan planned{};
    PpUpstreamSampleTeeProducerPreDepthPost producer_post{};
    PpUpstreamSampleTeeProducerPreDepthBinding selected_binding{};
    PpUpstreamDepthFirstWriterPostObservation first_writer_post{};
    u32 post_observation_count{};
    u32 selected_binding_count{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthScopeEntryOwned
BindPpUpstreamSampleTeeProducerPreDepthScopeEntryProvenance(
    const PpUpstreamSampleTeeProducerPreDepthScopeEntryBindDescriptor& descriptor) noexcept {
    const auto reject = [] {
        return PpUpstreamSampleTeeProducerPreDepthScopeEntryOwned{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1}};
    };
    const auto& plan = descriptor.planned;
    const auto& entry = plan.entry;
    const auto& target = plan.target_intent;
    const auto& post = descriptor.producer_post;
    const auto& output = post.output;
    const bool output_exact =
        post.status == FinalGuestSurfaceStatus::Complete && !post.loss.Any() && post.exact &&
        post.mutation_successor_exact && post.diagnostic_resume_host_successor_exact &&
        output.Valid() && target.Valid() && output.writer.hint == target.hint &&
        output.writer.link == target.link && output.writer.range == target.range &&
        output.writer.backing_generation == target.backing_generation &&
        target.mutation_epoch != std::numeric_limits<u64>::max() &&
        output.writer.mutation_epoch == target.mutation_epoch + 1 &&
        output.guest_rendering_serial == entry.guest_rendering_serial &&
        plan.pre_draw_host_rendering_instance_id != std::numeric_limits<u64>::max() &&
        output.host_rendering_instance_id == plan.pre_draw_host_rendering_instance_id + 1 &&
        output.after_diagnostic_resume;
    const auto first_writer =
        ClassifyPpUpstreamDepthFirstWriter(plan.first_writer_pre,
                                           descriptor.first_writer_post);
    const bool load_matches =
        (entry.load_op == PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Clear &&
         first_writer.target_scope_clear) ||
        (entry.load_op == PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Load &&
         first_writer.target_scope_load);
    const auto& selected = descriptor.selected_binding;
    const bool selected_exact = selected.status == FinalGuestSurfaceStatus::Complete &&
                                !selected.loss.Any() && selected.exact &&
                                selected.output_endpoint_exact &&
                                selected.semantic_layout_exact &&
                                selected.publish_producer_pre;
    if (plan.status != FinalGuestSurfaceStatus::Complete || !plan.record_metadata ||
        !plan.metadata_only || plan.copy_content || plan.additional_gpu_copy_count != 0 ||
        !output_exact || !first_writer.post_exact ||
        !first_writer.target_first_depth_writer || !load_matches || !selected_exact ||
        descriptor.post_observation_count != 1 || descriptor.selected_binding_count != 1) {
        return reject();
    }
    const bool clear =
        entry.load_op == PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Clear;
    return {.status = FinalGuestSurfaceStatus::Complete,
            .load_op = entry.load_op,
            .private_predecessor_endpoint = clear ? PpDepthPredecessorEndpointStamp{}
                                                  : entry.predecessor_endpoint,
            .private_depth_clear_bits = clear ? entry.private_depth_clear_bits : 0,
            .target_first_depth_writer = true,
            .scope_clear = clear,
            .scope_load = !clear,
            .prior_endpoint_owns_pre_content = !clear,
            .has_private_depth_clear_bits = clear,
            .raw_depth_clear_encoding_exact = clear && entry.raw_depth_clear_encoding_exact,
            .metadata_only = true,
            .exact = true,
            .retains_private_predecessor_endpoint = !clear};
}

struct PpUpstreamSampleTeeDownloadPrefixReservationDescriptor {
    bool producer_pre_depth_enabled{};
    u32 base_requested_max_bytes{};
    u32 input_requested_max_bytes{};
};

struct PpUpstreamSampleTeeDownloadPrefixReservation {
    u32 base_max_bytes{};
    u32 input_max_bytes{};
    bool globally_reserved{};
};

[[nodiscard]] constexpr PpUpstreamSampleTeeDownloadPrefixReservation
PlanPpUpstreamSampleTeeDownloadPrefixReservation(
    const PpUpstreamSampleTeeDownloadPrefixReservationDescriptor& descriptor) noexcept {
    const u32 cap = descriptor.producer_pre_depth_enabled
                        ? PpUpstreamSampleTeeDownloadPrefixBytes
                        : PpUpstreamInputSnapshotBytes;
    return {.base_max_bytes = std::min(descriptor.base_requested_max_bytes, cap),
            .input_max_bytes = std::min(descriptor.input_requested_max_bytes, cap),
            .globally_reserved = descriptor.producer_pre_depth_enabled};
}

struct PpUpstreamSampleTeeProducerPreDepthDownloadOrderingDescriptor {
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan layout{};
    u64 absolute_slot_offset{};
    bool slot_acquired_before_tail{};
    bool tail_copy_recorded{};
    bool base_copy_pending{};
};

struct PpUpstreamSampleTeeProducerPreDepthDownloadOrderingPlan {
    PpUpstreamSampleTeeBarrierPlan tail_pre_copy{};
    PpUpstreamSampleTeeBarrierPlan before_base_prefix{};
    PpUpstreamSampleTeeBarrierPlan host_read{};
    bool tail_is_first_copy{};
    bool base_prefix_is_first_use{};
    bool tail_before_base_prefix{};
    bool base_before_host_read{};
};

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthDownloadOrderingPlan
PlanPpUpstreamSampleTeeProducerPreDepthDownloadOrdering(
    const PpUpstreamSampleTeeProducerPreDepthDownloadOrderingDescriptor& descriptor) noexcept {
    if (!IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(descriptor.layout) ||
        descriptor.absolute_slot_offset > std::numeric_limits<u32>::max() ||
        descriptor.absolute_slot_offset + descriptor.layout.plane.plane_offset >
            std::numeric_limits<u32>::max()) {
        return {};
    }
    const auto initial = PpUpstreamSampleTeeStage::AllCommands | PpUpstreamSampleTeeStage::Host;
    const auto initial_access =
        PpUpstreamSampleTeeAccess::MemoryWrite | PpUpstreamSampleTeeAccess::HostRead;
    return {
        .tail_pre_copy = {.src_stages = initial,
                          .src_access = initial_access,
                          .dst_stages = PpUpstreamSampleTeeStage::Transfer,
                          .dst_access = PpUpstreamSampleTeeAccess::TransferWrite,
                          .offset = static_cast<u32>(descriptor.absolute_slot_offset) +
                                    descriptor.layout.plane.plane_offset,
                          .bytes = descriptor.layout.plane.plane_bytes},
        .before_base_prefix = {.src_stages = initial,
                               .src_access = initial_access,
                               .dst_stages = PpUpstreamSampleTeeStage::Transfer,
                               .dst_access = PpUpstreamSampleTeeAccess::TransferWrite,
                               .offset = static_cast<u32>(descriptor.absolute_slot_offset),
                               .bytes = PpUpstreamSampleTeeDownloadPrefixBytes},
        .host_read = {.src_stages = PpUpstreamSampleTeeStage::Transfer,
                      .src_access = PpUpstreamSampleTeeAccess::TransferWrite,
                      .dst_stages = PpUpstreamSampleTeeStage::Host,
                      .dst_access = PpUpstreamSampleTeeAccess::HostRead,
                      .offset = static_cast<u32>(descriptor.absolute_slot_offset) +
                                descriptor.layout.plane.plane_offset,
                      .bytes = descriptor.layout.plane.plane_bytes},
        .tail_is_first_copy = descriptor.slot_acquired_before_tail &&
                              descriptor.tail_copy_recorded,
        .base_prefix_is_first_use = descriptor.base_copy_pending,
        .tail_before_base_prefix = descriptor.tail_copy_recorded && descriptor.base_copy_pending,
        .base_before_host_read = descriptor.base_copy_pending,
    };
}

struct PpUpstreamSampleTeeDeferredReadbackDescriptor {
    bool enabled{};
    bool attempted{};
    FinalGuestSurfaceReadbackSlotPool::Token outer_slot{};
    bool release_only_outer_slot{};
    PpUpstreamSampleTeeCandidateRecord record{};
    PpUpstreamSampleTeeCapturePlan capture{};
    PpUpstreamSampleTeeCommandPlan commands{};
    PpUpstreamSampleTeeControllerState controller{};
    u32 selected_candidate_index{};
    FinalGuestSurfaceStatus capture_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss capture_loss{};
    bool capture_has_copy{};
    u64 mapped_bytes{};
    PpUpstreamSampleTeeTransferredDepthSourcePlan transferred_depth_source{};
    PpUpstreamSampleTeeSelectedDepthPlan selected_depth{};
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan producer_pre_depth{};
    PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan
        predecessor_writer_pre_depth{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding ordinal_content_binding{};
    PpUpstreamSampleTeeDepthDownloadTails ordinal_depth_download_tails{};
};

[[nodiscard]] inline bool IsExactPpUpstreamSampleTeeDeferredOrdinalContent(
    const PpUpstreamSampleTeeDeferredReadbackDescriptor& descriptor) noexcept {
    const auto& binding = descriptor.ordinal_content_binding;
    const auto& tails = descriptor.ordinal_depth_download_tails;
    const auto checked = ValidatePpUpstreamSampleTeeDepthDownloadTails(tails);
    const bool binding_mode_exact =
        IsExactPpUpstreamSampleTeeFreshSelectedPairBinding(binding) !=
            IsExactPpUpstreamSampleTeeDirectSelectedBinding(binding) &&
        (IsExactPpUpstreamSampleTeeFreshSelectedPairBinding(binding) ||
         IsExactPpUpstreamSampleTeeDirectSelectedBinding(binding));
    return descriptor.outer_slot && binding_mode_exact &&
           checked.status == FinalGuestSurfaceStatus::Complete && !checked.loss.Any() &&
           tails.predecessor_writer_pre.slot == descriptor.outer_slot.slot &&
           tails.producer_pre.slot == descriptor.outer_slot.slot &&
           tails.predecessor_writer_pre.slot == descriptor.predecessor_writer_pre_depth.slot &&
           tails.predecessor_writer_pre.slot_stride ==
               descriptor.predecessor_writer_pre_depth.slot_stride &&
           tails.predecessor_writer_pre.plane.plane_offset ==
               descriptor.predecessor_writer_pre_depth.plane.plane_offset &&
           IsPpUpstreamSampleTeeDepthPlaneSemanticLayoutExact(
               tails.predecessor_writer_pre.plane,
               descriptor.predecessor_writer_pre_depth.plane) &&
           tails.producer_pre.slot == descriptor.producer_pre_depth.slot &&
           tails.producer_pre.slot_stride == descriptor.producer_pre_depth.slot_stride &&
           tails.producer_pre.plane.plane_offset ==
               descriptor.producer_pre_depth.plane.plane_offset &&
           IsPpUpstreamSampleTeeDepthPlaneSemanticLayoutExact(
               tails.producer_pre.plane, descriptor.producer_pre_depth.plane);
}

struct PpUpstreamSampleTeeCaptureOutcome {
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy{};
};

[[nodiscard]] inline PpUpstreamSampleTeeCaptureOutcome
ResolvePpUpstreamSampleTeeCaptureOutcome(PpUpstreamSampleTeeCaptureOutcome tee,
                                         PpUpstreamSampleTeeCaptureOutcome base) noexcept {
    const bool tee_unset = tee.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                           !tee.loss.Any() && !tee.copy;
    const bool base_failed = base.status != FinalGuestSurfaceStatus::Complete || base.loss.Any() ||
                             !base.copy;
    return tee_unset && base_failed ? base : tee;
}

struct PpUpstreamSampleTeeDeferredDetachDescriptor {
    bool enabled{};
    bool attempted{};
    FinalGuestSurfaceReadbackSlotPool::Token outer_slot{};
    bool release_only_outer_slot{};
    PpUpstreamSampleTeeCandidateRecord record{};
    PpUpstreamSampleTeeCapturePlan capture{};
    PpUpstreamSampleTeeCommandPlan commands{};
    PpUpstreamSampleTeeControllerState controller{};
    u32 selected_candidate_index{};
    FinalGuestSurfaceStatus capture_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss capture_loss{};
    bool capture_has_copy{};
    u64 mapped_bytes{};
    PpUpstreamSampleTeeTransferredDepthSourcePlan transferred_depth_source{};
    PpUpstreamSampleTeeSelectedDepthPlan selected_depth{};
    PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan producer_pre_depth{};
    PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan
        predecessor_writer_pre_depth{};
    PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding ordinal_content_binding{};
    PpUpstreamSampleTeeDepthDownloadTails ordinal_depth_download_tails{};
};

[[nodiscard]] inline PpUpstreamSampleTeeDeferredReadbackDescriptor
DetachPpUpstreamSampleTeeDeferredReadback(
    const PpUpstreamSampleTeeDeferredDetachDescriptor& descriptor) noexcept {
    if (descriptor.release_only_outer_slot) {
        const bool payload_present =
            descriptor.attempted || descriptor.record != PpUpstreamSampleTeeCandidateRecord{} ||
            descriptor.capture != PpUpstreamSampleTeeCapturePlan{} ||
            descriptor.commands != PpUpstreamSampleTeeCommandPlan{} ||
            descriptor.controller != PpUpstreamSampleTeeControllerState{} ||
            descriptor.selected_candidate_index != 0 ||
            descriptor.capture_status != FinalGuestSurfaceStatus::AlreadyConsumed ||
            descriptor.capture_loss.Any() || descriptor.capture_has_copy ||
            descriptor.mapped_bytes != 0 || descriptor.transferred_depth_source.copy ||
            descriptor.transferred_depth_source.plane.region_count != 0 ||
            descriptor.selected_depth.copy || descriptor.selected_depth.plane.region_count != 0 ||
            descriptor.producer_pre_depth !=
                PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan{} ||
            descriptor.predecessor_writer_pre_depth !=
                PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan{} ||
            descriptor.ordinal_content_binding !=
                PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding{} ||
            descriptor.ordinal_depth_download_tails != PpUpstreamSampleTeeDepthDownloadTails{};
        const bool invalid = !descriptor.enabled || !descriptor.outer_slot || payload_present;
        return {.enabled = true,
                .outer_slot = descriptor.outer_slot,
                .release_only_outer_slot = true,
                .capture_status = invalid ? FinalGuestSurfaceStatus::InvalidationLoss
                                          : FinalGuestSurfaceStatus::AlreadyConsumed,
                .capture_loss = invalid ? FinalGuestSurfaceLoss{.invalidation = 1}
                                        : FinalGuestSurfaceLoss{}};
    }
    if (!descriptor.enabled) {
        return {.outer_slot = descriptor.outer_slot};
    }
    const auto terminal_failure = [](FinalGuestSurfaceStatus status) {
        switch (status) {
        case FinalGuestSurfaceStatus::Unsupported:
        case FinalGuestSurfaceStatus::CapacityLoss:
        case FinalGuestSurfaceStatus::BusyLoss:
        case FinalGuestSurfaceStatus::InvalidationLoss:
        case FinalGuestSurfaceStatus::GapLoss:
            return true;
        default:
            return false;
        }
    };
    const bool complete = descriptor.attempted &&
                          descriptor.capture_status == FinalGuestSurfaceStatus::Complete &&
                          !descriptor.capture_loss.Any() && descriptor.capture_has_copy;
    const bool failed = terminal_failure(descriptor.capture_status) &&
                        descriptor.capture_loss.Any() && !descriptor.capture_has_copy;
    const auto status = complete || failed ? descriptor.capture_status
                                           : FinalGuestSurfaceStatus::InvalidationLoss;
    const auto loss = complete || failed ? descriptor.capture_loss
                                         : FinalGuestSurfaceLoss{.invalidation = 1};
    if (!descriptor.outer_slot && !complete) {
        return {
            .enabled = true,
            .attempted = descriptor.attempted,
            .capture_status = status,
            .capture_loss = loss,
        };
    }
    return {
        .enabled = true,
        .attempted = descriptor.attempted,
        .outer_slot = descriptor.outer_slot,
        .release_only_outer_slot = false,
        .record = descriptor.record,
        .capture = descriptor.capture,
        .commands = descriptor.commands,
        .controller = descriptor.controller,
        .selected_candidate_index = descriptor.selected_candidate_index,
        .capture_status = status,
        .capture_loss = loss,
        .capture_has_copy = complete,
        .mapped_bytes = descriptor.mapped_bytes,
        .transferred_depth_source = descriptor.transferred_depth_source,
        .selected_depth = descriptor.selected_depth,
        .producer_pre_depth = descriptor.producer_pre_depth,
        .predecessor_writer_pre_depth = descriptor.predecessor_writer_pre_depth,
        .ordinal_content_binding = descriptor.ordinal_content_binding,
        .ordinal_depth_download_tails = descriptor.ordinal_depth_download_tails,
    };
}

struct PpUpstreamSampleTeeDeferredReadbackPlan {
    PpUpstreamSampleTeeReadbackHeader expected_header{};
    u64 mapped_bytes{};
    u64 invalidate_offset{};
    u64 invalidate_bytes{};
    u64 header_offset{};
    u64 header_bytes{};
    u64 records_offset{};
    u64 record_offset{};
    u64 records_bytes{};
    u32 record_count{};
    u32 expected_generation{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy{};
    bool borrows_outer_slot{};
    u32 outer_slot_release_attempts{};
    bool retains_mapped_span{};
    bool ordinal_content_exact{};
    u32 ordinal_outer_token_count{};
    u32 ordinal_download_invalidation_count{};
    u32 ordinal_outer_release_count{};
};

[[nodiscard]] inline PpUpstreamSampleTeeDeferredReadbackPlan
PlanPpUpstreamSampleTeeDeferredReadback(
    const PpUpstreamSampleTeeDeferredReadbackDescriptor& descriptor) noexcept {
    const u32 outer_slot_release_attempts = descriptor.outer_slot ? 1u : 0u;
    if (!descriptor.enabled) {
        return {.outer_slot_release_attempts = outer_slot_release_attempts};
    }
    const auto reject = [outer_slot_release_attempts](FinalGuestSurfaceStatus status,
                                                      u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeDeferredReadbackPlan result{
            .status = status,
            .outer_slot_release_attempts = outer_slot_release_attempts,
        };
        result.loss.*member = 1;
        return result;
    };
    if (descriptor.release_only_outer_slot) {
        if (!descriptor.outer_slot) {
            return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                          &FinalGuestSurfaceLoss::invalidation);
        }
        const bool release_only_is_clean =
            descriptor.capture_status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            !descriptor.capture_loss.Any() && !descriptor.capture_has_copy &&
            descriptor.record == PpUpstreamSampleTeeCandidateRecord{} &&
            descriptor.capture == PpUpstreamSampleTeeCapturePlan{} &&
            descriptor.commands == PpUpstreamSampleTeeCommandPlan{} &&
            descriptor.controller == PpUpstreamSampleTeeControllerState{} &&
            descriptor.selected_candidate_index == 0 && descriptor.mapped_bytes == 0 &&
            !descriptor.transferred_depth_source.copy &&
            descriptor.transferred_depth_source.plane.region_count == 0 &&
            !descriptor.selected_depth.copy &&
            descriptor.selected_depth.plane.region_count == 0 &&
            descriptor.producer_pre_depth ==
                PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan{} &&
            descriptor.predecessor_writer_pre_depth ==
                PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan{} &&
            descriptor.ordinal_content_binding ==
                PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding{} &&
            descriptor.ordinal_depth_download_tails == PpUpstreamSampleTeeDepthDownloadTails{};
        if (!release_only_is_clean) {
            if (descriptor.capture_status != FinalGuestSurfaceStatus::AlreadyConsumed &&
                descriptor.capture_loss.Any() && !descriptor.capture_has_copy) {
                return {
                    .status = descriptor.capture_status,
                    .loss = descriptor.capture_loss,
                    .outer_slot_release_attempts = outer_slot_release_attempts,
                };
            }
            return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                          &FinalGuestSurfaceLoss::invalidation);
        }
        return {.outer_slot_release_attempts = outer_slot_release_attempts};
    }
    if (descriptor.capture_status != FinalGuestSurfaceStatus::Complete ||
        descriptor.capture_loss.Any() || !descriptor.capture_has_copy) {
        if (descriptor.capture_status != FinalGuestSurfaceStatus::Complete &&
            descriptor.capture_loss.Any()) {
            return {
                .status = descriptor.capture_status,
                .loss = descriptor.capture_loss,
                .outer_slot_release_attempts = outer_slot_release_attempts,
            };
        }
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (!descriptor.outer_slot) {
        return reject(FinalGuestSurfaceStatus::BusyLoss, &FinalGuestSurfaceLoss::busy);
    }
    const auto& record = descriptor.record;
    const auto& capture = descriptor.capture;
    const auto& commands = descriptor.commands;
    const auto& controller = descriptor.controller;
    const auto propagate_specific_loss =
        [outer_slot_release_attempts](FinalGuestSurfaceStatus status,
                                      const FinalGuestSurfaceLoss& loss) {
            return PpUpstreamSampleTeeDeferredReadbackPlan{
                .status = status,
                .loss = loss,
                .outer_slot_release_attempts = outer_slot_release_attempts,
            };
        };
    const auto has_specific_loss = [](FinalGuestSurfaceStatus status,
                                      const FinalGuestSurfaceLoss& loss) {
        return status != FinalGuestSurfaceStatus::Complete &&
               status != FinalGuestSurfaceStatus::AlreadyConsumed && loss.Any();
    };
    if (has_specific_loss(record.status, record.loss)) {
        return propagate_specific_loss(record.status, record.loss);
    }
    if (has_specific_loss(record.shader.status, record.shader.loss)) {
        return propagate_specific_loss(record.shader.status, record.shader.loss);
    }
    if (has_specific_loss(record.window.status, record.window.loss)) {
        return propagate_specific_loss(record.window.status, record.window.loss);
    }
    if (has_specific_loss(capture.status, capture.loss)) {
        return propagate_specific_loss(capture.status, capture.loss);
    }
    if (has_specific_loss(commands.status, commands.loss)) {
        return propagate_specific_loss(commands.status, commands.loss);
    }
    if (has_specific_loss(controller.status, controller.loss)) {
        return propagate_specific_loss(controller.status, controller.loss);
    }
    const u64 expected_pixels = static_cast<u64>(record.window.width) * record.window.height;
    const u64 slot_offset =
        static_cast<u64>(descriptor.outer_slot.slot) * Shader::PpUpstreamSampleTeeSlotStrideBytes;
    const u64 header_bytes = sizeof(PpUpstreamSampleTeeReadbackHeader);
    const u64 records_bytes = expected_pixels * sizeof(PpUpstreamSampleTeePixelRecord);
    const auto expected_header = PpUpstreamSampleTeeReadbackHeader{
        .magic = Shader::PpUpstreamSampleTeeHeaderMagic,
        .generation = descriptor.outer_slot.generation,
        .candidate_index = descriptor.selected_candidate_index,
        .x = record.window.x,
        .y = record.window.y,
        .width = record.window.width,
        .height = record.window.height,
        .pixel_count = record.window.pixel_count,
    };
    const auto fits = [](u64 offset, u64 bytes, u64 size) {
        return offset <= size && bytes <= size - offset;
    };
    const auto canonical_commands = PlanPpUpstreamSampleTeeCommands({
        .enabled = true,
        .binding = {.status = FinalGuestSurfaceStatus::Complete, .bind = true},
        .capture = capture,
        .non_coherent_atom_size = Shader::PpUpstreamSampleTeeSlotStrideBytes,
        .output_candidate_count = 1,
        .direct_draw = true,
        .rendering_active = true,
        .observed_rendering_serial = controller.serial,
        .expected_rendering_serial = controller.serial,
    });
    const auto same_barrier = [](const PpUpstreamSampleTeeBarrierPlan& lhs,
                                 const PpUpstreamSampleTeeBarrierPlan& rhs) {
        return lhs.src_stages == rhs.src_stages && lhs.src_access == rhs.src_access &&
               lhs.dst_stages == rhs.dst_stages && lhs.dst_access == rhs.dst_access &&
               lhs.offset == rhs.offset && lhs.bytes == rhs.bytes;
    };
    const bool commands_are_canonical =
        canonical_commands.status == FinalGuestSurfaceStatus::Complete &&
        !canonical_commands.loss.Any() && commands.operations == canonical_commands.operations &&
        commands.operation_count == canonical_commands.operation_count &&
        same_barrier(commands.reuse_barrier, canonical_commands.reuse_barrier) &&
        same_barrier(commands.fragment_barrier, canonical_commands.fragment_barrier) &&
        same_barrier(commands.host_barrier, canonical_commands.host_barrier) &&
        commands.partial_push_stages == canonical_commands.partial_push_stages &&
        commands.partial_push_offset == canonical_commands.partial_push_offset &&
        commands.partial_push_bytes == canonical_commands.partial_push_bytes &&
        commands.serial == canonical_commands.serial &&
        commands.scheduler_waits == canonical_commands.scheduler_waits &&
        commands.scheduler_finishes == canonical_commands.scheduler_finishes &&
        commands.requires_allocation == canonical_commands.requires_allocation &&
        commands.bind_buffer == canonical_commands.bind_buffer &&
        commands.initialize == canonical_commands.initialize &&
        commands.arm == canonical_commands.arm &&
        commands.issue_draw == canonical_commands.issue_draw &&
        commands.require_post_barrier == canonical_commands.require_post_barrier &&
        commands.defer_readback == canonical_commands.defer_readback &&
        commands.force_load == canonical_commands.force_load &&
        commands.require_preserved_serial == canonical_commands.require_preserved_serial &&
        commands.logs_private_state == canonical_commands.logs_private_state;
    const u32 expected_base_dword = static_cast<u32>(slot_offset / sizeof(u32));
    const u32 expected_control =
        Shader::PpUpstreamSampleTeeArmMask | record.shader.resource_index |
        (descriptor.selected_candidate_index << Shader::PpUpstreamSampleTeeCandidateShift);
    if (!IsCanonicalPpUpstreamSampleTeeCandidateRecord(record,
                                                       descriptor.selected_candidate_index) ||
        controller.status != FinalGuestSurfaceStatus::Complete || controller.loss.Any() ||
        controller.active || !controller.ready || !controller.initialized || !controller.armed ||
        !controller.draw_executed || !controller.post_barrier_complete ||
        controller.candidate_index != descriptor.selected_candidate_index ||
        controller.serial == 0 || commands.status != FinalGuestSurfaceStatus::Complete ||
        commands.loss.Any() || commands.serial == 0 || commands.serial != controller.serial ||
        !commands_are_canonical || capture.status != FinalGuestSurfaceStatus::Complete ||
        capture.loss.Any() || !capture.initialize || !capture.arm ||
        !capture.require_post_barrier || capture.logs_private_state ||
        capture.header != expected_header || capture.slot_offset != slot_offset ||
        capture.slot_bytes != Shader::PpUpstreamSampleTeeSlotStrideBytes ||
        capture.clear_offset != slot_offset + header_bytes ||
        capture.clear_bytes != capture.slot_bytes - header_bytes ||
        capture.base_dword != expected_base_dword ||
        !Shader::IsValidPpUpstreamSampleTeeBaseDword(capture.base_dword) ||
        capture.control != expected_control ||
        capture.push_offset !=
            static_cast<u32>(offsetof(Shader::PushData, pp_upstream_sample_tee_control)) ||
        capture.push_bytes != sizeof(u32) * 2 || capture.image_operations != 0 ||
        capture.texture_operations != 0 || capture.scheduler_waits != 0 ||
        capture.scheduler_finishes != 0 ||
        descriptor.mapped_bytes != Shader::PpUpstreamSampleTeeBufferBytes ||
        !fits(slot_offset, capture.slot_bytes, descriptor.mapped_bytes) ||
        !fits(slot_offset, header_bytes, descriptor.mapped_bytes) ||
        !fits(capture.clear_offset, records_bytes, descriptor.mapped_bytes) ||
        records_bytes > capture.clear_bytes) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    return {
        .expected_header = expected_header,
        .mapped_bytes = descriptor.mapped_bytes,
        .invalidate_offset = slot_offset,
        .invalidate_bytes = capture.slot_bytes,
        .header_offset = slot_offset,
        .header_bytes = header_bytes,
        .records_offset = capture.clear_offset,
        .record_offset = capture.clear_offset,
        .records_bytes = records_bytes,
        .record_count = record.window.pixel_count,
        .expected_generation = descriptor.outer_slot.generation,
        .status = FinalGuestSurfaceStatus::Complete,
        .copy = true,
        .borrows_outer_slot = true,
        .outer_slot_release_attempts = outer_slot_release_attempts,
        .ordinal_content_exact =
            IsExactPpUpstreamSampleTeeDeferredOrdinalContent(descriptor),
        .ordinal_outer_token_count =
            IsExactPpUpstreamSampleTeeDeferredOrdinalContent(descriptor) ? 1u : 0u,
        .ordinal_download_invalidation_count =
            IsExactPpUpstreamSampleTeeDeferredOrdinalContent(descriptor) ? 1u : 0u,
        .ordinal_outer_release_count =
            IsExactPpUpstreamSampleTeeDeferredOrdinalContent(descriptor) ? 1u : 0u,
    };
}

struct PpUpstreamSampleTeeOwnedReadback {
    PpUpstreamSampleTeeReadbackHeader header{};
    std::vector<PpUpstreamSampleTeePixelRecord> records{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool mapped_invalidation_succeeded{};
    bool bounded_record_copy_succeeded{};
    bool retains_mapped_span{};
};

struct PpUpstreamSampleTeeTransferredDepthSourceOwned {
    PpUpstreamInputContentPlane plane{};
    std::vector<std::byte> bytes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool mapped_invalidation_succeeded{};
    bool bounded_copy_succeeded{};
    bool shared_release_deferred{};
    bool retains_source_token{};
    bool retains_values{};
};

[[nodiscard]] inline PpUpstreamSampleTeeTransferredDepthSourceOwned
CopyPpUpstreamSampleTeeTransferredDepthSource(
    const PpUpstreamSampleTeeTransferredDepthSourcePlan& plan,
    bool mapped_invalidation_succeeded, std::span<const std::byte> shared_slot) {
    const auto reject = [mapped_invalidation_succeeded] {
        return PpUpstreamSampleTeeTransferredDepthSourceOwned{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
            .mapped_invalidation_succeeded = mapped_invalidation_succeeded,
            .shared_release_deferred = true,
        };
    };
    if (!IsPpUpstreamSampleTeeTransferredDepthSourcePlanRecordable(plan) ||
        !mapped_invalidation_succeeded || plan.total_bytes > shared_slot.size() ||
        static_cast<u64>(plan.plane.plane_offset) + plan.plane.plane_bytes > shared_slot.size()) {
        return reject();
    }
    PpUpstreamSampleTeeTransferredDepthSourceOwned result{
        .plane = plan.plane,
        .status = FinalGuestSurfaceStatus::Complete,
        .mapped_invalidation_succeeded = true,
        .bounded_copy_succeeded = true,
        .shared_release_deferred = true,
    };
    result.bytes.resize(plan.plane.plane_bytes);
    std::ranges::copy(shared_slot.subspan(plan.plane.plane_offset, plan.plane.plane_bytes),
                      result.bytes.begin());
    return result;
}

struct PpUpstreamSampleTeeSelectedDepthOwned {
    PpUpstreamSampleTeeSelectedDepthPlane plane{};
    std::vector<std::byte> bytes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool mapped_invalidation_succeeded{};
    bool bounded_copy_succeeded{};
    bool shared_release_deferred{};
    bool retains_selected_token{};
    bool retains_values{};
};

struct PpUpstreamSampleTeeProducerPreDepthOwned {
    PpUpstreamSampleTeeSelectedDepthPlane plane{};
    std::vector<std::byte> bytes{};
    PpUpstreamSampleTeeProducerPreDepthScopeEntryOwned scope_entry{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool mapped_invalidation_succeeded{};
    bool bounded_copy_succeeded{};
    bool shared_release_deferred{};
    bool retains_source_token{};
    bool retains_private_hint{};
    bool retains_values{};
};

struct PpUpstreamSampleTeePredecessorWriterPreDepthOwned {
    PpUpstreamSampleTeeSelectedDepthPlane plane{};
    std::vector<std::byte> bytes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool mapped_invalidation_succeeded{};
    bool bounded_copy_succeeded{};
    bool shared_release_deferred{};
    bool exact_load_endpoint_bound{};
    bool current_producer_pre_is_post_proxy{};
    u32 additional_gpu_copy_count{};
    bool retains_source_token{};
    bool retains_private_hint{};
    bool retains_values{};
};

[[nodiscard]] inline bool IsCanonicalPpUpstreamSampleTeePredecessorWriterPreDepthOwned(
    const PpUpstreamSampleTeePredecessorWriterPreDepthOwned& value) noexcept {
    const auto& plane = value.plane;
    if (value.status != FinalGuestSurfaceStatus::Complete || value.loss.Any() ||
        !value.mapped_invalidation_succeeded || !value.bounded_copy_succeeded ||
        !value.shared_release_deferred || !value.exact_load_endpoint_bound ||
        !value.current_producer_pre_is_post_proxy || value.additional_gpu_copy_count != 1 ||
        value.retains_source_token || value.retains_private_hint || value.retains_values ||
        plane.status != FinalGuestSurfaceStatus::Complete || plane.loss.Any() ||
        plane.region_count == 0 ||
        plane.region_count > FinalGuestSurfaceWatchOrdinals::MaxOrdinals ||
        plane.plane_offset != PpUpstreamSampleTeePredecessorWriterPreDepthTailOffset ||
        plane.plane_bytes == 0 ||
        plane.plane_bytes > PpUpstreamSampleTeePredecessorWriterPreDepthTailBytes ||
        static_cast<u64>(plane.plane_offset) + plane.plane_bytes >
            PpUpstreamSampleTeeProducerPreDepthTailOffset ||
        plane.bytes_per_texel != 4 || plane.base_mip != 0 || plane.mip_count != 1 ||
        plane.base_layer != 0 || value.bytes.size() != plane.plane_bytes ||
        value.bytes.capacity() > PpUpstreamSampleTeePredecessorWriterPreDepthTailBytes) {
        return false;
    }
    u64 covered{};
    for (u32 index = 0; index < plane.region_count; ++index) {
        const auto& region = plane.regions[index];
        if (region.logical_ordinal == 0 || region.mip_level != 0 || region.width == 0 ||
            region.height == 0 || region.byte_size == 0 ||
            region.byte_size % plane.bytes_per_texel != 0 ||
            region.buffer_offset != covered ||
            static_cast<u64>(region.buffer_offset) + region.byte_size > plane.plane_bytes) {
            return false;
        }
        covered += region.byte_size;
    }
    return covered == plane.plane_bytes;
}

[[nodiscard]] inline PpUpstreamSampleTeePredecessorWriterPreDepthOwned
CopyPpUpstreamSampleTeePredecessorWriterPreDepth(
    const PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan& plan,
    bool mapped_invalidation_succeeded, std::span<const std::byte> download_slot) {
    const auto reject = [mapped_invalidation_succeeded, &plan] {
        return PpUpstreamSampleTeePredecessorWriterPreDepthOwned{
            .status = plan.status == FinalGuestSurfaceStatus::GapLoss
                          ? FinalGuestSurfaceStatus::GapLoss
                          : FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = plan.status == FinalGuestSurfaceStatus::GapLoss
                        ? FinalGuestSurfaceLoss{.gap = 1}
                        : FinalGuestSurfaceLoss{.invalidation = 1},
            .mapped_invalidation_succeeded = mapped_invalidation_succeeded,
            .shared_release_deferred = true};
    };
    if (!IsPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailRecordable(plan) ||
        !mapped_invalidation_succeeded || plan.plane.plane_offset > download_slot.size() ||
        plan.plane.plane_bytes > download_slot.size() - plan.plane.plane_offset) {
        return reject();
    }
    PpUpstreamSampleTeePredecessorWriterPreDepthOwned result{
        .plane = plan.plane,
        .status = FinalGuestSurfaceStatus::Complete,
        .mapped_invalidation_succeeded = true,
        .bounded_copy_succeeded = true,
        .shared_release_deferred = true,
        .exact_load_endpoint_bound = true,
        .current_producer_pre_is_post_proxy = true,
        .additional_gpu_copy_count = 1};
    result.bytes.resize(plan.plane.plane_bytes);
    std::ranges::copy(
        download_slot.subspan(plan.plane.plane_offset, plan.plane.plane_bytes),
        result.bytes.begin());
    return result;
}

[[nodiscard]] inline PpUpstreamSampleTeeProducerPreDepthOwned
CopyPpUpstreamSampleTeeProducerPreDepth(
    const PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan& plan,
    bool mapped_invalidation_succeeded, std::span<const std::byte> download_slot) {
    const auto reject = [mapped_invalidation_succeeded] {
        return PpUpstreamSampleTeeProducerPreDepthOwned{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
            .mapped_invalidation_succeeded = mapped_invalidation_succeeded,
            .shared_release_deferred = true,
        };
    };
    if (!IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(plan) ||
        !mapped_invalidation_succeeded || plan.plane.plane_offset > download_slot.size() ||
        plan.plane.plane_bytes > download_slot.size() - plan.plane.plane_offset) {
        return reject();
    }
    PpUpstreamSampleTeeProducerPreDepthOwned result{
        .plane = plan.plane,
        .status = FinalGuestSurfaceStatus::Complete,
        .mapped_invalidation_succeeded = true,
        .bounded_copy_succeeded = true,
        .shared_release_deferred = true,
    };
    result.bytes.resize(plan.plane.plane_bytes);
    std::ranges::copy(
        download_slot.subspan(plan.plane.plane_offset, plan.plane.plane_bytes),
        result.bytes.begin());
    return result;
}

[[nodiscard]] inline PpUpstreamSampleTeeSelectedDepthOwned
CopyPpUpstreamSampleTeeSelectedDepth(const PpUpstreamSampleTeeSelectedDepthPlan& plan,
                                     bool mapped_invalidation_succeeded,
                                     std::span<const std::byte> shared_slot) {
    const auto reject = [mapped_invalidation_succeeded] {
        return PpUpstreamSampleTeeSelectedDepthOwned{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
            .mapped_invalidation_succeeded = mapped_invalidation_succeeded,
            .shared_release_deferred = true,
        };
    };
    if (!IsPpUpstreamSampleTeeSelectedDepthPlanRecordable(plan) ||
        !mapped_invalidation_succeeded || plan.total_bytes > shared_slot.size() ||
        static_cast<u64>(plan.plane.plane_offset) + plan.plane.plane_bytes > shared_slot.size()) {
        return reject();
    }
    PpUpstreamSampleTeeSelectedDepthOwned result{
        .plane = plan.plane,
        .status = FinalGuestSurfaceStatus::Complete,
        .mapped_invalidation_succeeded = true,
        .bounded_copy_succeeded = true,
        .shared_release_deferred = true,
    };
    result.bytes.resize(plan.plane.plane_bytes);
    std::ranges::copy(shared_slot.subspan(plan.plane.plane_offset, plan.plane.plane_bytes),
                      result.bytes.begin());
    return result;
}

[[nodiscard]] inline bool IsCanonicalPpUpstreamSampleTeeDeferredReadbackPlan(
    const PpUpstreamSampleTeeDeferredReadbackPlan& plan) noexcept {
    const auto fits = [](u64 offset, u64 bytes, u64 size) {
        return offset <= size && bytes <= size - offset;
    };
    constexpr u64 MaxRecords = static_cast<u64>(Shader::PpUpstreamSampleTeeMaxDimension) *
                               Shader::PpUpstreamSampleTeeMaxDimension;
    const u64 expected_pixels =
        static_cast<u64>(plan.expected_header.width) * plan.expected_header.height;
    return plan.status == FinalGuestSurfaceStatus::Complete && !plan.loss.Any() && plan.copy &&
           plan.borrows_outer_slot && plan.outer_slot_release_attempts == 1 &&
           !plan.retains_mapped_span &&
           plan.mapped_bytes == Shader::PpUpstreamSampleTeeBufferBytes &&
           plan.invalidate_bytes == Shader::PpUpstreamSampleTeeSlotStrideBytes &&
           plan.invalidate_offset % Shader::PpUpstreamSampleTeeSlotStrideBytes == 0 &&
           plan.invalidate_offset / Shader::PpUpstreamSampleTeeSlotStrideBytes <
               Shader::PpUpstreamSampleTeeSlotCount &&
           plan.header_offset == plan.invalidate_offset &&
           plan.header_bytes == sizeof(PpUpstreamSampleTeeReadbackHeader) &&
           plan.records_offset == plan.header_offset + plan.header_bytes &&
           plan.record_count != 0 && plan.record_count <= MaxRecords &&
           plan.records_bytes ==
               static_cast<u64>(plan.record_count) * sizeof(PpUpstreamSampleTeePixelRecord) &&
           plan.records_bytes <= plan.invalidate_bytes - plan.header_bytes &&
           plan.expected_generation != 0 &&
           plan.expected_header.magic == Shader::PpUpstreamSampleTeeHeaderMagic &&
           plan.expected_header.generation == plan.expected_generation &&
           plan.expected_header.candidate_index < 4 && plan.expected_header.width != 0 &&
           plan.expected_header.height != 0 &&
           plan.expected_header.width <= Shader::PpUpstreamSampleTeeMaxDimension &&
           plan.expected_header.height <= Shader::PpUpstreamSampleTeeMaxDimension &&
           expected_pixels == plan.record_count &&
           plan.expected_header.pixel_count == plan.record_count &&
           fits(plan.invalidate_offset, plan.invalidate_bytes, plan.mapped_bytes) &&
           fits(plan.header_offset - plan.invalidate_offset, plan.header_bytes,
                plan.invalidate_bytes) &&
           fits(plan.records_offset - plan.invalidate_offset, plan.records_bytes,
                plan.invalidate_bytes);
}

[[nodiscard]] inline bool HasEmptyPpUpstreamSampleTeeDeferredReadbackPayload(
    const PpUpstreamSampleTeeDeferredReadbackPlan& plan) noexcept {
    return plan.expected_header == PpUpstreamSampleTeeReadbackHeader{} && plan.mapped_bytes == 0 &&
           plan.invalidate_offset == 0 && plan.invalidate_bytes == 0 && plan.header_offset == 0 &&
           plan.header_bytes == 0 && plan.records_offset == 0 && plan.records_bytes == 0 &&
           plan.record_count == 0 && plan.expected_generation == 0 && !plan.copy &&
           !plan.borrows_outer_slot && !plan.retains_mapped_span;
}

[[nodiscard]] inline PpUpstreamSampleTeeOwnedReadback CopyPpUpstreamSampleTeeDeferredReadback(
    const PpUpstreamSampleTeeDeferredReadbackPlan& plan, bool mapped_invalidation_succeeded,
    std::span<const std::byte> mapped) {
    const auto reject = [mapped_invalidation_succeeded] {
        return PpUpstreamSampleTeeOwnedReadback{
            .status = FinalGuestSurfaceStatus::InvalidationLoss,
            .loss = {.invalidation = 1},
            .mapped_invalidation_succeeded = mapped_invalidation_succeeded,
        };
    };
    if (plan.status != FinalGuestSurfaceStatus::Complete || plan.loss.Any() || !plan.copy) {
        if (plan.status != FinalGuestSurfaceStatus::Complete && plan.loss.Any()) {
            return {
                .status = plan.status,
                .loss = plan.loss,
                .mapped_invalidation_succeeded = mapped_invalidation_succeeded,
            };
        }
        return reject();
    }
    if (!IsCanonicalPpUpstreamSampleTeeDeferredReadbackPlan(plan) ||
        !mapped_invalidation_succeeded || mapped.size() != plan.mapped_bytes) {
        return reject();
    }
    PpUpstreamSampleTeeOwnedReadback result{
        .status = FinalGuestSurfaceStatus::Complete,
        .mapped_invalidation_succeeded = true,
    };
    std::memcpy(&result.header, mapped.data() + plan.header_offset, sizeof(result.header));
    if (result.header != plan.expected_header) {
        return reject();
    }
    result.records.resize(plan.record_count);
    std::memcpy(result.records.data(), mapped.data() + plan.records_offset, plan.records_bytes);
    result.bounded_record_copy_succeeded = true;
    return result;
}

struct PpUpstreamSampleTeeDeferredLifecycleState {
    PpUpstreamSampleTeeReadbackHeader expected_header{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    u32 expected_record_count{};
    u32 expected_outer_slot_release_attempts{};
    u32 owned_copy_observations{};
    u32 observed_outer_slot_release_attempts{};
    bool expects_owned_copy{};
    bool mapped_invalidation_succeeded{};
    bool bounded_record_copy_succeeded{};
    bool bounded_copy_completed_before_outer_slot_release{};
    bool slot_release_succeeded{};
    bool lifecycle_invalid{};
    bool finalized{};

    bool operator==(const PpUpstreamSampleTeeDeferredLifecycleState&) const = default;
};

[[nodiscard]] inline PpUpstreamSampleTeeDeferredLifecycleState
InvalidatePpUpstreamSampleTeeDeferredLifecycle(
    PpUpstreamSampleTeeDeferredLifecycleState state) noexcept {
    state.lifecycle_invalid = true;
    state.loss.invalidation = 1;
    if (state.status == FinalGuestSurfaceStatus::Complete ||
        state.status == FinalGuestSurfaceStatus::AlreadyConsumed) {
        state.status = FinalGuestSurfaceStatus::InvalidationLoss;
    }
    state.mapped_invalidation_succeeded = false;
    state.bounded_record_copy_succeeded = false;
    state.bounded_copy_completed_before_outer_slot_release = false;
    return state;
}

[[nodiscard]] inline PpUpstreamSampleTeeDeferredLifecycleState
BeginPpUpstreamSampleTeeDeferredLifecycle(
    const PpUpstreamSampleTeeDeferredReadbackPlan& plan) noexcept {
    PpUpstreamSampleTeeDeferredLifecycleState state{
        .expected_header = plan.expected_header,
        .status = plan.status,
        .loss = plan.loss,
        .expected_record_count = plan.record_count,
        .expected_outer_slot_release_attempts = plan.outer_slot_release_attempts,
        .expects_owned_copy =
            plan.status == FinalGuestSurfaceStatus::Complete && !plan.loss.Any() && plan.copy,
    };
    const bool valid_copy =
        !state.expects_owned_copy || IsCanonicalPpUpstreamSampleTeeDeferredReadbackPlan(plan);
    const bool valid_empty_payload =
        state.expects_owned_copy || HasEmptyPpUpstreamSampleTeeDeferredReadbackPayload(plan);
    const bool valid_loss =
        state.expects_owned_copy ||
        (plan.status == FinalGuestSurfaceStatus::AlreadyConsumed && !plan.loss.Any()) ||
        (plan.status != FinalGuestSurfaceStatus::Complete &&
         plan.status != FinalGuestSurfaceStatus::AlreadyConsumed && plan.loss.Any());
    if (plan.outer_slot_release_attempts > 1 || !valid_copy || !valid_empty_payload ||
        !valid_loss) {
        return InvalidatePpUpstreamSampleTeeDeferredLifecycle(state);
    }
    return state;
}

[[nodiscard]] inline PpUpstreamSampleTeeDeferredLifecycleState ObservePpUpstreamSampleTeeOwnedCopy(
    PpUpstreamSampleTeeDeferredLifecycleState state,
    const PpUpstreamSampleTeeOwnedReadback& owned) noexcept {
    if (state.owned_copy_observations != std::numeric_limits<u32>::max()) {
        ++state.owned_copy_observations;
    }
    if (!state.expects_owned_copy || state.owned_copy_observations != 1 ||
        state.observed_outer_slot_release_attempts != 0 || state.lifecycle_invalid ||
        state.finalized || state.status != FinalGuestSurfaceStatus::Complete || state.loss.Any()) {
        return InvalidatePpUpstreamSampleTeeDeferredLifecycle(state);
    }
    constexpr size_t MaxRecords = static_cast<size_t>(Shader::PpUpstreamSampleTeeMaxDimension) *
                                  Shader::PpUpstreamSampleTeeMaxDimension;
    if (owned.records.capacity() > MaxRecords) {
        return InvalidatePpUpstreamSampleTeeDeferredLifecycle(state);
    }
    if (owned.status != FinalGuestSurfaceStatus::Complete || owned.loss.Any() ||
        !owned.mapped_invalidation_succeeded || !owned.bounded_record_copy_succeeded ||
        owned.retains_mapped_span || owned.header != state.expected_header ||
        owned.records.size() != state.expected_record_count) {
        if (owned.status != FinalGuestSurfaceStatus::Complete && owned.loss.Any()) {
            state.status = owned.status;
            state.loss = owned.loss;
            return state;
        }
        return InvalidatePpUpstreamSampleTeeDeferredLifecycle(state);
    }
    state.mapped_invalidation_succeeded = true;
    state.bounded_record_copy_succeeded = true;
    return state;
}

[[nodiscard]] inline PpUpstreamSampleTeeDeferredLifecycleState
ObservePpUpstreamSampleTeeOuterSlotRelease(PpUpstreamSampleTeeDeferredLifecycleState state,
                                           bool release_succeeded) noexcept {
    if (state.observed_outer_slot_release_attempts != std::numeric_limits<u32>::max()) {
        ++state.observed_outer_slot_release_attempts;
    }
    const bool exact_attempt =
        state.observed_outer_slot_release_attempts == state.expected_outer_slot_release_attempts;
    const bool copy_phase_preceded_release =
        !state.expects_owned_copy || state.owned_copy_observations == 1;
    const bool successful_copy_preceded_release =
        state.status != FinalGuestSurfaceStatus::Complete || state.bounded_record_copy_succeeded;
    if (!exact_attempt || !copy_phase_preceded_release || !successful_copy_preceded_release ||
        !release_succeeded || state.lifecycle_invalid || state.finalized) {
        state.slot_release_succeeded = release_succeeded;
        return InvalidatePpUpstreamSampleTeeDeferredLifecycle(state);
    }
    state.bounded_copy_completed_before_outer_slot_release =
        state.expects_owned_copy && state.bounded_record_copy_succeeded;
    state.slot_release_succeeded = true;
    return state;
}

[[nodiscard]] inline PpUpstreamSampleTeeDeferredLifecycleState
FinalizePpUpstreamSampleTeeDeferredLifecycle(
    PpUpstreamSampleTeeDeferredLifecycleState state) noexcept {
    if (state.finalized) {
        return InvalidatePpUpstreamSampleTeeDeferredLifecycle(state);
    }
    state.finalized = true;
    const bool exact_release_count =
        state.observed_outer_slot_release_attempts == state.expected_outer_slot_release_attempts;
    const bool release_complete =
        state.expected_outer_slot_release_attempts == 0 || state.slot_release_succeeded;
    const bool copy_phase_complete =
        !state.expects_owned_copy || state.owned_copy_observations == 1;
    const bool successful_copy_complete =
        state.status != FinalGuestSurfaceStatus::Complete ||
        (state.mapped_invalidation_succeeded && state.bounded_record_copy_succeeded &&
         state.bounded_copy_completed_before_outer_slot_release);
    if (state.lifecycle_invalid || !exact_release_count || !release_complete ||
        !copy_phase_complete || !successful_copy_complete) {
        return InvalidatePpUpstreamSampleTeeDeferredLifecycle(state);
    }
    return state;
}

struct PpUpstreamSampleTeeSharedSlotReleaseResult {
    PpUpstreamInputPublication base{};
    PpUpstreamSampleTeeDeferredLifecycleState tee{};
};

[[nodiscard]] inline PpUpstreamSampleTeeSharedSlotReleaseResult
ReconcilePpUpstreamSampleTeeSharedSlotRelease(
    bool had_slot, bool shared_release_succeeded, FinalGuestSurfaceStatus base_status,
    FinalGuestSurfaceLoss base_loss, bool base_copy,
    PpUpstreamSampleTeeDeferredLifecycleState tee_after_owned_copy) noexcept {
    const bool incoherent_no_slot =
        !had_slot && (shared_release_succeeded || base_status == FinalGuestSurfaceStatus::Complete ||
                      base_copy);
    auto base = incoherent_no_slot
                    ? PpUpstreamInputPublication{
                          .status = FinalGuestSurfaceStatus::InvalidationLoss,
                          .loss = {.invalidation = 1},
                      }
                    : ReconcilePpUpstreamInputSlotRelease(
                          had_slot, shared_release_succeeded, base_status, base_loss, base_copy);
    auto tee = tee_after_owned_copy;
    if (had_slot) {
        tee = ObservePpUpstreamSampleTeeOuterSlotRelease(tee, shared_release_succeeded);
    }
    tee = FinalizePpUpstreamSampleTeeDeferredLifecycle(tee);
    return {.base = base, .tee = tee};
}

struct PpUpstreamSampleTeeRuntimeReconciliation {
    bool enabled{};
    PpUpstreamSampleTeeCandidateRecord record{};
    u32 selected_candidate_index{};
    FinalGuestSurfaceStatus capture_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss capture_loss{};
    bool capture_has_copy{};
    bool initialized{};
    bool armed{};
    bool draw_executed{};
    bool post_barrier_complete{};
    bool slot_present{};
    bool mapped_invalidation_succeeded{};
    bool bounded_record_copy_succeeded{};
    u64 rendering_serial{};
    u32 expected_generation{};
    PpUpstreamSampleTeeReadbackHeader header{};
    u32 observed_outer_slot_release_attempts{};
    bool bounded_copy_completed_before_outer_slot_release{};
    bool slot_release_succeeded{};
};

[[nodiscard]] inline PpUpstreamSampleTeeCandidatePublication ReconcilePpUpstreamSampleTeeRuntime(
    const PpUpstreamSampleTeeRuntimeReconciliation& reconciliation) noexcept {
    if (!reconciliation.enabled) {
        return {};
    }
    const auto reject = [](FinalGuestSurfaceStatus status, u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleTeeCandidatePublication result{.status = status};
        result.loss.*member = 1;
        return result;
    };
    if (reconciliation.capture_status != FinalGuestSurfaceStatus::Complete ||
        reconciliation.capture_loss.Any() || !reconciliation.capture_has_copy) {
        if (reconciliation.capture_status != FinalGuestSurfaceStatus::Complete &&
            reconciliation.capture_loss.Any()) {
            return {
                .status = reconciliation.capture_status,
                .loss = reconciliation.capture_loss,
            };
        }
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    if (!reconciliation.initialized || !reconciliation.armed || !reconciliation.draw_executed ||
        !reconciliation.post_barrier_complete) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    auto publication = ReconcilePpUpstreamSampleTeeCandidate({
        .enabled = true,
        .record = reconciliation.record,
        .selected_candidate_index = reconciliation.selected_candidate_index,
    });
    if (!publication.publish) {
        return publication;
    }
    if (!reconciliation.slot_present) {
        return reject(FinalGuestSurfaceStatus::BusyLoss, &FinalGuestSurfaceLoss::busy);
    }
    if (!reconciliation.mapped_invalidation_succeeded ||
        !reconciliation.bounded_record_copy_succeeded || reconciliation.rendering_serial == 0 ||
        reconciliation.expected_generation == 0) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    const auto& window = reconciliation.record.window;
    const PpUpstreamSampleTeeReadbackHeader expected{
        .magic = Shader::PpUpstreamSampleTeeHeaderMagic,
        .generation = reconciliation.expected_generation,
        .candidate_index = reconciliation.selected_candidate_index,
        .x = window.x,
        .y = window.y,
        .width = window.width,
        .height = window.height,
        .pixel_count = window.pixel_count,
    };
    if (reconciliation.header != expected ||
        reconciliation.observed_outer_slot_release_attempts != 1 ||
        !reconciliation.bounded_copy_completed_before_outer_slot_release ||
        !reconciliation.slot_release_succeeded) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    return publication;
}

struct PpUpstreamSampleTeeDeferredObservation {
    PpUpstreamSampleTeeDeferredLifecycleState lifecycle{};
    PpUpstreamSampleTeeCandidatePublication publication{};
    PpUpstreamSampleTeeCandidateRecord record{};
    PpUpstreamSampleTeeOwnedReadback owned{};
    PpUpstreamSampleTeeTransferredDepthSourceOwned transferred_depth_source{};
    PpUpstreamSampleTeeSelectedDepthOwned selected_depth{};
    PpUpstreamSampleTeeProducerPreDepthOwned producer_pre_depth{};
    PpUpstreamSampleTeePredecessorWriterPreDepthOwned predecessor_writer_pre_depth{};
    bool ordinal_content_binding_exact{};
    bool ordinal_tails_owned_before_outer_release{};
    bool observe_owned{};
};

enum class PpUpstreamFeedbackCubeMetadataDisposition : u8 {
    Idle = 0,
    ReadbackUnavailable = 1,
    WriterAuthorityLoss = 2,
    MetadataInvalid = 3,
    Exact = 4,
    InvalidTransition = 5,
};

struct PpUpstreamFeedbackCubeCompactLoss {
    u32 invalidation{};
    u32 gap{};

    [[nodiscard]] constexpr bool Any() const noexcept {
        return invalidation != 0 || gap != 0;
    }

    [[nodiscard]] friend constexpr bool operator==(
        const PpUpstreamFeedbackCubeCompactLoss& lhs,
        const FinalGuestSurfaceLoss& rhs) noexcept {
        return lhs.invalidation == rhs.invalidation && lhs.gap == rhs.gap &&
               rhs.unsupported_type == 0 && rhs.unsupported_samples == 0 &&
               rhs.unsupported_mip == 0 && rhs.unsupported_layer == 0 &&
               rhs.unsupported_aspect == 0 && rhs.unsupported_format == 0 &&
               rhs.invalid_extent == 0 && rhs.logical_mapping == 0 &&
               rhs.tile_capacity == 0 && rhs.byte_capacity == 0 &&
               rhs.ordinal_capacity == 0 && rhs.busy == 0 && rhs.history == 0 &&
               rhs.tile_detail == 0;
    }

    [[nodiscard]] friend constexpr bool operator==(
        const FinalGuestSurfaceLoss& lhs,
        const PpUpstreamFeedbackCubeCompactLoss& rhs) noexcept {
        return rhs == lhs;
    }

    bool operator==(const PpUpstreamFeedbackCubeCompactLoss&) const = default;
};

struct PpUpstreamFeedbackCubeMetadataReduction {
    static constexpr u32 MaxMetadataCount =
        Shader::PpUpstreamSampleTeeMaxDimension * Shader::PpUpstreamSampleTeeMaxDimension;

    u32 selected_candidate_index{};
    u32 image_resource_index{};
    u32 metadata_count{};
    u32 non_cube_count{};
    u32 identity_count{};
    u32 stride_compressed_count{};
    u32 reserved_count{};
    u32 invalid_count{};
    u32 missing_writer_count{};
    u32 writer_conflict_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    PpUpstreamFeedbackCubeCompactLoss loss{};
    PpUpstreamFeedbackCubeMetadataDisposition disposition{
        PpUpstreamFeedbackCubeMetadataDisposition::Idle};
    bool overflowed{};

    bool operator==(const PpUpstreamFeedbackCubeMetadataReduction&) const = default;
};

static_assert(std::is_trivially_copyable_v<PpUpstreamFeedbackCubeMetadataReduction>);
static_assert(sizeof(PpUpstreamFeedbackCubeMetadataReduction) <= 96);

[[nodiscard]] inline PpUpstreamFeedbackCubeMetadataReduction
ReducePpUpstreamFeedbackCubeOwnedReadback(
    const PpUpstreamSampleTeeDeferredObservation& observation,
    u32 selected_candidate_index) noexcept {
    using Disposition = PpUpstreamFeedbackCubeMetadataDisposition;
    PpUpstreamFeedbackCubeMetadataReduction result{
        .selected_candidate_index = selected_candidate_index,
    };
    const auto unavailable = [&]() {
        result.status = FinalGuestSurfaceStatus::InvalidationLoss;
        result.loss.invalidation = 1;
        result.disposition = Disposition::ReadbackUnavailable;
        return result;
    };
    if (observation.owned.records.size() > result.MaxMetadataCount) {
        result.overflowed = true;
        result.status = FinalGuestSurfaceStatus::InvalidationLoss;
        result.loss.invalidation = 1;
        result.disposition = Disposition::InvalidTransition;
        return result;
    }

    const auto& record = observation.record;
    const auto& lifecycle = observation.lifecycle;
    const auto& publication = observation.publication;
    const auto& owned = observation.owned;
    const PpUpstreamSampleTeeReadbackHeader expected_header{
        .magic = Shader::PpUpstreamSampleTeeHeaderMagic,
        .generation = owned.header.generation,
        .candidate_index = selected_candidate_index,
        .x = record.window.x,
        .y = record.window.y,
        .width = record.window.width,
        .height = record.window.height,
        .pixel_count = record.window.pixel_count,
    };
    if (!observation.observe_owned || !publication.publish ||
        publication.status != FinalGuestSurfaceStatus::Complete || publication.loss.Any() ||
        publication.shader != record.shader || publication.window != record.window ||
        !lifecycle.finalized || lifecycle.lifecycle_invalid ||
        lifecycle.status != FinalGuestSurfaceStatus::Complete || lifecycle.loss.Any() ||
        !lifecycle.expects_owned_copy || lifecycle.owned_copy_observations != 1 ||
        lifecycle.expected_outer_slot_release_attempts != 1 ||
        lifecycle.observed_outer_slot_release_attempts != 1 ||
        !lifecycle.mapped_invalidation_succeeded ||
        !lifecycle.bounded_record_copy_succeeded ||
        !lifecycle.bounded_copy_completed_before_outer_slot_release ||
        !lifecycle.slot_release_succeeded || owned.header.generation == 0 ||
        lifecycle.expected_record_count != record.window.pixel_count ||
        !IsCanonicalPpUpstreamSampleTeeCandidateRecord(record, selected_candidate_index) ||
        owned.status != FinalGuestSurfaceStatus::Complete || owned.loss.Any() ||
        !owned.mapped_invalidation_succeeded || !owned.bounded_record_copy_succeeded ||
        owned.retains_mapped_span || owned.header != expected_header ||
        lifecycle.expected_header != expected_header ||
        owned.records.size() != record.window.pixel_count) {
        return unavailable();
    }
    result.image_resource_index = record.shader.resource_index;

    for (const auto& pixel : owned.records) {
        result.missing_writer_count += pixel.writers == 0;
        result.writer_conflict_count += pixel.writers > 1;
    }
    if (result.missing_writer_count != 0 || result.writer_conflict_count != 0) {
        result.disposition = Disposition::WriterAuthorityLoss;
        if (result.writer_conflict_count != 0) {
            result.status = FinalGuestSurfaceStatus::InvalidationLoss;
            result.loss.invalidation = 1;
        } else {
            result.status = FinalGuestSurfaceStatus::GapLoss;
            result.loss.gap = 1;
        }
        return result;
    }

    for (const auto& pixel : owned.records) {
        const auto decoded =
            Shader::DecodePpUpstreamSampleTeeCubeMetadata(pixel.cube_metadata);
        if (!decoded.observed || decoded.malformed) {
            result.metadata_count = 0;
            result.non_cube_count = 0;
            result.identity_count = 0;
            result.stride_compressed_count = 0;
            result.reserved_count = 0;
            result.invalid_count = 0;
            result.status = FinalGuestSurfaceStatus::InvalidationLoss;
            result.loss.invalidation = 1;
            result.disposition = Disposition::MetadataInvalid;
            return result;
        }
        ++result.metadata_count;
        switch (decoded.classification) {
        case Shader::PpUpstreamSampleTeeCubeFaceClass::NonCube:
            ++result.non_cube_count;
            break;
        case Shader::PpUpstreamSampleTeeCubeFaceClass::Identity:
            ++result.identity_count;
            break;
        case Shader::PpUpstreamSampleTeeCubeFaceClass::StrideCompressed:
            ++result.stride_compressed_count;
            break;
        case Shader::PpUpstreamSampleTeeCubeFaceClass::Reserved:
            ++result.reserved_count;
            break;
        case Shader::PpUpstreamSampleTeeCubeFaceClass::Invalid:
            ++result.invalid_count;
            break;
        default:
            result.metadata_count = 0;
            result.non_cube_count = 0;
            result.identity_count = 0;
            result.stride_compressed_count = 0;
            result.reserved_count = 0;
            result.invalid_count = 0;
            result.status = FinalGuestSurfaceStatus::InvalidationLoss;
            result.loss.invalidation = 1;
            result.disposition = Disposition::MetadataInvalid;
            return result;
        }
    }
    result.status = FinalGuestSurfaceStatus::Complete;
    result.disposition = Disposition::Exact;
    return result;
}

enum class PpUpstreamFeedbackFragmentBufferDependencyDisposition : u8 {
    Idle = 0,
    ReadbackUnavailable = 1,
    WriterAuthorityLoss = 2,
    MetadataInvalid = 3,
    Varying = 4,
    Exact = 5,
};

struct PpUpstreamFeedbackFragmentBufferDependencyReduction {
    u32 metadata_count{};
    u32 fingerprint{};
    u32 missing_writer_count{};
    u32 writer_conflict_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    PpUpstreamFeedbackFragmentBufferDependencyDisposition disposition{
        PpUpstreamFeedbackFragmentBufferDependencyDisposition::Idle};
    bool uniform{};
    u32 additional_image_operation_count{};
    u32 additional_gpu_copy_count{};
    u32 additional_command_count{};
    u32 additional_barrier_count{};
    u32 additional_slot_count{};
    u32 additional_wait_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    bool retains_private_state{};
    bool affects_capture_behavior{};
    bool affects_renderer_behavior{};

    bool operator==(const PpUpstreamFeedbackFragmentBufferDependencyReduction&) const = default;
};

static_assert(std::is_trivially_copyable_v<
              PpUpstreamFeedbackFragmentBufferDependencyReduction>);
static_assert(sizeof(PpUpstreamFeedbackFragmentBufferDependencyReduction) <= 128);

[[nodiscard]] inline PpUpstreamFeedbackFragmentBufferDependencyReduction
ReducePpUpstreamFeedbackFragmentBufferDependencies(
    const PpUpstreamSampleTeeDeferredObservation& observation,
    u32 selected_candidate_index) noexcept {
    using Disposition = PpUpstreamFeedbackFragmentBufferDependencyDisposition;
    PpUpstreamFeedbackFragmentBufferDependencyReduction result{};
    const auto unavailable = [&]() {
        result.status = FinalGuestSurfaceStatus::InvalidationLoss;
        result.loss.invalidation = 1;
        result.disposition = Disposition::ReadbackUnavailable;
        return result;
    };
    if (observation.owned.records.size() >
        Shader::PpUpstreamSampleTeeMaxDimension *
            Shader::PpUpstreamSampleTeeMaxDimension) {
        return unavailable();
    }

    const auto& record = observation.record;
    const auto& lifecycle = observation.lifecycle;
    const auto& publication = observation.publication;
    const auto& owned = observation.owned;
    const PpUpstreamSampleTeeReadbackHeader expected_header{
        .magic = Shader::PpUpstreamSampleTeeHeaderMagic,
        .generation = owned.header.generation,
        .candidate_index = selected_candidate_index,
        .x = record.window.x,
        .y = record.window.y,
        .width = record.window.width,
        .height = record.window.height,
        .pixel_count = record.window.pixel_count,
    };
    if (!observation.observe_owned || !publication.publish ||
        publication.status != FinalGuestSurfaceStatus::Complete || publication.loss.Any() ||
        publication.shader != record.shader || publication.window != record.window ||
        !lifecycle.finalized || lifecycle.lifecycle_invalid ||
        lifecycle.status != FinalGuestSurfaceStatus::Complete || lifecycle.loss.Any() ||
        !lifecycle.expects_owned_copy || lifecycle.owned_copy_observations != 1 ||
        lifecycle.expected_outer_slot_release_attempts != 1 ||
        lifecycle.observed_outer_slot_release_attempts != 1 ||
        !lifecycle.mapped_invalidation_succeeded ||
        !lifecycle.bounded_record_copy_succeeded ||
        !lifecycle.bounded_copy_completed_before_outer_slot_release ||
        !lifecycle.slot_release_succeeded || owned.header.generation == 0 ||
        lifecycle.expected_record_count != record.window.pixel_count ||
        !IsCanonicalPpUpstreamSampleTeeCandidateRecord(record, selected_candidate_index) ||
        owned.status != FinalGuestSurfaceStatus::Complete || owned.loss.Any() ||
        !owned.mapped_invalidation_succeeded || !owned.bounded_record_copy_succeeded ||
        owned.retains_mapped_span || owned.header != expected_header ||
        lifecycle.expected_header != expected_header ||
        owned.records.size() != record.window.pixel_count) {
        return unavailable();
    }

    for (const auto& pixel : owned.records) {
        result.missing_writer_count += pixel.writers == 0;
        result.writer_conflict_count += pixel.writers > 1;
    }
    if (result.missing_writer_count != 0 || result.writer_conflict_count != 0) {
        result.disposition = Disposition::WriterAuthorityLoss;
        if (result.writer_conflict_count != 0) {
            result.status = FinalGuestSurfaceStatus::InvalidationLoss;
            result.loss.invalidation = 1;
        } else {
            result.status = FinalGuestSurfaceStatus::GapLoss;
            result.loss.gap = 1;
        }
        return result;
    }

    u32 fingerprint{};
    bool first = true;
    result.uniform = true;
    for (const auto& pixel : owned.records) {
        if ((pixel.dependency_metadata & 0x80000000u) == 0) {
            result.metadata_count = 0;
            result.fingerprint = 0;
            result.uniform = false;
            result.status = FinalGuestSurfaceStatus::InvalidationLoss;
            result.loss.invalidation = 1;
            result.disposition = Disposition::MetadataInvalid;
            return result;
        }
        if (first) {
            fingerprint = pixel.dependency_metadata;
            first = false;
        } else if (fingerprint != pixel.dependency_metadata) {
            result.uniform = false;
        }
        ++result.metadata_count;
    }
    if (first) {
        return unavailable();
    }
    result.status = FinalGuestSurfaceStatus::Complete;
    if (result.uniform || result.metadata_count == 1) {
        result.uniform = true;
        result.fingerprint = fingerprint;
        result.disposition = Disposition::Exact;
    } else {
        result.fingerprint = 0;
        result.disposition = Disposition::Varying;
    }
    return result;
}

[[nodiscard]] inline std::string FormatPpUpstreamFeedbackFragmentBufferDependencies(
    u64 sequence, const PpUpstreamFeedbackFragmentBufferDependencyReduction& reduction) {
    char hex[8]{};
    const auto [end, error] = std::to_chars(std::begin(hex), std::end(hex),
                                            reduction.fingerprint, 16);
    const std::string fingerprint = error == std::errc{} ? std::string{hex, end} : "0";
    return "FGSCTUB s=" + std::to_string(sequence) +
           " n=" + std::to_string(reduction.metadata_count) +
           " u=" + std::to_string(reduction.uniform) + " h=0x" + fingerprint +
           " st=" + std::to_string(static_cast<u32>(reduction.status)) +
           " lm=" + std::to_string(FinalGuestSurfaceLossMask(reduction.loss, 0));
}

enum class PpUpstreamFeedbackCubeQueryDisposition : u8 {
    Idle = 0,
    SourceAuthorityLoss = 1,
    ReadbackUnavailable = 2,
    StaticShapeMismatch = 3,
    Exact = 4,
};

struct PpUpstreamFeedbackCubeQueryReport {
    u64 sequence{};
    Shader::PpUpstreamSampleTeeCubeShape shape{
        Shader::PpUpstreamSampleTeeCubeShape::NonCube};
    u32 cube_count{};
    u32 base_layer{};
    u32 view_layer_count{};
    u32 metadata_count{};
    u32 non_cube_count{};
    u32 identity_count{};
    u32 stride_compressed_count{};
    u32 reserved_count{};
    u32 invalid_count{};
    PpUpstreamFeedbackCubeQueryDisposition disposition{
        PpUpstreamFeedbackCubeQueryDisposition::Idle};
    u32 additional_image_operation_count{};
    u32 additional_gpu_copy_count{};
    u32 additional_command_count{};
    u32 additional_barrier_count{};
    u32 additional_slot_count{};
    u32 additional_wait_count{};
    u32 additional_invalidation_count{};
    u32 additional_release_count{};
    u32 additional_ownership_transfer_count{};
    bool full_cube_range{};
    bool retains_private_state{};
    bool retains_raw_face{};
    bool retains_binding_key{};
    bool affects_capture_behavior{};
    bool affects_registry_behavior{};
    bool affects_renderer_behavior{};

    bool operator==(const PpUpstreamFeedbackCubeQueryReport&) const = default;
};

static_assert(std::is_trivially_copyable_v<PpUpstreamFeedbackCubeQueryReport>);
static_assert(sizeof(PpUpstreamFeedbackCubeQueryReport) <= 160);

[[nodiscard]] constexpr PpUpstreamFeedbackCubeQueryReport
MakePpUpstreamFeedbackCubeQueryReport(
    u64 sequence, const PpUpstreamSampleTeeSourceObservation& source,
    const PpUpstreamFeedbackCubeMetadataReduction& reduction) noexcept {
    using Disposition = PpUpstreamFeedbackCubeQueryDisposition;
    if (sequence == 0) {
        return {};
    }
    PpUpstreamFeedbackCubeQueryReport result{
        .sequence = sequence,
        .shape = source.cube_binding.fact.shape,
        .cube_count = source.cube_binding.fact.cube_count,
        .base_layer = source.cube_binding.fact.base_layer,
        .view_layer_count = source.cube_binding.fact.view_layer_count,
        .metadata_count = reduction.metadata_count,
        .non_cube_count = reduction.non_cube_count,
        .identity_count = reduction.identity_count,
        .stride_compressed_count = reduction.stride_compressed_count,
        .reserved_count = reduction.reserved_count,
        .invalid_count = reduction.invalid_count,
        .full_cube_range = source.cube_binding.fact.full_cube_range,
    };
    const bool source_exact =
        source.valid && source.route.valid && source.binding.valid &&
        source.binding.link.Valid() &&
        source.cube_binding.fact.disposition ==
            PpUpstreamFeedbackCubeBindingDisposition::Exact &&
        !source.cube_binding.overflowed && source.cube_binding.observation_count != 0 &&
        source.route.image_resource_index == reduction.image_resource_index &&
        source.cube_binding.selected_candidate_index == reduction.selected_candidate_index;
    if (!source_exact) {
        result.disposition = Disposition::SourceAuthorityLoss;
        return result;
    }
    if (reduction.disposition != PpUpstreamFeedbackCubeMetadataDisposition::Exact ||
        reduction.status != FinalGuestSurfaceStatus::Complete || reduction.loss.Any() ||
        reduction.overflowed) {
        result.disposition = Disposition::ReadbackUnavailable;
        return result;
    }
    const bool metadata_complete =
        reduction.metadata_count == reduction.non_cube_count + reduction.identity_count +
                                        reduction.stride_compressed_count +
                                        reduction.reserved_count + reduction.invalid_count;
    bool shape_matches{};
    switch (source.cube_binding.fact.shape) {
    case Shader::PpUpstreamSampleTeeCubeShape::NonCube:
        shape_matches = reduction.metadata_count == reduction.non_cube_count;
        break;
    case Shader::PpUpstreamSampleTeeCubeShape::SingleCube:
        shape_matches = reduction.non_cube_count == 0 &&
                        reduction.stride_compressed_count == 0 &&
                        reduction.reserved_count == 0 && reduction.invalid_count == 0;
        break;
    case Shader::PpUpstreamSampleTeeCubeShape::CubeArray:
        shape_matches = reduction.non_cube_count == 0;
        break;
    default:
        shape_matches = false;
        break;
    }
    result.disposition = metadata_complete && shape_matches
                             ? Disposition::Exact
                             : Disposition::StaticShapeMismatch;
    return result;
}

[[nodiscard]] inline std::string FormatPpUpstreamFeedbackCubeQueryReport(
    const PpUpstreamFeedbackCubeQueryReport& report) {
    return "FGSCTUQ s=" + std::to_string(report.sequence) +
           " sh=" + std::to_string(static_cast<u32>(report.shape)) +
           " cc=" + std::to_string(report.cube_count) +
           " bl=" + std::to_string(report.base_layer) +
           " vl=" + std::to_string(report.view_layer_count) +
           " fr=" + std::to_string(report.full_cube_range) +
           " mc=" + std::to_string(report.metadata_count) +
           " nc=" + std::to_string(report.non_cube_count) +
           " ic=" + std::to_string(report.identity_count) +
           " sc=" + std::to_string(report.stride_compressed_count) +
           " rc=" + std::to_string(report.reserved_count) +
           " vc=" + std::to_string(report.invalid_count) +
           " d=" + std::to_string(static_cast<u32>(report.disposition));
}

[[nodiscard]] inline PpUpstreamSampleTeeDeferredObservation
FinalizePpUpstreamSampleTeeDeferredObservation(
    const PpUpstreamSampleTeeDeferredReadbackDescriptor& attempt,
    PpUpstreamSampleTeeDeferredLifecycleState lifecycle, PpUpstreamSampleTeeOwnedReadback owned,
    PpUpstreamSampleTeeTransferredDepthSourceOwned transferred_depth_source = {},
    PpUpstreamSampleTeeSelectedDepthOwned selected_depth = {},
    PpUpstreamSampleTeeProducerPreDepthOwned producer_pre_depth = {}) {
    const auto clear_owned = [](PpUpstreamSampleTeeOwnedReadback& value,
                                FinalGuestSurfaceStatus status, const FinalGuestSurfaceLoss& loss) {
        std::vector<PpUpstreamSampleTeePixelRecord>{}.swap(value.records);
        value.header = {};
        value.status = status;
        value.loss = loss;
        value.mapped_invalidation_succeeded = false;
        value.bounded_record_copy_succeeded = false;
        value.retains_mapped_span = false;
    };
    if (!attempt.enabled) {
        clear_owned(owned, FinalGuestSurfaceStatus::AlreadyConsumed, {});
        return {.owned = std::move(owned),
                .transferred_depth_source = std::move(transferred_depth_source),
                .selected_depth = std::move(selected_depth),
                .producer_pre_depth = std::move(producer_pre_depth)};
    }

    const auto plan = PlanPpUpstreamSampleTeeDeferredReadback(attempt);
    constexpr size_t MaxRecords = static_cast<size_t>(Shader::PpUpstreamSampleTeeMaxDimension) *
                                  Shader::PpUpstreamSampleTeeMaxDimension;
    const FinalGuestSurfaceLoss invalidation_loss{.invalidation = 1};
    const bool capture_outcome_is_coherent =
        (attempt.capture_status == FinalGuestSurfaceStatus::Complete &&
         !attempt.capture_loss.Any() && attempt.capture_has_copy) ||
        (attempt.capture_status == FinalGuestSurfaceStatus::AlreadyConsumed &&
         !attempt.capture_loss.Any() && !attempt.capture_has_copy) ||
        (attempt.capture_status != FinalGuestSurfaceStatus::Complete &&
         attempt.capture_status != FinalGuestSurfaceStatus::AlreadyConsumed &&
         attempt.capture_loss.Any() && !attempt.capture_has_copy);
    const bool detached_state_is_empty =
        attempt.record == PpUpstreamSampleTeeCandidateRecord{} &&
        attempt.capture == PpUpstreamSampleTeeCapturePlan{} &&
        attempt.commands == PpUpstreamSampleTeeCommandPlan{} &&
        attempt.controller == PpUpstreamSampleTeeControllerState{} &&
        attempt.selected_candidate_index == 0 && attempt.mapped_bytes == 0;
    const bool empty_owned = owned.header == PpUpstreamSampleTeeReadbackHeader{} &&
                             owned.records.empty() && owned.records.capacity() == 0 &&
                             owned.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                             !owned.loss.Any() && !owned.mapped_invalidation_succeeded &&
                             !owned.bounded_record_copy_succeeded && !owned.retains_mapped_span;
    const bool complete_owned =
        owned.status == FinalGuestSurfaceStatus::Complete && !owned.loss.Any() &&
        owned.mapped_invalidation_succeeded && owned.bounded_record_copy_succeeded &&
        !owned.retains_mapped_span && owned.header == plan.expected_header &&
        owned.records.size() == plan.record_count && owned.records.capacity() <= MaxRecords;
    const bool invalidated_owned = owned.header == PpUpstreamSampleTeeReadbackHeader{} &&
                                   owned.records.empty() && owned.records.capacity() == 0 &&
                                   owned.status == FinalGuestSurfaceStatus::InvalidationLoss &&
                                   owned.loss == invalidation_loss &&
                                   !owned.bounded_record_copy_succeeded &&
                                   !owned.retains_mapped_span;
    const bool owned_matches_plan = plan.copy ? complete_owned || invalidated_owned : empty_owned;

    auto expected_lifecycle = BeginPpUpstreamSampleTeeDeferredLifecycle(plan);
    if (plan.copy && owned_matches_plan) {
        expected_lifecycle = ObservePpUpstreamSampleTeeOwnedCopy(expected_lifecycle, owned);
    }
    if (plan.outer_slot_release_attempts == 1) {
        expected_lifecycle = ObservePpUpstreamSampleTeeOuterSlotRelease(expected_lifecycle, true);
    }
    expected_lifecycle = FinalizePpUpstreamSampleTeeDeferredLifecycle(expected_lifecycle);

    bool detached_state_is_canonical = capture_outcome_is_coherent;
    if (plan.status != FinalGuestSurfaceStatus::Complete && plan.loss.Any()) {
        if (attempt.outer_slot) {
            auto canonical_attempt = attempt;
            canonical_attempt.capture_status = FinalGuestSurfaceStatus::Complete;
            canonical_attempt.capture_loss = {};
            canonical_attempt.capture_has_copy = true;
            canonical_attempt.record.status = FinalGuestSurfaceStatus::Complete;
            canonical_attempt.record.loss = {};
            canonical_attempt.record.shader.status = FinalGuestSurfaceStatus::Complete;
            canonical_attempt.record.shader.loss = {};
            canonical_attempt.record.window.status = FinalGuestSurfaceStatus::Complete;
            canonical_attempt.record.window.loss = {};
            canonical_attempt.capture.status = FinalGuestSurfaceStatus::Complete;
            canonical_attempt.capture.loss = {};
            canonical_attempt.commands.status = FinalGuestSurfaceStatus::Complete;
            canonical_attempt.commands.loss = {};
            canonical_attempt.controller.status = FinalGuestSurfaceStatus::Complete;
            canonical_attempt.controller.loss = {};
            const auto canonical_plan = PlanPpUpstreamSampleTeeDeferredReadback(canonical_attempt);
            detached_state_is_canonical &=
                canonical_plan.status == FinalGuestSurfaceStatus::Complete &&
                !canonical_plan.loss.Any() && canonical_plan.copy;
        } else {
            detached_state_is_canonical &=
                attempt.capture_status != FinalGuestSurfaceStatus::Complete &&
                attempt.capture_status != FinalGuestSurfaceStatus::AlreadyConsumed &&
                attempt.capture_loss.Any() && !attempt.capture_has_copy && detached_state_is_empty;
        }
    }
    if (!owned_matches_plan) {
        lifecycle = InvalidatePpUpstreamSampleTeeDeferredLifecycle(expected_lifecycle);
    } else if (lifecycle != expected_lifecycle || !detached_state_is_canonical) {
        lifecycle = InvalidatePpUpstreamSampleTeeDeferredLifecycle(lifecycle);
        if (!lifecycle.finalized) {
            lifecycle = FinalizePpUpstreamSampleTeeDeferredLifecycle(lifecycle);
        }
    }
    const auto& controller = attempt.controller;
    auto publication = ReconcilePpUpstreamSampleTeeRuntime({
        .enabled = true,
        .record = attempt.record,
        .selected_candidate_index = attempt.selected_candidate_index,
        .capture_status = lifecycle.status,
        .capture_loss = lifecycle.loss,
        .capture_has_copy = lifecycle.expects_owned_copy,
        .initialized = controller.initialized,
        .armed = controller.armed,
        .draw_executed = controller.draw_executed,
        .post_barrier_complete = controller.post_barrier_complete,
        .slot_present = static_cast<bool>(attempt.outer_slot),
        .mapped_invalidation_succeeded = lifecycle.mapped_invalidation_succeeded,
        .bounded_record_copy_succeeded = lifecycle.bounded_record_copy_succeeded,
        .rendering_serial = controller.serial,
        .expected_generation = attempt.outer_slot.generation,
        .header = owned.header,
        .observed_outer_slot_release_attempts = lifecycle.observed_outer_slot_release_attempts,
        .bounded_copy_completed_before_outer_slot_release =
            lifecycle.bounded_copy_completed_before_outer_slot_release,
        .slot_release_succeeded = lifecycle.slot_release_succeeded,
    });
    if (publication.publish && lifecycle.finalized && !lifecycle.lifecycle_invalid &&
        complete_owned) {
        return {
            .lifecycle = lifecycle,
            .publication = publication,
            .record = attempt.record,
            .owned = std::move(owned),
            .transferred_depth_source = std::move(transferred_depth_source),
            .selected_depth = std::move(selected_depth),
            .producer_pre_depth = std::move(producer_pre_depth),
            .observe_owned = true,
        };
    }
    if (publication.status == FinalGuestSurfaceStatus::Complete || !publication.loss.Any()) {
        publication.status = FinalGuestSurfaceStatus::InvalidationLoss;
        publication.loss = {.invalidation = 1};
    }
    clear_owned(owned, publication.status, publication.loss);
    return {
        .lifecycle = lifecycle,
        .publication = publication,
        .owned = std::move(owned),
        .transferred_depth_source = std::move(transferred_depth_source),
        .selected_depth = std::move(selected_depth),
        .producer_pre_depth = std::move(producer_pre_depth),
        .observe_owned = true,
    };
}

struct PpUpstreamSampleTeeDeferredConsumeResult {
    PpUpstreamInputPublication base{};
    PpUpstreamSampleTeeDeferredObservation tee{};
    bool release_attempted{};
    bool release_succeeded{};
};

template <typename Invalidate, typename Release>
[[nodiscard]] PpUpstreamSampleTeeDeferredConsumeResult
ConsumePpUpstreamSampleTeeDeferredReadback(
    const PpUpstreamSampleTeeDeferredReadbackDescriptor& attempt,
    PpUpstreamInputPublication base, std::span<const std::byte> mapped,
    Invalidate&& invalidate, Release&& release) {
    const auto plan = PlanPpUpstreamSampleTeeDeferredReadback(attempt);
    auto lifecycle = BeginPpUpstreamSampleTeeDeferredLifecycle(plan);
    PpUpstreamSampleTeeOwnedReadback owned{};
    PpUpstreamSampleTeeTransferredDepthSourceOwned transferred_depth_source{};
    PpUpstreamSampleTeeSelectedDepthOwned selected_depth{};
    bool mapped_invalidated{};
    if (plan.copy) {
        mapped_invalidated =
            std::forward<Invalidate>(invalidate)(plan.invalidate_offset, plan.invalidate_bytes);
        owned = CopyPpUpstreamSampleTeeDeferredReadback(plan, mapped_invalidated, mapped);
        lifecycle = ObservePpUpstreamSampleTeeOwnedCopy(lifecycle, owned);
    }
    const u64 slot_offset = static_cast<u64>(attempt.outer_slot.slot) *
                            Shader::PpUpstreamSampleTeeSlotStrideBytes;
    const bool slot_accessible =
        mapped_invalidated && attempt.outer_slot && slot_offset <= mapped.size() &&
        Shader::PpUpstreamSampleTeeSlotStrideBytes <= mapped.size() - slot_offset;
    const auto shared_slot = slot_accessible
                                 ? mapped.subspan(slot_offset,
                                                  Shader::PpUpstreamSampleTeeSlotStrideBytes)
                                 : std::span<const std::byte>{};
    const auto overlaps = [](u64 first_offset, u64 first_bytes, u64 second_offset,
                             u64 second_bytes) {
        return first_offset < second_offset + second_bytes &&
               second_offset < first_offset + first_bytes;
    };
    const u64 tee_absolute_end =
        std::max(plan.header_offset + plan.header_bytes, plan.records_offset + plan.records_bytes);
    const bool tee_range_valid =
        plan.copy && plan.header_offset >= slot_offset && plan.records_offset >= slot_offset &&
        tee_absolute_end >= slot_offset &&
        tee_absolute_end - slot_offset <= Shader::PpUpstreamSampleTeeSlotStrideBytes;
    const u64 tee_occupied_bytes = tee_range_valid ? tee_absolute_end - slot_offset : 0;
    const auto& depth_plan = attempt.transferred_depth_source;
    const bool transfer_overlaps_tee =
        depth_plan.status == FinalGuestSurfaceStatus::Complete && depth_plan.copy &&
        (!tee_range_valid || overlaps(0, tee_occupied_bytes, depth_plan.plane.plane_offset,
                                     depth_plan.plane.plane_bytes));
    if (depth_plan.status == FinalGuestSurfaceStatus::Complete && !depth_plan.loss.Any() &&
        depth_plan.copy) {
        if (slot_accessible && !transfer_overlaps_tee) {
            transferred_depth_source =
                CopyPpUpstreamSampleTeeTransferredDepthSource(depth_plan, true, shared_slot);
        } else {
            transferred_depth_source.status = FinalGuestSurfaceStatus::InvalidationLoss;
            transferred_depth_source.loss = {.invalidation = 1};
            transferred_depth_source.shared_release_deferred = true;
        }
    } else if (depth_plan.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
               depth_plan.loss.Any()) {
        transferred_depth_source.status = depth_plan.status;
        transferred_depth_source.loss = depth_plan.loss;
        transferred_depth_source.shared_release_deferred = true;
    }
    const auto& selected_plan = attempt.selected_depth;
    if (selected_plan.status == FinalGuestSurfaceStatus::Complete && !selected_plan.loss.Any() &&
        selected_plan.copy) {
        const bool transfer_present =
            depth_plan.status == FinalGuestSurfaceStatus::Complete && !depth_plan.loss.Any() &&
            depth_plan.copy && !transfer_overlaps_tee;
        const bool overlap = !tee_range_valid ||
                             overlaps(0, tee_occupied_bytes,
                                      selected_plan.plane.plane_offset,
                                      selected_plan.plane.plane_bytes) ||
                             (transfer_present &&
            overlaps(depth_plan.plane.plane_offset, depth_plan.plane.plane_bytes,
                     selected_plan.plane.plane_offset, selected_plan.plane.plane_bytes));
        if (overlap || !slot_accessible) {
            selected_depth.status = FinalGuestSurfaceStatus::InvalidationLoss;
            selected_depth.loss = {.invalidation = 1};
            selected_depth.shared_release_deferred = true;
        } else {
            selected_depth = CopyPpUpstreamSampleTeeSelectedDepth(selected_plan, true, shared_slot);
        }
    } else if (selected_plan.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
               selected_plan.loss.Any()) {
        selected_depth.status = selected_plan.status;
        selected_depth.loss = selected_plan.loss;
        selected_depth.shared_release_deferred = true;
    }
    const bool had_slot = bool{attempt.outer_slot};
    const bool released =
        had_slot && std::forward<Release>(release)(attempt.outer_slot);
    auto shared = ReconcilePpUpstreamSampleTeeSharedSlotRelease(
        had_slot, released, base.status, base.loss, base.copy, lifecycle);
    auto observation = FinalizePpUpstreamSampleTeeDeferredObservation(
        attempt, shared.tee, std::move(owned), std::move(transferred_depth_source),
        std::move(selected_depth));
    return {
        .base = shared.base,
        .tee = std::move(observation),
        .release_attempted = had_slot,
        .release_succeeded = released,
    };
}

template <typename DownloadInvalidate, typename TeeInvalidate, typename Release>
[[nodiscard]] PpUpstreamSampleTeeDeferredConsumeResult
ConsumePpUpstreamSampleTeeDeferredReadback(
    const PpUpstreamSampleTeeDeferredReadbackDescriptor& attempt,
    PpUpstreamInputPublication base, std::span<const std::byte> tee_mapped,
    std::span<const std::byte> download_mapped, u64 download_slot_stride,
    DownloadInvalidate&& download_invalidate, TeeInvalidate&& tee_invalidate,
    Release&& release) {
    const bool ordinal_requested =
        attempt.ordinal_content_binding.status != FinalGuestSurfaceStatus::AlreadyConsumed ||
        attempt.ordinal_content_binding.loss.Any() || attempt.ordinal_content_binding.exact ||
        attempt.ordinal_depth_download_tails.status !=
            FinalGuestSurfaceStatus::AlreadyConsumed ||
        attempt.ordinal_depth_download_tails.loss.Any();
    const bool ordinal_exact =
        ordinal_requested && IsExactPpUpstreamSampleTeeDeferredOrdinalContent(attempt);
    const auto& producer_plan = attempt.producer_pre_depth;
    const bool producer_recordable =
        IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(producer_plan) &&
        attempt.outer_slot && producer_plan.slot == attempt.outer_slot.slot &&
        producer_plan.slot_stride == download_slot_stride;
    const bool producer_has_loss =
        producer_plan.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
        producer_plan.status != FinalGuestSurfaceStatus::Complete && producer_plan.loss.Any();
    const auto& predecessor_plan = attempt.predecessor_writer_pre_depth;
    const bool predecessor_recordable =
        IsPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailRecordable(
            predecessor_plan) &&
        attempt.outer_slot && predecessor_plan.slot == attempt.outer_slot.slot &&
        predecessor_plan.slot_stride == download_slot_stride &&
        (!ordinal_requested || ordinal_exact);
    const bool predecessor_has_loss =
        predecessor_plan.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
        predecessor_plan.status != FinalGuestSurfaceStatus::Complete &&
        predecessor_plan.loss.Any();
    const bool need_download_invalidation =
        base.copy || producer_recordable || predecessor_recordable;
    bool download_invalidated{};
    const u64 download_slot_offset =
        static_cast<u64>(attempt.outer_slot.slot) * download_slot_stride;
    if (need_download_invalidation && attempt.outer_slot && download_slot_stride != 0 &&
        download_slot_offset <= download_mapped.size() &&
        download_slot_stride <= download_mapped.size() - download_slot_offset) {
        download_invalidated = std::forward<DownloadInvalidate>(download_invalidate)(
            download_slot_offset, download_slot_stride);
    }
    PpUpstreamSampleTeeProducerPreDepthOwned producer_owned{};
    PpUpstreamSampleTeePredecessorWriterPreDepthOwned predecessor_owned{};
    if (predecessor_recordable) {
        const auto download_slot = download_mapped.subspan(
            download_slot_offset, static_cast<size_t>(download_slot_stride));
        predecessor_owned = CopyPpUpstreamSampleTeePredecessorWriterPreDepth(
            predecessor_plan, download_invalidated, download_slot);
    } else if (predecessor_has_loss) {
        predecessor_owned.status = predecessor_plan.status;
        predecessor_owned.loss = predecessor_plan.loss;
        predecessor_owned.shared_release_deferred = true;
    } else if (predecessor_plan.status != FinalGuestSurfaceStatus::AlreadyConsumed ||
               predecessor_plan.copy || predecessor_plan.plane.region_count != 0) {
        predecessor_owned.status = FinalGuestSurfaceStatus::InvalidationLoss;
        predecessor_owned.loss = {.invalidation = 1};
        predecessor_owned.shared_release_deferred = true;
    }
    if (producer_recordable) {
        const auto download_slot = download_mapped.subspan(
            download_slot_offset, static_cast<size_t>(download_slot_stride));
        producer_owned = CopyPpUpstreamSampleTeeProducerPreDepth(
            producer_plan, download_invalidated, download_slot);
    } else if (producer_has_loss) {
        producer_owned.status = producer_plan.status;
        producer_owned.loss = producer_plan.loss;
        producer_owned.shared_release_deferred = true;
    } else if (producer_plan.status != FinalGuestSurfaceStatus::AlreadyConsumed ||
               producer_plan.copy || producer_plan.plane.region_count != 0) {
        producer_owned.status = FinalGuestSurfaceStatus::InvalidationLoss;
        producer_owned.loss = {.invalidation = 1};
        producer_owned.shared_release_deferred = true;
    }
    if (base.copy && !download_invalidated) {
        base.status = FinalGuestSurfaceStatus::InvalidationLoss;
        base.loss = {.invalidation = 1};
        base.copy = false;
    }
    auto result = ConsumePpUpstreamSampleTeeDeferredReadback(
        attempt, base, tee_mapped, std::forward<TeeInvalidate>(tee_invalidate),
        std::forward<Release>(release));
    result.tee.producer_pre_depth = std::move(producer_owned);
    result.tee.predecessor_writer_pre_depth = std::move(predecessor_owned);
    result.tee.ordinal_content_binding_exact =
        ordinal_exact &&
        result.tee.predecessor_writer_pre_depth.status ==
            FinalGuestSurfaceStatus::Complete &&
        result.tee.producer_pre_depth.status == FinalGuestSurfaceStatus::Complete;
    result.tee.ordinal_tails_owned_before_outer_release =
        result.tee.ordinal_content_binding_exact &&
        result.tee.predecessor_writer_pre_depth.bounded_copy_succeeded &&
        result.tee.producer_pre_depth.bounded_copy_succeeded;
    return result;
}

struct PpUpstreamSampleTeeReduction {
    u32 covered_pixels{};
    u32 aba_pixels{};
    u32 stable_pixels{};
    u32 ambiguous_pixels{};
    u32 writer_conflicts{};
    u32 missing_pixels{};
    u32 input_covered_pixels{};
    u32 input_aba_pixels{};
    u32 input_stable_pixels{};
    u32 input_ambiguous_pixels{};
    u32 coordinate_covered_pixels{};
    u32 coordinate_aba_pixels{};
    u32 coordinate_stable_pixels{};
    u32 coordinate_ambiguous_pixels{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool departed{};
    bool retains_values{};
};

[[nodiscard]] inline PpUpstreamSampleTeeReduction ReducePpUpstreamSampleTee(
    std::span<const PpUpstreamSampleTeePixelRecord> baseline,
    std::span<const PpUpstreamSampleTeePixelRecord> departure,
    std::span<const PpUpstreamSampleTeePixelRecord> returned) noexcept {
    PpUpstreamSampleTeeReduction result{};
    if (baseline.empty() || baseline.size() != departure.size() ||
        baseline.size() != returned.size()) {
        result.status = FinalGuestSurfaceStatus::InvalidationLoss;
        result.loss.invalidation = 1;
        return result;
    }
    constexpr size_t MaxRecords = static_cast<size_t>(Shader::PpUpstreamSampleTeeMaxDimension) *
                                  Shader::PpUpstreamSampleTeeMaxDimension;
    if (baseline.size() > MaxRecords) {
        result.status = FinalGuestSurfaceStatus::CapacityLoss;
        result.loss.byte_capacity = 1;
        return result;
    }
    for (size_t index = 0; index < baseline.size(); ++index) {
        const auto writers =
            std::array{baseline[index].writers, departure[index].writers, returned[index].writers};
        if (writers[0] == 0 || writers[1] == 0 || writers[2] == 0) {
            ++result.missing_pixels;
        }
        if (writers[0] > 1 || writers[1] > 1 || writers[2] > 1) {
            ++result.writer_conflicts;
        }
    }
    if (result.writer_conflicts != 0) {
        result.status = FinalGuestSurfaceStatus::InvalidationLoss;
        result.loss.invalidation = 1;
        return result;
    }
    if (result.missing_pixels != 0) {
        result.status = FinalGuestSurfaceStatus::GapLoss;
        result.loss.gap = 1;
        return result;
    }
    result.covered_pixels = static_cast<u32>(baseline.size());
    result.status = FinalGuestSurfaceStatus::Complete;
    bool input_metadata_valid = true;
    for (size_t index = 0; index < baseline.size(); ++index) {
        for (const u32 metadata : {baseline[index].dependency_metadata,
                                   departure[index].dependency_metadata,
                                   returned[index].dependency_metadata}) {
            input_metadata_valid &= (metadata & 0xc0000000u) == 0xc0000000u;
        }
    }
    for (size_t index = 0; index < baseline.size(); ++index) {
        const auto& first = baseline[index].value;
        const auto& middle = departure[index].value;
        const auto& last = returned[index].value;
        result.departed |= first != middle;
        if (first == middle && middle == last) {
            ++result.stable_pixels;
        } else if (first == last && first != middle) {
            ++result.aba_pixels;
        } else {
            ++result.ambiguous_pixels;
        }
        if (input_metadata_valid) {
            const u8 first_input = static_cast<u8>(baseline[index].dependency_metadata);
            const u8 middle_input = static_cast<u8>(departure[index].dependency_metadata);
            const u8 last_input = static_cast<u8>(returned[index].dependency_metadata);
            ++result.input_covered_pixels;
            if (first_input == middle_input && middle_input == last_input) {
                ++result.input_stable_pixels;
            } else if (first_input == last_input && first_input != middle_input) {
                ++result.input_aba_pixels;
            } else {
                ++result.input_ambiguous_pixels;
            }
            const u32 first_coordinate =
                (baseline[index].dependency_metadata >> 8) & 0x003fffffu;
            const u32 middle_coordinate =
                (departure[index].dependency_metadata >> 8) & 0x003fffffu;
            const u32 last_coordinate =
                (returned[index].dependency_metadata >> 8) & 0x003fffffu;
            ++result.coordinate_covered_pixels;
            if (first_coordinate == middle_coordinate && middle_coordinate == last_coordinate) {
                ++result.coordinate_stable_pixels;
            } else if (first_coordinate == last_coordinate &&
                       first_coordinate != middle_coordinate) {
                ++result.coordinate_aba_pixels;
            } else {
                ++result.coordinate_ambiguous_pixels;
            }
        }
    }
    return result;
}

enum class PpUpstreamSampleTeeProducerPreDepthClearValueRelation : u8 {
    Unavailable,
    Stable,
    Aba,
    Other,
};

enum class PpUpstreamSampleTeeProducerPreDepthWriterRelation : u8 {
    Unavailable,
    Stable,
    Aba,
    Other,
};

struct PpUpstreamSampleTeeCalibratedReport {
    u32 request_ordinal{};
    std::array<u64, 3> sequences{};
    PpUpstreamSampleTeeReduction reduction{};
    struct SourceReduction {
        bool available{};
        enum class TripletRelation : u8 {
            Unavailable,
            Stable,
            Aba,
            Other,
        } route{}, binding{};
        enum class EdgeRelation : u8 {
            Unavailable,
            Stable,
            Changed,
        } mutation_ab{}, mutation_bc{};
        std::array<VideoCore::ImageProducerClass, 3> producers{};
    } source{};
    struct TransferredDepthSourceReduction {
        u32 covered_regions{};
        u32 aba_regions{};
        u32 stable_regions{};
        u32 ambiguous_regions{};
        FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss loss{};
        bool available{};
        bool retains_values{};
    } transferred_depth_source{};
    struct SelectedDepthReduction {
        u32 covered_regions{};
        u32 aba_regions{};
        u32 stable_regions{};
        u32 ambiguous_regions{};
        FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss loss{};
        bool available{};
        bool retains_values{};
    } selected_depth{};
    struct ProducerPreDepthReduction {
        u32 covered_regions{};
        u32 aba_regions{};
        u32 stable_regions{};
        u32 ambiguous_regions{};
        FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss loss{};
        bool available{};
        bool retains_values{};
    } producer_pre_depth{};
    ProducerPreDepthReduction predecessor_writer_pre_depth{};
    struct PredecessorWriterPreProducerPreDepthReduction {
        u32 covered_regions{};
        u32 inherited_aba_regions{};
        u32 introduced_aba_regions{};
        u32 stable_regions{};
        u32 ambiguous_regions{};
        std::array<u32, 3> same_frame_equal_regions{};
        std::array<u32, 3> same_frame_unequal_regions{};
        FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss loss{};
        bool available{};
        u32 additional_gpu_copy_count{};
        u32 additional_slot_count{};
        u32 additional_release_count{};
        bool retains_values{};
        bool retains_private_endpoints{};
        bool retains_private_hints{};
        bool affects_capture_behavior{};
        bool affects_renderer_behavior{};
    } predecessor_writer_pre_producer_pre_depth{};
    struct ProducerPreDepthScopeEntryReduction {
        u32 covered_regions{};
        u32 clear_aba_regions{};
        u32 loaded_prior_writer_aba_regions{};
        u32 loaded_prior_content_aba_regions{};
        u32 stable_regions{};
        u32 ambiguous_regions{};
        u32 clear_materialization_match_regions{};
        u32 clear_materialization_mismatch_regions{};
        FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss loss{};
        PpUpstreamSampleTeeProducerPreDepthClearValueRelation clear_value_relation{
            PpUpstreamSampleTeeProducerPreDepthClearValueRelation::Unavailable};
        PpUpstreamSampleTeeProducerPreDepthWriterRelation predecessor_writer_relation{
            PpUpstreamSampleTeeProducerPreDepthWriterRelation::Unavailable};
        bool available{};
        bool retains_private_endpoints{};
        bool retains_clear_values{};
        u32 additional_gpu_copy_count{};
    } producer_pre_scope_entry{};
    struct ProducerPreSelectedDepthReduction {
        u32 covered_regions{};
        u32 inherited_aba_regions{};
        u32 introduced_aba_regions{};
        u32 stable_regions{};
        u32 ambiguous_regions{};
        std::array<u32, 3> same_frame_equal_regions{};
        std::array<u32, 3> same_frame_unequal_regions{};
        FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss loss{};
        bool available{};
        bool retains_values{};
    } producer_pre_selected_depth{};
};

using PpUpstreamSampleTeeTripletRelation =
    PpUpstreamSampleTeeCalibratedReport::SourceReduction::TripletRelation;
using PpUpstreamSampleTeeEdgeRelation =
    PpUpstreamSampleTeeCalibratedReport::SourceReduction::EdgeRelation;

struct PpUpstreamSampleTeeSourceDescriptor {
    bool enabled{};
    PpUpstreamSampleRouteSet routes{};
    u32 selected_input_index{};
    u32 selected_candidate_index{};
    PpUpstreamSampleTeeCandidateRecord candidate{};
    VideoCore::ImageColorScopeTrackedView binding{};
    u32 resolved_view_format{};
    u32 resolved_view_aspect{};
    VideoCore::ImageProducerClass producer{VideoCore::ImageProducerClass::Unknown};
    u32 sampled_physical_image_count{};
    std::span<const PpUpstreamFeedbackCubeLiveObservation> cube_observations{};
};

[[nodiscard]] constexpr PpUpstreamFeedbackCubeBindingSnapshot
SelectPpUpstreamFeedbackCubeBinding(
    std::span<const PpUpstreamFeedbackCubeLiveObservation> observations,
    u32 sampled_physical_image_count, VideoCore::ImageId selected_physical_image,
    const PpUpstreamSampleRoute& route, u32 selected_candidate_index) noexcept {
    using Disposition = PpUpstreamFeedbackCubeBindingDisposition;
    PpUpstreamFeedbackCubeBindingSnapshot result{
        .selected_candidate_index = selected_candidate_index,
    };
    if (observations.empty()) {
        return {};
    }
    if (sampled_physical_image_count != 1 || !selected_physical_image) {
        result.fact.disposition = Disposition::NoAuthority;
        return result;
    }

    const Shader::PpUpstreamSampleTeeCubeBindingFact* selected{};
    bool unauthorized_view{};
    bool conflicting_physical_image{};
    bool invalid_binding{};
    for (const auto& observation : observations) {
        if (observation.physical_image != selected_physical_image) {
            conflicting_physical_image = true;
            continue;
        }
        const auto& binding = observation.binding;
        if (static_cast<u32>(binding.logical_stage) != route.logical_stage ||
            binding.image_resource_index != route.image_resource_index ||
            binding.array_element != route.array_element) {
            continue;
        }
        if (result.observation_count == std::numeric_limits<u8>::max()) {
            result.overflowed = true;
            continue;
        }
        ++result.observation_count;
        unauthorized_view |=
            ClassifyPpUpstreamFeedbackSampledViewBindingFact(observation.view) !=
            PpUpstreamFeedbackSampledViewBindingDisposition::Exact;
        invalid_binding |= !binding.observed || !binding.valid;
        if (selected == nullptr) {
            selected = &binding;
        } else if (*selected != binding) {
            invalid_binding = true;
        }
    }

    if (result.overflowed) {
        result.fact.disposition = Disposition::InvalidTransition;
        return result;
    }
    if (selected == nullptr) {
        result.fact.disposition = Disposition::MissingBinding;
        return result;
    }
    if (unauthorized_view || conflicting_physical_image) {
        result.fact.disposition = Disposition::NoAuthority;
        return result;
    }
    if (invalid_binding) {
        result.fact.disposition = Disposition::InvalidTransition;
        return result;
    }
    result.fact = {
        .shape = selected->shape,
        .cube_count = selected->cube_count,
        .base_layer = selected->base_layer,
        .view_layer_count = selected->view_layer_count,
        .full_cube_range = selected->full_cube_range,
        .disposition = Disposition::Exact,
    };
    return result;
}

[[nodiscard]] constexpr PpUpstreamSampleTeeSourceObservation
PlanPpUpstreamSampleTeeSourceObservation(
    const PpUpstreamSampleTeeSourceDescriptor& descriptor) noexcept {
    if (!descriptor.enabled) {
        return {};
    }
    const auto& routes = descriptor.routes;
    if (routes.status != FinalGuestSurfaceStatus::Complete || routes.loss.Any() ||
        descriptor.selected_input_index >= PpTerminalScopeSampledInputs::MaxInputs ||
        routes.selected_input_index != descriptor.selected_input_index || routes.route_count != 1 ||
        routes.retains_image || routes.retains_view || routes.retains_sampler ||
        routes.logs_private_words ||
        !IsCanonicalPpUpstreamSampleTeeCandidateRecord(descriptor.candidate,
                                                       descriptor.selected_candidate_index)) {
        return {};
    }
    const auto& route = routes.routes[0];
    if (route.sampler_count > route.MaxSamplers) {
        return {};
    }
    for (u32 sampler_index = route.sampler_count; sampler_index < route.MaxSamplers;
         ++sampler_index) {
        if (route.samplers[sampler_index] != PpUpstreamSampleSamplerFingerprint{}) {
            return {};
        }
    }
    for (u32 route_index = 1; route_index < routes.MaxRoutes; ++route_index) {
        if (routes.routes[route_index] != PpUpstreamSampleRoute{}) {
            return {};
        }
    }
    PpUpstreamSampleTeeSourceObservation result{
        .route = route,
        .binding = descriptor.binding,
        .resolved_view_format = descriptor.resolved_view_format,
        .resolved_view_aspect = descriptor.resolved_view_aspect,
        .producer = descriptor.producer,
        .valid = true,
    };
    if (!IsCanonicalPpUpstreamSampleTeeSourceObservation(result, descriptor.candidate)) {
        return {};
    }
    result.cube_binding = SelectPpUpstreamFeedbackCubeBinding(
        descriptor.cube_observations, descriptor.sampled_physical_image_count,
        descriptor.binding.link.id, route, descriptor.selected_candidate_index);
    return result;
}

struct PpUpstreamSampleTeeCalibratedCoverage {
    u32 calibrations{};
    u32 outside{};
    u32 eligible{};
    u32 emitted{};
    u32 complete{};
    u32 loss{};
    bool ready{};
};

[[nodiscard]] inline std::string FormatPpUpstreamSampleTeeCalibratedReport(
    const PpUpstreamSampleTeeCalibratedReport& report) {
    const auto& reduction = report.reduction;
    return "FGSCTT q=" + std::to_string(report.request_ordinal) +
           " abc=" + std::to_string(report.sequences[0]) + "/" +
           std::to_string(report.sequences[1]) + "/" + std::to_string(report.sequences[2]) +
           " cov=" + std::to_string(reduction.covered_pixels) +
           " aba=" + std::to_string(reduction.aba_pixels) +
           " stable=" + std::to_string(reduction.stable_pixels) +
           " amb=" + std::to_string(reduction.ambiguous_pixels) +
           " writers=" + std::to_string(reduction.writer_conflicts) +
           " missing=" + std::to_string(reduction.missing_pixels) +
           " departed=" + std::to_string(reduction.departed) +
           " icov=" + std::to_string(reduction.input_covered_pixels) +
           " iaba=" + std::to_string(reduction.input_aba_pixels) +
           " istable=" + std::to_string(reduction.input_stable_pixels) +
           " iamb=" + std::to_string(reduction.input_ambiguous_pixels) +
           " ccov=" + std::to_string(reduction.coordinate_covered_pixels) +
           " caba=" + std::to_string(reduction.coordinate_aba_pixels) +
           " cstable=" + std::to_string(reduction.coordinate_stable_pixels) +
           " camb=" + std::to_string(reduction.coordinate_ambiguous_pixels) +
           " sv=" + std::to_string(report.source.available) +
           " rr=" + std::to_string(static_cast<u32>(report.source.route)) +
           " br=" + std::to_string(static_cast<u32>(report.source.binding)) +
           " eab=" + std::to_string(static_cast<u32>(report.source.mutation_ab)) +
           " ebc=" + std::to_string(static_cast<u32>(report.source.mutation_bc)) +
           " pc=" + std::to_string(static_cast<u32>(report.source.producers[0])) + "/" +
           std::to_string(static_cast<u32>(report.source.producers[1])) + "/" +
           std::to_string(static_cast<u32>(report.source.producers[2])) +
           " st=" + std::to_string(static_cast<u32>(reduction.status)) +
           " lm=" + std::to_string(FinalGuestSurfaceLossMask(reduction.loss, 0));
}

[[nodiscard]] inline std::string FormatPpUpstreamSampleTeeCalibratedCoverage(
    const PpUpstreamSampleTeeCalibratedCoverage& coverage) {
    return "FGSCTTC c=" + std::to_string(coverage.calibrations) +
           " o=" + std::to_string(coverage.outside) +
           " e=" + std::to_string(coverage.eligible) + "/" +
           std::to_string(coverage.emitted) + "/" + std::to_string(coverage.complete) +
           " lm=" + std::to_string(coverage.loss);
}

[[nodiscard]] inline std::string FormatPpUpstreamSampleTeeTransferredDepthSourceReport(
    const PpUpstreamSampleTeeCalibratedReport& report) {
    const auto& depth = report.transferred_depth_source;
    return "FGSCTTD q=" + std::to_string(report.request_ordinal) +
           " abc=" + std::to_string(report.sequences[0]) + "/" +
           std::to_string(report.sequences[1]) + "/" + std::to_string(report.sequences[2]) +
           " v=" + std::to_string(depth.available) +
           " cov=" + std::to_string(depth.covered_regions) +
           " aba=" + std::to_string(depth.aba_regions) +
           " stable=" + std::to_string(depth.stable_regions) +
           " amb=" + std::to_string(depth.ambiguous_regions) +
           " st=" + std::to_string(static_cast<u32>(depth.status)) +
           " lm=" + std::to_string(FinalGuestSurfaceLossMask(depth.loss, 0));
}

[[nodiscard]] inline std::string FormatPpUpstreamSampleTeeSelectedDepthReport(
    const PpUpstreamSampleTeeCalibratedReport& report) {
    const auto& depth = report.selected_depth;
    return "FGSCTSD q=" + std::to_string(report.request_ordinal) +
           " abc=" + std::to_string(report.sequences[0]) + "/" +
           std::to_string(report.sequences[1]) + "/" + std::to_string(report.sequences[2]) +
           " v=" + std::to_string(depth.available) +
           " cov=" + std::to_string(depth.covered_regions) +
           " aba=" + std::to_string(depth.aba_regions) +
           " stable=" + std::to_string(depth.stable_regions) +
           " amb=" + std::to_string(depth.ambiguous_regions) +
           " st=" + std::to_string(static_cast<u32>(depth.status)) +
           " lm=" + std::to_string(FinalGuestSurfaceLossMask(depth.loss, 0));
}

[[nodiscard]] inline std::string FormatPpUpstreamSampleTeeProducerPreDepthReport(
    const PpUpstreamSampleTeeCalibratedReport& report) {
    const auto& depth = report.producer_pre_depth;
    return "FGSCTPD q=" + std::to_string(report.request_ordinal) +
           " abc=" + std::to_string(report.sequences[0]) + "/" +
           std::to_string(report.sequences[1]) + "/" + std::to_string(report.sequences[2]) +
           " v=" + std::to_string(depth.available) +
           " cov=" + std::to_string(depth.covered_regions) +
           " aba=" + std::to_string(depth.aba_regions) +
           " stable=" + std::to_string(depth.stable_regions) +
           " amb=" + std::to_string(depth.ambiguous_regions) +
           " st=" + std::to_string(static_cast<u32>(depth.status)) +
           " lm=" + std::to_string(FinalGuestSurfaceLossMask(depth.loss, 0));
}

[[nodiscard]] inline std::string
FormatPpUpstreamSampleTeeProducerPreDepthScopeEntryReport(
    const PpUpstreamSampleTeeCalibratedReport& report) {
    const auto& scope = report.producer_pre_scope_entry;
    return "FGSCTPDS q=" + std::to_string(report.request_ordinal) +
           " abc=" + std::to_string(report.sequences[0]) + "/" +
           std::to_string(report.sequences[1]) + "/" +
           std::to_string(report.sequences[2]) + " v=" +
           std::to_string(scope.available) + " cov=" +
           std::to_string(scope.covered_regions) + " clearaba=" +
           std::to_string(scope.clear_aba_regions) + " loadwriter=" +
           std::to_string(scope.loaded_prior_writer_aba_regions) + " loadcontent=" +
           std::to_string(scope.loaded_prior_content_aba_regions) + " stable=" +
           std::to_string(scope.stable_regions) + " amb=" +
           std::to_string(scope.ambiguous_regions) + " match=" +
           std::to_string(scope.clear_materialization_match_regions) + " mismatch=" +
           std::to_string(scope.clear_materialization_mismatch_regions) + " cv=" +
           std::to_string(static_cast<u32>(scope.clear_value_relation)) + " wr=" +
           std::to_string(static_cast<u32>(scope.predecessor_writer_relation)) + " st=" +
           std::to_string(static_cast<u32>(scope.status)) + " lm=" +
           std::to_string(FinalGuestSurfaceLossMask(scope.loss, 0));
}

[[nodiscard]] inline std::string FormatPpUpstreamSampleTeeProducerPreSelectedDepthReport(
    const PpUpstreamSampleTeeCalibratedReport& report) {
    const auto& joint = report.producer_pre_selected_depth;
    return "FGSCTPDJ q=" + std::to_string(report.request_ordinal) +
           " abc=" + std::to_string(report.sequences[0]) + "/" +
           std::to_string(report.sequences[1]) + "/" + std::to_string(report.sequences[2]) +
           " v=" + std::to_string(joint.available) +
           " cov=" + std::to_string(joint.covered_regions) +
           " inherited=" + std::to_string(joint.inherited_aba_regions) +
           " introduced=" + std::to_string(joint.introduced_aba_regions) +
           " stable=" + std::to_string(joint.stable_regions) +
           " amb=" + std::to_string(joint.ambiguous_regions) +
           " eq=" + std::to_string(joint.same_frame_equal_regions[0]) + "/" +
           std::to_string(joint.same_frame_equal_regions[1]) + "/" +
           std::to_string(joint.same_frame_equal_regions[2]) + " neq=" +
           std::to_string(joint.same_frame_unequal_regions[0]) + "/" +
           std::to_string(joint.same_frame_unequal_regions[1]) + "/" +
           std::to_string(joint.same_frame_unequal_regions[2]) +
           " st=" + std::to_string(static_cast<u32>(joint.status)) +
           " lm=" + std::to_string(FinalGuestSurfaceLossMask(joint.loss, 0));
}

[[nodiscard]] inline std::string
FormatPpUpstreamSampleTeePredecessorWriterPreProducerPreDepthReport(
    const PpUpstreamSampleTeeCalibratedReport& report) {
    const auto& joint = report.predecessor_writer_pre_producer_pre_depth;
    return "FGSCTPDP q=" + std::to_string(report.request_ordinal) +
           " abc=" + std::to_string(report.sequences[0]) + "/" +
           std::to_string(report.sequences[1]) + "/" +
           std::to_string(report.sequences[2]) + " v=" +
           std::to_string(joint.available) + " cov=" +
           std::to_string(joint.covered_regions) + " inherited=" +
           std::to_string(joint.inherited_aba_regions) + " introduced=" +
           std::to_string(joint.introduced_aba_regions) + " stable=" +
           std::to_string(joint.stable_regions) + " amb=" +
           std::to_string(joint.ambiguous_regions) + " eq=" +
           std::to_string(joint.same_frame_equal_regions[0]) + "/" +
           std::to_string(joint.same_frame_equal_regions[1]) + "/" +
           std::to_string(joint.same_frame_equal_regions[2]) + " neq=" +
           std::to_string(joint.same_frame_unequal_regions[0]) + "/" +
           std::to_string(joint.same_frame_unequal_regions[1]) + "/" +
           std::to_string(joint.same_frame_unequal_regions[2]) + " st=" +
           std::to_string(static_cast<u32>(joint.status)) + " lm=" +
           std::to_string(FinalGuestSurfaceLossMask(joint.loss, 0));
}

class PpUpstreamSampleTeeCalibratedReducer {
public:
    static constexpr u32 MaxHistory = 32;
    static constexpr u32 MaxCalibrations = FinalGuestSurfaceMaxScreenshotRequests;
    static constexpr size_t MaxRetainedRecords =
        static_cast<size_t>(Shader::PpUpstreamSampleTeeMaxDimension) *
        Shader::PpUpstreamSampleTeeMaxDimension;

    explicit PpUpstreamSampleTeeCalibratedReducer(FinalGuestSurfaceCaptureWindow window_,
                                                  u32 history_capacity_ = MaxHistory,
                                                  u32 selected_candidate_index_ = 0,
                                                  u32 expected_calibrations_ = 0)
        : window{window_}, history_capacity{std::clamp(history_capacity_, 1u, MaxHistory)},
          selected_candidate_index{selected_candidate_index_},
          expected_calibrations{expected_calibrations_} {}

    void ObserveOwned(u64 sequence, const PpUpstreamSampleTeeCandidateRecord& candidate,
                      PpUpstreamSampleTeeOwnedReadback owned,
                      PpUpstreamSampleTeeSourceObservation source = {},
                      PpUpstreamSampleTeeTransferredDepthSourceOwned transferred_depth_source = {},
                      PpUpstreamSampleTeeSelectedDepthOwned selected_depth = {},
                      PpUpstreamSampleTeeProducerPreDepthOwned producer_pre_depth = {},
                      PpUpstreamSampleTeePredecessorWriterPreDepthOwned
                          predecessor_writer_pre_depth = {}) {
        if (finished || !window.Contains(sequence)) {
            return;
        }
        const u64 sequence_offset = sequence - window.frame_start;
        if (sequence_offset >= observed_sequences.size()) {
            SaturatingIncrement(coverage_loss);
            return;
        }
        if (observed_sequences[sequence_offset]) {
            if (const auto duplicate = std::find_if(
                    history.begin(), history.end(),
                    [sequence](const Observation& item) { return item.sequence == sequence; });
                duplicate != history.end()) {
                duplicate->status = FinalGuestSurfaceStatus::InvalidationLoss;
                duplicate->loss.invalidation = 1;
                std::vector<PpUpstreamSampleTeePixelRecord>{}.swap(duplicate->records);
                duplicate->source = {};
                duplicate->transferred_depth_source = {};
                duplicate->selected_depth = {};
                duplicate->producer_pre_depth = {};
                duplicate->predecessor_writer_pre_depth = {};
            }
            SaturatingIncrement(coverage_loss);
            last_content_sequence = std::max(last_content_sequence, sequence);
            Reconcile();
            return;
        }
        observed_sequences[sequence_offset] = true;
        Observation observation{
            .sequence = sequence,
            .key = {.candidate_index = candidate.candidate_index,
                    .resource_index = candidate.shader.resource_index,
                    .logical_ordinal = candidate.window.logical_ordinal,
                    .x = candidate.window.x,
                    .y = candidate.window.y,
                    .width = candidate.window.width,
                    .height = candidate.window.height,
                    .pixel_count = candidate.window.pixel_count},
            .status = owned.status,
            .loss = owned.loss,
        };
        const bool specific_loss = observation.status != FinalGuestSurfaceStatus::Complete &&
                                   observation.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
                                   observation.loss.Any();
        const bool canonical_candidate =
            IsCanonicalPpUpstreamSampleTeeCandidateRecord(candidate, selected_candidate_index);
        const bool canonical_owned =
            owned.status == FinalGuestSurfaceStatus::Complete && !owned.loss.Any() &&
            owned.mapped_invalidation_succeeded && owned.bounded_record_copy_succeeded &&
            !owned.retains_mapped_span &&
            owned.header.magic == Shader::PpUpstreamSampleTeeHeaderMagic &&
            owned.header.generation != 0 &&
            owned.header.candidate_index == candidate.candidate_index &&
            owned.header.x == candidate.window.x && owned.header.y == candidate.window.y &&
            owned.header.width == candidate.window.width &&
            owned.header.height == candidate.window.height &&
            owned.header.pixel_count == candidate.window.pixel_count &&
            owned.records.size() == candidate.window.pixel_count &&
            owned.records.capacity() <= MaxRetainedRecords;
        const bool canonical_source =
            IsCanonicalPpUpstreamSampleTeeSourceObservation(source, candidate);
        const auto depth_specific_loss = [&] {
            return transferred_depth_source.status != FinalGuestSurfaceStatus::Complete &&
                   transferred_depth_source.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
                   transferred_depth_source.loss.Any();
        }();
        const auto canonical_depth = [&] {
            const auto& plane = transferred_depth_source.plane;
            if (transferred_depth_source.status != FinalGuestSurfaceStatus::Complete ||
                transferred_depth_source.loss.Any() ||
                !transferred_depth_source.mapped_invalidation_succeeded ||
                !transferred_depth_source.bounded_copy_succeeded ||
                !transferred_depth_source.shared_release_deferred ||
                transferred_depth_source.retains_source_token ||
                transferred_depth_source.retains_values || plane.status != FinalGuestSurfaceStatus::Complete ||
                plane.loss.Any() || plane.region_count == 0 ||
                plane.region_count > FinalGuestSurfaceWatchOrdinals::MaxOrdinals ||
                plane.plane_bytes == 0 || plane.bytes_per_texel != 4 || plane.base_mip != 0 ||
                plane.mip_count != 1 || plane.base_layer != 0 ||
                transferred_depth_source.bytes.size() != plane.plane_bytes ||
                transferred_depth_source.bytes.capacity() >
                    Shader::PpUpstreamSampleTeeSlotStrideBytes) {
                return false;
            }
            u64 covered{};
            for (u32 index = 0; index < plane.region_count; ++index) {
                const auto& region = plane.regions[index];
                if (region.logical_ordinal == 0 || region.mip_level != 0 ||
                    region.byte_size == 0 ||
                    region.byte_size % plane.bytes_per_texel != 0 ||
                    region.buffer_offset != covered ||
                    static_cast<u64>(region.buffer_offset) + region.byte_size > plane.plane_bytes) {
                    return false;
                }
                covered += region.byte_size;
            }
            return covered == plane.plane_bytes;
        }();
        const bool empty_depth =
            transferred_depth_source.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            !transferred_depth_source.loss.Any() &&
            transferred_depth_source.plane.region_count == 0 &&
            transferred_depth_source.plane.plane_offset == 0 &&
            transferred_depth_source.plane.plane_bytes == 0 &&
            transferred_depth_source.bytes.empty() &&
            !transferred_depth_source.mapped_invalidation_succeeded &&
            !transferred_depth_source.bounded_copy_succeeded &&
            !transferred_depth_source.shared_release_deferred &&
            !transferred_depth_source.retains_source_token &&
            !transferred_depth_source.retains_values;
        const auto selected_specific_loss = [&] {
            return selected_depth.status != FinalGuestSurfaceStatus::Complete &&
                   selected_depth.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
                   selected_depth.loss.Any();
        }();
        const auto canonical_selected = [&] {
            const auto& plane = selected_depth.plane;
            if (selected_depth.status != FinalGuestSurfaceStatus::Complete ||
                selected_depth.loss.Any() || !selected_depth.mapped_invalidation_succeeded ||
                !selected_depth.bounded_copy_succeeded ||
                !selected_depth.shared_release_deferred || selected_depth.retains_selected_token ||
                selected_depth.retains_values ||
                plane.status != FinalGuestSurfaceStatus::Complete || plane.loss.Any() ||
                plane.region_count == 0 ||
                plane.region_count > FinalGuestSurfaceWatchOrdinals::MaxOrdinals ||
                plane.plane_bytes == 0 || plane.bytes_per_texel != 4 || plane.base_mip != 0 ||
                plane.mip_count != 1 || plane.base_layer != 0 ||
                selected_depth.bytes.size() != plane.plane_bytes ||
                selected_depth.bytes.capacity() > Shader::PpUpstreamSampleTeeSlotStrideBytes) {
                return false;
            }
            u64 covered{};
            for (u32 index = 0; index < plane.region_count; ++index) {
                const auto& region = plane.regions[index];
                if (region.logical_ordinal == 0 || region.mip_level != 0 ||
                    region.byte_size == 0 || region.byte_size % plane.bytes_per_texel != 0 ||
                    region.buffer_offset != covered ||
                    static_cast<u64>(region.buffer_offset) + region.byte_size >
                        plane.plane_bytes) {
                    return false;
                }
                covered += region.byte_size;
            }
            return covered == plane.plane_bytes;
        }();
        const bool empty_selected =
            selected_depth.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            !selected_depth.loss.Any() && selected_depth.plane.region_count == 0 &&
            selected_depth.plane.plane_offset == 0 && selected_depth.plane.plane_bytes == 0 &&
            selected_depth.bytes.empty() && !selected_depth.mapped_invalidation_succeeded &&
            !selected_depth.bounded_copy_succeeded && !selected_depth.shared_release_deferred &&
            !selected_depth.retains_selected_token && !selected_depth.retains_values;
        const auto producer_pre_specific_loss = [&] {
            return producer_pre_depth.status != FinalGuestSurfaceStatus::Complete &&
                   producer_pre_depth.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
                   producer_pre_depth.loss.Any();
        }();
        const auto canonical_producer_pre = [&] {
            const auto& plane = producer_pre_depth.plane;
            if (producer_pre_depth.status != FinalGuestSurfaceStatus::Complete ||
                producer_pre_depth.loss.Any() ||
                !producer_pre_depth.mapped_invalidation_succeeded ||
                !producer_pre_depth.bounded_copy_succeeded ||
                !producer_pre_depth.shared_release_deferred ||
                producer_pre_depth.retains_source_token ||
                producer_pre_depth.retains_private_hint || producer_pre_depth.retains_values ||
                plane.status != FinalGuestSurfaceStatus::Complete || plane.loss.Any() ||
                plane.region_count == 0 ||
                plane.region_count > FinalGuestSurfaceWatchOrdinals::MaxOrdinals ||
                plane.plane_offset != PpUpstreamSampleTeeProducerPreDepthPrefixBytes ||
                plane.plane_bytes == 0 ||
                plane.plane_bytes > PpUpstreamSampleTeeProducerPreDepthTailBytes ||
                static_cast<u64>(plane.plane_offset) + plane.plane_bytes >
                    PpUpstreamInputSnapshotBytes ||
                plane.bytes_per_texel != 4 || plane.base_mip != 0 ||
                plane.mip_count != 1 || plane.base_layer != 0 ||
                producer_pre_depth.bytes.size() != plane.plane_bytes ||
                producer_pre_depth.bytes.capacity() >
                    PpUpstreamSampleTeeProducerPreDepthTailBytes) {
                return false;
            }
            u64 covered{};
            for (u32 index = 0; index < plane.region_count; ++index) {
                const auto& region = plane.regions[index];
                if (region.logical_ordinal == 0 || region.mip_level != 0 ||
                    region.byte_size == 0 || region.byte_size % plane.bytes_per_texel != 0 ||
                    region.buffer_offset != covered ||
                    static_cast<u64>(region.buffer_offset) + region.byte_size >
                        plane.plane_bytes) {
                    return false;
                }
                covered += region.byte_size;
            }
            return covered == plane.plane_bytes;
        }();
        const bool empty_producer_pre =
            producer_pre_depth.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            !producer_pre_depth.loss.Any() && producer_pre_depth.plane.region_count == 0 &&
            producer_pre_depth.plane.plane_offset == 0 &&
            producer_pre_depth.plane.plane_bytes == 0 && producer_pre_depth.bytes.empty() &&
            !producer_pre_depth.mapped_invalidation_succeeded &&
            !producer_pre_depth.bounded_copy_succeeded &&
            !producer_pre_depth.shared_release_deferred &&
            !producer_pre_depth.retains_source_token &&
            !producer_pre_depth.retains_private_hint && !producer_pre_depth.retains_values;
        const auto scope_specific_loss = [&] {
            const auto& scope = producer_pre_depth.scope_entry;
            return scope.status != FinalGuestSurfaceStatus::Complete &&
                   scope.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
                   scope.loss.Any();
        }();
        const auto canonical_scope_entry = [&] {
            const auto& scope = producer_pre_depth.scope_entry;
            const bool clear = scope.load_op ==
                               PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Clear;
            const bool load = scope.load_op ==
                              PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Load;
            return scope.status == FinalGuestSurfaceStatus::Complete && !scope.loss.Any() &&
                   scope.exact && scope.metadata_only && scope.target_first_depth_writer &&
                   clear != load && scope.scope_clear == clear && scope.scope_load == load &&
                   scope.prior_endpoint_owns_pre_content == load &&
                   scope.has_private_depth_clear_bits == clear &&
                   (!load || (!scope.raw_depth_clear_encoding_exact &&
                              scope.private_depth_clear_bits == 0)) &&
                   scope.retains_private_predecessor_endpoint == load &&
                   (load ? scope.private_predecessor_endpoint.Valid()
                         : scope.private_predecessor_endpoint ==
                               PpDepthPredecessorEndpointStamp{}) &&
                   scope.additional_gpu_copy_count == 0;
        }();
        const auto empty_scope_entry = [&] {
            const auto& scope = producer_pre_depth.scope_entry;
            return scope.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                   !scope.loss.Any() &&
                   scope.load_op ==
                       PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::None &&
                   scope.private_predecessor_endpoint == PpDepthPredecessorEndpointStamp{} &&
                   scope.private_depth_clear_bits == 0 && !scope.target_first_depth_writer &&
                   !scope.scope_clear && !scope.scope_load &&
                   !scope.prior_endpoint_owns_pre_content &&
                   !scope.has_private_depth_clear_bits &&
                   !scope.raw_depth_clear_encoding_exact && !scope.metadata_only &&
                   !scope.exact && !scope.retains_private_predecessor_endpoint &&
                   scope.additional_gpu_copy_count == 0;
        }();
        const bool predecessor_pre_specific_loss =
            predecessor_writer_pre_depth.status != FinalGuestSurfaceStatus::Complete &&
            predecessor_writer_pre_depth.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
            predecessor_writer_pre_depth.loss.Any();
        const bool canonical_predecessor_pre =
            IsCanonicalPpUpstreamSampleTeePredecessorWriterPreDepthOwned(
                predecessor_writer_pre_depth);
        const bool empty_predecessor_pre =
            predecessor_writer_pre_depth.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            !predecessor_writer_pre_depth.loss.Any() &&
            predecessor_writer_pre_depth.plane.region_count == 0 &&
            predecessor_writer_pre_depth.plane.plane_offset == 0 &&
            predecessor_writer_pre_depth.plane.plane_bytes == 0 &&
            predecessor_writer_pre_depth.bytes.empty() &&
            !predecessor_writer_pre_depth.mapped_invalidation_succeeded &&
            !predecessor_writer_pre_depth.bounded_copy_succeeded &&
            !predecessor_writer_pre_depth.shared_release_deferred &&
            !predecessor_writer_pre_depth.exact_load_endpoint_bound &&
            !predecessor_writer_pre_depth.current_producer_pre_is_post_proxy &&
            predecessor_writer_pre_depth.additional_gpu_copy_count == 0 &&
            !predecessor_writer_pre_depth.retains_private_hint &&
            !predecessor_writer_pre_depth.retains_values;
        if (specific_loss) {
            // Preserve the concrete capture failure without retaining supplied record storage.
        } else if (!canonical_candidate || !canonical_owned) {
            observation.status = FinalGuestSurfaceStatus::InvalidationLoss;
            observation.loss = {.invalidation = 1};
        } else {
            observation.records = std::move(owned.records);
            if (canonical_source) {
                observation.source = std::move(source);
            }
            if (canonical_depth) {
                observation.transferred_depth_source = std::move(transferred_depth_source);
            } else if (depth_specific_loss) {
                observation.transferred_depth_source.status = transferred_depth_source.status;
                observation.transferred_depth_source.loss = transferred_depth_source.loss;
            } else if (!empty_depth) {
                observation.transferred_depth_source.status =
                    FinalGuestSurfaceStatus::InvalidationLoss;
                observation.transferred_depth_source.loss = {.invalidation = 1};
            }
            if (canonical_selected) {
                observation.selected_depth = std::move(selected_depth);
            } else if (selected_specific_loss) {
                observation.selected_depth.status = selected_depth.status;
                observation.selected_depth.loss = selected_depth.loss;
            } else if (!empty_selected) {
                observation.selected_depth.status = FinalGuestSurfaceStatus::InvalidationLoss;
                observation.selected_depth.loss = {.invalidation = 1};
            }
            if (canonical_producer_pre) {
                if (!canonical_scope_entry && !scope_specific_loss && !empty_scope_entry) {
                    producer_pre_depth.scope_entry = {
                        .status = FinalGuestSurfaceStatus::InvalidationLoss,
                        .loss = {.invalidation = 1}};
                }
                observation.producer_pre_depth = std::move(producer_pre_depth);
            } else if (producer_pre_specific_loss) {
                observation.producer_pre_depth.status = producer_pre_depth.status;
                observation.producer_pre_depth.loss = producer_pre_depth.loss;
            } else if (!empty_producer_pre) {
                observation.producer_pre_depth.status =
                    FinalGuestSurfaceStatus::InvalidationLoss;
                observation.producer_pre_depth.loss = {.invalidation = 1};
            }
            if (canonical_predecessor_pre) {
                observation.predecessor_writer_pre_depth =
                    std::move(predecessor_writer_pre_depth);
            } else if (predecessor_pre_specific_loss) {
                observation.predecessor_writer_pre_depth.status =
                    predecessor_writer_pre_depth.status;
                observation.predecessor_writer_pre_depth.loss =
                    predecessor_writer_pre_depth.loss;
            } else if (!empty_predecessor_pre) {
                observation.predecessor_writer_pre_depth.status =
                    FinalGuestSurfaceStatus::InvalidationLoss;
                observation.predecessor_writer_pre_depth.loss = {.invalidation = 1};
            }
        }
        if (history.size() == history_capacity) {
            history.pop_front();
        }
        history.push_back(std::move(observation));
        last_content_sequence = std::max(last_content_sequence, sequence);
        Reconcile();
    }

    void ObserveCalibration(FinalGuestSurfaceCalibratedStamp stamp) {
        if (finished) {
            return;
        }
        if (stamp.request_ordinal == 0 || stamp.request_ordinal > MaxCalibrations ||
            calibrations[stamp.request_ordinal].has_value()) {
            SaturatingIncrement(coverage_loss);
            return;
        }
        calibrations[stamp.request_ordinal] = stamp;
        SaturatingIncrement(calibration_count);
        Reconcile();
    }

    void Finish() {
        if (finished) {
            return;
        }
        finished = true;
        Reconcile();
        history.clear();
        observed_sequences.fill(false);
    }

    [[nodiscard]] std::vector<PpUpstreamSampleTeeCalibratedReport> TakeReports() {
        auto result = std::move(reports);
        reports.clear();
        return result;
    }

    [[nodiscard]] size_t RetainedObservationCount() const noexcept {
        return history.size();
    }

    [[nodiscard]] size_t
    RetainedProducerPreDepthScopePrivateEndpointCount() const noexcept {
        return std::ranges::count_if(history, [](const Observation& observation) {
            const auto& scope = observation.producer_pre_depth.scope_entry;
            return scope.retains_private_predecessor_endpoint &&
                   scope.private_predecessor_endpoint.Valid();
        });
    }

    [[nodiscard]] size_t RetainedPredecessorWriterPreDepthPrivateStateCount() const noexcept {
        return std::ranges::count_if(history, [](const Observation& observation) {
            return observation.predecessor_writer_pre_depth.retains_private_hint;
        });
    }

    [[nodiscard]] size_t RetainedRecordCapacity() const noexcept {
        size_t result{};
        for (const auto& observation : history) {
            if (observation.records.capacity() > std::numeric_limits<size_t>::max() - result) {
                return std::numeric_limits<size_t>::max();
            }
            result += observation.records.capacity();
        }
        return result;
    }

    [[nodiscard]] bool TerminalOutputReady() const noexcept {
        if (finished || expected_calibrations == 0 ||
            expected_calibrations > MaxCalibrations ||
            calibration_count != expected_calibrations) {
            return false;
        }
        for (u32 request = 1; request <= expected_calibrations; ++request) {
            if (!calibrations[request].has_value()) {
                return false;
            }
        }
        const u32 terminal_triplets = expected_calibrations > 2 ? expected_calibrations - 2 : 0;
        return outside_count + eligible_count == terminal_triplets &&
               emitted_count == eligible_count;
    }

    [[nodiscard]] PpUpstreamSampleTeeCalibratedCoverage GetCoverage() const noexcept {
        const u32 bounded_expected = std::min(expected_calibrations, MaxCalibrations);
        u32 present_expected{};
        for (u32 request = 1; request <= bounded_expected; ++request) {
            present_expected += calibrations[request].has_value();
        }
        u32 extras{};
        for (u32 request = bounded_expected + 1; request <= MaxCalibrations; ++request) {
            extras += calibrations[request].has_value();
        }
        const u32 missing = bounded_expected - present_expected;
        const u32 overflow =
            expected_calibrations > MaxCalibrations ? expected_calibrations - MaxCalibrations : 0;
        const u32 loss = SaturatingAdd(coverage_loss, missing, extras, overflow);
        return {
            .calibrations = calibration_count,
            .outside = outside_count,
            .eligible = eligible_count,
            .emitted = emitted_count,
            .complete = complete_count,
            .loss = loss,
            .ready = finished && missing == 0 && loss == 0 && emitted_count == eligible_count,
        };
    }

private:
    [[nodiscard]] static constexpr u32 SaturatingAdd(u32 first, u32 second, u32 third = 0,
                                                     u32 fourth = 0) noexcept {
        const auto add = [](u32 left, u32 right) {
            return right > std::numeric_limits<u32>::max() - left ? std::numeric_limits<u32>::max()
                                                                  : left + right;
        };
        return add(add(add(first, second), third), fourth);
    }

    static constexpr void SaturatingIncrement(u32& value) noexcept {
        value = SaturatingAdd(value, 1);
    }

    struct PrivateKey {
        u32 candidate_index{};
        u32 resource_index{};
        u32 logical_ordinal{};
        u32 x{};
        u32 y{};
        u32 width{};
        u32 height{};
        u32 pixel_count{};

        bool operator==(const PrivateKey&) const = default;
    };

    struct Observation {
        u64 sequence{};
        PrivateKey key{};
        std::vector<PpUpstreamSampleTeePixelRecord> records{};
        PpUpstreamSampleTeeSourceObservation source{};
        PpUpstreamSampleTeeTransferredDepthSourceOwned transferred_depth_source{};
        PpUpstreamSampleTeeSelectedDepthOwned selected_depth{};
        PpUpstreamSampleTeeProducerPreDepthOwned producer_pre_depth{};
        PpUpstreamSampleTeePredecessorWriterPreDepthOwned predecessor_writer_pre_depth{};
        FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss loss{};
    };

    struct SourceBindingKey {
        VideoCore::ImageColorScopePrivateLink link{};
        u64 backing_generation{};

        bool operator==(const SourceBindingKey&) const = default;
    };

    struct SourceRouteKey {
        PpUpstreamSampleRoute route{};
        u32 resolved_view_format{};
        u32 resolved_view_aspect{};

        bool operator==(const SourceRouteKey&) const = default;
    };

    template <typename Value>
    [[nodiscard]] static constexpr PpUpstreamSampleTeeTripletRelation ClassifyTriplet(
        const Value& a, const Value& b, const Value& c) noexcept {
        if (a == b && b == c) {
            return PpUpstreamSampleTeeTripletRelation::Stable;
        }
        if (a == c && a != b) {
            return PpUpstreamSampleTeeTripletRelation::Aba;
        }
        return PpUpstreamSampleTeeTripletRelation::Other;
    }

    [[nodiscard]] static constexpr SourceBindingKey SourceBinding(
        const PpUpstreamSampleTeeSourceObservation& source) noexcept {
        return {
            .link = source.binding.link,
            .backing_generation = source.binding.backing_generation,
        };
    }

    [[nodiscard]] static constexpr SourceRouteKey SourceRoute(
        const PpUpstreamSampleTeeSourceObservation& source) noexcept {
        return {
            .route = source.route,
            .resolved_view_format = source.resolved_view_format,
            .resolved_view_aspect = source.resolved_view_aspect,
        };
    }

    template <typename FirstPlane, typename SecondPlane>
    [[nodiscard]] static bool SameTransferredDepthSourceLayout(
        const FirstPlane& first, const SecondPlane& second) noexcept {
        if (first.region_count != second.region_count ||
            first.plane_offset != second.plane_offset || first.plane_bytes != second.plane_bytes ||
            first.bytes_per_texel != second.bytes_per_texel || first.base_mip != second.base_mip ||
            first.mip_count != second.mip_count || first.base_layer != second.base_layer ||
            first.status != second.status || first.loss != second.loss) {
            return false;
        }
        for (u32 index = 0; index < first.region_count; ++index) {
            const auto& lhs = first.regions[index];
            const auto& rhs = second.regions[index];
            if (lhs.logical_ordinal != rhs.logical_ordinal || lhs.mip_level != rhs.mip_level ||
                lhs.x != rhs.x || lhs.y != rhs.y || lhs.width != rhs.width ||
                lhs.height != rhs.height || lhs.buffer_offset != rhs.buffer_offset ||
                lhs.byte_size != rhs.byte_size) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static PpUpstreamSampleTeeCalibratedReport::TransferredDepthSourceReduction
    ReduceTransferredDepthSource(
        const PpUpstreamSampleTeeTransferredDepthSourceOwned& a,
        const PpUpstreamSampleTeeTransferredDepthSourceOwned& b,
        const PpUpstreamSampleTeeTransferredDepthSourceOwned& c) noexcept {
        using Reduction =
            PpUpstreamSampleTeeCalibratedReport::TransferredDepthSourceReduction;
        const auto specific = [](const auto& item) {
            return item.status != FinalGuestSurfaceStatus::Complete &&
                   item.status != FinalGuestSurfaceStatus::AlreadyConsumed && item.loss.Any();
        };
        if (specific(a) || specific(b) || specific(c)) {
            const auto& first = specific(a) ? a : specific(b) ? b : c;
            return {.status = first.status, .loss = first.loss};
        }
        const bool all_empty = a.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               b.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               c.status == FinalGuestSurfaceStatus::AlreadyConsumed;
        if (all_empty) {
            return {};
        }
        if (a.status != FinalGuestSurfaceStatus::Complete ||
            b.status != FinalGuestSurfaceStatus::Complete ||
            c.status != FinalGuestSurfaceStatus::Complete || a.loss.Any() || b.loss.Any() ||
            c.loss.Any() || !SameTransferredDepthSourceLayout(a.plane, b.plane) ||
            !SameTransferredDepthSourceLayout(a.plane, c.plane) ||
            a.bytes.size() != a.plane.plane_bytes || b.bytes.size() != b.plane.plane_bytes ||
            c.bytes.size() != c.plane.plane_bytes) {
            return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        Reduction result{
            .status = FinalGuestSurfaceStatus::Complete,
            .available = true,
        };
        for (u32 index = 0; index < a.plane.region_count; ++index) {
            const auto& region = a.plane.regions[index];
            const auto first = std::span<const std::byte>{a.bytes}.subspan(
                region.buffer_offset, region.byte_size);
            const auto middle = std::span<const std::byte>{b.bytes}.subspan(
                region.buffer_offset, region.byte_size);
            const auto last = std::span<const std::byte>{c.bytes}.subspan(
                region.buffer_offset, region.byte_size);
            switch (ClassifyPpUpstreamInputRawTriplet(first, middle, last)) {
            case PpUpstreamInputRawClass::Aba:
                ++result.aba_regions;
                break;
            case PpUpstreamInputRawClass::Stable:
                ++result.stable_regions;
                break;
            case PpUpstreamInputRawClass::Ambiguous:
                ++result.ambiguous_regions;
                break;
            case PpUpstreamInputRawClass::Unavailable:
                return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                        .loss = {.invalidation = 1}};
            }
            ++result.covered_regions;
        }
        return result;
    }

    [[nodiscard]] static PpUpstreamSampleTeeCalibratedReport::SelectedDepthReduction
    ReduceSelectedDepth(const PpUpstreamSampleTeeSelectedDepthOwned& a,
                        const PpUpstreamSampleTeeSelectedDepthOwned& b,
                        const PpUpstreamSampleTeeSelectedDepthOwned& c) noexcept {
        using Reduction = PpUpstreamSampleTeeCalibratedReport::SelectedDepthReduction;
        const auto specific = [](const auto& item) {
            return item.status != FinalGuestSurfaceStatus::Complete &&
                   item.status != FinalGuestSurfaceStatus::AlreadyConsumed && item.loss.Any();
        };
        if (specific(a) || specific(b) || specific(c)) {
            const auto& first = specific(a) ? a : specific(b) ? b : c;
            return {.status = first.status, .loss = first.loss};
        }
        const bool all_empty = a.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               b.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               c.status == FinalGuestSurfaceStatus::AlreadyConsumed;
        if (all_empty) {
            return {};
        }
        if (a.status != FinalGuestSurfaceStatus::Complete ||
            b.status != FinalGuestSurfaceStatus::Complete ||
            c.status != FinalGuestSurfaceStatus::Complete || a.loss.Any() || b.loss.Any() ||
            c.loss.Any() || !SameTransferredDepthSourceLayout(a.plane, b.plane) ||
            !SameTransferredDepthSourceLayout(a.plane, c.plane) ||
            a.bytes.size() != a.plane.plane_bytes || b.bytes.size() != b.plane.plane_bytes ||
            c.bytes.size() != c.plane.plane_bytes) {
            return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        Reduction result{.status = FinalGuestSurfaceStatus::Complete, .available = true};
        for (u32 index = 0; index < a.plane.region_count; ++index) {
            const auto& region = a.plane.regions[index];
            const auto first = std::span<const std::byte>{a.bytes}.subspan(
                region.buffer_offset, region.byte_size);
            const auto middle = std::span<const std::byte>{b.bytes}.subspan(
                region.buffer_offset, region.byte_size);
            const auto last = std::span<const std::byte>{c.bytes}.subspan(
                region.buffer_offset, region.byte_size);
            switch (ClassifyPpUpstreamInputRawTriplet(first, middle, last)) {
            case PpUpstreamInputRawClass::Aba:
                ++result.aba_regions;
                break;
            case PpUpstreamInputRawClass::Stable:
                ++result.stable_regions;
                break;
            case PpUpstreamInputRawClass::Ambiguous:
                ++result.ambiguous_regions;
                break;
            case PpUpstreamInputRawClass::Unavailable:
                return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                        .loss = {.invalidation = 1}};
            }
            ++result.covered_regions;
        }
        return result;
    }

    [[nodiscard]] static PpUpstreamSampleTeeCalibratedReport::ProducerPreDepthReduction
    ReduceProducerPreDepth(const PpUpstreamSampleTeeProducerPreDepthOwned& a,
                           const PpUpstreamSampleTeeProducerPreDepthOwned& b,
                           const PpUpstreamSampleTeeProducerPreDepthOwned& c) noexcept {
        using Reduction =
            PpUpstreamSampleTeeCalibratedReport::ProducerPreDepthReduction;
        const auto specific = [](const auto& item) {
            return item.status != FinalGuestSurfaceStatus::Complete &&
                   item.status != FinalGuestSurfaceStatus::AlreadyConsumed && item.loss.Any();
        };
        if (specific(a) || specific(b) || specific(c)) {
            const auto& first = specific(a) ? a : specific(b) ? b : c;
            return {.status = first.status, .loss = first.loss};
        }
        if (a.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            b.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            c.status == FinalGuestSurfaceStatus::AlreadyConsumed) {
            return {};
        }
        if (a.status != FinalGuestSurfaceStatus::Complete ||
            b.status != FinalGuestSurfaceStatus::Complete ||
            c.status != FinalGuestSurfaceStatus::Complete || a.loss.Any() || b.loss.Any() ||
            c.loss.Any() || !SameTransferredDepthSourceLayout(a.plane, b.plane) ||
            !SameTransferredDepthSourceLayout(a.plane, c.plane) ||
            a.bytes.size() != a.plane.plane_bytes || b.bytes.size() != b.plane.plane_bytes ||
            c.bytes.size() != c.plane.plane_bytes) {
            return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        Reduction result{.status = FinalGuestSurfaceStatus::Complete, .available = true};
        for (u32 index = 0; index < a.plane.region_count; ++index) {
            const auto& region = a.plane.regions[index];
            const auto first = std::span<const std::byte>{a.bytes}.subspan(
                region.buffer_offset, region.byte_size);
            const auto middle = std::span<const std::byte>{b.bytes}.subspan(
                region.buffer_offset, region.byte_size);
            const auto last = std::span<const std::byte>{c.bytes}.subspan(
                region.buffer_offset, region.byte_size);
            switch (ClassifyPpUpstreamInputRawTriplet(first, middle, last)) {
            case PpUpstreamInputRawClass::Aba:
                ++result.aba_regions;
                break;
            case PpUpstreamInputRawClass::Stable:
                ++result.stable_regions;
                break;
            case PpUpstreamInputRawClass::Ambiguous:
                ++result.ambiguous_regions;
                break;
            case PpUpstreamInputRawClass::Unavailable:
                return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                        .loss = {.invalidation = 1}};
            }
            ++result.covered_regions;
        }
        return result;
    }

    [[nodiscard]] static PpUpstreamSampleTeeCalibratedReport::ProducerPreDepthReduction
    ReducePredecessorWriterPreDepth(
        const PpUpstreamSampleTeePredecessorWriterPreDepthOwned& a,
        const PpUpstreamSampleTeePredecessorWriterPreDepthOwned& b,
        const PpUpstreamSampleTeePredecessorWriterPreDepthOwned& c) noexcept {
        using Reduction = PpUpstreamSampleTeeCalibratedReport::ProducerPreDepthReduction;
        const auto specific = [](const auto& item) {
            return item.status != FinalGuestSurfaceStatus::Complete &&
                   item.status != FinalGuestSurfaceStatus::AlreadyConsumed && item.loss.Any();
        };
        if (specific(a) || specific(b) || specific(c)) {
            const auto& first = specific(a) ? a : specific(b) ? b : c;
            return {.status = first.status, .loss = first.loss};
        }
        if (a.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            b.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            c.status == FinalGuestSurfaceStatus::AlreadyConsumed) {
            return {};
        }
        if (!IsCanonicalPpUpstreamSampleTeePredecessorWriterPreDepthOwned(a) ||
            !IsCanonicalPpUpstreamSampleTeePredecessorWriterPreDepthOwned(b) ||
            !IsCanonicalPpUpstreamSampleTeePredecessorWriterPreDepthOwned(c) ||
            !SameTransferredDepthSourceLayout(a.plane, b.plane) ||
            !SameTransferredDepthSourceLayout(a.plane, c.plane)) {
            return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        Reduction result{.status = FinalGuestSurfaceStatus::Complete, .available = true};
        for (u32 index = 0; index < a.plane.region_count; ++index) {
            const auto& region = a.plane.regions[index];
            const auto classification = ClassifyPpUpstreamInputRawTriplet(
                std::span<const std::byte>{a.bytes}.subspan(region.buffer_offset,
                                                            region.byte_size),
                std::span<const std::byte>{b.bytes}.subspan(region.buffer_offset,
                                                            region.byte_size),
                std::span<const std::byte>{c.bytes}.subspan(region.buffer_offset,
                                                            region.byte_size));
            switch (classification) {
            case PpUpstreamInputRawClass::Aba:
                ++result.aba_regions;
                break;
            case PpUpstreamInputRawClass::Stable:
                ++result.stable_regions;
                break;
            case PpUpstreamInputRawClass::Ambiguous:
                ++result.ambiguous_regions;
                break;
            case PpUpstreamInputRawClass::Unavailable:
                return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                        .loss = {.invalidation = 1}};
            }
            ++result.covered_regions;
        }
        return result;
    }

    [[nodiscard]] static PpUpstreamSampleTeeCalibratedReport::
        PredecessorWriterPreProducerPreDepthReduction
    ReducePredecessorWriterPreProducerPreDepth(
        const PpUpstreamSampleTeePredecessorWriterPreDepthOwned& xa,
        const PpUpstreamSampleTeePredecessorWriterPreDepthOwned& xb,
        const PpUpstreamSampleTeePredecessorWriterPreDepthOwned& xc,
        const PpUpstreamSampleTeeProducerPreDepthOwned& pa,
        const PpUpstreamSampleTeeProducerPreDepthOwned& pb,
        const PpUpstreamSampleTeeProducerPreDepthOwned& pc) noexcept {
        using Reduction = PpUpstreamSampleTeeCalibratedReport::
            PredecessorWriterPreProducerPreDepthReduction;
        const auto specific = [](const auto& item) {
            return item.status != FinalGuestSurfaceStatus::Complete &&
                   item.status != FinalGuestSurfaceStatus::AlreadyConsumed && item.loss.Any();
        };
        if (specific(xa) || specific(xb) || specific(xc)) {
            const auto& first = specific(xa) ? xa : specific(xb) ? xb : xc;
            return {.status = first.status, .loss = first.loss};
        }
        if (specific(pa) || specific(pb) || specific(pc)) {
            const auto& first = specific(pa) ? pa : specific(pb) ? pb : pc;
            return {.status = first.status, .loss = first.loss};
        }
        const bool all_empty = xa.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               xb.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               xc.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               pa.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               pb.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               pc.status == FinalGuestSurfaceStatus::AlreadyConsumed;
        if (all_empty) {
            return {};
        }
        const bool predecessor_all_empty =
            xa.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            xb.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            xc.status == FinalGuestSurfaceStatus::AlreadyConsumed;
        const bool producer_all_complete = pa.status == FinalGuestSurfaceStatus::Complete &&
                                           pb.status == FinalGuestSurfaceStatus::Complete &&
                                           pc.status == FinalGuestSurfaceStatus::Complete;
        if (predecessor_all_empty && producer_all_complete) {
            // The predecessor plane is optional. Its absence cannot invalidate the
            // independently complete producer-pre observation.
            return {};
        }
        const bool any_empty =
            xa.status == FinalGuestSurfaceStatus::AlreadyConsumed ||
            xb.status == FinalGuestSurfaceStatus::AlreadyConsumed ||
            xc.status == FinalGuestSurfaceStatus::AlreadyConsumed ||
            pa.status == FinalGuestSurfaceStatus::AlreadyConsumed ||
            pb.status == FinalGuestSurfaceStatus::AlreadyConsumed ||
            pc.status == FinalGuestSurfaceStatus::AlreadyConsumed;
        if (any_empty) {
            return {.status = FinalGuestSurfaceStatus::GapLoss,
                    .loss = {.gap = 1}};
        }
        if (!IsCanonicalPpUpstreamSampleTeePredecessorWriterPreDepthOwned(xa) ||
            !IsCanonicalPpUpstreamSampleTeePredecessorWriterPreDepthOwned(xb) ||
            !IsCanonicalPpUpstreamSampleTeePredecessorWriterPreDepthOwned(xc) ||
            pa.status != FinalGuestSurfaceStatus::Complete ||
            pb.status != FinalGuestSurfaceStatus::Complete ||
            pc.status != FinalGuestSurfaceStatus::Complete ||
            !SameTransferredDepthSourceLayout(xa.plane, xb.plane) ||
            !SameTransferredDepthSourceLayout(xa.plane, xc.plane) ||
            !SameTransferredDepthSourceLayout(pa.plane, pb.plane) ||
            !SameTransferredDepthSourceLayout(pa.plane, pc.plane) ||
            !IsPpUpstreamSampleTeeDepthPlaneSemanticLayoutExact(xa.plane, pa.plane) ||
            pa.bytes.size() != pa.plane.plane_bytes ||
            pb.bytes.size() != pb.plane.plane_bytes ||
            pc.bytes.size() != pc.plane.plane_bytes) {
            return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        Reduction result{.status = FinalGuestSurfaceStatus::Complete,
                         .available = true,
                         .additional_gpu_copy_count = 1};
        for (u32 index = 0; index < xa.plane.region_count; ++index) {
            const std::array x{
                std::span<const std::byte>{xa.bytes}.subspan(
                    xa.plane.regions[index].buffer_offset,
                    xa.plane.regions[index].byte_size),
                std::span<const std::byte>{xb.bytes}.subspan(
                    xb.plane.regions[index].buffer_offset,
                    xb.plane.regions[index].byte_size),
                std::span<const std::byte>{xc.bytes}.subspan(
                    xc.plane.regions[index].buffer_offset,
                    xc.plane.regions[index].byte_size),
            };
            const std::array p{
                std::span<const std::byte>{pa.bytes}.subspan(
                    pa.plane.regions[index].buffer_offset,
                    pa.plane.regions[index].byte_size),
                std::span<const std::byte>{pb.bytes}.subspan(
                    pb.plane.regions[index].buffer_offset,
                    pb.plane.regions[index].byte_size),
                std::span<const std::byte>{pc.bytes}.subspan(
                    pc.plane.regions[index].buffer_offset,
                    pc.plane.regions[index].byte_size),
            };
            const auto predecessor = ClassifyPpUpstreamInputRawTriplet(
                x[0], x[1], x[2]);
            const auto producer = ClassifyPpUpstreamInputRawTriplet(
                p[0], p[1], p[2]);
            if (predecessor == PpUpstreamInputRawClass::Aba &&
                producer == PpUpstreamInputRawClass::Aba) {
                ++result.inherited_aba_regions;
            } else if (predecessor != PpUpstreamInputRawClass::Aba &&
                       producer == PpUpstreamInputRawClass::Aba) {
                ++result.introduced_aba_regions;
            } else if (predecessor == PpUpstreamInputRawClass::Stable &&
                       producer == PpUpstreamInputRawClass::Stable) {
                ++result.stable_regions;
            } else if (predecessor == PpUpstreamInputRawClass::Unavailable ||
                       producer == PpUpstreamInputRawClass::Unavailable) {
                return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                        .loss = {.invalidation = 1}};
            } else {
                ++result.ambiguous_regions;
            }
            for (u32 frame = 0; frame < 3; ++frame) {
                auto& count = std::ranges::equal(x[frame], p[frame])
                                  ? result.same_frame_equal_regions[frame]
                                  : result.same_frame_unequal_regions[frame];
                ++count;
            }
            ++result.covered_regions;
        }
        return result;
    }

    [[nodiscard]] static PpUpstreamSampleTeeCalibratedReport::
        ProducerPreDepthScopeEntryReduction
        ReduceProducerPreDepthScopeEntry(
            const PpUpstreamSampleTeeProducerPreDepthOwned& a,
            const PpUpstreamSampleTeeProducerPreDepthOwned& b,
            const PpUpstreamSampleTeeProducerPreDepthOwned& c) noexcept {
        using Reduction = PpUpstreamSampleTeeCalibratedReport::
            ProducerPreDepthScopeEntryReduction;
        const auto specific = [](FinalGuestSurfaceStatus status,
                                 const FinalGuestSurfaceLoss& loss) {
            return status != FinalGuestSurfaceStatus::Complete &&
                   status != FinalGuestSurfaceStatus::AlreadyConsumed && loss.Any();
        };
        for (const auto* item : {&a, &b, &c}) {
            if (specific(item->status, item->loss)) {
                return {.status = item->status, .loss = item->loss};
            }
        }
        for (const auto* item : {&a.scope_entry, &b.scope_entry, &c.scope_entry}) {
            if (specific(item->status, item->loss)) {
                return {.status = item->status, .loss = item->loss};
            }
        }
        const bool all_empty =
            a.scope_entry.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            b.scope_entry.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
            c.scope_entry.status == FinalGuestSurfaceStatus::AlreadyConsumed;
        if (all_empty) {
            return {};
        }
        const auto canonical = [](const PpUpstreamSampleTeeProducerPreDepthOwned& item) {
            const auto& scope = item.scope_entry;
            const bool clear = scope.load_op ==
                               PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Clear;
            const bool load = scope.load_op ==
                              PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Load;
            return item.status == FinalGuestSurfaceStatus::Complete && !item.loss.Any() &&
                   scope.status == FinalGuestSurfaceStatus::Complete && !scope.loss.Any() &&
                   scope.exact && scope.metadata_only && scope.target_first_depth_writer &&
                   clear != load && scope.scope_clear == clear && scope.scope_load == load &&
                   scope.prior_endpoint_owns_pre_content == load &&
                   scope.has_private_depth_clear_bits == clear &&
                   (!load || (!scope.raw_depth_clear_encoding_exact &&
                              scope.private_depth_clear_bits == 0)) &&
                   scope.retains_private_predecessor_endpoint == load &&
                   (load ? scope.private_predecessor_endpoint.Valid()
                         : scope.private_predecessor_endpoint ==
                               PpDepthPredecessorEndpointStamp{}) &&
                   scope.additional_gpu_copy_count == 0 &&
                   item.bytes.size() == item.plane.plane_bytes;
        };
        if (!canonical(a) || !canonical(b) || !canonical(c) ||
            !SameTransferredDepthSourceLayout(a.plane, b.plane) ||
            !SameTransferredDepthSourceLayout(a.plane, c.plane) ||
            a.scope_entry.load_op != b.scope_entry.load_op ||
            a.scope_entry.load_op != c.scope_entry.load_op) {
            return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        Reduction result{.status = FinalGuestSurfaceStatus::Complete, .available = true};
        const bool clear = a.scope_entry.load_op ==
                           PpUpstreamSampleTeeProducerPreDepthScopeEntryLoadOp::Clear;
        if (clear && a.scope_entry.raw_depth_clear_encoding_exact &&
            b.scope_entry.raw_depth_clear_encoding_exact &&
            c.scope_entry.raw_depth_clear_encoding_exact) {
            const u32 first = a.scope_entry.private_depth_clear_bits;
            const u32 middle = b.scope_entry.private_depth_clear_bits;
            const u32 last = c.scope_entry.private_depth_clear_bits;
            result.clear_value_relation =
                first == middle && middle == last
                    ? PpUpstreamSampleTeeProducerPreDepthClearValueRelation::Stable
                : first == last
                    ? PpUpstreamSampleTeeProducerPreDepthClearValueRelation::Aba
                    : PpUpstreamSampleTeeProducerPreDepthClearValueRelation::Other;
        }
        if (!clear) {
            const auto& first = a.scope_entry.private_predecessor_endpoint.writer.hint;
            const auto& middle = b.scope_entry.private_predecessor_endpoint.writer.hint;
            const auto& last = c.scope_entry.private_predecessor_endpoint.writer.hint;
            result.predecessor_writer_relation =
                first == middle && middle == last
                    ? PpUpstreamSampleTeeProducerPreDepthWriterRelation::Stable
                : first == last
                    ? PpUpstreamSampleTeeProducerPreDepthWriterRelation::Aba
                    : PpUpstreamSampleTeeProducerPreDepthWriterRelation::Other;
        }
        const auto matches_clear = [](const PpUpstreamSampleTeeProducerPreDepthOwned& item,
                                      const PpUpstreamInputContentRegion& region) {
            if (!item.scope_entry.raw_depth_clear_encoding_exact ||
                region.byte_size == 0 || region.byte_size % sizeof(u32) != 0 ||
                static_cast<u64>(region.buffer_offset) + region.byte_size >
                    item.bytes.size()) {
                return false;
            }
            for (u32 offset = 0; offset < region.byte_size; offset += sizeof(u32)) {
                u32 word{};
                std::memcpy(&word, item.bytes.data() + region.buffer_offset + offset,
                            sizeof(word));
                if (word != item.scope_entry.private_depth_clear_bits) {
                    return false;
                }
            }
            return true;
        };
        for (u32 index = 0; index < a.plane.region_count; ++index) {
            const auto& region = a.plane.regions[index];
            const auto content = ClassifyPpUpstreamInputRawTriplet(
                std::span<const std::byte>{a.bytes}.subspan(region.buffer_offset,
                                                            region.byte_size),
                std::span<const std::byte>{b.bytes}.subspan(region.buffer_offset,
                                                            region.byte_size),
                std::span<const std::byte>{c.bytes}.subspan(region.buffer_offset,
                                                            region.byte_size));
            if (content == PpUpstreamInputRawClass::Unavailable) {
                return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                        .loss = {.invalidation = 1}};
            }
            if (clear) {
                const bool compare_clear =
                    result.clear_value_relation !=
                    PpUpstreamSampleTeeProducerPreDepthClearValueRelation::Unavailable;
                const bool materialized = compare_clear && matches_clear(a, region) &&
                                          matches_clear(b, region) && matches_clear(c, region);
                if (materialized) {
                    ++result.clear_materialization_match_regions;
                } else if (compare_clear) {
                    ++result.clear_materialization_mismatch_regions;
                }
                if (materialized &&
                    result.clear_value_relation ==
                        PpUpstreamSampleTeeProducerPreDepthClearValueRelation::Aba &&
                    content == PpUpstreamInputRawClass::Aba) {
                    ++result.clear_aba_regions;
                } else if (materialized &&
                           result.clear_value_relation ==
                               PpUpstreamSampleTeeProducerPreDepthClearValueRelation::Stable &&
                           content == PpUpstreamInputRawClass::Stable) {
                    ++result.stable_regions;
                } else {
                    ++result.ambiguous_regions;
                }
            } else if (content == PpUpstreamInputRawClass::Aba) {
                if (result.predecessor_writer_relation ==
                    PpUpstreamSampleTeeProducerPreDepthWriterRelation::Aba) {
                    ++result.loaded_prior_writer_aba_regions;
                } else if (result.predecessor_writer_relation ==
                           PpUpstreamSampleTeeProducerPreDepthWriterRelation::Stable) {
                    ++result.loaded_prior_content_aba_regions;
                } else {
                    ++result.ambiguous_regions;
                }
            } else if (content == PpUpstreamInputRawClass::Stable) {
                ++result.stable_regions;
            } else {
                ++result.ambiguous_regions;
            }
            ++result.covered_regions;
        }
        return result;
    }

    [[nodiscard]] static bool SameProducerSelectedSemanticLayout(
        const PpUpstreamSampleTeeSelectedDepthPlane& producer,
        const PpUpstreamSampleTeeSelectedDepthPlane& selected) noexcept {
        if (producer.region_count != selected.region_count ||
            producer.plane_bytes != selected.plane_bytes ||
            producer.bytes_per_texel != selected.bytes_per_texel ||
            producer.base_mip != selected.base_mip || producer.mip_count != selected.mip_count ||
            producer.base_layer != selected.base_layer) {
            return false;
        }
        for (u32 index = 0; index < producer.region_count; ++index) {
            const auto& a = producer.regions[index];
            const auto& b = selected.regions[index];
            if (a.logical_ordinal != b.logical_ordinal || a.mip_level != b.mip_level ||
                a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height ||
                a.buffer_offset != b.buffer_offset || a.byte_size != b.byte_size) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static PpUpstreamSampleTeeCalibratedReport::ProducerPreSelectedDepthReduction
    ReduceProducerPreSelectedDepth(
        const PpUpstreamSampleTeeProducerPreDepthOwned& pa,
        const PpUpstreamSampleTeeProducerPreDepthOwned& pb,
        const PpUpstreamSampleTeeProducerPreDepthOwned& pc,
        const PpUpstreamSampleTeeSelectedDepthOwned& sa,
        const PpUpstreamSampleTeeSelectedDepthOwned& sb,
        const PpUpstreamSampleTeeSelectedDepthOwned& sc) noexcept {
        using Reduction =
            PpUpstreamSampleTeeCalibratedReport::ProducerPreSelectedDepthReduction;
        const auto specific = [](const auto& item) {
            return item.status != FinalGuestSurfaceStatus::Complete &&
                   item.status != FinalGuestSurfaceStatus::AlreadyConsumed && item.loss.Any();
        };
        if (specific(pa) || specific(pb) || specific(pc)) {
            const auto& first = specific(pa) ? pa : specific(pb) ? pb : pc;
            return {.status = first.status, .loss = first.loss};
        }
        if (specific(sa) || specific(sb) || specific(sc)) {
            const auto& first = specific(sa) ? sa : specific(sb) ? sb : sc;
            return {.status = first.status, .loss = first.loss};
        }
        const bool all_empty = pa.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               pb.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               pc.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               sa.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               sb.status == FinalGuestSurfaceStatus::AlreadyConsumed &&
                               sc.status == FinalGuestSurfaceStatus::AlreadyConsumed;
        if (all_empty) {
            return {};
        }
        if (pa.status != FinalGuestSurfaceStatus::Complete ||
            pb.status != FinalGuestSurfaceStatus::Complete ||
            pc.status != FinalGuestSurfaceStatus::Complete ||
            sa.status != FinalGuestSurfaceStatus::Complete ||
            sb.status != FinalGuestSurfaceStatus::Complete ||
            sc.status != FinalGuestSurfaceStatus::Complete ||
            !SameTransferredDepthSourceLayout(pa.plane, pb.plane) ||
            !SameTransferredDepthSourceLayout(pa.plane, pc.plane) ||
            !SameTransferredDepthSourceLayout(sa.plane, sb.plane) ||
            !SameTransferredDepthSourceLayout(sa.plane, sc.plane) ||
            !SameProducerSelectedSemanticLayout(pa.plane, sa.plane) ||
            pa.bytes.size() != pa.plane.plane_bytes ||
            pb.bytes.size() != pb.plane.plane_bytes ||
            pc.bytes.size() != pc.plane.plane_bytes ||
            sa.bytes.size() != sa.plane.plane_bytes ||
            sb.bytes.size() != sb.plane.plane_bytes ||
            sc.bytes.size() != sc.plane.plane_bytes) {
            return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        Reduction result{.status = FinalGuestSurfaceStatus::Complete, .available = true};
        for (u32 index = 0; index < pa.plane.region_count; ++index) {
            const auto& p = pa.plane.regions[index];
            const auto& s = sa.plane.regions[index];
            const auto producer_class = ClassifyPpUpstreamInputRawTriplet(
                std::span<const std::byte>{pa.bytes}.subspan(p.buffer_offset, p.byte_size),
                std::span<const std::byte>{pb.bytes}.subspan(p.buffer_offset, p.byte_size),
                std::span<const std::byte>{pc.bytes}.subspan(p.buffer_offset, p.byte_size));
            const auto selected_class = ClassifyPpUpstreamInputRawTriplet(
                std::span<const std::byte>{sa.bytes}.subspan(s.buffer_offset, s.byte_size),
                std::span<const std::byte>{sb.bytes}.subspan(s.buffer_offset, s.byte_size),
                std::span<const std::byte>{sc.bytes}.subspan(s.buffer_offset, s.byte_size));
            if (producer_class == PpUpstreamInputRawClass::Aba &&
                selected_class == PpUpstreamInputRawClass::Aba) {
                ++result.inherited_aba_regions;
            } else if (producer_class != PpUpstreamInputRawClass::Aba &&
                       selected_class == PpUpstreamInputRawClass::Aba) {
                ++result.introduced_aba_regions;
            } else if (producer_class == PpUpstreamInputRawClass::Stable &&
                       selected_class == PpUpstreamInputRawClass::Stable) {
                ++result.stable_regions;
            } else if (producer_class == PpUpstreamInputRawClass::Unavailable ||
                       selected_class == PpUpstreamInputRawClass::Unavailable) {
                return {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                        .loss = {.invalidation = 1}};
            } else {
                ++result.ambiguous_regions;
            }
            const auto& pb_region = pb.plane.regions[index];
            const auto& pc_region = pc.plane.regions[index];
            const auto& sb_region = sb.plane.regions[index];
            const auto& sc_region = sc.plane.regions[index];
            const std::array producer_frames{
                std::span<const std::byte>{pa.bytes}.subspan(p.buffer_offset, p.byte_size),
                std::span<const std::byte>{pb.bytes}.subspan(pb_region.buffer_offset,
                                                             pb_region.byte_size),
                std::span<const std::byte>{pc.bytes}.subspan(pc_region.buffer_offset,
                                                             pc_region.byte_size),
            };
            const std::array selected_frames{
                std::span<const std::byte>{sa.bytes}.subspan(s.buffer_offset, s.byte_size),
                std::span<const std::byte>{sb.bytes}.subspan(sb_region.buffer_offset,
                                                             sb_region.byte_size),
                std::span<const std::byte>{sc.bytes}.subspan(sc_region.buffer_offset,
                                                             sc_region.byte_size),
            };
            for (u32 frame = 0; frame < producer_frames.size(); ++frame) {
                auto& count = std::ranges::equal(producer_frames[frame],
                                                 selected_frames[frame])
                                  ? result.same_frame_equal_regions[frame]
                                  : result.same_frame_unequal_regions[frame];
                ++count;
            }
            ++result.covered_regions;
        }
        return result;
    }

    [[nodiscard]] static constexpr PpUpstreamSampleTeeEdgeRelation ClassifyMutationEdge(
        const PpUpstreamSampleTeeSourceObservation& first,
        const PpUpstreamSampleTeeSourceObservation& second) noexcept {
        if (first.binding.link != second.binding.link) {
            return PpUpstreamSampleTeeEdgeRelation::Unavailable;
        }
        return first.binding.mutation_epoch == second.binding.mutation_epoch
                   ? PpUpstreamSampleTeeEdgeRelation::Stable
                   : PpUpstreamSampleTeeEdgeRelation::Changed;
    }

    [[nodiscard]] const Observation* Find(u64 sequence) const noexcept {
        const auto found =
            std::find_if(history.begin(), history.end(),
                         [sequence](const Observation& item) { return item.sequence == sequence; });
        return found == history.end() ? nullptr : &*found;
    }

    static void AddLoss(FinalGuestSurfaceLoss& target, const FinalGuestSurfaceLoss& source) {
        target.unsupported_type = SaturatingAdd(target.unsupported_type, source.unsupported_type);
        target.unsupported_samples =
            SaturatingAdd(target.unsupported_samples, source.unsupported_samples);
        target.unsupported_mip = SaturatingAdd(target.unsupported_mip, source.unsupported_mip);
        target.unsupported_layer =
            SaturatingAdd(target.unsupported_layer, source.unsupported_layer);
        target.unsupported_aspect =
            SaturatingAdd(target.unsupported_aspect, source.unsupported_aspect);
        target.unsupported_format =
            SaturatingAdd(target.unsupported_format, source.unsupported_format);
        target.invalid_extent = SaturatingAdd(target.invalid_extent, source.invalid_extent);
        target.logical_mapping = SaturatingAdd(target.logical_mapping, source.logical_mapping);
        target.tile_capacity = SaturatingAdd(target.tile_capacity, source.tile_capacity);
        target.byte_capacity = SaturatingAdd(target.byte_capacity, source.byte_capacity);
        target.ordinal_capacity = SaturatingAdd(target.ordinal_capacity, source.ordinal_capacity);
        target.busy = SaturatingAdd(target.busy, source.busy);
        target.invalidation = SaturatingAdd(target.invalidation, source.invalidation);
        target.gap = SaturatingAdd(target.gap, source.gap);
        target.history = SaturatingAdd(target.history, source.history);
        target.tile_detail = SaturatingAdd(target.tile_detail, source.tile_detail);
    }

    [[nodiscard]] PpUpstreamSampleTeeCalibratedReport Evaluate(u32 request, const Observation* a,
                                                               const Observation* b,
                                                               const Observation* c) const {
        PpUpstreamSampleTeeCalibratedReport report{
            .request_ordinal = request,
            .sequences = {calibrations[request - 2]->sequence, calibrations[request - 1]->sequence,
                          calibrations[request]->sequence},
            .reduction = {.status = FinalGuestSurfaceStatus::Complete},
        };
        const auto preserve_specific = [](const Observation* observation) {
            return observation && observation->status != FinalGuestSurfaceStatus::Complete &&
                   observation->status != FinalGuestSurfaceStatus::AlreadyConsumed &&
                   observation->loss.Any();
        };
        for (const Observation* observation : {a, b, c}) {
            if (observation) {
                AddLoss(report.reduction.loss, observation->loss);
            }
        }
        if (preserve_specific(a) || preserve_specific(b) || preserve_specific(c)) {
            const auto* first = preserve_specific(a) ? a : preserve_specific(b) ? b : c;
            report.reduction.status = first->status;
            return report;
        }
        if (!a || !b || !c) {
            report.reduction.status = FinalGuestSurfaceStatus::GapLoss;
            report.reduction.loss.history = 1;
            return report;
        }
        if (a->status != FinalGuestSurfaceStatus::Complete ||
            b->status != FinalGuestSurfaceStatus::Complete ||
            c->status != FinalGuestSurfaceStatus::Complete || report.reduction.loss.Any() ||
            a->key != b->key || a->key != c->key || a->records.size() != a->key.pixel_count ||
            b->records.size() != b->key.pixel_count || c->records.size() != c->key.pixel_count) {
            report.reduction.status = FinalGuestSurfaceStatus::InvalidationLoss;
            report.reduction.loss.invalidation = 1;
            return report;
        }
        report.reduction = ReducePpUpstreamSampleTee(a->records, b->records, c->records);
        if (a->source.valid && b->source.valid && c->source.valid) {
            report.source.available = true;
            report.source.route = ClassifyTriplet(SourceRoute(a->source), SourceRoute(b->source),
                                                  SourceRoute(c->source));
            report.source.binding = ClassifyTriplet(SourceBinding(a->source),
                                                    SourceBinding(b->source),
                                                    SourceBinding(c->source));
            report.source.mutation_ab = ClassifyMutationEdge(a->source, b->source);
            report.source.mutation_bc = ClassifyMutationEdge(b->source, c->source);
            report.source.producers = {a->source.producer, b->source.producer,
                                       c->source.producer};
        }
        report.transferred_depth_source =
            ReduceTransferredDepthSource(a->transferred_depth_source,
                                         b->transferred_depth_source,
                                         c->transferred_depth_source);
        report.selected_depth =
            ReduceSelectedDepth(a->selected_depth, b->selected_depth, c->selected_depth);
        report.producer_pre_depth = ReduceProducerPreDepth(
            a->producer_pre_depth, b->producer_pre_depth, c->producer_pre_depth);
        report.predecessor_writer_pre_depth = ReducePredecessorWriterPreDepth(
            a->predecessor_writer_pre_depth, b->predecessor_writer_pre_depth,
            c->predecessor_writer_pre_depth);
        report.predecessor_writer_pre_producer_pre_depth =
            ReducePredecessorWriterPreProducerPreDepth(
                a->predecessor_writer_pre_depth, b->predecessor_writer_pre_depth,
                c->predecessor_writer_pre_depth, a->producer_pre_depth,
                b->producer_pre_depth, c->producer_pre_depth);
        report.producer_pre_scope_entry = ReduceProducerPreDepthScopeEntry(
            a->producer_pre_depth, b->producer_pre_depth, c->producer_pre_depth);
        report.producer_pre_selected_depth = ReduceProducerPreSelectedDepth(
            a->producer_pre_depth, b->producer_pre_depth, c->producer_pre_depth,
            a->selected_depth, b->selected_depth, c->selected_depth);
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
            if (!window.Contains(a_stamp.sequence) || !window.Contains(b_stamp.sequence) ||
                !window.Contains(c_stamp.sequence)) {
                classified[request] = true;
                SaturatingIncrement(outside_count);
                continue;
            }
            if (!eligible_request[request]) {
                eligible_request[request] = true;
                SaturatingIncrement(eligible_count);
            }
            const bool stamps_valid = a_stamp.valid && b_stamp.valid && c_stamp.valid &&
                                      a_stamp.sequence < b_stamp.sequence &&
                                      b_stamp.sequence < c_stamp.sequence &&
                                      a_stamp.process_time_us < b_stamp.process_time_us &&
                                      b_stamp.process_time_us < c_stamp.process_time_us;
            const auto* a = Find(a_stamp.sequence);
            const auto* b = Find(b_stamp.sequence);
            const auto* c = Find(c_stamp.sequence);
            if (stamps_valid && (!a || !b || !c) && !finished &&
                last_content_sequence < c_stamp.sequence) {
                continue;
            }
            PpUpstreamSampleTeeCalibratedReport report{};
            if (!stamps_valid) {
                report.request_ordinal = request;
                report.sequences = {a_stamp.sequence, b_stamp.sequence, c_stamp.sequence};
                report.reduction.status = FinalGuestSurfaceStatus::InvalidationLoss;
                report.reduction.loss.invalidation = 1;
            } else {
                report = Evaluate(request, a, b, c);
            }
            SaturatingIncrement(emitted_count);
            if (report.reduction.status == FinalGuestSurfaceStatus::Complete &&
                !report.reduction.loss.Any()) {
                SaturatingIncrement(complete_count);
            }
            if (report.reduction.status != FinalGuestSurfaceStatus::Complete ||
                report.reduction.loss.Any()) {
                SaturatingIncrement(coverage_loss);
            }
            reports.push_back(std::move(report));
            classified[request] = true;
        }
    }

    FinalGuestSurfaceCaptureWindow window{};
    u32 history_capacity{};
    u32 selected_candidate_index{};
    u32 expected_calibrations{};
    std::deque<Observation> history{};
    std::array<bool, FinalGuestSurfaceCaptureWindow::MaxFrameCount> observed_sequences{};
    std::array<std::optional<FinalGuestSurfaceCalibratedStamp>, MaxCalibrations + 1> calibrations{};
    std::array<bool, MaxCalibrations + 1> classified{};
    std::array<bool, MaxCalibrations + 1> eligible_request{};
    std::vector<PpUpstreamSampleTeeCalibratedReport> reports{};
    u64 last_content_sequence{};
    u32 calibration_count{};
    u32 outside_count{};
    u32 eligible_count{};
    u32 emitted_count{};
    u32 complete_count{};
    u32 coverage_loss{};
    bool finished{};
};

struct PpUpstreamSampleTeeCalibratedOutput {
    PpUpstreamSampleTeeCalibratedCoverage coverage{};
    std::vector<PpUpstreamSampleTeeCalibratedReport> reports{};
};

[[nodiscard]] inline PpUpstreamSampleTeeCalibratedOutput
FinalizePpUpstreamSampleTeeCalibratedOutput(
    PpUpstreamSampleTeeCalibratedReducer& reducer) {
    reducer.Finish();
    PpUpstreamSampleTeeCalibratedOutput output{
        .coverage = reducer.GetCoverage(),
        .reports = reducer.TakeReports(),
    };
    if (!output.coverage.ready) {
        output.reports.clear();
    }
    return output;
}

} // namespace Vulkan::HostPasses
