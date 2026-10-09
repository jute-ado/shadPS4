// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bit>
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
#include "video_core/renderer_vulkan/host_passes/pp_upstream_input_content.h"
#include "video_core/renderer_vulkan/host_passes/pp_upstream_sample_route.h"

namespace Vulkan::HostPasses {

struct PpUpstreamSampleTeePipelinePolicy {
    bool instrument_profile{};
    bool allow_persistent_artifacts{true};
    bool allow_shader_patches{true};
};

[[nodiscard]] constexpr PpUpstreamSampleTeePipelinePolicy PlanPpUpstreamSampleTeePipelinePolicy(
    bool strict_tee) noexcept {
    return {
        .instrument_profile = strict_tee,
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
    constexpr u64 BytesPerPixelRecord = sizeof(std::array<u32, 4>) + sizeof(u32);
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
    u32 writers{};
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
static_assert(offsetof(PpUpstreamSampleTeePixelRecord, writers) == 4 * sizeof(u32));
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

struct PpUpstreamSampleTeeDeferredReadbackDescriptor {
    bool enabled{};
    FinalGuestSurfaceReadbackSlotPool::Token outer_slot{};
    PpUpstreamSampleTeeCandidateRecord record{};
    PpUpstreamSampleTeeCapturePlan capture{};
    PpUpstreamSampleTeeCommandPlan commands{};
    PpUpstreamSampleTeeControllerState controller{};
    u32 selected_candidate_index{};
    FinalGuestSurfaceStatus capture_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss capture_loss{};
    bool capture_has_copy{};
    u64 mapped_bytes{};
};

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
    PpUpstreamSampleTeeCandidateRecord record{};
    PpUpstreamSampleTeeCapturePlan capture{};
    PpUpstreamSampleTeeCommandPlan commands{};
    PpUpstreamSampleTeeControllerState controller{};
    u32 selected_candidate_index{};
    FinalGuestSurfaceStatus capture_status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss capture_loss{};
    bool capture_has_copy{};
    u64 mapped_bytes{};
};

[[nodiscard]] inline PpUpstreamSampleTeeDeferredReadbackDescriptor
DetachPpUpstreamSampleTeeDeferredReadback(
    const PpUpstreamSampleTeeDeferredDetachDescriptor& descriptor) noexcept {
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
            .capture_status = status,
            .capture_loss = loss,
        };
    }
    return {
        .enabled = true,
        .outer_slot = descriptor.outer_slot,
        .record = descriptor.record,
        .capture = descriptor.capture,
        .commands = descriptor.commands,
        .controller = descriptor.controller,
        .selected_candidate_index = descriptor.selected_candidate_index,
        .capture_status = status,
        .capture_loss = loss,
        .capture_has_copy = complete,
        .mapped_bytes = descriptor.mapped_bytes,
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
    u64 records_bytes{};
    u32 record_count{};
    u32 expected_generation{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool copy{};
    bool borrows_outer_slot{};
    u32 outer_slot_release_attempts{};
    bool retains_mapped_span{};
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
        .records_bytes = records_bytes,
        .record_count = record.window.pixel_count,
        .expected_generation = descriptor.outer_slot.generation,
        .status = FinalGuestSurfaceStatus::Complete,
        .copy = true,
        .borrows_outer_slot = true,
        .outer_slot_release_attempts = outer_slot_release_attempts,
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
    bool observe_owned{};
};

[[nodiscard]] inline PpUpstreamSampleTeeDeferredObservation
FinalizePpUpstreamSampleTeeDeferredObservation(
    const PpUpstreamSampleTeeDeferredReadbackDescriptor& attempt,
    PpUpstreamSampleTeeDeferredLifecycleState lifecycle, PpUpstreamSampleTeeOwnedReadback owned) {
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
        return {.owned = std::move(owned)};
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
    if (plan.copy) {
        const bool invalidated =
            std::forward<Invalidate>(invalidate)(plan.invalidate_offset, plan.invalidate_bytes);
        owned = CopyPpUpstreamSampleTeeDeferredReadback(plan, invalidated, mapped);
        lifecycle = ObservePpUpstreamSampleTeeOwnedCopy(lifecycle, owned);
    }
    const bool had_slot = bool{attempt.outer_slot};
    const bool released =
        had_slot && std::forward<Release>(release)(attempt.outer_slot);
    auto shared = ReconcilePpUpstreamSampleTeeSharedSlotRelease(
        had_slot, released, base.status, base.loss, base.copy, lifecycle);
    auto observation = FinalizePpUpstreamSampleTeeDeferredObservation(
        attempt, shared.tee, std::move(owned));
    return {
        .base = shared.base,
        .tee = std::move(observation),
        .release_attempted = had_slot,
        .release_succeeded = released,
    };
}

struct PpUpstreamSampleTeeReduction {
    u32 covered_pixels{};
    u32 aba_pixels{};
    u32 stable_pixels{};
    u32 ambiguous_pixels{};
    u32 writer_conflicts{};
    u32 missing_pixels{};
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
    }
    return result;
}

struct PpUpstreamSampleTeeCalibratedReport {
    u32 request_ordinal{};
    std::array<u64, 3> sequences{};
    PpUpstreamSampleTeeReduction reduction{};
};

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
                      PpUpstreamSampleTeeOwnedReadback owned) {
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
        if (specific_loss) {
            // Preserve the concrete capture failure without retaining supplied record storage.
        } else if (!canonical_candidate || !canonical_owned) {
            observation.status = FinalGuestSurfaceStatus::InvalidationLoss;
            observation.loss = {.invalidation = 1};
        } else {
            observation.records = std::move(owned.records);
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
        FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss loss{};
    };

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
