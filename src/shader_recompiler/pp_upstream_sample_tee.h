// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <span>

#include "common/types.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::IR {
enum class Opcode;
union TextureInstInfo;
} // namespace Shader::IR

namespace Shader {

enum class ImageResourceOperationUse : u32 {
    RegularSample = 1u << 0,
    DepthCompare = 1u << 1,
    Gather = 1u << 2,
    Fetch = 1u << 3,
    Query = 1u << 4,
    Storage = 1u << 5,
};

[[nodiscard]] u32 ClassifyImageResourceOperationUse(
    IR::Opcode opcode, const IR::TextureInstInfo& inst_info) noexcept;

[[nodiscard]] bool ImageResourcesShareBindingKey(const ImageResource& lhs,
                                                  const ImageResource& rhs) noexcept;

void ResetImageResourceOperationUses(Info& info) noexcept;

[[nodiscard]] bool MergeImageResourceOperationUse(Info& info, u32 binding_index,
                                                   u32 operation_use_mask) noexcept;

[[nodiscard]] bool RecordImageResourceOperationUse(
    Info& info, u32 binding_index, IR::Opcode opcode,
    const IR::TextureInstInfo& inst_info) noexcept;

void ResetSamplerImagePairUses(Info& info) noexcept;

[[nodiscard]] bool MergeSamplerImagePairUse(Info& info, u32 sampler_binding_index,
                                            u32 image_binding_index,
                                            u32 operation_use_mask,
                                            bool disable_aniso) noexcept;

inline constexpr u32 PpUpstreamSampleTeeMaxBytes = 128u << 10;
inline constexpr u32 PpUpstreamSampleTeeArmMask = 1u << 31;
inline constexpr u32 PpUpstreamSampleTeeResourceMask = NUM_IMAGES - 1;
inline constexpr u32 PpUpstreamSampleTeeCandidateShift = 6;
inline constexpr u32 PpUpstreamSampleTeeCandidateMask = 0x3u << PpUpstreamSampleTeeCandidateShift;
inline constexpr u32 PpUpstreamSampleTeeReservedMask =
    ~(PpUpstreamSampleTeeArmMask | PpUpstreamSampleTeeResourceMask |
      PpUpstreamSampleTeeCandidateMask);
inline constexpr u32 PpUpstreamSampleTeeHeaderMagic = 0x50505445;
inline constexpr u32 PpUpstreamSampleTeeHeaderDwords = 8;
inline constexpr u32 PpUpstreamSampleTeeRecordDwords = 7;
inline constexpr u32 PpUpstreamSampleTeeMaxDimension = 64;
inline constexpr u32 PpUpstreamSampleTeeSlotCount = 8;
inline constexpr u32 PpUpstreamSampleTeeSlotStrideBytes = PpUpstreamSampleTeeMaxBytes;
inline constexpr u32 PpUpstreamSampleTeeSlotStrideDwords =
    PpUpstreamSampleTeeSlotStrideBytes / sizeof(u32);
inline constexpr u32 PpUpstreamSampleTeeBufferBytes =
    PpUpstreamSampleTeeSlotCount * PpUpstreamSampleTeeSlotStrideBytes;
inline constexpr u32 PpUpstreamSampleTeeBufferDwords = PpUpstreamSampleTeeBufferBytes / sizeof(u32);
inline constexpr u32 PpUpstreamSampleTeeMaxRecordDwords =
    PpUpstreamSampleTeeHeaderDwords + PpUpstreamSampleTeeMaxDimension *
                                          PpUpstreamSampleTeeMaxDimension *
                                          PpUpstreamSampleTeeRecordDwords;

[[nodiscard]] constexpr bool IsValidPpUpstreamSampleTeeBaseDword(u32 base_dword) noexcept {
    return base_dword % PpUpstreamSampleTeeSlotStrideDwords == 0 &&
           base_dword <= PpUpstreamSampleTeeBufferDwords - PpUpstreamSampleTeeSlotStrideDwords;
}

static_assert(PpUpstreamSampleTeeMaxRecordDwords <= PpUpstreamSampleTeeSlotStrideDwords);

[[nodiscard]] constexpr bool IsPpUpstreamSampleTeeTarget(bool enabled, u64 target_pgm_hash,
                                                         u64 observed_pgm_hash,
                                                         LogicalStage stage) noexcept {
    return enabled && target_pgm_hash != 0 && target_pgm_hash == observed_pgm_hash &&
           stage == LogicalStage::Fragment;
}

[[nodiscard]] constexpr bool ShouldInstrumentPpUpstreamSampleResult(bool is_target,
                                                                    LogicalStage stage,
                                                                    u32 resource_index) noexcept {
    return is_target && stage == LogicalStage::Fragment && resource_index < NUM_IMAGES;
}

enum class PpUpstreamSampleTeeCubeShape : u8 {
    NonCube,
    SingleCube,
    CubeArray,
    PartialCube,
    Invalid,
};

struct PpUpstreamSampleTeeCubeBindingFact {
    bool observed{};
    bool valid{};
    LogicalStage logical_stage{};
    u32 image_resource_index{};
    u32 array_element{};
    PpUpstreamSampleTeeCubeShape shape{};
    u32 cube_count{};
    u32 base_layer{};
    u32 view_layer_count{};
    bool full_cube_range{};

    bool operator==(const PpUpstreamSampleTeeCubeBindingFact&) const noexcept = default;
};

[[nodiscard]] constexpr PpUpstreamSampleTeeCubeBindingFact
MakePpUpstreamSampleTeeCubeBindingFact(bool is_target, LogicalStage stage,
                                       u32 image_resource_index, u32 array_element,
                                       const ImageResource& resource,
                                       const AmdGpu::Image& sharp) noexcept {
    const u32 binding_count = resource.mip_fallback_mode == MipStorageFallbackMode::DynamicIndex
                                  ? (sharp.last_level >= sharp.base_level
                                         ? sharp.last_level - sharp.base_level + 1
                                         : 0)
                                  : 1;
    if (!ShouldInstrumentPpUpstreamSampleResult(is_target, stage, image_resource_index) ||
        array_element >= binding_count) {
        return {};
    }

    PpUpstreamSampleTeeCubeBindingFact fact{
        .observed = true,
        .valid = true,
        .logical_stage = stage,
        .image_resource_index = image_resource_index,
        .array_element = array_element,
        .shape = PpUpstreamSampleTeeCubeShape::NonCube,
    };
    if (!sharp.IsCube()) {
        return fact;
    }

    fact.cube_count = static_cast<u32>(sharp.depth) + 1;
    fact.base_layer = static_cast<u32>(sharp.base_array);
    const u32 last_layer = static_cast<u32>(sharp.last_array);
    const u32 available_layer_count = fact.cube_count * 6;
    if (last_layer < fact.base_layer || fact.base_layer >= available_layer_count) {
        fact.valid = false;
        fact.shape = PpUpstreamSampleTeeCubeShape::Invalid;
        return fact;
    }

    const u32 view_end = std::min(last_layer + 1, available_layer_count);
    fact.view_layer_count = view_end - fact.base_layer;
    fact.full_cube_range = fact.base_layer % 6 == 0 && fact.view_layer_count % 6 == 0;
    if (!fact.full_cube_range) {
        fact.valid = false;
        fact.shape = PpUpstreamSampleTeeCubeShape::PartialCube;
    } else {
        fact.shape = fact.view_layer_count == 6 ? PpUpstreamSampleTeeCubeShape::SingleCube
                                               : PpUpstreamSampleTeeCubeShape::CubeArray;
    }
    return fact;
}

[[nodiscard]] inline PpUpstreamSampleTeeCubeBindingFact
ObservePpUpstreamSampleTeeCubeBinding(const Info& info, bool is_target, LogicalStage stage,
                                      u32 image_resource_index,
                                      u32 array_element = 0) noexcept {
    if (!ShouldInstrumentPpUpstreamSampleResult(is_target, stage, image_resource_index) ||
        stage != info.l_stage || image_resource_index >= info.images.size()) {
        return {};
    }
    const auto& resource = info.images[image_resource_index];
    const auto sharp = resource.GetSharp(info);
    return MakePpUpstreamSampleTeeCubeBindingFact(is_target, stage, image_resource_index,
                                                  array_element, resource, sharp);
}

enum class PpUpstreamSampleTeeCubeFaceClass : u8 {
    NonCube,
    Identity,
    StrideCompressed,
    Reserved,
    Invalid,
};

struct PpUpstreamSampleTeeCubeFaceFact {
    PpUpstreamSampleTeeCubeFaceClass classification{};
    bool observed{};
    bool malformed{};

    bool operator==(const PpUpstreamSampleTeeCubeFaceFact&) const noexcept = default;
};

inline constexpr u32 PpUpstreamSampleTeeCubeMetadataClassMask = 0x7;
inline constexpr u32 PpUpstreamSampleTeeCubeMetadataPresentMask = 0x8;
inline constexpr u32 PpUpstreamSampleTeeCubeMetadataKnownMask =
    PpUpstreamSampleTeeCubeMetadataClassMask | PpUpstreamSampleTeeCubeMetadataPresentMask;

[[nodiscard]] inline PpUpstreamSampleTeeCubeFaceFact
ClassifyPpUpstreamSampleTeeCubeFace(bool is_cube, float face) noexcept {
    if (!is_cube) {
        return {.classification = PpUpstreamSampleTeeCubeFaceClass::NonCube,
                .observed = true};
    }
    if (!std::isfinite(face) || face < 0.0f || face > 4294967040.0f ||
        std::trunc(face) != face) {
        return {.classification = PpUpstreamSampleTeeCubeFaceClass::Invalid,
                .observed = true};
    }
    const auto code = static_cast<u32>(face);
    if (code % 8 >= 6) {
        return {.classification = PpUpstreamSampleTeeCubeFaceClass::Reserved,
                .observed = true};
    }
    return {.classification = code < 8 ? PpUpstreamSampleTeeCubeFaceClass::Identity
                                       : PpUpstreamSampleTeeCubeFaceClass::StrideCompressed,
            .observed = true};
}

[[nodiscard]] constexpr u32
PackPpUpstreamSampleTeeCubeMetadata(PpUpstreamSampleTeeCubeFaceFact fact) noexcept {
    const u32 classification = static_cast<u32>(fact.classification);
    if (!fact.observed || fact.malformed ||
        classification > static_cast<u32>(PpUpstreamSampleTeeCubeFaceClass::Invalid)) {
        return 0;
    }
    return PpUpstreamSampleTeeCubeMetadataPresentMask | classification;
}

[[nodiscard]] constexpr PpUpstreamSampleTeeCubeFaceFact
DecodePpUpstreamSampleTeeCubeMetadata(u32 metadata) noexcept {
    if (metadata == 0) {
        return {};
    }
    const u32 classification = metadata & PpUpstreamSampleTeeCubeMetadataClassMask;
    if ((metadata & ~PpUpstreamSampleTeeCubeMetadataKnownMask) != 0 ||
        (metadata & PpUpstreamSampleTeeCubeMetadataPresentMask) == 0 ||
        classification > static_cast<u32>(PpUpstreamSampleTeeCubeFaceClass::Invalid)) {
        return {.malformed = true};
    }
    return {.classification = static_cast<PpUpstreamSampleTeeCubeFaceClass>(classification),
            .observed = true};
}

[[nodiscard]] inline bool UsesPpUpstreamSampleTeePushData(
    LogicalStage stage, std::span<const BufferResource> buffers) noexcept {
    return stage == LogicalStage::Fragment &&
           std::ranges::any_of(buffers, [](const BufferResource& buffer) {
               return buffer.buffer_type == BufferType::PpUpstreamSampleTee;
           });
}

struct PpUpstreamSampleTeeCompilerResourceDescriptor {
    bool is_target{};
    LogicalStage stage{};
    std::span<const u32> result_counts{};
    size_t current_buffer_count{};
    size_t later_required_buffer_count{};
};

struct PpUpstreamSampleTeeCompilerResourcePlan {
    bool append_buffer{};
    bool clear_result_counts{};
    bool require_frag_coord{};
};

[[nodiscard]] constexpr PpUpstreamSampleTeeCompilerResourcePlan
PlanPpUpstreamSampleTeeCompilerResource(
    const PpUpstreamSampleTeeCompilerResourceDescriptor& descriptor) noexcept {
    bool has_result{};
    for (const u32 count : descriptor.result_counts) {
        has_result |= count != 0;
    }
    if (!has_result) {
        return {};
    }

    const bool eligible = descriptor.is_target && descriptor.stage == LogicalStage::Fragment;
    const bool has_capacity =
        descriptor.current_buffer_count <= NUM_BUFFERS &&
        descriptor.later_required_buffer_count <= NUM_BUFFERS &&
        descriptor.current_buffer_count + descriptor.later_required_buffer_count < NUM_BUFFERS;
    if (!eligible || !has_capacity) {
        return {.clear_result_counts = true};
    }
    return {.append_buffer = true, .require_frag_coord = true};
}

} // namespace Shader
