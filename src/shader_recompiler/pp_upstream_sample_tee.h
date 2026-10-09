// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "common/types.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader {

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
inline constexpr u32 PpUpstreamSampleTeeRecordDwords = 5;
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

[[nodiscard]] constexpr bool ShouldInstrumentPpUpstreamSampleResult(bool enabled,
                                                                    LogicalStage stage,
                                                                    u32 resource_index) noexcept {
    return enabled && stage == LogicalStage::Fragment && resource_index < NUM_IMAGES;
}

struct PpUpstreamSampleTeeCompilerResourceDescriptor {
    bool enabled{};
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

    const bool eligible = descriptor.enabled && descriptor.stage == LogicalStage::Fragment;
    const bool has_capacity = descriptor.current_buffer_count <= NUM_BUFFERS &&
                              descriptor.later_required_buffer_count <= NUM_BUFFERS &&
                              descriptor.current_buffer_count +
                                      descriptor.later_required_buffer_count <
                                  NUM_BUFFERS;
    if (!eligible || !has_capacity) {
        return {.clear_result_counts = true};
    }
    return {.append_buffer = true, .require_frag_coord = true};
}

} // namespace Shader
