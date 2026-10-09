// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/hash.h"
#include "shader_recompiler/specialization.h"

namespace Shader {

[[nodiscard]] constexpr u32 PackDrawUserDataLocations(
    const DrawUserDataLocations& locations) noexcept {
    return static_cast<u32>(locations.base_vertex) |
           (static_cast<u32>(locations.base_instance) << 8) |
           (static_cast<u32>(locations.draw_index) << 16);
}

inline constexpr u32 DisabledDrawUserDataKey = PackDrawUserDataLocations({});

/**
 * Identifies the translation context of one guest shader program.
 *
 * A guest program hash is not sufficient by itself: the same code may be used by different
 * physical and logical stages whose emitted SPIR-V is not interchangeable. CP-mapped user-data
 *
 * registers also change translation and host descriptor admission before specialization selection.

 */
struct ProgramCacheKey {
    u64 program_hash{};
    Stage stage{};
    LogicalStage logical_stage{};
    u32 draw_user_data_key = DisabledDrawUserDataKey;

    bool operator==(const ProgramCacheKey&) const noexcept = default;
};

struct ProgramCacheKeyHash {
    [[nodiscard]] size_t operator()(const ProgramCacheKey& key) const noexcept {
        const u64 stage_hash = HashCombine(key.program_hash, static_cast<u64>(key.stage));
        const u64 logical_hash = HashCombine(stage_hash, static_cast<u64>(key.logical_stage));
        // Preserve the existing hash for ordinary programs with no dynamic CP user-data mapping.
        return static_cast<size_t>(
            key.draw_user_data_key == DisabledDrawUserDataKey
                ? logical_hash
                : HashCombine(logical_hash, static_cast<u64>(key.draw_user_data_key)));
    }
};

// Draw-user-data substitution is currently supported only for ordinary physical/logical VS.
// Callers must not read the inactive vs_info union member for other physical stages.
[[nodiscard]] constexpr ProgramCacheKey MakeProgramCacheKey(
    u64 program_hash, Stage stage, LogicalStage logical_stage,
    const DrawUserDataLocations& draw_user_data = {}) noexcept {
    const u32 draw_key = stage == Stage::Vertex && logical_stage == LogicalStage::Vertex
                             ? PackDrawUserDataLocations(draw_user_data)
                             : DisabledDrawUserDataKey;
    return {program_hash, stage, logical_stage, draw_key};
}

[[nodiscard]] constexpr bool IsPhysicalStageCompatible(Stage stage,
                                                       LogicalStage logical_stage) noexcept {
    switch (logical_stage) {
    case LogicalStage::Fragment:
        return stage == Stage::Fragment;
    case LogicalStage::TessellationControl:
        return stage == Stage::Hull;
    case LogicalStage::TessellationEval:
        return stage == Stage::Vertex || stage == Stage::Export;
    case LogicalStage::Vertex:
        return stage == Stage::Vertex || stage == Stage::Export || stage == Stage::Local;
    case LogicalStage::Geometry:
        return stage == Stage::Geometry;
    case LogicalStage::Compute:
        return stage == Stage::Compute;
    case LogicalStage::NumLogicalStages:
        return false;
    }
    return false;
}

/**
 * Returns a stable identity for one compiled shader specialization.
 *
 * The identity is derived from explicit logical fields. It deliberately excludes native object
 * representation, pointers and runtime insertion order so it is suitable for persistent keys.
 */
[[nodiscard]] u64 ComputeShaderPermutationIdentity(u64 program_hash, LogicalStage logical_stage,
                                                   const StageSpecialization& spec);

} // namespace Shader
