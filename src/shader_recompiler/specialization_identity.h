// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/hash.h"
#include "shader_recompiler/specialization.h"

namespace Shader {

/**
 * Identifies the translation context of one guest shader program.
 *
 * A guest program hash is not sufficient by itself: the same code may be used by different
 * physical and logical stages whose emitted SPIR-V is not interchangeable.
 */
struct ProgramCacheKey {
    u64 program_hash{};
    Stage stage{};
    LogicalStage logical_stage{};

    bool operator==(const ProgramCacheKey&) const noexcept = default;
};

struct ProgramCacheKeyHash {
    [[nodiscard]] size_t operator()(const ProgramCacheKey& key) const noexcept {
        const u64 stage_hash = HashCombine(key.program_hash, static_cast<u64>(key.stage));
        return static_cast<size_t>(HashCombine(stage_hash, static_cast<u64>(key.logical_stage)));
    }
};

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
