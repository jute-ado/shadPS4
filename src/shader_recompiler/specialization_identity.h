// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/hash.h"
#include "shader_recompiler/specialization.h"

namespace Shader {

/**
 * Identifies the translation context of one guest shader program.
 *
 * The same guest code can legally be used by different physical or logical stages, whose runtime
 * information and emitted SPIR-V are not interchangeable.
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
        return static_cast<size_t>(
            HashCombine(stage_hash, static_cast<u64>(key.logical_stage)));
    }
};

/**
 * Returns whether a physical guest shader stage may legally back a logical pipeline stage.
 *
 * The physical stage is part of the emitted shader semantics and therefore must agree with the
 * logical slot that consumes it. Cached metadata that violates this relationship is corrupt or
 * belongs to a different pipeline configuration and must not be preloaded.
 */
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
