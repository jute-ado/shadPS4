// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_set>

#include <gtest/gtest.h>

#include "shader_recompiler/specialization_identity.h"

namespace {

using Shader::DrawUserDataLocations;
using Shader::LogicalStage;
using Shader::MakeProgramCacheKey;
using Shader::ProgramCacheKey;
using Shader::ProgramCacheKeyHash;
using Shader::Stage;

constexpr u64 ProgramHash = 0x123456789abcdef0ULL;

ProgramCacheKey VertexKey(const DrawUserDataLocations& locations = {}) {
    return MakeProgramCacheKey(ProgramHash, Stage::Vertex, LogicalStage::Vertex, locations);
}

TEST(DrawUserDataProgramCache, ExplicitDisabledMappingMatchesDefaultKey) {
    const auto default_key = VertexKey();
    const auto explicit_key =
        VertexKey({.base_vertex = 0xff, .base_instance = 0xff, .draw_index = 0xff});
    EXPECT_EQ(default_key, explicit_key);
    EXPECT_EQ(ProgramCacheKeyHash{}(default_key), ProgramCacheKeyHash{}(explicit_key));
    EXPECT_EQ(default_key, (ProgramCacheKey{ProgramHash, Stage::Vertex, LogicalStage::Vertex}));
    EXPECT_EQ(default_key.draw_user_data_key, 0x00ffffffU);
    EXPECT_EQ(
        ProgramCacheKeyHash{}(default_key),
        static_cast<size_t>(HashCombine(HashCombine(ProgramHash, static_cast<u64>(Stage::Vertex)),
                                        static_cast<u64>(LogicalStage::Vertex))));
}

TEST(DrawUserDataProgramCache, BaseVertexLocationChangesTranslationKeyAndHash) {
    const auto original = VertexKey();
    const auto mapped = VertexKey({.base_vertex = 2});
    EXPECT_NE(original, mapped);
    EXPECT_NE(ProgramCacheKeyHash{}(original), ProgramCacheKeyHash{}(mapped));
}

TEST(DrawUserDataProgramCache, BaseInstanceLocationChangesTranslationKeyAndHash) {
    const auto first = VertexKey({.base_instance = 2});
    const auto second = VertexKey({.base_instance = 3});
    EXPECT_NE(first, second);
    EXPECT_NE(ProgramCacheKeyHash{}(first), ProgramCacheKeyHash{}(second));
}

TEST(DrawUserDataProgramCache, DrawIndexLocationChangesTranslationKeyAndHash) {
    const auto original = VertexKey();
    const auto mapped = VertexKey({.draw_index = 4});
    EXPECT_NE(original, mapped);
    EXPECT_NE(ProgramCacheKeyHash{}(original), ProgramCacheKeyHash{}(mapped));
}

TEST(DrawUserDataProgramCache, RegisterZeroIsDistinctFromDisabledAndRoleOrderIsExplicit) {
    EXPECT_NE(VertexKey(), VertexKey({.base_instance = 0}));
    EXPECT_NE(VertexKey({.base_vertex = 2, .base_instance = 3}),
              VertexKey({.base_vertex = 3, .base_instance = 2}));
    EXPECT_NE(VertexKey({.base_vertex = 2}), VertexKey({.draw_index = 2}));
}

TEST(DrawUserDataProgramCache, EveryByteLocationRemainsDistinctWithoutTruncation) {
    for (unsigned role = 0; role < 3; ++role) {
        std::unordered_set<ProgramCacheKey, ProgramCacheKeyHash> keys;
        for (unsigned value = 0; value <= 255; ++value) {
            DrawUserDataLocations mapping;
            if (role == 0)
                mapping.base_vertex = static_cast<u8>(value);
            if (role == 1)
                mapping.base_instance = static_cast<u8>(value);
            if (role == 2)
                mapping.draw_index = static_cast<u8>(value);
            keys.emplace(VertexKey(mapping));
        }
        EXPECT_EQ(keys.size(), 256U);
    }
}

TEST(DrawUserDataProgramCache, MappedProgramsCoexistBeforeAggregateRefresh) {
    std::unordered_set<ProgramCacheKey, ProgramCacheKeyHash> programs;
    programs.emplace(VertexKey());
    programs.emplace(VertexKey({.base_instance = 2}));
    programs.emplace(VertexKey({.base_instance = 3}));
    programs.emplace(VertexKey({.base_vertex = 2, .base_instance = 3}));
    programs.emplace(VertexKey({.draw_index = 2}));
    EXPECT_EQ(programs.size(), 5U);
}

TEST(DrawUserDataProgramCache, PhysicalAndLogicalStageIdentityRemainsIndependent) {
    const DrawUserDataLocations map{.base_instance = 2};
    const auto vs = MakeProgramCacheKey(ProgramHash, Stage::Vertex, LogicalStage::Vertex, map);
    const auto es = MakeProgramCacheKey(ProgramHash, Stage::Export, LogicalStage::Vertex, map);
    const auto ls = MakeProgramCacheKey(ProgramHash, Stage::Local, LogicalStage::Vertex, map);
    EXPECT_NE(vs, es);
    EXPECT_NE(vs, ls);
    EXPECT_NE(es, ls);
    EXPECT_NE(vs, MakeProgramCacheKey(ProgramHash, Stage::Vertex, LogicalStage::TessellationEval));
}

TEST(DrawUserDataProgramCache, NonVertexLogicalStagesDoNotReadUnusedMappingState) {
    const DrawUserDataLocations unused{.base_vertex = 7, .base_instance = 8, .draw_index = 9};
    EXPECT_EQ(MakeProgramCacheKey(ProgramHash, Stage::Compute, LogicalStage::Compute),
              MakeProgramCacheKey(ProgramHash, Stage::Compute, LogicalStage::Compute, unused));
    EXPECT_EQ(MakeProgramCacheKey(ProgramHash, Stage::Fragment, LogicalStage::Fragment),
              MakeProgramCacheKey(ProgramHash, Stage::Fragment, LogicalStage::Fragment, unused));
}

TEST(DrawUserDataProgramCache, NonVertexPhysicalStagesIgnoreUnsupportedMappingState) {
    const DrawUserDataLocations unused{.base_vertex = 7, .base_instance = 8, .draw_index = 9};
    EXPECT_EQ(MakeProgramCacheKey(ProgramHash, Stage::Export, LogicalStage::Vertex),
              MakeProgramCacheKey(ProgramHash, Stage::Export, LogicalStage::Vertex, unused));
    EXPECT_EQ(MakeProgramCacheKey(ProgramHash, Stage::Local, LogicalStage::Vertex),
              MakeProgramCacheKey(ProgramHash, Stage::Local, LogicalStage::Vertex, unused));
}

TEST(DrawUserDataProgramCache, PackingUsesExplicitBytesAndNoNativePadding) {
    EXPECT_EQ(
        Shader::PackDrawUserDataLocations({.base_vertex = 2, .base_instance = 3, .draw_index = 4}),
        0x00040302U);
    EXPECT_EQ(VertexKey({.base_instance = 2}).draw_user_data_key, 0x00ff02ffU);
}

} // namespace
