// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <vector>

#include <gtest/gtest.h>

#include "shader_recompiler/specialization_identity.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

namespace {

using Shader::DrawUserDataLocations;
using Shader::LogicalStage;
using Shader::MakeProgramCacheKey;
using Shader::ProgramCacheKey;
using Shader::Stage;

Vulkan::PendingProgramPermutation MakePending(const ProgramCacheKey& key,
                                              DrawUserDataLocations module_mapping,
                                              Vulkan::Program* existing = nullptr,
                                              u64 permutation_id = 1) {
    Shader::Info info{};
    info.pgm_hash = key.program_hash;
    info.stage = key.stage;
    info.l_stage = key.logical_stage;
    Shader::StageSpecialization spec{};
    spec.runtime_info.Initialize(key.stage);
    if (key.stage == Stage::Vertex) {
        spec.runtime_info.vs_info.draw_user_data = module_mapping;
    }
    auto program = existing ? nullptr : std::make_unique<Vulkan::Program>();
    if (program) {
        program->info = info;
    }
    return {
        .cache_key = key,
        .existing_program = existing,
        .new_program = std::move(program),
        .module = Vulkan::Program::Module{{}, std::move(spec), std::move(info), permutation_id},
    };
}

ProgramCacheKey VertexKey(u64 hash, DrawUserDataLocations mapping = {}) {
    return MakeProgramCacheKey(hash, Stage::Vertex, LogicalStage::Vertex, mapping);
}

TEST(DrawUserDataPreload, WrongMappingCannotPublishUnderSameShaderAndStage) {
    Vulkan::ProgramCacheMap programs;
    std::vector<Vulkan::PendingProgramPermutation> pending;
    pending.emplace_back(MakePending(VertexKey(0x100, {.base_instance = 2}), {.base_instance = 3}));
    auto* info = pending[0].module.info.get();
    auto* program = pending[0].new_program.get();
    EXPECT_FALSE(Vulkan::PublishPreloadedPrograms(programs, pending));
    EXPECT_TRUE(programs.empty());
    EXPECT_EQ(pending[0].module.info.get(), info);
    EXPECT_EQ(pending[0].new_program.get(), program);
}

TEST(DrawUserDataPreload, DisabledAndEnabledMappingAreNotInterchangeable) {
    for (bool mapped_key : {false, true}) {
        Vulkan::ProgramCacheMap programs;
        std::vector<Vulkan::PendingProgramPermutation> pending;
        pending.emplace_back(MakePending(
            VertexKey(0x100, mapped_key ? DrawUserDataLocations{.base_vertex = 2}
                                        : DrawUserDataLocations{}),
            mapped_key ? DrawUserDataLocations{} : DrawUserDataLocations{.base_vertex = 2}));
        EXPECT_FALSE(Vulkan::PublishPreloadedPrograms(programs, pending));
        EXPECT_TRUE(programs.empty());
    }
}

TEST(DrawUserDataPreload, BatchMappingMismatchLeavesAllProgramsUnpublished) {
    Vulkan::ProgramCacheMap programs;
    std::vector<Vulkan::PendingProgramPermutation> pending;
    pending.emplace_back(MakePending(VertexKey(0x100, {.draw_index = 4}), {.draw_index = 4}));
    pending.emplace_back(
        MakePending(VertexKey(0x200, {.base_instance = 2}), {.base_instance = 3}, nullptr, 2));
    EXPECT_FALSE(Vulkan::PublishPreloadedPrograms(programs, pending));
    EXPECT_TRUE(programs.empty());
    EXPECT_NE(pending[0].new_program, nullptr);
    EXPECT_NE(pending[1].new_program, nullptr);
    EXPECT_NE(pending[0].module.info, nullptr);
    EXPECT_NE(pending[1].module.info, nullptr);
}

TEST(DrawUserDataPreload, ExistingProgramCannotReceiveWrongTranslationContext) {
    const auto key = VertexKey(0x100, {.base_instance = 2});
    Vulkan::ProgramCacheMap programs;
    auto pending_owner = MakePending(key, {.base_instance = 2});
    programs.try_emplace(key, std::move(pending_owner.new_program));
    auto* existing = programs.at(key).get();
    std::vector<Vulkan::PendingProgramPermutation> pending;
    pending.emplace_back(MakePending(key, {.base_instance = 3}, existing));
    EXPECT_FALSE(Vulkan::PublishPreloadedPrograms(programs, pending));
    EXPECT_EQ(programs.size(), 1U);
    EXPECT_TRUE(existing->modules.empty());
    EXPECT_NE(pending[0].module.info, nullptr);
}

TEST(DrawUserDataPreload, NoncanonicalUnusedKeyBitsCannotHideMappingMismatch) {
    auto key = VertexKey(0x100);
    key.draw_user_data_key |= 0x01000000;
    Vulkan::ProgramCacheMap programs;
    std::vector<Vulkan::PendingProgramPermutation> pending;
    pending.emplace_back(MakePending(key, {}));
    EXPECT_FALSE(Vulkan::PublishPreloadedPrograms(programs, pending));
    EXPECT_TRUE(programs.empty());
}

TEST(DrawUserDataPreload, MatchingMappedProgramPublishesAndPreservesOwnedInfo) {
    const DrawUserDataLocations mapping{.base_vertex = 1, .base_instance = 2, .draw_index = 3};
    const auto key = VertexKey(0x100, mapping);
    Vulkan::ProgramCacheMap programs;
    std::vector<Vulkan::PendingProgramPermutation> pending;
    pending.emplace_back(MakePending(key, mapping));
    auto* info = pending[0].module.info.get();
    ASSERT_TRUE(Vulkan::PublishPreloadedPrograms(programs, pending));
    ASSERT_EQ(programs.at(key)->modules.size(), 1U);
    EXPECT_EQ(programs.at(key)->modules.front().info.get(), info);
    EXPECT_EQ(programs.at(key)->modules.front().spec.info, info);
}

TEST(DrawUserDataPreload, SeparateMappingsOfSameProgramCoexist) {
    const auto first = VertexKey(0x100, {.base_instance = 2});
    const auto second = VertexKey(0x100, {.base_instance = 3});
    Vulkan::ProgramCacheMap programs;
    std::vector<Vulkan::PendingProgramPermutation> pending;
    pending.emplace_back(MakePending(first, {.base_instance = 2}));
    pending.emplace_back(MakePending(second, {.base_instance = 3}, nullptr, 2));
    ASSERT_TRUE(Vulkan::PublishPreloadedPrograms(programs, pending));
    EXPECT_EQ(programs.size(), 2U);
    EXPECT_EQ(programs.at(first)->modules.size(), 1U);
    EXPECT_EQ(programs.at(second)->modules.size(), 1U);
}

TEST(DrawUserDataPreload, UnmappedVertexAndNonVertexProgramsStillPublish) {
    for (const auto key :
         {VertexKey(0x100), MakeProgramCacheKey(0x200, Stage::Compute, LogicalStage::Compute),
          MakeProgramCacheKey(0x300, Stage::Export, LogicalStage::Vertex)}) {
        Vulkan::ProgramCacheMap programs;
        std::vector<Vulkan::PendingProgramPermutation> pending;
        pending.emplace_back(MakePending(key, {}));
        EXPECT_TRUE(Vulkan::PublishPreloadedPrograms(programs, pending));
        EXPECT_EQ(programs.size(), 1U);
    }
}

} // namespace
