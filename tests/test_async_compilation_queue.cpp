// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <future>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/async_compilation_queue.h"

namespace Vulkan {
namespace {

std::string ReadText(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(AsyncCompilationQueue, CoalescesDuplicateKeysAndPublishesOneResult) {
    std::promise<void> release;
    auto ready = release.get_future().share();
    std::atomic_uint compile_calls{};
    AsyncCompilationQueue<int, std::string, int> queue{[&](int payload) {
        ++compile_calls;
        ready.wait();
        return std::to_string(payload);
    }};

    EXPECT_TRUE(queue.Submit(7, 41));
    EXPECT_FALSE(queue.Submit(7, 99));
    release.set_value();

    auto completion = queue.WaitTake();
    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(completion->key, 7);
    ASSERT_TRUE(completion->result.has_value());
    EXPECT_EQ(*completion->result, "41");
    EXPECT_EQ(compile_calls, 1u);
}

TEST(AsyncCompilationQueue, FailedJobDoesNotKillWorkerOrPublishPartialResult) {
    AsyncCompilationQueue<int, int, int> queue{[](int payload) {
        if (payload < 0) {
            throw std::runtime_error{"compile failed"};
        }
        return payload * 2;
    }};

    EXPECT_TRUE(queue.Submit(1, -1));
    auto failed = queue.WaitTake();
    ASSERT_TRUE(failed.has_value());
    EXPECT_EQ(failed->key, 1);
    EXPECT_FALSE(failed->result.has_value());

    EXPECT_TRUE(queue.Submit(2, 6));
    auto succeeded = queue.WaitTake();
    ASSERT_TRUE(succeeded.has_value());
    ASSERT_TRUE(succeeded->result.has_value());
    EXPECT_EQ(*succeeded->result, 12);
}

TEST(AsyncCompilationQueue, ShutdownRejectsNewWorkAndJoinsTheWorker) {
    AsyncCompilationQueue<int, int, int> queue{[](int payload) { return payload; }};
    queue.Shutdown();

    EXPECT_FALSE(queue.Submit(1, 1));
    EXPECT_FALSE(queue.TryTake().has_value());
}

TEST(AsyncGraphicsPipelineCompilation, IsExplicitlyGatedAndPublishesOnTheOwnerThread) {
    const auto header = ReadText(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(header.empty());
    ASSERT_FALSE(source.empty());
    EXPECT_NE(header.find("AsyncGraphicsPipelineJob"), std::string::npos);
    EXPECT_NE(header.find("AsyncGraphicsPipelineResult"), std::string::npos);
    EXPECT_NE(header.find("info_copies"), std::string::npos);
    EXPECT_NE(source.find("EmulatorSettings.IsAsyncGraphicsPipelineCompilation()"),
              std::string::npos);
    const auto submit = source.find("async_graphics_compiler->Submit(");
    ASSERT_NE(submit, std::string::npos);
    EXPECT_NE(source.find("graphics_key", submit), std::string::npos);
    EXPECT_NE(source.find("PublishAsyncGraphicsPipelines()"), std::string::npos);
    EXPECT_NE(source.find("async_graphics_compiler->TryTake()"), std::string::npos);
    EXPECT_NE(source.find("WaitForAsyncGraphicsPipeline(graphics_key)"), std::string::npos);
    EXPECT_NE(source.find("result.pipeline->RebindStages(result.retained_infos)"),
              std::string::npos);

    const auto compute_begin = source.find("PipelineCache::GetComputePipeline()");
    const auto compute_end = source.find("PipelineCache::RefreshGraphicsKey()", compute_begin);
    ASSERT_NE(compute_begin, std::string::npos);
    ASSERT_NE(compute_end, std::string::npos);
    EXPECT_EQ(source.substr(compute_begin, compute_end - compute_begin).find("Submit("),
              std::string::npos);
}

TEST(AsyncGraphicsPipelineCompilation, HasAStableExperimentalConfigurationKey) {
    const auto settings_header = ReadText(SHADPS4_EMULATOR_SETTINGS_HEADER_PATH);
    const auto settings_source = ReadText(SHADPS4_EMULATOR_SETTINGS_SOURCE_PATH);

    ASSERT_FALSE(settings_header.empty());
    ASSERT_FALSE(settings_source.empty());
    EXPECT_NE(settings_header.find("async_graphics_pipeline_compilation"), std::string::npos);
    EXPECT_NE(settings_source.find("asyncGraphicsPipelineCompilation"), std::string::npos);
}

TEST(ShaderPermutationResources, RetainsTheInfoThatProducedEachCompiledModule) {
    const auto header = ReadText(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(header.empty());
    ASSERT_FALSE(source.empty());
    EXPECT_NE(header.find("std::unique_ptr<Shader::Info> owned_info"), std::string::npos);

    const auto get_program = source.find("PipelineCache::Result PipelineCache::GetProgram");
    const auto replace_shader = source.find("PipelineCache::ReplaceShader", get_program);
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(replace_shader, std::string::npos);
    const auto body = source.substr(get_program, replace_shader - get_program);
    EXPECT_NE(body.find("std::make_unique<Shader::Info>"), std::string::npos);
    EXPECT_NE(body.find("RegisterShaderMeta(*new_info"), std::string::npos);
    EXPECT_NE(body.find("std::move(new_info)"), std::string::npos);
    EXPECT_NE(body.find("selected->info"), std::string::npos);
}

TEST(ShaderPermutationResources, RecompilesWhenACurrentlyBoundImageChangesNumericClass) {
    Shader::StageSpecialization current{};
    Shader::StageSpecialization cached{};
    current.info = reinterpret_cast<const Shader::Info*>(1);
    cached.info = reinterpret_cast<const Shader::Info*>(1);
    current.images.emplace_back().is_integer = true;
    cached.images.emplace_back().is_integer = false;

    current.bitset.set(0);
    EXPECT_FALSE(current == cached);
}

TEST(ShaderPermutationResources, AllowsAnUnboundCurrentImageToReuseACompiledPermutation) {
    Shader::StageSpecialization current{};
    Shader::StageSpecialization cached{};
    current.info = reinterpret_cast<const Shader::Info*>(1);
    cached.info = reinterpret_cast<const Shader::Info*>(1);
    current.images.emplace_back().is_integer = false;
    cached.images.emplace_back().is_integer = true;

    cached.bitset.set(0);
    EXPECT_TRUE(current == cached);
}

TEST(ShaderPermutationResources, RecompilesWhenACurrentlyBoundBufferChangesStride) {
    Shader::StageSpecialization current{};
    Shader::StageSpecialization cached{};
    current.info = reinterpret_cast<const Shader::Info*>(1);
    cached.info = reinterpret_cast<const Shader::Info*>(1);
    current.buffers.emplace_back(Shader::BufferSpecialization{.stride = 16});
    cached.buffers.emplace_back(Shader::BufferSpecialization{.stride = 32});
    current.bitset.set(0);

    EXPECT_FALSE(current == cached);
}

TEST(ShaderPermutationResources, AllowsAnUnboundCurrentBufferToReuseACompiledPermutation) {
    Shader::StageSpecialization current{};
    Shader::StageSpecialization cached{};
    current.info = reinterpret_cast<const Shader::Info*>(1);
    cached.info = reinterpret_cast<const Shader::Info*>(1);
    current.buffers.emplace_back(Shader::BufferSpecialization{.stride = 16});
    cached.buffers.emplace_back(Shader::BufferSpecialization{.stride = 32});
    cached.bitset.set(0);

    EXPECT_TRUE(current == cached);
}

} // namespace
} // namespace Vulkan
