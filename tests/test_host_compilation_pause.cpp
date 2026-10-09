// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "common/types.h"
#include "video_core/renderer_vulkan/host_compilation_pause.h"

namespace Vulkan {
namespace {

struct FakeGuestPauseController {
    std::atomic_bool paused{};
    std::atomic<u32> pause_calls{};
    std::atomic<u32> resume_calls{};

    [[nodiscard]] bool IsGuestThreadsPaused() const noexcept {
        return paused.load();
    }

    void PauseGuestThreads() noexcept {
        paused.store(true);
        ++pause_calls;
    }

    void ResumeGuestThreads() noexcept {
        paused.store(false);
        ++resume_calls;
    }
};

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

bool ContainsBetween(const std::string& source, std::string_view begin, std::string_view end,
                     std::string_view expected) {
    const auto first = source.find(begin);
    if (first == std::string::npos) {
        return false;
    }
    const auto last = source.find(end, first + begin.size());
    return last != std::string::npos &&
           source.substr(first, last - first).find(expected) != std::string::npos;
}

TEST(HostCompilationPause, PausesGuestOnlyForTheHostCompilationLifetime) {
    FakeGuestPauseController controller;
    {
        ScopedHostCompilationGuestPause pause{controller};
        EXPECT_TRUE(controller.paused);
        EXPECT_EQ(controller.pause_calls, 1u);
        EXPECT_EQ(controller.resume_calls, 0u);
    }
    EXPECT_FALSE(controller.paused);
    EXPECT_EQ(controller.resume_calls, 1u);
}

TEST(HostCompilationPause, PreservesAnExistingGuestPause) {
    FakeGuestPauseController controller{.paused = true};
    {
        ScopedHostCompilationGuestPause pause{controller};
        EXPECT_EQ(controller.pause_calls, 0u);
    }
    EXPECT_TRUE(controller.paused);
    EXPECT_EQ(controller.resume_calls, 0u);
}

TEST(HostCompilationPause, OverlappingCompilersShareOneGuestPauseLease) {
    FakeGuestPauseController controller;
    auto first =
        std::make_unique<ScopedHostCompilationGuestPause<FakeGuestPauseController>>(controller);
    auto second =
        std::make_unique<ScopedHostCompilationGuestPause<FakeGuestPauseController>>(controller);

    EXPECT_TRUE(controller.paused);
    EXPECT_EQ(controller.pause_calls, 1u);
    first.reset();
    EXPECT_TRUE(controller.paused);
    EXPECT_EQ(controller.resume_calls, 0u);
    second.reset();
    EXPECT_FALSE(controller.paused);
    EXPECT_EQ(controller.resume_calls, 1u);
}

TEST(HostCompilationPause, DisabledLeaseDoesNotSuspendGuestWorkers) {
    FakeGuestPauseController controller;
    {
        ScopedHostCompilationGuestPause pause{controller, false};
        EXPECT_FALSE(controller.paused);
        EXPECT_EQ(controller.pause_calls, 0u);
    }
    EXPECT_FALSE(controller.paused);
    EXPECT_EQ(controller.resume_calls, 0u);
}

TEST(HostCompilationPause, SynchronousHostWorkStopsGuestFrameLifecycleProgress) {
    const auto source = ReadSource(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(source.empty());
    EXPECT_NE(source.find("host_compilation_pause.h"), std::string::npos);

    EXPECT_TRUE(ContainsBetween(source, "Compiling graphics pipeline", "RegisterPipelineData",
                                "ScopedHostCompilationGuestPause"));
    EXPECT_TRUE(ContainsBetween(source, "Compiling compute pipeline", "RegisterPipelineData",
                                "ScopedHostCompilationGuestPause"));
    EXPECT_TRUE(ContainsBetween(source, "vk::ShaderModule PipelineCache::CompileModule",
                                "RegisterShaderBinary", "ScopedHostCompilationGuestPause"));

    const auto compile_module = source.find("vk::ShaderModule PipelineCache::CompileModule");
    const auto register_shader = source.find("RegisterShaderBinary", compile_module);
    ASSERT_NE(compile_module, std::string::npos);
    ASSERT_NE(register_shader, std::string::npos);
    const auto module_compile = source.substr(compile_module, register_shader - compile_module);
    EXPECT_EQ(module_compile.find("std::chrono::seconds"), std::string::npos);
    EXPECT_NE(module_compile.find("std::vector<u32> code_snapshot{code.begin(), code.end()}"),
              std::string::npos);
    EXPECT_NE(module_compile.find("TranslateProgram(code_snapshot"), std::string::npos);
    EXPECT_EQ(module_compile.find("TranslateProgram(code,"), std::string::npos);
    EXPECT_NE(module_compile.find("!EmulatorSettings.IsAsyncGraphicsPipelineCompilation()"),
              std::string::npos);

    const auto async_compile = source.find("PipelineCache::CompileGraphicsPipelineAsync");
    const auto async_publish = source.find("PipelineCache::PublishAsyncGraphicsPipeline");
    ASSERT_NE(async_compile, std::string::npos);
    ASSERT_NE(async_publish, std::string::npos);
    EXPECT_EQ(source.substr(async_compile, async_publish - async_compile)
                  .find("ScopedHostCompilationGuestPause"),
              std::string::npos);
}

TEST(HostCompilationPause, AsyncGraphicsPreparationDoesNotSuspendGuestWorkers) {
    const auto source = ReadSource(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin = source.find("const GraphicsPipeline* PipelineCache::GetGraphicsPipeline");
    const auto end = source.find("const ComputePipeline* PipelineCache::GetComputePipeline", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto get_pipeline = source.substr(begin, end - begin);

    const auto async_branch = get_pipeline.find("if (async_graphics_compiler)");
    const auto snapshot = get_pipeline.find("MakeAsyncGraphicsPipelineJob()");
    const auto wait = get_pipeline.find("WaitForAsyncGraphicsPipeline");
    ASSERT_NE(async_branch, std::string::npos);
    ASSERT_NE(snapshot, std::string::npos);
    ASSERT_NE(wait, std::string::npos);
    ASSERT_LT(async_branch, snapshot);
    ASSERT_LT(snapshot, wait);
    const auto async_prepare = get_pipeline.substr(async_branch, wait - async_branch);
    EXPECT_EQ(async_prepare.find("ScopedHostCompilationGuestPause"), std::string::npos);
    EXPECT_NE(async_prepare.find("MakeAsyncGraphicsPipelineJob()"), std::string::npos);
}

TEST(AsyncGraphicsPipelineCompilation, InvalidVertexFormatsSkipTheDrawWithoutAsserting) {
    const auto pipeline = ReadSource(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto formats = ReadSource(SHADPS4_LIVERPOOL_TO_VK_SOURCE_PATH);
    const auto stages_begin = pipeline.find("bool PipelineCache::RefreshGraphicsStages");
    const auto stages_end = pipeline.find("bool PipelineCache::RefreshComputeKey", stages_begin);
    ASSERT_NE(stages_begin, std::string::npos);
    ASSERT_NE(stages_end, std::string::npos);
    const auto stages = pipeline.substr(stages_begin, stages_end - stages_begin);
    const auto supported = stages.find("IsSurfaceFormatSupported");
    const auto convert = stages.find("LiverpoolToVK::SurfaceFormat", supported);
    ASSERT_NE(supported, std::string::npos);
    ASSERT_NE(convert, std::string::npos);
    EXPECT_LT(supported, convert);
    EXPECT_NE(stages.substr(supported, convert - supported).find("return false"),
              std::string::npos);

    EXPECT_NE(formats.find("bool IsSurfaceFormatSupported"), std::string::npos);
}

TEST(AsyncGraphicsPipelineCompilation, InvalidColorFormatsSkipTheDrawWithoutAsserting) {
    const auto pipeline = ReadSource(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto key_begin = pipeline.find("bool PipelineCache::RefreshGraphicsKey");
    const auto key_end = pipeline.find("bool PipelineCache::RefreshGraphicsStages", key_begin);
    ASSERT_NE(key_begin, std::string::npos);
    ASSERT_NE(key_end, std::string::npos);
    const auto key = pipeline.substr(key_begin, key_end - key_begin);
    const auto supported = key.find("IsSurfaceFormatSupported");
    const auto assign = key.find("color_buffer.data_format");
    ASSERT_NE(supported, std::string::npos);
    ASSERT_NE(assign, std::string::npos);
    EXPECT_LT(supported, assign);
    EXPECT_NE(key.substr(supported, assign - supported).find("return false"), std::string::npos);
}

TEST(PersistentShaderCache, BackendSemanticChangesInvalidateStoredSpirv) {
    const auto source = ReadSource(SHADPS4_PIPELINE_SERIALIZATION_SOURCE_PATH);

    ASSERT_FALSE(source.empty());
    EXPECT_NE(source.find("static constexpr u32 ShaderBinaryVersion = 4u;"), std::string::npos);
}

} // namespace
} // namespace Vulkan
