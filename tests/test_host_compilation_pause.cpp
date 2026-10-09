// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "common/types.h"
#include "shader_recompiler/specialization_serialization.h"
#include "video_core/renderer_vulkan/host_compilation_pause.h"

namespace Vulkan {
namespace {

struct FakeGuestPauseController {
    bool paused{};
    u32 pause_calls{};
    u32 resume_calls{};

    [[nodiscard]] bool IsGuestThreadsPaused() const noexcept {
        return paused;
    }

    void PauseGuestThreads() noexcept {
        paused = true;
        ++pause_calls;
    }

    void ResumeGuestThreads() noexcept {
        paused = false;
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

TEST(HostCompilationPause, AsyncShaderCompilationDoesNotPauseGuestThreads) {
    FakeGuestPauseController controller;
    {
        ScopedHostCompilationGuestPause pause{controller, false};
        EXPECT_FALSE(controller.paused);
        EXPECT_EQ(controller.pause_calls, 0u);
    }
    EXPECT_FALSE(controller.paused);
    EXPECT_EQ(controller.resume_calls, 0u);
}

TEST(HostCompilationPause, AsyncComputePipelineCompilationDoesNotPauseGuestThreads) {
    const auto source = ReadSource(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    ASSERT_FALSE(source.empty());

    const auto begin = source.find("PipelineCache::GetComputePipeline()");
    const auto end = source.find("PipelineCache::RefreshGraphicsKey()", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto compute = source.substr(begin, end - begin);
    const auto pause = compute.find("ScopedHostCompilationGuestPause pause{");
    ASSERT_NE(pause, std::string::npos);
    const auto pause_end = compute.find("};", pause);
    ASSERT_NE(pause_end, std::string::npos);
    EXPECT_NE(compute.substr(pause, pause_end - pause)
                  .find("!EmulatorSettings.IsAsyncGraphicsPipelineCompilation()"),
              std::string::npos);

    // Compute dispatch still waits for the pipeline; async mode changes guest pause policy,
    // not the dispatch dependency or pipeline publication ordering.
    const auto construct = compute.find("std::make_unique<ComputePipeline>");
    const auto publish = compute.find("RegisterPipelineData", construct);
    const auto result = compute.find("return it->second.get()", publish);
    ASSERT_NE(construct, std::string::npos);
    ASSERT_NE(publish, std::string::npos);
    ASSERT_NE(result, std::string::npos);
    EXPECT_LT(construct, publish);
    EXPECT_LT(publish, result);
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
    EXPECT_NE(module_compile.find("IsAsyncGraphicsPipelineCompilation"), std::string::npos);
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

TEST(PersistentShaderCache, BackendSemanticChangesInvalidateStoredSpirv) {
    EXPECT_EQ(Shader::ShaderBinaryVersion, 8U);
}

} // namespace
} // namespace Vulkan
