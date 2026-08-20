// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "common/types.h"
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

    const auto async_compile = source.find("PipelineCache::CompileGraphicsPipelineAsync");
    const auto async_publish = source.find("PipelineCache::PublishAsyncGraphicsPipeline");
    ASSERT_NE(async_compile, std::string::npos);
    ASSERT_NE(async_publish, std::string::npos);
    EXPECT_EQ(source.substr(async_compile, async_publish - async_compile)
                  .find("ScopedHostCompilationGuestPause"),
              std::string::npos);
}

TEST(PersistentShaderCache, BackendSemanticChangesInvalidateStoredSpirv) {
    const auto source = ReadSource(SHADPS4_PIPELINE_SERIALIZATION_SOURCE_PATH);

    ASSERT_FALSE(source.empty());
    EXPECT_NE(source.find("static constexpr u32 ShaderBinaryVersion = 4u;"),
              std::string::npos);
}

} // namespace
} // namespace Vulkan
