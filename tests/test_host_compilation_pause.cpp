// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

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

TEST(HostCompilationPause, DoesNotResumeAnExistingGuestPause) {
    FakeGuestPauseController controller{.paused = true};
    {
        ScopedHostCompilationGuestPause pause{controller};
        EXPECT_EQ(controller.pause_calls, 0u);
    }
    EXPECT_TRUE(controller.paused);
    EXPECT_EQ(controller.resume_calls, 0u);
}

TEST(HostCompilationPause, NestedCompilationResumesOnlyTheOwningScope) {
    FakeGuestPauseController controller;
    {
        ScopedHostCompilationGuestPause outer{controller};
        {
            ScopedHostCompilationGuestPause inner{controller};
            EXPECT_TRUE(controller.paused);
            EXPECT_EQ(controller.pause_calls, 1u);
            EXPECT_EQ(controller.resume_calls, 0u);
        }
        EXPECT_TRUE(controller.paused);
        EXPECT_EQ(controller.resume_calls, 0u);
    }
    EXPECT_FALSE(controller.paused);
    EXPECT_EQ(controller.resume_calls, 1u);
}

TEST(HostCompilationPause, WrapsShaderAndPipelineCompilationSites) {
    const auto source = ReadSource(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(source.empty());
    EXPECT_NE(source.find("#include \"video_core/renderer_vulkan/host_compilation_pause.h\""),
              std::string::npos);

    const auto graphics_log = source.find("Compiling graphics pipeline");
    const auto compute_log = source.find("Compiling compute pipeline");
    const auto shader_log = source.find("Compiling {} shader");
    ASSERT_NE(graphics_log, std::string::npos);
    ASSERT_NE(compute_log, std::string::npos);
    ASSERT_NE(shader_log, std::string::npos);
    EXPECT_NE(source.find("ScopedHostCompilationGuestPause pause{DebugState}", graphics_log),
              std::string::npos);
    EXPECT_NE(source.find("ScopedHostCompilationGuestPause pause{DebugState}", compute_log),
              std::string::npos);
    EXPECT_NE(source.rfind("ScopedHostCompilationGuestPause pause{DebugState}", shader_log),
              std::string::npos);
}

} // namespace
} // namespace Vulkan
