// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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

} // namespace
} // namespace Vulkan
