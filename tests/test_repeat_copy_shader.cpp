// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/repeat_copy_shader.h"

TEST(RepeatCopyShader, CpuCurrentZeroControlCountIsNoOp) {
    EXPECT_TRUE(Vulkan::CanSkipRepeatCopyDispatch(0, 704, 704, false));
}

TEST(RepeatCopyShader, CpuCurrentZeroUserLimitIsNoOp) {
    EXPECT_TRUE(Vulkan::CanSkipRepeatCopyDispatch(100, 0, 704, false));
}

TEST(RepeatCopyShader, GpuOwnedControlCannotBeTrusted) {
    EXPECT_FALSE(Vulkan::CanSkipRepeatCopyDispatch(0, 704, 704, true));
}

TEST(RepeatCopyShader, NonzeroEffectiveCountUsesGuestShader) {
    EXPECT_FALSE(Vulkan::CanSkipRepeatCopyDispatch(100, 100, 704, false));
}
