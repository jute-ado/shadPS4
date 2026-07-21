// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/buffer_barrier_policy.h"

TEST(BufferBarrierPolicy, ReadOnlyBindingRequiresShaderReadAccess) {
    EXPECT_EQ(Vulkan::ShaderBufferAccess(false), vk::AccessFlagBits2::eShaderRead);
}

TEST(BufferBarrierPolicy, WritableBindingPreservesShaderReadAccess) {
    EXPECT_EQ(Vulkan::ShaderBufferAccess(true),
              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}

TEST(BufferBarrierPolicy, AliasedReadAndWriteBindingsMergeAccess) {
    auto access = Vulkan::ShaderBufferAccess(false);
    access = Vulkan::MergeShaderBufferAccess(access, true);

    EXPECT_EQ(access, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}
