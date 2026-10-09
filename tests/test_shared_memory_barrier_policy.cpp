// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/ir/passes/shared_memory_barrier_policy.h"

namespace Shader::Optimization {
namespace {

TEST(SharedMemoryBarrierPolicy, SeparatesReadWriteTransitions) {
    using enum SharedMemoryAccess;

    EXPECT_TRUE(NeedsSharedMemoryBarrier(Load, Write));
    EXPECT_TRUE(NeedsSharedMemoryBarrier(Write, Load));
    EXPECT_FALSE(NeedsSharedMemoryBarrier(Write, Write));
}

TEST(SharedMemoryBarrierPolicy, ConsecutiveLoadsRemainBarrierFree) {
    EXPECT_FALSE(NeedsSharedMemoryBarrier(SharedMemoryAccess::Load, SharedMemoryAccess::Load));
}

TEST(SharedMemoryBarrierPolicy, EmulatesLdsOrderingInsideMultiWaveWorkgroups) {
    EXPECT_TRUE(NeedsSharedMemoryBarrierEmulation(16, true));
    EXPECT_TRUE(NeedsSharedMemoryBarrierEmulation(64, true));
    EXPECT_TRUE(NeedsSharedMemoryBarrierEmulation(256, true));
}

TEST(SharedMemoryBarrierPolicy, RejectsEmptyOrUnsupportedGroups) {
    EXPECT_FALSE(NeedsSharedMemoryBarrierEmulation(0, true));
    EXPECT_FALSE(NeedsSharedMemoryBarrierEmulation(64, false));
}

} // namespace
} // namespace Shader::Optimization
