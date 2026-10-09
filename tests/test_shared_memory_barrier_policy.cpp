// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/ir/passes/shared_memory_barrier_policy.h"

using Shader::Optimization::NeedsSharedMemoryBarriers;

TEST(SharedMemoryBarrierPolicy, PreservesGuestWaveVisibilityAcrossHostSubgroupWidths) {
    EXPECT_TRUE(NeedsSharedMemoryBarriers(true, 256, 32, true));
    EXPECT_TRUE(NeedsSharedMemoryBarriers(true, 256, 64, true));
    EXPECT_TRUE(NeedsSharedMemoryBarriers(true, 1024, 256, true));
}

TEST(SharedMemoryBarrierPolicy, SkipsProgramsThatCannotNeedTheRepair) {
    EXPECT_FALSE(NeedsSharedMemoryBarriers(false, 256, 64, true));
    EXPECT_FALSE(NeedsSharedMemoryBarriers(true, 0, 64, true));
    EXPECT_FALSE(NeedsSharedMemoryBarriers(true, 256, 0, true));
    EXPECT_FALSE(NeedsSharedMemoryBarriers(true, 256, 64, false));
}
