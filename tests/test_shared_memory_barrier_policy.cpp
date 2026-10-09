// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/ir/passes/shared_memory_barrier_policy.h"

using Shader::Optimization::NeedsSharedMemoryBarriers;

TEST(SharedMemoryBarrierPolicy, RepairsWaveSynchronousTailInLargeWorkgroup) {
    EXPECT_TRUE(NeedsSharedMemoryBarriers(true, 1024, 256, true));
}

TEST(SharedMemoryBarrierPolicy, RepairsSingleWaveWorkgroup) {
    EXPECT_TRUE(NeedsSharedMemoryBarriers(true, 256, 64, true));
}

TEST(SharedMemoryBarrierPolicy, SkipsIrrelevantProgramsAndNativeDrivers) {
    EXPECT_FALSE(NeedsSharedMemoryBarriers(false, 1024, 256, true));
    EXPECT_FALSE(NeedsSharedMemoryBarriers(true, 0, 256, true));
    EXPECT_FALSE(NeedsSharedMemoryBarriers(true, 1024, 256, false));
}
