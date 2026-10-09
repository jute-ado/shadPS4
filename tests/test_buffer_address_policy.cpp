// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/ir/passes/buffer_address_policy.h"

TEST(BufferAddressPolicy, ThreadIdContributionIsComputeOnly) {
    EXPECT_TRUE(Shader::Optimization::ShouldApplyBufferThreadId(true, true));
    EXPECT_FALSE(Shader::Optimization::ShouldApplyBufferThreadId(true, false));
    EXPECT_FALSE(Shader::Optimization::ShouldApplyBufferThreadId(false, true));
}
