// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "shader_recompiler/read_const_access_policy.h"

TEST(ReadConstAccessPolicy, ImmediateOffsetsUseTheDrawLocalSnapshot) {
    EXPECT_EQ(Shader::SelectReadConstAccess(false, false), Shader::ReadConstAccess::FlatBuffer);
    EXPECT_EQ(Shader::SelectReadConstAccess(false, true), Shader::ReadConstAccess::FlatBuffer);
    EXPECT_FALSE(Shader::RequiresReadConstDma(false, false));
    EXPECT_FALSE(Shader::RequiresReadConstDma(false, true));
}

TEST(ReadConstAccessPolicy, DynamicOffsetsAlwaysUseGuardedDma) {
    EXPECT_EQ(Shader::SelectReadConstAccess(true, false), Shader::ReadConstAccess::DmaOnly);
    EXPECT_EQ(Shader::SelectReadConstAccess(true, true), Shader::ReadConstAccess::DmaOnly);
    EXPECT_TRUE(Shader::RequiresReadConstDma(true, false));
    EXPECT_TRUE(Shader::RequiresReadConstDma(true, true));
}
