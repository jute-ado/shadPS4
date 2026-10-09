// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/read_const_access_policy.h"

using Shader::ReadConstAccess;
using Shader::RequiresReadConstDma;
using Shader::SelectReadConstAccess;

TEST(ReadConstAccessPolicy, ImmediateOffsetsUseFlatBufferWhenDmaIsDisabled) {
    EXPECT_EQ(SelectReadConstAccess(false, false), ReadConstAccess::FlatBuffer);
    EXPECT_FALSE(RequiresReadConstDma(false, false));
}

TEST(ReadConstAccessPolicy, ImmediateOffsetsUseDmaWithFlatBufferFallbackWhenEnabled) {
    EXPECT_EQ(SelectReadConstAccess(false, true), ReadConstAccess::DmaWithFlatBufferFallback);
    EXPECT_TRUE(RequiresReadConstDma(false, true));
}

TEST(ReadConstAccessPolicy, DynamicOffsetsAlwaysRequireDma) {
    EXPECT_EQ(SelectReadConstAccess(true, false), ReadConstAccess::DmaOnly);
    EXPECT_EQ(SelectReadConstAccess(true, true), ReadConstAccess::DmaOnly);
    EXPECT_TRUE(RequiresReadConstDma(true, false));
    EXPECT_TRUE(RequiresReadConstDma(true, true));
}
