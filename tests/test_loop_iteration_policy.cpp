// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/backend/spirv/loop_iteration_policy.h"

TEST(LoopIterationPolicy, DynamicScalarMaskLoopsRequestBoundedPartialUnrolling) {
    EXPECT_EQ(Shader::Backend::SPIRV::LoopPartialCountForDynamicScalarMask(true), 16U);
}

TEST(LoopIterationPolicy, OrdinaryLoopsKeepDriverDefaultUnrollingPolicy) {
    EXPECT_EQ(Shader::Backend::SPIRV::LoopPartialCountForDynamicScalarMask(false), 0U);
}
