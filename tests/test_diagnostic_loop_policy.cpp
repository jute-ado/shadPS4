// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/backend/spirv/diagnostic_loop_policy.h"

using Shader::Backend::SPIRV::DiagnosticLoopBudget;
using Shader::Backend::SPIRV::DiagnosticLoopLimitForShader;
using Shader::Backend::SPIRV::ParseDiagnosticLoopBudget;

TEST(DiagnosticLoopPolicy, IsDisabledWithoutAnExplicitLimitAndShaderHash) {
    EXPECT_EQ(DiagnosticLoopLimitForShader({}, 0x4defcf7f, 0), 0);
    EXPECT_EQ(DiagnosticLoopLimitForShader(
                  DiagnosticLoopBudget{
                      .shader_hash = 0x4defcf7f, .loop_index = 0, .iteration_limit = 0},
                  0x4defcf7f, 0),
              0);
}

TEST(DiagnosticLoopPolicy, AppliesOnlyToTheExplicitlySelectedShader) {
    constexpr DiagnosticLoopBudget budget{
        .shader_hash = 0x4defcf7f,
        .loop_index = 2,
        .iteration_limit = 32,
    };

    EXPECT_EQ(DiagnosticLoopLimitForShader(budget, 0x4defcf7f, 2), 32);
    EXPECT_EQ(DiagnosticLoopLimitForShader(budget, 0x4defcf7f, 1), 0);
    EXPECT_EQ(DiagnosticLoopLimitForShader(budget, 0x4defcf80, 2), 0);
}

TEST(DiagnosticLoopPolicy, RejectsAnIncompleteSelection) {
    EXPECT_EQ(DiagnosticLoopLimitForShader(
                  DiagnosticLoopBudget{.shader_hash = 0, .loop_index = 0, .iteration_limit = 32},
                  0, 0),
              0);
}

TEST(DiagnosticLoopPolicy, ParsesAnExplicitEnvironmentSelection) {
    const auto budget = ParseDiagnosticLoopBudget("0x4defcf7f", "0", "32");

    EXPECT_EQ(budget.shader_hash, 0x4defcf7f);
    EXPECT_EQ(budget.loop_index, 0);
    EXPECT_EQ(budget.iteration_limit, 32);
}

TEST(DiagnosticLoopPolicy, RejectsMalformedOrPartialEnvironmentSelections) {
    EXPECT_EQ(ParseDiagnosticLoopBudget(nullptr, "0", "32"), DiagnosticLoopBudget{});
    EXPECT_EQ(ParseDiagnosticLoopBudget("0x4defcf7f", nullptr, "32"),
              DiagnosticLoopBudget{});
    EXPECT_EQ(ParseDiagnosticLoopBudget("0x4defcf7f", "0", nullptr),
              DiagnosticLoopBudget{});
    EXPECT_EQ(ParseDiagnosticLoopBudget("not-a-hash", "0", "32"),
              DiagnosticLoopBudget{});
    EXPECT_EQ(ParseDiagnosticLoopBudget("0x4defcf7f", "0", "0"), DiagnosticLoopBudget{});
}
