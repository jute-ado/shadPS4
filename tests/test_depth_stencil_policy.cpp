// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/depth_stencil_policy.h"

namespace {

using AmdGpu::StencilFunc;
using VideoCore::DepthStencilPolicy::SelectStencilReference;
using VideoCore::DepthStencilPolicy::ShouldClearDepth;
using VideoCore::DepthStencilPolicy::UsesStencilOpValue;

TEST(DepthStencilPolicy, RegisterDepthClearRequiresEnabledDepthWrites) {
    EXPECT_TRUE(ShouldClearDepth(true, true, true, false));
    EXPECT_FALSE(ShouldClearDepth(true, false, true, false));
    EXPECT_FALSE(ShouldClearDepth(true, true, false, false));
    EXPECT_FALSE(ShouldClearDepth(false, true, true, false));
}

TEST(DepthStencilPolicy, MetadataClearRemainsAuthoritative) {
    EXPECT_TRUE(ShouldClearDepth(false, false, false, true));
    EXPECT_TRUE(ShouldClearDepth(true, false, false, true));
}

TEST(DepthStencilPolicy, AnyReplaceOperationUsesStencilOpValue) {
    EXPECT_FALSE(UsesStencilOpValue(StencilFunc::Keep, StencilFunc::Zero,
                                    StencilFunc::AddClamp));
    EXPECT_TRUE(UsesStencilOpValue(StencilFunc::ReplaceOp, StencilFunc::Keep,
                                   StencilFunc::Keep));
    EXPECT_TRUE(UsesStencilOpValue(StencilFunc::Keep, StencilFunc::ReplaceOp,
                                   StencilFunc::Keep));
    EXPECT_TRUE(UsesStencilOpValue(StencilFunc::Keep, StencilFunc::Keep,
                                   StencilFunc::ReplaceOp));
}

TEST(DepthStencilPolicy, ReplaceSelectsOperationReferenceAndOtherOpsUseTestReference) {
    constexpr u8 op_value = 0x22;
    constexpr u8 test_value = 0x33;

    EXPECT_EQ(SelectStencilReference(true, op_value, test_value), op_value);
    EXPECT_EQ(SelectStencilReference(false, op_value, test_value), test_value);
}

} // namespace
