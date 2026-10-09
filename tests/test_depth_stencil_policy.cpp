// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/depth_stencil_policy.h"

namespace {

using AmdGpu::StencilFunc;
using VideoCore::DepthStencilPolicy::NeedsStencilWriteAccess;
using VideoCore::DepthStencilPolicy::SelectStencilReference;
using VideoCore::DepthStencilPolicy::SelectTargetLayout;
using VideoCore::DepthStencilPolicy::TargetLayout;
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

TEST(DepthStencilPolicy, KeepOnlyStencilTestDoesNotRequireWritableLayout) {
    AmdGpu::StencilControl operations{};
    AmdGpu::StencilRefMask front{.stencil_write_mask = 0xff};
    AmdGpu::StencilRefMask back{.stencil_write_mask = 0xff};

    EXPECT_FALSE(NeedsStencilWriteAccess(true, false, true, operations, front, back));
}

TEST(DepthStencilPolicy, WritableOperationRequiresEnabledFaceAndNonzeroMask) {
    AmdGpu::StencilControl operations{};
    operations.stencil_zpass_front = StencilFunc::ReplaceOp;
    operations.stencil_zpass_back = StencilFunc::Zero;
    AmdGpu::StencilRefMask front{.stencil_write_mask = 0xff};
    AmdGpu::StencilRefMask back{.stencil_write_mask = 0xff};

    EXPECT_TRUE(NeedsStencilWriteAccess(true, false, false, operations, front, back));

    front.stencil_write_mask = 0;
    EXPECT_FALSE(NeedsStencilWriteAccess(true, false, false, operations, front, back));
    EXPECT_TRUE(NeedsStencilWriteAccess(true, false, true, operations, front, back));
    EXPECT_FALSE(NeedsStencilWriteAccess(false, false, true, operations, front, back));
}

TEST(DepthStencilPolicy, StencilClearAlwaysRequiresWritableLayout) {
    EXPECT_TRUE(NeedsStencilWriteAccess(false, true, false, {}, {}, {}));
}

TEST(DepthStencilPolicy, SelectsLayoutsForIndependentDepthAndStencilAccess) {
    EXPECT_EQ(SelectTargetLayout(true, false, false), TargetLayout::DepthAttachment);
    EXPECT_EQ(SelectTargetLayout(true, true, false), TargetLayout::DepthStencilAttachment);
    EXPECT_EQ(SelectTargetLayout(false, true, true), TargetLayout::DepthReadStencilAttachment);
    EXPECT_EQ(SelectTargetLayout(false, true, false), TargetLayout::DepthStencilReadOnly);
    EXPECT_EQ(SelectTargetLayout(false, false, false), TargetLayout::DepthReadOnly);
}

} // namespace
