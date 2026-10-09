// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/ir/passes/dynamic_image_table_policy.h"
#include "shader_recompiler/ir/passes/sharp_source_policy.h"

namespace Shader::Optimization {
namespace {

TEST(SharpSourcePolicy, UserDataZeroRemainsAStaticUserDataLocation) {
    const auto source = ClassifySharpSource(true, 0);

    EXPECT_EQ(source.kind, SharpSourceKind::UserData);
    EXPECT_EQ(source.location, 0u);
}

TEST(SharpSourcePolicy, FlattenedReadConstHasAStaticFlatBufferLocation) {
    const auto source = ClassifySharpSource(false, 157);

    EXPECT_EQ(source.kind, SharpSourceKind::Flattened);
    EXPECT_EQ(source.location, 157u);
}

TEST(SharpSourcePolicy, UnflattenedReadConstIsDynamicInsteadOfUserDataZero) {
    const auto source = ClassifySharpSource(false, 0);

    EXPECT_EQ(source.kind, SharpSourceKind::Dynamic);
    EXPECT_EQ(source.location, 0u);
}

TEST(DynamicImageTablePolicy, AcceptsR128AffineDescriptorTable) {
    const auto plan = PlanDynamicImageTable({
        .base_pointer_location = 18,
        .byte_offset = 14480,
        .byte_stride = 32,
        .is_r128 = true,
        .supports_nonuniform_indexing = true,
    });

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->base_pointer_location, 18u);
    EXPECT_EQ(plan->offset_dw, 3620u);
    EXPECT_EQ(plan->stride_dw, 8u);
    EXPECT_EQ(plan->descriptor_count, 64u);
    EXPECT_EQ(plan->index_shift, 3u);
}

TEST(DynamicImageTablePolicy, AcceptsFullImageDescriptorsWhenStrideContainsAllEightDwords) {
    const auto plan = PlanDynamicImageTable({
        .base_pointer_location = 18,
        .byte_offset = 14480,
        .byte_stride = 32,
        .is_r128 = false,
        .supports_nonuniform_indexing = true,
    });

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->stride_dw, 8u);
}

TEST(DynamicImageTablePolicy, RejectsDescriptorsThatDoNotFitOrLackNonuniformSupport) {
    EXPECT_FALSE(PlanDynamicImageTable({
        .base_pointer_location = 18,
        .byte_offset = 14480,
        .byte_stride = 16,
        .is_r128 = false,
        .supports_nonuniform_indexing = true,
    }));
    EXPECT_FALSE(PlanDynamicImageTable({
        .base_pointer_location = 18,
        .byte_offset = 14480,
        .byte_stride = 32,
        .is_r128 = true,
        .supports_nonuniform_indexing = false,
    }));
}

} // namespace
} // namespace Shader::Optimization
