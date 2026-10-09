// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/ir/passes/dynamic_image_table_policy.h"
#include "shader_recompiler/resource.h"

namespace Shader::Optimization {
namespace {

constexpr DynamicImageTablePattern ValidPattern(DynamicImageAccess access, bool is_r128 = true) {
    return {
        .base_pointer_location = 18,
        .byte_offset = 14480,
        .byte_stride = 32,
        .is_r128 = is_r128,
        .access = access,
        .index_upper_bound = 5,
        .supported_descriptor_window = NUM_IMAGES,
        .supports_nonuniform_indexing = true,
        .supports_oob_guard = true,
    };
}

TEST(DynamicImageTablePolicy, AcceptsSampledExplicitLodTables) {
    const auto r128_plan =
        PlanDynamicImageTable(ValidPattern(DynamicImageAccess::SampledExplicitLod));
    ASSERT_TRUE(r128_plan.has_value());
    EXPECT_EQ(r128_plan->base_pointer_location, 18u);
    EXPECT_EQ(r128_plan->offset_dw, 3620u);
    EXPECT_EQ(r128_plan->stride_dw, 8u);
    EXPECT_EQ(r128_plan->descriptor_count, 6u);
    EXPECT_FALSE(r128_plan->requires_oob_guard);

    const auto full_plan =
        PlanDynamicImageTable(ValidPattern(DynamicImageAccess::SampledExplicitLod, false));
    ASSERT_TRUE(full_plan.has_value());
    EXPECT_EQ(full_plan->stride_dw, 8u);
}

TEST(DynamicImageTablePolicy, DeclinesFeatureOffWithoutGuessingABinding) {
    auto pattern = ValidPattern(DynamicImageAccess::SampledExplicitLod);
    pattern.supports_nonuniform_indexing = false;

    EXPECT_FALSE(PlanDynamicImageTable(pattern).has_value());
}

TEST(DynamicImageTablePolicy, PlansGuardedWindowForTheRealUnboundedAffineTargetShape) {
    auto pattern = ValidPattern(DynamicImageAccess::SampledExplicitLod);
    pattern.index_upper_bound.reset();

    const auto plan = PlanDynamicImageTable(pattern);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->base_pointer_location, 18u);
    EXPECT_EQ(plan->offset_dw, 3620u);
    EXPECT_EQ(plan->stride_dw, 8u);
    EXPECT_EQ(plan->descriptor_count, NUM_IMAGES);
    EXPECT_TRUE(plan->requires_oob_guard);
}

TEST(DynamicImageTablePolicy, DeclinesUnboundedIndexWhenBackendCannotEmitTheGuard) {
    auto pattern = ValidPattern(DynamicImageAccess::SampledExplicitLod);
    pattern.index_upper_bound.reset();
    pattern.supports_oob_guard = false;

    EXPECT_FALSE(PlanDynamicImageTable(pattern).has_value());
}

TEST(DynamicImageTablePolicy, DeclinesStorageAtomicAndImplicitLodAccess) {
    EXPECT_FALSE(
        PlanDynamicImageTable(ValidPattern(DynamicImageAccess::StorageRead)).has_value());
    EXPECT_FALSE(
        PlanDynamicImageTable(ValidPattern(DynamicImageAccess::StorageWrite)).has_value());
    EXPECT_FALSE(PlanDynamicImageTable(ValidPattern(DynamicImageAccess::Atomic)).has_value());
    EXPECT_FALSE(
        PlanDynamicImageTable(ValidPattern(DynamicImageAccess::SampledImplicitLod)).has_value());
}

TEST(DynamicImageTablePolicy, DeclinesMisalignedOrUndersizedDescriptors) {
    auto misaligned_offset = ValidPattern(DynamicImageAccess::SampledExplicitLod);
    misaligned_offset.byte_offset++;
    EXPECT_FALSE(PlanDynamicImageTable(misaligned_offset).has_value());

    auto misaligned_stride = ValidPattern(DynamicImageAccess::SampledExplicitLod);
    misaligned_stride.byte_stride = 30;
    EXPECT_FALSE(PlanDynamicImageTable(misaligned_stride).has_value());

    auto undersized_full = ValidPattern(DynamicImageAccess::SampledExplicitLod, false);
    undersized_full.byte_stride = 16;
    EXPECT_FALSE(PlanDynamicImageTable(undersized_full).has_value());
}

TEST(DynamicImageTablePolicy, DeclinesNonPowerOfTwoStride) {
    auto pattern = ValidPattern(DynamicImageAccess::SampledExplicitLod);
    pattern.byte_stride = 48;

    EXPECT_FALSE(PlanDynamicImageTable(pattern).has_value());
}

TEST(DynamicImageTablePolicy, GuardsBoundsBeyondTheSupportedBackendWindow) {
    auto pattern = ValidPattern(DynamicImageAccess::SampledExplicitLod);
    pattern.supported_descriptor_window = 23;
    pattern.index_upper_bound = 30;

    const auto plan = PlanDynamicImageTable(pattern);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->descriptor_count, 23u);
    EXPECT_TRUE(plan->requires_oob_guard);
}

TEST(DynamicImageTablePolicy, DeclinesMissingBackendWindow) {
    auto pattern = ValidPattern(DynamicImageAccess::SampledExplicitLod);
    pattern.supported_descriptor_window = 0;

    EXPECT_FALSE(PlanDynamicImageTable(pattern).has_value());
}

} // namespace
} // namespace Shader::Optimization
