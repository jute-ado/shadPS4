// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>

#include "shader_recompiler/buffer_access_range.h"

TEST(BufferAccessRange, NarrowsDeclaredRangeToStaticAccessPrefix) {
    Shader::BufferAccessRange range;
    range.Add(12, 16);
    range.Add(3, 4);
    range.Add(8, 16);

    EXPECT_TRUE(range.IsBounded());
    EXPECT_EQ(range.UpperBound(), 28);
    EXPECT_EQ(range.Fit(0xfffffffcULL), 28);
}

TEST(BufferAccessRange, PreservesSmallerDescriptorBound) {
    Shader::BufferAccessRange range;
    range.Add(12, 16);

    EXPECT_EQ(range.Fit(16), 16);
}

TEST(BufferAccessRange, DynamicAccessDisablesNarrowing) {
    Shader::BufferAccessRange range;
    range.Add(0, 16);
    range.MarkDynamic();
    range.Add(64, 4);

    EXPECT_FALSE(range.IsBounded());
    EXPECT_EQ(range.Fit(0xfffffffcULL), 0xfffffffcULL);
}

TEST(BufferAccessRange, OverflowDisablesNarrowing) {
    Shader::BufferAccessRange range;
    range.Add(std::numeric_limits<u64>::max() - 3, 8);

    EXPECT_FALSE(range.IsBounded());
    EXPECT_EQ(range.Fit(4_GB), 4_GB);
}

TEST(BufferAccessRange, EmptyAnalysisPreservesDeclaredRange) {
    const Shader::BufferAccessRange range;

    EXPECT_TRUE(range.IsBounded());
    EXPECT_EQ(range.UpperBound(), 0);
    EXPECT_EQ(range.Fit(4_GB), 4_GB);
}

TEST(BufferAccessRange, NarrowsOnlyMatchingAddressLayouts) {
    Shader::BufferAccessRange range;
    const Shader::BufferAddressLayout compiled_layout{
        .stride = 4,
        .data_format = 4,
        .element_size = 1,
        .index_stride = 2,
        .swizzle_enable = false,
        .add_tid_enable = false,
    };
    range.RecordAddressLayout(compiled_layout);
    range.Add(48, 16);

    constexpr u64 declared_size = 0xfffffffcULL;
    EXPECT_EQ(range.Fit(declared_size, compiled_layout), 64);

    auto changed_stride = compiled_layout;
    changed_stride.stride = 8;
    EXPECT_EQ(range.Fit(declared_size, changed_stride), declared_size);

    auto changed_swizzle = compiled_layout;
    changed_swizzle.swizzle_enable = true;
    EXPECT_EQ(range.Fit(declared_size, changed_swizzle), declared_size);
}

