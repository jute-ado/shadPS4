// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

#include "video_core/amdgpu/resource.h"
#include "video_core/texture_cache/types.h"

namespace {

using VideoCore::SubresourceExtent;

TEST(BufferResource, StructuredSizeDoesNotWrapAtFourGiB) {
    AmdGpu::Buffer buffer{};
    buffer.stride = 4096;
    buffer.num_records = 2 * 1024 * 1024;

    static_assert(std::is_same_v<decltype(buffer.GetSize()), u64>);
    EXPECT_EQ(buffer.GetSize(), 8_GB);
    EXPECT_EQ(buffer.NumDwords(), 2_GB);
}

TEST(BufferResource, MaximumStructuredSizeUsesTheFullDescriptorRange) {
    AmdGpu::Buffer buffer{};
    buffer.stride = (1u << 14) - 1;
    buffer.num_records = std::numeric_limits<u32>::max();

    const u64 expected = u64{buffer.stride} * buffer.num_records;
    EXPECT_EQ(buffer.GetSize(), expected);
    EXPECT_EQ(buffer.NumDwords(), Common::AlignUp(expected, sizeof(u32)) / sizeof(u32));
}

TEST(SubresourceExtent, ContainsEqualExtent) {
    constexpr SubresourceExtent available{.levels = 3, .layers = 6};

    EXPECT_TRUE(available.CanContain(SubresourceExtent{.levels = 3, .layers = 6}));
}

TEST(SubresourceExtent, ContainsSmallerExtentInBothDimensions) {
    constexpr SubresourceExtent available{.levels = 3, .layers = 6};

    EXPECT_TRUE(available.CanContain(SubresourceExtent{.levels = 2, .layers = 5}));
}

TEST(SubresourceExtent, RejectsMoreMipLevelsDespiteFewerLayers) {
    constexpr SubresourceExtent available{.levels = 1, .layers = 6};

    EXPECT_FALSE(available.CanContain(SubresourceExtent{.levels = 3, .layers = 1}));
}

TEST(SubresourceExtent, RejectsMoreLayersDespiteFewerMipLevels) {
    constexpr SubresourceExtent available{.levels = 3, .layers = 1};

    EXPECT_FALSE(available.CanContain(SubresourceExtent{.levels = 1, .layers = 6}));
}

} // namespace
