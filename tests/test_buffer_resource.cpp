// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <limits>
#include <type_traits>
#include <gtest/gtest.h>
#include "video_core/amdgpu/resource.h"
#include "video_core/buffer_cache/buffer_cache_policy.h"

TEST(BufferCachePolicy, StreamsSmallReadOnlyBuffersWithoutGpuWrites) {
    constexpr u64 MaxStreamSize = 64_KB;

    EXPECT_TRUE(VideoCore::ShouldUseStreamBuffer(false, MaxStreamSize, MaxStreamSize, false));
    EXPECT_FALSE(VideoCore::ShouldUseStreamBuffer(true, MaxStreamSize, MaxStreamSize, false));
    EXPECT_FALSE(VideoCore::ShouldUseStreamBuffer(false, MaxStreamSize + 1, MaxStreamSize, false));
    EXPECT_FALSE(VideoCore::ShouldUseStreamBuffer(false, MaxStreamSize, MaxStreamSize, true));
}

TEST(BufferResource, IgnoresAddressBitsOutsideTheGpuAddressSpace) {
    constexpr u64 Address = 0x12'3456'789a;
    constexpr std::array<u64, 2> RawDescriptor{Address | (0xfull << 40), 0};

    const auto buffer = std::bit_cast<AmdGpu::Buffer>(RawDescriptor);

    EXPECT_EQ(buffer.base_address, Address);
}

TEST(BufferResource, StructuredSizeDoesNotWrapAtFourGiB) {
    AmdGpu::Buffer buffer{};
    buffer.stride = 4096;
    buffer.num_records = 2 * 1024 * 1024;

    static_assert(std::is_same_v<decltype(buffer.GetSize()), u64>);
    EXPECT_EQ(buffer.GetSize(), 8_GB);
}

TEST(BufferResource, MaximumStructuredSizeUsesFullDescriptorRange) {
    AmdGpu::Buffer buffer{};
    buffer.stride = (1u << 14) - 1;
    buffer.num_records = std::numeric_limits<u32>::max();

    EXPECT_EQ(buffer.GetSize(), u64{buffer.stride} * buffer.num_records);
    EXPECT_EQ(buffer.NumDwords(), Common::AlignUp(buffer.GetSize(), sizeof(u32)) / sizeof(u32));
}

TEST(BufferResource, RawSizeRemainsTheRecordByteCount) {
    AmdGpu::Buffer buffer{};
    buffer.stride = 0;
    buffer.num_records = std::numeric_limits<u32>::max();

    EXPECT_EQ(buffer.GetSize(), std::numeric_limits<u32>::max());
}
