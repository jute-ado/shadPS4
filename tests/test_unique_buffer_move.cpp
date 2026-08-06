// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>
#include <utility>

#include <gtest/gtest.h>

#include "common/slot_vector.h"
#include "video_core/buffer_cache/buffer.h"

namespace VideoCore {

// These tests exercise the production inline move operations without constructing Vulkan/VMA
// resources. Real buffers are covered by the emulator app; the synthetic invariant here is that
// every piece of ownership metadata survives container moves.
UniqueBuffer::UniqueBuffer(vk::Device device_, VmaAllocator allocator_)
    : device{device_}, allocator{allocator_}, allocation{VK_NULL_HANDLE} {}

UniqueBuffer::~UniqueBuffer() = default;

TEST(UniqueBufferMove, DeviceAddressSurvivesMoveConstruction) {
    constexpr vk::DeviceAddress DeviceAddress = 0x1234'5678'9ABC'DEF0ULL;
    UniqueBuffer source{vk::Device{}, VK_NULL_HANDLE};
    source.bda_addr = DeviceAddress;

    UniqueBuffer destination{std::move(source)};

    EXPECT_EQ(destination.bda_addr, DeviceAddress);
    EXPECT_EQ(source.bda_addr, 0);
}

TEST(UniqueBufferMove, AllocationAndDeviceAddressSurviveMoveAssignment) {
    constexpr vk::DeviceAddress DeviceAddress = 0x0FED'CBA9'8765'4321ULL;
    const auto source_allocation = reinterpret_cast<VmaAllocation>(std::uintptr_t{0x1234});
    const auto stale_destination_allocation =
        reinterpret_cast<VmaAllocation>(std::uintptr_t{0x5678});
    UniqueBuffer source{vk::Device{}, VK_NULL_HANDLE};
    source.allocation = source_allocation;
    source.bda_addr = DeviceAddress;
    UniqueBuffer destination{vk::Device{}, VK_NULL_HANDLE};
    destination.allocation = stale_destination_allocation;

    destination = std::move(source);

    EXPECT_EQ(destination.allocation, source_allocation);
    EXPECT_EQ(destination.bda_addr, DeviceAddress);
    EXPECT_EQ(source.allocation, VK_NULL_HANDLE);
    EXPECT_EQ(source.bda_addr, 0);

    // No Vulkan buffer handles were installed, so the synthetic allocations are never freed.
    destination.allocation = VK_NULL_HANDLE;
    source.allocation = VK_NULL_HANDLE;
}

TEST(UniqueBufferMove, DeviceAddressSurvivesSlotVectorGrowth) {
    constexpr vk::DeviceAddress DeviceAddress = 0x1122'3344'5566'7788ULL;
    Common::SlotVector<UniqueBuffer> buffers;
    const auto original = buffers.insert(vk::Device{}, VK_NULL_HANDLE);
    buffers[original].bda_addr = DeviceAddress;

    // SlotVector starts with 2048 entries. The final insertion relocates every live buffer.
    for (std::size_t i = 1; i <= 2048; ++i) {
        static_cast<void>(buffers.insert(vk::Device{}, VK_NULL_HANDLE));
    }

    EXPECT_EQ(buffers[original].bda_addr, DeviceAddress);
}

} // namespace VideoCore
