// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "common/slot_vector.h"
#include "video_core/buffer_cache/buffer.h"

#include <vk_mem_alloc.h>

namespace {

struct DestroyedAllocation {
    VmaAllocator allocator;
    VkBuffer buffer;
    VmaAllocation allocation;

    bool operator==(const DestroyedAllocation&) const = default;
};

std::vector<DestroyedAllocation> destroyed;

template <typename T>
T Handle(std::uintptr_t value) {
    if constexpr (std::is_pointer_v<T>) {
        return reinterpret_cast<T>(value);
    } else {
        return static_cast<T>(value);
    }
}

class UniqueBufferOwnership : public ::testing::Test {
protected:
    void SetUp() override {
        destroyed.clear();
    }

    static void Populate(VideoCore::UniqueBuffer& value, std::uintptr_t id) {
        value.device = vk::Device{Handle<VkDevice>(id + 0x100)};
        value.allocator = Handle<VmaAllocator>(id + 0x200);
        value.allocation = Handle<VmaAllocation>(id + 0x300);
        value.buffer = vk::Buffer{Handle<VkBuffer>(id + 0x400)};
        value.bda_addr = id + 0x500;
    }

    static DestroyedAllocation Expected(std::uintptr_t id) {
        return {Handle<VmaAllocator>(id + 0x200), Handle<VkBuffer>(id + 0x400),
                Handle<VmaAllocation>(id + 0x300)};
    }

    static void ExpectIdentity(const VideoCore::UniqueBuffer& value, std::uintptr_t id) {
        EXPECT_EQ(value.device, vk::Device{Handle<VkDevice>(id + 0x100)});
        EXPECT_EQ(value.allocator, Handle<VmaAllocator>(id + 0x200));
        EXPECT_EQ(value.allocation, Handle<VmaAllocation>(id + 0x300));
        EXPECT_EQ(value.buffer, vk::Buffer{Handle<VkBuffer>(id + 0x400)});
        EXPECT_EQ(value.bda_addr, id + 0x500);
    }

    static void ExpectMovedFrom(const VideoCore::UniqueBuffer& value) {
        EXPECT_FALSE(value.buffer);
        EXPECT_EQ(value.allocator, VK_NULL_HANDLE);
        EXPECT_EQ(value.allocation, VK_NULL_HANDLE);
        EXPECT_EQ(value.bda_addr, 0);
    }
};

TEST_F(UniqueBufferOwnership, MoveConstructionTransfersDeviceAndAddressWithAllocation) {
    {
        VideoCore::UniqueBuffer source{vk::Device{}, VK_NULL_HANDLE};
        Populate(source, 1);
        VideoCore::UniqueBuffer destination{std::move(source)};
        ExpectIdentity(destination, 1);
        ExpectMovedFrom(source);
        EXPECT_TRUE(destroyed.empty());
    }
    EXPECT_EQ(destroyed, (std::vector{Expected(1)}));
}

TEST_F(UniqueBufferOwnership, MoveAssignmentReleasesOldAllocationBeforeTakingNewOwnership) {
    {
        VideoCore::UniqueBuffer source{vk::Device{}, VK_NULL_HANDLE};
        VideoCore::UniqueBuffer destination{vk::Device{}, VK_NULL_HANDLE};
        Populate(source, 1);
        Populate(destination, 2);
        destination = std::move(source);
        EXPECT_EQ(destroyed, (std::vector{Expected(2)}));
        ExpectIdentity(destination, 1);
        ExpectMovedFrom(source);
    }
    EXPECT_EQ(destroyed, (std::vector{Expected(2), Expected(1)}));
}

TEST_F(UniqueBufferOwnership, SelfMovePreservesLiveAllocation) {
    {
        VideoCore::UniqueBuffer value{vk::Device{}, VK_NULL_HANDLE};
        Populate(value, 1);
        auto& alias = value;
        value = std::move(alias);
        ExpectIdentity(value, 1);
        EXPECT_TRUE(destroyed.empty());
    }
    EXPECT_EQ(destroyed, (std::vector{Expected(1)}));
}

TEST_F(UniqueBufferOwnership, RepeatedMovesDestroyExactlyOnce) {
    {
        VideoCore::UniqueBuffer first{vk::Device{}, VK_NULL_HANDLE};
        Populate(first, 1);
        VideoCore::UniqueBuffer second{std::move(first)};
        VideoCore::UniqueBuffer third{std::move(second)};
        EXPECT_EQ(third.buffer, vk::Buffer{Handle<VkBuffer>(0x401)});
        EXPECT_TRUE(destroyed.empty());
    }
    EXPECT_EQ(destroyed, (std::vector{Expected(1)}));
}

TEST_F(UniqueBufferOwnership, NewlyConstructedEmptyBufferHasNoAllocationOrAddress) {
    {
        VideoCore::UniqueBuffer empty{vk::Device{}, VK_NULL_HANDLE};
        ExpectMovedFrom(empty);
        VideoCore::UniqueBuffer moved{std::move(empty)};
        ExpectMovedFrom(moved);
        ExpectMovedFrom(empty);
    }
    EXPECT_TRUE(destroyed.empty());
}

TEST_F(UniqueBufferOwnership, MovingAnExplicitlyEmptyAllocationReleasesDestination) {
    {
        VideoCore::UniqueBuffer empty{vk::Device{}, VK_NULL_HANDLE};
        // The old constructor did not initialize allocation. Set it explicitly so the RED
        // move test is deterministic and never reads indeterminate storage.
        empty.allocation = VK_NULL_HANDLE;
        VideoCore::UniqueBuffer destination{vk::Device{}, VK_NULL_HANDLE};
        Populate(destination, 2);
        destination = std::move(empty);
        EXPECT_EQ(destroyed, (std::vector{Expected(2)}));
        ExpectMovedFrom(destination);
    }
    EXPECT_EQ(destroyed, (std::vector{Expected(2)}));
}

TEST_F(UniqueBufferOwnership, SlotVectorGrowthPreservesExistingGpuAddresses) {
    {
        Common::SlotVector<VideoCore::UniqueBuffer> values;
        for (std::uintptr_t i = 1; i <= 2049; ++i) {
            const auto id = values.insert(vk::Device{}, VK_NULL_HANDLE);
            Populate(values[id], i);
        }
        // Inserting the 2049th buffer exercises the real production container's relocation.
        ExpectIdentity(values[Common::SlotId{0}], 1);
        ExpectIdentity(values[Common::SlotId{2047}], 2048);
        ExpectIdentity(values[Common::SlotId{2048}], 2049);
        EXPECT_TRUE(destroyed.empty());
    }
    ASSERT_EQ(destroyed.size(), 2049);
    for (std::uintptr_t i = 1; i <= 2049; ++i) {
        EXPECT_EQ(destroyed[i - 1], Expected(i));
    }
}

} // namespace

// The only substituted operation is the external destruction API. Constructors, moves,
// destructors and SlotVector relocation above are the production implementations.
extern "C" void vmaDestroyBuffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation) {
    destroyed.push_back({allocator, buffer, allocation});
}
