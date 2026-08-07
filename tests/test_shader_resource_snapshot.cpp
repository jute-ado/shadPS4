// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "shader_recompiler/resource_snapshot.h"

namespace Shader {
namespace {

using ResourceTable = std::array<u32, 8>;

constexpr ResourceTable GenerationA{0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7};
constexpr ResourceTable GenerationB{0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7};

TEST(ShaderResourceSnapshot, OwnsEveryRangeBeforeReadingOneDrawGeneration) {
    ResourceTable live = GenerationA;
    std::array<std::mutex, 2> page_locks;
    std::atomic<bool> first_descriptor_read{};
    std::atomic<bool> writer_started{};
    std::atomic<bool> writer_finished{};

    const std::array ranges{
        ResourceSnapshotRange{.address = 0x2000, .size = 16},
        ResourceSnapshotRange{.address = 0x1000, .size = 16},
    };

    std::jthread writer{[&] {
        while (!first_descriptor_read.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        writer_started.store(true, std::memory_order_release);
        std::scoped_lock lock{page_locks[0], page_locks[1]};
        live = GenerationB;
        writer_finished.store(true, std::memory_order_release);
    }};

    const auto snapshot = CaptureOwnedResourceSnapshot<ResourceTable>(
        ranges,
        [&](const ResourceSnapshotRange& range) {
            page_locks[(range.address - 0x1000) / 0x1000].lock();
            return true;
        },
        [&](ResourceTable& dst) {
            std::copy_n(live.begin(), 4, dst.begin());
            first_descriptor_read.store(true, std::memory_order_release);
            while (!writer_started.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            EXPECT_FALSE(writer_finished.load(std::memory_order_acquire));
            std::copy_n(live.begin() + 4, 4, dst.begin() + 4);
        },
        [&](const ResourceSnapshotRange& range) {
            page_locks[(range.address - 0x1000) / 0x1000].unlock();
        });

    ASSERT_TRUE(snapshot.has_value());
    EXPECT_EQ(*snapshot, GenerationA);
    EXPECT_NE(*snapshot, (ResourceTable{0xa0, 0xa1, 0xa2, 0xa3, 0xb4, 0xb5, 0xb6, 0xb7}));
    writer.join();
    EXPECT_TRUE(writer_finished.load(std::memory_order_acquire));
    EXPECT_EQ(live, GenerationB);
}

TEST(ShaderResourceSnapshot, FailsClosedAndReleasesOwnedRangesWhenOneRangeCannotBeOwned) {
    const std::array ranges{
        ResourceSnapshotRange{.address = 0x1000, .size = 16},
        ResourceSnapshotRange{.address = 0x2000, .size = 16},
        ResourceSnapshotRange{.address = 0x3000, .size = 16},
    };
    std::vector<VAddr> acquired;
    std::vector<VAddr> released;
    bool captured = false;

    const auto snapshot = CaptureOwnedResourceSnapshot<ResourceTable>(
        ranges,
        [&](const ResourceSnapshotRange& range) {
            if (range.address == 0x3000) {
                return false;
            }
            acquired.push_back(range.address);
            return true;
        },
        [&](ResourceTable&) { captured = true; },
        [&](const ResourceSnapshotRange& range) { released.push_back(range.address); });

    EXPECT_FALSE(snapshot.has_value());
    EXPECT_FALSE(captured);
    EXPECT_EQ(acquired, (std::vector<VAddr>{0x1000, 0x2000}));
    EXPECT_EQ(released, (std::vector<VAddr>{0x2000, 0x1000}));
}

TEST(ShaderResourceSnapshot, NormalizesOverlappingRangesBeforeOwnership) {
    const std::array ranges{
        ResourceSnapshotRange{.address = 0x2010, .size = 0x20},
        ResourceSnapshotRange{.address = 0x1008, .size = 8},
        ResourceSnapshotRange{.address = 0x2000, .size = 0x20},
        ResourceSnapshotRange{.address = 0x1010, .size = 0x10},
    };
    std::vector<ResourceSnapshotRange> acquired;
    std::vector<ResourceSnapshotRange> released;

    const auto snapshot = CaptureOwnedResourceSnapshot<ResourceTable>(
        ranges,
        [&](const ResourceSnapshotRange& range) {
            acquired.push_back(range);
            return true;
        },
        [&](ResourceTable& dst) { dst = GenerationA; },
        [&](const ResourceSnapshotRange& range) { released.push_back(range); });

    ASSERT_TRUE(snapshot.has_value());
    EXPECT_EQ(*snapshot, GenerationA);
    ASSERT_EQ(acquired.size(), 2);
    EXPECT_EQ(acquired[0], (ResourceSnapshotRange{.address = 0x1008, .size = 0x18}));
    EXPECT_EQ(acquired[1], (ResourceSnapshotRange{.address = 0x2000, .size = 0x30}));
    EXPECT_EQ(released, (std::vector<ResourceSnapshotRange>{acquired[1], acquired[0]}));
}

TEST(ShaderResourceSnapshot, BoundsGenerationMismatchDiagnosticsWithoutHidingOccurrences) {
    ResourceGenerationMismatchCounter counter{2};
    const ResourceTable torn{0xa0, 0xa1, 0xa2, 0xa3, 0xb4, 0xb5, 0xb6, 0xb7};

    const auto stable = counter.Observe(GenerationA, GenerationA);
    EXPECT_FALSE(stable.mismatched);
    EXPECT_FALSE(stable.should_report);
    EXPECT_EQ(stable.occurrence, 0);

    const auto first = counter.Observe(GenerationA, torn);
    EXPECT_TRUE(first.mismatched);
    EXPECT_TRUE(first.should_report);
    EXPECT_EQ(first.occurrence, 1);

    const auto second = counter.Observe(torn, GenerationB);
    EXPECT_TRUE(second.mismatched);
    EXPECT_TRUE(second.should_report);
    EXPECT_EQ(second.occurrence, 2);

    const auto third = counter.Observe(GenerationA, GenerationB);
    EXPECT_TRUE(third.mismatched);
    EXPECT_FALSE(third.should_report);
    EXPECT_EQ(third.occurrence, 3);
    EXPECT_EQ(counter.Total(), 3);
}

} // namespace
} // namespace Shader
