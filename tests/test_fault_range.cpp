// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <vector>

#include "video_core/buffer_cache/dma_dirty_ranges.h"
#include "video_core/buffer_cache/fault_range.h"

namespace {

enum class FaultRangeCacheCall {
    TransitionAuthoritativeTexture,
    Find,
    Synchronize,
};

struct RecordedFaultRangeCacheCall {
    FaultRangeCacheCall operation;
    VAddr address;
    u64 size;
};

class RecordingFaultRangeCache {
public:
    bool TransitionAuthoritativeTextureForDmaRead(VAddr address, u32 size) {
        calls.push_back({FaultRangeCacheCall::TransitionAuthoritativeTexture, address, size});
        return transition_succeeds;
    }

    void FindBuffer(VAddr address, u32 size) {
        calls.push_back({FaultRangeCacheCall::Find, address, size});
    }

    void SynchronizeBuffersInRange(VAddr address, u64 size) {
        calls.push_back({FaultRangeCacheCall::Synchronize, address, size});
    }

    bool transition_succeeds{true};
    std::vector<RecordedFaultRangeCacheCall> calls;
};

} // namespace

TEST(FaultRange, AcceptsMappedAddressSpaceRange) {
    EXPECT_TRUE(VideoCore::IsCacheableFaultRange(0x4000, 0x8000, 1ULL << 40));
}

TEST(FaultRange, RejectsNullPage) {
    EXPECT_FALSE(VideoCore::IsCacheableFaultRange(0, 0x4000, 1ULL << 40));
}

TEST(FaultRange, RejectsRangeOutsideAddressSpace) {
    EXPECT_FALSE(VideoCore::IsCacheableFaultRange(1ULL << 40, (1ULL << 40) + 0x4000, 1ULL << 40));
}

TEST(FaultRange, RejectsEmptyOrReversedRange) {
    EXPECT_FALSE(VideoCore::IsCacheableFaultRange(0x4000, 0x4000, 1ULL << 40));
    EXPECT_FALSE(VideoCore::IsCacheableFaultRange(0x8000, 0x4000, 1ULL << 40));
}

TEST(FaultRange, RejectsRangeTooLargeForBufferCache) {
    EXPECT_FALSE(VideoCore::IsCacheableFaultRange(0x4000, 0x4000 + (1ULL << 32), 1ULL << 40));
}

TEST(FaultRange, RejectsUnmappedGpuFaultRange) {
    bool mapping_checked = false;
    const auto is_mapped = [&mapping_checked](VAddr address, u64 size) {
        mapping_checked = true;
        EXPECT_EQ(address, 0x9C000);
        EXPECT_EQ(size, 0x4000);
        return false;
    };

    EXPECT_FALSE(VideoCore::IsProcessableDmaFaultRange(0x9C000, 0xA0000, 1ULL << 40, is_mapped));
    EXPECT_TRUE(mapping_checked);
}

TEST(FaultRange, AcceptsMappedGpuFaultRange) {
    const auto is_mapped = [](VAddr address, u64 size) {
        return address == 0x101E600000 && size == 0x4000;
    };

    EXPECT_TRUE(VideoCore::IsProcessableDmaFaultRange(0x101E600000, 0x101E604000,
                                                      1ULL << 40, is_mapped));
}

TEST(FaultRange, MakesDmaFaultRangeResidentBeforeDirectAccess) {
    RecordingFaultRangeCache cache;

    EXPECT_TRUE(VideoCore::MakeDmaFaultRangeResident(cache, 0x101E600000, 0x4000));

    ASSERT_EQ(cache.calls.size(), 3);
    EXPECT_EQ(cache.calls[0].operation, FaultRangeCacheCall::TransitionAuthoritativeTexture);
    EXPECT_EQ(cache.calls[0].address, 0x101E600000);
    EXPECT_EQ(cache.calls[0].size, 0x4000);
    EXPECT_EQ(cache.calls[1].operation, FaultRangeCacheCall::Find);
    EXPECT_EQ(cache.calls[1].address, 0x101E600000);
    EXPECT_EQ(cache.calls[1].size, 0x4000);
    EXPECT_EQ(cache.calls[2].operation, FaultRangeCacheCall::Synchronize);
    EXPECT_EQ(cache.calls[2].address, 0x101E600000);
    EXPECT_EQ(cache.calls[2].size, 0x4000);
}

TEST(FaultRange, RejectsDmaFaultWhenAuthoritativeTextureCannotTransition) {
    RecordingFaultRangeCache cache;
    cache.transition_succeeds = false;

    EXPECT_FALSE(VideoCore::MakeDmaFaultRangeResident(cache, 0x101E600000, 0x4000));

    ASSERT_EQ(cache.calls.size(), 1);
    EXPECT_EQ(cache.calls[0].operation, FaultRangeCacheCall::TransitionAuthoritativeTexture);
}

TEST(DmaDirtyRanges, CleanFramesProduceNoSynchronizationWork) {
    VideoCore::DmaDirtyRangeTracker tracker;

    for (int frame = 0; frame < 1'000; ++frame) {
        EXPECT_TRUE(tracker.Take().empty());
    }
}

TEST(DmaDirtyRanges, MergesOverlappingCpuWritesAndDrainsThemOnce) {
    VideoCore::DmaDirtyRangeTracker tracker;
    tracker.Mark(0x10'0000, 0x4000);
    tracker.Mark(0x10'2000, 0x6000);

    const auto ranges = tracker.Take();

    ASSERT_EQ(ranges.size(), 1);
    EXPECT_EQ(ranges[0].address, 0x10'0000);
    EXPECT_EQ(ranges[0].size, 0x8000);
    EXPECT_TRUE(tracker.Take().empty());
}

TEST(DmaDirtyRanges, CoalescesDisjointWritesWithinOneTrackedPage) {
    VideoCore::DmaDirtyRangeTracker tracker;
    tracker.Mark(0x10'0008, 8);
    tracker.Mark(0x10'0ff0, 8);

    const auto ranges = tracker.Take();

    ASSERT_EQ(ranges.size(), 1);
    EXPECT_EQ(ranges[0].address, 0x10'0000);
    EXPECT_EQ(ranges[0].size, 0x1000);
}

TEST(DmaDirtyRanges, RetainsWritesMarkedAfterAPreviousDrain) {
    VideoCore::DmaDirtyRangeTracker tracker;
    tracker.Mark(0x20'0000, 0x4000);
    ASSERT_EQ(tracker.Take().size(), 1);

    tracker.Mark(0x30'0000, 0x8000);
    const auto ranges = tracker.Take();

    ASSERT_EQ(ranges.size(), 1);
    EXPECT_EQ(ranges[0].address, 0x30'0000);
    EXPECT_EQ(ranges[0].size, 0x8000);
}
