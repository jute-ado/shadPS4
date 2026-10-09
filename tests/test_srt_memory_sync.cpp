// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/srt_memory_sync.h"

namespace VideoCore {
namespace {

struct SyncRecorder {
    bool mapped = true;
    bool dirty = true;
    bool readback_succeeds = true;
    std::vector<std::string_view> operations;
    VAddr read_address{};
    u64 read_size{};
    VAddr mapped_address{};
    u64 mapped_size{};

    bool IsGpuMapped(VAddr address, u64 size) {
        operations.emplace_back("mapped");
        mapped_address = address;
        mapped_size = size;
        return mapped;
    }

    bool IsRegionGpuModified(VAddr, u64) {
        operations.emplace_back("dirty");
        return dirty;
    }

    bool ReadMemoryExact(VAddr address, u64 size) {
        operations.emplace_back("read_exact");
        read_address = address;
        read_size = size;
        return readback_succeeds;
    }
};

TEST(SrtMemorySync, RejectsInvalidRangesBeforeMappingOrCacheQueries) {
    SyncRecorder recorder;
    constexpr VAddr MaxGpuAddress = 1ULL << 40;

    EXPECT_FALSE(SynchronizeSrtMemoryRange(recorder, recorder, 0, 8));
    EXPECT_FALSE(SynchronizeSrtMemoryRange(recorder, recorder, 0x1000, 0));
    EXPECT_FALSE(SynchronizeSrtMemoryRange(recorder, recorder, MaxGpuAddress, 4));
    EXPECT_FALSE(
        SynchronizeSrtMemoryRange(recorder, recorder, MaxGpuAddress - 4, 8));
    EXPECT_FALSE(SynchronizeSrtMemoryRange(recorder, recorder, 0x1000, 1ULL << 32));

    EXPECT_TRUE(recorder.operations.empty());
}

TEST(SrtMemorySync, UnmappedAndCleanRangesDoNotReadBack) {
    SyncRecorder recorder;
    recorder.mapped = false;
    EXPECT_FALSE(SynchronizeSrtMemoryRange(recorder, recorder, 0x2000, 8));
    EXPECT_EQ(recorder.operations, (std::vector<std::string_view>{"mapped"}));

    recorder.operations.clear();
    recorder.mapped = true;
    recorder.dirty = false;
    EXPECT_TRUE(SynchronizeSrtMemoryRange(recorder, recorder, 0x2000, 8));
    EXPECT_EQ(recorder.operations,
              (std::vector<std::string_view>{"mapped", "dirty"}));
}

TEST(SrtMemorySync, DirtyMappedRangeReadsBackInValidationOrder) {
    SyncRecorder recorder;

    EXPECT_TRUE(SynchronizeSrtMemoryRange(recorder, recorder, 0x242710318ULL, 8));

    EXPECT_EQ(recorder.operations,
              (std::vector<std::string_view>{"mapped", "dirty", "read_exact"}));
    EXPECT_EQ(recorder.read_address, 0x242710318ULL);
    EXPECT_EQ(recorder.read_size, 8U);
    EXPECT_EQ(recorder.mapped_address, 0x242710000ULL);
    EXPECT_EQ(recorder.mapped_size, 0x4000U);
}

TEST(SrtMemorySync, EightBytePointerRangeMayCrossATrackingPageBoundary) {
    SyncRecorder recorder;
    constexpr VAddr PointerAddress = 0x123FFFFCULL;

    EXPECT_TRUE(SynchronizeSrtMemoryRange(recorder, recorder, PointerAddress, 8));

    EXPECT_EQ(recorder.operations,
              (std::vector<std::string_view>{"mapped", "dirty", "read_exact"}));
    EXPECT_EQ(recorder.mapped_address, 0x123FC000ULL);
    EXPECT_EQ(recorder.mapped_size, 0x8000U);
    EXPECT_EQ(recorder.read_address, PointerAddress);
    EXPECT_EQ(recorder.read_size, 8U);
}

TEST(SrtMemorySync, FailedExactReadbackRefusesTheGuestRange) {
    SyncRecorder recorder;
    recorder.readback_succeeds = false;

    EXPECT_FALSE(SynchronizeSrtMemoryRange(recorder, recorder, 0x242710318ULL, 8));
    EXPECT_EQ(recorder.operations,
              (std::vector<std::string_view>{"mapped", "dirty", "read_exact"}));
}

} // namespace
} // namespace VideoCore
