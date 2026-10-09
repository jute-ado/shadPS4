// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/srt_memory_sync.h"

namespace VideoCore {
namespace {

struct SyncRecorder {
    bool mapped = true;
    bool readback_succeeds = true;
    std::vector<std::string_view> operations;
    VAddr read_address{};
    u64 read_size{};
    void* read_output{};
    VAddr mapped_address{};
    u64 mapped_size{};

    bool IsGpuMapped(VAddr address, u64 size) {
        operations.emplace_back("mapped");
        mapped_address = address;
        mapped_size = size;
        return mapped;
    }

    bool ReadMemorySnapshot(VAddr address, void* output, u64 size) {
        operations.emplace_back("read_snapshot");
        read_address = address;
        read_output = output;
        read_size = size;
        std::memset(output, 0x5a, size);
        return readback_succeeds;
    }
};

TEST(SrtMemorySync, RejectsInvalidRangesBeforeMappingOrCacheQueries) {
    SyncRecorder recorder;
    constexpr VAddr MaxGpuAddress = 1ULL << 40;

    std::array<u8, 8> output{};
    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, 0, output.data(), output.size()));
    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, 0x1000, output.data(), 0));
    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, MaxGpuAddress, output.data(), 4));
    EXPECT_FALSE(
        ReadSrtMemoryRange(recorder, recorder, MaxGpuAddress - 4, output.data(), 8));
    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, 0x1000, output.data(), 1ULL << 32));

    EXPECT_TRUE(recorder.operations.empty());
}

TEST(SrtMemorySync, UnmappedRangesAreRejectedAndMappedRangesAreSnapshotted) {
    SyncRecorder recorder;
    std::array<u8, 8> output{};
    recorder.mapped = false;
    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, 0x2000, output.data(), output.size()));
    EXPECT_EQ(recorder.operations, (std::vector<std::string_view>{"mapped"}));

    recorder.operations.clear();
    recorder.mapped = true;
    EXPECT_TRUE(ReadSrtMemoryRange(recorder, recorder, 0x2000, output.data(), output.size()));
    EXPECT_EQ(recorder.operations, (std::vector<std::string_view>{"mapped", "read_snapshot"}));
    EXPECT_EQ(output, (std::array<u8, 8>{0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a}));
}

TEST(SrtMemorySync, ReadsOnlyRequestedBytesIntoPrivateSnapshot) {
    SyncRecorder recorder;
    std::array<u8, 8> output{};

    EXPECT_TRUE(ReadSrtMemoryRange(recorder, recorder, 0x242710318ULL, output.data(), output.size()));

    EXPECT_EQ(recorder.operations,
              (std::vector<std::string_view>{"mapped", "read_snapshot"}));
    EXPECT_EQ(recorder.read_address, 0x242710318ULL);
    EXPECT_EQ(recorder.read_output, output.data());
    EXPECT_EQ(recorder.read_size, output.size());
    EXPECT_EQ(recorder.mapped_address, 0x242710000ULL);
    EXPECT_EQ(recorder.mapped_size, 0x4000U);
}

TEST(SrtMemorySync, EightBytePointerRangeMayCrossATrackingPageBoundary) {
    SyncRecorder recorder;
    constexpr VAddr PointerAddress = 0x123FFFFCULL;
    std::array<u8, 8> output{};

    EXPECT_TRUE(ReadSrtMemoryRange(recorder, recorder, PointerAddress, output.data(), output.size()));

    EXPECT_EQ(recorder.operations,
              (std::vector<std::string_view>{"mapped", "read_snapshot"}));
    EXPECT_EQ(recorder.mapped_address, 0x123FC000ULL);
    EXPECT_EQ(recorder.mapped_size, 0x8000U);
    EXPECT_EQ(recorder.read_address, PointerAddress);
    EXPECT_EQ(recorder.read_size, output.size());
}

TEST(SrtMemorySync, FailedExactReadbackRefusesTheGuestRange) {
    SyncRecorder recorder;
    recorder.readback_succeeds = false;
    std::array<u8, 8> output{};

    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, 0x242710318ULL, output.data(), output.size()));
    EXPECT_EQ(recorder.operations,
              (std::vector<std::string_view>{"mapped", "read_snapshot"}));
}

TEST(SrtMemoryReadSession, ReusesOnePrivateTrackingPageWithinSingleWalkerInvocation) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> first{};
    std::array<u8, 16> second{};

    EXPECT_TRUE(session.Read(0x242710318ULL, first.data(), first.size()));
    EXPECT_TRUE(session.Read(0x242710380ULL, second.data(), second.size()));

    EXPECT_EQ(recorder.read_address, 0x242710000ULL);
    EXPECT_EQ(recorder.read_size, 16_KB);
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"),
              1);
    EXPECT_EQ(first, (std::array<u8, 8>{0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a}));
    EXPECT_EQ(second, (std::array<u8, 16>{0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
                                              0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a}));
}

TEST(SrtMemoryReadSession, CrossPageReadKeepsOneExactCoherentSnapshot) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};

    EXPECT_TRUE(session.Read(0x242713FFCUL, output.data(), output.size()));

    EXPECT_EQ(recorder.read_address, 0x242713FFCUL);
    EXPECT_EQ(recorder.read_size, output.size());
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"),
              1);
}

TEST(SrtMemoryReadSession, FailedPageSnapshotIsNotReused) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};
    recorder.readback_succeeds = false;

    EXPECT_FALSE(session.Read(0x242710318ULL, output.data(), output.size()));
    EXPECT_FALSE(session.Read(0x242710318ULL, output.data(), output.size()));

    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"),
              2);
}

} // namespace
} // namespace VideoCore
