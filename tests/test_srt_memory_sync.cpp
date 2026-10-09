// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/srt_memory_sync.h"

namespace VideoCore {
namespace {

struct SyncRecorder {
    bool mapped = true;
    bool readback_succeeds = true;
    bool invalidate_during_read = false;
    bool epoch_valid = true;
    bool epoch_sealed = false;
    u8 snapshot_byte = 0x5a;
    std::unordered_map<VAddr, u32> snapshot_watchers;
    std::vector<std::string_view> operations;
    VAddr read_address{};
    u64 read_size{};
    VAddr mapped_address{};
    u64 mapped_size{};

    class SnapshotEpoch {
    public:
        explicit SnapshotEpoch(SyncRecorder& owner_) : owner{&owner_} {}
        SnapshotEpoch(const SnapshotEpoch&) = delete;
        SnapshotEpoch& operator=(const SnapshotEpoch&) = delete;
        SnapshotEpoch(SnapshotEpoch&& other) noexcept
            : owner{std::exchange(other.owner, nullptr)} {}
        SnapshotEpoch& operator=(SnapshotEpoch&& other) noexcept {
            if (this != &other) {
                (void)Seal();
                owner = std::exchange(other.owner, nullptr);
            }
            return *this;
        }
        ~SnapshotEpoch() {
            (void)Seal();
        }

        bool BeginPageCopy(VAddr address, u64 size) {
            if (!owner || !owner->epoch_valid || owner->epoch_sealed || size != 16_KB) {
                return false;
            }
            ++owner->snapshot_watchers[address];
            return true;
        }

        bool EndPageCopy(VAddr, u64, bool copied) {
            return owner && owner->epoch_valid && !owner->epoch_sealed && copied;
        }

        [[nodiscard]] bool Valid() const {
            return owner ? owner->epoch_valid : result;
        }

        bool Seal() {
            if (!owner) {
                return result;
            }
            owner->epoch_sealed = true;
            owner->snapshot_watchers.clear();
            result = owner->epoch_valid;
            owner = nullptr;
            return result;
        }

    private:
        SyncRecorder* owner{};
        bool result{};
    };

    SnapshotEpoch BeginSnapshotEpoch() {
        epoch_valid = true;
        epoch_sealed = false;
        return SnapshotEpoch{*this};
    }

    bool IsGpuMapped(VAddr address, u64 size) {
        operations.emplace_back("mapped");
        mapped_address = address;
        mapped_size = size;
        return mapped;
    }

    bool ReadMemorySnapshot(VAddr address, void* output, u64 size) {
        operations.emplace_back("read_snapshot");
        read_address = address;
        read_size = size;
        if (!readback_succeeds) {
            return false;
        }
        std::memset(output, snapshot_byte, size);
        if (invalidate_during_read) {
            epoch_valid = false;
        }
        return true;
    }
};

TEST(SrtMemorySync, RejectsInvalidRangesBeforeMappingOrCacheQueries) {
    SyncRecorder recorder;
    constexpr VAddr MaxGpuAddress = 1ULL << 40;
    std::array<u8, 8> output{};

    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, 0, output.data(), output.size()));
    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, 0x1000, output.data(), 0));
    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, MaxGpuAddress, output.data(), 4));
    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, MaxGpuAddress - 4, output.data(), 8));
    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, 0x1000, output.data(), 1ULL << 32));
    EXPECT_TRUE(recorder.operations.empty());
}

TEST(SrtMemorySync, UnmappedRangesAreRejectedAndMappedRangesUsePrivateOutput) {
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

TEST(SrtMemoryReadSession, ReusesOnePrivateTrackingPage) {
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
    EXPECT_TRUE(session.Seal());
}

TEST(SrtMemoryReadSession, CrossPageFailureDoesNotPublishPartialOutput) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output;
    output.fill(0xcc);
    recorder.readback_succeeds = false;

    EXPECT_FALSE(session.Read(0x242713FFCUL, output.data(), output.size()));
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 byte) { return byte == 0xcc; }));
    EXPECT_FALSE(session.Valid());
}

TEST(SrtMemoryReadSession, CpuInvalidationDuringCopyRejectsTheWholeEpoch) {
    SyncRecorder recorder;
    recorder.invalidate_during_read = true;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output;
    output.fill(0xcc);

    EXPECT_FALSE(session.Read(0x242710318ULL, output.data(), output.size()));
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 byte) { return byte == 0xcc; }));
    EXPECT_FALSE(session.Valid());
    EXPECT_FALSE(session.Seal());
    EXPECT_TRUE(recorder.snapshot_watchers.empty());
}

TEST(SrtMemoryReadSession, SealedSessionReplaysCapturedPagesButRefusesNewPages) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> first{};
    std::array<u8, 8> replay{};
    std::array<u8, 8> uncaptured;
    uncaptured.fill(0xcc);

    ASSERT_TRUE(session.Read(0x242710318ULL, first.data(), first.size()));
    ASSERT_TRUE(session.Seal());
    recorder.snapshot_byte = 0xa5;

    EXPECT_TRUE(session.Read(0x242710318ULL, replay.data(), replay.size()));
    EXPECT_EQ(first, replay);
    EXPECT_FALSE(session.Read(0x242750318ULL, uncaptured.data(), uncaptured.size()));
    EXPECT_TRUE(std::ranges::all_of(uncaptured, [](u8 byte) { return byte == 0xcc; }));
    EXPECT_FALSE(session.Valid());
}

} // namespace
} // namespace VideoCore
