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

#include "video_core/buffer_cache/download_readback_policy.h"
#include "video_core/buffer_cache/srt_memory_sync.h"

namespace VideoCore {
namespace {

struct SyncRecorder {
    bool mapped = true;
    bool readback_succeeds = true;
    bool change_mapping_during_snapshot = false;
    bool change_cpu_generation_during_snapshot = false;
    bool exact_gpu_modified = false;
    bool gpu_page_owned = false;
    u8 snapshot_byte = 0x5a;
    u8 gpu_snapshot_byte = 0xa5;
    u64 default_mapping_token = 1;
    std::unordered_map<VAddr, u64> mapping_tokens;
    std::unordered_map<VAddr, u32> snapshot_watchers;
    std::vector<std::string_view> operations;
    VAddr read_address{};
    u64 read_size{};
    void* read_output{};
    VAddr mapped_address{};
    u64 mapped_size{};
    bool epoch_valid = true;
    bool epoch_sealed = false;
    u32 provenance_requests = 0;
    std::optional<PotentialShaderWriteProvenance> latest_potential_writer;

    class SnapshotEpoch {
    public:
        explicit SnapshotEpoch(SyncRecorder& owner_) : owner{&owner_} {}
        SnapshotEpoch(const SnapshotEpoch&) = delete;
        SnapshotEpoch& operator=(const SnapshotEpoch&) = delete;
        SnapshotEpoch(SnapshotEpoch&& other) noexcept
            : owner{std::exchange(other.owner, nullptr)} {}
        SnapshotEpoch& operator=(SnapshotEpoch&& other) noexcept {
            if (this != &other) {
                Seal();
                owner = std::exchange(other.owner, nullptr);
            }
            return *this;
        }
        ~SnapshotEpoch() {
            Seal();
        }

        bool BeginPageCopy(VAddr address, u64 size) {
            if (!owner || !owner->epoch_valid || owner->epoch_sealed || size != 16_KB) {
                return false;
            }
            owner->snapshot_watchers.try_emplace(Common::AlignDown(address, 16_KB), 1);
            return true;
        }

        bool EndPageCopy(VAddr, u64, bool copied) {
            return owner && owner->epoch_valid && !owner->epoch_sealed && copied;
        }

        void Invalidate() {
            if (owner) {
                owner->epoch_valid = false;
            }
        }

        [[nodiscard]] bool Valid() const {
            return owner ? owner->epoch_valid : result;
        }

        bool Seal() {
            if (!owner) {
                return false;
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

    void CpuWrite(VAddr address, u8 value) {
        const VAddr page = Common::AlignDown(address, 16_KB);
        if (snapshot_watchers.contains(page)) {
            epoch_valid = false;
        }
        snapshot_byte = value;
    }

    bool IsGpuMapped(VAddr address, u64 size) {
        operations.emplace_back("mapped");
        mapped_address = address;
        mapped_size = size;
        return mapped;
    }

    u64 GetGpuMappingToken(VAddr address, u64 size) const {
        if (!mapped || size == 0) {
            return 0;
        }
        const VAddr page = Common::AlignDown(address, 16_KB);
        const auto token = mapping_tokens.find(page);
        return token != mapping_tokens.end() ? token->second : default_mapping_token;
    }

    void SetGpuMappingToken(VAddr address, u64 token) {
        mapping_tokens[Common::AlignDown(address, 16_KB)] = token;
    }

    bool ReadMemorySnapshot(VAddr address, void* output, u64 size,
                            SnapshotReadProvenance* provenance = nullptr) {
        const auto read = [&](u8 byte) {
            operations.emplace_back("read_snapshot");
            read_address = address;
            read_output = output;
            read_size = size;
            std::memset(output, byte, size);
            if (change_mapping_during_snapshot) {
                const VAddr page = Common::AlignDown(address, 16_KB);
                SetGpuMappingToken(page, GetGpuMappingToken(page, 16_KB) + 1);
            }
            if (change_cpu_generation_during_snapshot) {
                CpuWrite(address, static_cast<u8>(byte + 1));
            }
            return readback_succeeds;
        };
        if (provenance == nullptr) {
            return ReadAuthoritativeMemorySnapshot(exact_gpu_modified, gpu_page_owned,
                                                   [&] { return read(snapshot_byte); },
                                                   [&] { return read(gpu_snapshot_byte); });
        }
        ++provenance_requests;
        const auto result = ReadAuthoritativeMemorySnapshotWithProvenance(
            exact_gpu_modified, gpu_page_owned, [&] { return read(snapshot_byte); },
            [&] { return read(gpu_snapshot_byte); });
        *provenance = result;
        return result.completed;
    }

    std::optional<PotentialShaderWriteProvenance> LatestPotentialShaderWrite(
        VAddr, u64, u64 mapping_token) const {
        if (latest_potential_writer && latest_potential_writer->mapping_token == mapping_token) {
            return latest_potential_writer;
        }
        return std::nullopt;
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

TEST(SrtMemorySync, UnmappedRangesAreRejectedAndMappedRangesAreSnapshotted) {
    SyncRecorder recorder;
    std::array<u8, 8> output{};
    recorder.mapped = false;
    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, 0x2000, output.data(), output.size()));
    EXPECT_EQ(recorder.operations, (std::vector<std::string_view>{"mapped"}));

    recorder.operations.clear();
    recorder.mapped = true;
    EXPECT_TRUE(ReadSrtMemoryRange(recorder, recorder, 0x2000, output.data(), output.size()));
    EXPECT_EQ(recorder.operations,
              (std::vector<std::string_view>{"mapped", "read_snapshot"}));
    EXPECT_EQ(output, (std::array<u8, 8>{0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a}));
}

TEST(SrtMemorySync, ReadsOnlyRequestedBytesIntoPrivateSnapshot) {
    SyncRecorder recorder;
    std::array<u8, 8> output{};

    EXPECT_TRUE(ReadSrtMemoryRange(recorder, recorder, 0x242710318ULL, output.data(), output.size()));
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
    EXPECT_EQ(recorder.mapped_address, 0x123FC000ULL);
    EXPECT_EQ(recorder.mapped_size, 0x8000U);
    EXPECT_EQ(recorder.read_address, PointerAddress);
}

TEST(SrtMemorySync, FailedSnapshotRefusesTheGuestRange) {
    SyncRecorder recorder;
    recorder.readback_succeeds = false;
    std::array<u8, 8> output{};
    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, 0x242710318ULL, output.data(), output.size()));
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
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"), 1);
}

TEST(SrtMemoryReadSession, DisabledDiagnosticsDoNotRequestMetadataOrChangeReadback) {
    SyncRecorder without_diagnostics;
    SyncRecorder with_diagnostics;
    without_diagnostics.exact_gpu_modified = true;
    without_diagnostics.gpu_page_owned = true;
    with_diagnostics.exact_gpu_modified = true;
    with_diagnostics.gpu_page_owned = true;
    std::array<u8, 8> plain{};
    std::array<u8, 8> observed{};

    SrtMemoryReadSession<SyncRecorder, SyncRecorder> plain_session{without_diagnostics,
                                                                  without_diagnostics};
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> observed_session{with_diagnostics,
                                                                     with_diagnostics, true};
    ASSERT_TRUE(plain_session.Read(0x242710318ULL, plain.data(), plain.size()));
    ASSERT_TRUE(observed_session.Read(0x242710318ULL, observed.data(), observed.size()));

    EXPECT_EQ(plain, observed);
    EXPECT_EQ(without_diagnostics.operations, with_diagnostics.operations);
    EXPECT_EQ(without_diagnostics.provenance_requests, 0U);
    EXPECT_EQ(plain_session.Provenance().Ranges().size(), 0U);
    EXPECT_EQ(with_diagnostics.provenance_requests, 1U);
}

TEST(SrtMemoryReadSession, EnabledDiagnosticsRecordOwnershipSourceTokenRangeAndDigest) {
    SyncRecorder recorder;
    recorder.exact_gpu_modified = true;
    recorder.gpu_page_owned = true;
    recorder.default_mapping_token = 23;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder, true};
    std::array<u8, 8> output{};

    ASSERT_TRUE(session.Read(0x242710318ULL, output.data(), output.size()));

    ASSERT_EQ(session.Provenance().Ranges().size(), 1U);
    const auto& range = session.Provenance().Ranges().front();
    EXPECT_EQ(range.address, 0x242710000ULL);
    EXPECT_EQ(range.size, 16_KB);
    EXPECT_TRUE(range.exact_gpu_modified);
    EXPECT_TRUE(range.coarse_gpu_owned);
    EXPECT_EQ(range.source, ShaderCaptureSource::GpuBuffer);
    EXPECT_EQ(range.mapping_token, 23U);
    std::array<u8, 16_KB> expected_page;
    expected_page.fill(recorder.gpu_snapshot_byte);
    EXPECT_EQ(range.content_digest, DigestShaderCaptureContent(expected_page));
    EXPECT_NE(range.content_digest, 0U);
}

TEST(SrtMemoryReadSession, RefusedOwnershipIsRecordedWithoutAContentDigest) {
    SyncRecorder recorder;
    recorder.gpu_page_owned = true;
    recorder.default_mapping_token = 31;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder, true};
    std::array<u8, 8> output{};

    EXPECT_FALSE(session.Read(0x242710318ULL, output.data(), output.size()));

    ASSERT_EQ(session.Provenance().Ranges().size(), 1U);
    const auto& range = session.Provenance().Ranges().front();
    EXPECT_FALSE(range.exact_gpu_modified);
    EXPECT_TRUE(range.coarse_gpu_owned);
    EXPECT_EQ(range.source, ShaderCaptureSource::Refused);
    EXPECT_EQ(range.mapping_token, 31U);
    EXPECT_EQ(range.content_digest, 0U);
}

TEST(SrtMemoryReadSession, CrossPageReadReusesTheFirstSessionGeneration) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> first{};
    std::array<u8, 8> second{};

    EXPECT_TRUE(session.Read(0x242713FFCUL, first.data(), first.size()));
    recorder.snapshot_byte = 0xa5;
    EXPECT_TRUE(session.Read(0x242713FFCUL, second.data(), second.size()));

    EXPECT_EQ(first, second);
    EXPECT_TRUE(std::ranges::all_of(first, [](u8 byte) { return byte == 0x5a; }));
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"), 2);
}

TEST(SrtMemoryReadSession, FailedPageSnapshotIsNotReused) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};
    recorder.readback_succeeds = false;

    EXPECT_FALSE(session.Read(0x242710318ULL, output.data(), output.size()));
    EXPECT_FALSE(session.Read(0x242710318ULL, output.data(), output.size()));
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"), 2);
}

TEST(SrtMemoryReadSession,
     PendingGpuReadbackRefusesStaleBackingAndFreshRetryCapturesOneNewGeneration) {
    SyncRecorder recorder;
    recorder.snapshot_byte = 0x11;
    recorder.gpu_page_owned = true;
    std::array<u8, 8> output;
    output.fill(0xcc);
    u32 accepted_selections = 0;
    const auto select = [&](auto& session) {
        if (!session.Read(0x242710318ULL, output.data(), output.size())) {
            return false;
        }
        ++accepted_selections;
        return true;
    };

    SrtMemoryReadSession<SyncRecorder, SyncRecorder> pending_session{recorder, recorder};
    EXPECT_FALSE(select(pending_session));
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 byte) { return byte == 0xcc; }));
    EXPECT_EQ(accepted_selections, 0U);
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"),
              0);

    // The deferred readback has now published the authoritative generation into guest backing.
    recorder.gpu_page_owned = false;
    recorder.snapshot_byte = 0x22;
    EXPECT_TRUE(select(pending_session));
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 byte) { return byte == 0x22; }));
    EXPECT_EQ(accepted_selections, 1U);
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"),
              1);
}

TEST(SrtMemoryReadSession, ExactGpuModificationUsesTheGpuGenerationWhileThePageIsOwned) {
    SyncRecorder recorder;
    recorder.snapshot_byte = 0x11;
    recorder.gpu_snapshot_byte = 0x33;
    recorder.exact_gpu_modified = true;
    recorder.gpu_page_owned = true;
    std::array<u8, 8> output{};

    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    EXPECT_TRUE(session.Read(0x242710318ULL, output.data(), output.size()));
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 byte) { return byte == 0x33; }));
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"),
              1);
}

TEST(SrtMemorySync, SnapshotIsRejectedWhenMappingChangesDuringCopy) {
    SyncRecorder recorder;
    recorder.change_mapping_during_snapshot = true;
    std::array<u8, 8> output{};

    EXPECT_FALSE(ReadSrtMemoryRange(recorder, recorder, 0x242710318ULL, output.data(), output.size()));
}

TEST(SrtMemoryReadSession, SameMappingCpuWriteDuringCopyRejectsTheSnapshot) {
    SyncRecorder recorder;
    recorder.change_cpu_generation_during_snapshot = true;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};

    EXPECT_FALSE(session.Read(0x242710318ULL, output.data(), output.size()));
    EXPECT_FALSE(session.Valid());
    EXPECT_FALSE(session.Seal());
    EXPECT_TRUE(recorder.snapshot_watchers.empty());
}

TEST(SrtMemoryReadSession, CpuWriteAfterLastReadInvalidatesBeforeCapturePublication) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};

    ASSERT_TRUE(session.Read(0x242710318ULL, output.data(), output.size()));
    recorder.CpuWrite(0x242710318ULL, 0xa5);

    EXPECT_FALSE(session.Valid());
    EXPECT_FALSE(session.Seal());
    EXPECT_TRUE(recorder.snapshot_watchers.empty());
}

TEST(SrtMemoryReadSession, InvalidatedAttemptCannotRetryFromALaterCpuGeneration) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> generation_a{};
    std::array<u8, 8> later_attempt;
    later_attempt.fill(0xcc);

    ASSERT_TRUE(session.Read(0x242710318ULL, generation_a.data(), generation_a.size()));
    recorder.CpuWrite(0x242710318ULL, 0xa5);

    EXPECT_FALSE(session.Read(0x242710318ULL, later_attempt.data(), later_attempt.size()));
    EXPECT_TRUE(std::ranges::all_of(generation_a, [](u8 byte) { return byte == 0x5a; }));
    EXPECT_TRUE(std::ranges::all_of(later_attempt, [](u8 byte) { return byte == 0xcc; }));
}

TEST(SrtMemoryReadSession, UnrelatedCpuWriteDoesNotInvalidateCapturedPages) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};

    ASSERT_TRUE(session.Read(0x242710318ULL, output.data(), output.size()));
    recorder.CpuWrite(0x242750318ULL, 0xa5);

    EXPECT_TRUE(session.Valid());
    EXPECT_TRUE(session.Seal());
    EXPECT_TRUE(recorder.snapshot_watchers.empty());
}

TEST(SrtMemoryReadSession, CachedPageIsRejectedAfterSameAddressIsRemapped) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> first{};
    std::array<u8, 8> second{};

    EXPECT_TRUE(session.Read(0x242710318ULL, first.data(), first.size()));
    recorder.SetGpuMappingToken(0x242710318ULL, 2);
    recorder.snapshot_byte = 0xa5;

    EXPECT_FALSE(session.Read(0x242710318ULL, second.data(), second.size()));
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"), 1);
}

TEST(SrtMemoryReadSession, CrossPageReadsCannotMixMappingGenerations) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> first{};
    std::array<u8, 8> second{};

    EXPECT_TRUE(session.Read(0x242710318ULL, first.data(), first.size()));
    recorder.SetGpuMappingToken(0x242710318ULL, 2);

    EXPECT_FALSE(session.Read(0x242714318ULL, second.data(), second.size()));
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"), 1);
}

TEST(SrtMemoryReadSession, UnrelatedPageRemapDoesNotInvalidateCachedPage) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> first{};
    std::array<u8, 8> second{};

    EXPECT_TRUE(session.Read(0x242710318ULL, first.data(), first.size()));
    recorder.SetGpuMappingToken(0x242750318ULL, 2);
    recorder.snapshot_byte = 0xa5;

    EXPECT_TRUE(session.Read(0x242710318ULL, second.data(), second.size()));
    EXPECT_EQ(first, second);
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"), 1);
}

} // namespace
} // namespace VideoCore
