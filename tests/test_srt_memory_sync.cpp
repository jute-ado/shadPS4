// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/download_readback_policy.h"
#include "video_core/buffer_cache/srt_failure_flat_diagnostics.h"
#include "video_core/buffer_cache/srt_failure_flat_snapshot.h"
#include "video_core/buffer_cache/srt_memory_sync.h"
#include "video_core/buffer_cache/srt_preparation_retry.h"
#include "video_core/buffer_cache/srt_read_failure_diagnostics.h"
#include "video_core/srt_snapshot_epoch_coordinator.h"

namespace VideoCore {
namespace {

struct SyncRecorder {
    bool mapped = true;
    bool readback_succeeds = true;
    bool invalidate_during_read = false;
    bool epoch_valid = true;
    bool epoch_sealed = false;
    bool reject_page_copy = false;
    std::optional<VAddr> reject_page_copy_address;
    u8 snapshot_byte = 0x5a;
    std::unordered_map<VAddr, u32> snapshot_watchers;
    std::vector<std::string_view> operations;
    VAddr read_address{};
    u64 read_size{};
    VAddr mapped_address{};
    u64 mapped_size{};
    u32 receipt_capture_calls{};

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
            if (!owner || !owner->epoch_valid || owner->epoch_sealed || owner->reject_page_copy ||
                (owner->reject_page_copy_address && *owner->reject_page_copy_address == address) ||
                (size != 4_KB && size != 16_KB)) {
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

    bool ReadMemorySnapshot(VAddr address, void* output, u64 size,
                            SrtSnapshotCompositionReceiptRequest* request) {
        if (!ReadMemorySnapshot(address, output, size)) {
            return false;
        }
        if (request == nullptr) {
            return true;
        }
        ++receipt_capture_calls;
        request->result = CaptureSrtSnapshotCompositionReceipt(
            SrtSnapshotCompositionReceiptConfig{
                .snapshot_address = address,
                .snapshot_size = size,
                .focus_address = request->focus_address,
                .focus_size = request->focus_size,
                .route = SrtSnapshotCompositionRoute::CpuOnly,
            },
            {static_cast<const u8*>(output), static_cast<size_t>(size)}, {});
        return true;
    }
};

TEST(SrtFailureFlatSnapshot, OwnsUserDataAndFirstWalkerOutputsAfterSourceMutation) {
    std::array<u32, 24> flattened{};
    for (u32 i = 0; i < flattened.size(); ++i) {
        flattened[i] = 0x1000U + i;
    }
    const auto captured = CaptureSrtFailureFlatSnapshot(flattened);
    flattened.fill(0xdeadbeefU);

    EXPECT_EQ(captured.total_dwords, 24U);
    EXPECT_EQ(captured.copied_dwords, 22U);
    EXPECT_TRUE(captured.has_first_walker_prefix);
    for (u32 i = 0; i < captured.words.size(); ++i) {
        EXPECT_EQ(captured.words[i], 0x1000U + i);
    }
}

TEST(SrtFailureFlatSnapshot, ShortInputReportsExactCoverageAndZeroFillsTail) {
    const std::array<u32, 3> flattened{7U, 8U, 9U};
    const auto captured = CaptureSrtFailureFlatSnapshot(flattened);
    EXPECT_EQ(captured.total_dwords, 3U);
    EXPECT_EQ(captured.copied_dwords, 3U);
    EXPECT_FALSE(captured.has_first_walker_prefix);
    EXPECT_EQ(captured.words[0], 7U);
    EXPECT_EQ(captured.words[1], 8U);
    EXPECT_EQ(captured.words[2], 9U);
    EXPECT_TRUE(std::ranges::all_of(captured.words.begin() + 3, captured.words.end(),
                                    [](u32 value) { return value == 0; }));
}

TEST(SrtFailureFlatDiagnostics, UnsetIsOffAndMalformedSelectorsFailClosed) {
    const SrtFailureFlatDiagnostics unset;
    EXPECT_FALSE(unset.Configured());
    EXPECT_TRUE(unset.Valid());
    EXPECT_FALSE(unset.Enabled());

    for (const auto selector : {"", "0x", "xyz", "10000000000000000", "ddd376ab!"}) {
        const SrtFailureFlatDiagnostics malformed{selector};
        EXPECT_TRUE(malformed.Configured());
        EXPECT_FALSE(malformed.Valid()) << selector;
        EXPECT_FALSE(malformed.Enabled()) << selector;
    }
}

TEST(SrtFailureFlatDiagnostics, ExactHashHasIndependentEightRowBudget) {
    SrtFailureFlatDiagnostics diagnostics{"0xddd376ab"};
    ASSERT_TRUE(diagnostics.Configured());
    ASSERT_TRUE(diagnostics.Valid());
    ASSERT_TRUE(diagnostics.Enabled());
    EXPECT_TRUE(diagnostics.Matches(0xddd376ab));
    EXPECT_FALSE(diagnostics.Matches(0x1234));

    std::vector<unsigned> rows;
    diagnostics.Emit(0x1234, [&](unsigned row) { rows.push_back(row); });
    for (unsigned attempt = 0; attempt < SrtFailureFlatDiagnostics::MaxRows + 2; ++attempt) {
        diagnostics.Emit(0xddd376ab, [&](unsigned row) { rows.push_back(row); });
    }
    ASSERT_EQ(rows.size(), SrtFailureFlatDiagnostics::MaxRows);
    for (unsigned row = 0; row < rows.size(); ++row) {
        EXPECT_EQ(rows[row], row);
    }
}

TEST(SrtFailureFlatDiagnostics, LoggerFailureIsContainedAndConsumesOneTargetSlot) {
    SrtFailureFlatDiagnostics diagnostics{"ddd376ab"};
    EXPECT_NO_THROW(diagnostics.Emit(0xddd376ab, [](unsigned) { throw 7; }));
    unsigned next = 99;
    diagnostics.Emit(0xddd376ab, [&](unsigned row) { next = row; });
    EXPECT_EQ(next, 1U);
}

TEST(SrtFailureFlatDiagnostics, FailureBudgetIsIndependentFromSuccessRows) {
    SrtFailureFlatDiagnostics diagnostics{"ddd376ab"};
    std::vector<unsigned> success_rows;
    std::vector<unsigned> failure_rows;

    for (unsigned attempt = 0; attempt < SrtFailureFlatDiagnostics::MaxRows + 2; ++attempt) {
        diagnostics.Emit(0xddd376ab, [&](unsigned row) { success_rows.push_back(row); });
    }
    for (unsigned attempt = 0; attempt < SrtFailureFlatDiagnostics::MaxRows + 2; ++attempt) {
        diagnostics.EmitFailure(0xddd376ab, [&](unsigned row) { failure_rows.push_back(row); });
    }

    ASSERT_EQ(success_rows.size(), SrtFailureFlatDiagnostics::MaxRows);
    ASSERT_EQ(failure_rows.size(), SrtFailureFlatDiagnostics::MaxRows);
    for (unsigned row = 0; row < SrtFailureFlatDiagnostics::MaxRows; ++row) {
        EXPECT_EQ(success_rows[row], row);
        EXPECT_EQ(failure_rows[row], row);
    }
}

TEST(DownloadReadbackPolicy, FocusOverlapUsesOverflowSafeRanges) {
    EXPECT_TRUE(SnapshotOverlayIntersectsFocus(0x70, 8, 0x70, 8));
    EXPECT_TRUE(SnapshotOverlayIntersectsFocus(0x40, 0x40, 0x70, 8));
    EXPECT_FALSE(SnapshotOverlayIntersectsFocus(0x78, 8, 0x70, 8));
    EXPECT_FALSE(SnapshotOverlayIntersectsFocus(std::numeric_limits<u64>::max() - 3, 8,
                                                std::numeric_limits<u64>::max() - 1, 1));
}

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

TEST(SrtMemoryReadSession, ReusesOnePrivateHostPage) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> first{};
    std::array<u8, 16> second{};

    EXPECT_TRUE(session.Read(0x242710318ULL, first.data(), first.size()));
    EXPECT_TRUE(session.Read(0x242710380ULL, second.data(), second.size()));
    EXPECT_EQ(recorder.read_address, 0x242710000ULL);
    EXPECT_EQ(recorder.read_size, 4_KB);
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"),
              1);
    EXPECT_TRUE(session.Seal());
}

TEST(SrtMemoryReadSession, BatchesAdjacentUncapturedPagesIntoOneReadback) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 2 * SrtSnapshotPageSize> output{};

    ASSERT_TRUE(session.Read(0x4000, output.data(), output.size()));
    EXPECT_EQ(recorder.read_address, 0x4000U);
    EXPECT_EQ(recorder.read_size, 2 * SrtSnapshotPageSize);
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"),
              1);
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 byte) { return byte == 0x5a; }));
    EXPECT_TRUE(session.Seal());
}

TEST(SrtMemoryReadSession, BoundsOneReadbackBatchBeforeCapturingTheRemainder) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::vector<u8> output(17 * SrtSnapshotPageSize);

    ASSERT_TRUE(session.Read(0x4000, output.data(), output.size()));
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"),
              2);
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 byte) { return byte == 0x5a; }));
    EXPECT_TRUE(session.Seal());
}

TEST(SrtMemoryReadSession, MidBatchAdmissionFailureReleasesEveryBegunCopy) {
    SyncRecorder recorder;
    recorder.reject_page_copy_address = 0x5000;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 2 * SrtSnapshotPageSize> output;
    output.fill(0xcc);

    EXPECT_FALSE(session.Read(0x4000, output.data(), output.size()));
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 byte) { return byte == 0xcc; }));
    EXPECT_TRUE(recorder.operations.empty());
    EXPECT_FALSE(session.Seal());
    EXPECT_TRUE(recorder.snapshot_watchers.empty());
}

TEST(SrtSnapshotCompositionSession, ArmRetainsFirstAcceptedRootAcrossLaterFailure) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder};
    std::array<u8, 8> output{};

    session.ArmCompositionReceipt();
    ASSERT_TRUE(session.Read(0x4018, output.data(), output.size()));
    ASSERT_TRUE(session.CompositionReceipt());
    const auto receipt = *session.CompositionReceipt();
    EXPECT_EQ(receipt.snapshot_address, 0x4000U);
    EXPECT_EQ(receipt.snapshot_size, SrtSnapshotPageSize);
    EXPECT_EQ(receipt.focus_address, 0x4018U);
    EXPECT_EQ(receipt.focus_size, output.size());
    EXPECT_EQ(receipt.route, SrtSnapshotCompositionRoute::CpuOnly);
    EXPECT_EQ(recorder.receipt_capture_calls, 1U);

    EXPECT_FALSE(session.Read(0, output.data(), output.size()));
    ASSERT_TRUE(session.CompositionReceipt());
    EXPECT_EQ(session.CompositionReceipt()->snapshot_address, receipt.snapshot_address);
    EXPECT_EQ(session.CompositionReceipt()->focus_address, receipt.focus_address);
    EXPECT_EQ(session.CompositionReceipt()->route, receipt.route);
}

TEST(SrtSnapshotCompositionSession,
     AcceptedRootSurvivesLaterFailureWithoutAttributingItsRouteToTheFailure) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder, true};
    std::array<u8, 8> output{};

    session.ArmCompositionReceipt();
    ASSERT_TRUE(session.Read(0x4018, output.data(), output.size()));
    ASSERT_TRUE(session.CompositionReceipt());
    const auto root = *session.CompositionReceipt();

    // The root composition remains the accepted snapshot, while a later malformed source
    // request must be recorded independently and must not inherit the root's CPU-only route.
    EXPECT_FALSE(session.Read(0, nullptr, output.size()));
    ASSERT_TRUE(session.CompositionReceipt());
    EXPECT_EQ(session.CompositionReceipt()->snapshot_address, root.snapshot_address);
    EXPECT_EQ(session.CompositionReceipt()->focus_address, root.focus_address);
    EXPECT_EQ(session.CompositionReceipt()->route, root.route);
    EXPECT_EQ(session.FirstFailureRequest(),
              (SrtReadFailureRequest{true, 0, output.size(), true, 1U | 2U}));

    const auto reads = session.SourceReads();
    ASSERT_EQ(reads.size(), 2U);
    EXPECT_EQ(reads[0].route, SrtSnapshotCompositionRoute::CpuOnly);
    EXPECT_EQ(reads[1].failure, SrtMemoryReadFailure::InvalidRequest);
    EXPECT_EQ(reads[1].route, SrtSnapshotCompositionRoute::Unavailable);
    EXPECT_EQ(reads[1].snapshot_address, 0U);
    EXPECT_EQ(reads[1].snapshot_size, 0U);
    EXPECT_EQ(reads[1].bytes_captured, 0U);
}

TEST(SrtSnapshotCompositionSession, LaterReadsCannotOverwriteFirstAcceptedReceipt) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder};
    std::array<u8, 8> output{};

    session.ArmCompositionReceipt();
    ASSERT_TRUE(session.Read(0x4018, output.data(), output.size()));
    ASSERT_TRUE(session.Read(0x8010, output.data(), output.size()));

    ASSERT_TRUE(session.CompositionReceipt());
    EXPECT_EQ(session.CompositionReceipt()->focus_address, 0x4018U);
    EXPECT_EQ(recorder.receipt_capture_calls, 1U);
}

TEST(SrtSnapshotCompositionSession,
     SourceReadRecordsDoNotReuseFirstCompositionReceiptForLaterCachedReads) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder, true};
    std::array<u8, 8> output{};

    session.ArmCompositionReceipt();
    ASSERT_TRUE(session.Read(0x4018, output.data(), output.size()));
    ASSERT_TRUE(session.Read(0x4020, output.data(), output.size()));

    const auto reads = session.SourceReads();
    ASSERT_EQ(reads.size(), 2U);
    EXPECT_EQ(reads[0].route, SrtSnapshotCompositionRoute::CpuOnly);
    EXPECT_EQ(reads[0].snapshot_address, 0x4000U);
    EXPECT_EQ(reads[1].route, SrtSnapshotCompositionRoute::Unavailable);
    EXPECT_EQ(reads[1].snapshot_address, 0U);
    EXPECT_EQ(reads[1].diagnostic_snapshot_id, 0U);
}

TEST(SrtSnapshotCompositionSession,
     SourceReadRecordsAttachCompositionOnlyToTheArmedReadAfterAnUnarmedRead) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder, true};
    std::array<u8, 8> output{};

    ASSERT_TRUE(session.Read(0x4018, output.data(), output.size()));
    session.ArmCompositionReceipt();
    ASSERT_TRUE(session.Read(0x8010, output.data(), output.size()));

    const auto reads = session.SourceReads();
    ASSERT_EQ(reads.size(), 2U);
    EXPECT_EQ(reads[0].route, SrtSnapshotCompositionRoute::Unavailable);
    EXPECT_EQ(reads[0].snapshot_address, 0U);
    EXPECT_EQ(reads[1].route, SrtSnapshotCompositionRoute::CpuOnly);
    EXPECT_EQ(reads[1].snapshot_address, 0x8000U);
}

TEST(SrtSnapshotCompositionSession, RejectedPageCopyDiscardsCandidateReceipt) {
    SyncRecorder recorder;
    recorder.invalidate_during_read = true;
    SrtMemoryReadSession session{recorder, recorder};
    std::array<u8, 8> output{};

    session.ArmCompositionReceipt();
    EXPECT_FALSE(session.Read(0x4018, output.data(), output.size()));
    EXPECT_FALSE(session.CompositionReceipt());
    EXPECT_EQ(recorder.receipt_capture_calls, 1U);
}

TEST(SrtSourceRead, FailedReadRecordRetainsUnderlyingReadClassification) {
    for (const auto failure : {SrtMemoryReadFailure::UnmappedRange,
                               SrtMemoryReadFailure::ReadbackUnavailable}) {
        SyncRecorder recorder;
        recorder.mapped = failure != SrtMemoryReadFailure::UnmappedRange;
        recorder.readback_succeeds = failure != SrtMemoryReadFailure::ReadbackUnavailable;
        SrtMemoryReadSession session{recorder, recorder, true};
        std::array<u8, 8> output{};

        ASSERT_FALSE(session.Read(0x4018, output.data(), output.size()));
        ASSERT_EQ(session.LastFailure(), failure);
        const auto reads = session.SourceReads();
        ASSERT_EQ(reads.size(), 1U);
        EXPECT_EQ(reads.front().failure, failure);
        EXPECT_FALSE(reads.front().source_bytes_known);
        EXPECT_EQ(reads.front().bytes_captured, 0U);
    }
}

TEST(SrtSnapshotCompositionSession, CachedFirstPageReportsUnavailableWithoutReread) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder};
    std::array<u8, 8> output{};
    ASSERT_TRUE(session.Read(0x4018, output.data(), output.size()));
    const auto initial_reads =
        std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot");

    session.ArmCompositionReceipt();
    ASSERT_TRUE(session.Read(0x4020, output.data(), output.size()));
    ASSERT_TRUE(session.CompositionReceipt());
    EXPECT_EQ(session.CompositionReceipt()->route, SrtSnapshotCompositionRoute::Unavailable);
    EXPECT_EQ(session.CompositionReceipt()->snapshot_address, 0x4000U);
    EXPECT_EQ(session.CompositionReceipt()->focus_address, 0x4020U);
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"),
              initial_reads);
    EXPECT_EQ(recorder.receipt_capture_calls, 0U);

    ASSERT_TRUE(session.Read(0x8010, output.data(), output.size()));
    EXPECT_EQ(session.CompositionReceipt()->focus_address, 0x4020U);
    EXPECT_EQ(recorder.receipt_capture_calls, 0U);
}

TEST(SrtSnapshotCompositionSession, CrossPageFocusIsClippedToStartingPage) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder};
    std::array<u8, 8> output{};

    session.ArmCompositionReceipt();
    ASSERT_TRUE(session.Read(0x4ffc, output.data(), output.size()));

    ASSERT_TRUE(session.CompositionReceipt());
    EXPECT_EQ(session.CompositionReceipt()->snapshot_address, 0x4000U);
    EXPECT_EQ(session.CompositionReceipt()->focus_address, 0x4ffcU);
    EXPECT_EQ(session.CompositionReceipt()->focus_size, 4U);
    EXPECT_EQ(recorder.receipt_capture_calls, 1U);
    EXPECT_EQ(std::count(recorder.operations.begin(), recorder.operations.end(), "read_snapshot"),
              2);
}

TEST(SrtSnapshotCompositionSession, DefaultOffDoesNotRequestOrPublishReceipt) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder};
    std::array<u8, 8> output{};

    ASSERT_TRUE(session.Read(0x4018, output.data(), output.size()));
    EXPECT_FALSE(session.CompositionReceipt());
    EXPECT_EQ(recorder.receipt_capture_calls, 0U);
}

TEST(SrtSnapshotCompositionSession, InvalidRequestPublishesNoReceipt) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder};
    std::array<u8, 8> output{};

    session.ArmCompositionReceipt();
    EXPECT_FALSE(session.Read(0, output.data(), output.size()));
    EXPECT_FALSE(session.CompositionReceipt());
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

TEST(SrtMemoryReadSession, InvalidationBetweenCachedPageReadsDoesNotPublishSecondRange) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> first{};
    std::array<u8, 8> second;
    second.fill(0xcc);

    ASSERT_TRUE(session.Read(0x4010, first.data(), first.size()));
    EXPECT_TRUE(std::ranges::all_of(first, [](u8 byte) { return byte == 0x5a; }));
    // The second request is in the already captured page. It must still observe epoch
    // invalidation before copying cached bytes into the caller's output.
    recorder.epoch_valid = false;
    EXPECT_FALSE(session.Read(0x4020, second.data(), second.size()));
    EXPECT_TRUE(std::ranges::all_of(second, [](u8 byte) { return byte == 0xcc; }));
    EXPECT_FALSE(session.Valid());
    EXPECT_TRUE(*session.FailureFlag());
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

template <typename Session>
std::string_view FailureReason(const Session& session) {
    if constexpr (requires { session.LastFailure(); }) {
        return SrtMemoryReadFailureName(session.LastFailure());
    } else {
        return "unclassified";
    }
}

TEST(SrtMemoryReadDiagnostics, SuccessfulCaptureAndSealHaveNoFailure) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};
    ASSERT_TRUE(session.Read(0x4010, output.data(), output.size()));
    EXPECT_EQ(FailureReason(session), "none");
    ASSERT_TRUE(session.Seal());
    EXPECT_EQ(FailureReason(session), "none");
    EXPECT_FALSE(*session.FailureFlag());
}

TEST(SrtMemoryReadDiagnostics, InvalidRequestIsRetainedAcrossLaterReadAttempts) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};
    EXPECT_FALSE(session.Read(0, output.data(), output.size()));
    EXPECT_EQ(FailureReason(session), "invalid_request");
    recorder.mapped = false;
    EXPECT_FALSE(session.Read(0x4010, output.data(), output.size()));
    EXPECT_EQ(FailureReason(session), "invalid_request");
    EXPECT_TRUE(recorder.operations.empty());
    EXPECT_TRUE(*session.FailureFlag());
}

TEST(SrtMemoryReadDiagnostics, UnmappedRangeNeverAttemptsReadback) {
    SyncRecorder recorder;
    recorder.mapped = false;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};
    EXPECT_FALSE(session.Read(0x4010, output.data(), output.size()));
    EXPECT_EQ(FailureReason(session), "unmapped_range");
    EXPECT_EQ(recorder.operations, (std::vector<std::string_view>{"mapped"}));
    EXPECT_TRUE(*session.FailureFlag());
}

TEST(SrtMemoryReadDiagnostics, ReadbackFailureIsNotReportedAsEpochInvalidation) {
    SyncRecorder recorder;
    recorder.readback_succeeds = false;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};
    EXPECT_FALSE(session.Read(0x4010, output.data(), output.size()));
    EXPECT_EQ(FailureReason(session), "readback_unavailable");
    EXPECT_TRUE(recorder.epoch_valid);
    EXPECT_TRUE(*session.FailureFlag());
}

TEST(SrtMemoryReadDiagnostics, InvalidationDuringCopyIsDistinctFromUnreadableBacking) {
    SyncRecorder recorder;
    recorder.invalidate_during_read = true;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};
    EXPECT_FALSE(session.Read(0x4010, output.data(), output.size()));
    EXPECT_EQ(FailureReason(session), "epoch_invalidated");
    EXPECT_EQ(output, (std::array<u8, 8>{}));
    EXPECT_TRUE(*session.FailureFlag());
}

TEST(SrtMemoryReadDiagnostics, LateInvalidationDoesNotChangeExistingReaderFailureFlag) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};
    ASSERT_TRUE(session.Read(0x4010, output.data(), output.size()));
    recorder.epoch_valid = false;
    EXPECT_FALSE(session.Valid());
    EXPECT_EQ(FailureReason(session), "epoch_invalidated");
    EXPECT_FALSE(*session.FailureFlag());
    EXPECT_FALSE(session.Seal());
    EXPECT_EQ(FailureReason(session), "epoch_invalidated");
    EXPECT_FALSE(*session.FailureFlag());
}

TEST(SrtMemoryReadDiagnostics, RefusedPageCopyIsDistinctFromMappingAndReadbackFailure) {
    SyncRecorder recorder;
    recorder.reject_page_copy = true;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};
    EXPECT_FALSE(session.Read(0x4010, output.data(), output.size()));
    EXPECT_EQ(FailureReason(session), "page_copy_rejected");
    EXPECT_TRUE(recorder.operations.empty());
}

TEST(SrtMemoryReadDiagnostics, SealedPageDiscoveryDoesNotChangeCapturedReplay) {
    SyncRecorder recorder;
    SrtMemoryReadSession<SyncRecorder, SyncRecorder> session{recorder, recorder};
    std::array<u8, 8> output{};
    ASSERT_TRUE(session.Read(0x4010, output.data(), output.size()));
    ASSERT_TRUE(session.Seal());
    ASSERT_TRUE(session.Read(0x4010, output.data(), output.size()));
    EXPECT_EQ(FailureReason(session), "none");
    EXPECT_FALSE(session.Read(0x8010, output.data(), output.size()));
    EXPECT_EQ(FailureReason(session), "new_page_after_seal");
    EXPECT_TRUE(*session.FailureFlag());
}

TEST(SrtReadFailureMetadata, InvalidReasonsCoverIndependentAndOverlappingAdmissionChecks) {
    EXPECT_EQ(SrtInvalidReadReasons(0x1000, 4, false), 0U);
    EXPECT_EQ(SrtInvalidReadReasons(0x1000, 4, true), 1U);
    EXPECT_EQ(SrtInvalidReadReasons(0, 4, false), 2U);
    EXPECT_EQ(SrtInvalidReadReasons(0x1000, 0, false), 4U);
    EXPECT_EQ(SrtInvalidReadReasons(0x1000, 1ULL << 32, false), 8U);
    EXPECT_EQ(SrtInvalidReadReasons(MaxSrtGpuAddress, 4, false), 16U);
    EXPECT_EQ(SrtInvalidReadReasons(MaxSrtGpuAddress - 2, 4, false), 32U);
    EXPECT_EQ(SrtInvalidReadReasons(0, 0, true), 1U | 2U | 4U);
    EXPECT_EQ(SrtInvalidReadReasons(~0ULL, ~0ULL, false), 8U | 16U);
}

TEST(SrtReadFailureMetadata, ExactInvalidRequestSurvivesLaterReads) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder, true};
    EXPECT_FALSE(session.Read(MaxSrtGpuAddress + 7, nullptr, 11));
    const auto first = session.FirstFailureRequest();
    EXPECT_EQ(first, (SrtReadFailureRequest{true, MaxSrtGpuAddress + 7, 11, true, 17}));
    std::array<u8, 8> output{};
    EXPECT_FALSE(session.Read(0x4010, output.data(), output.size()));
    EXPECT_EQ(session.FirstFailureRequest(), first);
    EXPECT_EQ(session.LastFailure(), SrtMemoryReadFailure::InvalidRequest);
    EXPECT_TRUE(recorder.operations.empty());
    EXPECT_TRUE(*session.FailureFlag());
    EXPECT_FALSE(session.Seal());
    EXPECT_EQ(session.FirstFailureRequest(), first);
}

TEST(SrtReadFailureMetadata, RecordsOriginalRequestNotExpandedMappingOrCopyPage) {
    for (const auto failure :
         {SrtMemoryReadFailure::UnmappedRange, SrtMemoryReadFailure::ReadbackUnavailable,
          SrtMemoryReadFailure::PageCopyRejected, SrtMemoryReadFailure::EpochInvalidated}) {
        SCOPED_TRACE(SrtMemoryReadFailureName(failure));
        SyncRecorder recorder;
        recorder.mapped = failure != SrtMemoryReadFailure::UnmappedRange;
        recorder.readback_succeeds = failure != SrtMemoryReadFailure::ReadbackUnavailable;
        recorder.reject_page_copy = failure == SrtMemoryReadFailure::PageCopyRejected;
        recorder.invalidate_during_read = failure == SrtMemoryReadFailure::EpochInvalidated;
        SrtMemoryReadSession session{recorder, recorder, true};
        std::array<u8, 8> output{};
        EXPECT_FALSE(session.Read(0x4ffc, output.data(), output.size()));
        EXPECT_EQ(session.FirstFailureRequest(),
                  (SrtReadFailureRequest{true, 0x4ffc, 8, false, 0}));
        EXPECT_EQ(session.LastFailure(), failure);
        EXPECT_EQ(output, (std::array<u8, 8>{}));
    }
}

TEST(SrtReadFailureMetadata, FailedSealKeepsUnknownRequestDespiteLaterInvalidRead) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder, true};
    std::array<u8, 8> output{};
    ASSERT_TRUE(session.Read(0x4010, output.data(), output.size()));
    recorder.epoch_valid = false;
    EXPECT_FALSE(session.Seal());
    EXPECT_EQ(session.FirstFailureRequest(), SrtReadFailureRequest{});
    EXPECT_EQ(session.LastFailure(), SrtMemoryReadFailure::EpochInvalidated);
    EXPECT_FALSE(*session.FailureFlag());
    EXPECT_FALSE(session.Read(0, nullptr, 0));
    EXPECT_EQ(session.FirstFailureRequest(), SrtReadFailureRequest{});
}

TEST(SrtReadFailureMetadata, ExpandedZeroPageRefusalDoesNotInventInvalidOriginalRequestBits) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder, true};
    std::array<u8, 8> output{};
    EXPECT_FALSE(session.Read(1, output.data(), output.size()));
    EXPECT_EQ(session.LastFailure(), SrtMemoryReadFailure::InvalidRequest);
    EXPECT_EQ(session.FirstFailureRequest(), (SrtReadFailureRequest{true, 1, 8, false, 0}));
    EXPECT_TRUE(recorder.operations.empty());
}

TEST(SrtReadFailureMetadata, ZeroPageExpansionRetainsOriginalAddressAndSize) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder, true};
    std::array<u8, 16> output;
    output.fill(0xcc);

    // This is the observed small descriptor request. The host-page capture aligns its internal
    // read down to address zero, but diagnostics must describe the guest request (0x32, 16), not
    // the rejected expanded page (0, 4096).
    EXPECT_FALSE(session.Read(0x32, output.data(), output.size()));
    EXPECT_EQ(session.LastFailure(), SrtMemoryReadFailure::InvalidRequest);
    EXPECT_EQ(session.FirstFailureRequest(), (SrtReadFailureRequest{true, 0x32, 16, false, 0}));
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 byte) { return byte == 0xcc; }));
    EXPECT_TRUE(recorder.operations.empty());
}

TEST(SrtReadFailureMetadata, NewPageAfterSuccessfulSealRetainsActualRequest) {
    SyncRecorder recorder;
    SrtMemoryReadSession session{recorder, recorder, true};
    std::array<u8, 8> output{};
    ASSERT_TRUE(session.Read(0x4010, output.data(), output.size()));
    ASSERT_TRUE(session.Seal());
    EXPECT_FALSE(session.Read(0x8010, output.data(), output.size()));
    EXPECT_EQ(session.FirstFailureRequest(), (SrtReadFailureRequest{true, 0x8010, 8, false, 0}));
    EXPECT_EQ(session.LastFailure(), SrtMemoryReadFailure::NewPageAfterSeal);
}

TEST(SrtReadFailureMetadata, DefaultOffAndNoReadAdmissionFailureHaveNoRequest) {
    SyncRecorder recorder;
    SrtMemoryReadSession disabled{recorder, recorder};
    EXPECT_FALSE(disabled.Read(0, nullptr, 0));
    EXPECT_EQ(disabled.FirstFailureRequest(), SrtReadFailureRequest{});
    SyncRecorder other;
    SrtMemoryReadSession enabled{other, other, true};
    EXPECT_EQ(enabled.FirstFailureRequest(), SrtReadFailureRequest{});
    EXPECT_EQ(enabled.LastFailure(), SrtMemoryReadFailure::None);
    EXPECT_FALSE(*enabled.FailureFlag());
    std::array<u8, 8> output{};
    ASSERT_TRUE(enabled.Read(0x4010, output.data(), output.size()));
    EXPECT_EQ(enabled.FirstFailureRequest(), SrtReadFailureRequest{});
}

TEST(SrtReadFailureDiagnosticBudget, OnlyExactOneEnables) {
    EXPECT_TRUE(ParseSrtReadFailureDiagnostics("1"));
    for (const auto value : {"", "0", "01", "true", " 1", "1 ", "1\n"}) {
        EXPECT_FALSE(ParseSrtReadFailureDiagnostics(value));
    }
}

TEST(SrtReadFailureDiagnosticBudget, DisabledDoesNotObserveOrLog) {
    SrtReadFailureDiagnostics diagnostics{false};
    unsigned records{};
    diagnostics.Emit([&](unsigned) { ++records; });
    EXPECT_EQ(records, 0U);
    EXPECT_FALSE(diagnostics.Enabled());
}

TEST(SrtReadFailureDiagnosticBudget, ThirtyTwoRowsThenNoMoreObservation) {
    SrtReadFailureDiagnostics diagnostics{true};
    unsigned records{};
    for (unsigned i = 0; i < 100; ++i) {
        diagnostics.Emit([&](unsigned row) { EXPECT_EQ(row, records++); });
    }
    EXPECT_EQ(records, 32U);
}

TEST(SrtReadFailureDiagnosticBudget, LoggerFailureIsContainedAndConsumesOnlyOneSlot) {
    SrtReadFailureDiagnostics diagnostics{true};
    unsigned records{};
    EXPECT_NO_THROW(diagnostics.Emit([&](unsigned row) {
        ++records;
        EXPECT_EQ(row, 0U);
        throw 9;
    }));
    diagnostics.Emit([&](unsigned row) {
        ++records;
        EXPECT_EQ(row, 1U);
    });
    EXPECT_EQ(records, 2U);
}

TEST(SrtReadFailureDiagnosticBudget, ConcurrentAttemptsCannotExceedProcessBudget) {
    SrtReadFailureDiagnostics diagnostics{true};
    std::array<std::atomic<unsigned>, 32> hits{};
    std::atomic<unsigned> out_of_range{};
    std::vector<std::thread> producers;
    for (unsigned producer = 0; producer < 8; ++producer) {
        producers.emplace_back([&] {
            for (unsigned call = 0; call < 64; ++call) {
                diagnostics.Emit([&](unsigned row) {
                    if (row < hits.size()) {
                        ++hits[row];
                    } else {
                        ++out_of_range;
                    }
                });
            }
        });
    }
    for (auto& producer : producers) {
        producer.join();
    }
    EXPECT_EQ(out_of_range, 0U);
    for (const auto& hit : hits) {
        EXPECT_EQ(hit, 1U);
    }
}

// The coordinator and session are production classes. Only the logical mapping query,
// host-watcher sink and CPU/GPU byte source are substituted; no Vulkan or guest process runs.
struct PartitionMemory {
    static constexpr VAddr Base = 0x40000;
    std::array<u8, 32_KB> backing{};
    std::array<bool, 32_KB / SrtSnapshotPageSize> mapped_pages{};
    SrtSnapshotEpochCoordinator coordinator;
    std::vector<std::pair<VAddr, u64>> mapping_queries;
    std::vector<std::pair<VAddr, u64>> copies;
    u32 watchers{};
    u32 epochs{};
    bool completely_mapped = true;

    PartitionMemory() {
        mapped_pages.fill(true);
        for (u32 page = 0; page < backing.size() / 4_KB; ++page) {
            std::fill_n(backing.begin() + page * 4_KB, 4_KB, static_cast<u8>(0x11 + page));
        }
    }

    struct Epoch {
        PartitionMemory& memory;
        u64 id;
        bool sealed_result{};
        ~Epoch() {
            (void)Seal();
        }
        bool BeginPageCopy(VAddr address, u64 size) {
            return memory.coordinator.BeginPageCopy(id, address, size,
                                                    [&](VAddr, u64) { ++memory.watchers; });
        }
        bool EndPageCopy(VAddr address, u64 size, bool copied) {
            return memory.coordinator.EndPageCopy(id, address, size, copied);
        }
        bool Valid() const {
            return id ? memory.coordinator.IsEpochValid(id) : sealed_result;
        }
        bool Seal() {
            if (id) {
                sealed_result =
                    memory.coordinator.SealEpoch(id, [&](VAddr, u64) { --memory.watchers; });
                id = 0;
            }
            return sealed_result;
        }
    };

    Epoch BeginSnapshotEpoch() {
        ++epochs;
        return {*this, coordinator.BeginEpoch()};
    }
    bool IsGpuMapped(VAddr address, u64 size) {
        mapping_queries.emplace_back(address, size);
        if (!completely_mapped || address < Base || size == 0) {
            return false;
        }
        const VAddr offset = address - Base;
        if (offset > backing.size() || size > backing.size() - offset ||
            offset % SrtSnapshotPageSize != 0 || size % SrtSnapshotPageSize != 0) {
            return false;
        }
        const auto first_page = static_cast<size_t>(offset / SrtSnapshotPageSize);
        const auto page_count = static_cast<size_t>(size / SrtSnapshotPageSize);
        return std::all_of(mapped_pages.begin() + first_page,
                           mapped_pages.begin() + first_page + page_count,
                           [](bool mapped) { return mapped; });
    }
    bool ReadMemorySnapshot(VAddr address, void* output, u64 size) {
        copies.emplace_back(address, size);
        if (address < Base || address - Base > backing.size() ||
            size > backing.size() - (address - Base)) {
            return false;
        }
        std::memcpy(output, backing.data() + (address - Base), size);
        return true;
    }
    void Write(VAddr address) {
        coordinator.InvalidateRangeAndWait(address, 8, [&](VAddr, u64) { --watchers; });
        backing.at(address - Base) ^= 0xff;
    }
    void Unmap(VAddr address, u64 size) {
        coordinator.InvalidateRangeAndWait(address, size, [&](VAddr, u64) { --watchers; });
        completely_mapped = false;
    }
};

TEST(SrtSnapshotCompositionSession, LegacyThreeArgumentCacheReportsUnavailableWithoutReread) {
    PartitionMemory memory;
    SrtMemoryReadSession session{memory, memory};
    std::array<u8, 8> descriptor{};

    session.ArmCompositionReceipt();
    ASSERT_TRUE(session.Read(PartitionMemory::Base + 0x100, descriptor.data(), descriptor.size()));
    ASSERT_TRUE(session.CompositionReceipt());
    EXPECT_EQ(session.CompositionReceipt()->route, SrtSnapshotCompositionRoute::Unavailable);
    EXPECT_EQ(session.CompositionReceipt()->snapshot_address, PartitionMemory::Base);
    EXPECT_EQ(session.CompositionReceipt()->focus_address, PartitionMemory::Base + 0x100);
    EXPECT_EQ(memory.copies.size(), 1U);
}

TEST(SrtSnapshotHostPages, IndependentHostPageWritesDoNotExhaustPreparation) {
    PartitionMemory memory;
    u32 binding = 5;
    const auto result = RetrySrtPreparation(
        binding,
        [&](u32& next_binding, SrtMemoryReadFailure& failure, bool&) -> std::optional<u8> {
            SrtMemoryReadSession session{memory, memory};
            std::array<u8, 32> descriptor{};
            EXPECT_TRUE(
                session.Read(PartitionMemory::Base + 0x100, descriptor.data(), descriptor.size()));
            // Same 16KiB guest mapping, different 4KiB host page, never consumed by this shader.
            // Repeating this on every attempt must not starve an unchanged descriptor.
            memory.Write(PartitionMemory::Base + 0x3100);
            ++next_binding;
            if (!session.Seal()) {
                failure = session.LastFailure();
                return std::nullopt;
            }
            return descriptor[0];
        },
        [](const auto& value) { return value.has_value(); });
    EXPECT_EQ(result, 0x11);
    EXPECT_EQ(memory.epochs, 1u);
    EXPECT_EQ(binding, 6u);
    EXPECT_EQ(memory.watchers, 0u);
}

TEST(SrtSnapshotHostPages, HostPageCopyStillRequiresWholeGuestMappingCoverage) {
    PartitionMemory memory;
    SrtMemoryReadSession session{memory, memory};
    std::array<u8, 8> descriptor{};
    ASSERT_TRUE(session.Read(PartitionMemory::Base + 0x1100, descriptor.data(), descriptor.size()));
    EXPECT_EQ(memory.mapping_queries,
              (std::vector<std::pair<VAddr, u64>>{{PartitionMemory::Base, 16_KB}}));
    EXPECT_EQ(memory.copies,
              (std::vector<std::pair<VAddr, u64>>{{PartitionMemory::Base + 4_KB, 4_KB}}));
    EXPECT_TRUE(session.Seal());
    EXPECT_EQ(memory.watchers, 0u);
}

TEST(SrtSnapshotHostPages, StraddlingReadRetainsBothPagesUntilSeal) {
    PartitionMemory memory;
    SrtMemoryReadSession session{memory, memory};
    std::array<u8, 16> descriptor{};
    ASSERT_TRUE(
        session.Read(PartitionMemory::Base + 4_KB - 8, descriptor.data(), descriptor.size()));
    EXPECT_TRUE(std::all_of(descriptor.begin(), descriptor.begin() + 8,
                            [](u8 value) { return value == 0x11; }));
    EXPECT_TRUE(std::all_of(descriptor.begin() + 8, descriptor.end(),
                            [](u8 value) { return value == 0x12; }));
    EXPECT_EQ(memory.copies,
              (std::vector<std::pair<VAddr, u64>>{{PartitionMemory::Base, 2 * 4_KB}}));
    memory.Write(PartitionMemory::Base + 0x1100);
    EXPECT_FALSE(session.Seal());
    EXPECT_EQ(memory.watchers, 0u);
}

TEST(SrtSnapshotHostPages, EarlierPageMutationRejectsLaterReadWithoutPublishingOutput) {
    PartitionMemory memory;
    SrtMemoryReadSession session{memory, memory};
    std::array<u8, 8> descriptor{};
    ASSERT_TRUE(session.Read(PartitionMemory::Base + 0x100, descriptor.data(), descriptor.size()));
    memory.Write(PartitionMemory::Base + 0x100);
    descriptor.fill(0xcc);
    EXPECT_FALSE(
        session.Read(PartitionMemory::Base + 0x1100, descriptor.data(), descriptor.size()));
    EXPECT_TRUE(std::ranges::all_of(descriptor, [](u8 value) { return value == 0xcc; }));
    EXPECT_EQ(session.LastFailure(), SrtMemoryReadFailure::EpochInvalidated);
    EXPECT_FALSE(session.Seal());
}

TEST(SrtSnapshotHostPages, SameHostPageWriteAndOverlappingUnmapStillInvalidate) {
    for (const bool unmap : {false, true}) {
        PartitionMemory memory;
        SrtMemoryReadSession session{memory, memory};
        std::array<u8, 8> descriptor{};
        ASSERT_TRUE(
            session.Read(PartitionMemory::Base + 0x100, descriptor.data(), descriptor.size()));
        if (unmap) {
            memory.Unmap(PartitionMemory::Base, 16_KB);
        } else {
            // Retain page precision: a fault is not a decoded store-width proof.
            memory.Write(PartitionMemory::Base + 0x200);
        }
        EXPECT_FALSE(session.Seal());
        EXPECT_EQ(session.LastFailure(), SrtMemoryReadFailure::EpochInvalidated);
        EXPECT_EQ(memory.watchers, 0u);
    }
}

TEST(SrtSnapshotHostPages, SealedCaptureReplaysButRefusesUncapturedSiblingHostPage) {
    PartitionMemory memory;
    SrtMemoryReadSession session{memory, memory};
    std::array<u8, 8> descriptor{};
    ASSERT_TRUE(session.Read(PartitionMemory::Base + 0x100, descriptor.data(), descriptor.size()));
    ASSERT_TRUE(session.Seal());
    memory.Write(PartitionMemory::Base + 0x100);
    ASSERT_TRUE(session.Read(PartitionMemory::Base + 0x100, descriptor.data(), descriptor.size()));
    EXPECT_TRUE(std::ranges::all_of(descriptor, [](u8 value) { return value == 0x11; }));
    descriptor.fill(0xcc);
    EXPECT_FALSE(
        session.Read(PartitionMemory::Base + 0x3100, descriptor.data(), descriptor.size()));
    EXPECT_EQ(session.LastFailure(), SrtMemoryReadFailure::NewPageAfterSeal);
    EXPECT_TRUE(std::ranges::all_of(descriptor, [](u8 value) { return value == 0xcc; }));
}

TEST(SrtSnapshotHostPages, UnrelatedGuestMappingWriteDoesNotInvalidate) {
    PartitionMemory memory;
    SrtMemoryReadSession session{memory, memory};
    std::array<u8, 8> descriptor{};
    ASSERT_TRUE(session.Read(PartitionMemory::Base + 0x100, descriptor.data(), descriptor.size()));
    memory.Write(PartitionMemory::Base + 16_KB);
    EXPECT_TRUE(session.Seal());
    EXPECT_EQ(memory.watchers, 0u);
}

TEST(SrtSnapshotHostPages, IncompleteGuestMappingRejectsBeforeReadingCpuOrGpuBytes) {
    PartitionMemory memory;
    memory.completely_mapped = false;
    SrtMemoryReadSession session{memory, memory};
    std::array<u8, 8> descriptor;
    descriptor.fill(0xcc);
    EXPECT_FALSE(
        session.Read(PartitionMemory::Base + 0x1100, descriptor.data(), descriptor.size()));
    EXPECT_EQ(session.LastFailure(), SrtMemoryReadFailure::UnmappedRange);
    EXPECT_TRUE(memory.copies.empty());
    EXPECT_EQ(memory.mapping_queries,
              (std::vector<std::pair<VAddr, u64>>{{PartitionMemory::Base, 16_KB}}));
    EXPECT_TRUE(std::ranges::all_of(descriptor, [](u8 value) { return value == 0xcc; }));
    EXPECT_FALSE(session.Seal());
    EXPECT_EQ(memory.watchers, 0u);
}

TEST(SrtSnapshotHostPages, InteriorFourKiBHoleRejectsWholeSixteenKiBAdmission) {
    PartitionMemory memory;
    // The requested bytes are in host page 1, but the logical GPU mapping also requires
    // sibling pages 0, 2, and 3. Keep only sibling page 2 absent to exercise that distinction.
    memory.mapped_pages[2] = false;
    SrtMemoryReadSession session{memory, memory};
    std::array<u8, 8> descriptor;
    descriptor.fill(0xcc);

    EXPECT_FALSE(
        session.Read(PartitionMemory::Base + 0x1100, descriptor.data(), descriptor.size()));
    EXPECT_EQ(session.LastFailure(), SrtMemoryReadFailure::UnmappedRange);
    EXPECT_EQ(memory.mapping_queries,
              (std::vector<std::pair<VAddr, u64>>{{PartitionMemory::Base, 16_KB}}));
    EXPECT_TRUE(memory.copies.empty());
    EXPECT_TRUE(std::ranges::all_of(descriptor, [](u8 value) { return value == 0xcc; }));
    EXPECT_FALSE(session.Seal());
    EXPECT_EQ(memory.watchers, 0u);
}

} // namespace
} // namespace VideoCore
