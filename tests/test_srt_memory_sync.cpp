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
            if (!owner || !owner->epoch_valid || owner->epoch_sealed || owner->reject_page_copy ||
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
    SrtSnapshotEpochCoordinator coordinator;
    std::vector<std::pair<VAddr, u64>> mapping_queries;
    std::vector<std::pair<VAddr, u64>> copies;
    u32 watchers{};
    u32 epochs{};
    bool completely_mapped = true;

    PartitionMemory() {
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
        return completely_mapped;
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
              (std::vector<std::pair<VAddr, u64>>{{PartitionMemory::Base, 4_KB},
                                                  {PartitionMemory::Base + 4_KB, 4_KB}}));
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

} // namespace
} // namespace VideoCore
