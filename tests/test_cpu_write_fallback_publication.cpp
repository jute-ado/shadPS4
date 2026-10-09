// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/cpu_page_write_snapshot.h"
#include "video_core/buffer_cache/gpu_readback_cpu_publication.h"

namespace {

using VideoCore::CpuPageUploadRange;
using VideoCore::CpuPageWriteTracker;
using VideoCore::TRACKER_BYTES_PER_PAGE;
using Page = std::array<u8, TRACKER_BYTES_PER_PAGE>;

constexpr VAddr PageAddress = 0x20000000;
constexpr VAddr FaultingNeighbor = PageAddress + TRACKER_BYTES_PER_PAGE;
constexpr size_t FirstGpuWord = 0x100;
constexpr size_t PendingCpuWord = 0x200;
constexpr size_t LastGpuWord = 0x300;
constexpr u32 CpuValue = 0x12345678;
constexpr u32 StaleGpuValue = 0x87654321;

void PutWord(Page& page, size_t offset, u32 value) {
    std::memcpy(page.data() + offset, &value, sizeof(value));
}

u32 GetWord(const Page& page, size_t offset) {
    u32 value;
    std::memcpy(&value, page.data() + offset, sizeof(value));
    return value;
}

void PublishSynchronousBatch(CpuPageWriteTracker& tracker, Page& cpu, const Page& gpu,
                             std::span<const VideoCore::GpuReadbackPublicationRange> ranges) {
    const auto prepared = VideoCore::PrepareGpuReadbackPublication(
        tracker, ranges, gpu.size(), [&](VAddr address, std::span<u8> output) {
            std::memcpy(output.data(), cpu.data() + address - PageAddress, output.size());
            return true;
        });
    ASSERT_TRUE(prepared);
    ASSERT_TRUE(VideoCore::PublishPreparedGpuReadback(
        *prepared, gpu,
        [&](const auto& range, std::span<const u8> bytes) {
            std::memcpy(cpu.data() + range.address - PageAddress, bytes.data(), bytes.size());
            return true;
        },
        [&](const auto& range) {
            EXPECT_TRUE(tracker.AcknowledgePublished(
                range.address, std::span{gpu}.subspan(range.source_offset, range.size)));
        }));
}

void PublishCurrentSynchronousIsland(CpuPageWriteTracker& tracker, Page& cpu, const Page& gpu,
                                     size_t offset, size_t size) {
    const VideoCore::GpuReadbackPublicationRange range{PageAddress + offset, size, offset, 0};
    PublishSynchronousBatch(tracker, cpu, gpu, std::span{&range, 1});
}

bool UploadCpuPage(CpuPageWriteTracker& tracker, const Page& cpu, Page& gpu) {
    const bool captured =
        tracker.Consume(PageAddress, cpu, 0, cpu.size(), [&](CpuPageUploadRange range) {
            std::memcpy(gpu.data() + range.offset, cpu.data() + range.offset, range.size);
        });
    if (!captured) {
        // SynchronizeBuffer's existing no-before-image fallback uploads the query.
        std::ranges::copy(cpu, gpu.begin());
    }
    return captured;
}

class CpuWriteFallbackPublication : public testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(tracker.Capture(PageAddress, cpu, PendingCpuWord, sizeof(u32)));
        PutWord(cpu, PendingCpuWord, CpuValue);
        PutWord(gpu, FirstGpuWord, 0x11112222);
        PutWord(gpu, PendingCpuWord, StaleGpuValue);
        PutWord(gpu, LastGpuWord, 0x33334444);

        // Use the real capacity. A and B share a 512-KiB readback window; filler
        // snapshots live elsewhere. The next B fault requests synchronous fallback
        // while A still has an unpublished CPU write. No mapped guest VA is needed
        // by the tracker, which captures bytes from the supplied immutable span.
        constexpr VAddr FillAddress = 0x40000000;
        const Page before{};
        for (size_t i = 1; i < CpuPageWriteTracker::MaximumSnapshots; ++i) {
            ASSERT_TRUE(
                tracker.Capture(FillAddress + i * TRACKER_BYTES_PER_PAGE, before, 0, sizeof(u32)));
        }
        ASSERT_FALSE(tracker.Capture(FaultingNeighbor, before, 0, sizeof(u32)));
        static_assert((PageAddress / (512 * 1024)) == (FaultingNeighbor / (512 * 1024)));
    }

    CpuPageWriteTracker tracker;
    Page cpu{};
    Page gpu{};
};

TEST_F(CpuWriteFallbackPublication, PendingCpuWordSurvivesOverlappingReadbackIsland) {
    PublishCurrentSynchronousIsland(tracker, cpu, gpu, PendingCpuWord, sizeof(u32));

    EXPECT_EQ(GetWord(cpu, PendingCpuWord), CpuValue);
}

TEST_F(CpuWriteFallbackPublication, EarlierGpuIslandDoesNotForgetLaterPendingCpuWord) {
    PublishCurrentSynchronousIsland(tracker, cpu, gpu, FirstGpuWord, sizeof(u32));
    PublishCurrentSynchronousIsland(tracker, cpu, gpu, PendingCpuWord, sizeof(u32));

    EXPECT_EQ(GetWord(cpu, FirstGpuWord), GetWord(gpu, FirstGpuWord));
    EXPECT_EQ(GetWord(cpu, PendingCpuWord), CpuValue);
}

TEST_F(CpuWriteFallbackPublication, CompleteFallbackBatchAndUploadKeepCpuAndGpuResults) {
    // Every GPU-owned island on this page is published, matching the widened
    // synchronous path. This does not invent an unqueried GPU-owned neighbour.
    const std::array<VideoCore::GpuReadbackPublicationRange, 3> ranges{{
        {PageAddress + FirstGpuWord, sizeof(u32), FirstGpuWord, 0},
        {PageAddress + PendingCpuWord, sizeof(u32), PendingCpuWord, 0},
        {PageAddress + LastGpuWord, sizeof(u32), LastGpuWord, 0},
    }};
    PublishSynchronousBatch(tracker, cpu, gpu, ranges);
    static_cast<void>(UploadCpuPage(tracker, cpu, gpu));

    EXPECT_EQ(GetWord(gpu, PendingCpuWord), CpuValue);
    EXPECT_EQ(GetWord(gpu, FirstGpuWord), 0x11112222);
    EXPECT_EQ(GetWord(gpu, LastGpuWord), 0x33334444);
}

TEST_F(CpuWriteFallbackPublication, CpuWordOutsidePublishedIslandsRemainsIntact) {
    // Control: there is no overlapping GPU owner for the pending CPU word.
    PutWord(gpu, PendingCpuWord, 0);
    PublishCurrentSynchronousIsland(tracker, cpu, gpu, FirstGpuWord, sizeof(u32));
    PublishCurrentSynchronousIsland(tracker, cpu, gpu, LastGpuWord, sizeof(u32));
    static_cast<void>(UploadCpuPage(tracker, cpu, gpu));

    EXPECT_EQ(GetWord(gpu, PendingCpuWord), CpuValue);
    EXPECT_EQ(GetWord(gpu, FirstGpuWord), 0x11112222);
    EXPECT_EQ(GetWord(gpu, LastGpuWord), 0x33334444);
}

TEST_F(CpuWriteFallbackPublication, ExistingBeforeImageCanBeReusedAtCapacity) {
    EXPECT_TRUE(tracker.Capture(PageAddress, cpu, PendingCpuWord + sizeof(u32), sizeof(u32)));
    EXPECT_FALSE(tracker.Capture(FaultingNeighbor, cpu, 0, sizeof(u32)));
}

TEST_F(CpuWriteFallbackPublication, PublishedGpuNeighborDoesNotBecomeCpuUploadIntent) {
    PublishCurrentSynchronousIsland(tracker, cpu, gpu, FirstGpuWord, sizeof(u32));
    PutWord(gpu, FirstGpuWord, 0xaabbccdd);
    ASSERT_TRUE(UploadCpuPage(tracker, cpu, gpu));

    EXPECT_EQ(GetWord(gpu, PendingCpuWord), CpuValue);
    EXPECT_EQ(GetWord(gpu, FirstGpuWord), 0xaabbccdd);
}

TEST(CpuWriteFallbackPublicationControl, KnownCpuWordCanChangeAgainDuringPublication) {
    CpuPageWriteTracker tracker;
    Page cpu{};
    Page gpu{};
    ASSERT_TRUE(tracker.Capture(PageAddress, cpu, PendingCpuWord, sizeof(u32)));
    PutWord(cpu, PendingCpuWord, CpuValue);
    const VideoCore::GpuReadbackPublicationRange range{PageAddress, cpu.size(), 0, 0};
    const auto prepared = VideoCore::PrepareGpuReadbackPublication(
        tracker, std::span{&range, 1}, gpu.size(), [&](VAddr address, std::span<u8> output) {
            std::memcpy(output.data(), cpu.data() + address - PageAddress, output.size());
            return true;
        });
    ASSERT_TRUE(prepared);
    ASSERT_TRUE(VideoCore::PublishPreparedGpuReadback(
        *prepared, gpu,
        [&](const auto& published, std::span<const u8> bytes) {
            PutWord(cpu, PendingCpuWord, 0xaabbccdd);
            // The public callback can safely inspect/update tracker metadata: its lock is gone.
            EXPECT_TRUE(tracker.Capture(PageAddress, cpu, PendingCpuWord, sizeof(u32)));
            std::memcpy(cpu.data() + published.address - PageAddress, bytes.data(), bytes.size());
            return true;
        },
        [&](const auto& published) {
            tracker.DiscardPublished(published.address, published.size);
        }));
    EXPECT_EQ(GetWord(cpu, PendingCpuWord), 0xaabbccdd);
}

TEST(CpuWriteFallbackPublicationControl, GpuOnlyPagesNeverInvokeTheCpuReader) {
    CpuPageWriteTracker tracker;
    const VideoCore::GpuReadbackPublicationRange range{PageAddress, 2 * TRACKER_BYTES_PER_PAGE, 0,
                                                       7};
    const auto prepared = VideoCore::PrepareGpuReadbackPublication(
        tracker, std::span{&range, 1}, range.size, [](VAddr, std::span<u8>) {
            ADD_FAILURE() << "GPU-only precise pages must not recursively fault a CPU read";
            return false;
        });
    ASSERT_TRUE(prepared);
    ASSERT_EQ(prepared->size(), 1);
    EXPECT_EQ(prepared->front().address, range.address);
    EXPECT_EQ(prepared->front().size, range.size);
    EXPECT_EQ(prepared->front().owner_index, 7);
}

TEST(CpuWriteFallbackPublicationControl, ExactSubwordRangeProtectsOnlyItsCpuIntersection) {
    CpuPageWriteTracker tracker;
    Page cpu{};
    ASSERT_TRUE(tracker.Capture(PageAddress, cpu, PendingCpuWord, sizeof(u32)));
    const VideoCore::GpuReadbackPublicationRange range{PageAddress + PendingCpuWord - 1, 6, 10, 0};
    const auto prepared = VideoCore::PrepareGpuReadbackPublication(
        tracker, std::span{&range, 1}, 16, [&](VAddr address, std::span<u8> output) {
            EXPECT_EQ(address, PageAddress + PendingCpuWord - 4);
            EXPECT_EQ(output.size(), 12);
            std::memcpy(output.data(), cpu.data() + address - PageAddress, output.size());
            return true;
        });
    ASSERT_TRUE(prepared);
    ASSERT_EQ(prepared->size(), 2);
    EXPECT_EQ((*prepared)[0].address, range.address);
    EXPECT_EQ((*prepared)[0].size, 1);
    EXPECT_EQ((*prepared)[0].source_offset, 10);
    EXPECT_EQ((*prepared)[1].address, PageAddress + PendingCpuWord + 4);
    EXPECT_EQ((*prepared)[1].size, 1);
    EXPECT_EQ((*prepared)[1].source_offset, 15);
}

TEST(CpuWriteFallbackPublicationControl, FailedLaterCaptureDoesNotPublishEarlierIsland) {
    CpuPageWriteTracker tracker;
    Page cpu{};
    ASSERT_TRUE(tracker.Capture(PageAddress, cpu, PendingCpuWord, sizeof(u32)));
    const std::array<VideoCore::GpuReadbackPublicationRange, 2> ranges{{
        {FaultingNeighbor, sizeof(u32), 0, 0},
        {PageAddress + PendingCpuWord, sizeof(u32), 4, 1},
    }};
    int reads = 0;
    const auto prepared =
        VideoCore::PrepareGpuReadbackPublication(tracker, ranges, 8, [&](VAddr, std::span<u8>) {
            ++reads;
            return false;
        });
    EXPECT_FALSE(prepared);
    EXPECT_EQ(reads, 1);
    EXPECT_TRUE(tracker.Consume(PageAddress, cpu, 0, cpu.size(), [](auto) {}));
}

TEST(CpuWriteFallbackPublicationControl, FailedPublicationDoesNotReconcileUnwrittenBytes) {
    Page gpu{};
    const std::array<VideoCore::GpuReadbackPublicationRange, 3> prepared{{
        {PageAddress, 4, 0, 0},
        {PageAddress + 8, 4, 8, 0},
        {PageAddress + 16, 4, 16, 0},
    }};
    int writes = 0;
    std::vector<VAddr> reconciled;
    EXPECT_FALSE(VideoCore::PublishPreparedGpuReadback(
        prepared, gpu, [&](const auto&, auto) { return ++writes != 2; },
        [&](const auto& range) { reconciled.push_back(range.address); }));
    EXPECT_EQ(writes, 2);
    EXPECT_EQ(reconciled, std::vector<VAddr>{PageAddress});
}

TEST(CpuWriteFallbackPublicationControl, InvalidLaterRangeIsRejectedBeforeAnyCallback) {
    CpuPageWriteTracker tracker;
    Page gpu{};
    const std::array<VideoCore::GpuReadbackPublicationRange, 2> ranges{{
        {PageAddress, 4, 0, 0},
        {std::numeric_limits<VAddr>::max() - 1, 4, 4, 0},
    }};
    const auto prepared =
        VideoCore::PrepareGpuReadbackPublication(tracker, ranges, gpu.size(), [](VAddr, auto) {
            ADD_FAILURE();
            return true;
        });
    EXPECT_FALSE(prepared);
    EXPECT_FALSE(VideoCore::PublishPreparedGpuReadback(
        ranges, gpu,
        [](const auto&, auto) {
            ADD_FAILURE();
            return true;
        },
        [](const auto&) { ADD_FAILURE(); }));
}

TEST(CpuWriteFallbackPublicationControl, ChangedUnforcedWordAndCrossPageRangesRemainProtected) {
    CpuPageWriteTracker tracker;
    std::array<Page, 2> cpu{};
    ASSERT_TRUE(tracker.Capture(PageAddress, cpu[0], 0, 4));
    ASSERT_TRUE(tracker.Capture(FaultingNeighbor, cpu[1], 0, 4));
    PutWord(cpu[0], TRACKER_BYTES_PER_PAGE - 4, CpuValue);
    const VideoCore::GpuReadbackPublicationRange range{FaultingNeighbor - 5, 10, 0, 3};
    const auto prepared = VideoCore::PrepareGpuReadbackPublication(
        tracker, std::span{&range, 1}, 10, [&](VAddr address, std::span<u8> output) {
            const auto index = (address - PageAddress) / TRACKER_BYTES_PER_PAGE;
            const auto offset = (address - PageAddress) % TRACKER_BYTES_PER_PAGE;
            // This reader is outside the tracker lock, including across page boundaries.
            EXPECT_TRUE(
                tracker.Capture(PageAddress + index * TRACKER_BYTES_PER_PAGE, cpu[index], 0, 4));
            std::memcpy(output.data(), cpu[index].data() + offset, output.size());
            return true;
        });
    ASSERT_TRUE(prepared);
    ASSERT_EQ(prepared->size(), 2);
    EXPECT_EQ((*prepared)[0].address, FaultingNeighbor - 5);
    EXPECT_EQ((*prepared)[0].size, 1);
    EXPECT_EQ((*prepared)[1].address, FaultingNeighbor + 4);
    EXPECT_EQ((*prepared)[1].size, 1);
    EXPECT_EQ((*prepared)[1].source_offset, 9);
}

TEST(CpuWriteFallbackPublicationControl, MappingUploadWithoutSnapshotNeedsNoCpuRead) {
    CpuPageWriteTracker tracker;
    tracker.ForceMappingUpload(PageAddress, TRACKER_BYTES_PER_PAGE);
    const VideoCore::GpuReadbackPublicationRange range{PageAddress + 9, 11, 0, 0};
    const auto prepared = VideoCore::PrepareGpuReadbackPublication(tracker, std::span{&range, 1},
                                                                   11, [](VAddr, auto) {
                                                                       ADD_FAILURE();
                                                                       return false;
                                                                   });
    ASSERT_TRUE(prepared);
    EXPECT_TRUE(prepared->empty());
}

TEST(CpuWriteFallbackPublicationControl, SourceBoundsAreCheckedBeforePreparationAndPublication) {
    CpuPageWriteTracker tracker;
    const VideoCore::GpuReadbackPublicationRange range{PageAddress, 4, 2, 0};
    const Page source{};
    EXPECT_FALSE(
        VideoCore::PrepareGpuReadbackPublication(tracker, std::span{&range, 1}, 4, [](VAddr, auto) {
            ADD_FAILURE();
            return true;
        }));
    EXPECT_FALSE(VideoCore::PublishPreparedGpuReadback(
        std::span{&range, 1}, std::span{source}.first(4),
        [](const auto&, auto) {
            ADD_FAILURE();
            return true;
        },
        [](const auto&) { ADD_FAILURE(); }));
}

TEST(CpuWriteFallbackPublicationControl, ProductionPublishesBeforeExactOwnershipCleanup) {
    std::ifstream stream{SHADPS4_INVALIDATION_BUFFER_SOURCE_PATH};
    ASSERT_TRUE(stream);
    const std::string source{std::istreambuf_iterator<char>{stream}, {}};
    const auto begin = source.find("void BufferCache::DownloadBufferMemory(");
    const auto end = source.find("bool BufferCache::DownloadBufferMemorySnapshot(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_EQ(body.find("cpu_page_write_tracker.Discard("), std::string::npos);
    EXPECT_EQ(body.find("gpu_modified_ranges.Subtract("), std::string::npos);
    const auto prepare = body.find("PrepareGpuReadbackPublication(");
    const auto publish = body.find("PublishPreparedGpuReadback(");
    const auto write = body.find("return memory->TryWriteBacking(");
    const auto reconcile = body.find("ReconcilePublishedGpuOwnerRanges(");
    ASSERT_NE(prepare, std::string::npos);
    ASSERT_NE(publish, std::string::npos);
    ASSERT_NE(write, std::string::npos);
    ASSERT_NE(reconcile, std::string::npos);
    EXPECT_LT(prepare, publish);
    EXPECT_LT(publish, write);
    EXPECT_LT(write, reconcile);
    EXPECT_NE(body.find("completed && published"), std::string::npos);
    EXPECT_NE(body.find("cpu_page_write_tracker.AcknowledgePublished("), std::string::npos);
}

TEST(CpuWriteFallbackPublicationControl, AcknowledgementChangesOnlyPublishedBaselineBytes) {
    Page before{};
    before.fill(0x11);
    VideoCore::CpuPageWriteSnapshot snapshot{before, PendingCpuWord, 4};
    const std::array<u8, 2> published{0x22, 0x33};
    ASSERT_TRUE(snapshot.AcknowledgePublished(FirstGpuWord + 1, published));
    EXPECT_EQ(snapshot.DiagnosticBeforeWord(FirstGpuWord / 4), 0x11332211u);
    EXPECT_EQ(snapshot.DiagnosticBeforeWord(FirstGpuWord / 4 - 1), 0x11111111u);
    EXPECT_TRUE(snapshot.DiagnosticForcedWord(PendingCpuWord / 4));
    EXPECT_FALSE(snapshot.AcknowledgePublished(TRACKER_BYTES_PER_PAGE - 1, published));
    EXPECT_EQ(snapshot.DiagnosticBeforeWord((TRACKER_BYTES_PER_PAGE - 4) / 4), 0x11111111u);
}

TEST(CpuWriteFallbackPublicationControl, EmptyAcknowledgementIsValidOnlyAtContainedOffsets) {
    Page before{};
    VideoCore::CpuPageWriteSnapshot snapshot{before, PendingCpuWord, 4};
    EXPECT_TRUE(snapshot.AcknowledgePublished(0, {}));
    EXPECT_TRUE(snapshot.AcknowledgePublished(TRACKER_BYTES_PER_PAGE, {}));
    EXPECT_FALSE(snapshot.AcknowledgePublished(TRACKER_BYTES_PER_PAGE + 1, {}));
    EXPECT_TRUE(snapshot.DiagnosticForcedWord(PendingCpuWord / 4));
    CpuPageWriteTracker tracker;
    EXPECT_TRUE(tracker.AcknowledgePublished(PageAddress, {}));
}

TEST(CpuWriteFallbackPublicationControl, FailedWriterDoesNotAcknowledgeBaseline) {
    CpuPageWriteTracker tracker;
    Page cpu{};
    Page gpu{};
    ASSERT_TRUE(tracker.Capture(PageAddress, cpu, PendingCpuWord, sizeof(u32)));
    PutWord(gpu, FirstGpuWord, StaleGpuValue);
    const VideoCore::GpuReadbackPublicationRange range{PageAddress + FirstGpuWord, 4, FirstGpuWord,
                                                       0};
    const auto prepared = VideoCore::PrepareGpuReadbackPublication(
        tracker, std::span{&range, 1}, gpu.size(), [&](VAddr address, std::span<u8> output) {
            std::memcpy(output.data(), cpu.data() + address - PageAddress, output.size());
            return true;
        });
    ASSERT_TRUE(prepared);
    EXPECT_FALSE(VideoCore::PublishPreparedGpuReadback(
        *prepared, gpu, [](const auto&, auto) { return false; },
        [&](const auto& published) {
            EXPECT_TRUE(tracker.AcknowledgePublished(
                published.address,
                std::span{gpu}.subspan(published.source_offset, published.size)));
        }));
    VideoCore::CpuWriteFocusDiagnostic diagnostic;
    Page output{};
    ASSERT_TRUE(tracker.ApplyPendingWrites(PageAddress, cpu, PageAddress, output, &diagnostic,
                                           PageAddress + FirstGpuWord, 4));
    EXPECT_EQ(diagnostic.before[0], 0u);
    EXPECT_EQ(diagnostic.changed_word_mask, 0u);
}

TEST(CpuWriteFallbackPublicationControl,
     AcknowledgementRetainsNewExplicitCpuWriteAndMappingIntent) {
    CpuPageWriteTracker tracker;
    Page cpu{};
    ASSERT_TRUE(tracker.Capture(PageAddress, cpu, PendingCpuWord, 4));
    PutWord(cpu, FirstGpuWord, StaleGpuValue);
    // This mark can arrive after a publication plan was prepared. Acknowledgement must not clear
    // it.
    ASSERT_TRUE(tracker.Capture(PageAddress, cpu, FirstGpuWord, 4));
    ASSERT_TRUE(tracker.AcknowledgePublished(PageAddress + FirstGpuWord,
                                             std::span{cpu}.subspan(FirstGpuWord, 4)));
    std::vector<CpuPageUploadRange> uploads;
    ASSERT_TRUE(tracker.Consume(PageAddress, cpu, 0, cpu.size(),
                                [&](auto range) { uploads.push_back(range); }));
    EXPECT_NE(std::ranges::find(uploads, CpuPageUploadRange{FirstGpuWord, 4}), uploads.end());

    tracker.ForceMappingUpload(PageAddress, cpu.size());
    ASSERT_TRUE(tracker.AcknowledgePublished(PageAddress, cpu));
    const VideoCore::GpuReadbackPublicationRange range{PageAddress, cpu.size(), 0, 0};
    const auto prepared = VideoCore::PrepareGpuReadbackPublication(tracker, std::span{&range, 1},
                                                                   cpu.size(), [](VAddr, auto) {
                                                                       ADD_FAILURE();
                                                                       return false;
                                                                   });
    ASSERT_TRUE(prepared);
    EXPECT_TRUE(prepared->empty());
}

TEST(CpuWriteFallbackPublicationControl, GpuOnlyPageUsesTheDownloadedBytes) {
    CpuPageWriteTracker tracker;
    Page cpu{};
    Page gpu{};
    PutWord(gpu, FirstGpuWord, 0x11112222);
    PutWord(gpu, LastGpuWord, 0x33334444);
    PublishCurrentSynchronousIsland(tracker, cpu, gpu, FirstGpuWord, sizeof(u32));
    PublishCurrentSynchronousIsland(tracker, cpu, gpu, LastGpuWord, sizeof(u32));
    EXPECT_FALSE(UploadCpuPage(tracker, cpu, gpu));

    EXPECT_EQ(GetWord(gpu, FirstGpuWord), 0x11112222);
    EXPECT_EQ(GetWord(gpu, LastGpuWord), 0x33334444);
}

} // namespace
