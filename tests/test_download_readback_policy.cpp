// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/cpu_page_write_snapshot.h"
#include "video_core/buffer_cache/download_readback_policy.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/buffer_cache/srt_snapshot_capture.h"

namespace VideoCore {
namespace {

std::string ReadText(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(DownloadReadbackPolicy, NonCoherentReservationsAreAtomIsolated) {
    const auto reservation = PlanDownloadReservation(65, 16, 64, false);

    EXPECT_EQ(reservation.size, 128U);
    EXPECT_EQ(reservation.alignment, 64U);
}

TEST(DownloadReadbackPolicy, CoherentReservationsKeepRequestedShape) {
    const auto reservation = PlanDownloadReservation(65, 16, 64, true);

    EXPECT_EQ(reservation.size, 65U);
    EXPECT_EQ(reservation.alignment, 16U);
}

TEST(DownloadReadbackPolicy, ExactReadbackClipsTrackerPageToRequestedBytes) {
    const auto range = ClipDownloadRange(0x242710000ULL, 0x1000, 0x242710318ULL, 8);

    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->address, 0x242710318ULL);
    EXPECT_EQ(range->size, 8U);
}

TEST(DownloadReadbackPolicy, ExactReadbackRejectsDisjointTrackerRange) {
    EXPECT_FALSE(ClipDownloadRange(0x1000, 0x1000, 0x3000, 8).has_value());
}

TEST(DownloadReadbackPolicy, FocusedOverlayDiagnosticsBypassExhaustedNoiseBudget) {
    struct Diagnostic {
        bool focus_intersects;
        int id;
    };
    const std::array diagnostics{
        Diagnostic{false, 1},
        Diagnostic{false, 2},
        Diagnostic{true, 3},
        Diagnostic{false, 4},
    };
    std::vector<int> emitted;
    bool budget_available = false;
    ForEachSnapshotOverlayDiagnostic(
        diagnostics, [&](const Diagnostic& diagnostic) { emitted.push_back(diagnostic.id); },
        [&] {
            const bool available = budget_available;
            budget_available = false;
            return available;
        });

    EXPECT_EQ(emitted, (std::vector<int>{3}));
}

TEST(DownloadReadbackPolicy, OwnedRangesResolveEachDirtyIntervalToItsSourceBuffer) {
    struct Owner {
        int id;
        VAddr base;
        u64 size;
    };
    const std::map<int, Owner> owners{
        {1, Owner{1, 0x1000, 0x100}},
        {2, Owner{2, 0x1080, 0x100}},
    };
    const std::array dirty{
        std::tuple{VAddr{0x1040}, VAddr{0x1080}, 1},
        std::tuple{VAddr{0x1080}, VAddr{0x10c0}, 2},
    };
    std::vector<PlannedDownloadRange<Owner>> planned;
    u64 total_size = 0;
    const bool valid = PlanOwnedDownloadRanges<Owner>(
        [&](auto&& callback) {
            for (const auto& [begin, end, owner_id] : dirty) {
                callback(begin, end, owner_id);
            }
        },
        [&](int owner_id) -> std::optional<DownloadRangeOwner<Owner>> {
            const auto it = owners.find(owner_id);
            if (it == owners.end()) {
                return std::nullopt;
            }
            return DownloadRangeOwner<Owner>{it->second, it->second.base, it->second.size};
        },
        [&](const auto& range) { planned.push_back(range); }, total_size, 0x1000);

    ASSERT_TRUE(valid);
    ASSERT_EQ(planned.size(), 2U);
    EXPECT_EQ(planned[0].owner.id, 1);
    EXPECT_EQ(planned[0].address, 0x1040U);
    EXPECT_EQ(planned[0].source_offset, 0x40U);
    EXPECT_EQ(planned[0].staging_offset, 0U);
    EXPECT_EQ(planned[1].owner.id, 2);
    EXPECT_EQ(planned[1].address, 0x1080U);
    EXPECT_EQ(planned[1].source_offset, 0U);
    EXPECT_EQ(planned[1].staging_offset, 0x40U);
    EXPECT_EQ(total_size, 0x80U);
}

TEST(DownloadReadbackPolicy, OwnedRangesFailClosedWhenSourceBufferDoesNotContainDirtyBytes) {
    const std::array dirty{
        std::tuple{VAddr{0x10f0}, VAddr{0x1110}, 7},
    };
    std::vector<PlannedDownloadRange<int>> planned;
    u64 total_size = 0;
    const bool valid = PlanOwnedDownloadRanges<int>(
        [&](auto&& callback) {
            for (const auto& [begin, end, owner] : dirty) {
                callback(begin, end, owner);
            }
        },
        [](int) -> std::optional<DownloadRangeOwner<int>> {
            return DownloadRangeOwner<int>{7, 0x1000, 0xe0};
        },
        [&](const auto& range) { planned.push_back(range); }, total_size, 0x1000);

    EXPECT_FALSE(valid);
    EXPECT_TRUE(planned.empty());
    EXPECT_EQ(total_size, 0U);
}

TEST(DownloadReadbackPolicy, SnapshotOwnerValidationChecksEveryCachingPage) {
    const std::array page_owners{7, 7, 7, 7};
    const auto owner_at = [&](VAddr address) {
        return page_owners[(address >> 12) & (page_owners.size() - 1)];
    };

    EXPECT_TRUE(AllPagesOwnedBy(7, 0x1010, 0x2fe0, 0x1000, owner_at));
}

TEST(DownloadReadbackPolicy, SnapshotOwnerValidationRejectsAReboundMiddlePage) {
    const std::array page_owners{7, 7, 9, 7};
    const auto owner_at = [&](VAddr address) {
        return page_owners[(address >> 12) & (page_owners.size() - 1)];
    };

    EXPECT_FALSE(AllPagesOwnedBy(7, 0x1010, 0x2fe0, 0x1000, owner_at));
}

TEST(DownloadReadbackPolicy, CpuConsumptionFollowsFinishThenInvalidation) {
    std::vector<std::string_view> operations;

    const bool completed = CompleteDownloadReadback([&] { operations.emplace_back("finish"); },
                                                    [&] {
                                                        operations.emplace_back("invalidate");
                                                        return true;
                                                    },
                                                    [&] { operations.emplace_back("consume"); });

    EXPECT_TRUE(completed);
    EXPECT_EQ(operations, (std::vector<std::string_view>{"finish", "invalidate", "consume"}));
}

TEST(DownloadReadbackPolicy, PartialPublicationKeepsUnpublishedGpuNeighbourOwned) {
    constexpr VAddr Page = 0x4000;
    RangeMap<int> dirty_ranges;
    dirty_ranges.Add(Page + 0x100, 4, 1);
    dirty_ranges.Add(Page + 0x200, 4, 1);
    const std::array published{DownloadRange{Page + 0x100, 4}};
    std::vector<VAddr> unmarked_pages;

    ASSERT_TRUE(ReconcilePublishedGpuRanges(dirty_ranges, std::span<const DownloadRange>{published},
                                            TRACKER_BYTES_PER_PAGE,
                                            [&](VAddr page) { unmarked_pages.push_back(page); }));
    EXPECT_TRUE(dirty_ranges.Intersects(Page + 0x200, 4));
    EXPECT_TRUE(unmarked_pages.empty());
}

TEST(DownloadReadbackPolicy, StaleReadbackDoesNotClearReplacementOwner) {
    constexpr VAddr Page = 0x8000;
    constexpr VAddr Address = Page + 0x100;
    constexpr int OriginalOwner = 1;
    constexpr int ReplacementOwner = 2;
    RangeMap<int> dirty_ranges;
    // The readback was captured while OriginalOwner produced the bytes. A replacement can
    // become the current owner before completion, even when the guest address is unchanged.
    dirty_ranges.Add(Address, 4, OriginalOwner);
    dirty_ranges.Assign(Address, 4, ReplacementOwner);
    const std::array published{
        OwnedDownloadRange<int>{OriginalOwner, Address, 4},
    };
    std::vector<VAddr> unmarked_pages;

    ASSERT_TRUE(ReconcilePublishedGpuOwnerRanges(
        dirty_ranges, std::span<const OwnedDownloadRange<int>>{published}, TRACKER_BYTES_PER_PAGE,
        [&](VAddr page) { unmarked_pages.push_back(page); }));

    // Completion from OriginalOwner must not clear the still-dirty ReplacementOwner range.
    EXPECT_TRUE(dirty_ranges.Intersects(Address, 4));
    EXPECT_TRUE(unmarked_pages.empty());
}

TEST(DownloadReadbackPolicy, StaleReadbackOwnerRejectsBackingPublication) {
    constexpr VAddr Address = 0x9100;
    RangeMap<int> dirty_ranges;
    dirty_ranges.Add(Address, 4, 2);
    const std::array captured{
        OwnedDownloadRange<int>{1, Address, 4},
    };

    EXPECT_FALSE(AreGpuReadbackOwnersCompatible(
        dirty_ranges, std::span<const OwnedDownloadRange<int>>{captured}));
}

TEST(DownloadReadbackPolicy, CompletePublicationReleasesGpuPageAfterDirtyRangesAreRemoved) {
    constexpr VAddr Page = 0x4000;
    RangeMap<int> dirty_ranges;
    dirty_ranges.Add(Page + 0x100, 4, 1);
    const std::array published{DownloadRange{Page + 0x100, 4}};
    std::vector<VAddr> unmarked_pages;

    ASSERT_TRUE(ReconcilePublishedGpuRanges(dirty_ranges, std::span<const DownloadRange>{published},
                                            TRACKER_BYTES_PER_PAGE,
                                            [&](VAddr page) { unmarked_pages.push_back(page); }));
    EXPECT_FALSE(dirty_ranges.Intersects(Page, TRACKER_BYTES_PER_PAGE));
    EXPECT_EQ(unmarked_pages, std::vector<VAddr>{Page});
}

TEST(DownloadReadbackPolicy, FailedInvalidationPreventsCpuConsumption) {
    std::vector<std::string_view> operations;

    const bool completed = CompleteDownloadReadback([&] { operations.emplace_back("finish"); },
                                                    [&] {
                                                        operations.emplace_back("invalidate");
                                                        return false;
                                                    },
                                                    [&] { operations.emplace_back("consume"); });

    EXPECT_FALSE(completed);
    EXPECT_EQ(operations, (std::vector<std::string_view>{"finish", "invalidate"}));
}

TEST(DownloadReadbackPolicy, PrivateSnapshotComposesCpuAndExactGpuBytesAfterCompletion) {
    std::array<u8, 8> output{};
    const std::array<u8, 8> cpu{0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17};
    const std::array<u8, 4> gpu{0xa1, 0xa2, 0xb1, 0xb2};
    const std::array overlays{
        MemorySnapshotOverlay{.output_offset = 1, .download_offset = 0, .size = 2},
        MemorySnapshotOverlay{.output_offset = 5, .download_offset = 2, .size = 2},
    };
    std::vector<std::string_view> operations;

    const bool completed = CompletePrivateSnapshotReadback(
        [&] { operations.emplace_back("finish"); },
        [&] {
            operations.emplace_back("invalidate");
            return true;
        },
        [&] {
            operations.emplace_back("cpu");
            output = cpu;
            return true;
        },
        [&] {
            operations.emplace_back("gpu");
            return ApplyMemorySnapshotOverlays(output, gpu, overlays);
        });

    EXPECT_TRUE(completed);
    EXPECT_EQ(operations, (std::vector<std::string_view>{"finish", "invalidate", "cpu", "gpu"}));
    EXPECT_EQ(output, (std::array<u8, 8>{0x10, 0xa1, 0xa2, 0x13, 0x14, 0xb1, 0xb2, 0x17}));
}

TEST(DownloadReadbackPolicy, InvalidOverlayLeavesDestinationUnchanged) {
    std::array<u8, 4> output{1, 2, 3, 4};
    const std::array<u8, 2> gpu{5, 6};
    const std::array overlays{
        MemorySnapshotOverlay{.output_offset = 3, .download_offset = 0, .size = 2},
    };

    EXPECT_FALSE(ApplyMemorySnapshotOverlays(output, gpu, overlays));
    EXPECT_EQ(output, (std::array<u8, 4>{1, 2, 3, 4}));
}

TEST(DownloadReadbackPolicy, OverlayDownloadOffsetsAreRelativeToTheMappedView) {
    // StreamBuffer::Map returns mapped_data.data() + offset. The overlay offsets intentionally
    // remain relative to that returned pointer, while Vulkan copy destinations include offset.
    std::array<u8, 64> staging_storage{};
    constexpr size_t staging_offset = 24;
    auto mapped_view = std::span<u8>{staging_storage}.subspan(staging_offset, 32);
    for (size_t i = 0; i < mapped_view.size(); ++i) {
        mapped_view[i] = static_cast<u8>(0xa0 + i);
    }

    std::array<u8, 8> output{0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc};
    const std::array overlays{
        MemorySnapshotOverlay{.output_offset = 1, .download_offset = 4, .size = 4},
    };

    ASSERT_TRUE(ApplyMemorySnapshotOverlays(output, mapped_view, overlays));
    EXPECT_EQ(output, (std::array<u8, 8>{0xcc, 0xa4, 0xa5, 0xa6, 0xa7, 0xcc, 0xcc, 0xcc}));
    EXPECT_EQ(staging_storage[staging_offset + 4], 0xa4);
}

TEST(DownloadReadbackPolicy, PendingCpuDescriptorWinsOverOlderGpuPayloadWithoutLosingNeighbours) {
    constexpr VAddr PageAddress = 0x4000;
    constexpr size_t DescriptorOffset = 0x30;
    constexpr size_t DescriptorSize = 16;
    std::array<u8, TRACKER_BYTES_PER_PAGE> cpu{};
    cpu.fill(0x11);
    CpuPageWriteTracker writes;
    ASSERT_TRUE(writes.Capture(PageAddress, cpu, DescriptorOffset, DescriptorSize));
    std::fill_n(cpu.begin() + DescriptorOffset, DescriptorSize, 0x22);
    const auto cpu_after_write = cpu;

    // Disabled readbacks preserve the CPU before-image, then allow this new write while
    // the exact GPU-owned range still includes the descriptor and its unchanged neighbours.
    std::array<u8, TRACKER_BYTES_PER_PAGE> gpu{};
    gpu.fill(0x77);
    std::array<u8, TRACKER_BYTES_PER_PAGE> output{};
    const std::array overlays{
        MemorySnapshotOverlay{.output_offset = 0, .download_offset = 0, .size = gpu.size()},
    };
    ASSERT_TRUE(ReadAuthoritativeMemorySnapshot(
        true, true, [] { return false; },
        [&] {
            return CompletePrivateSnapshotReadback(
                [] {}, [] { return true; },
                [&] {
                    output = cpu;
                    return true;
                },
                [&] {
                    return ApplyMemorySnapshotOverlays(output, gpu, overlays) &&
                           writes.ApplyPendingWrites(PageAddress, cpu_after_write, PageAddress,
                                                     output);
                });
        }));

    EXPECT_TRUE(std::ranges::all_of(std::span{output}.subspan(DescriptorOffset, DescriptorSize),
                                    [](u8 value) { return value == 0x22; }));
    EXPECT_EQ(output[DescriptorOffset - 1], 0x77);
    EXPECT_EQ(output[DescriptorOffset + DescriptorSize], 0x77);
    EXPECT_EQ(cpu, cpu_after_write);
    std::vector<CpuPageUploadRange> pending;
    ASSERT_TRUE(writes.Consume(PageAddress, cpu, 0, cpu.size(),
                               [&](auto range) { pending.push_back(range); }));
    EXPECT_EQ(pending, (std::vector{CpuPageUploadRange{DescriptorOffset, DescriptorSize}}));
}

TEST(DownloadReadbackPolicy, ExactGpuWritesUsePrivateGpuReadback) {
    std::vector<std::string_view> operations;
    const bool completed = ReadAuthoritativeMemorySnapshot(
        true, true,
        [&] {
            operations.emplace_back("cpu");
            return true;
        },
        [&] {
            operations.emplace_back("gpu");
            return true;
        });

    EXPECT_TRUE(completed);
    EXPECT_EQ(operations, (std::vector<std::string_view>{"gpu"}));
}

TEST(DownloadReadbackPolicy, PendingGpuOwnershipRefusesStaleCpuBacking) {
    std::vector<std::string_view> operations;
    const bool completed = ReadAuthoritativeMemorySnapshot(
        false, true,
        [&] {
            operations.emplace_back("cpu");
            return true;
        },
        [&] {
            operations.emplace_back("gpu");
            return true;
        });

    EXPECT_FALSE(completed);
    EXPECT_TRUE(operations.empty());
}

TEST(DownloadReadbackPolicy, CpuOwnedBytesUseGuestBacking) {
    std::vector<std::string_view> operations;
    const bool completed = ReadAuthoritativeMemorySnapshot(
        false, false,
        [&] {
            operations.emplace_back("cpu");
            return true;
        },
        [&] {
            operations.emplace_back("gpu");
            return true;
        });

    EXPECT_TRUE(completed);
    EXPECT_EQ(operations, (std::vector<std::string_view>{"cpu"}));
}

TEST(DownloadReadbackPolicy, ProductionInvalidatesAfterCompletionBeforeCpuConsumption) {
    const auto buffer = ReadText(SHADPS4_BUFFER_SOURCE_PATH);
    const auto cache = ReadText(SHADPS4_BUFFER_CACHE_SOURCE_PATH);

    EXPECT_NE(buffer.find("PlanDownloadReservation(size, alignment"), std::string::npos);
    EXPECT_NE(buffer.find("bool Buffer::InvalidateMappedRange("), std::string::npos);

    const auto write_data = cache.find("const auto write_data =");
    const auto finish = cache.find("scheduler.Finish()", write_data);
    const auto invalidate = cache.find("download_buffer.InvalidateMappedRange", finish);
    const auto consume = cache.find("write_data", invalidate);
    ASSERT_NE(write_data, std::string::npos);
    ASSERT_NE(finish, std::string::npos);
    ASSERT_NE(invalidate, std::string::npos);
    ASSERT_NE(consume, std::string::npos);
    EXPECT_LT(finish, invalidate);
    EXPECT_LT(invalidate, consume);
}

TEST(SrtSnapshotCapture, SingleTrackerPageFitsInlineCaptureBudget) {
    const auto capture = PlanCpuSnapshotCapture(0x401, TRACKER_BYTES_PER_PAGE);

    ASSERT_TRUE(capture.has_value());
    EXPECT_LE(capture->size, SrtSnapshotCaptureInlineBytes);

    SrtSnapshotCaptureBuffer storage(static_cast<std::size_t>(capture->size));
    EXPECT_EQ(storage.size(), capture->size);
    EXPECT_GE(storage.capacity(), SrtSnapshotCaptureInlineBytes);
}

TEST(SrtSnapshotCapture, LargerPublicSnapshotStillUsesCompleteDynamicStorage) {
    const auto capture = PlanCpuSnapshotCapture(0x401, TRACKER_BYTES_PER_PAGE * 2);

    ASSERT_TRUE(capture.has_value());
    ASSERT_GT(capture->size, SrtSnapshotCaptureInlineBytes);

    SrtSnapshotCaptureBuffer storage(static_cast<std::size_t>(capture->size));
    EXPECT_EQ(storage.size(), capture->size);
    EXPECT_GE(storage.capacity(), storage.size());
}

TEST(SrtSnapshotCapture, ProductionUsesInlineCaptureBufferWithoutTruncatingVectors) {
    const auto source = ReadText(SHADPS4_BUFFER_CACHE_SOURCE_PATH);

    EXPECT_NE(source.find("SrtSnapshotCaptureBuffer cpu_capture"), std::string::npos);
    EXPECT_EQ(source.find("std::vector<u8> cpu_capture"), std::string::npos);
}

} // namespace
} // namespace VideoCore
