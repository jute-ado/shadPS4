// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "video_core/buffer_cache/srt_snapshot_timing.h"

#if __has_include("video_core/buffer_cache/srt_overlay_reuse.h")
#include "video_core/buffer_cache/srt_overlay_reuse.h"
#define HAS_SRT_OVERLAY_REUSE 1
#else
#define HAS_SRT_OVERLAY_REUSE 0
#endif

namespace {

#if HAS_SRT_OVERLAY_REUSE
using VideoCore::SrtOverlayReuseCache;
using VideoCore::SrtOverlayReuseKey;
using VideoCore::SrtOverlayReuseSlice;

constexpr SrtOverlayReuseKey Key{0x4000, 8};
constexpr std::array Slices{SrtOverlayReuseSlice{
    .source_address = 0x4000,
    .source_size = 4096,
    .source_offset = 0,
    .output_offset = 0,
    .download_offset = 0,
    .size = 8,
    .owner_generation = 9,
    .buffer_id = 3,
    .mapped_buffer_id = 3,
}};

TEST(SrtOverlayReuse, OnlyOwnedRawBytesSurviveSourceMutation) {
    SrtOverlayReuseCache cache{true};
    auto scope = cache.BeginSelection();
    const auto token = cache.CaptureToken();
    std::array<std::uint8_t, 8> source{1, 2, 3, 4, 5, 6, 7, 8};
    ASSERT_TRUE(cache.Store(token, Key, Slices, source));
    source.fill(0xee);
    std::array<std::uint8_t, 8> output{};
    ASSERT_TRUE(cache.TryRead(token, Key, Slices, output));
    EXPECT_EQ(output, (std::array<std::uint8_t, 8>{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(SrtOverlayReuse, FreshCpuCompositionIsRequiredForEachRawHit) {
    SrtOverlayReuseCache cache{true};
    auto scope = cache.BeginSelection();
    const auto token = cache.CaptureToken();
    const std::array<std::uint8_t, 8> gpu{1, 2, 3, 4, 5, 6, 7, 8};
    ASSERT_TRUE(cache.Store(token, Key, Slices, gpu));
    unsigned cpu_reads{};
    std::uint8_t pending_cpu = 0xa1;
    const auto compose = [&] {
        std::array<std::uint8_t, 8> output{};
        EXPECT_TRUE(cache.TryRead(token, Key, Slices, output));
        ++cpu_reads;
        output[0] = pending_cpu;
        return output;
    };
    EXPECT_EQ(compose()[0], 0xa1);
    pending_cpu = 0xb2;
    const auto second = compose();
    EXPECT_EQ(second[0], 0xb2);
    EXPECT_EQ(second[4], 5);
    EXPECT_EQ(cpu_reads, 2U);
}

TEST(SrtOverlayReuse, DisjointSlicesRetainCanonicalPaddedDownloadOffsets) {
    SrtOverlayReuseCache cache{true};
    auto scope = cache.BeginSelection();
    auto slices = std::array{Slices[0], Slices[0]};
    slices[0].source_offset = 2;
    slices[0].output_offset = 2;
    slices[0].download_offset = 16;
    slices[0].size = 4;
    slices[1].source_address += 8;
    slices[1].source_offset = 4;
    slices[1].output_offset = 12;
    slices[1].download_offset = 64;
    slices[1].size = 4;
    ++slices[1].owner_generation;
    ++slices[1].buffer_id;
    ++slices[1].mapped_buffer_id;
    std::array<std::uint8_t, 128> raw{};
    for (std::size_t index = 0; index < raw.size(); ++index) {
        raw[index] = static_cast<std::uint8_t>(index);
    }
    ASSERT_TRUE(cache.Store(cache.CaptureToken(), {Key.snapshot_address, 16}, slices, raw));
    std::array<std::uint8_t, 128> output{};
    ASSERT_TRUE(cache.TryRead(cache.CaptureToken(), {Key.snapshot_address, 16}, slices, output));
    EXPECT_EQ(output, raw);
}

TEST(SrtOverlayReuse, OwnerMappingAndEveryRangeFieldParticipateInIdentity) {
    SrtOverlayReuseCache cache{true};
    auto scope = cache.BeginSelection();
    const auto token = cache.CaptureToken();
    const std::array<std::uint8_t, 8> bytes{};
    ASSERT_TRUE(cache.Store(token, Key, Slices, bytes));
    std::array<std::uint8_t, 8> output{};
    for (unsigned field = 0; field < 9; ++field) {
        auto changed = Slices;
        switch (field) {
        case 0:
            ++changed[0].source_address;
            break;
        case 1:
            ++changed[0].source_size;
            break;
        case 2:
            ++changed[0].source_offset;
            break;
        case 3:
            ++changed[0].output_offset;
            break;
        case 4:
            ++changed[0].download_offset;
            break;
        case 5:
            --changed[0].size;
            break;
        case 6:
            ++changed[0].owner_generation;
            break;
        case 7:
            ++changed[0].buffer_id;
            break;
        case 8:
            ++changed[0].mapped_buffer_id;
            break;
        }
        EXPECT_FALSE(cache.TryRead(token, Key, changed, output)) << field;
    }
    EXPECT_FALSE(
        cache.TryRead(token, {Key.snapshot_address + 1, Key.snapshot_size}, Slices, output));
    EXPECT_FALSE(
        cache.TryRead(token, {Key.snapshot_address, Key.snapshot_size + 1}, Slices, output));
}

TEST(SrtOverlayReuse, ScopeExitNewSelectionAndStaleTokenAllMiss) {
    SrtOverlayReuseCache cache{true};
    const std::array<std::uint8_t, 8> bytes{};
    std::uint64_t old_token{};
    {
        auto scope = cache.BeginSelection();
        old_token = cache.CaptureToken();
        ASSERT_TRUE(cache.Store(old_token, Key, Slices, bytes));
    }
    EXPECT_EQ(cache.CaptureToken(), 0U);
    EXPECT_EQ(cache.RetainedBytes(), 0U);
    auto next = cache.BeginSelection();
    std::array<std::uint8_t, 8> output{};
    EXPECT_FALSE(cache.TryRead(old_token, Key, Slices, output));
    EXPECT_FALSE(cache.TryRead(cache.CaptureToken(), Key, Slices, output));
    EXPECT_FALSE(cache.Store(old_token, Key, Slices, bytes));
}

TEST(SrtOverlayReuse, NestedSelectionDisablesOuterUntilItsExit) {
    SrtOverlayReuseCache cache{true};
    const std::array<std::uint8_t, 8> bytes{};
    {
        auto outer = cache.BeginSelection();
        const auto token = cache.CaptureToken();
        ASSERT_TRUE(cache.Store(token, Key, Slices, bytes));
        {
            auto inner = cache.BeginSelection();
            EXPECT_EQ(cache.CaptureToken(), 0U);
            EXPECT_FALSE(cache.Store(token, Key, Slices, bytes));
        }
        EXPECT_EQ(cache.CaptureToken(), 0U);
        EXPECT_FALSE(cache.Store(token, Key, Slices, bytes));
    }
    auto fresh = cache.BeginSelection();
    EXPECT_NE(cache.CaptureToken(), 0U);
}

TEST(SrtOverlayReuse, RetryFailureOrUnexpectedWriteInvalidatesRemainingSelection) {
    SrtOverlayReuseCache cache{true};
    auto scope = cache.BeginSelection();
    const auto token = cache.CaptureToken();
    const std::array<std::uint8_t, 8> bytes{};
    ASSERT_TRUE(cache.Store(token, Key, Slices, bytes));
    cache.InvalidateSelection();
    std::array<std::uint8_t, 8> output{};
    EXPECT_EQ(cache.CaptureToken(), 0U);
    EXPECT_FALSE(cache.TryRead(token, Key, Slices, output));
    EXPECT_FALSE(cache.Store(token, Key, Slices, bytes));
    EXPECT_EQ(cache.RetainedBytes(), 0U);
}

TEST(SrtOverlayReuse, DisabledAndInactiveCacheRetainNothing) {
    const std::array<std::uint8_t, 8> bytes{};
    SrtOverlayReuseCache disabled{false};
    auto scope = disabled.BeginSelection();
    EXPECT_EQ(disabled.CaptureToken(), 0U);
    EXPECT_FALSE(disabled.Store(1, Key, Slices, bytes));
    EXPECT_EQ(disabled.RetainedBytes(), 0U);
    SrtOverlayReuseCache inactive{true};
    EXPECT_EQ(inactive.CaptureToken(), 0U);
    EXPECT_FALSE(inactive.Store(1, Key, Slices, bytes));
}

TEST(SrtOverlayReuse, MalformedFootprintsFailWithoutPublishingOrChangingOutput) {
    SrtOverlayReuseCache cache{true};
    auto scope = cache.BeginSelection();
    const auto token = cache.CaptureToken();
    const std::array<std::uint8_t, 8> bytes{};
    auto invalid = Slices;
    invalid[0].download_offset = std::numeric_limits<std::uint64_t>::max();
    EXPECT_FALSE(cache.Store(token, Key, invalid, bytes));
    invalid = Slices;
    invalid[0].source_offset = invalid[0].source_size;
    EXPECT_FALSE(cache.Store(token, Key, invalid, bytes));
    invalid = Slices;
    invalid[0].size = 0;
    EXPECT_FALSE(cache.Store(token, Key, invalid, bytes));
    EXPECT_FALSE(cache.Store(token, {std::numeric_limits<std::uint64_t>::max(), 8}, Slices, bytes));
    std::array<std::uint8_t, 8> output{};
    output.fill(0xab);
    EXPECT_FALSE(cache.TryRead(token, Key, Slices, output));
    for (const auto byte : output) {
        EXPECT_EQ(byte, 0xab);
    }
    EXPECT_EQ(cache.RetainedBytes(), 0U);
}

TEST(SrtOverlayReuse, StorageBudgetAndEntryLimitAreHardBounds) {
    SrtOverlayReuseCache cache{true};
    auto scope = cache.BeginSelection();
    const auto token = cache.CaptureToken();
    std::vector<std::uint8_t> too_large(SrtOverlayReuseCache::MaximumRetainedBytes + 1);
    EXPECT_FALSE(cache.Store(token, Key, Slices, too_large));
    const std::array<std::uint8_t, 8> bytes{};
    for (std::uint32_t index = 0; index < SrtOverlayReuseCache::MaximumEntries; ++index) {
        auto slices = Slices;
        slices[0].source_address += index * 4096;
        ASSERT_TRUE(cache.Store(token, {Key.snapshot_address + index * 4096, 8}, slices, bytes));
    }
    EXPECT_EQ(cache.EntryCount(), SrtOverlayReuseCache::MaximumEntries);
    EXPECT_FALSE(cache.Store(token, {0x8000000, 8}, Slices, bytes));
    EXPECT_LE(cache.RetainedBytes(), SrtOverlayReuseCache::MaximumRetainedBytes);
}

TEST(SrtOverlayReuse, PayloadBudgetFailureRetainsEarlierEntries) {
    SrtOverlayReuseCache cache{true};
    auto scope = cache.BeginSelection();
    const auto token = cache.CaptureToken();
    const std::vector<std::uint8_t> bytes(65536, 0x5a);
    unsigned stored{};
    for (unsigned index = 0; index < 32; ++index) {
        auto slices = Slices;
        slices[0].source_address += index * 65536;
        slices[0].source_size = bytes.size();
        slices[0].size = bytes.size();
        if (!cache.Store(token, {slices[0].source_address, bytes.size()}, slices, bytes)) {
            break;
        }
        ++stored;
    }
    EXPECT_GT(stored, 0U);
    EXPECT_LT(stored, 16U); // Payload plus retained slice metadata share the one-MiB budget.
    EXPECT_LE(cache.RetainedBytes(), SrtOverlayReuseCache::MaximumRetainedBytes);
    auto first = Slices;
    first[0].source_size = bytes.size();
    first[0].size = bytes.size();
    std::vector<std::uint8_t> output(bytes.size());
    ASSERT_TRUE(cache.TryRead(token, {Key.snapshot_address, bytes.size()}, first, output));
    EXPECT_EQ(output, bytes);
}

TEST(SrtOverlayReuse, SliceCountAndInconsistentGuestAddressFailClosed) {
    SrtOverlayReuseCache cache{true};
    auto scope = cache.BeginSelection();
    const auto token = cache.CaptureToken();
    const std::array<std::uint8_t, 8> bytes{};
    const std::vector<SrtOverlayReuseSlice> too_many(SrtOverlayReuseCache::MaximumSlices + 1,
                                                     Slices[0]);
    EXPECT_FALSE(cache.Store(token, Key, too_many, bytes));
    auto shifted = Slices;
    ++shifted[0].source_address;
    EXPECT_FALSE(cache.Store(token, Key, shifted, bytes));
    EXPECT_EQ(cache.RetainedBytes(), 0U);
}
#else
TEST(SrtOverlayReuse, ScopedBoundedProductionHelperExists) {
    FAIL() << "Scoped raw GPU overlay reuse helper is absent";
}
#endif

std::string ReadSource(const char* path) {
    std::ifstream input{path};
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

TEST(SrtOverlayReuseWiring, ScopeIsInsideStageSelectionAndFailuresInvalidateIt) {
#ifdef SHADPS4_OVERLAY_REUSE_PIPELINE_SOURCE_PATH
    const auto source = ReadSource(SHADPS4_OVERLAY_REUSE_PIPELINE_SOURCE_PATH);
    const auto stages = source.find("bool PipelineCache::RefreshGraphicsStages(");
    ASSERT_NE(stages, std::string::npos);
    const auto begin = source.find("buffer_cache.BeginSrtOverlayReuseSelection()", stages);
    ASSERT_NE(begin, std::string::npos);
    EXPECT_LT(begin, source.find("const auto bind_stage", stages));
    const auto failure = source.find("const auto fail_snapshot");
    ASSERT_NE(failure, std::string::npos);
    EXPECT_NE(source.find("buffer_cache.InvalidateSrtOverlayReuse()", failure), std::string::npos);
#else
    FAIL() << "Pipeline source contract is not configured";
#endif
}

TEST(SrtOverlayReuseWiring, ExactPlanPrecedesReuseAndFreshCpuCompositionRemains) {
#ifdef SHADPS4_OVERLAY_REUSE_BUFFER_SOURCE_PATH
    const auto source = ReadSource(SHADPS4_OVERLAY_REUSE_BUFFER_SOURCE_PATH);
    const auto begin = source.find("bool BufferCache::DownloadBufferMemorySnapshot(");
    ASSERT_NE(begin, std::string::npos);
    const auto plan = source.find("gpu_modified_ranges.ForEachInRange(", begin);
    ASSERT_NE(plan, std::string::npos);
    const auto reuse = source.find("srt_overlay_reuse.TryRead(", plan);
    ASSERT_NE(reuse, std::string::npos);
    EXPECT_LT(reuse, source.find("download_buffer.Map(total_size_bytes)", plan));
    EXPECT_NE(source.find("cpu_page_write_tracker.ApplyPendingWrites(", reuse), std::string::npos);
    EXPECT_NE(source.find("SHADPS4_SRT_OVERLAY_REUSE"), std::string::npos);
#else
    FAIL() << "Buffer source contract is not configured";
#endif
}

TEST(SrtOverlayReuseWiring, PotentialContentMutationsInvalidateBeforePreparingWork) {
#ifdef SHADPS4_OVERLAY_REUSE_BUFFER_SOURCE_PATH
    const auto source = ReadSource(SHADPS4_OVERLAY_REUSE_BUFFER_SOURCE_PATH);
    for (const char* signature :
         {"void BufferCache::FillBuffer(", "void BufferCache::CopyBuffer(",
          "bool BufferCache::SynchronizeBuffer(", "void BufferCache::WriteDataBuffer(",
          "void BufferCache::DeleteBuffer("}) {
        const auto begin = source.find(signature);
        ASSERT_NE(begin, std::string::npos) << signature;
        const auto body = source.find('{', begin);
        ASSERT_NE(body, std::string::npos);
        EXPECT_NE(source.substr(body, 160).find("InvalidateSrtOverlayReuse();"), std::string::npos)
            << signature;
    }
#else
    FAIL() << "Buffer source contract is not configured";
#endif
}

template <typename Sample>
void MarkReused(Sample& sample) {
    if constexpr (requires {
                      sample.raw_overlay_reused;
                      sample.reused_overlay_bytes;
                  }) {
        sample.raw_overlay_reused = true;
        sample.reused_overlay_bytes = 24;
    }
}

template <typename Window>
void CheckReuseTotals(const Window& window) {
    if constexpr (requires {
                      window.reuse_hits;
                      window.reused_overlay_bytes;
                  }) {
        EXPECT_EQ(window.reuse_hits, 1U);
        EXPECT_EQ(window.reused_overlay_bytes, 24U);
    } else {
        FAIL() << "Timing receipt has no raw reuse counters";
    }
}

TEST(SrtOverlayReuseTiming, ReusedRawBytesAreCountedWithoutInventingFinishTime) {
    VideoCore::SrtSnapshotTiming timing{true};
    std::vector<VideoCore::SrtSnapshotTimingWindow> rows;
    auto emit = [&](auto row) { rows.push_back(row); };
    std::uint64_t clock{};
    auto now = [&] { return ++clock; };
    auto sample = timing.BeginSnapshot(5, 32, now, emit);
    ASSERT_TRUE(sample);
    MarkReused(*sample);
    sample->gpu_composed = true;
    sample->success = true;
    timing.CompleteSnapshot(*sample, now);
    (void)timing.BeginSnapshot(6, 32, now, emit);
    ASSERT_EQ(rows.size(), 1U);
    CheckReuseTotals(rows[0]);
    EXPECT_EQ(rows[0].finish_us, 0U);
}

} // namespace
