// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>

#if __has_include("video_core/buffer_cache/srt_snapshot_timing.h")
#include "video_core/buffer_cache/srt_snapshot_timing.h"
#define HAS_SRT_SNAPSHOT_TIMING 1
#else
#define HAS_SRT_SNAPSHOT_TIMING 0
#endif

namespace {

std::string ReadBufferSource() {
    std::ifstream input{SHADPS4_INVALIDATION_BUFFER_SOURCE_PATH};
    EXPECT_TRUE(input.is_open());
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

std::size_t CountOccurrences(std::string_view source, std::string_view needle) {
    if (needle.empty()) {
        return 0;
    }
    std::size_t count{};
    for (std::size_t offset{};;) {
        offset = source.find(needle, offset);
        if (offset == std::string_view::npos) {
            return count;
        }
        ++count;
        offset += needle.size();
    }
}

TEST(SrtSnapshotTimingSource, OptionalTimingIsOwnedByTheGpuCommandCallback) {
    const auto source = ReadBufferSource();
    const auto start = source.find("bool BufferCache::ReadMemorySnapshot(");
    const auto stop = source.find("template <bool async, bool exact>", start);
    ASSERT_NE(start, std::string::npos);
    ASSERT_NE(stop, std::string::npos);
    const auto body = source.substr(start, stop - start);
    const auto dispatch = body.find("liverpool->SendCommand<true>");
    const auto begin = body.find("srt_snapshot_timing.BeginSnapshot(");
    ASSERT_NE(begin, std::string::npos);
    EXPECT_LT(dispatch, begin);
    EXPECT_NE(body.find("DebugState.GetFrameNum()"), std::string::npos);
    EXPECT_NE(body.find("srt_snapshot_timing.CompleteSnapshot("), std::string::npos);
}

TEST(SrtSnapshotTimingSource, MeasuresTheExistingFinishWithoutAddingAWait) {
    const auto source = ReadBufferSource();
    const auto start = source.find("bool BufferCache::DownloadBufferMemorySnapshot(");
    const auto stop = source.find("BufferCache::PreparedVertexBuffers", start);
    ASSERT_NE(start, std::string::npos);
    ASSERT_NE(stop, std::string::npos);
    const auto body = source.substr(start, stop - start);
    EXPECT_NE(body.find("&SrtSnapshotTimingSample::finish_us"), std::string::npos);
    EXPECT_NE(body.find("&SrtSnapshotTimingSample::map_us"), std::string::npos);
    EXPECT_NE(body.find("&SrtSnapshotTimingSample::compose_us"), std::string::npos);
    const auto finish = body.find("scheduler.Finish()");
    ASSERT_NE(finish, std::string::npos);
    EXPECT_EQ(body.find("scheduler.Finish()", finish + 1), std::string::npos);
}

TEST(SrtSnapshotTimingSource, ASnapshotCannotAccumulateDuplicateFinishesOrNestedWaits) {
    const auto source = ReadBufferSource();
    const auto start = source.find("bool BufferCache::DownloadBufferMemorySnapshot(");
    const auto stop = source.find("BufferCache::PreparedVertexBuffers", start);
    ASSERT_NE(start, std::string::npos);
    ASSERT_NE(stop, std::string::npos);
    const auto body = source.substr(start, stop - start);

    // The current path has one completion fence per GPU-composed snapshot. A future batched
    // implementation may move that fence outside this per-request function, so zero is also
    // valid here; a second Finish or an internal Wait would recreate the measured stall.
    EXPECT_LE(CountOccurrences(body, "scheduler.Finish()"), 1U);
    EXPECT_EQ(CountOccurrences(body, "scheduler.Wait("), 0U);
    EXPECT_NE(body.find("CompletePrivateSnapshotReadback"), std::string::npos);
}

TEST(SrtSnapshotTimingSource, DiagnosticsAreExplicitlyOptInAndBounded) {
    const auto source = ReadBufferSource();
    EXPECT_NE(source.find("SHADPS4_SRT_SNAPSHOT_TIMING"), std::string::npos);
    EXPECT_NE(source.find("SHADPS4_SRT_SNAPSHOT_TIMING_START_FRAME"), std::string::npos);
    EXPECT_NE(source.find("SrtSnapshotTimingV1"), std::string::npos);
    EXPECT_NE(source.find("SrtSnapshotTimingTruncatedV1"), std::string::npos);
}

TEST(SrtSnapshotTiming, DisabledDoesNotReadClockOrEmitRows) {
#if HAS_SRT_SNAPSHOT_TIMING
    VideoCore::SrtSnapshotTiming timing{false};
    unsigned clock_reads{};
    unsigned rows{};
    const auto clock = [&] {
        ++clock_reads;
        return 10ULL;
    };
    auto sample = timing.BeginSnapshot(1, 4096, clock, [&](const auto&) { ++rows; });
    EXPECT_FALSE(sample.has_value());
    unsigned operation_calls{};
    const auto result = VideoCore::MeasureSrtSnapshotPhase(
        nullptr, &VideoCore::SrtSnapshotTimingSample::finish_us, clock, [&] {
            ++operation_calls;
            return 7;
        });
    EXPECT_EQ(result, 7);
    EXPECT_EQ(operation_calls, 1);
    EXPECT_EQ(clock_reads, 0);
    EXPECT_EQ(rows, 0);
#else
    FAIL() << "Default-off timing helper and its no-clock path are not implemented.";
#endif
}

#if HAS_SRT_SNAPSHOT_TIMING
using VideoCore::SrtSnapshotTiming;
using VideoCore::SrtSnapshotTimingConfig;
using VideoCore::SrtSnapshotTimingSample;
using VideoCore::SrtSnapshotTimingWindow;

TEST(SrtSnapshotTiming, ExactOptInAndStrictStartFrameParsing) {
    EXPECT_FALSE(SrtSnapshotTimingConfig::Parse("", "").enabled);
    EXPECT_FALSE(SrtSnapshotTimingConfig::Parse("true", "470").enabled);
    EXPECT_FALSE(SrtSnapshotTimingConfig::Parse("10", "470").enabled);
    EXPECT_FALSE(SrtSnapshotTimingConfig::Parse("1", "-1").enabled);
    EXPECT_FALSE(SrtSnapshotTimingConfig::Parse("1", "470x").enabled);
    EXPECT_FALSE(SrtSnapshotTimingConfig::Parse("1", "50000001").enabled);
    EXPECT_FALSE(SrtSnapshotTimingConfig::Parse("1", "99999999999999999").enabled);
    EXPECT_EQ(SrtSnapshotTimingConfig::Parse("1", "").start_frame, 0);
    EXPECT_TRUE(SrtSnapshotTimingConfig::Parse("1", "50000000").enabled);
    EXPECT_EQ(SrtSnapshotTimingConfig::Parse("1", "470").start_frame, 470);
}

TEST(SrtSnapshotTiming, StartCutoffDoesNotReadClockAndIsInclusive) {
    SrtSnapshotTiming timing{true, 470};
    unsigned clock_reads{};
    const auto clock = [&] {
        ++clock_reads;
        return 100ULL;
    };
    const auto emit = [](const auto&) { FAIL() << "No completed window yet"; };
    EXPECT_FALSE(timing.BeginSnapshot(469, 4096, clock, emit));
    EXPECT_EQ(clock_reads, 0);
    EXPECT_TRUE(timing.BeginSnapshot(470, 4096, clock, emit));
    EXPECT_EQ(clock_reads, 1);
}

TEST(SrtSnapshotTiming, AggregatesEveryPhaseAndRouteWithoutMixingFrameWindows) {
    SrtSnapshotTiming timing{true};
    std::uint64_t now = 100;
    const auto clock = [&] { return now; };
    std::vector<SrtSnapshotTimingWindow> rows;
    const auto emit = [&](const auto& row) { rows.push_back(row); };
    auto first = timing.BeginSnapshot(10, 4096, clock, emit);
    ASSERT_TRUE(first);
    first->cpu_only = true;
    first->success = true;
    first->compose_us = 20;
    now = 150;
    timing.CompleteSnapshot(*first, clock);
    now = 200;
    auto second = timing.BeginSnapshot(10, 4096, clock, emit);
    ASSERT_TRUE(second);
    second->gpu_composed = true;
    second->success = true;
    second->overlay_bytes = 2048;
    second->copies = 3;
    second->map_us = 4;
    second->finish_us = 400;
    second->compose_us = 30;
    now = 700;
    timing.CompleteSnapshot(*second, clock);
    EXPECT_TRUE(rows.empty());
    now = 1000;
    ASSERT_TRUE(timing.BeginSnapshot(11, 16, clock, emit));
    ASSERT_EQ(rows.size(), 1);
    const auto& row = rows[0];
    EXPECT_EQ(row.presented_frame, 10);
    EXPECT_EQ(row.begin_us, 100);
    EXPECT_EQ(row.end_us, 700);
    EXPECT_EQ(row.snapshots, 2);
    EXPECT_EQ(row.cpu_only, 1);
    EXPECT_EQ(row.gpu_composed, 1);
    EXPECT_EQ(row.failures, 0);
    EXPECT_EQ(row.requested_bytes, 8192);
    EXPECT_EQ(row.overlay_bytes, 2048);
    EXPECT_EQ(row.copies, 3);
    EXPECT_EQ(row.map_us, 4);
    EXPECT_EQ(row.finish_us, 400);
    EXPECT_EQ(row.compose_us, 50);
    EXPECT_EQ(row.total_us, 550);
    EXPECT_EQ(row.max_total_us, 500);
    EXPECT_EQ(row.max_finish_us, 400);
    EXPECT_FALSE(row.saturated);
}

TEST(SrtSnapshotTiming, FailedCaptureStillAccountsElapsedTime) {
    SrtSnapshotTiming timing{true};
    std::uint64_t now = 10;
    const auto clock = [&] { return now; };
    std::vector<SrtSnapshotTimingWindow> rows;
    const auto emit = [&](const auto& row) { rows.push_back(row); };
    auto sample = timing.BeginSnapshot(1, 4096, clock, emit);
    ASSERT_TRUE(sample);
    sample->gpu_composed = true;
    sample->finish_us = 80;
    now = 100;
    timing.CompleteSnapshot(*sample, clock);
    ASSERT_TRUE(timing.BeginSnapshot(2, 4096, clock, emit));
    ASSERT_EQ(rows.size(), 1);
    EXPECT_EQ(rows[0].failures, 1);
    EXPECT_EQ(rows[0].total_us, 90);
    EXPECT_EQ(rows[0].finish_us, 80);
}

TEST(SrtSnapshotTiming, BoundedRowsEmitOneTruncationAndThenNeverReadTheClock) {
    SrtSnapshotTiming timing{true};
    unsigned clock_reads{};
    const auto clock = [&] { return ++clock_reads; };
    std::vector<SrtSnapshotTimingWindow> rows;
    const auto emit = [&](const auto& row) { rows.push_back(row); };
    for (unsigned frame = 0; frame < SrtSnapshotTiming::MaximumRows; ++frame) {
        auto sample = timing.BeginSnapshot(frame, 4, clock, emit);
        ASSERT_TRUE(sample);
        sample->success = true;
        timing.CompleteSnapshot(*sample, clock);
    }
    const auto reads_before_truncation = clock_reads;
    EXPECT_FALSE(timing.BeginSnapshot(256, 4, clock, emit));
    EXPECT_FALSE(timing.BeginSnapshot(257, 4, clock, emit));
    EXPECT_FALSE(timing.BeginSnapshot(900, 4, clock, emit));
    EXPECT_EQ(clock_reads, reads_before_truncation);
    ASSERT_EQ(rows.size(), 257);
    EXPECT_FALSE(rows[255].truncated);
    EXPECT_EQ(rows[255].presented_frame, 255);
    EXPECT_TRUE(rows[256].truncated);
    EXPECT_EQ(rows[256].presented_frame, 256);
}

TEST(SrtSnapshotTiming, FinalPartialWindowDoesNotEmitFromDestruction) {
    unsigned rows{};
    {
        SrtSnapshotTiming timing{true};
        auto sample =
            timing.BeginSnapshot(1, 4, [] { return 10ULL; }, [&](const auto&) { ++rows; });
        ASSERT_TRUE(sample);
        timing.CompleteSnapshot(*sample, [] { return 20ULL; });
    }
    EXPECT_EQ(rows, 0);
}

TEST(SrtSnapshotTiming, SaturatesCountersRatherThanWrapping) {
    SrtSnapshotTiming timing{true};
    std::vector<SrtSnapshotTimingWindow> rows;
    const auto emit = [&](const auto& row) { rows.push_back(row); };
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    auto sample = timing.BeginSnapshot(1, maximum, [] { return 0ULL; }, emit);
    ASSERT_TRUE(sample);
    sample->finish_us = maximum;
    sample->copies = maximum;
    timing.CompleteSnapshot(*sample, [=] { return maximum; });
    sample = timing.BeginSnapshot(1, 1, [] { return 0ULL; }, emit);
    ASSERT_TRUE(sample);
    sample->finish_us = 1;
    sample->copies = 1;
    timing.CompleteSnapshot(*sample, [] { return 1ULL; });
    ASSERT_TRUE(timing.BeginSnapshot(2, 1, [] { return 1ULL; }, emit));
    ASSERT_EQ(rows.size(), 1);
    EXPECT_EQ(rows[0].requested_bytes, maximum);
    EXPECT_EQ(rows[0].finish_us, maximum);
    EXPECT_EQ(rows[0].copies, maximum);
    EXPECT_EQ(rows[0].total_us, maximum);
    EXPECT_EQ(rows[0].end_us, maximum);
    EXPECT_TRUE(rows[0].saturated);
}

TEST(SrtSnapshotTiming, PhaseTimerRetainsReturnValuesAndFailures) {
    SrtSnapshotTimingSample sample;
    std::uint64_t now = 10;
    const auto clock = [&] { return std::exchange(now, now + 7); };
    EXPECT_FALSE(VideoCore::MeasureSrtSnapshotPhase(&sample, &SrtSnapshotTimingSample::map_us,
                                                    clock, [] { return false; }));
    EXPECT_EQ(sample.map_us, 7);
    VideoCore::MeasureSrtSnapshotPhase(&sample, &SrtSnapshotTimingSample::finish_us, clock, [] {});
    EXPECT_EQ(sample.finish_us, 7);
    EXPECT_THROW(
        VideoCore::MeasureSrtSnapshotPhase(&sample, &SrtSnapshotTimingSample::compose_us, clock,
                                           [] { throw std::runtime_error{"capture failed"}; }),
        std::runtime_error);
    EXPECT_EQ(sample.compose_us, 7);
}

TEST(SrtSnapshotTiming, ClockRegressionCannotUnderflowIntoHugeDuration) {
    EXPECT_EQ(VideoCore::SrtSnapshotElapsed(100, 99), 0);
    EXPECT_EQ(VideoCore::SrtSnapshotElapsed(100, 100), 0);
    EXPECT_EQ(VideoCore::SrtSnapshotElapsed(100, 130), 30);
}
#endif

} // namespace
