// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include "video_core/buffer_cache/srt_memory_sync.h"

#if __has_include("video_core/buffer_cache/srt_session_diagnostics.h")
#include "video_core/buffer_cache/srt_session_diagnostics.h"
#define HAS_SRT_SESSION_DIAGNOSTICS 1
#else
#define HAS_SRT_SESSION_DIAGNOSTICS 0
#endif

namespace {

struct PageSource {
    unsigned snapshots{};
    bool valid{true};
    struct Epoch {
        PageSource& source;
        bool BeginPageCopy(VAddr, u64) {
            return source.valid;
        }
        bool EndPageCopy(VAddr, u64, bool copied) {
            return copied && source.valid;
        }
        bool Valid() const {
            return source.valid;
        }
        bool Seal() {
            return source.valid;
        }
    };
    Epoch BeginSnapshotEpoch() {
        return {*this};
    }
    bool IsGpuMapped(VAddr, u64) {
        return true;
    }
    bool ReadMemorySnapshot(VAddr, void* output, u64 size) {
        ++snapshots;
        std::memset(output, 0x5a, size);
        return true;
    }
};

using Session = VideoCore::SrtMemoryReadSession<PageSource, PageSource>;

template <typename T = Session>
T MakeSession(PageSource& source, bool enabled) {
    if constexpr (std::is_constructible_v<T, PageSource&, PageSource&, bool, bool>) {
        return T{source, source, false, enabled};
    } else {
        return T{source, source};
    }
}

struct EmptyCounts {
    u64 read_calls{};
    u64 new_pages{};
    u64 cache_hits{};
    u64 requested_bytes{};
};

template <typename T>
auto Counts(const T& session) {
    if constexpr (requires { session.DiagnosticCounts(); }) {
        return session.DiagnosticCounts();
    } else {
        return EmptyCounts{};
    }
}

TEST(SrtSessionDiagnostics, RealSessionCountsCallsPagesAndCachedHits) {
    PageSource source;
    auto session = MakeSession(source, true);
    std::array<u8, 8> output{};
    ASSERT_TRUE(session.Read(0x4010, output.data(), 8));
    ASSERT_TRUE(session.Read(0x4020, output.data(), 4));
    EXPECT_EQ(source.snapshots, 1);
    const auto counts = Counts(session);
    EXPECT_EQ(counts.read_calls, 2);
    EXPECT_EQ(counts.new_pages, 1);
    EXPECT_EQ(counts.cache_hits, 1);
    EXPECT_EQ(counts.requested_bytes, 12);
}

TEST(SrtSessionDiagnostics, CrossPageRequestRetainsTwoDistinctCaptures) {
    PageSource source;
    auto session = MakeSession(source, true);
    std::array<u8, 8> output{};
    ASSERT_TRUE(session.Read(0x4ffc, output.data(), output.size()));
    EXPECT_EQ(source.snapshots, 2);
    const auto counts = Counts(session);
    EXPECT_EQ(counts.read_calls, 1);
    EXPECT_EQ(counts.new_pages, 2);
    EXPECT_EQ(counts.cache_hits, 0);
    EXPECT_EQ(counts.requested_bytes, 8);
}

TEST(SrtSessionDiagnostics, DisabledKeepsOperationsAndReturnsEmptyCounts) {
    PageSource source;
    auto session = MakeSession(source, false);
    std::array<u8, 8> output{};
    ASSERT_TRUE(session.Read(0x4010, output.data(), output.size()));
    EXPECT_EQ(source.snapshots, 1);
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 value) { return value == 0x5a; }));
    EXPECT_EQ(Counts(session).read_calls, 0);
    EXPECT_EQ(Counts(session).new_pages, 0);
}

TEST(SrtSessionDiagnosticsSource, GetProgramOwnsOneDiagnosticScopePerRetryAttempt) {
    std::ifstream input{SHADPS4_PIPELINE_CACHE_SOURCE_PATH};
    ASSERT_TRUE(input.is_open());
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto start = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto stop =
        source.find("std::optional<vk::ShaderModule> PipelineCache::ReplaceShader", start);
    ASSERT_NE(start, std::string::npos);
    ASSERT_NE(stop, std::string::npos);
    const auto body = source.substr(start, stop - start);
    const auto attempt = body.find("bool& attempt_translation_rejected) -> Result {");
    const auto begin = body.find("session_diagnostics.BeginAttempt(");
    ASSERT_NE(begin, std::string::npos);
    EXPECT_LT(attempt, begin);
    EXPECT_NE(body.find("session_diagnostics.CompleteSession("), std::string::npos);
    EXPECT_NE(body.find("srt_read_session.DiagnosticCounts()"), std::string::npos);
    EXPECT_NE(source.find("SrtSessionCountsV1"), std::string::npos);
    EXPECT_NE(source.find("SrtSessionCountsTruncatedV1"), std::string::npos);
}

TEST(SrtSessionDiagnostics, DisabledAggregationDoesNotReadClockOrEmit) {
#if HAS_SRT_SESSION_DIAGNOSTICS
    VideoCore::SrtSessionDiagnostics diagnostics{false};
    unsigned clocks{};
    unsigned rows{};
    EXPECT_FALSE(
        diagnostics.BeginAttempt(1, 0, 1, [&] { return ++clocks; }, [&](const auto&) { ++rows; }));
    EXPECT_EQ(clocks, 0);
    EXPECT_EQ(rows, 0);
#else
    FAIL() << "The default-off session accumulator is not implemented.";
#endif
}

#if HAS_SRT_SESSION_DIAGNOSTICS
using VideoCore::SrtSessionDiagnostics;
using VideoCore::SrtSessionDiagnosticWindow;
using VideoCore::SrtSessionReadCounts;

TEST(SrtSessionDiagnostics, InvalidReadIsCountedWithoutReadingMemory) {
    PageSource source;
    auto session = MakeSession(source, true);
    std::array<u8, 8> output;
    output.fill(0xcc);
    EXPECT_FALSE(session.Read(0, output.data(), output.size()));
    EXPECT_FALSE(session.Valid());
    EXPECT_EQ(source.snapshots, 0);
    EXPECT_EQ(Counts(session).read_calls, 1);
    EXPECT_EQ(Counts(session).new_pages, 0);
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 byte) { return byte == 0xcc; }));
}

TEST(SrtSessionDiagnostics, CachedPageStillRejectsInvalidatedEpochWithCountingEnabled) {
    PageSource source;
    auto session = MakeSession(source, true);
    std::array<u8, 8> output{};
    ASSERT_TRUE(session.Read(0x4010, output.data(), output.size()));
    source.valid = false;
    output.fill(0xcc);
    EXPECT_FALSE(session.Read(0x4020, output.data(), output.size()));
    EXPECT_EQ(source.snapshots, 1);
    EXPECT_EQ(Counts(session).read_calls, 2);
    EXPECT_EQ(Counts(session).new_pages, 1);
    EXPECT_EQ(Counts(session).cache_hits, 1);
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 byte) { return byte == 0xcc; }));
}

TEST(SrtSessionDiagnostics, StageDistributionAndRetryCountsUseOneWindow) {
    SrtSessionDiagnostics diagnostics{true};
    std::uint64_t now = 10;
    const auto clock = [&] { return now; };
    std::vector<SrtSessionDiagnosticWindow> rows;
    const auto emit = [&](const auto& row) { rows.push_back(row); };
    const SrtSessionReadCounts fs_counts{
        .read_calls = 10, .new_pages = 2, .cache_hits = 8, .requested_bytes = 160};
    auto first = diagnostics.BeginAttempt(470, 0, 1, clock, emit);
    ASSERT_TRUE(first);
    now = 30;
    diagnostics.CompleteSession(*first, fs_counts, true, clock);
    auto retry = diagnostics.BeginAttempt(470, 0, 2, clock, emit);
    ASSERT_TRUE(retry);
    now = 40;
    diagnostics.CompleteSession(*retry, fs_counts, false, clock);
    auto vertex = diagnostics.BeginAttempt(470, 1, 1, clock, emit);
    ASSERT_TRUE(vertex);
    now = 100;
    diagnostics.CompleteSession(
        *vertex, {.read_calls = 5, .new_pages = 4, .cache_hits = 1, .requested_bytes = 80}, false,
        clock);
    EXPECT_TRUE(rows.empty());
    ASSERT_TRUE(diagnostics.BeginAttempt(471, 6, 1, clock, emit));
    ASSERT_EQ(rows.size(), 1);
    const auto& row = rows.front();
    EXPECT_EQ(row.presented_frame, 470);
    EXPECT_EQ(row.begin_us, 10);
    EXPECT_EQ(row.end_us, 100);
    EXPECT_EQ(row.totals.sessions, 3);
    EXPECT_EQ(row.totals.selections, 2);
    EXPECT_EQ(row.totals.retries, 1);
    EXPECT_EQ(row.totals.invalid_sessions, 1);
    EXPECT_EQ(row.totals.read_calls, 25);
    EXPECT_EQ(row.totals.new_pages, 8);
    EXPECT_EQ(row.totals.cache_hits, 17);
    EXPECT_EQ(row.totals.requested_bytes, 400);
    EXPECT_EQ(row.totals.max_pages_per_session, 4);
    EXPECT_EQ(row.totals.total_us, 90);
    EXPECT_EQ(row.totals.max_session_us, 60);
    EXPECT_EQ(row.stages[0].sessions, 2);
    EXPECT_EQ(row.stages[0].new_pages, 4);
    EXPECT_EQ(row.stages[1].sessions, 1);
    EXPECT_EQ(row.stages[1].new_pages, 4);
    EXPECT_EQ(row.stages[6].sessions, 0);
}

TEST(SrtSessionDiagnostics, CutoffAndInvalidStageNeverReadClock) {
    SrtSessionDiagnostics diagnostics{true, 470};
    unsigned clocks{};
    const auto clock = [&] { return ++clocks; };
    const auto emit = [](const auto&) { FAIL() << "No window completed"; };
    EXPECT_FALSE(diagnostics.BeginAttempt(469, 0, 1, clock, emit));
    EXPECT_FALSE(diagnostics.BeginAttempt(470, 7, 1, clock, emit));
    EXPECT_FALSE(diagnostics.BeginAttempt(470, 0, 0, clock, emit));
    EXPECT_EQ(clocks, 0);
    EXPECT_TRUE(diagnostics.BeginAttempt(470, 6, 1, clock, emit));
    EXPECT_EQ(clocks, 1);
}

TEST(SrtSessionDiagnostics, RowsAreBoundedWithOneTruncationNotice) {
    SrtSessionDiagnostics diagnostics{true};
    unsigned clocks{};
    std::vector<SrtSessionDiagnosticWindow> rows;
    const auto clock = [&] { return ++clocks; };
    const auto emit = [&](const auto& row) { rows.push_back(row); };
    for (unsigned frame = 0; frame < SrtSessionDiagnostics::MaximumRows; ++frame) {
        const auto token = diagnostics.BeginAttempt(frame, frame % 7, 1, clock, emit);
        ASSERT_TRUE(token);
        diagnostics.CompleteSession(*token, {.read_calls = 1, .new_pages = 1}, false, clock);
    }
    const auto previous_clocks = clocks;
    EXPECT_FALSE(diagnostics.BeginAttempt(256, 0, 1, clock, emit));
    EXPECT_FALSE(diagnostics.BeginAttempt(300, 0, 1, clock, emit));
    EXPECT_EQ(clocks, previous_clocks);
    ASSERT_EQ(rows.size(), 257);
    EXPECT_FALSE(rows[255].truncated);
    EXPECT_TRUE(rows[256].truncated);
}

TEST(SrtSessionDiagnostics, TotalsSaturateAndReaderOverflowIsExplicit) {
    VideoCore::SrtSessionReadCounter reader{true};
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    reader.Read(maximum);
    reader.Read(1);
    EXPECT_EQ(reader.Counts().requested_bytes, maximum);
    EXPECT_TRUE(reader.Counts().saturated);
    SrtSessionDiagnostics diagnostics{true};
    std::vector<SrtSessionDiagnosticWindow> rows;
    const auto emit = [&](const auto& row) { rows.push_back(row); };
    const auto clock = [] { return 10ULL; };
    auto token = diagnostics.BeginAttempt(1, 0, 1, clock, emit);
    ASSERT_TRUE(token);
    diagnostics.CompleteSession(*token, reader.Counts(), false, clock);
    token = diagnostics.BeginAttempt(1, 0, 1, clock, emit);
    ASSERT_TRUE(token);
    diagnostics.CompleteSession(*token, {.requested_bytes = 1}, false, clock);
    ASSERT_TRUE(diagnostics.BeginAttempt(2, 0, 1, clock, emit));
    ASSERT_EQ(rows.size(), 1);
    EXPECT_EQ(rows[0].totals.requested_bytes, maximum);
    EXPECT_EQ(rows[0].stages[0].requested_bytes, maximum);
    EXPECT_TRUE(rows[0].saturated);
}

TEST(SrtSessionDiagnostics, UnflushedFinalWindowDoesNotUseDestructorCallback) {
    unsigned rows{};
    {
        SrtSessionDiagnostics diagnostics{true};
        const auto token =
            diagnostics.BeginAttempt(1, 0, 1, [] { return 10ULL; }, [&](const auto&) { ++rows; });
        ASSERT_TRUE(token);
        diagnostics.CompleteSession(*token, {}, false, [] { return 20ULL; });
    }
    EXPECT_EQ(rows, 0);
}
#endif

} // namespace
