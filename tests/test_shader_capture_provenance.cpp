// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "video_core/shader_capture_provenance.h"

namespace VideoCore {
namespace {

SrtCaptureProvenance MakeCapture(ShaderCaptureSource source, u64 mapping_token,
                                 u64 content_digest) {
    SrtCaptureProvenance capture;
    capture.Record({
        .address = 0x242710000ULL,
        .size = 16_KB,
        .exact_gpu_modified = source == ShaderCaptureSource::GpuBuffer,
        .coarse_gpu_owned = source != ShaderCaptureSource::CpuBacking,
        .source = source,
        .mapping_token = mapping_token,
        .content_digest = content_digest,
    });
    return capture;
}

TEST(ShaderCaptureProvenance, RangeDigestContainsNoGuestPayloadAndChangesWithContent) {
    static_assert(std::is_trivially_copyable_v<ShaderCaptureRangeProvenance>);

    std::array<u8, 16> first{};
    std::array<u8, 16> second{};
    second.back() = 1;

    const auto first_digest = DigestShaderCaptureContent(first);
    const auto second_digest = DigestShaderCaptureContent(second);

    EXPECT_NE(first_digest, second_digest);
    const ShaderCaptureRangeProvenance range{
        .address = 0x242710000ULL,
        .size = first.size(),
        .exact_gpu_modified = false,
        .coarse_gpu_owned = false,
        .source = ShaderCaptureSource::CpuBacking,
        .mapping_token = 17,
        .content_digest = first_digest,
    };
    EXPECT_EQ(range.content_digest, first_digest);
    EXPECT_EQ(sizeof(range.content_digest), sizeof(u64));
}

TEST(ShaderCaptureProvenance, RangeCollectorIsBoundedAndMarksTruncation) {
    SrtCaptureProvenance capture;
    for (size_t index = 0; index <= SrtCaptureProvenance::MaxRanges; ++index) {
        capture.Record({
            .address = 0x1000 + index * 0x1000,
            .size = 0x1000,
            .source = ShaderCaptureSource::CpuBacking,
            .mapping_token = index + 1,
            .content_digest = index + 10,
        });
    }

    EXPECT_EQ(capture.Ranges().size(), SrtCaptureProvenance::MaxRanges);
    EXPECT_TRUE(capture.Truncated());
    EXPECT_EQ(capture.Ranges().front().address, 0x1000U);
}

TEST(ShaderCaptureProvenance, ProgramStoreIsBoundedAndCapsDiagnosticOutput) {
    ShaderCaptureProvenanceStore store;
    SrtCaptureProvenance capture;
    capture.Record({
        .address = 0x4000,
        .size = 16_KB,
        .source = ShaderCaptureSource::GpuBuffer,
        .mapping_token = 9,
        .content_digest = 0x1234,
    });

    for (size_t index = 0; index <= ShaderCaptureProvenanceStore::MaxPrograms; ++index) {
        store.Record(index, 2, true, capture);
    }

    ASSERT_EQ(store.Records().size(), ShaderCaptureProvenanceStore::MaxPrograms);
    EXPECT_EQ(store.DroppedPrograms(), 1U);
    EXPECT_EQ(store.Records().front().program_hash, 0U);
    EXPECT_EQ(store.Records().back().program_hash,
              ShaderCaptureProvenanceStore::MaxPrograms - 1);
    ASSERT_EQ(store.Records().back().ranges.size(), 1U);
    EXPECT_EQ(store.Records().back().ranges.front().mapping_token, 9U);
}

TEST(ShaderCaptureProvenance, ExactDuplicatesDoNotConsumeCapacityOrCountAsDropped) {
    ShaderCaptureProvenanceStore store;
    const auto capture = MakeCapture(ShaderCaptureSource::CpuBacking, 7, 0x1234);

    for (size_t index = 0; index < ShaderCaptureProvenanceStore::MaxPrograms * 2; ++index) {
        store.Record(0x1234, 2, true, capture);
    }

    ASSERT_EQ(store.Records().size(), 1U);
    EXPECT_EQ(store.DroppedPrograms(), 0U);

    for (size_t index = 1; index < ShaderCaptureProvenanceStore::MaxPrograms; ++index) {
        store.Record(0x1234 + index, 2, true, capture);
    }
    ASSERT_EQ(store.Records().size(), ShaderCaptureProvenanceStore::MaxPrograms);

    store.Record(0x1234, 2, true, capture);
    EXPECT_EQ(store.DroppedPrograms(), 0U);
    store.Record(0xffff, 2, true, capture);
    EXPECT_EQ(store.DroppedPrograms(), 1U);
}

TEST(ShaderCaptureProvenance, ChangedSourceTokenDigestOrOutcomeProducesANewRecord) {
    ShaderCaptureProvenanceStore store;
    const auto cpu = MakeCapture(ShaderCaptureSource::CpuBacking, 7, 0x1234);
    const auto gpu = MakeCapture(ShaderCaptureSource::GpuBuffer, 7, 0x1234);
    const auto new_token = MakeCapture(ShaderCaptureSource::CpuBacking, 8, 0x1234);
    const auto new_digest = MakeCapture(ShaderCaptureSource::CpuBacking, 7, 0x5678);

    EXPECT_TRUE(store.Record(0x1234, 2, true, cpu));
    EXPECT_TRUE(store.Record(0x1234, 2, true, gpu));
    EXPECT_TRUE(store.Record(0x1234, 2, true, new_token));
    EXPECT_TRUE(store.Record(0x1234, 2, true, new_digest));
    EXPECT_TRUE(store.Record(0x1234, 2, false, cpu));

    EXPECT_EQ(store.Records().size(), 5U);
    EXPECT_EQ(store.DroppedPrograms(), 0U);
}

TEST(ShaderCaptureProvenance, StructuredLogIsObservableOnlyWhenEnabled) {
    SrtCaptureProvenance capture;
    capture.Record({
        .address = 0x242710000ULL,
        .size = 16_KB,
        .exact_gpu_modified = false,
        .coarse_gpu_owned = true,
        .source = ShaderCaptureSource::Refused,
        .mapping_token = 42,
        .content_digest = 0,
    });
    ShaderCaptureProvenanceRecorder recorder;

    for (size_t index = 0; index < ShaderCaptureProvenanceStore::MaxPrograms * 2; ++index) {
        const auto disabled = recorder.Record(false, 0x1234, 3, false, capture);
        EXPECT_FALSE(disabled.has_value());
    }
    EXPECT_TRUE(recorder.Store().Records().empty());
    EXPECT_EQ(recorder.Store().DroppedPrograms(), 0U);

    const auto enabled = recorder.Record(true, 0x1234, 3, false, capture);
    ASSERT_TRUE(enabled.has_value());
    EXPECT_NE(enabled->find("shader_capture_provenance={"), std::string::npos);
    EXPECT_NE(enabled->find("\"program\":\"0x0000000000001234\""), std::string::npos);
    EXPECT_NE(enabled->find("\"stage\":3"), std::string::npos);
    EXPECT_NE(enabled->find("\"outcome\":\"refused\""), std::string::npos);
    EXPECT_NE(enabled->find("\"address\":\"0x0000000242710000\""), std::string::npos);
    EXPECT_NE(enabled->find("\"size\":16384"), std::string::npos);
    EXPECT_NE(enabled->find("\"exact\":false"), std::string::npos);
    EXPECT_NE(enabled->find("\"coarse\":true"), std::string::npos);
    EXPECT_NE(enabled->find("\"source\":\"refused\""), std::string::npos);
    EXPECT_NE(enabled->find("\"token\":42"), std::string::npos);
    EXPECT_NE(enabled->find("\"digest\":\"0x0000000000000000\""), std::string::npos);
}

TEST(PotentialShaderWriteProvenance, DisabledRecordingHasNoStateOrSequenceSideEffect) {
    PotentialShaderWriteTracker tracker;

    tracker.Record(false, 0x1234, 5, 0, 17, 0x20000, 0x1000);

    EXPECT_EQ(tracker.RecordCount(), 0U);
    EXPECT_FALSE(tracker.LatestOverlap(0x20000, 0x1000, 17).has_value());
}

TEST(PotentialShaderWriteProvenance, LatestOverlappingWriterSurvivesBoundedRingTurnover) {
    PotentialShaderWriteTracker tracker;
    for (size_t index = 0; index < PotentialShaderWriteTracker::MaxWrites + 3; ++index) {
        tracker.Record(true, 0x1000 + index, 5, 2, 17, 0x24000, 0x2000);
    }
    tracker.Record(true, 0xDEAD, 5, 3, 17, 0x50000, 0x1000);

    ASSERT_EQ(tracker.RecordCount(), PotentialShaderWriteTracker::MaxWrites);
    const auto latest = tracker.LatestOverlap(0x25000, 0x100, 17);
    ASSERT_TRUE(latest.has_value());
    EXPECT_EQ(latest->program_hash,
              0x1000U + PotentialShaderWriteTracker::MaxWrites + 2U);
    EXPECT_EQ(latest->address, 0x24000U);
    EXPECT_EQ(latest->size, 0x2000U);
    EXPECT_FALSE(tracker.LatestOverlap(0x70000, 0x1000, 17).has_value());
}

TEST(PotentialShaderWriteProvenance, SameAddressFromOldMappingGenerationCannotMatch) {
    PotentialShaderWriteTracker tracker;
    tracker.Record(true, 0x50784801, 5, 2, 17, 0x242710110ULL, 0xB00);

    EXPECT_TRUE(tracker.LatestOverlap(0x242710000ULL, 16_KB, 17).has_value());
    EXPECT_FALSE(tracker.LatestOverlap(0x242710000ULL, 16_KB, 18).has_value());
}

TEST(PotentialShaderWriteProvenance, StructuredRangeLogNamesWriterWithoutGuestPayload) {
    SrtCaptureProvenance capture;
    capture.Record({
        .address = 0x24000,
        .size = 0x4000,
        .source = ShaderCaptureSource::GpuBuffer,
        .potential_writer = PotentialShaderWriteProvenance{
            .program_hash = 0x50784801,
            .stage = 5,
            .descriptor_index = 2,
            .mapping_token = 17,
            .address = 0x24110,
            .size = 0xB00,
            .sequence = 42,
        },
    });
    const ShaderCaptureProgramProvenance program{
        .program_hash = 0xFCD79FD1,
        .stage = 5,
        .succeeded = true,
        .ranges = {capture.Ranges().begin(), capture.Ranges().end()},
    };

    const auto formatted = FormatShaderCaptureProvenance(program);
    EXPECT_NE(formatted.find("\"writer\":{"), std::string::npos);
    EXPECT_NE(formatted.find("0000000050784801"), std::string::npos);
    EXPECT_NE(formatted.find("0000000000024110"), std::string::npos);
    EXPECT_NE(formatted.find("\"kind\":\"admitted-potential-write\""), std::string::npos);
    EXPECT_NE(formatted.find("\"descriptor\":2"), std::string::npos);
    EXPECT_NE(formatted.find("\"mapping_token\":17"), std::string::npos);
    EXPECT_EQ(formatted.find("guest_payload"), std::string::npos);
}

TEST(PotentialShaderWriteProvenance, SnapshotCopyDoesNotGuessWriterWithoutMappingGeneration) {
    std::ifstream input{SHADPS4_BUFFER_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input.is_open());
    const std::string source{std::istreambuf_iterator<char>{input},
                             std::istreambuf_iterator<char>{}};
    const auto function = source.find("bool BufferCache::ReadMemorySnapshot(");
    const auto next_function = source.find("template <bool async", function);
    ASSERT_NE(function, std::string::npos);
    ASSERT_NE(next_function, std::string::npos);
    const auto body = source.substr(function, next_function - function);
    const auto writer_query = body.find("potential_shader_writes.LatestOverlap");
    EXPECT_EQ(writer_query, std::string::npos);
}

TEST(PotentialShaderWriteProvenance, SrtCaptureQueriesWriterWithAcceptedMappingToken) {
    std::ifstream input{SHADPS4_SRT_MEMORY_SYNC_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input.is_open());
    const std::string source{std::istreambuf_iterator<char>{input},
                             std::istreambuf_iterator<char>{}};
    const auto function = source.find("class SrtMemoryReadSession");
    ASSERT_NE(function, std::string::npos);
    const auto body = source.substr(function);
    const auto query = body.find("LatestPotentialShaderWrite(");
    ASSERT_NE(query, std::string::npos);
    const auto query_body = body.substr(query, 180);
    EXPECT_NE(query_body.find("page_address"), std::string::npos);
    EXPECT_NE(query_body.find("TrackingPageSize"), std::string::npos);
    EXPECT_NE(query_body.find("mapping_token"), std::string::npos);
}

TEST(PotentialShaderWriteProvenance, ComputeWriterIsRecordedOnlyAfterDispatchAdmission) {
    std::ifstream input{SHADPS4_RASTERIZER_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input.is_open());
    const std::string source{std::istreambuf_iterator<char>{input},
                             std::istreambuf_iterator<char>{}};
    const auto direct = source.find("void Rasterizer::DispatchDirect()");
    const auto indirect = source.find("void Rasterizer::DispatchIndirect(", direct);
    ASSERT_NE(direct, std::string::npos);
    ASSERT_NE(indirect, std::string::npos);
    const auto body = source.substr(direct, indirect - direct);
    const auto admission = body.find("AreComputeBufferBindingsCurrent(");
    const auto record = body.find("RecordPotentialShaderWrites(cs)");
    const auto dispatch = body.find("cmdbuf.dispatch(");

    ASSERT_NE(admission, std::string::npos);
    ASSERT_NE(record, std::string::npos);
    ASSERT_NE(dispatch, std::string::npos);
    EXPECT_LT(admission, record);
    EXPECT_LT(record, dispatch);

    const auto indirect_end = source.find("void Rasterizer::DispatchIndirect", indirect + 1);
    const auto indirect_body = source.substr(indirect, indirect_end - indirect);
    const auto indirect_admission = indirect_body.find("AreComputeBufferBindingsCurrent(");
    const auto indirect_record = indirect_body.find("RecordPotentialShaderWrites(");
    const auto indirect_dispatch = indirect_body.find("cmdbuf.dispatchIndirect(");
    ASSERT_NE(indirect_admission, std::string::npos);
    ASSERT_NE(indirect_record, std::string::npos);
    ASSERT_NE(indirect_dispatch, std::string::npos);
    EXPECT_LT(indirect_admission, indirect_record);
    EXPECT_LT(indirect_record, indirect_dispatch);
}

} // namespace
} // namespace VideoCore
