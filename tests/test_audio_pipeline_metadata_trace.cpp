// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ranges>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "core/libraries/audio/audio_pipeline_metadata_trace.h"

namespace Libraries::AudioPipelineTrace {
namespace {

u64 CurrentTestProcessId() {
#ifdef _WIN32
    return static_cast<u64>(_getpid());
#else
    return static_cast<u64>(getpid());
#endif
}

std::string UniqueTraceFileToken() {
    static std::atomic<u64> sequence{};
    return std::to_string(CurrentTestProcessId()) + "-" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

class AudioPipelineMetadataTraceTest : public testing::Test {
protected:
    void SetUp() override {
        Testing::Reset();
        ClearTracePath();
        path = std::filesystem::temp_directory_path() /
               ("shadps4-audio-pipeline-trace-" + UniqueTraceFileToken() + ".jsonl");
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    void TearDown() override {
        Testing::Reset();
        ClearTracePath();
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    void EnableTrace() {
#ifdef _WIN32
        ASSERT_EQ(_putenv_s("SHADPS4_AUDIO_PIPELINE_TRACE_PATH", path.string().c_str()), 0);
#else
        ASSERT_EQ(setenv("SHADPS4_AUDIO_PIPELINE_TRACE_PATH", path.string().c_str(), 1), 0);
#endif
        Testing::Reset();
    }

    static void ClearTracePath() {
#ifdef _WIN32
        ASSERT_EQ(_putenv_s("SHADPS4_AUDIO_PIPELINE_TRACE_PATH", ""), 0);
#else
        ASSERT_EQ(unsetenv("SHADPS4_AUDIO_PIPELINE_TRACE_PATH"), 0);
#endif
    }

    std::vector<nlohmann::json> ReadLines() const {
        std::ifstream input(path, std::ios::binary);
        std::vector<nlohmann::json> lines;
        std::string line;
        while (std::getline(input, line)) {
            lines.push_back(nlohmann::json::parse(line));
        }
        return lines;
    }

    std::filesystem::path path;
};

TEST(AudioPipelineSignalAnalysis, ClassifiesPcmAndFloatWithoutRetainingSamples) {
    SignalAnalyzer pcm;
    pcm.ObservePcmS16(0);
    pcm.ObservePcmS16(-1);
    pcm.ObservePcmS16(32767);
    EXPECT_EQ(pcm.Result().samples, 3);
    EXPECT_EQ(pcm.Result().zero_samples, 1);
    EXPECT_EQ(pcm.Result().nonzero_samples, 2);
    EXPECT_EQ(pcm.Result().nonfinite_samples, 0);
    EXPECT_EQ(pcm.Result().Classification(), SignalClass::Nonzero);

    SignalAnalyzer floating;
    floating.ObserveFloat32(0.0F);
    floating.ObserveFloat32(-0.0F);
    floating.ObserveFloat32(0.25F);
    floating.ObserveFloat32(std::numeric_limits<float>::infinity());
    floating.ObserveFloat32(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(floating.Result().samples, 5);
    EXPECT_EQ(floating.Result().zero_samples, 2);
    EXPECT_EQ(floating.Result().nonzero_samples, 1);
    EXPECT_EQ(floating.Result().nonfinite_samples, 2);
    EXPECT_EQ(floating.Result().Classification(), SignalClass::NonFinite);

    SignalAnalyzer subnormal;
    subnormal.ObserveFloat32(std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(subnormal.Result().nonzero_samples, 1);
    EXPECT_EQ(subnormal.Result().Classification(), SignalClass::Nonzero);
}

TEST_F(AudioPipelineMetadataTraceTest, IsOptInCreateExclusiveAndUsesPrivacySafeSchema) {
    EXPECT_FALSE(IsEnabled());
    RecordRackCreate({0x1000, 2, 3, 4, 5, true});
    Testing::Finalize();
    EXPECT_FALSE(std::filesystem::exists(path));

    EnableTrace();
    EXPECT_TRUE(IsEnabled());
    RecordRackCreate({0x1000, 2, 3, 4, 5, true});
    RecordVoiceControl({6, 12, -8, true, 4, true, 0x1000, 7, true});
    RecordSamplerEvent({0x1000, 7, 0, true});
    RecordSamplerSetup({
        .source_rack_id = 0x1000,
        .source_voice_index = 7,
        .waveform_type = 8,
        .channels = 2,
        .sample_rate = 48000,
        .codec_config = 0x10203040,
        .frame_offset = 16,
        .frame_margin = 4,
        .flags = 1,
        .atrac9_applicable = true,
        .atrac9_config_valid = true,
        .atrac9_declared_format_matches = true,
        .atrac9_frame_bytes = 240,
        .atrac9_frames_per_superframe = 4,
        .atrac9_frame_samples = 256,
        .atrac9_superframe_bytes = 960,
        .atrac9_superframe_samples = 1024,
    });
    RecordWaveformBlocks({
        .source_rack_id = 0x1000,
        .source_voice_index = 7,
        .flags = 3,
        .data_present = true,
        .block_count = 3,
        .aggregate_data_bytes = 4096,
        .aggregate_samples = 2048,
        .aggregate_skip_samples = 64,
        .minimum_data_offset = 32,
        .maximum_data_end = 4128,
        .data_range_arithmetic_valid = true,
        .has_repeats = true,
        .metadata_complete = true,
    });
    RecordUserFx({0x20000004, true});
    RecordVoiceMatrixLevels({0x1000, 7, 3, 64, true});
    RecordVoicePortVolume({0x1000, 7, 2, 1.0F});
    RecordVoicePortMatrix({0x1000, 7, 2, -1});
    RecordVoicePatch({0x1000, 7, 2, 4, 0x2000, 1, true, true, PatchEndpointKind::Voice, true});
    RecordVoicePatchState({
        .source_rack_id = 0x1000,
        .source_voice_index = 7,
        .source_port = 2,
        .destination_input_id = 4,
        .destination_rack_id = 0x2000,
        .destination_voice_index = 1,
        .destination_present = true,
        .destination_resolved = true,
        .destination_same_system = true,
        .destination_exact_match = DestinationExactMatch::Voice,
        .destination_inside_live_rack = true,
        .destination_offset_class = DestinationOffsetClass::VoiceSlotAligned,
        .destination_candidate_voice_index_valid = true,
        .destination_candidate_voice_index = 1,
        .destination_previously_returned = true,
        .destination_live_generation = 11,
        .destination_tombstone_kind = DestinationTombstoneKind::Voice,
        .destination_tombstone_generation = 7,
        .destination_generation_changed_since_commit = true,
        .destination_endpoint_kind = PatchEndpointKind::Voice,
        .destination_voice_index_valid = true,
        .phase = PatchResolutionPhase::Committed,
    });
    RecordRackLifecycle({
        .rack_id = 0x2000,
        .phase = RackLifecyclePhase::Created,
        .generation = 11,
        .voice_count = 2,
        .reused_identity = true,
        .previous_generation = 7,
    });
    RecordSamplerVoiceState({0x1000, 7, SamplerPlaybackState::Empty, SamplerPlaybackState::Playing,
                             SamplerStateObservation::RenderTransition});
    RecordMasteringOutput({0x3000, 1, 2});
    RecordRenderBuffer({0, 8, 2, 2048});
    RecordRenderBuffer({1, 8, 2, 2048});
    RecordSystemRender({2, 2, true, 4096, true, true, true});
    RecordAudioOutput({SampleFormat::PcmS16, 2, 48000, 1024, {2048, 1024, 1024, 0}});
    Testing::Finalize();

    const auto first_contents = [&] {
        std::ifstream input(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>{input}, {});
    }();
    ASSERT_FALSE(first_contents.empty());
    EXPECT_EQ(first_contents.find("pointer"), std::string::npos);
    EXPECT_EQ(first_contents.find("handle"), std::string::npos);
    EXPECT_EQ(first_contents.find("address"), std::string::npos);
    EXPECT_EQ(first_contents.find("thread"), std::string::npos);
    EXPECT_EQ(first_contents.find("title"), std::string::npos);
    EXPECT_EQ(first_contents.find("payload"), std::string::npos);
    EXPECT_EQ(first_contents.find("hash"), std::string::npos);

    const auto lines = ReadLines();
    ASSERT_GE(lines.size(), 6);
    EXPECT_EQ(lines.front().at("event"), "trace_header");
    EXPECT_EQ(lines.front().at("schema_version"), 8);
    EXPECT_EQ(lines.back().at("event"), "trace_summary");

    const std::unordered_map<std::string, std::unordered_set<std::string>> allowlists{
        {"trace_header",
         {"event", "schema_version", "max_lines", "max_bytes", "max_shapes_per_kind",
          "max_transitions_per_kind", "max_render_buffers_per_call"}},
        {"rack_create",
         {"event", "observation", "occurrence", "rack_id", "max_voices", "max_input_delay_blocks",
          "max_matrices", "max_ports", "succeeded"}},
        {"rack_lifecycle",
         {"event", "occurrence", "rack_id", "phase", "generation", "voice_count", "reused_identity",
          "previous_generation"}},
        {"voice_control",
         {"event", "observation", "occurrence", "control_id", "command_size", "next_offset",
          "has_event", "event_id", "accepted", "source_rack_id", "source_voice_index",
          "source_resolved"}},
        {"sampler_event",
         {"event", "occurrence", "source_rack_id", "source_voice_index", "event_id", "accepted"}},
        {"sampler_setup",
         {"event",
          "observation",
          "occurrence",
          "source_rack_id",
          "source_voice_index",
          "waveform_type",
          "channels",
          "sample_rate",
          "codec_config",
          "frame_offset",
          "frame_margin",
          "flags",
          "atrac9_applicable",
          "atrac9_config_valid",
          "atrac9_declared_format_matches",
          "atrac9_frame_bytes",
          "atrac9_frames_per_superframe",
          "atrac9_frame_samples",
          "atrac9_superframe_bytes",
          "atrac9_superframe_samples"}},
        {"waveform_blocks",
         {"event", "observation", "occurrence", "source_rack_id", "source_voice_index", "flags",
          "data_present", "block_count", "aggregate_data_bytes", "aggregate_samples",
          "aggregate_skip_samples", "minimum_data_offset", "maximum_data_end",
          "data_range_arithmetic_valid", "has_repeats", "metadata_complete"}},
        {"user_fx", {"event", "observation", "occurrence", "control_id", "callback_present"}},
        {"voice_matrix_levels",
         {"event", "observation", "occurrence", "source_rack_id", "source_voice_index", "matrix_id",
          "level_count", "levels_present"}},
        {"voice_port_volume",
         {"event", "observation", "occurrence", "source_rack_id", "source_voice_index", "port",
          "level_shape"}},
        {"voice_port_matrix",
         {"event", "observation", "occurrence", "source_rack_id", "source_voice_index", "port",
          "matrix_id"}},
        {"voice_patch",
         {"event", "observation", "occurrence", "source_rack_id", "source_voice_index",
          "source_port", "destination_input_id", "destination_rack_id", "destination_voice_index",
          "destination_resolved", "destination_present", "destination_endpoint_kind",
          "destination_voice_index_valid"}},
        {"voice_patch_state",
         {"event",
          "occurrence",
          "phase",
          "source_rack_id",
          "source_voice_index",
          "source_port",
          "destination_input_id",
          "destination_rack_id",
          "destination_voice_index",
          "destination_resolved",
          "destination_present",
          "destination_same_system",
          "destination_exact_match",
          "destination_inside_live_rack",
          "destination_offset_class",
          "destination_candidate_voice_index_valid",
          "destination_candidate_voice_index",
          "destination_previously_returned",
          "destination_live_generation",
          "destination_tombstone_kind",
          "destination_tombstone_generation",
          "destination_generation_changed_since_commit",
          "destination_endpoint_kind",
          "destination_voice_index_valid"}},
        {"sampler_voice_state",
         {"event", "occurrence", "observation", "source_rack_id", "source_voice_index",
          "previous_state", "current_state"}},
        {"mastering_output",
         {"event", "observation", "occurrence", "source_rack_id", "source_voice_index",
          "output_id"}},
        {"render_buffer",
         {"event", "observation", "occurrence", "output_index", "waveform_type", "channels",
          "buffer_bytes"}},
        {"system_render",
         {"event", "observation", "occurrence", "buffer_count", "traced_buffer_count",
          "buffer_metadata_complete", "output_bytes", "staged", "render_succeeded",
          "commit_succeeded"}},
        {"audio_output",
         {"event", "observation", "occurrence", "format", "channels", "sample_rate", "frames",
          "samples", "zero_samples", "nonzero_samples", "nonfinite_samples", "signal"}},
        {"render_transition", {"event", "occurrence", "from", "to"}},
        {"audio_transition", {"event", "occurrence", "from", "to"}},
        {"trace_summary",
         {"event",
          "rack_creates",
          "voice_controls",
          "sampler_events",
          "sampler_setups",
          "waveform_block_lists",
          "user_fx_controls",
          "voice_matrix_levels",
          "voice_port_volumes",
          "voice_port_matrices",
          "voice_patches",
          "voice_patch_states",
          "sampler_voice_patch_states",
          "sampler_voice_patch_commits",
          "sampler_voice_patch_resolution_transitions",
          "rack_lifecycle_events",
          "sampler_voice_states",
          "mastering_outputs",
          "render_buffers",
          "system_renders",
          "audio_outputs",
          "written_lines",
          "written_bytes",
          "dropped_lines",
          "rack_shapes",
          "voice_shapes",
          "sampler_shapes",
          "waveform_block_shapes",
          "user_fx_shapes",
          "voice_matrix_level_shapes",
          "voice_port_volume_shapes",
          "voice_port_matrix_shapes",
          "voice_patch_shapes",
          "mastering_output_shapes",
          "render_buffer_shapes",
          "render_shapes",
          "audio_shapes",
          "rack_shapes_saturated",
          "voice_shapes_saturated",
          "sampler_shapes_saturated",
          "waveform_block_shapes_saturated",
          "user_fx_shapes_saturated",
          "voice_matrix_level_shapes_saturated",
          "voice_port_volume_shapes_saturated",
          "voice_port_matrix_shapes_saturated",
          "voice_patch_shapes_saturated",
          "mastering_output_shapes_saturated",
          "render_buffer_shapes_saturated",
          "render_shapes_saturated",
          "audio_shapes_saturated",
          "render_transitions",
          "audio_transitions",
          "render_transitions_saturated",
          "audio_transitions_saturated",
          "sampler_events_saturated",
          "voice_patch_states_saturated",
          "voice_patch_commits_saturated",
          "voice_patch_resolution_transitions_saturated",
          "sampler_voice_patch_states_saturated",
          "sampler_voice_patch_commits_saturated",
          "sampler_voice_patch_resolution_transitions_saturated",
          "rack_lifecycle_events_saturated",
          "sampler_voice_states_saturated",
          "sampler_voice_state_polls_saturated",
          "sampler_voice_state_transitions_saturated"}},
    };
    for (const auto& line : lines) {
        const auto& allowed = allowlists.at(line.at("event").get<std::string>());
        for (const auto& [key, ignored] : line.items()) {
            EXPECT_TRUE(allowed.contains(key)) << key << " in " << line.dump();
        }
    }

    Testing::Reset();
    RecordRackCreate({0x2000, 8, 0, 1, 1, true});
    Testing::Finalize();
    std::ifstream second_input(path, std::ios::binary);
    const std::string second_contents(std::istreambuf_iterator<char>{second_input}, {});
    EXPECT_EQ(second_contents, first_contents);
}

TEST_F(AudioPipelineMetadataTraceTest, DisabledFastPathAvoidsRecorderLocksAndResetRechecksOptIn) {
    EXPECT_FALSE(IsEnabled());
    const u64 lock_count_after_discovery = Testing::StateLockAcquisitions();

    for (u32 occurrence = 0; occurrence < 64; ++occurrence) {
        EXPECT_FALSE(IsEnabled());
        RecordVoiceControl({6, 12, 0, false, 0, true});
    }
    EXPECT_EQ(Testing::StateLockAcquisitions(), lock_count_after_discovery);

    EnableTrace();
    EXPECT_TRUE(IsEnabled());
    RecordVoiceControl({6, 12, 0, false, 0, true});
    Testing::Finalize();
    EXPECT_TRUE(std::filesystem::exists(path));
}

TEST_F(AudioPipelineMetadataTraceTest, ProcessExitFinalizerWritesExactlyOneTerminalSummary) {
    EnableTrace();
    EXPECT_TRUE(Testing::ProcessExitFinalizerRegistered());
    RecordRackCreate({0x1000, 2, 3, 4, 5, true});

    Testing::FinalizeForProcessExit();
    Testing::FinalizeForProcessExit();
    RecordRackCreate({0x2000, 8, 0, 1, 1, true});

    const auto lines = ReadLines();
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ(lines.back().at("event"), "trace_summary");
    EXPECT_EQ(std::ranges::count_if(
                  lines, [](const auto& line) { return line.at("event") == "trace_summary"; }),
              1);
    EXPECT_EQ(lines.back().at("rack_creates"), 1);
}

TEST_F(AudioPipelineMetadataTraceTest, EmitsOnlyPowerOfTwoOccurrenceCheckpoints) {
    EnableTrace();
    for (u32 occurrence = 1; occurrence <= 8; ++occurrence) {
        RecordRackCreate({1, 2, 3, 4, 5, true});
    }
    Testing::Finalize();

    std::vector<u64> checkpoints;
    for (const auto& line : ReadLines()) {
        if (line.at("event") == "rack_create" && line.at("observation") == "checkpoint") {
            checkpoints.push_back(line.at("occurrence"));
        }
    }
    EXPECT_EQ(checkpoints, (std::vector<u64>{1, 2, 4, 8}));
}

TEST_F(AudioPipelineMetadataTraceTest, RecordsOnlyBoundedSamplerAndCallbackShape) {
    EnableTrace();
    RecordSamplerSetup({
        .source_rack_id = 0x1000,
        .source_voice_index = 31,
        .waveform_type = 9,
        .channels = 6,
        .sample_rate = 48000,
        .codec_config = 0x12345678,
        .frame_offset = 1024,
        .frame_margin = 32,
        .flags = 5,
    });
    RecordWaveformBlocks({
        .source_rack_id = 0x1000,
        .source_voice_index = 31,
        .flags = 7,
        .data_present = true,
        .block_count = 4,
        .aggregate_data_bytes = 65536,
        .aggregate_samples = 32768,
        .has_repeats = true,
        .metadata_complete = true,
    });
    RecordUserFx({0x10000008, false});
    Testing::Finalize();

    const auto lines = ReadLines();
    const auto sampler = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "sampler_setup"; });
    ASSERT_NE(sampler, lines.end());
    EXPECT_EQ(sampler->at("source_rack_id"), 0x1000);
    EXPECT_EQ(sampler->at("source_voice_index"), 31);
    EXPECT_EQ(sampler->at("waveform_type"), 9);
    EXPECT_EQ(sampler->at("channels"), 6);
    EXPECT_EQ(sampler->at("sample_rate"), 48000);
    EXPECT_EQ(sampler->at("codec_config"), 0x12345678);
    EXPECT_EQ(sampler->at("frame_offset"), 1024);
    EXPECT_EQ(sampler->at("frame_margin"), 32);
    EXPECT_EQ(sampler->at("flags"), 5);

    const auto blocks = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "waveform_blocks"; });
    ASSERT_NE(blocks, lines.end());
    EXPECT_EQ(blocks->at("source_rack_id"), 0x1000);
    EXPECT_EQ(blocks->at("source_voice_index"), 31);
    EXPECT_EQ(blocks->at("flags"), 7);
    EXPECT_TRUE(blocks->at("data_present"));
    EXPECT_EQ(blocks->at("block_count"), 4);
    EXPECT_EQ(blocks->at("aggregate_data_bytes"), 65536);
    EXPECT_EQ(blocks->at("aggregate_samples"), 32768);
    EXPECT_TRUE(blocks->at("has_repeats"));
    EXPECT_TRUE(blocks->at("metadata_complete"));

    const auto callback =
        std::ranges::find_if(lines, [](const auto& line) { return line.at("event") == "user_fx"; });
    ASSERT_NE(callback, lines.end());
    EXPECT_EQ(callback->at("control_id"), 0x10000008);
    EXPECT_FALSE(callback->at("callback_present"));

    const auto contents = [&] {
        std::ifstream input(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>{input}, {});
    }();
    EXPECT_EQ(contents.find("pointer"), std::string::npos);
    EXPECT_EQ(contents.find("address"), std::string::npos);
    EXPECT_EQ(contents.find("payload"), std::string::npos);
}

TEST_F(AudioPipelineMetadataTraceTest, BoundsPrivacySafeRenderBufferEvidencePerCall) {
    const auto empty = PlanRenderBufferTrace(0);
    EXPECT_EQ(empty.traced_buffer_count, 0);
    EXPECT_TRUE(empty.metadata_complete);

    const auto exact = PlanRenderBufferTrace(MaxRenderBuffersPerCall);
    EXPECT_EQ(exact.traced_buffer_count, MaxRenderBuffersPerCall);
    EXPECT_TRUE(exact.metadata_complete);

    const auto truncated = PlanRenderBufferTrace(MaxRenderBuffersPerCall + 1);
    EXPECT_EQ(truncated.traced_buffer_count, MaxRenderBuffersPerCall);
    EXPECT_FALSE(truncated.metadata_complete);

    EnableTrace();
    RecordRenderBuffer({2, 9, 6, 8192});
    RecordSystemRender({MaxRenderBuffersPerCall + 1, truncated.traced_buffer_count,
                        truncated.metadata_complete, 8192, true, true, true});
    Testing::Finalize();

    const auto lines = ReadLines();
    const auto output = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "render_buffer"; });
    ASSERT_NE(output, lines.end());
    EXPECT_EQ(output->at("output_index"), 2);
    EXPECT_EQ(output->at("waveform_type"), 9);
    EXPECT_EQ(output->at("channels"), 6);
    EXPECT_EQ(output->at("buffer_bytes"), 8192);

    const auto render = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "system_render"; });
    ASSERT_NE(render, lines.end());
    EXPECT_EQ(render->at("buffer_count"), MaxRenderBuffersPerCall + 1);
    EXPECT_EQ(render->at("traced_buffer_count"), MaxRenderBuffersPerCall);
    EXPECT_FALSE(render->at("buffer_metadata_complete"));

    const std::string contents = [&] {
        std::ifstream input(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>{input}, {});
    }();
    EXPECT_EQ(contents.find("pointer"), std::string::npos);
    EXPECT_EQ(contents.find("address"), std::string::npos);
    EXPECT_EQ(contents.find("payload"), std::string::npos);
    EXPECT_EQ(contents.find("pcm"), std::string::npos);
}

TEST_F(AudioPipelineMetadataTraceTest, RecordsOnlyPrivacySafeRoutingTopologyShapes) {
    EnableTrace();
    RecordVoiceMatrixLevels({0x1000, 31, 9, 64, true});
    RecordVoicePortVolume({0x1000, 31, 5, 0.0F});
    RecordVoicePortVolume({0x1000, 31, 6, 1.0F});
    RecordVoicePortVolume({0x1000, 31, 7, 0.5F});
    RecordVoicePortVolume({0x1000, 31, 8, std::numeric_limits<float>::quiet_NaN()});
    RecordVoicePortMatrix({0x1000, 31, 6, -1});
    RecordVoicePatch({0x1000, 31, 6, 4, 0x2000, 2, true, true, PatchEndpointKind::Voice, true});
    RecordVoicePatch({0x1000, 31, 7, 5, 0, 0, false, false, PatchEndpointKind::None, false});
    RecordMasteringOutput({0x3000, 2, 1});
    Testing::Finalize();

    const auto lines = ReadLines();
    const auto matrix_levels = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "voice_matrix_levels"; });
    ASSERT_NE(matrix_levels, lines.end());
    EXPECT_EQ(matrix_levels->at("source_rack_id"), 0x1000);
    EXPECT_EQ(matrix_levels->at("source_voice_index"), 31);
    EXPECT_EQ(matrix_levels->at("matrix_id"), 9);
    EXPECT_EQ(matrix_levels->at("level_count"), 64);
    EXPECT_TRUE(matrix_levels->at("levels_present"));

    const auto zero_volume = std::ranges::find_if(lines, [](const auto& line) {
        return line.at("event") == "voice_port_volume" && line.at("port") == 5;
    });
    ASSERT_NE(zero_volume, lines.end());
    EXPECT_EQ(zero_volume->at("level_shape"), "zero");
    const auto unity_volume = std::ranges::find_if(lines, [](const auto& line) {
        return line.at("event") == "voice_port_volume" && line.at("port") == 6;
    });
    ASSERT_NE(unity_volume, lines.end());
    EXPECT_EQ(unity_volume->at("level_shape"), "unity");
    const auto other_finite_volume = std::ranges::find_if(lines, [](const auto& line) {
        return line.at("event") == "voice_port_volume" && line.at("port") == 7;
    });
    ASSERT_NE(other_finite_volume, lines.end());
    EXPECT_EQ(other_finite_volume->at("level_shape"), "other_finite");
    const auto nonfinite_volume = std::ranges::find_if(lines, [](const auto& line) {
        return line.at("event") == "voice_port_volume" && line.at("port") == 8;
    });
    ASSERT_NE(nonfinite_volume, lines.end());
    EXPECT_EQ(nonfinite_volume->at("level_shape"), "nonfinite");

    const auto resolved_patch = std::ranges::find_if(lines, [](const auto& line) {
        return line.at("event") == "voice_patch" && line.at("destination_resolved") == true;
    });
    ASSERT_NE(resolved_patch, lines.end());
    EXPECT_EQ(resolved_patch->at("source_port"), 6);
    EXPECT_EQ(resolved_patch->at("destination_input_id"), 4);
    EXPECT_EQ(resolved_patch->at("destination_rack_id"), 0x2000);
    EXPECT_EQ(resolved_patch->at("destination_voice_index"), 2);
    EXPECT_TRUE(resolved_patch->at("destination_present"));

    const auto unresolved_patch = std::ranges::find_if(lines, [](const auto& line) {
        return line.at("event") == "voice_patch" && line.at("destination_resolved") == false;
    });
    ASSERT_NE(unresolved_patch, lines.end());
    EXPECT_EQ(unresolved_patch->at("destination_rack_id"), 0);
    EXPECT_TRUE(unresolved_patch->at("destination_voice_index").is_null());
    EXPECT_FALSE(unresolved_patch->at("destination_present"));

    const auto output = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "mastering_output"; });
    ASSERT_NE(output, lines.end());
    EXPECT_EQ(output->at("source_voice_index"), 2);
    EXPECT_EQ(output->at("output_id"), 1);

    const auto contents = [&] {
        std::ifstream input(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>{input}, {});
    }();
    EXPECT_EQ(contents.find("handle"), std::string::npos);
    EXPECT_EQ(contents.find("pointer"), std::string::npos);
    EXPECT_EQ(contents.find("address"), std::string::npos);
    EXPECT_EQ(contents.find("payload"), std::string::npos);
}

TEST_F(AudioPipelineMetadataTraceTest, BoundsShapesTransitionsLinesAndBytesUnderConcurrency) {
    EnableTrace();
    for (u32 index = 0; index < 100; ++index) {
        RecordRackCreate({index + 1, index, index, index, index, true});
        RecordVoicePatch({0x1000, index, index, index, 0x2000, index, true, true});
    }
    for (u32 index = 0; index < 1000; ++index) {
        const bool success = (index & 1) == 0;
        RecordSystemRender({
            .buffer_count = 1,
            .traced_buffer_count = 1,
            .buffer_metadata_complete = true,
            .output_bytes = 2048,
            .staged = true,
            .render_succeeded = success,
            .commit_succeeded = success,
        });
        RecordAudioOutput(
            {SampleFormat::PcmS16, 2, 48000, 1024,
             success ? SignalAnalysis{2048, 2048, 0, 0} : SignalAnalysis{2048, 0, 2048, 0}});
    }

    std::vector<std::thread> threads;
    for (u32 thread = 0; thread < 8; ++thread) {
        threads.emplace_back([] {
            for (u32 index = 0; index < 1000; ++index) {
                RecordVoiceControl({6, 12, 0, true, index % 6, true});
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    Testing::Finalize();

    const auto lines = ReadLines();
    ASSERT_FALSE(lines.empty());
    EXPECT_LE(lines.size(), MaxTraceLines);
    EXPECT_LE(std::filesystem::file_size(path), MaxTraceBytes);
    const auto& summary = lines.back();
    ASSERT_EQ(summary.at("event"), "trace_summary");
    EXPECT_EQ(summary.at("rack_creates"), 100);
    EXPECT_EQ(summary.at("voice_patches"), 100);
    EXPECT_EQ(summary.at("voice_controls"), 8000);
    EXPECT_EQ(summary.at("system_renders"), 1000);
    EXPECT_EQ(summary.at("audio_outputs"), 1000);
    EXPECT_EQ(summary.at("rack_shapes"), MaxFirstSeenShapesPerKind);
    EXPECT_TRUE(summary.at("rack_shapes_saturated"));
    EXPECT_EQ(summary.at("voice_patch_shapes"), MaxFirstSeenShapesPerKind);
    EXPECT_TRUE(summary.at("voice_patch_shapes_saturated"));
    EXPECT_EQ(summary.at("render_transitions"), MaxTransitionsPerKind);
    EXPECT_EQ(summary.at("audio_transitions"), MaxTransitionsPerKind);
    EXPECT_TRUE(summary.at("render_transitions_saturated"));
    EXPECT_TRUE(summary.at("audio_transitions_saturated"));
}

TEST_F(AudioPipelineMetadataTraceTest, DedicatedSamplerLifecycleSurvivesGenericShapeSaturation) {
    EnableTrace();
    for (u32 index = 0; index < 100; ++index) {
        RecordVoiceControl({index, 12, 0, false, 0, true, 0x2000, index, true});
    }
    for (u32 index = 0; index < 100; ++index) {
        RecordSamplerEvent({0x1000, index, index % 6, true});
        RecordVoicePatchState({
            .source_rack_id = 0x1000,
            .source_voice_index = index,
            .source_is_sampler = true,
            .destination_rack_id = 0x2000,
            .destination_present = true,
            .destination_resolved = (index & 1) == 0,
            .destination_same_system = (index & 1) == 0,
            .phase = PatchResolutionPhase::Committed,
        });
        RecordSamplerVoiceState({0x1000, index, SamplerPlaybackState::Empty,
                                 SamplerPlaybackState::Empty, SamplerStateObservation::Poll});
    }
    RecordSamplerVoiceState({0x1000, 100, SamplerPlaybackState::Playing,
                             SamplerPlaybackState::Playing, SamplerStateObservation::Poll});
    RecordSamplerVoiceState({0x1000, 101, SamplerPlaybackState::Paused,
                             SamplerPlaybackState::Paused, SamplerStateObservation::Poll});
    RecordSamplerVoiceState({0x1000, 102, SamplerPlaybackState::Stopped,
                             SamplerPlaybackState::Stopped, SamplerStateObservation::Poll});
    RecordVoicePatchState({
        .source_rack_id = 0x1000,
        .source_is_sampler = true,
        .destination_rack_id = 0x2000,
        .destination_present = true,
        .destination_resolved = true,
        .destination_same_system = true,
        .phase = PatchResolutionPhase::RenderTransition,
    });
    RecordSamplerVoiceState({0x1000, 0, SamplerPlaybackState::Empty, SamplerPlaybackState::Playing,
                             SamplerStateObservation::RenderTransition});
    Testing::Finalize();

    const auto lines = ReadLines();
    EXPECT_EQ(std::ranges::count_if(
                  lines, [](const auto& line) { return line.at("event") == "sampler_event"; }),
              MaxTransitionsPerKind);
    EXPECT_EQ(std::ranges::count_if(
                  lines, [](const auto& line) { return line.at("event") == "voice_patch_state"; }),
              MaxSamplerPatchStatesPerPhase + 1);
    EXPECT_EQ(
        std::ranges::count_if(
            lines, [](const auto& line) { return line.at("event") == "sampler_voice_state"; }),
        MaxTransitionsPerKind / 4 + 4);
    EXPECT_TRUE(std::ranges::any_of(lines, [](const auto& line) {
        return line.at("event") == "voice_patch_state" && line.at("phase") == "render_transition";
    }));
    EXPECT_TRUE(std::ranges::any_of(lines, [](const auto& line) {
        return line.at("event") == "sampler_voice_state" &&
               line.at("observation") == "render_transition";
    }));
    for (const char* state : {"playing", "paused", "stopped"}) {
        EXPECT_TRUE(std::ranges::any_of(lines, [state](const auto& line) {
            return line.at("event") == "sampler_voice_state" && line.at("observation") == "poll" &&
                   line.at("current_state") == state;
        })) << state;
    }
    const auto& summary = lines.back();
    ASSERT_EQ(summary.at("event"), "trace_summary");
    EXPECT_EQ(summary.at("sampler_events"), 100);
    EXPECT_EQ(summary.at("voice_patch_states"), 101);
    EXPECT_EQ(summary.at("sampler_voice_states"), 104);
    EXPECT_TRUE(summary.at("voice_shapes_saturated"));
    EXPECT_TRUE(summary.at("sampler_events_saturated"));
    EXPECT_TRUE(summary.at("voice_patch_states_saturated"));
    EXPECT_TRUE(summary.at("voice_patch_commits_saturated"));
    EXPECT_FALSE(summary.at("voice_patch_resolution_transitions_saturated"));
    EXPECT_TRUE(summary.at("sampler_voice_states_saturated"));
    EXPECT_TRUE(summary.at("sampler_voice_state_polls_saturated"));
    EXPECT_FALSE(summary.at("sampler_voice_state_transitions_saturated"));
}

TEST_F(AudioPipelineMetadataTraceTest, DedicatedSamplerPatchStateSurvivesGenericCommitSaturation) {
    EnableTrace();
    for (u32 index = 0; index < 100; ++index) {
        RecordVoicePatchState({
            .source_rack_id = 0x2000,
            .source_voice_index = index,
            .destination_rack_id = 0x2000,
            .destination_present = true,
            .destination_resolved = true,
            .destination_same_system = true,
            .phase = PatchResolutionPhase::Committed,
        });
    }
    for (u32 index = 0; index < 4; ++index) {
        RecordVoicePatchState({
            .source_rack_id = 0x1000,
            .source_voice_index = index * 6,
            .source_is_sampler = true,
            .destination_present = true,
            .destination_resolved = false,
            .destination_offset_class = DestinationOffsetClass::VoiceSlotInterior,
            .destination_candidate_voice_index_valid = true,
            .destination_candidate_voice_index = index,
            .phase = PatchResolutionPhase::Committed,
        });
    }
    Testing::Finalize();

    const auto lines = ReadLines();
    EXPECT_EQ(std::ranges::count_if(lines,
                                    [](const auto& line) {
                                        return line.at("event") == "voice_patch_state" &&
                                               line.at("source_rack_id") == 0x1000;
                                    }),
              4);
    for (u32 index = 0; index < 4; ++index) {
        EXPECT_TRUE(std::ranges::any_of(lines, [index](const auto& line) {
            return line.at("event") == "voice_patch_state" && line.at("source_rack_id") == 0x1000 &&
                   line.at("source_voice_index") == index * 6 &&
                   line.at("destination_offset_class") == "voice_slot_interior" &&
                   line.at("destination_candidate_voice_index_valid") &&
                   line.at("destination_candidate_voice_index") == index;
        })) << index;
    }
    const auto& summary = lines.back();
    ASSERT_EQ(summary.at("event"), "trace_summary");
    EXPECT_EQ(summary.at("voice_patch_states"), 104);
    EXPECT_EQ(summary.at("sampler_voice_patch_states"), 4);
    EXPECT_TRUE(summary.at("voice_patch_commits_saturated"));
    EXPECT_FALSE(summary.at("sampler_voice_patch_states_saturated"));
}

} // namespace
} // namespace Libraries::AudioPipelineTrace
