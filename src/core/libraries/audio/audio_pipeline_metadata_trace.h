// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::AudioPipelineTrace {

constexpr u32 MaxTraceLines = 512;
constexpr u32 MaxTraceBytes = 256 * 1024;
constexpr u32 MaxFirstSeenShapesPerKind = 32;
constexpr u32 MaxTransitionsPerKind = 64;
constexpr u32 MaxRenderBuffersPerCall = 32;

enum class SampleFormat : u8 {
    PcmS16,
    Float32,
};

enum class SignalClass : u8 {
    Silent,
    Nonzero,
    NonFinite,
};

struct SignalAnalysis {
    u64 samples{};
    u64 zero_samples{};
    u64 nonzero_samples{};
    u64 nonfinite_samples{};

    [[nodiscard]] SignalClass Classification() const noexcept;
};

class SignalAnalyzer {
public:
    void ObservePcmS16(s16 sample) noexcept;
    void ObserveFloat32(float sample) noexcept;
    [[nodiscard]] SignalAnalysis Result() const noexcept;

private:
    SignalAnalysis result{};
};

struct RackCreateMetadata {
    u32 rack_id{};
    u32 max_voices{};
    u32 max_input_delay_blocks{};
    u32 max_matrices{};
    u32 max_ports{};
    bool succeeded{};
};

struct VoiceControlMetadata {
    u32 control_id{};
    u16 command_size{};
    s16 next_offset{};
    bool has_event{};
    u32 event_id{};
    bool accepted{};
    u32 source_rack_id{};
    u32 source_voice_index{};
    bool source_resolved{};
};

struct SamplerEventMetadata {
    u32 source_rack_id{};
    u32 source_voice_index{};
    u32 event_id{};
    bool accepted{};
};

struct SamplerSetupMetadata {
    u32 source_rack_id{};
    u32 source_voice_index{};
    u32 waveform_type{};
    u32 channels{};
    u32 sample_rate{};
    // Opaque ABI codec configuration. Its meaning is deliberately not inferred here.
    u32 codec_config{};
    u32 frame_offset{};
    u32 frame_margin{};
    u32 flags{};
    bool atrac9_applicable{};
    bool atrac9_config_valid{};
    bool atrac9_declared_format_matches{};
    u32 atrac9_frame_bytes{};
    u32 atrac9_frames_per_superframe{};
    u32 atrac9_frame_samples{};
    u32 atrac9_superframe_bytes{};
    u32 atrac9_superframe_samples{};
};

struct WaveformBlocksMetadata {
    u32 source_rack_id{};
    u32 source_voice_index{};
    u32 flags{};
    bool data_present{};
    u32 block_count{};
    u64 aggregate_data_bytes{};
    u64 aggregate_samples{};
    u64 aggregate_skip_samples{};
    u32 minimum_data_offset{};
    u64 maximum_data_end{};
    // True only when base + offset + size arithmetic is representable. This
    // does not claim that the guest range is mapped or readable.
    bool data_range_arithmetic_valid{};
    bool has_repeats{};
    bool metadata_complete{};
};

struct UserFxMetadata {
    u32 control_id{};
    bool callback_present{};
};

struct VoiceMatrixLevelsMetadata {
    u32 source_rack_id{};
    u32 source_voice_index{};
    u32 matrix_id{};
    u32 level_count{};
    bool levels_present{};
};

struct VoicePortVolumeMetadata {
    u32 source_rack_id{};
    u32 source_voice_index{};
    u32 port{};
    float level{};
};

struct VoicePortMatrixMetadata {
    u32 source_rack_id{};
    u32 source_voice_index{};
    u32 port{};
    s32 matrix_id{};
};

struct VoicePatchMetadata {
    u32 source_rack_id{};
    u32 source_voice_index{};
    u32 source_port{};
    u32 destination_input_id{};
    u32 destination_rack_id{};
    u32 destination_voice_index{};
    bool destination_resolved{};
    bool destination_present{};
};

enum class PatchResolutionPhase : u8 {
    Committed,
    RenderTransition,
};

enum class DestinationExactMatch : u8 {
    None,
    System,
    Rack,
    Voice,
};

enum class DestinationOffsetClass : u8 {
    OutsideLiveRack,
    RackBase,
    RackHeader,
    VoiceSlotAligned,
    VoiceSlotInterior,
    RackTail,
};

enum class DestinationTombstoneKind : u8 {
    None,
    System,
    Rack,
    Voice,
};

struct VoicePatchStateMetadata {
    u32 source_rack_id{};
    u32 source_voice_index{};
    u32 source_port{};
    u32 destination_input_id{};
    u32 destination_rack_id{};
    u32 destination_voice_index{};
    bool destination_present{};
    bool destination_resolved{};
    bool destination_same_system{};
    DestinationExactMatch destination_exact_match{};
    bool destination_inside_live_rack{};
    DestinationOffsetClass destination_offset_class{};
    bool destination_candidate_voice_index_valid{};
    u32 destination_candidate_voice_index{};
    bool destination_previously_returned{};
    u64 destination_live_generation{};
    DestinationTombstoneKind destination_tombstone_kind{};
    u64 destination_tombstone_generation{};
    bool destination_generation_changed_since_commit{};
    PatchResolutionPhase phase{};
};

enum class RackLifecyclePhase : u8 {
    Created,
    Destroyed,
};

struct RackLifecycleMetadata {
    u32 rack_id{};
    RackLifecyclePhase phase{};
    u64 generation{};
    u32 voice_count{};
    bool reused_identity{};
    u64 previous_generation{};
};

enum class SamplerPlaybackState : u8 {
    Empty,
    Playing,
    Paused,
    Stopped,
};

enum class SamplerStateObservation : u8 {
    RenderTransition,
    Poll,
};

struct SamplerVoiceStateMetadata {
    u32 source_rack_id{};
    u32 source_voice_index{};
    SamplerPlaybackState previous_state{};
    SamplerPlaybackState current_state{};
    SamplerStateObservation observation{};
};

struct MasteringOutputMetadata {
    u32 source_rack_id{};
    u32 source_voice_index{};
    u32 output_id{};
};

struct RenderBufferMetadata {
    u32 output_index{};
    u32 waveform_type{};
    u32 channels{};
    u64 buffer_bytes{};
};

struct RenderBufferTracePlan {
    u32 traced_buffer_count{};
    bool metadata_complete{};
};

[[nodiscard]] constexpr RenderBufferTracePlan PlanRenderBufferTrace(u32 buffer_count) noexcept {
    const u32 traced_buffer_count =
        buffer_count < MaxRenderBuffersPerCall ? buffer_count : MaxRenderBuffersPerCall;
    return {
        .traced_buffer_count = traced_buffer_count,
        .metadata_complete = traced_buffer_count == buffer_count,
    };
}

struct SystemRenderMetadata {
    u32 buffer_count{};
    u32 traced_buffer_count{};
    bool buffer_metadata_complete{};
    u64 output_bytes{};
    bool staged{};
    bool render_succeeded{};
    bool commit_succeeded{};
};

struct AudioOutputMetadata {
    SampleFormat format{};
    u32 channels{};
    u32 sample_rate{};
    u32 frames{};
    SignalAnalysis signal{};
};

void RecordRackCreate(RackCreateMetadata metadata) noexcept;
void RecordVoiceControl(VoiceControlMetadata metadata) noexcept;
void RecordSamplerEvent(SamplerEventMetadata metadata) noexcept;
void RecordSamplerSetup(SamplerSetupMetadata metadata) noexcept;
void RecordWaveformBlocks(WaveformBlocksMetadata metadata) noexcept;
void RecordUserFx(UserFxMetadata metadata) noexcept;
void RecordVoiceMatrixLevels(VoiceMatrixLevelsMetadata metadata) noexcept;
void RecordVoicePortVolume(VoicePortVolumeMetadata metadata) noexcept;
void RecordVoicePortMatrix(VoicePortMatrixMetadata metadata) noexcept;
void RecordVoicePatch(VoicePatchMetadata metadata) noexcept;
void RecordVoicePatchState(VoicePatchStateMetadata metadata) noexcept;
void RecordRackLifecycle(RackLifecycleMetadata metadata) noexcept;
void RecordSamplerVoiceState(SamplerVoiceStateMetadata metadata) noexcept;
void RecordMasteringOutput(MasteringOutputMetadata metadata) noexcept;
void RecordRenderBuffer(RenderBufferMetadata metadata) noexcept;
void RecordSystemRender(SystemRenderMetadata metadata) noexcept;
void RecordAudioOutput(AudioOutputMetadata metadata) noexcept;
[[nodiscard]] bool IsEnabled() noexcept;

namespace Testing {

void Reset() noexcept;
void Finalize() noexcept;
void FinalizeForProcessExit() noexcept;
[[nodiscard]] bool ProcessExitFinalizerRegistered() noexcept;
[[nodiscard]] u64 StateLockAcquisitions() noexcept;

} // namespace Testing

} // namespace Libraries::AudioPipelineTrace
