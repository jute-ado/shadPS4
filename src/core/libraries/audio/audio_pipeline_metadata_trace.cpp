// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/audio/audio_pipeline_metadata_trace.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include <nlohmann/json.hpp>

namespace Libraries::AudioPipelineTrace {
namespace {

using Json = nlohmann::ordered_json;

constexpr size_t SummaryReserveBytes = 4096;
constexpr size_t SamplerPlaybackStateCount = 4;
constexpr size_t MaxSamplerPollShapesPerState = MaxTransitionsPerKind / SamplerPlaybackStateCount;
static_assert(MaxTransitionsPerKind % SamplerPlaybackStateCount == 0);

template <typename T>
void SaturatingIncrement(T& value) noexcept {
    if (value != std::numeric_limits<T>::max()) {
        ++value;
    }
}

bool IsPowerOfTwo(u64 value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

const char* SignalName(SignalClass signal) noexcept {
    switch (signal) {
    case SignalClass::Silent:
        return "silent";
    case SignalClass::Nonzero:
        return "nonzero";
    case SignalClass::NonFinite:
        return "nonfinite";
    }
    return "nonfinite";
}

const char* FormatName(SampleFormat format) noexcept {
    switch (format) {
    case SampleFormat::PcmS16:
        return "pcm_s16";
    case SampleFormat::Float32:
        return "float32";
    }
    return "unknown";
}

const char* PatchResolutionPhaseName(PatchResolutionPhase phase) noexcept {
    switch (phase) {
    case PatchResolutionPhase::Committed:
        return "committed";
    case PatchResolutionPhase::RenderTransition:
        return "render_transition";
    }
    return "committed";
}

const char* DestinationExactMatchName(DestinationExactMatch match) noexcept {
    switch (match) {
    case DestinationExactMatch::None:
        return "none";
    case DestinationExactMatch::System:
        return "system";
    case DestinationExactMatch::Rack:
        return "rack";
    case DestinationExactMatch::Voice:
        return "voice";
    }
    return "none";
}

const char* DestinationOffsetClassName(DestinationOffsetClass offset_class) noexcept {
    switch (offset_class) {
    case DestinationOffsetClass::OutsideLiveRack:
        return "outside_live_rack";
    case DestinationOffsetClass::RackBase:
        return "rack_base";
    case DestinationOffsetClass::RackHeader:
        return "rack_header";
    case DestinationOffsetClass::VoiceSlotAligned:
        return "voice_slot_aligned";
    case DestinationOffsetClass::VoiceSlotInterior:
        return "voice_slot_interior";
    case DestinationOffsetClass::RackTail:
        return "rack_tail";
    }
    return "outside_live_rack";
}

const char* DestinationTombstoneKindName(DestinationTombstoneKind kind) noexcept {
    switch (kind) {
    case DestinationTombstoneKind::None:
        return "none";
    case DestinationTombstoneKind::System:
        return "system";
    case DestinationTombstoneKind::Rack:
        return "rack";
    case DestinationTombstoneKind::Voice:
        return "voice";
    }
    return "none";
}

const char* RackLifecyclePhaseName(RackLifecyclePhase phase) noexcept {
    switch (phase) {
    case RackLifecyclePhase::Created:
        return "created";
    case RackLifecyclePhase::Destroyed:
        return "destroyed";
    }
    return "destroyed";
}

const char* SamplerPlaybackStateName(SamplerPlaybackState state) noexcept {
    switch (state) {
    case SamplerPlaybackState::Empty:
        return "empty";
    case SamplerPlaybackState::Playing:
        return "playing";
    case SamplerPlaybackState::Paused:
        return "paused";
    case SamplerPlaybackState::Stopped:
        return "stopped";
    }
    return "empty";
}

size_t SamplerPlaybackStateIndex(SamplerPlaybackState state) noexcept {
    const auto index = static_cast<size_t>(state);
    return index < SamplerPlaybackStateCount ? index : 0;
}

const char* SamplerStateObservationName(SamplerStateObservation observation) noexcept {
    switch (observation) {
    case SamplerStateObservation::RenderTransition:
        return "render_transition";
    case SamplerStateObservation::Poll:
        return "poll";
    }
    return "poll";
}

enum class ScalarShape : u8 {
    Zero,
    Unity,
    OtherFinite,
    NonFinite,
};

ScalarShape ClassifyScalar(float value) noexcept {
    if (!std::isfinite(value)) {
        return ScalarShape::NonFinite;
    }
    if (value == 0.0F) {
        return ScalarShape::Zero;
    }
    if (value == 1.0F) {
        return ScalarShape::Unity;
    }
    return ScalarShape::OtherFinite;
}

const char* ScalarShapeName(ScalarShape shape) noexcept {
    switch (shape) {
    case ScalarShape::Zero:
        return "zero";
    case ScalarShape::Unity:
        return "unity";
    case ScalarShape::OtherFinite:
        return "other_finite";
    case ScalarShape::NonFinite:
        return "nonfinite";
    }
    return "nonfinite";
}

enum class RenderState : u8 {
    NotStaged,
    RenderFailed,
    CommitFailed,
    Committed,
};

RenderState ClassifyRender(const SystemRenderMetadata& metadata) noexcept {
    if (!metadata.staged) {
        return RenderState::NotStaged;
    }
    if (!metadata.render_succeeded) {
        return RenderState::RenderFailed;
    }
    if (!metadata.commit_succeeded) {
        return RenderState::CommitFailed;
    }
    return RenderState::Committed;
}

const char* RenderStateName(RenderState state) noexcept {
    switch (state) {
    case RenderState::NotStaged:
        return "not_staged";
    case RenderState::RenderFailed:
        return "render_failed";
    case RenderState::CommitFailed:
        return "commit_failed";
    case RenderState::Committed:
        return "committed";
    }
    return "render_failed";
}

class Recorder {
public:
    explicit Recorder(const std::filesystem::path& path) noexcept {
        try {
            output.open(path, std::ios::out | std::ios::binary | std::ios::noreplace);
            enabled = output.is_open() && output.good();
            if (!enabled) {
                return;
            }
            WriteData(Json{{"event", "trace_header"},
                           {"schema_version", 6},
                           {"max_lines", MaxTraceLines},
                           {"max_bytes", MaxTraceBytes},
                           {"max_shapes_per_kind", MaxFirstSeenShapesPerKind},
                           {"max_transitions_per_kind", MaxTransitionsPerKind},
                           {"max_render_buffers_per_call", MaxRenderBuffersPerCall}});
        } catch (...) {
            Disable();
        }
    }

    ~Recorder() noexcept {
        Finalize();
    }

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled && !finalized;
    }

    void FailClosed() noexcept {
        finalized = true;
        Disable();
    }

    void RecordRackCreate(const RackCreateMetadata& metadata) {
        SaturatingIncrement(rack_creates);
        const RackShape shape{
            metadata.rack_id,      metadata.max_voices, metadata.max_input_delay_blocks,
            metadata.max_matrices, metadata.max_ports,  metadata.succeeded};
        if (RememberShape(rack_shapes, rack_shapes_saturated, shape)) {
            WriteData(RackDocument(metadata, "first_seen", rack_creates));
        }
        if (IsPowerOfTwo(rack_creates)) {
            WriteData(RackDocument(metadata, "checkpoint", rack_creates));
        }
    }

    void RecordVoiceControl(const VoiceControlMetadata& metadata) {
        SaturatingIncrement(voice_controls);
        const VoiceShape shape{
            metadata.control_id,     metadata.command_size,       metadata.next_offset,
            metadata.has_event,      metadata.event_id,           metadata.accepted,
            metadata.source_rack_id, metadata.source_voice_index, metadata.source_resolved,
        };
        if (RememberShape(voice_shapes, voice_shapes_saturated, shape)) {
            WriteData(VoiceDocument(metadata, "first_seen", voice_controls));
        }
        if (IsPowerOfTwo(voice_controls)) {
            WriteData(VoiceDocument(metadata, "checkpoint", voice_controls));
        }
    }

    void RecordSamplerEvent(const SamplerEventMetadata& metadata) {
        SaturatingIncrement(sampler_events);
        if (sampler_events <= MaxTransitionsPerKind) {
            WriteData(SamplerEventDocument(metadata, sampler_events));
        } else {
            sampler_events_saturated = true;
        }
    }

    void RecordSamplerSetup(const SamplerSetupMetadata& metadata) {
        SaturatingIncrement(sampler_setups);
        const SamplerShape shape{
            metadata.source_rack_id,
            metadata.source_voice_index,
            metadata.waveform_type,
            metadata.channels,
            metadata.sample_rate,
            metadata.codec_config,
            metadata.frame_offset,
            metadata.frame_margin,
            metadata.flags,
            metadata.atrac9_applicable,
            metadata.atrac9_config_valid,
            metadata.atrac9_declared_format_matches,
            metadata.atrac9_frame_bytes,
            metadata.atrac9_frames_per_superframe,
            metadata.atrac9_frame_samples,
            metadata.atrac9_superframe_bytes,
            metadata.atrac9_superframe_samples,
        };
        if (RememberShape(sampler_shapes, sampler_shapes_saturated, shape)) {
            WriteData(SamplerDocument(metadata, "first_seen", sampler_setups));
        }
        if (IsPowerOfTwo(sampler_setups)) {
            WriteData(SamplerDocument(metadata, "checkpoint", sampler_setups));
        }
    }

    void RecordWaveformBlocks(const WaveformBlocksMetadata& metadata) {
        SaturatingIncrement(waveform_block_lists);
        const WaveformBlockShape shape{
            metadata.source_rack_id,
            metadata.source_voice_index,
            metadata.flags,
            metadata.data_present,
            metadata.block_count,
            metadata.aggregate_data_bytes,
            metadata.aggregate_samples,
            metadata.aggregate_skip_samples,
            metadata.minimum_data_offset,
            metadata.maximum_data_end,
            metadata.data_range_arithmetic_valid,
            metadata.has_repeats,
            metadata.metadata_complete,
        };
        if (RememberShape(waveform_block_shapes, waveform_block_shapes_saturated, shape)) {
            WriteData(WaveformBlocksDocument(metadata, "first_seen", waveform_block_lists));
        }
        if (IsPowerOfTwo(waveform_block_lists)) {
            WriteData(WaveformBlocksDocument(metadata, "checkpoint", waveform_block_lists));
        }
    }

    void RecordUserFx(const UserFxMetadata& metadata) {
        SaturatingIncrement(user_fx_controls);
        const UserFxShape shape{metadata.control_id, metadata.callback_present};
        if (RememberShape(user_fx_shapes, user_fx_shapes_saturated, shape)) {
            WriteData(UserFxDocument(metadata, "first_seen", user_fx_controls));
        }
        if (IsPowerOfTwo(user_fx_controls)) {
            WriteData(UserFxDocument(metadata, "checkpoint", user_fx_controls));
        }
    }

    void RecordVoiceMatrixLevels(const VoiceMatrixLevelsMetadata& metadata) {
        SaturatingIncrement(voice_matrix_levels);
        const VoiceMatrixLevelShape shape{metadata.source_rack_id, metadata.source_voice_index,
                                          metadata.matrix_id, metadata.level_count,
                                          metadata.levels_present};
        if (RememberShape(voice_matrix_level_shapes, voice_matrix_level_shapes_saturated, shape)) {
            WriteData(VoiceMatrixLevelsDocument(metadata, "first_seen", voice_matrix_levels));
        }
        if (IsPowerOfTwo(voice_matrix_levels)) {
            WriteData(VoiceMatrixLevelsDocument(metadata, "checkpoint", voice_matrix_levels));
        }
    }

    void RecordVoicePortVolume(const VoicePortVolumeMetadata& metadata) {
        SaturatingIncrement(voice_port_volumes);
        const ScalarShape level_shape = ClassifyScalar(metadata.level);
        const VoicePortVolumeShape shape{metadata.source_rack_id, metadata.source_voice_index,
                                         metadata.port, level_shape};
        if (RememberShape(voice_port_volume_shapes, voice_port_volume_shapes_saturated, shape)) {
            WriteData(
                VoicePortVolumeDocument(metadata, level_shape, "first_seen", voice_port_volumes));
        }
        if (IsPowerOfTwo(voice_port_volumes)) {
            WriteData(
                VoicePortVolumeDocument(metadata, level_shape, "checkpoint", voice_port_volumes));
        }
    }

    void RecordVoicePortMatrix(const VoicePortMatrixMetadata& metadata) {
        SaturatingIncrement(voice_port_matrices);
        const VoicePortMatrixShape shape{metadata.source_rack_id, metadata.source_voice_index,
                                         metadata.port, metadata.matrix_id};
        if (RememberShape(voice_port_matrix_shapes, voice_port_matrix_shapes_saturated, shape)) {
            WriteData(VoicePortMatrixDocument(metadata, "first_seen", voice_port_matrices));
        }
        if (IsPowerOfTwo(voice_port_matrices)) {
            WriteData(VoicePortMatrixDocument(metadata, "checkpoint", voice_port_matrices));
        }
    }

    void RecordVoicePatch(const VoicePatchMetadata& metadata) {
        SaturatingIncrement(voice_patches);
        const VoicePatchShape shape{
            metadata.source_rack_id,       metadata.source_voice_index,
            metadata.source_port,          metadata.destination_input_id,
            metadata.destination_rack_id,  metadata.destination_voice_index,
            metadata.destination_resolved, metadata.destination_present,
        };
        if (RememberShape(voice_patch_shapes, voice_patch_shapes_saturated, shape)) {
            WriteData(VoicePatchDocument(metadata, "first_seen", voice_patches));
        }
        if (IsPowerOfTwo(voice_patches)) {
            WriteData(VoicePatchDocument(metadata, "checkpoint", voice_patches));
        }
    }

    void RecordVoicePatchState(const VoicePatchStateMetadata& metadata) {
        SaturatingIncrement(voice_patch_states);
        u64& phase_count = metadata.phase == PatchResolutionPhase::Committed
                               ? voice_patch_commits
                               : voice_patch_resolution_transitions;
        bool& phase_saturated = metadata.phase == PatchResolutionPhase::Committed
                                    ? voice_patch_commits_saturated
                                    : voice_patch_resolution_transitions_saturated;
        SaturatingIncrement(phase_count);
        if (phase_count <= MaxTransitionsPerKind) {
            WriteData(VoicePatchStateDocument(metadata, voice_patch_states));
        } else {
            phase_saturated = true;
            voice_patch_states_saturated = true;
        }
    }

    void RecordRackLifecycle(const RackLifecycleMetadata& metadata) {
        SaturatingIncrement(rack_lifecycle_events);
        if (rack_lifecycle_events <= MaxTransitionsPerKind) {
            WriteData(RackLifecycleDocument(metadata, rack_lifecycle_events));
        } else {
            rack_lifecycle_events_saturated = true;
        }
    }

    void RecordSamplerVoiceState(const SamplerVoiceStateMetadata& metadata) {
        SaturatingIncrement(sampler_voice_states);
        if (metadata.observation == SamplerStateObservation::RenderTransition) {
            SaturatingIncrement(sampler_voice_state_transitions);
            if (sampler_voice_state_transitions <= MaxTransitionsPerKind) {
                WriteData(SamplerVoiceStateDocument(metadata, sampler_voice_states));
            } else {
                sampler_voice_state_transitions_saturated = true;
                sampler_voice_states_saturated = true;
            }
            return;
        }

        SaturatingIncrement(sampler_voice_state_polls);
        const SamplerPollShape shape{metadata.source_rack_id, metadata.source_voice_index};
        auto& state_shapes =
            sampler_voice_state_poll_shapes[SamplerPlaybackStateIndex(metadata.current_state)];
        if (state_shapes.contains(shape)) {
            return;
        }
        if (state_shapes.size() < MaxSamplerPollShapesPerState) {
            state_shapes.insert(shape);
            WriteData(SamplerVoiceStateDocument(metadata, sampler_voice_states));
        } else {
            sampler_voice_state_polls_saturated = true;
            sampler_voice_states_saturated = true;
        }
    }

    void RecordMasteringOutput(const MasteringOutputMetadata& metadata) {
        SaturatingIncrement(mastering_outputs);
        const MasteringOutputShape shape{metadata.source_rack_id, metadata.source_voice_index,
                                         metadata.output_id};
        if (RememberShape(mastering_output_shapes, mastering_output_shapes_saturated, shape)) {
            WriteData(MasteringOutputDocument(metadata, "first_seen", mastering_outputs));
        }
        if (IsPowerOfTwo(mastering_outputs)) {
            WriteData(MasteringOutputDocument(metadata, "checkpoint", mastering_outputs));
        }
    }

    void RecordRenderBuffer(const RenderBufferMetadata& metadata) {
        SaturatingIncrement(render_buffers);
        const RenderBufferShape shape{metadata.output_index, metadata.waveform_type,
                                      metadata.channels, metadata.buffer_bytes};
        if (RememberShape(render_buffer_shapes, render_buffer_shapes_saturated, shape)) {
            WriteData(RenderBufferDocument(metadata, "first_seen", render_buffers));
        }
        if (IsPowerOfTwo(render_buffers)) {
            WriteData(RenderBufferDocument(metadata, "checkpoint", render_buffers));
        }
    }

    void RecordSystemRender(const SystemRenderMetadata& metadata) {
        SaturatingIncrement(system_renders);
        const RenderShape shape{
            metadata.buffer_count,
            metadata.traced_buffer_count,
            metadata.buffer_metadata_complete,
            metadata.output_bytes,
            metadata.staged,
            metadata.render_succeeded,
            metadata.commit_succeeded,
        };
        if (RememberShape(render_shapes, render_shapes_saturated, shape)) {
            WriteData(RenderDocument(metadata, "first_seen", system_renders));
        }
        if (IsPowerOfTwo(system_renders)) {
            WriteData(RenderDocument(metadata, "checkpoint", system_renders));
        }

        const RenderState current = ClassifyRender(metadata);
        if (last_render_state && *last_render_state != current) {
            RecordTransition("render_transition", RenderStateName(*last_render_state),
                             RenderStateName(current), render_transitions,
                             render_transitions_saturated);
        }
        last_render_state = current;
    }

    void RecordAudioOutput(const AudioOutputMetadata& metadata) {
        SaturatingIncrement(audio_outputs);
        const SignalClass signal = metadata.signal.Classification();
        const AudioShape shape{metadata.format, metadata.channels, metadata.sample_rate,
                               metadata.frames, signal};
        if (RememberShape(audio_shapes, audio_shapes_saturated, shape)) {
            WriteData(AudioDocument(metadata, "first_seen", audio_outputs));
        }
        if (IsPowerOfTwo(audio_outputs)) {
            WriteData(AudioDocument(metadata, "checkpoint", audio_outputs));
        }

        if (last_audio_state && *last_audio_state != signal) {
            RecordTransition("audio_transition", SignalName(*last_audio_state), SignalName(signal),
                             audio_transitions, audio_transitions_saturated);
        }
        last_audio_state = signal;
    }

    void Finalize() noexcept {
        if (!enabled || finalized) {
            return;
        }
        finalized = true;
        try {
            Json summary{
                {"event", "trace_summary"},
                {"rack_creates", rack_creates},
                {"voice_controls", voice_controls},
                {"sampler_events", sampler_events},
                {"sampler_setups", sampler_setups},
                {"waveform_block_lists", waveform_block_lists},
                {"user_fx_controls", user_fx_controls},
                {"voice_matrix_levels", voice_matrix_levels},
                {"voice_port_volumes", voice_port_volumes},
                {"voice_port_matrices", voice_port_matrices},
                {"voice_patches", voice_patches},
                {"voice_patch_states", voice_patch_states},
                {"rack_lifecycle_events", rack_lifecycle_events},
                {"sampler_voice_states", sampler_voice_states},
                {"mastering_outputs", mastering_outputs},
                {"render_buffers", render_buffers},
                {"system_renders", system_renders},
                {"audio_outputs", audio_outputs},
                {"written_lines", line_count + 1},
                {"written_bytes", byte_count},
                {"dropped_lines", dropped_lines},
                {"rack_shapes", rack_shapes.size()},
                {"voice_shapes", voice_shapes.size()},
                {"sampler_shapes", sampler_shapes.size()},
                {"waveform_block_shapes", waveform_block_shapes.size()},
                {"user_fx_shapes", user_fx_shapes.size()},
                {"voice_matrix_level_shapes", voice_matrix_level_shapes.size()},
                {"voice_port_volume_shapes", voice_port_volume_shapes.size()},
                {"voice_port_matrix_shapes", voice_port_matrix_shapes.size()},
                {"voice_patch_shapes", voice_patch_shapes.size()},
                {"mastering_output_shapes", mastering_output_shapes.size()},
                {"render_buffer_shapes", render_buffer_shapes.size()},
                {"render_shapes", render_shapes.size()},
                {"audio_shapes", audio_shapes.size()},
                {"rack_shapes_saturated", rack_shapes_saturated},
                {"voice_shapes_saturated", voice_shapes_saturated},
                {"sampler_shapes_saturated", sampler_shapes_saturated},
                {"waveform_block_shapes_saturated", waveform_block_shapes_saturated},
                {"user_fx_shapes_saturated", user_fx_shapes_saturated},
                {"voice_matrix_level_shapes_saturated", voice_matrix_level_shapes_saturated},
                {"voice_port_volume_shapes_saturated", voice_port_volume_shapes_saturated},
                {"voice_port_matrix_shapes_saturated", voice_port_matrix_shapes_saturated},
                {"voice_patch_shapes_saturated", voice_patch_shapes_saturated},
                {"mastering_output_shapes_saturated", mastering_output_shapes_saturated},
                {"render_buffer_shapes_saturated", render_buffer_shapes_saturated},
                {"render_shapes_saturated", render_shapes_saturated},
                {"audio_shapes_saturated", audio_shapes_saturated},
                {"render_transitions", render_transitions},
                {"audio_transitions", audio_transitions},
                {"render_transitions_saturated", render_transitions_saturated},
                {"audio_transitions_saturated", audio_transitions_saturated}};

            summary["sampler_events_saturated"] = sampler_events_saturated;
            summary["voice_patch_states_saturated"] = voice_patch_states_saturated;
            summary["voice_patch_commits_saturated"] = voice_patch_commits_saturated;
            summary["voice_patch_resolution_transitions_saturated"] =
                voice_patch_resolution_transitions_saturated;
            summary["rack_lifecycle_events_saturated"] = rack_lifecycle_events_saturated;
            summary["sampler_voice_states_saturated"] = sampler_voice_states_saturated;
            summary["sampler_voice_state_polls_saturated"] = sampler_voice_state_polls_saturated;
            summary["sampler_voice_state_transitions_saturated"] =
                sampler_voice_state_transitions_saturated;

            std::string line;
            for (int iteration = 0; iteration < 3; ++iteration) {
                line = summary.dump();
                summary["written_bytes"] = byte_count + line.size() + 1;
            }
            line = summary.dump();
            if (line_count < MaxTraceLines && line.size() + 1 <= MaxTraceBytes - byte_count) {
                WriteRaw(line);
            }
            output.flush();
            if (!output.good()) {
                Disable();
            }
        } catch (...) {
            Disable();
        }
    }

private:
    using RackShape = std::tuple<u32, u32, u32, u32, u32, bool>;
    using VoiceShape = std::tuple<u32, u16, s16, bool, u32, bool, u32, u32, bool>;
    using SamplerShape = std::tuple<u32, u32, u32, u32, u32, u32, u32, u32, u32, bool, bool, bool,
                                    u32, u32, u32, u32, u32>;
    using WaveformBlockShape =
        std::tuple<u32, u32, u32, bool, u32, u64, u64, u64, u32, u64, bool, bool, bool>;
    using UserFxShape = std::tuple<u32, bool>;
    using VoiceMatrixLevelShape = std::tuple<u32, u32, u32, u32, bool>;
    using VoicePortVolumeShape = std::tuple<u32, u32, u32, ScalarShape>;
    using VoicePortMatrixShape = std::tuple<u32, u32, u32, s32>;
    using VoicePatchShape = std::tuple<u32, u32, u32, u32, u32, u32, bool, bool>;
    using SamplerPollShape = std::tuple<u32, u32>;
    using MasteringOutputShape = std::tuple<u32, u32, u32>;
    using RenderBufferShape = std::tuple<u32, u32, u32, u64>;
    using RenderShape = std::tuple<u32, u32, bool, u64, bool, bool, bool>;
    using AudioShape = std::tuple<SampleFormat, u32, u32, u32, SignalClass>;

    template <typename Shape>
    bool RememberShape(std::set<Shape>& shapes, bool& saturated, const Shape& shape) {
        if (shapes.contains(shape)) {
            return false;
        }
        if (shapes.size() >= MaxFirstSeenShapesPerKind) {
            saturated = true;
            return false;
        }
        shapes.insert(shape);
        return true;
    }

    Json RackDocument(const RackCreateMetadata& metadata, const char* observation,
                      u64 occurrence) const {
        return Json{{"event", "rack_create"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"rack_id", metadata.rack_id},
                    {"max_voices", metadata.max_voices},
                    {"max_input_delay_blocks", metadata.max_input_delay_blocks},
                    {"max_matrices", metadata.max_matrices},
                    {"max_ports", metadata.max_ports},
                    {"succeeded", metadata.succeeded}};
    }

    Json VoiceDocument(const VoiceControlMetadata& metadata, const char* observation,
                       u64 occurrence) const {
        return Json{{"event", "voice_control"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"control_id", metadata.control_id},
                    {"command_size", metadata.command_size},
                    {"next_offset", metadata.next_offset},
                    {"has_event", metadata.has_event},
                    {"event_id", metadata.has_event ? metadata.event_id : 0},
                    {"accepted", metadata.accepted},
                    {"source_rack_id", metadata.source_rack_id},
                    {"source_voice_index", metadata.source_voice_index},
                    {"source_resolved", metadata.source_resolved}};
    }

    Json SamplerEventDocument(const SamplerEventMetadata& metadata, u64 occurrence) const {
        return Json{{"event", "sampler_event"},
                    {"occurrence", occurrence},
                    {"source_rack_id", metadata.source_rack_id},
                    {"source_voice_index", metadata.source_voice_index},
                    {"event_id", metadata.event_id},
                    {"accepted", metadata.accepted}};
    }

    Json SamplerDocument(const SamplerSetupMetadata& metadata, const char* observation,
                         u64 occurrence) const {
        return Json{{"event", "sampler_setup"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"source_rack_id", metadata.source_rack_id},
                    {"source_voice_index", metadata.source_voice_index},
                    {"waveform_type", metadata.waveform_type},
                    {"channels", metadata.channels},
                    {"sample_rate", metadata.sample_rate},
                    {"codec_config", metadata.codec_config},
                    {"frame_offset", metadata.frame_offset},
                    {"frame_margin", metadata.frame_margin},
                    {"flags", metadata.flags},
                    {"atrac9_applicable", metadata.atrac9_applicable},
                    {"atrac9_config_valid", metadata.atrac9_config_valid},
                    {"atrac9_declared_format_matches", metadata.atrac9_declared_format_matches},
                    {"atrac9_frame_bytes", metadata.atrac9_frame_bytes},
                    {"atrac9_frames_per_superframe", metadata.atrac9_frames_per_superframe},
                    {"atrac9_frame_samples", metadata.atrac9_frame_samples},
                    {"atrac9_superframe_bytes", metadata.atrac9_superframe_bytes},
                    {"atrac9_superframe_samples", metadata.atrac9_superframe_samples}};
    }

    Json WaveformBlocksDocument(const WaveformBlocksMetadata& metadata, const char* observation,
                                u64 occurrence) const {
        return Json{{"event", "waveform_blocks"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"source_rack_id", metadata.source_rack_id},
                    {"source_voice_index", metadata.source_voice_index},
                    {"flags", metadata.flags},
                    {"data_present", metadata.data_present},
                    {"block_count", metadata.block_count},
                    {"aggregate_data_bytes", metadata.aggregate_data_bytes},
                    {"aggregate_samples", metadata.aggregate_samples},
                    {"aggregate_skip_samples", metadata.aggregate_skip_samples},
                    {"minimum_data_offset", metadata.minimum_data_offset},
                    {"maximum_data_end", metadata.maximum_data_end},
                    {"data_range_arithmetic_valid", metadata.data_range_arithmetic_valid},
                    {"has_repeats", metadata.has_repeats},
                    {"metadata_complete", metadata.metadata_complete}};
    }

    Json UserFxDocument(const UserFxMetadata& metadata, const char* observation,
                        u64 occurrence) const {
        return Json{{"event", "user_fx"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"control_id", metadata.control_id},
                    {"callback_present", metadata.callback_present}};
    }

    Json VoiceMatrixLevelsDocument(const VoiceMatrixLevelsMetadata& metadata,
                                   const char* observation, u64 occurrence) const {
        return Json{{"event", "voice_matrix_levels"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"source_rack_id", metadata.source_rack_id},
                    {"source_voice_index", metadata.source_voice_index},
                    {"matrix_id", metadata.matrix_id},
                    {"level_count", metadata.level_count},
                    {"levels_present", metadata.levels_present}};
    }

    Json VoicePortVolumeDocument(const VoicePortVolumeMetadata& metadata, ScalarShape level_shape,
                                 const char* observation, u64 occurrence) const {
        return Json{{"event", "voice_port_volume"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"source_rack_id", metadata.source_rack_id},
                    {"source_voice_index", metadata.source_voice_index},
                    {"port", metadata.port},
                    {"level_shape", ScalarShapeName(level_shape)}};
    }

    Json VoicePortMatrixDocument(const VoicePortMatrixMetadata& metadata, const char* observation,
                                 u64 occurrence) const {
        return Json{{"event", "voice_port_matrix"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"source_rack_id", metadata.source_rack_id},
                    {"source_voice_index", metadata.source_voice_index},
                    {"port", metadata.port},
                    {"matrix_id", metadata.matrix_id}};
    }

    Json VoicePatchDocument(const VoicePatchMetadata& metadata, const char* observation,
                            u64 occurrence) const {
        return Json{{"event", "voice_patch"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"source_rack_id", metadata.source_rack_id},
                    {"source_voice_index", metadata.source_voice_index},
                    {"source_port", metadata.source_port},
                    {"destination_input_id", metadata.destination_input_id},
                    {"destination_rack_id", metadata.destination_rack_id},
                    {"destination_voice_index", metadata.destination_voice_index},
                    {"destination_resolved", metadata.destination_resolved},
                    {"destination_present", metadata.destination_present}};
    }

    Json VoicePatchStateDocument(const VoicePatchStateMetadata& metadata, u64 occurrence) const {
        return Json{
            {"event", "voice_patch_state"},
            {"occurrence", occurrence},
            {"phase", PatchResolutionPhaseName(metadata.phase)},
            {"source_rack_id", metadata.source_rack_id},
            {"source_voice_index", metadata.source_voice_index},
            {"source_port", metadata.source_port},
            {"destination_input_id", metadata.destination_input_id},
            {"destination_rack_id", metadata.destination_rack_id},
            {"destination_voice_index", metadata.destination_voice_index},
            {"destination_resolved", metadata.destination_resolved},
            {"destination_present", metadata.destination_present},
            {"destination_same_system", metadata.destination_same_system},
            {"destination_exact_match",
             DestinationExactMatchName(metadata.destination_exact_match)},
            {"destination_inside_live_rack", metadata.destination_inside_live_rack},
            {"destination_offset_class",
             DestinationOffsetClassName(metadata.destination_offset_class)},
            {"destination_candidate_voice_index_valid",
             metadata.destination_candidate_voice_index_valid},
            {"destination_candidate_voice_index", metadata.destination_candidate_voice_index_valid
                                                      ? metadata.destination_candidate_voice_index
                                                      : 0},
            {"destination_previously_returned", metadata.destination_previously_returned},
            {"destination_live_generation", metadata.destination_live_generation},
            {"destination_tombstone_kind",
             DestinationTombstoneKindName(metadata.destination_tombstone_kind)},
            {"destination_tombstone_generation", metadata.destination_tombstone_generation},
            {"destination_generation_changed_since_commit",
             metadata.destination_generation_changed_since_commit}};
    }

    Json RackLifecycleDocument(const RackLifecycleMetadata& metadata, u64 occurrence) const {
        return Json{
            {"event", "rack_lifecycle"},
            {"occurrence", occurrence},
            {"rack_id", metadata.rack_id},
            {"phase", RackLifecyclePhaseName(metadata.phase)},
            {"generation", metadata.generation},
            {"voice_count", metadata.voice_count},
            {"reused_identity", metadata.reused_identity},
            {"previous_generation", metadata.reused_identity ? metadata.previous_generation : 0}};
    }

    Json SamplerVoiceStateDocument(const SamplerVoiceStateMetadata& metadata,
                                   u64 occurrence) const {
        return Json{{"event", "sampler_voice_state"},
                    {"occurrence", occurrence},
                    {"observation", SamplerStateObservationName(metadata.observation)},
                    {"source_rack_id", metadata.source_rack_id},
                    {"source_voice_index", metadata.source_voice_index},
                    {"previous_state", SamplerPlaybackStateName(metadata.previous_state)},
                    {"current_state", SamplerPlaybackStateName(metadata.current_state)}};
    }

    Json MasteringOutputDocument(const MasteringOutputMetadata& metadata, const char* observation,
                                 u64 occurrence) const {
        return Json{{"event", "mastering_output"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"source_rack_id", metadata.source_rack_id},
                    {"source_voice_index", metadata.source_voice_index},
                    {"output_id", metadata.output_id}};
    }

    Json RenderBufferDocument(const RenderBufferMetadata& metadata, const char* observation,
                              u64 occurrence) const {
        return Json{{"event", "render_buffer"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"output_index", metadata.output_index},
                    {"waveform_type", metadata.waveform_type},
                    {"channels", metadata.channels},
                    {"buffer_bytes", metadata.buffer_bytes}};
    }

    Json RenderDocument(const SystemRenderMetadata& metadata, const char* observation,
                        u64 occurrence) const {
        return Json{{"event", "system_render"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"buffer_count", metadata.buffer_count},
                    {"traced_buffer_count", metadata.traced_buffer_count},
                    {"buffer_metadata_complete", metadata.buffer_metadata_complete},
                    {"output_bytes", metadata.output_bytes},
                    {"staged", metadata.staged},
                    {"render_succeeded", metadata.render_succeeded},
                    {"commit_succeeded", metadata.commit_succeeded}};
    }

    Json AudioDocument(const AudioOutputMetadata& metadata, const char* observation,
                       u64 occurrence) const {
        return Json{{"event", "audio_output"},
                    {"observation", observation},
                    {"occurrence", occurrence},
                    {"format", FormatName(metadata.format)},
                    {"channels", metadata.channels},
                    {"sample_rate", metadata.sample_rate},
                    {"frames", metadata.frames},
                    {"samples", metadata.signal.samples},
                    {"zero_samples", metadata.signal.zero_samples},
                    {"nonzero_samples", metadata.signal.nonzero_samples},
                    {"nonfinite_samples", metadata.signal.nonfinite_samples},
                    {"signal", SignalName(metadata.signal.Classification())}};
    }

    void RecordTransition(const char* event, const char* from, const char* to, u64& count,
                          bool& saturated) {
        if (count >= MaxTransitionsPerKind) {
            saturated = true;
            return;
        }
        SaturatingIncrement(count);
        WriteData(Json{{"event", event}, {"occurrence", count}, {"from", from}, {"to", to}});
    }

    void WriteData(const Json& document) {
        if (!enabled || finalized) {
            return;
        }
        const std::string line = document.dump();
        if (line_count >= MaxTraceLines - 1 ||
            line.size() + 1 + SummaryReserveBytes > MaxTraceBytes - byte_count) {
            SaturatingIncrement(dropped_lines);
            return;
        }
        WriteRaw(line);
    }

    void WriteRaw(const std::string& line) {
        output.write(line.data(), static_cast<std::streamsize>(line.size()));
        output.put('\n');
        if (!output.good()) {
            Disable();
            return;
        }
        SaturatingIncrement(line_count);
        byte_count += line.size() + 1;
    }

    void Disable() noexcept {
        enabled = false;
        if (output.is_open()) {
            output.close();
        }
    }

    std::ofstream output;
    bool enabled{};
    bool finalized{};
    u64 line_count{};
    u64 byte_count{};
    u64 dropped_lines{};
    u64 rack_creates{};
    u64 voice_controls{};
    u64 sampler_events{};
    u64 sampler_setups{};
    u64 waveform_block_lists{};
    u64 user_fx_controls{};
    u64 voice_matrix_levels{};
    u64 voice_port_volumes{};
    u64 voice_port_matrices{};
    u64 voice_patches{};
    u64 voice_patch_states{};
    u64 voice_patch_commits{};
    u64 voice_patch_resolution_transitions{};
    u64 rack_lifecycle_events{};
    u64 sampler_voice_states{};
    u64 sampler_voice_state_polls{};
    u64 sampler_voice_state_transitions{};
    u64 mastering_outputs{};
    u64 render_buffers{};
    u64 system_renders{};
    u64 audio_outputs{};
    u64 render_transitions{};
    u64 audio_transitions{};
    bool rack_shapes_saturated{};
    bool voice_shapes_saturated{};
    bool sampler_shapes_saturated{};
    bool waveform_block_shapes_saturated{};
    bool user_fx_shapes_saturated{};
    bool voice_matrix_level_shapes_saturated{};
    bool voice_port_volume_shapes_saturated{};
    bool voice_port_matrix_shapes_saturated{};
    bool voice_patch_shapes_saturated{};
    bool mastering_output_shapes_saturated{};
    bool render_buffer_shapes_saturated{};
    bool render_shapes_saturated{};
    bool audio_shapes_saturated{};
    bool render_transitions_saturated{};
    bool audio_transitions_saturated{};
    bool sampler_events_saturated{};
    bool voice_patch_states_saturated{};
    bool voice_patch_commits_saturated{};
    bool voice_patch_resolution_transitions_saturated{};
    bool rack_lifecycle_events_saturated{};
    bool sampler_voice_states_saturated{};
    bool sampler_voice_state_polls_saturated{};
    bool sampler_voice_state_transitions_saturated{};
    std::set<RackShape> rack_shapes;
    std::set<VoiceShape> voice_shapes;
    std::set<SamplerShape> sampler_shapes;
    std::set<WaveformBlockShape> waveform_block_shapes;
    std::set<UserFxShape> user_fx_shapes;
    std::set<VoiceMatrixLevelShape> voice_matrix_level_shapes;
    std::set<VoicePortVolumeShape> voice_port_volume_shapes;
    std::set<VoicePortMatrixShape> voice_port_matrix_shapes;
    std::set<VoicePatchShape> voice_patch_shapes;
    std::array<std::set<SamplerPollShape>, SamplerPlaybackStateCount>
        sampler_voice_state_poll_shapes;
    std::set<MasteringOutputShape> mastering_output_shapes;
    std::set<RenderBufferShape> render_buffer_shapes;
    std::set<RenderShape> render_shapes;
    std::set<AudioShape> audio_shapes;
    std::optional<RenderState> last_render_state;
    std::optional<SignalClass> last_audio_state;
};

enum class Availability : u8 {
    Uninitialized,
    Disabled,
    Enabled,
};

struct GlobalState {
    ~GlobalState() noexcept {
        try {
            recorder.reset();
        } catch (...) {
        }
    }

    [[nodiscard]] std::unique_lock<std::mutex> Lock() {
        lock_acquisitions.fetch_add(1, std::memory_order_relaxed);
        return std::unique_lock{mutex};
    }

    std::mutex mutex;
    std::unique_ptr<Recorder> recorder;
    std::atomic<Availability> availability{Availability::Uninitialized};
    std::atomic<u64> lock_acquisitions{};
    std::atomic<bool> process_exit_finalizer_registered{};
};

void FinalizeAtProcessExit() noexcept;

GlobalState& State() {
    static GlobalState state;
    static const bool registered = std::at_quick_exit(FinalizeAtProcessExit) == 0;
    state.process_exit_finalizer_registered.store(registered, std::memory_order_release);
    return state;
}

void FinalizeState(GlobalState& state) noexcept {
    try {
        auto lock = state.Lock();
        if (state.recorder) {
            state.recorder->Finalize();
        }
        state.availability.store(Availability::Disabled, std::memory_order_release);
    } catch (...) {
    }
}

void FinalizeAtProcessExit() noexcept {
    FinalizeState(State());
}

Recorder* GetRecorder(GlobalState& state) {
    if (state.availability.load(std::memory_order_relaxed) == Availability::Uninitialized) {
        const char* path = std::getenv("SHADPS4_AUDIO_PIPELINE_TRACE_PATH");
        if (path && path[0] != '\0') {
            state.recorder = std::make_unique<Recorder>(std::filesystem::path{path});
        }
        const bool enabled = state.recorder && state.recorder->Enabled();
        state.availability.store(enabled ? Availability::Enabled : Availability::Disabled,
                                 std::memory_order_release);
    }
    return state.recorder.get();
}

template <typename Callback>
void WithRecorder(Callback&& callback) noexcept {
    try {
        auto& state = State();
        if (state.availability.load(std::memory_order_acquire) == Availability::Disabled) {
            return;
        }

        auto lock = state.Lock();
        if (auto* recorder = GetRecorder(state); recorder && recorder->Enabled()) {
            try {
                std::forward<Callback>(callback)(*recorder);
            } catch (...) {
                recorder->FailClosed();
            }
            state.availability.store(recorder->Enabled() ? Availability::Enabled
                                                         : Availability::Disabled,
                                     std::memory_order_release);
        }
    } catch (...) {
    }
}

} // namespace

SignalClass SignalAnalysis::Classification() const noexcept {
    if (nonfinite_samples != 0) {
        return SignalClass::NonFinite;
    }
    if (nonzero_samples != 0) {
        return SignalClass::Nonzero;
    }
    return SignalClass::Silent;
}

void SignalAnalyzer::ObservePcmS16(s16 sample) noexcept {
    SaturatingIncrement(result.samples);
    if (sample == 0) {
        SaturatingIncrement(result.zero_samples);
    } else {
        SaturatingIncrement(result.nonzero_samples);
    }
}

void SignalAnalyzer::ObserveFloat32(float sample) noexcept {
    SaturatingIncrement(result.samples);
    if (!std::isfinite(sample)) {
        SaturatingIncrement(result.nonfinite_samples);
    } else if (sample == 0.0F) {
        SaturatingIncrement(result.zero_samples);
    } else {
        SaturatingIncrement(result.nonzero_samples);
    }
}

SignalAnalysis SignalAnalyzer::Result() const noexcept {
    return result;
}

void RecordRackCreate(RackCreateMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordRackCreate(metadata); });
}

void RecordVoiceControl(VoiceControlMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordVoiceControl(metadata); });
}

void RecordSamplerEvent(SamplerEventMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordSamplerEvent(metadata); });
}

void RecordSamplerSetup(SamplerSetupMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordSamplerSetup(metadata); });
}

void RecordWaveformBlocks(WaveformBlocksMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordWaveformBlocks(metadata); });
}

void RecordUserFx(UserFxMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordUserFx(metadata); });
}

void RecordVoiceMatrixLevels(VoiceMatrixLevelsMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordVoiceMatrixLevels(metadata); });
}

void RecordVoicePortVolume(VoicePortVolumeMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordVoicePortVolume(metadata); });
}

void RecordVoicePortMatrix(VoicePortMatrixMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordVoicePortMatrix(metadata); });
}

void RecordVoicePatch(VoicePatchMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordVoicePatch(metadata); });
}

void RecordVoicePatchState(VoicePatchStateMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordVoicePatchState(metadata); });
}

void RecordRackLifecycle(RackLifecycleMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordRackLifecycle(metadata); });
}

void RecordSamplerVoiceState(SamplerVoiceStateMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordSamplerVoiceState(metadata); });
}

void RecordMasteringOutput(MasteringOutputMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordMasteringOutput(metadata); });
}

void RecordRenderBuffer(RenderBufferMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordRenderBuffer(metadata); });
}

void RecordSystemRender(SystemRenderMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordSystemRender(metadata); });
}

void RecordAudioOutput(AudioOutputMetadata metadata) noexcept {
    WithRecorder([&](Recorder& recorder) { recorder.RecordAudioOutput(metadata); });
}

bool IsEnabled() noexcept {
    try {
        auto& state = State();
        const Availability availability = state.availability.load(std::memory_order_acquire);
        if (availability != Availability::Uninitialized) {
            return availability == Availability::Enabled;
        }

        auto lock = state.Lock();
        const auto* recorder = GetRecorder(state);
        return recorder && recorder->Enabled();
    } catch (...) {
        return false;
    }
}

namespace Testing {

void Reset() noexcept {
    try {
        auto& state = State();
        auto lock = state.Lock();
        state.recorder.reset();
        state.availability.store(Availability::Uninitialized, std::memory_order_release);
    } catch (...) {
    }
}

void Finalize() noexcept {
    FinalizeState(State());
}

void FinalizeForProcessExit() noexcept {
    FinalizeAtProcessExit();
}

bool ProcessExitFinalizerRegistered() noexcept {
    return State().process_exit_finalizer_registered.load(std::memory_order_acquire);
}

u64 StateLockAcquisitions() noexcept {
    return State().lock_acquisitions.load(std::memory_order_relaxed);
}

} // namespace Testing

} // namespace Libraries::AudioPipelineTrace
