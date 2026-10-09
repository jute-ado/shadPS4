// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/ngs2/ngs2_render.h"

#include <algorithm>
#include <array>
#include <compare>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <tuple>
#include <utility>

#include "core/libraries/audio/audio_pipeline_metadata_trace.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/ngs2/ngs2_atrac9.h"
#include "core/libraries/ngs2/ngs2_error.h"

namespace Libraries::Ngs2 {
namespace {

constexpr u32 Float32WaveformType = 24;
constexpr u32 MaxRenderBuffers = 16;
constexpr size_t MaxRenderOutputBytes = 64 * 1024 * 1024;

struct AudioBlock {
    u32 channels{};
    u32 frames{};
    std::vector<float> samples;
};

struct BusKey {
    u32 rack_id{};
    u64 endpoint_generation{};
    u32 input_id{};
    bool voice_specific{};
    u32 voice_index{};

    auto operator<=>(const BusKey&) const = default;
};

struct LockedSampler {
    const SystemRenderVoiceSnapshot* voice{};
    std::unique_lock<std::mutex> lock;
};

struct DecodedSampler {
    const SystemRenderVoiceSnapshot* voice{};
    SamplerDecodedBlock decoded;
};

bool IsSupportedChannelCount(u32 channels) noexcept {
    return channels == 1 || channels == 2 || channels == 6 || channels == 8;
}

bool HasUnsupportedEnabledUserFx(const SystemRenderGraphSnapshot& graph) noexcept {
    return std::ranges::any_of(graph.voices, [](const auto& voice) {
        if (voice.playback_state != VoiceRenderPlaybackState::Playing) {
            return false;
        }
        return (voice.control.sampler && voice.control.sampler->user_fx_enabled) ||
               (voice.control.submixer && voice.control.submixer->user_fx_enabled);
    });
}

s32 PreflightOutputs(const SystemRenderGraphSnapshot& graph,
                     const OrbisNgs2RenderBufferInfo* buffer_info, u32 count,
                     std::vector<AudioBlock>& outputs) {
    if (count == 0) {
        outputs.clear();
        return ORBIS_OK;
    }
    if (!buffer_info || count > MaxRenderBuffers) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_INFO;
    }
    size_t total_output_bytes = 0;
    try {
        outputs.clear();
        outputs.reserve(count);
        for (u32 index = 0; index < count; ++index) {
            const auto& descriptor = buffer_info[index];
            if (descriptor.waveformType != Float32WaveformType) {
                return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_TYPE;
            }
            if (!IsSupportedChannelCount(descriptor.numChannels)) {
                return ORBIS_NGS2_ERROR_INVALID_NUM_CHANNELS;
            }
            if (graph.system.num_grain_samples >
                std::numeric_limits<size_t>::max() / descriptor.numChannels / sizeof(float)) {
                return ORBIS_NGS2_ERROR_INVALID_BUFFER_SIZE;
            }
            const size_t required_bytes = static_cast<size_t>(graph.system.num_grain_samples) *
                                          descriptor.numChannels * sizeof(float);
            if (descriptor.bufferSize < required_bytes) {
                return ORBIS_NGS2_ERROR_INVALID_BUFFER_SIZE;
            }
            if (!descriptor.buffer) {
                return ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS;
            }
            if (descriptor.bufferSize > MaxRenderOutputBytes - total_output_bytes) {
                return ORBIS_NGS2_ERROR_INVALID_BUFFER_SIZE;
            }
            total_output_bytes += descriptor.bufferSize;
            outputs.push_back({
                .channels = descriptor.numChannels,
                .frames = graph.system.num_grain_samples,
                .samples = std::vector<float>(static_cast<size_t>(graph.system.num_grain_samples) *
                                              descriptor.numChannels),
            });
        }
    } catch (const std::bad_alloc&) {
        outputs.clear();
        return ORBIS_NGS2_ERROR_FAIL;
    }
    return ORBIS_OK;
}

std::optional<u32> InputChannels(const SystemRenderVoiceSnapshot& voice) noexcept {
    if (voice.playback_state != VoiceRenderPlaybackState::Playing) {
        return std::nullopt;
    }
    if (voice.token.rack_id == 0x2000 && voice.control.submixer &&
        voice.control.submixer->setup_configured && voice.control.submixer->flags == 0) {
        return voice.control.submixer->num_io_channels;
    }
    if (voice.token.rack_id == 0x3000 && voice.control.mastering &&
        voice.control.mastering->setup_configured && voice.control.mastering->setup_flags == 0) {
        return voice.control.mastering->num_input_channels;
    }
    return std::nullopt;
}

bool RouteTargetsVoice(const VoiceRouteSnapshot& route,
                       const SystemRenderVoiceSnapshot& destination) noexcept {
    if (!route.connected || route.destination_rack_id != destination.token.rack_id ||
        route.destination_input_id >= destination.control.max_inputs) {
        return false;
    }
    if (route.endpoint_kind == VoiceRouteEndpointKind::Rack) {
        return !route.destination_voice_index_valid &&
               route.destination_generation == destination.token.rack_generation;
    }
    return route.endpoint_kind == VoiceRouteEndpointKind::Voice &&
           route.destination_voice_index_valid &&
           route.destination_voice_index == destination.token.voice_index &&
           route.destination_generation == destination.token.voice_generation;
}

std::optional<u32> DestinationChannels(const SystemRenderGraphSnapshot& graph,
                                       const VoiceRouteSnapshot& route) noexcept {
    std::optional<u32> channels;
    u32 destination_count = 0;
    for (const auto& destination : graph.voices) {
        if (!RouteTargetsVoice(route, destination)) {
            continue;
        }
        const auto candidate = InputChannels(destination);
        if (!candidate) {
            continue;
        }
        if (channels && *channels != *candidate) {
            return std::nullopt;
        }
        channels = candidate;
        ++destination_count;
        if (route.endpoint_kind == VoiceRouteEndpointKind::Voice) {
            break;
        }
    }
    // A rack endpoint is a distinct input-bus target. Its mapping onto multiple active voices is
    // not equivalent to fanout and remains unknown, so only the unambiguous one-voice case is
    // rendered for now.
    return route.endpoint_kind == VoiceRouteEndpointKind::Rack && destination_count != 1
               ? std::nullopt
               : channels;
}

std::optional<std::vector<const SystemRenderVoiceSnapshot*>> PlanSubmixerOrder(
    const SystemRenderGraphSnapshot& graph) {
    std::vector<const SystemRenderVoiceSnapshot*> submixers;
    for (const auto& voice : graph.voices) {
        if (voice.token.rack_id == 0x2000 && InputChannels(voice)) {
            submixers.push_back(&voice);
        }
    }

    std::vector<std::vector<size_t>> successors(submixers.size());
    std::vector<size_t> predecessors(submixers.size());
    for (size_t source = 0; source < submixers.size(); ++source) {
        for (const auto& route : submixers[source]->control.routes) {
            // Use the actual routing admission: an ambiguous rack bus is not guessed as fanout,
            // and inactive, unsupported, disconnected or stale destinations add no dependency.
            if (!DestinationChannels(graph, route)) {
                continue;
            }
            for (size_t destination = 0; destination < submixers.size(); ++destination) {
                if (RouteTargetsVoice(route, *submixers[destination])) {
                    successors[source].push_back(destination);
                    ++predecessors[destination];
                }
            }
        }
    }

    std::vector<size_t> ready;
    ready.reserve(submixers.size());
    for (size_t index = 0; index < submixers.size(); ++index) {
        if (predecessors[index] == 0) {
            ready.push_back(index);
        }
    }
    std::vector<const SystemRenderVoiceSnapshot*> order;
    order.reserve(submixers.size());
    for (size_t index = 0; index < ready.size(); ++index) {
        const size_t source = ready[index];
        order.push_back(submixers[source]);
        for (const size_t destination : successors[source]) {
            if (--predecessors[destination] == 0) {
                ready.push_back(destination);
            }
        }
    }
    // Feedback requires separate delay/iteration semantics. Do not consume a decoder grain or
    // publish partial output merely because an unsupported cycle has no ready node.
    if (order.size() != submixers.size()) {
        return std::nullopt;
    }
    return order;
}

BusKey KeyForRoute(const VoiceRouteSnapshot& route) noexcept {
    return {
        .rack_id = route.destination_rack_id,
        .endpoint_generation = route.destination_generation,
        .input_id = route.destination_input_id,
        .voice_specific = route.endpoint_kind == VoiceRouteEndpointKind::Voice,
        .voice_index = route.destination_voice_index,
    };
}

const VoicePortSnapshot* FindPort(const VoiceControlSnapshot& voice, u32 port) noexcept {
    const auto found = std::ranges::lower_bound(voice.ports, port, {}, &VoicePortSnapshot::port);
    return found != voice.ports.end() && found->port == port ? &*found : nullptr;
}

const VoiceMatrixSnapshot* FindMatrix(const VoiceControlSnapshot& voice, u32 matrix) noexcept {
    const auto found =
        std::ranges::lower_bound(voice.matrices, matrix, {}, &VoiceMatrixSnapshot::matrix_id);
    return found != voice.matrices.end() && found->matrix_id == matrix ? &*found : nullptr;
}

void AddIdentity(const AudioBlock& source, float volume, AudioBlock& destination) {
    const size_t count = std::min(source.samples.size(), destination.samples.size());
    for (size_t index = 0; index < count; ++index) {
        destination.samples[index] += source.samples[index] * volume;
    }
}

void AddGenericMatrix(const AudioBlock& source, u32 destination_channels,
                      std::span<const float> levels, float volume, AudioBlock& destination) {
    const u32 frames = std::min(source.frames, destination.frames);
    for (u32 frame = 0; frame < frames; ++frame) {
        const size_t source_base = static_cast<size_t>(frame) * source.channels;
        const size_t destination_base = static_cast<size_t>(frame) * destination_channels;
        for (u32 destination_channel = 0; destination_channel < destination_channels;
             ++destination_channel) {
            float mixed = 0.0F;
            const size_t level_base = static_cast<size_t>(destination_channel) * source.channels;
            for (u32 source_channel = 0; source_channel < source.channels; ++source_channel) {
                mixed += source.samples[source_base + source_channel] *
                         levels[level_base + source_channel];
            }
            destination.samples[destination_base + destination_channel] += mixed * volume;
        }
    }
}

void RouteVoiceBlock(const SystemRenderGraphSnapshot& graph,
                     const SystemRenderVoiceSnapshot& source_voice, const AudioBlock& source,
                     std::map<BusKey, AudioBlock>& buses) {
    for (const VoiceRouteSnapshot& route : source_voice.control.routes) {
        const auto destination_channels = DestinationChannels(graph, route);
        if (!destination_channels || source.samples.empty()) {
            continue;
        }
        const VoicePortSnapshot* port = FindPort(source_voice.control, route.source_port);
        const float volume = port && port->volume_configured ? port->volume : 1.0F;
        auto [bus, inserted] = buses.try_emplace(
            KeyForRoute(route),
            AudioBlock{.channels = *destination_channels,
                       .frames = source.frames,
                       .samples = std::vector<float>(static_cast<size_t>(source.frames) *
                                                     *destination_channels)});
        if (bus->second.channels != *destination_channels || bus->second.frames != source.frames) {
            continue;
        }
        if (!port || !port->matrix_configured || port->matrix_id == -1) {
            if (source.channels == *destination_channels) {
                AddIdentity(source, volume, bus->second);
            }
            continue;
        }
        const auto* matrix = FindMatrix(source_voice.control, static_cast<u32>(port->matrix_id));
        const size_t expected_levels = static_cast<size_t>(source.channels) * *destination_channels;
        if (!matrix || matrix->levels.size() != expected_levels) {
            continue;
        }
        AddGenericMatrix(source, *destination_channels, matrix->levels, volume, bus->second);
    }
}

AudioBlock GatherVoiceInputs(const SystemRenderVoiceSnapshot& voice,
                             const std::map<BusKey, AudioBlock>& buses, u32 channels, u32 frames) {
    AudioBlock input{.channels = channels,
                     .frames = frames,
                     .samples = std::vector<float>(static_cast<size_t>(channels) * frames)};
    for (u32 input_id = 0; input_id < voice.control.max_inputs; ++input_id) {
        for (const BusKey key : std::array{
                 BusKey{voice.token.rack_id, voice.token.rack_generation, input_id, false, 0},
                 BusKey{voice.token.rack_id, voice.token.voice_generation, input_id, true,
                        voice.token.voice_index},
             }) {
            const auto bus = buses.find(key);
            if (bus != buses.end() && bus->second.channels == channels &&
                bus->second.frames == frames) {
                AddIdentity(bus->second, 1.0F, input);
            }
        }
    }
    return input;
}

std::optional<AudioBlock> ApplyMastering(const SystemRenderVoiceSnapshot& voice,
                                         const AudioBlock& input, u32 output_channels) {
    const auto& mastering = *voice.control.mastering;
    AudioBlock output{.channels = output_channels,
                      .frames = input.frames,
                      .samples =
                          std::vector<float>(static_cast<size_t>(output_channels) * input.frames)};
    if (mastering.matrix_configured) {
        const size_t expected = static_cast<size_t>(input.channels) * output_channels;
        if (mastering.matrix_input_channels != input.channels ||
            mastering.matrix_output_channels != output_channels ||
            mastering.matrix_levels.size() != expected) {
            return std::nullopt;
        }
        for (u32 frame = 0; frame < input.frames; ++frame) {
            const size_t input_base = static_cast<size_t>(frame) * input.channels;
            const size_t output_base = static_cast<size_t>(frame) * output_channels;
            for (u32 input_channel = 0; input_channel < input.channels; ++input_channel) {
                const size_t level_base = static_cast<size_t>(input_channel) * output_channels;
                for (u32 output_channel = 0; output_channel < output_channels; ++output_channel) {
                    output.samples[output_base + output_channel] +=
                        input.samples[input_base + input_channel] *
                        mastering.matrix_levels[level_base + output_channel];
                }
            }
        }
    } else {
        if (input.channels != output_channels) {
            return std::nullopt;
        }
        output.samples = input.samples;
    }

    if (mastering.gain_configured) {
        if (output.channels >= 6 && mastering.fbw_level != mastering.lfe_level) {
            return std::nullopt;
        }
        for (float& sample : output.samples) {
            sample *= mastering.fbw_level;
        }
    }
    return output;
}

s32 DecodeError(SamplerDecodeStatus status) noexcept {
    switch (status) {
    case SamplerDecodeStatus::InvalidFormat:
        return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_FORMAT;
    case SamplerDecodeStatus::InvalidWaveform:
        return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_DATA;
    case SamplerDecodeStatus::UnsupportedLoop:
        return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_BLOCK_NUM_REPEATS;
    case SamplerDecodeStatus::GuestReadFailed:
        return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_ADDRESS;
    case SamplerDecodeStatus::DecoderUnavailable:
        return ORBIS_NGS2_ERROR_EMPTY_CODEC_DECODER;
    case SamplerDecodeStatus::DecodeFailed:
        return ORBIS_NGS2_ERROR_CODEC_DECODE_FAIL;
    default:
        return ORBIS_NGS2_ERROR_FAIL;
    }
}

void AcceptEngineProgress(u64 revision, void* context) noexcept {
    static_cast<SamplerDecodeEngine*>(context)->AcceptStaged(revision);
}

void RejectAll(std::span<LockedSampler> samplers) noexcept {
    for (auto& sampler : samplers) {
        if (sampler.voice->decode_runtime && sampler.voice->decode_runtime->engine) {
            sampler.voice->decode_runtime->engine->RejectStaged();
        }
    }
}

bool TryPublish(SystemRenderPublishHandler publish_handler, void* publish_context) noexcept {
    try {
        return !publish_handler || publish_handler(publish_context);
    } catch (...) {
        return false;
    }
}

void WritePreparedOutputs(const OrbisNgs2RenderBufferInfo* buffer_info,
                          std::span<const AudioBlock> outputs) {
    for (size_t index = 0; index < outputs.size(); ++index) {
        std::memset(buffer_info[index].buffer, 0, buffer_info[index].bufferSize);
        std::memcpy(buffer_info[index].buffer, outputs[index].samples.data(),
                    outputs[index].samples.size() * sizeof(float));
    }
}

s32 CommitRenderedAudio(const SystemRenderGraphSnapshot& graph,
                        std::span<const VoiceRenderProgressCommit> progress,
                        const OrbisNgs2RenderBufferInfo* buffer_info,
                        std::span<const AudioBlock> outputs,
                        SystemRenderPublishHandler publish_handler, void* publish_context) {
    if (publish_handler) {
        if (!ValidateSystemRenderProgress(graph, progress)) {
            return ORBIS_NGS2_ERROR_FAIL;
        }
        WritePreparedOutputs(buffer_info, outputs);
        // Public-export descriptors refer to owned staging. Guest writes may fault, so invoke
        // the batched writer outside the registry lock while decoder state remains staged.
        if (!TryPublish(publish_handler, publish_context)) {
            return ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS;
        }
    }
    // Revalidate after publication. A concurrent graph change can reject progress even when
    // guest bytes are already visible; this is not a global graph/memory transaction.
    if (!CommitSystemRenderProgress(graph, progress)) {
        return ORBIS_NGS2_ERROR_FAIL;
    }
    if (!publish_handler) {
        WritePreparedOutputs(buffer_info, outputs);
    }
    return ORBIS_OK;
}

} // namespace

s32 RenderSystemAudio(OrbisNgs2Handle system_handle, const OrbisNgs2RenderBufferInfo* buffer_info,
                      u32 num_buffer_info, OrbisNgs2RuntimeReadHandler read_handler,
                      void* read_context, Atrac9FrameDecoderFactory decoder_factory,
                      SystemRenderPublishHandler publish_handler, void* publish_context) {
    std::optional<SystemRenderGraphSnapshot> graph;
    try {
        graph = SnapshotSystemRenderGraph(system_handle);
    } catch (const std::bad_alloc&) {
        return ORBIS_NGS2_ERROR_FAIL;
    }
    if (!graph) {
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    std::vector<AudioBlock> outputs;
    if (const s32 preflight = PreflightOutputs(*graph, buffer_info, num_buffer_info, outputs);
        preflight < 0) {
        return preflight;
    }
    if (num_buffer_info == 0) {
        return TryPublish(publish_handler, publish_context)
                   ? ORBIS_OK
                   : ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS;
    }

    std::optional<std::vector<const SystemRenderVoiceSnapshot*>> submixer_order;
    try {
        submixer_order = PlanSubmixerOrder(*graph);
    } catch (const std::bad_alloc&) {
        return ORBIS_NGS2_ERROR_FAIL;
    }
    if (!submixer_order) {
        return ORBIS_NGS2_ERROR_INVALID_PATCH;
    }
    if (!ValidateSystemRenderProgress(*graph, {})) {
        return ORBIS_NGS2_ERROR_FAIL;
    }

    // Guest callbacks are retained as configuration, but cannot be invoked safely here yet.
    // Preserve the pre-existing silent-render behavior without decoding or advancing cursors.
    if (HasUnsupportedEnabledUserFx(*graph)) {
        AudioPipelineTrace::RecordRenderDecision(
            AudioPipelineTrace::RenderDecision::UnsupportedUserFx);
        return CommitRenderedAudio(*graph, {}, buffer_info, outputs, publish_handler,
                                   publish_context);
    }

    std::vector<LockedSampler> locked_samplers;
    std::vector<DecodedSampler> decoded_samplers;
    std::vector<VoiceRenderProgressCommit> progress;
    try {
        for (const auto& voice : graph->voices) {
            if (voice.token.rack_id != 0x1000 ||
                voice.playback_state != VoiceRenderPlaybackState::Playing ||
                !voice.control.sampler || !voice.control.sampler->format ||
                !voice.control.sampler->waveform_configured ||
                voice.control.sampler->format->waveform_type != Atrac9WaveformType ||
                !voice.decode_runtime) {
                continue;
            }
            locked_samplers.push_back({
                .voice = &voice,
                .lock = std::unique_lock<std::mutex>{voice.decode_runtime->mutex},
            });
        }
        // This reports the selected path, not successful decoding or an audible output signal.
        AudioPipelineTrace::RecordRenderDecision(
            locked_samplers.empty() ? AudioPipelineTrace::RenderDecision::NoEligibleSampler
                                    : AudioPipelineTrace::RenderDecision::DecodeAttempt);
        decoded_samplers.reserve(locked_samplers.size());
        progress.reserve(locked_samplers.size());
        for (auto& locked : locked_samplers) {
            const auto& voice = *locked.voice;
            AppliedSamplerVoiceSnapshot applied{
                .token = voice.token,
                .playback_state = voice.playback_state,
                .max_grain_samples = graph->system.max_grain_samples,
                .num_grain_samples = graph->system.num_grain_samples,
                .system_sample_rate = graph->system.sample_rate,
                .sampler_source_epoch = voice.sampler_source_epoch,
                .sampler = *voice.control.sampler,
                .decode_runtime = voice.decode_runtime,
            };
            auto decoded =
                StageAtrac9SamplerVoiceLocked(applied, graph->system.num_grain_samples,
                                              read_handler, read_context, decoder_factory);
            if (decoded.status != SamplerDecodeStatus::Produced &&
                decoded.status != SamplerDecodeStatus::EndOfStream) {
                RejectAll(locked_samplers);
                return DecodeError(decoded.status);
            }
            if (voice.decode_runtime->engine->HasStagedProgress()) {
                progress.push_back({
                    .token = voice.token,
                    .decode_runtime = voice.decode_runtime,
                    .handler = AcceptEngineProgress,
                    .context = voice.decode_runtime->engine.get(),
                });
            }
            decoded_samplers.push_back({&voice, std::move(decoded)});
        }

        std::map<BusKey, AudioBlock> buses;
        for (const auto& sampler : decoded_samplers) {
            AudioBlock block{
                .channels = sampler.decoded.channels,
                .frames = graph->system.num_grain_samples,
                .samples = std::vector<float>(static_cast<size_t>(graph->system.num_grain_samples) *
                                              sampler.decoded.channels),
            };
            const size_t copy_count =
                std::min(sampler.decoded.interleaved_samples.size(), block.samples.size());
            std::ranges::copy_n(sampler.decoded.interleaved_samples.begin(), copy_count,
                                block.samples.begin());
            RouteVoiceBlock(*graph, *sampler.voice, block, buses);
        }

        for (const auto* ordered_voice : *submixer_order) {
            const auto& voice = *ordered_voice;
            const auto channels = InputChannels(voice);
            if (!channels) {
                continue;
            }
            auto input =
                GatherVoiceInputs(voice, buses, *channels, graph->system.num_grain_samples);
            RouteVoiceBlock(*graph, voice, input, buses);
        }

        for (const auto& voice : graph->voices) {
            if (voice.token.rack_id != 0x3000 || !voice.control.mastering) {
                continue;
            }
            const auto channels = InputChannels(voice);
            const auto& mastering = *voice.control.mastering;
            if (!channels || !mastering.output_configured ||
                mastering.output_id >= outputs.size()) {
                continue;
            }
            auto input =
                GatherVoiceInputs(voice, buses, *channels, graph->system.num_grain_samples);
            const auto mastered =
                ApplyMastering(voice, input, outputs[mastering.output_id].channels);
            if (mastered) {
                AddIdentity(*mastered, 1.0F, outputs[mastering.output_id]);
            }
        }

        const auto result = CommitRenderedAudio(*graph, progress, buffer_info, outputs,
                                                publish_handler, publish_context);
        if (result < 0) {
            RejectAll(locked_samplers);
        }
        return result;
    } catch (const std::bad_alloc&) {
        RejectAll(locked_samplers);
        return ORBIS_NGS2_ERROR_FAIL;
    }
}

} // namespace Libraries::Ngs2
