// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "ngs2.h"
#include "ngs2_atrac9.h"
#include "ngs2_custom.h"
#include "ngs2_decode.h"
#include "ngs2_error.h"
#include "ngs2_impl.h"
#include "ngs2_mastering.h"
#include "ngs2_render.h"
#include "ngs2_reverb.h"
#include "ngs2_sampler.h"
#include "ngs2_submixer.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/libraries/audio/audio_pipeline_metadata_trace.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/kernel/kernel.h"

using namespace Libraries::Kernel;

namespace Libraries::Ngs2 {
namespace {

constexpr size_t SystemBufferSize = 64;
constexpr size_t RackBufferHeaderSize = 64;
constexpr size_t VoiceTokenStride = 16;
constexpr size_t ContextBufferAlignment = 8;
constexpr u32 MaxVoiceCount = 65535;
constexpr u32 MaxPersistentControlSlots = 4096;
constexpr size_t MaxVoiceControlCommands = 4096;
constexpr size_t MaxVoiceControlBytes = 1024 * 1024;
constexpr u32 MaxPersistedWaveformBlocks = 4096;
constexpr size_t MaxPersistedMatrixLevels = MaxVoiceControlBytes / sizeof(float);
constexpr size_t MaxVoiceStateBytes = 1024 * 1024;
constexpr u32 VoiceMatrixLevelsControlId = 0x00000001;
constexpr u32 VoicePortVolumeControlId = 0x00000002;
constexpr u32 VoicePortMatrixControlId = 0x00000003;
constexpr u32 VoicePatchControlId = 0x00000005;
constexpr u32 VoiceEventControlId = 0x00000006;
constexpr u32 SamplerSetupControlId = 0x10000000;
constexpr u32 SamplerWaveformBlocksControlId = 0x10000001;
constexpr u32 SamplerUserFxControlId = 0x10000008;
constexpr u32 SubmixerSetupControlId = 0x20000000;
constexpr u32 SubmixerUserFxControlId = 0x20000004;
constexpr u32 MasteringSetupControlId = 0x30000000;
constexpr u32 MasteringMatrixControlId = 0x30000001;
constexpr u32 MasteringGainControlId = 0x30000004;
constexpr u32 MasteringOutputControlId = 0x30000005;

// Native mastering matrix types enumerate every input/output layout pair in this order.
constexpr std::array<u32, 16> MasteringMatrixInputChannels{
    1, 1, 1, 1, 2, 2, 2, 2, 6, 6, 6, 6, 8, 8, 8, 8,
};
constexpr std::array<u32, 16> MasteringMatrixOutputChannels{
    1, 2, 6, 8, 1, 2, 6, 8, 1, 2, 6, 8, 1, 2, 6, 8,
};

enum class VoicePlayState {
    Empty,
    Playing,
    Paused,
    Stopped,
};

struct SystemRuntime {
    OrbisNgs2ContextBufferInfo buffer_info{};
    OrbisNgs2BufferFreeHandler host_free{};
    u32 max_grain_samples{};
    u32 num_grain_samples{};
    u32 sample_rate{};
    u64 generation{};
};

struct RackRuntime {
    OrbisNgs2Handle system_handle{};
    OrbisNgs2ContextBufferInfo buffer_info{};
    OrbisNgs2BufferFreeHandler host_free{};
    u32 rack_id{};
    u32 max_inputs{};
    u32 max_matrices{};
    u32 max_ports{};
    std::vector<OrbisNgs2Handle> voices;
    u64 generation{};
};

struct DestinationClassification {
    AudioPipelineTrace::DestinationExactMatch exact_match{};
    bool inside_live_rack{};
    AudioPipelineTrace::DestinationOffsetClass offset_class{};
    bool candidate_voice_index_valid{};
    u32 candidate_voice_index{};
    bool previously_returned{};
    u64 live_generation{};
    AudioPipelineTrace::DestinationTombstoneKind tombstone_kind{};
    u64 tombstone_generation{};

    auto operator<=>(const DestinationClassification&) const = default;
};

struct VoiceRuntime {
    OrbisNgs2Handle rack_handle{};
    u32 voice_index{};
    VoicePlayState state{VoicePlayState::Empty};
    std::vector<u32> pending_events;
    bool was_returned{};
    struct Patch {
        u32 destination_input_id{};
        OrbisNgs2Handle destination_handle{};
        bool last_observed{};
        bool last_destination_resolved{};
        bool last_destination_same_system{};
        u64 last_destination_generation{};
        DestinationClassification last_classification{};
        AudioPipelineTrace::PatchEndpointKind committed_endpoint_kind{};
        u32 committed_destination_rack_id{};
        bool committed_destination_voice_index_valid{};
        u32 committed_destination_voice_index{};
        u64 committed_destination_generation{};
        bool connected{};
    };
    std::unordered_map<u32, Patch> patches;
    SamplerControlSnapshot sampler;
    std::unordered_map<u32, std::vector<float>> matrices;
    std::unordered_map<u32, VoicePortSnapshot> ports;
    SubmixerControlSnapshot submixer;
    MasteringControlSnapshot mastering;
    u64 generation{};
    u64 sampler_source_epoch{1};
    std::shared_ptr<VoiceDecodeRuntime> decode_runtime{std::make_shared<VoiceDecodeRuntime>()};
    // Scoped by generation, so a same-address recreation cannot accept a stale render commit.
    u64 render_state_revision{1};
};

struct HandleTombstone {
    OrbisNgs2Handle identity{};
    AudioPipelineTrace::DestinationTombstoneKind kind{};
    u64 generation{};
    bool was_returned{};
};

struct RuntimeRegistry {
    std::mutex mutex;
    std::unordered_map<OrbisNgs2Handle, SystemRuntime> systems;
    std::unordered_map<OrbisNgs2Handle, RackRuntime> racks;
    std::unordered_map<OrbisNgs2Handle, VoiceRuntime> voices;
    std::deque<HandleTombstone> tombstones;
    u64 next_generation{1};
};

struct VoiceTraceIdentity {
    u32 rack_id{};
    u32 voice_index{};
    bool resolved{};
    OrbisNgs2Handle system_handle{};
    u64 generation{};
};

struct VoiceControlTarget {
    u32 rack_id{};
    u32 max_matrices{};
    u32 max_ports{};
};

RuntimeRegistry& Runtime() {
    static RuntimeRegistry runtime;
    return runtime;
}

void AddTombstoneLocked(RuntimeRegistry& runtime, HandleTombstone tombstone) {
    if (tombstone.identity == 0 || tombstone.generation == 0) {
        return;
    }
    runtime.tombstones.push_back(tombstone);
    if (runtime.tombstones.size() > MaxDestroyedHandleTombstones) {
        runtime.tombstones.pop_front();
    }
}

const HandleTombstone* FindTombstoneLocked(
    const RuntimeRegistry& runtime, OrbisNgs2Handle identity,
    std::optional<AudioPipelineTrace::DestinationTombstoneKind> kind = std::nullopt) {
    const auto tombstone = std::ranges::find_if(
        runtime.tombstones.rbegin(), runtime.tombstones.rend(), [&](const auto& candidate) {
            return candidate.identity == identity && (!kind || candidate.kind == *kind);
        });
    return tombstone == runtime.tombstones.rend() ? nullptr : &*tombstone;
}

VoiceTraceIdentity ResolveVoiceTraceIdentityLocked(const RuntimeRegistry& runtime,
                                                   OrbisNgs2Handle handle) {
    const auto voice_it = runtime.voices.find(handle);
    if (voice_it == runtime.voices.end()) {
        return {};
    }
    const auto rack_it = runtime.racks.find(voice_it->second.rack_handle);
    if (rack_it == runtime.racks.end()) {
        return {};
    }
    return {
        .rack_id = rack_it->second.rack_id,
        .voice_index = voice_it->second.voice_index,
        .resolved = true,
        .system_handle = rack_it->second.system_handle,
        .generation = voice_it->second.generation,
    };
}

VoiceTraceIdentity ResolveVoiceTraceIdentity(OrbisNgs2Handle handle) {
    auto& runtime = Runtime();
    std::scoped_lock lock{runtime.mutex};
    return ResolveVoiceTraceIdentityLocked(runtime, handle);
}

AudioPipelineTrace::SamplerPlaybackState TracePlaybackState(VoicePlayState state) {
    switch (state) {
    case VoicePlayState::Empty:
        return AudioPipelineTrace::SamplerPlaybackState::Empty;
    case VoicePlayState::Playing:
        return AudioPipelineTrace::SamplerPlaybackState::Playing;
    case VoicePlayState::Paused:
        return AudioPipelineTrace::SamplerPlaybackState::Paused;
    case VoicePlayState::Stopped:
        return AudioPipelineTrace::SamplerPlaybackState::Stopped;
    }
    return AudioPipelineTrace::SamplerPlaybackState::Empty;
}

struct PendingPatch {
    u32 source_port{};
    u32 destination_input_id{};
    OrbisNgs2Handle destination_handle{};
};

struct PendingSamplerSetup {
    WaveformFormatSnapshot format;
    u32 flags{};
};

struct PendingWaveformBlocks {
    uintptr_t guest_data_token{};
    u32 flags{};
    std::vector<WaveformBlockSnapshot> blocks;
};

struct PendingMatrixLevels {
    u32 matrix_id{};
    std::vector<float> levels;
};

struct PendingPortVolume {
    u32 port{};
    float level{};
};

struct PendingPortMatrix {
    u32 port{};
    s32 matrix_id{};
};

struct PendingSubmixerSetup {
    u32 num_io_channels{};
    u32 flags{};
};

struct PendingUserFx {
    u32 rack_id{};
    bool enabled{};
};

struct PendingMasteringSetup {
    u32 num_input_channels{};
    u32 flags{};
};

struct PendingMasteringMatrix {
    u32 type{};
    u32 input_channels{};
    u32 output_channels{};
    std::vector<float> levels;
};

struct PendingMasteringGain {
    float fbw_level{};
    float lfe_level{};
};

struct PendingMasteringOutput {
    u32 output_id{};
};

using PendingControlMutation =
    std::variant<PendingSamplerSetup, PendingWaveformBlocks, PendingMatrixLevels, PendingPortVolume,
                 PendingPortMatrix, PendingSubmixerSetup, PendingUserFx, PendingMasteringSetup,
                 PendingMasteringMatrix, PendingMasteringGain, PendingMasteringOutput>;

struct PatchEndpoint {
    AudioPipelineTrace::PatchEndpointKind kind{};
    u32 rack_id{};
    u32 voice_index{};
    bool voice_index_valid{};
    OrbisNgs2Handle system_handle{};
    u64 generation{};
    u32 max_inputs{};

    [[nodiscard]] bool IsResolved() const {
        return kind == AudioPipelineTrace::PatchEndpointKind::Rack ||
               kind == AudioPipelineTrace::PatchEndpointKind::Voice;
    }
};

struct PatchResolution {
    PatchEndpoint endpoint{};
    DestinationClassification classification{};
    bool present{};
    bool same_system{};
    bool destination_input_valid{};

    [[nodiscard]] bool IsValid() const {
        return !present || (endpoint.IsResolved() && same_system && destination_input_valid);
    }
};

DestinationClassification ClassifyDestinationLocked(const RuntimeRegistry& runtime,
                                                    OrbisNgs2Handle destination_handle) {
    DestinationClassification classification;
    if (destination_handle == 0) {
        return classification;
    }

    if (const auto system_it = runtime.systems.find(destination_handle);
        system_it != runtime.systems.end()) {
        classification.exact_match = AudioPipelineTrace::DestinationExactMatch::System;
        classification.live_generation = system_it->second.generation;
    } else if (const auto rack_it = runtime.racks.find(destination_handle);
               rack_it != runtime.racks.end()) {
        classification.exact_match = AudioPipelineTrace::DestinationExactMatch::Rack;
        classification.live_generation = rack_it->second.generation;
    } else if (const auto voice_it = runtime.voices.find(destination_handle);
               voice_it != runtime.voices.end()) {
        classification.exact_match = AudioPipelineTrace::DestinationExactMatch::Voice;
        classification.live_generation = voice_it->second.generation;
        classification.previously_returned = voice_it->second.was_returned;
    }

    for (const auto& [rack_handle, rack] : runtime.racks) {
        if (destination_handle < rack_handle) {
            continue;
        }
        const uintptr_t offset = destination_handle - rack_handle;
        if (offset >= rack.buffer_info.hostBufferSize) {
            continue;
        }

        classification.inside_live_rack = true;
        if (offset == 0) {
            classification.offset_class = AudioPipelineTrace::DestinationOffsetClass::RackBase;
        } else if (offset < RackBufferHeaderSize) {
            classification.offset_class = AudioPipelineTrace::DestinationOffsetClass::RackHeader;
        } else {
            const uintptr_t voice_offset = offset - RackBufferHeaderSize;
            const size_t voice_table_size = rack.voices.size() * VoiceTokenStride;
            if (voice_offset < voice_table_size) {
                classification.candidate_voice_index_valid = true;
                classification.candidate_voice_index =
                    static_cast<u32>(voice_offset / VoiceTokenStride);
                classification.offset_class =
                    voice_offset % VoiceTokenStride == 0
                        ? AudioPipelineTrace::DestinationOffsetClass::VoiceSlotAligned
                        : AudioPipelineTrace::DestinationOffsetClass::VoiceSlotInterior;
            } else {
                classification.offset_class = AudioPipelineTrace::DestinationOffsetClass::RackTail;
            }
        }
        break;
    }

    if (const auto* tombstone = FindTombstoneLocked(runtime, destination_handle)) {
        classification.tombstone_kind = tombstone->kind;
        classification.tombstone_generation = tombstone->generation;
        classification.previously_returned |= tombstone->was_returned;
    }
    return classification;
}

PatchEndpoint ResolvePatchEndpointLocked(const RuntimeRegistry& runtime,
                                         OrbisNgs2Handle destination_handle) {
    if (const auto rack_it = runtime.racks.find(destination_handle);
        rack_it != runtime.racks.end()) {
        const RackRuntime& rack = rack_it->second;
        return {
            .kind = AudioPipelineTrace::PatchEndpointKind::Rack,
            .rack_id = rack.rack_id,
            .system_handle = rack.system_handle,
            .generation = rack.generation,
            .max_inputs = rack.max_inputs,
        };
    }

    const auto voice_it = runtime.voices.find(destination_handle);
    if (voice_it == runtime.voices.end()) {
        return {};
    }
    const auto rack_it = runtime.racks.find(voice_it->second.rack_handle);
    if (rack_it == runtime.racks.end()) {
        return {};
    }
    const RackRuntime& rack = rack_it->second;
    return {
        .kind = AudioPipelineTrace::PatchEndpointKind::Voice,
        .rack_id = rack.rack_id,
        .voice_index = voice_it->second.voice_index,
        .voice_index_valid = true,
        .system_handle = rack.system_handle,
        .generation = voice_it->second.generation,
        .max_inputs = rack.max_inputs,
    };
}

PatchResolution ResolvePatchLocked(const RuntimeRegistry& runtime, const VoiceTraceIdentity& source,
                                   OrbisNgs2Handle destination_handle, u32 destination_input_id) {
    const PatchEndpoint endpoint = ResolvePatchEndpointLocked(runtime, destination_handle);
    return {
        .endpoint = endpoint,
        .classification = ClassifyDestinationLocked(runtime, destination_handle),
        .present = destination_handle != 0,
        .same_system = source.resolved && endpoint.IsResolved() &&
                       source.system_handle == endpoint.system_handle,
        .destination_input_valid =
            endpoint.IsResolved() && destination_input_id < endpoint.max_inputs,
    };
}

s32 PatchValidationResult(const PatchResolution& resolution) {
    if (!resolution.present) {
        return ORBIS_OK;
    }
    if (resolution.endpoint.IsResolved()) {
        return resolution.same_system && resolution.destination_input_valid
                   ? ORBIS_OK
                   : ORBIS_NGS2_ERROR_INVALID_PATCH;
    }

    switch (resolution.classification.exact_match) {
    case AudioPipelineTrace::DestinationExactMatch::System:
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    case AudioPipelineTrace::DestinationExactMatch::Rack:
        return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
    case AudioPipelineTrace::DestinationExactMatch::Voice:
        return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
    case AudioPipelineTrace::DestinationExactMatch::None:
        break;
    }
    switch (resolution.classification.tombstone_kind) {
    case AudioPipelineTrace::DestinationTombstoneKind::System:
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    case AudioPipelineTrace::DestinationTombstoneKind::Rack:
        return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
    case AudioPipelineTrace::DestinationTombstoneKind::Voice:
        return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
    case AudioPipelineTrace::DestinationTombstoneKind::None:
        return ORBIS_NGS2_ERROR_INVALID_HANDLE;
    }
    return ORBIS_NGS2_ERROR_INVALID_HANDLE;
}

PatchResolution ResolveCommittedPatchLocked(const RuntimeRegistry& runtime,
                                            const VoiceTraceIdentity& source,
                                            const VoiceRuntime::Patch& patch) {
    PatchResolution resolution =
        ResolvePatchLocked(runtime, source, patch.destination_handle, patch.destination_input_id);
    if (!patch.connected || resolution.endpoint.kind != patch.committed_endpoint_kind ||
        resolution.endpoint.generation != patch.committed_destination_generation) {
        resolution.endpoint = {};
        resolution.same_system = false;
        resolution.destination_input_valid = false;
    }
    return resolution;
}

AudioPipelineTrace::VoicePatchStateMetadata PatchStateMetadata(
    const VoiceTraceIdentity& source, u32 source_port, u32 destination_input_id,
    const PatchResolution& resolution, u64 committed_destination_generation,
    AudioPipelineTrace::PatchResolutionPhase phase) {
    const auto& classification = resolution.classification;
    return {
        .source_rack_id = source.rack_id,
        .source_voice_index = source.voice_index,
        .source_is_sampler = source.rack_id == 0x1000,
        .source_port = source_port,
        .destination_input_id = destination_input_id,
        .destination_rack_id = resolution.endpoint.rack_id,
        .destination_voice_index = resolution.endpoint.voice_index,
        .destination_present = resolution.present,
        .destination_resolved = resolution.endpoint.IsResolved(),
        .destination_same_system = resolution.same_system,
        .destination_exact_match = classification.exact_match,
        .destination_inside_live_rack = classification.inside_live_rack,
        .destination_offset_class = classification.offset_class,
        .destination_candidate_voice_index_valid = classification.candidate_voice_index_valid,
        .destination_candidate_voice_index = classification.candidate_voice_index,
        .destination_previously_returned = classification.previously_returned,
        .destination_live_generation = classification.live_generation,
        .destination_tombstone_kind = classification.tombstone_kind,
        .destination_tombstone_generation = classification.tombstone_generation,
        .destination_generation_changed_since_commit =
            committed_destination_generation != 0 &&
            committed_destination_generation != classification.live_generation,
        .destination_endpoint_kind = resolution.endpoint.kind,
        .destination_voice_index_valid = resolution.endpoint.voice_index_valid,
        .phase = phase,
    };
}

using PendingTraceRecord = std::variant<
    AudioPipelineTrace::SamplerSetupMetadata, AudioPipelineTrace::WaveformBlocksMetadata,
    AudioPipelineTrace::UserFxMetadata, AudioPipelineTrace::VoiceMatrixLevelsMetadata,
    AudioPipelineTrace::VoicePortVolumeMetadata, AudioPipelineTrace::VoicePortMatrixMetadata,
    AudioPipelineTrace::MasteringOutputMetadata>;

void EmitTraceRecord(const PendingTraceRecord& record) {
    std::visit(
        [](const auto& metadata) {
            using Metadata = std::remove_cvref_t<decltype(metadata)>;
            if constexpr (std::same_as<Metadata, AudioPipelineTrace::SamplerSetupMetadata>) {
                AudioPipelineTrace::RecordSamplerSetup(metadata);
            } else if constexpr (std::same_as<Metadata,
                                              AudioPipelineTrace::WaveformBlocksMetadata>) {
                AudioPipelineTrace::RecordWaveformBlocks(metadata);
            } else if constexpr (std::same_as<Metadata, AudioPipelineTrace::UserFxMetadata>) {
                AudioPipelineTrace::RecordUserFx(metadata);
            } else if constexpr (std::same_as<Metadata,
                                              AudioPipelineTrace::VoiceMatrixLevelsMetadata>) {
                AudioPipelineTrace::RecordVoiceMatrixLevels(metadata);
            } else if constexpr (std::same_as<Metadata,
                                              AudioPipelineTrace::VoicePortVolumeMetadata>) {
                AudioPipelineTrace::RecordVoicePortVolume(metadata);
            } else if constexpr (std::same_as<Metadata,
                                              AudioPipelineTrace::VoicePortMatrixMetadata>) {
                AudioPipelineTrace::RecordVoicePortMatrix(metadata);
            } else if constexpr (std::same_as<Metadata,
                                              AudioPipelineTrace::MasteringOutputMetadata>) {
                AudioPipelineTrace::RecordMasteringOutput(metadata);
            }
        },
        record);
}

bool HandleIsInUse(const RuntimeRegistry& runtime, OrbisNgs2Handle handle) {
    return runtime.systems.contains(handle) || runtime.racks.contains(handle) ||
           runtime.voices.contains(handle);
}

bool RangeEnd(uintptr_t base, size_t size, uintptr_t* out_end) {
    if (size == 0 || size > std::numeric_limits<uintptr_t>::max() - base) {
        return false;
    }
    *out_end = base + size;
    return true;
}

bool BackingRangeOverlaps(const RuntimeRegistry& runtime, uintptr_t base, size_t size) {
    uintptr_t end = 0;
    if (!RangeEnd(base, size, &end)) {
        return true;
    }
    const auto overlaps = [base, end](const OrbisNgs2ContextBufferInfo& info) {
        const auto other_base = reinterpret_cast<uintptr_t>(info.hostBuffer);
        uintptr_t other_end = 0;
        return !RangeEnd(other_base, info.hostBufferSize, &other_end) ||
               (base < other_end && other_base < end);
    };
    return std::ranges::any_of(
               runtime.systems,
               [&](const auto& entry) { return overlaps(entry.second.buffer_info); }) ||
           std::ranges::any_of(runtime.racks, [&](const auto& entry) {
               return overlaps(entry.second.buffer_info);
           });
}

bool AddWouldOverflow(uintptr_t value, size_t addend) {
    return addend > std::numeric_limits<uintptr_t>::max() - value;
}

struct VoiceStateSizeRequirement {
    size_t minimum_size;
    bool exact;
};

VoiceStateSizeRequirement VoiceStateSizeForRack(u32 rack_id) {
    switch (rack_id) {
    case 0x1000:
        return {sizeof(OrbisNgs2SamplerVoiceState), true};
    case 0x2000:
        return {sizeof(OrbisNgs2SubmixerVoiceState), true};
    case 0x2001:
        return {sizeof(OrbisNgs2ReverbVoiceState), true};
    case 0x3000:
        return {sizeof(OrbisNgs2MasteringVoiceState), true};
    case 0x4002:
        return {sizeof(OrbisNgs2CustomSubmixerVoiceState), false};
    default:
        return {sizeof(OrbisNgs2VoiceState), false};
    }
}

struct NormalizedRackOption {
    OrbisNgs2RackOption base{};
    u32 max_inputs{};
    u32 max_matrices{};
    u32 max_ports{};
};

s32 NormalizeRackOption(u32 rack_id, const OrbisNgs2RackOption* option, bool option_required,
                        NormalizedRackOption* out_option) {
    if (!option) {
        if (option_required) {
            return ORBIS_NGS2_ERROR_INVALID_OPTION_ADDRESS;
        }
    } else {
        if (option->size < sizeof(OrbisNgs2RackOption)) {
            return ORBIS_NGS2_ERROR_INVALID_OPTION_SIZE;
        }
        out_option->base = *option;
    }
    out_option->max_matrices = out_option->base.maxMatrices;
    if (out_option->max_matrices > MaxPersistentControlSlots) {
        return ORBIS_NGS2_ERROR_INVALID_MAX_MATRICES;
    }
    out_option->max_ports = out_option->base.maxPorts;
    if (out_option->max_ports > MaxPersistentControlSlots) {
        return ORBIS_NGS2_ERROR_INVALID_MAX_PORTS;
    }

    switch (rack_id) {
    case 0x1000: // Sampler racks are sources and expose no patch inputs.
        out_option->max_inputs = 0;
        break;
    case 0x2000: { // Standard submixer: an absent typed option defaults to one input.
        u32 requested_inputs = 1;
        if (option) {
            if (option->size < sizeof(OrbisNgs2SubmixerRackOption)) {
                return ORBIS_NGS2_ERROR_INVALID_OPTION_SIZE;
            }
            OrbisNgs2SubmixerRackOption typed{};
            std::memcpy(&typed, option, sizeof(typed));
            requested_inputs = typed.maxInputs;
        }
        out_option->max_inputs = requested_inputs;
        if (out_option->max_inputs < 1 || out_option->max_inputs > 2) {
            return ORBIS_NGS2_ERROR_INVALID_MAX_INPUTS;
        }
        break;
    }
    case 0x2001: // Standard reverb exposes one fixed input.
        if (option && option->size < sizeof(OrbisNgs2ReverbRackOption)) {
            return ORBIS_NGS2_ERROR_INVALID_OPTION_SIZE;
        }
        out_option->max_inputs = 1;
        break;
    case 0x3000: // Standard mastering exposes one fixed input.
        if (option && option->size < sizeof(OrbisNgs2MasteringRackOption)) {
            return ORBIS_NGS2_ERROR_INVALID_OPTION_SIZE;
        }
        out_option->max_inputs = 1;
        break;
    default:
        // Keep existing lifecycle support, but fail closed for routing until a typed capacity is
        // proven for this rack kind.
        out_option->max_inputs = 0;
        break;
    }
    return ORBIS_OK;
}

bool RequiredRackBufferSize(u32 max_voices, size_t* out_size) {
    if (max_voices > MaxVoiceCount) {
        return false;
    }
    *out_size = RackBufferHeaderSize + static_cast<size_t>(max_voices) * VoiceTokenStride;
    return true;
}

s32 ValidateSystemOption(const OrbisNgs2SystemOption* option) {
    if (!option) {
        return ORBIS_OK;
    }
    if (option->size != sizeof(OrbisNgs2SystemOption)) {
        return ORBIS_NGS2_ERROR_INVALID_OPTION_SIZE;
    }

    const u32 max_grain_samples = option->maxGrainSamples;
    const u32 num_grain_samples = option->numGrainSamples;
    const u32 sample_rate = option->sampleRate;
    if (max_grain_samples < 64 || max_grain_samples > 1024 || (max_grain_samples & 63) != 0) {
        return ORBIS_NGS2_ERROR_INVALID_MAX_GRAIN_SAMPLES;
    }
    if (num_grain_samples < 64 || num_grain_samples > 1024 || (num_grain_samples & 63) != 0) {
        return ORBIS_NGS2_ERROR_INVALID_NUM_GRAIN_SAMPLES;
    }
    if (sample_rate != 11025 && sample_rate != 12000 && sample_rate != 22050 &&
        sample_rate != 24000 && sample_rate != 44100 && sample_rate != 48000 &&
        sample_rate != 88200 && sample_rate != 96000 && sample_rate != 176400 &&
        sample_rate != 192000) {
        return ORBIS_NGS2_ERROR_INVALID_SAMPLE_RATE;
    }
    return ORBIS_OK;
}

SystemRuntimeSnapshot EffectiveSystemRuntime(const OrbisNgs2SystemOption* option) {
    if (!option) {
        return {
            .max_grain_samples = OrbisNgs2DefaultMaxGrainSamples,
            .num_grain_samples = OrbisNgs2DefaultNumGrainSamples,
            .sample_rate = OrbisNgs2DefaultSampleRate,
        };
    }
    return {
        .max_grain_samples = option->maxGrainSamples,
        .num_grain_samples = option->numGrainSamples,
        .sample_rate = option->sampleRate,
    };
}

u32 StateFlags(VoicePlayState state) {
    switch (state) {
    case VoicePlayState::Empty:
        return 0;
    case VoicePlayState::Playing:
        return 0x3;
    case VoicePlayState::Paused:
        return 0x5;
    case VoicePlayState::Stopped:
        return 0xB;
    }
    return 0;
}

void AdvanceRenderStateRevisionLocked(VoiceRuntime& voice) {
    // A 64-bit revision cannot exhaust within a practical voice lifetime. Failing closed preserves
    // strict monotonicity and keeps zero reserved for invalid/uninitialized snapshots.
    if (voice.render_state_revision == std::numeric_limits<u64>::max()) [[unlikely]] {
        UNREACHABLE_MSG("NGS2 voice render-state revision exhausted");
    }
    ++voice.render_state_revision;
}

void AdvanceSamplerSourceEpochLocked(VoiceRuntime& voice) {
    if (voice.sampler_source_epoch == std::numeric_limits<u64>::max()) [[unlikely]] {
        UNREACHABLE_MSG("NGS2 sampler source epoch exhausted");
    }
    ++voice.sampler_source_epoch;
}

bool ApplyEvent(VoiceRuntime& voice, u32 event_id) {
    const VoicePlayState previous_state = voice.state;
    switch (event_id) {
    case 0: // Play
        if (voice.state == VoicePlayState::Empty) {
            voice.state = VoicePlayState::Playing;
        }
        break;
    case 1: // Stop
        if (voice.state == VoicePlayState::Playing || voice.state == VoicePlayState::Paused) {
            voice.state = VoicePlayState::Stopped;
        }
        break;
    case 2: // Stop immediately
    case 3: // Kill
        voice.state = VoicePlayState::Empty;
        break;
    case 4: // Pause
        if (voice.state == VoicePlayState::Playing) {
            voice.state = VoicePlayState::Paused;
        }
        break;
    case 5: // Resume
        if (voice.state == VoicePlayState::Paused) {
            voice.state = VoicePlayState::Playing;
        }
        break;
    default:
        break;
    }
    return voice.state != previous_state;
}

VoiceRenderPlaybackState RenderPlaybackState(VoicePlayState state) noexcept {
    switch (state) {
    case VoicePlayState::Empty:
        return VoiceRenderPlaybackState::Empty;
    case VoicePlayState::Playing:
        return VoiceRenderPlaybackState::Playing;
    case VoicePlayState::Paused:
        return VoiceRenderPlaybackState::Paused;
    case VoicePlayState::Stopped:
        return VoiceRenderPlaybackState::Stopped;
    }
    return VoiceRenderPlaybackState::Empty;
}

bool AdvanceAddress(uintptr_t address, s16 delta, uintptr_t* out_address) {
    if (delta >= 0) {
        const auto positive_delta = static_cast<size_t>(delta);
        if (AddWouldOverflow(address, positive_delta)) {
            return false;
        }
        *out_address = address + positive_delta;
        return true;
    }

    const auto magnitude = static_cast<uintptr_t>(-static_cast<s32>(delta));
    if (address < magnitude) {
        return false;
    }
    *out_address = address - magnitude;
    return true;
}

u64 SaturatingAdd(u64 left, u64 right) {
    return right > std::numeric_limits<u64>::max() - left ? std::numeric_limits<u64>::max()
                                                          : left + right;
}

AudioPipelineTrace::WaveformBlocksMetadata TraceWaveformBlocks(
    const VoiceTraceIdentity& source, const OrbisNgs2SamplerVoiceWaveformBlocksParam& param,
    std::span<const WaveformBlockSnapshot> blocks) {
    AudioPipelineTrace::WaveformBlocksMetadata metadata{
        .source_rack_id = source.rack_id,
        .source_voice_index = source.voice_index,
        .flags = param.flags,
        .data_present = param.data != nullptr,
        .block_count = param.numBlocks,
        .data_range_arithmetic_valid = param.numBlocks == 0 || param.data != nullptr,
        .metadata_complete = true,
    };
    const uintptr_t data_base = reinterpret_cast<uintptr_t>(param.data);
    bool saw_block = false;
    for (const WaveformBlockSnapshot& block : blocks) {
        if (!saw_block) {
            metadata.minimum_data_offset = block.data_offset;
            saw_block = true;
        } else {
            metadata.minimum_data_offset =
                std::min(metadata.minimum_data_offset, block.data_offset);
        }
        metadata.maximum_data_end = std::max(metadata.maximum_data_end,
                                             static_cast<u64>(block.data_offset) + block.data_size);
        metadata.aggregate_data_bytes =
            SaturatingAdd(metadata.aggregate_data_bytes, block.data_size);
        metadata.aggregate_samples = SaturatingAdd(metadata.aggregate_samples, block.num_samples);
        metadata.aggregate_skip_samples =
            SaturatingAdd(metadata.aggregate_skip_samples, block.num_skip_samples);
        metadata.has_repeats |= block.num_repeats != 0;
        if (AddWouldOverflow(data_base, block.data_offset) ||
            AddWouldOverflow(data_base + block.data_offset, block.data_size)) {
            metadata.data_range_arithmetic_valid = false;
        }
    }
    return metadata;
}

s32 ParsePersistentVoiceCommand(const OrbisNgs2VoiceParamHeader& header,
                                std::span<const u8> command,
                                OrbisNgs2RuntimeReadHandler read_handler, void* read_context,
                                const VoiceControlTarget& target, u32& remaining_waveform_blocks,
                                size_t& remaining_matrix_levels,
                                std::vector<PendingControlMutation>& mutations) {
    switch (header.id) {
    case VoiceMatrixLevelsControlId: {
        if (command.size() != sizeof(OrbisNgs2VoiceMatrixLevelsParam)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        OrbisNgs2VoiceMatrixLevelsParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        if (param.matrixId >= target.max_matrices) {
            return ORBIS_NGS2_ERROR_INVALID_MATRIX_INDEX;
        }
        if (param.numLevels > ORBIS_NGS2_MAX_MATRIX_LEVELS ||
            param.numLevels > remaining_matrix_levels) {
            return ORBIS_NGS2_ERROR_INVALID_NUM_MATRIX_LEVELS;
        }
        if (param.numLevels != 0 && param.aLevel == nullptr) {
            return ORBIS_NGS2_ERROR_INVALID_MATRIX_LEVEL_ADDRESS;
        }
        std::vector<float> levels(param.numLevels);
        if (param.numLevels != 0) {
            const size_t byte_size = static_cast<size_t>(param.numLevels) * sizeof(float);
            const uintptr_t level_address = reinterpret_cast<uintptr_t>(param.aLevel);
            if (AddWouldOverflow(level_address, byte_size) ||
                !read_handler(level_address, levels.data(), byte_size, read_context)) {
                return ORBIS_NGS2_ERROR_INVALID_MATRIX_LEVEL_ADDRESS;
            }
            remaining_matrix_levels -= param.numLevels;
        }
        mutations.emplace_back(PendingMatrixLevels{
            .matrix_id = param.matrixId,
            .levels = std::move(levels),
        });
        return ORBIS_OK;
    }
    case VoicePortVolumeControlId: {
        if (command.size() != sizeof(OrbisNgs2VoicePortVolumeParam)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        OrbisNgs2VoicePortVolumeParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        if (param.port >= target.max_ports) {
            return ORBIS_NGS2_ERROR_INVALID_PORT_INDEX;
        }
        mutations.emplace_back(PendingPortVolume{param.port, param.level});
        return ORBIS_OK;
    }
    case VoicePortMatrixControlId: {
        if (command.size() != sizeof(OrbisNgs2VoicePortMatrixParam)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        OrbisNgs2VoicePortMatrixParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        if (param.port >= target.max_ports) {
            return ORBIS_NGS2_ERROR_INVALID_PORT_INDEX;
        }
        if (param.matrixId < -1 ||
            (param.matrixId >= 0 && static_cast<u32>(param.matrixId) >= target.max_matrices)) {
            return ORBIS_NGS2_ERROR_INVALID_MATRIX_INDEX;
        }
        mutations.emplace_back(PendingPortMatrix{param.port, param.matrixId});
        return ORBIS_OK;
    }
    case SamplerSetupControlId: {
        if (command.size() != sizeof(OrbisNgs2SamplerVoiceSetupParam)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        OrbisNgs2SamplerVoiceSetupParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        if (target.rack_id != 0x1000) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
        }
        mutations.emplace_back(PendingSamplerSetup{
            .format =
                {
                    .waveform_type = param.format.waveformType,
                    .num_channels = param.format.numChannels,
                    .sample_rate = param.format.sampleRate,
                    .config_data = param.format.configData,
                    .frame_offset = param.format.frameOffset,
                    .frame_margin = param.format.frameMargin,
                },
            .flags = param.flags,
        });
        return ORBIS_OK;
    }
    case SamplerWaveformBlocksControlId: {
        if (command.size() != sizeof(OrbisNgs2SamplerVoiceWaveformBlocksParam)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        OrbisNgs2SamplerVoiceWaveformBlocksParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        if (target.rack_id != 0x1000) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
        }
        if (param.numBlocks > MaxPersistedWaveformBlocks ||
            param.numBlocks > remaining_waveform_blocks) {
            return ORBIS_NGS2_ERROR_INVALID_NUM_WAVEFORM_BLOCKS;
        }
        if (param.numBlocks != 0 && param.data == nullptr) {
            return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_ADDRESS;
        }
        if (param.numBlocks != 0 && param.aBlock == nullptr) {
            return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_BLOCK_ADDRESS;
        }

        std::vector<WaveformBlockSnapshot> blocks;
        blocks.reserve(param.numBlocks);
        uintptr_t block_address = reinterpret_cast<uintptr_t>(param.aBlock);
        const size_t descriptor_bytes =
            static_cast<size_t>(param.numBlocks) * sizeof(OrbisNgs2WaveformBlock);
        if (descriptor_bytes != 0 && AddWouldOverflow(block_address, descriptor_bytes)) {
            return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_BLOCK_ADDRESS;
        }
        const uintptr_t data_base = reinterpret_cast<uintptr_t>(param.data);
        for (u32 block_index = 0; block_index < param.numBlocks; ++block_index) {
            OrbisNgs2WaveformBlock block{};
            if (!read_handler(block_address, &block, sizeof(block), read_context)) {
                return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_BLOCK_ADDRESS;
            }
            if (AddWouldOverflow(data_base, block.dataOffset)) {
                return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_ADDRESS;
            }
            const uintptr_t block_data = data_base + block.dataOffset;
            if (AddWouldOverflow(block_data, block.dataSize)) {
                return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_ADDRESS;
            }
            blocks.push_back({
                .data_offset = block.dataOffset,
                .data_size = block.dataSize,
                .num_repeats = block.numRepeats,
                .num_skip_samples = block.numSkipSamples,
                .num_samples = block.numSamples,
            });
            block_address += sizeof(block);
        }
        remaining_waveform_blocks -= param.numBlocks;
        mutations.emplace_back(PendingWaveformBlocks{
            .guest_data_token = reinterpret_cast<uintptr_t>(param.data),
            .flags = param.flags,
            .blocks = std::move(blocks),
        });
        return ORBIS_OK;
    }
    case SamplerUserFxControlId: {
        if (command.size() != sizeof(OrbisNgs2SamplerVoiceUserFxParam)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        if (target.rack_id != 0x1000) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
        }
        OrbisNgs2SamplerVoiceUserFxParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        mutations.emplace_back(PendingUserFx{target.rack_id, param.handler != nullptr});
        return ORBIS_OK;
    }
    case SubmixerSetupControlId: {
        if (command.size() != sizeof(OrbisNgs2SubmixerVoiceSetupParam)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        OrbisNgs2SubmixerVoiceSetupParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        if (target.rack_id != 0x2000) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
        }
        if (param.numIoChannels == 0 || param.numIoChannels > ORBIS_NGS2_MAX_VOICE_CHANNELS) {
            return ORBIS_NGS2_ERROR_INVALID_NUM_CHANNELS;
        }
        mutations.emplace_back(PendingSubmixerSetup{param.numIoChannels, param.flags});
        return ORBIS_OK;
    }
    case SubmixerUserFxControlId: {
        if (command.size() != sizeof(OrbisNgs2SubmixerVoiceUserFxParam)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        if (target.rack_id != 0x2000) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
        }
        OrbisNgs2SubmixerVoiceUserFxParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        mutations.emplace_back(PendingUserFx{target.rack_id, param.handler != nullptr});
        return ORBIS_OK;
    }
    case MasteringSetupControlId: {
        if (command.size() != sizeof(OrbisNgs2MasteringVoiceSetupParam)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        if (target.rack_id != 0x3000) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
        }
        OrbisNgs2MasteringVoiceSetupParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        if (param.numInputChannels == 0 || param.numInputChannels > ORBIS_NGS2_MAX_VOICE_CHANNELS) {
            return ORBIS_NGS2_ERROR_INVALID_NUM_CHANNELS;
        }
        mutations.emplace_back(PendingMasteringSetup{param.numInputChannels, param.flags});
        return ORBIS_OK;
    }
    case MasteringMatrixControlId: {
        if (command.size() != sizeof(OrbisNgs2MasteringVoiceMatrixParam)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        if (target.rack_id != 0x3000) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
        }
        OrbisNgs2MasteringVoiceMatrixParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        if (param.type >= MasteringMatrixInputChannels.size()) {
            return ORBIS_NGS2_ERROR_INVALID_MATRIX_INDEX;
        }
        const u32 input_channels = MasteringMatrixInputChannels[param.type];
        const u32 output_channels = MasteringMatrixOutputChannels[param.type];
        const u32 expected_levels = input_channels * output_channels;
        if (param.numLevels != expected_levels || param.numLevels > remaining_matrix_levels) {
            return ORBIS_NGS2_ERROR_INVALID_NUM_MATRIX_LEVELS;
        }
        if (!param.aLevel) {
            return ORBIS_NGS2_ERROR_INVALID_MATRIX_LEVEL_ADDRESS;
        }
        std::vector<float> levels(param.numLevels);
        const size_t byte_size = static_cast<size_t>(param.numLevels) * sizeof(float);
        const uintptr_t level_address = reinterpret_cast<uintptr_t>(param.aLevel);
        if (AddWouldOverflow(level_address, byte_size) ||
            !read_handler(level_address, levels.data(), byte_size, read_context)) {
            return ORBIS_NGS2_ERROR_INVALID_MATRIX_LEVEL_ADDRESS;
        }
        remaining_matrix_levels -= param.numLevels;
        mutations.emplace_back(PendingMasteringMatrix{
            .type = param.type,
            .input_channels = input_channels,
            .output_channels = output_channels,
            .levels = std::move(levels),
        });
        return ORBIS_OK;
    }
    case MasteringGainControlId: {
        if (command.size() != sizeof(OrbisNgs2MasteringVoiceGainParam)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        OrbisNgs2MasteringVoiceGainParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        if (target.rack_id != 0x3000) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
        }
        mutations.emplace_back(PendingMasteringGain{param.fbwLevel, param.lfeLevel});
        return ORBIS_OK;
    }
    case MasteringOutputControlId: {
        if (command.size() != sizeof(OrbisNgs2MasteringVoiceOutputParam)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        OrbisNgs2MasteringVoiceOutputParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        if (target.rack_id != 0x3000) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
        }
        mutations.emplace_back(PendingMasteringOutput{param.outputId});
        return ORBIS_OK;
    }
    default:
        return ORBIS_OK;
    }
}

void TraceVoiceCommand(const VoiceTraceIdentity& source, const OrbisNgs2VoiceParamHeader& header,
                       std::span<const u8> command,
                       const PendingControlMutation* persistent_mutation,
                       std::vector<PendingTraceRecord>& records) {
    switch (header.id) {
    case VoiceMatrixLevelsControlId: {
        if (!source.resolved || command.size() != sizeof(OrbisNgs2VoiceMatrixLevelsParam)) {
            return;
        }
        OrbisNgs2VoiceMatrixLevelsParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        records.emplace_back(AudioPipelineTrace::VoiceMatrixLevelsMetadata{
            .source_rack_id = source.rack_id,
            .source_voice_index = source.voice_index,
            .matrix_id = param.matrixId,
            .level_count = param.numLevels,
            .levels_present = param.aLevel != nullptr,
        });
        return;
    }
    case VoicePortVolumeControlId: {
        if (!source.resolved || command.size() != sizeof(OrbisNgs2VoicePortVolumeParam)) {
            return;
        }
        OrbisNgs2VoicePortVolumeParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        records.emplace_back(AudioPipelineTrace::VoicePortVolumeMetadata{
            .source_rack_id = source.rack_id,
            .source_voice_index = source.voice_index,
            .port = param.port,
            .level = param.level,
        });
        return;
    }
    case VoicePortMatrixControlId: {
        if (!source.resolved || command.size() != sizeof(OrbisNgs2VoicePortMatrixParam)) {
            return;
        }
        OrbisNgs2VoicePortMatrixParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        records.emplace_back(AudioPipelineTrace::VoicePortMatrixMetadata{
            .source_rack_id = source.rack_id,
            .source_voice_index = source.voice_index,
            .port = param.port,
            .matrix_id = param.matrixId,
        });
        return;
    }
    case VoicePatchControlId: {
        return;
    }
    case SamplerSetupControlId: {
        if (!source.resolved || command.size() != sizeof(OrbisNgs2SamplerVoiceSetupParam)) {
            return;
        }
        OrbisNgs2SamplerVoiceSetupParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        const Atrac9FormatInspection atrac9 = InspectAtrac9Format(param.format);
        const bool atrac9_config_valid =
            atrac9.status == Atrac9FormatStatus::DeclaredFormatMismatch ||
            atrac9.status == Atrac9FormatStatus::Supported;
        records.emplace_back(AudioPipelineTrace::SamplerSetupMetadata{
            .source_rack_id = source.rack_id,
            .source_voice_index = source.voice_index,
            .waveform_type = param.format.waveformType,
            .channels = param.format.numChannels,
            .sample_rate = param.format.sampleRate,
            .codec_config = param.format.configData,
            .frame_offset = param.format.frameOffset,
            .frame_margin = param.format.frameMargin,
            .flags = param.flags,
            .atrac9_applicable = atrac9.status != Atrac9FormatStatus::UnsupportedWaveform,
            .atrac9_config_valid = atrac9_config_valid,
            .atrac9_declared_format_matches = atrac9.status == Atrac9FormatStatus::Supported,
            .atrac9_frame_bytes = atrac9.info.frame_bytes,
            .atrac9_frames_per_superframe = atrac9.info.frames_per_superframe,
            .atrac9_frame_samples = atrac9.info.frame_samples,
            .atrac9_superframe_bytes = atrac9.info.superframe_bytes,
            .atrac9_superframe_samples = atrac9.info.superframe_samples,
        });
        return;
    }
    case SamplerWaveformBlocksControlId: {
        if (!source.resolved ||
            command.size() != sizeof(OrbisNgs2SamplerVoiceWaveformBlocksParam)) {
            return;
        }
        OrbisNgs2SamplerVoiceWaveformBlocksParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        const auto* waveform =
            persistent_mutation ? std::get_if<PendingWaveformBlocks>(persistent_mutation) : nullptr;
        if (waveform) {
            records.emplace_back(TraceWaveformBlocks(source, param, waveform->blocks));
        }
        return;
    }
    case SamplerUserFxControlId: {
        if (command.size() != sizeof(OrbisNgs2SamplerVoiceUserFxParam)) {
            return;
        }
        OrbisNgs2SamplerVoiceUserFxParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        records.emplace_back(
            AudioPipelineTrace::UserFxMetadata{header.id, param.handler != nullptr});
        return;
    }
    case SubmixerUserFxControlId: {
        if (command.size() != sizeof(OrbisNgs2SubmixerVoiceUserFxParam)) {
            return;
        }
        OrbisNgs2SubmixerVoiceUserFxParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        records.emplace_back(
            AudioPipelineTrace::UserFxMetadata{header.id, param.handler != nullptr});
        return;
    }
    case MasteringOutputControlId: {
        if (!source.resolved || command.size() != sizeof(OrbisNgs2MasteringVoiceOutputParam)) {
            return;
        }
        OrbisNgs2MasteringVoiceOutputParam param{};
        std::memcpy(&param, command.data(), sizeof(param));
        records.emplace_back(AudioPipelineTrace::MasteringOutputMetadata{
            .source_rack_id = source.rack_id,
            .source_voice_index = source.voice_index,
            .output_id = param.outputId,
        });
        return;
    }
    default:
        return;
    }
}

} // namespace

s32 HandleReportInvalid(OrbisNgs2Handle handle, u32 handleType) {
    switch (handleType) {
    case 1:
        LOG_ERROR(Lib_Ngs2, "Invalid system handle {}", handle);
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    case 2:
        LOG_ERROR(Lib_Ngs2, "Invalid rack handle {}", handle);
        return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
    case 4:
        LOG_ERROR(Lib_Ngs2, "Invalid voice handle {}", handle);
        return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
    case 8:
        LOG_ERROR(Lib_Ngs2, "Invalid report handle {}", handle);
        return ORBIS_NGS2_ERROR_INVALID_REPORT_HANDLE;
    default:
        LOG_ERROR(Lib_Ngs2, "Invalid handle {}", handle);
        return ORBIS_NGS2_ERROR_INVALID_HANDLE;
    }
}

void* MemoryClear(void* buffer, size_t size) {
    return std::memset(buffer, 0, size);
}

s32 StackBufferClose(StackBuffer* stackBuffer, size_t* outTotalSize) {
    if (outTotalSize) {
        *outTotalSize = stackBuffer->usedSize + stackBuffer->alignment;
    }
    return ORBIS_OK;
}

s32 StackBufferOpen(StackBuffer* stackBuffer, void* bufferStart, size_t bufferSize,
                    void** outBuffer, u8 flags) {
    stackBuffer->top = outBuffer;
    stackBuffer->base = bufferStart;
    stackBuffer->size = reinterpret_cast<size_t>(bufferStart);
    stackBuffer->currentOffset = reinterpret_cast<size_t>(bufferStart);
    stackBuffer->usedSize = 0;
    stackBuffer->totalSize = bufferSize;
    stackBuffer->alignment = 8;
    stackBuffer->flags = flags;

    if (outBuffer) {
        *outBuffer = nullptr;
    }
    return ORBIS_OK;
}

s32 SystemSetupCore(StackBuffer* stackBuffer, const OrbisNgs2SystemOption* option,
                    SystemInternal* outSystem) {
    const s32 result = ValidateSystemOption(option);
    if (result < 0) {
        return result;
    }
    if (stackBuffer) {
        stackBuffer->usedSize = std::max(stackBuffer->usedSize, SystemBufferSize - 8);
    }
    if (outSystem) {
        outSystem->systemHandle = reinterpret_cast<OrbisNgs2Handle>(stackBuffer->base);
    }
    return ORBIS_OK;
}

s32 SystemSetup(const OrbisNgs2SystemOption* option, OrbisNgs2ContextBufferInfo* hostBufferInfo,
                OrbisNgs2BufferFreeHandler hostFree, OrbisNgs2Handle* outHandle) {
    if (!hostBufferInfo) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_INFO;
    }
    const s32 option_result = ValidateSystemOption(option);
    if (option_result < 0) {
        return option_result;
    }

    if (!outHandle) {
        hostBufferInfo->hostBuffer = nullptr;
        hostBufferInfo->hostBufferSize = SystemBufferSize;
        MemoryClear(hostBufferInfo->reserved, sizeof(hostBufferInfo->reserved));
        return ORBIS_OK;
    }
    if (!hostBufferInfo->hostBuffer) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS;
    }
    if (hostBufferInfo->hostBufferSize < SystemBufferSize) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_SIZE;
    }

    const auto handle = reinterpret_cast<OrbisNgs2Handle>(hostBufferInfo->hostBuffer);
    if ((handle & (ContextBufferAlignment - 1)) != 0) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_ALIGN;
    }
    auto& runtime = Runtime();
    {
        std::scoped_lock lock{runtime.mutex};
        if (HandleIsInUse(runtime, handle) ||
            BackingRangeOverlaps(runtime, handle, hostBufferInfo->hostBufferSize)) {
            return ORBIS_NGS2_ERROR_BUFFER_BROKEN;
        }
        const SystemRuntimeSnapshot effective = EffectiveSystemRuntime(option);
        runtime.systems.emplace(handle, SystemRuntime{
                                            .buffer_info = *hostBufferInfo,
                                            .host_free = hostFree,
                                            .max_grain_samples = effective.max_grain_samples,
                                            .num_grain_samples = effective.num_grain_samples,
                                            .sample_rate = effective.sample_rate,
                                            .generation = runtime.next_generation++,
                                        });
    }
    *outHandle = handle;
    return ORBIS_OK;
}

s32 SystemCreateWithAllocator(const OrbisNgs2SystemOption* option,
                              const OrbisNgs2BufferAllocator* allocator,
                              OrbisNgs2Handle* outHandle) {
    if (!allocator || !allocator->allocHandler) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_ALLOCATOR;
    }
    if (!outHandle) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }

    const OrbisNgs2BufferAllocator allocator_copy = *allocator;
    OrbisNgs2SystemOption option_copy{};
    const OrbisNgs2SystemOption* stable_option = nullptr;
    if (option) {
        option_copy = *option;
        stable_option = &option_copy;
    }

    OrbisNgs2ContextBufferInfo buffer_info{};
    s32 result = SystemSetup(stable_option, &buffer_info, nullptr, nullptr);
    if (result < 0) {
        return result;
    }

    buffer_info.userData = allocator_copy.userData;
    result = allocator_copy.allocHandler(&buffer_info);
    if (result < 0) {
        return result;
    }

    result = SystemSetup(stable_option, &buffer_info, allocator_copy.freeHandler, outHandle);
    if (result < 0 && allocator_copy.freeHandler) {
        allocator_copy.freeHandler(&buffer_info);
    }
    return result;
}

s32 SystemCleanup(OrbisNgs2Handle systemHandle, OrbisNgs2ContextBufferInfo* outInfo) {
    OrbisNgs2ContextBufferInfo system_buffer_info{};
    OrbisNgs2BufferFreeHandler system_free = nullptr;
    std::vector<std::pair<OrbisNgs2BufferFreeHandler, OrbisNgs2ContextBufferInfo>> rack_frees;
    std::vector<AudioPipelineTrace::RackLifecycleMetadata> rack_lifecycle_traces;

    auto& runtime = Runtime();
    {
        std::scoped_lock lock{runtime.mutex};
        const auto system_it = runtime.systems.find(systemHandle);
        if (system_it == runtime.systems.end()) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }

        system_buffer_info = system_it->second.buffer_info;
        system_free = system_it->second.host_free;
        for (auto rack_it = runtime.racks.begin(); rack_it != runtime.racks.end();) {
            if (rack_it->second.system_handle != systemHandle) {
                ++rack_it;
                continue;
            }
            for (const auto voice_handle : rack_it->second.voices) {
                const auto voice_it = runtime.voices.find(voice_handle);
                if (voice_it != runtime.voices.end()) {
                    AddTombstoneLocked(
                        runtime, {
                                     .identity = voice_handle,
                                     .kind = AudioPipelineTrace::DestinationTombstoneKind::Voice,
                                     .generation = voice_it->second.generation,
                                     .was_returned = voice_it->second.was_returned,
                                 });
                    runtime.voices.erase(voice_it);
                }
            }
            AddTombstoneLocked(runtime,
                               {
                                   .identity = rack_it->first,
                                   .kind = AudioPipelineTrace::DestinationTombstoneKind::Rack,
                                   .generation = rack_it->second.generation,
                               });
            rack_lifecycle_traces.push_back({
                .rack_id = rack_it->second.rack_id,
                .phase = AudioPipelineTrace::RackLifecyclePhase::Destroyed,
                .generation = rack_it->second.generation,
                .voice_count = static_cast<u32>(rack_it->second.voices.size()),
            });
            if (rack_it->second.host_free) {
                rack_frees.emplace_back(rack_it->second.host_free, rack_it->second.buffer_info);
            }
            rack_it = runtime.racks.erase(rack_it);
        }
        AddTombstoneLocked(runtime,
                           {
                               .identity = systemHandle,
                               .kind = AudioPipelineTrace::DestinationTombstoneKind::System,
                               .generation = system_it->second.generation,
                           });
        runtime.systems.erase(system_it);
    }

    for (const auto& metadata : rack_lifecycle_traces) {
        AudioPipelineTrace::RecordRackLifecycle(metadata);
    }

    if (outInfo) {
        *outInfo = system_buffer_info;
    }
    s32 result = ORBIS_OK;
    for (auto& [free_handler, buffer_info] : rack_frees) {
        const s32 free_result = free_handler(&buffer_info);
        if (result >= 0 && free_result < 0) {
            result = free_result;
        }
    }
    if (system_free) {
        const s32 free_result = system_free(&system_buffer_info);
        if (result >= 0 && free_result < 0) {
            result = free_result;
        }
    }
    return result;
}

s32 RackQueryBufferSize(u32 rackId, const OrbisNgs2RackOption* option,
                        OrbisNgs2ContextBufferInfo* outBufferInfo) {
    if (!outBufferInfo) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS;
    }
    if (rackId == 0) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_ID;
    }

    NormalizedRackOption normalized;
    const s32 normalize_result = NormalizeRackOption(rackId, option, false, &normalized);
    if (normalize_result < 0) {
        return normalize_result;
    }

    size_t required_buffer_size = 0;
    if (!RequiredRackBufferSize(normalized.base.maxVoices, &required_buffer_size)) {
        return ORBIS_NGS2_ERROR_INVALID_MAX_VOICES;
    }

    outBufferInfo->hostBuffer = nullptr;
    outBufferInfo->hostBufferSize = required_buffer_size;
    MemoryClear(outBufferInfo->reserved, sizeof(outBufferInfo->reserved));
    return ORBIS_OK;
}

s32 RackQueryBufferSize(const OrbisNgs2RackOption* option,
                        OrbisNgs2ContextBufferInfo* outBufferInfo) {
    return RackQueryBufferSize(1, option, outBufferInfo);
}

s32 RackCreateWithExpectedSystemGeneration(OrbisNgs2Handle systemHandle,
                                           u64 expectedSystemGeneration, u32 rackId,
                                           const NormalizedRackOption& option,
                                           const OrbisNgs2ContextBufferInfo* bufferInfo,
                                           OrbisNgs2BufferFreeHandler hostFree,
                                           OrbisNgs2Handle* outHandle) {
    if (!outHandle) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    if (rackId == 0) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_ID;
    }
    if (!bufferInfo) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_INFO;
    }
    if (!bufferInfo->hostBuffer) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS;
    }

    size_t required_buffer_size = 0;
    if (!RequiredRackBufferSize(option.base.maxVoices, &required_buffer_size)) {
        return ORBIS_NGS2_ERROR_INVALID_MAX_VOICES;
    }
    if (bufferInfo->hostBufferSize < required_buffer_size) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_SIZE;
    }

    const auto rack_handle = reinterpret_cast<OrbisNgs2Handle>(bufferInfo->hostBuffer);
    if ((rack_handle & (ContextBufferAlignment - 1)) != 0) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_ALIGN;
    }
    if (AddWouldOverflow(rack_handle, required_buffer_size)) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS;
    }

    const AudioPipelineTrace::RackCreateMetadata trace_metadata{
        .rack_id = rackId,
        .max_voices = option.base.maxVoices,
        .max_input_delay_blocks = option.base.maxInputDelayBlocks,
        .max_matrices = option.base.maxMatrices,
        .max_ports = option.base.maxPorts,
        .succeeded = true,
    };
    AudioPipelineTrace::RackLifecycleMetadata lifecycle_metadata{};

    std::vector<OrbisNgs2Handle> voice_handles;
    voice_handles.reserve(option.base.maxVoices);
    for (u32 index = 0; index < option.base.maxVoices; ++index) {
        voice_handles.push_back(rack_handle + RackBufferHeaderSize +
                                static_cast<size_t>(index) * VoiceTokenStride);
    }

    auto& runtime = Runtime();
    {
        std::scoped_lock lock{runtime.mutex};
        const auto system_it = runtime.systems.find(systemHandle);
        if (system_it == runtime.systems.end() ||
            (expectedSystemGeneration != 0 &&
             system_it->second.generation != expectedSystemGeneration)) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }
        if (HandleIsInUse(runtime, rack_handle) ||
            BackingRangeOverlaps(runtime, rack_handle, bufferInfo->hostBufferSize)) {
            return ORBIS_NGS2_ERROR_BUFFER_BROKEN;
        }
        if (std::ranges::any_of(voice_handles, [&](OrbisNgs2Handle voice_handle) {
                return voice_handle == 0 || HandleIsInUse(runtime, voice_handle);
            })) {
            return ORBIS_NGS2_ERROR_BUFFER_BROKEN;
        }

        RackRuntime rack;
        rack.system_handle = systemHandle;
        rack.buffer_info = *bufferInfo;
        rack.host_free = hostFree;
        rack.rack_id = rackId;
        rack.max_inputs = option.max_inputs;
        rack.max_matrices = option.max_matrices;
        rack.max_ports = option.max_ports;
        rack.voices = voice_handles;
        rack.generation = runtime.next_generation++;
        const auto* previous = FindTombstoneLocked(
            runtime, rack_handle, AudioPipelineTrace::DestinationTombstoneKind::Rack);
        lifecycle_metadata = {
            .rack_id = rackId,
            .phase = AudioPipelineTrace::RackLifecyclePhase::Created,
            .generation = rack.generation,
            .voice_count = static_cast<u32>(voice_handles.size()),
            .reused_identity = previous != nullptr,
            .previous_generation = previous ? previous->generation : 0,
        };
        runtime.racks.emplace(rack_handle, std::move(rack));
        for (u32 index = 0; index < voice_handles.size(); ++index) {
            runtime.voices.emplace(voice_handles[index],
                                   VoiceRuntime{
                                       .rack_handle = rack_handle,
                                       .voice_index = index,
                                       .state = VoicePlayState::Empty,
                                       .generation = runtime.next_generation++,
                                   });
        }
    }

    *outHandle = rack_handle;
    AudioPipelineTrace::RecordRackCreate(trace_metadata);
    AudioPipelineTrace::RecordRackLifecycle(lifecycle_metadata);
    return ORBIS_OK;
}

s32 RackCreate(OrbisNgs2Handle systemHandle, u32 rackId, const OrbisNgs2RackOption* option,
               const OrbisNgs2ContextBufferInfo* bufferInfo, OrbisNgs2BufferFreeHandler hostFree,
               OrbisNgs2Handle* outHandle) {
    if (!outHandle) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    if (rackId == 0) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_ID;
    }
    NormalizedRackOption normalized;
    const s32 normalize_result = NormalizeRackOption(rackId, option, true, &normalized);
    if (normalize_result < 0) {
        return normalize_result;
    }
    return RackCreateWithExpectedSystemGeneration(systemHandle, 0, rackId, normalized, bufferInfo,
                                                  hostFree, outHandle);
}

s32 RackCreateWithAllocator(OrbisNgs2Handle systemHandle, u32 rackId,
                            const OrbisNgs2RackOption* option,
                            const OrbisNgs2BufferAllocator* allocator, OrbisNgs2Handle* outHandle) {
    if (!allocator || !allocator->allocHandler) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_ALLOCATOR;
    }
    if (!outHandle) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    if (!option) {
        return ORBIS_NGS2_ERROR_INVALID_OPTION_ADDRESS;
    }

    if (rackId == 0) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_ID;
    }
    const OrbisNgs2BufferAllocator allocator_copy = *allocator;
    NormalizedRackOption normalized;
    const s32 normalize_result = NormalizeRackOption(rackId, option, true, &normalized);
    if (normalize_result < 0) {
        return normalize_result;
    }
    u64 system_generation = 0;
    {
        auto& runtime = Runtime();
        std::scoped_lock lock{runtime.mutex};
        const auto system_it = runtime.systems.find(systemHandle);
        if (system_it == runtime.systems.end()) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }
        system_generation = system_it->second.generation;
    }

    OrbisNgs2ContextBufferInfo buffer_info{};
    size_t required_buffer_size = 0;
    if (!RequiredRackBufferSize(normalized.base.maxVoices, &required_buffer_size)) {
        return ORBIS_NGS2_ERROR_INVALID_MAX_VOICES;
    }
    buffer_info.hostBufferSize = required_buffer_size;
    buffer_info.userData = allocator_copy.userData;
    s32 result = allocator_copy.allocHandler(&buffer_info);
    if (result < 0) {
        return result;
    }
    result =
        RackCreateWithExpectedSystemGeneration(systemHandle, system_generation, rackId, normalized,
                                               &buffer_info, allocator_copy.freeHandler, outHandle);
    if (result < 0 && allocator_copy.freeHandler) {
        allocator_copy.freeHandler(&buffer_info);
    }
    return result;
}

s32 RackDestroy(OrbisNgs2Handle rackHandle, OrbisNgs2ContextBufferInfo* outBufferInfo) {
    OrbisNgs2ContextBufferInfo buffer_info{};
    OrbisNgs2BufferFreeHandler host_free = nullptr;
    AudioPipelineTrace::RackLifecycleMetadata lifecycle_metadata{};

    auto& runtime = Runtime();
    {
        std::scoped_lock lock{runtime.mutex};
        const auto rack_it = runtime.racks.find(rackHandle);
        if (rack_it == runtime.racks.end()) {
            return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
        }
        buffer_info = rack_it->second.buffer_info;
        host_free = rack_it->second.host_free;
        std::unordered_set<OrbisNgs2Handle> destroyed_endpoints;
        destroyed_endpoints.reserve(rack_it->second.voices.size() + 1);
        destroyed_endpoints.insert(rackHandle);
        destroyed_endpoints.insert(rack_it->second.voices.begin(), rack_it->second.voices.end());
        for (auto& source_entry : runtime.voices) {
            if (destroyed_endpoints.contains(source_entry.first)) {
                continue;
            }
            bool route_invalidated = false;
            for (auto& patch_entry : source_entry.second.patches) {
                auto& patch = patch_entry.second;
                if (patch.connected && destroyed_endpoints.contains(patch.destination_handle)) {
                    patch.connected = false;
                    route_invalidated = true;
                }
            }
            if (route_invalidated) {
                AdvanceRenderStateRevisionLocked(source_entry.second);
            }
        }
        for (const auto voice_handle : rack_it->second.voices) {
            const auto voice_it = runtime.voices.find(voice_handle);
            if (voice_it != runtime.voices.end()) {
                AddTombstoneLocked(runtime,
                                   {
                                       .identity = voice_handle,
                                       .kind = AudioPipelineTrace::DestinationTombstoneKind::Voice,
                                       .generation = voice_it->second.generation,
                                       .was_returned = voice_it->second.was_returned,
                                   });
                runtime.voices.erase(voice_it);
            }
        }
        AddTombstoneLocked(runtime, {
                                        .identity = rackHandle,
                                        .kind = AudioPipelineTrace::DestinationTombstoneKind::Rack,
                                        .generation = rack_it->second.generation,
                                    });
        lifecycle_metadata = {
            .rack_id = rack_it->second.rack_id,
            .phase = AudioPipelineTrace::RackLifecyclePhase::Destroyed,
            .generation = rack_it->second.generation,
            .voice_count = static_cast<u32>(rack_it->second.voices.size()),
        };
        runtime.racks.erase(rack_it);
    }

    AudioPipelineTrace::RecordRackLifecycle(lifecycle_metadata);

    if (outBufferInfo) {
        *outBufferInfo = buffer_info;
    }
    return host_free ? host_free(&buffer_info) : ORBIS_OK;
}

s32 RackGetVoiceHandle(OrbisNgs2Handle rackHandle, u32 voiceIndex, OrbisNgs2Handle* outHandle) {
    if (!outHandle) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }

    auto& runtime = Runtime();
    std::scoped_lock lock{runtime.mutex};
    const auto rack_it = runtime.racks.find(rackHandle);
    if (rack_it == runtime.racks.end()) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
    }
    if (voiceIndex >= rack_it->second.voices.size()) {
        return ORBIS_NGS2_ERROR_INVALID_VOICE_INDEX;
    }
    *outHandle = rack_it->second.voices[voiceIndex];
    if (const auto voice_it = runtime.voices.find(*outHandle); voice_it != runtime.voices.end()) {
        voice_it->second.was_returned = true;
    }
    return ORBIS_OK;
}

s32 VoiceControl(OrbisNgs2Handle voiceHandle, const OrbisNgs2VoiceParamHeader* paramList,
                 OrbisNgs2RuntimeReadHandler readHandler, void* readContext) {
    if (!paramList || !readHandler) {
        return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS;
    }

    u64 voice_generation = 0;
    VoiceTraceIdentity source_identity{};
    VoiceControlTarget control_target{};
    auto& runtime = Runtime();
    {
        std::scoped_lock lock{runtime.mutex};
        const auto voice_it = runtime.voices.find(voiceHandle);
        if (voice_it == runtime.voices.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        voice_generation = voice_it->second.generation;
        const auto rack_it = runtime.racks.find(voice_it->second.rack_handle);
        if (rack_it == runtime.racks.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        source_identity = {
            .rack_id = rack_it->second.rack_id,
            .voice_index = voice_it->second.voice_index,
            .resolved = true,
            .system_handle = rack_it->second.system_handle,
            .generation = voice_it->second.generation,
        };
        control_target = {
            .rack_id = rack_it->second.rack_id,
            .max_matrices = rack_it->second.max_matrices,
            .max_ports = rack_it->second.max_ports,
        };
    }

    std::vector<u32> parsed_events;
    std::vector<PendingPatch> parsed_patches;
    std::vector<PendingControlMutation> parsed_control_mutations;
    std::vector<AudioPipelineTrace::VoiceControlMetadata> pending_voice_traces;
    std::vector<PendingTraceRecord> pending_command_traces;
    const bool trace_enabled = AudioPipelineTrace::IsEnabled();
    std::unordered_set<uintptr_t> visited;
    uintptr_t address = reinterpret_cast<uintptr_t>(paramList);
    size_t total_bytes = 0;
    u32 remaining_persisted_waveform_blocks = MaxPersistedWaveformBlocks;
    size_t remaining_persisted_matrix_levels = MaxPersistedMatrixLevels;

    for (size_t command_count = 0;; ++command_count) {
        if (command_count >= MaxVoiceControlCommands) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        if (!visited.insert(address).second) {
            return ORBIS_NGS2_ERROR_DETECTED_CIRCULAR_VOICE_CONTROL;
        }

        OrbisNgs2VoiceParamHeader initial_header{};
        if (!readHandler(address, &initial_header, sizeof(initial_header), readContext)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS;
        }
        if (initial_header.size < sizeof(OrbisNgs2VoiceParamHeader)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        if (initial_header.size > MaxVoiceControlBytes - total_bytes) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }

        std::vector<u8> command(initial_header.size);
        if (!readHandler(address, command.data(), command.size(), readContext)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS;
        }
        OrbisNgs2VoiceParamHeader header{};
        std::memcpy(&header, command.data(), sizeof(header));
        if (header.size != initial_header.size) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        total_bytes += command.size();

        const bool has_event = header.id == VoiceEventControlId;
        u32 event_id = 0;
        if (has_event) {
            if (header.size != sizeof(OrbisNgs2VoiceEventParam)) {
                return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
            }
            OrbisNgs2VoiceEventParam event{};
            std::memcpy(&event, command.data(), sizeof(event));
            event_id = event.eventId;
            if (event.eventId > 5) {
                if (trace_enabled) {
                    AudioPipelineTrace::RecordVoiceControl({
                        .control_id = header.id,
                        .command_size = header.size,
                        .next_offset = header.next,
                        .has_event = true,
                        .event_id = event.eventId,
                        .accepted = false,
                        .source_rack_id = source_identity.rack_id,
                        .source_voice_index = source_identity.voice_index,
                        .source_resolved = source_identity.resolved,
                    });
                    if (source_identity.rack_id == 0x1000) {
                        AudioPipelineTrace::RecordSamplerEvent({source_identity.rack_id,
                                                                source_identity.voice_index,
                                                                event.eventId, false});
                    }
                }
                return ORBIS_NGS2_ERROR_INVALID_EVENT_TYPE;
            }
            parsed_events.push_back(event.eventId);
        }

        if (header.id == VoicePatchControlId) {
            if (command.size() != sizeof(OrbisNgs2VoicePatchParam)) {
                return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
            }
            OrbisNgs2VoicePatchParam patch{};
            std::memcpy(&patch, command.data(), sizeof(patch));
            parsed_patches.push_back({patch.port, patch.destInputId, patch.destHandle});
        }

        const size_t mutation_count_before = parsed_control_mutations.size();
        if (const s32 parse_result = ParsePersistentVoiceCommand(
                header, command, readHandler, readContext, control_target,
                remaining_persisted_waveform_blocks, remaining_persisted_matrix_levels,
                parsed_control_mutations);
            parse_result < 0) {
            return parse_result;
        }
        const PendingControlMutation* persistent_mutation =
            parsed_control_mutations.size() != mutation_count_before
                ? &parsed_control_mutations.back()
                : nullptr;

        if (trace_enabled) {
            pending_voice_traces.push_back({
                .control_id = header.id,
                .command_size = header.size,
                .next_offset = header.next,
                .has_event = has_event,
                .event_id = event_id,
                .accepted = true,
                .source_rack_id = source_identity.rack_id,
                .source_voice_index = source_identity.voice_index,
                .source_resolved = source_identity.resolved,
            });
            TraceVoiceCommand(source_identity, header, command, persistent_mutation,
                              pending_command_traces);
        }

        if (header.next == 0) {
            break;
        }
        if (!AdvanceAddress(address, header.next, &address)) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS;
        }
    }

    std::vector<AudioPipelineTrace::VoicePatchMetadata> committed_patch_traces;
    std::vector<AudioPipelineTrace::VoicePatchStateMetadata> committed_patch_state_traces;
    {
        std::scoped_lock lock{runtime.mutex};
        const auto voice_it = runtime.voices.find(voiceHandle);
        if (voice_it == runtime.voices.end() || voice_it->second.generation != voice_generation) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        if (parsed_events.size() >
            MaxVoiceControlCommands - voice_it->second.pending_events.size()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE;
        }
        const auto source_rack_it = runtime.racks.find(voice_it->second.rack_handle);
        if (source_rack_it == runtime.racks.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }

        for (const PendingControlMutation& mutation : parsed_control_mutations) {
            if ((std::holds_alternative<PendingSamplerSetup>(mutation) ||
                 std::holds_alternative<PendingWaveformBlocks>(mutation)) &&
                source_rack_it->second.rack_id != 0x1000) {
                return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
            }
            if (const auto* volume = std::get_if<PendingPortVolume>(&mutation);
                volume && volume->port >= source_rack_it->second.max_ports) {
                return ORBIS_NGS2_ERROR_INVALID_PORT_INDEX;
            }
            if (const auto* levels = std::get_if<PendingMatrixLevels>(&mutation);
                levels && levels->matrix_id >= source_rack_it->second.max_matrices) {
                return ORBIS_NGS2_ERROR_INVALID_MATRIX_INDEX;
            }
            if (const auto* matrix = std::get_if<PendingPortMatrix>(&mutation);
                matrix && matrix->port >= source_rack_it->second.max_ports) {
                return ORBIS_NGS2_ERROR_INVALID_PORT_INDEX;
            }
            if (const auto* matrix = std::get_if<PendingPortMatrix>(&mutation);
                matrix && (matrix->matrix_id < -1 ||
                           (matrix->matrix_id >= 0 && static_cast<u32>(matrix->matrix_id) >=
                                                          source_rack_it->second.max_matrices))) {
                return ORBIS_NGS2_ERROR_INVALID_MATRIX_INDEX;
            }
            if (std::holds_alternative<PendingSubmixerSetup>(mutation) &&
                source_rack_it->second.rack_id != 0x2000) {
                return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
            }
            if (const auto* user_fx = std::get_if<PendingUserFx>(&mutation);
                user_fx && source_rack_it->second.rack_id != user_fx->rack_id) {
                return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
            }
            if ((std::holds_alternative<PendingMasteringSetup>(mutation) ||
                 std::holds_alternative<PendingMasteringMatrix>(mutation) ||
                 std::holds_alternative<PendingMasteringGain>(mutation) ||
                 std::holds_alternative<PendingMasteringOutput>(mutation)) &&
                source_rack_it->second.rack_id != 0x3000) {
                return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ID;
            }
        }

        std::vector<PatchResolution> patch_resolutions;
        patch_resolutions.reserve(parsed_patches.size());
        for (const PendingPatch& patch : parsed_patches) {
            if (patch.source_port >= source_rack_it->second.max_ports) {
                return ORBIS_NGS2_ERROR_INVALID_PORT_INDEX;
            }
            PatchResolution resolution = ResolvePatchLocked(
                runtime, source_identity, patch.destination_handle, patch.destination_input_id);
            if (const s32 validation_result = PatchValidationResult(resolution);
                validation_result < 0) {
                return validation_result;
            }
            patch_resolutions.push_back(std::move(resolution));
        }

        voice_it->second.pending_events.insert(voice_it->second.pending_events.end(),
                                               parsed_events.begin(), parsed_events.end());
        bool sampler_source_changed = false;
        for (PendingControlMutation& mutation : parsed_control_mutations) {
            if (auto* setup = std::get_if<PendingSamplerSetup>(&mutation)) {
                sampler_source_changed = true;
                voice_it->second.sampler.format = setup->format;
                voice_it->second.sampler.setup_flags = setup->flags;
            } else if (auto* waveform = std::get_if<PendingWaveformBlocks>(&mutation)) {
                sampler_source_changed = true;
                voice_it->second.sampler.waveform_configured = true;
                voice_it->second.sampler.guest_data_token = waveform->guest_data_token;
                voice_it->second.sampler.waveform_flags = waveform->flags;
                voice_it->second.sampler.waveform_blocks = std::move(waveform->blocks);
            } else if (auto* matrix = std::get_if<PendingMatrixLevels>(&mutation)) {
                voice_it->second.matrices[matrix->matrix_id] = std::move(matrix->levels);
            } else if (const auto* volume = std::get_if<PendingPortVolume>(&mutation)) {
                auto& port = voice_it->second.ports[volume->port];
                port.port = volume->port;
                port.volume_configured = true;
                port.volume = volume->level;
            } else if (const auto* matrix = std::get_if<PendingPortMatrix>(&mutation)) {
                auto& port = voice_it->second.ports[matrix->port];
                port.port = matrix->port;
                port.matrix_configured = true;
                port.matrix_id = matrix->matrix_id;
            } else if (const auto* setup = std::get_if<PendingSubmixerSetup>(&mutation)) {
                voice_it->second.submixer = {
                    .setup_configured = true,
                    .num_io_channels = setup->num_io_channels,
                    .flags = setup->flags,
                    .user_fx_configured = voice_it->second.submixer.user_fx_configured,
                    .user_fx_enabled = voice_it->second.submixer.user_fx_enabled,
                };
            } else if (const auto* user_fx = std::get_if<PendingUserFx>(&mutation)) {
                if (user_fx->rack_id == 0x1000) {
                    voice_it->second.sampler.user_fx_configured = true;
                    voice_it->second.sampler.user_fx_enabled = user_fx->enabled;
                } else {
                    voice_it->second.submixer.user_fx_configured = true;
                    voice_it->second.submixer.user_fx_enabled = user_fx->enabled;
                }
            } else if (const auto* setup = std::get_if<PendingMasteringSetup>(&mutation)) {
                voice_it->second.mastering.setup_configured = true;
                voice_it->second.mastering.num_input_channels = setup->num_input_channels;
                voice_it->second.mastering.setup_flags = setup->flags;
            } else if (auto* matrix = std::get_if<PendingMasteringMatrix>(&mutation)) {
                voice_it->second.mastering.matrix_configured = true;
                voice_it->second.mastering.matrix_type = matrix->type;
                voice_it->second.mastering.matrix_input_channels = matrix->input_channels;
                voice_it->second.mastering.matrix_output_channels = matrix->output_channels;
                voice_it->second.mastering.matrix_levels = std::move(matrix->levels);
            } else if (const auto* gain = std::get_if<PendingMasteringGain>(&mutation)) {
                voice_it->second.mastering.gain_configured = true;
                voice_it->second.mastering.fbw_level = gain->fbw_level;
                voice_it->second.mastering.lfe_level = gain->lfe_level;
            } else if (const auto* output = std::get_if<PendingMasteringOutput>(&mutation)) {
                voice_it->second.mastering.output_configured = true;
                voice_it->second.mastering.output_id = output->output_id;
            }
        }
        if (sampler_source_changed) {
            AdvanceSamplerSourceEpochLocked(voice_it->second);
        }
        for (size_t index = 0; index < parsed_patches.size(); ++index) {
            const PendingPatch& patch = parsed_patches[index];
            const PatchResolution& resolution = patch_resolutions[index];
            auto& stored = voice_it->second.patches[patch.source_port];
            stored = {
                .destination_input_id = patch.destination_input_id,
                .destination_handle = patch.destination_handle,
                .last_observed = true,
                .last_destination_resolved = resolution.endpoint.IsResolved(),
                .last_destination_same_system = resolution.same_system,
                .last_destination_generation = resolution.endpoint.generation,
                .last_classification = resolution.classification,
                .committed_endpoint_kind = resolution.endpoint.kind,
                .committed_destination_rack_id = resolution.endpoint.rack_id,
                .committed_destination_voice_index_valid = resolution.endpoint.voice_index_valid,
                .committed_destination_voice_index = resolution.endpoint.voice_index,
                .committed_destination_generation = resolution.endpoint.generation,
                .connected = resolution.present,
            };
            if (trace_enabled) {
                committed_patch_traces.push_back({
                    .source_rack_id = source_identity.rack_id,
                    .source_voice_index = source_identity.voice_index,
                    .source_port = patch.source_port,
                    .destination_input_id = patch.destination_input_id,
                    .destination_rack_id = resolution.endpoint.rack_id,
                    .destination_voice_index = resolution.endpoint.voice_index,
                    .destination_resolved = resolution.endpoint.IsResolved(),
                    .destination_present = resolution.present,
                    .destination_endpoint_kind = resolution.endpoint.kind,
                    .destination_voice_index_valid = resolution.endpoint.voice_index_valid,
                });
                committed_patch_state_traces.push_back(PatchStateMetadata(
                    source_identity, patch.source_port, patch.destination_input_id, resolution,
                    stored.committed_destination_generation,
                    AudioPipelineTrace::PatchResolutionPhase::Committed));
            }
        }
        if (!parsed_events.empty() || !parsed_patches.empty() ||
            !parsed_control_mutations.empty()) {
            AdvanceRenderStateRevisionLocked(voice_it->second);
        }
    }

    for (const auto& metadata : pending_voice_traces) {
        AudioPipelineTrace::RecordVoiceControl(metadata);
    }
    for (const auto& record : pending_command_traces) {
        EmitTraceRecord(record);
    }
    if (trace_enabled && source_identity.rack_id == 0x1000) {
        for (const u32 event_id : parsed_events) {
            AudioPipelineTrace::RecordSamplerEvent(
                {source_identity.rack_id, source_identity.voice_index, event_id, true});
        }
    }
    for (const auto& metadata : committed_patch_traces) {
        AudioPipelineTrace::RecordVoicePatch(metadata);
    }
    for (const auto& metadata : committed_patch_state_traces) {
        AudioPipelineTrace::RecordVoicePatchState(metadata);
    }
    return ORBIS_OK;
}

s32 VoiceGetState(OrbisNgs2Handle voiceHandle, void* outState, size_t stateSize) {
    u32 state_flags = 0;
    VoiceStateSizeRequirement state_size_requirement{};
    VoiceTraceIdentity identity{};
    VoicePlayState state = VoicePlayState::Empty;
    auto& runtime = Runtime();
    {
        std::scoped_lock lock{runtime.mutex};
        const auto voice_it = runtime.voices.find(voiceHandle);
        if (voice_it == runtime.voices.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        const auto rack_it = runtime.racks.find(voice_it->second.rack_handle);
        if (rack_it == runtime.racks.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        state_flags = StateFlags(voice_it->second.state);
        state = voice_it->second.state;
        state_size_requirement = VoiceStateSizeForRack(rack_it->second.rack_id);
        identity = {
            .rack_id = rack_it->second.rack_id,
            .voice_index = voice_it->second.voice_index,
            .resolved = true,
            .system_handle = rack_it->second.system_handle,
            .generation = voice_it->second.generation,
        };
    }
    if (!outState) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    if ((state_size_requirement.exact && stateSize != state_size_requirement.minimum_size) ||
        (!state_size_requirement.exact &&
         (stateSize < state_size_requirement.minimum_size || stateSize > MaxVoiceStateBytes))) {
        return ORBIS_NGS2_ERROR_INVALID_VOICE_STATE_SIZE;
    }

    MemoryClear(outState, stateSize);
    std::memcpy(outState, &state_flags, sizeof(state_flags));
    if (identity.rack_id == 0x1000) {
        AudioPipelineTrace::RecordSamplerVoiceState({
            .source_rack_id = identity.rack_id,
            .source_voice_index = identity.voice_index,
            .previous_state = TracePlaybackState(state),
            .current_state = TracePlaybackState(state),
            .observation = AudioPipelineTrace::SamplerStateObservation::Poll,
        });
    }
    return ORBIS_OK;
}

s32 VoiceGetStateFlags(OrbisNgs2Handle voiceHandle, u32* outStateFlags) {
    VoiceTraceIdentity identity{};
    VoicePlayState state = VoicePlayState::Empty;
    auto& runtime = Runtime();
    {
        std::scoped_lock lock{runtime.mutex};
        const auto voice_it = runtime.voices.find(voiceHandle);
        if (voice_it == runtime.voices.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        if (!outStateFlags) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        const auto rack_it = runtime.racks.find(voice_it->second.rack_handle);
        if (rack_it == runtime.racks.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        state = voice_it->second.state;
        *outStateFlags = StateFlags(state);
        identity = {
            .rack_id = rack_it->second.rack_id,
            .voice_index = voice_it->second.voice_index,
            .resolved = true,
            .system_handle = rack_it->second.system_handle,
            .generation = voice_it->second.generation,
        };
    }
    if (identity.rack_id == 0x1000) {
        AudioPipelineTrace::RecordSamplerVoiceState({
            .source_rack_id = identity.rack_id,
            .source_voice_index = identity.voice_index,
            .previous_state = TracePlaybackState(state),
            .current_state = TracePlaybackState(state),
            .observation = AudioPipelineTrace::SamplerStateObservation::Poll,
        });
    }
    return ORBIS_OK;
}

static s32 SystemRenderImpl(OrbisNgs2Handle systemHandle,
                            const OrbisNgs2RenderBufferInfo* bufferInfo, u32 numBufferInfo,
                            OrbisNgs2RuntimeReadHandler readHandler, void* readContext,
                            Atrac9FrameDecoderFactory decoderFactory,
                            SystemRenderPublishHandler publishHandler = nullptr,
                            void* publishContext = nullptr) {
    if (numBufferInfo != 0 && !bufferInfo) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_INFO;
    }
    for (u32 index = 0; index < numBufferInfo; ++index) {
        if (bufferInfo[index].bufferSize != 0 && !bufferInfo[index].buffer) {
            return ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS;
        }
    }

    const bool trace_enabled = AudioPipelineTrace::IsEnabled();
    std::vector<AudioPipelineTrace::SamplerVoiceStateMetadata> sampler_state_traces;
    std::vector<AudioPipelineTrace::VoicePatchStateMetadata> patch_state_traces;
    auto& runtime = Runtime();
    {
        std::scoped_lock lock{runtime.mutex};
        if (!runtime.systems.contains(systemHandle)) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }
        for (const auto& [rack_handle, rack] : runtime.racks) {
            if (rack.system_handle != systemHandle) {
                continue;
            }
            for (const auto voice_handle : rack.voices) {
                auto voice_it = runtime.voices.find(voice_handle);
                if (voice_it == runtime.voices.end()) {
                    continue;
                }
                const VoiceTraceIdentity source_identity{
                    .rack_id = rack.rack_id,
                    .voice_index = voice_it->second.voice_index,
                    .resolved = true,
                    .system_handle = rack.system_handle,
                    .generation = voice_it->second.generation,
                };
                const VoicePlayState previous_state = voice_it->second.state;
                bool state_transition_applied = false;
                bool decoder_reset_applied = false;
                for (const u32 event_id : voice_it->second.pending_events) {
                    const VoicePlayState before_event = voice_it->second.state;
                    state_transition_applied |= ApplyEvent(voice_it->second, event_id);
                    decoder_reset_applied |= before_event != VoicePlayState::Empty &&
                                             voice_it->second.state == VoicePlayState::Empty;
                }
                voice_it->second.pending_events.clear();
                if (state_transition_applied) {
                    AdvanceRenderStateRevisionLocked(voice_it->second);
                }
                if (decoder_reset_applied) {
                    AdvanceSamplerSourceEpochLocked(voice_it->second);
                }
                if (trace_enabled && rack.rack_id == 0x1000 &&
                    previous_state != voice_it->second.state) {
                    sampler_state_traces.push_back({
                        .source_rack_id = source_identity.rack_id,
                        .source_voice_index = source_identity.voice_index,
                        .previous_state = TracePlaybackState(previous_state),
                        .current_state = TracePlaybackState(voice_it->second.state),
                        .observation =
                            AudioPipelineTrace::SamplerStateObservation::RenderTransition,
                    });
                }
                if (!trace_enabled) {
                    continue;
                }
                for (auto& [source_port, patch] : voice_it->second.patches) {
                    const PatchResolution resolution =
                        ResolveCommittedPatchLocked(runtime, source_identity, patch);
                    const bool changed =
                        !patch.last_observed ||
                        patch.last_destination_resolved != resolution.endpoint.IsResolved() ||
                        patch.last_destination_same_system != resolution.same_system ||
                        patch.last_destination_generation != resolution.endpoint.generation ||
                        patch.last_classification != resolution.classification;
                    if (!changed) {
                        continue;
                    }
                    patch.last_observed = true;
                    patch.last_destination_resolved = resolution.endpoint.IsResolved();
                    patch.last_destination_same_system = resolution.same_system;
                    patch.last_destination_generation = resolution.endpoint.generation;
                    patch.last_classification = resolution.classification;
                    patch_state_traces.push_back(PatchStateMetadata(
                        source_identity, source_port, patch.destination_input_id, resolution,
                        patch.committed_destination_generation,
                        AudioPipelineTrace::PatchResolutionPhase::RenderTransition));
                }
            }
        }
    }

    for (const auto& metadata : sampler_state_traces) {
        AudioPipelineTrace::RecordSamplerVoiceState(metadata);
    }
    for (const auto& metadata : patch_state_traces) {
        AudioPipelineTrace::RecordVoicePatchState(metadata);
    }

    return RenderSystemAudio(systemHandle, bufferInfo, numBufferInfo, readHandler, readContext,
                             std::move(decoderFactory), publishHandler, publishContext);
}

s32 SystemRender(OrbisNgs2Handle systemHandle, const OrbisNgs2RenderBufferInfo* bufferInfo,
                 u32 numBufferInfo) {
    return SystemRenderImpl(systemHandle, bufferInfo, numBufferInfo, nullptr, nullptr,
                            LibAtrac9FrameDecoderFactory());
}

s32 SystemRender(OrbisNgs2Handle systemHandle, const OrbisNgs2RenderBufferInfo* bufferInfo,
                 u32 numBufferInfo, OrbisNgs2RuntimeReadHandler readHandler, void* readContext,
                 SystemRenderPublishHandler publishHandler, void* publishContext) {
    return SystemRenderImpl(systemHandle, bufferInfo, numBufferInfo, readHandler, readContext,
                            LibAtrac9FrameDecoderFactory(), publishHandler, publishContext);
}

s32 SystemRenderWithFactoryForTesting(OrbisNgs2Handle systemHandle,
                                      const OrbisNgs2RenderBufferInfo* bufferInfo,
                                      u32 numBufferInfo, OrbisNgs2RuntimeReadHandler readHandler,
                                      void* readContext, Atrac9FrameDecoderFactory decoderFactory,
                                      SystemRenderPublishHandler publishHandler,
                                      void* publishContext) {
    return SystemRenderImpl(systemHandle, bufferInfo, numBufferInfo, readHandler, readContext,
                            std::move(decoderFactory), publishHandler, publishContext);
}

std::optional<SystemRuntimeSnapshot> SnapshotSystemRuntime(OrbisNgs2Handle systemHandle) {
    auto& runtime = Runtime();
    std::scoped_lock lock{runtime.mutex};
    const auto system_it = runtime.systems.find(systemHandle);
    if (system_it == runtime.systems.end()) {
        return std::nullopt;
    }
    const SystemRuntime& system = system_it->second;
    return SystemRuntimeSnapshot{
        .max_grain_samples = system.max_grain_samples,
        .num_grain_samples = system.num_grain_samples,
        .sample_rate = system.sample_rate,
        .generation = system.generation,
    };
}

std::optional<SystemRuntimeSnapshot> SnapshotSystemRuntime(OrbisNgs2Handle systemHandle,
                                                           u64 expectedGeneration) {
    auto& runtime = Runtime();
    std::scoped_lock lock{runtime.mutex};
    const auto system_it = runtime.systems.find(systemHandle);
    if (system_it == runtime.systems.end() || system_it->second.generation != expectedGeneration) {
        return std::nullopt;
    }
    const SystemRuntime& system = system_it->second;
    return SystemRuntimeSnapshot{
        .max_grain_samples = system.max_grain_samples,
        .num_grain_samples = system.num_grain_samples,
        .sample_rate = system.sample_rate,
        .generation = system.generation,
    };
}

static VoiceControlSnapshot BuildVoiceControlSnapshotLocked(const SystemRuntime& system,
                                                            const RackRuntime& rack,
                                                            const VoiceRuntime& voice) {
    VoiceControlSnapshot snapshot{
        .rack_id = rack.rack_id,
        .voice_index = voice.voice_index,
        .system_generation = system.generation,
        .rack_generation = rack.generation,
        .voice_generation = voice.generation,
        .render_state_revision = voice.render_state_revision,
        .max_inputs = rack.max_inputs,
        .max_matrices = rack.max_matrices,
        .max_ports = rack.max_ports,
    };
    if (rack.rack_id == 0x1000) {
        snapshot.sampler = voice.sampler;
    } else if (rack.rack_id == 0x2000) {
        snapshot.submixer = voice.submixer;
    } else if (rack.rack_id == 0x3000) {
        snapshot.mastering = voice.mastering;
    }

    snapshot.matrices.reserve(voice.matrices.size());
    for (const auto& [matrix_id, levels] : voice.matrices) {
        snapshot.matrices.push_back({matrix_id, levels});
    }
    std::ranges::sort(snapshot.matrices, {}, &VoiceMatrixSnapshot::matrix_id);

    snapshot.ports.reserve(voice.ports.size());
    for (const auto& [port_id, port] : voice.ports) {
        snapshot.ports.push_back(port);
    }
    std::ranges::sort(snapshot.ports, {}, &VoicePortSnapshot::port);

    snapshot.routes.reserve(voice.patches.size());
    for (const auto& [source_port, patch] : voice.patches) {
        VoiceRouteEndpointKind endpoint_kind = VoiceRouteEndpointKind::None;
        if (patch.committed_endpoint_kind == AudioPipelineTrace::PatchEndpointKind::Rack) {
            endpoint_kind = VoiceRouteEndpointKind::Rack;
        } else if (patch.committed_endpoint_kind == AudioPipelineTrace::PatchEndpointKind::Voice) {
            endpoint_kind = VoiceRouteEndpointKind::Voice;
        }
        snapshot.routes.push_back({
            .source_port = source_port,
            .destination_input_id = patch.destination_input_id,
            .endpoint_kind = endpoint_kind,
            .destination_rack_id = patch.committed_destination_rack_id,
            .destination_voice_index_valid = patch.committed_destination_voice_index_valid,
            .destination_voice_index = patch.committed_destination_voice_index,
            .destination_generation = patch.committed_destination_generation,
            .connected = patch.connected,
        });
    }
    std::ranges::sort(snapshot.routes, {}, &VoiceRouteSnapshot::source_port);
    return snapshot;
}

std::optional<VoiceControlSnapshot> SnapshotVoiceControl(OrbisNgs2Handle voiceHandle) {
    auto& runtime = Runtime();
    std::scoped_lock lock{runtime.mutex};
    const auto voice_it = runtime.voices.find(voiceHandle);
    if (voice_it == runtime.voices.end()) {
        return std::nullopt;
    }
    const auto rack_it = runtime.racks.find(voice_it->second.rack_handle);
    if (rack_it == runtime.racks.end()) {
        return std::nullopt;
    }
    const auto system_it = runtime.systems.find(rack_it->second.system_handle);
    if (system_it == runtime.systems.end()) {
        return std::nullopt;
    }
    return BuildVoiceControlSnapshotLocked(system_it->second, rack_it->second, voice_it->second);
}

std::optional<AppliedSamplerVoiceSnapshot> SnapshotAppliedSamplerVoice(
    OrbisNgs2Handle voiceHandle) {
    auto& runtime = Runtime();
    std::scoped_lock lock{runtime.mutex};
    const auto voice_it = runtime.voices.find(voiceHandle);
    if (voice_it == runtime.voices.end()) {
        return std::nullopt;
    }
    const auto rack_it = runtime.racks.find(voice_it->second.rack_handle);
    if (rack_it == runtime.racks.end() || rack_it->second.rack_id != 0x1000) {
        return std::nullopt;
    }
    const auto system_it = runtime.systems.find(rack_it->second.system_handle);
    if (system_it == runtime.systems.end()) {
        return std::nullopt;
    }
    return AppliedSamplerVoiceSnapshot{
        .token =
            {
                .system_handle = rack_it->second.system_handle,
                .rack_handle = voice_it->second.rack_handle,
                .voice_handle = voiceHandle,
                .rack_id = rack_it->second.rack_id,
                .voice_index = voice_it->second.voice_index,
                .system_generation = system_it->second.generation,
                .rack_generation = rack_it->second.generation,
                .voice_generation = voice_it->second.generation,
                .render_state_revision = voice_it->second.render_state_revision,
            },
        .playback_state = RenderPlaybackState(voice_it->second.state),
        .max_grain_samples = system_it->second.max_grain_samples,
        .num_grain_samples = system_it->second.num_grain_samples,
        .system_sample_rate = system_it->second.sample_rate,
        .sampler_source_epoch = voice_it->second.sampler_source_epoch,
        .sampler = voice_it->second.sampler,
        .decode_runtime = voice_it->second.decode_runtime,
    };
}

std::optional<SystemRenderGraphSnapshot> SnapshotSystemRenderGraph(OrbisNgs2Handle systemHandle) {
    auto& runtime = Runtime();
    std::scoped_lock lock{runtime.mutex};
    const auto system_it = runtime.systems.find(systemHandle);
    if (system_it == runtime.systems.end()) {
        return std::nullopt;
    }
    SystemRenderGraphSnapshot snapshot{
        .system_handle = systemHandle,
        .system =
            {
                .max_grain_samples = system_it->second.max_grain_samples,
                .num_grain_samples = system_it->second.num_grain_samples,
                .sample_rate = system_it->second.sample_rate,
                .generation = system_it->second.generation,
            },
    };
    for (const auto& [rack_handle, rack] : runtime.racks) {
        if (rack.system_handle != systemHandle) {
            continue;
        }
        for (const OrbisNgs2Handle voice_handle : rack.voices) {
            const auto voice_it = runtime.voices.find(voice_handle);
            if (voice_it == runtime.voices.end()) {
                continue;
            }
            const VoiceRuntime& voice = voice_it->second;
            snapshot.voices.push_back({
                .token =
                    {
                        .system_handle = systemHandle,
                        .rack_handle = rack_handle,
                        .voice_handle = voice_handle,
                        .rack_id = rack.rack_id,
                        .voice_index = voice.voice_index,
                        .system_generation = system_it->second.generation,
                        .rack_generation = rack.generation,
                        .voice_generation = voice.generation,
                        .render_state_revision = voice.render_state_revision,
                    },
                .playback_state = RenderPlaybackState(voice.state),
                .sampler_source_epoch = voice.sampler_source_epoch,
                .control = BuildVoiceControlSnapshotLocked(system_it->second, rack, voice),
                .decode_runtime = rack.rack_id == 0x1000 ? voice.decode_runtime : nullptr,
            });
        }
    }
    std::ranges::sort(snapshot.voices, [](const auto& left, const auto& right) {
        return std::tie(left.token.rack_id, left.token.rack_generation, left.token.voice_index) <
               std::tie(right.token.rack_id, right.token.rack_generation, right.token.voice_index);
    });
    return snapshot;
}

static bool MatchesRenderTokenLocked(const RuntimeRegistry& runtime,
                                     const VoiceRenderCommitToken& token,
                                     const std::shared_ptr<VoiceDecodeRuntime>* decodeRuntime) {
    const auto system_it = runtime.systems.find(token.system_handle);
    const auto rack_it = runtime.racks.find(token.rack_handle);
    const auto voice_it = runtime.voices.find(token.voice_handle);
    return system_it != runtime.systems.end() && rack_it != runtime.racks.end() &&
           voice_it != runtime.voices.end() &&
           system_it->second.generation == token.system_generation &&
           rack_it->second.generation == token.rack_generation &&
           rack_it->second.rack_id == token.rack_id &&
           rack_it->second.system_handle == token.system_handle &&
           token.voice_index < rack_it->second.voices.size() &&
           rack_it->second.voices[token.voice_index] == token.voice_handle &&
           voice_it->second.generation == token.voice_generation &&
           voice_it->second.render_state_revision == token.render_state_revision &&
           voice_it->second.rack_handle == token.rack_handle &&
           voice_it->second.voice_index == token.voice_index &&
           (!decodeRuntime || voice_it->second.decode_runtime == *decodeRuntime);
}

std::optional<u64> CommitVoiceRenderProgress(
    const VoiceRenderCommitToken& token, const std::shared_ptr<VoiceDecodeRuntime>& decodeRuntime,
    VoiceRenderProgressCommitHandler commitHandler, void* commitContext) {
    if (!decodeRuntime || !commitHandler) {
        return std::nullopt;
    }
    auto& runtime = Runtime();
    std::scoped_lock lock{runtime.mutex};
    const auto voice_it = runtime.voices.find(token.voice_handle);
    if (!MatchesRenderTokenLocked(runtime, token, &decodeRuntime) ||
        voice_it->second.state != VoicePlayState::Playing) {
        return std::nullopt;
    }
    AdvanceRenderStateRevisionLocked(voice_it->second);
    commitHandler(voice_it->second.render_state_revision, commitContext);
    return voice_it->second.render_state_revision;
}

static bool ValidateSystemRenderProgressLocked(
    const RuntimeRegistry& runtime, const SystemRenderGraphSnapshot& snapshot,
    std::span<const VoiceRenderProgressCommit> progress) {
    const auto system_it = runtime.systems.find(snapshot.system_handle);
    if (system_it == runtime.systems.end() ||
        system_it->second.generation != snapshot.system.generation) {
        return false;
    }

    size_t live_voice_count = 0;
    for (const auto& [rack_handle, rack] : runtime.racks) {
        if (rack.system_handle == snapshot.system_handle) {
            live_voice_count += rack.voices.size();
        }
    }
    if (live_voice_count != snapshot.voices.size()) {
        return false;
    }
    for (const SystemRenderVoiceSnapshot& voice : snapshot.voices) {
        if (!MatchesRenderTokenLocked(runtime, voice.token, nullptr)) {
            return false;
        }
    }

    for (size_t index = 0; index < progress.size(); ++index) {
        const VoiceRenderProgressCommit& request = progress[index];
        if (!request.decode_runtime || !request.handler ||
            !MatchesRenderTokenLocked(runtime, request.token, &request.decode_runtime)) {
            return false;
        }
        const auto snapshot_voice =
            std::ranges::find(snapshot.voices, request.token, &SystemRenderVoiceSnapshot::token);
        if (snapshot_voice == snapshot.voices.end() ||
            snapshot_voice->playback_state != VoiceRenderPlaybackState::Playing) {
            return false;
        }
        for (size_t previous = 0; previous < index; ++previous) {
            if (progress[previous].token.voice_handle == request.token.voice_handle) {
                return false;
            }
        }
    }

    return true;
}

bool ValidateSystemRenderProgress(const SystemRenderGraphSnapshot& snapshot,
                                  std::span<const VoiceRenderProgressCommit> progress) {
    auto& runtime = Runtime();
    std::scoped_lock lock{runtime.mutex};
    return ValidateSystemRenderProgressLocked(runtime, snapshot, progress);
}

bool CommitSystemRenderProgress(const SystemRenderGraphSnapshot& snapshot,
                                std::span<const VoiceRenderProgressCommit> progress) {
    auto& runtime = Runtime();
    std::scoped_lock lock{runtime.mutex};
    if (!ValidateSystemRenderProgressLocked(runtime, snapshot, progress)) {
        return false;
    }
    for (const VoiceRenderProgressCommit& request : progress) {
        auto& voice = runtime.voices.find(request.token.voice_handle)->second;
        AdvanceRenderStateRevisionLocked(voice);
        request.handler(voice.render_state_revision, request.context);
    }
    return true;
}

void ResetRuntimeForTesting() {
    auto& runtime = Runtime();
    std::scoped_lock lock{runtime.mutex};
    runtime.voices.clear();
    runtime.racks.clear();
    runtime.systems.clear();
    runtime.tombstones.clear();
    runtime.next_generation = 1;
}

} // namespace Libraries::Ngs2
