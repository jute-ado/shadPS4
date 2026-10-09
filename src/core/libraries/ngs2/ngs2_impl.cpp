// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "ngs2.h"
#include "ngs2_atrac9.h"
#include "ngs2_custom.h"
#include "ngs2_error.h"
#include "ngs2_impl.h"
#include "ngs2_mastering.h"
#include "ngs2_reverb.h"
#include "ngs2_sampler.h"
#include "ngs2_submixer.h"

#include <algorithm>
#include <concepts>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

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
constexpr size_t MaxVoiceControlCommands = 4096;
constexpr size_t MaxVoiceControlBytes = 1024 * 1024;
constexpr u32 MaxTracedWaveformBlocks = 4096;
constexpr size_t MaxVoiceStateBytes = 1024 * 1024;
constexpr u32 VoiceMatrixLevelsControlId = 0x00000001;
constexpr u32 VoicePortVolumeControlId = 0x00000002;
constexpr u32 VoicePortMatrixControlId = 0x00000003;
constexpr u32 VoicePatchControlId = 0x00000005;
constexpr u32 VoiceEventControlId = 0x00000006;
constexpr u32 SamplerSetupControlId = 0x10000000;
constexpr u32 SamplerWaveformBlocksControlId = 0x10000001;
constexpr u32 SamplerUserFxControlId = 0x10000008;
constexpr u32 SubmixerUserFxControlId = 0x20000004;
constexpr u32 MasteringOutputControlId = 0x30000005;

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
        u64 committed_destination_generation{};
    };
    std::unordered_map<u32, Patch> patches;
    u64 generation{};
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

struct PatchResolution {
    VoiceTraceIdentity destination{};
    DestinationClassification classification{};
    bool present{};
    bool same_system{};
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
                classification.offset_class = voice_offset % VoiceTokenStride == 0
                                                  ? AudioPipelineTrace::DestinationOffsetClass::
                                                        VoiceSlotAligned
                                                  : AudioPipelineTrace::DestinationOffsetClass::
                                                        VoiceSlotInterior;
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

PatchResolution ResolvePatchLocked(const RuntimeRegistry& runtime, const VoiceTraceIdentity& source,
                                   OrbisNgs2Handle destination_handle) {
    const VoiceTraceIdentity destination =
        ResolveVoiceTraceIdentityLocked(runtime, destination_handle);
    return {
        .destination = destination,
        .classification = ClassifyDestinationLocked(runtime, destination_handle),
        .present = destination_handle != 0,
        .same_system = source.resolved && destination.resolved &&
                       source.system_handle == destination.system_handle,
    };
}

AudioPipelineTrace::VoicePatchStateMetadata PatchStateMetadata(
    const VoiceTraceIdentity& source, u32 source_port, u32 destination_input_id,
    const PatchResolution& resolution, u64 committed_destination_generation,
    AudioPipelineTrace::PatchResolutionPhase phase) {
    const auto& classification = resolution.classification;
    return {
        .source_rack_id = source.rack_id,
        .source_voice_index = source.voice_index,
        .source_port = source_port,
        .destination_input_id = destination_input_id,
        .destination_rack_id = resolution.destination.rack_id,
        .destination_voice_index = resolution.destination.voice_index,
        .destination_present = resolution.present,
        .destination_resolved = resolution.destination.resolved,
        .destination_same_system = resolution.same_system,
        .destination_exact_match = classification.exact_match,
        .destination_inside_live_rack = classification.inside_live_rack,
        .destination_offset_class = classification.offset_class,
        .destination_candidate_voice_index_valid =
            classification.candidate_voice_index_valid,
        .destination_candidate_voice_index = classification.candidate_voice_index,
        .destination_previously_returned = classification.previously_returned,
        .destination_live_generation = classification.live_generation,
        .destination_tombstone_kind = classification.tombstone_kind,
        .destination_tombstone_generation = classification.tombstone_generation,
        .destination_generation_changed_since_commit =
            committed_destination_generation != 0 &&
            committed_destination_generation != classification.live_generation,
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

void ApplyEvent(VoiceRuntime& voice, u32 event_id) {
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
    OrbisNgs2RuntimeReadHandler read_handler, void* read_context, u32& remaining_block_reads) {
    AudioPipelineTrace::WaveformBlocksMetadata metadata{
        .source_rack_id = source.rack_id,
        .source_voice_index = source.voice_index,
        .flags = param.flags,
        .data_present = param.data != nullptr,
        .block_count = param.numBlocks,
        .data_range_arithmetic_valid = param.numBlocks == 0 || param.data != nullptr,
        .metadata_complete = true,
    };
    if (param.numBlocks > remaining_block_reads ||
        (param.numBlocks != 0 && param.aBlock == nullptr)) {
        metadata.metadata_complete = false;
        metadata.data_range_arithmetic_valid = false;
        return metadata;
    }

    uintptr_t block_address = reinterpret_cast<uintptr_t>(param.aBlock);
    const uintptr_t data_base = reinterpret_cast<uintptr_t>(param.data);
    bool saw_block = false;
    for (u32 block_index = 0; block_index < param.numBlocks; ++block_index) {
        OrbisNgs2WaveformBlock block{};
        if (AddWouldOverflow(block_address, sizeof(block))) {
            metadata.metadata_complete = false;
            metadata.data_range_arithmetic_valid = false;
            break;
        }
        --remaining_block_reads;
        if (!read_handler(block_address, &block, sizeof(block), read_context)) {
            metadata.metadata_complete = false;
            metadata.data_range_arithmetic_valid = false;
            break;
        }
        if (!saw_block) {
            metadata.minimum_data_offset = block.dataOffset;
            saw_block = true;
        } else {
            metadata.minimum_data_offset = std::min(metadata.minimum_data_offset, block.dataOffset);
        }
        metadata.maximum_data_end = std::max(metadata.maximum_data_end,
                                             static_cast<u64>(block.dataOffset) + block.dataSize);
        metadata.aggregate_data_bytes =
            SaturatingAdd(metadata.aggregate_data_bytes, block.dataSize);
        metadata.aggregate_samples = SaturatingAdd(metadata.aggregate_samples, block.numSamples);
        metadata.aggregate_skip_samples =
            SaturatingAdd(metadata.aggregate_skip_samples, block.numSkipSamples);
        metadata.has_repeats |= block.numRepeats != 0;
        if (AddWouldOverflow(data_base, block.dataOffset) ||
            AddWouldOverflow(data_base + block.dataOffset, block.dataSize)) {
            metadata.data_range_arithmetic_valid = false;
        }

        if (block_index + 1 < param.numBlocks) {
            block_address += sizeof(block);
        }
    }
    return metadata;
}

void TraceVoiceCommand(const VoiceTraceIdentity& source, const OrbisNgs2VoiceParamHeader& header,
                       std::span<const u8> command, OrbisNgs2RuntimeReadHandler read_handler,
                       void* read_context, u32& remaining_waveform_block_reads,
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
        records.emplace_back(TraceWaveformBlocks(source, param, read_handler, read_context,
                                                 remaining_waveform_block_reads));
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
    if (option && option->size < sizeof(OrbisNgs2RackOption)) {
        return ORBIS_NGS2_ERROR_INVALID_OPTION_SIZE;
    }
    if (rackId == 0) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_ID;
    }

    const u32 max_voices = option ? option->maxVoices : 0;
    size_t required_buffer_size = 0;
    if (!RequiredRackBufferSize(max_voices, &required_buffer_size)) {
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
                                           const OrbisNgs2RackOption* option,
                                           const OrbisNgs2ContextBufferInfo* bufferInfo,
                                           OrbisNgs2BufferFreeHandler hostFree,
                                           OrbisNgs2Handle* outHandle) {
    if (!outHandle) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    if (!option) {
        return ORBIS_NGS2_ERROR_INVALID_OPTION_ADDRESS;
    }
    if (option->size < sizeof(OrbisNgs2RackOption)) {
        return ORBIS_NGS2_ERROR_INVALID_OPTION_SIZE;
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
    if (!RequiredRackBufferSize(option->maxVoices, &required_buffer_size)) {
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
        .max_voices = option->maxVoices,
        .max_input_delay_blocks = option->maxInputDelayBlocks,
        .max_matrices = option->maxMatrices,
        .max_ports = option->maxPorts,
        .succeeded = true,
    };
    AudioPipelineTrace::RackLifecycleMetadata lifecycle_metadata{};

    std::vector<OrbisNgs2Handle> voice_handles;
    voice_handles.reserve(option->maxVoices);
    for (u32 index = 0; index < option->maxVoices; ++index) {
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
    return RackCreateWithExpectedSystemGeneration(systemHandle, 0, rackId, option, bufferInfo,
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

    const OrbisNgs2BufferAllocator allocator_copy = *allocator;
    const OrbisNgs2RackOption option_copy = *option;
    if (option_copy.size < sizeof(OrbisNgs2RackOption)) {
        return ORBIS_NGS2_ERROR_INVALID_OPTION_SIZE;
    }
    if (rackId == 0) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_ID;
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
    s32 result = RackQueryBufferSize(rackId, &option_copy, &buffer_info);
    if (result < 0) {
        return result;
    }
    buffer_info.userData = allocator_copy.userData;
    result = allocator_copy.allocHandler(&buffer_info);
    if (result < 0) {
        return result;
    }
    result = RackCreateWithExpectedSystemGeneration(systemHandle, system_generation, rackId,
                                                    &option_copy, &buffer_info,
                                                    allocator_copy.freeHandler, outHandle);
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
        AddTombstoneLocked(runtime,
                           {
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
    auto& runtime = Runtime();
    {
        std::scoped_lock lock{runtime.mutex};
        const auto voice_it = runtime.voices.find(voiceHandle);
        if (voice_it == runtime.voices.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        voice_generation = voice_it->second.generation;
        const auto rack_it = runtime.racks.find(voice_it->second.rack_handle);
        if (rack_it != runtime.racks.end()) {
            source_identity = {
                .rack_id = rack_it->second.rack_id,
                .voice_index = voice_it->second.voice_index,
                .resolved = true,
                .system_handle = rack_it->second.system_handle,
                .generation = voice_it->second.generation,
            };
        }
    }

    std::vector<u32> parsed_events;
    std::vector<PendingPatch> parsed_patches;
    std::vector<AudioPipelineTrace::VoiceControlMetadata> pending_voice_traces;
    std::vector<PendingTraceRecord> pending_command_traces;
    const bool trace_enabled = AudioPipelineTrace::IsEnabled();
    std::unordered_set<uintptr_t> visited;
    uintptr_t address = reinterpret_cast<uintptr_t>(paramList);
    size_t total_bytes = 0;
    u32 remaining_traced_waveform_block_reads = MaxTracedWaveformBlocks;

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

        if (header.id == VoicePatchControlId &&
            command.size() == sizeof(OrbisNgs2VoicePatchParam)) {
            OrbisNgs2VoicePatchParam patch{};
            std::memcpy(&patch, command.data(), sizeof(patch));
            parsed_patches.push_back({patch.port, patch.destInputId, patch.destHandle});
        }

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
            TraceVoiceCommand(source_identity, header, command, readHandler, readContext,
                              remaining_traced_waveform_block_reads, pending_command_traces);
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
        voice_it->second.pending_events.insert(voice_it->second.pending_events.end(),
                                               parsed_events.begin(), parsed_events.end());
        for (const PendingPatch& patch : parsed_patches) {
            const PatchResolution resolution =
                ResolvePatchLocked(runtime, source_identity, patch.destination_handle);
            auto& stored = voice_it->second.patches[patch.source_port];
            stored = {
                .destination_input_id = patch.destination_input_id,
                .destination_handle = patch.destination_handle,
                .last_observed = true,
                .last_destination_resolved = resolution.destination.resolved,
                .last_destination_same_system = resolution.same_system,
                .last_destination_generation = resolution.destination.generation,
                .last_classification = resolution.classification,
                .committed_destination_generation = resolution.classification.live_generation,
            };
            if (trace_enabled) {
                committed_patch_traces.push_back({
                    .source_rack_id = source_identity.rack_id,
                    .source_voice_index = source_identity.voice_index,
                    .source_port = patch.source_port,
                    .destination_input_id = patch.destination_input_id,
                    .destination_rack_id = resolution.destination.rack_id,
                    .destination_voice_index = resolution.destination.voice_index,
                    .destination_resolved = resolution.destination.resolved,
                    .destination_present = resolution.present,
                });
                committed_patch_state_traces.push_back(PatchStateMetadata(
                    source_identity, patch.source_port, patch.destination_input_id, resolution,
                    stored.committed_destination_generation,
                    AudioPipelineTrace::PatchResolutionPhase::Committed));
            }
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

s32 SystemRender(OrbisNgs2Handle systemHandle, const OrbisNgs2RenderBufferInfo* bufferInfo,
                 u32 numBufferInfo) {
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
                for (const u32 event_id : voice_it->second.pending_events) {
                    ApplyEvent(voice_it->second, event_id);
                }
                voice_it->second.pending_events.clear();
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
                        ResolvePatchLocked(runtime, source_identity, patch.destination_handle);
                    const bool changed =
                        !patch.last_observed ||
                        patch.last_destination_resolved != resolution.destination.resolved ||
                        patch.last_destination_same_system != resolution.same_system ||
                        patch.last_destination_generation != resolution.destination.generation ||
                        patch.last_classification != resolution.classification;
                    if (!changed) {
                        continue;
                    }
                    patch.last_observed = true;
                    patch.last_destination_resolved = resolution.destination.resolved;
                    patch.last_destination_same_system = resolution.same_system;
                    patch.last_destination_generation = resolution.destination.generation;
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

    for (u32 index = 0; index < numBufferInfo; ++index) {
        if (bufferInfo[index].bufferSize != 0) {
            MemoryClear(bufferInfo[index].buffer, bufferInfo[index].bufferSize);
        }
    }
    return ORBIS_OK;
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
