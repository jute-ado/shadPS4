// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/ngs2/ngs2_atrac9.h"

#include <array>
#include <cstring>

#include "core/libraries/audio/atrac9_codec_lock.h"

extern "C" {
#include <libatrac9.h>
}

namespace Libraries::Ngs2 {
namespace {

constexpr u32 Atrac9Header = 0xFE;
constexpr u32 MaxAtrac9ChannelConfigIndex = 5;

std::array<u8, ATRAC9_CONFIG_DATA_SIZE> ConfigBytes(u32 config) noexcept {
    return {
        static_cast<u8>(config >> 24U),
        static_cast<u8>(config >> 16U),
        static_cast<u8>(config >> 8U),
        static_cast<u8>(config),
    };
}

bool ConfigPreflight(u32 config) noexcept {
    const u32 header = config >> 24U;
    const u32 channel_config_index = (config >> 17U) & 0x7U;
    const u32 validation_bit = (config >> 16U) & 0x1U;
    return header == Atrac9Header && channel_config_index <= MaxAtrac9ChannelConfigIndex &&
           validation_bit == 0;
}

bool CodecInfoIsSane(const Atrac9CodecInfo& info) noexcept {
    if (info.channels <= 0 || info.channels > ORBIS_NGS2_MAX_VOICE_CHANNELS ||
        info.samplingRate <= 0 || info.superframeSize <= 0 || info.framesInSuperframe <= 0 ||
        info.framesInSuperframe > 8 || info.frameSamples <= 0 || info.frameSamples > 256) {
        return false;
    }
    const u32 frames = static_cast<u32>(info.framesInSuperframe);
    return (frames & (frames - 1U)) == 0 && static_cast<u32>(info.superframeSize) % frames == 0;
}

} // namespace

Atrac9FormatInspection InspectAtrac9Format(const OrbisNgs2WaveformFormat& format) noexcept {
    Atrac9FormatInspection result;
    if (format.waveformType != Atrac9WaveformType) {
        return result;
    }
    result.status = Atrac9FormatStatus::InvalidConfig;
    if (!ConfigPreflight(format.configData)) {
        return result;
    }

    auto config = ConfigBytes(format.configData);
    void* const handle = Atrac9GetHandle();
    if (!handle) {
        return result;
    }
    const int init_result = Audio::Atrac9InitDecoderLocked(handle, config.data());
    Atrac9CodecInfo codec_info{};
    const int info_result = init_result == 0 ? Atrac9GetCodecInfo(handle, &codec_info) : -1;
    Atrac9ReleaseHandle(handle);
    if (init_result != 0 || info_result != 0 || !CodecInfoIsSane(codec_info) ||
        std::memcmp(codec_info.configData, config.data(), config.size()) != 0) {
        return result;
    }

    const u32 frames = static_cast<u32>(codec_info.framesInSuperframe);
    result.info = {
        .channels = static_cast<u32>(codec_info.channels),
        .sample_rate = static_cast<u32>(codec_info.samplingRate),
        .frame_bytes = static_cast<u32>(codec_info.superframeSize) / frames,
        .frames_per_superframe = frames,
        .frame_samples = static_cast<u32>(codec_info.frameSamples),
        .superframe_bytes = static_cast<u32>(codec_info.superframeSize),
        .superframe_samples = static_cast<u32>(codec_info.frameSamples) * frames,
    };
    result.status =
        result.info.channels == format.numChannels && result.info.sample_rate == format.sampleRate
            ? Atrac9FormatStatus::Supported
            : Atrac9FormatStatus::DeclaredFormatMismatch;
    return result;
}

Atrac9BlockCursorInspection InspectAtrac9BlockCursor(const Atrac9FormatInfo& format,
                                                     const OrbisNgs2WaveformBlock& block) noexcept {
    Atrac9BlockCursorInspection result{
        .data_end = static_cast<u64>(block.dataOffset) + block.dataSize,
    };
    if (format.frame_bytes == 0 || format.frames_per_superframe == 0 || format.frame_samples == 0 ||
        format.superframe_bytes == 0 || format.superframe_samples == 0) {
        return result;
    }

    result.frame_offset_aligned = block.dataOffset % format.frame_bytes == 0;
    result.complete_frames = block.dataSize != 0 && block.dataSize % format.frame_bytes == 0;
    result.complete_superframes = block.dataSize != 0 &&
                                  block.dataOffset % format.superframe_bytes == 0 &&
                                  block.dataSize % format.superframe_bytes == 0;
    if (result.complete_frames) {
        result.encoded_frames = block.dataSize / format.frame_bytes;
        result.decoded_samples = result.encoded_frames * format.frame_samples;
    }
    result.sample_bounds_valid = block.numSkipSamples <= result.decoded_samples &&
                                 block.numSamples <= result.decoded_samples - block.numSkipSamples;
    if (result.sample_bounds_valid) {
        result.playable_samples = block.numSamples;
    }
    const bool exact_sample_accounting =
        static_cast<u64>(block.numSkipSamples) + block.numSamples == result.decoded_samples;
    const bool explicit_empty = block.dataSize == 0 && block.numSkipSamples == 0 &&
                                block.numSamples == 0;
    result.decodable = block.numRepeats == 0 &&
                       (explicit_empty || (result.complete_superframes &&
                                           result.sample_bounds_valid &&
                                           exact_sample_accounting));
    return result;
}

} // namespace Libraries::Ngs2
