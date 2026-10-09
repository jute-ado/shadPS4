// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/audio/atrac9_codec_lock.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

#include "../../../../externals/LibAtrac9/C/src/structures.h"

namespace Libraries::Audio {

std::shared_mutex& Atrac9CodecTableMutex() noexcept {
    static std::shared_mutex mutex;
    return mutex;
}

void* Atrac9CloneHandleLocked(const void* handle) noexcept {
    std::shared_lock lock{Atrac9CodecTableMutex()};

    if (handle == nullptr) {
        return nullptr;
    }

    const auto* source = static_cast<const Atrac9Handle*>(handle);
    const int block_count = source->Config.ChannelConfig.BlockCount;
    const int channel_count = source->Config.ChannelCount;
    if (source->Initialized != 1 || block_count < 1 || block_count > MAX_BLOCK_COUNT ||
        channel_count < 1 || channel_count > MAX_CHANNEL_COUNT ||
        source->Config.FramesPerSuperframe < 1 ||
        source->Frame.IndexInSuperframe < 0 ||
        source->Frame.IndexInSuperframe >= source->Config.FramesPerSuperframe) {
        return nullptr;
    }

    auto* clone = static_cast<Atrac9Handle*>(std::malloc(sizeof(Atrac9Handle)));
    if (clone == nullptr) {
        return nullptr;
    }
    std::memcpy(clone, source, sizeof(Atrac9Handle));

    clone->Frame.Config = &clone->Config;
    for (auto& channel : clone->Frame.Channels) {
        channel = nullptr;
    }

    int next_channel = 0;
    for (int block_index = 0; block_index < block_count; ++block_index) {
        auto& block = clone->Frame.Blocks[block_index];
        if (block.ChannelCount < 0 || block.ChannelCount > MAX_BLOCK_CHANNEL_COUNT ||
            next_channel > channel_count - block.ChannelCount) {
            std::free(clone);
            return nullptr;
        }

        block.Frame = &clone->Frame;
        block.Config = &clone->Config;
        for (int channel_index = 0; channel_index < block.ChannelCount; ++channel_index) {
            auto& channel = block.Channels[channel_index];
            channel.Frame = &clone->Frame;
            channel.Block = &block;
            channel.Config = &clone->Config;
            clone->Frame.Channels[next_channel++] = &channel;
        }
    }

    if (next_channel != channel_count) {
        std::free(clone);
        return nullptr;
    }
    return clone;
}

int Atrac9InitDecoderLocked(void* handle, unsigned char* config_data) noexcept {
    std::unique_lock lock{Atrac9CodecTableMutex()};
    return Atrac9InitDecoder(handle, config_data);
}

int Atrac9DecodeLocked(void* handle, const unsigned char* input, int input_size, short* output,
                       int* bytes_used, int noninterleaved) noexcept {
    std::shared_lock lock{Atrac9CodecTableMutex()};
    return Atrac9Decode(handle, input, input_size, output, bytes_used, noninterleaved);
}

int Atrac9DecodeS32Locked(void* handle, const unsigned char* input, int input_size, int* output,
                          int* bytes_used, int noninterleaved) noexcept {
    std::shared_lock lock{Atrac9CodecTableMutex()};
    return Atrac9DecodeS32(handle, input, input_size, output, bytes_used, noninterleaved);
}

int Atrac9DecodeF32Locked(void* handle, const unsigned char* input, int input_size, float* output,
                          int* bytes_used, int noninterleaved) noexcept {
    std::shared_lock lock{Atrac9CodecTableMutex()};
    return Atrac9DecodeF32(handle, input, input_size, output, bytes_used, noninterleaved);
}

} // namespace Libraries::Audio
