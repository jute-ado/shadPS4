// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/audio/atrac9_codec_lock.h"

#include <mutex>

namespace Libraries::Audio {

std::shared_mutex& Atrac9CodecTableMutex() noexcept {
    static std::shared_mutex mutex;
    return mutex;
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
