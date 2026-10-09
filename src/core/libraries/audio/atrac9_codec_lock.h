// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <shared_mutex>

#include <libatrac9.h>

namespace Libraries::Audio {

// LibAtrac9 decoder initialization writes process-global decode tables while
// decode calls read them. Callers must hold this lock exclusively around
// Atrac9InitDecoder and shared around every Atrac9Decode variant.
[[nodiscard]] std::shared_mutex& Atrac9CodecTableMutex() noexcept;

// Returns an independent decoder checkpoint, or null for an invalid/uninitialized handle or
// allocation failure. Release a successful clone with Atrac9ReleaseHandle.
[[nodiscard]] void* Atrac9CloneHandleLocked(const void* handle) noexcept;

int Atrac9InitDecoderLocked(void* handle, unsigned char* config_data) noexcept;
int Atrac9DecodeLocked(void* handle, const unsigned char* input, int input_size, short* output,
                       int* bytes_used, int noninterleaved) noexcept;
int Atrac9DecodeS32Locked(void* handle, const unsigned char* input, int input_size, int* output,
                          int* bytes_used, int noninterleaved) noexcept;
int Atrac9DecodeF32Locked(void* handle, const unsigned char* input, int input_size, float* output,
                          int* bytes_used, int noninterleaved) noexcept;

} // namespace Libraries::Audio
