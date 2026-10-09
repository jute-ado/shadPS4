// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/ngs2/ngs2.h"

namespace Libraries::Ngs2 {

// NGS2's ATRAC9 waveform discriminator. A matching value is necessary but is
// never sufficient: the codec configuration and declared stream shape must
// also pass InspectAtrac9Format.
constexpr u32 Atrac9WaveformType = 0x40;

enum class Atrac9FormatStatus : u8 {
    UnsupportedWaveform,
    InvalidConfig,
    DeclaredFormatMismatch,
    Supported,
};

struct Atrac9FormatInfo {
    u32 channels{};
    u32 sample_rate{};
    u32 frame_bytes{};
    u32 frames_per_superframe{};
    u32 frame_samples{};
    u32 superframe_bytes{};
    u32 superframe_samples{};
};

struct Atrac9FormatInspection {
    Atrac9FormatStatus status{Atrac9FormatStatus::UnsupportedWaveform};
    Atrac9FormatInfo info{};
};

// Validates the NGS2 discriminator, the inline big-endian ATRAC9 config, the
// decoder-reported frame shape, and the NGS2-declared channel/rate contract.
[[nodiscard]] Atrac9FormatInspection InspectAtrac9Format(
    const OrbisNgs2WaveformFormat& format) noexcept;

struct Atrac9BlockCursorInspection {
    u64 data_end{};
    u64 encoded_frames{};
    u64 decoded_samples{};
    u64 playable_samples{};
    bool frame_offset_aligned{};
    bool complete_frames{};
    bool complete_superframes{};
    bool sample_bounds_valid{};
    bool decodable{};
};

// Examines only public block metadata. Encoded bytes are deliberately not read
// here; this is the fail-closed boundary required before a decoder owns a
// payload cursor. Complete superframes are reported for scheduling but are not
// required: LibAtrac9 consumes individual, complete frames.
[[nodiscard]] Atrac9BlockCursorInspection InspectAtrac9BlockCursor(
    const Atrac9FormatInfo& format, const OrbisNgs2WaveformBlock& block) noexcept;

} // namespace Libraries::Ngs2
