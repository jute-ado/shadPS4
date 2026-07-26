// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdio>
#include <filesystem>
#include <span>
#include <vector>

#include "common/types.h"

namespace Libraries::AudioOut {

constexpr auto TEST_LAB_AUDIO_CAPTURE_ENV = "EMULATOR_TEST_LAB_AUDIO_PCM16";

class TestLabPcm16CaptureFile {
public:
    explicit TestLabPcm16CaptureFile(const std::filesystem::path& path);
    ~TestLabPcm16CaptureFile();

    TestLabPcm16CaptureFile(const TestLabPcm16CaptureFile&) = delete;
    TestLabPcm16CaptureFile& operator=(const TestLabPcm16CaptureFile&) = delete;

    void Append(std::span<const s16> stereo_pcm16);

private:
    std::FILE* file_ = nullptr;
};

struct TestLabAudioFormat {
    bool is_float;
    u8 num_channels;
    std::array<int, 8> channel_layout;
};

[[nodiscard]] std::vector<s16> ConvertTestLabStereoPcm16(
    const void* source, u32 frames, const TestLabAudioFormat& format);

void CaptureTestLabMainPortAudio(const void* source, u32 frames,
                                 const TestLabAudioFormat& format, bool is_main);

} // namespace Libraries::AudioOut
