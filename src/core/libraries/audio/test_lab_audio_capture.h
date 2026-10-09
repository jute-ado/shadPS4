// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "common/types.h"

namespace Libraries::AudioOut {

constexpr auto TEST_LAB_AUDIO_CAPTURE_ENV = "EMULATOR_TEST_LAB_AUDIO_PCM16";
constexpr auto TEST_LAB_AUDIO_PORT_CAPTURE_ENV = "EMULATOR_TEST_LAB_AUDIO_PORT_PCM16_DIR";

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

[[nodiscard]] std::unique_ptr<TestLabPcm16CaptureFile> CreateTestLabPortPcm16Capture(
    const std::filesystem::path& output_directory, u64 capture_index, u32 port_type,
    u32 sample_rate);

[[nodiscard]] std::unique_ptr<TestLabPcm16CaptureFile> CreateTestLabPortPcm16Capture(
    u32 port_type, u32 sample_rate);

} // namespace Libraries::AudioOut
