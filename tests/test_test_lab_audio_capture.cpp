// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include "core/libraries/audio/test_lab_audio_capture.h"

using namespace Libraries::AudioOut;

TEST(TestLabAudioCapture, ConvertsStereoS16WithoutChangingSamples) {
    const std::array<s16, 4> input{-1200, 2400, 32767, -32768};
    const TestLabAudioFormat format{
        .is_float = false,
        .num_channels = 2,
        .channel_layout = {0, 1},
    };

    EXPECT_EQ(ConvertTestLabStereoPcm16(input.data(), 2, format),
              (std::vector<s16>{-1200, 2400, 32767, -32768}));
}

TEST(TestLabAudioCapture, DuplicatesMonoFloatAndClampsToPcm16) {
    const std::array<float, 3> input{0.5f, -1.0f, 2.0f};
    const TestLabAudioFormat format{
        .is_float = true,
        .num_channels = 1,
        .channel_layout = {0},
    };

    EXPECT_EQ(ConvertTestLabStereoPcm16(input.data(), 3, format),
              (std::vector<s16>{16384, 16384, -32768, -32768, 32767, 32767}));
}

TEST(TestLabAudioCapture, CreatesAndAppendsFlushedEvidence) {
    const auto root = std::filesystem::temp_directory_path() /
                      ("shadps4-audio-capture-" +
                       std::to_string(
                           std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count()));
    const auto path = root / "captures" / "audio.s16le";
    {
        TestLabPcm16CaptureFile capture{path};
        const std::array<s16, 2> first{1, 2};
        const std::array<s16, 2> second{3, 4};
        capture.Append(first);
        capture.Append(second);
    }

    {
        std::ifstream stream{path, std::ios::binary};
        const std::vector<char> bytes{
            std::istreambuf_iterator<char>{stream},
            std::istreambuf_iterator<char>{},
        };
        ASSERT_EQ(bytes.size(), 8);
        EXPECT_EQ(bytes[0], 1);
        EXPECT_EQ(bytes[2], 2);
        EXPECT_EQ(bytes[4], 3);
        EXPECT_EQ(bytes[6], 4);
    }
    std::filesystem::remove_all(root);
}

TEST(TestLabAudioCapture, RefusesToOverwriteExistingEvidence) {
    const auto root = std::filesystem::temp_directory_path() /
                      ("shadps4-audio-capture-existing-" +
                       std::to_string(
                           std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count()));
    std::filesystem::create_directories(root);
    const auto path = root / "audio.s16le";
    {
        std::ofstream existing{path, std::ios::binary};
        existing << "keep";
    }

    EXPECT_THROW(TestLabPcm16CaptureFile{path}, std::runtime_error);
    std::filesystem::remove_all(root);
}
