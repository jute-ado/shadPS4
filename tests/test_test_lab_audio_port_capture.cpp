// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/libraries/audio/test_lab_audio_capture.h"

namespace {

using namespace Libraries::AudioOut;

class TestLabAudioPortCapture : public ::testing::Test {
protected:
    void SetUp() override {
        root = std::filesystem::temp_directory_path() /
               ("shadps4-port-audio-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    }

    void TearDown() override {
        std::filesystem::remove_all(root);
    }

    std::vector<s16> ReadSamples(const std::filesystem::path& path) {
        std::ifstream input{path, std::ios::binary};
        EXPECT_TRUE(input.is_open());
        std::vector<s16> samples;
        for (s16 sample; input.read(reinterpret_cast<char*>(&sample), sizeof(sample));) {
            samples.push_back(sample);
        }
        return samples;
    }

    std::filesystem::path root;
};

class ScopedEnvironmentValue {
public:
    ScopedEnvironmentValue(const char* name, const std::string& value)
        : name_(name), previous_(ReadEnvironment(name)) {
#ifdef _WIN32
        if (_putenv_s(name_.c_str(), value.c_str()) != 0) {
            throw std::runtime_error("failed to set test environment");
        }
#else
        if (setenv(name_.c_str(), value.c_str(), 1) != 0) {
            throw std::runtime_error("failed to set test environment");
        }
#endif
    }

    ~ScopedEnvironmentValue() {
#ifdef _WIN32
        _putenv_s(name_.c_str(), previous_ ? previous_->c_str() : "");
#else
        if (previous_) {
            setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
#endif
    }

    ScopedEnvironmentValue(const ScopedEnvironmentValue&) = delete;
    ScopedEnvironmentValue& operator=(const ScopedEnvironmentValue&) = delete;

private:
    static std::optional<std::string> ReadEnvironment(const char* name) {
        if (const char* value = std::getenv(name); value != nullptr) {
            return std::string{value};
        }
        return std::nullopt;
    }

    std::string name_;
    std::optional<std::string> previous_;
};

TEST_F(TestLabAudioPortCapture, SilentSecondaryPortCannotInterleaveWithAudibleMainPort) {
    const auto audible_path = root / "port-1-type-0-48000hz.s16le";
    const auto silent_path = root / "port-2-type-0-48000hz.s16le";
    auto audible = CreateTestLabPortPcm16Capture(root, 1, 0, 48000);
    auto silent = CreateTestLabPortPcm16Capture(root, 2, 0, 48000);
    ASSERT_NE(audible, nullptr);
    ASSERT_NE(silent, nullptr);

    const std::array<s16, 4> first{110, -220, 330, -440};
    const std::array<s16, 4> zeros{};
    const std::array<s16, 2> second{550, -660};
    audible->Append(first);
    silent->Append(zeros);
    audible->Append(second);
    audible.reset();
    silent.reset();

    EXPECT_EQ(ReadSamples(audible_path), (std::vector<s16>{110, -220, 330, -440, 550, -660}));
    EXPECT_EQ(ReadSamples(silent_path), (std::vector<s16>{0, 0, 0, 0}));
}

TEST_F(TestLabAudioPortCapture, CapturesBgmAndPadSpeakerIndependentlyWithExplicitRates) {
    auto bgm = CreateTestLabPortPcm16Capture(root, 3, 1, 48000);
    auto pad = CreateTestLabPortPcm16Capture(root, 4, 4, 24000);
    ASSERT_NE(bgm, nullptr);
    ASSERT_NE(pad, nullptr);
    bgm->Append(std::array<s16, 2>{100, -100});
    pad->Append(std::array<s16, 2>{200, 200});
    bgm.reset();
    pad.reset();

    EXPECT_EQ(ReadSamples(root / "port-3-type-1-48000hz.s16le"), (std::vector<s16>{100, -100}));
    EXPECT_EQ(ReadSamples(root / "port-4-type-4-24000hz.s16le"), (std::vector<s16>{200, 200}));
}

TEST_F(TestLabAudioPortCapture, ExistingPortEvidenceIsNeverOverwritten) {
    const auto path = root / "port-5-type-0-48000hz.s16le";
    {
        auto capture = CreateTestLabPortPcm16Capture(root, 5, 0, 48000);
        ASSERT_NE(capture, nullptr);
        capture->Append(std::array<s16, 2>{1234, -1234});
    }

    EXPECT_THROW((void)CreateTestLabPortPcm16Capture(root, 5, 0, 48000), std::runtime_error);
    EXPECT_EQ(ReadSamples(path), (std::vector<s16>{1234, -1234}));
}

TEST_F(TestLabAudioPortCapture, AutomaticCaptureSkipsStaleIndexWithoutOverwritingEvidence) {
    ScopedEnvironmentValue capture_directory{
        TEST_LAB_AUDIO_PORT_CAPTURE_ENV,
        root.string(),
    };

    auto first = CreateTestLabPortPcm16Capture(0, 48000);
    ASSERT_NE(first, nullptr);
    first->Append(std::array<s16, 2>{1234, -1234});
    first.reset();

    const auto first_files = std::vector<std::filesystem::directory_entry>{
        std::filesystem::directory_iterator{root},
        std::filesystem::directory_iterator{},
    };
    ASSERT_EQ(first_files.size(), 1U);
    const auto first_name = first_files.front().path().filename().string();
    const auto first_prefix = std::string{"port-"};
    const auto first_suffix = std::string{"-type-0-48000hz.s16le"};
    ASSERT_TRUE(first_name.starts_with(first_prefix));
    ASSERT_TRUE(first_name.ends_with(first_suffix));
    const auto first_index_text = first_name.substr(
        first_prefix.size(), first_name.size() - first_prefix.size() - first_suffix.size());
    const auto first_index = std::stoull(first_index_text);

    const auto stale_path = root / ("port-" + std::to_string(first_index + 1) +
                                    "-type-0-48000hz.s16le");
    {
        std::ofstream stale{stale_path, std::ios::binary};
        ASSERT_TRUE(stale.is_open());
        stale << "keep";
    }

    auto second = CreateTestLabPortPcm16Capture(0, 48000);
    ASSERT_NE(second, nullptr);
    second->Append(std::array<s16, 2>{5678, -5678});
    second.reset();

    std::ifstream preserved{stale_path, std::ios::binary};
    const std::string preserved_bytes{std::istreambuf_iterator<char>{preserved},
                                      std::istreambuf_iterator<char>{}};
    EXPECT_EQ(preserved_bytes, "keep");
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator{root},
                            std::filesystem::directory_iterator{}),
              3);
}

TEST_F(TestLabAudioPortCapture, DisabledCaptureCreatesNoDirectory) {
    EXPECT_EQ(CreateTestLabPortPcm16Capture({}, 1, 0, 48000), nullptr);
    EXPECT_FALSE(std::filesystem::exists(root));
}

TEST_F(TestLabAudioPortCapture, InvalidCaptureMetadataFailsBeforeCreatingEvidence) {
    EXPECT_THROW((void)CreateTestLabPortPcm16Capture("relative-port-capture", 1, 0, 48000),
                 std::invalid_argument);
    EXPECT_THROW((void)CreateTestLabPortPcm16Capture(root, 0, 0, 48000), std::invalid_argument);
    EXPECT_THROW((void)CreateTestLabPortPcm16Capture(root, 1, 0, 0), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(root));
}

} // namespace
