// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/audio/test_lab_audio_capture.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <system_error>

namespace Libraries::AudioOut {
namespace {

float ReadNormalizedSample(const void* source, std::size_t index, bool is_float) {
    if (is_float) {
        float sample;
        std::memcpy(&sample,
                    static_cast<const std::byte*>(source) + index * sizeof(float),
                    sizeof(sample));
        return std::isfinite(sample) ? sample : 0.0f;
    }

    s16 sample;
    std::memcpy(&sample,
                static_cast<const std::byte*>(source) + index * sizeof(s16),
                sizeof(sample));
    return static_cast<float>(sample) / 32768.0f;
}

s16 ToPcm16(float sample) {
    const auto clamped = std::clamp(sample, -1.0f, 1.0f);
    if (clamped <= -1.0f) {
        return -32768;
    }
    return static_cast<s16>(std::lround(clamped * 32767.0f));
}

float Downmix(const void* source, std::size_t frame, const TestLabAudioFormat& format,
              bool left) {
    const auto channels = static_cast<std::size_t>(format.num_channels);
    const auto at = [&](int channel) {
        return ReadNormalizedSample(source, frame * channels + channel, format.is_float);
    };
    if (channels == 1) {
        return at(0);
    }
    if (channels == 2) {
        return at(left ? 0 : 1);
    }

    const auto& layout = format.channel_layout;
    const float front = at(layout[left ? 0 : 1]);
    const float center = at(layout[2]) * 0.70710678f;
    const float lfe = at(layout[3]) * 0.5f;
    const float back = at(layout[left ? 4 : 5]) * 0.70710678f;
    const float side = at(layout[left ? 6 : 7]) * 0.70710678f;
    return front + center + lfe + back + side;
}

std::mutex capture_mutex;
std::once_flag capture_init;
std::unique_ptr<TestLabPcm16CaptureFile> capture_file;
std::atomic<u64> next_port_capture_index{1};

} // namespace

TestLabPcm16CaptureFile::TestLabPcm16CaptureFile(const std::filesystem::path& path) {
    const auto absolute = std::filesystem::absolute(path);
    if (absolute.has_parent_path()) {
        std::filesystem::create_directories(absolute.parent_path());
    }
#ifdef _WIN32
    if (_wfopen_s(&file_, absolute.c_str(), L"wbx") != 0) {
#else
    file_ = std::fopen(absolute.c_str(), "wbx");
    if (file_ == nullptr) {
#endif
        throw std::runtime_error("failed to create audio capture evidence");
    }
}

TestLabPcm16CaptureFile::~TestLabPcm16CaptureFile() {
    if (file_ != nullptr) {
        std::fclose(file_);
    }
}

void TestLabPcm16CaptureFile::Append(std::span<const s16> stereo_pcm16) {
    if ((stereo_pcm16.size() & 1) != 0) {
        throw std::invalid_argument("stereo PCM16 must contain complete frames");
    }
    if (std::fwrite(stereo_pcm16.data(), 1, stereo_pcm16.size_bytes(), file_) !=
            stereo_pcm16.size_bytes() ||
        std::fflush(file_) != 0) {
        throw std::runtime_error("failed to append audio capture evidence");
    }
}

std::vector<s16> ConvertTestLabStereoPcm16(const void* source, u32 frames,
                                          const TestLabAudioFormat& format) {
    if (source == nullptr) {
        throw std::invalid_argument("audio source is null");
    }
    if (format.num_channels != 1 && format.num_channels != 2 &&
        format.num_channels != 8) {
        throw std::invalid_argument("unsupported audio channel count");
    }

    std::vector<s16> output(static_cast<std::size_t>(frames) * 2);
    if (!format.is_float && format.num_channels == 2) {
        std::memcpy(output.data(), source, output.size() * sizeof(s16));
        return output;
    }
    for (std::size_t frame = 0; frame < frames; ++frame) {
        output[frame * 2] = ToPcm16(Downmix(source, frame, format, true));
        output[frame * 2 + 1] = ToPcm16(Downmix(source, frame, format, false));
    }
    return output;
}

void CaptureTestLabMainPortAudio(const void* source, u32 frames,
                                 const TestLabAudioFormat& format, bool is_main) {
    if (!is_main) {
        return;
    }
    std::call_once(capture_init, [] {
        const char* path = std::getenv(TEST_LAB_AUDIO_CAPTURE_ENV);
        if (path != nullptr && path[0] != '\0') {
            capture_file = std::make_unique<TestLabPcm16CaptureFile>(path);
        }
    });
    if (!capture_file) {
        return;
    }

    auto pcm = ConvertTestLabStereoPcm16(
        source, frames, format);
    std::scoped_lock lock{capture_mutex};
    capture_file->Append(pcm);
}

std::unique_ptr<TestLabPcm16CaptureFile> CreateTestLabPortPcm16Capture(
    const std::filesystem::path& output_directory, u64 capture_index, u32 port_type,
    u32 sample_rate) {
    if (output_directory.empty()) {
        return {};
    }
    if (!output_directory.is_absolute() || capture_index == 0 || sample_rate == 0) {
        throw std::invalid_argument("invalid per-port audio capture directory or metadata");
    }
    const auto filename = "port-" + std::to_string(capture_index) + "-type-" +
                          std::to_string(port_type) + "-" + std::to_string(sample_rate) +
                          "hz.s16le";
    return std::make_unique<TestLabPcm16CaptureFile>(output_directory / filename);
}

std::unique_ptr<TestLabPcm16CaptureFile> CreateTestLabPortPcm16Capture(u32 port_type,
                                                                 u32 sample_rate) {
    const char* path = std::getenv(TEST_LAB_AUDIO_PORT_CAPTURE_ENV);
    if (path == nullptr || path[0] == '\0') {
        return {};
    }

    const auto output_directory = std::filesystem::path{path};
    for (;;) {
        const auto capture_index =
            next_port_capture_index.fetch_add(1, std::memory_order_relaxed);
        const auto filename = "port-" + std::to_string(capture_index) + "-type-" +
                              std::to_string(port_type) + "-" +
                              std::to_string(sample_rate) + "hz.s16le";
        const auto candidate = output_directory / filename;

        std::error_code exists_error;
        if (std::filesystem::exists(candidate, exists_error)) {
            continue;
        }
        if (exists_error) {
            throw std::system_error(exists_error,
                                    "failed to inspect audio capture evidence");
        }

        try {
            return CreateTestLabPortPcm16Capture(output_directory, capture_index, port_type,
                                                 sample_rate);
        } catch (const std::runtime_error&) {
            // Another emulator process may have claimed this index between
            // the existence check and the exclusive create. Retry only when
            // the candidate is now present; preserve all other I/O errors.
            exists_error.clear();
            if (std::filesystem::exists(candidate, exists_error)) {
                continue;
            }
            if (exists_error) {
                throw std::system_error(exists_error,
                                        "failed to inspect audio capture evidence");
            }
            throw;
        }
    }
}

} // namespace Libraries::AudioOut
