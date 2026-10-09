// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <future>
#include <mutex>

#include <gtest/gtest.h>

#include "core/libraries/audio/atrac9_codec_lock.h"
#include "core/libraries/ngs2/ngs2_atrac9.h"
#include "../externals/LibAtrac9/C/src/structures.h"

namespace Libraries::Ngs2 {
namespace {

constexpr OrbisNgs2WaveformFormat SixChannelFormat(u32 codec_config) {
    return {
        .waveformType = Atrac9WaveformType,
        .numChannels = 6,
        .sampleRate = 48000,
        .configData = codec_config,
    };
}

constexpr u32 Atrac9Config(u32 sample_rate_index, u32 channel_config_index, u32 frame_bytes,
                           u32 superframe_index) {
    return (0xFEU << 24U) | (sample_rate_index << 20U) | (channel_config_index << 17U) |
           ((frame_bytes - 1U) << 5U) | (superframe_index << 3U);
}

std::array<u8, ATRAC9_CONFIG_DATA_SIZE> ConfigBytesForTest(u32 config) {
    return {
        static_cast<u8>(config >> 24U),
        static_cast<u8>(config >> 16U),
        static_cast<u8>(config >> 8U),
        static_cast<u8>(config),
    };
}

void ExpectCloneTopologyIsSelfContained(const Atrac9Handle& clone) {
    ASSERT_EQ(clone.Initialized, 1);
    ASSERT_EQ(clone.Frame.Config, &clone.Config);
    int next_channel = 0;
    for (int block_index = 0; block_index < clone.Config.ChannelConfig.BlockCount;
         ++block_index) {
        const auto& block = clone.Frame.Blocks[block_index];
        EXPECT_EQ(block.Frame, &clone.Frame);
        EXPECT_EQ(block.Config, &clone.Config);
        for (int channel_index = 0; channel_index < block.ChannelCount; ++channel_index) {
            const auto& channel = block.Channels[channel_index];
            EXPECT_EQ(channel.Frame, &clone.Frame);
            EXPECT_EQ(channel.Block, &block);
            EXPECT_EQ(channel.Config, &clone.Config);
            ASSERT_LT(next_channel, clone.Config.ChannelCount);
            EXPECT_EQ(clone.Frame.Channels[next_channel++], &channel);
        }
    }
    EXPECT_EQ(next_channel, clone.Config.ChannelCount);
}

TEST(Ngs2Atrac9Contract, AcceptsObservedSixChannelConfigsAndReportsExactFrameShape) {
    const auto four_frame = InspectAtrac9Format(SixChannelFormat(0xFE761DF0));
    ASSERT_EQ(four_frame.status, Atrac9FormatStatus::Supported);
    EXPECT_EQ(four_frame.info.channels, 6);
    EXPECT_EQ(four_frame.info.sample_rate, 48000);
    EXPECT_EQ(four_frame.info.frame_bytes, 240);
    EXPECT_EQ(four_frame.info.frames_per_superframe, 4);
    EXPECT_EQ(four_frame.info.frame_samples, 256);
    EXPECT_EQ(four_frame.info.superframe_bytes, 960);
    EXPECT_EQ(four_frame.info.superframe_samples, 1024);

    const auto one_frame = InspectAtrac9Format(SixChannelFormat(0xFE761DE0));
    ASSERT_EQ(one_frame.status, Atrac9FormatStatus::Supported);
    EXPECT_EQ(one_frame.info.channels, 6);
    EXPECT_EQ(one_frame.info.sample_rate, 48000);
    EXPECT_EQ(one_frame.info.frame_bytes, 240);
    EXPECT_EQ(one_frame.info.frames_per_superframe, 1);
    EXPECT_EQ(one_frame.info.frame_samples, 256);
    EXPECT_EQ(one_frame.info.superframe_bytes, 240);
    EXPECT_EQ(one_frame.info.superframe_samples, 256);
}

TEST(Ngs2Atrac9Contract, AcceptsGenericMonoAndStereoConfigsAtLegalRates) {
    const OrbisNgs2WaveformFormat mono{
        .waveformType = Atrac9WaveformType,
        .numChannels = 1,
        .sampleRate = 44100,
        .configData = Atrac9Config(6, 0, 80, 1),
    };
    const auto mono_inspection = InspectAtrac9Format(mono);
    ASSERT_EQ(mono_inspection.status, Atrac9FormatStatus::Supported);
    EXPECT_EQ(mono_inspection.info.channels, 1);
    EXPECT_EQ(mono_inspection.info.sample_rate, 44100);
    EXPECT_EQ(mono_inspection.info.frame_bytes, 80);
    EXPECT_EQ(mono_inspection.info.frames_per_superframe, 2);

    const OrbisNgs2WaveformFormat stereo{
        .waveformType = Atrac9WaveformType,
        .numChannels = 2,
        .sampleRate = 48000,
        .configData = Atrac9Config(7, 2, 160, 0),
    };
    const auto stereo_inspection = InspectAtrac9Format(stereo);
    ASSERT_EQ(stereo_inspection.status, Atrac9FormatStatus::Supported);
    EXPECT_EQ(stereo_inspection.info.channels, 2);
    EXPECT_EQ(stereo_inspection.info.sample_rate, 48000);
    EXPECT_EQ(stereo_inspection.info.frame_bytes, 160);
    EXPECT_EQ(stereo_inspection.info.frames_per_superframe, 1);
}

TEST(Ngs2Atrac9Contract, SharedAudioWrapperInitializesLibAtrac9ForAllCallers) {
    auto config = std::array<u8, ATRAC9_CONFIG_DATA_SIZE>{0xFE, 0x76, 0x1D, 0xF0};
    void* const handle = Atrac9GetHandle();
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(Audio::Atrac9InitDecoderLocked(handle, config.data()), 0);
    Atrac9CodecInfo info{};
    EXPECT_EQ(Atrac9GetCodecInfo(handle, &info), 0);
    EXPECT_EQ(info.channels, 6);
    EXPECT_EQ(info.samplingRate, 48000);
    Atrac9ReleaseHandle(handle);
}

TEST(Ngs2Atrac9Contract, ClonedDecoderHasIndependentRebasedStateAndEquivalentOutput) {
    // Generated mono silence-like frame: minimum legal band/gradient shape, CLC scale-factor
    // mode with zero factors, and the first valid spectrum symbols. This is synthetic test data,
    // not a captured game payload.
    auto config = std::array<u8, ATRAC9_CONFIG_DATA_SIZE>{0xFE, 0x70, 0x03, 0xE0};
    void* const uninitialized = Atrac9GetHandle();
    ASSERT_NE(uninitialized, nullptr);
    EXPECT_EQ(Audio::Atrac9CloneHandleLocked(uninitialized), nullptr);
    Atrac9ReleaseHandle(uninitialized);

    void* const original = Atrac9GetHandle();
    ASSERT_NE(original, nullptr);
    ASSERT_EQ(Audio::Atrac9InitDecoderLocked(original, config.data()), 0);

    std::array<u8, 32> encoded{};
    encoded[4] = 0x04;
    encoded[11] = 0x11;
    std::array<float, 256> checkpoint_pcm{};
    int checkpoint_bytes = 0;
    ASSERT_EQ(Audio::Atrac9DecodeF32Locked(original, encoded.data(),
                                           static_cast<int>(encoded.size()),
                                           checkpoint_pcm.data(), &checkpoint_bytes, 0),
              0);
    ASSERT_GT(checkpoint_bytes, 0);
    ASSERT_LE(checkpoint_bytes, static_cast<int>(encoded.size()));

    void* const clone_a = Audio::Atrac9CloneHandleLocked(original);
    void* const clone_b = Audio::Atrac9CloneHandleLocked(original);
    ASSERT_NE(clone_a, nullptr);
    ASSERT_NE(clone_b, nullptr);
    ASSERT_NE(clone_a, original);
    ASSERT_NE(clone_b, original);

    std::array<float, 256> original_next_pcm{};
    std::array<float, 256> clone_a_pcm{};
    std::array<float, 256> clone_b_pcm{};
    int original_next_bytes = 0;
    int clone_a_bytes = 0;
    int clone_b_bytes = 0;
    ASSERT_EQ(Audio::Atrac9DecodeF32Locked(original, encoded.data(),
                                           static_cast<int>(encoded.size()),
                                           original_next_pcm.data(), &original_next_bytes, 0),
              0);
    Atrac9ReleaseHandle(original);
    ASSERT_EQ(Audio::Atrac9DecodeF32Locked(clone_a, encoded.data(),
                                           static_cast<int>(encoded.size()), clone_a_pcm.data(),
                                           &clone_a_bytes, 0),
              0);
    ASSERT_EQ(Audio::Atrac9DecodeF32Locked(clone_b, encoded.data(),
                                           static_cast<int>(encoded.size()), clone_b_pcm.data(),
                                           &clone_b_bytes, 0),
              0);
    EXPECT_EQ(clone_a_bytes, original_next_bytes);
    EXPECT_EQ(clone_b_bytes, original_next_bytes);
    EXPECT_EQ(clone_a_pcm, original_next_pcm);
    EXPECT_EQ(clone_b_pcm, original_next_pcm);

    Atrac9ReleaseHandle(clone_b);
    Atrac9ReleaseHandle(clone_a);
}

TEST(Ngs2Atrac9Contract, CloneRebasesAllSupportedTopologiesAndOwnsStereoState) {
    for (const u32 channel_config_index : {0U, 2U, 3U, 4U}) {
        auto config = ConfigBytesForTest(Atrac9Config(7, channel_config_index, 32, 0));
        void* const original = Atrac9GetHandle();
        ASSERT_NE(original, nullptr);
        ASSERT_EQ(Audio::Atrac9InitDecoderLocked(original, config.data()), 0);
        void* const clone_a = Audio::Atrac9CloneHandleLocked(original);
        void* const clone_b = Audio::Atrac9CloneHandleLocked(original);
        ASSERT_NE(clone_a, nullptr);
        ASSERT_NE(clone_b, nullptr);

        auto& first = *static_cast<Atrac9Handle*>(clone_a);
        auto& second = *static_cast<Atrac9Handle*>(clone_b);
        ExpectCloneTopologyIsSelfContained(first);
        ExpectCloneTopologyIsSelfContained(second);
        const auto second_rng = second.Frame.Blocks[0].Channels[0].Rng.StateA;
        const auto second_history = second.Frame.Blocks[0].Channels[0].Mdct.ImdctPrevious[0];
        first.Frame.Blocks[0].Channels[0].Rng.StateA ^= 0xFFFFU;
        first.Frame.Blocks[0].Channels[0].Mdct.ImdctPrevious[0] = 123.0;
        EXPECT_EQ(second.Frame.Blocks[0].Channels[0].Rng.StateA, second_rng);
        EXPECT_EQ(second.Frame.Blocks[0].Channels[0].Mdct.ImdctPrevious[0], second_history);

        Atrac9ReleaseHandle(clone_b);
        Atrac9ReleaseHandle(clone_a);
        Atrac9ReleaseHandle(original);
    }
}

TEST(Ngs2Atrac9Contract, FailsClosedForWrongTypeMalformedConfigAndDeclaredFormatMismatch) {
    auto format = SixChannelFormat(0xFE761DF0);
    format.waveformType = 9;
    EXPECT_EQ(InspectAtrac9Format(format).status, Atrac9FormatStatus::UnsupportedWaveform);

    format = SixChannelFormat(0xDEADBEEF);
    EXPECT_EQ(InspectAtrac9Format(format).status, Atrac9FormatStatus::InvalidConfig);

    format = SixChannelFormat(0xF01D76FE);
    EXPECT_EQ(InspectAtrac9Format(format).status, Atrac9FormatStatus::InvalidConfig);

    format = SixChannelFormat((0xFE761DF0U & ~(0x7U << 17U)) | (0x6U << 17U));
    EXPECT_EQ(InspectAtrac9Format(format).status, Atrac9FormatStatus::InvalidConfig);

    format = SixChannelFormat(0xFE761DF0);
    format.numChannels = 2;
    const auto channel_mismatch = InspectAtrac9Format(format);
    EXPECT_EQ(channel_mismatch.status, Atrac9FormatStatus::DeclaredFormatMismatch);
    EXPECT_EQ(channel_mismatch.info.channels, 6);

    format = SixChannelFormat(0xFE761DF0);
    format.sampleRate = 44100;
    EXPECT_EQ(InspectAtrac9Format(format).status, Atrac9FormatStatus::DeclaredFormatMismatch);
}

TEST(Ngs2Atrac9Contract, InspectionWaitsForExclusiveLibAtrac9TableAccess) {
    using namespace std::chrono_literals;

    std::shared_lock decoder_lock{Audio::Atrac9CodecTableMutex()};
    std::promise<void> started;
    auto started_future = started.get_future();
    auto inspection_future = std::async(std::launch::async, [&] {
        started.set_value();
        return InspectAtrac9Format(SixChannelFormat(0xFE761DF0));
    });
    ASSERT_EQ(started_future.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(inspection_future.wait_for(50ms), std::future_status::timeout);

    decoder_lock.unlock();
    ASSERT_EQ(inspection_future.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(inspection_future.get().status, Atrac9FormatStatus::Supported);
}

TEST(Ngs2Atrac9Contract, ValidatesObservedBlockCursorWithoutReadingEncodedPayload) {
    const auto four_frame = InspectAtrac9Format(SixChannelFormat(0xFE761DF0));
    ASSERT_EQ(four_frame.status, Atrac9FormatStatus::Supported);
    const OrbisNgs2WaveformBlock block{
        .dataOffset = 0,
        .dataSize = 130560,
        .numRepeats = 0,
        .numSkipSamples = 256,
        .numSamples = 139008,
    };

    const auto cursor = InspectAtrac9BlockCursor(four_frame.info, block);
    EXPECT_TRUE(cursor.frame_offset_aligned);
    EXPECT_TRUE(cursor.complete_superframes);
    EXPECT_TRUE(cursor.sample_bounds_valid);
    EXPECT_TRUE(cursor.decodable);
    EXPECT_EQ(cursor.data_end, 130560);
    EXPECT_EQ(cursor.encoded_frames, 544);
    EXPECT_EQ(cursor.decoded_samples, 139264);
    EXPECT_EQ(cursor.playable_samples, 139008);

    const auto one_frame = InspectAtrac9Format(SixChannelFormat(0xFE761DE0));
    ASSERT_EQ(one_frame.status, Atrac9FormatStatus::Supported);
    auto second_block = block;
    second_block.dataSize = 131040;
    second_block.numSamples = 139520;
    const auto second_cursor = InspectAtrac9BlockCursor(one_frame.info, second_block);
    EXPECT_TRUE(second_cursor.decodable);
    EXPECT_EQ(second_cursor.encoded_frames, 546);
    EXPECT_EQ(second_cursor.decoded_samples, 139776);
    EXPECT_EQ(second_cursor.playable_samples, 139520);
}

TEST(Ngs2Atrac9Contract, RejectsFrameSplitRepeatAndInexactSampleMetadata) {
    const auto inspection = InspectAtrac9Format(SixChannelFormat(0xFE761DF0));
    ASSERT_EQ(inspection.status, Atrac9FormatStatus::Supported);

    OrbisNgs2WaveformBlock block{
        .dataOffset = 0,
        .dataSize = 960,
        .numSkipSamples = 256,
        .numSamples = 768,
    };
    EXPECT_TRUE(InspectAtrac9BlockCursor(inspection.info, block).decodable);

    block.dataSize = 959;
    const auto partial = InspectAtrac9BlockCursor(inspection.info, block);
    EXPECT_FALSE(partial.complete_frames);
    EXPECT_FALSE(partial.complete_superframes);
    EXPECT_FALSE(partial.decodable);

    block.dataOffset = 240;
    block.dataSize = 240;
    block.numSkipSamples = 0;
    block.numSamples = 256;
    const auto frame_split = InspectAtrac9BlockCursor(inspection.info, block);
    EXPECT_TRUE(frame_split.frame_offset_aligned);
    EXPECT_TRUE(frame_split.complete_frames);
    EXPECT_FALSE(frame_split.complete_superframes);
    EXPECT_FALSE(frame_split.decodable);

    block.dataSize = 960;
    block.numSamples = 1024;
    const auto full_size_at_frame_offset = InspectAtrac9BlockCursor(inspection.info, block);
    EXPECT_FALSE(full_size_at_frame_offset.complete_superframes);
    EXPECT_FALSE(full_size_at_frame_offset.decodable);

    block.dataOffset = 1;
    const auto misaligned = InspectAtrac9BlockCursor(inspection.info, block);
    EXPECT_FALSE(misaligned.frame_offset_aligned);
    EXPECT_FALSE(misaligned.decodable);

    block.dataOffset = 0;
    block.numSkipSamples = 257;
    block.numSamples = 768;
    const auto impossible = InspectAtrac9BlockCursor(inspection.info, block);
    EXPECT_FALSE(impossible.sample_bounds_valid);
    EXPECT_FALSE(impossible.decodable);

    block.numSkipSamples = 0;
    block.numSamples = 1023;
    EXPECT_FALSE(InspectAtrac9BlockCursor(inspection.info, block).decodable);
    block.numSamples = 1024;
    block.numRepeats = 1;
    EXPECT_FALSE(InspectAtrac9BlockCursor(inspection.info, block).decodable);

    const OrbisNgs2WaveformBlock empty{};
    EXPECT_TRUE(InspectAtrac9BlockCursor(inspection.info, empty).decodable);
}

} // namespace
} // namespace Libraries::Ngs2
