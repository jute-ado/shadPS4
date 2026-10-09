// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <future>
#include <mutex>

#include <gtest/gtest.h>

#include "core/libraries/audio/atrac9_codec_lock.h"
#include "core/libraries/ngs2/ngs2_atrac9.h"

namespace Libraries::Ngs2 {
namespace {

constexpr OrbisNgs2WaveformFormat ShadowSixChannelFormat(u32 codec_config) {
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

TEST(Ngs2Atrac9Contract, AcceptsObservedSixChannelConfigsAndReportsExactFrameShape) {
    const auto four_frame = InspectAtrac9Format(ShadowSixChannelFormat(0xFE761DF0));
    ASSERT_EQ(four_frame.status, Atrac9FormatStatus::Supported);
    EXPECT_EQ(four_frame.info.channels, 6);
    EXPECT_EQ(four_frame.info.sample_rate, 48000);
    EXPECT_EQ(four_frame.info.frame_bytes, 240);
    EXPECT_EQ(four_frame.info.frames_per_superframe, 4);
    EXPECT_EQ(four_frame.info.frame_samples, 256);
    EXPECT_EQ(four_frame.info.superframe_bytes, 960);
    EXPECT_EQ(four_frame.info.superframe_samples, 1024);

    const auto one_frame = InspectAtrac9Format(ShadowSixChannelFormat(0xFE761DE0));
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

TEST(Ngs2Atrac9Contract, FailsClosedForWrongTypeMalformedConfigAndDeclaredFormatMismatch) {
    auto format = ShadowSixChannelFormat(0xFE761DF0);
    format.waveformType = 9;
    EXPECT_EQ(InspectAtrac9Format(format).status, Atrac9FormatStatus::UnsupportedWaveform);

    format = ShadowSixChannelFormat(0xDEADBEEF);
    EXPECT_EQ(InspectAtrac9Format(format).status, Atrac9FormatStatus::InvalidConfig);

    format = ShadowSixChannelFormat(0xF01D76FE);
    EXPECT_EQ(InspectAtrac9Format(format).status, Atrac9FormatStatus::InvalidConfig);

    format = ShadowSixChannelFormat((0xFE761DF0U & ~(0x7U << 17U)) | (0x6U << 17U));
    EXPECT_EQ(InspectAtrac9Format(format).status, Atrac9FormatStatus::InvalidConfig);

    format = ShadowSixChannelFormat(0xFE761DF0);
    format.numChannels = 2;
    const auto channel_mismatch = InspectAtrac9Format(format);
    EXPECT_EQ(channel_mismatch.status, Atrac9FormatStatus::DeclaredFormatMismatch);
    EXPECT_EQ(channel_mismatch.info.channels, 6);

    format = ShadowSixChannelFormat(0xFE761DF0);
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
        return InspectAtrac9Format(ShadowSixChannelFormat(0xFE761DF0));
    });
    ASSERT_EQ(started_future.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(inspection_future.wait_for(50ms), std::future_status::timeout);

    decoder_lock.unlock();
    ASSERT_EQ(inspection_future.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(inspection_future.get().status, Atrac9FormatStatus::Supported);
}

TEST(Ngs2Atrac9Contract, ValidatesObservedBlockCursorWithoutReadingEncodedPayload) {
    const auto four_frame = InspectAtrac9Format(ShadowSixChannelFormat(0xFE761DF0));
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

    const auto one_frame = InspectAtrac9Format(ShadowSixChannelFormat(0xFE761DE0));
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

TEST(Ngs2Atrac9Contract, AcceptsFrameSplitBlocksButRejectsPartialFramesAndImpossibleSamples) {
    const auto inspection = InspectAtrac9Format(ShadowSixChannelFormat(0xFE761DF0));
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
    EXPECT_TRUE(frame_split.decodable);

    block.dataSize = 960;
    block.numSamples = 1024;
    const auto full_size_at_frame_offset = InspectAtrac9BlockCursor(inspection.info, block);
    EXPECT_FALSE(full_size_at_frame_offset.complete_superframes);
    EXPECT_TRUE(full_size_at_frame_offset.decodable);

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
}

} // namespace
} // namespace Libraries::Ngs2
