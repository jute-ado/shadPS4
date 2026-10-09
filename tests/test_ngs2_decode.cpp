// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "core/libraries/error_codes.h"
#include "core/libraries/ngs2/ngs2.h"
#include "core/libraries/ngs2/ngs2_custom.h"
#include "core/libraries/ngs2/ngs2_decode.h"
#include "core/libraries/ngs2/ngs2_error.h"
#include "core/libraries/ngs2/ngs2_sampler.h"

namespace Libraries::Ngs2 {
namespace {

struct ReadRegion {
    uintptr_t base{};
    size_t size{};
};

struct ReadMap {
    std::vector<ReadRegion> regions;
    u32 reads{};
    OrbisNgs2Handle inspect_voice{};
    bool observed_registry_unlocked{};
};

bool ReadMappedBytes(uintptr_t address, void* destination, size_t size, void* context) {
    auto& map = *static_cast<ReadMap*>(context);
    ++map.reads;
    if (map.inspect_voice != 0) {
        map.observed_registry_unlocked = SnapshotVoiceControl(map.inspect_voice).has_value();
    }
    for (const ReadRegion region : map.regions) {
        if (address < region.base) {
            continue;
        }
        const uintptr_t offset = address - region.base;
        if (offset <= region.size && size <= region.size - offset) {
            std::memcpy(destination, reinterpret_cast<const void*>(address), size);
            return true;
        }
    }
    return false;
}

struct MutatingPayloadRead {
    ReadMap payload;
    OrbisNgs2Handle voice{};
    OrbisNgs2VoicePortVolumeParam volume{};
    bool mutated{};
    s32 mutation_result{ORBIS_NGS2_ERROR_FAIL};
};

bool ReadPayloadAndMutateControl(uintptr_t address, void* destination, size_t size, void* context) {
    auto& mutation = *static_cast<MutatingPayloadRead*>(context);
    if (!mutation.mutated) {
        mutation.mutated = true;
        ReadMap command{{{reinterpret_cast<uintptr_t>(&mutation.volume), sizeof(mutation.volume)}}};
        mutation.mutation_result =
            VoiceControl(mutation.voice, &mutation.volume.header, ReadMappedBytes, &command);
    }
    return ReadMappedBytes(address, destination, size, &mutation.payload);
}

struct FakeDecoderStats {
    u32 factories{};
    u32 clones{};
    u32 decodes{};
    std::optional<u32> fail_at_frame;
    u32 bytes_used_override{};
};

class PatternFrameDecoder final : public Atrac9FrameDecoder {
public:
    PatternFrameDecoder(std::shared_ptr<FakeDecoderStats> stats, Atrac9FormatInfo format,
                        u32 frame_index = 0)
        : stats{std::move(stats)}, format{format}, frame_index{frame_index} {}

    [[nodiscard]] std::unique_ptr<Atrac9FrameDecoder> Clone() const override {
        ++stats->clones;
        return std::make_unique<PatternFrameDecoder>(stats, format, frame_index);
    }

    [[nodiscard]] Atrac9FrameDecodeResult Decode(
        std::span<const u8> encoded, std::span<float> output) override {
        ++stats->decodes;
        if (stats->fail_at_frame && frame_index == *stats->fail_at_frame) {
            return {};
        }
        if (output.size() != static_cast<size_t>(format.channels) * format.frame_samples) {
            return {};
        }
        for (u32 sample = 0; sample < format.frame_samples; ++sample) {
            for (u32 channel = 0; channel < format.channels; ++channel) {
                output[static_cast<size_t>(sample) * format.channels + channel] =
                    static_cast<float>(frame_index * 1000 + sample * 10 + channel);
            }
        }
        ++frame_index;
        const u32 bytes_used = stats->bytes_used_override != 0
                                   ? stats->bytes_used_override
                                   : format.frame_bytes;
        return {.succeeded = bytes_used <= encoded.size(), .bytes_used = bytes_used};
    }

private:
    std::shared_ptr<FakeDecoderStats> stats;
    Atrac9FormatInfo format;
    u32 frame_index{};
};

class ThrowingFrameDecoder final : public Atrac9FrameDecoder {
public:
    [[nodiscard]] std::unique_ptr<Atrac9FrameDecoder> Clone() const override {
        return std::make_unique<ThrowingFrameDecoder>();
    }

    [[nodiscard]] Atrac9FrameDecodeResult Decode(std::span<const u8>,
                                                  std::span<float>) override {
        throw std::runtime_error{"injected decoder failure"};
    }
};

Atrac9FrameDecoderFactory PatternFactory(const std::shared_ptr<FakeDecoderStats>& stats) {
    return [stats](u32, const Atrac9FormatInfo& format) {
        ++stats->factories;
        return std::make_unique<PatternFrameDecoder>(stats, format);
    };
}

Atrac9SamplerSource SmallSource(std::span<const u8> encoded) {
    return {
        .config_data = 0xFE000000,
        .format =
            {
                .channels = 2,
                .sample_rate = 48000,
                .frame_bytes = 2,
                .frames_per_superframe = 1,
                .frame_samples = 4,
                .superframe_bytes = 2,
                .superframe_samples = 4,
            },
        .guest_data_base = reinterpret_cast<uintptr_t>(encoded.data()),
        .blocks =
            {{
                .data_offset = 0,
                .data_size = static_cast<u32>(encoded.size()),
                .num_repeats = 0,
                .num_skip_samples = 1,
                .num_samples = 7,
            }},
    };
}

TEST(Ngs2SamplerDecodeEngine, PreservesExactInterleavedChannelsCursorPendingAndEos) {
    const std::array<u8, 4> encoded{1, 2, 3, 4};
    ReadMap reads{{{reinterpret_cast<uintptr_t>(encoded.data()), encoded.size()}}};
    auto stats = std::make_shared<FakeDecoderStats>();
    SamplerDecodeEngine engine{PatternFactory(stats)};

    auto first = engine.Stage(SmallSource(encoded), 2, 8, ReadMappedBytes, &reads);
    ASSERT_EQ(first.status, SamplerDecodeStatus::Produced);
    EXPECT_EQ(first.channels, 2u);
    EXPECT_EQ(first.samples_per_channel, 2u);
    EXPECT_EQ(first.interleaved_samples, (std::vector<float>{10, 11, 20, 21}));
    EXPECT_EQ(first.cursor.byte_offset, 2u);
    EXPECT_EQ(first.cursor.skip_samples_remaining, 0u);
    EXPECT_EQ(first.cursor.playable_samples_remaining, 4u);
    EXPECT_EQ(first.cursor.buffered_samples, 1u);
    EXPECT_FALSE(first.cursor.end_of_stream);
    ASSERT_TRUE(engine.HasStagedProgress());
    engine.AcceptStaged(7);
    EXPECT_EQ(engine.CommittedRevision(), 7u);

    auto second = engine.Stage(SmallSource(encoded), 5, 8, ReadMappedBytes, &reads);
    ASSERT_EQ(second.status, SamplerDecodeStatus::Produced);
    EXPECT_EQ(second.samples_per_channel, 5u);
    EXPECT_EQ(second.interleaved_samples,
              (std::vector<float>{30, 31, 1000, 1001, 1010, 1011, 1020, 1021, 1030, 1031}));
    EXPECT_TRUE(second.cursor.end_of_stream);
    EXPECT_EQ(second.cursor.block_index, 1u);
    EXPECT_EQ(second.cursor.buffered_samples, 0u);
    engine.AcceptStaged(8);

    const auto eos = engine.Stage(SmallSource(encoded), 1, 8, ReadMappedBytes, &reads);
    EXPECT_EQ(eos.status, SamplerDecodeStatus::EndOfStream);
    EXPECT_TRUE(eos.cursor.end_of_stream);
    EXPECT_FALSE(engine.HasStagedProgress());
    EXPECT_EQ(engine.CommittedRevision(), 8u);
    EXPECT_EQ(stats->factories, 1u);
    EXPECT_EQ(stats->clones, 1u);
    EXPECT_EQ(stats->decodes, 2u);
}

TEST(Ngs2SamplerDecodeEngine, RejectLeavesDecoderCursorAndPendingAtLastCommit) {
    const std::array<u8, 4> encoded{1, 2, 3, 4};
    ReadMap reads{{{reinterpret_cast<uintptr_t>(encoded.data()), encoded.size()}}};
    auto source = SmallSource(encoded);
    source.blocks[0].num_skip_samples = 0;
    source.blocks[0].num_samples = 8;
    auto stats = std::make_shared<FakeDecoderStats>();
    SamplerDecodeEngine engine{PatternFactory(stats)};

    const auto rejected = engine.Stage(source, 2, 8, ReadMappedBytes, &reads);
    ASSERT_EQ(rejected.status, SamplerDecodeStatus::Produced);
    engine.RejectStaged();
    EXPECT_EQ(engine.CommittedCursor(), SamplerDecodeCursor{});
    EXPECT_EQ(engine.CommittedRevision(), 0u);

    const auto retried = engine.Stage(source, 2, 8, ReadMappedBytes, &reads);
    EXPECT_EQ(retried.status, SamplerDecodeStatus::Produced);
    EXPECT_EQ(retried.interleaved_samples, rejected.interleaved_samples);
    EXPECT_EQ(retried.cursor, rejected.cursor);
    EXPECT_EQ(stats->factories, 2u);
}

TEST(Ngs2SamplerDecodeEngine, DecodeFailureDiscardsCandidateDecoderAndAllProgress) {
    const std::array<u8, 4> encoded{1, 2, 3, 4};
    ReadMap reads{{{reinterpret_cast<uintptr_t>(encoded.data()), encoded.size()}}};
    auto source = SmallSource(encoded);
    source.blocks[0].num_skip_samples = 0;
    source.blocks[0].num_samples = 8;
    auto stats = std::make_shared<FakeDecoderStats>();
    stats->fail_at_frame = 1;
    SamplerDecodeEngine engine{PatternFactory(stats)};

    const auto failed = engine.Stage(source, 5, 8, ReadMappedBytes, &reads);
    EXPECT_EQ(failed.status, SamplerDecodeStatus::DecodeFailed);
    EXPECT_TRUE(failed.interleaved_samples.empty());
    EXPECT_FALSE(engine.HasStagedProgress());
    EXPECT_EQ(engine.CommittedCursor(), SamplerDecodeCursor{});

    stats->fail_at_frame.reset();
    const auto retried = engine.Stage(source, 2, 8, ReadMappedBytes, &reads);
    EXPECT_EQ(retried.status, SamplerDecodeStatus::Produced);
    EXPECT_EQ(retried.interleaved_samples, (std::vector<float>{0, 1, 10, 11}));
}

TEST(Ngs2SamplerDecodeEngine, UsesDecoderReportedBytesForTheStagedCursor) {
    const std::array<u8, 4> encoded{1, 2, 3, 4};
    ReadMap reads{{{reinterpret_cast<uintptr_t>(encoded.data()), encoded.size()}}};
    auto source = SmallSource(encoded);
    source.blocks[0].num_skip_samples = 0;
    source.blocks[0].num_samples = 8;
    source.format.frames_per_superframe = 2;
    source.format.superframe_bytes = 4;
    source.format.superframe_samples = 8;
    auto stats = std::make_shared<FakeDecoderStats>();
    stats->bytes_used_override = 1;
    SamplerDecodeEngine engine{PatternFactory(stats)};

    const auto staged = engine.Stage(source, 1, 8, ReadMappedBytes, &reads);
    ASSERT_EQ(staged.status, SamplerDecodeStatus::Produced);
    EXPECT_EQ(staged.cursor.byte_offset, 1u);
    EXPECT_EQ(staged.cursor.decoded_frame_count, 1u);
}

TEST(Ngs2SamplerDecodeEngine, ExplicitEmptyWaveformIsValidEosWithoutReaderOrDecoder) {
    auto stats = std::make_shared<FakeDecoderStats>();
    SamplerDecodeEngine engine{PatternFactory(stats)};
    Atrac9SamplerSource empty{
        .config_data = 0xFE000000,
        .format = {.channels = 2,
                   .sample_rate = 48000,
                   .frame_bytes = 2,
                   .frames_per_superframe = 1,
                   .frame_samples = 4,
                   .superframe_bytes = 2,
                   .superframe_samples = 4},
    };
    ReadMap reads;

    const auto first = engine.Stage(empty, 1, 8, ReadMappedBytes, &reads);
    EXPECT_EQ(first.status, SamplerDecodeStatus::EndOfStream);
    EXPECT_TRUE(first.cursor.end_of_stream);
    ASSERT_TRUE(engine.HasStagedProgress());
    engine.AcceptStaged(3);
    const auto repeated = engine.Stage(empty, 1, 8, ReadMappedBytes, &reads);
    EXPECT_EQ(repeated.status, SamplerDecodeStatus::EndOfStream);
    EXPECT_FALSE(engine.HasStagedProgress());
    EXPECT_EQ(reads.reads, 0u);
    EXPECT_EQ(stats->factories, 0u);
}

TEST(Ngs2SamplerDecodeEngine, LoopMetadataFailsClosedUntilNativeSemanticsAreKnown) {
    const std::array<u8, 2> encoded{1, 2};
    ReadMap reads{{{reinterpret_cast<uintptr_t>(encoded.data()), encoded.size()}}};
    auto source = SmallSource(encoded);
    source.blocks[0].num_repeats = 1;
    source.blocks[0].num_skip_samples = 0;
    source.blocks[0].num_samples = 4;
    auto stats = std::make_shared<FakeDecoderStats>();
    SamplerDecodeEngine engine{PatternFactory(stats)};

    const auto result = engine.Stage(source, 1, 8, ReadMappedBytes, &reads);
    EXPECT_EQ(result.status, SamplerDecodeStatus::UnsupportedLoop);
    EXPECT_EQ(result.cursor, SamplerDecodeCursor{});
    EXPECT_FALSE(engine.HasStagedProgress());
    EXPECT_EQ(reads.reads, 0u);
    EXPECT_EQ(stats->factories, 0u);
}

TEST(Ngs2SamplerDecodeEngine, InvalidBaseOverflowAndRequestCapsFailBeforeGuestReads) {
    const std::array<u8, 2> encoded{1, 2};
    auto source = SmallSource(encoded);
    source.guest_data_base = std::numeric_limits<uintptr_t>::max() - 1;
    source.blocks[0].data_offset = 2;
    source.blocks[0].data_size = 2;
    source.blocks[0].num_skip_samples = 0;
    source.blocks[0].num_samples = 4;
    ReadMap reads;
    auto stats = std::make_shared<FakeDecoderStats>();
    SamplerDecodeEngine engine{PatternFactory(stats)};

    EXPECT_EQ(engine.Stage(source, 1, 8, ReadMappedBytes, &reads).status,
              SamplerDecodeStatus::InvalidWaveform);
    source.guest_data_base = 0;
    source.blocks[0].data_offset = 0;
    EXPECT_EQ(engine.Stage(source, 1, 8, ReadMappedBytes, &reads).status,
              SamplerDecodeStatus::InvalidWaveform);
    source.guest_data_base = reinterpret_cast<uintptr_t>(encoded.data());
    EXPECT_EQ(engine.Stage(source, 0, 8, ReadMappedBytes, &reads).status,
              SamplerDecodeStatus::InvalidRequest);
    EXPECT_EQ(engine.Stage(source, 9, 8, ReadMappedBytes, &reads).status,
              SamplerDecodeStatus::InvalidRequest);
    EXPECT_EQ(reads.reads, 0u);
    EXPECT_EQ(stats->factories, 0u);
}

TEST(Ngs2SamplerDecodeEngine, OversizedSuperframeFailsBeforeGuestReadsOrDecoderCreation) {
    std::vector<u8> encoded(16385);
    auto source = SmallSource(encoded);
    source.format.frame_bytes = static_cast<u32>(encoded.size());
    source.format.superframe_bytes = static_cast<u32>(encoded.size());
    source.blocks[0].num_skip_samples = 0;
    source.blocks[0].num_samples = source.format.frame_samples;
    ReadMap reads{{{reinterpret_cast<uintptr_t>(encoded.data()), encoded.size()}}};
    auto stats = std::make_shared<FakeDecoderStats>();
    SamplerDecodeEngine engine{PatternFactory(stats)};

    const auto result = engine.Stage(source, 1, 8, ReadMappedBytes, &reads);
    EXPECT_EQ(result.status, SamplerDecodeStatus::InvalidFormat);
    EXPECT_EQ(reads.reads, 0u);
    EXPECT_EQ(stats->factories, 0u);
}

TEST(Ngs2SamplerDecodeEngine, MissingOrThrowingDecoderFailsClosedWithoutProgress) {
    const std::array<u8, 2> encoded{1, 2};
    ReadMap reads{{{reinterpret_cast<uintptr_t>(encoded.data()), encoded.size()}}};
    auto source = SmallSource(encoded);
    source.blocks[0].num_skip_samples = 0;
    source.blocks[0].num_samples = 4;

    SamplerDecodeEngine missing{Atrac9FrameDecoderFactory{}};
    SamplerDecodedBlock missing_result;
    EXPECT_NO_THROW(missing_result = missing.Stage(source, 1, 8, ReadMappedBytes, &reads));
    EXPECT_EQ(missing_result.status, SamplerDecodeStatus::DecoderUnavailable);
    EXPECT_FALSE(missing.HasStagedProgress());

    SamplerDecodeEngine throwing_factory{
        [](u32, const Atrac9FormatInfo&) -> std::unique_ptr<Atrac9FrameDecoder> {
            throw std::bad_alloc{};
        }};
    SamplerDecodedBlock factory_result;
    EXPECT_NO_THROW(
        factory_result = throwing_factory.Stage(source, 1, 8, ReadMappedBytes, &reads));
    EXPECT_EQ(factory_result.status, SamplerDecodeStatus::DecoderUnavailable);
    EXPECT_FALSE(throwing_factory.HasStagedProgress());

    SamplerDecodeEngine throwing_decoder{
        [](u32, const Atrac9FormatInfo&) { return std::make_unique<ThrowingFrameDecoder>(); }};
    SamplerDecodedBlock decode_result;
    EXPECT_NO_THROW(
        decode_result = throwing_decoder.Stage(source, 1, 8, ReadMappedBytes, &reads));
    EXPECT_EQ(decode_result.status, SamplerDecodeStatus::DecodeFailed);
    EXPECT_TRUE(decode_result.interleaved_samples.empty());
    EXPECT_FALSE(throwing_decoder.HasStagedProgress());
    EXPECT_EQ(throwing_decoder.CommittedCursor(), SamplerDecodeCursor{});
}

TEST(Ngs2SamplerDecodeEngine, RealLibAtrac9WrapperDecodesAndClonesSyntheticFrames) {
    const OrbisNgs2WaveformFormat native_format{
        .waveformType = Atrac9WaveformType,
        .numChannels = 1,
        .sampleRate = 48000,
        .configData = 0xFE7003E0,
    };
    const auto inspected = InspectAtrac9Format(native_format);
    ASSERT_EQ(inspected.status, Atrac9FormatStatus::Supported);
    ASSERT_EQ(inspected.info.frame_bytes, 32u);
    ASSERT_EQ(inspected.info.frames_per_superframe, 1u);

    // Two generated minimum-band mono frames. Bit 37 selects legal CLC scale factors; byte 11
    // packs the first valid symbols for the final two spectrum codebooks.
    std::array<u8, 64> encoded{};
    encoded[4] = 0x04;
    encoded[11] = 0x11;
    encoded[36] = 0x04;
    encoded[43] = 0x11;
    Atrac9SamplerSource source{
        .config_data = native_format.configData,
        .format = inspected.info,
        .guest_data_base = reinterpret_cast<uintptr_t>(encoded.data()),
        .blocks = {{.data_size = static_cast<u32>(encoded.size()), .num_samples = 512}},
    };
    ReadMap reads{{{reinterpret_cast<uintptr_t>(encoded.data()), encoded.size()}}};
    SamplerDecodeEngine engine{LibAtrac9FrameDecoderFactory()};

    const auto first = engine.Stage(source, 256, 256, ReadMappedBytes, &reads);
    ASSERT_EQ(first.status, SamplerDecodeStatus::Produced);
    EXPECT_EQ(first.channels, 1u);
    EXPECT_EQ(first.samples_per_channel, 256u);
    EXPECT_EQ(first.cursor.byte_offset, 32u);
    EXPECT_TRUE(std::ranges::all_of(first.interleaved_samples,
                                    [](float value) { return std::isfinite(value); }));
    engine.AcceptStaged(2);

    const auto second = engine.Stage(source, 256, 256, ReadMappedBytes, &reads);
    ASSERT_EQ(second.status, SamplerDecodeStatus::Produced);
    EXPECT_EQ(second.samples_per_channel, 256u);
    EXPECT_TRUE(second.cursor.end_of_stream);
    EXPECT_EQ(second.cursor.byte_offset, 0u);
    EXPECT_EQ(second.cursor.block_index, 1u);
    EXPECT_TRUE(std::ranges::all_of(second.interleaved_samples,
                                    [](float value) { return std::isfinite(value); }));
    engine.AcceptStaged(3);
    EXPECT_EQ(engine.Stage(source, 1, 256, ReadMappedBytes, &reads).status,
              SamplerDecodeStatus::EndOfStream);
}

void RecordCommittedRevision(u64 revision, void* context) noexcept {
    *static_cast<u64*>(context) = revision;
}

class Ngs2DecodeRuntimeTest : public testing::Test {
protected:
    void SetUp() override {
        ResetRuntimeForTesting();
    }
    void TearDown() override {
        ResetRuntimeForTesting();
    }
};

TEST_F(Ngs2DecodeRuntimeTest, AppliedSnapshotAndCommitCheckEveryGraphIdentityField) {
    alignas(8) std::array<u8, 64> system_storage{};
    OrbisNgs2ContextBufferInfo system_info{system_storage.data(), system_storage.size()};
    OrbisNgs2Handle system = 0;
    ASSERT_EQ(SystemSetup(nullptr, &system_info, nullptr, &system), ORBIS_OK);

    OrbisNgs2RackOption option{};
    option.size = sizeof(option);
    option.maxVoices = 1;
    option.maxPorts = 1;
    OrbisNgs2ContextBufferInfo rack_info{};
    ASSERT_EQ(RackQueryBufferSize(0x1000, &option, &rack_info), ORBIS_OK);
    std::vector<u8> rack_storage(rack_info.hostBufferSize);
    rack_info.hostBuffer = rack_storage.data();
    OrbisNgs2Handle rack = 0;
    ASSERT_EQ(RackCreate(system, 0x1000, &option, &rack_info, nullptr, &rack), ORBIS_OK);
    OrbisNgs2Handle voice = 0;
    ASSERT_EQ(RackGetVoiceHandle(rack, 0, &voice), ORBIS_OK);

    OrbisNgs2VoiceEventParam play{};
    play.header = {sizeof(play), 0, 6};
    ReadMap command_reads{{{reinterpret_cast<uintptr_t>(&play), sizeof(play)}}};
    ASSERT_EQ(VoiceControl(voice, &play.header, ReadMappedBytes, &command_reads), ORBIS_OK);
    ASSERT_EQ(SystemRender(system, nullptr, 0), ORBIS_OK);

    const auto applied = SnapshotAppliedSamplerVoice(voice);
    ASSERT_TRUE(applied.has_value());
    EXPECT_EQ(applied->playback_state, VoiceRenderPlaybackState::Playing);
    EXPECT_EQ(applied->token.system_handle, system);
    EXPECT_EQ(applied->token.rack_handle, rack);
    EXPECT_EQ(applied->token.voice_handle, voice);
    EXPECT_EQ(applied->token.rack_id, 0x1000u);
    EXPECT_EQ(applied->token.voice_index, 0u);
    ASSERT_NE(applied->decode_runtime, nullptr);

    const auto original = applied->token;
    auto expect_rejected = [&](auto mutate) {
        auto token = original;
        mutate(token);
        u64 called = 0;
        EXPECT_FALSE(CommitVoiceRenderProgress(token, applied->decode_runtime,
                                               RecordCommittedRevision, &called)
                         .has_value());
        EXPECT_EQ(called, 0u);
    };
    expect_rejected([](auto& token) { ++token.system_handle; });
    expect_rejected([](auto& token) { ++token.system_generation; });
    expect_rejected([](auto& token) { ++token.rack_handle; });
    expect_rejected([](auto& token) { ++token.rack_generation; });
    expect_rejected([](auto& token) { ++token.rack_id; });
    expect_rejected([](auto& token) { ++token.voice_handle; });
    expect_rejected([](auto& token) { ++token.voice_generation; });
    expect_rejected([](auto& token) { ++token.voice_index; });
    expect_rejected([](auto& token) { ++token.render_state_revision; });
    u64 called = 0;
    EXPECT_FALSE(CommitVoiceRenderProgress(original, std::make_shared<VoiceDecodeRuntime>(),
                                           RecordCommittedRevision, &called)
                     .has_value());
    EXPECT_EQ(called, 0u);

    const auto committed = CommitVoiceRenderProgress(
        original, applied->decode_runtime, RecordCommittedRevision, &called);
    ASSERT_TRUE(committed.has_value());
    EXPECT_EQ(*committed, original.render_state_revision + 1);
    EXPECT_EQ(called, *committed);
    EXPECT_FALSE(CommitVoiceRenderProgress(original, applied->decode_runtime,
                                           RecordCommittedRevision, &called)
                     .has_value());
}

TEST_F(Ngs2DecodeRuntimeTest,
       DecodeUsesOwnedAppliedSnapshotCheckedReadsAndTransactionalPersistentCursor) {
    std::vector<u8> system_storage(64);
    OrbisNgs2ContextBufferInfo system_info{system_storage.data(), system_storage.size()};
    OrbisNgs2Handle system = 0;
    ASSERT_EQ(SystemSetup(nullptr, &system_info, nullptr, &system), ORBIS_OK);

    OrbisNgs2RackOption option{};
    option.size = sizeof(option);
    option.maxVoices = 1;
    option.maxPorts = 1;
    OrbisNgs2ContextBufferInfo rack_info{};
    ASSERT_EQ(RackQueryBufferSize(0x1000, &option, &rack_info), ORBIS_OK);
    std::vector<u8> rack_storage(rack_info.hostBufferSize);
    rack_info.hostBuffer = rack_storage.data();
    OrbisNgs2Handle rack = 0;
    ASSERT_EQ(RackCreate(system, 0x1000, &option, &rack_info, nullptr, &rack), ORBIS_OK);
    OrbisNgs2Handle voice = 0;
    ASSERT_EQ(RackGetVoiceHandle(rack, 0, &voice), ORBIS_OK);

    std::array<u8, 480> encoded{};
    struct Commands {
        OrbisNgs2SamplerVoiceSetupParam setup{};
        OrbisNgs2SamplerVoiceWaveformBlocksParam waveform{};
        OrbisNgs2VoiceEventParam play{};
        OrbisNgs2WaveformBlock block{};
    } commands;
    commands.setup.header = {
        sizeof(commands.setup),
        static_cast<s16>(offsetof(Commands, waveform) - offsetof(Commands, setup)),
        0x10000000,
    };
    commands.setup.format = {
        .waveformType = Atrac9WaveformType,
        .numChannels = 6,
        .sampleRate = 48000,
        .configData = 0xFE761DE0,
    };
    commands.waveform.header = {
        sizeof(commands.waveform),
        static_cast<s16>(offsetof(Commands, play) - offsetof(Commands, waveform)),
        0x10000001,
    };
    commands.waveform.data = encoded.data();
    commands.waveform.numBlocks = 1;
    commands.waveform.aBlock = &commands.block;
    commands.play.header = {sizeof(commands.play), 0, 6};
    commands.play.eventId = 0;
    commands.block = {
        .dataOffset = 0,
        .dataSize = static_cast<u32>(encoded.size()),
        .numRepeats = 0,
        .numSkipSamples = 1,
        .numSamples = 511,
    };
    ReadMap controls{{{reinterpret_cast<uintptr_t>(&commands), sizeof(commands)}}};
    ASSERT_EQ(VoiceControl(voice, &commands.setup.header, ReadMappedBytes, &controls), ORBIS_OK);
    ASSERT_EQ(SystemRender(system, nullptr, 0), ORBIS_OK);
    const auto before_decode = SnapshotAppliedSamplerVoice(voice);
    ASSERT_TRUE(before_decode.has_value());
    ASSERT_EQ(before_decode->playback_state, VoiceRenderPlaybackState::Playing);

    auto stats = std::make_shared<FakeDecoderStats>();
    ReadMap payload{{{reinterpret_cast<uintptr_t>(encoded.data()), encoded.size()}}, 0, voice};
    const auto first = DecodeAtrac9SamplerVoiceWithFactoryForTesting(
        voice, 2, ReadMappedBytes, &payload, PatternFactory(stats));
    ASSERT_EQ(first.status, SamplerDecodeStatus::Produced);
    EXPECT_EQ(first.channels, 6u);
    EXPECT_EQ(first.samples_per_channel, 2u);
    EXPECT_EQ(first.interleaved_samples,
              (std::vector<float>{10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25}));
    EXPECT_TRUE(payload.observed_registry_unlocked);
    EXPECT_EQ(first.committed_render_state_revision,
              before_decode->token.render_state_revision + 1);
    EXPECT_EQ(first.cursor.byte_offset, 240u);
    EXPECT_EQ(first.cursor.buffered_samples, 253u);

    MutatingPayloadRead mutation{
        .payload = {{{reinterpret_cast<uintptr_t>(encoded.data()), encoded.size()}}, 0, voice},
        .voice = voice,
    };
    mutation.volume.header = {sizeof(mutation.volume), 0, 2};
    mutation.volume.port = 0;
    mutation.volume.level = 0.5F;
    const auto stale = DecodeAtrac9SamplerVoiceWithFactoryForTesting(
        voice, 256, ReadPayloadAndMutateControl, &mutation, PatternFactory(stats));
    EXPECT_TRUE(mutation.mutated);
    EXPECT_EQ(mutation.mutation_result, ORBIS_OK);
    EXPECT_TRUE(mutation.payload.observed_registry_unlocked);
    EXPECT_EQ(stale.status, SamplerDecodeStatus::StaleSnapshot);
    EXPECT_TRUE(stale.interleaved_samples.empty());
    EXPECT_EQ(stale.cursor, first.cursor);
    const auto after_mutation = SnapshotAppliedSamplerVoice(voice);
    ASSERT_TRUE(after_mutation.has_value());
    EXPECT_EQ(after_mutation->token.render_state_revision,
              first.committed_render_state_revision + 1);

    payload.reads = 0;
    payload.observed_registry_unlocked = false;
    const auto retried = DecodeAtrac9SamplerVoiceWithFactoryForTesting(
        voice, 256, ReadMappedBytes, &payload, PatternFactory(stats));
    ASSERT_EQ(retried.status, SamplerDecodeStatus::Produced);
    ASSERT_EQ(retried.interleaved_samples.size(), 256u * 6u);
    EXPECT_EQ(std::vector<float>(retried.interleaved_samples.begin(),
                                 retried.interleaved_samples.begin() + 6),
              (std::vector<float>{30, 31, 32, 33, 34, 35}));
    const auto second_frame = retried.interleaved_samples.begin() + 253 * 6;
    EXPECT_EQ(std::vector<float>(second_frame, second_frame + 6),
              (std::vector<float>{1000, 1001, 1002, 1003, 1004, 1005}));
    EXPECT_TRUE(retried.cursor.end_of_stream);
    EXPECT_EQ(retried.cursor.buffered_samples, 253u);
    EXPECT_EQ(retried.committed_render_state_revision,
              after_mutation->token.render_state_revision + 1);
    EXPECT_EQ(stats->factories, 1u);
    EXPECT_EQ(stats->clones, 2u);
    EXPECT_EQ(stats->decodes, 3u);

    const auto drained = DecodeAtrac9SamplerVoiceWithFactoryForTesting(
        voice, 253, ReadMappedBytes, &payload, PatternFactory(stats));
    ASSERT_EQ(drained.status, SamplerDecodeStatus::Produced);
    EXPECT_EQ(drained.samples_per_channel, 253u);
    EXPECT_TRUE(drained.cursor.end_of_stream);
    EXPECT_EQ(drained.cursor.buffered_samples, 0u);
    EXPECT_EQ(drained.committed_render_state_revision,
              retried.committed_render_state_revision + 1);
    const auto eos_before = SnapshotAppliedSamplerVoice(voice);
    ASSERT_TRUE(eos_before.has_value());
    const auto eos = DecodeAtrac9SamplerVoiceWithFactoryForTesting(
        voice, 1, ReadMappedBytes, &payload, PatternFactory(stats));
    EXPECT_EQ(eos.status, SamplerDecodeStatus::EndOfStream);
    EXPECT_TRUE(eos.interleaved_samples.empty());
    EXPECT_EQ(eos.cursor, drained.cursor);
    EXPECT_EQ(eos.committed_render_state_revision,
              drained.committed_render_state_revision);
    const auto eos_after = SnapshotAppliedSamplerVoice(voice);
    ASSERT_TRUE(eos_after.has_value());
    EXPECT_EQ(eos_after->token.render_state_revision,
              eos_before->token.render_state_revision);
    EXPECT_EQ(stats->clones, 3u);

    // Reissuing byte-identical source controls is still a new sampler source lifetime.
    controls.reads = 0;
    ASSERT_EQ(VoiceControl(voice, &commands.setup.header, ReadMappedBytes, &controls), ORBIS_OK);
    ASSERT_EQ(SystemRender(system, nullptr, 0), ORBIS_OK);
    const auto identical_restart = DecodeAtrac9SamplerVoiceWithFactoryForTesting(
        voice, 1, ReadMappedBytes, &payload, PatternFactory(stats));
    ASSERT_EQ(identical_restart.status, SamplerDecodeStatus::Produced);
    EXPECT_EQ(identical_restart.interleaved_samples,
              (std::vector<float>{10, 11, 12, 13, 14, 15}));
    EXPECT_EQ(stats->factories, 2u);

    OrbisNgs2VoiceEventParam kill{};
    kill.header = {sizeof(kill), 0, 6};
    kill.eventId = 3;
    ReadMap kill_read{{{reinterpret_cast<uintptr_t>(&kill), sizeof(kill)}}};
    ASSERT_EQ(VoiceControl(voice, &kill.header, ReadMappedBytes, &kill_read), ORBIS_OK);
    ASSERT_EQ(SystemRender(system, nullptr, 0), ORBIS_OK);
    const auto killed = SnapshotAppliedSamplerVoice(voice);
    ASSERT_TRUE(killed.has_value());
    const auto cursor_before_not_playing = identical_restart.cursor;
    const auto not_playing = DecodeAtrac9SamplerVoiceWithFactoryForTesting(
        voice, 1, ReadMappedBytes, &payload, PatternFactory(stats));
    EXPECT_EQ(not_playing.status, SamplerDecodeStatus::NotPlaying);
    const auto killed_after = SnapshotAppliedSamplerVoice(voice);
    ASSERT_TRUE(killed_after.has_value());
    EXPECT_EQ(killed_after->token.render_state_revision, killed->token.render_state_revision);
    {
        std::scoped_lock lock{killed->decode_runtime->mutex};
        ASSERT_NE(killed->decode_runtime->engine, nullptr);
        EXPECT_EQ(killed->decode_runtime->engine->CommittedCursor(), cursor_before_not_playing);
    }

    OrbisNgs2VoiceEventParam play{};
    play.header = {sizeof(play), 0, 6};
    play.eventId = 0;
    ReadMap play_read{{{reinterpret_cast<uintptr_t>(&play), sizeof(play)}}};
    ASSERT_EQ(VoiceControl(voice, &play.header, ReadMappedBytes, &play_read), ORBIS_OK);
    ASSERT_EQ(SystemRender(system, nullptr, 0), ORBIS_OK);
    const auto replayed = DecodeAtrac9SamplerVoiceWithFactoryForTesting(
        voice, 1, ReadMappedBytes, &payload, PatternFactory(stats));
    ASSERT_EQ(replayed.status, SamplerDecodeStatus::Produced);
    EXPECT_EQ(replayed.interleaved_samples,
              (std::vector<float>{10, 11, 12, 13, 14, 15}));
    EXPECT_EQ(stats->factories, 3u);
}

} // namespace
} // namespace Libraries::Ngs2
