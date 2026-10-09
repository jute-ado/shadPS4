// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/ngs2/ngs2_decode.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

#include "core/libraries/audio/atrac9_codec_lock.h"

extern "C" {
#include <libatrac9.h>
}

namespace Libraries::Ngs2 {
namespace {

constexpr size_t MaxSamplerDecodeBlocks = 4096;
constexpr u32 MaxFramesPerDecodeTransaction = 4096;
constexpr u32 MaxAtrac9FrameBytes = 2048;
constexpr u32 MaxAtrac9FramesPerSuperframe = 8;
constexpr u32 MaxAtrac9SuperframeBytes = MaxAtrac9FrameBytes * MaxAtrac9FramesPerSuperframe;

std::array<u8, ATRAC9_CONFIG_DATA_SIZE> ConfigBytes(u32 config) noexcept {
    return {
        static_cast<u8>(config >> 24U),
        static_cast<u8>(config >> 16U),
        static_cast<u8>(config >> 8U),
        static_cast<u8>(config),
    };
}

bool AddAddress(uintptr_t base, u64 offset, uintptr_t* result) noexcept {
    if (!result || offset > std::numeric_limits<uintptr_t>::max() ||
        base > std::numeric_limits<uintptr_t>::max() - static_cast<uintptr_t>(offset)) {
        return false;
    }
    *result = base + static_cast<uintptr_t>(offset);
    return true;
}

bool FormatIsSane(const Atrac9FormatInfo& format) noexcept {
    if (format.channels == 0 || format.channels > ORBIS_NGS2_MAX_VOICE_CHANNELS ||
        format.sample_rate == 0 || format.frame_bytes == 0 || format.frames_per_superframe == 0 ||
        format.frame_samples == 0 || format.frame_samples > 256 || format.superframe_bytes == 0 ||
        format.superframe_samples == 0 || format.frame_bytes > MaxAtrac9FrameBytes ||
        format.frames_per_superframe > MaxAtrac9FramesPerSuperframe ||
        (format.frames_per_superframe & (format.frames_per_superframe - 1U)) != 0 ||
        format.superframe_bytes > MaxAtrac9SuperframeBytes) {
        return false;
    }
    return static_cast<u64>(format.frame_bytes) * format.frames_per_superframe ==
               format.superframe_bytes &&
           static_cast<u64>(format.frame_samples) * format.frames_per_superframe ==
               format.superframe_samples;
}

class LibAtrac9FrameDecoder final : public Atrac9FrameDecoder {
public:
    LibAtrac9FrameDecoder(void* handle, Atrac9FormatInfo format) : handle{handle}, format{format} {}

    ~LibAtrac9FrameDecoder() override {
        Atrac9ReleaseHandle(handle);
    }

    [[nodiscard]] std::unique_ptr<Atrac9FrameDecoder> Clone() const override {
        std::unique_ptr<void, decltype(&Atrac9ReleaseHandle)> cloned{
            Audio::Atrac9CloneHandleLocked(handle), Atrac9ReleaseHandle};
        if (!cloned) {
            return nullptr;
        }
        auto result = std::make_unique<LibAtrac9FrameDecoder>(cloned.get(), format);
        cloned.release();
        return result;
    }

    [[nodiscard]] Atrac9FrameDecodeResult Decode(std::span<const u8> encoded,
                                                 std::span<float> interleaved_output) override {
        const u64 required_output = static_cast<u64>(format.channels) * format.frame_samples;
        if (!handle || encoded.empty() ||
            encoded.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            required_output != interleaved_output.size()) {
            return {};
        }
        std::vector<float> scratch(interleaved_output.size());
        int bytes_used = 0;
        const int result =
            Audio::Atrac9DecodeF32Locked(handle, encoded.data(), static_cast<int>(encoded.size()),
                                         scratch.data(), &bytes_used, 0);
        if (result != 0 || bytes_used <= 0 || static_cast<size_t>(bytes_used) > encoded.size()) {
            return {};
        }
        std::ranges::copy(scratch, interleaved_output.begin());
        return {.succeeded = true, .bytes_used = static_cast<u32>(bytes_used)};
    }

private:
    void* handle{};
    Atrac9FormatInfo format{};
};

std::unique_ptr<Atrac9FrameDecoder> CreateLibAtrac9FrameDecoder(
    u32 config_data, const Atrac9FormatInfo& expected_format) {
    if (!FormatIsSane(expected_format)) {
        return nullptr;
    }
    std::unique_ptr<void, decltype(&Atrac9ReleaseHandle)> handle{Atrac9GetHandle(),
                                                                 Atrac9ReleaseHandle};
    if (!handle) {
        return nullptr;
    }
    auto config = ConfigBytes(config_data);
    if (Audio::Atrac9InitDecoderLocked(handle.get(), config.data()) != 0) {
        return nullptr;
    }
    Atrac9CodecInfo info{};
    if (Atrac9GetCodecInfo(handle.get(), &info) != 0 || info.channels <= 0 ||
        info.samplingRate <= 0 || info.framesInSuperframe <= 0 || info.frameSamples <= 0 ||
        info.superframeSize <= 0) {
        return nullptr;
    }
    const u32 frames = static_cast<u32>(info.framesInSuperframe);
    const Atrac9FormatInfo actual{
        .channels = static_cast<u32>(info.channels),
        .sample_rate = static_cast<u32>(info.samplingRate),
        .frame_bytes = static_cast<u32>(info.superframeSize) / frames,
        .frames_per_superframe = frames,
        .frame_samples = static_cast<u32>(info.frameSamples),
        .superframe_bytes = static_cast<u32>(info.superframeSize),
        .superframe_samples = static_cast<u32>(info.frameSamples) * frames,
    };
    if (actual != expected_format ||
        std::memcmp(info.configData, config.data(), config.size()) != 0) {
        return nullptr;
    }
    auto result = std::make_unique<LibAtrac9FrameDecoder>(handle.get(), actual);
    handle.release();
    return result;
}

struct SourcePreflight {
    SamplerDecodeStatus status{SamplerDecodeStatus::InvalidWaveform};
    bool has_encoded_data{};
};

SourcePreflight PreflightSource(const Atrac9SamplerSource& source) noexcept {
    if (!FormatIsSane(source.format) || source.blocks.size() > MaxSamplerDecodeBlocks) {
        return {.status = SamplerDecodeStatus::InvalidFormat};
    }
    bool has_data = false;
    for (const auto& block : source.blocks) {
        if (block.num_repeats != 0) {
            return {.status = SamplerDecodeStatus::UnsupportedLoop};
        }
        if (block.data_size == 0) {
            if (block.num_skip_samples != 0 || block.num_samples != 0) {
                return {.status = SamplerDecodeStatus::InvalidWaveform};
            }
            continue;
        }
        has_data = true;
        if (source.guest_data_base == 0 ||
            block.data_offset % source.format.superframe_bytes != 0 ||
            block.data_size % source.format.superframe_bytes != 0) {
            return {.status = SamplerDecodeStatus::InvalidWaveform};
        }
        const u64 block_end = static_cast<u64>(block.data_offset) + block.data_size;
        uintptr_t ignored = 0;
        if (!AddAddress(source.guest_data_base, block_end, &ignored)) {
            return {.status = SamplerDecodeStatus::InvalidWaveform};
        }
        const u64 decoded_samples =
            (static_cast<u64>(block.data_size) / source.format.superframe_bytes) *
            source.format.superframe_samples;
        if (static_cast<u64>(block.num_skip_samples) + block.num_samples != decoded_samples) {
            return {.status = SamplerDecodeStatus::InvalidWaveform};
        }
    }
    return {.status = SamplerDecodeStatus::Produced, .has_encoded_data = has_data};
}

bool NormalizeCursor(const Atrac9SamplerSource& source, SamplerDecodeCursor& cursor) noexcept {
    while (cursor.block_index < source.blocks.size()) {
        const auto& block = source.blocks[cursor.block_index];
        if (block.data_size == 0) {
            ++cursor.block_index;
            cursor.byte_offset = 0;
            continue;
        }
        if (cursor.byte_offset == 0 && cursor.skip_samples_remaining == 0 &&
            cursor.playable_samples_remaining == 0) {
            cursor.skip_samples_remaining = block.num_skip_samples;
            cursor.playable_samples_remaining = block.num_samples;
        }
        if (cursor.byte_offset < block.data_size) {
            return true;
        }
        if (cursor.byte_offset != block.data_size || cursor.skip_samples_remaining != 0 ||
            cursor.playable_samples_remaining != 0) {
            return false;
        }
        ++cursor.block_index;
        cursor.byte_offset = 0;
        cursor.skip_samples_remaining = 0;
        cursor.playable_samples_remaining = 0;
    }
    cursor.end_of_stream = true;
    return true;
}

void SetBufferedSamples(SamplerDecodeCursor& cursor, const std::vector<float>& pending,
                        u32 channels) noexcept {
    cursor.buffered_samples = channels == 0 ? 0 : static_cast<u32>(pending.size() / channels);
}

void DrainPending(std::vector<float>& pending, u32 channels, u32 requested_samples,
                  std::vector<float>& output) {
    if (pending.empty() || channels == 0) {
        return;
    }
    const size_t produced_samples = output.size() / channels;
    const size_t wanted_samples = requested_samples - produced_samples;
    const size_t available_samples = pending.size() / channels;
    const size_t take_values = std::min(wanted_samples, available_samples) * channels;
    output.insert(output.end(), pending.begin(), pending.begin() + take_values);
    pending.erase(pending.begin(), pending.begin() + take_values);
}

Atrac9SamplerSource SourceFromSnapshot(const SamplerControlSnapshot& sampler,
                                       const Atrac9FormatInfo& format, u64 source_epoch) {
    return {
        .source_epoch = source_epoch,
        .config_data = sampler.format->config_data,
        .format = format,
        .guest_data_base = sampler.guest_data_token,
        .blocks = sampler.waveform_blocks,
    };
}

VoiceRenderPlaybackState PlaybackStateFromSnapshot(VoiceRenderPlaybackState state) noexcept {
    return state;
}

} // namespace

struct SamplerDecodeEngine::Impl {
    struct State {
        std::optional<Atrac9SamplerSource> source;
        SamplerDecodeCursor cursor{};
        std::vector<float> pending;
        std::unique_ptr<Atrac9FrameDecoder> decoder;
        u64 revision{};
    };

    Atrac9FrameDecoderFactory factory;
    State committed;
    std::optional<State> staged;
};

Atrac9FrameDecoderFactory LibAtrac9FrameDecoderFactory() {
    return CreateLibAtrac9FrameDecoder;
}

SamplerDecodeEngine::SamplerDecodeEngine(Atrac9FrameDecoderFactory decoder_factory)
    : impl{std::make_unique<Impl>()} {
    impl->factory = std::move(decoder_factory);
}

SamplerDecodeEngine::~SamplerDecodeEngine() = default;
SamplerDecodeEngine::SamplerDecodeEngine(SamplerDecodeEngine&&) noexcept = default;
SamplerDecodeEngine& SamplerDecodeEngine::operator=(SamplerDecodeEngine&&) noexcept = default;

SamplerDecodedBlock SamplerDecodeEngine::Stage(const Atrac9SamplerSource& source,
                                               u32 requested_samples, u32 max_samples,
                                               OrbisNgs2RuntimeReadHandler read_handler,
                                               void* read_context) noexcept {
    try {
        return StageUnchecked(source, requested_samples, max_samples, read_handler, read_context);
    } catch (const std::bad_alloc&) {
        impl->staged.reset();
        return {
            .status = SamplerDecodeStatus::DecoderUnavailable,
            .channels = source.format.channels,
            .sample_rate = source.format.sample_rate,
            .cursor = impl->committed.cursor,
            .committed_render_state_revision = impl->committed.revision,
        };
    } catch (...) {
        impl->staged.reset();
        return {
            .status = SamplerDecodeStatus::DecodeFailed,
            .channels = source.format.channels,
            .sample_rate = source.format.sample_rate,
            .cursor = impl->committed.cursor,
            .committed_render_state_revision = impl->committed.revision,
        };
    }
}

SamplerDecodedBlock SamplerDecodeEngine::StageUnchecked(const Atrac9SamplerSource& source,
                                                        u32 requested_samples, u32 max_samples,
                                                        OrbisNgs2RuntimeReadHandler read_handler,
                                                        void* read_context) {
    impl->staged.reset();
    SamplerDecodedBlock result{
        .channels = source.format.channels,
        .sample_rate = source.format.sample_rate,
        .committed_render_state_revision = impl->committed.revision,
    };
    if (requested_samples == 0 || max_samples == 0 || requested_samples > max_samples) {
        result.status = SamplerDecodeStatus::InvalidRequest;
        result.cursor = impl->committed.cursor;
        return result;
    }
    const SourcePreflight preflight = PreflightSource(source);
    if (preflight.status != SamplerDecodeStatus::Produced) {
        result.status = preflight.status;
        result.cursor = impl->committed.cursor;
        return result;
    }

    Impl::State candidate;
    const bool same_source = impl->committed.source && *impl->committed.source == source;
    candidate.source = source;
    if (same_source) {
        candidate.cursor = impl->committed.cursor;
        candidate.pending = impl->committed.pending;
        candidate.revision = impl->committed.revision;
    }
    result.interleaved_samples.reserve(static_cast<size_t>(requested_samples) *
                                       source.format.channels);
    DrainPending(candidate.pending, source.format.channels, requested_samples,
                 result.interleaved_samples);
    if (!NormalizeCursor(source, candidate.cursor)) {
        result.status = SamplerDecodeStatus::InvalidWaveform;
        result.interleaved_samples.clear();
        result.cursor = impl->committed.cursor;
        return result;
    }

    const bool needs_decoder =
        result.interleaved_samples.size() / source.format.channels < requested_samples &&
        !candidate.cursor.end_of_stream;
    if (needs_decoder) {
        if (!read_handler) {
            result.status = SamplerDecodeStatus::GuestReadFailed;
            result.interleaved_samples.clear();
            result.cursor = impl->committed.cursor;
            return result;
        }
        if (!same_source && !impl->factory) {
            result.status = SamplerDecodeStatus::DecoderUnavailable;
            result.interleaved_samples.clear();
            result.cursor = impl->committed.cursor;
            return result;
        }
        try {
            candidate.decoder = same_source && impl->committed.decoder
                                    ? impl->committed.decoder->Clone()
                                    : impl->factory(source.config_data, source.format);
        } catch (...) {
            result.status = SamplerDecodeStatus::DecoderUnavailable;
            result.interleaved_samples.clear();
            result.cursor = impl->committed.cursor;
            return result;
        }
        if (!candidate.decoder) {
            result.status = SamplerDecodeStatus::DecoderUnavailable;
            result.interleaved_samples.clear();
            result.cursor = impl->committed.cursor;
            return result;
        }
    } else if (same_source && impl->committed.decoder &&
               (candidate.cursor != impl->committed.cursor ||
                candidate.pending != impl->committed.pending)) {
        try {
            candidate.decoder = impl->committed.decoder->Clone();
        } catch (...) {
            result.status = SamplerDecodeStatus::DecoderUnavailable;
            result.interleaved_samples.clear();
            result.cursor = impl->committed.cursor;
            return result;
        }
        if (!candidate.decoder) {
            result.status = SamplerDecodeStatus::DecoderUnavailable;
            result.interleaved_samples.clear();
            result.cursor = impl->committed.cursor;
            return result;
        }
    }

    u32 decoded_in_transaction = 0;
    while (result.interleaved_samples.size() / source.format.channels < requested_samples &&
           !candidate.cursor.end_of_stream) {
        if (++decoded_in_transaction > MaxFramesPerDecodeTransaction) {
            result.status = SamplerDecodeStatus::WorkLimitExceeded;
            result.interleaved_samples.clear();
            result.cursor = impl->committed.cursor;
            return result;
        }
        const auto& block = source.blocks[candidate.cursor.block_index];
        const u32 in_superframe = candidate.cursor.byte_offset % source.format.superframe_bytes;
        const u32 encoded_size = source.format.superframe_bytes - in_superframe;
        const u64 payload_offset =
            static_cast<u64>(block.data_offset) + candidate.cursor.byte_offset;
        uintptr_t payload_address = 0;
        if (candidate.cursor.byte_offset > block.data_size ||
            encoded_size > block.data_size - candidate.cursor.byte_offset ||
            !AddAddress(source.guest_data_base, payload_offset, &payload_address)) {
            result.status = SamplerDecodeStatus::InvalidWaveform;
            result.interleaved_samples.clear();
            result.cursor = impl->committed.cursor;
            return result;
        }
        std::vector<u8> encoded(encoded_size);
        if (!read_handler(payload_address, encoded.data(), encoded.size(), read_context)) {
            result.status = SamplerDecodeStatus::GuestReadFailed;
            result.interleaved_samples.clear();
            result.cursor = impl->committed.cursor;
            return result;
        }
        std::vector<float> frame_pcm(static_cast<size_t>(source.format.channels) *
                                     source.format.frame_samples);
        const auto decoded = candidate.decoder->Decode(encoded, frame_pcm);
        if (!decoded.succeeded || decoded.bytes_used == 0 || decoded.bytes_used > encoded.size()) {
            result.status = SamplerDecodeStatus::DecodeFailed;
            result.interleaved_samples.clear();
            result.cursor = impl->committed.cursor;
            return result;
        }

        const u32 next_frame_in_superframe = static_cast<u32>(
            (candidate.cursor.decoded_frame_count + 1) % source.format.frames_per_superframe);
        const u64 consumed_end =
            static_cast<u64>(candidate.cursor.byte_offset) + decoded.bytes_used;
        if (consumed_end > block.data_size ||
            (next_frame_in_superframe != 0 &&
             consumed_end >= static_cast<u64>(candidate.cursor.byte_offset) - in_superframe +
                                 source.format.superframe_bytes)) {
            result.status = SamplerDecodeStatus::DecodeFailed;
            result.interleaved_samples.clear();
            result.cursor = impl->committed.cursor;
            return result;
        }
        if (next_frame_in_superframe == 0) {
            const u64 superframe_end = static_cast<u64>(candidate.cursor.byte_offset) -
                                       in_superframe + source.format.superframe_bytes;
            if (consumed_end > superframe_end || superframe_end > block.data_size) {
                result.status = SamplerDecodeStatus::DecodeFailed;
                result.interleaved_samples.clear();
                result.cursor = impl->committed.cursor;
                return result;
            }
            candidate.cursor.byte_offset = static_cast<u32>(superframe_end);
        } else {
            candidate.cursor.byte_offset = static_cast<u32>(consumed_end);
        }
        ++candidate.cursor.decoded_frame_count;

        const u32 skipped = static_cast<u32>(
            std::min<u64>(candidate.cursor.skip_samples_remaining, source.format.frame_samples));
        candidate.cursor.skip_samples_remaining -= skipped;
        const u32 available = source.format.frame_samples - skipped;
        const u32 playable =
            static_cast<u32>(std::min<u64>(candidate.cursor.playable_samples_remaining, available));
        candidate.cursor.playable_samples_remaining -= playable;
        const auto begin =
            frame_pcm.begin() + static_cast<size_t>(skipped) * source.format.channels;
        const auto end = begin + static_cast<size_t>(playable) * source.format.channels;
        candidate.pending.insert(candidate.pending.end(), begin, end);
        DrainPending(candidate.pending, source.format.channels, requested_samples,
                     result.interleaved_samples);
        if (!NormalizeCursor(source, candidate.cursor)) {
            result.status = SamplerDecodeStatus::InvalidWaveform;
            result.interleaved_samples.clear();
            result.cursor = impl->committed.cursor;
            return result;
        }
    }

    SetBufferedSamples(candidate.cursor, candidate.pending, source.format.channels);
    result.samples_per_channel =
        static_cast<u32>(result.interleaved_samples.size() / source.format.channels);
    result.cursor = candidate.cursor;
    result.status = result.samples_per_channel != 0 ? SamplerDecodeStatus::Produced
                                                    : SamplerDecodeStatus::EndOfStream;

    const bool changed = !same_source || candidate.cursor != impl->committed.cursor ||
                         candidate.pending != impl->committed.pending;
    if (changed) {
        impl->staged = std::move(candidate);
    }
    return result;
}

void SamplerDecodeEngine::AcceptStaged(u64 committed_revision) noexcept {
    if (!impl->staged) {
        return;
    }
    impl->staged->revision = committed_revision;
    impl->committed = std::move(*impl->staged);
    impl->staged.reset();
}

void SamplerDecodeEngine::RejectStaged() noexcept {
    impl->staged.reset();
}

bool SamplerDecodeEngine::HasStagedProgress() const noexcept {
    return impl->staged.has_value();
}

SamplerDecodeCursor SamplerDecodeEngine::CommittedCursor() const noexcept {
    return impl->committed.cursor;
}

u64 SamplerDecodeEngine::CommittedRevision() const noexcept {
    return impl->committed.revision;
}

SamplerDecodedBlock StageAtrac9SamplerVoiceLocked(const AppliedSamplerVoiceSnapshot& snapshot,
                                                  u32 requested_samples,
                                                  OrbisNgs2RuntimeReadHandler read_handler,
                                                  void* read_context,
                                                  Atrac9FrameDecoderFactory decoder_factory) {
    if (PlaybackStateFromSnapshot(snapshot.playback_state) != VoiceRenderPlaybackState::Playing) {
        return {.status = SamplerDecodeStatus::NotPlaying};
    }
    if (!snapshot.sampler.format || !snapshot.sampler.waveform_configured) {
        return {.status = SamplerDecodeStatus::NotConfigured};
    }
    const auto& format_snapshot = *snapshot.sampler.format;
    if (format_snapshot.frame_offset != 0 || format_snapshot.frame_margin != 0 ||
        snapshot.sampler.setup_flags != 0 || snapshot.sampler.waveform_flags != 0) {
        return {.status = SamplerDecodeStatus::InvalidWaveform};
    }
    OrbisNgs2WaveformFormat native_format{
        .waveformType = format_snapshot.waveform_type,
        .numChannels = format_snapshot.num_channels,
        .sampleRate = format_snapshot.sample_rate,
        .configData = format_snapshot.config_data,
        .frameOffset = format_snapshot.frame_offset,
        .frameMargin = format_snapshot.frame_margin,
    };
    const auto inspected = InspectAtrac9Format(native_format);
    if (inspected.status == Atrac9FormatStatus::UnsupportedWaveform) {
        return {.status = SamplerDecodeStatus::UnsupportedFormat};
    }
    if (inspected.status != Atrac9FormatStatus::Supported ||
        inspected.info.sample_rate != snapshot.system_sample_rate) {
        return {.status = SamplerDecodeStatus::InvalidFormat};
    }
    if (!snapshot.decode_runtime) {
        return {.status = SamplerDecodeStatus::DecoderUnavailable};
    }
    if (!snapshot.decode_runtime->engine) {
        snapshot.decode_runtime->engine =
            std::make_unique<SamplerDecodeEngine>(std::move(decoder_factory));
    }
    return snapshot.decode_runtime->engine->Stage(
        SourceFromSnapshot(snapshot.sampler, inspected.info, snapshot.sampler_source_epoch),
        requested_samples, snapshot.max_grain_samples, read_handler, read_context);
}

namespace {

void AcceptEngineProgress(u64 revision, void* context) noexcept {
    static_cast<SamplerDecodeEngine*>(context)->AcceptStaged(revision);
}

SamplerDecodedBlock DecodeAtrac9SamplerVoiceImpl(OrbisNgs2Handle voice_handle,
                                                 u32 requested_samples,
                                                 OrbisNgs2RuntimeReadHandler read_handler,
                                                 void* read_context,
                                                 Atrac9FrameDecoderFactory decoder_factory) {
    const auto snapshot = SnapshotAppliedSamplerVoice(voice_handle);
    if (!snapshot) {
        return {.status = SamplerDecodeStatus::NotConfigured};
    }
    if (!snapshot->decode_runtime) {
        return {.status = SamplerDecodeStatus::DecoderUnavailable};
    }

    std::scoped_lock decode_lock{snapshot->decode_runtime->mutex};
    auto result = StageAtrac9SamplerVoiceLocked(*snapshot, requested_samples, read_handler,
                                                read_context, std::move(decoder_factory));
    if (!snapshot->decode_runtime->engine) {
        return result;
    }
    auto& engine = *snapshot->decode_runtime->engine;
    if (!engine.HasStagedProgress()) {
        return result;
    }
    const auto committed = CommitVoiceRenderProgress(snapshot->token, snapshot->decode_runtime,
                                                     AcceptEngineProgress, &engine);
    if (!committed) {
        engine.RejectStaged();
        result.status = SamplerDecodeStatus::StaleSnapshot;
        result.samples_per_channel = 0;
        result.interleaved_samples.clear();
        result.cursor = engine.CommittedCursor();
        result.committed_render_state_revision = engine.CommittedRevision();
        return result;
    }
    result.committed_render_state_revision = *committed;
    return result;
}

} // namespace

SamplerDecodedBlock DecodeAtrac9SamplerVoice(OrbisNgs2Handle voice_handle, u32 requested_samples,
                                             OrbisNgs2RuntimeReadHandler read_handler,
                                             void* read_context) {
    return DecodeAtrac9SamplerVoiceImpl(voice_handle, requested_samples, read_handler, read_context,
                                        LibAtrac9FrameDecoderFactory());
}

SamplerDecodedBlock DecodeAtrac9SamplerVoiceWithFactoryForTesting(
    OrbisNgs2Handle voice_handle, u32 requested_samples, OrbisNgs2RuntimeReadHandler read_handler,
    void* read_context, Atrac9FrameDecoderFactory decoder_factory) {
    return DecodeAtrac9SamplerVoiceImpl(voice_handle, requested_samples, read_handler, read_context,
                                        std::move(decoder_factory));
}

} // namespace Libraries::Ngs2
