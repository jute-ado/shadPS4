// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "core/libraries/ngs2/ngs2_atrac9.h"
#include "core/libraries/ngs2/ngs2_impl.h"

namespace Libraries::Ngs2 {

enum class SamplerDecodeStatus : u8 {
    Produced,
    EndOfStream,
    NotPlaying,
    NotConfigured,
    InvalidRequest,
    UnsupportedFormat,
    InvalidFormat,
    InvalidWaveform,
    UnsupportedLoop,
    GuestReadFailed,
    DecoderUnavailable,
    DecodeFailed,
    WorkLimitExceeded,
    StaleSnapshot,
};

struct SamplerDecodeCursor {
    u32 block_index{};
    u32 repeat_index{};
    u32 byte_offset{};
    u64 skip_samples_remaining{};
    u64 playable_samples_remaining{};
    u64 decoded_frame_count{};
    u32 buffered_samples{};
    bool end_of_stream{};

    auto operator<=>(const SamplerDecodeCursor&) const = default;
};

struct Atrac9SamplerSource {
    u64 source_epoch{};
    u32 config_data{};
    Atrac9FormatInfo format{};
    uintptr_t guest_data_base{};
    std::vector<WaveformBlockSnapshot> blocks;

    bool operator==(const Atrac9SamplerSource&) const = default;
};

struct SamplerDecodedBlock {
    SamplerDecodeStatus status{SamplerDecodeStatus::InvalidRequest};
    u32 channels{};
    u32 sample_rate{};
    u32 samples_per_channel{};
    std::vector<float> interleaved_samples;
    SamplerDecodeCursor cursor{};
    u64 committed_render_state_revision{};
};

struct Atrac9FrameDecodeResult {
    bool succeeded{};
    u32 bytes_used{};
};

class Atrac9FrameDecoder {
public:
    virtual ~Atrac9FrameDecoder() = default;
    [[nodiscard]] virtual std::unique_ptr<Atrac9FrameDecoder> Clone() const = 0;
    [[nodiscard]] virtual Atrac9FrameDecodeResult Decode(std::span<const u8> encoded,
                                                         std::span<float> interleaved_output) = 0;
};

using Atrac9FrameDecoderFactory = std::function<std::unique_ptr<Atrac9FrameDecoder>(
    u32 config_data, const Atrac9FormatInfo& format)>;

[[nodiscard]] Atrac9FrameDecoderFactory LibAtrac9FrameDecoderFactory();

// Transactional scheduler for one sampler voice. Stage never changes the committed cursor or
// decoder. AcceptStaged installs the staged decoder/cursor without allocation; RejectStaged
// discards it. Callers provide a complete-range checked guest reader.
class SamplerDecodeEngine {
public:
    explicit SamplerDecodeEngine(Atrac9FrameDecoderFactory decoder_factory);
    ~SamplerDecodeEngine();
    SamplerDecodeEngine(SamplerDecodeEngine&&) noexcept;
    SamplerDecodeEngine& operator=(SamplerDecodeEngine&&) noexcept;
    SamplerDecodeEngine(const SamplerDecodeEngine&) = delete;
    SamplerDecodeEngine& operator=(const SamplerDecodeEngine&) = delete;

    [[nodiscard]] SamplerDecodedBlock Stage(const Atrac9SamplerSource& source,
                                            u32 requested_samples, u32 max_samples,
                                            OrbisNgs2RuntimeReadHandler read_handler,
                                            void* read_context) noexcept;
    void AcceptStaged(u64 committed_revision) noexcept;
    void RejectStaged() noexcept;
    [[nodiscard]] bool HasStagedProgress() const noexcept;
    [[nodiscard]] SamplerDecodeCursor CommittedCursor() const noexcept;
    [[nodiscard]] u64 CommittedRevision() const noexcept;

private:
    [[nodiscard]] SamplerDecodedBlock StageUnchecked(const Atrac9SamplerSource& source,
                                                     u32 requested_samples, u32 max_samples,
                                                     OrbisNgs2RuntimeReadHandler read_handler,
                                                     void* read_context);

    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct VoiceDecodeRuntime {
    std::mutex mutex;
    std::unique_ptr<SamplerDecodeEngine> engine;
};

// Stages one applied sampler snapshot without committing it. The caller must hold
// snapshot.decode_runtime->mutex until the staged state is either accepted or rejected.
[[nodiscard]] SamplerDecodedBlock StageAtrac9SamplerVoiceLocked(
    const AppliedSamplerVoiceSnapshot& snapshot, u32 requested_samples,
    OrbisNgs2RuntimeReadHandler read_handler, void* read_context,
    Atrac9FrameDecoderFactory decoder_factory);

// Produces one transactional per-voice float block. The system renderer composes this boundary
// with routing and mixing; standalone callers still receive only the decoded block.
[[nodiscard]] SamplerDecodedBlock DecodeAtrac9SamplerVoice(OrbisNgs2Handle voice_handle,
                                                           u32 requested_samples,
                                                           OrbisNgs2RuntimeReadHandler read_handler,
                                                           void* read_context);

// Internal deterministic seam used by the CPU suite. It is identical to the production path
// except for the injected frame-decoder factory.
[[nodiscard]] SamplerDecodedBlock DecodeAtrac9SamplerVoiceWithFactoryForTesting(
    OrbisNgs2Handle voice_handle, u32 requested_samples, OrbisNgs2RuntimeReadHandler read_handler,
    void* read_context, Atrac9FrameDecoderFactory decoder_factory);

} // namespace Libraries::Ngs2
