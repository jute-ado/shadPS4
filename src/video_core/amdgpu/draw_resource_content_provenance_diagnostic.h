// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "common/types.h"

namespace AmdGpu {

enum class DrawResourceContentProbeMode : u8 {
    FullProvenance,
    StagedOnly,
};

enum class DrawResourceUploadProvenance : u8 {
    Coherent,
    ConcurrentWriterStagedBefore,
    ConcurrentWriterStagedAfter,
    ConcurrentWriterOther,
    StagedMismatch,
};

[[nodiscard]] constexpr DrawResourceUploadProvenance ClassifyDrawResourceUpload(
    u64 before, u64 staged, u64 after) noexcept {
    if (before == after) {
        return staged == before ? DrawResourceUploadProvenance::Coherent
                                : DrawResourceUploadProvenance::StagedMismatch;
    }
    if (staged == before) {
        return DrawResourceUploadProvenance::ConcurrentWriterStagedBefore;
    }
    if (staged == after) {
        return DrawResourceUploadProvenance::ConcurrentWriterStagedAfter;
    }
    return DrawResourceUploadProvenance::ConcurrentWriterOther;
}

struct DrawResourceContentOrdinal {
    u32 draw_ordinal{};
    u32 resource_ordinal{};
};

struct DrawResourceContentProvenanceSnapshot {
    static constexpr u32 MaxReportedResources = 16;

    bool should_report{};
    u64 sequence{};
    u32 draws{};
    u32 observations{};
    u32 bytes_probed{};
    u32 coherent_changed_resources{};
    u32 reported_coherent_changed_resources{};
    std::array<DrawResourceContentOrdinal, MaxReportedResources>
        first_coherent_changed_resources{};
    u32 coherent_content_aba_resources{};
    u32 reported_coherent_content_aba_resources{};
    std::array<DrawResourceContentOrdinal, MaxReportedResources>
        first_coherent_content_aba_resources{};
    u32 staged_changed_resources{};
    u32 reported_staged_changed_resources{};
    std::array<DrawResourceContentOrdinal, MaxReportedResources>
        first_staged_changed_resources{};
    u32 staged_content_aba_resources{};
    u32 reported_staged_content_aba_resources{};
    std::array<DrawResourceContentOrdinal, MaxReportedResources>
        first_staged_content_aba_resources{};
    u32 concurrent_write_resources{};
    u32 endpoint_capture_resources{};
    u32 staged_mismatch_resources{};
    u32 resident_unobserved_resources{};
    u32 truncated_resources{};
    u64 truncated_bytes{};
};

/**
 * Bounded, diagnostic-only classification of CPU upload coherence and temporal equality.
 * Tokens remain process-local; snapshots expose only counts and logical draw/resource ordinals.
 */
class DrawResourceContentProvenanceDiagnostic {
public:
    static constexpr u32 MaxObservationsPerFrame = 16384;
    static constexpr u32 MaxProbeBytesPerFrame = 4_MB;
    static constexpr u32 MaxFullProbeBytes = 16_KB;

    explicit DrawResourceContentProvenanceDiagnostic(u64 report_limit_)
        : report_limit{report_limit_}, capture_count{report_limit_},
          current_frame{std::make_unique<Frame>()},
          previous_frame{std::make_unique<Frame>()},
          previous_previous_frame{std::make_unique<Frame>()} {}

    void ConfigureCaptureWindow(u64 start, u64 count) noexcept {
        capture_start = start;
        capture_count = count;
    }

    void ConfigureProbeMode(DrawResourceContentProbeMode mode) noexcept {
        probe_mode = mode;
    }

    void ConfigureProbeByteLimit(u32 limit) noexcept {
        max_probe_bytes_per_frame =
            limit < MaxProbeBytesPerFrame ? limit : MaxProbeBytesPerFrame;
    }

    [[nodiscard]] DrawResourceContentProbeMode GetProbeMode() const noexcept {
        return probe_mode;
    }

    void BeginDraw() noexcept {
        if (draw_active) {
            AbortDraw();
        }
        if (!IsInCaptureWindow(reports_emitted + 1)) {
            return;
        }
        draw_active = true;
        draw_observation_begin = current_observations;
        current_resource_ordinal = 0;
    }

    void EndDraw() noexcept {
        if (!draw_active) {
            return;
        }
        ++current_draws;
        draw_active = false;
    }

    void AbortDraw() noexcept {
        if (!draw_active) {
            return;
        }
        current_observations = draw_observation_begin;
        RecountCurrentProbeBytes();
        current_resource_ordinal = 0;
        draw_active = false;
    }

    [[nodiscard]] bool CanProbeCpuUpload(u32 size) const noexcept {
        return draw_active && current_observations < MaxObservationsPerFrame &&
               size <= MaxFullProbeBytes && current_probe_bytes <= max_probe_bytes_per_frame &&
               size <= max_probe_bytes_per_frame - current_probe_bytes;
    }

    [[nodiscard]] bool IsDrawActive() const noexcept {
        return draw_active;
    }

    [[nodiscard]] u64 FingerprintBytes(const void* data, size_t size) const noexcept {
        u64 hash = HashSeed;
        const auto* bytes = static_cast<const u8*>(data);
        for (size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    void RecordCpuUpload(u64 before, u64 staged, u64 after, u32 size) noexcept {
        Observation observation{
            .key = CurrentKey(),
            .token = staged,
            .size = size,
            .kind = ObservationKind::CpuUpload,
            .provenance = ClassifyDrawResourceUpload(before, staged, after),
        };
        if (Append(observation)) {
            current_probe_bytes += size;
        }
        ++current_resource_ordinal;
    }

    void RecordStagedUpload(u64 staged, u32 size) noexcept {
        if (Append({
                .key = CurrentKey(),
                .token = staged,
                .size = size,
                .kind = ObservationKind::StagedOnly,
            })) {
            current_probe_bytes += size;
        }
        ++current_resource_ordinal;
    }

    void RecordResidentUnobserved() noexcept {
        Append(Observation{
            .key = CurrentKey(),
            .kind = ObservationKind::ResidentUnobserved,
        });
        ++current_resource_ordinal;
    }

    void RecordTruncated(u32 size) noexcept {
        Append(Observation{
            .key = CurrentKey(),
            .size = size,
            .kind = ObservationKind::Truncated,
        });
        ++current_resource_ordinal;
    }

    [[nodiscard]] DrawResourceContentProvenanceSnapshot TakeSnapshot() noexcept {
        if (draw_active) {
            AbortDraw();
        }
        if (reports_emitted >= report_limit) {
            ResetCurrent();
            return {};
        }

        const u64 sequence = ++reports_emitted;
        if (!IsInCaptureWindow(sequence)) {
            ResetCurrent();
            return {.sequence = sequence};
        }

        DrawResourceContentProvenanceSnapshot snapshot{
            .should_report = true,
            .sequence = sequence,
            .draws = current_draws,
            .observations = current_observations,
            .bytes_probed = current_probe_bytes,
            .truncated_resources = overflow_resources,
            .truncated_bytes = overflow_bytes,
        };

        for (u32 index = 0; index < current_observations; ++index) {
            const auto& current = (*current_frame)[index];
            switch (current.kind) {
            case ObservationKind::CpuUpload: {
                if (current.provenance == DrawResourceUploadProvenance::Coherent) {
                    const auto* previous = MatchingAt(*previous_frame, previous_observations,
                                                      index, current.key);
                    if (previous != nullptr && IsCoherent(*previous) &&
                        current.token != previous->token) {
                        ++snapshot.coherent_changed_resources;
                        Report(current.key, snapshot.reported_coherent_changed_resources,
                               snapshot.first_coherent_changed_resources);
                    }
                    const auto* previous_previous =
                        MatchingAt(*previous_previous_frame, previous_previous_observations, index,
                                   current.key);
                    if (previous != nullptr && previous_previous != nullptr &&
                        IsCoherent(*previous) && IsCoherent(*previous_previous) &&
                        current.token == previous_previous->token &&
                        current.token != previous->token) {
                        ++snapshot.coherent_content_aba_resources;
                        Report(current.key, snapshot.reported_coherent_content_aba_resources,
                               snapshot.first_coherent_content_aba_resources);
                    }
                    break;
                }
                if (current.provenance == DrawResourceUploadProvenance::StagedMismatch) {
                    ++snapshot.staged_mismatch_resources;
                    break;
                }
                ++snapshot.concurrent_write_resources;
                if (current.provenance ==
                        DrawResourceUploadProvenance::ConcurrentWriterStagedBefore ||
                    current.provenance ==
                        DrawResourceUploadProvenance::ConcurrentWriterStagedAfter) {
                    ++snapshot.endpoint_capture_resources;
                }
                break;
            }
            case ObservationKind::StagedOnly: {
                const auto* previous = MatchingAt(*previous_frame, previous_observations, index,
                                                  current.key);
                if (previous != nullptr && IsStagedOnly(*previous) &&
                    current.token != previous->token) {
                    ++snapshot.staged_changed_resources;
                    Report(current.key, snapshot.reported_staged_changed_resources,
                           snapshot.first_staged_changed_resources);
                }
                const auto* previous_previous =
                    MatchingAt(*previous_previous_frame, previous_previous_observations, index,
                               current.key);
                if (previous != nullptr && previous_previous != nullptr &&
                    IsStagedOnly(*previous) && IsStagedOnly(*previous_previous) &&
                    current.token == previous_previous->token &&
                    current.token != previous->token) {
                    ++snapshot.staged_content_aba_resources;
                    Report(current.key, snapshot.reported_staged_content_aba_resources,
                           snapshot.first_staged_content_aba_resources);
                }
                break;
            }
            case ObservationKind::ResidentUnobserved:
                ++snapshot.resident_unobserved_resources;
                break;
            case ObservationKind::Truncated:
                ++snapshot.truncated_resources;
                snapshot.truncated_bytes += current.size;
                break;
            }
        }

        *previous_previous_frame = *previous_frame;
        previous_previous_observations = previous_observations;
        *previous_frame = *current_frame;
        previous_observations = current_observations;
        ResetCurrent();
        return snapshot;
    }

private:
    enum class ObservationKind : u8 {
        CpuUpload,
        StagedOnly,
        ResidentUnobserved,
        Truncated,
    };

    struct Observation {
        DrawResourceContentOrdinal key{};
        u64 token{};
        u32 size{};
        ObservationKind kind{};
        DrawResourceUploadProvenance provenance{};
    };

    using Frame = std::array<Observation, MaxObservationsPerFrame>;

    static constexpr u64 HashSeed = 14695981039346656037ULL;

    [[nodiscard]] DrawResourceContentOrdinal CurrentKey() const noexcept {
        return {current_draws, current_resource_ordinal};
    }

    [[nodiscard]] bool IsInCaptureWindow(u64 sequence) const noexcept {
        return sequence >= capture_start && sequence - capture_start < capture_count;
    }

    bool Append(const Observation& observation) noexcept {
        if (!draw_active) {
            return false;
        }
        if (current_observations >= MaxObservationsPerFrame) {
            ++overflow_resources;
            overflow_bytes += observation.size;
            return false;
        }
        (*current_frame)[current_observations++] = observation;
        return true;
    }

    [[nodiscard]] static const Observation* MatchingAt(
        const Frame& frame, u32 observations, u32 index,
        DrawResourceContentOrdinal key) noexcept {
        if (index >= observations || frame[index].key.draw_ordinal != key.draw_ordinal ||
            frame[index].key.resource_ordinal != key.resource_ordinal) {
            return nullptr;
        }
        return &frame[index];
    }

    [[nodiscard]] static bool IsCoherent(const Observation& observation) noexcept {
        return observation.kind == ObservationKind::CpuUpload &&
               observation.provenance == DrawResourceUploadProvenance::Coherent;
    }

    [[nodiscard]] static bool IsStagedOnly(const Observation& observation) noexcept {
        return observation.kind == ObservationKind::StagedOnly;
    }

    static void Report(
        DrawResourceContentOrdinal key, u32& reported,
        std::array<DrawResourceContentOrdinal,
                   DrawResourceContentProvenanceSnapshot::MaxReportedResources>& output) noexcept {
        if (reported < output.size()) {
            output[reported++] = key;
        }
    }

    void RecountCurrentProbeBytes() noexcept {
        current_probe_bytes = 0;
        for (u32 index = 0; index < current_observations; ++index) {
            if ((*current_frame)[index].kind == ObservationKind::CpuUpload ||
                (*current_frame)[index].kind == ObservationKind::StagedOnly) {
                current_probe_bytes += (*current_frame)[index].size;
            }
        }
    }

    void ResetCurrent() noexcept {
        *current_frame = {};
        current_draws = 0;
        current_observations = 0;
        current_probe_bytes = 0;
        current_resource_ordinal = 0;
        draw_observation_begin = 0;
        overflow_resources = 0;
        overflow_bytes = 0;
        draw_active = false;
    }

    const u64 report_limit;
    u64 reports_emitted{};
    u64 capture_start{1};
    u64 capture_count{};
    std::unique_ptr<Frame> current_frame;
    std::unique_ptr<Frame> previous_frame;
    std::unique_ptr<Frame> previous_previous_frame;
    u64 overflow_bytes{};
    u32 current_draws{};
    u32 current_observations{};
    u32 previous_observations{};
    u32 previous_previous_observations{};
    u32 current_probe_bytes{};
    u32 current_resource_ordinal{};
    u32 draw_observation_begin{};
    u32 overflow_resources{};
    bool draw_active{};
    DrawResourceContentProbeMode probe_mode{DrawResourceContentProbeMode::FullProvenance};
    u32 max_probe_bytes_per_frame{MaxProbeBytesPerFrame};
};

} // namespace AmdGpu
