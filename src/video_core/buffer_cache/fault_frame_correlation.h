// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "common/types.h"

namespace VideoCore {

enum class FaultFrameCorrelationStatus : u32 {
    Complete,
    NoBatches,
    Incomplete,
    Gap,
};

enum class FaultFrameCorrelationLoss : u32 {
    None = 0,
    DownloadOverflow = 1U << 0,
    PageCapacity = 1U << 1,
    SequenceGap = 1U << 2,
    MissingBatch = 1U << 3,
};

[[nodiscard]] constexpr FaultFrameCorrelationLoss operator|(FaultFrameCorrelationLoss lhs,
                                                             FaultFrameCorrelationLoss rhs) {
    return static_cast<FaultFrameCorrelationLoss>(static_cast<u32>(lhs) | static_cast<u32>(rhs));
}

[[nodiscard]] constexpr FaultFrameCorrelationLoss operator&(FaultFrameCorrelationLoss lhs,
                                                             FaultFrameCorrelationLoss rhs) {
    return static_cast<FaultFrameCorrelationLoss>(static_cast<u32>(lhs) & static_cast<u32>(rhs));
}

struct FaultFrameCorrelationConfiguration {
    bool enabled{};
    u64 first_frame{1};
    u32 frame_count{2048};
    u32 page_cap{4096};
};

struct FaultFrameCorrelationStamp {
    u64 frame_sequence{};
    u64 process_time_us{};
    bool selected{};
};

struct FaultFrameCorrelationObservation {
    u64 frame_sequence{};
    u64 process_time_us{};
    u32 page_count{};
    u32 batch_count{};
    bool stable{};
    bool changed{};
    bool exact_aba{};
    FaultFrameCorrelationStatus status{FaultFrameCorrelationStatus::Complete};
    FaultFrameCorrelationLoss loss{FaultFrameCorrelationLoss::None};
};

struct FaultFrameCorrelationCoverage {
    u32 selected_frames{};
    u32 complete_frames{};
    u32 no_batch_frames{};
    u32 incomplete_frames{};
    u32 gap_frames{};
    u32 stable_frames{};
    u32 changed_frames{};
    u32 exact_aba_frames{};
    u64 total_batches{};
    u64 total_unique_pages{};
    u64 dropped_pages{};

    bool operator==(const FaultFrameCorrelationCoverage&) const = default;
};

/**
 * Bounded, CPU-only reducer for correlating the already-downloaded GPU fault page set with parsed
 * flip frames. Page identities remain private reducer state and never appear in observations.
 */
class FaultFrameCorrelation {
public:
    static constexpr u32 HardMaxFrames = 2048;
    static constexpr u32 HardMaxPagesPerFrame = 4096;

    explicit FaultFrameCorrelation(FaultFrameCorrelationConfiguration config_) : config{config_} {
        config.frame_count = std::min(config.frame_count, HardMaxFrames);
        config.page_cap = std::min(config.page_cap, HardMaxPagesPerFrame);
        if (config.frame_count == 0 || config.page_cap == 0) {
            config.enabled = false;
        }
        frames.resize(config.frame_count);
    }

    void RecordFlip(u64 frame_sequence, u64 process_time_us) {
        std::scoped_lock lock{mutex};
        current_stamp = {};
        if (finished || !IsSelected(frame_sequence)) {
            return;
        }
        auto& frame = frames[static_cast<size_t>(frame_sequence - config.first_frame)];
        if (!frame.seen) {
            frame.seen = true;
            frame.frame_sequence = frame_sequence;
            frame.process_time_us = process_time_us;
            ++coverage.selected_frames;
        }
        current_stamp = {
            .frame_sequence = frame_sequence,
            .process_time_us = frame.process_time_us,
            .selected = true,
        };
    }

    [[nodiscard]] FaultFrameCorrelationStamp CaptureStamp() const {
        std::scoped_lock lock{mutex};
        return current_stamp;
    }

    void ObserveBatch(FaultFrameCorrelationStamp stamp, std::span<const u64> private_page_ids,
                      bool download_overflow) {
        std::scoped_lock lock{mutex};
        if (finished || !stamp.selected || !IsSelected(stamp.frame_sequence)) {
            return;
        }
        auto& frame = frames[static_cast<size_t>(stamp.frame_sequence - config.first_frame)];
        if (!frame.seen || frame.process_time_us != stamp.process_time_us) {
            return;
        }

        ++frame.batch_count;
        ++coverage.total_batches;
        if (download_overflow) {
            frame.loss = frame.loss | FaultFrameCorrelationLoss::DownloadOverflow;
        }

        std::vector<u64> incoming{private_page_ids.begin(), private_page_ids.end()};
        std::ranges::sort(incoming);
        incoming.erase(std::ranges::unique(incoming).begin(), incoming.end());

        std::vector<u64> merged;
        merged.reserve(std::min<size_t>(config.page_cap, frame.private_page_ids.size() +
                                                            incoming.size()));
        std::ranges::set_union(frame.private_page_ids, incoming, std::back_inserter(merged));
        if (merged.size() > config.page_cap) {
            coverage.dropped_pages += merged.size() - config.page_cap;
            merged.resize(config.page_cap);
            frame.loss = frame.loss | FaultFrameCorrelationLoss::PageCapacity;
        }
        frame.private_page_ids = std::move(merged);
    }

    [[nodiscard]] std::vector<FaultFrameCorrelationObservation> Finish() {
        std::scoped_lock lock{mutex};
        if (finished) {
            return observations;
        }
        finished = true;
        current_stamp = {};
        if (!config.enabled) {
            return {};
        }

        const Frame* previous{};
        const Frame* previous_previous{};
        FaultFrameCorrelationStatus previous_status{};
        FaultFrameCorrelationStatus previous_previous_status{};
        for (auto& frame : frames) {
            if (!frame.seen) {
                continue;
            }

            FaultFrameCorrelationObservation observation{
                .frame_sequence = frame.frame_sequence,
                .process_time_us = frame.process_time_us,
                .page_count = static_cast<u32>(frame.private_page_ids.size()),
                .batch_count = frame.batch_count,
                .loss = frame.loss,
            };
            const bool has_gap = previous && previous->frame_sequence + 1 != frame.frame_sequence;
            if (frame.batch_count == 0) {
                observation.status = FaultFrameCorrelationStatus::NoBatches;
                observation.loss = observation.loss | FaultFrameCorrelationLoss::MissingBatch;
                ++coverage.no_batch_frames;
            } else if (frame.loss != FaultFrameCorrelationLoss::None) {
                observation.status = FaultFrameCorrelationStatus::Incomplete;
                ++coverage.incomplete_frames;
            } else if (has_gap) {
                observation.status = FaultFrameCorrelationStatus::Gap;
                observation.loss = observation.loss | FaultFrameCorrelationLoss::SequenceGap;
                ++coverage.gap_frames;
            } else {
                observation.status = FaultFrameCorrelationStatus::Complete;
                ++coverage.complete_frames;
            }

            if (previous && observation.status == FaultFrameCorrelationStatus::Complete &&
                previous_status == FaultFrameCorrelationStatus::Complete &&
                previous->frame_sequence + 1 == frame.frame_sequence) {
                observation.stable = previous->private_page_ids == frame.private_page_ids;
                observation.changed = !observation.stable;
                coverage.stable_frames += observation.stable;
                coverage.changed_frames += observation.changed;
            }
            if (previous_previous && previous &&
                observation.status == FaultFrameCorrelationStatus::Complete &&
                previous_status == FaultFrameCorrelationStatus::Complete &&
                previous_previous_status == FaultFrameCorrelationStatus::Complete &&
                previous_previous->frame_sequence + 1 == previous->frame_sequence &&
                previous->frame_sequence + 1 == frame.frame_sequence) {
                observation.exact_aba =
                    previous_previous->private_page_ids == frame.private_page_ids &&
                    previous_previous->private_page_ids != previous->private_page_ids;
                coverage.exact_aba_frames += observation.exact_aba;
            }
            coverage.total_unique_pages += observation.page_count;
            observations.push_back(observation);
            previous_previous = previous;
            previous_previous_status = previous_status;
            previous = &frame;
            previous_status = observation.status;
        }
        return observations;
    }

    [[nodiscard]] FaultFrameCorrelationCoverage GetCoverage() const {
        std::scoped_lock lock{mutex};
        return coverage;
    }

    [[nodiscard]] FaultFrameCorrelationConfiguration GetConfiguration() const {
        return config;
    }

private:
    struct Frame {
        bool seen{};
        u64 frame_sequence{};
        u64 process_time_us{};
        u32 batch_count{};
        FaultFrameCorrelationLoss loss{FaultFrameCorrelationLoss::None};
        std::vector<u64> private_page_ids;
    };

    [[nodiscard]] bool IsSelected(u64 frame_sequence) const {
        return config.enabled && frame_sequence >= config.first_frame &&
               frame_sequence - config.first_frame < config.frame_count;
    }

    FaultFrameCorrelationConfiguration config;
    mutable std::mutex mutex;
    std::vector<Frame> frames;
    FaultFrameCorrelationStamp current_stamp{};
    FaultFrameCorrelationCoverage coverage{};
    std::vector<FaultFrameCorrelationObservation> observations;
    bool finished{};
};

class FaultFrameCorrelationReportGate {
public:
    void MarkDeferredCallbacksDrained() noexcept {
        drained.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool ClaimAfterDrain() noexcept {
        if (!drained.load(std::memory_order_acquire)) {
            return false;
        }
        bool expected = false;
        return claimed.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
    }

private:
    std::atomic<bool> drained{};
    std::atomic<bool> claimed{};
};

class FaultFrameCorrelationRuntime {
public:
    FaultFrameCorrelationRuntime() : reducer{ReadConfiguration()} {}

    void RecordPatchedFlip(u64 process_time_us) {
        if (!reducer.GetConfiguration().enabled) {
            return;
        }
        const u64 sequence = frame_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
        reducer.RecordFlip(sequence, process_time_us);
    }

    [[nodiscard]] FaultFrameCorrelationStamp CaptureStamp() const {
        return reducer.CaptureStamp();
    }

    void ObserveBatch(FaultFrameCorrelationStamp stamp, std::span<const u64> private_page_ids,
                      bool download_overflow) {
        reducer.ObserveBatch(stamp, private_page_ids, download_overflow);
    }

    void MarkDeferredCallbacksDrained() noexcept {
        report_gate.MarkDeferredCallbacksDrained();
    }

    [[nodiscard]] std::optional<std::vector<FaultFrameCorrelationObservation>>
    FinishAfterDeferredCallbacksDrained() {
        if (!report_gate.ClaimAfterDrain()) {
            return std::nullopt;
        }
        return reducer.Finish();
    }

    [[nodiscard]] FaultFrameCorrelationCoverage GetCoverage() const {
        return reducer.GetCoverage();
    }

    [[nodiscard]] FaultFrameCorrelationConfiguration GetConfiguration() const {
        return reducer.GetConfiguration();
    }

private:
    [[nodiscard]] static u64 ReadU64(const char* name, u64 fallback, u64 maximum) {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0') {
            return fallback;
        }
        char* end{};
        const auto parsed = std::strtoull(value, &end, 10);
        return end != value && *end == '\0' ? std::min(parsed, maximum) : fallback;
    }

    [[nodiscard]] static FaultFrameCorrelationConfiguration ReadConfiguration() {
        const char* enabled_value = std::getenv("SHADPS4_DIAGNOSTIC_FAULT_FRAME_CORRELATION");
        const bool enabled = enabled_value != nullptr && std::string_view{enabled_value} == "1";
        return {
            .enabled = enabled,
            .first_frame = ReadU64("SHADPS4_DIAGNOSTIC_FAULT_FRAME_CORRELATION_FIRST_FRAME", 1,
                                   UINT64_MAX),
            .frame_count = static_cast<u32>(ReadU64(
                "SHADPS4_DIAGNOSTIC_FAULT_FRAME_CORRELATION_FRAME_COUNT", HardMaxFrames,
                FaultFrameCorrelation::HardMaxFrames)),
            .page_cap = static_cast<u32>(ReadU64(
                "SHADPS4_DIAGNOSTIC_FAULT_FRAME_CORRELATION_PAGE_CAP", HardMaxPagesPerFrame,
                FaultFrameCorrelation::HardMaxPagesPerFrame)),
        };
    }

    static constexpr u64 HardMaxFrames = FaultFrameCorrelation::HardMaxFrames;
    static constexpr u64 HardMaxPagesPerFrame = FaultFrameCorrelation::HardMaxPagesPerFrame;
    std::atomic<u64> frame_sequence{};
    FaultFrameCorrelation reducer;
    FaultFrameCorrelationReportGate report_gate;
};

[[nodiscard]] inline FaultFrameCorrelationRuntime& GetFaultFrameCorrelationRuntime() {
    // The renderer is held by a process-global presenter and reports during global teardown. Keep
    // the diagnostic alive until process exit so static destruction order cannot invalidate it.
    static auto* runtime = new FaultFrameCorrelationRuntime;
    return *runtime;
}

} // namespace VideoCore
