// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "video_core/buffer_cache/srt_snapshot_timing.h"

namespace VideoCore {

struct SrtSessionReadCounts {
    std::uint64_t read_calls{};
    std::uint64_t new_pages{};
    std::uint64_t cache_hits{};
    std::uint64_t requested_bytes{};
    bool saturated{};
};

class SrtSessionReadCounter {
public:
    explicit SrtSessionReadCounter(bool enabled = false) : enabled{enabled} {}

    void Read(std::uint64_t size) noexcept {
        if (enabled) {
            AddSrtSnapshotTiming(counts.read_calls, 1, counts.saturated);
            AddSrtSnapshotTiming(counts.requested_bytes, size, counts.saturated);
        }
    }

    void NewPage() noexcept {
        if (enabled) {
            AddSrtSnapshotTiming(counts.new_pages, 1, counts.saturated);
        }
    }

    void CacheHit() noexcept {
        if (enabled) {
            AddSrtSnapshotTiming(counts.cache_hits, 1, counts.saturated);
        }
    }

    [[nodiscard]] const SrtSessionReadCounts& Counts() const noexcept {
        return counts;
    }

private:
    bool enabled{};
    SrtSessionReadCounts counts;
};

struct SrtSessionStageCounts {
    std::uint64_t sessions{};
    std::uint64_t selections{};
    std::uint64_t retries{};
    std::uint64_t invalid_sessions{};
    std::uint64_t read_calls{};
    std::uint64_t new_pages{};
    std::uint64_t cache_hits{};
    std::uint64_t requested_bytes{};
    std::uint64_t max_pages_per_session{};
    std::uint64_t total_us{};
    std::uint64_t max_session_us{};
};

struct SrtSessionDiagnosticWindow {
    static constexpr std::size_t StageCount = 7;
    bool truncated{};
    bool saturated{};
    std::uint32_t presented_frame{};
    std::uint64_t begin_us{};
    std::uint64_t end_us{};
    SrtSessionStageCounts totals;
    std::array<SrtSessionStageCounts, StageCount> stages{};
};

struct SrtSessionDiagnosticToken {
    std::uint32_t stage{};
    bool first_attempt{};
    std::uint64_t begin_us{};
};

// Owner-thread only; final partial window is intentionally not emitted by destruction.
class SrtSessionDiagnostics {
public:
    static constexpr std::uint32_t MaximumRows = 256;

    explicit SrtSessionDiagnostics(bool enabled = false, std::uint32_t start_frame = 0)
        : config{enabled, start_frame} {}
    explicit SrtSessionDiagnostics(SrtSnapshotTimingConfig config_) : config{config_} {}

    template <typename Clock, typename Emit>
    std::optional<SrtSessionDiagnosticToken> BeginAttempt(std::uint32_t frame, std::uint32_t stage,
                                                          std::uint32_t attempt, Clock&& clock,
                                                          Emit&& emit) {
        if (!config.enabled || truncated || frame < config.start_frame ||
            stage >= SrtSessionDiagnosticWindow::StageCount || attempt == 0) {
            return std::nullopt;
        }
        if (active && frame != window.presented_frame) {
            std::invoke(emit, window);
            ++emitted_rows;
            active = false;
            if (emitted_rows == MaximumRows) {
                truncated = true;
                std::invoke(emit, SrtSessionDiagnosticWindow{
                                      .truncated = true,
                                      .presented_frame = frame,
                                  });
                return std::nullopt;
            }
        }
        const auto now = static_cast<std::uint64_t>(std::invoke(clock));
        if (!active) {
            window = {.presented_frame = frame, .begin_us = now, .end_us = now};
            active = true;
        }
        return SrtSessionDiagnosticToken{stage, attempt == 1, now};
    }

    template <typename Clock>
    void CompleteSession(const SrtSessionDiagnosticToken& token, const SrtSessionReadCounts& counts,
                         bool invalid, Clock&& clock) {
        if (!config.enabled || truncated || !active ||
            token.stage >= SrtSessionDiagnosticWindow::StageCount) {
            return;
        }
        const auto now = static_cast<std::uint64_t>(std::invoke(clock));
        window.end_us = std::max(window.end_us, now);
        window.saturated |= counts.saturated;
        const auto accumulate = [&](SrtSessionStageCounts& stage) {
            const auto add = [&](std::uint64_t& value, std::uint64_t increment) {
                AddSrtSnapshotTiming(value, increment, window.saturated);
            };
            add(stage.sessions, 1);
            add(stage.selections, token.first_attempt);
            add(stage.retries, !token.first_attempt);
            add(stage.invalid_sessions, invalid);
            add(stage.read_calls, counts.read_calls);
            add(stage.new_pages, counts.new_pages);
            add(stage.cache_hits, counts.cache_hits);
            add(stage.requested_bytes, counts.requested_bytes);
            stage.max_pages_per_session = std::max(stage.max_pages_per_session, counts.new_pages);
            const auto elapsed = SrtSnapshotElapsed(token.begin_us, now);
            add(stage.total_us, elapsed);
            stage.max_session_us = std::max(stage.max_session_us, elapsed);
        };
        accumulate(window.totals);
        accumulate(window.stages[token.stage]);
    }

private:
    SrtSnapshotTimingConfig config;
    SrtSessionDiagnosticWindow window;
    std::uint32_t emitted_rows{};
    bool active{};
    bool truncated{};
};

} // namespace VideoCore
