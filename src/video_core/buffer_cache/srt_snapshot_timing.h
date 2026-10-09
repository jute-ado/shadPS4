// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace VideoCore {

struct SrtSnapshotTimingConfig {
    bool enabled{};
    std::uint32_t start_frame{};

    static SrtSnapshotTimingConfig Parse(std::string_view enabled_value,
                                         std::string_view start_frame_value) noexcept {
        if (enabled_value != "1") {
            return {};
        }
        std::uint32_t start{};
        if (!start_frame_value.empty()) {
            const auto result =
                std::from_chars(start_frame_value.data(),
                                start_frame_value.data() + start_frame_value.size(), start);
            if (result.ec != std::errc{} ||
                result.ptr != start_frame_value.data() + start_frame_value.size() ||
                start > 50'000'000) {
                return {};
            }
        }
        return {true, start};
    }
};

inline void AddSrtSnapshotTiming(std::uint64_t& destination, std::uint64_t value,
                                 bool& saturated) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - destination) {
        destination = std::numeric_limits<std::uint64_t>::max();
        saturated = true;
    } else {
        destination += value;
    }
}

inline std::uint64_t SrtSnapshotElapsed(std::uint64_t begin, std::uint64_t end) noexcept {
    return end >= begin ? end - begin : 0;
}

struct SrtSnapshotTimingSample {
    std::uint64_t start_us{};
    std::uint64_t requested_bytes{};
    std::uint64_t overlay_bytes{};
    std::uint64_t copies{};
    std::uint64_t reused_overlay_bytes{};
    std::uint64_t map_us{};
    std::uint64_t finish_us{};
    std::uint64_t compose_us{};
    bool cpu_only{};
    bool gpu_composed{};
    bool raw_overlay_reused{};
    bool success{};
    bool saturated{};
};

struct SrtSnapshotTimingWindow {
    bool truncated{};
    bool saturated{};
    std::uint32_t presented_frame{};
    std::uint64_t begin_us{};
    std::uint64_t end_us{};
    std::uint64_t snapshots{};
    std::uint64_t cpu_only{};
    std::uint64_t gpu_composed{};
    std::uint64_t failures{};
    std::uint64_t requested_bytes{};
    std::uint64_t overlay_bytes{};
    std::uint64_t copies{};
    std::uint64_t reuse_hits{};
    std::uint64_t reused_overlay_bytes{};
    std::uint64_t map_us{};
    std::uint64_t finish_us{};
    std::uint64_t compose_us{};
    std::uint64_t total_us{};
    std::uint64_t max_finish_us{};
    std::uint64_t max_total_us{};
};

// GPU-owner-thread only. Windows are keyed by the presented counter observed at snapshot entry,
// not by GPU timestamps. The final active window is intentionally not emitted on destruction.
class SrtSnapshotTiming {
public:
    static constexpr std::uint32_t MaximumRows = 256;

    explicit SrtSnapshotTiming(bool enabled = false, std::uint32_t start_frame = 0)
        : config{enabled, start_frame} {}

    explicit SrtSnapshotTiming(SrtSnapshotTimingConfig config_) : config{config_} {}

    template <typename Clock, typename Emit>
    std::optional<SrtSnapshotTimingSample> BeginSnapshot(std::uint32_t presented_frame,
                                                         std::uint64_t requested_bytes,
                                                         Clock&& clock, Emit&& emit) {
        if (!config.enabled || truncated || presented_frame < config.start_frame) {
            return std::nullopt;
        }
        if (active && presented_frame != window.presented_frame) {
            std::invoke(emit, window);
            ++emitted_rows;
            active = false;
            if (emitted_rows == MaximumRows) {
                truncated = true;
                std::invoke(emit, SrtSnapshotTimingWindow{
                                      .truncated = true,
                                      .presented_frame = presented_frame,
                                  });
                return std::nullopt;
            }
        }
        const auto now = static_cast<std::uint64_t>(std::invoke(clock));
        if (!active) {
            window = {.presented_frame = presented_frame, .begin_us = now, .end_us = now};
            active = true;
        }
        return SrtSnapshotTimingSample{.start_us = now, .requested_bytes = requested_bytes};
    }

    template <typename Clock>
    void CompleteSnapshot(const SrtSnapshotTimingSample& sample, Clock&& clock) {
        if (!config.enabled || truncated || !active) {
            return;
        }
        const auto now = static_cast<std::uint64_t>(std::invoke(clock));
        window.end_us = std::max(window.end_us, now);
        window.saturated |= sample.saturated;
        const auto add = [&](std::uint64_t& destination, std::uint64_t value) {
            AddSrtSnapshotTiming(destination, value, window.saturated);
        };
        add(window.snapshots, 1);
        add(window.cpu_only, sample.cpu_only);
        add(window.gpu_composed, sample.gpu_composed);
        add(window.failures, !sample.success);
        add(window.requested_bytes, sample.requested_bytes);
        add(window.overlay_bytes, sample.overlay_bytes);
        add(window.copies, sample.copies);
        add(window.reuse_hits, sample.raw_overlay_reused);
        add(window.reused_overlay_bytes, sample.reused_overlay_bytes);
        add(window.map_us, sample.map_us);
        add(window.finish_us, sample.finish_us);
        add(window.compose_us, sample.compose_us);
        const auto total = SrtSnapshotElapsed(sample.start_us, now);
        add(window.total_us, total);
        window.max_finish_us = std::max(window.max_finish_us, sample.finish_us);
        window.max_total_us = std::max(window.max_total_us, total);
    }

private:
    SrtSnapshotTimingConfig config;
    SrtSnapshotTimingWindow window;
    std::uint32_t emitted_rows{};
    bool active{};
    bool truncated{};
};

template <typename Clock, typename Operation>
auto MeasureSrtSnapshotPhase(SrtSnapshotTimingSample* sample,
                             std::uint64_t SrtSnapshotTimingSample::* phase, Clock&& clock,
                             Operation&& operation) -> std::invoke_result_t<Operation&> {
    if (sample == nullptr) {
        return std::invoke(operation);
    }
    const auto begin = static_cast<std::uint64_t>(std::invoke(clock));
    const auto finish = [&] {
        const auto end = static_cast<std::uint64_t>(std::invoke(clock));
        AddSrtSnapshotTiming(sample->*phase, SrtSnapshotElapsed(begin, end), sample->saturated);
    };
    try {
        if constexpr (std::is_void_v<std::invoke_result_t<Operation&>>) {
            std::invoke(operation);
            finish();
        } else {
            decltype(auto) result = std::invoke(operation);
            finish();
            return result;
        }
    } catch (...) {
        finish();
        throw;
    }
}

} // namespace VideoCore
