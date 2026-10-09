// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>

#include "common/types.h"

namespace AmdGpu {

constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;

class ZpassCounterState {
public:
    [[nodiscard]] u64 Snapshot() const {
        return samples | OcclusionCounterValidMask;
    }

    [[nodiscard]] u64 AddSamples(const u64 count) {
        samples += count;
        return Snapshot();
    }

    void Reset() {
        samples = 0;
    }

private:
    u64 samples{};
};

[[nodiscard]] inline std::optional<bool> EvaluateZpassVisibility(
    const std::span<const u64> counters, const u32 num_counter_pairs) {
    if (counters.size() < static_cast<size_t>(num_counter_pairs) * 2) {
        return std::nullopt;
    }
    bool visible = false;
    for (u32 pair = 0; pair < num_counter_pairs; ++pair) {
        const u64 begin = counters[pair * 2];
        const u64 end = counters[pair * 2 + 1];
        if ((begin & end & OcclusionCounterValidMask) == 0) {
            return std::nullopt;
        }
        visible |= (begin & ~OcclusionCounterValidMask) !=
                   (end & ~OcclusionCounterValidMask);
    }
    return visible;
}

class StaticPredicationState {
public:
    void Disable() {
        enabled = false;
        execute = true;
        visible.reset();
    }

    void Set(std::optional<bool> new_visible, const bool draw_visible, const bool combine) {
        if (combine) {
            if (visible && new_visible) {
                new_visible = *visible || *new_visible;
            } else {
                new_visible.reset();
            }
        }
        enabled = true;
        visible = new_visible;
        execute = !visible || (draw_visible ? *visible : !*visible);
    }

    [[nodiscard]] bool ShouldSkip(const bool packet_predicated,
                                  const bool force_execute = false) const {
        return !force_execute && packet_predicated && enabled && !execute;
    }

private:
    bool enabled{};
    bool execute{true};
    std::optional<bool> visible;
};

} // namespace AmdGpu
