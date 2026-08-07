// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_set>

#include "common/types.h"
#include "video_core/amdgpu/pixel_pipe_stat_control.h"

namespace AmdGpu {

struct OcclusionQueryReuseSnapshot {
    u64 sequence{};
    u64 dumps{};
    u64 fresh_targets{};
    u64 reused_targets{};
    u64 unknown_targets{};
    u64 no_prior_valid{};
    u64 partial_prior_valid{};
    u64 all_prior_valid{};
    u64 distinct_targets{};
    u32 min_counter_pairs{};
    u32 max_counter_pairs{};
    u64 controls{};
    u64 control_changes{};
    u64 dumps_without_control{};
    u64 hardcoded_layout_mismatches{};
    u32 counter_id_min{};
    u32 counter_id_max{};
    u32 stride_bytes_min{};
    u32 stride_bytes_max{};
    u32 enabled_instances_min{};
    u32 enabled_instances_max{};
    u64 instance_mask_and{};
    u64 instance_mask_or{};
};

class OcclusionQueryReuseDiagnostic {
public:
    static constexpr u64 ValidMask = 1ULL << 63;

    OcclusionQueryReuseDiagnostic(u64 report_interval_, u64 report_limit_, u64 target_limit_)
        : report_interval{std::max<u64>(report_interval_, 1)}, report_limit{report_limit_},
          target_limit{target_limit_} {
        seen_targets.reserve(static_cast<size_t>(target_limit));
    }

    void ObserveControl(const PixelPipeStatControl& control) noexcept {
        if (active_control.has_value() && active_control.value() != control) {
            ++current.control_changes;
        }
        active_control = control;
        ++current.controls;
    }

    [[nodiscard]] std::optional<OcclusionQueryReuseSnapshot> Observe(
        std::uintptr_t target, const u64* results, u32 counter_pairs) {
        if (reports_emitted >= report_limit) {
            return std::nullopt;
        }

        if (seen_targets.contains(target)) {
            ++current.reused_targets;
        } else if (seen_targets.size() < target_limit) {
            seen_targets.insert(target);
            ++current.fresh_targets;
        } else {
            ++current.unknown_targets;
        }

        u32 valid_count{};
        for (u32 i = 0; i < counter_pairs; ++i) {
            valid_count += (results[i * 2] & ValidMask) != 0;
        }
        if (valid_count == 0) {
            ++current.no_prior_valid;
        } else if (valid_count == counter_pairs) {
            ++current.all_prior_valid;
        } else {
            ++current.partial_prior_valid;
        }


        if (!active_control.has_value()) {
            ++current.dumps_without_control;
        } else {
            const auto& control = active_control.value();
            const u64 expected_instance_mask =
                counter_pairs >= 64 ? std::numeric_limits<u64>::max()
                                    : ((1ULL << counter_pairs) - 1);
            if (control.stride_bytes != 16 ||
                control.instance_enable_mask != expected_instance_mask) {
                ++current.hardcoded_layout_mismatches;
            }
            const u32 enabled_instances = control.EnabledInstanceCount();
            current.counter_id_min = std::min(current.counter_id_min, control.counter_id);
            current.counter_id_max = std::max(current.counter_id_max, control.counter_id);
            current.stride_bytes_min = std::min(current.stride_bytes_min, control.stride_bytes);
            current.stride_bytes_max = std::max(current.stride_bytes_max, control.stride_bytes);
            current.enabled_instances_min =
                std::min(current.enabled_instances_min, enabled_instances);
            current.enabled_instances_max =
                std::max(current.enabled_instances_max, enabled_instances);
            current.instance_mask_and &= control.instance_enable_mask;
            current.instance_mask_or |= control.instance_enable_mask;
        }

        current.min_counter_pairs = std::min(current.min_counter_pairs, counter_pairs);
        current.max_counter_pairs = std::max(current.max_counter_pairs, counter_pairs);
        ++current.dumps;
        if (current.dumps < report_interval) {
            return std::nullopt;
        }

        current.sequence = ++reports_emitted;
        current.distinct_targets = seen_targets.size();
        const auto snapshot = current;
        ResetCurrent();
        return snapshot;
    }

private:
    void ResetCurrent() noexcept {
        current = {};
        current.min_counter_pairs = std::numeric_limits<u32>::max();
        current.counter_id_min = std::numeric_limits<u32>::max();
        current.stride_bytes_min = std::numeric_limits<u32>::max();
        current.enabled_instances_min = std::numeric_limits<u32>::max();
        current.instance_mask_and = std::numeric_limits<u64>::max();
    }

    const u64 report_interval;
    const u64 report_limit;
    const u64 target_limit;
    u64 reports_emitted{};
    std::unordered_set<std::uintptr_t> seen_targets;
    std::optional<PixelPipeStatControl> active_control;
    OcclusionQueryReuseSnapshot current{
        .min_counter_pairs = std::numeric_limits<u32>::max(),
        .counter_id_min = std::numeric_limits<u32>::max(),
        .stride_bytes_min = std::numeric_limits<u32>::max(),
        .enabled_instances_min = std::numeric_limits<u32>::max(),
        .instance_mask_and = std::numeric_limits<u64>::max(),
    };
};

} // namespace AmdGpu
