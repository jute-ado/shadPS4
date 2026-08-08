// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

#include "common/types.h"

namespace VideoCore {

constexpr u32 NumBdaFallbackLogicalStages = 6;
constexpr u32 MaxBdaFallbackWindowFrames = 512;
constexpr u32 MaxBdaFallbackOperationsPerFrame = 1U << 16;
constexpr u32 DisabledBdaFallbackToken = std::numeric_limits<u32>::max();

constexpr std::string_view BdaFallbackEnabledEnv = "SHADPS4_DIAGNOSTIC_BDA_FALLBACK_CONSUMPTION";
constexpr std::string_view BdaFallbackFirstFrameEnv = "SHADPS4_DIAGNOSTIC_BDA_FALLBACK_FIRST_FRAME";
constexpr std::string_view BdaFallbackFrameCountEnv = "SHADPS4_DIAGNOSTIC_BDA_FALLBACK_FRAME_COUNT";
constexpr std::string_view BdaFallbackMaxOperationsEnv =
    "SHADPS4_DIAGNOSTIC_BDA_FALLBACK_MAX_OPERATIONS_PER_FRAME";

struct BdaFallbackWindowConfig {
    u64 first_frame{};
    u32 frame_count{};
    u32 max_operations_per_frame{};
};

inline bool BdaFallbackConsumptionDiagnosticEnabled();

struct BdaFallbackParsedFrame {
    u64 sequence{};
    u64 process_time_us{};
};

class BdaFallbackParsedFrameSequence {
public:
    u64 ObserveFlip(u64 process_time_us) noexcept {
        publication.fetch_add(1, std::memory_order_acq_rel);
        const u64 completed = frame_sequence.fetch_add(1, std::memory_order_relaxed);
        frame_process_time_us.store(process_time_us, std::memory_order_relaxed);
        publication.fetch_add(1, std::memory_order_release);
        return completed;
    }

    [[nodiscard]] BdaFallbackParsedFrame Read() const noexcept {
        for (;;) {
            const u64 before = publication.load(std::memory_order_acquire);
            if ((before & 1) != 0) {
                continue;
            }
            const BdaFallbackParsedFrame result{
                .sequence = frame_sequence.load(std::memory_order_relaxed),
                .process_time_us = frame_process_time_us.load(std::memory_order_relaxed),
            };
            const u64 after = publication.load(std::memory_order_acquire);
            if (before == after) {
                return result;
            }
        }
    }

private:
    std::atomic<u64> publication{};
    std::atomic<u64> frame_sequence{1};
    std::atomic<u64> frame_process_time_us{};
};

inline BdaFallbackParsedFrameSequence& GetBdaFallbackParsedFrameSequence() {
    static BdaFallbackParsedFrameSequence sequence;
    return sequence;
}

inline bool BdaFallbackConsumptionDiagnosticEnabled() {
    const char* value = std::getenv(BdaFallbackEnabledEnv.data());
    return value != nullptr && std::string_view{value} == "1";
}

constexpr bool ShouldPreloadPipelineCacheForBdaFallbackDiagnostic(bool cache_enabled,
                                                                  bool diagnostic_enabled) {
    return cache_enabled && !diagnostic_enabled;
}

constexpr bool ShouldApplyShaderPatchForBdaFallbackDiagnostic(bool patch_available,
                                                              bool patches_enabled,
                                                              bool diagnostic_enabled) {
    return patch_available && patches_enabled && !diagnostic_enabled;
}

inline bool ParseBdaFallbackU64(std::string_view name, u64& result) {
    const char* value = std::getenv(name.data());
    if (value == nullptr) {
        return false;
    }
    const std::string_view text{value};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    return error == std::errc{} && end == text.data() + text.size();
}

inline BdaFallbackWindowConfig ReadBdaFallbackWindowConfig() {
    BdaFallbackWindowConfig config{};
    u64 frame_count{};
    u64 max_operations{};
    if (!ParseBdaFallbackU64(BdaFallbackFirstFrameEnv, config.first_frame) ||
        !ParseBdaFallbackU64(BdaFallbackFrameCountEnv, frame_count) ||
        !ParseBdaFallbackU64(BdaFallbackMaxOperationsEnv, max_operations) ||
        frame_count > std::numeric_limits<u32>::max() ||
        max_operations > std::numeric_limits<u32>::max()) {
        return {};
    }
    config.frame_count = static_cast<u32>(frame_count);
    config.max_operations_per_frame = static_cast<u32>(max_operations);
    return config;
}

enum class BdaFallbackMarkStatus : u8 {
    Valid,
    Disabled,
    OutsideWindow,
    OperationOverflow,
    StageOverflow,
    CapacityExceeded,
};

struct BdaFallbackMarkPlan {
    BdaFallbackMarkStatus status{BdaFallbackMarkStatus::Disabled};
    u64 bit_index{};
    u64 word_index{};
    u32 bit_mask{};
};

constexpr bool IsBdaFallbackConfigValid(const BdaFallbackWindowConfig& config) {
    if (config.frame_count == 0 || config.frame_count > MaxBdaFallbackWindowFrames ||
        config.max_operations_per_frame == 0 ||
        config.max_operations_per_frame > MaxBdaFallbackOperationsPerFrame) {
        return false;
    }
    if (config.first_frame > std::numeric_limits<u64>::max() - config.frame_count) {
        return false;
    }
    const u64 operations = static_cast<u64>(config.frame_count) * config.max_operations_per_frame;
    return operations <= std::numeric_limits<u32>::max() / NumBdaFallbackLogicalStages;
}

constexpr u64 BdaFallbackWindowBitCount(const BdaFallbackWindowConfig& config) {
    return IsBdaFallbackConfigValid(config)
               ? static_cast<u64>(config.frame_count) * config.max_operations_per_frame *
                     NumBdaFallbackLogicalStages
               : 0;
}

constexpr size_t BdaFallbackWindowWordCount(const BdaFallbackWindowConfig& config) {
    return static_cast<size_t>((BdaFallbackWindowBitCount(config) + 31) / 32);
}

constexpr size_t BdaFallbackFrameWordCount(const BdaFallbackWindowConfig& config) {
    if (!IsBdaFallbackConfigValid(config)) {
        return 0;
    }
    const u64 bits =
        static_cast<u64>(config.max_operations_per_frame) * NumBdaFallbackLogicalStages;
    return static_cast<size_t>((bits + 31) / 32);
}

constexpr BdaFallbackMarkPlan PlanBdaFallbackMark(const BdaFallbackWindowConfig& config,
                                                  bool enabled, u64 frame, u32 operation,
                                                  u32 stage) {
    if (!enabled) {
        return {.status = BdaFallbackMarkStatus::Disabled};
    }
    if (!IsBdaFallbackConfigValid(config)) {
        return {.status = BdaFallbackMarkStatus::CapacityExceeded};
    }
    if (frame < config.first_frame || frame - config.first_frame >= config.frame_count) {
        return {.status = BdaFallbackMarkStatus::OutsideWindow};
    }
    if (operation >= config.max_operations_per_frame) {
        return {.status = BdaFallbackMarkStatus::OperationOverflow};
    }
    if (stage >= NumBdaFallbackLogicalStages) {
        return {.status = BdaFallbackMarkStatus::StageOverflow};
    }
    const u64 frame_index = frame - config.first_frame;
    const u64 bit_index =
        (frame_index * config.max_operations_per_frame + operation) * NumBdaFallbackLogicalStages +
        stage;
    return {
        .status = BdaFallbackMarkStatus::Valid,
        .bit_index = bit_index,
        .word_index = bit_index / 32,
        .bit_mask = 1U << (bit_index % 32),
    };
}

inline bool MarkBdaFallback(std::span<std::atomic<u32>> words, const BdaFallbackMarkPlan& plan) {
    if (plan.status != BdaFallbackMarkStatus::Valid || plan.word_index >= words.size()) {
        return false;
    }
    words[plan.word_index].fetch_or(plan.bit_mask, std::memory_order_relaxed);
    return true;
}

enum class BdaFallbackFrameAvailability : u8 {
    Complete,
    Unavailable,
    Incomplete,
};

enum class BdaFallbackFrameStatus : u8 {
    Complete,
    Unavailable,
    Incomplete,
    OperationOverflow,
    CapacityLoss,
};

struct BdaFallbackFrameObservation {
    BdaFallbackFrameStatus status{BdaFallbackFrameStatus::Unavailable};
    u64 frame{};
    u64 previous_frame{};
    bool has_previous{};
    bool changed_from_previous{};
    bool exact_aba_return{};
    u64 aba_middle_frame{};
};

class BdaFallbackFrameReducer {
public:
    explicit BdaFallbackFrameReducer(BdaFallbackWindowConfig config_) : config{config_} {}

    BdaFallbackFrameObservation Observe(u64 frame, std::span<const u32> words,
                                        BdaFallbackFrameAvailability availability,
                                        bool operation_overflow) {
        if (availability != BdaFallbackFrameAvailability::Complete) {
            ResetHistory();
            return {.status = availability == BdaFallbackFrameAvailability::Unavailable
                                  ? BdaFallbackFrameStatus::Unavailable
                                  : BdaFallbackFrameStatus::Incomplete,
                    .frame = frame};
        }
        if (operation_overflow) {
            ResetHistory();
            return {.status = BdaFallbackFrameStatus::OperationOverflow, .frame = frame};
        }
        const size_t expected_words = BdaFallbackFrameWordCount(config);
        if (expected_words == 0 || words.size() != expected_words) {
            ResetHistory();
            return {.status = BdaFallbackFrameStatus::CapacityLoss, .frame = frame};
        }

        BdaFallbackFrameObservation result{.status = BdaFallbackFrameStatus::Complete,
                                           .frame = frame};
        if (previous_valid) {
            result.previous_frame = previous_frame;
            result.has_previous = true;
            result.changed_from_previous = !std::ranges::equal(previous, words);
            if (result.changed_from_previous && prior_valid && std::ranges::equal(prior, words) &&
                previous_frame + 1 == frame && prior_frame + 1 == previous_frame) {
                result.exact_aba_return = true;
                result.aba_middle_frame = previous_frame;
            }
        }

        prior = std::move(previous);
        prior_frame = previous_frame;
        prior_valid = previous_valid;
        previous.assign(words.begin(), words.end());
        previous_frame = frame;
        previous_valid = true;
        return result;
    }

private:
    void ResetHistory() {
        previous.clear();
        prior.clear();
        previous_valid = false;
        prior_valid = false;
    }

    BdaFallbackWindowConfig config;
    std::vector<u32> previous;
    std::vector<u32> prior;
    u64 previous_frame{};
    u64 prior_frame{};
    bool previous_valid{};
    bool prior_valid{};
};

} // namespace VideoCore
