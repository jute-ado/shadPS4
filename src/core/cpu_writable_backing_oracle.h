// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <charconv>
#include <mutex>
#include <optional>
#include <string_view>

#include "common/types.h"
#include "core/physical_backing_provenance.h"

namespace Core {

enum class CpuWritableBackingOracleLoss : u32 {
    None = 0,
    CandidateCapacity = 1U << 0,
    MultipleSelection = 1U << 1,
};

[[nodiscard]] constexpr CpuWritableBackingOracleLoss operator|(CpuWritableBackingOracleLoss lhs,
                                                                CpuWritableBackingOracleLoss rhs) {
    return static_cast<CpuWritableBackingOracleLoss>(static_cast<u32>(lhs) |
                                                       static_cast<u32>(rhs));
}

[[nodiscard]] constexpr CpuWritableBackingOracleLoss operator&(CpuWritableBackingOracleLoss lhs,
                                                                CpuWritableBackingOracleLoss rhs) {
    return static_cast<CpuWritableBackingOracleLoss>(static_cast<u32>(lhs) &
                                                       static_cast<u32>(rhs));
}

struct CpuWritableBackingOracleConfiguration {
    bool enabled{};
    u32 selector{};
    u32 candidate_cap{65'536};
};

struct CpuWritableBackingOracleCandidate {
    PhysicalBackingMappingClass mapping_class{PhysicalBackingMappingClass::Unsupported};
    bool cpu_read{};
    bool cpu_write{};
    bool physical_backing_eligible{};
    bool complete_provenance{};
    bool owned_allocation{};
    u64 page_count{};
};

struct CpuWritableBackingOracleDecision {
    bool selected{};
    u32 candidate_ordinal{};
    u32 selected_count{};
};

struct CpuWritableBackingOracleSelectionEvent {
    u32 selector{};
    PhysicalBackingMappingClass mapping_class{PhysicalBackingMappingClass::Unsupported};
    u64 page_count{};
    u32 candidate_ordinal{};
    u32 selected_count{};
};

struct CpuWritableBackingOracleCoverage {
    u32 valid_candidates{};
    u64 invalid_candidates{};
    u64 dropped_candidates{};
    u32 selected_count{};
    u64 rejected_after_selection{};
    CpuWritableBackingOracleLoss loss{CpuWritableBackingOracleLoss::None};

    bool operator==(const CpuWritableBackingOracleCoverage&) const = default;
};

class CpuWritableBackingOracle {
public:
    static constexpr u32 HardMaxCandidates = 65'536;

    explicit CpuWritableBackingOracle(CpuWritableBackingOracleConfiguration config_)
        : config{config_} {
        config.candidate_cap = std::min(config.candidate_cap, HardMaxCandidates);
        if (config.selector == 0 || config.selector > HardMaxCandidates ||
            config.candidate_cap == 0) {
            config.enabled = false;
        }
    }

    /// Considers only a fully validated CPU-read/write Direct or Pooled mapping. Invalid mappings
    /// never consume a selector ordinal. The selector is one-based and at most one mapping can be
    /// admitted; all later mappings remain on the ordinary fail-closed path.
    [[nodiscard]] CpuWritableBackingOracleDecision Consider(
        const CpuWritableBackingOracleCandidate& candidate) {
        if (!config.enabled) {
            return {};
        }
        std::scoped_lock lock{mutex};
        if (!IsValidCandidate(candidate)) {
            ++coverage.invalid_candidates;
            return {};
        }
        if (coverage.valid_candidates >= config.candidate_cap) {
            ++coverage.dropped_candidates;
            coverage.loss = coverage.loss | CpuWritableBackingOracleLoss::CandidateCapacity;
            return {};
        }

        const u32 ordinal = ++coverage.valid_candidates;
        if (ordinal != config.selector) {
            coverage.rejected_after_selection += coverage.selected_count != 0;
            return {.candidate_ordinal = ordinal, .selected_count = coverage.selected_count};
        }
        if (coverage.selected_count != 0) {
            coverage.loss = coverage.loss | CpuWritableBackingOracleLoss::MultipleSelection;
            return {.candidate_ordinal = ordinal, .selected_count = coverage.selected_count};
        }

        coverage.selected_count = 1;
        pending_event = CpuWritableBackingOracleSelectionEvent{
            .selector = config.selector,
            .mapping_class = candidate.mapping_class,
            .page_count = candidate.page_count,
            .candidate_ordinal = ordinal,
            .selected_count = coverage.selected_count,
        };
        return {
            .selected = true,
            .candidate_ordinal = ordinal,
            .selected_count = coverage.selected_count,
        };
    }

    [[nodiscard]] std::optional<CpuWritableBackingOracleSelectionEvent> TakeSelectionEvent() {
        if (!config.enabled) {
            return std::nullopt;
        }
        std::scoped_lock lock{mutex};
        auto event = pending_event;
        pending_event.reset();
        return event;
    }

    [[nodiscard]] CpuWritableBackingOracleCoverage GetCoverage() const {
        if (!config.enabled) {
            return {};
        }
        std::scoped_lock lock{mutex};
        return coverage;
    }

    [[nodiscard]] CpuWritableBackingOracleConfiguration GetConfiguration() const noexcept {
        return config;
    }

private:
    [[nodiscard]] static constexpr bool IsValidCandidate(
        const CpuWritableBackingOracleCandidate& candidate) noexcept {
        const bool supported_class =
            candidate.mapping_class == PhysicalBackingMappingClass::Direct ||
            candidate.mapping_class == PhysicalBackingMappingClass::Pooled;
        return supported_class && candidate.cpu_read && candidate.cpu_write &&
               candidate.physical_backing_eligible && candidate.complete_provenance &&
               candidate.owned_allocation && candidate.page_count != 0;
    }

    CpuWritableBackingOracleConfiguration config;
    mutable std::mutex mutex;
    CpuWritableBackingOracleCoverage coverage{};
    std::optional<CpuWritableBackingOracleSelectionEvent> pending_event;
};

[[nodiscard]] inline CpuWritableBackingOracleConfiguration
ParseCpuWritableBackingOracleConfiguration(const char* enabled_value,
                                            const char* selector_value) noexcept {
    if (enabled_value == nullptr || std::string_view{enabled_value} != "1" ||
        selector_value == nullptr || *selector_value == '\0') {
        return {};
    }

    u64 selector{};
    const std::string_view text{selector_value};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), selector);
    if (error != std::errc{} || end != text.data() + text.size() || selector == 0 ||
        selector > CpuWritableBackingOracle::HardMaxCandidates) {
        return {};
    }
    return {
        .enabled = true,
        .selector = static_cast<u32>(selector),
        .candidate_cap = CpuWritableBackingOracle::HardMaxCandidates,
    };
}

[[nodiscard]] inline constexpr std::string_view CpuWritableBackingOracleClassName(
    PhysicalBackingMappingClass mapping_class) noexcept {
    switch (mapping_class) {
    case PhysicalBackingMappingClass::Direct:
        return "direct";
    case PhysicalBackingMappingClass::Pooled:
        return "pooled";
    default:
        return "rejected";
    }
}

} // namespace Core
