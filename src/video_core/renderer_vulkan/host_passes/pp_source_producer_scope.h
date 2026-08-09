// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string>

#include "video_core/renderer_vulkan/host_passes/pp_source_publication.h"

namespace Vulkan::HostPasses {

enum class PpSourceProducerScopeClass : u8 {
    ActiveAtFlip,
    EndedEarlier,
};

[[nodiscard]] constexpr PpSourceProducerScopeClass ClassifyPpSourceProducerScope(
    bool rendering_before_flip) noexcept {
    return rendering_before_flip ? PpSourceProducerScopeClass::ActiveAtFlip
                                 : PpSourceProducerScopeClass::EndedEarlier;
}

struct PpSourceProducerScopeObservation {
    u64 sequence{};
    PpSourceProducerScopeClass classification{PpSourceProducerScopeClass::EndedEarlier};
    u32 selected{};
    u32 emitted{};
    u32 expected{};
    u32 active_at_flip{};
    u32 ended_earlier{};
    u32 loss{};
    bool final{};
};

class PpSourceProducerScopeCoverage {
public:
    explicit constexpr PpSourceProducerScopeCoverage(PpSourcePublicationWindow window_) noexcept
        : window{window_} {}

    [[nodiscard]] std::optional<PpSourceProducerScopeObservation> Observe(
        u64 sequence, PpSourceProducerScopeClass classification) noexcept {
        if (!window.Contains(sequence)) {
            return std::nullopt;
        }
        ++selected;
        ++emitted;
        if (has_last && sequence != last_sequence + 1) {
            ++loss;
        }
        has_last = true;
        last_sequence = sequence;
        classification == PpSourceProducerScopeClass::ActiveAtFlip ? ++active_at_flip
                                                                   : ++ended_earlier;
        const bool final = window.IsFinal(sequence);
        if (final && emitted != window.count) {
            ++loss;
        }
        return PpSourceProducerScopeObservation{
            .sequence = sequence,
            .classification = classification,
            .selected = selected,
            .emitted = emitted,
            .expected = window.count,
            .active_at_flip = active_at_flip,
            .ended_earlier = ended_earlier,
            .loss = loss,
            .final = final,
        };
    }

private:
    PpSourcePublicationWindow window{};
    u64 last_sequence{};
    u32 selected{};
    u32 emitted{};
    u32 active_at_flip{};
    u32 ended_earlier{};
    u32 loss{};
    bool has_last{};
};

[[nodiscard]] inline std::string FormatPpSourceProducerScopeObservation(
    const PpSourceProducerScopeObservation& observation) {
    return "FGSCPS s=" + std::to_string(observation.sequence) +
           " r=" + std::to_string(static_cast<u32>(observation.classification));
}

[[nodiscard]] inline std::string FormatPpSourceProducerScopeCoverage(
    const PpSourceProducerScopeObservation& observation) {
    return "FGSCPSC s=" + std::to_string(observation.sequence) +
           " n=" + std::to_string(observation.emitted) + '/' +
           std::to_string(observation.selected) + '/' + std::to_string(observation.expected) +
           " a=" + std::to_string(observation.active_at_flip) +
           " e=" + std::to_string(observation.ended_earlier) +
           " l=" + std::to_string(observation.loss);
}

} // namespace Vulkan::HostPasses
