// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string>

#include "video_core/renderer_vulkan/host_passes/pp_source_publication.h"
#include "video_core/texture_cache/image_color_scope_producer.h"

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
    VideoCore::ImageColorScopeProducerObservation draw_scope{};
    u32 selected{};
    u32 emitted{};
    u32 expected{};
    u32 active_at_flip{};
    u32 ended_earlier{};
    u32 valid_draw_scopes{};
    u32 invalid_draw_scopes{};
    u32 overflow_draw_scopes{};
    u32 single_sampled_input{};
    u32 zero_sampled_inputs{};
    u32 multiple_sampled_inputs{};
    u32 writable_image_draws{};
    u32 loss{};
    bool final{};
};

class PpSourceProducerScopeCoverage {
public:
    explicit constexpr PpSourceProducerScopeCoverage(PpSourcePublicationWindow window_) noexcept
        : window{window_} {}

    [[nodiscard]] std::optional<PpSourceProducerScopeObservation> Observe(
        u64 sequence, PpSourceProducerScopeClass classification,
        VideoCore::ImageColorScopeProducerObservation draw_scope) noexcept {
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
        if (draw_scope.valid) {
            ++valid_draw_scopes;
            if (draw_scope.draw_count == 1) {
                if (draw_scope.sampled_images == 0) {
                    ++zero_sampled_inputs;
                } else if (draw_scope.sampled_images == 1) {
                    ++single_sampled_input;
                } else {
                    ++multiple_sampled_inputs;
                }
                writable_image_draws += draw_scope.storage_writes != 0;
            }
        } else if (draw_scope.overflow) {
            ++overflow_draw_scopes;
            ++loss;
        } else {
            ++invalid_draw_scopes;
            ++loss;
        }
        const bool final = window.IsFinal(sequence);
        if (final && emitted != window.count) {
            ++loss;
        }
        return PpSourceProducerScopeObservation{
            .sequence = sequence,
            .classification = classification,
            .draw_scope = draw_scope,
            .selected = selected,
            .emitted = emitted,
            .expected = window.count,
            .active_at_flip = active_at_flip,
            .ended_earlier = ended_earlier,
            .valid_draw_scopes = valid_draw_scopes,
            .invalid_draw_scopes = invalid_draw_scopes,
            .overflow_draw_scopes = overflow_draw_scopes,
            .single_sampled_input = single_sampled_input,
            .zero_sampled_inputs = zero_sampled_inputs,
            .multiple_sampled_inputs = multiple_sampled_inputs,
            .writable_image_draws = writable_image_draws,
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
    u32 valid_draw_scopes{};
    u32 invalid_draw_scopes{};
    u32 overflow_draw_scopes{};
    u32 single_sampled_input{};
    u32 zero_sampled_inputs{};
    u32 multiple_sampled_inputs{};
    u32 writable_image_draws{};
    u32 loss{};
    bool has_last{};
};

[[nodiscard]] inline std::string FormatPpSourceProducerScopeObservation(
    const PpSourceProducerScopeObservation& observation) {
    return "FGSCPS s=" + std::to_string(observation.sequence) +
           " r=" + std::to_string(static_cast<u32>(observation.classification)) +
           " d=" + std::to_string(observation.draw_scope.draw_count) +
           " k=" + std::to_string(static_cast<u32>(observation.draw_scope.last_draw)) +
           " c=" + std::to_string(observation.draw_scope.clear_at_begin ? 1 : 0) +
           " x=" + std::to_string(observation.draw_scope.valid ? 0 : 1) +
           " j=" + std::to_string(observation.draw_scope.indexed ? 1 : 0) +
           " e=" + std::to_string(observation.draw_scope.element_count) +
           " n=" + std::to_string(observation.draw_scope.instance_count) +
           " b=" + std::to_string(observation.draw_scope.sampled_bindings) +
           " u=" + std::to_string(observation.draw_scope.sampled_images) +
           " w=" + std::to_string(observation.draw_scope.storage_writes);
}

[[nodiscard]] inline std::string FormatPpSourceProducerScopeCoverage(
    const PpSourceProducerScopeObservation& observation) {
    return "FGSCPSC s=" + std::to_string(observation.sequence) +
           " n=" + std::to_string(observation.emitted) + '/' +
           std::to_string(observation.selected) + '/' + std::to_string(observation.expected) +
           " a=" + std::to_string(observation.active_at_flip) +
           " e=" + std::to_string(observation.ended_earlier) +
           " v=" + std::to_string(observation.valid_draw_scopes) +
           " i=" + std::to_string(observation.invalid_draw_scopes) +
           " x=" + std::to_string(observation.overflow_draw_scopes) +
           " s=" + std::to_string(observation.single_sampled_input) +
           " z=" + std::to_string(observation.zero_sampled_inputs) +
           " m=" + std::to_string(observation.multiple_sampled_inputs) +
           " w=" + std::to_string(observation.writable_image_draws) +
           " l=" + std::to_string(observation.loss);
}

} // namespace Vulkan::HostPasses
