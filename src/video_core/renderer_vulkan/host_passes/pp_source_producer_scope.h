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
    u32 input_color_attachment{};
    u32 input_storage_image{};
    u32 input_transfer{};
    u32 input_cpu_upload{};
    u32 input_unknown{};
    u32 input_fresh{};
    u32 input_reused{};
    u32 input_alias{};
    u32 invalid_single_input{};
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
                    if (!draw_scope.sampled_input_valid) {
                        ++invalid_single_input;
                        ++loss;
                    } else {
                        switch (draw_scope.sampled_input_producer) {
                        case VideoCore::ImageProducerClass::ColorAttachment:
                            ++input_color_attachment;
                            break;
                        case VideoCore::ImageProducerClass::StorageImage:
                            ++input_storage_image;
                            break;
                        case VideoCore::ImageProducerClass::Transfer:
                            ++input_transfer;
                            break;
                        case VideoCore::ImageProducerClass::CpuUpload:
                            ++input_cpu_upload;
                            break;
                        case VideoCore::ImageProducerClass::Unknown:
                            ++input_unknown;
                            break;
                        }
                        draw_scope.sampled_input_fresh ? ++input_fresh : ++input_reused;
                        input_alias += draw_scope.sampled_input_alias;
                    }
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
            .input_color_attachment = input_color_attachment,
            .input_storage_image = input_storage_image,
            .input_transfer = input_transfer,
            .input_cpu_upload = input_cpu_upload,
            .input_unknown = input_unknown,
            .input_fresh = input_fresh,
            .input_reused = input_reused,
            .input_alias = input_alias,
            .invalid_single_input = invalid_single_input,
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
    u32 input_color_attachment{};
    u32 input_storage_image{};
    u32 input_transfer{};
    u32 input_cpu_upload{};
    u32 input_unknown{};
    u32 input_fresh{};
    u32 input_reused{};
    u32 input_alias{};
    u32 invalid_single_input{};
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
           " w=" + std::to_string(observation.draw_scope.storage_writes) + " ip=" +
           std::to_string(static_cast<u32>(observation.draw_scope.sampled_input_producer)) +
           " in=" + std::to_string(observation.draw_scope.sampled_input_fresh ? 1 : 0) +
           " ia=" + std::to_string(observation.draw_scope.sampled_input_alias ? 1 : 0) +
           " iv=" + std::to_string(observation.draw_scope.sampled_input_valid ? 1 : 0);
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
           " ic=" + std::to_string(observation.input_color_attachment) +
           " is=" + std::to_string(observation.input_storage_image) +
           " it=" + std::to_string(observation.input_transfer) +
           " iu=" + std::to_string(observation.input_cpu_upload) +
           " ix=" + std::to_string(observation.input_unknown) +
           " if=" + std::to_string(observation.input_fresh) +
           " ir=" + std::to_string(observation.input_reused) +
           " ia=" + std::to_string(observation.input_alias) +
           " il=" + std::to_string(observation.invalid_single_input) +
           " l=" + std::to_string(observation.loss);
}

} // namespace Vulkan::HostPasses
