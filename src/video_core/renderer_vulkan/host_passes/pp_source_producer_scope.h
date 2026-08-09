// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
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
    u32 valid_input_scopes{};
    u32 invalid_input_scopes{};
    u32 overflow_input_scopes{};
    u32 zero_input_scope_draws{};
    u32 single_input_scope_draw{};
    u32 multiple_input_scope_draws{};
    u32 input_scope_direct{};
    u32 input_scope_indirect{};
    u32 input_scope_clear{};
    u32 scope_input_color_attachment{};
    u32 scope_input_storage_image{};
    u32 scope_input_transfer{};
    u32 scope_input_cpu_upload{};
    u32 scope_input_unknown{};
    u32 scope_input_fresh{};
    u32 scope_input_reused{};
    u32 scope_input_alias{};
    u32 invalid_scope_input{};
    u32 ancestry_observed{};
    u32 ancestry_max_depth{};
    u32 ancestry_truncated{};
    u32 ancestry_loss{};
    u32 ancestry_draw_summary_loss{};
    std::array<u32, static_cast<u32>(VideoCore::ImageColorScopeAncestryTerminal::Count)>
        ancestry_terminals{};
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
                            if (draw_scope.sampled_input_scope_valid) {
                                ++valid_input_scopes;
                                if (draw_scope.sampled_input_scope_draw_count == 0) {
                                    ++zero_input_scope_draws;
                                } else if (draw_scope.sampled_input_scope_draw_count == 1) {
                                    ++single_input_scope_draw;
                                } else {
                                    ++multiple_input_scope_draws;
                                }
                                input_scope_direct += draw_scope.sampled_input_scope_last_draw ==
                                                      VideoCore::ImageColorScopeDrawKind::Direct;
                                input_scope_indirect +=
                                    draw_scope.sampled_input_scope_last_draw ==
                                    VideoCore::ImageColorScopeDrawKind::Indirect;
                                input_scope_clear += draw_scope.sampled_input_scope_clear_at_begin;
                                if (draw_scope.sampled_input_scope_sampled_images == 1) {
                                    if (!draw_scope.sampled_input_scope_input_valid) {
                                        ++invalid_scope_input;
                                        ++loss;
                                    } else {
                                        switch (draw_scope.sampled_input_scope_input_producer) {
                                        case VideoCore::ImageProducerClass::ColorAttachment:
                                            ++scope_input_color_attachment;
                                            break;
                                        case VideoCore::ImageProducerClass::StorageImage:
                                            ++scope_input_storage_image;
                                            break;
                                        case VideoCore::ImageProducerClass::Transfer:
                                            ++scope_input_transfer;
                                            break;
                                        case VideoCore::ImageProducerClass::CpuUpload:
                                            ++scope_input_cpu_upload;
                                            break;
                                        case VideoCore::ImageProducerClass::Unknown:
                                            ++scope_input_unknown;
                                            break;
                                        }
                                        draw_scope.sampled_input_scope_input_fresh
                                            ? ++scope_input_fresh
                                            : ++scope_input_reused;
                                        scope_input_alias +=
                                            draw_scope.sampled_input_scope_input_alias;
                                    }
                                }
                            } else if (draw_scope.sampled_input_scope_overflow) {
                                ++overflow_input_scopes;
                                ++loss;
                            } else {
                                ++invalid_input_scopes;
                                ++loss;
                            }
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
        if (draw_scope.ancestry.depth != 0) {
            ++ancestry_observed;
            ancestry_max_depth = draw_scope.ancestry.depth > ancestry_max_depth
                                     ? draw_scope.ancestry.depth
                                     : ancestry_max_depth;
            ancestry_truncated += draw_scope.ancestry.truncated;
            const u32 terminal = static_cast<u32>(draw_scope.ancestry.terminal);
            if (terminal < ancestry_terminals.size()) {
                ++ancestry_terminals[terminal];
            } else {
                ++ancestry_loss;
                ++loss;
            }
            if (VideoCore::IsImageColorScopeAncestryLoss(draw_scope.ancestry.terminal)) {
                ++ancestry_loss;
                ++loss;
            }
            if (draw_scope.ancestry.terminal_draws_truncated) {
                ++ancestry_draw_summary_loss;
                ++loss;
            }
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
            .valid_input_scopes = valid_input_scopes,
            .invalid_input_scopes = invalid_input_scopes,
            .overflow_input_scopes = overflow_input_scopes,
            .zero_input_scope_draws = zero_input_scope_draws,
            .single_input_scope_draw = single_input_scope_draw,
            .multiple_input_scope_draws = multiple_input_scope_draws,
            .input_scope_direct = input_scope_direct,
            .input_scope_indirect = input_scope_indirect,
            .input_scope_clear = input_scope_clear,
            .scope_input_color_attachment = scope_input_color_attachment,
            .scope_input_storage_image = scope_input_storage_image,
            .scope_input_transfer = scope_input_transfer,
            .scope_input_cpu_upload = scope_input_cpu_upload,
            .scope_input_unknown = scope_input_unknown,
            .scope_input_fresh = scope_input_fresh,
            .scope_input_reused = scope_input_reused,
            .scope_input_alias = scope_input_alias,
            .invalid_scope_input = invalid_scope_input,
            .ancestry_observed = ancestry_observed,
            .ancestry_max_depth = ancestry_max_depth,
            .ancestry_truncated = ancestry_truncated,
            .ancestry_loss = ancestry_loss,
            .ancestry_draw_summary_loss = ancestry_draw_summary_loss,
            .ancestry_terminals = ancestry_terminals,
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
    u32 valid_input_scopes{};
    u32 invalid_input_scopes{};
    u32 overflow_input_scopes{};
    u32 zero_input_scope_draws{};
    u32 single_input_scope_draw{};
    u32 multiple_input_scope_draws{};
    u32 input_scope_direct{};
    u32 input_scope_indirect{};
    u32 input_scope_clear{};
    u32 scope_input_color_attachment{};
    u32 scope_input_storage_image{};
    u32 scope_input_transfer{};
    u32 scope_input_cpu_upload{};
    u32 scope_input_unknown{};
    u32 scope_input_fresh{};
    u32 scope_input_reused{};
    u32 scope_input_alias{};
    u32 invalid_scope_input{};
    u32 ancestry_observed{};
    u32 ancestry_max_depth{};
    u32 ancestry_truncated{};
    u32 ancestry_loss{};
    u32 ancestry_draw_summary_loss{};
    std::array<u32, static_cast<u32>(VideoCore::ImageColorScopeAncestryTerminal::Count)>
        ancestry_terminals{};
    u32 loss{};
    bool has_last{};
};

[[nodiscard]] inline std::string FormatImageColorScopeDrawSummaries(
    const VideoCore::ImageColorScopeAncestry& ancestry) {
    std::string result;
    for (u32 index = 0; index < ancestry.terminal_draw_count; ++index) {
        if (index != 0) {
            result += '/';
        }
        const auto& draw = ancestry.terminal_draws[index];
        result += std::to_string(static_cast<u32>(draw.kind)) + '.' +
                  std::to_string(draw.indexed ? 1 : 0) + '.' + std::to_string(draw.element_count) +
                  '.' + std::to_string(draw.instance_count) + '.' +
                  std::to_string(draw.sampled_images) + '.' + std::to_string(draw.storage_writes) +
                  '.' + std::to_string(static_cast<u32>(draw.sampled_input_producer)) + '.' +
                  std::to_string(draw.sampled_input_fresh ? 1 : 0) + '.' +
                  std::to_string(draw.sampled_input_alias ? 1 : 0) + '.' +
                  std::to_string(draw.sampled_input_valid ? 1 : 0);
    }
    return result;
}

[[nodiscard]] inline std::string FormatImageColorScopeAncestry(
    const VideoCore::ImageColorScopeAncestry& ancestry) {
    std::string result;
    for (u32 index = 0; index < ancestry.depth; ++index) {
        if (index != 0) {
            result += '/';
        }
        const auto& node = ancestry.nodes[index];
        result +=
            std::to_string(static_cast<u32>(node.producer)) + '.' +
            std::to_string(node.fresh ? 1 : 0) + '.' + std::to_string(node.alias ? 1 : 0) + '.' +
            std::to_string(node.producer_valid ? 1 : 0) + '.' + std::to_string(node.draw_count) +
            '.' + std::to_string(static_cast<u32>(node.last_draw)) + '.' +
            std::to_string(node.indexed ? 1 : 0) + '.' + std::to_string(node.element_count) + '.' +
            std::to_string(node.instance_count) + '.' + std::to_string(node.sampled_images) + '.' +
            std::to_string(node.storage_writes) + '.' +
            std::to_string(node.clear_at_begin ? 1 : 0) + '.' +
            std::to_string(node.scope_valid ? 1 : 0) + '.' + std::to_string(node.overflow ? 1 : 0);
    }
    return result;
}

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
           " iv=" + std::to_string(observation.draw_scope.sampled_input_valid ? 1 : 0) +
           " sd=" + std::to_string(observation.draw_scope.sampled_input_scope_draw_count) + " sk=" +
           std::to_string(static_cast<u32>(observation.draw_scope.sampled_input_scope_last_draw)) +
           " sc=" +
           std::to_string(observation.draw_scope.sampled_input_scope_clear_at_begin ? 1 : 0) +
           " sx=" +
           std::to_string(observation.draw_scope.sampled_input_valid &&
                                  observation.draw_scope.sampled_input_producer ==
                                      VideoCore::ImageProducerClass::ColorAttachment &&
                                  !observation.draw_scope.sampled_input_scope_valid
                              ? 1
                              : 0) +
           " sj=" + std::to_string(observation.draw_scope.sampled_input_scope_indexed ? 1 : 0) +
           " se=" + std::to_string(observation.draw_scope.sampled_input_scope_element_count) +
           " sn=" + std::to_string(observation.draw_scope.sampled_input_scope_instance_count) +
           " su=" + std::to_string(observation.draw_scope.sampled_input_scope_sampled_images) +
           " sw=" + std::to_string(observation.draw_scope.sampled_input_scope_storage_writes) +
           " np=" +
           std::to_string(
               static_cast<u32>(observation.draw_scope.sampled_input_scope_input_producer)) +
           " nn=" + std::to_string(observation.draw_scope.sampled_input_scope_input_fresh ? 1 : 0) +
           " na=" + std::to_string(observation.draw_scope.sampled_input_scope_input_alias ? 1 : 0) +
           " nv=" + std::to_string(observation.draw_scope.sampled_input_scope_input_valid ? 1 : 0) +
           (observation.draw_scope.ancestry.depth == 0
                ? std::string{}
                : " ad=" + std::to_string(observation.draw_scope.ancestry.depth) + " at=" +
                      std::to_string(static_cast<u32>(observation.draw_scope.ancestry.terminal)) +
                      " ax=" + std::to_string(observation.draw_scope.ancestry.truncated ? 1 : 0) +
                      " ch=" + FormatImageColorScopeAncestry(observation.draw_scope.ancestry) +
                      (observation.draw_scope.ancestry.terminal_draw_count == 0
                           ? std::string{}
                           : " td=" +
                                 std::to_string(
                                     observation.draw_scope.ancestry.terminal_draw_count) +
                                 " tx=" +
                                 std::to_string(
                                     observation.draw_scope.ancestry.terminal_draws_truncated ? 1
                                                                                              : 0) +
                                 " th=" +
                                 FormatImageColorScopeDrawSummaries(
                                     observation.draw_scope.ancestry)));
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
           " sv=" + std::to_string(observation.valid_input_scopes) +
           " si=" + std::to_string(observation.invalid_input_scopes) +
           " sx=" + std::to_string(observation.overflow_input_scopes) +
           " sz=" + std::to_string(observation.zero_input_scope_draws) +
           " ss=" + std::to_string(observation.single_input_scope_draw) +
           " sm=" + std::to_string(observation.multiple_input_scope_draws) +
           " sd=" + std::to_string(observation.input_scope_direct) +
           " sn=" + std::to_string(observation.input_scope_indirect) +
           " sc=" + std::to_string(observation.input_scope_clear) +
           " dc=" + std::to_string(observation.scope_input_color_attachment) +
           " ds=" + std::to_string(observation.scope_input_storage_image) +
           " dt=" + std::to_string(observation.scope_input_transfer) +
           " du=" + std::to_string(observation.scope_input_cpu_upload) +
           " dx=" + std::to_string(observation.scope_input_unknown) +
           " df=" + std::to_string(observation.scope_input_fresh) +
           " dr=" + std::to_string(observation.scope_input_reused) +
           " da=" + std::to_string(observation.scope_input_alias) +
           " dl=" + std::to_string(observation.invalid_scope_input) +
           (observation.ancestry_observed == 0
                ? std::string{}
                : " ao=" + std::to_string(observation.ancestry_observed) +
                      " am=" + std::to_string(observation.ancestry_max_depth) +
                      " ax=" + std::to_string(observation.ancestry_truncated) +
                      " al=" + std::to_string(observation.ancestry_loss) +
                      " ay=" + std::to_string(observation.ancestry_draw_summary_loss) + " at=" +
                      [&] {
                          std::string terminals;
                          for (u32 index = 0; index < observation.ancestry_terminals.size();
                               ++index) {
                              if (index != 0) {
                                  terminals += ',';
                              }
                              terminals += std::to_string(observation.ancestry_terminals[index]);
                          }
                          return terminals;
                      }()) +
           " l=" + std::to_string(observation.loss);
}

} // namespace Vulkan::HostPasses
