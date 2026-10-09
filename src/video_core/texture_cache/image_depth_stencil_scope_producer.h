// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>

#include "common/types.h"
#include "video_core/texture_cache/image_color_scope_producer.h"

namespace VideoCore {

struct ImageDepthWriterShape {
    ImageColorScopeDrawKind kind{ImageColorScopeDrawKind::Unknown};
    u64 fragment_pgm_hash{};
    u32 element_count{};
    u32 instance_count{};
    u32 sampled_images{};
    u32 storage_writes{};
    u8 guest_primitive_type{};
    u8 effective_primitive_type{};
    u8 cull_mode{};
    u8 front_face{};
    u8 fragment_mrt_mask{};
    u8 depth_compare_op{};
    bool indexed{};
    bool quad_list_converted{};
    bool fragment_writes_depth{};
    bool fragment_has_discard{};
    bool depth_test_enabled{};
    bool depth_bounds_test_enabled{};
    bool depth_bias_enabled{};
    bool stencil_test_enabled{};

    [[nodiscard]] constexpr bool Valid() const noexcept {
        return kind != ImageColorScopeDrawKind::Unknown;
    }

    auto operator<=>(const ImageDepthWriterShape&) const = default;
};

[[nodiscard]] constexpr ImageDepthWriterShape MakeImageDepthWriterShape(
    const ImageColorScopeDrawDescriptor& draw) noexcept {
    return {
        .kind = draw.kind,
        .fragment_pgm_hash = draw.fragment_pgm_hash,
        .element_count = draw.element_count,
        .instance_count = draw.instance_count,
        .sampled_images = draw.sampled_images,
        .storage_writes = draw.storage_writes,
        .guest_primitive_type = draw.guest_primitive_type,
        .effective_primitive_type = draw.effective_primitive_type,
        .cull_mode = draw.cull_mode,
        .front_face = draw.front_face,
        .fragment_mrt_mask = draw.fragment_mrt_mask,
        .depth_compare_op = draw.depth_compare_op,
        .indexed = draw.indexed,
        .quad_list_converted = draw.quad_list_converted,
        .fragment_writes_depth = draw.fragment_writes_depth,
        .fragment_has_discard = draw.fragment_has_discard,
        .depth_test_enabled = draw.depth_test_enabled,
        .depth_bounds_test_enabled = draw.depth_bounds_test_enabled,
        .depth_bias_enabled = draw.depth_bias_enabled,
        .stencil_test_enabled = draw.stencil_test_enabled,
    };
}

struct ImageDepthWriterScopeState {
    u64 rendering_serial{};
    u64 backing_generation{};
    u64 latest_mutation_epoch{};
    ImageDepthWriterShape first_shape{};
    u32 writer_count{};
    bool first_valid{};
    bool writer_count_overflow{};
    bool depth_clear{};
    bool stencil_clear{};
    bool valid{};

    auto operator<=>(const ImageDepthWriterScopeState&) const = default;
};

struct ImageDepthWriterScopeMark {
    u64 rendering_serial{};
    u64 backing_generation{};
    u64 previous_mutation_epoch{};
    u64 next_mutation_epoch{};
    ImageDepthWriterShape shape{};
    bool depth_clear{};
    bool stencil_clear{};
    bool depth_write{};
    bool stencil_write{};
};

struct ImageDepthWriterScopeAdvanceResult {
    ImageDepthWriterScopeState state{};
    bool preserved{};
    bool history_exact{};
};

[[nodiscard]] constexpr ImageDepthWriterScopeState InvalidateImageDepthWriterScope(
    const ImageDepthWriterScopeState& previous) noexcept {
    return {
        .rendering_serial = previous.rendering_serial,
    };
}

[[nodiscard]] constexpr ImageDepthWriterScopeAdvanceResult AdvanceImageDepthWriterScope(
    const ImageDepthWriterScopeState& previous, const ImageDepthWriterScopeMark& mark) noexcept {
    const bool mark_valid = mark.rendering_serial != 0 && mark.backing_generation != 0 &&
                            mark.next_mutation_epoch != 0 && mark.shape.Valid() &&
                            (mark.depth_write || mark.stencil_write);
    if (!mark_valid) {
        return {};
    }

    const bool same_serial = previous.rendering_serial != 0 &&
                             previous.rendering_serial == mark.rendering_serial;
    if (same_serial &&
        (!previous.valid || previous.backing_generation != mark.backing_generation ||
         previous.latest_mutation_epoch != mark.previous_mutation_epoch)) {
        return {
            .state = {.rendering_serial = mark.rendering_serial,
                      .backing_generation = mark.backing_generation,
                      .latest_mutation_epoch = mark.next_mutation_epoch},
        };
    }

    ImageDepthWriterScopeState state = same_serial
                                           ? previous
                                           : ImageDepthWriterScopeState{
                                                 .rendering_serial = mark.rendering_serial,
                                                 .backing_generation = mark.backing_generation,
                                                 .latest_mutation_epoch = mark.next_mutation_epoch,
                                                 .depth_clear = mark.depth_clear,
                                                 .stencil_clear = mark.stencil_clear,
                                                 .valid = true,
                                             };
    state.latest_mutation_epoch = mark.next_mutation_epoch;
    if (mark.depth_write) {
        if (!state.first_valid) {
            state.first_shape = mark.shape;
            state.first_valid = true;
        }
        if (state.writer_count == std::numeric_limits<u32>::max()) {
            state.writer_count_overflow = true;
        } else {
            ++state.writer_count;
        }
    }
    return {
        .state = state,
        .preserved = same_serial,
        .history_exact = true,
    };
}

} // namespace VideoCore
