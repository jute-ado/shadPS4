// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/texture_cache/image_producer.h"

namespace VideoCore {

enum class ImageColorScopeDrawKind : u8 {
    Unknown,
    Direct,
    Indirect,
};

struct ImageColorScopeDrawDescriptor {
    ImageColorScopeDrawKind kind{ImageColorScopeDrawKind::Unknown};
    bool indexed{};
    u32 element_count{};
    u32 instance_count{};
    u32 sampled_bindings{};
    u32 sampled_images{};
    u32 storage_writes{};
    ImageProducerClass sampled_input_producer{ImageProducerClass::Unknown};
    bool sampled_input_fresh{};
    bool sampled_input_alias{};
    bool sampled_input_valid{};
    u32 sampled_input_scope_draw_count{};
    ImageColorScopeDrawKind sampled_input_scope_last_draw{ImageColorScopeDrawKind::Unknown};
    bool sampled_input_scope_indexed{};
    u32 sampled_input_scope_element_count{};
    u32 sampled_input_scope_instance_count{};
    u32 sampled_input_scope_sampled_images{};
    u32 sampled_input_scope_storage_writes{};
    bool sampled_input_scope_clear_at_begin{};
    bool sampled_input_scope_valid{};
    bool sampled_input_scope_overflow{};
    ImageProducerClass sampled_input_scope_input_producer{ImageProducerClass::Unknown};
    bool sampled_input_scope_input_fresh{};
    bool sampled_input_scope_input_alias{};
    bool sampled_input_scope_input_valid{};
};

struct ImageColorScopeProducerObservation {
    u32 draw_count{};
    ImageColorScopeDrawKind last_draw{ImageColorScopeDrawKind::Unknown};
    bool indexed{};
    u32 element_count{};
    u32 instance_count{};
    u32 sampled_bindings{};
    u32 sampled_images{};
    u32 storage_writes{};
    ImageProducerClass sampled_input_producer{ImageProducerClass::Unknown};
    bool sampled_input_fresh{};
    bool sampled_input_alias{};
    bool sampled_input_valid{};
    u32 sampled_input_scope_draw_count{};
    ImageColorScopeDrawKind sampled_input_scope_last_draw{ImageColorScopeDrawKind::Unknown};
    bool sampled_input_scope_indexed{};
    u32 sampled_input_scope_element_count{};
    u32 sampled_input_scope_instance_count{};
    u32 sampled_input_scope_sampled_images{};
    u32 sampled_input_scope_storage_writes{};
    bool sampled_input_scope_clear_at_begin{};
    bool sampled_input_scope_valid{};
    bool sampled_input_scope_overflow{};
    ImageProducerClass sampled_input_scope_input_producer{ImageProducerClass::Unknown};
    bool sampled_input_scope_input_fresh{};
    bool sampled_input_scope_input_alias{};
    bool sampled_input_scope_input_valid{};
    bool clear_at_begin{};
    bool valid{};
    bool overflow{};

    auto operator<=>(const ImageColorScopeProducerObservation&) const = default;
};

class ImageColorScopeProducerState {
public:
    static constexpr u32 MaxTrackedDraws = 2048;
    static constexpr u32 MaxTrackedImageBindings = 32;

    void BeginScope(u64 scope_serial_, bool clear_at_begin_) noexcept {
        if (scope_serial_ == 0) {
            observation = {};
            scope_serial = 0;
            return;
        }
        if (scope_serial == scope_serial_) {
            return;
        }
        scope_serial = scope_serial_;
        observation = {
            .clear_at_begin = clear_at_begin_,
            .valid = true,
        };
    }

    void MarkDraw(u64 scope_serial_, ImageColorScopeDrawDescriptor descriptor) noexcept {
        if (scope_serial == 0 || scope_serial_ != scope_serial ||
            descriptor.kind == ImageColorScopeDrawKind::Unknown) {
            observation.valid = false;
            return;
        }
        observation.last_draw = descriptor.kind;
        observation.indexed = descriptor.indexed;
        observation.element_count = descriptor.element_count;
        observation.instance_count = descriptor.instance_count;
        observation.sampled_bindings = descriptor.sampled_bindings;
        observation.sampled_images = descriptor.sampled_images;
        observation.storage_writes = descriptor.storage_writes;
        observation.sampled_input_producer = descriptor.sampled_input_producer;
        observation.sampled_input_fresh = descriptor.sampled_input_fresh;
        observation.sampled_input_alias = descriptor.sampled_input_alias;
        observation.sampled_input_valid = descriptor.sampled_input_valid;
        observation.sampled_input_scope_draw_count = descriptor.sampled_input_scope_draw_count;
        observation.sampled_input_scope_last_draw = descriptor.sampled_input_scope_last_draw;
        observation.sampled_input_scope_indexed = descriptor.sampled_input_scope_indexed;
        observation.sampled_input_scope_element_count =
            descriptor.sampled_input_scope_element_count;
        observation.sampled_input_scope_instance_count =
            descriptor.sampled_input_scope_instance_count;
        observation.sampled_input_scope_sampled_images =
            descriptor.sampled_input_scope_sampled_images;
        observation.sampled_input_scope_storage_writes =
            descriptor.sampled_input_scope_storage_writes;
        observation.sampled_input_scope_clear_at_begin =
            descriptor.sampled_input_scope_clear_at_begin;
        observation.sampled_input_scope_valid = descriptor.sampled_input_scope_valid;
        observation.sampled_input_scope_overflow = descriptor.sampled_input_scope_overflow;
        observation.sampled_input_scope_input_producer =
            descriptor.sampled_input_scope_input_producer;
        observation.sampled_input_scope_input_fresh = descriptor.sampled_input_scope_input_fresh;
        observation.sampled_input_scope_input_alias = descriptor.sampled_input_scope_input_alias;
        observation.sampled_input_scope_input_valid = descriptor.sampled_input_scope_input_valid;
        if (descriptor.sampled_images > descriptor.sampled_bindings ||
            descriptor.sampled_bindings > MaxTrackedImageBindings ||
            descriptor.storage_writes > MaxTrackedImageBindings) {
            observation.valid = false;
        }
        if (observation.draw_count == MaxTrackedDraws) {
            observation.overflow = true;
            observation.valid = false;
            return;
        }
        ++observation.draw_count;
    }

    [[nodiscard]] ImageColorScopeProducerObservation Observe() const noexcept {
        return observation;
    }

    void Reset() noexcept {
        scope_serial = 0;
        observation = {};
    }

private:
    u64 scope_serial{};
    ImageColorScopeProducerObservation observation{};
};

} // namespace VideoCore
