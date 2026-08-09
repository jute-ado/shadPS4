// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

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
