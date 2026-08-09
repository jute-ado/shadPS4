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

struct ImageColorScopeProducerObservation {
    u32 draw_count{};
    ImageColorScopeDrawKind last_draw{ImageColorScopeDrawKind::Unknown};
    bool clear_at_begin{};
    bool valid{};
    bool overflow{};

    auto operator<=>(const ImageColorScopeProducerObservation&) const = default;
};

class ImageColorScopeProducerState {
public:
    static constexpr u32 MaxTrackedDraws = 2048;

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

    void MarkDraw(u64 scope_serial_, ImageColorScopeDrawKind kind) noexcept {
        if (scope_serial == 0 || scope_serial_ != scope_serial ||
            kind == ImageColorScopeDrawKind::Unknown) {
            observation.valid = false;
            return;
        }
        observation.last_draw = kind;
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
