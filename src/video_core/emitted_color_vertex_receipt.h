// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string_view>

#include "video_core/renderdoc_compute_selector.h"

namespace VideoCore {

// Default-off, one-shot proof that an exact guest vertex shader reached a color draw. This
// deliberately records command emission only; it does not imply GPU completion or visible output.
class EmittedColorVertexReceipt {
public:
    [[nodiscard]] static EmittedColorVertexReceipt Parse(
        std::optional<std::string_view> configured) noexcept {
        EmittedColorVertexReceipt receipt;
        if (!configured) {
            return receipt;
        }
        receipt.configured = true;
        receipt.selector = RenderDocComputeSelector::Parse(configured);
        receipt.valid = receipt.selector.Valid();
        return receipt;
    }

    [[nodiscard]] bool Valid() const noexcept {
        return valid;
    }

    [[nodiscard]] bool Enabled() const noexcept {
        return configured && valid && !matched;
    }

    [[nodiscard]] bool Observe(u64 vertex_hash, bool has_color_attachment) noexcept {
        if (!Enabled() || !has_color_attachment || !selector.Allows(vertex_hash)) {
            return false;
        }
        matched = true;
        return true;
    }

private:
    RenderDocComputeSelector selector;
    bool configured{};
    bool valid{true};
    bool matched{};
};

} // namespace VideoCore
