// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderdoc_compute_selector.h"

namespace VideoCore {

class RenderDocShaderSelector {
public:
    // A launch selects at most one command kind. Even empty configured values count as present:
    // malformed or ambiguous settings must never broaden a requested capture.
    [[nodiscard]] static RenderDocShaderSelector Parse(
        std::optional<std::string_view> compute, std::optional<std::string_view> vertex) noexcept {
        RenderDocShaderSelector result;
        if (compute && vertex) {
            result.target = Target::Invalid;
            return result;
        }
        result.target = vertex ? Target::Vertex : compute ? Target::Compute : Target::Any;
        result.selector = RenderDocComputeSelector::Parse(vertex ? vertex : compute);
        return result;
    }

    [[nodiscard]] bool Valid() const noexcept {
        return target != Target::Invalid && selector.Valid();
    }

    [[nodiscard]] bool Allows(std::optional<u64> compute_shader_hash,
                              std::optional<u64> vertex_shader_hash) const noexcept {
        if (!Valid() || (compute_shader_hash && vertex_shader_hash)) {
            return false;
        }
        return selector.Allows(target == Target::Vertex ? vertex_shader_hash : compute_shader_hash);
    }

private:
    enum class Target { Any, Compute, Vertex, Invalid };
    Target target{Target::Any};
    RenderDocComputeSelector selector;
};

} // namespace VideoCore
