// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>

#include "video_core/renderdoc_compute_selector.h"

namespace VideoCore {

struct RenderDocColorDrawProof {
    bool has_bound_color_attachment{};
    bool has_presentable_color_attachment{};
};

template <typename Attachments>
[[nodiscard]] bool HasBoundRenderDocColorAttachment(const Attachments& attachments,
                                                   std::size_t count) noexcept {
    if (count > attachments.size()) {
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (static_cast<bool>(attachments[i].image_view)) {
            return true;
        }
    }
    return false;
}

class RenderDocShaderSelector {
public:
    // A launch selects at most one command kind. Even empty configured values count as present:
    // malformed or ambiguous settings must never broaden a requested capture.
    [[nodiscard]] static RenderDocShaderSelector Parse(
        std::optional<std::string_view> compute, std::optional<std::string_view> vertex,
        std::optional<std::string_view> color = std::nullopt) noexcept {
        RenderDocShaderSelector result;
        if (color && *color != "0" && *color != "1") {
            result.target = Target::Invalid;
            return result;
        }
        result.color_draw = color && *color == "1";
        if ((compute && vertex) || (compute && result.color_draw)) {
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

    [[nodiscard]] bool RequiresColorAttachment() const noexcept {
        return Valid() && color_draw;
    }

    [[nodiscard]] bool Allows(std::optional<u64> compute_shader_hash,
                              std::optional<u64> vertex_shader_hash,
                              bool has_color_attachment = false) const noexcept {
        // Keep the boolean overload as a compatibility boundary for callers that only have the
        // old attachment proof. A strict color-draw request uses the proof overload below and
        // therefore cannot consume a depth-only or off-screen submission.
        return Allows(compute_shader_hash, vertex_shader_hash,
                      RenderDocColorDrawProof{has_color_attachment, has_color_attachment});
    }

    [[nodiscard]] bool Allows(std::optional<u64> compute_shader_hash,
                              std::optional<u64> vertex_shader_hash,
                              RenderDocColorDrawProof proof) const noexcept {
        if (!Valid() || (compute_shader_hash && vertex_shader_hash)) {
            return false;
        }
        if (color_draw && (!vertex_shader_hash || !proof.has_bound_color_attachment ||
                           !proof.has_presentable_color_attachment)) {
            return false;
        }
        return selector.Allows(target == Target::Vertex ? vertex_shader_hash : compute_shader_hash);
    }

private:
    enum class Target { Any, Compute, Vertex, Invalid };
    Target target{Target::Any};
    RenderDocComputeSelector selector;
    bool color_draw{};
};

} // namespace VideoCore
