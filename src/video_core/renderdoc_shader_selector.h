// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <charconv>
#include <cstddef>

#include "video_core/renderdoc_compute_selector.h"

namespace VideoCore {

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
        std::optional<std::string_view> color = std::nullopt,
        std::optional<std::string_view> match_skip = std::nullopt) noexcept {
        RenderDocShaderSelector result;
        if (match_skip) {
            if (match_skip->empty()) {
                result.target = Target::Invalid;
                return result;
            }
            const auto [end, error] =
                std::from_chars(match_skip->data(), match_skip->data() + match_skip->size(),
                                result.match_skip_count);
            if (error != std::errc{} || end != match_skip->data() + match_skip->size() ||
                result.match_skip_count > 65535) {
                result.target = Target::Invalid;
                return result;
            }
        }
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

    [[nodiscard]] u32 MatchSkip() const noexcept {
        return match_skip_count;
    }

    [[nodiscard]] bool Allows(std::optional<u64> compute_shader_hash,
                              std::optional<u64> vertex_shader_hash,
                              bool has_color_attachment = false) const noexcept {
        if (!Valid() || (compute_shader_hash && vertex_shader_hash)) {
            return false;
        }
        if (color_draw && (!vertex_shader_hash || !has_color_attachment)) {
            return false;
        }
        return selector.Allows(target == Target::Vertex ? vertex_shader_hash : compute_shader_hash);
    }

private:
    enum class Target { Any, Compute, Vertex, Invalid };
    Target target{Target::Any};
    RenderDocComputeSelector selector;
    bool color_draw{};
    u32 match_skip_count{};
};

} // namespace VideoCore
