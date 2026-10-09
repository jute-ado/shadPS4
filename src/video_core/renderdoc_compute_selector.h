// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <charconv>
#include <optional>
#include <string_view>

#include "common/types.h"

namespace VideoCore {

class RenderDocComputeSelector {
public:
    // An unset selector preserves ordinary guest-work capture. An explicitly configured empty,
    // malformed, or overflowing value must not silently broaden the requested capture.
    [[nodiscard]] static RenderDocComputeSelector Parse(
        std::optional<std::string_view> configured) noexcept {
        RenderDocComputeSelector selector;
        if (!configured) {
            return selector;
        }
        auto text = *configured;
        if (text.starts_with("0x") || text.starts_with("0X")) {
            text.remove_prefix(2);
        }
        selector.valid = false;
        if (text.empty() || text.size() > 16) {
            return selector;
        }
        u64 hash{};
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), hash, 16);
        if (error != std::errc{} || end != text.data() + text.size()) {
            return selector;
        }
        selector.hash = hash;
        selector.valid = true;
        return selector;
    }

    [[nodiscard]] bool Valid() const noexcept {
        return valid;
    }
    [[nodiscard]] bool Allows(std::optional<u64> compute_shader_hash) const noexcept {
        return valid && (!hash || (compute_shader_hash && *compute_shader_hash == *hash));
    }

private:
    std::optional<u64> hash;
    bool valid{true};
};

} // namespace VideoCore
