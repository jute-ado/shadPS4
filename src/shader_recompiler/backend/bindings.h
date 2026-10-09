// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <optional>

#include "common/types.h"

namespace Shader::Backend {

struct Bindings {
    u32 unified{};
    u32 buffer{};
    u32 user_data{};

    constexpr std::optional<u32> TryAddDescriptorBinding(u32 descriptor_count = 1) noexcept {
        if (descriptor_count == 0 || unified == std::numeric_limits<u32>::max()) {
            return std::nullopt;
        }
        return unified++;
    }

    constexpr std::optional<u32> TryAddBufferBinding() noexcept {
        if (buffer == std::numeric_limits<u32>::max()) {
            return std::nullopt;
        }
        const auto result = TryAddDescriptorBinding();
        if (result) {
            ++buffer;
        }
        return result;
    }

    constexpr std::optional<u32> TryAddImageBinding(u32 descriptor_count) noexcept {
        return TryAddDescriptorBinding(descriptor_count);
    }

    constexpr std::optional<u32> TryAddSamplerBinding() noexcept {
        return TryAddDescriptorBinding();
    }

    auto operator<=>(const Bindings&) const = default;
};

} // namespace Shader::Backend
