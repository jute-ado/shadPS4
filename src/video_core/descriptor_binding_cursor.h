// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <optional>

#include "common/types.h"

namespace VideoCore {

class DescriptorBindingCursor {
public:
    constexpr std::optional<u32> TryAdd(u32 descriptor_count = 1) noexcept {
        if (descriptor_count == 0 || next_binding == std::numeric_limits<u32>::max() ||
            descriptor_count > std::numeric_limits<u32>::max() - total_descriptors) {
            return std::nullopt;
        }
        const u32 binding = next_binding++;
        total_descriptors += descriptor_count;
        return binding;
    }

    [[nodiscard]] constexpr u32 BindingCount() const noexcept {
        return next_binding;
    }

    [[nodiscard]] constexpr u32 DescriptorCount() const noexcept {
        return total_descriptors;
    }

    [[nodiscard]] constexpr bool CanUsePushDescriptors(u32 maximum) const noexcept {
        return total_descriptors <= maximum;
    }

private:
    u32 next_binding{};
    u32 total_descriptors{};
};

} // namespace VideoCore
