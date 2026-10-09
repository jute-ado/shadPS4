// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

constexpr u32 ConsumeDescriptorBinding(u32& next_binding) noexcept {
    const u32 binding = next_binding++;
    return binding;
}

class DescriptorBindingCursor {
public:
    constexpr u32 Add(u32 descriptor_count = 1) noexcept {
        const u32 binding = ConsumeDescriptorBinding(next_binding);
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
