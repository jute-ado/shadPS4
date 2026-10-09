// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Vulkan {

[[nodiscard]] constexpr bool ShouldRequireGuestSubgroupSize(
    bool supports_guest_size, bool uses_subgroup_operations) noexcept {
    return supports_guest_size && uses_subgroup_operations;
}

} // namespace Vulkan
