// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>
#include <string_view>

namespace Vulkan {

// Memory snapshots can be expensive and are intentionally opt-in. Keep the parser strict so a
// normal environment value can never turn this diagnostic on accidentally.
[[nodiscard]] constexpr bool ParseVulkanMemoryDiagnostics(std::string_view value) noexcept {
    return value == "1";
}

[[nodiscard]] inline bool VulkanMemoryDiagnosticsEnabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_VK_MEMORY_DIAGNOSTICS");
        return value && ParseVulkanMemoryDiagnostics(value);
    }();
    return enabled;
}

} // namespace Vulkan
