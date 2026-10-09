// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>

namespace Vulkan {

// The worker normally uses the owner-created driver pipeline cache.  An explicit launch-time
// opt-in can bypass that cache for A/B diagnosis of drivers with problematic concurrent cache
// access.  Keep the parser strict so an accidental environment value cannot silently change the
// default execution path.
[[nodiscard]] constexpr bool ParseAsyncGraphicsWorkerNoCache(
    std::string_view value) noexcept {
    return value == "1";
}

} // namespace Vulkan
