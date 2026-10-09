// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Vulkan {

[[nodiscard]] constexpr bool ShouldClearDepthAttachment(bool depth_clear_enabled,
                                                        bool depth_enabled,
                                                        bool depth_write_enabled,
                                                        bool metadata_cleared) noexcept {
    return (depth_clear_enabled && depth_enabled && depth_write_enabled) || metadata_cleared;
}

} // namespace Vulkan
