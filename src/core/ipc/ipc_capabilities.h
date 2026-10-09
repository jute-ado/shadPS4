// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <span>
#include <string_view>

namespace Core::Ipc {

inline constexpr std::array BaseCapabilities{
    std::string_view{"ENABLE_MEMORY_PATCH"},
    std::string_view{"ENABLE_EMU_CONTROL"},
    std::string_view{"ENABLE_SCREENSHOT"},
    std::string_view{"ENABLE_GAMEPAD"},
    std::string_view{"ENABLE_PRESENTED_FRAME_GAMEPAD"},
    std::string_view{"ENABLE_PRESENTED_FRAME_SCREENSHOT"},
};

inline constexpr std::array RenderDocCapability{
    std::string_view{"ENABLE_MEMORY_PATCH"}, std::string_view{"ENABLE_EMU_CONTROL"},
    std::string_view{"ENABLE_SCREENSHOT"},   std::string_view{"ENABLE_RENDERDOC_CAPTURE"},
    std::string_view{"ENABLE_GAMEPAD"}, std::string_view{"ENABLE_PRESENTED_FRAME_GAMEPAD"},
    std::string_view{"ENABLE_PRESENTED_FRAME_SCREENSHOT"},
};

[[nodiscard]] constexpr std::span<const std::string_view> IpcCapabilities(
    const bool renderdoc_loaded) {
    return renderdoc_loaded ? std::span<const std::string_view>{RenderDocCapability}
                            : std::span<const std::string_view>{BaseCapabilities};
}

} // namespace Core::Ipc
