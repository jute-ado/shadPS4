// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>

namespace Vulkan::LayerSettings {

inline constexpr std::string_view CrashDiagnosticLayer =
    "VK_LAYER_LUNARG_crash_diagnostic";
inline constexpr bool InstrumentAllCrashCommands = true;
inline constexpr std::string_view CrashShaderDumpMode = "on_crash";
inline constexpr std::string_view CrashCommandDumpMode = "all";
inline constexpr std::string_view CrashCommandBufferDumpMode = "pending";
inline constexpr std::string_view CrashQueueSubmitDumpMode = "pending";

} // namespace Vulkan::LayerSettings
