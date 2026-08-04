// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <string>

namespace Core {

inline std::string FormatInternalScreenResolution(const std::uint32_t width,
                                                  const std::uint32_t height) {
    return "GPU internalScreenWidth: " + std::to_string(width) +
           " internalScreenHeight: " + std::to_string(height);
}

} // namespace Core
