// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>

namespace Libraries::SysModule {

enum class PreloadRequirement {
    None,
    Libc,
    Fios2,
};

[[nodiscard]] constexpr PreloadRequirement ClassifyPreloadRequirement(
    std::string_view library, std::string_view stub) noexcept {
    if (library == "libc" && stub == "Need_sceLibc") {
        return PreloadRequirement::Libc;
    }
    if (library == "libSceFios2" && stub == "sceFiosInitialize") {
        return PreloadRequirement::Fios2;
    }
    return PreloadRequirement::None;
}

} // namespace Libraries::SysModule
