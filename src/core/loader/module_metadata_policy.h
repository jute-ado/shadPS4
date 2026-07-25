// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <string_view>

namespace Core::Loader {

template <std::size_t Size>
constexpr void WriteModuleFilename(std::array<char, Size>& destination,
                                   std::string_view module_name) {
    destination.fill('\0');
    if constexpr (Size <= 1) {
        return;
    }

    constexpr std::string_view suffix = ".sprx";
    const std::size_t content_capacity = Size - 1;
    const std::size_t suffix_size = std::min(suffix.size(), content_capacity);
    const std::size_t stem_size =
        std::min(module_name.size(), content_capacity - suffix_size);
    std::copy_n(module_name.begin(), stem_size, destination.begin());
    std::copy_n(suffix.begin(), suffix_size, destination.begin() + stem_size);
}

} // namespace Core::Loader
