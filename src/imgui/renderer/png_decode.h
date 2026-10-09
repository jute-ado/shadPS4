// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>

#include "common/types.h"

namespace ImGui::Core::TextureManager {

struct PngPixelsDeleter {
    void operator()(u8* pixels) const noexcept;
};

struct DecodedPng {
    std::unique_ptr<u8, PngPixelsDeleter> pixels;
    u32 width{};
    u32 height{};

    [[nodiscard]] std::size_t SizeBytes() const noexcept {
        return static_cast<std::size_t>(width) * height * 4;
    }
};

[[nodiscard]] std::optional<DecodedPng> DecodePngRgba(std::span<const u8> encoded);

} // namespace ImGui::Core::TextureManager
