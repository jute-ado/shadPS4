// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "png_decode.h"

#include <limits>

#include "common/stb.h"

namespace ImGui::Core::TextureManager {

void PngPixelsDeleter::operator()(u8* pixels) const noexcept {
    stbi_image_free(pixels);
}

std::optional<DecodedPng> DecodePngRgba(std::span<const u8> encoded) {
    if (encoded.empty() || encoded.size() > std::numeric_limits<int>::max()) {
        return std::nullopt;
    }

    int width = 0;
    int height = 0;
    auto pixels = std::unique_ptr<u8, PngPixelsDeleter>{stbi_load_from_memory(
        encoded.data(), static_cast<int>(encoded.size()), &width, &height, nullptr, 4)};
    if (pixels == nullptr || width <= 0 || height <= 0) {
        return std::nullopt;
    }

    const auto pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (pixel_count > std::numeric_limits<std::size_t>::max() / 4) {
        return std::nullopt;
    }

    return DecodedPng{
        .pixels = std::move(pixels),
        .width = static_cast<u32>(width),
        .height = static_cast<u32>(height),
    };
}

} // namespace ImGui::Core::TextureManager
