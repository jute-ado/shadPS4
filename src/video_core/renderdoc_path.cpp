// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderdoc_path.h"

#include <system_error>

namespace VideoCore {

std::string_view RenderDocModuleFilename() {
#ifdef _WIN32
    return "renderdoc.dll";
#elif defined(__APPLE__)
    return "librenderdoc.dylib";
#else
    return "librenderdoc.so";
#endif
}

std::optional<std::filesystem::path> ResolveRenderDocModulePath(
    const std::filesystem::path& configured_path) {
    if (configured_path.empty()) {
        return std::nullopt;
    }

    std::error_code error;
    if (std::filesystem::is_directory(configured_path, error)) {
        const auto library_path = configured_path / RenderDocModuleFilename();
        if (std::filesystem::is_regular_file(library_path, error)) {
            return library_path;
        }
        return std::nullopt;
    }
    if (!error && std::filesystem::is_regular_file(configured_path, error)) {
        return configured_path;
    }
    return std::nullopt;
}

} // namespace VideoCore
