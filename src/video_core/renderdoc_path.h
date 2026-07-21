// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <optional>
#include <string_view>

namespace VideoCore {

std::string_view RenderDocModuleFilename();

std::optional<std::filesystem::path> ResolveRenderDocModulePath(
    const std::filesystem::path& configured_path);

} // namespace VideoCore
