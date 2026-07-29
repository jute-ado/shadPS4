// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>

namespace Core::UserMigration {

/// Returns true when the portable user directory contains data supported by
/// the legacy save/trophy migration flow.
bool HasLegacyData(const std::filesystem::path& user_dir);

} // namespace Core::UserMigration
