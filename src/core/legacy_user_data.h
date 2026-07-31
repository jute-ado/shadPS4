// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>

namespace Core::UserMigration {

bool HasLegacySaveData(const std::filesystem::path& legacy_save_directory);

bool HasLegacyTrophyData(const std::filesystem::path& legacy_trophy_base_directory);

bool HasMigratableData(const std::filesystem::path& legacy_save_directory,
                       const std::filesystem::path& legacy_trophy_base_directory);

} // namespace Core::UserMigration
