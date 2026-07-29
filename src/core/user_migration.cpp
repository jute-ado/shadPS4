// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "user_migration.h"

namespace Core::UserMigration {
namespace fs = std::filesystem;

static bool HasLegacySaves(const fs::path& user_dir) {
    std::error_code error;
    const auto saves = user_dir / "savedata" / "1";
    return fs::is_directory(saves, error) && !fs::is_empty(saves, error) && !error;
}

static bool HasLegacyTrophies(const fs::path& user_dir) {
    std::error_code error;
    const auto games = user_dir / "game_data";
    fs::directory_iterator game_entries{games, error};
    if (error) {
        return false;
    }

    for (const auto& game : game_entries) {
        fs::directory_iterator trophy_entries{game.path() / "TrophyFiles", error};
        if (error) {
            error.clear();
            continue;
        }
        for (const auto& trophy : trophy_entries) {
            if (fs::is_regular_file(trophy.path() / "Xml" / "TROP.XML", error) && !error) {
                return true;
            }
            error.clear();
        }
    }
    return false;
}

bool HasLegacyData(const fs::path& user_dir) {
    return HasLegacySaves(user_dir) || HasLegacyTrophies(user_dir);
}

} // namespace Core::UserMigration
