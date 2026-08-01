// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "legacy_user_data.h"

namespace fs = std::filesystem;

namespace Core::UserMigration {
namespace {

bool IsDirectory(const fs::path& path) {
    std::error_code ec;
    return fs::is_directory(path, ec);
}

} // namespace

bool HasLegacySaveData(const fs::path& legacy_save_directory) {
    if (!IsDirectory(legacy_save_directory)) {
        return false;
    }

    std::error_code ec;
    const fs::directory_iterator first{legacy_save_directory,
                                       fs::directory_options::skip_permission_denied, ec};
    return !ec && first != fs::directory_iterator{};
}

bool HasLegacyTrophyData(const fs::path& legacy_trophy_base_directory) {
    if (!IsDirectory(legacy_trophy_base_directory)) {
        return false;
    }

    std::error_code ec;
    for (fs::directory_iterator game_it{legacy_trophy_base_directory,
                                        fs::directory_options::skip_permission_denied, ec},
         end;
         !ec && game_it != end; game_it.increment(ec)) {
        if (!IsDirectory(game_it->path())) {
            continue;
        }

        const auto trophy_files = game_it->path() / "TrophyFiles";
        if (!IsDirectory(trophy_files)) {
            continue;
        }

        std::error_code trophy_ec;
        for (fs::directory_iterator
                 trophy_it{trophy_files, fs::directory_options::skip_permission_denied, trophy_ec},
             trophy_end;
             !trophy_ec && trophy_it != trophy_end; trophy_it.increment(trophy_ec)) {
            if (!IsDirectory(trophy_it->path())) {
                continue;
            }

            std::error_code file_ec;
            if (fs::is_regular_file(trophy_it->path() / "Xml" / "TROP.XML", file_ec)) {
                return true;
            }
        }
    }

    return false;
}

bool HasMigratableData(const fs::path& legacy_save_directory,
                       const fs::path& legacy_trophy_base_directory) {
    return HasLegacySaveData(legacy_save_directory) ||
           HasLegacyTrophyData(legacy_trophy_base_directory);
}

} // namespace Core::UserMigration
