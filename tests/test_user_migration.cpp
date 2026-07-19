// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "core/user_migration.h"

namespace {
namespace fs = std::filesystem;

class UserMigrationTest : public testing::Test {
protected:
    void SetUp() override {
        root = fs::temp_directory_path() / "shadps4-user-migration-test";
        fs::remove_all(root);
        fs::create_directories(root);
    }

    void TearDown() override {
        fs::remove_all(root);
    }

    fs::path root;
};

TEST_F(UserMigrationTest, FreshPortableProfileDoesNotNeedMigration) {
    EXPECT_FALSE(Core::UserMigration::HasLegacyData(root));
}

TEST_F(UserMigrationTest, EmptyLegacyDirectoriesDoNotNeedMigration) {
    fs::create_directories(root / "savedata" / "1");
    fs::create_directories(root / "game_data");

    EXPECT_FALSE(Core::UserMigration::HasLegacyData(root));
}

TEST_F(UserMigrationTest, ExistingLegacySaveNeedsMigration) {
    const auto save = root / "savedata" / "1" / "CUSA00001" / "save.bin";
    fs::create_directories(save.parent_path());
    std::ofstream(save) << "save";

    EXPECT_TRUE(Core::UserMigration::HasLegacyData(root));
}

TEST_F(UserMigrationTest, ExistingLegacyTrophyNeedsMigration) {
    const auto trophy =
        root / "game_data" / "CUSA00001" / "TrophyFiles" / "NPWR00001_00" / "Xml" / "TROP.XML";
    fs::create_directories(trophy.parent_path());
    std::ofstream(trophy) << "<trophyconf/>";

    EXPECT_TRUE(Core::UserMigration::HasLegacyData(root));
}

TEST_F(UserMigrationTest, UnrelatedGameDataDoesNotNeedMigration) {
    const auto unrelated = root / "game_data" / "CUSA00001" / "cache.bin";
    fs::create_directories(unrelated.parent_path());
    std::ofstream(unrelated) << "cache";

    EXPECT_FALSE(Core::UserMigration::HasLegacyData(root));
}

} // namespace
