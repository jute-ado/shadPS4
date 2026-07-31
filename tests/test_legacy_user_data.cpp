// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "core/legacy_user_data.h"

namespace fs = std::filesystem;

class LegacyUserDataTest : public testing::Test {
protected:
    void SetUp() override {
        root = fs::temp_directory_path() /
               ("shadps4-legacy-user-data-test-" +
                std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        std::error_code ec;
        fs::remove_all(root, ec);
        ASSERT_TRUE(fs::create_directories(root));
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    fs::path root;
};

TEST_F(LegacyUserDataTest, MissingAndEmptyRootsDoNotRequireMigration) {
    const auto saves = root / "savedata" / "1";
    const auto trophies = root / "game_data";

    EXPECT_FALSE(Core::UserMigration::HasMigratableData(saves, trophies));

    ASSERT_TRUE(fs::create_directories(saves));
    ASSERT_TRUE(fs::create_directories(trophies));
    EXPECT_FALSE(Core::UserMigration::HasMigratableData(saves, trophies));
}

TEST_F(LegacyUserDataTest, SaveEntryRequiresMigration) {
    const auto saves = root / "savedata" / "1";
    ASSERT_TRUE(fs::create_directories(saves));
    std::ofstream(saves / "slot.dat") << "synthetic save";

    EXPECT_TRUE(Core::UserMigration::HasLegacySaveData(saves));
    EXPECT_TRUE(Core::UserMigration::HasMigratableData(saves, root / "game_data"));
}

TEST_F(LegacyUserDataTest, TrophyXmlRequiresMigration) {
    const auto trophies = root / "game_data";
    const auto xml = trophies / "CUSA00000" / "TrophyFiles" / "NPWR00000_00" / "Xml";
    ASSERT_TRUE(fs::create_directories(xml));
    std::ofstream(xml / "TROP.XML") << "<trophyconf npcommid=\"NPWR00000_00\"/>";

    EXPECT_TRUE(Core::UserMigration::HasLegacyTrophyData(trophies));
    EXPECT_TRUE(Core::UserMigration::HasMigratableData(root / "savedata" / "1", trophies));
}

TEST_F(LegacyUserDataTest, UnrelatedGameDataDoesNotRequireMigration) {
    const auto trophies = root / "game_data";
    ASSERT_TRUE(fs::create_directories(trophies / "CUSA00000" / "cache"));
    std::ofstream(trophies / "CUSA00000" / "cache" / "metadata.bin") << "synthetic";

    EXPECT_FALSE(Core::UserMigration::HasLegacyTrophyData(trophies));
    EXPECT_FALSE(Core::UserMigration::HasMigratableData(root / "savedata" / "1", trophies));
}
