// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "core/emulator_settings.h"

TEST(LogSettings, FlushLevelDefaultsToDisabled) {
    const LogSettings settings;

    EXPECT_TRUE(settings.flush_level.value.empty());
}

TEST(LogSettings, FlushLevelRoundTripsThroughJson) {
    LogSettings original;
    original.flush_level.set("Warning");

    const nlohmann::json encoded = original;
    ASSERT_TRUE(encoded.contains("flush_level"));
    EXPECT_EQ(encoded["flush_level"].get<std::string>(), "Warning");

    const auto decoded = encoded.get<LogSettings>();
    EXPECT_EQ(decoded.flush_level.value, "Warning");
}

TEST(LogSettings, FlushLevelIsGameOverrideable) {
    const LogSettings settings;
    const auto fields = settings.GetOverrideableFields();

    EXPECT_TRUE(std::ranges::any_of(fields, [](const OverrideItem& field) {
        return std::string(field.key) == "flush_level";
    }));
}
