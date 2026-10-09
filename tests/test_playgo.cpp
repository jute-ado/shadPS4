// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/playgo/playgo.h"

namespace Libraries::PlayGo {
namespace {

TEST(PlayGoLanguageMask, MapsSupportedSystemLanguagesAcrossTheFullMaskWidth) {
    EXPECT_EQ(ConvertLanguageToMask(0), u64{1} << 63);
    EXPECT_EQ(ConvertLanguageToMask(31), u64{1} << 32);
    EXPECT_EQ(ConvertLanguageToMask(32), u64{1} << 31);
    EXPECT_EQ(ConvertLanguageToMask(47), u64{1} << 16);
}

TEST(PlayGoLanguageMask, RejectsSystemLanguagesOutsideTheSupportedRange) {
    EXPECT_EQ(ConvertLanguageToMask(-1), 0);
    EXPECT_EQ(ConvertLanguageToMask(48), 0);
}

} // namespace
} // namespace Libraries::PlayGo
