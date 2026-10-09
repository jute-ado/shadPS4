// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/save_data/dialog/save_dialog_input.h"

TEST(SaveDialogInput, DetectsNewVirtualButtonPress) {
    constexpr u32 cross = 0x4000;

    EXPECT_TRUE(Libraries::SaveData::Dialog::IsNewButtonPress(cross, 0, cross));
}

TEST(SaveDialogInput, IgnoresHeldAndReleasedVirtualButton) {
    constexpr u32 cross = 0x4000;

    EXPECT_FALSE(Libraries::SaveData::Dialog::IsNewButtonPress(cross, cross, cross));
    EXPECT_FALSE(Libraries::SaveData::Dialog::IsNewButtonPress(0, cross, cross));
}

TEST(SaveDialogInput, IgnoresUnrelatedVirtualButton) {
    constexpr u32 cross = 0x4000;
    constexpr u32 circle = 0x2000;

    EXPECT_FALSE(Libraries::SaveData::Dialog::IsNewButtonPress(circle, 0, cross));
}

TEST(SaveDialogInput, DoesNotCarryHeldButtonIntoNewDialog) {
    constexpr u32 cross = 0x4000;
    using Libraries::SaveData::Dialog::IsNewButtonPressAfterInitialization;

    EXPECT_FALSE(IsNewButtonPressAfterInitialization(false, cross, 0, cross));
    EXPECT_TRUE(IsNewButtonPressAfterInitialization(true, cross, 0, cross));
}

TEST(SaveDialogInput, MovesSelectionWithinListBounds) {
    using Libraries::SaveData::Dialog::MoveSelection;

    EXPECT_EQ(MoveSelection(1, 3, -1), 0);
    EXPECT_EQ(MoveSelection(1, 3, 1), 2);
    EXPECT_EQ(MoveSelection(0, 3, -1), 0);
    EXPECT_EQ(MoveSelection(2, 3, 1), 2);
}

TEST(SaveDialogInput, RejectsSelectionForEmptyList) {
    EXPECT_EQ(Libraries::SaveData::Dialog::MoveSelection(0, 0, 1), -1);
}
