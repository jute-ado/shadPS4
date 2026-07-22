// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/save_data/dialog/savedatadialog_navigation.h"

namespace SaveDataDialog = Libraries::SaveData::Dialog;

TEST(SaveDataDialogNavigation, EstablishesInitialListFocusWithoutReplayingHeldConfirm) {
    EXPECT_EQ(SaveDataDialog::ChooseListFocusAction(true, false, false),
              SaveDataDialog::ListFocusAction::Restore);
    EXPECT_EQ(SaveDataDialog::ChooseListFocusAction(true, false, true),
              SaveDataDialog::ListFocusAction::Restore);
}

TEST(SaveDataDialogNavigation, RestoresFocusAfterOpeningInputClearsNavigationId) {
    EXPECT_EQ(SaveDataDialog::ChooseListFocusAction(false, false, false),
              SaveDataDialog::ListFocusAction::Restore);
}

TEST(SaveDataDialogNavigation, PreservesConfirmEdgeWhenRestoringLostFocus) {
    EXPECT_EQ(SaveDataDialog::ChooseListFocusAction(false, false, true),
              SaveDataDialog::ListFocusAction::RestoreAndActivate);
}

TEST(SaveDataDialogNavigation, LeavesExistingNavigationFocusToImGui) {
    EXPECT_EQ(SaveDataDialog::ChooseListFocusAction(false, true, true),
              SaveDataDialog::ListFocusAction::None);
}
