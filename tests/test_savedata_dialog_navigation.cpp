// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/save_data/dialog/gamepad_input_capture.h"
#include "core/libraries/save_data/dialog/savedatadialog_navigation.h"

namespace SaveDataDialog = Libraries::SaveData::Dialog;

namespace {

int acquire_count;
int release_count;

void AcquireInput() {
    ++acquire_count;
}

void ReleaseInput() {
    ++release_count;
}

} // namespace

TEST(SaveDataDialogInputCapture, ReleasesExactlyOnceWhenMovedAndReset) {
    acquire_count = 0;
    release_count = 0;

    {
        SaveDataDialog::GamepadInputCapture capture{AcquireInput, ReleaseInput};
        EXPECT_EQ(acquire_count, 1);
        EXPECT_EQ(release_count, 0);

        SaveDataDialog::GamepadInputCapture moved{std::move(capture)};
        EXPECT_EQ(acquire_count, 1);
        EXPECT_EQ(release_count, 0);

        capture.Reset();
        EXPECT_EQ(release_count, 0);

        moved.Reset();
        EXPECT_EQ(release_count, 1);
    }

    EXPECT_EQ(release_count, 1);
}

TEST(SaveDataDialogInputCapture, MoveAssignmentReleasesThePreviousLease) {
    acquire_count = 0;
    release_count = 0;

    SaveDataDialog::GamepadInputCapture first{AcquireInput, ReleaseInput};
    SaveDataDialog::GamepadInputCapture second{AcquireInput, ReleaseInput};
    EXPECT_EQ(acquire_count, 2);

    second = std::move(first);
    EXPECT_EQ(release_count, 1);

    second.Reset();
    EXPECT_EQ(release_count, 2);
}

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
