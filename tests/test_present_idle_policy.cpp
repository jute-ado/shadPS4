// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/videoout/present_idle_policy.h"

namespace Libraries::VideoOut {
namespace {

TEST(PresentIdlePolicy, ServicesPresentedScreenshotWithoutGuestFlipOrSparePacingBudget) {
    EXPECT_EQ(SelectPresentIdleAction(false, true, false, true),
              PresentIdleAction::LastFrame);
}

TEST(PresentIdlePolicy, LeavesAnIdleOpenPortAloneWithoutAHostRedrawReason) {
    EXPECT_EQ(SelectPresentIdleAction(true, true, false, false), PresentIdleAction::None);
}

TEST(PresentIdlePolicy, PreservesExistingHostRedrawAndClosedPortBehavior) {
    EXPECT_EQ(SelectPresentIdleAction(true, true, true, false),
              PresentIdleAction::LastFrame);
    EXPECT_EQ(SelectPresentIdleAction(true, false, false, false),
              PresentIdleAction::BlankFrame);
}

} // namespace
} // namespace Libraries::VideoOut
