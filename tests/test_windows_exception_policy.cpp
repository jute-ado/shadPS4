// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/windows_exception_policy.h"

namespace {

TEST(WindowsExceptionPolicy, CaughtMsvcCppExceptionDoesNotBeginEmulatorShutdown) {
    EXPECT_FALSE(Core::WindowsException::ShouldShutdownForUnclaimedException(
        Core::WindowsException::MsvcCppExceptionCode));
}

TEST(WindowsExceptionPolicy, UnclaimedAccessViolationStillBeginsEmulatorShutdown) {
    EXPECT_TRUE(Core::WindowsException::ShouldShutdownForUnclaimedException(0xc0000005));
}

TEST(WindowsExceptionPolicy, BreakpointRemainsOwnedByAssertPath) {
    EXPECT_FALSE(Core::WindowsException::ShouldShutdownForUnclaimedException(
        Core::WindowsException::BreakpointExceptionCode));
}

} // namespace
