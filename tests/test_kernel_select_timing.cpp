// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <gtest/gtest.h>

#include "core/libraries/kernel/select_timing.h"

namespace {
using namespace std::chrono_literals;

TEST(KernelSelectTiming, WaitsForEmptySelectTimeout) {
    const auto started = std::chrono::steady_clock::now();

    const bool valid = Libraries::Kernel::WaitForSelectTimeout(0, 16'000);

    const auto elapsed = std::chrono::steady_clock::now() - started;
    EXPECT_TRUE(valid);
    EXPECT_GE(elapsed, 12ms);
    EXPECT_LT(elapsed, 500ms);
}

TEST(KernelSelectTiming, ZeroTimeoutReturnsImmediately) {
    const auto started = std::chrono::steady_clock::now();

    const bool valid = Libraries::Kernel::WaitForSelectTimeout(0, 0);

    EXPECT_TRUE(valid);
    EXPECT_LT(std::chrono::steady_clock::now() - started, 100ms);
}

TEST(KernelSelectTiming, RejectsInvalidTimeout) {
    EXPECT_FALSE(Libraries::Kernel::WaitForSelectTimeout(-1, 0));
    EXPECT_FALSE(Libraries::Kernel::WaitForSelectTimeout(0, -1));
    EXPECT_FALSE(Libraries::Kernel::WaitForSelectTimeout(0, 1'000'000));
}

} // namespace
