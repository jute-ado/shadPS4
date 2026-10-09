// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>

#include <gtest/gtest.h>

#include "core/windows_exception_stack.h"

TEST(WindowsExceptionStack, ProvidesAlignedIndependentStack) {
    Core::WindowsExceptionStack stack;

    void* const top = stack.TryAcquire();

    ASSERT_NE(top, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(top) % 16, 0);
    EXPECT_TRUE(stack.Contains(top));
    EXPECT_TRUE(stack.IsActive());
}

TEST(WindowsExceptionStack, NestedHandlerStaysOnAlreadyActiveStack) {
    Core::WindowsExceptionStack stack;

    ASSERT_NE(stack.TryAcquire(), nullptr);
    EXPECT_EQ(stack.TryAcquire(), nullptr);
    EXPECT_TRUE(stack.IsActive());
}

TEST(WindowsExceptionStack, CanBeReusedAfterRelease) {
    Core::WindowsExceptionStack stack;

    void* const first = stack.TryAcquire();
    ASSERT_NE(first, nullptr);
    stack.Release();

    EXPECT_FALSE(stack.IsActive());
    EXPECT_EQ(stack.TryAcquire(), first);
}

TEST(WindowsExceptionStack, PreparedThreadLocalStackSupportsHandlerEntryAndNesting) {
    Core::PrepareWindowsExceptionStack();

    void* const top = Core::AcquireWindowsExceptionStack();
    ASSERT_NE(top, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(top) % 16, 0);
    EXPECT_EQ(Core::AcquireWindowsExceptionStack(), nullptr);

    Core::ReleaseWindowsExceptionStack();
    EXPECT_EQ(Core::AcquireWindowsExceptionStack(), top);
    Core::ReleaseWindowsExceptionStack();
}
