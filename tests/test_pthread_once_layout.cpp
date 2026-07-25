// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstddef>
#include <type_traits>

#include <gtest/gtest.h>

#include "core/libraries/kernel/threads/pthread.h"

namespace {

using Libraries::Kernel::PthreadMutexT;
using Libraries::Kernel::PthreadOnce;

TEST(PthreadOnceLayout, UsesGuestPthreadMutexHandle) {
    EXPECT_TRUE((std::is_same_v<decltype(PthreadOnce::mutex), PthreadMutexT>));
}

TEST(PthreadOnceLayout, MatchesGuestAbi) {
    EXPECT_TRUE(std::is_standard_layout_v<PthreadOnce>);
    EXPECT_EQ(offsetof(PthreadOnce, state), 0);
    EXPECT_EQ(offsetof(PthreadOnce, mutex), 8);
    EXPECT_EQ(sizeof(PthreadOnce), 16);
    EXPECT_EQ(alignof(PthreadOnce), 8);
}

} // namespace
