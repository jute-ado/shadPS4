// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads.h"
#include "core/libraries/kernel/threads/thread_state.h"

namespace Libraries::Kernel {

int PS4_SYSV_ABI posix_pthread_attr_getinheritsched(const PthreadAttrT* attr, int* sched_inherit);
int PS4_SYSV_ABI scePthreadAttrGetaffinity(PthreadAttrT* attr, u64* mask);

ThreadState::ThreadState() = default;

int ThreadState::FindThread(Pthread*, bool) {
    return POSIX_ESRCH;
}

} // namespace Libraries::Kernel

using Libraries::Kernel::Cpuset;
using Libraries::Kernel::posix_pthread_attr_destroy;
using Libraries::Kernel::posix_pthread_attr_getaffinity_np;
using Libraries::Kernel::posix_pthread_attr_getinheritsched;
using Libraries::Kernel::posix_pthread_attr_init;
using Libraries::Kernel::posix_pthread_attr_setaffinity_np;
using Libraries::Kernel::PthreadAttrT;
using Libraries::Kernel::scePthreadAttrGetaffinity;

TEST(PthreadAttrApi, InitRejectsNullOutput) {
    EXPECT_EQ(posix_pthread_attr_init(nullptr), POSIX_EINVAL);
}

TEST(PthreadAttrApi, InitAndDestroyManageTheOpaqueAttribute) {
    PthreadAttrT attr = nullptr;

    ASSERT_EQ(posix_pthread_attr_init(&attr), 0);
    ASSERT_NE(attr, nullptr);
    EXPECT_EQ(posix_pthread_attr_destroy(&attr), 0);
    EXPECT_EQ(attr, nullptr);
}

TEST(PthreadAttrApi, GetAffinityRejectsNullOutput) {
    PthreadAttrT attr = nullptr;
    ASSERT_EQ(posix_pthread_attr_init(&attr), 0);

    EXPECT_EQ(posix_pthread_attr_getaffinity_np(&attr, sizeof(Libraries::Kernel::Cpuset), nullptr),
              POSIX_EINVAL);

    EXPECT_EQ(posix_pthread_attr_destroy(&attr), 0);
}

TEST(PthreadAttrApi, AffinityRoundTripsThroughTheOpaqueAttribute) {
    PthreadAttrT attr = nullptr;
    ASSERT_EQ(posix_pthread_attr_init(&attr), 0);
    const Cpuset requested{.bits = 0x15, ._reserved = 0x1234};

    ASSERT_EQ(posix_pthread_attr_setaffinity_np(&attr, sizeof(requested), &requested), 0);
    Cpuset actual{};
    ASSERT_EQ(posix_pthread_attr_getaffinity_np(&attr, sizeof(actual), &actual), 0);
    EXPECT_EQ(actual.bits, requested.bits);
    EXPECT_EQ(actual._reserved, requested._reserved);

    EXPECT_EQ(posix_pthread_attr_destroy(&attr), 0);
}

TEST(PthreadAttrApi, GetInheritScheduleRejectsNullOutput) {
    PthreadAttrT attr = nullptr;
    ASSERT_EQ(posix_pthread_attr_init(&attr), 0);

    EXPECT_EQ(posix_pthread_attr_getinheritsched(&attr, nullptr), POSIX_EINVAL);

    EXPECT_EQ(posix_pthread_attr_destroy(&attr), 0);
}

TEST(PthreadAttrApi, SceGetAffinityRejectsNullOutput) {
    PthreadAttrT attr = nullptr;
    ASSERT_EQ(posix_pthread_attr_init(&attr), 0);

    EXPECT_EQ(scePthreadAttrGetaffinity(&attr, nullptr), POSIX_EINVAL);

    EXPECT_EQ(posix_pthread_attr_destroy(&attr), 0);
}
