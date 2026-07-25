// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/kernel/threads/pthread_attr_storage.h"

using Libraries::Kernel::ClonePthreadAttr;
using Libraries::Kernel::Cpuset;
using Libraries::Kernel::PthreadAttr;
using Libraries::Kernel::ReleasePthreadAttr;

TEST(PthreadAttrStorage, CloneOwnsAnIndependentAffinityCopy) {
    Cpuset source_cpuset{.bits = 0x15, ._reserved = 0x1234};
    PthreadAttr source{};
    source.cpusetsize = sizeof(source_cpuset);
    source.cpuset = &source_cpuset;
    PthreadAttr clone{};

    ASSERT_TRUE(ClonePthreadAttr(clone, source));
    ASSERT_NE(clone.cpuset, nullptr);
    EXPECT_NE(clone.cpuset, source.cpuset);
    EXPECT_EQ(clone.cpusetsize, sizeof(Cpuset));
    EXPECT_EQ(clone.cpuset->bits, 0x15);
    EXPECT_EQ(clone.cpuset->_reserved, 0x1234);

    source_cpuset.bits = 0x2a;
    EXPECT_EQ(clone.cpuset->bits, 0x15);

    ReleasePthreadAttr(clone);
    EXPECT_EQ(clone.cpuset, nullptr);
    EXPECT_EQ(clone.cpusetsize, 0);
}

TEST(PthreadAttrStorage, ReplacingCloneReleasesStaleAffinityState) {
    Cpuset source_cpuset{.bits = 0x3f};
    PthreadAttr with_affinity{};
    with_affinity.cpusetsize = sizeof(source_cpuset);
    with_affinity.cpuset = &source_cpuset;
    PthreadAttr destination{};

    ASSERT_TRUE(ClonePthreadAttr(destination, with_affinity));
    ASSERT_NE(destination.cpuset, nullptr);

    PthreadAttr without_affinity{};
    ASSERT_TRUE(ClonePthreadAttr(destination, without_affinity));
    EXPECT_EQ(destination.cpuset, nullptr);
    EXPECT_EQ(destination.cpusetsize, 0);
}

TEST(PthreadAttrStorage, SelfClonePreservesAffinity) {
    Cpuset source_cpuset{.bits = 0x30};
    PthreadAttr attr{};
    attr.cpusetsize = sizeof(source_cpuset);
    attr.cpuset = &source_cpuset;

    ASSERT_TRUE(ClonePthreadAttr(attr, attr));
    EXPECT_EQ(attr.cpuset, &source_cpuset);
    EXPECT_EQ(attr.cpuset->bits, 0x30);
}
