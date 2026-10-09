// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/ipc/heap_size_summary.h"

TEST(HeapSizeSummary, RanksAllocationSizesByTotalBytesWithoutAllocatingOnInsert) {
    Core::Ipc::HeapSizeSummary summary;
    summary.Insert(128 * 1024, 0x1000);
    summary.Insert(128 * 1024, 0x2000);
    summary.Insert(1024 * 1024);
    summary.Insert(64);

    const auto top = summary.TopByTotalBytes();
    ASSERT_GE(top.size(), 3);
    EXPECT_EQ(top[0].size, 1024 * 1024);
    EXPECT_EQ(top[0].count, 1);
    EXPECT_EQ(top[1].size, 128 * 1024);
    EXPECT_EQ(top[1].count, 2);
    EXPECT_EQ(top[1].total_bytes, 256 * 1024);
    EXPECT_EQ(top[1].sample_address, 0x1000);
    EXPECT_EQ(top[1].last_sample_address, 0x2000);
}
