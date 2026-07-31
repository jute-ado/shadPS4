// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/memory.h"

TEST(SparseMemory, RejectsCommittedNoAccessRangeAsCopySource) {
    const Core::VirtualMemoryArea area{
        .type = Core::VMAType::Direct,
        .prot = Core::MemoryProt::NoAccess,
    };

    EXPECT_TRUE(area.IsMapped());
    EXPECT_FALSE(area.HasReadableBacking());
}

TEST(SparseMemory, AcceptsGpuReadableRangeAsCopySource) {
    const Core::VirtualMemoryArea area{
        .type = Core::VMAType::Direct,
        .prot = Core::MemoryProt::GpuRead,
    };

    EXPECT_TRUE(area.HasReadableBacking());
}

TEST(SparseMemory, RejectsReservedRangeEvenWithReadFlag) {
    const Core::VirtualMemoryArea area{
        .type = Core::VMAType::Reserved,
        .prot = Core::MemoryProt::CpuRead,
    };

    EXPECT_FALSE(area.HasReadableBacking());
}
