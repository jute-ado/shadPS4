// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <tuple>
#include <vector>

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

TEST(SparseMemory, ZeroFillsRangeBeforeFirstUsableVma) {
    const std::map<VAddr, Core::VirtualMemoryArea> areas{
        {0x1000000,
         Core::VirtualMemoryArea{
             .base = 0x1000000,
             .size = 0x10000,
             .type = Core::VMAType::Direct,
             .prot = Core::MemoryProt::GpuRead,
         }},
    };
    std::vector<std::tuple<VAddr, u64, bool>> segments;

    ASSERT_TRUE(Core::ForEachSparseMemorySegment(
        areas, 0xffc000, 0x4000, [&](VAddr address, u64 size, bool readable) {
            segments.emplace_back(address, size, readable);
        }));

    EXPECT_EQ(segments,
              (std::vector<std::tuple<VAddr, u64, bool>>{{0xffc000, 0x4000, false}}));
}

TEST(SparseMemory, SplitsReadableVmasFromUnmappedGaps) {
    const std::map<VAddr, Core::VirtualMemoryArea> areas{
        {0x2000,
         Core::VirtualMemoryArea{
             .base = 0x2000,
             .size = 0x1000,
             .type = Core::VMAType::Direct,
             .prot = Core::MemoryProt::GpuRead,
         }},
        {0x4000,
         Core::VirtualMemoryArea{
             .base = 0x4000,
             .size = 0x1000,
             .type = Core::VMAType::Reserved,
             .prot = Core::MemoryProt::CpuRead,
         }},
    };
    std::vector<std::tuple<VAddr, u64, bool>> segments;

    ASSERT_TRUE(Core::ForEachSparseMemorySegment(
        areas, 0x1800, 0x3800, [&](VAddr address, u64 size, bool readable) {
            segments.emplace_back(address, size, readable);
        }));

    EXPECT_EQ(segments,
              (std::vector<std::tuple<VAddr, u64, bool>>{
                  {0x1800, 0x800, false},
                  {0x2000, 0x1000, true},
                  {0x3000, 0x1000, false},
                  {0x4000, 0x1000, false},
              }));
}

TEST(SparseMemory, RejectsOverflowingRanges) {
    const std::map<VAddr, Core::VirtualMemoryArea> areas;

    EXPECT_FALSE(Core::ForEachSparseMemorySegment(
        areas, std::numeric_limits<VAddr>::max() - 3, 8, [](VAddr, u64, bool) {}));
}
