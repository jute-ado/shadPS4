// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <map>
#include <span>
#include <type_traits>
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

TEST(SparseMemory, CompleteReadableRangeCanCrossContiguousBackedAreas) {
    const std::map<VAddr, Core::VirtualMemoryArea> areas{
        {0x1000,
         Core::VirtualMemoryArea{
             .base = 0x1000,
             .size = 0x1000,
             .type = Core::VMAType::Direct,
             .prot = Core::MemoryProt::CpuRead,
         }},
        {0x2000,
         Core::VirtualMemoryArea{
             .base = 0x2000,
             .size = 0x1000,
             .type = Core::VMAType::File,
             .prot = Core::MemoryProt::CpuRead,
         }},
    };

    EXPECT_TRUE(Core::Detail::IsCompleteReadableBackingRange(areas, 0x1800, 0x1000));
}

TEST(SparseMemory, CompleteReadableRangeRejectsUnreadableTail) {
    const std::map<VAddr, Core::VirtualMemoryArea> areas{
        {0x1000,
         Core::VirtualMemoryArea{
             .base = 0x1000,
             .size = 0x1000,
             .type = Core::VMAType::Direct,
             .prot = Core::MemoryProt::CpuRead,
         }},
        {0x2000,
         Core::VirtualMemoryArea{
             .base = 0x2000,
             .size = 0x1000,
             .type = Core::VMAType::Free,
             .prot = Core::MemoryProt::NoAccess,
         }},
    };

    EXPECT_FALSE(Core::Detail::IsCompleteReadableBackingRange(areas, 0x1800, 0x1000));
}

TEST(SparseMemory, CompleteReadableRangeRejectsGapAndOverflow) {
    const std::map<VAddr, Core::VirtualMemoryArea> areas{
        {0x1000,
         Core::VirtualMemoryArea{
             .base = 0x1000,
             .size = 0x1000,
             .type = Core::VMAType::Direct,
             .prot = Core::MemoryProt::CpuRead,
         }},
        {0x3000,
         Core::VirtualMemoryArea{
             .base = 0x3000,
             .size = 0x1000,
             .type = Core::VMAType::Direct,
             .prot = Core::MemoryProt::CpuRead,
         }},
    };

    EXPECT_FALSE(Core::Detail::IsCompleteReadableBackingRange(areas, 0x1800, 0x1000));
    EXPECT_FALSE(Core::Detail::IsCompleteReadableBackingRange(
        areas, std::numeric_limits<VAddr>::max() - 3, 8));
}

TEST(SparseMemory, ExposesLockProtectedReadBackingApi) {
    using TryReadBacking = bool (Core::MemoryManager::*)(const void*, void*, u64);
    EXPECT_TRUE((std::is_same_v<decltype(&Core::MemoryManager::TryReadBacking), TryReadBacking>));
}

TEST(SparseMemory, CompleteReadableRangesRequireEveryCommandBufferByte) {
    const std::map<VAddr, Core::VirtualMemoryArea> areas{
        {0x1000,
         Core::VirtualMemoryArea{
             .base = 0x1000,
             .size = 0x1000,
             .type = Core::VMAType::Direct,
             .prot = Core::MemoryProt::CpuRead,
         }},
        {0x2000,
         Core::VirtualMemoryArea{
             .base = 0x2000,
             .size = 0x1000,
             .type = Core::VMAType::Direct,
             .prot = Core::MemoryProt::GpuRead,
         }},
    };
    const std::array readable_ranges{
        Core::GuestMemoryRange{0x1100, 0x200},
        Core::GuestMemoryRange{0x1f00, 0x200},
    };
    const std::array overflowing_ranges{
        Core::GuestMemoryRange{std::numeric_limits<VAddr>::max() - 3, 8},
    };

    EXPECT_TRUE(Core::Detail::AreCompleteReadableBackingRanges(areas, readable_ranges));
    EXPECT_FALSE(Core::Detail::AreCompleteReadableBackingRanges(areas, overflowing_ranges));

    auto unreadable_ranges = readable_ranges;
    unreadable_ranges[1] = Core::GuestMemoryRange{0x2f00, 0x200};
    EXPECT_FALSE(Core::Detail::AreCompleteReadableBackingRanges(areas, unreadable_ranges));
}

TEST(SparseMemory, ExposesBatchedReadableBackingApi) {
    using GuestRanges = std::span<const Core::GuestMemoryRange>;
    using CanReadBacking = bool (Core::MemoryManager::*)(GuestRanges);
    EXPECT_TRUE(
        (std::is_same_v<decltype(&Core::MemoryManager::CanReadBacking), CanReadBacking>));
}

TEST(SparseMemory, CompleteWritableRangesRejectAnyUnreadableOrOverflowingOutput) {
    const std::map<VAddr, Core::VirtualMemoryArea> areas{
        {0x1000,
         Core::VirtualMemoryArea{
             .base = 0x1000,
             .size = 0x1000,
             .type = Core::VMAType::Direct,
             .prot = Core::MemoryProt::CpuReadWrite,
         }},
        {0x2000,
         Core::VirtualMemoryArea{
             .base = 0x2000,
             .size = 0x1000,
             .type = Core::VMAType::Direct,
             .prot = Core::MemoryProt::CpuRead,
         }},
    };
    const std::array valid_ranges{
        Core::GuestMemoryRange{0x1100, 0x200},
        Core::GuestMemoryRange{0x1800, 0x400},
    };
    const std::array invalid_ranges{
        Core::GuestMemoryRange{0x1100, 0x200},
        Core::GuestMemoryRange{0x1f00, 0x200},
    };
    const std::array overflowing_ranges{
        Core::GuestMemoryRange{std::numeric_limits<VAddr>::max() - 3, 8},
    };

    EXPECT_TRUE(Core::Detail::AreCompleteWritableBackingRanges(areas, valid_ranges));
    EXPECT_FALSE(Core::Detail::AreCompleteWritableBackingRanges(areas, invalid_ranges));
    EXPECT_FALSE(Core::Detail::AreCompleteWritableBackingRanges(areas, overflowing_ranges));
}

TEST(SparseMemory, ExposesCheckedBatchedGuestOutputApis) {
    using GuestRanges = std::span<const Core::GuestMemoryRange>;
    using GuestWrites = std::span<const Core::GuestMemoryWrite>;
    using CanWriteBacking = bool (Core::MemoryManager::*)(GuestRanges);
    using TryClearBacking = bool (Core::MemoryManager::*)(GuestRanges);
    using TryWriteGuest = bool (Core::MemoryManager::*)(GuestWrites);

    EXPECT_TRUE(
        (std::is_same_v<decltype(&Core::MemoryManager::CanWriteBacking), CanWriteBacking>));
    EXPECT_TRUE(
        (std::is_same_v<decltype(&Core::MemoryManager::TryClearBacking), TryClearBacking>));
    EXPECT_TRUE((std::is_same_v<decltype(&Core::MemoryManager::TryWriteGuest), TryWriteGuest>));
}

TEST(SparseMemory, CompleteGuestWritesRejectNullSourceBeforeAnyCommit) {
    const std::map<VAddr, Core::VirtualMemoryArea> areas{
        {0x1000,
         Core::VirtualMemoryArea{
             .base = 0x1000,
             .size = 0x1000,
             .type = Core::VMAType::Direct,
             .prot = Core::MemoryProt::CpuReadWrite,
         }},
    };
    const std::array<u8, 4> source{};
    const std::array valid_writes{
        Core::GuestMemoryWrite{0x1100, source.data(), source.size()},
        Core::GuestMemoryWrite{0x1200, source.data(), source.size()},
    };
    const std::array invalid_writes{
        Core::GuestMemoryWrite{0x1100, source.data(), source.size()},
        Core::GuestMemoryWrite{0x1200, nullptr, source.size()},
    };

    EXPECT_TRUE(Core::Detail::AreCompleteWritableGuestWrites(areas, valid_writes));
    EXPECT_FALSE(Core::Detail::AreCompleteWritableGuestWrites(areas, invalid_writes));
}
