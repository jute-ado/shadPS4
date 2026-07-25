// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>

#include "core/libraries/kernel/memory.h"
#include "core/memory.h"

namespace Libraries::Kernel {
namespace {

TEST(KernelMemoryValidation, RejectsMissingNamedDirectMemoryPointers) {
    void* address{};

    EXPECT_FALSE(IsMemoryAddressStorageValid(nullptr));
    EXPECT_FALSE(AreNamedMemoryPointersValid(nullptr, "anon"));
    EXPECT_FALSE(AreNamedMemoryPointersValid(&address, nullptr));
}

TEST(KernelMemoryValidation, AcceptsAddressStorageAndName) {
    void* address{};

    EXPECT_TRUE(IsMemoryAddressStorageValid(&address));
    EXPECT_TRUE(AreNamedMemoryPointersValid(&address, "anon"));
}

TEST(KernelMemoryValidation, RejectsWrappedVirtualMemoryContainmentRange) {
    const Core::VirtualMemoryArea area{
        .base = std::numeric_limits<VAddr>::max() - 0xff,
        .size = 0x80,
    };

    EXPECT_FALSE(area.Contains(std::numeric_limits<VAddr>::max() - 0x3f, 0x80));
}

TEST(KernelMemoryValidation, DetectsOverlapBeforeWrappedRangeEnd) {
    const Core::VirtualMemoryArea area{
        .base = std::numeric_limits<VAddr>::max() - 0x3f,
        .size = 0x30,
    };

    EXPECT_TRUE(area.Overlaps(std::numeric_limits<VAddr>::max() - 0x7f, 0x100));
}

TEST(KernelMemoryValidation, EmptyVirtualMemoryRangeDoesNotOverlap) {
    const Core::VirtualMemoryArea area{
        .base = 0x1000,
        .size = 0x1000,
    };

    EXPECT_FALSE(area.Overlaps(0x1800, 0));
}

TEST(KernelMemoryValidation, VirtualMemoryRangesUseHalfOpenEndpoints) {
    const Core::VirtualMemoryArea area{
        .base = 0x1000,
        .size = 0x1000,
    };

    EXPECT_TRUE(area.Contains(0x1000, 0x1000));
    EXPECT_TRUE(area.Contains(0x2000, 0));
    EXPECT_FALSE(area.Contains(0x2000, 1));
    EXPECT_FALSE(area.Overlaps(0x0800, 0x800));
    EXPECT_FALSE(area.Overlaps(0x2000, 0x800));
    EXPECT_TRUE(area.Overlaps(0x0800, 0x801));
    EXPECT_TRUE(area.Overlaps(0x1fff, 1));
}

TEST(KernelMemoryValidation, GpuRangeMayEndAtFortyBitBoundary) {
    constexpr VAddr gpu_address_limit = 0x10000000000;

    EXPECT_TRUE(Core::MemoryManager::IsValidGpuMapping(gpu_address_limit - 1, 1));
    EXPECT_TRUE(Core::MemoryManager::IsValidGpuMapping(0, gpu_address_limit));
    EXPECT_FALSE(Core::MemoryManager::IsValidGpuMapping(gpu_address_limit - 1, 2));
}

TEST(KernelMemoryValidation, RejectsWrappedGpuAddressRange) {
    EXPECT_FALSE(Core::MemoryManager::IsValidGpuMapping(
        std::numeric_limits<VAddr>::max() - 1, 4));
}

TEST(KernelMemoryValidation, ValidatesRangesAndBudgetsWithoutOverflow) {
    EXPECT_TRUE(Core::IsMemoryRangeWithinLimit(0x1000, 0x400, 0x600));
    EXPECT_TRUE(Core::IsMemoryRangeWithinLimit(0x1000, 0x1000, 0));

    EXPECT_FALSE(Core::IsMemoryRangeWithinLimit(0x1000, 0x1001, 0));
    EXPECT_FALSE(Core::IsMemoryRangeWithinLimit(0x1000, 0xf00, 0x101));
    EXPECT_FALSE(Core::IsMemoryRangeWithinLimit(
        std::numeric_limits<u64>::max(), std::numeric_limits<u64>::max() - 1, 2));
}

TEST(KernelMemoryValidation, RejectsWrappedPhysicalAreaAdjacency) {
    Core::PhysicalMemoryArea area{
        .base = std::numeric_limits<PAddr>::max() - 0xff,
        .size = 0x200,
    };
    Core::PhysicalMemoryArea low_area{
        .base = 0x100,
        .size = 0x100,
    };

    EXPECT_FALSE(area.CanMergeWith(low_area));
}

TEST(KernelMemoryValidation, RejectsWrappedVirtualAreaAdjacency) {
    Core::VirtualMemoryArea area{
        .base = std::numeric_limits<VAddr>::max() - 0xff,
        .size = 0x200,
    };
    Core::VirtualMemoryArea low_area{
        .base = 0x100,
        .size = 0x100,
    };

    EXPECT_FALSE(area.CanMergeWith(low_area));
}

TEST(KernelMemoryValidation, AdjacentMemoryAreasRemainMergeable) {
    Core::PhysicalMemoryArea physical{
        .base = 0x1000,
        .size = 0x1000,
    };
    Core::PhysicalMemoryArea next_physical{
        .base = 0x2000,
        .size = 0x1000,
    };
    Core::VirtualMemoryArea virtual_area{
        .base = 0x1000,
        .size = 0x1000,
    };
    Core::VirtualMemoryArea next_virtual{
        .base = 0x2000,
        .size = 0x1000,
    };

    EXPECT_TRUE(physical.CanMergeWith(next_physical));
    EXPECT_TRUE(virtual_area.CanMergeWith(next_virtual));
}

} // namespace
} // namespace Libraries::Kernel
