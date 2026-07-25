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

TEST(KernelMemoryValidation, ResolvesAlignedMemoryRangesWithoutOverflow) {
    EXPECT_EQ(Core::ResolveAlignedMemoryRangeStart(0x11000, 0xc000, 0x4000, 0x40000), 0x18000u);
    EXPECT_EQ(Core::ResolveAlignedMemoryRangeStart(0x18000, 0xc000, 0x4000, 0x40000), 0x18000u);

    EXPECT_FALSE(Core::ResolveAlignedMemoryRangeStart(0x1000, 0, 0x1000, 0x40000).has_value());
    EXPECT_FALSE(Core::ResolveAlignedMemoryRangeStart(
                     std::numeric_limits<u64>::max() - 0x1000, 0x4000, 0x1000,
                     std::numeric_limits<u64>::max())
                     .has_value());
    EXPECT_FALSE(
        Core::ResolveAlignedMemoryRangeStart(0x3f000, 0x1000, 0x2000, 0x40000).has_value());
}

TEST(KernelMemoryValidation, ResolvesBoundedAvailableMemorySpansAfterAlignment) {
    const auto aligned =
        Core::ResolveAvailableMemorySpan(0x1000, 0x9000, 0x2500, 0x8000, 0x2000);
    ASSERT_TRUE(aligned.has_value());
    EXPECT_EQ(aligned->base, 0x4000u);
    EXPECT_EQ(aligned->size, 0x4000u);

    const auto unaligned =
        Core::ResolveAvailableMemorySpan(0x1000, 0x9000, 0x2500, 0x8000, 0);
    ASSERT_TRUE(unaligned.has_value());
    EXPECT_EQ(unaligned->base, 0x2500u);
    EXPECT_EQ(unaligned->size, 0x5b00u);

    EXPECT_FALSE(
        Core::ResolveAvailableMemorySpan(0x1000, 0x1000, 0x3000, 0x4000, 0x1000).has_value());
    EXPECT_FALSE(Core::ResolveAvailableMemorySpan(
                     std::numeric_limits<u64>::max() - 0x1000, 0x2000, 0,
                     std::numeric_limits<u64>::max(), 0x1000)
                     .has_value());
    EXPECT_FALSE(Core::ResolveAvailableMemorySpan(
                     std::numeric_limits<u64>::max() - 0x1000, 0x1000, 0,
                     std::numeric_limits<u64>::max(), 0x4000)
                     .has_value());
}

TEST(KernelMemoryValidation, ClipsAndIntersectsDirectMemoryReleaseSpans) {
    const auto clipped = Core::ClipMemorySpanToLimit(0x3000, 0x3000, 0x4000);
    ASSERT_TRUE(clipped.has_value());
    EXPECT_EQ(clipped->base, 0x3000u);
    EXPECT_EQ(clipped->size, 0x1000u);

    const auto wrapped = Core::ClipMemorySpanToLimit(
        std::numeric_limits<u64>::max() - 0x1000, 0x4000,
        std::numeric_limits<u64>::max());
    ASSERT_TRUE(wrapped.has_value());
    EXPECT_EQ(wrapped->base, std::numeric_limits<u64>::max() - 0x1000);
    EXPECT_EQ(wrapped->size, 0x1000u);

    const auto allocated_overlap =
        Core::IntersectMemorySpans({.base = 0x1000, .size = 0x3000},
                                   {.base = 0x2800, .size = 0x2000});
    ASSERT_TRUE(allocated_overlap.has_value());
    EXPECT_EQ(allocated_overlap->base, 0x2800u);
    EXPECT_EQ(allocated_overlap->size, 0x1800u);

    const auto mapping_overlap =
        Core::IntersectMemorySpans({.base = 0x1000, .size = 0x1000},
                                   {.base = 0x1800, .size = 0x1000});
    ASSERT_TRUE(mapping_overlap.has_value());
    EXPECT_EQ(mapping_overlap->base, 0x1800u);
    EXPECT_EQ(mapping_overlap->size, 0x800u);

    EXPECT_FALSE(Core::ClipMemorySpanToLimit(0x4000, 0x1000, 0x4000).has_value());
    EXPECT_FALSE(Core::IntersectMemorySpans({.base = 0x1000, .size = 0x1000},
                                            {.base = 0x2000, .size = 0x1000})
                     .has_value());
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
