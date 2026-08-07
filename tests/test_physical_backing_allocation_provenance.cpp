// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <limits>
#include <map>

#include "core/memory.h"

namespace {

using Core::AcquirePhysicalBacking;
using Core::CollectPhysicalBackingSpans;
using Core::PhysicalBackingGenerationSource;
using Core::PhysicalBackingMappingClass;
using Core::PhysicalMemoryArea;
using Core::PhysicalMemoryType;
using Core::RetirePhysicalBacking;
using Core::SplitPhysicalBackingArea;

constexpr u64 PageSize = 16_KB;
constexpr VAddr GuestBase = 0x1000'0000;

PhysicalMemoryArea MakeArea(PAddr base, u64 size, u64 generation,
                            PhysicalMemoryType type = PhysicalMemoryType::Allocated) {
    return {
        .base = base,
        .size = size,
        .memory_type = 3,
        .dma_type = type,
        .allocation_generation = generation,
    };
}

PhysicalMemoryArea MakeMappedArea(PAddr base, u64 size, u64 generation) {
    return MakeArea(base, size, generation, PhysicalMemoryType::Mapped);
}

} // namespace

TEST(PhysicalBackingAllocationProvenance, DirectAndPoolOwnershipUseFreshGlobalGenerations) {
    PhysicalBackingGenerationSource generations;
    auto direct = MakeArea(0, PageSize, 0, PhysicalMemoryType::Free);
    auto pool = MakeArea(PageSize, PageSize, 0, PhysicalMemoryType::Free);

    AcquirePhysicalBacking(direct, generations);
    direct.dma_type = PhysicalMemoryType::Allocated;
    AcquirePhysicalBacking(pool, generations);
    pool.dma_type = PhysicalMemoryType::Pooled;

    EXPECT_NE(direct.allocation_generation, 0u);
    EXPECT_GT(pool.allocation_generation, direct.allocation_generation);
}

TEST(PhysicalBackingAllocationProvenance, AliasSplitAndOwnershipTransitionsPreserveGeneration) {
    const auto original = MakeArea(4 * PageSize, 4 * PageSize, 17);
    const auto alias = original;
    auto [first, second] = SplitPhysicalBackingArea(original, 2 * PageSize);

    EXPECT_EQ(alias.allocation_generation, 17u);
    EXPECT_EQ(first.allocation_generation, 17u);
    EXPECT_EQ(second.allocation_generation, 17u);
    EXPECT_TRUE(first.CanMergeWith(second));

    second.dma_type = PhysicalMemoryType::Mapped;
    EXPECT_EQ(second.allocation_generation, 17u);
    second.dma_type = PhysicalMemoryType::Committed;
    EXPECT_EQ(second.allocation_generation, 17u);
    second.dma_type = PhysicalMemoryType::Pooled;
    EXPECT_EQ(second.allocation_generation, 17u);
}

TEST(PhysicalBackingAllocationProvenance, PartialFreeRetiresOnlyFreedSpanAndReuseIsNewer) {
    PhysicalBackingGenerationSource generations{40};
    auto original = MakeArea(8 * PageSize, 4 * PageSize, generations.Acquire());
    auto [retained, freed] = SplitPhysicalBackingArea(original, 2 * PageSize);

    const auto retirement = RetirePhysicalBacking(freed);
    ASSERT_TRUE(retirement.has_value());
    EXPECT_EQ(retirement->physical_offset, 10 * PageSize);
    EXPECT_EQ(retirement->size, 2 * PageSize);
    EXPECT_EQ(retirement->allocation_generation, original.allocation_generation);
    EXPECT_EQ(retained.allocation_generation, original.allocation_generation);
    EXPECT_EQ(freed.allocation_generation, 0u);

    AcquirePhysicalBacking(freed, generations);
    EXPECT_GT(freed.allocation_generation, retained.allocation_generation);
}

TEST(PhysicalBackingAllocationProvenance, AdjacentDifferentAllocationsNeverMerge) {
    const auto first = MakeArea(0, PageSize, 1);
    const auto second = MakeArea(PageSize, PageSize, 2);
    EXPECT_FALSE(first.CanMergeWith(second));
}

TEST(PhysicalBackingAllocationProvenance, OverflowingPhysicalAreasNeverMerge) {
    constexpr u64 Max = std::numeric_limits<u64>::max();
    const auto wrapping_first = MakeArea(Max - PageSize + 1, PageSize, 1);
    const auto wrapped_second = MakeArea(0, PageSize, 1);
    EXPECT_FALSE(wrapping_first.CanMergeWith(wrapped_second));

    const auto oversized_first = MakeArea(0, Max - 7, 2);
    const auto oversized_second = MakeArea(Max - 7, 16, 2);
    EXPECT_FALSE(oversized_first.CanMergeWith(oversized_second));

    const auto valid_first = MakeArea(100, 50, 3);
    const auto valid_second = MakeArea(150, 25, 3);
    EXPECT_TRUE(valid_first.CanMergeWith(valid_second));
}

TEST(PhysicalBackingAllocationProvenance, CollectsExactMultiplePhysicalSpans) {
    const std::map<uintptr_t, PhysicalMemoryArea> areas{
        {0, MakeArea(5 * PageSize, 2 * PageSize, 7, PhysicalMemoryType::Committed)},
        {2 * PageSize, MakeArea(20 * PageSize, PageSize, 8, PhysicalMemoryType::Committed)},
    };

    const auto spans = CollectPhysicalBackingSpans(
        GuestBase, 3 * PageSize, PhysicalBackingMappingClass::Pooled, true, areas);
    ASSERT_TRUE(spans.has_value());
    ASSERT_EQ(spans->size(), 2u);
    EXPECT_EQ((*spans)[0].guest_base, GuestBase);
    EXPECT_EQ((*spans)[0].physical_offset, 5 * PageSize);
    EXPECT_EQ((*spans)[0].size, 2 * PageSize);
    EXPECT_EQ((*spans)[0].allocation_generation, 7u);
    EXPECT_EQ((*spans)[1].guest_base, GuestBase + 2 * PageSize);
    EXPECT_EQ((*spans)[1].physical_offset, 20 * PageSize);
    EXPECT_EQ((*spans)[1].allocation_generation, 8u);
}

TEST(PhysicalBackingAllocationProvenance, AliasesCollectTheSameCanonicalPhysicalPage) {
    const std::map<uintptr_t, PhysicalMemoryArea> areas{
        {0, MakeMappedArea(9 * PageSize, PageSize, 3)},
    };

    const auto first = CollectPhysicalBackingSpans(
        GuestBase, PageSize, PhysicalBackingMappingClass::Direct, true, areas);
    const auto second = CollectPhysicalBackingSpans(
        GuestBase + 8 * PageSize, PageSize, PhysicalBackingMappingClass::Direct, true, areas);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(first->front().physical_offset, second->front().physical_offset);
    EXPECT_EQ(first->front().allocation_generation, second->front().allocation_generation);
    EXPECT_NE(first->front().guest_base, second->front().guest_base);
}

TEST(PhysicalBackingAllocationProvenance, VirtualMemoryAreaCollectsOnlyEligibleBackingClasses) {
    Core::VirtualMemoryArea area{};
    area.base = GuestBase;
    area.size = PageSize;
    area.type = Core::VMAType::Direct;
    area.physical_backing_eligible = true;
    area.phys_areas.emplace(0, MakeMappedArea(6 * PageSize, PageSize, 4));

    const auto direct = area.CollectPhysicalBackingSpans();
    ASSERT_TRUE(direct.has_value());
    ASSERT_EQ(direct->size(), 1u);
    EXPECT_EQ(direct->front().physical_offset, 6 * PageSize);

    area.physical_backing_eligible = false;
    EXPECT_FALSE(area.CollectPhysicalBackingSpans());
    area.physical_backing_eligible = true;
    area.type = Core::VMAType::Flexible;
    EXPECT_FALSE(area.CollectPhysicalBackingSpans());
    area.type = Core::VMAType::File;
    EXPECT_FALSE(area.CollectPhysicalBackingSpans());
}

TEST(PhysicalBackingAllocationProvenance,
     UntrackedCpuWritableMappingCannotPublishImportedBacking) {
    Core::VirtualMemoryArea area{};
    area.base = GuestBase;
    area.size = PageSize;
    area.type = Core::VMAType::Direct;
    area.prot = Core::MemoryProt::CpuReadWrite | Core::MemoryProt::GpuReadWrite;
    area.physical_backing_eligible = true;
    area.phys_areas.emplace(0, MakeMappedArea(6 * PageSize, PageSize, 4));

    EXPECT_FALSE(area.CollectPhysicalBackingSpans());
}

TEST(PhysicalBackingAllocationProvenance, MappingClassRejectsEveryMismatchedDmaState) {
    constexpr PhysicalMemoryType Types[]{
        PhysicalMemoryType::Free,   PhysicalMemoryType::Allocated, PhysicalMemoryType::Mapped,
        PhysicalMemoryType::Pooled, PhysicalMemoryType::Committed, PhysicalMemoryType::Flexible,
    };

    for (const auto type : Types) {
        const std::map<uintptr_t, PhysicalMemoryArea> areas{
            {0, MakeArea(0, PageSize, 1, type)},
        };
        const bool direct_expected = type == PhysicalMemoryType::Mapped;
        const bool pooled_expected = type == PhysicalMemoryType::Committed;
        EXPECT_EQ(CollectPhysicalBackingSpans(GuestBase, PageSize,
                                              PhysicalBackingMappingClass::Direct, true, areas)
                      .has_value(),
                  direct_expected);
        EXPECT_EQ(CollectPhysicalBackingSpans(GuestBase, PageSize,
                                              PhysicalBackingMappingClass::Pooled, true, areas)
                      .has_value(),
                  pooled_expected);
    }
}

TEST(PhysicalBackingAllocationProvenance, UnsupportedAndPrivateMappingsFailClosed) {
    const std::map<uintptr_t, PhysicalMemoryArea> areas{
        {0, MakeMappedArea(0, PageSize, 1)},
    };

    EXPECT_FALSE(CollectPhysicalBackingSpans(GuestBase, PageSize,
                                             PhysicalBackingMappingClass::Direct, false, areas));
    EXPECT_FALSE(CollectPhysicalBackingSpans(GuestBase, PageSize,
                                             PhysicalBackingMappingClass::Flexible, true, areas));
    EXPECT_FALSE(CollectPhysicalBackingSpans(GuestBase, PageSize, PhysicalBackingMappingClass::File,
                                             true, areas));
    EXPECT_FALSE(CollectPhysicalBackingSpans(
        GuestBase, PageSize, PhysicalBackingMappingClass::Unsupported, true, areas));
}

TEST(PhysicalBackingAllocationProvenance, MalformedOrIncompleteMappingsFailClosed) {
    const std::map<uintptr_t, PhysicalMemoryArea> valid{
        {0, MakeMappedArea(0, PageSize, 1)},
    };
    const std::map<uintptr_t, PhysicalMemoryArea> gap{
        {0, MakeMappedArea(0, PageSize, 1)},
        {2 * PageSize, MakeMappedArea(2 * PageSize, PageSize, 1)},
    };
    const std::map<uintptr_t, PhysicalMemoryArea> zero_generation{
        {0, MakeMappedArea(0, PageSize, 0)},
    };
    const std::map<uintptr_t, PhysicalMemoryArea> overcomplete{
        {0, MakeMappedArea(0, PageSize, 1)},
        {PageSize, MakeMappedArea(PageSize, PageSize, 1)},
    };
    const std::map<uintptr_t, PhysicalMemoryArea> unaligned_physical{
        {0, MakeMappedArea(1, PageSize, 1)},
    };
    const std::map<uintptr_t, PhysicalMemoryArea> physical_overflow{
        {0, MakeMappedArea(std::numeric_limits<u64>::max() - (PageSize - 1), 2 * PageSize, 1)},
    };
    const std::map<uintptr_t, PhysicalMemoryArea> two_pages{
        {0, MakeMappedArea(0, 2 * PageSize, 1)},
    };

    EXPECT_FALSE(CollectPhysicalBackingSpans(GuestBase + 1, PageSize,
                                             PhysicalBackingMappingClass::Direct, true, valid));
    EXPECT_FALSE(CollectPhysicalBackingSpans(GuestBase, PageSize - 1,
                                             PhysicalBackingMappingClass::Direct, true, valid));
    EXPECT_FALSE(CollectPhysicalBackingSpans(GuestBase, 2 * PageSize,
                                             PhysicalBackingMappingClass::Direct, true, gap));
    EXPECT_FALSE(CollectPhysicalBackingSpans(
        GuestBase, PageSize, PhysicalBackingMappingClass::Direct, true, zero_generation));
    EXPECT_FALSE(CollectPhysicalBackingSpans(
        GuestBase, PageSize, PhysicalBackingMappingClass::Direct, true, overcomplete));
    EXPECT_FALSE(CollectPhysicalBackingSpans(
        GuestBase, PageSize, PhysicalBackingMappingClass::Direct, true, unaligned_physical));
    EXPECT_FALSE(CollectPhysicalBackingSpans(
        GuestBase, 2 * PageSize, PhysicalBackingMappingClass::Direct, true, physical_overflow));
    EXPECT_FALSE(CollectPhysicalBackingSpans(std::numeric_limits<VAddr>::max() - (PageSize - 1),
                                             2 * PageSize, PhysicalBackingMappingClass::Direct,
                                             true, two_pages));
}

TEST(PhysicalBackingAllocationProvenance, ExhaustionDisablesProvenanceWithoutChangingOwnership) {
    PhysicalBackingGenerationSource generations{std::numeric_limits<u64>::max() - 1};
    auto last = MakeArea(0, PageSize, 0, PhysicalMemoryType::Free);
    auto exhausted = MakeArea(PageSize, PageSize, 0, PhysicalMemoryType::Free);

    AcquirePhysicalBacking(last, generations);
    AcquirePhysicalBacking(exhausted, generations);

    EXPECT_EQ(last.allocation_generation, std::numeric_limits<u64>::max());
    EXPECT_EQ(exhausted.allocation_generation, 0u);
    EXPECT_EQ(exhausted.dma_type, PhysicalMemoryType::Free);
    EXPECT_EQ(generations.Acquire(), 0u);
}

TEST(PhysicalBackingAllocationProvenance, GenerationSourceIsOneBoundedScalar) {
    EXPECT_EQ(sizeof(PhysicalBackingGenerationSource), sizeof(u64));
}
