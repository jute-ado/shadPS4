// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <gtest/gtest.h>
#include "core/memory.h"
#include "core/memory_retirement_plan.h"

namespace {
using namespace Core;

struct MappingFixture {
    std::map<VAddr, VirtualMemoryArea> vmas;
    MemoryRangeIdentity virtual_identity;
    MemoryRangeIdentity physical_identity;

    void Map(VAddr address, u64 size, VMAType type, MemoryProt prot, PAddr physical = 0) {
        VirtualMemoryArea area{address, size};
        area.type = type;
        area.prot = prot;
        if (area.HasPhysicalBacking()) {
            area.phys_areas.emplace(0, PhysicalMemoryArea{physical, size});
            EXPECT_TRUE(physical_identity.Stamp(physical, size));
        }
        if (area.IsMapped()) {
            EXPECT_TRUE(virtual_identity.Stamp(address, size));
        }
        vmas.insert_or_assign(address, std::move(area));
    }

    auto Capture(u64 address, u64 size, bool physical = false) {
        return CaptureMemoryRetirementPlan(vmas, virtual_identity, physical_identity, address, size,
                                           physical);
    }
};

// Historical policy characterizations only. Production uses the independent owner-lifetime
// gate; these tests do not assert a PS4 requirement to drain outstanding submissions.
TEST(HistoricalMemoryRetirementPlan, ExistingCpuOnlyBackingAlsoRequiresRetirement) {
    MappingFixture f;
    f.Map(0x10000, 0x4000, VMAType::Flexible, MemoryProt::CpuReadWrite);
    const auto plan = f.Capture(0x10000, 0x4000);
    ASSERT_TRUE(plan);
    EXPECT_TRUE(plan->requires_retirement);
    EXPECT_EQ(plan->physical_ranges.size(), 1u);
}

TEST(HistoricalMemoryRetirementPlan, FreshAndReservedGpuFlaggedRangesDoNotWait) {
    for (auto type : {VMAType::Free, VMAType::Reserved, VMAType::PoolReserved}) {
        MappingFixture f;
        f.Map(0x10000, 0x4000, type, MemoryProt::GpuReadWrite);
        const auto plan = f.Capture(0x10000, 0x4000);
        ASSERT_TRUE(plan);
        EXPECT_FALSE(plan->requires_retirement);
        EXPECT_TRUE(plan->physical_ranges.empty());
    }
}

TEST(HistoricalMemoryRetirementPlan, SparsePlaceholderWithoutHostBackingDoesNotInventRetirement) {
    MappingFixture f;
    const auto plan = f.Capture(0x10000, 0x4000);
    ASSERT_TRUE(plan);
    EXPECT_FALSE(plan->requires_retirement);
    ASSERT_EQ(plan->virtual_ranges.size(), 1u);
    EXPECT_EQ(plan->virtual_ranges.front().parts.front().identity, 0u);
}

TEST(HistoricalMemoryRetirementPlan, AnyMappedPortionOfMixedRangeRequiresRetirement) {
    MappingFixture f;
    f.Map(0x10000, 0x4000, VMAType::Reserved, MemoryProt::NoAccess);
    f.Map(0x14000, 0x4000, VMAType::Direct, MemoryProt::CpuReadWrite, 0x4000);
    const auto plan = f.Capture(0x10000, 0x8000);
    ASSERT_TRUE(plan);
    EXPECT_TRUE(plan->requires_retirement);
    ASSERT_EQ(plan->physical_ranges.size(), 1u);
    EXPECT_EQ(plan->physical_ranges.front().address, 0x4000u);
}

TEST(HistoricalMemoryRetirementPlan, PhysicalFreeCapturesEveryAliasAndOnlyClippedBytes) {
    MappingFixture f;
    f.Map(0x10000, 0x8000, VMAType::Direct, MemoryProt::CpuReadWrite, 0x4000);
    f.Map(0x30000, 0x8000, VMAType::Direct, MemoryProt::GpuRead, 0x4000);
    const auto plan = f.Capture(0x6000, 0x2000, true);
    ASSERT_TRUE(plan);
    EXPECT_TRUE(plan->requires_retirement);
    ASSERT_EQ(plan->virtual_ranges.size(), 2u);
    EXPECT_EQ(plan->virtual_ranges[0].address, 0x12000u);
    EXPECT_EQ(plan->virtual_ranges[1].address, 0x32000u);
    EXPECT_EQ(plan->virtual_ranges[0].size, 0x2000u);
    EXPECT_EQ(plan->physical_ranges.front().address, 0x6000u);
}

TEST(HistoricalMemoryRetirementPlan, NewAliasInvalidatesPhysicalRetirementWithoutGlobalRevision) {
    MappingFixture f;
    f.Map(0x10000, 0x4000, VMAType::Direct, MemoryProt::GpuRead, 0x4000);
    const auto plan = f.Capture(0x4000, 0x4000, true);
    ASSERT_TRUE(plan);
    f.Map(0x30000, 0x4000, VMAType::Direct, MemoryProt::GpuRead, 0x4000);
    EXPECT_FALSE(f.physical_identity.Matches(plan->physical_ranges.front()));
    EXPECT_TRUE(f.virtual_identity.Matches(plan->virtual_ranges.front()));
}

TEST(HistoricalMemoryRetirementPlan, VirtualReplacementCapturesOnlyOverlappingPhysicalFragments) {
    MappingFixture f;
    f.Map(0x10000, 0x4000, VMAType::Pooled, MemoryProt::GpuRead, 0x8000);
    const auto plan = f.Capture(0x11000, 0x1000);
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan->physical_ranges.size(), 1u);
    EXPECT_EQ(plan->physical_ranges.front().address, 0x9000u);
    EXPECT_EQ(plan->physical_ranges.front().size, 0x1000u);
}

TEST(HistoricalMemoryRetirementPlan, PhysicalAllocationWithoutAliasesNeedsOnlyLifetimeLease) {
    MappingFixture f;
    ASSERT_TRUE(f.physical_identity.Stamp(0x4000, 0x4000));
    const auto plan = f.Capture(0x4000, 0x4000, true);
    ASSERT_TRUE(plan);
    EXPECT_FALSE(plan->requires_retirement);
    EXPECT_TRUE(plan->virtual_ranges.empty());
    ASSERT_EQ(plan->physical_ranges.size(), 1u);
    EXPECT_NE(plan->physical_ranges.front().parts.front().identity, 0u);
}

TEST(HistoricalMemoryRetirementPlan, OutOfGpuAddressSpaceBackingDoesNotBlockFreshGuestProgress) {
    MappingFixture f;
    f.Map(0x20000000000ULL, 0x4000, VMAType::System, MemoryProt::CpuReadWrite);
    const auto plan = f.Capture(0x20000000000ULL, 0x4000);
    ASSERT_TRUE(plan);
    EXPECT_FALSE(plan->requires_retirement);
}

TEST(HistoricalMemoryRetirementPlan, OverflowAndInvalidPhysicalMetadataFailClosed) {
    MappingFixture f;
    EXPECT_FALSE(f.Capture(std::numeric_limits<u64>::max() - 3, 8));
    f.Map(0x10000, 0x4000, VMAType::Direct, MemoryProt::GpuRead);
    f.vmas.begin()->second.phys_areas.begin()->second.size = 0x8000;
    EXPECT_FALSE(f.Capture(0x10000, 0x4000));
    EXPECT_FALSE(f.Capture(0, 0x4000, true));
}

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input.good());
    std::string result{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    std::erase(result, '\r');
    return result;
}

std::string Function(const std::string& source, const std::string& signature) {
    const auto begin = source.find(signature);
    EXPECT_NE(begin, std::string::npos);
    if (begin == std::string::npos)
        return {};
    const auto end = source.find("\n}\n", begin);
    EXPECT_NE(end, std::string::npos);
    return source.substr(begin, end - begin);
}

TEST(GpuMappingAdmission, EveryDestructiveEntryAcquiresBeforeGpuUnmapAndWriterLock) {
    const auto source = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    for (const char* name : {"Free", "MapMemory", "MapFile", "PoolDecommit", "UnmapMemory"}) {
        const auto body = Function(source, "s32 MemoryManager::" + std::string{name} + "(");
        const auto acquire = body.find("AcquireGpuOwnerLease(");
        ASSERT_NE(acquire, std::string::npos) << name;
        const auto unmap = body.find("UnmapGpuMappings(");
        ASSERT_NE(unmap, std::string::npos) << name;
        EXPECT_LT(acquire, unmap) << name;
        EXPECT_NE(body.find("ORBIS_KERNEL_ERROR_ECANCELED"), std::string::npos) << name;
    }
    const auto helper = Function(source, "MemoryManager::AcquireGpuOwnerLease(");
    EXPECT_NE(helper.find("gpu_owner_lifetime->Acquire()"), std::string::npos);
    EXPECT_EQ(helper.find("CaptureMemoryRetirementPlan("), std::string::npos);
    EXPECT_EQ(helper.find("AcquireOptimisticMemoryRetirement("), std::string::npos);
    EXPECT_EQ(helper.find("unlock"), std::string::npos);
}

TEST(GpuMappingAdmission, EarlyGpuUnmapIsClippedToExistingLogicalMappings) {
    const auto source = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    const auto helper = Function(source, "void MemoryManager::UnmapGpuMappings(");
    ASSERT_NE(helper.find("rasterizer->UnmapMappedMemory("), std::string::npos);
    EXPECT_NE(helper.find("IsValidGpuMapping(virtual_addr, size)"), std::string::npos);

    for (const char* name : {"Free", "MapMemory", "MapFile", "PoolDecommit", "UnmapMemory"}) {
        const auto body = Function(source, "s32 MemoryManager::" + std::string{name} + "(");
        EXPECT_NE(body.find("UnmapGpuMappings("), std::string::npos) << name;
        EXPECT_EQ(body.find("if (IsGpuMapped("), std::string::npos) << name;
    }
}

TEST(GpuMappingAdmission, DestructivePathsUnmapEveryMappedSubrange) {
    const auto memory = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    const auto rasterizer = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto page_manager = ReadSource(SHADPS4_PAGE_MANAGER_SOURCE_PATH);

    const auto helper = Function(memory, "void MemoryManager::UnmapGpuMappings(");
    ASSERT_NE(helper.find("rasterizer->UnmapMappedMemory("), std::string::npos);
    EXPECT_NE(helper.find("IsValidGpuMapping("), std::string::npos);

    for (const char* name : {"Free", "MapMemory", "MapFile", "PoolDecommit", "UnmapMemory"}) {
        const auto body = Function(memory, "s32 MemoryManager::" + std::string{name} + "(");
        EXPECT_NE(body.find("UnmapGpuMappings("), std::string::npos) << name;
        EXPECT_EQ(body.find("if (IsGpuMapped("), std::string::npos) << name;
    }

    EXPECT_NE(page_manager.find("logical_mappings.ForEachMappedSubrange("), std::string::npos);
    EXPECT_NE(rasterizer.find("page_manager.GetGpuMappedSubranges("), std::string::npos);
    EXPECT_NE(rasterizer.find("void Rasterizer::UnmapMappedMemory("), std::string::npos);
}

TEST(GpuMappingAdmission, KernelDoesNotWaitBeforeFreshMappingCanMakeGuestProgress) {
    const auto source = ReadSource(SHADPS4_KERNEL_MEMORY_SOURCE_PATH);
    EXPECT_EQ(source.find("AcquireGpuSubmissionBoundary"), std::string::npos);
    const auto body = Function(source, "s32 PS4_SYSV_ABI sceKernelMunmap(");
    EXPECT_NE(body.find("memory->UnmapMemory(virtual_addr, len)"), std::string::npos);
}

TEST(GpuMappingAdmission, ActualOwnerLifetimeClosesBeforeStopWithoutRetirementCallback) {
    const auto gnm = ReadSource(SHADPS4_GNM_SOURCE_PATH);
    EXPECT_EQ(gnm.find("AcquireSubmissionRetirement("), std::string::npos);
    EXPECT_EQ(gnm.find("[guard = std::move(*admission)] {}"), std::string::npos);
    EXPECT_NE(gnm.find("Core::Memory::Instance()->SetGpuOwnerLifetime"), std::string::npos);
    const auto liverpool = ReadSource(SHADPS4_LIVERPOOL_SOURCE_PATH);
    const auto capture = Function(liverpool, "Liverpool::CaptureRetirementTicket()");
    EXPECT_NE(capture.find("submission_completions.CaptureCompletionTicket()"), std::string::npos);
    EXPECT_NE(capture.find("submit_cv.notify_one()"), std::string::npos);
    EXPECT_EQ(capture.find("DrainReadyBoundaries()"), std::string::npos);
    const auto destroy = Function(liverpool, "Liverpool::~Liverpool()");
    EXPECT_LT(destroy.find("CloseAndWait()"), destroy.find("process_thread.request_stop()"));
}
} // namespace
