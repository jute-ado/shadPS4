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
        const auto unmap = body.find("rasterizer->UnmapMemory(");
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

TEST(GpuMappingAdmission, EarlyGpuUnmapRequiresAnExistingLogicalMapping) {
    const auto source = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    const auto helper = Function(source, "bool MemoryManager::IsGpuMapped(");
    ASSERT_NE(helper.find("rasterizer->IsMapped(virtual_addr, size)"), std::string::npos);
    EXPECT_NE(helper.find("IsValidGpuMapping(virtual_addr, size)"), std::string::npos);

    const std::array<std::pair<const char*, const char*>, 5> expected{{
        {"Free", "if (IsGpuMapped(addr, unmap_size))"},
        {"MapMemory", "if (IsGpuMapped(virtual_addr, size))"},
        {"MapFile", "if (IsGpuMapped(virtual_addr, size))"},
        {"PoolDecommit", "if (IsGpuMapped(virtual_addr, size))"},
        {"UnmapMemory", "const bool gpu_mapped = IsGpuMapped(virtual_addr, size)"}}};
    for (const auto& [name, condition] : expected) {
        const auto body = Function(source, "s32 MemoryManager::" + std::string{name} + "(");
        ASSERT_NE(body.find(condition), std::string::npos) << name;
    }
}

TEST(GpuMappingAdmission, BroadAddressChangesRetainScopedAdmissionAcrossGpuUnmap) {
    const auto source = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    const std::array<const char*, 4> operations{{"Free", "MapMemory", "MapFile",
                                                  "PoolDecommit"}};
    for (const char* name : operations) {
        const auto body = Function(source, "s32 MemoryManager::" + std::string{name} + "(");
        ASSERT_NE(body.find("AcquireGpuAddressSpaceGuard acquire_gpu_address_space_guard"),
                  std::string::npos)
            << name;
        const auto admission = body.find("AcquireGpuMappingAdmission(");
        const auto gpu_unmap = body.find("rasterizer->UnmapMemory(");
        ASSERT_NE(admission, std::string::npos) << name;
        ASSERT_NE(gpu_unmap, std::string::npos) << name;
        EXPECT_LT(admission, gpu_unmap) << name;
        EXPECT_NE(body.find("submission_admission"), std::string::npos) << name;
    }
}

TEST(GpuMappingAdmission, KernelWiresScopedAdmissionIntoAllBroadAddressChanges) {
    const auto source = ReadSource(SHADPS4_KERNEL_MEMORY_SOURCE_PATH);
    const std::array<std::pair<const char*, const char*>, 7> expected{{
        {"sceKernelCheckedReleaseDirectMemory(", "memory->Free(start, len, true"},
        {"sceKernelReleaseDirectMemory(", "memory->Free(start, len, false"},
        {"sceKernelMapNamedDirectMemory(", "memory->MapMemory(addr, in_addr"},
        {"sceKernelMapDirectMemory2(", "memory->MapMemory(addr, in_addr"},
        {"sceKernelMapNamedFlexibleMemory(", "memory->MapMemory("},
        {"sceKernelMapNamedSystemFlexibleMemory(", "memory->MapMemory("},
        {"sceKernelMemoryPoolDecommit(", "memory->PoolDecommit(pool_addr, len"},
    }};
    for (const auto& [signature, call] : expected) {
        const auto begin = source.find(signature);
        ASSERT_NE(begin, std::string::npos) << signature;
        const auto next = source.find("\ns32 PS4_SYSV_ABI ", begin + 1);
        const auto body = source.substr(begin, next == std::string::npos ? std::string::npos
                                                                          : next - begin);
        const auto call_pos = body.find(call);
        ASSERT_NE(call_pos, std::string::npos) << signature;
        EXPECT_NE(body.find("GnmDriver::AcquireGpuAddressSpaceGuard()", call_pos),
                  std::string::npos)
            << signature;
    }
}

TEST(GpuMappingAdmission, KernelDoesNotUseTheRetiredGlobalFenceForUnmap) {
    const auto source = ReadSource(SHADPS4_KERNEL_MEMORY_SOURCE_PATH);
    EXPECT_EQ(source.find("AcquireGpuSubmissionBoundary"), std::string::npos);
    const auto body = Function(source, "s32 PS4_SYSV_ABI sceKernelMunmap(");
    EXPECT_NE(body.find("memory->UnmapMemory("), std::string::npos);
    EXPECT_NE(body.find("GnmDriver::AcquireGpuAddressSpaceGuard()"), std::string::npos);
    EXPECT_EQ(body.find("WaitGpuCompletion"), std::string::npos);
}

TEST(GpuMappingAdmission, GpuVisibleUnmapOwnsOnlyAScopedSubmissionAdmission) {
    const auto kernel = ReadSource(SHADPS4_KERNEL_MEMORY_SOURCE_PATH);
    const auto kernel_body = Function(kernel, "s32 PS4_SYSV_ABI sceKernelMunmap(");
    const auto provider = kernel_body.find("GnmDriver::AcquireGpuAddressSpaceGuard()");
    const auto unmap = kernel_body.find("memory->UnmapMemory(");
    ASSERT_NE(provider, std::string::npos);
    ASSERT_NE(unmap, std::string::npos);
    EXPECT_LT(unmap, provider);
    EXPECT_EQ(kernel_body.find("AcquireGpuSubmissionBoundary"), std::string::npos);
    EXPECT_EQ(kernel_body.find("WaitGpuCompletion"), std::string::npos);

    const auto gnm = ReadSource(SHADPS4_GNM_SOURCE_PATH);
    const auto guard = Function(gnm, "Common::UniqueFunction<void> AcquireGpuAddressSpaceGuard(");
    ASSERT_NE(guard.find("submission_gate->Enter()"), std::string::npos);
    EXPECT_EQ(guard.find("WaitForBoundaryAndEnter"), std::string::npos);
    EXPECT_EQ(guard.find("WaitGpuCompletion"), std::string::npos);

    const auto memory = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    const auto memory_body = Function(memory, "s32 MemoryManager::UnmapMemory(");
    const auto mapping_check = memory_body.find("const bool gpu_mapped = IsGpuMapped(");
    const auto admission = memory_body.find("acquire_gpu_address_space_guard()");
    const auto gpu_unmap = memory_body.find("rasterizer->UnmapMemory(");
    const auto backing_unmap = memory_body.find("UnmapMemoryImpl(");
    ASSERT_NE(mapping_check, std::string::npos);
    ASSERT_NE(admission, std::string::npos);
    ASSERT_NE(gpu_unmap, std::string::npos);
    ASSERT_NE(backing_unmap, std::string::npos);
    EXPECT_NE(memory_body.find("if (gpu_mapped && acquire_gpu_address_space_guard)"),
              std::string::npos);
    EXPECT_LT(mapping_check, admission);
    EXPECT_LT(admission, gpu_unmap);
    EXPECT_LT(gpu_unmap, backing_unmap);
    EXPECT_EQ(memory_body.find("WaitForBoundaryAndEnter"), std::string::npos);
    EXPECT_EQ(memory_body.find("WaitGpuCompletion"), std::string::npos);
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
