// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <gtest/gtest.h>
#include "core/libraries/gnmdriver/vmm_gpu_completion_diagnostic.h"
#include "core/memory.h"
#include "core/memory_retirement_plan.h"

namespace {
using namespace Core;

struct RecordingSubmissionGate {
    struct Guard {};

    Guard Enter() {
        ++enter_count;
        return {};
    }

    Guard WaitForBoundaryAndEnter() {
        ++boundary_wait_count;
        return {};
    }

    int enter_count{};
    int boundary_wait_count{};
};

struct RecordingGpu {
    explicit RecordingGpu(const RecordingSubmissionGate& gate_) : gate{gate_} {}

    void WaitGpuCompletion() {
        boundary_was_acquired = gate.boundary_wait_count != 0;
        ++completion_wait_count;
    }

    const RecordingSubmissionGate& gate;
    int completion_wait_count{};
    bool boundary_was_acquired{};
};

TEST(VmmGpuCompletionDiagnostic, ParserRequiresTheExactOneValue) {
    using Libraries::GnmDriver::IsVmmGpuCompletionWaitEnabled;

    EXPECT_TRUE(IsVmmGpuCompletionWaitEnabled("1"));
    for (const char* value : {static_cast<const char*>(nullptr), "", "0", "01", "true", " 1",
                              "1 "}) {
        EXPECT_FALSE(IsVmmGpuCompletionWaitEnabled(value));
    }
}

TEST(VmmGpuCompletionDiagnostic, DisabledPathUsesAdmissionOnly) {
    RecordingSubmissionGate gate;
    RecordingGpu gpu{gate};

    auto guard = Libraries::GnmDriver::AcquireVmmGpuAddressSpaceGuard(gate, &gpu, false);
    static_cast<void>(guard);

    EXPECT_EQ(gate.enter_count, 1);
    EXPECT_EQ(gate.boundary_wait_count, 0);
    EXPECT_EQ(gpu.completion_wait_count, 0);
}

TEST(VmmGpuCompletionDiagnostic, EnabledPathWaitsForBoundaryThenCompletesGpu) {
    RecordingSubmissionGate gate;
    RecordingGpu gpu{gate};

    auto guard = Libraries::GnmDriver::AcquireVmmGpuAddressSpaceGuard(gate, &gpu, true);
    static_cast<void>(guard);

    EXPECT_EQ(gate.enter_count, 0);
    EXPECT_EQ(gate.boundary_wait_count, 1);
    EXPECT_EQ(gpu.completion_wait_count, 1);
    EXPECT_TRUE(gpu.boundary_was_acquired);
}

TEST(VmmGpuCompletionDiagnostic, EnabledPathWithNoGpuFallsBackWithoutWaiting) {
    RecordingSubmissionGate gate;

    auto guard = Libraries::GnmDriver::AcquireVmmGpuAddressSpaceGuard(
        gate, static_cast<RecordingGpu*>(nullptr), true);
    static_cast<void>(guard);

    EXPECT_EQ(gate.enter_count, 1);
    EXPECT_EQ(gate.boundary_wait_count, 0);
}

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

TEST(VmmGpuCompletionDiagnosticWiring, LaunchTimeToggleIsolatedFromProductionGuardBody) {
    const auto gnm = ReadSource(SHADPS4_GNM_SOURCE_PATH);
    const auto helper = ReadSource(SHADPS4_VMM_GPU_COMPLETION_HEADER_PATH);
    const auto guard = Function(gnm, "Common::UniqueFunction<void> AcquireGpuAddressSpaceGuard(");

    EXPECT_NE(gnm.find("SHADPS4_VMM_WAIT_GPU_COMPLETION"), std::string::npos);
    EXPECT_NE(guard.find("static const bool"), std::string::npos);
    EXPECT_NE(guard.find("std::getenv(\"SHADPS4_VMM_WAIT_GPU_COMPLETION\")"),
              std::string::npos);
    EXPECT_NE(guard.find("submission_gate->Enter()"), std::string::npos);
    // Existing production source contracts intentionally keep the retired sequence out of this
    // body. The opt-in implementation lives in a helper that can be tested independently.
    EXPECT_EQ(guard.find("WaitForBoundaryAndEnter"), std::string::npos);
    EXPECT_EQ(guard.find("WaitGpuCompletion"), std::string::npos);

    EXPECT_NE(helper.find("SHADPS4_VMM_WAIT_GPU_COMPLETION"), std::string::npos);
    EXPECT_NE(helper.find("WaitForBoundaryAndEnter"), std::string::npos);
    EXPECT_NE(helper.find("WaitGpuCompletion"), std::string::npos);
    EXPECT_NE(helper.find("gpu != nullptr"), std::string::npos);
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

TEST(GpuMappingAdmission, EarlyGpuUnmapRequiresAnExistingLogicalMapping) {
    const auto source = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    const auto helper = Function(source, "bool MemoryManager::IsGpuMapped(");
    ASSERT_NE(helper.find("rasterizer->IsMapped(virtual_addr, size)"), std::string::npos);
    EXPECT_NE(helper.find("IsValidGpuMapping(virtual_addr, size)"), std::string::npos);

    const std::array<std::pair<const char*, const char*>, 5> expected{{
        {"Free", "HasGpuMappedSubrange(addr, unmap_size, VideoCore::GpuMappingKind::HostMapped)"},
        {"MapMemory", "HasGpuMappedSubrange(virtual_addr, size, VideoCore::GpuMappingKind::HostMapped)"},
        {"MapFile", "HasGpuMappedSubrange(virtual_addr, size, VideoCore::GpuMappingKind::HostMapped)"},
        {"PoolDecommit", "HasGpuMappedSubrange(virtual_addr, size, VideoCore::GpuMappingKind::HostMapped)"},
        {"UnmapMemory", "HasGpuMappedSubrange(virtual_addr, size, VideoCore::GpuMappingKind::HostMapped)"}}};
    for (const auto& [name, condition] : expected) {
        const auto body = Function(source, "s32 MemoryManager::" + std::string{name} + "(");
        ASSERT_NE(body.find(condition), std::string::npos) << name;
    }
}

TEST(GpuMappingAdmission, EarlyGpuUnmapIsClippedToExistingLogicalMappings) {
    const auto memory = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    const auto helper = Function(memory, "void MemoryManager::UnmapGpuMappings(");
    ASSERT_NE(helper.find("rasterizer->UnmapMappedMemory("), std::string::npos);
    EXPECT_NE(helper.find("GPU_ADDRESS_SPACE_LIMIT - virtual_addr"), std::string::npos);

    for (const char* name : {"Free", "MapMemory", "MapFile", "PoolDecommit", "UnmapMemory"}) {
        const auto body = Function(memory, "s32 MemoryManager::" + std::string{name} + "(");
        EXPECT_NE(body.find("HasGpuMappedSubrange("), std::string::npos) << name;
        EXPECT_NE(body.find("UnmapGpuMappings("), std::string::npos) << name;
    }

    const auto rasterizer = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto page_manager = ReadSource(SHADPS4_PAGE_MANAGER_SOURCE_PATH);
    EXPECT_NE(rasterizer.find("page_manager.GetGpuMappedSubranges("), std::string::npos);
    EXPECT_NE(rasterizer.find("page_manager.GetGpuMappedSubranges(addr, size, kind)"),
              std::string::npos);
    EXPECT_NE(rasterizer.find("void Rasterizer::UnmapMappedMemory("), std::string::npos);
    EXPECT_NE(page_manager.find("logical_mappings.ForEachMappedSubrange("), std::string::npos);
    EXPECT_NE(page_manager.find("ForEachHostMappedSubrange("), std::string::npos);
    EXPECT_NE(page_manager.find("ForEachSparsePlaceholderSubrange("), std::string::npos);
}

TEST(GpuMappingAdmission, PartialLogicalOverlapTriggersFragmentTeardown) {
    const auto rasterizer = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto helper = Function(rasterizer, "bool Rasterizer::HasGpuMappedSubrange(");
    ASSERT_NE(helper.find("page_manager.GetGpuMappedSubranges(addr, size, kind)"),
              std::string::npos);
    EXPECT_NE(helper.find("!page_manager.GetGpuMappedSubranges(addr, size, kind).empty()"),
              std::string::npos);
}

TEST(GpuMappingAdmission, MappingQueriesSerializeWithLifecycleTransitions) {
    const auto page_manager = ReadSource(SHADPS4_PAGE_MANAGER_SOURCE_PATH);
    for (const auto* signature : {"bool PageManager::IsGpuMapped(",
                                  "u64 PageManager::GetGpuMappedExtent(",
                                  "std::vector<std::pair<VAddr, u64>> PageManager::GetGpuMappedSubranges("}) {
        const auto body = Function(page_manager, signature);
        ASSERT_NE(body.find("mapping_lifecycle_mutex"), std::string::npos) << signature;
        EXPECT_NE(body.find("scoped_lock"), std::string::npos) << signature;
    }
}

TEST(GpuMappingAdmission, FragmentTeardownQueuesOneOrderedTextureRetirementBatch) {
    const auto rasterizer = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto helper = Function(rasterizer, "void Rasterizer::UnmapMappedMemory(");
    ASSERT_NE(helper.find("GetGpuMappedSubranges(addr, size, kind)"), std::string::npos);
    EXPECT_EQ(helper.find("UnmapMemory(mapped_addr, mapped_size, kind)"), std::string::npos);
    ASSERT_NE(helper.find("SendCommand<false>"), std::string::npos);
    EXPECT_NE(helper.find("texture_cache.UnmapMemory"), std::string::npos);
}

TEST(GpuMappingAdmission, SingleRangeTeardownCapturesImageGenerationBeforeAsyncUnmap) {
    const auto rasterizer = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto helper = Function(rasterizer, "void Rasterizer::UnmapMemory(");
    ASSERT_NE(helper.find("texture_cache.CaptureUnmapTargets(addr, size)"), std::string::npos);

    const auto capture = helper.find("texture_cache.CaptureUnmapTargets(addr, size)");
    const auto prepare = helper.find("page_manager.PrepareForGpuUnmap(addr, size)");
    const auto queue = helper.find("liverpool->SendCommand<false>");
    const auto callback = helper.find("texture_cache.UnmapMemory(unmap_targets)");
    ASSERT_NE(prepare, std::string::npos);
    ASSERT_NE(queue, std::string::npos);
    ASSERT_NE(callback, std::string::npos);
    EXPECT_LT(capture, prepare);
    EXPECT_LT(capture, queue);
    EXPECT_LT(queue, callback);
    EXPECT_NE(helper.find("unmap_targets = std::move(unmap_targets)"), std::string::npos);
    EXPECT_EQ(helper.find("texture_cache.UnmapMemory(addr, size)"), std::string::npos);
}

TEST(GpuMappingAdmission, TextureRetirementCapturesImageGenerationBeforeAsyncUnmap) {
    const auto texture_header = ReadSource(SHADPS4_TEXTURE_CACHE_HEADER_PATH);
    EXPECT_NE(texture_header.find("struct ImageUnmapTarget"), std::string::npos);
    EXPECT_NE(texture_header.find("CaptureUnmapTargets"), std::string::npos);
    EXPECT_NE(texture_header.find("std::span<const ImageUnmapTarget>"), std::string::npos);

    const auto texture_source = ReadSource(SHADPS4_TEXTURE_CACHE_SOURCE_PATH);
    const auto capture = Function(texture_source, "TextureCache::CaptureUnmapTargets(");
    ASSERT_NE(capture.find("std::scoped_lock"), std::string::npos);
    EXPECT_NE(capture.find("image.image_uid"), std::string::npos);

    const auto unmap =
        Function(texture_source, "void TextureCache::UnmapMemory(std::span<const ImageUnmapTarget>");
    ASSERT_NE(unmap.find("image_uid"), std::string::npos);
    EXPECT_NE(unmap.find("TryGetImage"), std::string::npos);

    const auto rasterizer = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto helper = Function(rasterizer, "void Rasterizer::UnmapMappedMemory(");
    const auto capture_request = helper.find("CaptureUnmapTargets");
    const auto queue = helper.find("SendCommand<false>");
    ASSERT_NE(capture_request, std::string::npos);
    ASSERT_NE(queue, std::string::npos);
    EXPECT_LT(capture_request, queue);
    EXPECT_NE(helper.find("unmap_targets"), std::string::npos);
    EXPECT_EQ(helper.find("texture_cache.UnmapMemory(mapped_addr, mapped_size)"),
              std::string::npos);
}

TEST(GpuMappingAdmission, EveryRegisteredImageGenerationGetsAUniqueRetirementUid) {
    const auto image = ReadSource(SHADPS4_IMAGE_SOURCE_PATH);
    const auto constructor = Function(image, "Image::Image(const Vulkan::Instance& instance_");
    const auto uid = constructor.find("image_uid = global_image_uid.Next()");
    const auto undefined = constructor.find("if (info.pixel_format == vk::Format::eUndefined)");
    ASSERT_NE(uid, std::string::npos);
    ASSERT_NE(undefined, std::string::npos);
    EXPECT_LT(uid, undefined);
}

TEST(GpuMappingAdmission, GpuRangeAllowsExactFortyBitExclusiveEnd) {
    constexpr VAddr limit = 1ULL << 40;
    EXPECT_TRUE(IsValidGpuAddressRange(limit - 0x4000, 0x4000));
    EXPECT_TRUE(IsValidGpuAddressRange(limit - 0x4000, 0));
    EXPECT_FALSE(IsValidGpuAddressRange(limit, 0));
    EXPECT_FALSE(IsValidGpuAddressRange(limit - 0x4000, 0x4001));
    EXPECT_FALSE(IsValidGpuAddressRange(std::numeric_limits<VAddr>::max(), 1));
}

TEST(GpuMappingAdmission, DestructiveQueriesClipToGpuApertureAndRetainOwnerKind) {
    const auto memory = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    const auto has = Function(memory, "bool MemoryManager::HasGpuMappedSubrange(");
    const auto unmap = Function(memory, "void MemoryManager::UnmapGpuMappings(");
    ASSERT_NE(has.find("VideoCore::GpuMappingKind kind"), std::string::npos);
    EXPECT_NE(has.find("GPU_ADDRESS_SPACE_LIMIT - virtual_addr"), std::string::npos);
    EXPECT_NE(has.find("std::min<u64>(size, GPU_ADDRESS_SPACE_LIMIT - virtual_addr)"),
              std::string::npos);
    EXPECT_NE(unmap.find("std::min<u64>(size, GPU_ADDRESS_SPACE_LIMIT - virtual_addr)"),
              std::string::npos);

    const auto set_prt = Function(memory, "void MemoryManager::SetPrtArea(");
    EXPECT_NE(set_prt.find(
                  "HasGpuMappedSubrange(area.start, area.end - area.start,\n                             VideoCore::GpuMappingKind::SparsePlaceholder)"),
              std::string::npos);
}

TEST(GpuMappingAdmission, MappingValidityUsesLockTakingAndLockedVariants) {
    const auto source = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    const auto wrapper = Function(source, "bool MemoryManager::IsValidMapping(");
    ASSERT_NE(wrapper.find("std::shared_lock lk{mutex}"), std::string::npos);
    EXPECT_NE(wrapper.find("return IsValidMappingLocked(virtual_addr, size)"),
              std::string::npos);

    const auto wrapper_end = source.find("\n}\n", source.find("bool MemoryManager::IsValidMapping("));
    ASSERT_NE(wrapper_end, std::string::npos);
    const auto after_wrapper = source.substr(wrapper_end + 3);
    // Internal callers must not re-enter the non-recursive VMA mutex through the public wrapper.
    EXPECT_EQ(after_wrapper.find("IsValidMapping("), std::string::npos);

    const auto locked = Function(source, "bool MemoryManager::IsValidMappingLocked(");
    EXPECT_EQ(locked.find("std::shared_lock"), std::string::npos);
    EXPECT_EQ(locked.find("std::unique_lock"), std::string::npos);

    const auto search = Function(source, "VAddr MemoryManager::SearchFree(");
    EXPECT_NE(search.find("std::shared_lock lk{mutex}"), std::string::npos);
    EXPECT_NE(search.find("IsValidMappingLocked(virtual_addr)"), std::string::npos);
}

TEST(GpuMappingAdmission, WatcherLedgerValidatesTrackedBoundsBeforeIndexing) {
    const auto page_manager = ReadSource(SHADPS4_PAGE_MANAGER_SOURCE_PATH);
    const auto watcher = page_manager.find("bool UpdatePageWatchers(VAddr addr, u64 size)");
    ASSERT_NE(watcher, std::string::npos);
    const auto guard = page_manager.find("if (!IsValidTrackedRange(addr, size))", watcher);
    const auto page_index = page_manager.find("size_t page = addr >> PM_PAGE_BITS", watcher);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(page_index, std::string::npos);
    EXPECT_LT(guard, page_index);

    const auto masked = page_manager.find("void UpdatePageWatchersForRegion(VAddr base_addr",
                                          watcher);
    ASSERT_NE(masked, std::string::npos);
    const auto region_guard =
        page_manager.find("if (!IsValidTrackedRange(base_addr, TRACKER_HIGHER_PAGE_SIZE))",
                          masked);
    const auto mask_read = page_manager.find("auto start_range = mask.FirstRange()", masked);
    ASSERT_NE(region_guard, std::string::npos);
    ASSERT_NE(mask_read, std::string::npos);
    EXPECT_LT(region_guard, mask_read);
}

TEST(GpuMappingAdmission, PendingGpuUnmapHidesRangeFromSnapshotMappingQueries) {
    const auto page_manager = ReadSource(SHADPS4_PAGE_MANAGER_SOURCE_PATH);

    // The VMM keeps the logical mapping alive while cache invalidation and texture retirement run.
    // Publish a pending-unmap marker first so an SRT session that starts in that interval cannot
    // treat the still-visible interval as a fresh mapping.
    const auto prepare = page_manager.find("void PrepareForGpuUnmap(VAddr address, size_t size)");
    const auto marker = page_manager.find("BeginPendingGpuUnmap(address, size)", prepare);
    const auto invalidate = page_manager.find("InvalidateSnapshotRange(address, size)", marker);
    ASSERT_NE(prepare, std::string::npos);
    ASSERT_NE(marker, std::string::npos);
    ASSERT_NE(invalidate, std::string::npos);
    EXPECT_LT(marker, invalidate);

    const auto query = page_manager.find("bool PageManager::IsGpuMapped(VAddr address, size_t size)");
    const auto pending_check = page_manager.find("HasPendingGpuUnmapLocked(address, size)", query);
    const auto logical_check = page_manager.find("logical_mappings.Contains(address, size)", query);
    ASSERT_NE(query, std::string::npos);
    ASSERT_NE(pending_check, std::string::npos);
    ASSERT_NE(logical_check, std::string::npos);
    EXPECT_LT(pending_check, logical_check);

    const auto retire = page_manager.find("void OnUnmap(VAddr address, size_t size");
    const auto logical_retire = page_manager.find("RetireHostMappedRange(address, size)", retire);
    const auto clear_marker = page_manager.find("pending_gpu_unmaps.erase(it)", logical_retire);
    ASSERT_NE(retire, std::string::npos);
    ASSERT_NE(logical_retire, std::string::npos);
    ASSERT_NE(clear_marker, std::string::npos);
    EXPECT_LT(logical_retire, clear_marker);
}

TEST(GpuMappingAdmission, SparsePrtRetirementSerializesOwnerSnapshotAndPublication) {
    const auto source = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    const auto body = Function(source, "void MemoryManager::SetPrtArea(");
    EXPECT_NE(body.find("AcquireGpuAddressSpaceGuard acquire_gpu_address_space_guard"),
              std::string::npos);
    const auto lock = body.find("std::unique_lock lk{unmap_mutex}");
    const auto lifetime = body.find("AcquireGpuOwnerLease()");
    const auto gpu_mapped = body.find("const bool gpu_mapped =");
    const auto mapping_query = body.find("HasGpuMappedSubrange(", gpu_mapped);
    const auto admission = body.find(
        "AcquireGpuMappingAdmission(gpu_mapped, acquire_gpu_address_space_guard)");
    const auto unmap = body.find("UnmapGpuMappings(");
    const auto publish = body.find("rasterizer->MapMemory(");
    ASSERT_NE(lock, std::string::npos);
    ASSERT_NE(lifetime, std::string::npos);
    ASSERT_NE(gpu_mapped, std::string::npos);
    ASSERT_NE(mapping_query, std::string::npos);
    ASSERT_NE(admission, std::string::npos);
    ASSERT_NE(unmap, std::string::npos);
    ASSERT_NE(publish, std::string::npos);
    EXPECT_LT(lock, lifetime);
    EXPECT_LT(gpu_mapped, mapping_query);
    EXPECT_LT(gpu_mapped, admission);
    EXPECT_LT(admission, unmap);
    EXPECT_LT(lifetime, unmap);
    EXPECT_LT(unmap, publish);
}

TEST(GpuMappingAdmission, KernelWiresScopedAdmissionIntoPrtRetirement) {
    const auto source = ReadSource(SHADPS4_KERNEL_MEMORY_SOURCE_PATH);
    const auto body = Function(source, "s32 PS4_SYSV_ABI sceKernelSetPrtAperture(");
    const auto set_prt = body.find("memory->SetPrtArea(");
    const auto provider = body.find("GnmDriver::AcquireGpuAddressSpaceGuard()", set_prt);
    const auto call_end = body.find(");", set_prt);
    ASSERT_NE(set_prt, std::string::npos);
    ASSERT_NE(provider, std::string::npos);
    ASSERT_NE(call_end, std::string::npos);
    EXPECT_LT(set_prt, provider);
    EXPECT_LT(provider, call_end);
}

TEST(GpuMappingAdmission, PrtApertureRangeCheckRejectsAddressOverflow) {
    const auto source = ReadSource(SHADPS4_KERNEL_MEMORY_SOURCE_PATH);
    const auto body = Function(source, "s32 PS4_SYSV_ABI sceKernelSetPrtAperture(");
    const auto range_check = body.find("PRT_AREA_START_ADDR");
    ASSERT_NE(range_check, std::string::npos);
    EXPECT_NE(body.find("PRT_AREA_END_ADDR", range_check), std::string::npos);
    EXPECT_NE(body.find("size > PRT_AREA_END_ADDR - address", range_check), std::string::npos);
    EXPECT_EQ(body.find("address + size >", range_check), std::string::npos);
}

TEST(GpuMappingAdmission, GetPrtApertureRejectsNullOutputPointersBeforeWriting) {
    const auto source = ReadSource(SHADPS4_KERNEL_MEMORY_SOURCE_PATH);
    const auto body = Function(source, "s32 PS4_SYSV_ABI sceKernelGetPrtAperture(");
    const auto null_guard = body.find("address == nullptr || size == nullptr");
    const auto output_write = body.find("std::tie(*address, *size)");
    ASSERT_NE(null_guard, std::string::npos);
    ASSERT_NE(output_write, std::string::npos);
    EXPECT_LT(null_guard, output_write);
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
        const auto gpu_unmap = body.find("UnmapGpuMappings(");
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

TEST(GpuMappingAdmission, KernelWiresScopedAdmissionIntoPoolReserveReplacement) {
    const auto source = ReadSource(SHADPS4_KERNEL_MEMORY_SOURCE_PATH);
    const auto body = Function(source, "s32 PS4_SYSV_ABI sceKernelMemoryPoolReserve(");
    const auto map = body.find("memory->MapMemory(");
    const auto provider = body.find("GnmDriver::AcquireGpuAddressSpaceGuard()", map);
    const auto call_end = body.find(");", map);
    ASSERT_NE(map, std::string::npos);
    ASSERT_NE(provider, std::string::npos);
    ASSERT_NE(call_end, std::string::npos);
    EXPECT_LT(map, provider);
    EXPECT_LT(provider, call_end);
}

TEST(GpuMappingAdmission, PoolCommitIsFreshPoolReservedPublication) {
    const auto source = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    const auto body = Function(source, "s32 MemoryManager::PoolCommit(");
    ASSERT_NE(body.find("vma.type != VMAType::PoolReserved"), std::string::npos);
    EXPECT_EQ(body.find("HasGpuMappedSubrange"), std::string::npos);
    EXPECT_EQ(body.find("UnmapGpuMappings"), std::string::npos);
    EXPECT_EQ(body.find("AcquireGpuMappingAdmission"), std::string::npos);
    const auto stamp = body.find("ASSERT(virtual_identity.Stamp(");
    const auto publish = body.find("rasterizer->MapMemory(");
    ASSERT_NE(stamp, std::string::npos);
    ASSERT_NE(publish, std::string::npos);
    EXPECT_LT(stamp, publish);
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
    const auto mapping_check = memory_body.find("HasGpuMappedSubrange(");
    const auto admission = memory_body.find("acquire_gpu_address_space_guard()");
    const auto gpu_unmap = memory_body.find("UnmapGpuMappings(");
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
