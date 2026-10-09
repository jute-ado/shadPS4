// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <limits>
#include <map>
#include <vector>

#include "shader_recompiler/addr64_pointer_residency.h"
#include "shader_recompiler/direct_memory_access_policy.h"
#include "shader_recompiler/ir/passes/srt_pointer_use_policy.h"
#include "video_core/buffer_cache/buffer_residency.h"
#include "video_core/buffer_cache/fault_range.h"
#include "video_core/renderer_vulkan/vk_resource_retirement.h"

namespace {

enum class ResidencyCall {
    Synchronize,
    Publish,
    Download,
};

struct RecordedResidencyCall {
    ResidencyCall operation;
    VAddr address;
    u32 size;
};

class ExpandedDmaBuffer {
public:
    [[nodiscard]] VAddr CpuAddr() const {
        return 0x101E600000;
    }

    [[nodiscard]] size_t SizeBytes() const {
        return 0x200000;
    }
};

class RecordedBufferCache {
public:
    void FindBuffer(VAddr address, u32 size) {
        calls.push_back({ResidencyCall::Publish, address, size});
    }

    void SynchronizeBuffersInRange(VAddr address, u32 size) {
        calls.push_back({ResidencyCall::Synchronize, address, size});
    }

    void SynchronizeGpuModifiedMemoryForCpuRead(VAddr address, u32 size) {
        calls.push_back({ResidencyCall::Download, address, size});
    }

    std::vector<RecordedResidencyCall> calls;
};

class RecordedRetirementScheduler {
public:
    template <typename Callback>
    void DeferOperationAfterNextSubmit(Callback&& callback) {
        deferred_after_next_submit = true;
        std::forward<Callback>(callback)();
    }

    bool deferred_after_next_submit{};
};

class RecordedPageTableBuffer {
public:
    void Fill(u64 offset, u32 size, u32 value) {
        fill_offset = offset;
        fill_size = size;
        fill_value = value;
        fill_count++;
    }

    u64 fill_offset{};
    u32 fill_size{};
    u32 fill_value{0xffffffffU};
    u32 fill_count{};
};

} // namespace

TEST(DirectMemoryAccessPolicy, UsesDirectMemoryForEveryStageWhenEnabled) {
    EXPECT_TRUE(Shader::ShouldUseDirectMemoryAccess(true, true, 0x79704cbaULL));
    EXPECT_TRUE(Shader::ShouldUseDirectMemoryAccess(true, true, 0x729c541eULL));
    EXPECT_TRUE(Shader::ShouldUseDirectMemoryAccess(true, true, 0xbf4600a0ULL));
    EXPECT_TRUE(Shader::ShouldUseDirectMemoryAccess(true, false));
}

TEST(DirectMemoryAccessPolicy, HonorsDisabledSettingForEveryStage) {
    EXPECT_FALSE(Shader::ShouldUseDirectMemoryAccess(false, true));
    EXPECT_FALSE(Shader::ShouldUseDirectMemoryAccess(false, false));
}

TEST(BufferResidency, PublishesExpandedDmaMappingOnlyAfterFullSpanIsResident) {
    ExpandedDmaBuffer buffer;
    std::vector<RecordedResidencyCall> calls;

    VideoCore::PublishDmaBufferAfterSynchronization(
        buffer,
        [&](ExpandedDmaBuffer&, VAddr address, u32 size) {
            calls.push_back({ResidencyCall::Synchronize, address, size});
        },
        [&] { calls.push_back({ResidencyCall::Publish, 0, 0}); });

    ASSERT_EQ(calls.size(), 2);
    EXPECT_EQ(calls[0].operation, ResidencyCall::Synchronize);
    EXPECT_EQ(calls[0].address, buffer.CpuAddr());
    EXPECT_EQ(calls[0].size, buffer.SizeBytes());
    EXPECT_EQ(calls[1].operation, ResidencyCall::Publish);
}

TEST(BufferResidency, DoesNotTouchUnpublishedBufferAfterResidencyUpload) {
    u32 touch_count = 0;

    VideoCore::TouchBufferAfterUploadIfRegistered(false, [&] { ++touch_count; });
    EXPECT_EQ(touch_count, 0);

    VideoCore::TouchBufferAfterUploadIfRegistered(true, [&] { ++touch_count; });
    EXPECT_EQ(touch_count, 1);
}

TEST(BufferResidency, InitializesEveryGpuPageTableEntryAsUnmapped) {
    RecordedPageTableBuffer page_table;
    constexpr u32 page_table_size = 0x20000;

    VideoCore::InitializeDmaPageTable(page_table, page_table_size);

    EXPECT_EQ(page_table.fill_count, 1U);
    EXPECT_EQ(page_table.fill_offset, 0U);
    EXPECT_EQ(page_table.fill_size, page_table_size);
    EXPECT_EQ(page_table.fill_value, 0U);
}

TEST(BufferResidency, BindsGpuCacheBackedGuestBuffersWithoutCpuMappedBacking) {
    constexpr VAddr address = 0x282bd0a10ULL;
    constexpr u64 size = 191744;

    EXPECT_TRUE(VideoCore::IsGuestBufferDescriptorBindable(false, address, size, true, false));
    EXPECT_TRUE(VideoCore::IsGuestBufferDescriptorBindable(false, address, size, true, true));
    EXPECT_FALSE(VideoCore::IsGuestBufferDescriptorBindable(true, address, size, true, true));
    EXPECT_FALSE(VideoCore::IsGuestBufferDescriptorBindable(false, 0, size, true, true));
    EXPECT_FALSE(VideoCore::IsGuestBufferDescriptorBindable(false, address, 0, true, true));
    EXPECT_FALSE(VideoCore::IsGuestBufferDescriptorBindable(false, address, size, false, true));
}

TEST(BufferResidency, StreamsSmallCpuReadWhenOnlyANeighboringPageRangeIsGpuOwned) {
    constexpr u32 page_size = 4096;

    EXPECT_TRUE(VideoCore::CanUseCpuStreamCopy(false, 28, page_size, false));
    EXPECT_FALSE(VideoCore::CanUseCpuStreamCopy(false, 28, page_size, true));
    EXPECT_FALSE(VideoCore::CanUseCpuStreamCopy(true, 28, page_size, false));
    EXPECT_FALSE(VideoCore::CanUseCpuStreamCopy(false, page_size + 1, page_size, false));
}

TEST(BufferResidency, WidensDetectedCpuChangeToItsSmallDescriptorTransaction) {
    constexpr u32 page_size = 4096;

    EXPECT_TRUE(VideoCore::ShouldWidenCpuUploadToDescriptor(28, page_size, true));
    EXPECT_FALSE(VideoCore::ShouldWidenCpuUploadToDescriptor(28, page_size, false));
    EXPECT_FALSE(
        VideoCore::ShouldWidenCpuUploadToDescriptor(page_size + 1, page_size, true));
}

TEST(BufferResidency, DiscardsCpuSnapshotsBeforeCheckingBufferRegistration) {
    std::vector<int> operations;

    const bool registered = VideoCore::DiscardCpuSnapshotsBeforeBufferLookup(
        [&] { operations.push_back(1); },
        [&] {
            operations.push_back(2);
            return false;
        });

    EXPECT_FALSE(registered);
    EXPECT_EQ(operations, (std::vector<int>{1, 2}));
}

TEST(BufferResidency, DoesNotCaptureCpuSnapshotAcrossPartiallyMappedTrackingPage) {
    std::vector<int> operations;

    const bool captured = VideoCore::CaptureCpuSnapshotIfReadable(
        0x494310000ULL,
        [&](VAddr, size_t) {
            operations.push_back(1);
            return false;
        },
        [&](VAddr, std::span<const u8>, size_t, size_t) {
            operations.push_back(2);
            return true;
        },
        [&](VAddr, size_t) { operations.push_back(3); }, 0x3000, 8);

    EXPECT_FALSE(captured);
    EXPECT_EQ(operations, (std::vector<int>{1, 3}));
}

TEST(BufferResidency, RetiresTransientResourceOnlyAfterItsFinalCommandIsRecorded) {
    std::vector<int> operations;

    VideoCore::RecordThenRetireTransientResource(
        [&] { operations.push_back(1); }, [&] { operations.push_back(2); });

    EXPECT_EQ(operations, (std::vector<int>{1, 2}));
}

TEST(BufferResidency, RetiresPublishedDmaBufferAfterAnAdditionalSubmission) {
    RecordedRetirementScheduler scheduler;
    bool retired = false;

    VideoCore::RetirePublishedDmaBuffer(scheduler, [&] { retired = true; });

    EXPECT_TRUE(scheduler.deferred_after_next_submit);
    EXPECT_TRUE(retired);
}

TEST(BufferResidency, EstablishesDynamicVertexStateAfterPotentiallyFlushingDrawBuffers) {
    std::vector<int> operations;

    VideoCore::PrepareDrawBuffersThenBindDynamicVertexState(
        [&] { operations.push_back(1); }, [&] { operations.push_back(2); },
        [&] { operations.push_back(3); });

    EXPECT_EQ(operations, (std::vector<int>{1, 2, 3}));
}

TEST(BufferResidency, CacheResourceDeletionWaitsForItsLatestRecordedUse) {
    EXPECT_FALSE(Vulkan::ResourceUseHasRetired(42, 41));
    EXPECT_TRUE(Vulkan::ResourceUseHasRetired(42, 42));
}

TEST(BufferResidency, SelectsOnlyAdjacentFlattenedPointerWordsFromAddr64Dependencies) {
    constexpr std::array flattened_dependencies{116U, 29U, 28U, 29U, 74U};

    const auto roots = Shader::SelectAddr64PointerRoots(3, flattened_dependencies);

    ASSERT_FALSE(roots.overflow);
    ASSERT_EQ(roots.count, 1U);
    EXPECT_EQ(roots.values[0].buffer_resource_index, 3U);
    EXPECT_EQ(roots.values[0].pointer_lo_flat_index, 28U);
}

TEST(BufferResidency, Addr64PointerRootSelectionIsBoundedDeduplicatedAndFailClosed) {
    constexpr std::array isolated_dependencies{1U, 3U, 5U, 7U};
    const auto isolated = Shader::SelectAddr64PointerRoots(0, isolated_dependencies);
    EXPECT_FALSE(isolated.overflow);
    EXPECT_EQ(isolated.count, 0U);

    constexpr std::array duplicated_pair{20U, 21U, 20U, 21U};
    const auto duplicate = Shader::SelectAddr64PointerRoots(2, duplicated_pair);
    ASSERT_FALSE(duplicate.overflow);
    ASSERT_EQ(duplicate.count, 1U);
    EXPECT_EQ(duplicate.values[0].pointer_lo_flat_index, 20U);

    constexpr std::array too_many_pairs{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U,
                                        10U, 11U, 12U, 13U, 14U, 15U, 16U, 17U};
    const auto overflow = Shader::SelectAddr64PointerRoots(1, too_many_pairs);
    EXPECT_TRUE(overflow.overflow);
    EXPECT_EQ(overflow.count, 0U);
}

TEST(BufferResidency, Addr64PointerTraversalSetBoundsQueueGrowthAndDeduplicatesNodes) {
    Shader::Addr64PointerTraversalSet traversal;
    for (u32 node = 1; node <= Shader::MaxAddr64PointerTraversalNodes; ++node) {
        EXPECT_EQ(traversal.TryInsert(node), Shader::Addr64PointerTraversalInsert::Inserted);
    }
    EXPECT_EQ(traversal.TryInsert(1), Shader::Addr64PointerTraversalInsert::Duplicate);
    EXPECT_EQ(traversal.TryInsert(Shader::MaxAddr64PointerTraversalNodes + 1U),
              Shader::Addr64PointerTraversalInsert::Overflow);
    EXPECT_TRUE(traversal.overflow());
    EXPECT_EQ(traversal.size(), Shader::MaxAddr64PointerTraversalNodes);
}

TEST(BufferResidency, Addr64PointerResidencyPlansOneExactBoundedGuestTable) {
    constexpr u64 address_space_size = 1ULL << 40U;

    const auto exact = Shader::PlanAddr64PointerResidency(0x06622E30U, 0x10U,
                                                          address_space_size);
    EXPECT_TRUE(exact.valid);
    EXPECT_EQ(exact.address, 0x1006622E30ULL);
    EXPECT_EQ(exact.size, Shader::Addr64PointerTableBytes);

    EXPECT_FALSE(Shader::PlanAddr64PointerResidency(0U, 0U, address_space_size).valid);
    EXPECT_FALSE(Shader::PlanAddr64PointerResidency(
                     0xFFFFFF80U, 0xFFU, address_space_size)
                     .valid);
}

TEST(BufferResidency, ReadConstPointerResidencyCoversKnownOffsetsByCachePage) {
    constexpr u64 page_size = 0x4000;
    constexpr u64 address_space_size = 1ULL << 40;

    const auto exact = Shader::PlanReadConstPointerResidency(
        0x42710110U, 0x2U, 2U, 148U, false, page_size, address_space_size);
    ASSERT_TRUE(exact.valid);
    EXPECT_EQ(exact.address, 0x242710000ULL);
    EXPECT_EQ(exact.size, page_size);

    const auto dynamic = Shader::PlanReadConstPointerResidency(
        0x42710110U, 0x2U, 0U, 0U, true, page_size, address_space_size);
    ASSERT_TRUE(dynamic.valid);
    EXPECT_EQ(dynamic.address, 0x242710000ULL);
    EXPECT_EQ(dynamic.size, page_size);

    EXPECT_FALSE(Shader::PlanReadConstPointerResidency(
                     0U, 0U, 0U, 0U, true, page_size, address_space_size)
                     .valid);
}

TEST(BufferResidency, RefreshesNestedReadConstPointersToABoundedFixedPoint) {
    u32 pointer_generation = 0;
    std::vector<u32> prepared_generations;

    Shader::PrepareReadConstPointersThenRefresh(
        4, [&] { return pointer_generation; },
        [&] { prepared_generations.push_back(pointer_generation); },
        [&] {
            if (pointer_generation < 2) {
                ++pointer_generation;
            }
        });

    EXPECT_EQ(prepared_generations, (std::vector<u32>{0, 1, 2}));
}

TEST(BufferResidency, Addr64PointerResidencyUsesExistingFindThenSynchronizePathExactlyOnce) {
    RecordedBufferCache cache;
    constexpr VAddr address = 0x1006622E30ULL;

    VideoCore::MakeDmaFaultRangeResident(cache, address, Shader::Addr64PointerTableBytes);

    ASSERT_EQ(cache.calls.size(), 2U);
    EXPECT_EQ(cache.calls[0].operation, ResidencyCall::Publish);
    EXPECT_EQ(cache.calls[0].address, address);
    EXPECT_EQ(cache.calls[0].size, Shader::Addr64PointerTableBytes);
    EXPECT_EQ(cache.calls[1].operation, ResidencyCall::Synchronize);
    EXPECT_EQ(cache.calls[1].address, address);
    EXPECT_EQ(cache.calls[1].size, Shader::Addr64PointerTableBytes);
}

TEST(BufferResidency, CpuWalkedReadConstRangeIsResidentBeforeGpuWritesAreDownloaded) {
    RecordedBufferCache cache;
    constexpr VAddr address = 0x242710000ULL;
    constexpr u32 size = 0x4000;

    VideoCore::MakeDmaFaultRangeCpuVisible(cache, address, size);

    ASSERT_EQ(cache.calls.size(), 3U);
    EXPECT_EQ(cache.calls[0].operation, ResidencyCall::Publish);
    EXPECT_EQ(cache.calls[1].operation, ResidencyCall::Synchronize);
    EXPECT_EQ(cache.calls[2].operation, ResidencyCall::Download);
    EXPECT_EQ(cache.calls[2].address, address);
    EXPECT_EQ(cache.calls[2].size, size);
}

TEST(BufferResidency, KeepsEveryReadConstSharingAnSrtPointerOffset) {
    std::map<u32, std::vector<const void*>> uses;
    const int first{};
    const int second{};

    Shader::Optimization::AddSrtPointerUse(uses, 12, &first);
    Shader::Optimization::AddSrtPointerUse(uses, 12, &second);

    ASSERT_EQ(uses.size(), 1);
    ASSERT_EQ(uses.at(12).size(), 2);
    EXPECT_EQ(uses.at(12)[0], &first);
    EXPECT_EQ(uses.at(12)[1], &second);
}
