// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "shader_recompiler/addr64_pointer_residency.h"
#include "video_core/buffer_cache/buffer_residency.h"
#include "video_core/buffer_cache/fault_range.h"
#include "video_core/buffer_cache/readback_prefetch.h"

namespace {

std::string ReadTextFile(const char* path) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input);
    return {std::istreambuf_iterator<char>{input}, {}};
}

std::string_view FunctionBody(const std::string& source, std::string_view signature,
                              std::string_view next_signature) {
    const auto begin = source.find(signature);
    EXPECT_NE(begin, std::string::npos);
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = source.find(next_signature, begin + signature.size());
    EXPECT_NE(end, std::string::npos);
    return std::string_view{source}.substr(begin, end - begin);
}

enum class ResidencyCall {
    Synchronize,
    Publish,
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

    std::vector<RecordedResidencyCall> calls;
};

} // namespace

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

TEST(BufferResidency, Addr64PointerResidencyNeverCreatesACrossPageOverlapMerge) {
    RecordedBufferCache cache;
    constexpr u64 cache_page_size = 16_KB;
    constexpr VAddr address = 0x100663FFF0ULL;

    VideoCore::MakeDmaFaultRangeResident(cache, address, Shader::Addr64PointerTableBytes,
                                         cache_page_size);

    ASSERT_EQ(cache.calls.size(), 3U);
    EXPECT_EQ(cache.calls[0].operation, ResidencyCall::Publish);
    EXPECT_EQ(cache.calls[0].address, address);
    EXPECT_EQ(cache.calls[0].size, 0x10U);
    EXPECT_EQ(cache.calls[1].operation, ResidencyCall::Publish);
    EXPECT_EQ(cache.calls[1].address, address + 0x10U);
    EXPECT_EQ(cache.calls[1].size, Shader::Addr64PointerTableBytes - 0x10U);
    EXPECT_EQ(cache.calls[2].operation, ResidencyCall::Synchronize);
    EXPECT_EQ(cache.calls[2].address, address);
    EXPECT_EQ(cache.calls[2].size, Shader::Addr64PointerTableBytes);
}

TEST(RelaxedReadbackPrefetch, PlansOnlyARecentFaultHotGpuWrittenWindow) {
    VideoCore::RelaxedReadbackPrefetchTracker tracker;
    constexpr VAddr hot_address = 0x1023456788ULL;

    EXPECT_EQ(tracker.PlanSubmit().count, 0U);
    tracker.ObserveGpuWrite(hot_address, 8);
    EXPECT_EQ(tracker.PlanSubmit().count, 0U);

    tracker.ObserveSynchronousFault(hot_address, 8);
    EXPECT_EQ(tracker.PlanSubmit().count, 0U);
    tracker.ObserveGpuWrite(hot_address + 64, 8);

    const auto batch = tracker.PlanSubmit();
    ASSERT_EQ(batch.count, 1U);
    EXPECT_EQ(batch.plans[0].address,
              hot_address & ~(VideoCore::ReadbackPrefetchWindowSize - 1ULL));
    EXPECT_EQ(batch.plans[0].size, VideoCore::ReadbackPrefetchWindowSize);
    EXPECT_EQ(batch.plans[0].generation, 1U);
    EXPECT_NE(batch.plans[0].token, 0U);
}

TEST(RelaxedReadbackPrefetch, PreservesTheExactClampedSynchronousWindow) {
    VideoCore::RelaxedReadbackPrefetchTracker tracker;
    constexpr VAddr aligned_window = 0x1030000000ULL;
    constexpr VAddr clamped_start = aligned_window + 0x4000;
    constexpr u64 clamped_size = 0x28000;

    tracker.ObserveSynchronousWindow(clamped_start, clamped_size);
    tracker.ObserveGpuWrite(clamped_start + 0x100, 8);
    const auto batch = tracker.PlanSubmit();
    ASSERT_EQ(batch.count, 1U);
    EXPECT_EQ(batch.plans[0].address, clamped_start);
    EXPECT_EQ(batch.plans[0].size, clamped_size);

    ASSERT_TRUE(tracker.CommitSubmitted(batch.plans[0], 44));
    EXPECT_EQ(tracker.ClassifyFault(clamped_start + clamped_size - 8, 8).disposition,
              VideoCore::ReadbackPrefetchFaultDisposition::PendingExact);
    EXPECT_EQ(tracker.ClassifyFault(clamped_start - 8, 8).disposition,
              VideoCore::ReadbackPrefetchFaultDisposition::None);
    EXPECT_EQ(tracker.ClassifyFault(clamped_start + clamped_size, 8).disposition,
              VideoCore::ReadbackPrefetchFaultDisposition::None);
}

TEST(RelaxedReadbackPrefetch, FaultConsumesOnlyTheExactSubmittedGeneration) {
    VideoCore::RelaxedReadbackPrefetchTracker tracker;
    constexpr VAddr address = 0x1040000120ULL;
    tracker.ObserveSynchronousFault(address, 8);
    tracker.ObserveGpuWrite(address, 8);
    const auto plan = tracker.PlanSubmit().plans[0];

    EXPECT_TRUE(tracker.CommitSubmitted(plan, 91));
    const auto pending = tracker.ClassifyFault(address, 8);
    EXPECT_EQ(pending.disposition,
              VideoCore::ReadbackPrefetchFaultDisposition::PendingExact);
    EXPECT_EQ(pending.token, plan.token);
    EXPECT_EQ(pending.submission_tick, 91U);

    EXPECT_EQ(tracker.Complete(plan.token, plan.generation),
              VideoCore::ReadbackPrefetchCompletionDisposition::Publish);
    EXPECT_EQ(tracker.ClassifyFault(address, 8).disposition,
              VideoCore::ReadbackPrefetchFaultDisposition::None);
    EXPECT_EQ(tracker.PlanSubmit().count, 0U);
}

TEST(RelaxedReadbackPrefetch, LaterGpuWriteMakesPendingGenerationStale) {
    VideoCore::RelaxedReadbackPrefetchTracker tracker;
    constexpr VAddr address = 0x1048000200ULL;
    tracker.ObserveSynchronousFault(address, 8);
    tracker.ObserveGpuWrite(address, 8);
    const auto plan = tracker.PlanSubmit().plans[0];
    ASSERT_TRUE(tracker.CommitSubmitted(plan, 12));

    tracker.ObserveGpuWrite(address + 128, 8);
    EXPECT_EQ(tracker.ClassifyFault(address, 8).disposition,
              VideoCore::ReadbackPrefetchFaultDisposition::Stale);
    EXPECT_EQ(tracker.Complete(plan.token, plan.generation),
              VideoCore::ReadbackPrefetchCompletionDisposition::Stale);

    const auto replacement = tracker.PlanSubmit();
    ASSERT_EQ(replacement.count, 1U);
    EXPECT_EQ(replacement.plans[0].generation, 2U);
    EXPECT_NE(replacement.plans[0].token, plan.token);
}

TEST(RelaxedReadbackPrefetch, FailedCpuPublicationKeepsTheGenerationDirty) {
    VideoCore::RelaxedReadbackPrefetchTracker tracker;
    constexpr VAddr address = 0x104C000200ULL;
    tracker.ObserveSynchronousFault(address, 8);
    tracker.ObserveGpuWrite(address, 8);
    const auto plan = tracker.PlanSubmit().plans[0];
    ASSERT_TRUE(tracker.CommitSubmitted(plan, 22));
    ASSERT_TRUE(tracker.CanPublish(plan.token, plan.generation));

    EXPECT_EQ(tracker.Complete(plan.token, plan.generation, false),
              VideoCore::ReadbackPrefetchCompletionDisposition::Stale);
    EXPECT_EQ(tracker.ClassifyFault(address, 8).disposition,
              VideoCore::ReadbackPrefetchFaultDisposition::Stale);
    EXPECT_EQ(tracker.PlanSubmit().count, 1U);
}

TEST(RelaxedReadbackPrefetch, PlanningIsBoundedAndEvictsOnlyNonPendingLruWindows) {
    static_assert(VideoCore::MaxReadbackPrefetchHotWindows == 64);
    static_assert(VideoCore::MaxReadbackPrefetchPlansPerSubmit == 16);
    VideoCore::RelaxedReadbackPrefetchTracker tracker;
    constexpr VAddr base = 0x1080000000ULL;
    for (u32 index = 0; index < VideoCore::MaxReadbackPrefetchHotWindows; ++index) {
        const VAddr address = base + index * VideoCore::ReadbackPrefetchWindowSize;
        tracker.ObserveSynchronousFault(address, 8);
        tracker.ObserveGpuWrite(address, 8);
    }

    const auto first = tracker.PlanSubmit();
    ASSERT_EQ(first.count, VideoCore::MaxReadbackPrefetchPlansPerSubmit);
    for (u32 index = 0; index < first.count; ++index) {
        EXPECT_TRUE(tracker.CommitSubmitted(first.plans[index], 100 + index));
    }

    const VAddr newcomer = base + VideoCore::MaxReadbackPrefetchHotWindows *
                                      VideoCore::ReadbackPrefetchWindowSize;
    tracker.ObserveSynchronousFault(newcomer, 8);
    tracker.ObserveGpuWrite(newcomer, 8);
    EXPECT_EQ(tracker.HotWindowCount(), VideoCore::MaxReadbackPrefetchHotWindows);
    for (u32 index = 0; index < first.count; ++index) {
        EXPECT_EQ(tracker.ClassifyFault(first.plans[index].address, 8).disposition,
                  VideoCore::ReadbackPrefetchFaultDisposition::PendingExact);
    }
    EXPECT_NE(tracker.ClassifyFault(newcomer, 8).disposition,
              VideoCore::ReadbackPrefetchFaultDisposition::None);
}

TEST(RelaxedReadbackPrefetch, RejectsMalformedAndCrossWindowObservations) {
    VideoCore::RelaxedReadbackPrefetchTracker tracker;
    constexpr VAddr address = 0x10A0000000ULL;
    tracker.ObserveSynchronousFault(0, 8);
    tracker.ObserveSynchronousFault(address, 0);
    tracker.ObserveSynchronousFault(address + VideoCore::ReadbackPrefetchWindowSize - 4, 8);
    EXPECT_EQ(tracker.HotWindowCount(), 0U);

    tracker.ObserveSynchronousFault(address + 32, 8);
    tracker.ObserveGpuWrite(address - 16, 8);
    EXPECT_EQ(tracker.PlanSubmit().count, 0U);
    EXPECT_FALSE(tracker.CommitSubmitted({}, 1));
    EXPECT_EQ(tracker.Complete(0, 0),
              VideoCore::ReadbackPrefetchCompletionDisposition::Unknown);
}

TEST(RelaxedReadbackPrefetch, NoRemainingGpuBytesCompletesTheExactPlanWithoutReplanning) {
    VideoCore::RelaxedReadbackPrefetchTracker tracker;
    constexpr VAddr address = 0x10A8000400ULL;
    tracker.ObserveSynchronousFault(address, 8);
    tracker.ObserveGpuWrite(address, 8);
    const auto plan = tracker.PlanSubmit().plans[0];

    EXPECT_TRUE(tracker.CompleteNoWork(plan));
    EXPECT_EQ(tracker.PlanSubmit().count, 0U);
    EXPECT_EQ(tracker.ClassifyFault(address, 8).disposition,
              VideoCore::ReadbackPrefetchFaultDisposition::None);

    tracker.ObserveGpuWrite(address + 64, 8);
    const auto next = tracker.PlanSubmit();
    ASSERT_EQ(next.count, 1U);
    EXPECT_EQ(next.plans[0].generation, plan.generation + 1);
}

TEST(RelaxedReadbackPrefetch, ResetDropsPlannedAndPendingAuthorityWithoutPublishing) {
    VideoCore::RelaxedReadbackPrefetchTracker tracker;
    constexpr VAddr planned_address = 0x10B0000100ULL;
    constexpr VAddr pending_address = planned_address + VideoCore::ReadbackPrefetchWindowSize;
    tracker.ObserveSynchronousFault(planned_address, 8);
    tracker.ObserveGpuWrite(planned_address, 8);
    tracker.ObserveSynchronousFault(pending_address, 8);
    tracker.ObserveGpuWrite(pending_address, 8);
    const auto batch = tracker.PlanSubmit();
    ASSERT_EQ(batch.count, 2U);
    ASSERT_TRUE(tracker.CommitSubmitted(batch.plans[1], 77));

    tracker.Reset();
    EXPECT_EQ(tracker.HotWindowCount(), 0U);
    EXPECT_EQ(tracker.PlanSubmit().count, 0U);
    EXPECT_EQ(tracker.ClassifyFault(pending_address, 8).disposition,
              VideoCore::ReadbackPrefetchFaultDisposition::None);
    EXPECT_EQ(tracker.Complete(batch.plans[1].token, batch.plans[1].generation),
              VideoCore::ReadbackPrefetchCompletionDisposition::Unknown);

    tracker.ObserveSynchronousFault(planned_address, 8);
    tracker.ObserveGpuWrite(planned_address, 8);
    const auto after_reset = tracker.PlanSubmit();
    ASSERT_EQ(after_reset.count, 1U);
    EXPECT_NE(after_reset.plans[0].token, batch.plans[0].token);
    EXPECT_NE(after_reset.plans[0].token, batch.plans[1].token);
}

TEST(RelaxedReadbackPrefetch, ResidencyIsSplitAtEveryBufferCachePageBoundary) {
    struct Slice {
        VAddr address;
        u64 size;
    };
    std::vector<Slice> slices;

    constexpr VAddr address = 0x1020003ff0;
    constexpr u64 size = 0x8020;
    ASSERT_TRUE(VideoCore::ForEachReadbackPrefetchCacheSlice(
        address, size, 16_KB,
        [&](VAddr slice_address, u64 slice_size) { slices.push_back({slice_address, slice_size}); }));

    ASSERT_EQ(slices.size(), 4U);
    EXPECT_EQ(slices[0].address, address);
    EXPECT_EQ(slices[0].size, 0x10U);
    EXPECT_EQ(slices[1].address, 0x1020004000U);
    EXPECT_EQ(slices[1].size, 0x4000U);
    EXPECT_EQ(slices[2].address, 0x1020008000U);
    EXPECT_EQ(slices[2].size, 0x4000U);
    EXPECT_EQ(slices[3].address, 0x102000c000U);
    EXPECT_EQ(slices[3].size, 0x10U);
}

TEST(RelaxedReadbackPrefetch, ResidencySlicePlannerFailsClosedOnMalformedRanges) {
    u32 calls = 0;
    const auto record = [&](VAddr, u64) { ++calls; };

    EXPECT_FALSE(VideoCore::ForEachReadbackPrefetchCacheSlice(0x1000, 0, 16_KB, record));
    EXPECT_FALSE(VideoCore::ForEachReadbackPrefetchCacheSlice(0x1000, 32, 0, record));
    EXPECT_FALSE(VideoCore::ForEachReadbackPrefetchCacheSlice(0x1000, 32, 12_KB, record));
    EXPECT_FALSE(VideoCore::ForEachReadbackPrefetchCacheSlice(
        std::numeric_limits<VAddr>::max() - 3, 8, 16_KB, record));
    EXPECT_EQ(calls, 0U);
}

TEST(RelaxedReadbackPrefetch, UnsafeSpeculativePublicationIsQuarantinedFromProductionSubmit) {
    const std::string buffer_source = ReadTextFile(SHADPS4_BUFFER_CACHE_SOURCE_PATH);
    const std::string rasterizer_source = ReadTextFile(SHADPS4_RASTERIZER_SOURCE_PATH);
    const std::string liverpool_source = ReadTextFile(SHADPS4_LIVERPOOL_SOURCE_PATH);

    const auto prefetch = FunctionBody(buffer_source, "void BufferCache::PrefetchHotReadbacks()",
                                       "template <bool async>");
    EXPECT_NE(prefetch.find("GpuReadbacksMode::Relaxed"), std::string_view::npos);
    EXPECT_NE(prefetch.find("readback_prefetch_tracker.Reset()"), std::string_view::npos);
    const auto inactive_guard = prefetch.find("if (readback_prefetch_active)");
    const auto reset = prefetch.find("readback_prefetch_tracker.Reset()", inactive_guard);
    const auto deactivate = prefetch.find("readback_prefetch_active = false", reset);
    const auto activate = prefetch.find("readback_prefetch_active = true", deactivate);
    EXPECT_NE(inactive_guard, std::string_view::npos);
    EXPECT_LT(inactive_guard, reset);
    EXPECT_LT(reset, deactivate);
    EXPECT_LT(deactivate, activate);
    EXPECT_EQ(prefetch.find("scheduler.Finish("), std::string_view::npos);
    EXPECT_EQ(prefetch.find("scheduler.Wait("), std::string_view::npos);
    EXPECT_EQ(prefetch.find("SendCommand"), std::string_view::npos);
    EXPECT_EQ(prefetch.find("CUSA"), std::string_view::npos);
    EXPECT_EQ(prefetch.find("Uncharted"), std::string_view::npos);

    const auto submit_prefetch = FunctionBody(
        buffer_source, "BufferCache::ReadbackPrefetchSubmitDisposition BufferCache::SubmitReadbackPrefetch(",
        "void BufferCache::FinalizeReadbackPrefetch(");
    const auto source_barrier = submit_prefetch.find("buffer.GetBarrier(");
    const auto copy = submit_prefetch.find("copyBuffer(");
    EXPECT_NE(submit_prefetch.find("ForEachReadbackPrefetchCacheSlice("), std::string_view::npos);
    EXPECT_EQ(submit_prefetch.find("FindBuffer(plan.address"), std::string_view::npos);
    EXPECT_NE(source_barrier, std::string_view::npos);
    EXPECT_NE(submit_prefetch.find("vk::AccessFlagBits2::eTransferRead", source_barrier),
              std::string_view::npos);
    EXPECT_NE(submit_prefetch.find("vk::PipelineStageFlagBits2::eTransfer", source_barrier),
              std::string_view::npos);
    EXPECT_LT(source_barrier, copy);

    const auto read_memory = FunctionBody(buffer_source, "void BufferCache::ReadMemory(",
                                          "void BufferCache::MarkGpuModifiedRange(");
    const auto classify = read_memory.find("ClassifyFault");
    const auto wait = read_memory.find("scheduler.Wait", classify);
    const auto finalize = read_memory.find("FinalizeReadbackPrefetch", wait);
    const auto fallback = read_memory.find("DownloadBufferMemory<false>", finalize);
    EXPECT_NE(classify, std::string_view::npos);
    EXPECT_NE(wait, std::string_view::npos);
    EXPECT_NE(finalize, std::string_view::npos);
    EXPECT_NE(fallback, std::string_view::npos);

    EXPECT_EQ(std::count(buffer_source.begin(), buffer_source.end(), '\0'), 0);
    EXPECT_EQ(buffer_source.find("gpu_modified_ranges.Add", buffer_source.find(
                  "void BufferCache::MarkGpuModifiedRange")),
              buffer_source.rfind("gpu_modified_ranges.Add"));

    const auto on_submit = FunctionBody(rasterizer_source, "void Rasterizer::OnSubmit()",
                                        "bool Rasterizer::BindResources(");
    const auto prefetch_call = on_submit.find("buffer_cache.PrefetchHotReadbacks()");
    EXPECT_EQ(prefetch_call, std::string_view::npos);
    EXPECT_EQ(on_submit.find("scheduler.Finish("), std::string_view::npos);

    const auto submit_done = FunctionBody(liverpool_source, "void Liverpool::SubmitDone(",
                                          "} // namespace AmdGpu");
    EXPECT_LT(submit_done.find("rasterizer->OnSubmit()"),
              submit_done.find("rasterizer->Flush()"));
}
