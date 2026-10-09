// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/deferred_ready_queue.h"
#include "video_core/buffer_cache/fault_range.h"
#include "video_core/buffer_cache/fault_record_snapshot.h"

namespace {

struct FakeBuffer {
    u32 value;
};

std::string ReadSource(const char* path) {
    std::ifstream input{path};
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

std::string_view FunctionSource(const std::string& source, std::string_view name) {
    const auto begin = source.find(name);
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = source.find("\n}", begin + name.size());
    return std::string_view{source}.substr(begin, end - begin + 2);
}

TEST(DeferredFaultResidency, CompletionCannotStrandResumedUploadInSupersededBuffer) {
    auto current = std::make_shared<FakeBuffer>(FakeBuffer{11});
    auto active_upload = current;
    VideoCore::DeferredReadyQueue<u64> queue{
        [&](std::span<const u64>) { current = std::make_shared<FakeBuffer>(*current); }};
    const std::array<u64, 1> records{0x4000};
    queue.Enqueue(records); // Completion during the active upload's stream-buffer wait.
    active_upload->value = 22;
    queue.Drain(); // A later outer draw/dispatch boundary, after the upload finished.
    EXPECT_EQ(current->value, 22U);
}

TEST(DeferredFaultResidency, SrtFinishCollectsRecordsWithoutMutatingResidency) {
    bool inside_snapshot = true;
    u32 inside_calls = 0;
    u32 calls = 0;
    VideoCore::DeferredReadyQueue<u64> queue{[&](std::span<const u64> records) {
        inside_calls += inside_snapshot;
        calls += records.size();
    }};
    const std::array<u64, 1> records{0x4000};
    queue.Enqueue(records);
    inside_snapshot = false;
    EXPECT_EQ(inside_calls, 0U);
    EXPECT_EQ(calls, 0U);
    queue.Drain();
    EXPECT_EQ(calls, 1U);
}

TEST(DeferredFaultResidency, CompletedSlotCanBeReusedBeforeTheBoundary) {
    std::array<u64, 2> slot_records{0x4000, 0x8000};
    std::vector<u64> observed;
    VideoCore::DeferredReadyQueue<u64> queue{[&](std::span<const u64> records) {
        observed.insert(observed.end(), records.begin(), records.end());
    }};
    queue.Enqueue(slot_records);
    slot_records.fill(0xDEAD0000);
    EXPECT_TRUE(observed.empty());
    queue.Drain();
    EXPECT_EQ(observed, (std::vector<u64>{0x4000, 0x8000}));
}

TEST(DeferredFaultResidency, EnqueueDuringDrainSettlesAtOuterBoundaryAndCannotRecurse) {
    std::vector<u64> observed;
    bool nested_drain_result = true;
    VideoCore::DeferredReadyQueue<u64>* queue_pointer{};
    VideoCore::DeferredReadyQueue<u64> queue{[&](std::span<const u64> records) {
        observed.insert(observed.end(), records.begin(), records.end());
        if (records.front() == 1) {
            const std::array<u64, 1> next{2};
            queue_pointer->Enqueue(next);
            nested_drain_result = queue_pointer->Drain();
        }
    }};
    queue_pointer = &queue;
    const std::array<u64, 1> first{1};
    queue.Enqueue(first);
    EXPECT_TRUE(queue.Drain());
    EXPECT_FALSE(nested_drain_result);
    EXPECT_EQ(observed, (std::vector<u64>{1, 2}));
    EXPECT_TRUE(queue.Drain());
    EXPECT_EQ(observed, (std::vector<u64>{1, 2}));
}

TEST(DeferredFaultResidency, NestedCompletionChainSettlesAtSameOuterBoundary) {
    std::vector<u64> observed;
    bool nested_drain_result = true;
    VideoCore::DeferredReadyQueue<u64>* queue_pointer{};
    VideoCore::DeferredReadyQueue<u64> queue{[&](std::span<const u64> records) {
        observed.insert(observed.end(), records.begin(), records.end());
        if (records.front() < 3) {
            const std::array<u64, 1> next{records.front() + 1};
            queue_pointer->Enqueue(next);
            nested_drain_result = queue_pointer->Drain();
        }
    }};
    queue_pointer = &queue;
    const std::array<u64, 1> first{1};
    queue.Enqueue(first);

    EXPECT_TRUE(queue.Drain());
    EXPECT_FALSE(nested_drain_result);
    EXPECT_EQ(observed, (std::vector<u64>{1, 2, 3}));
}

TEST(DeferredFaultResidency, EmptyDrainDoesNotCallConsumerAndBatchesPreserveOrder) {
    u32 calls{};
    std::vector<u64> observed;
    VideoCore::DeferredReadyQueue<u64> queue{[&](std::span<const u64> records) {
        ++calls;
        observed.insert(observed.end(), records.begin(), records.end());
    }};
    queue.Enqueue({});
    EXPECT_TRUE(queue.Drain());
    EXPECT_EQ(calls, 0U);
    const std::array<u64, 2> first{1, 2};
    const std::array<u64, 1> second{3};
    queue.Enqueue(first);
    queue.Enqueue(second);
    EXPECT_TRUE(queue.Drain());
    EXPECT_EQ(calls, 1U);
    EXPECT_EQ(observed, (std::vector<u64>{1, 2, 3}));
}

TEST(DeferredFaultResidency, ExceptionalConsumerDoesNotLeaveQueuePermanentlyDraining) {
    bool should_throw = true;
    u32 calls{};
    VideoCore::DeferredReadyQueue<u64> queue{[&](std::span<const u64>) {
        if (should_throw) {
            throw std::runtime_error{"synthetic consumer failure"};
        }
        ++calls;
    }};
    const std::array<u64, 1> record{1};
    queue.Enqueue(record);
    EXPECT_THROW(queue.Drain(), std::runtime_error);
    should_throw = false;
    queue.Enqueue(record);
    EXPECT_TRUE(queue.Drain());
    EXPECT_EQ(calls, 1U);
}

TEST(DeferredFaultResidency, MappingIsRevalidatedWhenResidencyActuallyRuns) {
    bool mapped = true;
    u32 resident_calls{};
    VideoCore::DeferredReadyQueue<u64> queue{[&](std::span<const u64> records) {
        for (const u64 address : records) {
            if (VideoCore::IsProcessableDmaFaultRange(address, address + 0x1000, 0x100000,
                                                      [&](u64, u64) { return mapped; })) {
                ++resident_calls;
            }
        }
    }};
    const std::array<u64, 1> record{0x4000};
    queue.Enqueue(record);
    mapped = false;
    queue.Drain();
    EXPECT_EQ(resident_calls, 0U);
    mapped = true;
    queue.Enqueue(record);
    queue.Drain();
    EXPECT_EQ(resident_calls, 1U);
}

TEST(GpuFaultRecordPlan, ReservesFirstQwordForCounter) {
    const auto plan = VideoCore::PlanGpuFaultRecords(8192, 1024);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->read_count, 1023U);
    EXPECT_EQ(plan->reported_count, 1024U);
    EXPECT_TRUE(plan->truncated);
}

TEST(GpuFaultRecordPlan, CorruptCounterCannotReadIntoFollowingSlots) {
    const auto plan = VideoCore::PlanGpuFaultRecords(8192, 0xffffffffU);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->read_count, 1023U);
    EXPECT_TRUE(plan->truncated);
}

TEST(GpuFaultRecordPlan, ExistingEmptyAndFullValidSlotsAreUnchanged) {
    EXPECT_EQ(VideoCore::PlanGpuFaultRecords(8192, 0)->read_count, 0U);
    EXPECT_EQ(VideoCore::PlanGpuFaultRecords(8192, 1023)->read_count, 1023U);
    EXPECT_FALSE(VideoCore::PlanGpuFaultRecords(8192, 1023)->truncated);
    EXPECT_FALSE(VideoCore::PlanGpuFaultRecords(0, 0));
    EXPECT_FALSE(VideoCore::PlanGpuFaultRecords(16384, 1024));
}

void SetSlotHeaderAndRecord(std::span<u8> slot, u64 header, u64 first) {
    std::memcpy(slot.data(), &header, sizeof(header));
    std::memcpy(slot.data() + sizeof(header), &first, sizeof(first));
}

TEST(GpuFaultRecordSnapshot, InvalidatesBeforeReadingCounterAndPayload) {
    std::array<u8, VideoCore::GpuFaultSlotBytes> slot{};
    SetSlotHeaderAndRecord(slot, 0, 11);
    u32 invalidate_calls{};
    const auto snapshot = VideoCore::CaptureGpuFaultRecords(slot, [&] {
        ++invalidate_calls;
        SetSlotHeaderAndRecord(slot, 1, 22);
        return true;
    });
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(invalidate_calls, 1U);
    ASSERT_EQ(snapshot->Records().size(), 1U);
    EXPECT_EQ(snapshot->Records().front(), 22U);
}

TEST(GpuFaultRecordSnapshot, FailedInvalidationNeverReturnsStaleRecords) {
    std::array<u8, VideoCore::GpuFaultSlotBytes> slot{};
    SetSlotHeaderAndRecord(slot, 1, 11);
    EXPECT_FALSE(VideoCore::CaptureGpuFaultRecords(slot, [] { return false; }));
}

TEST(GpuFaultRecordSnapshot, OwnsPayloadAndUsesOnlyLowCounterDword) {
    std::array<u8, VideoCore::GpuFaultSlotBytes> slot{};
    SetSlotHeaderAndRecord(slot, 0xDEAD000000000001ULL, 0x12345000);
    const auto snapshot = VideoCore::CaptureGpuFaultRecords(slot, [] { return true; });
    ASSERT_TRUE(snapshot);
    slot.fill(0);
    EXPECT_EQ(snapshot->plan.reported_count, 1U);
    ASSERT_EQ(snapshot->Records().size(), 1U);
    EXPECT_EQ(snapshot->Records()[0], 0x12345000U);
}

TEST(GpuFaultRecordSnapshot, CorruptCounterIsBoundedAtLastQwordOfOneSlot) {
    std::array<u64, VideoCore::GpuFaultSlotBytes / sizeof(u64)> slot{};
    slot[0] = 0xffffffffU;
    slot.back() = 0xFFFF0000;
    const auto bytes = std::span<const u8>{reinterpret_cast<const u8*>(slot.data()), sizeof(slot)};
    const auto snapshot = VideoCore::CaptureGpuFaultRecords(bytes, [] { return true; });
    ASSERT_TRUE(snapshot);
    EXPECT_TRUE(snapshot->plan.truncated);
    ASSERT_EQ(snapshot->Records().size(), VideoCore::MaxGpuFaultRecords);
    EXPECT_EQ(snapshot->Records().back(), slot.back());
}

TEST(GpuFaultRecordSnapshot, UnalignedInputIsCopiedWithoutTypedPointerLoads) {
    std::array<u8, VideoCore::GpuFaultSlotBytes + 1> backing{};
    const auto slot = std::span<u8>{backing}.subspan(1);
    SetSlotHeaderAndRecord(slot, 1, 0xAABBCCDDEEFF0011);
    const auto snapshot = VideoCore::CaptureGpuFaultRecords(slot, [] { return true; });
    ASSERT_TRUE(snapshot);
    ASSERT_EQ(snapshot->Records().size(), 1U);
    EXPECT_EQ(snapshot->Records()[0], 0xAABBCCDDEEFF0011);
}

TEST(GpuFaultRecordSnapshot, RejectsWrongSlotSizeBeforeInvalidation) {
    std::array<u8, VideoCore::GpuFaultSlotBytes * 2> slots{};
    u32 invalidate_calls{};
    const auto invalidate = [&] {
        ++invalidate_calls;
        return true;
    };
    EXPECT_FALSE(VideoCore::CaptureGpuFaultRecords({}, invalidate));
    EXPECT_FALSE(VideoCore::CaptureGpuFaultRecords(slots, invalidate));
    EXPECT_FALSE(VideoCore::CaptureGpuFaultRecords(std::span{slots}.first(8191), invalidate));
    EXPECT_EQ(invalidate_calls, 0U);
    const auto empty = VideoCore::CaptureGpuFaultRecords(std::span{slots}.first(8192), invalidate);
    ASSERT_TRUE(empty);
    EXPECT_TRUE(empty->Records().empty());
    EXPECT_EQ(invalidate_calls, 1U);
}

TEST(FaultResidencyWiring, CompletedReadbackInvalidatesTheExactSlotBeforeEnqueue) {
    const auto source = ReadSource(SHADPS4_FAULT_RESIDENCY_SOURCE_PATH);
    const auto collect = FunctionSource(source, "void FaultManager::CollectCompletedFaults(");
    ASSERT_FALSE(collect.empty());
    const auto invalidate =
        collect.find("download_buffer.InvalidateMappedRange(offset, PageFaultAreaSize)");
    const auto enqueue = collect.find("completed_faults.Enqueue(");
    ASSERT_NE(invalidate, std::string_view::npos);
    ASSERT_NE(enqueue, std::string_view::npos);
    EXPECT_LT(invalidate, enqueue);
    EXPECT_NE(collect.find("CaptureGpuFaultRecords("), std::string_view::npos);
    EXPECT_NE(collect.find("fault_areas[area] = 0", enqueue), std::string_view::npos);
    EXPECT_EQ(collect.find("MakeDmaFaultRangeResident"), std::string_view::npos);
    EXPECT_EQ(collect.find("DrainReadyFaults"), std::string_view::npos);
}

TEST(FaultResidencyWiring, SlotReuseCollectsOwnedRecordsWithoutGenericCallbacks) {
    const auto source = ReadSource(SHADPS4_FAULT_RESIDENCY_SOURCE_PATH);
    const auto process = FunctionSource(source, "void FaultManager::ProcessFaultBuffer()");
    ASSERT_FALSE(process.empty());
    EXPECT_NE(process.find("CollectCompletedFaults(current_area)"), std::string_view::npos);
    EXPECT_EQ(process.find("scheduler.DeferOperation("), std::string_view::npos);
    EXPECT_EQ(process.find("scheduler.PopPendingOperations()"), std::string_view::npos);
    EXPECT_EQ(process.find("MakeDmaFaultRangeResident"), std::string_view::npos);
    EXPECT_EQ(process.find("completed_faults.Drain"), std::string_view::npos);
}

TEST(FaultResidencyWiring, DrainPollsOnlyFaultSlotsAfterEachResidencyBatch) {
    const auto source = ReadSource(SHADPS4_FAULT_RESIDENCY_SOURCE_PATH);
    const auto constructor_begin = source.find("FaultManager::FaultManager(");
    const auto constructor_end = source.find("void FaultManager::ProcessFaultBuffer()",
                                             constructor_begin);
    ASSERT_NE(constructor_begin, std::string::npos);
    ASSERT_NE(constructor_end, std::string::npos);
    const std::string_view constructor{source.data() + constructor_begin,
                                       constructor_end - constructor_begin};
    const auto resident = constructor.find("MakeCompletedFaultsResident(records)");
    ASSERT_NE(resident, std::string_view::npos);
    EXPECT_EQ(constructor.find("scheduler.PopPendingOperations()", resident),
              std::string_view::npos);

    const auto drain = FunctionSource(source, "void FaultManager::DrainReadyFaults()");
    ASSERT_FALSE(drain.empty());
    const auto ready = drain.find("scheduler.IsFree(fault_areas[area])");
    const auto collect = drain.find("CollectCompletedFaults(area)", ready);
    const auto consume = drain.find("completed_faults.Drain()", collect);
    const auto repoll = drain.find("while (collect_ready())", consume);
    ASSERT_NE(ready, std::string_view::npos);
    ASSERT_NE(collect, std::string_view::npos);
    ASSERT_NE(consume, std::string_view::npos);
    ASSERT_NE(repoll, std::string_view::npos);
    EXPECT_LT(ready, collect);
    EXPECT_LT(collect, consume);
    EXPECT_LT(consume, repoll);
    EXPECT_EQ(drain.find("scheduler.PopPendingOperations()"), std::string_view::npos);
}

TEST(FaultResidencyWiring, FaultSlotResetAndReadbackHaveExplicitGpuVisibility) {
    const auto source = ReadSource(SHADPS4_FAULT_RESIDENCY_SOURCE_PATH);
    const auto process = FunctionSource(source, "void FaultManager::ProcessFaultBuffer()");
    ASSERT_FALSE(process.empty());
    EXPECT_EQ(process.find("memset("), std::string_view::npos);
    const auto fill =
        process.find("cmdbuf.fillBuffer(download_buffer.Handle(), offset, PageFaultAreaSize, 0)");
    const auto reset_visible = process.find(".pBufferMemoryBarriers = &reset_barrier");
    const auto dispatch = process.find("cmdbuf.dispatch(");
    const auto host_visible = process.find(".pBufferMemoryBarriers = &readback_barrier");
    const auto completion = process.find("fault_areas[current_area++] = scheduler.CurrentTick()");
    ASSERT_NE(fill, std::string_view::npos);
    ASSERT_NE(reset_visible, std::string_view::npos);
    ASSERT_NE(dispatch, std::string_view::npos);
    ASSERT_NE(host_visible, std::string_view::npos);
    ASSERT_NE(completion, std::string_view::npos);
    EXPECT_LT(fill, reset_visible);
    EXPECT_LT(reset_visible, dispatch);
    EXPECT_LT(dispatch, host_visible);
    EXPECT_LT(host_visible, completion);
    EXPECT_EQ(process.find("scheduler.DeferOperation("), std::string_view::npos);
    const auto reset_start = process.find("const vk::BufferMemoryBarrier2 reset_barrier");
    const auto reset = process.substr(reset_start, process.find("};", reset_start) - reset_start);
    EXPECT_NE(reset.find(".srcStageMask = vk::PipelineStageFlagBits2::eTransfer"),
              std::string_view::npos);
    EXPECT_NE(reset.find(".srcAccessMask = vk::AccessFlagBits2::eTransferWrite"),
              std::string_view::npos);
    EXPECT_NE(reset.find(".dstStageMask = vk::PipelineStageFlagBits2::eComputeShader"),
              std::string_view::npos);
    EXPECT_NE(reset.find("vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite"),
              std::string_view::npos);
    EXPECT_NE(reset.find(".offset = offset"), std::string_view::npos);
    EXPECT_NE(reset.find(".size = PageFaultAreaSize"), std::string_view::npos);
    const auto read_start = process.find("const vk::BufferMemoryBarrier2 readback_barrier");
    const auto read = process.substr(read_start, process.find("};", read_start) - read_start);
    EXPECT_NE(read.find(".srcStageMask = vk::PipelineStageFlagBits2::eComputeShader"),
              std::string_view::npos);
    EXPECT_NE(read.find(".srcAccessMask = vk::AccessFlagBits2::eShaderWrite"),
              std::string_view::npos);
    EXPECT_NE(read.find(".dstStageMask = vk::PipelineStageFlagBits2::eHost"),
              std::string_view::npos);
    EXPECT_NE(read.find(".dstAccessMask = vk::AccessFlagBits2::eHostRead"), std::string_view::npos);
    EXPECT_NE(read.find(".offset = offset"), std::string_view::npos);
    EXPECT_NE(read.find(".size = PageFaultAreaSize"), std::string_view::npos);
}

TEST(FaultResidencyWiring, OuterCommandBoundariesDrainBeforeResourcePreparation) {
    const auto source = ReadSource(SHADPS4_FAULT_RESIDENCY_RASTERIZER_PATH);
    for (const auto function :
         {"void Rasterizer::Draw(", "void Rasterizer::DrawIndirect(",
          "void Rasterizer::DispatchDirect(", "void Rasterizer::DispatchIndirect("}) {
        SCOPED_TRACE(function);
        const auto body = FunctionSource(source, function);
        ASSERT_FALSE(body.empty());
        const auto collect = body.find("scheduler.PopPendingOperations()");
        const auto drain = body.find("buffer_cache.DrainReadyFaults()");
        const auto prepare = body.find("pipeline_cache.");
        ASSERT_NE(collect, std::string_view::npos);
        ASSERT_NE(drain, std::string_view::npos);
        ASSERT_NE(prepare, std::string_view::npos);
        EXPECT_LT(collect, drain);
        EXPECT_LT(drain, prepare);
    }
    const auto submit = FunctionSource(source, "void Rasterizer::OnSubmit()");
    const auto process_faults = submit.find("buffer_cache.ProcessFaultBuffer()");
    const auto drain_faults = submit.find("buffer_cache.DrainReadyFaults()");
    ASSERT_NE(process_faults, std::string_view::npos);
    ASSERT_NE(drain_faults, std::string_view::npos);
    EXPECT_LT(process_faults, drain_faults);
    const auto cache = ReadSource(SHADPS4_FAULT_RESIDENCY_CACHE_PATH);
    EXPECT_NE(FunctionSource(cache, "void BufferCache::DrainReadyFaults()")
                  .find("fault_manager.DrainReadyFaults()"),
              std::string_view::npos);
    for (const auto function :
         {"bool BufferCache::ReadMemorySnapshot(",
          "bool BufferCache::DownloadBufferMemorySnapshot(", "bool BufferCache::SynchronizeBuffer(",
          "vk::Buffer BufferCache::UploadCopies("}) {
        SCOPED_TRACE(function);
        const auto body = FunctionSource(cache, function);
        ASSERT_FALSE(body.empty());
        EXPECT_EQ(body.find("DrainReadyFaults"), std::string_view::npos);
    }
}

} // namespace
