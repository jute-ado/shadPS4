// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <latch>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <queue>

#include <gtest/gtest.h>

#include "common/types.h"
#include "core/libraries/videoout/eop_flip_presentation.h"
#include "video_core/amdgpu/eop_completion.h"
#include "video_core/amdgpu/eop_flip_tracker.h"
#include "video_core/amdgpu/eop_submission_batch.h"
#include "video_core/renderer_vulkan/gpu_completion_submission.h"

namespace {

struct TestEopPacket {
    u32 data;
    bool write = true;
    bool interrupt = true;

    void SignalFence(auto&& write_memory, auto&& signal_interrupt) const {
        if (write) {
            write_memory(data);
        }
        if (interrupt) {
            signal_interrupt();
        }
    }
};

struct TestGpuTimeline {
    struct Operation {
        u64 tick;
        Common::UniqueFunction<void> callback;
    };

    u64 next_tick = 41;
    u64 completed_tick = 40;
    u32 submissions = 0;
    std::queue<Operation> operations;

    void Submit(Common::UniqueFunction<void, u64>&& on_submitted,
                Common::UniqueFunction<void>&& wake) {
        const u64 tick = next_tick++;
        on_submitted(u64{tick});
        operations.emplace(tick, std::move(wake));
        ++submissions;
    }

    void Drain(AmdGpu::EopSubmissionBatch& batch) {
        batch.DrainCompleted([&](u64 tick) { return tick <= completed_tick; });
    }

    void Complete(u64 tick) {
        completed_tick = tick;
        while (!operations.empty() && operations.front().tick <= completed_tick) {
            auto callback = std::move(operations.front().callback);
            operations.pop();
            callback();
        }
    }
};

std::string ReadEopSource(const char* relative_path) {
    // Read-only negative controls may select an older source snapshot without
    // replacing production files in a running worktree.
    const char* override_root = std::getenv("SHADPS4_EOP_TEST_SOURCE_ROOT");
    const std::filesystem::path root = override_root ? override_root : SHADPS4_EOP_SOURCE_ROOT;
    std::ifstream input{root / relative_path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string_view SourceSection(std::string_view source, std::string_view begin,
                               std::string_view end) {
    const auto first = source.find(begin);
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = source.find(end, first + begin.size());
    return last == std::string_view::npos ? std::string_view{} : source.substr(first, last - first);
}

TEST(EventWriteEop, InterruptFlipWaitsForFollowingEopPublication) {
    std::vector<std::string> operations;
    AmdGpu::EopFlipTracker tracker;

    ASSERT_TRUE(tracker.QueueFlip(AmdGpu::DecodeFlipEopPosition(0x33),
                                  [&] { operations.emplace_back("flip"); }));
    EXPECT_TRUE(operations.empty());

    auto complete_eop = tracker.BeginEop();
    AmdGpu::PublishEop(
        TestEopPacket{.data = 1}, [&](u32) { operations.emplace_back("fence"); },
        [&] { operations.emplace_back("interrupt"); },
        [complete_eop = std::move(complete_eop)]() mutable { complete_eop(); });

    EXPECT_EQ(operations, (std::vector<std::string>{"fence", "interrupt", "flip"}));
}

TEST(EventWriteEop, PlainAndLabelFlipsFollowTheirPrecedingEopPublication) {
    for (const u32 nop_count : {0x34u, 0x39u}) {
        std::vector<std::string> operations;
        AmdGpu::EopFlipTracker tracker;

        auto complete_eop = tracker.BeginEop();
        AmdGpu::PublishEop(
            TestEopPacket{.data = 1}, [&](u32) { operations.emplace_back("fence"); },
            [&] { operations.emplace_back("interrupt"); },
            [complete_eop = std::move(complete_eop)]() mutable { complete_eop(); });

        ASSERT_TRUE(tracker.QueueFlip(AmdGpu::DecodeFlipEopPosition(nop_count),
                                      [&] { operations.emplace_back("flip"); }));
        EXPECT_EQ(operations, (std::vector<std::string>{"fence", "interrupt", "flip"}));
    }
}

TEST(EventWriteEop, RetainsEveryFlipAssociatedWithTheSameEop) {
    std::vector<int> flips;
    AmdGpu::EopFlipTracker tracker;

    ASSERT_TRUE(
        tracker.QueueFlip(AmdGpu::FlipEopPosition::Following, [&] { flips.emplace_back(1); }));
    ASSERT_TRUE(
        tracker.QueueFlip(AmdGpu::FlipEopPosition::Following, [&] { flips.emplace_back(2); }));
    auto complete_eop = tracker.BeginEop();

    EXPECT_TRUE(flips.empty());
    complete_eop();
    complete_eop();
    EXPECT_EQ(flips, (std::vector<int>{1, 2}));
}

TEST(EventWriteEop, EopFlipWaitsForTheNextVblankIteration) {
    using Libraries::VideoOut::IsFlipPresentationEligible;

    EXPECT_FALSE(IsFlipPresentationEligible(true, 41, 41));
    EXPECT_TRUE(IsFlipPresentationEligible(true, 41, 42));
    EXPECT_TRUE(IsFlipPresentationEligible(false, 41, 41));
}

TEST(EventWriteEop, DoesNotReleaseBusyFrameResourcesWhenPacketIsOnlyParsed) {
    // The guest treats this fence/IRQ as permission to reuse frame storage.
    // Parsing or submitting a packet is not completion of its preceding work.
    TestGpuTimeline gpu;
    AmdGpu::EopSubmissionBatch batch;
    const u64 frame_resource_last_use_tick = 41;
    u32 guest_fence = 0;
    bool guest_reused_busy_resource = false;
    std::vector<std::string> operations;

    batch.Add(
        TestEopPacket{.data = 42},
        [&](u32 value) {
            guest_fence = value;
            operations.emplace_back("fence");
        },
        [&] {
            EXPECT_EQ(guest_fence, 42u);
            guest_reused_busy_resource = gpu.completed_tick < frame_resource_last_use_tick;
            operations.emplace_back("interrupt");
        },
        [&] { operations.emplace_back("flip"); });

    EXPECT_EQ(guest_fence, 0u);
    EXPECT_FALSE(guest_reused_busy_resource);
    EXPECT_TRUE(operations.empty());

    batch.Flush([] {}, [&](auto&& on_submitted,
                           auto&& wake) { gpu.Submit(std::move(on_submitted), std::move(wake)); });
    EXPECT_EQ(gpu.next_tick, 42u);
    EXPECT_EQ(gpu.completed_tick, 40u);
    EXPECT_EQ(guest_fence, 0u);
    gpu.Complete(40);
    EXPECT_TRUE(operations.empty());
    gpu.Complete(41);
    gpu.Drain(batch);
    EXPECT_EQ(guest_fence, 42u);
    EXPECT_FALSE(guest_reused_busy_resource);
    EXPECT_EQ(operations, (std::vector<std::string>{"fence", "interrupt", "flip"}));
    gpu.Complete(42);
    EXPECT_EQ(operations.size(), 3u);
}

TEST(EventWriteEop, OwnsPacketAndMoveOnlyCallbacksUntilCompletion) {
    TestGpuTimeline gpu;
    AmdGpu::EopSubmissionBatch batch;
    TestEopPacket packet{.data = 42};
    u32 fence = 0;
    u32 notification = 0;
    batch.Add(
        packet, [&](u32 value) { fence = value; }, [] {},
        [owned = std::make_unique<u32>(7), &notification] { notification = *owned; });
    packet.data = 99;
    batch.Flush([] {}, [&](auto&& on_submitted,
                           auto&& wake) { gpu.Submit(std::move(on_submitted), std::move(wake)); });
    EXPECT_EQ(fence, 0u);
    gpu.Complete(41);
    gpu.Drain(batch);
    EXPECT_EQ(fence, 42u);
    EXPECT_EQ(notification, 7u);
}

TEST(EventWriteEop, AdjacentPacketsShareOneSubmissionAndPublishFifo) {
    TestGpuTimeline gpu;
    AmdGpu::EopSubmissionBatch batch;
    std::vector<u32> observations;
    u32 fence = 0;
    u32 preparations = 0;
    for (u32 data : {41u, 42u}) {
        batch.Add(
            TestEopPacket{.data = data}, [&](u32 value) { fence = value; },
            [&] { observations.emplace_back(fence); }, [] {});
    }
    ASSERT_TRUE(batch.HasPending());
    batch.Flush([&] { ++preparations; },
                [&](auto&& on_submitted, auto&& wake) {
                    gpu.Submit(std::move(on_submitted), std::move(wake));
                });
    EXPECT_FALSE(batch.HasPending());
    batch.Flush([&] { ++preparations; },
                [&](auto&& on_submitted, auto&& wake) {
                    gpu.Submit(std::move(on_submitted), std::move(wake));
                });
    EXPECT_EQ(preparations, 1u);
    EXPECT_EQ(gpu.submissions, 1u);
    EXPECT_TRUE(observations.empty());
    gpu.Complete(41);
    gpu.Drain(batch);
    EXPECT_EQ(observations, (std::vector<u32>{41, 42}));
}

TEST(EventWriteEop, PreservesWriteOnlyAndInterruptOnlyPacketSemantics) {
    TestGpuTimeline gpu;
    AmdGpu::EopSubmissionBatch batch;
    u32 writes = 0;
    u32 interrupts = 0;
    u32 completions = 0;
    batch.Add(
        TestEopPacket{.data = 41, .interrupt = false}, [&](u32) { ++writes; },
        [&] { ++interrupts; }, [&] { ++completions; });
    batch.Add(
        TestEopPacket{.data = 42, .write = false}, [&](u32) { ++writes; }, [&] { ++interrupts; },
        [&] { ++completions; });
    batch.Flush([] {}, [&](auto&& on_submitted,
                           auto&& wake) { gpu.Submit(std::move(on_submitted), std::move(wake)); });
    EXPECT_EQ(writes + interrupts + completions, 0u);
    gpu.Complete(41);
    gpu.Drain(batch);
    EXPECT_EQ(writes, 1u);
    EXPECT_EQ(interrupts, 1u);
    EXPECT_EQ(completions, 2u);
}

TEST(EventWriteEop, CompletionCanWakeIdleOwnerWithoutRunningFlipOnCompletionThread) {
    TestGpuTimeline gpu;
    AmdGpu::EopSubmissionBatch batch;
    AmdGpu::EopFlipTracker tracker;
    std::queue<Common::UniqueFunction<void>> owner_commands;
    bool presentation_label = false;
    u32 fence = 0;
    ASSERT_TRUE(
        tracker.QueueFlip(AmdGpu::FlipEopPosition::Following, [&] { presentation_label = true; }));
    batch.Add(
        TestEopPacket{.data = 42}, [&](u32 value) { fence = value; }, [] {}, tracker.BeginEop());

    // Flush before entering WAIT_REG_MEM, including a final EOP with no later
    // submission. Otherwise waiting for this fence would prevent its own submit.
    batch.Flush([] {},
                [&](auto&& on_submitted, auto&& wake) {
                    gpu.Submit(std::move(on_submitted), std::move(wake));
                },
                [&] { owner_commands.emplace([&] { gpu.Drain(batch); }); });
    EXPECT_TRUE(owner_commands.empty());
    gpu.Complete(41);
    ASSERT_EQ(owner_commands.size(), 1u);
    EXPECT_FALSE(presentation_label);
    EXPECT_EQ(fence, 0u);

    // A presentation-label wait must yield to owner commands, not block that
    // owner in a sleep whose wakeup requires this still-queued flip.
    owner_commands.front()();
    owner_commands.pop();
    EXPECT_TRUE(presentation_label);
    EXPECT_EQ(fence, 42u);
    gpu.Drain(batch);
    EXPECT_TRUE(owner_commands.empty());
}

TEST(EventWriteEop, CancellationDisarmsPendingFenceInterruptAndOwnerCallbacks) {
    TestGpuTimeline gpu;
    AmdGpu::EopSubmissionBatch batch;
    u32 operations = 0;
    u32 wakeups = 0;
    batch.Add(
        TestEopPacket{.data = 42}, [&](u32) { ++operations; }, [&] { ++operations; },
        [&] { ++operations; });
    batch.Flush([] {},
                [&](auto&& on_submitted, auto&& wake) {
                    gpu.Submit(std::move(on_submitted), std::move(wake));
                },
                [&] { ++wakeups; });
    batch.Cancel();
    gpu.Complete(41);
    gpu.Drain(batch);
    EXPECT_EQ(operations, 0u);
    EXPECT_EQ(wakeups, 0u);
}

TEST(EventWriteEop, GpuCompletionBarrierRetiresHostFenceBeforeUnmappingItsDestination) {
    TestGpuTimeline gpu;
    AmdGpu::EopSubmissionBatch batch;
    bool destination_mapped = true;
    bool wrote_after_unmap = false;
    u32 fence = 0;
    batch.Add(
        TestEopPacket{.data = 42},
        [&](u32 value) {
            wrote_after_unmap = !destination_mapped;
            fence = value;
        },
        [] {}, [] {});
    batch.Flush([] {}, [&](auto&& on_submitted,
                           auto&& wake) { gpu.Submit(std::move(on_submitted), std::move(wake)); });

    // Finish waits for the physical GPU, but the priority worker can be delayed.
    // A destructive address barrier must retire the host EOP publication too.
    gpu.completed_tick = 41;
    gpu.Drain(batch);
    EXPECT_EQ(fence, 42u);
    destination_mapped = false;
    gpu.Complete(41);
    gpu.Drain(batch);
    EXPECT_FALSE(wrote_after_unmap);
}

TEST(EventWriteEop, OwnerDestructionDisarmsAlreadySubmittedCompletion) {
    TestGpuTimeline gpu;
    u32 operations = 0;
    u32 wakeups = 0;
    {
        AmdGpu::EopSubmissionBatch batch;
        batch.Add(
            TestEopPacket{.data = 42}, [&](u32) { ++operations; }, [&] { ++operations; },
            [&] { ++operations; });
        batch.Flush([] {},
                    [&](auto&& on_submitted, auto&& wake) {
                        gpu.Submit(std::move(on_submitted), std::move(wake));
                    },
                    [&] { ++wakeups; });
    }
    gpu.Complete(41);
    EXPECT_EQ(operations, 0u);
    EXPECT_EQ(wakeups, 0u);
}

TEST(EventWriteEop, NestedOwnerDrainDoesNotRepublishAnActiveFenceOrOvertakeItsInterrupt) {
    TestGpuTimeline gpu;
    AmdGpu::EopSubmissionBatch batch;
    std::vector<u32> operations;
    for (u32 data : {41u, 42u}) {
        batch.Add(
            TestEopPacket{.data = data},
            [&](u32 value) {
                operations.emplace_back(value);
                // Simulate a fence page fault which synchronously enters
                // owner work again. The current atom must finish first.
                gpu.Drain(batch);
            },
            [&, data] { operations.emplace_back(data + 100); }, [] {});
    }
    batch.Flush([] {}, [&](auto&& on_submitted,
                           auto&& wake) { gpu.Submit(std::move(on_submitted), std::move(wake)); });
    gpu.completed_tick = 41;
    gpu.Drain(batch);
    gpu.Drain(batch);
    gpu.Complete(41);
    gpu.Drain(batch);
    EXPECT_EQ(operations, (std::vector<u32>{41, 141, 42, 142}));
}

TEST(EventWriteEop, InternalGpuWaitDoesNotPublishEopInsideActiveResourceSnapshot) {
    TestGpuTimeline gpu;
    AmdGpu::EopSubmissionBatch batch;
    bool snapshot_copy_active = true;
    u32 fence = 0;
    batch.Add(
        TestEopPacket{.data = 42},
        [&](u32 value) {
            EXPECT_FALSE(snapshot_copy_active);
            fence = value;
        },
        [] {}, [] {});
    batch.Flush([] {}, [&](auto&& on_submitted,
                           auto&& wake) { gpu.Submit(std::move(on_submitted), std::move(wake)); });
    // An internal GPU wait/readback is not an address-retirement boundary and
    // must not insert host writes into the resource snapshot it is preparing.
    gpu.Complete(41);
    EXPECT_EQ(fence, 0u);
    snapshot_copy_active = false;
    gpu.Drain(batch);
    EXPECT_EQ(fence, 42u);
}

TEST(GpuCompletionSubmission, BindsCallbackToSuccessfulSubmissionNotNextTick) {
    std::mutex mutex;
    Vulkan::GpuCompletionSubmission sequencer{mutex};
    u64 current_tick = 41;
    u64 deferred_tick = 0;
    bool executed = false;
    Common::UniqueFunction<void> deferred;
    const auto submitted_tick = sequencer.Submit(
        Common::UniqueFunction<void>{[&] { executed = true; }}, [&] { return current_tick++; },
        [&](auto&& callback, u64 tick) {
            deferred_tick = tick;
            deferred = std::move(callback);
        });
    EXPECT_EQ(submitted_tick, 41u);
    EXPECT_EQ(deferred_tick, 41u);
    EXPECT_EQ(current_tick, 42u);
    EXPECT_FALSE(executed);
    deferred();
    EXPECT_TRUE(executed);
}

TEST(GpuCompletionSubmission, FailedSubmissionDoesNotQueueAnUnsignalableFutureTick) {
    std::mutex mutex;
    Vulkan::GpuCompletionSubmission sequencer{mutex};
    bool enqueued = false;
    EXPECT_THROW(static_cast<void>(sequencer.Submit(
                     [] {}, []() -> u64 { throw std::runtime_error("submit failed"); },
                     [&](auto&&, u64) { enqueued = true; })),
                 std::runtime_error);
    EXPECT_FALSE(enqueued);
    ASSERT_TRUE(mutex.try_lock());
    mutex.unlock();
}

TEST(GpuCompletionSubmission, ExactTickRegistrationSerializesWithOrdinarySubmissions) {
    std::mutex mutex;
    Vulkan::GpuCompletionSubmission sequencer{mutex};
    std::atomic<u64> next_tick{41};
    u64 deferred_tick = 0;
    u64 ordinary_tick = 0;
    std::latch submit_entered{1};
    std::latch release_submit{1};
    std::jthread exact{[&] {
        static_cast<void>(sequencer.Submit([] {},
                                           [&] {
                                               const auto tick = next_tick.fetch_add(1);
                                               submit_entered.count_down();
                                               release_submit.wait();
                                               return tick;
                                           },
                                           [&](auto&&, u64 tick) { deferred_tick = tick; }));
    }};
    submit_entered.wait();
    std::jthread ordinary{[&] {
        std::scoped_lock lock{mutex};
        ordinary_tick = next_tick.fetch_add(1);
    }};
    release_submit.count_down();
    exact.join();
    ordinary.join();
    EXPECT_EQ(deferred_tick, 41u);
    EXPECT_EQ(ordinary_tick, 42u);
}

TEST(EventWriteEopWiring, GraphicsParserFlushesEveryBoundaryBeforeItCanWaitOrYield) {
    const auto source = ReadEopSource("video_core/amdgpu/liverpool.cpp");
    ASSERT_FALSE(source.empty());
    const auto graphics = SourceSection(source, "Liverpool::Task Liverpool::ProcessGraphics(",
                                        "Liverpool::Task Liverpool::ProcessCompute(");
    ASSERT_FALSE(graphics.empty());
    const auto packet_start = SourceSection(graphics, "while (!dcb.empty())", "switch (type)");
    ASSERT_FALSE(packet_start.empty());
    const auto boundary = packet_start.find("if (!is_eop && rasterizer)");
    const auto flush = packet_start.find("FlushPendingEops();");
    const auto command_guard = packet_start.find("if (!continues_eop_batch)");
    const auto commands = packet_start.find("ProcessCommands();");
    ASSERT_NE(boundary, std::string_view::npos);
    ASSERT_NE(flush, std::string_view::npos);
    ASSERT_NE(command_guard, std::string_view::npos);
    ASSERT_NE(commands, std::string_view::npos);
    EXPECT_LT(boundary, flush);
    EXPECT_LT(flush, command_guard);
    EXPECT_LT(command_guard, commands);

    const auto eop =
        SourceSection(graphics, "case PM4ItOpcode::EventWriteEop:", "case PM4ItOpcode::DmaData:");
    ASSERT_FALSE(eop.empty());
    const auto rasterizer_path = SourceSection(eop, "if (rasterizer)", "} else {");
    ASSERT_FALSE(rasterizer_path.empty());
    EXPECT_NE(rasterizer_path.find("eop_submission_batch.Add("), std::string_view::npos);
    EXPECT_EQ(rasterizer_path.find("PublishEop("), std::string_view::npos);

    const auto wait = SourceSection(
        graphics, "case PM4ItOpcode::WaitRegMem:", "case PM4ItOpcode::IndirectBuffer:");
    ASSERT_FALSE(wait.empty());
    EXPECT_NE(wait.find("YIELD_GFX();"), std::string_view::npos);
    EXPECT_EQ(wait.find("WaitVoLabel("), std::string_view::npos);
    const auto tail_flush = graphics.rfind("FlushPendingEops();");
    const auto ce_drain = graphics.rfind("if (ce_task.handle)");
    ASSERT_NE(tail_flush, std::string_view::npos);
    ASSERT_NE(ce_drain, std::string_view::npos);
    EXPECT_LT(tail_flush, ce_drain);
    EXPECT_GT(tail_flush, graphics.find("case PM4ItOpcode::IndirectBuffer:"));
}

TEST(EventWriteEopWiring, SchedulerUsesExactSuccessfulTickWithoutBlockingSubmissionAdmission) {
    const auto source = ReadEopSource("video_core/renderer_vulkan/vk_scheduler.cpp");
    ASSERT_FALSE(source.empty());
    const auto flush = SourceSection(source, "void Scheduler::FlushWithGpuCompletion(",
                                     "void Scheduler::Finish()");
    ASSERT_FALSE(flush.empty());
    const auto submit = flush.find("sequencer.Submit(");
    const auto register_owner = flush.find("on_submitted(u64{completion_tick});");
    const auto enqueue = flush.find("QueuePriorityOperationAtTick(");
    const auto notify = flush.find("priority_pending_ops_cv.notify_one();");
    const auto drain = flush.find("PopPendingOperations();");
    ASSERT_NE(submit, std::string_view::npos);
    ASSERT_NE(register_owner, std::string_view::npos);
    ASSERT_NE(enqueue, std::string_view::npos);
    ASSERT_NE(notify, std::string_view::npos);
    ASSERT_NE(drain, std::string_view::npos);
    EXPECT_LT(submit, register_owner);
    EXPECT_LT(register_owner, enqueue);
    EXPECT_LT(enqueue, notify);
    EXPECT_LT(notify, drain);
    EXPECT_NE(flush.find("return SubmitExecutionWithSubmitLockHeld(info);"),
              std::string_view::npos);
    EXPECT_NE(flush.find("std::move(callback), completion_tick"), std::string_view::npos);
    EXPECT_EQ(flush.find("CurrentTick()"), std::string_view::npos);
    EXPECT_EQ(flush.find("Wait("), std::string_view::npos);

    const auto execution =
        SourceSection(source, "u64 Scheduler::SubmitExecutionWithSubmitLockHeld(",
                      "void Scheduler::PriorityPendingOpsThread(");
    ASSERT_FALSE(execution.empty());
    const auto tick = execution.find("const u64 signal_value = master_semaphore.NextTick();");
    const auto driver_submit = execution.find("GetGraphicsQueue().submit(");
    const auto returned_tick = execution.find("return signal_value;");
    ASSERT_NE(tick, std::string_view::npos);
    ASSERT_NE(driver_submit, std::string_view::npos);
    ASSERT_NE(returned_tick, std::string_view::npos);
    EXPECT_LT(tick, driver_submit);
    EXPECT_LT(driver_submit, returned_tick);
    EXPECT_EQ(execution.find("PopPendingOperations();"), std::string_view::npos);

    const auto liverpool = ReadEopSource("video_core/amdgpu/liverpool.cpp");
    const auto submit_done = SourceSection(liverpool, "void Liverpool::SubmitDone(",
                                           "void Liverpool::WaitGpuCompletion(");
    ASSERT_FALSE(submit_done.empty());
    EXPECT_EQ(submit_done.find("FlushWithGpuCompletion"), std::string_view::npos);
}

TEST(EventWriteEopWiring, OwnerCancellationPrecedesStopAndFlipStillUsesOwnerQueue) {
    const auto source = ReadEopSource("video_core/amdgpu/liverpool.cpp");
    ASSERT_FALSE(source.empty());
    const auto destructor = SourceSection(source, "Liverpool::~Liverpool()", "void Liverpool::");
    ASSERT_FALSE(destructor.empty());
    const auto cancel = destructor.find("eop_submission_batch.Cancel();");
    const auto stop = destructor.find("process_thread.request_stop();");
    const auto join = destructor.find("process_thread.join();");
    ASSERT_NE(cancel, std::string_view::npos);
    ASSERT_NE(stop, std::string_view::npos);
    ASSERT_NE(join, std::string_view::npos);
    EXPECT_LT(cancel, stop);
    EXPECT_LT(stop, join);
    EXPECT_NE(source.find("while (!stoken.stop_requested() && (num_submits || num_commands))"),
              std::string::npos);
    const auto flip = SourceSection(source, "const auto eop_position = DecodeFlipEopPosition(",
                                    "case PM4CmdNop::PayloadType::DebugMarkerPush:");
    ASSERT_FALSE(flip.empty());
    EXPECT_NE(flip.find("SendCommand("), std::string_view::npos);
    EXPECT_NE(flip.find("Platform::InterruptId::GfxFlip"), std::string_view::npos);
    EXPECT_EQ(flip.find("SendCommand<true>"), std::string_view::npos);
}

TEST(EventWriteEopWiring, OnlyExplicitOwnerRetirementBarrierDrainsAfterPhysicalFinish) {
    const auto liverpool = ReadEopSource("video_core/amdgpu/liverpool.cpp");
    ASSERT_FALSE(liverpool.empty());
    const auto barrier =
        SourceSection(liverpool, "void Liverpool::WaitGpuCompletion()", "} // namespace AmdGpu");
    ASSERT_FALSE(barrier.empty());
    const auto finish = barrier.find("rasterizer->Finish();");
    const auto drain = barrier.find("DrainCompletedEops();");
    ASSERT_NE(finish, std::string_view::npos);
    ASSERT_NE(drain, std::string_view::npos);
    EXPECT_LT(finish, drain);
    EXPECT_NE(barrier.find("SendCommand<true>"), std::string_view::npos);

    const auto flush = SourceSection(liverpool, "void Liverpool::FlushPendingEops()",
                                     "void Liverpool::DrainCompletedEops()");
    ASSERT_FALSE(flush.empty());
    EXPECT_NE(flush.find("SendCommand([this] { DrainCompletedEops(); });"), std::string_view::npos);
    EXPECT_EQ(flush.find("PublishEop("), std::string_view::npos);
    const auto owner_drain = SourceSection(liverpool, "void Liverpool::DrainCompletedEops()",
                                           "void Liverpool::Process(");
    EXPECT_NE(owner_drain.find("eop_submission_batch.DrainCompleted("), std::string_view::npos);
    EXPECT_NE(owner_drain.find("GetScheduler().IsFree(tick)"), std::string_view::npos);

    const auto scheduler = ReadEopSource("video_core/renderer_vulkan/vk_scheduler.cpp");
    const auto internal_finish =
        SourceSection(scheduler, "void Scheduler::Finish()", "void Scheduler::Wait(");
    ASSERT_FALSE(internal_finish.empty());
    EXPECT_EQ(internal_finish.find("DrainCompleted"), std::string_view::npos);
}

} // namespace
