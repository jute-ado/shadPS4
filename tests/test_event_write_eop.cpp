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
#include "core/libraries/videoout/flip_label_generation.h"
#include "core/libraries/videoout/videoout_lifecycle.h"
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

TEST(EventWriteEop, FollowingFlipsStayBoundToTheirNextEop) {
    std::vector<int> flips;
    AmdGpu::EopFlipTracker tracker;

    ASSERT_TRUE(tracker.QueueFlip(AmdGpu::FlipEopPosition::Following,
                                  [&] { flips.emplace_back(1); }));
    auto complete_first_eop = tracker.BeginEop();

    ASSERT_TRUE(tracker.QueueFlip(AmdGpu::FlipEopPosition::Following,
                                  [&] { flips.emplace_back(2); }));
    auto complete_second_eop = tracker.BeginEop();

    EXPECT_TRUE(flips.empty());
    complete_first_eop();
    EXPECT_EQ(flips, (std::vector<int>{1}));
    complete_second_eop();
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

TEST(EventWriteEop, UnmappedFenceSkipsGuestWriteButCompletesInterruptAndFlip) {
    TestGpuTimeline gpu;
    AmdGpu::EopSubmissionBatch batch;
    AmdGpu::EopFlipTracker tracker;
    std::vector<std::string> operations;
    u32 guest_fence = 0xdeadbeef;
    const bool fence_destination_captured = false;

    ASSERT_TRUE(tracker.QueueFlip(AmdGpu::FlipEopPosition::Following,
                                  [&] { operations.emplace_back("flip"); }));
    batch.Add(
        TestEopPacket{.data = 42},
        [&](u32 value) {
            // CaptureReadbackDestinations returns no token for an unreadable or unmapped
            // destination. Do not fall back to the raw guest pointer or overwrite an old value.
            if (!fence_destination_captured) {
                return;
            }
            guest_fence = value;
            operations.emplace_back("fence");
        },
        [&] { operations.emplace_back("interrupt"); }, tracker.BeginEop());

    batch.Flush([] {}, [&](auto&& on_submitted,
                           auto&& wake) { gpu.Submit(std::move(on_submitted), std::move(wake)); });
    EXPECT_TRUE(operations.empty());
    gpu.Complete(41);
    gpu.Drain(batch);

    EXPECT_EQ(guest_fence, 0xdeadbeefu);
    EXPECT_EQ(operations, (std::vector<std::string>{"interrupt", "flip"}));
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
    // Every graphics-owner wait must remain cooperative, including the last-submission VO-label
    // path whose flip completion is serviced by the same GPU worker.
    EXPECT_NE(wait.find("is_vo_label_range"), std::string_view::npos);
    EXPECT_EQ(wait.find("WaitVoLabel("), std::string_view::npos);
    EXPECT_NE(wait.find("while (vo_port->lifecycle.IsCurrent(wait_lifecycle_epoch)"),
              std::string_view::npos);
    EXPECT_NE(wait.find("test_wait_reg_mem"), std::string_view::npos);
    const auto tail_flush = graphics.rfind("FlushPendingEops();");
    const auto ce_drain = graphics.rfind("if (ce_task.handle)");
    ASSERT_NE(tail_flush, std::string_view::npos);
    ASSERT_NE(ce_drain, std::string_view::npos);
    EXPECT_LT(tail_flush, ce_drain);
    EXPECT_GT(tail_flush, graphics.find("case PM4ItOpcode::IndirectBuffer:"));
}

TEST(EventWriteEopWiring, VoLabelWaitAndWritersShareCompletionMutex) {
    const auto liverpool = ReadEopSource("video_core/amdgpu/liverpool.cpp");
    ASSERT_FALSE(liverpool.empty());
    const auto graphics = SourceSection(liverpool, "Liverpool::Task Liverpool::ProcessGraphics(",
                                        "Liverpool::Task Liverpool::ProcessCompute(");
    ASSERT_FALSE(graphics.empty());

    const auto wait =
        SourceSection(graphics, "case PM4ItOpcode::WaitRegMem:", "case PM4ItOpcode::IndirectBuffer:");
    ASSERT_FALSE(wait.empty());
    // Memory-space polls whose addresses are inside the host-owned label array must retain the
    // VideoOut lifetime they observed. Register polls remain on the checked generic path.
    EXPECT_NE(wait.find("wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory"),
              std::string_view::npos);
    EXPECT_EQ(wait.find("vo_port->WaitVoLabel"), std::string_view::npos);
    EXPECT_NE(wait.find("const auto wait_lifecycle_epoch ="), std::string_view::npos);
    EXPECT_NE(wait.find("wait_is_vo_label ? vo_port->lifecycle.CurrentEpoch()"),
              std::string_view::npos);
    EXPECT_NE(wait.find("while (vo_port->lifecycle.IsCurrent(wait_lifecycle_epoch)"),
              std::string_view::npos);
    EXPECT_NE(wait.find("!vo_port->lifecycle.IsCurrent(wait_lifecycle_epoch)"),
              std::string_view::npos);
    EXPECT_NE(wait.find("YIELD_GFX();"), std::string_view::npos);
    EXPECT_NE(wait.find("mapped_queues[GfxQueueId].m_access"), std::string_view::npos);

    const auto write_guest =
        SourceSection(graphics, "const auto try_write_guest =", "const auto test_wait_reg_mem =");
    ASSERT_FALSE(write_guest.empty());
    // PM4 writes can update the labels while a graphics owner is waiting on them. They must
    // participate in the same mutex/condition-variable protocol as presenter resets.
    EXPECT_NE(write_guest.find("vo_port->vo_mutex"), std::string_view::npos);

    const auto driver = ReadEopSource("core/libraries/videoout/driver.cpp");
    ASSERT_FALSE(driver.empty());
    const auto flip = SourceSection(driver, "void VideoOutDriver::Flip(", "bool VideoOutDriver::SubmitFlip(");
    ASSERT_FALSE(flip.empty());
    EXPECT_NE(flip.find("port->vo_mutex"), std::string_view::npos);

    const auto video_out = ReadEopSource("core/libraries/videoout/video_out.cpp");
    ASSERT_FALSE(video_out.empty());
    const auto eop_flip = SourceSection(video_out, "s32 sceVideoOutSubmitEopFlip(",
                                        "s32 PS4_SYSV_ABI sceVideoOutGetDeviceCapabilityInfo(");
    ASSERT_FALSE(eop_flip.empty());
    EXPECT_NE(eop_flip.find("port->vo_mutex"), std::string_view::npos);
}

TEST(EventWriteEopWiring, GenericVoLabelWaitEscapesAfterPortCloseAndReopen) {
    const auto source = ReadEopSource("video_core/amdgpu/liverpool.cpp");
    ASSERT_FALSE(source.empty());
    const auto graphics = SourceSection(source, "Liverpool::Task Liverpool::ProcessGraphics(",
                                        "Liverpool::Task Liverpool::ProcessCompute(");
    ASSERT_FALSE(graphics.empty());
    const auto wait =
        SourceSection(graphics, "case PM4ItOpcode::WaitRegMem:", "case PM4ItOpcode::IndirectBuffer:");
    ASSERT_FALSE(wait.empty());

    const auto epoch_capture = wait.find("const auto wait_lifecycle_epoch =");
    const auto epoch_source = wait.find("wait_is_vo_label ? vo_port->lifecycle.CurrentEpoch()");
    const auto last_submission = wait.find("const bool wait_is_last_submission =");
    const auto generic_loop = wait.find("while (true) {");
    const auto lifecycle_escape =
        wait.find("!vo_port->lifecycle.IsCurrent(wait_lifecycle_epoch)", generic_loop);
    ASSERT_NE(epoch_capture, std::string_view::npos);
    ASSERT_NE(epoch_source, std::string_view::npos);
    ASSERT_NE(last_submission, std::string_view::npos);
    ASSERT_NE(generic_loop, std::string_view::npos);
    ASSERT_NE(lifecycle_escape, std::string_view::npos);
    EXPECT_LT(epoch_capture, generic_loop);
    EXPECT_LT(epoch_source, generic_loop);
    EXPECT_LT(epoch_source, last_submission);
    EXPECT_LT(generic_loop, lifecycle_escape);

    // The guard must be in the generic loop, so a close/reopen invalidates a stale label wait even
    // when this submission is not the queue's final graphics submission.
    const auto generic_result = wait.find("const auto result = test_wait_reg_mem", lifecycle_escape);
    ASSERT_NE(generic_result, std::string_view::npos);
    EXPECT_LT(lifecycle_escape, generic_result);
}

TEST(FlipLabelGeneration, OlderPresentationCannotClearNewerReservation) {
    Libraries::VideoOut::FlipLabelGeneration generation;
    const auto older = generation.Reserve();
    const auto newer = generation.Reserve();
    generation.Complete(newer);
    generation.Display(older);

    EXPECT_NE(older, newer);
    EXPECT_FALSE(generation.CanReleaseDisplayed());

    generation.Display(newer);
    EXPECT_TRUE(generation.CanReleaseDisplayed());
}

TEST(FlipLabelGeneration, CompletedPresentationCanReleaseBeforeNewerReservationDisplays) {
    Libraries::VideoOut::FlipLabelGeneration generation;
    const auto older = generation.Reserve();
    generation.Complete(older);
    generation.Display(older);
    const auto newer = generation.Reserve();

    // The next flip may already be reserved for this two-buffer slot, but the completed older
    // presentation must still release its label so the guest can submit the newer one.
    EXPECT_NE(older, newer);
    EXPECT_TRUE(generation.CanReleaseDisplayed());
}

TEST(VideoOutLifecycle, CloseAndReopenInvalidateCapturedEpoch) {
    Libraries::VideoOut::VideoOutLifecycle lifecycle;
    const auto first_epoch = lifecycle.Open();

    EXPECT_TRUE(lifecycle.IsOpen());
    EXPECT_TRUE(lifecycle.IsCurrent(first_epoch));

    lifecycle.Close();
    EXPECT_FALSE(lifecycle.IsOpen());
    EXPECT_FALSE(lifecycle.IsCurrent(first_epoch));

    const auto second_epoch = lifecycle.Open();
    EXPECT_NE(first_epoch, second_epoch);
    EXPECT_TRUE(lifecycle.IsCurrent(second_epoch));
    EXPECT_FALSE(lifecycle.IsCurrent(first_epoch));
}

TEST(VideoOutLifecycle, StaleCallbackCannotUseReopenedPort) {
    Libraries::VideoOut::VideoOutLifecycle lifecycle;
    const auto captured_epoch = lifecycle.Open();
    u32 submissions = 0;

    const auto callback = [&] {
        if (lifecycle.IsCurrent(captured_epoch)) {
            ++submissions;
        }
    };

    lifecycle.Close();
    lifecycle.Open();
    callback();

    EXPECT_EQ(submissions, 0u);
}

TEST(VideoOutLifecycle, CloseDrainsQueuedRequests) {
    std::queue<u32> requests;
    requests.push(1);
    requests.push(2);

    Libraries::VideoOut::DrainVideoOutRequests(requests);

    EXPECT_TRUE(requests.empty());
}

TEST(EventWriteEopWiring, VideoOutCloseInvalidatesEpochAndDrainsQueue) {
    const auto driver = ReadEopSource("core/libraries/videoout/driver.cpp");
    const auto header = ReadEopSource("core/libraries/videoout/driver.h");
    ASSERT_FALSE(driver.empty());
    ASSERT_FALSE(header.empty());

    const auto close =
        SourceSection(driver, "void VideoOutDriver::Close(", "VideoOutPort* VideoOutDriver::GetPort(");
    const auto request = SourceSection(header, "struct Request {", "operator bool()");
    ASSERT_FALSE(close.empty());
    ASSERT_FALSE(request.empty());

    EXPECT_NE(close.find("main_port.lifecycle.Close()"), std::string_view::npos);
    EXPECT_NE(close.find("DrainVideoOutRequests(requests)"), std::string_view::npos);
    EXPECT_NE(request.find("lifecycle_epoch"), std::string_view::npos);
}

TEST(EventWriteEopWiring, GfxFlipCallbackRejectsStalePortEpoch) {
    const auto source = ReadEopSource("core/libraries/videoout/video_out.cpp");
    ASSERT_FALSE(source.empty());
    const auto eop = SourceSection(source, "s32 sceVideoOutSubmitEopFlip(",
                                   "s32 PS4_SYSV_ABI sceVideoOutGetDeviceCapabilityInfo(");
    ASSERT_FALSE(eop.empty());

    const auto capture = eop.find("lifecycle_epoch");
    const auto current = eop.find("lifecycle.IsCurrent(lifecycle_epoch)");
    const auto submit = eop.find("SubmitFlip(port, buf_id, flip_arg, true, generation,");
    ASSERT_NE(capture, std::string_view::npos);
    ASSERT_NE(current, std::string_view::npos);
    ASSERT_NE(submit, std::string_view::npos);
    EXPECT_LT(capture, current);
    EXPECT_LT(current, submit);
}

TEST(EventWriteEopWiring, QueuedFlipChecksPortEpochBeforePresenting) {
    const auto source = ReadEopSource("core/libraries/videoout/driver.cpp");
    ASSERT_FALSE(source.empty());
    const auto flip =
        SourceSection(source, "void VideoOutDriver::Flip(", "void VideoOutDriver::DrawBlankFrame(");
    const auto internal = SourceSection(source, "void VideoOutDriver::SubmitFlipInternal(",
                                        "void VideoOutDriver::PresentThread(");
    ASSERT_FALSE(flip.empty());
    ASSERT_FALSE(internal.empty());

    EXPECT_NE(flip.find("lifecycle.IsCurrent(req.lifecycle_epoch)"), std::string_view::npos);
    EXPECT_NE(internal.find("lifecycle.IsCurrent(lifecycle_epoch)"), std::string_view::npos);
    EXPECT_NE(internal.find(".lifecycle_epoch = lifecycle_epoch"), std::string_view::npos);
}

TEST(EventWriteEopWiring, DroppedQueuedFlipReleasesSameLifetimePendingAccounting) {
    const auto source = ReadEopSource("core/libraries/videoout/driver.cpp");
    ASSERT_FALSE(source.empty());
    const auto internal = SourceSection(source, "void VideoOutDriver::SubmitFlipInternal(",
                                        "void VideoOutDriver::PresentThread(");
    ASSERT_FALSE(internal.empty());

    const auto discard = internal.find("const auto discard_pending");
    const auto pending = internal.find("flip_status.flip_pending_num", discard);
    const auto gc = internal.find("flip_status.gc_queue_num", discard);
    const auto first_return = internal.find("discard_pending();\n            return;", discard);
    ASSERT_NE(discard, std::string_view::npos);
    ASSERT_NE(pending, std::string_view::npos);
    ASSERT_NE(gc, std::string_view::npos);
    ASSERT_NE(first_return, std::string_view::npos);
    EXPECT_LT(discard, pending);
    EXPECT_LT(pending, first_return);
    EXPECT_LT(discard, gc);
    EXPECT_NE(internal.find("if (buffer.group_index < 0)", discard), std::string_view::npos);
    EXPECT_NE(internal.find("discard_pending();", first_return + 1), std::string_view::npos);
}

TEST(EventWriteEopWiring, FlipEventVectorsSharePortLifecycleLock) {
    const auto driver = ReadEopSource("core/libraries/videoout/driver.cpp");
    const auto video_out = ReadEopSource("core/libraries/videoout/video_out.cpp");
    ASSERT_FALSE(driver.empty());
    ASSERT_FALSE(video_out.empty());

    const auto flip =
        SourceSection(driver, "void VideoOutDriver::Flip(", "void VideoOutDriver::DrawBlankFrame(");
    const auto close =
        SourceSection(driver, "void VideoOutDriver::Close(", "VideoOutPort* VideoOutDriver::GetPort(");
    const auto add = SourceSection(video_out, "s32 PS4_SYSV_ABI sceVideoOutAddFlipEvent(",
                                   "s32 PS4_SYSV_ABI sceVideoOutDeleteFlipEvent(");
    const auto remove = SourceSection(video_out, "s32 PS4_SYSV_ABI sceVideoOutDeleteFlipEvent(",
                                      "s32 PS4_SYSV_ABI sceVideoOutAddVblankEvent(");
    ASSERT_FALSE(flip.empty());
    ASSERT_FALSE(close.empty());
    ASSERT_FALSE(add.empty());
    ASSERT_FALSE(remove.empty());

    EXPECT_NE(flip.find("port->port_mutex"), std::string_view::npos);
    EXPECT_NE(close.find("main_port.port_mutex"), std::string_view::npos);
    EXPECT_NE(add.find("port->port_mutex"), std::string_view::npos);
    EXPECT_NE(remove.find("port->port_mutex"), std::string_view::npos);
}

TEST(EventWriteEopWiring, VideoOutPresentationStateUsesPortLifecycleLock) {
    const auto driver = ReadEopSource("core/libraries/videoout/driver.cpp");
    const auto video_out = ReadEopSource("core/libraries/videoout/video_out.cpp");
    ASSERT_FALSE(driver.empty());
    ASSERT_FALSE(video_out.empty());

    const auto present =
        SourceSection(driver, "void VideoOutDriver::PresentThread(", "} // namespace");
    const auto set_rate = SourceSection(video_out, "s32 PS4_SYSV_ABI sceVideoOutSetFlipRate(",
                                        "s32 PS4_SYSV_ABI sceVideoOutIsFlipPending(");
    const auto configure = SourceSection(
        video_out, "s32 PS4_SYSV_ABI sceVideoOutConfigureOutputMode_(",
        "s32 PS4_SYSV_ABI sceVideoOutSubmitChangeBufferAttribute(");
    ASSERT_FALSE(present.empty());
    ASSERT_FALSE(set_rate.empty());
    ASSERT_FALSE(configure.empty());

    const auto present_lock = present.find("std::scoped_lock vo_lock{main_port.vo_mutex};");
    const auto present_rate = present.find("flip_rate = main_port.flip_rate");
    ASSERT_NE(present_lock, std::string_view::npos);
    ASSERT_NE(present_rate, std::string_view::npos);
    EXPECT_LT(present_lock, present_rate);
    EXPECT_NE(present.substr(present_lock, present.find("if (current_vblank", present_lock) -
                              present_lock)
                  .find("flip_rate"),
              std::string_view::npos);

    const auto rate_port = set_rate.find("auto* port = driver->GetPort(handle);");
    const auto rate_guard = set_rate.find("if (port == nullptr)", rate_port);
    const auto rate_lock = set_rate.find("port->port_mutex", rate_port);
    const auto rate_write = set_rate.find("port->flip_rate = rate", rate_port);
    ASSERT_NE(rate_port, std::string_view::npos);
    ASSERT_NE(rate_guard, std::string_view::npos);
    ASSERT_NE(rate_lock, std::string_view::npos);
    ASSERT_NE(rate_write, std::string_view::npos);
    EXPECT_LT(rate_port, rate_guard);
    EXPECT_LT(rate_guard, rate_lock);
    EXPECT_LT(rate_lock, rate_write);

    const auto configure_lock = configure.find("port->port_mutex");
    const auto configure_switch = configure.find("switch (mode->colorimetry)");
    const auto configure_write = configure.find("port->is_hdr");
    ASSERT_NE(configure_lock, std::string_view::npos);
    ASSERT_NE(configure_switch, std::string_view::npos);
    ASSERT_NE(configure_write, std::string_view::npos);
    EXPECT_LT(configure_lock, configure_switch);
    EXPECT_LT(configure_lock, configure_write);
}

TEST(EventWriteEopWiring, ChangeBufferAttributeDoesNotRelockPortMutex) {
    const auto driver = ReadEopSource("core/libraries/videoout/driver.cpp");
    ASSERT_FALSE(driver.empty());

    const auto change = SourceSection(driver, "int VideoOutDriver::ChangeBufferAttribute(",
                                      "void VideoOutDriver::Flip(");
    ASSERT_FALSE(change.empty());

    // Validation and the final copy run under the one lock acquired at function entry. Relocking
    // this non-recursive mutex deadlocks the guest thread on every valid attribute submission.
    EXPECT_NE(change.find("std::scoped_lock port_lock{port->port_mutex}"), std::string_view::npos);
    EXPECT_NE(change.find("attributeIndex < 0"), std::string_view::npos);
    EXPECT_EQ(change.find("std::unique_lock lock{port->port_mutex}"), std::string_view::npos);
}

TEST(EventWriteEopWiring, VideoOutLifecycleLocksHaveOneGlobalOrder) {
    const auto driver = ReadEopSource("core/libraries/videoout/driver.cpp");
    ASSERT_FALSE(driver.empty());

    const auto open =
        SourceSection(driver, "int VideoOutDriver::Open(", "void VideoOutDriver::Close(");
    const auto close =
        SourceSection(driver, "void VideoOutDriver::Close(", "VideoOutPort* VideoOutDriver::GetPort(");
    ASSERT_FALSE(open.empty());
    ASSERT_FALSE(close.empty());

    const auto open_port = open.find("std::scoped_lock port_lock{main_port.port_mutex}");
    const auto open_queue = open.find("std::scoped_lock lock{mutex}");
    const auto close_port = close.find("std::scoped_lock port_lock{main_port.port_mutex}");
    const auto close_queue = close.find("std::scoped_lock lock{mutex}");
    ASSERT_NE(open_port, std::string_view::npos);
    ASSERT_NE(open_queue, std::string_view::npos);
    ASSERT_NE(close_port, std::string_view::npos);
    ASSERT_NE(close_queue, std::string_view::npos);
    EXPECT_LT(open_port, open_queue);
    EXPECT_LT(close_port, close_queue);
}

TEST(EventWriteEopWiring, VideoOutCloseCancelsVblankWait) {
    const auto driver = ReadEopSource("core/libraries/videoout/driver.cpp");
    const auto video_out = ReadEopSource("core/libraries/videoout/video_out.cpp");
    ASSERT_FALSE(driver.empty());
    ASSERT_FALSE(video_out.empty());

    const auto close =
        SourceSection(driver, "void VideoOutDriver::Close(", "VideoOutPort* VideoOutDriver::GetPort(");
    const auto wait = SourceSection(video_out, "s32 PS4_SYSV_ABI sceVideoOutWaitVblank(",
                                    "s32 PS4_SYSV_ABI sceVideoOutColorSettingsSetGamma(");
    ASSERT_FALSE(close.empty());
    ASSERT_FALSE(wait.empty());

    const auto reset = close.find("std::memset(&main_port.vblank_status");
    const auto notify = close.find("main_port.vblank_cv.notify_all()");
    ASSERT_NE(reset, std::string_view::npos);
    ASSERT_NE(notify, std::string_view::npos);
    EXPECT_LT(reset, notify);
    EXPECT_NE(wait.find("const auto wait_epoch = port->lifecycle.CurrentEpoch()"),
              std::string_view::npos);
    EXPECT_NE(wait.find("!port->lifecycle.IsCurrent(wait_epoch)"), std::string_view::npos);
    EXPECT_NE(wait.find("ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE"), std::string_view::npos);
}

TEST(EventWriteEopWiring, FlipLabelGenerationGuardsOlderPresentationClear) {
    const auto driver = ReadEopSource("core/libraries/videoout/driver.cpp");
    ASSERT_FALSE(driver.empty());

    const auto flip =
        SourceSection(driver, "void VideoOutDriver::Flip(", "void VideoOutDriver::DrawBlankFrame(");
    ASSERT_FALSE(flip.empty());
    const auto release_check = flip.find("CanReleaseDisplayed");
    const auto label_clear = flip.find("buffer_labels[previous_index] = 0");
    const auto display = flip.find("Display(req.generation)");
    const auto update_previous = flip.find("port->prev_index = req.index");
    ASSERT_NE(release_check, std::string_view::npos);
    ASSERT_NE(label_clear, std::string_view::npos);
    ASSERT_NE(display, std::string_view::npos);
    ASSERT_NE(update_previous, std::string_view::npos);
    EXPECT_LT(release_check, label_clear);
    EXPECT_LT(display, update_previous);
    EXPECT_NE(flip.find("flip_label_generations"), std::string_view::npos);

    const auto submit = SourceSection(driver, "bool VideoOutDriver::SubmitFlip(",
                                      "void VideoOutDriver::SubmitFlipInternal(");
    ASSERT_FALSE(submit.empty());
    EXPECT_NE(submit.find("generation == 0"), std::string_view::npos);
    EXPECT_NE(submit.find("flip_label_generations[index].Reserve()"), std::string_view::npos);
    EXPECT_NE(submit.find("SubmitFlipInternal(port, index, flip_arg, is_eop, generation, request_epoch)"),
              std::string_view::npos);

    const auto internal = SourceSection(driver, "void VideoOutDriver::SubmitFlipInternal(",
                                        "void VideoOutDriver::PresentThread(");
    ASSERT_FALSE(internal.empty());
    EXPECT_NE(internal.find(".generation = generation"), std::string_view::npos);

    const auto video_out = ReadEopSource("core/libraries/videoout/video_out.cpp");
    ASSERT_FALSE(video_out.empty());
    const auto eop_flip = SourceSection(video_out, "s32 sceVideoOutSubmitEopFlip(",
                                        "s32 PS4_SYSV_ABI sceVideoOutGetDeviceCapabilityInfo(");
    ASSERT_FALSE(eop_flip.empty());
    const auto reserve = eop_flip.find("ReserveFlipLabel");
    const auto register_irq = eop_flip.find("RegisterOnce");
    ASSERT_NE(reserve, std::string_view::npos);
    ASSERT_NE(register_irq, std::string_view::npos);
    EXPECT_LT(reserve, register_irq);
    EXPECT_NE(eop_flip.find("flip_arg, true, generation, lifecycle_epoch"), std::string_view::npos);
}

TEST(EventWriteEopWiring, EopReservationCompletesAtIrqSubmitBoundary) {
    const auto driver = ReadEopSource("core/libraries/videoout/driver.cpp");
    ASSERT_FALSE(driver.empty());
    const auto submit = SourceSection(driver, "bool VideoOutDriver::SubmitFlip(",
                                      "void VideoOutDriver::SubmitFlipInternal(");
    ASSERT_FALSE(submit.empty());

    const auto fallback_reserve =
        submit.find("if (is_eop && index != -1 && generation == 0)");
    const auto completion_guard = submit.find("if (is_eop && index != -1)");
    const auto complete = submit.find("flip_label_generations[index].Complete(generation)");
    ASSERT_NE(fallback_reserve, std::string_view::npos);
    ASSERT_NE(completion_guard, std::string_view::npos);
    ASSERT_NE(complete, std::string_view::npos);
    EXPECT_LT(fallback_reserve, completion_guard);
    EXPECT_LT(completion_guard, complete);
    EXPECT_EQ(submit.find("if (index != -1 && generation == 0)"), std::string_view::npos);

    const auto video_out = ReadEopSource("core/libraries/videoout/video_out.cpp");
    ASSERT_FALSE(video_out.empty());
    const auto eop_flip = SourceSection(video_out, "s32 sceVideoOutSubmitEopFlip(",
                                        "s32 PS4_SYSV_ABI sceVideoOutGetDeviceCapabilityInfo(");
    ASSERT_FALSE(eop_flip.empty());
    // The reservation remains at API registration, while completion is deferred until the
    // interrupt callback reaches SubmitFlip.
    EXPECT_LT(eop_flip.find("ReserveFlipLabel"), eop_flip.find("RegisterOnce"));
}

TEST(EventWriteEopWiring, FenceStoresCaptureAndPublishExactBackingOwnership) {
    const auto source = ReadEopSource("video_core/amdgpu/liverpool.cpp");
    ASSERT_FALSE(source.empty());
    const auto graphics = SourceSection(source, "Liverpool::Task Liverpool::ProcessGraphics(",
                                        "Liverpool::Task Liverpool::ProcessCompute(");
    const auto eos = SourceSection(source, "case PM4ItOpcode::EventWriteEos:",
                                   "case PM4ItOpcode::EventWriteEop:");
    const auto eop = SourceSection(source, "case PM4ItOpcode::EventWriteEop:",
                                   "case PM4ItOpcode::DmaData:");
    const auto publish_fence = SourceSection(graphics, "const auto publish_fence =",
                                             "const auto test_wait_reg_mem =");
    ASSERT_FALSE(graphics.empty());
    ASSERT_FALSE(eos.empty());
    ASSERT_FALSE(eop.empty());
    ASSERT_FALSE(publish_fence.empty());
    EXPECT_NE(eos.find("CaptureReadbackDestinations"), std::string_view::npos);
    EXPECT_NE(eos.find("publish_fence"), std::string_view::npos);
    EXPECT_NE(eop.find("CaptureReadbackDestinations"), std::string_view::npos);
    EXPECT_NE(eop.find("publish_fence"), std::string_view::npos);
    EXPECT_EQ(eos.find("memcpy(address"), std::string_view::npos);
    EXPECT_EQ(eop.find("memcpy(address"), std::string_view::npos);

    // Video-out labels are host-owned and therefore cannot produce a VMA-backed token. They must
    // use the checked label path; ordinary guest fences retain exact deferred backing validation.
    const auto label_check = publish_fence.find("is_vo_label_range");
    const auto label_lock = publish_fence.find("vo_port->vo_mutex");
    const auto label_wake = publish_fence.find("vo_port->vo_cv.notify_all()");
    const auto guest_publish = publish_fence.find("TryPublishFence");
    ASSERT_NE(label_check, std::string_view::npos);
    ASSERT_NE(label_lock, std::string_view::npos);
    ASSERT_NE(label_wake, std::string_view::npos);
    ASSERT_NE(guest_publish, std::string_view::npos);
    EXPECT_LT(label_check, guest_publish);
    EXPECT_LT(label_lock, label_wake);
}

TEST(EventWriteEopWiring, GdsStoreUsesCheckedGuestWrite) {
    const auto source = ReadEopSource("video_core/amdgpu/liverpool.cpp");
    ASSERT_FALSE(source.empty());
    const auto graphics = SourceSection(source, "Liverpool::Task Liverpool::ProcessGraphics(",
                                        "Liverpool::Task Liverpool::ProcessCompute(");
    const auto eos = SourceSection(source, "case PM4ItOpcode::EventWriteEos:",
                                   "case PM4ItOpcode::EventWriteEop:");
    const auto write_guest = SourceSection(graphics, "const auto try_write_guest =",
                                           "const auto publish_fence =");
    ASSERT_FALSE(graphics.empty());
    ASSERT_FALSE(eos.empty());
    ASSERT_FALSE(write_guest.empty());
    const auto gds = SourceSection(eos, "if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore)",
                                   "break;");
    ASSERT_FALSE(gds.empty());
    EXPECT_NE(gds.find("try_write_guest"), std::string_view::npos);
    EXPECT_EQ(gds.find("*event_eos->Address()"), std::string_view::npos);
    EXPECT_NE(write_guest.find("vo_port->vo_mutex"), std::string_view::npos);
    EXPECT_NE(write_guest.find("vo_port->vo_cv.notify_all()"), std::string_view::npos);
    EXPECT_NE(write_guest.find("TryWriteGuest"), std::string_view::npos);
    EXPECT_NE(write_guest.find("TryWriteBacking"), std::string_view::npos);
}

TEST(GuestMemoryWiring, ConstantEngineDumpUsesCheckedGuestWrite) {
    const auto source = ReadEopSource("video_core/amdgpu/liverpool.cpp");
    ASSERT_FALSE(source.empty());
    const auto ce = SourceSection(source, "Liverpool::Task Liverpool::ProcessCeUpdate(",
                                  "Liverpool::Task Liverpool::ProcessGraphics(");
    ASSERT_FALSE(ce.empty());

    const auto dump = SourceSection(ce, "case PM4ItOpcode::DumpConstRam:", "case PM4ItOpcode::");
    ASSERT_FALSE(dump.empty());
    EXPECT_NE(dump.find("TryWriteGuest"), std::string_view::npos);
    EXPECT_EQ(dump.find("memcpy(dump_const->Address"), std::string_view::npos);
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
