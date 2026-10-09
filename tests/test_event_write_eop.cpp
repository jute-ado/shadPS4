// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/types.h"
#include "core/libraries/videoout/eop_flip_presentation.h"
#include "video_core/amdgpu/eop_completion.h"
#include "video_core/amdgpu/eop_flip_tracker.h"

namespace {

std::string ReadLiverpoolSource() {
    std::ifstream stream{SHADPS4_LIVERPOOL_SOURCE_PATH, std::ios::binary};
    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

struct TestEopPacket {
    u32 data;

    void SignalFence(auto&& write_memory, auto&& signal_interrupt) const {
        write_memory(data);
        signal_interrupt();
    }
};

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
    EXPECT_EQ(flips, (std::vector<int>{1, 2}));
}

TEST(EventWriteEop, EopFlipWaitsForTheNextVblankIteration) {
    using Libraries::VideoOut::IsFlipPresentationEligible;

    EXPECT_FALSE(IsFlipPresentationEligible(true, 41, 41));
    EXPECT_TRUE(IsFlipPresentationEligible(true, 41, 42));
    EXPECT_TRUE(IsFlipPresentationEligible(false, 41, 41));
}

TEST(EventWriteEop, ExternalFenceWritesNotifySnapshotsBeforeBypassingVirtualProtection) {
    const auto source = ReadLiverpoolSource();
    ASSERT_FALSE(source.empty());

    const auto writer_begin = source.find("const auto write_guest_backing");
    const auto writer_end = source.find("const auto base_addr", writer_begin);
    ASSERT_NE(writer_begin, std::string::npos);
    ASSERT_NE(writer_end, std::string::npos);
    const auto writer = source.substr(writer_begin, writer_end - writer_begin);

    const auto notify = writer.find("rasterizer->NotifyGuestBackingWrite(");
    const auto backing_write = writer.find("memory->TryWriteBacking(");
    const auto fallback_write = writer.find("memcpy(", backing_write);
    ASSERT_NE(notify, std::string::npos);
    ASSERT_NE(backing_write, std::string::npos);
    ASSERT_NE(fallback_write, std::string::npos);
    EXPECT_LT(notify, backing_write);
    EXPECT_LT(backing_write, fallback_write);

    const auto eos = source.find("event_eos->SignalFence(write_guest_backing)");
    const auto eop = source.find("*event_eop, write_guest_backing");
    EXPECT_NE(eos, std::string::npos);
    EXPECT_NE(eop, std::string::npos);
}

} // namespace
