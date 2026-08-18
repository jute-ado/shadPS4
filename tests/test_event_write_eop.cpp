// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/types.h"
#include "core/libraries/videoout/eop_flip_presentation.h"
#include "video_core/amdgpu/eop_completion.h"
#include "video_core/amdgpu/eop_flip_tracker.h"

namespace {

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

} // namespace
