// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/ipc/presented_frame_input_queue.h"

namespace {

using Core::Ipc::PresentedFrameInput;
using Core::Ipc::PresentedFrameInputQueue;
using Core::Ipc::PresentedFrameInputType;

PresentedFrameInput Button(const u64 frame, const u32 code, const u8 value) {
    return {.frame = frame,
            .type = PresentedFrameInputType::Button,
            .code = code,
            .value = value};
}

PresentedFrameInput Screenshot(const u64 frame) {
    return {.frame = frame,
            .type = PresentedFrameInputType::ScreenshotWithOverlays};
}

} // namespace

TEST(PresentedFrameInputQueue, DoesNotReleaseInputBeforeItsFrame) {
    PresentedFrameInputQueue queue;
    ASSERT_TRUE(queue.Push(Button(10, 1, 1)));

    EXPECT_TRUE(queue.PopDue(9).empty());
    ASSERT_EQ(queue.Size(), 1);

    const auto due = queue.PopDue(10);
    ASSERT_EQ(due.size(), 1);
    EXPECT_EQ(due.front(), Button(10, 1, 1));
    EXPECT_EQ(queue.Size(), 0);
}

TEST(PresentedFrameInputQueue, ReleasesOverdueInputAndPreservesSameFrameOrder) {
    PresentedFrameInputQueue queue;
    ASSERT_TRUE(queue.Push(Button(3, 7, 1)));
    ASSERT_TRUE(queue.Push(Button(3, 8, 0)));
    ASSERT_TRUE(queue.Push(Button(5, 9, 1)));

    const auto due = queue.PopDue(4);
    ASSERT_EQ(due.size(), 2);
    EXPECT_EQ(due[0], Button(3, 7, 1));
    EXPECT_EQ(due[1], Button(3, 8, 0));
    EXPECT_EQ(queue.Size(), 1);
}

TEST(PresentedFrameInputQueue, RejectsOutOfOrderAndExcessInput) {
    PresentedFrameInputQueue queue{2};
    ASSERT_TRUE(queue.Push(Button(10, 1, 1)));
    EXPECT_FALSE(queue.Push(Button(9, 2, 1)));
    ASSERT_TRUE(queue.Push(Button(10, 3, 1)));
    EXPECT_FALSE(queue.Push(Button(11, 4, 1)));
    EXPECT_EQ(queue.Size(), 2);
}

TEST(PresentedFrameInputQueue, EveryPresentationAdvancesTheSchedulingClock) {
    PresentedFrameInputQueue queue;
    ASSERT_TRUE(queue.Push(Button(2, 1, 1)));

    EXPECT_TRUE(queue.Advance().empty());
    const auto due = queue.Advance();
    ASSERT_EQ(due.size(), 1);
    EXPECT_EQ(due.front(), Button(2, 1, 1));
}

TEST(PresentedFrameInputQueue, ReleasesScreenshotAtRequestedPresentation) {
    PresentedFrameInputQueue queue;
    ASSERT_TRUE(queue.Push(Button(2, 1, 1)));
    ASSERT_TRUE(queue.Push(Screenshot(2)));

    EXPECT_TRUE(queue.Advance().empty());
    const auto due = queue.Advance();
    ASSERT_EQ(due.size(), 2);
    EXPECT_EQ(due[0], Button(2, 1, 1));
    EXPECT_EQ(due[1], Screenshot(2));
}

TEST(PresentedFrameInputQueue, ReusedHostPresentationDoesNotAdvanceGuestFrameClock) {
    EXPECT_TRUE(Core::Ipc::AdvancesPresentedFrameAutomationClock(false));
    EXPECT_FALSE(Core::Ipc::AdvancesPresentedFrameAutomationClock(true));
}
