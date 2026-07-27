// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <thread>

#include <gtest/gtest.h>

#include "video_core/renderdoc_capture_state.h"

TEST(RenderDocCaptureState, TriggerSchedulesExactlyOnePresentedFrame) {
    VideoCore::RenderDocCaptureState state;

    EXPECT_TRUE(state.Trigger());
    EXPECT_TRUE(state.ConsumePresentedFrameTrigger());
    EXPECT_FALSE(state.ConsumePresentedFrameTrigger());
}

TEST(RenderDocCaptureState, TriggerIsPublishedAcrossThreads) {
    VideoCore::RenderDocCaptureState state;

    std::thread trigger_thread{[&state] { EXPECT_TRUE(state.Trigger()); }};
    trigger_thread.join();

    EXPECT_TRUE(state.ConsumePresentedFrameTrigger());
}

TEST(RenderDocCaptureState, RepeatedTriggerDoesNotQueueAnotherCapture) {
    VideoCore::RenderDocCaptureState state;

    ASSERT_TRUE(state.Trigger());
    EXPECT_FALSE(state.Trigger());
    ASSERT_TRUE(state.ConsumePresentedFrameTrigger());
    EXPECT_FALSE(state.ConsumePresentedFrameTrigger());
}
