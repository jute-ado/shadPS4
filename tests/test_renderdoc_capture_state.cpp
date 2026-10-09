// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <thread>

#include <gtest/gtest.h>

#include "video_core/renderdoc_capture_state.h"

TEST(RenderDocCaptureState, BeginCaptureSpansACompletePresentedFrameInterval) {
    VideoCore::RenderDocCaptureState state;

    EXPECT_TRUE(state.TryBeginCapture());
    EXPECT_FALSE(state.ConsumeCaptureForPresentedFrame(false));
    EXPECT_TRUE(state.ConsumeCaptureForPresentedFrame(false));
    EXPECT_FALSE(state.ConsumeCaptureForPresentedFrame(false));
}

TEST(RenderDocCaptureState, CaptureStartIsPublishedAcrossThreads) {
    VideoCore::RenderDocCaptureState state;

    std::thread trigger_thread{[&state] { EXPECT_TRUE(state.TryBeginCapture()); }};
    trigger_thread.join();

    EXPECT_FALSE(state.ConsumeCaptureForPresentedFrame(false));
    EXPECT_TRUE(state.ConsumeCaptureForPresentedFrame(false));
}

TEST(RenderDocCaptureState, RepeatedStartDoesNotOverlapCapture) {
    VideoCore::RenderDocCaptureState state;

    ASSERT_TRUE(state.TryBeginCapture());
    EXPECT_FALSE(state.TryBeginCapture());
    EXPECT_FALSE(state.ConsumeCaptureForPresentedFrame(false));
    ASSERT_TRUE(state.ConsumeCaptureForPresentedFrame(false));
    EXPECT_FALSE(state.ConsumeCaptureForPresentedFrame(false));
}

TEST(RenderDocCaptureState, ReusedHostFramesDoNotAdvanceGuestFrameCapture) {
    VideoCore::RenderDocCaptureState state;

    ASSERT_TRUE(state.TryBeginCapture());
    EXPECT_FALSE(state.ConsumeCaptureForPresentedFrame(true));
    EXPECT_FALSE(state.ConsumeCaptureForPresentedFrame(true));
    EXPECT_FALSE(state.ConsumeCaptureForPresentedFrame(false));
    EXPECT_FALSE(state.ConsumeCaptureForPresentedFrame(true));
    EXPECT_TRUE(state.ConsumeCaptureForPresentedFrame(false));
}
