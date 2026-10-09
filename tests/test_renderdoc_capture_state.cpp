// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <thread>

#include <gtest/gtest.h>

#include "video_core/renderdoc_capture_state.h"

TEST(RenderDocCaptureState, TriggerSchedulesExactlyOneGuestCommandBufferFrame) {
    VideoCore::RenderDocCaptureState state;

    EXPECT_TRUE(state.Trigger());
    EXPECT_FALSE(state.ConsumeGuestFrameTrigger());
    EXPECT_TRUE(state.ArmForPreparedGuestWork());
    EXPECT_TRUE(state.ConsumeGuestFrameTrigger());
    EXPECT_TRUE(state.IsCapturing());
    EXPECT_FALSE(state.ConsumeGuestFrameTrigger());
    EXPECT_TRUE(state.FinishPresentedFrameCapture());
    EXPECT_FALSE(state.IsCapturing());
    EXPECT_FALSE(state.FinishPresentedFrameCapture());
}

TEST(RenderDocCaptureState, TriggerIsPublishedAcrossThreads) {
    VideoCore::RenderDocCaptureState state;

    std::thread trigger_thread{[&state] { EXPECT_TRUE(state.Trigger()); }};
    trigger_thread.join();

    EXPECT_TRUE(state.ArmForPreparedGuestWork());
    EXPECT_TRUE(state.ConsumeGuestFrameTrigger());
}

TEST(RenderDocCaptureState, RepeatedTriggerDoesNotQueueAnotherCapture) {
    VideoCore::RenderDocCaptureState state;

    ASSERT_TRUE(state.Trigger());
    EXPECT_FALSE(state.Trigger());
    EXPECT_TRUE(state.ArmForPreparedGuestWork());
    EXPECT_FALSE(state.ArmForPreparedGuestWork());
    ASSERT_TRUE(state.ConsumeGuestFrameTrigger());
    EXPECT_FALSE(state.ConsumeGuestFrameTrigger());
    EXPECT_FALSE(state.Trigger());
    EXPECT_TRUE(state.FinishPresentedFrameCapture());
    EXPECT_TRUE(state.Trigger());
}

TEST(RenderDocCaptureState, BoundedGuestSubmissionBatchAlwaysTerminates) {
    VideoCore::RenderDocCaptureState state;

    // Large capture batches can exhaust driver resources in mature scenes before RenderDoc has
    // a chance to serialize the capture. Keep the diagnostic window deliberately narrow.
    EXPECT_LE(VideoCore::RenderDocCaptureState::GuestSubmissionLimit, 16u);

    ASSERT_TRUE(state.Trigger());
    ASSERT_TRUE(state.ArmForPreparedGuestWork());
    ASSERT_TRUE(state.ConsumeGuestFrameTrigger());
    for (unsigned submission = 1;
         submission < VideoCore::RenderDocCaptureState::GuestSubmissionLimit; ++submission) {
        EXPECT_FALSE(state.ShouldFinishAfterGuestSubmission());
    }
    EXPECT_TRUE(state.ShouldFinishAfterGuestSubmission());
}

TEST(RenderDocCaptureState, TransferOnlyCommandBuffersCannotConsumePendingCapture) {
    VideoCore::RenderDocCaptureState state;

    ASSERT_TRUE(state.Trigger());
    for (unsigned transfer_submission = 0; transfer_submission < 32; ++transfer_submission) {
        EXPECT_FALSE(state.ConsumeGuestFrameTrigger());
        EXPECT_FALSE(state.IsCapturing());
    }

    EXPECT_TRUE(state.ArmForPreparedGuestWork());
    EXPECT_TRUE(state.ConsumeGuestFrameTrigger());
    EXPECT_TRUE(state.IsCapturing());
}

TEST(RenderDocCaptureState, PreparedComputeCanArmCaptureBeforeDependentGraphics) {
    VideoCore::RenderDocCaptureState state;

    ASSERT_TRUE(state.Trigger());
    EXPECT_TRUE(state.ArmForPreparedGuestWork());
    EXPECT_TRUE(state.ConsumeGuestFrameTrigger());
    EXPECT_TRUE(state.IsCapturing());

    // The dependent graphics command buffer belongs to the same bounded capture rather than
    // consuming a second trigger.
    EXPECT_FALSE(state.ArmForPreparedGuestWork());
    EXPECT_FALSE(state.ConsumeGuestFrameTrigger());
}
