// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <future>

#include <gtest/gtest.h>

#include "video_core/screenshot_request_queue.h"
#include "video_core/screenshot_writer_queue.h"

namespace VideoCore {
namespace {

using namespace std::chrono_literals;

TEST(ScreenshotRequestQueue, KeepsAutomatedCapturesSeparateFromUserNotifications) {
    ScreenshotRequestQueue queue;

    queue.Push(ScreenshotRequest::WithOverlays, ScreenshotRequestOrigin::User);
    queue.Push(ScreenshotRequest::WithOverlays, ScreenshotRequestOrigin::Automation);
    queue.Push(ScreenshotRequest::WithOverlays, ScreenshotRequestOrigin::Automation);

    const auto requests = queue.ConsumeWithOverlays();

    EXPECT_EQ(requests.notifying_count, 1);
    EXPECT_EQ(requests.silent_count, 2);
}

TEST(ScreenshotRequestQueue, ConsumingOneKindDoesNotConsumeTheOther) {
    ScreenshotRequestQueue queue;

    queue.Push(ScreenshotRequest::GameOnly, ScreenshotRequestOrigin::Automation);
    queue.Push(ScreenshotRequest::WithOverlays, ScreenshotRequestOrigin::User);

    const auto game_only = queue.ConsumeGameOnly();
    const auto with_overlays = queue.ConsumeWithOverlays();

    EXPECT_EQ(game_only.notifying_count, 0);
    EXPECT_EQ(game_only.silent_count, 1);
    EXPECT_EQ(with_overlays.notifying_count, 1);
    EXPECT_EQ(with_overlays.silent_count, 0);
}

TEST(ScreenshotRequestQueue, ConsumeClearsPendingRequests) {
    ScreenshotRequestQueue queue;
    queue.Push(ScreenshotRequest::GameOnly, ScreenshotRequestOrigin::User);

    EXPECT_EQ(queue.ConsumeGameOnly().notifying_count, 1);
    const auto empty = queue.ConsumeGameOnly();
    EXPECT_EQ(empty.notifying_count, 0);
    EXPECT_EQ(empty.silent_count, 0);
}

TEST(ScreenshotWriterQueue, WaitIdleTracksWorkUntilTheWriterCompletesIt) {
    ScreenshotWriterQueue<int> queue;
    queue.Start();
    ASSERT_TRUE(queue.Push(42));
    ASSERT_EQ(queue.WaitPop(), 42);

    auto idle = std::async(std::launch::async, [&queue] { queue.WaitIdle(); });
    EXPECT_EQ(idle.wait_for(20ms), std::future_status::timeout);

    queue.Complete();

    EXPECT_EQ(idle.wait_for(1s), std::future_status::ready);
    queue.Stop();
}

TEST(ScreenshotWriterQueue, WaitIdleIncludesGpuCompletionsReservedBeforeWriterSubmission) {
    ScreenshotWriterQueue<int> queue;
    queue.Start();
    ASSERT_TRUE(queue.Reserve());

    auto idle = std::async(std::launch::async, [&queue] { queue.WaitIdle(); });
    EXPECT_EQ(idle.wait_for(20ms), std::future_status::timeout);

    queue.PushReserved(7);
    ASSERT_EQ(queue.WaitPop(), 7);
    queue.Complete();

    EXPECT_EQ(idle.wait_for(1s), std::future_status::ready);
    queue.Stop();
}

} // namespace
} // namespace VideoCore
