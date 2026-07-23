// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/screenshot_request_queue.h"

namespace VideoCore {
namespace {

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

} // namespace
} // namespace VideoCore
