// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/present_frame_ownership.h"

namespace Vulkan {
namespace {

struct TestFrame {
    int id;
};

TEST(PresentFrameOwnership, KeepsMostRecentPresentedFrameOutOfProducerQueue) {
    std::queue<TestFrame*> free_queue;
    TestFrame first{.id = 1};
    TestFrame* last_presented_frame{};

    const bool released = CompletePresentFrameOwnership(free_queue, last_presented_frame, &first,
                                                        false, true);

    EXPECT_EQ(last_presented_frame, &first);
    EXPECT_FALSE(released);
    EXPECT_TRUE(free_queue.empty());
}

TEST(PresentFrameOwnership, ReleasesPreviousFrameWhenANewFrameBecomesLast) {
    std::queue<TestFrame*> free_queue;
    TestFrame first{.id = 1};
    TestFrame second{.id = 2};
    TestFrame* last_presented_frame = &first;

    const bool released = CompletePresentFrameOwnership(free_queue, last_presented_frame, &second,
                                                        false, true);

    ASSERT_TRUE(released);
    ASSERT_EQ(free_queue.size(), 1);
    EXPECT_EQ(free_queue.front(), &first);
    EXPECT_EQ(last_presented_frame, &second);
}

TEST(PresentFrameOwnership, ReturnsSkippedFrameWithoutReplacingLastPresentedFrame) {
    std::queue<TestFrame*> free_queue;
    TestFrame first{.id = 1};
    TestFrame skipped{.id = 2};
    TestFrame* last_presented_frame = &first;

    const bool released = CompletePresentFrameOwnership(free_queue, last_presented_frame, &skipped,
                                                        false, false);

    ASSERT_TRUE(released);
    ASSERT_EQ(free_queue.size(), 1);
    EXPECT_EQ(free_queue.front(), &skipped);
    EXPECT_EQ(last_presented_frame, &first);
}

TEST(PresentFrameOwnership, LeavesReusedLastFrameReserved) {
    std::queue<TestFrame*> free_queue;
    TestFrame first{.id = 1};
    TestFrame* last_presented_frame = &first;

    const bool released = CompletePresentFrameOwnership(free_queue, last_presented_frame, &first,
                                                        true, true);

    EXPECT_FALSE(released);
    EXPECT_TRUE(free_queue.empty());
    EXPECT_EQ(last_presented_frame, &first);
}

} // namespace
} // namespace Vulkan
