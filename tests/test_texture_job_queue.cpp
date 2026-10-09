// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <future>

#include <gtest/gtest.h>

#include "imgui/renderer/texture_job_queue.h"

using namespace std::chrono_literals;

TEST(TextureJobQueue, DeliversWorkQueuedBeforeConsumerWaits) {
    ImGui::Core::TextureManager::JobQueue<int> queue;
    queue.Start();

    ASSERT_TRUE(queue.Push(42));
    auto result = std::async(std::launch::async, [&queue] { return queue.WaitPop(); });

    const auto status = result.wait_for(1s);
    queue.Stop();

    ASSERT_EQ(status, std::future_status::ready);
    const auto job = result.get();
    ASSERT_TRUE(job.has_value());
    EXPECT_EQ(*job, 42);
}

TEST(TextureJobQueue, StopUnblocksAnIdleConsumer) {
    ImGui::Core::TextureManager::JobQueue<int> queue;
    queue.Start();
    auto result = std::async(std::launch::async, [&queue] { return queue.WaitPop(); });

    queue.Stop();

    ASSERT_EQ(result.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(result.get(), std::nullopt);
}

TEST(TextureJobQueue, StopDrainsQueuedWorkBeforeClosing) {
    ImGui::Core::TextureManager::JobQueue<int> queue;
    queue.Start();
    ASSERT_TRUE(queue.Push(7));
    queue.Stop();

    ASSERT_EQ(queue.WaitPop(), 7);
    EXPECT_EQ(queue.WaitPop(), std::nullopt);
    EXPECT_FALSE(queue.Push(8));
}
