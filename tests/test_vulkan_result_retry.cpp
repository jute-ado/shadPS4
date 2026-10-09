// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_result_retry.h"

namespace {

std::string ReadText(const char* path) {
    std::ifstream source{path, std::ios::binary};
    EXPECT_TRUE(source.is_open());
    std::string text{std::istreambuf_iterator<char>{source}, std::istreambuf_iterator<char>{}};
    std::erase(text, '\r');
    return text;
}

TEST(VulkanResultRetry, SuccessCompletesOnFirstAttempt) {
    unsigned attempts = 0;
    const auto result = Vulkan::RetryOnTimeout([&] {
        ++attempts;
        return vk::Result::eSuccess;
    });
    EXPECT_EQ(result, vk::Result::eSuccess);
    EXPECT_EQ(attempts, 1u);
}

TEST(VulkanResultRetry, TimeoutsContinueUntilSuccess) {
    unsigned attempts = 0;
    const auto result = Vulkan::RetryOnTimeout(
        [&] { return ++attempts < 3 ? vk::Result::eTimeout : vk::Result::eSuccess; });
    EXPECT_EQ(result, vk::Result::eSuccess);
    EXPECT_EQ(attempts, 3u);
}

TEST(VulkanResultRetry, PendingWorkHasNoFixedRetryLimit) {
    unsigned attempts = 0;
    const auto result = Vulkan::RetryOnTimeout(
        [&] { return ++attempts < 4096 ? vk::Result::eTimeout : vk::Result::eSuccess; });
    EXPECT_EQ(result, vk::Result::eSuccess);
    EXPECT_EQ(attempts, 4096u);
}

class VulkanTerminalResult : public testing::TestWithParam<vk::Result> {};

TEST_P(VulkanTerminalResult, FirstTerminalResultIsReturnedWithoutRetry) {
    unsigned attempts = 0;
    const auto result =
        Vulkan::RetryOnTimeout([&] { return ++attempts == 1 ? GetParam() : vk::Result::eSuccess; });
    EXPECT_EQ(result, GetParam());
    EXPECT_EQ(attempts, 1u);
}

TEST_P(VulkanTerminalResult, TimeoutDoesNotHideSubsequentTerminalResult) {
    unsigned attempts = 0;
    const auto result = Vulkan::RetryOnTimeout([&] {
        if (++attempts == 1) {
            return vk::Result::eTimeout;
        }
        return attempts == 2 ? GetParam() : vk::Result::eSuccess;
    });
    EXPECT_EQ(result, GetParam());
    EXPECT_EQ(attempts, 2u);
}

INSTANTIATE_TEST_SUITE_P(NonTimeout, VulkanTerminalResult,
                         testing::Values(vk::Result::eErrorDeviceLost,
                                         vk::Result::eErrorOutOfDeviceMemory,
                                         vk::Result::eErrorOutOfHostMemory,
                                         vk::Result::eErrorUnknown, vk::Result::eNotReady,
                                         vk::Result::eIncomplete));

TEST(VulkanResultHandlingWiring, MasterWaitChecksTerminalResultBeforeRefreshing) {
    const auto text = ReadText(SHADPS4_MASTER_SEMAPHORE_SOURCE_PATH);
    const auto wait = text.find("void MasterSemaphore::Wait(u64 tick)");
    ASSERT_NE(wait, std::string::npos);
    const auto retry = text.find("RetryOnTimeout(", wait);
    ASSERT_NE(retry, std::string::npos);
    const auto check = text.find("Check<\"wait for master semaphore\">(result);", retry);
    ASSERT_NE(check, std::string::npos);
    const auto refresh = text.find("Refresh();", retry);
    ASSERT_NE(refresh, std::string::npos);
    EXPECT_LT(check, refresh);
}

TEST(VulkanResultHandlingWiring, MasterRefreshChecksCounterResultBeforeCasRetry) {
    const auto text = ReadText(SHADPS4_MASTER_SEMAPHORE_SOURCE_PATH);
    const auto refresh = text.find("void MasterSemaphore::Refresh()");
    ASSERT_NE(refresh, std::string::npos);
    const auto counter_result = text.find("counter_result", refresh);
    ASSERT_NE(counter_result, std::string::npos);
    const auto terminal_check = text.find("ASSERT_MSG(counter_result == vk::Result::eSuccess",
                                                counter_result);
    ASSERT_NE(terminal_check, std::string::npos);
    const auto cas_retry = text.find("compare_exchange_weak", terminal_check);
    ASSERT_NE(cas_retry, std::string::npos);
    EXPECT_LT(terminal_check, cas_retry);
}

TEST(VulkanResultHandlingWiring, SubmissionChecksEveryResultBeforeAdvancing) {
    const auto text = ReadText(SHADPS4_SCHEDULER_SOURCE_PATH);
    const auto submit = text.find("auto submit_result = instance.GetGraphicsQueue().submit(");
    ASSERT_NE(submit, std::string::npos);
    const auto check = text.find("Check<\"submit GPU command buffer\">(submit_result);", submit);
    ASSERT_NE(check, std::string::npos);
    const auto refresh = text.find("master_semaphore.Refresh();", submit);
    ASSERT_NE(refresh, std::string::npos);
    EXPECT_LT(check, refresh);
}

TEST(VulkanResultHandlingWiring, PresenterFrameWaitChecksTerminalResultBeforeRecreating) {
    const auto text = ReadText(SHADPS4_PRESENTER_SOURCE_PATH);
    const auto get_frame = text.find("Frame* Presenter::GetRenderFrame()");
    ASSERT_NE(get_frame, std::string::npos);
    const auto retry = text.find("RetryOnTimeout(", get_frame);
    ASSERT_NE(retry, std::string::npos);
    const auto fence_wait = text.find("waitForFences(", retry);
    ASSERT_NE(fence_wait, std::string::npos);
    const auto check = text.find("Check<\"wait for present frame\">(result);", fence_wait);
    ASSERT_NE(check, std::string::npos);
    const auto recreate = text.find("RecreateFrame(", check);
    ASSERT_NE(recreate, std::string::npos);
    EXPECT_LT(fence_wait, check);
    EXPECT_LT(check, recreate);
}

TEST(VulkanResultHandlingWiring, PresenterLastFrameWaitChecksTerminalResultBeforeReturning) {
    const auto text = ReadText(SHADPS4_PRESENTER_SOURCE_PATH);
    const auto prepare_last = text.find("Frame* Presenter::PrepareLastFrame()");
    ASSERT_NE(prepare_last, std::string::npos);
    const auto retry = text.find("RetryOnTimeout(", prepare_last);
    ASSERT_NE(retry, std::string::npos);
    const auto fence_wait = text.find("waitForFences(", retry);
    ASSERT_NE(fence_wait, std::string::npos);
    const auto check = text.find("Check<\"wait for present frame\">(result);", fence_wait);
    ASSERT_NE(check, std::string::npos);
    const auto return_frame = text.find("return frame;", check);
    ASSERT_NE(return_frame, std::string::npos);
    EXPECT_LT(fence_wait, check);
    EXPECT_LT(check, return_frame);
}

TEST(VulkanResultHandlingWiring, PresenterResetFenceChecksTerminalResultBeforeRecording) {
    const auto text = ReadText(SHADPS4_PRESENTER_SOURCE_PATH);
    const auto present = text.find("void Presenter::Present(Frame* frame");
    ASSERT_NE(present, std::string::npos);
    const auto reset = text.find("resetFences(frame->present_done)", present);
    ASSERT_NE(reset, std::string::npos);
    const auto check = text.find("ASSERT_MSG(reset_result == vk::Result::eSuccess", reset);
    ASSERT_NE(check, std::string::npos);
    const auto acquire = text.find("ImGuiID dockId", check);
    ASSERT_NE(acquire, std::string::npos);
    EXPECT_LT(check, acquire);
}

} // namespace
