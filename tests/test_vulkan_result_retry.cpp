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

} // namespace
