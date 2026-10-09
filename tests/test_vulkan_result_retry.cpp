// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/submit_wait_diagnostic.h"
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
    const auto budget = text.find("instance.LogDeviceMemoryBudget(\"semaphore_wait\", result);",
                                 retry);
    ASSERT_NE(budget, std::string::npos);
    const auto refresh = text.find("Refresh();", retry);
    ASSERT_NE(refresh, std::string::npos);
    EXPECT_LT(retry, budget);
    EXPECT_LT(budget, check);
    EXPECT_LT(check, refresh);
}

TEST(VulkanResultHandlingWiring, MasterRefreshChecksCounterResultBeforeCasRetry) {
    const auto text = ReadText(SHADPS4_MASTER_SEMAPHORE_SOURCE_PATH);
    const auto refresh = text.find("void MasterSemaphore::Refresh()");
    ASSERT_NE(refresh, std::string::npos);
    const auto counter_result = text.find("counter_result", refresh);
    ASSERT_NE(counter_result, std::string::npos);
    const auto budget = text.find(
        "instance.LogDeviceMemoryBudget(\"semaphore_counter\", counter_result);", counter_result);
    ASSERT_NE(budget, std::string::npos);
    const auto terminal_check = text.find("ASSERT_MSG(counter_result == vk::Result::eSuccess",
                                                counter_result);
    ASSERT_NE(terminal_check, std::string::npos);
    const auto cas_retry = text.find("compare_exchange_weak", terminal_check);
    ASSERT_NE(cas_retry, std::string::npos);
    EXPECT_LT(counter_result, budget);
    EXPECT_LT(budget, terminal_check);
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

TEST(SubmitWaitDiagnostic, StrictLaunchParser) {
    EXPECT_FALSE(Vulkan::ParseSubmitWaitDiagnostic(""));
    EXPECT_FALSE(Vulkan::ParseSubmitWaitDiagnostic("0"));
    EXPECT_FALSE(Vulkan::ParseSubmitWaitDiagnostic("true"));
    EXPECT_FALSE(Vulkan::ParseSubmitWaitDiagnostic("01"));
    EXPECT_FALSE(Vulkan::ParseSubmitWaitDiagnostic(" 1"));
    EXPECT_FALSE(Vulkan::ParseSubmitWaitDiagnostic("1 "));
    EXPECT_TRUE(Vulkan::ParseSubmitWaitDiagnostic("1"));
}

TEST(SubmitWaitDiagnosticWiring, SchedulerLogsResultAndTimelineTicksAfterSubmit) {
    const auto text = ReadText(SHADPS4_SCHEDULER_SOURCE_PATH);
    const auto submit = text.find("auto submit_result = instance.GetGraphicsQueue().submit(");
    ASSERT_NE(submit, std::string::npos);
    const auto result = text.find("submit_wait submit_result", submit);
    const auto refresh = text.find("master_semaphore.Refresh();", submit);
    const auto end = text.find("submit_wait submit_end", submit);
    const auto known = text.find("KnownGpuTick()", refresh);
    ASSERT_NE(result, std::string::npos);
    ASSERT_NE(refresh, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    ASSERT_NE(known, std::string::npos);
    EXPECT_LT(submit, result);
    EXPECT_LT(result, refresh);
    EXPECT_LT(refresh, end);
    EXPECT_NE(text.substr(submit, end - submit).find("signal_value"), std::string::npos);
    EXPECT_LT(refresh, known);
}

TEST(SubmitWaitDiagnosticWiring, MasterSemaphoreLogsWaitTargetAndResult) {
    const auto text = ReadText(SHADPS4_MASTER_SEMAPHORE_SOURCE_PATH);
    const auto wait = text.find("void MasterSemaphore::Wait(u64 tick)");
    ASSERT_NE(wait, std::string::npos);
    const auto begin = text.find("submit_wait wait_begin", wait);
    const auto result = text.find("submit_wait wait_result", wait);
    const auto end = text.find("submit_wait wait_end", result);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(result, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    EXPECT_LT(begin, result);
    EXPECT_LT(result, end);
    EXPECT_NE(text.substr(wait, end - wait).find("target_tick"), std::string::npos);
    EXPECT_NE(text.substr(wait, end - wait).find("KnownGpuTick"), std::string::npos);
}

TEST(SubmitWaitDiagnosticWiring, SchedulerKeepsHighVolumeTraceOptIn) {
    const auto text = ReadText(SHADPS4_SCHEDULER_SOURCE_PATH);
    const auto submit = text.find("u64 Scheduler::SubmitExecutionWithSubmitLockHeld(");
    ASSERT_NE(submit, std::string::npos);
    const auto opt_in = text.find("const bool trace_submit = SubmitWaitDiagnosticEnabled();", submit);
    const auto first_guard = text.find("if (trace_submit)", submit);
    const auto result = text.find("submit_wait submit_result", submit);
    const auto end = text.find("submit_wait submit_end", submit);
    ASSERT_NE(opt_in, std::string::npos);
    ASSERT_NE(first_guard, std::string::npos);
    ASSERT_NE(result, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    EXPECT_LT(opt_in, first_guard);
    EXPECT_LT(first_guard, result);
    EXPECT_LT(result, end);
    EXPECT_LT(text.rfind("if (trace_submit)", result), result);
    EXPECT_LT(text.rfind("if (trace_submit)", end), end);
}

TEST(SubmitWaitDiagnosticWiring, MasterSemaphoreKeepsHighVolumeTraceOptIn) {
    const auto text = ReadText(SHADPS4_MASTER_SEMAPHORE_SOURCE_PATH);
    const auto wait = text.find("void MasterSemaphore::Wait(u64 tick)");
    ASSERT_NE(wait, std::string::npos);
    const auto opt_in = text.find("const bool trace_wait = SubmitWaitDiagnosticEnabled();", wait);
    const auto first_guard = text.find("if (trace_wait)", wait);
    const auto begin = text.find("submit_wait wait_begin", wait);
    const auto result = text.find("submit_wait wait_result", wait);
    const auto end = text.find("submit_wait wait_end", result);
    ASSERT_NE(opt_in, std::string::npos);
    ASSERT_NE(first_guard, std::string::npos);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(result, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    EXPECT_LT(opt_in, first_guard);
    EXPECT_LT(first_guard, begin);
    EXPECT_LT(begin, result);
    EXPECT_LT(result, end);
    EXPECT_LT(text.rfind("if (trace_wait)", result), result);
    EXPECT_LT(text.rfind("if (trace_wait)", end), end);
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
