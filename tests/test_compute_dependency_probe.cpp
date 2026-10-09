// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/compute_dependency_probe.h"

TEST(ComputeDependencyProbe, OnlyExactOneEnables) {
    EXPECT_TRUE(Vulkan::IsComputeDependencyProbeEnabled("1"));
    EXPECT_FALSE(Vulkan::IsComputeDependencyProbeEnabled(nullptr));
    for (const char* value : {"", "0", "true", "01", "1 ", " 1", "1\n", "2"}) {
        EXPECT_FALSE(Vulkan::IsComputeDependencyProbeEnabled(value));
    }
}

TEST(ComputeDependencyProbe, DisabledDirectAndIndirectOnlyEmitCommand) {
    Vulkan::ComputeDependencyProbe probe{false};
    int commands = 0;
    int barriers = 0;
    int logs = 0;
    for (const bool indirect : {false, true}) {
        SCOPED_TRACE(indirect);
        probe.Dispatch([&] { ++commands; }, [&](const auto&) { ++barriers; }, [&] { ++logs; });
    }
    EXPECT_EQ(commands, 2);
    EXPECT_EQ(barriers, 0);
    EXPECT_EQ(logs, 0);
}

TEST(ComputeDependencyProbe, BothCommandsEmitDependencyAfterActualDispatch) {
    Vulkan::ComputeDependencyProbe probe{true};
    std::vector<std::string> calls;
    for (const bool indirect : {false, true}) {
        probe.Dispatch(
            [&] { calls.emplace_back(indirect ? "indirect" : "direct"); },
            [&](const vk::MemoryBarrier2& barrier) {
                EXPECT_EQ(calls.back(), indirect ? "indirect" : "direct");
                EXPECT_EQ(barrier.srcStageMask, vk::PipelineStageFlagBits2::eAllCommands);
                EXPECT_EQ(barrier.dstStageMask, vk::PipelineStageFlagBits2::eAllCommands);
                const auto rw = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
                EXPECT_EQ(barrier.srcAccessMask, rw);
                EXPECT_EQ(barrier.dstAccessMask, rw);
                calls.emplace_back("barrier");
            },
            [&] { calls.emplace_back("log"); });
    }
    EXPECT_EQ(calls, (std::vector<std::string>{"direct", "barrier", "log", "indirect", "barrier"}));
}

TEST(ComputeDependencyProbe, FailedCommandDoesNotPublishDependencyOrLog) {
    Vulkan::ComputeDependencyProbe probe{true};
    int barriers = 0;
    int logs = 0;
    EXPECT_THROW(probe.Dispatch([] { throw std::runtime_error("dispatch"); },
                                [&](const auto&) { ++barriers; }, [&] { ++logs; }),
                 std::runtime_error);
    EXPECT_EQ(barriers, 0);
    EXPECT_EQ(logs, 0);
    probe.Dispatch([] {}, [&](const auto&) { ++barriers; }, [&] { ++logs; });
    EXPECT_EQ(barriers, 1);
    EXPECT_EQ(logs, 1);
}

TEST(ComputeDependencyProbe, EnabledLoggingIsBoundedAcrossRepeatedCommands) {
    Vulkan::ComputeDependencyProbe probe{true};
    int barriers = 0;
    int logs = 0;
    for (int i = 0; i < 100; ++i) {
        probe.Dispatch([] {}, [&](const auto&) { ++barriers; }, [&] { ++logs; });
    }
    EXPECT_EQ(barriers, 100);
    EXPECT_EQ(logs, 1);
}

TEST(ComputeDependencyProbe, LoggerFailureCannotInterruptCommandsOrRetryLogging) {
    Vulkan::ComputeDependencyProbe probe{true};
    int commands = 0;
    int barriers = 0;
    int logs = 0;
    for (int i = 0; i < 2; ++i) {
        EXPECT_NO_THROW(probe.Dispatch([&] { ++commands; }, [&](const auto&) { ++barriers; }, [&] {
            ++logs;
            throw std::runtime_error("logger");
        }));
    }
    EXPECT_EQ(commands, 2);
    EXPECT_EQ(barriers, 2);
    EXPECT_EQ(logs, 1);
}

TEST(ComputeDependencyProbe, DependencyFailureStillPropagatesWithoutLogging) {
    Vulkan::ComputeDependencyProbe probe{true};
    int commands = 0;
    int logs = 0;
    EXPECT_THROW(probe.Dispatch([&] { ++commands; },
                                [](const auto&) { throw std::runtime_error("dependency"); },
                                [&] { ++logs; }),
                 std::runtime_error);
    EXPECT_EQ(commands, 1);
    EXPECT_EQ(logs, 0);
}

TEST(ComputeDependencyProbe, ActualDirectAndIndirectHooksFollowEndRenderingAndWrapCommands) {
    std::ifstream file{std::string{COMPUTE_PROBE_SOURCE_ROOT} +
                       "/src/video_core/renderer_vulkan/vk_rasterizer.cpp"};
    const std::string source{std::istreambuf_iterator<char>{file}, {}};
    ASSERT_FALSE(source.empty());
    for (const auto& [name, command] :
         {std::pair{"DispatchDirect", "cmdbuf.dispatch("},
          std::pair{"DispatchIndirect", "cmdbuf.dispatchIndirect("}}) {
        SCOPED_TRACE(name);
        const auto start = source.find(std::string{"void Rasterizer::"} + name + "(");
        ASSERT_NE(start, std::string::npos);
        const auto finish = source.find("\n}", start);
        ASSERT_NE(finish, std::string::npos);
        const auto body = source.substr(start, finish - start);
        const auto end_rendering = body.find("scheduler.EndRendering();");
        const auto wrapper = body.find("GetComputeDependencyProbe().Dispatch(");
        const auto emit = body.find(command);
        const auto barrier = body.find("cmdbuf.pipelineBarrier2(");
        const auto mark = body.find("scheduler.MarkGuestWork(");
        ASSERT_NE(end_rendering, std::string::npos);
        ASSERT_NE(wrapper, std::string::npos);
        ASSERT_NE(emit, std::string::npos);
        ASSERT_NE(barrier, std::string::npos);
        ASSERT_NE(mark, std::string::npos);
        EXPECT_LT(end_rendering, wrapper);
        EXPECT_LT(wrapper, emit);
        EXPECT_LT(emit, barrier);
        EXPECT_LT(barrier, mark);
        EXPECT_EQ(body.find("scheduler.EndRendering();", end_rendering + 1), std::string::npos);
        EXPECT_EQ(body.find("scheduler.BeginRendering("), std::string::npos);
        EXPECT_EQ(body.find(command, emit + 1), std::string::npos);
    }
}
