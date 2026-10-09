// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "video_core/renderer_vulkan/presented_frame_timing_trace.h"

namespace {

std::vector<std::string> ReadLines(const std::filesystem::path& path) {
    std::ifstream input{path};
    std::vector<std::string> lines;
    for (std::string line; std::getline(input, line);) {
        lines.emplace_back(std::move(line));
    }
    return lines;
}

} // namespace

TEST(PresentedFrameTimingTrace, ParsesBoundedWindow) {
    const auto request =
        Vulkan::PresentedFrameTimingTraceRequest::Parse("trace.jsonl", "100", "31");

    ASSERT_TRUE(request.has_value());
    EXPECT_EQ(request->start_frame, 100);
    EXPECT_EQ(request->sample_count, 31);
}

TEST(PresentedFrameTimingTrace, RejectsIncompleteOrUnboundedWindow) {
    EXPECT_FALSE(
        Vulkan::PresentedFrameTimingTraceRequest::Parse("", "1", "2").has_value());
    EXPECT_FALSE(
        Vulkan::PresentedFrameTimingTraceRequest::Parse("trace", "0", "2").has_value());
    EXPECT_FALSE(
        Vulkan::PresentedFrameTimingTraceRequest::Parse("trace", "1", "1").has_value());
    EXPECT_FALSE(Vulkan::PresentedFrameTimingTraceRequest::Parse("trace", "1", "1000002")
                     .has_value());
    EXPECT_FALSE(
        Vulkan::PresentedFrameTimingTraceRequest::Parse("trace", "one", "2").has_value());
}

TEST(PresentedFrameTimingTrace, WritesOnlyRequestedFramesAndFlushesFinalSample) {
    const auto root =
        std::filesystem::temp_directory_path() /
        ("shadps4-performance-trace-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(root);
    const auto path = root / "trace.jsonl";
    {
        Vulkan::PresentedFrameTimingTrace trace{{path, 10, 3}};
        trace.Record(9, 900);
        trace.Record(10, 1000);
        trace.Record(11, 1100);
        trace.Record(12, 1200);
        trace.Record(13, 1300);
    }

    const auto lines = ReadLines(path);
    ASSERT_EQ(lines.size(), 4);
    EXPECT_EQ(nlohmann::json::parse(lines[0]).at("source"), "presented_frame");
    EXPECT_EQ(nlohmann::json::parse(lines[1]).at("presentedFrame"), 10);
    EXPECT_EQ(nlohmann::json::parse(lines[2]).at("presentedFrame"), 11);
    EXPECT_EQ(nlohmann::json::parse(lines[3]).at("presentedFrame"), 12);
    std::filesystem::remove_all(root);
}
