// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "video_core/buffer_cache/buffer_access_interval_provenance.h"
#include "video_core/buffer_cache/buffer_access_provenance_trace.h"

namespace {

using Tracker = VideoCore::BasicBufferAccessIntervalProvenance<std::uint32_t>;
using Role = VideoCore::BufferAccessRole;

std::vector<nlohmann::json> ReadJsonLines(const std::filesystem::path& path) {
    std::ifstream input{path};
    std::vector<nlohmann::json> lines;
    for (std::string line; std::getline(input, line);) {
        lines.push_back(nlohmann::json::parse(line));
    }
    return lines;
}

TEST(BufferAccessIntervalProvenance, DisjointRolesKeepThePriorProducerAndOverlapAggregates) {
    Tracker tracker;

    tracker.BeginCommand(11, 50);
    tracker.Observe(7, 0, 128, Role::TransferWrite, 0x01, 0x10, true);
    ASSERT_EQ(tracker.CommitCommand().size(), 1);

    tracker.BeginCommand(12, 51);
    tracker.Observe(7, 0, 64, Role::ShaderReadWrite, 0x06, 0x20, true);
    tracker.Observe(7, 64, 64, Role::VertexRead, 0x08, 0x40, false);
    tracker.Observe(7, 64, 64, Role::IndexRead, 0x10, 0x80, false);

    const auto transitions = tracker.CommitCommand();
    ASSERT_EQ(transitions.size(), 2);

    const auto& shader = transitions[0];
    EXPECT_EQ(shader.offset, 0);
    EXPECT_EQ(shader.size, 64);
    ASSERT_TRUE(shader.prior.has_value());
    EXPECT_EQ(shader.prior->command_id, 11);
    EXPECT_EQ(shader.prior->tick, 50);
    EXPECT_EQ(shader.prior->roles, Role::TransferWrite);
    EXPECT_EQ(shader.observations.size(), 1);
    EXPECT_EQ(shader.current_roles, Role::ShaderReadWrite);

    const auto& geometry = transitions[1];
    EXPECT_EQ(geometry.offset, 64);
    EXPECT_EQ(geometry.size, 64);
    ASSERT_TRUE(geometry.prior.has_value());
    EXPECT_EQ(geometry.prior->command_id, 11);
    EXPECT_EQ(geometry.prior->tick, 50);
    EXPECT_EQ(geometry.prior->roles, Role::TransferWrite);
    EXPECT_EQ(geometry.observations.size(), 2);
    EXPECT_EQ(geometry.current_roles, Role::VertexRead | Role::IndexRead);
    EXPECT_EQ(geometry.current_access, 0x18);
    EXPECT_EQ(geometry.current_stages, 0xC0);
}

TEST(BufferAccessProvenanceTrace, ParsesOnlyExplicitBoundedRequests) {
    const auto valid =
        VideoCore::BufferAccessProvenanceTraceRequest::Parse("buffer-access.jsonl", "16", "1024");
    ASSERT_TRUE(valid.has_value());
    EXPECT_EQ(valid->record_limit, 16);
    EXPECT_EQ(valid->observation_limit, 1024);

    EXPECT_FALSE(VideoCore::BufferAccessProvenanceTraceRequest::Parse("", "16", "1024"));
    EXPECT_FALSE(
        VideoCore::BufferAccessProvenanceTraceRequest::Parse("buffer-access.jsonl", "0", "1024"));
    EXPECT_FALSE(VideoCore::BufferAccessProvenanceTraceRequest::Parse("buffer-access.jsonl", "16",
                                                                      "unbounded"));
}

TEST(BufferAccessProvenanceTrace, WritesExactIntervalsForMultiRoleCommandsWithinLimits) {
    const auto root =
        std::filesystem::temp_directory_path() /
        ("shadps4-buffer-access-provenance-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(root);
    const auto path = root / "trace.jsonl";

    {
        VideoCore::BasicBufferAccessProvenanceTrace<std::uint32_t> trace{{path, 2, 16}};
        trace.BeginCommand(11, 50);
        trace.Observe(7, 0, 128, Role::TransferWrite, 0x01, 0x10, true);
        trace.CommitCommand();

        trace.BeginCommand(12, 51);
        trace.Observe(7, 0, 64, Role::ShaderReadWrite, 0x06, 0x20, true);
        trace.Observe(7, 64, 64, Role::VertexRead, 0x08, 0x40, false);
        trace.Observe(7, 64, 64, Role::IndexRead, 0x10, 0x80, false);
        trace.CommitCommand();
    }

    const auto lines = ReadJsonLines(path);
    ASSERT_EQ(lines.size(), 3);
    EXPECT_EQ(lines[0].at("source"), "buffer_access_interval_provenance");

    EXPECT_EQ(lines[1].at("commandId"), 12);
    EXPECT_EQ(lines[1].at("tick"), 51);
    EXPECT_EQ(lines[1].at("bufferId"), 1);
    EXPECT_EQ(lines[1].at("offset"), 0);
    EXPECT_EQ(lines[1].at("size"), 64);
    EXPECT_EQ(lines[1].at("prior").at("commandId"), 11);
    EXPECT_EQ(lines[1].at("observations").size(), 1);

    EXPECT_EQ(lines[2].at("offset"), 64);
    EXPECT_EQ(lines[2].at("size"), 64);
    EXPECT_EQ(lines[2].at("prior").at("commandId"), 11);
    EXPECT_EQ(lines[2].at("observations").size(), 2);
    EXPECT_TRUE(lines[2].at("interesting").at("multiRole"));
    std::filesystem::remove_all(root);
}

} // namespace
