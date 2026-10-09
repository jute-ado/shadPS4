// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/emitted_color_command_ledger.h"
#include "video_core/emitted_color_vertex_receipt.h"

namespace {
using namespace VideoCore;

struct Harness {
    EmittedColorCommandLedger ledger{{true, true, 1000, 100}};
    EmittedColorCommand command{
        .key = {.vertex_hash = 1,
                .fragment_hash = 2,
                .kind = ColorCommandKind::Indexed,
                .colors = {{{true, 98, 1920, 1080, 1}}}},
        .elements = 42,
        .instances = 1,
    };
    std::size_t builds{};
    std::vector<EmittedColorCommand> rows;
    std::vector<std::uint64_t> timestamps;
    std::vector<ColorLedgerSummary> ends;
    bool has_color{true};

    void Observe(std::uint64_t elapsed) {
        ledger.Observe(
            elapsed,
            [&]() -> std::optional<EmittedColorCommand> {
                ++builds;
                return has_color ? std::optional{command} : std::nullopt;
            },
            [&](std::size_t ordinal, std::uint64_t time, const EmittedColorCommand& value) {
                EXPECT_EQ(ordinal, rows.size());
                timestamps.push_back(time);
                rows.push_back(value);
            },
            [&](const ColorLedgerSummary& end) { ends.push_back(end); });
    }

    void Close(std::uint64_t elapsed) {
        ledger.Close(elapsed, [&](const ColorLedgerSummary& end) { ends.push_back(end); });
    }
};

TEST(EmittedColorCommandLedger, DefaultAndExplicitOffDoNotReadIrrelevantWindowPolicy) {
    for (const auto enabled :
         {std::optional<std::string_view>{}, std::optional<std::string_view>{"0"}}) {
        const auto policy = ColorCommandLedgerPolicy::Parse(enabled, "garbage", "garbage");
        EXPECT_TRUE(policy.valid);
        EXPECT_FALSE(policy.enabled);
    }
}

TEST(EmittedColorCommandLedger, ExplicitEnableRequiresStrictBoundedWindow) {
    const auto low = ColorCommandLedgerPolicy::Parse("1", "0", "1");
    EXPECT_TRUE(low.enabled);
    EXPECT_TRUE(low.valid);
    EXPECT_EQ(low.start_ms, 0);
    EXPECT_EQ(low.duration_ms, 1);
    const auto high = ColorCommandLedgerPolicy::Parse("1", "600000", "5000");
    EXPECT_TRUE(high.enabled);
    EXPECT_EQ(high.start_ms, 600000);
    EXPECT_EQ(high.duration_ms, 5000);
}

TEST(EmittedColorCommandLedger, MalformedExplicitPolicyNeverBroadensWindow) {
    for (const auto value : {"", "true", "01", " 1", "1 "}) {
        const auto p = ColorCommandLedgerPolicy::Parse(value, "0", "1");
        EXPECT_FALSE(p.valid) << value;
        EXPECT_FALSE(p.enabled) << value;
    }
    for (const auto value : {"", "+1", "-1", " 1", "1 ", "1ms", "18446744073709551616"}) {
        EXPECT_FALSE(ColorCommandLedgerPolicy::Parse("1", value, "1").valid) << value;
        EXPECT_FALSE(ColorCommandLedgerPolicy::Parse("1", "0", value).valid) << value;
    }
    EXPECT_FALSE(ColorCommandLedgerPolicy::Parse("1", std::nullopt, "1").valid);
    EXPECT_FALSE(ColorCommandLedgerPolicy::Parse("1", "0", std::nullopt).valid);
    EXPECT_FALSE(ColorCommandLedgerPolicy::Parse("1", "600001", "1").valid);
    EXPECT_FALSE(ColorCommandLedgerPolicy::Parse("1", "0", "0").valid);
    EXPECT_FALSE(ColorCommandLedgerPolicy::Parse("1", "0", "5001").valid);
}

TEST(EmittedColorCommandLedger, DisabledNeverBuildsLogsOrCloses) {
    Harness h;
    h.ledger = EmittedColorCommandLedger{{}};
    h.Observe(1001);
    h.Close(1002);
    EXPECT_EQ(h.builds, 0);
    EXPECT_TRUE(h.rows.empty());
    EXPECT_TRUE(h.ends.empty());
}

TEST(EmittedColorCommandLedger, WindowIsStartInclusiveEndExclusiveAndClosesOnce) {
    Harness h;
    h.Observe(999);
    EXPECT_EQ(h.builds, 0);
    h.Observe(1000);
    EXPECT_EQ(h.rows.size(), 1);
    EXPECT_EQ(h.timestamps, (std::vector<std::uint64_t>{1000}));
    h.Observe(1100);
    h.Observe(1101);
    h.Close(1102);
    ASSERT_EQ(h.ends.size(), 1);
    EXPECT_EQ(h.ends[0].reason, ColorLedgerEnd::Window);
    EXPECT_EQ(h.ends[0].unique, 1);
    EXPECT_EQ(h.builds, 1);
}

TEST(EmittedColorCommandLedger, ChangingCountsDoesNotConsumeAnotherPipelineKey) {
    Harness h;
    h.Observe(1001);
    h.command.elements = 999;
    h.command.instances = 2;
    h.Observe(1002);
    h.Close(1003);
    ASSERT_EQ(h.rows.size(), 1);
    EXPECT_EQ(h.rows[0].elements, 42);
    ASSERT_EQ(h.ends.size(), 1);
    EXPECT_EQ(h.ends[0].observed, 2);
    EXPECT_EQ(h.ends[0].duplicates, 1);
}

TEST(EmittedColorCommandLedger, ShaderKindAndBoundTargetMetadataArePartOfKey) {
    Harness h;
    h.Observe(1000);
    ++h.command.key.vertex_hash;
    h.Observe(1001);
    ++h.command.key.fragment_hash;
    h.Observe(1002);
    h.command.key.kind = ColorCommandKind::Direct;
    h.Observe(1003);
    ++h.command.key.colors[0].format;
    h.Observe(1004);
    ++h.command.key.colors[0].width;
    h.Observe(1005);
    ++h.command.key.colors[0].height;
    h.Observe(1006);
    ++h.command.key.colors[0].layers;
    h.Observe(1007);
    h.command.key.colors[1] = h.command.key.colors[0];
    h.Observe(1008);
    h.command.key.has_fragment = !h.command.key.has_fragment;
    h.Observe(1009);
    EXPECT_EQ(h.rows.size(), 10);
}

TEST(EmittedColorCommandLedger, SameShaderPairAndTargetsWithDifferentPipelineKeyRemainDistinct) {
    Harness h;
    h.command.key.pipeline_key_hash = 0x1234;
    h.Observe(1000);
    h.command.key.pipeline_key_hash = 0x5678;
    h.Observe(1001);
    ASSERT_EQ(h.rows.size(), 2);
    EXPECT_EQ(h.rows[0].key.pipeline_key_hash, 0x1234);
    EXPECT_EQ(h.rows[1].key.pipeline_key_hash, 0x5678);
}

TEST(EmittedColorCommandLedger, ThirtyThirdDistinctKeyTruncatesAndStopsBuilding) {
    Harness h;
    for (std::size_t i = 0; i < 33; ++i) {
        h.command.key.vertex_hash = i;
        h.Observe(1000 + i);
    }
    ASSERT_EQ(h.rows.size(), 32);
    ASSERT_EQ(h.ends.size(), 1);
    EXPECT_EQ(h.ends[0].reason, ColorLedgerEnd::Capacity);
    EXPECT_EQ(h.ends[0].observed, 33);
    EXPECT_EQ(h.ends[0].unique, 32);
    h.Observe(1099);
    h.Close(1100);
    EXPECT_EQ(h.builds, 33);
    EXPECT_EQ(h.ends.size(), 1);
}

TEST(EmittedColorCommandLedger, DuplicateAtCapacityDoesNotFalselyTruncate) {
    Harness h;
    for (std::size_t i = 0; i < 32; ++i) {
        h.command.key.vertex_hash = i;
        h.Observe(1000 + i);
    }
    h.Observe(1032);
    EXPECT_TRUE(h.ends.empty());
    h.Close(1033);
    ASSERT_EQ(h.ends.size(), 1);
    EXPECT_EQ(h.ends[0].reason, ColorLedgerEnd::Teardown);
    EXPECT_EQ(h.ends[0].duplicates, 1);
}

TEST(EmittedColorCommandLedger, NonColorDoesNotCreateARecord) {
    Harness h;
    h.has_color = false;
    h.Observe(1000);
    h.Close(1001);
    EXPECT_TRUE(h.rows.empty());
    ASSERT_EQ(h.ends.size(), 1);
    EXPECT_EQ(h.ends[0].observed, 0);
}

TEST(EmittedColorCommandLedger, IndirectMetadataDoesNotPretendToBeElementsOrInstances) {
    Harness h;
    h.command = {.key = h.command.key, .command_count = 3, .stride = 20};
    h.command.key.kind = ColorCommandKind::IndexedIndirect;
    h.Observe(1000);
    h.command.key.kind = ColorCommandKind::IndexedIndirectCount;
    h.command.command_count = 0;
    h.command.max_command_count = 17;
    h.Observe(1001);
    ASSERT_EQ(h.rows.size(), 2);
    EXPECT_EQ(h.rows[0].elements, 0);
    EXPECT_EQ(h.rows[0].instances, 0);
    EXPECT_EQ(h.rows[0].command_count, 3);
    EXPECT_EQ(h.rows[1].command_count, 0);
    EXPECT_EQ(h.rows[1].max_command_count, 17);
}

TEST(EmittedColorCommandLedger, EarlyTeardownIsExplicitNotCompletedWindow) {
    Harness h;
    h.Close(500);
    h.Close(501);
    ASSERT_EQ(h.ends.size(), 1);
    EXPECT_EQ(h.ends[0].reason, ColorLedgerEnd::Teardown);
    EXPECT_EQ(h.ends[0].elapsed_ms, 500);
    EXPECT_EQ(h.ends[0].observed, 0);
}

TEST(EmittedColorCommandLedger, CloseFailureCannotReportAgainAndDisableStopsCallbacks) {
    Harness h;
    EXPECT_THROW(h.ledger.Close(1000, [](const ColorLedgerSummary&) { throw 7; }), int);
    h.Observe(1001);
    h.Close(1002);
    EXPECT_EQ(h.builds, 0);
    EXPECT_TRUE(h.ends.empty());
    Harness disabled;
    disabled.ledger.Disable();
    disabled.Observe(1000);
    disabled.Close(1001);
    EXPECT_EQ(disabled.builds, 0);
    EXPECT_TRUE(disabled.ends.empty());
}

TEST(EmittedColorVertexReceipt, UnsetIsOffAndMalformedConfigurationFailsClosed) {
    const auto unset = EmittedColorVertexReceipt::Parse(std::nullopt);
    EXPECT_TRUE(unset.Valid());
    EXPECT_FALSE(unset.Enabled());

    for (const auto value : {"", "0x", "xyz", "10000000000000000"}) {
        const auto invalid = EmittedColorVertexReceipt::Parse(value);
        EXPECT_FALSE(invalid.Valid()) << value;
        EXPECT_FALSE(invalid.Enabled()) << value;
    }
}

TEST(EmittedColorVertexReceipt, ExactColorDrawMatchesOnce) {
    auto receipt = EmittedColorVertexReceipt::Parse("0xddd376ab");
    ASSERT_TRUE(receipt.Valid());
    ASSERT_TRUE(receipt.Enabled());
    EXPECT_FALSE(receipt.Observe(0xddd376ab, false));
    EXPECT_FALSE(receipt.Observe(0x1234, true));
    EXPECT_TRUE(receipt.Observe(0xddd376ab, true));
    EXPECT_FALSE(receipt.Enabled());
    EXPECT_FALSE(receipt.Observe(0xddd376ab, true));
}

TEST(EmittedColorCommandLedger, ProductionHooksArePostEmissionAndUseOwnedBoundMetadata) {
    std::ifstream stream{SHADPS4_RASTERIZER_SOURCE_PATH};
    ASSERT_TRUE(stream.good());
    const std::string source{std::istreambuf_iterator<char>{stream}, {}};
    const auto direct = source.find("void Rasterizer::Draw(bool");
    const auto indirect = source.find("void Rasterizer::DrawIndirect(bool");
    const auto dispatch = source.find("void Rasterizer::DispatchDirect()");
    ASSERT_NE(direct, std::string::npos);
    ASSERT_NE(indirect, std::string::npos);
    ASSERT_NE(dispatch, std::string::npos);
    const auto d = source.substr(direct, indirect - direct);
    const auto i = source.substr(indirect, dispatch - indirect);
    EXPECT_NE(d.find("RecordEmittedColorCommand("), std::string::npos);
    EXPECT_NE(i.find("RecordEmittedColorCommand("), std::string::npos);
    EXPECT_LT(d.find("cmdbuf.draw("), d.find("RecordEmittedColorCommand("));
    EXPECT_LT(i.find("cmdbuf.drawIndirect("), i.find("RecordEmittedColorCommand("));
    const auto record = source.find("void Rasterizer::RecordEmittedColorCommand(");
    ASSERT_NE(record, std::string::npos);
    const auto end = source.find("\n}\n", record);
    const auto body = source.substr(record, end - record);
    EXPECT_NE(body.find("color_command_ledger.Enabled()"), std::string::npos);
    EXPECT_NE(body.find("state.color_attachments"), std::string::npos);
    EXPECT_NE(body.find("desc.view_info.format"), std::string::npos);
    EXPECT_NE(body.find("image.GetHostExtent()"), std::string::npos);
    EXPECT_NE(body.find("std::hash<GraphicsPipelineKey>{}(pipeline.GetGraphicsKey())"),
              std::string::npos);
    for (const auto forbidden : {"FindImage(", "FindView(", "FindRenderTarget(", "ReadMemory(",
                                 "Download", "Wait", "Flush", "Address()"}) {
        EXPECT_EQ(body.find(forbidden), std::string::npos) << forbidden;
    }
    EXPECT_NE(source.find("SHADPS4_EMITTED_COLOR_VERTEX_HASH"), std::string::npos);
    EXPECT_NE(body.find("emitted_color_vertex_receipt.Observe(vertex_hash, has_color)"),
              std::string::npos);
}

} // namespace
