// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "video_core/color_attachment_provenance_receipt.h"

namespace {
using namespace VideoCore;

constexpr std::string_view SelectorA = "0123456789abcdef";
constexpr std::string_view SelectorB = "fedcba9876543210";

ColorAttachmentProvenancePolicy Parse(std::optional<std::string_view> enable = "1",
                                      std::optional<std::string_view> start = "1000",
                                      std::optional<std::string_view> duration = "100",
                                      std::optional<std::string_view> points = "484:694,516:694",
                                      std::optional<std::string_view> selectors = SelectorA,
                                      std::optional<std::string_view> capacity = "2") {
    return ColorAttachmentProvenancePolicy::Parse(enable, start, duration, points, selectors,
                                                  capacity);
}

ColorAttachmentProvenanceTarget ValidTarget() {
    return {
        .image_uid = 17,
        .format = 97,
        .width = 1920,
        .height = 1080,
        .levels = 4,
        .layers = 3,
        .mip = 1,
        .layer = 2,
        .bits_per_texel = 64,
        .samples = 1,
    };
}

std::string ReadRasterizerSource() {
    std::ifstream input{SHADPS4_COLOR_ATTACHMENT_PROVENANCE_RASTERIZER_PATH,
                        std::ios::binary};
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

TEST(ColorAttachmentProvenancePolicy, DefaultAndExplicitOffIgnoreIrrelevantFields) {
    for (const auto enable :
         {std::optional<std::string_view>{}, std::optional<std::string_view>{"0"}}) {
        const auto policy = ColorAttachmentProvenancePolicy::Parse(enable, "garbage", "garbage",
                                                                   "garbage", "garbage", "garbage");
        EXPECT_TRUE(policy.valid);
        EXPECT_FALSE(policy.enabled);
        EXPECT_EQ(policy.point_count, 0);
        EXPECT_EQ(policy.selector_count, 0);
    }
}

TEST(ColorAttachmentProvenancePolicy, ExplicitEnableAcceptsOnlyStrictBoundedConfiguration) {
    const auto low =
        ColorAttachmentProvenancePolicy::Parse("1", "0", "1", "0:0", "0000000000000001", "1");
    ASSERT_TRUE(low.valid);
    EXPECT_TRUE(low.enabled);
    EXPECT_EQ(low.start_ms, 0);
    EXPECT_EQ(low.duration_ms, 1);
    EXPECT_EQ(low.command_capacity, 1);

    const auto high = ColorAttachmentProvenancePolicy::Parse(
        "1", "600000", "5000", "0:0,250:250,750:750,1000:1000",
        "0000000000000001,0000000000000002,0000000000000003,0000000000000004,"
        "0000000000000005,0000000000000006,0000000000000007,0000000000000008",
        "64");
    ASSERT_TRUE(high.valid);
    EXPECT_EQ(high.start_ms, 600000);
    EXPECT_EQ(high.duration_ms, 5000);
    EXPECT_EQ(high.point_count, 4);
    EXPECT_EQ(high.selector_count, 8);
    EXPECT_EQ(high.command_capacity, 64);
}

TEST(ColorAttachmentProvenancePolicy, MalformedConfigurationFailsClosed) {
    for (const auto value : {"", "true", "01", " 1", "1 "}) {
        const auto policy = Parse(value);
        EXPECT_FALSE(policy.valid) << value;
        EXPECT_FALSE(policy.enabled) << value;
    }
    for (const auto value : {"", "+1", "-1", " 1", "1 ", "1ms", "18446744073709551616"}) {
        EXPECT_FALSE(Parse("1", value).valid) << value;
        EXPECT_FALSE(Parse("1", "0", value).valid) << value;
        EXPECT_FALSE(Parse("1", "0", "1", "0:0", SelectorA, value).valid) << value;
    }
    EXPECT_FALSE(Parse("1", std::nullopt).valid);
    EXPECT_FALSE(Parse("1", "0", std::nullopt).valid);
    EXPECT_FALSE(Parse("1", "600001", "1").valid);
    EXPECT_FALSE(Parse("1", "0", "0").valid);
    EXPECT_FALSE(Parse("1", "0", "5001").valid);
    EXPECT_FALSE(Parse("1", "0", "1", "0:0", SelectorA, "0").valid);
    EXPECT_FALSE(Parse("1", "0", "1", "0:0", SelectorA, "65").valid);

    for (const auto points :
         {"", "0", "0:", ":0", "0:0:", "-1:0", "0:-1", "1001:0", "0:1001", "0:0,1:1,2:2,3:3,4:4"}) {
        EXPECT_FALSE(Parse("1", "0", "1", points).valid) << points;
    }
    EXPECT_FALSE(Parse("1", "0", "1", "0:0,0:0,0:0,0:0,0:0").valid);
    for (const auto selectors :
         {"", "0", "0x0123456789abcdef", "0123456789abcde", "0123456789abcdef0", "0123456789abcdeg",
          "0000000000000001,0000000000000002,0000000000000003,"
          "0000000000000004,0000000000000005,0000000000000006,"
          "0000000000000007,0000000000000008,0000000000000009"}) {
        EXPECT_FALSE(Parse("1", "0", "1", "0:0", selectors).valid) << selectors;
    }
    EXPECT_FALSE(Parse("1", "0", "1", "0:0",
                       "0123456789abcdef,0123456789abcdef,0123456789abcdef,"
                       "0123456789abcdef,0123456789abcdef,0123456789abcdef,"
                       "0123456789abcdef,0123456789abcdef,0123456789abcdef")
                     .valid);
}

TEST(ColorAttachmentProvenancePolicy, DeduplicatesPointsAndSelectorsInFirstSeenOrder) {
    const auto policy = ColorAttachmentProvenancePolicy::Parse(
        "1", "0", "1", "484:694,516:694,484:694,400:694",
        "0123456789abcdef,fedcba9876543210,0123456789ABCDEF", "1");
    ASSERT_TRUE(policy.valid);
    ASSERT_EQ(policy.point_count, 3);
    EXPECT_EQ(policy.points[0], (ColorAttachmentProvenancePoint{484, 694}));
    EXPECT_EQ(policy.points[1], (ColorAttachmentProvenancePoint{516, 694}));
    EXPECT_EQ(policy.points[2], (ColorAttachmentProvenancePoint{400, 694}));
    ASSERT_EQ(policy.selector_count, 2);
    EXPECT_EQ(policy.selectors[0], 0x0123456789abcdefULL);
    EXPECT_EQ(policy.selectors[1], 0xfedcba9876543210ULL);
}

TEST(ColorAttachmentProvenanceReceipt, SelectsExactHashesAndClosesAtCompletedCapacity) {
    auto policy = Parse();
    ASSERT_TRUE(policy.valid);
    ColorAttachmentProvenanceReceipt receipt{policy};

    EXPECT_FALSE(receipt.Begin(999, 0x0123456789abcdefULL).capture.has_value());
    EXPECT_FALSE(receipt.Begin(1000, 0x0123456789abcdeeULL).capture.has_value());

    const auto first = receipt.Begin(1000, 0x0123456789abcdefULL);
    ASSERT_TRUE(first.capture.has_value());
    EXPECT_EQ(first.capture->ordinal, 0);
    EXPECT_EQ(first.capture->selector, 0x0123456789abcdefULL);
    EXPECT_FALSE(receipt.Finish(*first.capture, 1001).has_value());
    EXPECT_FALSE(receipt.Finish(*first.capture, 1002).has_value());

    const auto second = receipt.Begin(1002, 0x0123456789abcdefULL);
    ASSERT_TRUE(second.capture.has_value());
    EXPECT_EQ(second.capture->ordinal, 1);
    const auto end = receipt.Finish(*second.capture, 1003);
    ASSERT_TRUE(end.has_value());
    EXPECT_EQ(end->reason, ColorAttachmentProvenanceEnd::Capacity);
    EXPECT_EQ(end->completed_commands, 2);
    EXPECT_EQ(end->matched_commands, 2);
    EXPECT_FALSE(receipt.Begin(1004, 0x0123456789abcdefULL).capture.has_value());
}

TEST(ColorAttachmentProvenanceReceipt, WindowAndTeardownClosureAreDeterministic) {
    ColorAttachmentProvenanceReceipt window{Parse()};
    const auto at_end = window.Begin(1100, 0x0123456789abcdefULL);
    ASSERT_TRUE(at_end.end.has_value());
    EXPECT_EQ(at_end.end->reason, ColorAttachmentProvenanceEnd::Window);
    EXPECT_EQ(at_end.end->elapsed_ms, 1100);
    EXPECT_FALSE(window.Begin(1101, 0x0123456789abcdefULL).end.has_value());
    EXPECT_FALSE(window.Close(1102).has_value());

    ColorAttachmentProvenanceReceipt teardown{Parse()};
    const auto end = teardown.Close(500);
    ASSERT_TRUE(end.has_value());
    EXPECT_EQ(end->reason, ColorAttachmentProvenanceEnd::Teardown);
    EXPECT_FALSE(teardown.Close(501).has_value());

    ColorAttachmentProvenanceReceipt disabled{ColorAttachmentProvenancePolicy{}};
    EXPECT_FALSE(disabled.Begin(1000, 0x0123456789abcdefULL).capture.has_value());
    EXPECT_FALSE(disabled.Close(1001).has_value());
}

TEST(ColorAttachmentProvenancePlanner, ScalesNormalizedPointByMipAndSelectsExactLayer) {
    ColorAttachmentProvenanceSlotAllocator slots{192};
    const auto result = PlanColorAttachmentProvenanceSample(
        ValidTarget(), ColorAttachmentProvenancePoint{500, 1000}, slots);
    ASSERT_TRUE(result.plan.has_value());
    EXPECT_EQ(result.rejection, ColorAttachmentProvenanceRejection::None);
    EXPECT_EQ(result.plan->width, 960);
    EXPECT_EQ(result.plan->height, 540);
    EXPECT_EQ(result.plan->x, 480);
    EXPECT_EQ(result.plan->y, 539);
    EXPECT_EQ(result.plan->mip, 1);
    EXPECT_EQ(result.plan->layer, 2);
    EXPECT_EQ(result.plan->payload_bytes, 8);
    EXPECT_EQ(result.plan->pre_offset, 0);
    EXPECT_EQ(result.plan->post_offset, 48);

    auto twelve_byte_target = ValidTarget();
    twelve_byte_target.bits_per_texel = 96;
    const auto twelve_byte = PlanColorAttachmentProvenanceSample(
        twelve_byte_target, ColorAttachmentProvenancePoint{0, 0}, slots);
    ASSERT_TRUE(twelve_byte.plan.has_value());
    EXPECT_EQ(twelve_byte.plan->payload_bytes, 12);
    EXPECT_EQ(twelve_byte.plan->pre_offset, 96);
    EXPECT_EQ(twelve_byte.plan->post_offset, 144);
    EXPECT_EQ(twelve_byte.plan->pre_offset % 48, 0);
    EXPECT_EQ(twelve_byte.plan->post_offset % 48, 0);
}

TEST(ColorAttachmentProvenancePlanner, PairReservationsAreFortyEightByteAlignedAndAtomic) {
    ColorAttachmentProvenanceSlotAllocator slots{96};
    const auto first = slots.ReservePair();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->pre_offset, 0);
    EXPECT_EQ(first->post_offset, 48);
    EXPECT_EQ(first->pre_offset % 48, 0);
    EXPECT_EQ(first->post_offset % 48, 0);
    EXPECT_EQ(slots.UsedBytes(), 96);
    EXPECT_FALSE(slots.ReservePair().has_value());
    EXPECT_EQ(slots.UsedBytes(), 96);

    ColorAttachmentProvenanceSlotAllocator short_buffer{95};
    EXPECT_FALSE(short_buffer.ReservePair().has_value());
    EXPECT_EQ(short_buffer.UsedBytes(), 0);
}

TEST(ColorAttachmentProvenancePlanner, RuntimeAtomStrideKeepsPreAndPostInSeparateAtoms) {
    ColorAttachmentProvenanceSlotAllocator slots{256, 128};
    const auto first = slots.ReservePair();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->pre_offset, 0);
    EXPECT_EQ(first->post_offset, 128);
    EXPECT_EQ(slots.UsedBytes(), 256);
    EXPECT_FALSE(slots.ReservePair().has_value());
}

TEST(ColorAttachmentProvenancePlanner, RejectsUnsupportedTargetsWithoutConsumingSlots) {
    const auto expect_rejected = [](ColorAttachmentProvenanceTarget target,
                                    ColorAttachmentProvenanceRejection expected) {
        ColorAttachmentProvenanceSlotAllocator slots{96};
        const auto result = PlanColorAttachmentProvenanceSample(
            target, ColorAttachmentProvenancePoint{500, 500}, slots);
        EXPECT_FALSE(result.plan.has_value());
        EXPECT_EQ(result.rejection, expected);
        EXPECT_EQ(slots.UsedBytes(), 0);
    };

    auto target = ValidTarget();
    target.format = 0;
    expect_rejected(target, ColorAttachmentProvenanceRejection::UndefinedFormat);
    target = ValidTarget();
    target.samples = 2;
    expect_rejected(target, ColorAttachmentProvenanceRejection::Multisampled);
    target = ValidTarget();
    target.compressed = true;
    expect_rejected(target, ColorAttachmentProvenanceRejection::Compressed);
    target = ValidTarget();
    target.depth = true;
    expect_rejected(target, ColorAttachmentProvenanceRejection::Depth);
    target = ValidTarget();
    target.is_3d = true;
    expect_rejected(target, ColorAttachmentProvenanceRejection::ThreeDimensional);
    target = ValidTarget();
    target.bits_per_texel = 136;
    expect_rejected(target, ColorAttachmentProvenanceRejection::InvalidTexelSize);
    target = ValidTarget();
    target.bits_per_texel = 7;
    expect_rejected(target, ColorAttachmentProvenanceRejection::InvalidTexelSize);
    target = ValidTarget();
    target.mip = target.levels;
    expect_rejected(target, ColorAttachmentProvenanceRejection::InvalidMip);
    target = ValidTarget();
    target.layer = target.layers;
    expect_rejected(target, ColorAttachmentProvenanceRejection::InvalidLayer);
    target = ValidTarget();
    target.width = 0;
    expect_rejected(target, ColorAttachmentProvenanceRejection::InvalidExtent);
    ColorAttachmentProvenanceSlotAllocator invalid_point_slots{96};
    const auto invalid_point = PlanColorAttachmentProvenanceSample(
        ValidTarget(), ColorAttachmentProvenancePoint{1001, 0}, invalid_point_slots);
    EXPECT_FALSE(invalid_point.plan.has_value());
    EXPECT_EQ(invalid_point.rejection, ColorAttachmentProvenanceRejection::InvalidPoint);
    EXPECT_EQ(invalid_point_slots.UsedBytes(), 0);
}

TEST(ColorAttachmentProvenancePlanner, DeduplicatesExactSubresourcesInFirstSeenOrder) {
    auto a = ValidTarget();
    auto b = ValidTarget();
    b.image_uid = 3;
    auto a_alias = a;
    a_alias.format = 83;
    auto c = a;
    c.layer = 1;
    auto b_alias = b;
    b_alias.bits_per_texel = 32;
    const std::array targets{a, b, a_alias, c, b_alias};

    const auto indices = UniqueColorAttachmentProvenanceTargets(targets);
    EXPECT_EQ(indices, (std::vector<std::size_t>{0, 1, 3}));
}

TEST(ColorAttachmentProvenanceReceipt, ProductionWiringSurroundsDirectAndIndirectDrawsWithoutWaits) {
    const auto source = ReadRasterizerSource();
    const auto direct_begin = source.find("void Rasterizer::Draw(bool is_indexed");
    const auto direct_end = source.find("void Rasterizer::DrawIndirect(", direct_begin);
    const auto indirect_end = source.find("void Rasterizer::DispatchDirect(", direct_end);
    ASSERT_NE(direct_begin, std::string::npos);
    ASSERT_NE(direct_end, std::string::npos);
    ASSERT_NE(indirect_end, std::string::npos);

    const auto direct = source.substr(direct_begin, direct_end - direct_begin);
    const auto indirect = source.substr(direct_end, indirect_end - direct_end);
    for (const auto& body : {direct, indirect}) {
        const auto begin_rendering = body.find("scheduler.BeginRendering(state)");
        const auto prepare = body.find("PrepareColorAttachmentProvenance(");
        const auto draw = body.find("cmdbuf.draw", begin_rendering);
        const auto complete = body.find("CompleteColorAttachmentProvenance(");
        const auto reset = body.rfind("ResetBindings()");
        EXPECT_NE(prepare, std::string::npos);
        EXPECT_NE(complete, std::string::npos);
        EXPECT_LT(begin_rendering, prepare);
        EXPECT_LT(prepare, draw);
        EXPECT_LT(draw, complete);
        EXPECT_LT(complete, reset);
    }

    const auto implementation_begin =
        source.find("Rasterizer::PrepareColorAttachmentProvenance(");
    const auto implementation_end = source.find("void Rasterizer::CpSync(",
                                                implementation_begin);
    ASSERT_NE(implementation_begin, std::string::npos);
    ASSERT_NE(implementation_end, std::string::npos);
    const auto implementation =
        source.substr(implementation_begin, implementation_end - implementation_begin);
    for (const auto required : {"copyImageToBuffer", "pipelineBarrier2(", "DeferOperation(",
                                "InvalidateMappedRange(", "depth_clear = false",
                                "stencil_clear = false"}) {
        EXPECT_NE(implementation.find(required), std::string::npos) << required;
    }
    for (const auto forbidden : {"scheduler.Finish(", "scheduler.Wait(", "scheduler.Flush(",
                                 ".Download(", "DownloadImageMemory(", "TryWriteBacking(",
                                 ".GetBarriers("}) {
        EXPECT_EQ(implementation.find(forbidden), std::string::npos) << forbidden;
    }
    for (const auto environment : {"SHADPS4_COLOR_ATTACHMENT_PROVENANCE",
                                   "SHADPS4_COLOR_ATTACHMENT_PROVENANCE_START_MS",
                                   "SHADPS4_COLOR_ATTACHMENT_PROVENANCE_DURATION_MS",
                                   "SHADPS4_COLOR_ATTACHMENT_PROVENANCE_POINTS",
                                   "SHADPS4_COLOR_ATTACHMENT_PROVENANCE_SELECTORS",
                                   "SHADPS4_COLOR_ATTACHMENT_PROVENANCE_CAPACITY"}) {
        EXPECT_NE(source.find(environment), std::string::npos) << environment;
    }
}

} // namespace
