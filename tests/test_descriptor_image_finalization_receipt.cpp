// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/descriptor_image_finalization_receipt.h"

namespace {

using Vulkan::DescriptorImageCommandContext;
using Vulkan::DescriptorImageCommandKind;
using Vulkan::DescriptorImageFinalizationFailure;
using Vulkan::DescriptorImageFinalizationObservation;
using Vulkan::DescriptorImageFinalizationReceiptPolicy;
using Vulkan::DescriptorImageFinalizationReceipt;

DescriptorImageFinalizationObservation Example() {
    return {
        .failure = DescriptorImageFinalizationFailure::StaleGeneration,
        .command = {.kind = DescriptorImageCommandKind::IndexedDraw,
                    .vertex_hash = 0x1122334455667788ULL,
                    .fragment_hash = 0x8877665544332211ULL},
        .descriptor_index = 7,
        .pending_index = 5,
        .image_id = 19,
        .image_uid = 0x123456789abcdef0ULL,
    };
}

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input.is_open());
    std::string text{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    std::erase(text, '\r');
    return text;
}

TEST(DescriptorImageFinalizationReceipt, RequiresExactExplicitOptIn) {
    EXPECT_TRUE(Vulkan::ParseDescriptorImageFinalizationReceipt("1"));
    for (const auto value : {"", "0", "true", "yes", " 1", "1 ", "01", "+1"}) {
        EXPECT_FALSE(Vulkan::ParseDescriptorImageFinalizationReceipt(value)) << value;
    }
}

TEST(DescriptorImageFinalizationReceipt, DefaultOffDoesNotParseIrrelevantWindowValues) {
    for (const auto enabled :
         {std::optional<std::string_view>{}, std::optional<std::string_view>{"0"}}) {
        const auto policy =
            DescriptorImageFinalizationReceiptPolicy::Parse(enabled, "garbage", "garbage");
        EXPECT_TRUE(policy.valid);
        EXPECT_FALSE(policy.enabled);
    }
}

TEST(DescriptorImageFinalizationReceipt, ExplicitEnableRequiresStrictBoundedWindow) {
    const auto low = DescriptorImageFinalizationReceiptPolicy::Parse("1", "0", "1");
    EXPECT_TRUE(low.valid);
    EXPECT_TRUE(low.enabled);
    EXPECT_EQ(low.start_ms, 0U);
    EXPECT_EQ(low.duration_ms, 1U);
    const auto high =
        DescriptorImageFinalizationReceiptPolicy::Parse("1", "600000", "5000");
    EXPECT_TRUE(high.valid);
    EXPECT_TRUE(high.enabled);
    EXPECT_EQ(high.start_ms, 600000U);
    EXPECT_EQ(high.duration_ms, 5000U);
}

TEST(DescriptorImageFinalizationReceipt, MalformedExplicitWindowFailsClosed) {
    for (const auto value : {"", "true", "01", " 1", "1 "}) {
        const auto policy = DescriptorImageFinalizationReceiptPolicy::Parse(value, "0", "1");
        EXPECT_FALSE(policy.valid) << value;
        EXPECT_FALSE(policy.enabled) << value;
    }
    for (const auto value : {"", "+1", "-1", " 1", "1 ", "1ms",
                             "18446744073709551616"}) {
        EXPECT_FALSE(DescriptorImageFinalizationReceiptPolicy::Parse("1", value, "1").valid)
            << value;
        EXPECT_FALSE(DescriptorImageFinalizationReceiptPolicy::Parse("1", "0", value).valid)
            << value;
    }
    EXPECT_FALSE(DescriptorImageFinalizationReceiptPolicy::Parse("1", std::nullopt, "1").valid);
    EXPECT_FALSE(DescriptorImageFinalizationReceiptPolicy::Parse("1", "0", std::nullopt).valid);
    EXPECT_FALSE(DescriptorImageFinalizationReceiptPolicy::Parse("1", "600001", "1").valid);
    EXPECT_FALSE(DescriptorImageFinalizationReceiptPolicy::Parse("1", "0", "0").valid);
    EXPECT_FALSE(DescriptorImageFinalizationReceiptPolicy::Parse("1", "0", "5001").valid);
}

TEST(DescriptorImageFinalizationReceipt, ClassificationHasDeterministicFailurePrecedence) {
    using Failure = DescriptorImageFinalizationFailure;
    EXPECT_EQ(Vulkan::ClassifyDescriptorImageFinalization(false, false, true, false),
              Failure::PendingIndexOutOfBounds);
    EXPECT_EQ(Vulkan::ClassifyDescriptorImageFinalization(true, false, true, false),
              Failure::StaleGeneration);
    EXPECT_EQ(Vulkan::ClassifyDescriptorImageFinalization(true, true, true, false),
              Failure::NeedsRebind);
    EXPECT_EQ(Vulkan::ClassifyDescriptorImageFinalization(true, true, false, false),
              Failure::ViewMissing);
    EXPECT_EQ(Vulkan::ClassifyDescriptorImageFinalization(true, true, false, true), std::nullopt);
}

TEST(DescriptorImageFinalizationReceipt, NamesDrawDropsForRouteDiagnostics) {
    EXPECT_EQ(Vulkan::DescriptorImageFinalizationFailureName(
                  DescriptorImageFinalizationFailure::DrawDropped),
              "draw_dropped");
}

TEST(DescriptorImageFinalizationReceipt, CountsRepeatedFailuresWithinTheBoundedWindow) {
    DescriptorImageFinalizationReceipt receipt{{true, true, 0, 5000}};
    std::uint32_t descriptor_index = 0;

    auto observe = [&](DescriptorImageFinalizationFailure failure) {
        auto value = Example();
        value.failure = failure;
        value.descriptor_index = descriptor_index++;
        static_cast<void>(receipt.Observe(1, [&] { return value; }));
    };

    observe(DescriptorImageFinalizationFailure::StaleGeneration);
    observe(DescriptorImageFinalizationFailure::StaleGeneration);
    observe(DescriptorImageFinalizationFailure::NeedsRebind);
    observe(DescriptorImageFinalizationFailure::ViewMissing);
    observe(DescriptorImageFinalizationFailure::DrawDropped);
    observe(DescriptorImageFinalizationFailure::DrawDropped);

    EXPECT_EQ(receipt.FailureOccurrences(DescriptorImageFinalizationFailure::StaleGeneration), 2U);
    EXPECT_EQ(receipt.FailureOccurrences(DescriptorImageFinalizationFailure::NeedsRebind), 1U);
    EXPECT_EQ(receipt.FailureOccurrences(DescriptorImageFinalizationFailure::ViewMissing), 1U);
    EXPECT_EQ(receipt.FailureOccurrences(DescriptorImageFinalizationFailure::DrawDropped), 2U);
}

TEST(DescriptorImageFinalizationReceipt, DisabledDoesNotBuildOrRetainObservation) {
    DescriptorImageFinalizationReceipt receipt{{}};
    unsigned builds = 0;
    EXPECT_FALSE(receipt.Observe(178000, [&] {
        ++builds;
        return Example();
    }));
    EXPECT_EQ(builds, 0U);
    EXPECT_EQ(receipt.RecordedCount(), 0U);
}

TEST(DescriptorImageFinalizationReceipt, RendererWindowIsStartInclusiveAndEndExclusive) {
    DescriptorImageFinalizationReceipt receipt{{true, true, 178000, 5000}};
    unsigned builds = 0;
    const auto observe = [&](std::uint64_t elapsed) {
        return receipt.Observe(elapsed, [&] {
            auto value = Example();
            value.descriptor_index = builds++;
            return value;
        });
    };
    EXPECT_FALSE(observe(177999));
    EXPECT_TRUE(observe(178000));
    EXPECT_TRUE(observe(182999));
    EXPECT_FALSE(observe(183000));
    EXPECT_FALSE(observe(183001));
    EXPECT_EQ(builds, 2U);
    EXPECT_EQ(receipt.RecordedCount(), 2U);
}

TEST(DescriptorImageFinalizationReceipt, EnabledOwnsEveryContextAndIdentityField) {
    DescriptorImageFinalizationReceipt receipt{{true, true, 0, 5000}};
    auto source = Example();
    const auto first = receipt.Observe(0, [&] { return source; });
    ASSERT_TRUE(first);
    source.command.vertex_hash = 0;
    source.command.fragment_hash = 0;
    source.image_uid = 0;
    EXPECT_EQ(*first, Example());
    EXPECT_EQ(receipt.RecordedCount(), 1U);
}

TEST(DescriptorImageFinalizationReceipt, DuplicateDoesNotConsumeCapacity) {
    DescriptorImageFinalizationReceipt receipt{{true, true, 0, 5000}};
    ASSERT_TRUE(receipt.Observe(0, Example));
    for (unsigned repeat = 0; repeat < 100; ++repeat) {
        EXPECT_FALSE(receipt.Observe(1, Example));
    }
    EXPECT_EQ(receipt.RecordedCount(), 1U);
}

TEST(DescriptorImageFinalizationReceipt, EveryDiagnosticKeyFieldAffectsDeduplication) {
    DescriptorImageFinalizationReceipt receipt{{true, true, 0, 5000}};
    ASSERT_TRUE(receipt.Observe(0, Example));
    for (unsigned field = 0; field < 10; ++field) {
        auto value = Example();
        switch (field) {
        case 0:
            value.failure = DescriptorImageFinalizationFailure::NeedsRebind;
            break;
        case 1:
            value.command.kind = DescriptorImageCommandKind::DirectDraw;
            break;
        case 2:
            ++value.command.vertex_hash;
            break;
        case 3:
            ++value.command.fragment_hash;
            break;
        case 4:
            ++value.command.compute_hash;
            break;
        case 5:
            ++value.descriptor_index;
            break;
        case 6:
            ++value.pending_index;
            break;
        case 7:
            ++value.image_id;
            break;
        case 8:
            ++value.image_uid;
            break;
        case 9:
            value.command.kind = DescriptorImageCommandKind::IndirectDispatch;
            ++value.command.compute_hash;
            break;
        }
        EXPECT_TRUE(receipt.Observe(1 + field, [&] { return value; })) << field;
    }
    EXPECT_EQ(receipt.RecordedCount(), 11U);
}

TEST(DescriptorImageFinalizationReceipt, CapacityStopsBeforeBuildingThirtyThirdUniqueTuple) {
    DescriptorImageFinalizationReceipt receipt{{true, true, 0, 5000}};
    for (std::size_t index = 0; index < DescriptorImageFinalizationReceipt::Capacity; ++index) {
        auto value = Example();
        value.descriptor_index = static_cast<std::uint32_t>(index);
        ASSERT_TRUE(receipt.Observe(index, [&] { return value; }));
    }
    unsigned builds = 0;
    EXPECT_FALSE(receipt.Observe(32, [&] {
        ++builds;
        auto value = Example();
        value.descriptor_index = 999;
        return value;
    }));
    EXPECT_EQ(builds, 0U);
    EXPECT_EQ(receipt.RecordedCount(), DescriptorImageFinalizationReceipt::Capacity);
}

TEST(DescriptorImageFinalizationReceipt, ProductionWiringClassifiesWithoutRepairing) {
    const auto source = ReadSource(SHADPS4_DESCRIPTOR_FINALIZATION_RASTERIZER_PATH);
    const auto begin = source.find(
        "bool Rasterizer::FinalizeImageDescriptors(const DescriptorImageCommandContext& context)");
    const auto end = source.find("void Rasterizer::TrackBoundImage(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    for (const auto field : {"PendingIndexOutOfBounds", "StaleGeneration", "NeedsRebind",
                             "ViewMissing", "context", "descriptor_index", "pending_index",
                             "image_uid", "descriptor_image_finalization_receipt.Observe",
                             "descriptor_finalization_origin", "FailureOccurrences(failure)"}) {
        EXPECT_NE(body.find(field), std::string::npos) << field;
    }
    for (const auto forbidden : {"FindImage(", "FindTexture(", "UpdateImage(", "scheduler.",
                                 "DrainReadyFaults", "Wait", "Finish("}) {
        EXPECT_EQ(body.find(forbidden), std::string::npos) << forbidden;
    }
}

TEST(DescriptorImageFinalizationReceipt, ProductionUsesStrictLaunchWindowEnvironment) {
    const auto source = ReadSource(SHADPS4_DESCRIPTOR_FINALIZATION_RASTERIZER_PATH);
    for (const auto variable : {"SHADPS4_DESCRIPTOR_IMAGE_FINALIZATION_RECEIPT",
                                "SHADPS4_DESCRIPTOR_IMAGE_FINALIZATION_RECEIPT_START_MS",
                                "SHADPS4_DESCRIPTOR_IMAGE_FINALIZATION_RECEIPT_DURATION_MS"}) {
        EXPECT_NE(source.find(variable), std::string::npos) << variable;
    }
    EXPECT_NE(source.find("DescriptorImageFinalizationReceiptPolicy::Parse"), std::string::npos);
}

TEST(DescriptorImageFinalizationReceipt, ProductionRecordsImageFinalizationDrawDrops) {
    const auto source = ReadSource(SHADPS4_DESCRIPTOR_FINALIZATION_RASTERIZER_PATH);
    EXPECT_NE(source.find("void Rasterizer::ObserveDescriptorImageDrawDrop("),
              std::string::npos);
    EXPECT_NE(source.find("DescriptorImageFinalizationFailure::DrawDropped"),
              std::string::npos);
    EXPECT_NE(source.find("ObserveDescriptorImageDrawDrop(descriptor_context)"),
              std::string::npos);
}

} // namespace
