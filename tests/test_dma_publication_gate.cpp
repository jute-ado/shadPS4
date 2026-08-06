// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/dma_publication_gate.h"
#include "video_core/renderer_vulkan/dma_discovery_policy.h"

namespace {

using VideoCore::DmaAttemptResult;
using VideoCore::DmaFaultEpoch;
using VideoCore::DmaPublicationGate;
using VideoCore::DmaWorkTraits;

TEST(DmaPublicationGate, FaultedDiscoveryRetriesAndPublishesExactlyOnceWhenClean) {
    DmaPublicationGate gate{
        DmaPublicationGate::Config{.maximum_attempts = 3, .binding_generation = 42}};

    EXPECT_EQ(gate.BeginAttempt(42), DmaAttemptResult::DiscoverWithoutPublication);
    EXPECT_EQ(gate.CompleteAttempt(DmaFaultEpoch::Faults({0x1000})),
              DmaAttemptResult::RetryWithoutPublication);
    EXPECT_EQ(gate.BeginAttempt(42), DmaAttemptResult::DiscoverWithoutPublication);
    EXPECT_EQ(gate.CompleteAttempt(DmaFaultEpoch::Faults({0x2000})),
              DmaAttemptResult::RetryWithoutPublication);
    EXPECT_EQ(gate.BeginAttempt(42), DmaAttemptResult::DiscoverWithoutPublication);
    EXPECT_EQ(gate.CompleteAttempt(DmaFaultEpoch::Clean()), DmaAttemptResult::PublishExactlyOnce);

    EXPECT_EQ(gate.AttemptCount(), 3);
    EXPECT_EQ(gate.PublicationCount(), 1);
    EXPECT_EQ(gate.BeginAttempt(42), DmaAttemptResult::Complete);
}

TEST(DmaPublicationGate, BindingMutationAbortsWithoutPublication) {
    DmaPublicationGate gate{
        DmaPublicationGate::Config{.maximum_attempts = 3, .binding_generation = 17}};

    EXPECT_EQ(gate.BeginAttempt(17), DmaAttemptResult::DiscoverWithoutPublication);
    EXPECT_EQ(gate.CompleteAttempt(DmaFaultEpoch::Faults({0x3000})),
              DmaAttemptResult::RetryWithoutPublication);
    EXPECT_EQ(gate.BeginAttempt(18), DmaAttemptResult::AbortWithoutPublication);
    EXPECT_EQ(gate.PublicationCount(), 0);
}

TEST(DmaPublicationGate, InvalidOverflowAndRetryExhaustionNeverPublish) {
    DmaPublicationGate invalid{
        DmaPublicationGate::Config{.maximum_attempts = 3, .binding_generation = 1}};
    EXPECT_EQ(invalid.BeginAttempt(1), DmaAttemptResult::DiscoverWithoutPublication);
    EXPECT_EQ(invalid.CompleteAttempt(DmaFaultEpoch::Invalid()),
              DmaAttemptResult::AbortWithoutPublication);
    EXPECT_EQ(invalid.PublicationCount(), 0);

    DmaPublicationGate overflow{
        DmaPublicationGate::Config{.maximum_attempts = 3, .binding_generation = 2}};
    EXPECT_EQ(overflow.BeginAttempt(2), DmaAttemptResult::DiscoverWithoutPublication);
    EXPECT_EQ(overflow.CompleteAttempt(DmaFaultEpoch::Overflow()),
              DmaAttemptResult::AbortWithoutPublication);
    EXPECT_EQ(overflow.PublicationCount(), 0);

    DmaPublicationGate exhausted{
        DmaPublicationGate::Config{.maximum_attempts = 2, .binding_generation = 3}};
    EXPECT_EQ(exhausted.BeginAttempt(3), DmaAttemptResult::DiscoverWithoutPublication);
    EXPECT_EQ(exhausted.CompleteAttempt(DmaFaultEpoch::Faults({0x4000})),
              DmaAttemptResult::RetryWithoutPublication);
    EXPECT_EQ(exhausted.BeginAttempt(3), DmaAttemptResult::DiscoverWithoutPublication);
    EXPECT_EQ(exhausted.CompleteAttempt(DmaFaultEpoch::Faults({0x5000})),
              DmaAttemptResult::AbortWithoutPublication);
    EXPECT_EQ(exhausted.PublicationCount(), 0);
}

TEST(DmaPublicationGate, EligibilityExcludesWorkWithUnreplayableSideEffects) {
    const DmaWorkTraits eligible{.vertex_dma_reads = true};
    EXPECT_TRUE(VideoCore::IsDmaDiscoveryEligible(eligible));

    auto traits = eligible;
    traits.fragment_dma_reads = true;
    EXPECT_FALSE(VideoCore::IsDmaDiscoveryEligible(traits));
    traits = eligible;
    traits.guest_or_gds_writes = true;
    EXPECT_FALSE(VideoCore::IsDmaDiscoveryEligible(traits));
    traits = eligible;
    traits.atomics = true;
    EXPECT_FALSE(VideoCore::IsDmaDiscoveryEligible(traits));
    traits = eligible;
    traits.storage_image_writes = true;
    EXPECT_FALSE(VideoCore::IsDmaDiscoveryEligible(traits));
    traits = eligible;
    traits.unsupported_stage_dma_reads = true;
    EXPECT_FALSE(VideoCore::IsDmaDiscoveryEligible(traits));
}

TEST(DmaPublicationGate, RebuildsEveryDiscoveryAttemptBeforePublishingOnce) {
    DmaPublicationGate gate{
        DmaPublicationGate::Config{.maximum_attempts = 3, .binding_generation = 23}};
    const std::array epochs{DmaFaultEpoch::FaultCount(1), DmaFaultEpoch::Clean()};
    std::vector<char> operations;

    const auto result = VideoCore::ExecuteDmaPublicationGate(
        gate, 23,
        [&](std::uint32_t attempt) {
            operations.push_back('D');
            return epochs.at(attempt - 1);
        },
        [&] {
            operations.push_back('P');
            return true;
        });

    EXPECT_EQ(result, VideoCore::DmaExecutionResult::Published);
    EXPECT_EQ(operations, (std::vector<char>{'D', 'D', 'P'}));
    EXPECT_EQ(gate.AttemptCount(), 2);
    EXPECT_EQ(gate.PublicationCount(), 1);
}

TEST(DmaDiscoveryPolicy, DiscoveryNeverClearsOrConsumesAttachmentMetadata) {
    const auto cleared =
        VideoCore::ResolveDmaAttachmentPolicy(true, VideoCore::DmaAttachmentMode::Discovery);
    EXPECT_FALSE(cleared.load_clear);
    EXPECT_FALSE(cleared.consume_metadata);

    const auto loaded =
        VideoCore::ResolveDmaAttachmentPolicy(false, VideoCore::DmaAttachmentMode::Discovery);
    EXPECT_FALSE(loaded.load_clear);
    EXPECT_FALSE(loaded.consume_metadata);
}

TEST(DmaDiscoveryPolicy, PublicationPreservesClearAndConsumesMetadata) {
    const auto cleared =
        VideoCore::ResolveDmaAttachmentPolicy(true, VideoCore::DmaAttachmentMode::Publication);
    EXPECT_TRUE(cleared.load_clear);
    EXPECT_TRUE(cleared.consume_metadata);

    const auto loaded =
        VideoCore::ResolveDmaAttachmentPolicy(false, VideoCore::DmaAttachmentMode::Publication);
    EXPECT_FALSE(loaded.load_clear);
    EXPECT_TRUE(loaded.consume_metadata);
}

} // namespace
