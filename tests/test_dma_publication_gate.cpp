// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/dma_publication_gate.h"

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
}

} // namespace
