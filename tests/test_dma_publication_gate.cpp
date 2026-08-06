// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/dma_gpu_publication_predicate.h"
#include "video_core/buffer_cache/dma_publication_gate.h"
#include "video_core/renderer_vulkan/dma_discovery_policy.h"

namespace {

using VideoCore::DmaAttemptResult;
using VideoCore::DmaFaultEpoch;
using VideoCore::DmaPublicationGate;
using VideoCore::DmaWorkTraits;
using VideoCore::GpuDmaPredicateCommand;
using VideoCore::GpuDmaPredicateSlotPool;
using VideoCore::GpuDmaPredicateSupport;

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

TEST(DmaDiscoveryPolicy, DiscoveryNeverClearsOrConsumesAttachmentMetadata) {
    const auto cleared = VideoCore::ResolveDmaAttachmentPolicy(
        true, VideoCore::DmaAttachmentMode::Discovery);
    EXPECT_FALSE(cleared.load_clear);
    EXPECT_FALSE(cleared.consume_metadata);

    const auto loaded = VideoCore::ResolveDmaAttachmentPolicy(
        false, VideoCore::DmaAttachmentMode::Discovery);
    EXPECT_FALSE(loaded.load_clear);
    EXPECT_FALSE(loaded.consume_metadata);
}

TEST(DmaDiscoveryPolicy, PublicationPreservesClearAndConsumesMetadata) {
    const auto cleared = VideoCore::ResolveDmaAttachmentPolicy(
        true, VideoCore::DmaAttachmentMode::Publication);
    EXPECT_TRUE(cleared.load_clear);
    EXPECT_TRUE(cleared.consume_metadata);

    const auto loaded = VideoCore::ResolveDmaAttachmentPolicy(
        false, VideoCore::DmaAttachmentMode::Publication);
    EXPECT_FALSE(loaded.load_clear);
    EXPECT_TRUE(loaded.consume_metadata);
}

TEST(GpuDmaPublicationPredicate, FailsClosedWithoutSupportOrOwnedSlot) {
    constexpr GpuDmaPredicateSupport supported{
        .extension_available = true,
        .conditional_rendering_feature = true,
    };
    constexpr GpuDmaPredicateSupport no_extension{
        .extension_available = false,
        .conditional_rendering_feature = true,
    };
    constexpr GpuDmaPredicateSupport no_feature{
        .extension_available = true,
        .conditional_rendering_feature = false,
    };
    GpuDmaPredicateSlotPool<1> slots;
    const auto owned = slots.Acquire(0);
    ASSERT_TRUE(owned.has_value());

    EXPECT_FALSE(VideoCore::BuildGpuDmaPublicationPlan(no_extension, owned, slots).has_value());
    EXPECT_FALSE(VideoCore::BuildGpuDmaPublicationPlan(no_feature, owned, slots).has_value());
    EXPECT_FALSE(
        VideoCore::BuildGpuDmaPublicationPlan(supported, std::nullopt, slots).has_value());

    const auto stale = *owned;
    ASSERT_TRUE(slots.ReleaseUnsubmitted(*owned));
    EXPECT_FALSE(VideoCore::BuildGpuDmaPublicationPlan(supported, stale, slots).has_value());
}

TEST(GpuDmaPublicationPredicate, CleanAttemptUsesGpuOrderedConditionalPublication) {
    constexpr GpuDmaPredicateSupport supported{
        .extension_available = true,
        .conditional_rendering_feature = true,
    };
    GpuDmaPredicateSlotPool<1> slots;
    const auto owned = slots.Acquire(0);
    ASSERT_TRUE(owned.has_value());

    const auto plan = VideoCore::BuildGpuDmaPublicationPlan(supported, owned, slots);
    ASSERT_TRUE(plan.has_value());
    constexpr std::array expected{
        GpuDmaPredicateCommand::ResetClean,
        GpuDmaPredicateCommand::TransferWriteToShaderReadWriteBarrier,
        GpuDmaPredicateCommand::RasterDiscardDiscovery,
        GpuDmaPredicateCommand::ShaderWriteToConditionalReadBarrier,
        GpuDmaPredicateCommand::BeginConditionalPublication,
        GpuDmaPredicateCommand::PublicationDraw,
        GpuDmaPredicateCommand::EndConditionalPublication,
    };
    EXPECT_EQ(plan->commands, expected);
    EXPECT_EQ(plan->predicate_value_for_clean, 1U);
    EXPECT_EQ(plan->predicate_value_for_fault, 0U);
    EXPECT_FALSE(plan->conditional_inverted);
    EXPECT_FALSE(plan->requires_cpu_wait);
}

TEST(GpuDmaPublicationPredicate, SlotsRemainUniqueUntilTheirGpuTickCompletes) {
    GpuDmaPredicateSlotPool<2> slots;
    const auto first = slots.Acquire(0);
    const auto second = slots.Acquire(0);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_NE(first->index, second->index);
    EXPECT_FALSE(slots.Acquire(0).has_value());

    ASSERT_TRUE(slots.Retire(*first, 7));
    EXPECT_FALSE(slots.Acquire(6).has_value());
    const auto reused = slots.Acquire(7);
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(reused->index, first->index);
    EXPECT_NE(reused->generation, first->generation);

    EXPECT_FALSE(slots.Retire(*first, 8));
    EXPECT_TRUE(slots.Owns(*reused));
}

TEST(GpuDmaPublicationPredicate, UnsubmittedReleaseRejectsStaleOwnership) {
    GpuDmaPredicateSlotPool<1> slots;
    const auto first = slots.Acquire(0);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(slots.ReleaseUnsubmitted(*first));

    const auto second = slots.Acquire(0);
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->index, first->index);
    EXPECT_NE(second->generation, first->generation);
    EXPECT_FALSE(slots.ReleaseUnsubmitted(*first));
    EXPECT_TRUE(slots.Owns(*second));
}

} // namespace
