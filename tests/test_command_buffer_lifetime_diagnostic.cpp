// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <span>

#include <gtest/gtest.h>

#include "video_core/amdgpu/command_buffer_lifetime_diagnostic.h"

namespace AmdGpu {
namespace {

TEST(CommandBufferLifetimeDiagnostic, StableBufferRemainsStableAcrossEveryPhase) {
    CommandBufferLifetimeDiagnostic diagnostic{
        {.enabled = true, .max_words_per_signature = 16, .max_resume_checks = 4}};
    std::array<u32, 4> words{1, 2, 3, 4};

    auto probe = diagnostic.Begin(CommandBufferKind::TopLevelDcb, words);
    probe.ObserveInitial(words);
    probe.Suspend(std::span<const u32>{words}.subspan(2));
    probe.Resume(std::span<const u32>{words}.subspan(2));
    probe.ObserveFinal();

    const auto snapshot = diagnostic.Read();
    EXPECT_EQ(snapshot.observed_buffers, 1);
    EXPECT_EQ(snapshot.resume_checks, 1);
    EXPECT_EQ(snapshot.initial_mutations, 0);
    EXPECT_EQ(snapshot.later_resume_mutations, 0);
    EXPECT_EQ(snapshot.final_mutations, 0);
}

TEST(CommandBufferLifetimeDiagnostic, DetectsMutationBeforeInitialParserResume) {
    CommandBufferLifetimeDiagnostic diagnostic{
        {.enabled = true, .max_words_per_signature = 16, .max_resume_checks = 4}};
    std::array<u32, 4> words{1, 2, 3, 4};
    auto probe = diagnostic.Begin(CommandBufferKind::TopLevelDcb, words);

    words[2] = 99;
    probe.ObserveInitial(words);

    EXPECT_EQ(diagnostic.Read().initial_mutations, 1);
}

TEST(CommandBufferLifetimeDiagnostic, DetectsMutationAcrossALaterCoroutineResume) {
    CommandBufferLifetimeDiagnostic diagnostic{
        {.enabled = true, .max_words_per_signature = 16, .max_resume_checks = 4}};
    std::array<u32, 5> words{1, 2, 3, 4, 5};
    auto probe = diagnostic.Begin(CommandBufferKind::TopLevelDcb, words);
    probe.ObserveInitial(words);
    const auto remaining = std::span<const u32>{words}.subspan(2);
    probe.Suspend(remaining);

    words[4] = 77;
    probe.Resume(remaining);

    const auto snapshot = diagnostic.Read();
    EXPECT_EQ(snapshot.later_resume_mutations, 1);
    EXPECT_EQ(snapshot.last_mutation_phase, CommandBufferMutationPhase::LaterResume);
    EXPECT_EQ(snapshot.last_logical_word_offset, 2);
    EXPECT_EQ(snapshot.last_remaining_words, 3);
}

TEST(CommandBufferLifetimeDiagnostic, LaterResumeComparesAgainstTheSubmitBaseline) {
    CommandBufferLifetimeDiagnostic diagnostic{
        {.enabled = true, .max_words_per_signature = 16, .max_resume_checks = 4}};
    std::array<u32, 5> words{1, 2, 3, 4, 5};
    auto probe = diagnostic.Begin(CommandBufferKind::TopLevelDcb, words);
    probe.ObserveInitial(words);

    words[0] = 66;
    const auto remaining = std::span<const u32>{words}.subspan(3);
    probe.Suspend(remaining);
    probe.Resume(remaining);

    const auto snapshot = diagnostic.Read();
    EXPECT_EQ(snapshot.later_resume_mutations, 1);
    EXPECT_EQ(snapshot.last_mutation_phase, CommandBufferMutationPhase::LaterResume);
    EXPECT_EQ(snapshot.last_logical_word_offset, 3);
    EXPECT_EQ(snapshot.last_remaining_words, 2);
}

TEST(CommandBufferLifetimeDiagnostic, FinalCheckCatchesMutationOfAlreadyConsumedPrefix) {
    CommandBufferLifetimeDiagnostic diagnostic{
        {.enabled = true, .max_words_per_signature = 16, .max_resume_checks = 4}};
    std::array<u32, 5> words{1, 2, 3, 4, 5};
    auto probe = diagnostic.Begin(CommandBufferKind::TopLevelDcb, words);
    probe.ObserveInitial(words);
    const auto remaining = std::span<const u32>{words}.subspan(3);
    probe.Suspend(remaining);
    probe.Resume(remaining);

    words[0] = 88;
    probe.ObserveFinal();

    const auto snapshot = diagnostic.Read();
    EXPECT_EQ(snapshot.later_resume_mutations, 0);
    EXPECT_EQ(snapshot.final_mutations, 1);
    EXPECT_EQ(snapshot.last_mutation_phase, CommandBufferMutationPhase::Final);
}

TEST(CommandBufferLifetimeDiagnostic, ResumeChecksHaveAnExplicitFixedCapacity) {
    CommandBufferLifetimeDiagnostic diagnostic{
        {.enabled = true, .max_words_per_signature = 16, .max_resume_checks = 1}};
    std::array<u32, 4> words{1, 2, 3, 4};
    auto probe = diagnostic.Begin(CommandBufferKind::TopLevelDcb, words);
    probe.ObserveInitial(words);

    const auto remaining = std::span<const u32>{words}.subspan(1);
    probe.Suspend(remaining);
    probe.Resume(remaining);
    probe.Suspend(remaining);
    probe.Resume(remaining);

    const auto snapshot = diagnostic.Read();
    EXPECT_EQ(snapshot.resume_checks, 1);
    EXPECT_EQ(snapshot.resume_check_capacity_loss, 1);
}

TEST(CommandBufferLifetimeDiagnostic, OversizedAndDisabledBuffersAreNeverSampled) {
    std::array<u32, 4> words{1, 2, 3, 4};
    CommandBufferLifetimeDiagnostic bounded{
        {.enabled = true, .max_words_per_signature = 3, .max_resume_checks = 4}};
    EXPECT_FALSE(bounded.Begin(CommandBufferKind::TopLevelCcb, words).Active());
    EXPECT_EQ(bounded.Read().oversized_buffers, 1);
    EXPECT_EQ(bounded.Read().observed_buffers, 1);
    EXPECT_EQ(bounded.Read().selected_buffers, 1);
    EXPECT_EQ(bounded.Read().hashed_bytes, 0);

    CommandBufferLifetimeDiagnostic disabled{
        {.enabled = false, .max_words_per_signature = 16, .max_resume_checks = 4}};
    EXPECT_FALSE(disabled.Begin(CommandBufferKind::IndirectDcb, words).Active());
    EXPECT_EQ(disabled.Read().observed_buffers, 0);
    EXPECT_EQ(disabled.Read().oversized_buffers, 0);
}

TEST(CommandBufferLifetimeDiagnostic, MutationReportsRetainSemanticBufferKind) {
    CommandBufferLifetimeDiagnostic diagnostic{
        {.enabled = true, .max_words_per_signature = 16, .max_resume_checks = 4}};
    std::array<u32, 3> words{1, 2, 3};
    auto probe = diagnostic.Begin(CommandBufferKind::IndirectCcb, words);
    probe.ObserveInitial(words);
    probe.Suspend(words);
    words[1] = 44;
    probe.Resume(words);

    const auto snapshot = diagnostic.Read();
    EXPECT_EQ(snapshot.last_buffer_kind, CommandBufferKind::IndirectCcb);
    EXPECT_EQ(snapshot.last_buffer_ordinal, 1);
}

TEST(CommandBufferLifetimeDiagnostic, BuffersBeforeTheSelectedOrdinalWindowAreCountedNotHashed) {
    CommandBufferLifetimeDiagnostic diagnostic{{.enabled = true,
                                                .minimum_buffer_ordinal = 3,
                                                .selected_buffer_count = 2,
                                                .total_hash_byte_budget = 1024,
                                                .max_words_per_signature = 16,
                                                .max_resume_checks = 4}};
    std::array<u32, 4> words{1, 2, 3, 4};

    EXPECT_FALSE(diagnostic.Begin(CommandBufferKind::TopLevelDcb, words).Active());
    EXPECT_FALSE(diagnostic.Begin(CommandBufferKind::IndirectDcb, words).Active());
    auto selected = diagnostic.Begin(CommandBufferKind::TopLevelCcb, words);

    const auto snapshot = diagnostic.Read();
    EXPECT_TRUE(selected.Active());
    EXPECT_EQ(snapshot.observed_buffers, 3);
    EXPECT_EQ(snapshot.last_observed_buffer_ordinal, 3);
    EXPECT_EQ(snapshot.pre_window_buffers, 2);
    EXPECT_EQ(snapshot.selected_buffers, 1);
    EXPECT_EQ(snapshot.hashed_bytes, words.size() * sizeof(u32));
}

TEST(CommandBufferLifetimeDiagnostic, SelectedWindowHasAConfiguredAndHardCapacity) {
    constexpr u32 requested = CommandBufferLifetimeDiagnostic::HardMaxSelectedBuffers + 10;
    CommandBufferLifetimeDiagnostic diagnostic{{.enabled = true,
                                                .selected_buffer_count = requested,
                                                .total_hash_byte_budget = 4096,
                                                .max_words_per_signature = 16,
                                                .max_resume_checks = 4}};
    std::array<u32, 1> words{1};
    for (u32 i = 0; i < CommandBufferLifetimeDiagnostic::HardMaxSelectedBuffers; ++i) {
        EXPECT_TRUE(diagnostic.Begin(CommandBufferKind::TopLevelDcb, words).Active());
    }
    EXPECT_FALSE(diagnostic.Begin(CommandBufferKind::TopLevelDcb, words).Active());

    const auto snapshot = diagnostic.Read();
    EXPECT_EQ(snapshot.selected_buffer_limit,
              CommandBufferLifetimeDiagnostic::HardMaxSelectedBuffers);
    EXPECT_EQ(snapshot.selected_buffers, CommandBufferLifetimeDiagnostic::HardMaxSelectedBuffers);
    EXPECT_EQ(snapshot.selection_capacity_loss, 1);
}

TEST(CommandBufferLifetimeDiagnostic, GlobalByteBudgetReportsTheExactPhaseThatLostCoverage) {
    CommandBufferLifetimeDiagnostic diagnostic{{.enabled = true,
                                                .selected_buffer_count = 1,
                                                .total_hash_byte_budget = 32,
                                                .max_words_per_signature = 16,
                                                .max_resume_checks = 4}};
    std::array<u32, 4> words{1, 2, 3, 4};
    auto probe = diagnostic.Begin(CommandBufferKind::TopLevelDcb, words);
    probe.ObserveInitial(words);
    probe.Suspend(words);
    probe.Resume(words);

    const auto snapshot = diagnostic.Read();
    EXPECT_EQ(snapshot.hashed_bytes, 32);
    EXPECT_EQ(snapshot.submit_hashes, 1);
    EXPECT_EQ(snapshot.initial_hashes, 1);
    EXPECT_EQ(snapshot.later_resume_hashes, 0);
    EXPECT_EQ(snapshot.later_resume_budget_exhaustions, 1);
}

TEST(CommandBufferLifetimeDiagnostic, CountOnlyModeAssignsOrdinalsWithoutHashing) {
    CommandBufferLifetimeDiagnostic diagnostic{{.enabled = true,
                                                .count_only = true,
                                                .selected_buffer_count = 4,
                                                .total_hash_byte_budget = 1024,
                                                .max_words_per_signature = 16,
                                                .max_resume_checks = 4}};
    std::array<u32, 3> words{1, 2, 3};

    EXPECT_FALSE(diagnostic.Begin(CommandBufferKind::TopLevelDcb, words).Active());
    EXPECT_FALSE(diagnostic.Begin(CommandBufferKind::IndirectCcb, words).Active());

    const auto snapshot = diagnostic.Read();
    EXPECT_TRUE(snapshot.count_only_mode);
    EXPECT_EQ(snapshot.observed_buffers, 2);
    EXPECT_EQ(snapshot.last_observed_buffer_ordinal, 2);
    EXPECT_EQ(snapshot.count_only_buffers, 2);
    EXPECT_EQ(snapshot.selected_buffers, 0);
    EXPECT_EQ(snapshot.hashed_bytes, 0);
}

TEST(CommandBufferLifetimeDiagnostic, FreezeProducesAStableSnapshotAndStopsEveryCounter) {
    CommandBufferLifetimeDiagnostic diagnostic{{.enabled = true,
                                                .selected_buffer_count = 2,
                                                .total_hash_byte_budget = 1024,
                                                .max_words_per_signature = 16,
                                                .max_resume_checks = 4}};
    std::array<u32, 3> words{1, 2, 3};
    auto probe = diagnostic.Begin(CommandBufferKind::TopLevelDcb, words);
    probe.ObserveInitial(words);

    const auto frozen = diagnostic.FreezeAndRead();
    ASSERT_TRUE(frozen.frozen);
    ASSERT_TRUE(frozen.stable);

    words[0] = 99;
    probe.Suspend(words);
    probe.Resume(words);
    probe.ObserveFinal();
    EXPECT_FALSE(diagnostic.Begin(CommandBufferKind::TopLevelCcb, words).Active());

    const auto after = diagnostic.Read();
    EXPECT_TRUE(after.frozen);
    EXPECT_TRUE(after.stable);
    EXPECT_EQ(after.observed_buffers, frozen.observed_buffers);
    EXPECT_EQ(after.hashed_bytes, frozen.hashed_bytes);
    EXPECT_EQ(after.resume_checks, frozen.resume_checks);
    EXPECT_EQ(after.later_resume_mutations, frozen.later_resume_mutations);
    EXPECT_EQ(after.final_mutations, frozen.final_mutations);
}

} // namespace
} // namespace AmdGpu
