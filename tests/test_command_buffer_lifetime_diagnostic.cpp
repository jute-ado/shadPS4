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
    EXPECT_EQ(bounded.Read().observed_buffers, 0);

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

} // namespace
} // namespace AmdGpu
