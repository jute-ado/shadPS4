// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/attachment_publication_diagnostic.h"

namespace {

using Vulkan::AttachmentPublicationDiagnostic;
using Vulkan::AttachmentSubresource;

constexpr AttachmentSubresource WholeImage{0, 4, 0, 6};
constexpr AttachmentSubresource FirstMip{0, 1, 0, 6};
constexpr AttachmentSubresource SecondMip{1, 1, 0, 6};

TEST(AttachmentPublicationDiagnostic, AcceptsCompleteAttachmentToSampleLifecycle) {
    AttachmentPublicationDiagnostic diagnostic;
    constexpr std::array targets{Vulkan::AttachmentTarget{7, WholeImage}};

    diagnostic.BeginScope(targets);
    diagnostic.RecordDrawIssued();
    diagnostic.EndScope();
    diagnostic.RecordBarrier(7, WholeImage);
    diagnostic.RecordSample(7, FirstMip);

    const auto snapshot = diagnostic.TakeSnapshot();
    EXPECT_EQ(snapshot.samples, 1);
    EXPECT_EQ(snapshot.attachment_samples, 1);
    EXPECT_EQ(snapshot.without_issued_producer, 0);
    EXPECT_EQ(snapshot.before_scope_end, 0);
    EXPECT_EQ(snapshot.without_covering_barrier, 0);
    EXPECT_EQ(snapshot.after_destructive_write, 0);
}

TEST(AttachmentPublicationDiagnostic, DistinguishesIntentFromAnIssuedDraw) {
    AttachmentPublicationDiagnostic diagnostic;
    constexpr std::array targets{Vulkan::AttachmentTarget{7, WholeImage}};

    diagnostic.BeginScope(targets);
    diagnostic.EndScope();
    diagnostic.RecordBarrier(7, WholeImage);
    diagnostic.RecordSample(7, FirstMip);

    EXPECT_EQ(diagnostic.TakeSnapshot().without_issued_producer, 1);
}

TEST(AttachmentPublicationDiagnostic, RejectsSamplingBeforeRenderScopeEnds) {
    AttachmentPublicationDiagnostic diagnostic;
    constexpr std::array targets{Vulkan::AttachmentTarget{7, WholeImage}};

    diagnostic.BeginScope(targets);
    diagnostic.RecordDrawIssued();
    diagnostic.RecordSample(7, FirstMip);

    const auto snapshot = diagnostic.TakeSnapshot();
    EXPECT_EQ(snapshot.before_scope_end, 1);
    EXPECT_EQ(snapshot.without_covering_barrier, 1);
}

TEST(AttachmentPublicationDiagnostic, RequiresBarrierToCoverSampledSubresource) {
    AttachmentPublicationDiagnostic diagnostic;
    constexpr std::array targets{Vulkan::AttachmentTarget{7, WholeImage}};

    diagnostic.BeginScope(targets);
    diagnostic.RecordDrawIssued();
    diagnostic.EndScope();
    diagnostic.RecordBarrier(7, SecondMip);
    diagnostic.RecordSample(7, FirstMip);

    EXPECT_EQ(diagnostic.TakeSnapshot().without_covering_barrier, 1);
}

TEST(AttachmentPublicationDiagnostic, DetectsMutationAfterAttachmentPublication) {
    AttachmentPublicationDiagnostic diagnostic;
    constexpr std::array targets{Vulkan::AttachmentTarget{7, WholeImage}};

    diagnostic.BeginScope(targets);
    diagnostic.RecordDrawIssued();
    diagnostic.EndScope();
    diagnostic.RecordBarrier(7, WholeImage);
    diagnostic.RecordDestructiveWrite(7, FirstMip);
    diagnostic.RecordSample(7, FirstMip);

    const auto snapshot = diagnostic.TakeSnapshot();
    EXPECT_EQ(snapshot.after_destructive_write, 1);
    EXPECT_EQ(snapshot.without_covering_barrier, 1);
}

TEST(AttachmentPublicationDiagnostic, IgnoresOrdinaryTextureSamples) {
    AttachmentPublicationDiagnostic diagnostic;

    diagnostic.RecordSample(99, WholeImage);

    const auto snapshot = diagnostic.TakeSnapshot();
    EXPECT_EQ(snapshot.samples, 1);
    EXPECT_EQ(snapshot.attachment_samples, 0);
    EXPECT_EQ(snapshot.without_issued_producer, 0);
    EXPECT_EQ(snapshot.before_scope_end, 0);
    EXPECT_EQ(snapshot.without_covering_barrier, 0);
    EXPECT_EQ(snapshot.after_destructive_write, 0);
}

TEST(AttachmentPublicationDiagnostic, KeepsRepeatedDrawsInsideOneScopeValid) {
    AttachmentPublicationDiagnostic diagnostic;
    constexpr std::array targets{Vulkan::AttachmentTarget{7, WholeImage}};

    diagnostic.BeginScope(targets);
    diagnostic.RecordDrawIssued();
    diagnostic.RecordDrawIssued();
    diagnostic.EndScope();
    diagnostic.RecordBarrier(7, WholeImage);
    diagnostic.RecordSample(7, FirstMip);

    const auto snapshot = diagnostic.TakeSnapshot();
    EXPECT_EQ(snapshot.draws_issued, 2);
    EXPECT_EQ(snapshot.without_issued_producer, 0);
    EXPECT_EQ(snapshot.before_scope_end, 0);
    EXPECT_EQ(snapshot.without_covering_barrier, 0);
}

} // namespace
