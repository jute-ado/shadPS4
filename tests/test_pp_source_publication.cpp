// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/host_passes/pp_source_producer_scope.h"
#include "video_core/renderer_vulkan/host_passes/pp_source_publication.h"
#include "video_core/texture_cache/image_color_scope_producer.h"
#include "video_core/texture_cache/image_producer.h"

namespace {

using namespace Vulkan::HostPasses;

TEST(PpSourcePublication, ImageRefreshBranchesExposeTheirActualMutationOutcome) {
    const auto clean = VideoCore::PlanImageRefreshStart(false, 1);
    EXPECT_FALSE(clean.refresh);
    EXPECT_EQ(clean.result, VideoCore::ImageRefreshResult::Clean);

    const auto multisampled = VideoCore::PlanImageRefreshStart(true, 4);
    EXPECT_FALSE(multisampled.refresh);
    EXPECT_EQ(multisampled.result, VideoCore::ImageRefreshResult::MultisampledDirty);

    const auto dirty = VideoCore::PlanImageRefreshStart(true, 1);
    EXPECT_TRUE(dirty.refresh);
    EXPECT_EQ(VideoCore::CompleteImageRefresh(true),
              VideoCore::ImageRefreshResult::GpuModifiedUnchanged);
    EXPECT_EQ(VideoCore::CompleteImageRefresh(false), VideoCore::ImageRefreshResult::Uploaded);
}

TEST(PpSourcePublication, ClassifiesWhetherFlipRefreshMutatedTheSource) {
    EXPECT_EQ(ClassifyPpSourcePublication(
                  {.refresh = VideoCore::ImageRefreshResult::Clean, .gpu_modified_before = true}),
              PpSourcePublicationClass::CleanGpuTracked);
    EXPECT_EQ(ClassifyPpSourcePublication(
                  {.refresh = VideoCore::ImageRefreshResult::Clean, .gpu_modified_before = false}),
              PpSourcePublicationClass::CleanResident);
    EXPECT_EQ(ClassifyPpSourcePublication(
                  {.refresh = VideoCore::ImageRefreshResult::MaybeCpuDirtyUnchanged,
                   .gpu_modified_before = true}),
              PpSourcePublicationClass::DirtyNoUpload);
    EXPECT_EQ(
        ClassifyPpSourcePublication({.refresh = VideoCore::ImageRefreshResult::GpuModifiedUnchanged,
                                     .gpu_modified_before = true}),
        PpSourcePublicationClass::DirtyNoUpload);
    EXPECT_EQ(ClassifyPpSourcePublication({.refresh = VideoCore::ImageRefreshResult::Uploaded,
                                           .gpu_modified_before = true}),
              PpSourcePublicationClass::CpuUpload);
    EXPECT_EQ(
        ClassifyPpSourcePublication({.refresh = VideoCore::ImageRefreshResult::MultisampledDirty,
                                     .gpu_modified_before = true}),
        PpSourcePublicationClass::Unsupported);
}

TEST(PpSourcePublication, ReportsEverySelectedSequenceExactlyOnce) {
    PpSourcePublicationCoverage coverage{{.start = 4000, .count = 4}};
    const auto first = coverage.Observe(3999, PpSourcePublicationClass::CleanResident);
    EXPECT_FALSE(first.has_value());

    ASSERT_TRUE(coverage.Observe(4000, PpSourcePublicationClass::CleanGpuTracked));
    ASSERT_TRUE(coverage.Observe(4001, PpSourcePublicationClass::CleanGpuTracked));
    ASSERT_TRUE(coverage.Observe(4002, PpSourcePublicationClass::DirtyNoUpload));
    const auto final = coverage.Observe(4003, PpSourcePublicationClass::CpuUpload);
    ASSERT_TRUE(final);
    EXPECT_TRUE(final->final);
    EXPECT_EQ(final->sequence, 4003u);
    EXPECT_EQ(final->selected, 4u);
    EXPECT_EQ(final->emitted, 4u);
    EXPECT_EQ(final->clean_gpu_tracked, 2u);
    EXPECT_EQ(final->clean_resident, 0u);
    EXPECT_EQ(final->dirty_no_upload, 1u);
    EXPECT_EQ(final->cpu_upload, 1u);
    EXPECT_EQ(final->unsupported, 0u);
    EXPECT_EQ(final->loss, 0u);
}

TEST(PpSourcePublication, GapsAndDuplicatesFailClosed) {
    PpSourcePublicationCoverage coverage{{.start = 10, .count = 3}};
    ASSERT_TRUE(coverage.Observe(10, PpSourcePublicationClass::CleanGpuTracked));
    ASSERT_TRUE(coverage.Observe(10, PpSourcePublicationClass::CleanGpuTracked));
    const auto final = coverage.Observe(12, PpSourcePublicationClass::Unsupported);
    ASSERT_TRUE(final);
    EXPECT_EQ(final->selected, 3u);
    EXPECT_EQ(final->emitted, 3u);
    EXPECT_EQ(final->unsupported, 1u);
    EXPECT_EQ(final->loss, 2u);
}

TEST(PpSourcePublication, MissingWindowPrefixFailsFinalCoverage) {
    PpSourcePublicationCoverage coverage{{.start = 10, .count = 3}};
    ASSERT_TRUE(coverage.Observe(11, PpSourcePublicationClass::CleanGpuTracked));
    const auto final = coverage.Observe(12, PpSourcePublicationClass::CleanGpuTracked);
    ASSERT_TRUE(final);
    EXPECT_TRUE(final->final);
    EXPECT_EQ(final->expected, 3u);
    EXPECT_EQ(final->selected, 2u);
    EXPECT_EQ(final->emitted, 2u);
    EXPECT_EQ(final->loss, 1u);
}

TEST(PpSourcePublication, CompactOutputContainsOnlySequenceClassAndCounts) {
    const auto line = FormatPpSourcePublicationObservation(
        {.sequence = 4579, .classification = PpSourcePublicationClass::CleanGpuTracked});
    EXPECT_EQ(line, "FGSCP s=4579 r=0");

    PpSourcePublicationCoverage coverage{{.start = 1, .count = 1}};
    const auto final = coverage.Observe(1, PpSourcePublicationClass::CleanResident);
    ASSERT_TRUE(final);
    const auto coverage_line = FormatPpSourcePublicationCoverage(*final);
    EXPECT_EQ(coverage_line, "FGSCPC s=1 n=1/1/1 g=0 r=1 d=0 u=0 x=0 l=0");
    EXPECT_LT(line.size(), size_t{40});
    EXPECT_LT(coverage_line.size(), size_t{100});
}

TEST(PpSourceProducer, RecordsOnlySuccessfullyEncodedWriterClasses) {
    VideoCore::ImageProducerState state;
    EXPECT_EQ(state.Observe(), (VideoCore::ImageProducerObservation{
                                   .classification = VideoCore::ImageProducerClass::Unknown,
                                   .produced_since_last_observation = false,
                               }));

    state.Mark(VideoCore::ImageProducerClass::ColorAttachment);
    EXPECT_EQ(state.Observe(), (VideoCore::ImageProducerObservation{
                                   .classification = VideoCore::ImageProducerClass::ColorAttachment,
                                   .produced_since_last_observation = true,
                               }));
    EXPECT_EQ(state.Observe(), (VideoCore::ImageProducerObservation{
                                   .classification = VideoCore::ImageProducerClass::ColorAttachment,
                                   .produced_since_last_observation = false,
                               }));

    state.Mark(VideoCore::ImageProducerClass::StorageImage);
    state.Mark(VideoCore::ImageProducerClass::Transfer);
    EXPECT_EQ(state.Observe(), (VideoCore::ImageProducerObservation{
                                   .classification = VideoCore::ImageProducerClass::Transfer,
                                   .produced_since_last_observation = true,
                               }));
}

TEST(PpSourceProducer, ResetAndGenerationChangesFailClosed) {
    VideoCore::ImageProducerState state;
    state.Mark(VideoCore::ImageProducerClass::ColorAttachment);
    ASSERT_TRUE(state.Observe().produced_since_last_observation);
    state.Reset();
    EXPECT_EQ(state.Observe(), (VideoCore::ImageProducerObservation{
                                   .classification = VideoCore::ImageProducerClass::Unknown,
                                   .produced_since_last_observation = false,
                               }));
}

TEST(PpSourceProducer, CompactProducerOutputDoesNotExposeImageIdentity) {
    const auto line = FormatPpSourceProducerObservation(
        {.sequence = 4579,
         .producer = {.classification = VideoCore::ImageProducerClass::ColorAttachment,
                      .produced_since_last_observation = true}});
    EXPECT_EQ(line, "FGSCPR s=4579 r=1 n=1");
    EXPECT_LT(line.size(), size_t{40});
}

TEST(PpSourceProducer, CoverageCountsEveryFlipAndSeparatesNewProductionFromReuse) {
    PpSourceProducerCoverage coverage{{.start = 4000, .count = 2}};
    EXPECT_TRUE(
        coverage.Observe(4000, {.classification = VideoCore::ImageProducerClass::ColorAttachment,
                                .produced_since_last_observation = true}));
    const auto final =
        coverage.Observe(4001, {.classification = VideoCore::ImageProducerClass::ColorAttachment,
                                .produced_since_last_observation = false});
    ASSERT_TRUE(final);
    EXPECT_TRUE(final->final);
    EXPECT_EQ(final->expected, 2u);
    EXPECT_EQ(final->selected, 2u);
    EXPECT_EQ(final->emitted, 2u);
    EXPECT_EQ(final->color_attachment, 2u);
    EXPECT_EQ(final->new_production, 1u);
    EXPECT_EQ(final->reused_production, 1u);
    EXPECT_EQ(final->loss, 0u);
    EXPECT_EQ(FormatPpSourceProducerCoverage(*final),
              "FGSCPRC s=4001 n=2/2/2 c=2 s=0 t=0 u=0 x=0 p=1 r=1 l=0");
}

TEST(PpSourceProducerScope, DistinguishesFlipEndedScopeFromAlreadyEndedProducer) {
    EXPECT_EQ(ClassifyPpSourceProducerScope(true), PpSourceProducerScopeClass::ActiveAtFlip);
    EXPECT_EQ(ClassifyPpSourceProducerScope(false), PpSourceProducerScopeClass::EndedEarlier);
}

TEST(PpSourceProducerScope, CoverageIsExactAndPrivacySafe) {
    PpSourceProducerScopeCoverage coverage{{.start = 4000, .count = 2}};
    ASSERT_TRUE(coverage.Observe(4000, PpSourceProducerScopeClass::ActiveAtFlip,
                                 {.draw_count = 3,
                                  .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
                                  .clear_at_begin = true,
                                  .valid = true}));
    const auto final = coverage.Observe(4001, PpSourceProducerScopeClass::EndedEarlier,
                                        {.draw_count = 2,
                                         .last_draw = VideoCore::ImageColorScopeDrawKind::Indirect,
                                         .valid = true});
    ASSERT_TRUE(final);
    EXPECT_TRUE(final->final);
    EXPECT_EQ(final->active_at_flip, 1u);
    EXPECT_EQ(final->ended_earlier, 1u);
    EXPECT_EQ(final->valid_draw_scopes, 2u);
    EXPECT_EQ(final->invalid_draw_scopes, 0u);
    EXPECT_EQ(final->overflow_draw_scopes, 0u);
    EXPECT_EQ(final->loss, 0u);
    EXPECT_EQ(FormatPpSourceProducerScopeObservation(
                  {.sequence = 4001,
                   .classification = PpSourceProducerScopeClass::EndedEarlier,
                   .draw_scope = {.draw_count = 2,
                                  .last_draw = VideoCore::ImageColorScopeDrawKind::Indirect,
                                  .valid = true}}),
              "FGSCPS s=4001 r=1 d=2 k=2 c=0 x=0 j=0 e=0 n=0 b=0 u=0 w=0 ip=0 in=0 "
              "ia=0 iv=0 sd=0 sk=0 sc=0 sx=0 sj=0 se=0 sn=0 su=0 sw=0 np=0 nn=0 na=0 nv=0");
    EXPECT_EQ(FormatPpSourceProducerScopeCoverage(*final),
              "FGSCPSC s=4001 n=2/2/2 a=1 e=1 v=2 i=0 x=0 s=0 z=0 m=0 w=0 ic=0 "
              "is=0 it=0 iu=0 ix=0 if=0 ir=0 ia=0 il=0 sv=0 si=0 sx=0 sz=0 ss=0 "
              "sm=0 sd=0 sn=0 sc=0 dc=0 ds=0 dt=0 du=0 dx=0 df=0 dr=0 da=0 dl=0 l=0");
}

TEST(PpSourceColorScopeDraw, CountsOnlyEncodedDrawsInTheCurrentScope) {
    VideoCore::ImageColorScopeProducerState state;
    state.BeginScope(11, true);
    state.MarkDraw(11, {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                        .indexed = true,
                        .element_count = 36,
                        .instance_count = 2,
                        .sampled_bindings = 2,
                        .sampled_images = 1,
                        .sampled_input_producer = VideoCore::ImageProducerClass::ColorAttachment,
                        .sampled_input_fresh = true,
                        .sampled_input_valid = true});
    state.MarkDraw(11, {.kind = VideoCore::ImageColorScopeDrawKind::Indirect,
                        .sampled_bindings = 3,
                        .sampled_images = 2,
                        .storage_writes = 1});

    EXPECT_EQ(state.Observe(), (VideoCore::ImageColorScopeProducerObservation{
                                   .draw_count = 2,
                                   .last_draw = VideoCore::ImageColorScopeDrawKind::Indirect,
                                   .indexed = false,
                                   .element_count = 0,
                                   .instance_count = 0,
                                   .sampled_bindings = 3,
                                   .sampled_images = 2,
                                   .storage_writes = 1,
                                   .sampled_input_producer = VideoCore::ImageProducerClass::Unknown,
                                   .sampled_input_fresh = false,
                                   .sampled_input_alias = false,
                                   .sampled_input_valid = false,
                                   .clear_at_begin = true,
                                   .valid = true,
                               }));

    state.BeginScope(12, false);
    EXPECT_EQ(state.Observe(), (VideoCore::ImageColorScopeProducerObservation{
                                   .draw_count = 0,
                                   .last_draw = VideoCore::ImageColorScopeDrawKind::Unknown,
                                   .clear_at_begin = false,
                                   .valid = true,
                               }));
}

TEST(PpSourceColorScopeDraw, StaleScopeAndCapacityOverflowFailClosed) {
    VideoCore::ImageColorScopeProducerState state;
    state.BeginScope(21, false);
    state.MarkDraw(20, {.kind = VideoCore::ImageColorScopeDrawKind::Direct});
    EXPECT_FALSE(state.Observe().valid);

    state.BeginScope(22, false);
    for (u32 index = 0; index <= VideoCore::ImageColorScopeProducerState::MaxTrackedDraws;
         ++index) {
        state.MarkDraw(22, {.kind = VideoCore::ImageColorScopeDrawKind::Direct});
    }
    const auto overflow = state.Observe();
    EXPECT_EQ(overflow.draw_count, VideoCore::ImageColorScopeProducerState::MaxTrackedDraws);
    EXPECT_TRUE(overflow.overflow);
    EXPECT_FALSE(overflow.valid);

    state.Reset();
    EXPECT_EQ(state.Observe(), VideoCore::ImageColorScopeProducerObservation{});
}

TEST(PpSourceColorScopeDraw, InvalidDrawShapeFailsClosedWithoutIdentity) {
    VideoCore::ImageColorScopeProducerState state;
    state.BeginScope(31, false);
    state.MarkDraw(31, {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                        .sampled_bindings = 1,
                        .sampled_images = 2});
    EXPECT_FALSE(state.Observe().valid);

    state.BeginScope(32, false);
    state.MarkDraw(32, {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                        .sampled_bindings =
                            VideoCore::ImageColorScopeProducerState::MaxTrackedImageBindings + 1,
                        .sampled_images = 1});
    EXPECT_FALSE(state.Observe().valid);
}

TEST(PpSourceColorScopeDraw, CompactOutputAndCoverageClassifySingleSampledInput) {
    PpSourceProducerScopeCoverage coverage{{.start = 50, .count = 2}};
    ASSERT_TRUE(coverage.Observe(
        50, PpSourceProducerScopeClass::ActiveAtFlip,
        {.draw_count = 1,
         .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
         .element_count = 3,
         .instance_count = 1,
         .sampled_bindings = 1,
         .sampled_images = 1,
         .sampled_input_producer = VideoCore::ImageProducerClass::ColorAttachment,
         .sampled_input_fresh = true,
         .sampled_input_valid = true,
         .sampled_input_scope_draw_count = 1,
         .sampled_input_scope_last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
         .sampled_input_scope_valid = true,
         .valid = true}));
    const auto final = coverage.Observe(51, PpSourceProducerScopeClass::ActiveAtFlip,
                                        {.draw_count = 1,
                                         .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
                                         .indexed = true,
                                         .element_count = 6,
                                         .instance_count = 1,
                                         .sampled_bindings = 2,
                                         .sampled_images = 2,
                                         .storage_writes = 1,
                                         .sampled_input_valid = false,
                                         .valid = true});
    ASSERT_TRUE(final);
    EXPECT_EQ(final->single_sampled_input, 1u);
    EXPECT_EQ(final->multiple_sampled_inputs, 1u);
    EXPECT_EQ(final->writable_image_draws, 1u);
    EXPECT_EQ(final->input_color_attachment, 1u);
    EXPECT_EQ(final->input_storage_image, 0u);
    EXPECT_EQ(final->input_transfer, 0u);
    EXPECT_EQ(final->input_cpu_upload, 0u);
    EXPECT_EQ(final->input_unknown, 0u);
    EXPECT_EQ(final->input_fresh, 1u);
    EXPECT_EQ(final->input_reused, 0u);
    EXPECT_EQ(final->input_alias, 0u);
    EXPECT_EQ(final->invalid_single_input, 0u);
    EXPECT_EQ(FormatPpSourceProducerScopeObservation(*final),
              "FGSCPS s=51 r=0 d=1 k=1 c=0 x=0 j=1 e=6 n=1 b=2 u=2 w=1 ip=0 in=0 "
              "ia=0 iv=0 sd=0 sk=0 sc=0 sx=0 sj=0 se=0 sn=0 su=0 sw=0 np=0 nn=0 na=0 nv=0");
    EXPECT_EQ(FormatPpSourceProducerScopeCoverage(*final),
              "FGSCPSC s=51 n=2/2/2 a=2 e=0 v=2 i=0 x=0 s=1 z=0 m=1 w=1 ic=1 "
              "is=0 it=0 iu=0 ix=0 if=1 ir=0 ia=0 il=0 sv=1 si=0 sx=0 sz=0 ss=1 "
              "sm=0 sd=1 sn=0 sc=0 dc=0 ds=0 dt=0 du=0 dx=0 df=0 dr=0 da=0 dl=0 l=0");
}

TEST(PpSourceProducer, FlipAndSampledInputObserversHaveIndependentFreshnessEpochs) {
    VideoCore::ImageProducerState state;
    state.Mark(VideoCore::ImageProducerClass::ColorAttachment);
    EXPECT_TRUE(state.Observe().produced_since_last_observation);
    EXPECT_TRUE(state.ObserveSampledInput().produced_since_last_observation);
    EXPECT_FALSE(state.Observe().produced_since_last_observation);
    EXPECT_FALSE(state.ObserveSampledInput().produced_since_last_observation);

    state.Mark(VideoCore::ImageProducerClass::Transfer);
    EXPECT_TRUE(state.ObserveSampledInput().produced_since_last_observation);
    EXPECT_TRUE(state.Observe().produced_since_last_observation);
    state.Reset();
    EXPECT_FALSE(state.ObserveSampledInput().produced_since_last_observation);
}

TEST(PpSourceColorScopeDraw, SingleSampledInputProducerClassesAreBoundedAndFailClosed) {
    PpSourceProducerScopeCoverage coverage{{.start = 70, .count = 4}};
    ASSERT_TRUE(coverage.Observe(70, PpSourceProducerScopeClass::ActiveAtFlip,
                                 {.draw_count = 1,
                                  .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
                                  .sampled_bindings = 1,
                                  .sampled_images = 1,
                                  .sampled_input_producer = VideoCore::ImageProducerClass::Transfer,
                                  .sampled_input_valid = true,
                                  .valid = true}));
    ASSERT_TRUE(
        coverage.Observe(71, PpSourceProducerScopeClass::ActiveAtFlip,
                         {.draw_count = 1,
                          .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
                          .sampled_bindings = 1,
                          .sampled_images = 1,
                          .sampled_input_producer = VideoCore::ImageProducerClass::CpuUpload,
                          .sampled_input_fresh = true,
                          .sampled_input_valid = true,
                          .valid = true}));
    ASSERT_TRUE(
        coverage.Observe(72, PpSourceProducerScopeClass::ActiveAtFlip,
                         {.draw_count = 1,
                          .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
                          .sampled_bindings = 1,
                          .sampled_images = 1,
                          .sampled_input_producer = VideoCore::ImageProducerClass::StorageImage,
                          .sampled_input_alias = true,
                          .sampled_input_valid = true,
                          .valid = true}));
    const auto final =
        coverage.Observe(73, PpSourceProducerScopeClass::ActiveAtFlip,
                         {.draw_count = 1,
                          .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
                          .sampled_bindings = 1,
                          .sampled_images = 1,
                          .sampled_input_producer = VideoCore::ImageProducerClass::Unknown,
                          .sampled_input_valid = false,
                          .valid = true});
    ASSERT_TRUE(final);
    EXPECT_EQ(final->input_transfer, 1u);
    EXPECT_EQ(final->input_cpu_upload, 1u);
    EXPECT_EQ(final->input_storage_image, 1u);
    EXPECT_EQ(final->input_unknown, 0u);
    EXPECT_EQ(final->input_fresh, 1u);
    EXPECT_EQ(final->input_reused, 2u);
    EXPECT_EQ(final->input_alias, 1u);
    EXPECT_EQ(final->invalid_single_input, 1u);
    EXPECT_EQ(final->loss, 1u);
}

TEST(PpSourceColorScopeDraw, RetainsSampledInputColorScopeWithoutIdentity) {
    VideoCore::ImageColorScopeProducerState state;
    state.BeginScope(81, false);
    state.MarkDraw(
        81, {
                .kind = VideoCore::ImageColorScopeDrawKind::Direct,
                .indexed = true,
                .element_count = 4,
                .instance_count = 1,
                .sampled_bindings = 1,
                .sampled_images = 1,
                .sampled_input_producer = VideoCore::ImageProducerClass::ColorAttachment,
                .sampled_input_fresh = true,
                .sampled_input_valid = true,
                .sampled_input_scope_draw_count = 2,
                .sampled_input_scope_last_draw = VideoCore::ImageColorScopeDrawKind::Indirect,
                .sampled_input_scope_indexed = true,
                .sampled_input_scope_element_count = 18,
                .sampled_input_scope_instance_count = 2,
                .sampled_input_scope_sampled_images = 3,
                .sampled_input_scope_storage_writes = 1,
                .sampled_input_scope_clear_at_begin = true,
                .sampled_input_scope_valid = true,
                .sampled_input_scope_input_producer = VideoCore::ImageProducerClass::Transfer,
                .sampled_input_scope_input_fresh = true,
                .sampled_input_scope_input_alias = true,
                .sampled_input_scope_input_valid = true,
            });

    const auto observed = state.Observe();
    EXPECT_EQ(observed.sampled_input_scope_draw_count, 2u);
    EXPECT_EQ(observed.sampled_input_scope_last_draw, VideoCore::ImageColorScopeDrawKind::Indirect);
    EXPECT_TRUE(observed.sampled_input_scope_indexed);
    EXPECT_EQ(observed.sampled_input_scope_element_count, 18u);
    EXPECT_EQ(observed.sampled_input_scope_instance_count, 2u);
    EXPECT_EQ(observed.sampled_input_scope_sampled_images, 3u);
    EXPECT_EQ(observed.sampled_input_scope_storage_writes, 1u);
    EXPECT_TRUE(observed.sampled_input_scope_clear_at_begin);
    EXPECT_TRUE(observed.sampled_input_scope_valid);
    EXPECT_FALSE(observed.sampled_input_scope_overflow);
    EXPECT_EQ(observed.sampled_input_scope_input_producer, VideoCore::ImageProducerClass::Transfer);
    EXPECT_TRUE(observed.sampled_input_scope_input_fresh);
    EXPECT_TRUE(observed.sampled_input_scope_input_alias);
    EXPECT_TRUE(observed.sampled_input_scope_input_valid);
}

TEST(PpSourceColorScopeDraw, InputColorScopeCoverageIsBoundedAndFailClosed) {
    PpSourceProducerScopeCoverage coverage{{.start = 90, .count = 3}};
    ASSERT_TRUE(coverage.Observe(
        90, PpSourceProducerScopeClass::ActiveAtFlip,
        {.draw_count = 1,
         .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
         .sampled_images = 1,
         .sampled_input_producer = VideoCore::ImageProducerClass::ColorAttachment,
         .sampled_input_valid = true,
         .sampled_input_scope_draw_count = 1,
         .sampled_input_scope_last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
         .sampled_input_scope_element_count = 3,
         .sampled_input_scope_instance_count = 1,
         .sampled_input_scope_sampled_images = 1,
         .sampled_input_scope_valid = true,
         .sampled_input_scope_input_producer = VideoCore::ImageProducerClass::ColorAttachment,
         .sampled_input_scope_input_fresh = true,
         .sampled_input_scope_input_valid = true,
         .valid = true}));
    ASSERT_TRUE(coverage.Observe(
        91, PpSourceProducerScopeClass::ActiveAtFlip,
        {.draw_count = 1,
         .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
         .sampled_images = 1,
         .sampled_input_producer = VideoCore::ImageProducerClass::ColorAttachment,
         .sampled_input_valid = true,
         .sampled_input_scope_draw_count = 2,
         .sampled_input_scope_last_draw = VideoCore::ImageColorScopeDrawKind::Indirect,
         .sampled_input_scope_sampled_images = 1,
         .sampled_input_scope_clear_at_begin = true,
         .sampled_input_scope_valid = true,
         .sampled_input_scope_input_producer = VideoCore::ImageProducerClass::StorageImage,
         .sampled_input_scope_input_alias = true,
         .sampled_input_scope_input_valid = true,
         .valid = true}));
    const auto final = coverage.Observe(
        92, PpSourceProducerScopeClass::ActiveAtFlip,
        {.draw_count = 1,
         .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
         .sampled_images = 1,
         .sampled_input_producer = VideoCore::ImageProducerClass::ColorAttachment,
         .sampled_input_valid = true,
         .sampled_input_scope_draw_count = VideoCore::ImageColorScopeProducerState::MaxTrackedDraws,
         .sampled_input_scope_last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
         .sampled_input_scope_valid = false,
         .sampled_input_scope_overflow = true,
         .valid = true});
    ASSERT_TRUE(final);
    EXPECT_EQ(final->valid_input_scopes, 2u);
    EXPECT_EQ(final->invalid_input_scopes, 0u);
    EXPECT_EQ(final->overflow_input_scopes, 1u);
    EXPECT_EQ(final->single_input_scope_draw, 1u);
    EXPECT_EQ(final->multiple_input_scope_draws, 1u);
    EXPECT_EQ(final->input_scope_direct, 1u);
    EXPECT_EQ(final->input_scope_indirect, 1u);
    EXPECT_EQ(final->input_scope_clear, 1u);
    EXPECT_EQ(final->scope_input_color_attachment, 1u);
    EXPECT_EQ(final->scope_input_storage_image, 1u);
    EXPECT_EQ(final->scope_input_transfer, 0u);
    EXPECT_EQ(final->scope_input_cpu_upload, 0u);
    EXPECT_EQ(final->scope_input_unknown, 0u);
    EXPECT_EQ(final->scope_input_fresh, 1u);
    EXPECT_EQ(final->scope_input_reused, 1u);
    EXPECT_EQ(final->scope_input_alias, 1u);
    EXPECT_EQ(final->invalid_scope_input, 0u);
    EXPECT_EQ(final->loss, 1u);
    EXPECT_EQ(FormatPpSourceProducerScopeObservation(*final),
              "FGSCPS s=92 r=0 d=1 k=1 c=0 x=0 j=0 e=0 n=0 b=0 u=1 w=0 ip=1 in=0 "
              "ia=0 iv=1 sd=2048 sk=1 sc=0 sx=1 sj=0 se=0 sn=0 su=0 sw=0 np=0 nn=0 "
              "na=0 nv=0");
    EXPECT_EQ(FormatPpSourceProducerScopeCoverage(*final),
              "FGSCPSC s=92 n=3/3/3 a=3 e=0 v=3 i=0 x=0 s=3 z=0 m=0 w=0 ic=3 "
              "is=0 it=0 iu=0 ix=0 if=0 ir=3 ia=0 il=0 sv=2 si=0 sx=1 sz=0 ss=1 "
              "sm=1 sd=1 sn=1 sc=1 dc=1 ds=1 dt=0 du=0 dx=0 df=1 dr=1 da=1 dl=0 l=1");
}

TEST(PpSourceColorScopeDraw, InvalidAndOverflowScopesAreExplicitCoverageLoss) {
    PpSourceProducerScopeCoverage coverage{{.start = 10, .count = 2}};
    ASSERT_TRUE(coverage.Observe(10, PpSourceProducerScopeClass::ActiveAtFlip,
                                 {.draw_count = 1,
                                  .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
                                  .valid = false}));
    const auto final =
        coverage.Observe(11, PpSourceProducerScopeClass::ActiveAtFlip,
                         {.draw_count = VideoCore::ImageColorScopeProducerState::MaxTrackedDraws,
                          .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
                          .valid = false,
                          .overflow = true});
    ASSERT_TRUE(final);
    EXPECT_EQ(final->valid_draw_scopes, 0u);
    EXPECT_EQ(final->invalid_draw_scopes, 1u);
    EXPECT_EQ(final->overflow_draw_scopes, 1u);
    EXPECT_EQ(final->loss, 2u);
}

TEST(PpSourceColorScopeAncestry, BuildsOrderedChainAndStopsAtNamedBoundary) {
    VideoCore::ImageColorScopeProducerObservation inner_scope{
        .draw_count = 1,
        .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
        .indexed = true,
        .element_count = 6,
        .instance_count = 1,
        .sampled_images = 1,
        .storage_writes = 0,
        .ancestry =
            {
                .nodes = {{
                    {.producer = VideoCore::ImageProducerClass::Transfer,
                     .fresh = true,
                     .producer_valid = true},
                }},
                .depth = 1,
                .terminal = VideoCore::ImageColorScopeAncestryTerminal::NonColorProducer,
            },
        .valid = true,
    };

    const auto ancestry = VideoCore::BuildImageColorScopeAncestry(
        {.classification = VideoCore::ImageProducerClass::ColorAttachment,
         .produced_since_last_observation = true},
        true, false, &inner_scope);

    ASSERT_EQ(ancestry.depth, 2u);
    EXPECT_EQ(ancestry.terminal, VideoCore::ImageColorScopeAncestryTerminal::NonColorProducer);
    EXPECT_FALSE(ancestry.truncated);
    EXPECT_EQ(ancestry.nodes[0].producer, VideoCore::ImageProducerClass::ColorAttachment);
    EXPECT_TRUE(ancestry.nodes[0].fresh);
    EXPECT_TRUE(ancestry.nodes[0].producer_valid);
    EXPECT_EQ(ancestry.nodes[0].draw_count, 1u);
    EXPECT_EQ(ancestry.nodes[0].last_draw, VideoCore::ImageColorScopeDrawKind::Direct);
    EXPECT_TRUE(ancestry.nodes[0].indexed);
    EXPECT_EQ(ancestry.nodes[0].element_count, 6u);
    EXPECT_EQ(ancestry.nodes[0].sampled_images, 1u);
    EXPECT_EQ(ancestry.nodes[1].producer, VideoCore::ImageProducerClass::Transfer);
}

TEST(PpSourceColorScopeAncestry, StopsAtMultiDrawAndFailsClosedAtDepthCap) {
    const VideoCore::ImageColorScopeProducerObservation multi_draw{
        .draw_count = 2,
        .last_draw = VideoCore::ImageColorScopeDrawKind::Indirect,
        .sampled_images = 1,
        .valid = true,
    };
    const auto stopped = VideoCore::BuildImageColorScopeAncestry(
        {.classification = VideoCore::ImageProducerClass::ColorAttachment,
         .produced_since_last_observation = true},
        true, false, &multi_draw);
    ASSERT_EQ(stopped.depth, 1u);
    EXPECT_EQ(stopped.terminal, VideoCore::ImageColorScopeAncestryTerminal::MultipleDraws);
    EXPECT_FALSE(stopped.truncated);

    VideoCore::ImageColorScopeProducerObservation deep_scope{
        .draw_count = 1,
        .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
        .sampled_images = 1,
        .valid = true,
    };
    deep_scope.ancestry.depth = VideoCore::MaxImageColorScopeAncestryDepth;
    deep_scope.ancestry.terminal = VideoCore::ImageColorScopeAncestryTerminal::NonColorProducer;
    for (u32 index = 0; index < deep_scope.ancestry.depth; ++index) {
        deep_scope.ancestry.nodes[index] = {
            .producer = VideoCore::ImageProducerClass::ColorAttachment,
            .fresh = true,
            .producer_valid = true,
            .draw_count = 1,
            .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
            .sampled_images = 1,
            .scope_valid = true,
        };
    }
    const auto capped = VideoCore::BuildImageColorScopeAncestry(
        {.classification = VideoCore::ImageProducerClass::ColorAttachment,
         .produced_since_last_observation = true},
        true, false, &deep_scope);
    EXPECT_EQ(capped.depth, VideoCore::MaxImageColorScopeAncestryDepth);
    EXPECT_EQ(capped.terminal, VideoCore::ImageColorScopeAncestryTerminal::DepthCap);
    EXPECT_TRUE(capped.truncated);
    EXPECT_TRUE(VideoCore::IsImageColorScopeAncestryLoss(capped.terminal));
}

TEST(PpSourceColorScopeAncestry, InvalidUnknownAndMissingHistoryFailClosed) {
    const auto invalid_producer = VideoCore::BuildImageColorScopeAncestry(
        {.classification = VideoCore::ImageProducerClass::Transfer}, false, false, nullptr);
    EXPECT_EQ(invalid_producer.terminal,
              VideoCore::ImageColorScopeAncestryTerminal::InvalidProducer);

    const auto unknown_producer = VideoCore::BuildImageColorScopeAncestry(
        {.classification = VideoCore::ImageProducerClass::Unknown}, true, false, nullptr);
    EXPECT_EQ(unknown_producer.terminal,
              VideoCore::ImageColorScopeAncestryTerminal::InvalidProducer);

    const VideoCore::ImageColorScopeProducerObservation invalid_scope{};
    const auto invalid = VideoCore::BuildImageColorScopeAncestry(
        {.classification = VideoCore::ImageProducerClass::ColorAttachment}, true, false,
        &invalid_scope);
    EXPECT_EQ(invalid.terminal, VideoCore::ImageColorScopeAncestryTerminal::InvalidScope);

    const VideoCore::ImageColorScopeProducerObservation missing_history{
        .draw_count = 1,
        .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
        .sampled_images = 1,
        .valid = true,
    };
    const auto missing = VideoCore::BuildImageColorScopeAncestry(
        {.classification = VideoCore::ImageProducerClass::ColorAttachment}, true, false,
        &missing_history);
    EXPECT_EQ(missing.terminal, VideoCore::ImageColorScopeAncestryTerminal::HistoryUnavailable);
    EXPECT_TRUE(VideoCore::IsImageColorScopeAncestryLoss(missing.terminal));

    const auto alias = VideoCore::BuildImageColorScopeAncestry(
        {.classification = VideoCore::ImageProducerClass::ColorAttachment}, true, true,
        &missing_history);
    EXPECT_EQ(alias.terminal, VideoCore::ImageColorScopeAncestryTerminal::Alias);
    EXPECT_FALSE(VideoCore::IsImageColorScopeAncestryLoss(alias.terminal));
}

TEST(PpSourceColorScopeAncestry, StateAndCompactReportRetainBoundedPrivacySafeChain) {
    VideoCore::ImageColorScopeProducerState state;
    state.BeginScope(101, false);
    state.MarkDraw(
        101, {
                 .kind = VideoCore::ImageColorScopeDrawKind::Direct,
                 .sampled_bindings = 1,
                 .sampled_images = 1,
                 .sampled_input_producer = VideoCore::ImageProducerClass::CpuUpload,
                 .sampled_input_fresh = true,
                 .sampled_input_valid = true,
                 .ancestry =
                     {
                         .nodes = {{
                             {.producer = VideoCore::ImageProducerClass::CpuUpload,
                              .fresh = true,
                              .producer_valid = true},
                         }},
                         .depth = 1,
                         .terminal = VideoCore::ImageColorScopeAncestryTerminal::NonColorProducer,
                     },
             });
    const auto observed = state.Observe();
    ASSERT_EQ(observed.ancestry.depth, 1u);
    EXPECT_EQ(observed.ancestry.nodes[0].producer, VideoCore::ImageProducerClass::CpuUpload);

    PpSourceProducerScopeCoverage coverage{{.start = 101, .count = 1}};
    const auto final = coverage.Observe(101, PpSourceProducerScopeClass::ActiveAtFlip, observed);
    ASSERT_TRUE(final);
    EXPECT_EQ(final->ancestry_max_depth, 1u);
    EXPECT_EQ(final->ancestry_terminals[static_cast<u32>(
                  VideoCore::ImageColorScopeAncestryTerminal::NonColorProducer)],
              1u);
    EXPECT_EQ(final->loss, 0u);
    const auto line = FormatPpSourceProducerScopeObservation(*final);
    EXPECT_NE(line.find(" ad=1 at=1 ax=0 ch="), std::string::npos);
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
}

} // namespace
