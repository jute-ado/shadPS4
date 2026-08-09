// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/host_passes/pp_source_producer_scope.h"
#include "video_core/renderer_vulkan/host_passes/pp_source_publication.h"
#include "video_core/renderer_vulkan/host_passes/pp_terminal_scope_content.h"
#include "video_core/texture_cache/image_color_scope_producer.h"
#include "video_core/texture_cache/image_producer.h"

namespace {

using namespace Vulkan;
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

    EXPECT_EQ(state.Observe(),
              (VideoCore::ImageColorScopeProducerObservation{
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
                  .draw_summaries = {{
                      {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                       .indexed = true,
                       .element_count = 36,
                       .instance_count = 2,
                       .sampled_images = 1,
                       .sampled_input_producer = VideoCore::ImageProducerClass::ColorAttachment,
                       .sampled_input_fresh = true,
                       .sampled_input_valid = true},
                      {.kind = VideoCore::ImageColorScopeDrawKind::Indirect,
                       .sampled_images = 2,
                       .storage_writes = 1},
                  }},
                  .draw_summary_count = 2,
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

TEST(PpSourceTerminalScopeDraws, RetainsDrawOrderAndGenericInputShape) {
    VideoCore::ImageColorScopeProducerState state;
    state.BeginScope(201, false);
    state.MarkDraw(201, {
                            .kind = VideoCore::ImageColorScopeDrawKind::Direct,
                            .indexed = false,
                            .element_count = 3,
                            .instance_count = 1,
                            .sampled_bindings = 1,
                            .sampled_images = 1,
                            .sampled_input_producer = VideoCore::ImageProducerClass::Transfer,
                            .sampled_input_fresh = true,
                            .sampled_input_valid = true,
                        });
    state.MarkDraw(201, {
                            .kind = VideoCore::ImageColorScopeDrawKind::Indirect,
                            .indexed = true,
                            .element_count = 24,
                            .instance_count = 1,
                            .sampled_bindings = 2,
                            .sampled_images = 2,
                        });

    const auto scope = state.Observe();
    ASSERT_EQ(scope.draw_summary_count, 2u);
    EXPECT_FALSE(scope.draw_summaries_truncated);
    EXPECT_EQ(scope.draw_summaries[0].kind, VideoCore::ImageColorScopeDrawKind::Direct);
    EXPECT_FALSE(scope.draw_summaries[0].indexed);
    EXPECT_EQ(scope.draw_summaries[0].element_count, 3u);
    EXPECT_EQ(scope.draw_summaries[0].sampled_images, 1u);
    EXPECT_EQ(scope.draw_summaries[0].sampled_input_producer,
              VideoCore::ImageProducerClass::Transfer);
    EXPECT_TRUE(scope.draw_summaries[0].sampled_input_fresh);
    EXPECT_TRUE(scope.draw_summaries[0].sampled_input_valid);
    EXPECT_EQ(scope.draw_summaries[1].kind, VideoCore::ImageColorScopeDrawKind::Indirect);
    EXPECT_TRUE(scope.draw_summaries[1].indexed);
    EXPECT_EQ(scope.draw_summaries[1].element_count, 24u);
    EXPECT_EQ(scope.draw_summaries[1].sampled_images, 2u);
}

TEST(PpSourceTerminalScopeDraws, MultiDrawBoundaryCarriesBoundedOrderedSummaries) {
    VideoCore::ImageColorScopeProducerObservation scope{
        .draw_count = 2,
        .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
        .draw_summaries = {{
            {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
             .element_count = 6,
             .instance_count = 1,
             .sampled_images = 1,
             .sampled_input_producer = VideoCore::ImageProducerClass::ColorAttachment,
             .sampled_input_fresh = true,
             .sampled_input_valid = true},
            {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
             .indexed = true,
             .element_count = 24,
             .instance_count = 1,
             .sampled_images = 2},
        }},
        .draw_summary_count = 2,
        .valid = true,
    };
    const auto ancestry = VideoCore::BuildImageColorScopeAncestry(
        {.classification = VideoCore::ImageProducerClass::ColorAttachment,
         .produced_since_last_observation = true},
        true, false, &scope);
    ASSERT_EQ(ancestry.terminal, VideoCore::ImageColorScopeAncestryTerminal::MultipleDraws);
    ASSERT_EQ(ancestry.terminal_draw_count, 2u);
    EXPECT_FALSE(ancestry.terminal_draws_truncated);
    EXPECT_EQ(ancestry.terminal_draws[0].element_count, 6u);
    EXPECT_EQ(ancestry.terminal_draws[1].element_count, 24u);

    PpSourceProducerScopeCoverage coverage{{.start = 202, .count = 1}};
    const auto report =
        coverage.Observe(202, PpSourceProducerScopeClass::ActiveAtFlip,
                         {.draw_count = 1,
                          .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
                          .sampled_bindings = 1,
                          .sampled_images = 1,
                          .sampled_input_producer = VideoCore::ImageProducerClass::ColorAttachment,
                          .sampled_input_valid = true,
                          .sampled_input_scope_draw_count = 2,
                          .sampled_input_scope_valid = true,
                          .ancestry = ancestry,
                          .valid = true});
    ASSERT_TRUE(report);
    EXPECT_EQ(report->ancestry_draw_summary_loss, 0u);
    const auto line = FormatPpSourceProducerScopeObservation(*report);
    EXPECT_NE(line.find(" td=2 tx=0 th="), std::string::npos);
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
}

TEST(PpSourceTerminalScopeDraws, SummaryCapacityFailsClosedWithoutPartialClaim) {
    VideoCore::ImageColorScopeProducerState state;
    state.BeginScope(203, false);
    for (u32 index = 0; index <= VideoCore::MaxImageColorScopeTerminalDraws; ++index) {
        state.MarkDraw(203, {.kind = VideoCore::ImageColorScopeDrawKind::Direct});
    }
    const auto scope = state.Observe();
    EXPECT_EQ(scope.draw_summary_count, VideoCore::MaxImageColorScopeTerminalDraws);
    EXPECT_TRUE(scope.draw_summaries_truncated);

    const auto ancestry = VideoCore::BuildImageColorScopeAncestry(
        {.classification = VideoCore::ImageProducerClass::ColorAttachment}, true, false, &scope);
    EXPECT_EQ(ancestry.terminal, VideoCore::ImageColorScopeAncestryTerminal::MultipleDraws);
    EXPECT_TRUE(ancestry.terminal_draws_truncated);

    PpSourceProducerScopeCoverage coverage{{.start = 203, .count = 1}};
    const auto report =
        coverage.Observe(203, PpSourceProducerScopeClass::ActiveAtFlip,
                         {.draw_count = 1,
                          .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
                          .sampled_bindings = 1,
                          .sampled_images = 1,
                          .sampled_input_producer = VideoCore::ImageProducerClass::ColorAttachment,
                          .sampled_input_valid = true,
                          .sampled_input_scope_draw_count = 2,
                          .sampled_input_scope_valid = true,
                          .ancestry = ancestry,
                          .valid = true});
    ASSERT_TRUE(report);
    EXPECT_EQ(report->ancestry_draw_summary_loss, 1u);
    EXPECT_EQ(report->loss, 1u);
}

TEST(PpTerminalScopeContent, ExternalShapeSelectorArmsExactlyTwoOrderedDraws) {
    const PpTerminalScopeContentConfig config{
        .enabled = true,
        .first = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                  .indexed = true,
                  .element_count = 696,
                  .instance_count = 1,
                  .sampled_images = 1,
                  .storage_writes = 0},
        .second = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                   .indexed = true,
                   .element_count = 24,
                   .instance_count = 1,
                   .sampled_images = 2,
                   .storage_writes = 0},
    };
    PpTerminalScopeContentGate gate{config};
    EXPECT_TRUE(gate.Arm(17, 3));
    EXPECT_EQ(gate.ObserveDraw(16, 90, config.first), PpTerminalScopeContentAction::None);
    EXPECT_EQ(gate.ObserveDraw(17, 90, config.first), PpTerminalScopeContentAction::CaptureFirst);
    EXPECT_EQ(gate.ObserveDraw(17, 91, config.second), PpTerminalScopeContentAction::ShapeLoss);
    EXPECT_TRUE(gate.Arm(17, 4));
    EXPECT_EQ(gate.ObserveDraw(17, 92, config.first), PpTerminalScopeContentAction::CaptureFirst);
    EXPECT_EQ(gate.ObserveDraw(17, 92, config.second), PpTerminalScopeContentAction::CaptureSecond);
    const auto complete = gate.Take(17, 4);
    EXPECT_EQ(complete.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(complete.draw_count, 2u);
    EXPECT_FALSE(complete.cpu_wait);
    EXPECT_FALSE(complete.finish);
    EXPECT_FALSE(complete.retains_image);
    EXPECT_FALSE(complete.retains_vk_image);
}

TEST(PpTerminalScopeContent, LatestScopeSupersedesEarlierCompleteOrInvalidShape) {
    const PpTerminalScopeContentConfig config{
        .enabled = true,
        .first = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                  .indexed = true,
                  .element_count = 696,
                  .instance_count = 1,
                  .sampled_images = 1},
        .second = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                   .indexed = true,
                   .element_count = 24,
                   .instance_count = 1,
                   .sampled_images = 2},
    };
    PpTerminalScopeContentGate gate{config};
    ASSERT_TRUE(gate.Arm(17, 4));
    EXPECT_EQ(gate.ObserveDraw(17, 90, config.first), PpTerminalScopeContentAction::CaptureFirst);
    EXPECT_EQ(gate.ObserveDraw(17, 90, config.second), PpTerminalScopeContentAction::CaptureSecond);

    const PpTerminalScopeDrawSelector unrelated{
        .kind = VideoCore::ImageColorScopeDrawKind::Direct,
        .indexed = true,
        .element_count = 3,
        .instance_count = 1,
        .sampled_images = 1,
    };
    EXPECT_EQ(gate.ObserveDraw(17, 91, unrelated), PpTerminalScopeContentAction::ShapeLoss);
    EXPECT_EQ(gate.ObserveDraw(17, 92, config.first), PpTerminalScopeContentAction::CaptureFirst);
    EXPECT_EQ(gate.ObserveDraw(17, 92, config.second), PpTerminalScopeContentAction::CaptureSecond);
    EXPECT_EQ(gate.Take(17, 4).status, FinalGuestSurfaceStatus::Complete);

    const auto reset = ApplyPpTerminalScopeContentAction(
        FinalGuestSurfaceStatus::GapLoss, {.gap = 1}, PpTerminalScopeContentAction::CaptureFirst);
    EXPECT_EQ(reset.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_FALSE(reset.loss.Any());

    const auto acquire = PlanPpTerminalScopePlaneSlot(0, false);
    EXPECT_TRUE(acquire.acquire);
    EXPECT_FALSE(acquire.reuse);
    const auto reuse = PlanPpTerminalScopePlaneSlot(0, true);
    EXPECT_FALSE(reuse.acquire);
    EXPECT_TRUE(reuse.reuse);
    const auto missing = PlanPpTerminalScopePlaneSlot(1, false);
    EXPECT_EQ(missing.status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(missing.loss.gap, 1u);
}

TEST(PpTerminalScopeContent, SelectedLogicalWindowsProduceTwoBoundedPlanes) {
    FinalGuestSurfaceWatchOrdinals selector{};
    selector.status = FinalGuestSurfaceStatus::Complete;
    selector.count = 2;
    selector.ordinals[0] = 1024;
    selector.ordinals[1] = 1299;
    const auto plan = PlanPpTerminalScopeContent({
        .enabled = true,
        .armed = true,
        .target_width = 1920,
        .target_height = 1080,
        .final_source_width = 1920,
        .final_source_height = 1080,
        .logical_width = 1280,
        .logical_height = 720,
        .format = FinalGuestSurfaceFormat::Bgra8,
        .samples = 1,
        .selector = selector,
        .buffer_alignment = 16,
        .max_regions = 32,
        .max_bytes = PpTerminalScopeSnapshotBytes,
    });
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(plan.region_count, 2u);
    EXPECT_EQ(plan.copy_region_count, 4u);
    EXPECT_EQ(plan.first_plane_offset, 0u);
    EXPECT_GE(plan.second_plane_offset, plan.plane_bytes);
    EXPECT_LE(plan.total_bytes, PpTerminalScopeSnapshotBytes);
    EXPECT_EQ(plan.image_barriers_per_draw, 2u);
    EXPECT_TRUE(plan.ends_rendering);
    EXPECT_TRUE(plan.resumes_rendering_with_load);
    EXPECT_TRUE(plan.preserves_rendering_serial);
    EXPECT_TRUE(plan.callback_payload_is_scalar_only);
    EXPECT_FALSE(plan.cpu_wait);
    EXPECT_FALSE(plan.finish);
}

TEST(PpTerminalScopeContent, MappingCapacityAndStaleArmFailClosed) {
    FinalGuestSurfaceWatchOrdinals selector{};
    selector.status = FinalGuestSurfaceStatus::Complete;
    selector.count = 1;
    selector.ordinals[0] = 1024;
    const auto wrong_target = PlanPpTerminalScopeContent({
        .enabled = true,
        .armed = true,
        .target_width = 1280,
        .target_height = 720,
        .final_source_width = 1920,
        .final_source_height = 1080,
        .logical_width = 1280,
        .logical_height = 720,
        .format = FinalGuestSurfaceFormat::Bgra8,
        .samples = 1,
        .selector = selector,
        .buffer_alignment = 16,
        .max_regions = 32,
        .max_bytes = PpTerminalScopeSnapshotBytes,
    });
    EXPECT_EQ(wrong_target.status, FinalGuestSurfaceStatus::Unsupported);
    EXPECT_EQ(wrong_target.loss.logical_mapping, 1u);
    EXPECT_FALSE(wrong_target.copy);

    PpTerminalScopeContentGate gate{
        {.enabled = true,
         .first = {.kind = VideoCore::ImageColorScopeDrawKind::Direct},
         .second = {.kind = VideoCore::ImageColorScopeDrawKind::Direct}}};
    ASSERT_TRUE(gate.Arm(7, 10));
    EXPECT_EQ(gate.ObserveDraw(7, 100, {.kind = VideoCore::ImageColorScopeDrawKind::Direct}),
              PpTerminalScopeContentAction::CaptureFirst);
    const auto stale = gate.Take(7, 11);
    EXPECT_EQ(stale.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(stale.loss.invalidation, 1u);
}

TEST(PpTerminalScopeContent, CompactGrammarNeverExposesPrivateTargetToken) {
    const PpTerminalScopeContentReport report{
        .sequence = 4100,
        .status = FinalGuestSurfaceStatus::Complete,
        .draw_count = 2,
        .region_count = 14,
        .first_aba = 1,
        .first_stable = 2,
        .second_aba = 3,
        .second_stable = 4,
    };
    const auto line = FormatPpTerminalScopeContentReport(report);
    EXPECT_EQ(line, "FGSCTS s=4100 st=0 d=2 r=14 a0=1 s0=2 a1=3 s1=4 lm=0");
    EXPECT_EQ(line.find("token"), std::string::npos);
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
}

TEST(PpTerminalScopeContent, PrivateLinksRetainExactSoleInputAndFailClosedOnReuse) {
    VideoCore::ImageColorScopeProducerState state;
    state.BeginScope(301, false);
    state.MarkDraw(301, {
                            .kind = VideoCore::ImageColorScopeDrawKind::Direct,
                            .sampled_bindings = 1,
                            .sampled_images = 1,
                            .sampled_input_image = {.id = VideoCore::ImageId{41}, .uid = 7001},
                        });
    const auto sole = state.Observe();
    EXPECT_EQ(sole.sampled_input_image.id, VideoCore::ImageId{41});
    EXPECT_EQ(sole.sampled_input_image.uid, 7001u);
    EXPECT_TRUE(VideoCore::ValidateImageColorScopePrivateLink(sole.sampled_input_image,
                                                              VideoCore::ImageId{41}, 7001));
    EXPECT_FALSE(VideoCore::ValidateImageColorScopePrivateLink(sole.sampled_input_image,
                                                               VideoCore::ImageId{41}, 7002));

    state.BeginScope(302, false);
    state.MarkDraw(302, {
                            .kind = VideoCore::ImageColorScopeDrawKind::Direct,
                            .sampled_bindings = 2,
                            .sampled_images = 2,
                            .sampled_input_image = {.id = VideoCore::ImageId{42}, .uid = 7003},
                        });
    EXPECT_FALSE(state.Observe().sampled_input_image.Valid());
}

TEST(PpTerminalScopeContent, ExistingCompactScopeOutputNeverFormatsPrivateLinks) {
    PpSourceProducerScopeCoverage coverage{{.start = 302, .count = 1}};
    const auto report =
        coverage.Observe(302, PpSourceProducerScopeClass::ActiveAtFlip,
                         {.draw_count = 1,
                          .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
                          .sampled_bindings = 1,
                          .sampled_images = 1,
                          .sampled_input_producer = VideoCore::ImageProducerClass::Transfer,
                          .sampled_input_valid = true,
                          .sampled_input_image = {.id = VideoCore::ImageId{1234}, .uid = 998877},
                          .valid = true});
    ASSERT_TRUE(report);
    const auto line = FormatPpSourceProducerScopeObservation(*report);
    EXPECT_EQ(line.find("1234"), std::string::npos);
    EXPECT_EQ(line.find("998877"), std::string::npos);
    EXPECT_EQ(line.find("uid"), std::string::npos);
}

TEST(PpTerminalScopeContent, PrivateLinkResolutionNeverIndexesAFreeOrReusedSlot) {
    const VideoCore::ImageColorScopePrivateLink link{
        .id = VideoCore::ImageId{17},
        .uid = 4401,
    };
    u32 uid_queries{};
    const auto free_slot = VideoCore::ResolveImageColorScopePrivateLink(
        link, [](VideoCore::ImageId) { return false; },
        [&](VideoCore::ImageId) {
            ++uid_queries;
            return 4401;
        });
    EXPECT_FALSE(free_slot);
    EXPECT_EQ(uid_queries, 0u);

    const auto reused_slot = VideoCore::ResolveImageColorScopePrivateLink(
        link, [](VideoCore::ImageId) { return true; },
        [&](VideoCore::ImageId id) {
            ++uid_queries;
            EXPECT_EQ(id, VideoCore::ImageId{17});
            return 4402;
        });
    EXPECT_FALSE(reused_slot);
    EXPECT_EQ(uid_queries, 1u);

    const auto current = VideoCore::ResolveImageColorScopePrivateLink(
        link, [](VideoCore::ImageId) { return true; },
        [&](VideoCore::ImageId) {
            ++uid_queries;
            return 4401;
        });
    ASSERT_TRUE(current);
    EXPECT_EQ(*current, VideoCore::ImageId{17});
    EXPECT_EQ(uid_queries, 2u);
}

TEST(PpTerminalScopeContent, ExactCalibratedTripletClassifiesBothDrawPlanesPerOrdinal) {
    PpTerminalScopeContentReducer reducer{{.frame_start = 100, .frame_count = 3}, 8};
    const PpTerminalScopeContentHistoryLayout layout{
        .region_count = 2,
        .plane_bytes = 8,
        .second_plane_offset = 8,
        .total_bytes = 16,
        .regions = {{{.logical_ordinal = 11, .buffer_offset = 0, .byte_size = 4},
                     {.logical_ordinal = 12, .buffer_offset = 4, .byte_size = 4}}},
    };
    const auto bytes = [](std::array<u8, 4> first0, std::array<u8, 4> first1,
                          std::array<u8, 4> second0, std::array<u8, 4> second1) {
        std::array<std::byte, 16> result{};
        const std::array planes{first0, first1, second0, second1};
        for (u32 plane = 0; plane < planes.size(); ++plane) {
            for (u32 index = 0; index < planes[plane].size(); ++index) {
                result[plane * 4 + index] = std::byte{planes[plane][index]};
            }
        }
        return result;
    };
    const auto a = bytes({1, 2, 3, 255}, {4, 5, 6, 255}, {7, 8, 9, 255}, {10, 11, 12, 255});
    const auto b = bytes({9, 9, 9, 255}, {4, 5, 6, 255}, {7, 8, 9, 255}, {99, 98, 97, 255});
    const auto c = bytes({1, 2, 3, 1}, {8, 8, 8, 255}, {7, 8, 9, 0}, {10, 11, 12, 255});
    reducer.ObserveContent(100, layout, a, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(101, layout, b, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(102, layout, c, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveCalibration({.request_ordinal = 1, .sequence = 100, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 2, .sequence = 101, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 3, .sequence = 102, .valid = true});
    const auto reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports[0].request_ordinal, 3u);
    EXPECT_EQ(reports[0].sequences, (std::array<u64, 3>{100, 101, 102}));
    EXPECT_EQ(reports[0].first_aba_ordinals, (std::vector<u32>{11}));
    EXPECT_EQ(reports[0].first_stable_ordinals, (std::vector<u32>{}));
    EXPECT_EQ(reports[0].first_ambiguous_ordinals, (std::vector<u32>{12}));
    EXPECT_EQ(reports[0].second_aba_ordinals, (std::vector<u32>{12}));
    EXPECT_EQ(reports[0].second_stable_ordinals, (std::vector<u32>{11}));
    EXPECT_EQ(reports[0].second_ambiguous_ordinals, (std::vector<u32>{}));
    EXPECT_FALSE(reports[0].loss.Any());
}

TEST(PpTerminalScopeContent, MissingChangedOrEvictedContentFailsClosed) {
    PpTerminalScopeContentReducer reducer{{.frame_start = 200, .frame_count = 40}, 3};
    const PpTerminalScopeContentHistoryLayout layout{
        .region_count = 1,
        .plane_bytes = 4,
        .second_plane_offset = 4,
        .total_bytes = 8,
        .regions = {{{.logical_ordinal = 21, .buffer_offset = 0, .byte_size = 4}}},
    };
    const std::array<std::byte, 8> bytes{};
    reducer.ObserveContent(200, layout, bytes, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(201, layout, bytes, FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
    reducer.ObserveContent(202, layout, bytes, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveCalibration({.request_ordinal = 1, .sequence = 200, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 2, .sequence = 201, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 3, .sequence = 202, .valid = true});
    auto reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports[0].status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(reports[0].loss.gap, 1u);

    reducer.ObserveContent(203, layout, bytes, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(204, layout, bytes, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(205, layout, bytes, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveCalibration({.request_ordinal = 4, .sequence = 203, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 5, .sequence = 204, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 6, .sequence = 205, .valid = true});
    reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 3u);
    EXPECT_EQ(reports.back().request_ordinal, 6u);
    EXPECT_EQ(reports.back().status, FinalGuestSurfaceStatus::Complete);

    reducer.ObserveCalibration({.request_ordinal = 7, .sequence = 200, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 8, .sequence = 204, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 9, .sequence = 205, .valid = true});
    reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 3u);
    EXPECT_EQ(reports.back().request_ordinal, 9u);
    EXPECT_EQ(reports.back().status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(reports.back().loss.history, 1u);
}

TEST(PpTerminalScopeContent, CompactCalibratedOutputContainsOnlyOrdinalsAndStatus) {
    const PpTerminalScopeCalibratedReport report{
        .request_ordinal = 19,
        .sequences = {700, 706, 711},
        .first_aba_ordinals = {11, 12},
        .first_stable_ordinals = {13},
        .second_ambiguous_ordinals = {14},
        .status = FinalGuestSurfaceStatus::Complete,
    };
    const auto line = FormatPpTerminalScopeCalibratedReport(report);
    EXPECT_NE(line.find("FGSCTST q=19 abc=700/706/711"), std::string::npos);
    EXPECT_NE(line.find("a0=11,12"), std::string::npos);
    EXPECT_NE(line.find("s0=13"), std::string::npos);
    EXPECT_NE(line.find("x1=14"), std::string::npos);
    EXPECT_EQ(line.find("pixel"), std::string::npos);
    EXPECT_EQ(line.find("hash"), std::string::npos);
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
}

TEST(PpTerminalScopeContent, ExternalRuntimeConfigRequiresGenericDrawSelectors) {
    const std::map<std::string_view, std::string_view> values{
        {"SHADPS4_FINAL_GUEST_SURFACE_CONTENT", "1"},
        {"SHADPS4_FINAL_GUEST_SURFACE_STAGE", "pp_source_publication_reconstruction"},
        {"SHADPS4_FINAL_GUEST_SURFACE_CALIBRATED_TRIPLETS", "1"},
        {"SHADPS4_FINAL_GUEST_SURFACE_EXPECTED_CALIBRATIONS", "300"},
        {"SHADPS4_FINAL_GUEST_SURFACE_FRAME_START", "4000"},
        {"SHADPS4_FINAL_GUEST_SURFACE_FRAME_COUNT", "800"},
        {"SHADPS4_FINAL_GUEST_SURFACE_WATCH_ORDINALS", "390,1024,1299"},
        {"SHADPS4_PP_TERMINAL_SCOPE_CONTENT", "1"},
        {"SHADPS4_PP_TERMINAL_SCOPE_FIRST", "direct,indexed,696,1,1,0"},
        {"SHADPS4_PP_TERMINAL_SCOPE_SECOND", "direct,indexed,24,1,2,0"},
    };
    const auto config = ResolvePpTerminalScopeRuntimeConfig([&](const char* name) {
        const auto found = values.find(name);
        return found == values.end() ? std::optional<std::string_view>{}
                                     : std::optional<std::string_view>{found->second};
    });
    ASSERT_TRUE(config);
    EXPECT_EQ(config->window.frame_start, 4000u);
    EXPECT_EQ(config->window.frame_count, 800u);
    EXPECT_EQ(config->expected_calibrations, 300u);
    EXPECT_EQ(config->watch_ordinals.count, 3u);
    EXPECT_EQ(config->content.first,
              (PpTerminalScopeDrawSelector{VideoCore::ImageColorScopeDrawKind::Direct, true, 696, 1,
                                           1, 0}));
    EXPECT_EQ(config->content.second,
              (PpTerminalScopeDrawSelector{VideoCore::ImageColorScopeDrawKind::Direct, true, 24, 1,
                                           2, 0}));
}

TEST(PpTerminalScopeContent, RuntimeConfigRejectsDisabledMalformedOrImplicitSelection) {
    const auto missing = [](const char*) { return std::optional<std::string_view>{}; };
    EXPECT_FALSE(ResolvePpTerminalScopeRuntimeConfig(missing));
    EXPECT_FALSE(ParsePpTerminalScopeDrawSelector("direct,indexed,696,1,1"));
    EXPECT_FALSE(ParsePpTerminalScopeDrawSelector("direct,indexed,696,1,1,0,trailing"));
    EXPECT_FALSE(ParsePpTerminalScopeDrawSelector("title,indexed,696,1,1,0"));
    EXPECT_FALSE(ParsePpTerminalScopeDrawSelector("direct,maybe,696,1,1,0"));
    EXPECT_FALSE(ParsePpTerminalScopeDrawSelector("direct,indexed,0,1,1,0"));
    EXPECT_FALSE(ParsePpTerminalScopeDrawSelector("direct,indexed,696,0,1,0"));
}

TEST(PpTerminalScopeContent, RenderingSplitResumesWithLoadAndPreservesGuestScopeSerial) {
    const auto complete = PlanPpTerminalScopeRenderingSplit(true, 77, 77);
    EXPECT_EQ(complete.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_TRUE(complete.end_rendering);
    EXPECT_TRUE(complete.resume_rendering);
    EXPECT_TRUE(complete.force_load);
    EXPECT_TRUE(complete.preserve_serial);
    EXPECT_EQ(complete.serial, 77u);

    const auto not_rendering = PlanPpTerminalScopeRenderingSplit(false, 77, 77);
    EXPECT_EQ(not_rendering.status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(not_rendering.loss.gap, 1u);
    EXPECT_FALSE(not_rendering.resume_rendering);

    const auto stale = PlanPpTerminalScopeRenderingSplit(true, 78, 77);
    EXPECT_EQ(stale.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(stale.loss.invalidation, 1u);
    EXPECT_FALSE(stale.resume_rendering);
}

TEST(PpTerminalScopeContent, FlipDecisionEmitsAtMostOneFailClosedObservation) {
    const auto existing = PlanPpTerminalScopeFlipDecision(true, true, true, true);
    EXPECT_TRUE(existing.use_existing_capture);
    EXPECT_FALSE(existing.synthesize_loss);
    EXPECT_TRUE(existing.arm_next);

    const auto warmup = PlanPpTerminalScopeFlipDecision(false, true, true, true);
    EXPECT_FALSE(warmup.use_existing_capture);
    EXPECT_TRUE(warmup.synthesize_loss);
    EXPECT_EQ(warmup.status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(warmup.loss.gap, 1u);
    EXPECT_TRUE(warmup.arm_next);

    const auto capacity = PlanPpTerminalScopeFlipDecision(false, true, false, true);
    EXPECT_TRUE(capacity.synthesize_loss);
    EXPECT_EQ(capacity.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(capacity.loss.tile_capacity, 1u);
    EXPECT_FALSE(capacity.arm_next);

    const auto stale = PlanPpTerminalScopeFlipDecision(false, false, false, true);
    EXPECT_TRUE(stale.synthesize_loss);
    EXPECT_EQ(stale.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(stale.loss.invalidation, 1u);
    EXPECT_FALSE(stale.arm_next);

    const auto outside = PlanPpTerminalScopeFlipDecision(false, true, true, false);
    EXPECT_FALSE(outside.synthesize_loss);
    EXPECT_TRUE(outside.arm_next);
}

TEST(PpTerminalScopeContent, CalibratedCoverageIsExactAndFailsClosedWhenIncomplete) {
    PpTerminalScopeContentReducer reducer{{.frame_start = 300, .frame_count = 3}, 8};
    const PpTerminalScopeContentHistoryLayout layout{
        .region_count = 1,
        .plane_bytes = 4,
        .second_plane_offset = 4,
        .total_bytes = 8,
        .regions = {{{.logical_ordinal = 31, .buffer_offset = 0, .byte_size = 4}}},
    };
    const std::array<std::byte, 8> bytes{};
    for (u64 sequence = 300; sequence <= 302; ++sequence) {
        reducer.ObserveContent(sequence, layout, bytes, FinalGuestSurfaceStatus::Complete, {});
    }
    reducer.ObserveCalibration({.request_ordinal = 1, .sequence = 300, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 2, .sequence = 301, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 3, .sequence = 302, .valid = true});
    const auto complete = reducer.GetCoverage(3);
    EXPECT_TRUE(complete.ready);
    EXPECT_EQ(complete.calibrations, 3u);
    EXPECT_EQ(complete.outside, 0u);
    EXPECT_EQ(complete.eligible, 1u);
    EXPECT_EQ(complete.emitted, 1u);
    EXPECT_EQ(complete.complete, 1u);
    EXPECT_EQ(complete.loss, 0u);

    const auto incomplete = reducer.GetCoverage(5);
    EXPECT_FALSE(incomplete.ready);
    EXPECT_EQ(incomplete.calibrations, 3u);
    EXPECT_EQ(incomplete.loss, 2u);
    const auto line = FormatPpTerminalScopeCalibratedCoverage(incomplete);
    EXPECT_NE(line.find("FGSCTSTC c=3"), std::string::npos);
    EXPECT_NE(line.find("e=1/1/1"), std::string::npos);
    EXPECT_NE(line.find("lm=2"), std::string::npos);
}

} // namespace
