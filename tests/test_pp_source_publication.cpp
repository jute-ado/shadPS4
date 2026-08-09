// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/host_passes/pp_source_producer_scope.h"
#include "video_core/renderer_vulkan/host_passes/pp_source_publication.h"
#include "video_core/renderer_vulkan/host_passes/pp_terminal_scope_content.h"
#include "video_core/renderer_vulkan/host_passes/pp_upstream_feedback_pre_post.h"
#include "video_core/renderer_vulkan/host_passes/pp_upstream_input_content.h"
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

TEST(PpSourceProducer, PeekingAtThePreviousProducerDoesNotConsumeFreshness) {
    VideoCore::ImageProducerState state;
    state.Mark(VideoCore::ImageProducerClass::ColorAttachment);

    EXPECT_EQ(state.Peek(), (VideoCore::ImageProducerObservation{
                                .classification = VideoCore::ImageProducerClass::ColorAttachment,
                                .produced_since_last_observation = true,
                            }));
    EXPECT_TRUE(state.Peek().produced_since_last_observation);
    EXPECT_TRUE(state.Observe().produced_since_last_observation)
        << "A diagnostic predecessor snapshot must not consume the Present observer";
    EXPECT_FALSE(state.Peek().produced_since_last_observation);

    state.Mark(VideoCore::ImageProducerClass::Transfer);
    EXPECT_EQ(state.Peek().classification, VideoCore::ImageProducerClass::Transfer);
    EXPECT_TRUE(state.Peek().produced_since_last_observation);
    EXPECT_TRUE(state.Observe().produced_since_last_observation);
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
        .consumer = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                     .indexed = true,
                     .element_count = 4,
                     .instance_count = 1,
                     .sampled_images = 1,
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
    EXPECT_EQ(gate.ObserveConsumer(17, config.consumer),
              PpTerminalScopeConsumerAction::CaptureConsumer);
    const auto complete = gate.Take(17, 4);
    EXPECT_EQ(complete.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(complete.draw_count, 2u);
    EXPECT_FALSE(complete.cpu_wait);
    EXPECT_FALSE(complete.finish);
    EXPECT_FALSE(complete.retains_image);
    EXPECT_FALSE(complete.retains_vk_image);
}

TEST(PpTerminalScopeContent, FirstProducerPreviewCapturesBeforeTheDrawWithoutAdvancingIt) {
    const PpTerminalScopeContentConfig config{
        .enabled = true,
        .capture_pre_first = true,
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
        .consumer = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                     .indexed = true,
                     .element_count = 4,
                     .instance_count = 1,
                     .sampled_images = 1},
    };
    PpTerminalScopeContentGate gate{config};
    ASSERT_TRUE(gate.Arm(17, 3));
    EXPECT_EQ(gate.PreviewDraw(16, 90, config.first), PpTerminalScopePreDrawAction::None);
    EXPECT_EQ(gate.PreviewDraw(17, 90, config.first),
              PpTerminalScopePreDrawAction::CaptureBeforeFirst);
    EXPECT_EQ(gate.ObserveDraw(17, 90, config.first), PpTerminalScopeContentAction::CaptureFirst);
    EXPECT_EQ(gate.PreviewDraw(17, 90, config.second), PpTerminalScopePreDrawAction::None);
    EXPECT_EQ(gate.ObserveDraw(17, 90, config.second), PpTerminalScopeContentAction::CaptureSecond);

    ASSERT_TRUE(gate.Arm(17, 4));
    EXPECT_EQ(gate.ObserveDraw(17, 91, config.first), PpTerminalScopeContentAction::ShapeLoss);
    ASSERT_TRUE(gate.Arm(17, 5));
    EXPECT_EQ(gate.PreviewDraw(17, 92, config.first),
              PpTerminalScopePreDrawAction::CaptureBeforeFirst);
    EXPECT_EQ(gate.PreviewDraw(17, 92, config.first), PpTerminalScopePreDrawAction::ShapeLoss);
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
        .consumer = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                     .indexed = true,
                     .element_count = 4,
                     .instance_count = 1,
                     .sampled_images = 1},
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
    EXPECT_EQ(gate.ObserveConsumer(17, config.consumer),
              PpTerminalScopeConsumerAction::CaptureConsumer);
    EXPECT_EQ(gate.Take(17, 4).status, FinalGuestSurfaceStatus::Complete);

    const auto reset = ApplyPpTerminalScopeContentAction(
        FinalGuestSurfaceStatus::GapLoss, {.gap = 1}, PpTerminalScopeContentAction::CaptureFirst);
    EXPECT_EQ(reset.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_FALSE(reset.loss.Any());
    const auto preserve_pre_loss =
        ApplyPpTerminalScopeContentAction(FinalGuestSurfaceStatus::BusyLoss, {.busy = 1},
                                          PpTerminalScopeContentAction::CaptureFirst, true);
    EXPECT_EQ(preserve_pre_loss.status, FinalGuestSurfaceStatus::BusyLoss);
    EXPECT_EQ(preserve_pre_loss.loss.busy, 1u);

    const auto acquire = PlanPpTerminalScopePlaneSlot(4, false);
    EXPECT_TRUE(acquire.acquire);
    EXPECT_FALSE(acquire.reuse);
    const auto missing_pre = PlanPpTerminalScopePlaneSlot(0, false);
    EXPECT_EQ(missing_pre.status, FinalGuestSurfaceStatus::GapLoss);
    const auto reuse = PlanPpTerminalScopePlaneSlot(0, true);
    EXPECT_FALSE(reuse.acquire);
    EXPECT_TRUE(reuse.reuse);
    const auto missing = PlanPpTerminalScopePlaneSlot(1, false);
    EXPECT_EQ(missing.status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(missing.loss.gap, 1u);
    const auto consumer = PlanPpTerminalScopePlaneSlot(2, true);
    EXPECT_FALSE(consumer.acquire);
    EXPECT_TRUE(consumer.reuse);
    const auto consumer_only = PlanPpTerminalScopePlaneSlot(2, false);
    EXPECT_TRUE(consumer_only.acquire);
    EXPECT_FALSE(consumer_only.reuse);
    const auto output = PlanPpTerminalScopePlaneSlot(3, true);
    EXPECT_TRUE(output.reuse);
    EXPECT_EQ(output.status, FinalGuestSurfaceStatus::Complete);
    const auto output_without_input = PlanPpTerminalScopePlaneSlot(3, false);
    EXPECT_EQ(output_without_input.status, FinalGuestSurfaceStatus::GapLoss);
}

TEST(PpTerminalScopeContent, ExactConsumerFreezesTheReferencedEarlierScope) {
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
        .consumer = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                     .indexed = true,
                     .element_count = 4,
                     .instance_count = 1,
                     .sampled_images = 1},
    };
    PpTerminalScopeContentGate gate{config};
    ASSERT_TRUE(gate.Arm(17, 4));
    ASSERT_EQ(gate.ObserveDraw(17, 90, config.first), PpTerminalScopeContentAction::CaptureFirst);
    ASSERT_EQ(gate.ObserveDraw(17, 90, config.second), PpTerminalScopeContentAction::CaptureSecond);
    auto wrong_consumer = config.consumer;
    wrong_consumer.element_count = 5;
    EXPECT_FALSE(gate.ObserveConsumer(17, wrong_consumer));
    EXPECT_TRUE(gate.ObserveConsumer(17, config.consumer));

    const PpTerminalScopeDrawSelector later_scope{
        .kind = VideoCore::ImageColorScopeDrawKind::Direct,
        .indexed = true,
        .element_count = 3,
        .instance_count = 1,
        .sampled_images = 1,
    };
    EXPECT_EQ(gate.ObserveDraw(17, 91, later_scope), PpTerminalScopeContentAction::None);
    const auto take = gate.Take(17, 4);
    EXPECT_EQ(take.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(take.consumer_observations, 2u);
    EXPECT_EQ(take.consumer_phase_mask, 1u << 2);
    EXPECT_EQ(take.consumer_shape_matches, 1u);
    EXPECT_TRUE(take.consumer_frozen);
}

TEST(PpTerminalScopeContent, ConsumerProbePreservesOrderingAndMismatchWithoutPrivateIdentity) {
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
        .consumer = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                     .indexed = true,
                     .element_count = 4,
                     .instance_count = 1,
                     .sampled_images = 1},
    };
    PpTerminalScopeContentGate gate{config};
    ASSERT_TRUE(gate.Arm(17, 4));
    EXPECT_FALSE(gate.ObserveConsumer(17, config.consumer));
    ASSERT_EQ(gate.ObserveDraw(17, 90, config.first), PpTerminalScopeContentAction::CaptureFirst);
    ASSERT_EQ(gate.ObserveDraw(17, 90, config.second), PpTerminalScopeContentAction::CaptureSecond);
    auto mismatch = config.consumer;
    mismatch.element_count = 5;
    EXPECT_FALSE(gate.ObserveConsumer(17, mismatch));

    const auto take = gate.Take(17, 4);
    EXPECT_EQ(take.consumer_observations, 2u);
    EXPECT_EQ(take.consumer_phase_mask, (1u << 0) | (1u << 2));
    EXPECT_EQ(take.consumer_shape_matches, 1u);
    EXPECT_FALSE(take.consumer_frozen);
}

TEST(PpTerminalScopeContent, ExactLateConsumerFreezesAndRequestsThirdPlane) {
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
        .consumer = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                     .indexed = true,
                     .element_count = 4,
                     .instance_count = 1,
                     .sampled_images = 1},
    };
    PpTerminalScopeContentGate gate{config};
    ASSERT_TRUE(gate.Arm(17, 4));
    ASSERT_EQ(gate.ObserveDraw(17, 90, config.first), PpTerminalScopeContentAction::CaptureFirst);
    ASSERT_EQ(gate.ObserveDraw(17, 90, config.second), PpTerminalScopeContentAction::CaptureSecond);
    EXPECT_EQ(gate.ObserveDraw(17, 90, config.first), PpTerminalScopeContentAction::ShapeLoss);
    EXPECT_EQ(gate.ObserveConsumer(17, config.consumer),
              PpTerminalScopeConsumerAction::CaptureConsumer);
    const auto take = gate.Take(17, 4);
    EXPECT_EQ(take.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(take.draw_count, 3u);
    EXPECT_TRUE(take.consumer_frozen);
}

TEST(PpTerminalScopeContent, SelectedLogicalWindowsProduceFiveBoundedPlanes) {
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
    EXPECT_EQ(plan.copy_region_count, 10u);
    EXPECT_EQ(plan.first_plane_offset, 0u);
    EXPECT_GE(plan.second_plane_offset, plan.plane_bytes);
    EXPECT_GE(plan.consumer_plane_offset, plan.second_plane_offset + plan.plane_bytes);
    EXPECT_GE(plan.output_plane_offset, plan.consumer_plane_offset + plan.plane_bytes);
    EXPECT_GE(plan.pre_first_plane_offset, plan.output_plane_offset + plan.plane_bytes);
    EXPECT_GE(plan.total_bytes, plan.pre_first_plane_offset + plan.plane_bytes);
    EXPECT_LE(plan.total_bytes, PpTerminalScopeSnapshotBytes);
    EXPECT_EQ(plan.image_barriers_per_draw, 2u);
    EXPECT_TRUE(plan.ends_rendering);
    EXPECT_TRUE(plan.resumes_rendering_with_load);
    EXPECT_TRUE(plan.preserves_rendering_serial);
    EXPECT_TRUE(plan.callback_payload_is_scalar_only);
    EXPECT_FALSE(plan.cpu_wait);
    EXPECT_FALSE(plan.finish);
}

TEST(PpTerminalScopeContent, PreFirstPlaneUsesTheExactCalibratedVisualPredicate) {
    constexpr u32 RegionBytes = 4;
    const PpTerminalScopeContentHistoryLayout layout{
        .region_count = 1,
        .plane_bytes = RegionBytes,
        .second_plane_offset = RegionBytes,
        .consumer_plane_offset = RegionBytes * 2,
        .output_plane_offset = RegionBytes * 3,
        .pre_first_plane_offset = RegionBytes * 4,
        .total_bytes = RegionBytes * 5,
        .plane_mask = 0x1f,
        .regions = {{{.logical_ordinal = 77, .buffer_offset = 0, .byte_size = RegionBytes}}},
        .format = FinalGuestSurfaceFormat::Rgba8,
    };
    const auto observation = [](std::array<u8, 4> pre) {
        std::array<std::byte, RegionBytes * 5> bytes{};
        for (u32 index = 0; index < pre.size(); ++index) {
            bytes[RegionBytes * 4 + index] = std::byte{pre[index]};
        }
        return bytes;
    };
    PpTerminalScopeContentReducer reducer{{.frame_start = 100, .frame_count = 3}, 8};
    reducer.ObserveContent(100, layout, observation({1, 2, 3, 255}),
                           FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(101, layout, observation({32, 33, 34, 255}),
                           FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(102, layout, observation({1, 2, 3, 0}),
                           FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveCalibration({.request_ordinal = 1, .sequence = 100, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 2, .sequence = 101, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 3, .sequence = 102, .valid = true});
    const auto reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports[0].pre_first_aba_ordinals, (std::vector<u32>{77}));
    EXPECT_TRUE(reports[0].pre_first_stable_ordinals.empty());
    EXPECT_TRUE(reports[0].pre_first_ambiguous_ordinals.empty());
    EXPECT_EQ(reports[0].pre_first_localized_visual_return_ordinals, (std::vector<u32>{77}));
    EXPECT_NE(FormatPpTerminalScopeCalibratedReport(reports[0]).find(" y4=77"), std::string::npos);
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
         .second = {.kind = VideoCore::ImageColorScopeDrawKind::Direct},
         .consumer = {.kind = VideoCore::ImageColorScopeDrawKind::Direct}}};
    ASSERT_TRUE(gate.Arm(7, 10));
    EXPECT_EQ(gate.ObserveDraw(7, 100, {.kind = VideoCore::ImageColorScopeDrawKind::Direct}),
              PpTerminalScopeContentAction::CaptureFirst);
    const auto stale = gate.Take(7, 11);
    EXPECT_EQ(stale.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(stale.loss.invalidation, 1u);
}

TEST(PpTerminalScopeContent, CompactGrammarNeverExposesPrivateTargetToken) {
    auto report = MakePpTerminalScopeContentReport(4100, FinalGuestSurfaceStatus::Complete, {}, 2,
                                                   14, 3, 5, 2, true);
    report.plane_mask = 7;
    report.first_aba = 1;
    report.first_stable = 2;
    report.second_aba = 3;
    report.second_stable = 4;
    report.lineage_hops = 1;
    report.lineage_status = FinalGuestSurfaceStatus::Complete;
    const auto line = FormatPpTerminalScopeContentReport(report);
    EXPECT_EQ(line, "FGSCTS s=4100 st=0 d=2 r=14 pm=7 co=3 cp=5 cm=2 cf=1 a0=1 s0=2 a1=3 "
                    "s1=4 lh=1 ls=0 ll=0 lm=0");
    EXPECT_EQ(line.find("token"), std::string::npos);
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);

    const auto shape_loss =
        MakePpTerminalScopeContentReport(4101, FinalGuestSurfaceStatus::GapLoss, {.gap = 1}, 3, 14);
    EXPECT_EQ(shape_loss.draw_count, 3u);
    EXPECT_NE(FormatPpTerminalScopeContentReport(shape_loss).find("st=5 d=3"), std::string::npos);
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

TEST(PpTerminalScopeContent, PrivateLineageJoinsExactRotatingInputOutputGenerations) {
    const VideoCore::ImageColorScopePrivateLink root_a{VideoCore::ImageId{41}, 7001};
    const VideoCore::ImageColorScopePrivateLink output_a{VideoCore::ImageId{51}, 8001};
    const VideoCore::ImageColorScopePrivateLink root_b{VideoCore::ImageId{42}, 7002};
    const VideoCore::ImageColorScopePrivateLink output_b{VideoCore::ImageId{52}, 8002};
    PpTerminalScopePrivateLineage lineage_a{};
    PpTerminalScopePrivateLineage lineage_b{};
    ASSERT_TRUE(lineage_a.Start(root_a, 11));
    ASSERT_TRUE(lineage_b.Start(root_b, 12));
    EXPECT_EQ(lineage_a.Extend({root_a, 11, output_a, 21, true, true}).status,
              FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(lineage_b.Extend({root_b, 12, output_b, 22, true, true}).status,
              FinalGuestSurfaceStatus::Complete);

    const auto resolved_a = lineage_a.Resolve(output_a, 21);
    EXPECT_TRUE(resolved_a.matched);
    EXPECT_EQ(resolved_a.hops, 1u);
    EXPECT_FALSE(resolved_a.retains_pointer);
    EXPECT_FALSE(resolved_a.retains_image);
    EXPECT_FALSE(resolved_a.retains_vk_image);
    EXPECT_FALSE(lineage_a.Resolve(output_b, 22).matched);
    EXPECT_FALSE(lineage_b.Resolve(output_a, 21).matched);
}

TEST(PpTerminalScopeContent, PrivateLineageFailsClosedOnReuseAmbiguityCycleAndDepth) {
    const VideoCore::ImageColorScopePrivateLink root{VideoCore::ImageId{61}, 9001};
    const VideoCore::ImageColorScopePrivateLink output{VideoCore::ImageId{62}, 9002};
    PpTerminalScopePrivateLineage stale{};
    ASSERT_TRUE(stale.Start(root, 31));
    EXPECT_EQ(stale.Extend({root, 32, output, 41, true, true}).status,
              FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(stale.Loss().invalidation, 1u);

    PpTerminalScopePrivateLineage ambiguous{};
    ASSERT_TRUE(ambiguous.Start(root, 31));
    EXPECT_EQ(ambiguous.Extend({root, 31, output, 41, false, true}).status,
              FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(ambiguous.Loss().gap, 1u);

    PpTerminalScopePrivateLineage cycle{};
    ASSERT_TRUE(cycle.Start(root, 31));
    ASSERT_EQ(cycle.Extend({root, 31, output, 41, true, true}).status,
              FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(cycle.Extend({output, 41, root, 31, true, true}).status,
              FinalGuestSurfaceStatus::InvalidationLoss);

    PpTerminalScopePrivateLineage capped{};
    ASSERT_TRUE(capped.Start({VideoCore::ImageId{70}, 1000}, 100));
    for (u32 index = 1; index < PpTerminalScopePrivateLineage::MaxDepth; ++index) {
        const auto tail = VideoCore::ImageColorScopePrivateLink{VideoCore::ImageId{70 + index - 1},
                                                                1000 + index - 1};
        const auto next =
            VideoCore::ImageColorScopePrivateLink{VideoCore::ImageId{70 + index}, 1000 + index};
        ASSERT_EQ(capped.Extend({tail, 100 + index - 1, next, 100 + index, true, true}).status,
                  FinalGuestSurfaceStatus::Complete);
    }
    const auto overflow =
        capped.Extend({{VideoCore::ImageId{70 + PpTerminalScopePrivateLineage::MaxDepth - 1},
                        1000 + PpTerminalScopePrivateLineage::MaxDepth - 1},
                       100 + PpTerminalScopePrivateLineage::MaxDepth - 1,
                       {VideoCore::ImageId{90}, 2000},
                       200,
                       true,
                       true});
    EXPECT_EQ(overflow.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(overflow.loss.tile_capacity, 1u);
}

TEST(PpTerminalScopeContent, PrivateLineageReportFormatsOnlyHopStatusAndLoss) {
    const auto line = FormatPpTerminalScopePrivateLineageReport({
        .hops = 2,
        .status = FinalGuestSurfaceStatus::Complete,
    });
    EXPECT_EQ(line, "lh=2 ls=0 ll=0");
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("image"), std::string::npos);
    EXPECT_EQ(line.find("generation"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
}

TEST(PpTerminalScopeContent, LineageHandoffRequiresPreAndTwoProducerPlanesAndOneOutput) {
    const auto complete =
        PlanPpTerminalScopeLineageHandoff(true, FinalGuestSurfaceStatus::Complete, {}, 0x13, true);
    EXPECT_EQ(complete.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_TRUE(complete.capture_consumer);
    EXPECT_TRUE(complete.capture_output);
    EXPECT_TRUE(complete.publish_flip_alias);

    const auto missing_plane =
        PlanPpTerminalScopeLineageHandoff(true, FinalGuestSurfaceStatus::Complete, {}, 1, true);
    EXPECT_EQ(missing_plane.status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(missing_plane.loss.gap, 1u);
    EXPECT_FALSE(missing_plane.capture_consumer);
    EXPECT_FALSE(missing_plane.capture_output);
    EXPECT_FALSE(missing_plane.publish_flip_alias);

    const auto ambiguous_output =
        PlanPpTerminalScopeLineageHandoff(true, FinalGuestSurfaceStatus::Complete, {}, 0x13, false);
    EXPECT_EQ(ambiguous_output.status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(ambiguous_output.loss.gap, 1u);
    EXPECT_FALSE(ambiguous_output.publish_flip_alias);
}

TEST(PpTerminalScopeContent, LineageHandoffPropagatesStaleStateAndIgnoresOtherDraws) {
    const auto stale = PlanPpTerminalScopeLineageHandoff(
        true, FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1}, 3, true);
    EXPECT_EQ(stale.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(stale.loss.invalidation, 1u);
    EXPECT_FALSE(stale.capture_consumer);

    const auto unrelated =
        PlanPpTerminalScopeLineageHandoff(false, FinalGuestSurfaceStatus::Complete, {}, 3, true);
    EXPECT_EQ(unrelated.status, FinalGuestSurfaceStatus::AlreadyConsumed);
    EXPECT_FALSE(unrelated.capture_consumer);
    EXPECT_FALSE(unrelated.publish_flip_alias);
}

TEST(PpTerminalScopeContent, ConsumedLineageLeavesCapacityForTheNextDiscoveredProducer) {
    EXPECT_FALSE(ShouldArmPpTerminalScopeFallbackAfterFlip(true));
    EXPECT_TRUE(ShouldArmPpTerminalScopeFallbackAfterFlip(false));
}

TEST(PpTerminalScopeContent, ExactCalibratedTripletClassifiesAllFourPlanesPerOrdinal) {
    PpTerminalScopeContentReducer reducer{{.frame_start = 100, .frame_count = 3}, 8};
    const PpTerminalScopeContentHistoryLayout layout{
        .region_count = 2,
        .plane_bytes = 8,
        .second_plane_offset = 8,
        .consumer_plane_offset = 16,
        .output_plane_offset = 24,
        .total_bytes = 32,
        .plane_mask = 15,
        .regions = {{{.logical_ordinal = 11, .buffer_offset = 0, .byte_size = 4},
                     {.logical_ordinal = 12, .buffer_offset = 4, .byte_size = 4}}},
        .format = FinalGuestSurfaceFormat::Rgba8,
    };
    const auto bytes = [](std::array<u8, 4> first0, std::array<u8, 4> first1,
                          std::array<u8, 4> second0, std::array<u8, 4> second1,
                          std::array<u8, 4> consumer0, std::array<u8, 4> consumer1,
                          std::array<u8, 4> output0, std::array<u8, 4> output1) {
        std::array<std::byte, 32> result{};
        const std::array planes{first0,    first1,    second0, second1,
                                consumer0, consumer1, output0, output1};
        for (u32 plane = 0; plane < planes.size(); ++plane) {
            for (u32 index = 0; index < planes[plane].size(); ++index) {
                result[plane * 4 + index] = std::byte{planes[plane][index]};
            }
        }
        return result;
    };
    const auto a =
        bytes({1, 2, 3, 255}, {4, 5, 6, 255}, {7, 8, 9, 255}, {10, 11, 12, 255}, {13, 14, 15, 255},
              {16, 17, 18, 255}, {21, 22, 23, 255}, {24, 25, 26, 255});
    const auto b =
        bytes({9, 9, 9, 255}, {4, 5, 6, 255}, {7, 8, 9, 255}, {99, 98, 97, 255}, {88, 87, 86, 255},
              {16, 17, 18, 255}, {77, 76, 75, 255}, {24, 25, 26, 255});
    const auto c = bytes({1, 2, 3, 1}, {8, 8, 8, 255}, {7, 8, 9, 0}, {10, 11, 12, 255},
                         {13, 14, 15, 0}, {19, 20, 21, 255}, {21, 22, 23, 0}, {24, 25, 26, 255});
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
    EXPECT_EQ(reports[0].consumer_aba_ordinals, (std::vector<u32>{11}));
    EXPECT_EQ(reports[0].consumer_stable_ordinals, (std::vector<u32>{}));
    EXPECT_EQ(reports[0].consumer_ambiguous_ordinals, (std::vector<u32>{12}));
    EXPECT_EQ(reports[0].output_aba_ordinals, (std::vector<u32>{11}));
    EXPECT_EQ(reports[0].output_stable_ordinals, (std::vector<u32>{12}));
    EXPECT_EQ(reports[0].output_ambiguous_ordinals, (std::vector<u32>{}));
    EXPECT_FALSE(reports[0].loss.Any());
}

TEST(PpTerminalScopeContent, ExactCalibratedTripletClassifiesCapturedInputRegionsIndependently) {
    PpTerminalScopeContentReducer reducer{{.frame_start = 200, .frame_count = 3}, 8};
    const PpTerminalScopeContentHistoryLayout layout{
        .total_bytes = 4,
        .input_count = 2,
        .input_capture_mask = 0b10,
        .input_unavailable_mask = 0b01,
        .input_alias_mask = 0b10,
        .input_planes =
            {{{.status = FinalGuestSurfaceStatus::Unsupported, .loss = {.unsupported_format = 1}},
              {.region_count = 1,
               .plane_offset = 0,
               .plane_bytes = 4,
               .regions = {{{.logical_ordinal = 42, .buffer_offset = 0, .byte_size = 4}}},
               .format = FinalGuestSurfaceFormat::Rgba8,
               .status = FinalGuestSurfaceStatus::Complete}}},
        .format = FinalGuestSurfaceFormat::Rgba8,
    };
    const std::array<std::byte, 4> a{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{255}};
    const std::array<std::byte, 4> b{std::byte{9}, std::byte{8}, std::byte{7}, std::byte{255}};
    const std::array<std::byte, 4> c{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{0}};
    reducer.ObserveContent(200, layout, a, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(201, layout, b, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(202, layout, c, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveCalibration({.request_ordinal = 1, .sequence = 200, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 2, .sequence = 201, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 3, .sequence = 202, .valid = true});
    const auto reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports[0].input_capture_mask, 0b10u);
    EXPECT_EQ(reports[0].input_unavailable_mask, 0b01u);
    EXPECT_EQ(reports[0].input_alias_mask, 0b10u);
    EXPECT_EQ(reports[0].sampled_input_aba_ordinals[1], (std::vector<u32>{42}));
    EXPECT_TRUE(reports[0].sampled_input_stable_ordinals[1].empty());
    EXPECT_TRUE(reports[0].sampled_input_ambiguous_ordinals[1].empty());
    EXPECT_FALSE(reports[0].loss.Any());

    const auto line = FormatPpTerminalScopeCalibratedReport(reports[0]);
    EXPECT_NE(line.find(" im=2/2/1/2"), std::string::npos);
    EXPECT_NE(line.find(" z1a=42 z1s=- z1x=-"), std::string::npos);
    EXPECT_EQ(line.find("byte"), std::string::npos);
    EXPECT_EQ(line.find("hash"), std::string::npos);
    EXPECT_EQ(line.find("uid"), std::string::npos);
}

TEST(PpTerminalScopeContent, CapturedInputUsesTheLocalizedVisualPredicateAtItsOwnExtent) {
    PpTerminalScopeContentReducer reducer{{.frame_start = 300, .frame_count = 3}, 8};
    constexpr u32 RegionBytes = 4 * 4;
    const PpTerminalScopeContentHistoryLayout layout{
        .total_bytes = RegionBytes,
        .input_count = 2,
        .input_capture_mask = 0b10,
        .input_unavailable_mask = 0b01,
        .input_planes =
            {{{.status = FinalGuestSurfaceStatus::Unsupported, .loss = {.unsupported_format = 1}},
              {.region_count = 1,
               .plane_offset = 0,
               .plane_bytes = RegionBytes,
               .regions = {{{.logical_ordinal = 77, .buffer_offset = 0, .byte_size = RegionBytes}}},
               .format = FinalGuestSurfaceFormat::Bgra8,
               .status = FinalGuestSurfaceStatus::Complete}}},
        .format = FinalGuestSurfaceFormat::Rgba8,
    };
    const auto pixels = [](u8 color) {
        std::array<std::byte, RegionBytes> bytes{};
        for (u32 offset = 0; offset < RegionBytes; offset += 4) {
            bytes[offset + 0] = std::byte{color};
            bytes[offset + 1] = std::byte{color};
            bytes[offset + 2] = std::byte{color};
            bytes[offset + 3] = std::byte{255};
        }
        return bytes;
    };
    reducer.ObserveContent(300, layout, pixels(10), FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(301, layout, pixels(100), FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(302, layout, pixels(11), FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveCalibration({.request_ordinal = 1, .sequence = 300, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 2, .sequence = 301, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 3, .sequence = 302, .valid = true});
    const auto reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports[0].sampled_input_ambiguous_ordinals[1], (std::vector<u32>{77}))
        << "A and C are deliberately not byte-identical";
    EXPECT_EQ(reports[0].sampled_input_localized_visual_return_ordinals[1], (std::vector<u32>{77}));
    EXPECT_FALSE(reports[0].loss.Any());
    const auto line = FormatPpTerminalScopeCalibratedReport(reports[0]);
    EXPECT_NE(line.find(" z1y=77"), std::string::npos);
}

TEST(PpTerminalScopeContent, OutputPlaneUsesTheExactLocalizedVisualReturnPredicate) {
    constexpr u32 PixelCount = 32 * 32;
    constexpr u32 RegionBytes = PixelCount * 4;
    std::array<std::byte, RegionBytes> baseline{};
    std::array<std::byte, RegionBytes> departure{};
    std::array<std::byte, RegionBytes> returned{};
    for (u32 pixel = 0; pixel < PixelCount; ++pixel) {
        const u32 offset = pixel * 4;
        baseline[offset + 0] = std::byte{10};
        baseline[offset + 1] = std::byte{20};
        baseline[offset + 2] = std::byte{30};
        baseline[offset + 3] = std::byte{255};
        departure[offset + 0] = std::byte{10};
        departure[offset + 1] = std::byte{20};
        departure[offset + 2] = std::byte{30};
        departure[offset + 3] = std::byte{0};
        returned[offset + 0] = std::byte{10};
        returned[offset + 1] = std::byte{20};
        returned[offset + 2] = std::byte{30};
        returned[offset + 3] = std::byte{1};
    }
    // Three hundred departure pixels leave enough headroom for ten legitimate return-motion
    // pixels while satisfying the exact 25% / 25% / 1% Test Lab predicate.
    for (u32 pixel = 0; pixel < 300; ++pixel) {
        departure[pixel * 4 + 0] = std::byte{26};
        departure[pixel * 4 + 1] = std::byte{36};
        departure[pixel * 4 + 2] = std::byte{46};
    }
    for (u32 pixel = 0; pixel < 10; ++pixel) {
        returned[pixel * 4 + 0] = std::byte{26};
        returned[pixel * 4 + 1] = std::byte{36};
        returned[pixel * 4 + 2] = std::byte{46};
    }

    ASSERT_EQ(IsPpTerminalScopeLocalizedVisualReturn(FinalGuestSurfaceFormat::Rgba8, baseline,
                                                     departure, returned),
              std::optional<bool>{true});

    PpTerminalScopeContentReducer reducer{{.frame_start = 100, .frame_count = 3}, 8};
    const PpTerminalScopeContentHistoryLayout layout{
        .region_count = 1,
        .plane_bytes = RegionBytes,
        .second_plane_offset = RegionBytes,
        .consumer_plane_offset = RegionBytes * 2,
        .output_plane_offset = RegionBytes * 3,
        .total_bytes = RegionBytes * 4,
        .plane_mask = 1u << 3,
        .regions = {{{.logical_ordinal = 41, .buffer_offset = 0, .byte_size = RegionBytes}}},
        .format = FinalGuestSurfaceFormat::Rgba8,
    };
    const auto observation = [](const auto& output) {
        std::array<std::byte, RegionBytes * 4> bytes{};
        std::ranges::copy(output, bytes.begin() + RegionBytes * 3);
        return bytes;
    };
    reducer.ObserveContent(100, layout, observation(baseline), FinalGuestSurfaceStatus::Complete,
                           {});
    reducer.ObserveContent(101, layout, observation(departure), FinalGuestSurfaceStatus::Complete,
                           {});
    reducer.ObserveContent(102, layout, observation(returned), FinalGuestSurfaceStatus::Complete,
                           {});
    reducer.ObserveCalibration({.request_ordinal = 1, .sequence = 100, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 2, .sequence = 101, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 3, .sequence = 102, .valid = true});
    const auto reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports[0].output_ambiguous_ordinals, (std::vector<u32>{41}));
    EXPECT_EQ(reports[0].output_localized_visual_return_ordinals, (std::vector<u32>{41}));
    const auto line = FormatPpTerminalScopeCalibratedReport(reports[0]);
    EXPECT_NE(line.find(" y3=41"), std::string::npos);
}

TEST(PpTerminalScopeContent, EveryRetainedPlaneUsesTheExactLocalizedVisualReturnPredicate) {
    constexpr u32 PixelCount = 32 * 32;
    constexpr u32 RegionBytes = PixelCount * 4;
    constexpr u32 PlaneBytes = RegionBytes * 2;
    std::array<std::byte, RegionBytes> departure{};
    std::array<std::byte, RegionBytes> returned{};
    for (u32 pixel = 0; pixel < 300; ++pixel) {
        departure[pixel * 4 + 0] = std::byte{16};
        departure[pixel * 4 + 1] = std::byte{16};
        departure[pixel * 4 + 2] = std::byte{16};
    }
    for (u32 pixel = 0; pixel < 10; ++pixel) {
        returned[pixel * 4 + 0] = std::byte{16};
        returned[pixel * 4 + 1] = std::byte{16};
        returned[pixel * 4 + 2] = std::byte{16};
    }

    const PpTerminalScopeContentHistoryLayout layout{
        .region_count = 2,
        .plane_bytes = PlaneBytes,
        .second_plane_offset = PlaneBytes,
        .consumer_plane_offset = PlaneBytes * 2,
        .output_plane_offset = PlaneBytes * 3,
        .total_bytes = PlaneBytes * 4,
        .plane_mask = 0xf,
        .regions =
            {{{.logical_ordinal = 41, .buffer_offset = 0, .byte_size = RegionBytes},
              {.logical_ordinal = 42, .buffer_offset = RegionBytes, .byte_size = RegionBytes}}},
        .format = FinalGuestSurfaceFormat::Rgba8,
    };
    const auto observation = [&](bool is_departure, bool is_returned) {
        std::array<std::byte, PlaneBytes * 4> bytes{};
        const auto& changed = is_departure ? departure : returned;
        if (is_departure || is_returned) {
            // Plane 0 returns only in region 41; plane 1 only in region 42; plane 2 in both;
            // plane 3 remains stable. Distinct patterns prove that each field reads its plane.
            std::ranges::copy(changed, bytes.begin());
            std::ranges::copy(changed, bytes.begin() + PlaneBytes + RegionBytes);
            std::ranges::copy(changed, bytes.begin() + PlaneBytes * 2);
            std::ranges::copy(changed, bytes.begin() + PlaneBytes * 2 + RegionBytes);
        }
        return bytes;
    };

    PpTerminalScopeContentReducer reducer{{.frame_start = 100, .frame_count = 3}, 8};
    reducer.ObserveContent(100, layout, observation(false, false),
                           FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(101, layout, observation(true, false), FinalGuestSurfaceStatus::Complete,
                           {});
    reducer.ObserveContent(102, layout, observation(false, true), FinalGuestSurfaceStatus::Complete,
                           {});
    reducer.ObserveCalibration({.request_ordinal = 1, .sequence = 100, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 2, .sequence = 101, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 3, .sequence = 102, .valid = true});

    const auto reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports[0].first_localized_visual_return_ordinals, (std::vector<u32>{41}));
    EXPECT_EQ(reports[0].second_localized_visual_return_ordinals, (std::vector<u32>{42}));
    EXPECT_EQ(reports[0].consumer_localized_visual_return_ordinals, (std::vector<u32>{41, 42}));
    EXPECT_TRUE(reports[0].output_localized_visual_return_ordinals.empty());
    const auto line = FormatPpTerminalScopeCalibratedReport(reports[0]);
    EXPECT_NE(line.find(" y0=41"), std::string::npos);
    EXPECT_NE(line.find(" y1=42"), std::string::npos);
    EXPECT_NE(line.find(" y2=41,42"), std::string::npos);
    EXPECT_NE(line.find(" y3=-"), std::string::npos);
}

TEST(PpTerminalScopeContent, LocalizedVisualReturnHonorsExactThresholdsAndFailsClosed) {
    constexpr u32 PixelCount = 32 * 32;
    std::array<std::byte, PixelCount * 4> a{};
    std::array<std::byte, PixelCount * 4> b{};
    std::array<std::byte, PixelCount * 4> c{};
    const auto set_changed = [](auto& bytes, u32 pixels, u8 red, u8 green, u8 blue) {
        for (u32 pixel = 0; pixel < pixels; ++pixel) {
            bytes[pixel * 4 + 0] = std::byte{red};
            bytes[pixel * 4 + 1] = std::byte{green};
            bytes[pixel * 4 + 2] = std::byte{blue};
        }
    };

    set_changed(b, 256, 16, 16, 16);
    EXPECT_EQ(IsPpTerminalScopeLocalizedVisualReturn(FinalGuestSurfaceFormat::Bgra8, a, b, c),
              std::optional<bool>{true});

    b = {};
    set_changed(b, 255, 16, 16, 16); // One pixel below the exact 25% floor.
    EXPECT_EQ(IsPpTerminalScopeLocalizedVisualReturn(FinalGuestSurfaceFormat::Rgba8, a, b, c),
              std::optional<bool>{false});
    b = {};
    set_changed(b, 256, 16, 16, 15); // RGB sum 47 is below the per-pixel threshold.
    EXPECT_EQ(IsPpTerminalScopeLocalizedVisualReturn(FinalGuestSurfaceFormat::Rgba8, a, b, c),
              std::optional<bool>{false});
    set_changed(b, 256, 16, 16, 16);
    set_changed(c, 11, 16, 16, 16); // Eleven of 1024 pixels exceed the 1% return ceiling.
    EXPECT_EQ(IsPpTerminalScopeLocalizedVisualReturn(FinalGuestSurfaceFormat::Rgba8, a, b, c),
              std::optional<bool>{false});

    c = {};
    for (u32 pixel = 0; pixel < PixelCount; ++pixel) {
        a[pixel * 4 + 3] = std::byte{255};
        b[pixel * 4 + 3] = std::byte{1};
        c[pixel * 4 + 3] = std::byte{127};
    }
    EXPECT_EQ(IsPpTerminalScopeLocalizedVisualReturn(FinalGuestSurfaceFormat::Rgba8, a, b, c),
              std::optional<bool>{true});
    EXPECT_EQ(IsPpTerminalScopeLocalizedVisualReturn(FinalGuestSurfaceFormat::Unsupported, a, b, c),
              std::nullopt);
    EXPECT_EQ(IsPpTerminalScopeLocalizedVisualReturn(FinalGuestSurfaceFormat::Rgba8,
                                                     std::span{a}.first(a.size() - 1), b, c),
              std::nullopt);
}

TEST(PpTerminalScopeContent, ExactFinalBackingJoinCoversEveryCalibratedOutputEndpoint) {
    PpTerminalScopeContentReducer reducer{{.frame_start = 100, .frame_count = 3}, 8, true};
    const PpTerminalScopeContentHistoryLayout content_layout{
        .region_count = 2,
        .plane_bytes = 8,
        .second_plane_offset = 8,
        .consumer_plane_offset = 16,
        .output_plane_offset = 24,
        .total_bytes = 32,
        .plane_mask = 15,
        .regions = {{{.logical_ordinal = 11, .buffer_offset = 0, .byte_size = 4},
                     {.logical_ordinal = 12, .buffer_offset = 4, .byte_size = 4}}},
        .format = FinalGuestSurfaceFormat::Bgra8,
    };
    const PpTerminalScopeFinalBackingLayout backing_layout{
        .region_count = 2,
        .total_bytes = 8,
        .regions = {{{.logical_ordinal = 11, .buffer_offset = 0, .byte_size = 4},
                     {.logical_ordinal = 12, .buffer_offset = 4, .byte_size = 4}}},
        .format = FinalGuestSurfaceFormat::Bgra8,
    };
    const auto content = [](std::array<u8, 4> output0, std::array<u8, 4> output1) {
        std::array<std::byte, 32> bytes{};
        for (u32 index = 0; index < 4; ++index) {
            bytes[24 + index] = std::byte{output0[index]};
            bytes[28 + index] = std::byte{output1[index]};
        }
        return bytes;
    };
    const auto backing = [](std::array<u8, 4> region0, std::array<u8, 4> region1) {
        std::array<std::byte, 8> bytes{};
        for (u32 index = 0; index < 4; ++index) {
            bytes[index] = std::byte{region0[index]};
            bytes[4 + index] = std::byte{region1[index]};
        }
        return bytes;
    };

    const auto a = content({1, 2, 3, 4}, {5, 6, 7, 8});
    const auto b = content({9, 10, 11, 12}, {13, 14, 15, 16});
    const auto c = content({17, 18, 19, 20}, {21, 22, 23, 24});
    reducer.ObserveContent(100, content_layout, a, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(101, content_layout, b, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(102, content_layout, c, FinalGuestSurfaceStatus::Complete, {});

    // Alpha is not guest-visible at this boundary and must not create a false mismatch.
    const auto backing_a = backing({1, 2, 3, 255}, {5, 6, 7, 8});
    const auto backing_b = backing({9, 10, 11, 0}, {99, 14, 15, 16});
    const auto backing_c = backing({17, 18, 19, 1}, {21, 22, 23, 24});
    reducer.ObserveFinalBacking(101, backing_layout, backing_b, FinalGuestSurfaceStatus::Complete,
                                {});
    reducer.ObserveFinalBacking(100, backing_layout, backing_a, FinalGuestSurfaceStatus::Complete,
                                {});
    reducer.ObserveFinalBacking(102, backing_layout, backing_c, FinalGuestSurfaceStatus::Complete,
                                {});
    reducer.ObserveCalibration({.request_ordinal = 1, .sequence = 100, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 2, .sequence = 101, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 3, .sequence = 102, .valid = true});

    const auto reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports[0].output_final_backing_equal_ordinals, (std::vector<u32>{11}));
    EXPECT_EQ(reports[0].output_final_backing_different_ordinals, (std::vector<u32>{12}));
    EXPECT_FALSE(reports[0].loss.Any());
    const auto line = FormatPpTerminalScopeCalibratedReport(reports[0]);
    EXPECT_NE(line.find(" e3=11 d3=12"), std::string::npos);
    EXPECT_EQ(line.find("pixel"), std::string::npos);
    EXPECT_EQ(line.find("hash"), std::string::npos);
    EXPECT_EQ(line.find("uid"), std::string::npos);
}

TEST(PpTerminalScopeContent, FinalBackingJoinFailsClosedOnMissingLossOrLayoutMismatch) {
    PpTerminalScopeContentReducer missing{{.frame_start = 200, .frame_count = 3}, 8, true};
    const PpTerminalScopeContentHistoryLayout content_layout{
        .region_count = 1,
        .plane_bytes = 4,
        .second_plane_offset = 4,
        .consumer_plane_offset = 8,
        .output_plane_offset = 12,
        .total_bytes = 16,
        .plane_mask = 15,
        .regions = {{{.logical_ordinal = 31, .buffer_offset = 0, .byte_size = 4}}},
        .format = FinalGuestSurfaceFormat::Rgba8,
    };
    const PpTerminalScopeFinalBackingLayout backing_layout{
        .region_count = 1,
        .total_bytes = 4,
        .regions = {{{.logical_ordinal = 31, .buffer_offset = 0, .byte_size = 4}}},
        .format = FinalGuestSurfaceFormat::Rgba8,
    };
    const std::array<std::byte, 16> content{};
    const std::array<std::byte, 4> backing{};
    for (u64 sequence = 200; sequence <= 202; ++sequence) {
        missing.ObserveContent(sequence, content_layout, content, FinalGuestSurfaceStatus::Complete,
                               {});
    }
    missing.ObserveFinalBacking(200, backing_layout, backing, FinalGuestSurfaceStatus::Complete,
                                {});
    missing.ObserveFinalBacking(202, backing_layout, backing, FinalGuestSurfaceStatus::Complete,
                                {});
    missing.ObserveCalibration({.request_ordinal = 1, .sequence = 200, .valid = true});
    missing.ObserveCalibration({.request_ordinal = 2, .sequence = 201, .valid = true});
    missing.ObserveCalibration({.request_ordinal = 3, .sequence = 202, .valid = true});
    auto reports = missing.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports[0].status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(reports[0].loss.history, 1u);

    PpTerminalScopeContentReducer invalid{{.frame_start = 300, .frame_count = 3}, 8, true};
    auto mismatched = backing_layout;
    mismatched.regions[0].byte_size = 3;
    for (u64 sequence = 300; sequence <= 302; ++sequence) {
        invalid.ObserveContent(sequence, content_layout, content, FinalGuestSurfaceStatus::Complete,
                               {});
        invalid.ObserveFinalBacking(sequence, mismatched, backing,
                                    FinalGuestSurfaceStatus::Complete, {});
    }
    invalid.ObserveCalibration({.request_ordinal = 1, .sequence = 300, .valid = true});
    invalid.ObserveCalibration({.request_ordinal = 2, .sequence = 301, .valid = true});
    invalid.ObserveCalibration({.request_ordinal = 3, .sequence = 302, .valid = true});
    reports = invalid.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports[0].status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(reports[0].loss.invalidation, 1u);
}

TEST(PpTerminalScopeContent, ConsumerOnlyPlaneIsCompleteWithoutClassifyingMissingPlanes) {
    PpTerminalScopeContentReducer reducer{{.frame_start = 300, .frame_count = 3}, 8};
    const PpTerminalScopeContentHistoryLayout layout{
        .region_count = 1,
        .plane_bytes = 4,
        .second_plane_offset = 4,
        .consumer_plane_offset = 8,
        .total_bytes = 12,
        .plane_mask = 1u << 2,
        .regions = {{{.logical_ordinal = 31, .buffer_offset = 0, .byte_size = 4}}},
        .format = FinalGuestSurfaceFormat::Rgba8,
    };
    const std::array<std::byte, 12> a{std::byte{99}, std::byte{99}, std::byte{99}, std::byte{99},
                                      std::byte{88}, std::byte{88}, std::byte{88}, std::byte{88},
                                      std::byte{1},  std::byte{2},  std::byte{3},  std::byte{255}};
    auto b = a;
    b[8] = std::byte{9};
    auto c = a;
    c[11] = std::byte{0};
    reducer.ObserveContent(300, layout, a, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(301, layout, b, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(302, layout, c, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveCalibration({.request_ordinal = 1, .sequence = 300, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 2, .sequence = 301, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 3, .sequence = 302, .valid = true});
    const auto reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_TRUE(reports[0].first_aba_ordinals.empty());
    EXPECT_TRUE(reports[0].second_aba_ordinals.empty());
    EXPECT_EQ(reports[0].consumer_aba_ordinals, (std::vector<u32>{31}));
    EXPECT_FALSE(reports[0].loss.Any());
}

TEST(PpTerminalScopeContent, MissingChangedOrEvictedContentFailsClosed) {
    PpTerminalScopeContentReducer reducer{{.frame_start = 200, .frame_count = 40}, 3};
    const PpTerminalScopeContentHistoryLayout layout{
        .region_count = 1,
        .plane_bytes = 4,
        .second_plane_offset = 4,
        .consumer_plane_offset = 8,
        .total_bytes = 12,
        .plane_mask = 7,
        .regions = {{{.logical_ordinal = 21, .buffer_offset = 0, .byte_size = 4}}},
        .format = FinalGuestSurfaceFormat::Rgba8,
    };
    const std::array<std::byte, 12> bytes{};
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
        .consumer_aba_ordinals = {15},
        .consumer_stable_ordinals = {16},
        .consumer_ambiguous_ordinals = {17},
        .output_aba_ordinals = {18},
        .output_stable_ordinals = {19},
        .output_ambiguous_ordinals = {20},
        .status = FinalGuestSurfaceStatus::Complete,
    };
    const auto line = FormatPpTerminalScopeCalibratedReport(report);
    EXPECT_NE(line.find("FGSCTST q=19 abc=700/706/711"), std::string::npos);
    EXPECT_NE(line.find("a0=11,12"), std::string::npos);
    EXPECT_NE(line.find("s0=13"), std::string::npos);
    EXPECT_NE(line.find("x1=14"), std::string::npos);
    EXPECT_NE(line.find("a2=15"), std::string::npos);
    EXPECT_NE(line.find("s2=16"), std::string::npos);
    EXPECT_NE(line.find("x2=17"), std::string::npos);
    EXPECT_NE(line.find("a3=18"), std::string::npos);
    EXPECT_NE(line.find("s3=19"), std::string::npos);
    EXPECT_NE(line.find("x3=20"), std::string::npos);
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
        {"SHADPS4_PP_TERMINAL_SCOPE_INPUT_CONTENT", "1"},
        {"SHADPS4_PP_TERMINAL_SCOPE_PREDECESSOR", "direct,indexed,4,1,3,0"},
        {"SHADPS4_PP_TERMINAL_SCOPE_UPSTREAM", "direct,indexed,4,1,6,0"},
        {"SHADPS4_PP_TERMINAL_SCOPE_UPSTREAM_INPUT_INDEX", "1"},
        {"SHADPS4_PP_TERMINAL_SCOPE_UPSTREAM_PRE_POST", "1"},
        {"SHADPS4_PP_TERMINAL_SCOPE_UPSTREAM_INPUT_CONTENT", "1"},
        {"SHADPS4_PP_TERMINAL_SCOPE_UPSTREAM_CANDIDATE", "0"},
        {"SHADPS4_PP_TERMINAL_FINAL_BACKING_JOIN", "1"},
        {"SHADPS4_PP_TERMINAL_SCOPE_FIRST", "direct,indexed,696,1,1,0"},
        {"SHADPS4_PP_TERMINAL_SCOPE_SECOND", "direct,indexed,24,1,2,0"},
        {"SHADPS4_PP_TERMINAL_SCOPE_CONSUMER", "direct,indexed,4,1,1,0"},
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
    EXPECT_TRUE(config->join_final_backing);
    EXPECT_TRUE(config->content.capture_pre_first);
    EXPECT_TRUE(config->content.capture_predecessor);
    EXPECT_TRUE(config->content.capture_sampled_input_content);
    EXPECT_TRUE(config->content.capture_upstream_inputs);
    EXPECT_TRUE(config->content.capture_upstream_pre_post);
    EXPECT_TRUE(config->content.capture_upstream_input_content);
    EXPECT_EQ(config->content.upstream_candidate_index, 0u);
    EXPECT_EQ(config->content.upstream_input_index, 1u);
    EXPECT_EQ(config->content.upstream.sampled_images, 6u);
    EXPECT_EQ(config->watch_ordinals.count, 3u);
    EXPECT_EQ(config->content.first,
              (PpTerminalScopeDrawSelector{VideoCore::ImageColorScopeDrawKind::Direct, true, 696, 1,
                                           1, 0}));
    EXPECT_EQ(config->content.second,
              (PpTerminalScopeDrawSelector{VideoCore::ImageColorScopeDrawKind::Direct, true, 24, 1,
                                           2, 0}));
    EXPECT_EQ(config->content.consumer,
              (PpTerminalScopeDrawSelector{VideoCore::ImageColorScopeDrawKind::Direct, true, 4, 1,
                                           1, 0}));
}

TEST(PpTerminalScopeContent, FinalBackingObservationReusesTheExistingPackedFootprint) {
    FinalGuestSurfaceTilePlan plan{
        .paired_backing_offset = 4096,
        .paired_backing_bytes = 128,
        .paired_backing_region_count = 2,
        .paired_backing_regions =
            {{{.logical_ordinal = 11, .buffer_offset = 0, .byte_size = 64, .width = 4, .height = 4},
              {.logical_ordinal = 12,
               .buffer_offset = 64,
               .byte_size = 64,
               .width = 4,
               .height = 4}}},
        .paired_backing_format = FinalGuestSurfaceFormat::Bgra8,
        .status = FinalGuestSurfaceStatus::Complete,
    };
    const auto observation = PlanPpTerminalScopeFinalBackingObservation(
        true, FinalGuestSurfaceStage::PpSourcePublicationReconstruction, plan, true);
    ASSERT_TRUE(observation.observe);
    EXPECT_EQ(observation.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(observation.layout.region_count, 2u);
    EXPECT_EQ(observation.layout.total_bytes, 128u);
    EXPECT_EQ(observation.layout.format, FinalGuestSurfaceFormat::Bgra8);
    EXPECT_EQ(observation.layout.regions[0], (PpTerminalScopeContentHistoryRegion{11, 0, 64}));
    EXPECT_EQ(observation.layout.regions[1], (PpTerminalScopeContentHistoryRegion{12, 64, 64}));
    EXPECT_EQ(observation.source_offset, 4096u);
    EXPECT_EQ(observation.source_bytes, 128u);
    EXPECT_FALSE(observation.gpu_copy);
    EXPECT_FALSE(observation.cpu_wait);
    EXPECT_FALSE(observation.finish);

    const auto disabled = PlanPpTerminalScopeFinalBackingObservation(
        false, FinalGuestSurfaceStage::PpSourcePublicationReconstruction, plan, true);
    EXPECT_FALSE(disabled.observe);
    EXPECT_EQ(disabled.status, FinalGuestSurfaceStatus::AlreadyConsumed);

    const auto wrong_stage = PlanPpTerminalScopeFinalBackingObservation(
        true, FinalGuestSurfaceStage::PostPp, plan, true);
    EXPECT_FALSE(wrong_stage.observe);
    EXPECT_EQ(wrong_stage.status, FinalGuestSurfaceStatus::Unsupported);
    EXPECT_EQ(wrong_stage.loss.unsupported_type, 1u);

    plan.paired_backing_regions[1].byte_size = 65;
    const auto malformed = PlanPpTerminalScopeFinalBackingObservation(
        true, FinalGuestSurfaceStage::PpSourcePublicationReconstruction, plan, true);
    EXPECT_FALSE(malformed.observe);
    EXPECT_EQ(malformed.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(malformed.loss.invalidation, 1u);
}

TEST(PpTerminalScopeContent, FinalBackingCallbacksDrainBeforeTerminalCoverageFinalizes) {
    const auto joined = PlanPpTerminalScopeFinalizeOrder(true, true);
    EXPECT_TRUE(joined.drain_draw_callbacks_before_terminal);
    EXPECT_TRUE(joined.finalize_terminal);
    EXPECT_FALSE(joined.drain_draw_callbacks_after_terminal);

    const auto legacy = PlanPpTerminalScopeFinalizeOrder(false, true);
    EXPECT_FALSE(legacy.drain_draw_callbacks_before_terminal);
    EXPECT_TRUE(legacy.finalize_terminal);
    EXPECT_TRUE(legacy.drain_draw_callbacks_after_terminal);

    const auto present_owned = PlanPpTerminalScopeFinalizeOrder(true, false);
    EXPECT_FALSE(present_owned.drain_draw_callbacks_before_terminal);
    EXPECT_TRUE(present_owned.finish_present_before_terminal);
    EXPECT_TRUE(present_owned.drain_present_callbacks_before_terminal);
    EXPECT_TRUE(present_owned.finalize_terminal);
    EXPECT_FALSE(present_owned.drain_draw_callbacks_after_terminal);

    const auto legacy_present_owned = PlanPpTerminalScopeFinalizeOrder(false, false);
    EXPECT_FALSE(legacy_present_owned.finish_present_before_terminal);
    EXPECT_FALSE(legacy_present_owned.drain_present_callbacks_before_terminal);
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

    const std::map<std::string_view, std::string_view> out_of_range_upstream{
        {"SHADPS4_FINAL_GUEST_SURFACE_CONTENT", "1"},
        {"SHADPS4_FINAL_GUEST_SURFACE_STAGE", "pp_source_publication_reconstruction"},
        {"SHADPS4_FINAL_GUEST_SURFACE_CALIBRATED_TRIPLETS", "1"},
        {"SHADPS4_FINAL_GUEST_SURFACE_EXPECTED_CALIBRATIONS", "300"},
        {"SHADPS4_FINAL_GUEST_SURFACE_FRAME_START", "4000"},
        {"SHADPS4_FINAL_GUEST_SURFACE_FRAME_COUNT", "800"},
        {"SHADPS4_FINAL_GUEST_SURFACE_WATCH_ORDINALS", "390,1024,1299"},
        {"SHADPS4_PP_TERMINAL_SCOPE_CONTENT", "1"},
        {"SHADPS4_PP_TERMINAL_SCOPE_PREDECESSOR", "direct,indexed,4,1,3,0"},
        {"SHADPS4_PP_TERMINAL_SCOPE_UPSTREAM", "direct,indexed,4,1,6,0"},
        {"SHADPS4_PP_TERMINAL_SCOPE_UPSTREAM_INPUT_INDEX", "3"},
        {"SHADPS4_PP_TERMINAL_SCOPE_FIRST", "direct,indexed,696,1,1,0"},
        {"SHADPS4_PP_TERMINAL_SCOPE_SECOND", "direct,indexed,24,1,2,0"},
        {"SHADPS4_PP_TERMINAL_SCOPE_CONSUMER", "direct,indexed,4,1,1,0"},
    };
    EXPECT_FALSE(ResolvePpTerminalScopeRuntimeConfig([&](const char* name) {
        const auto found = out_of_range_upstream.find(name);
        return found == out_of_range_upstream.end()
                   ? std::optional<std::string_view>{}
                   : std::optional<std::string_view>{found->second};
    }));

    const std::map<std::string_view, std::string_view> orphan_pre_post{
        {"SHADPS4_FINAL_GUEST_SURFACE_CONTENT", "1"},
        {"SHADPS4_FINAL_GUEST_SURFACE_STAGE", "pp_source_publication_reconstruction"},
        {"SHADPS4_FINAL_GUEST_SURFACE_CALIBRATED_TRIPLETS", "1"},
        {"SHADPS4_FINAL_GUEST_SURFACE_EXPECTED_CALIBRATIONS", "300"},
        {"SHADPS4_FINAL_GUEST_SURFACE_FRAME_START", "4000"},
        {"SHADPS4_FINAL_GUEST_SURFACE_FRAME_COUNT", "800"},
        {"SHADPS4_FINAL_GUEST_SURFACE_WATCH_ORDINALS", "390,1024,1299"},
        {"SHADPS4_PP_TERMINAL_SCOPE_CONTENT", "1"},
        {"SHADPS4_PP_TERMINAL_SCOPE_UPSTREAM_PRE_POST", "1"},
        {"SHADPS4_PP_TERMINAL_SCOPE_FIRST", "direct,indexed,696,1,1,0"},
        {"SHADPS4_PP_TERMINAL_SCOPE_SECOND", "direct,indexed,24,1,2,0"},
        {"SHADPS4_PP_TERMINAL_SCOPE_CONSUMER", "direct,indexed,4,1,1,0"},
    };
    EXPECT_FALSE(ResolvePpTerminalScopeRuntimeConfig([&](const char* name) {
        const auto found = orphan_pre_post.find(name);
        return found == orphan_pre_post.end() ? std::optional<std::string_view>{}
                                              : std::optional<std::string_view>{found->second};
    }));
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

TEST(PpTerminalScopeContent, DiscoveryArmsAndCapturesTheExactFirstDraw) {
    const auto discover = PlanPpTerminalScopeDiscoveryDecision(true, false, true, true, true, true);
    EXPECT_EQ(discover.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_TRUE(discover.allocate);
    EXPECT_TRUE(discover.arm);
    EXPECT_TRUE(discover.capture_current_draw);
    EXPECT_FALSE(discover.loss.Any());

    const auto tracked = PlanPpTerminalScopeDiscoveryDecision(true, true, true, true, true, true);
    EXPECT_FALSE(tracked.allocate);
    EXPECT_FALSE(tracked.arm);
    EXPECT_FALSE(tracked.capture_current_draw);

    const auto unrelated =
        PlanPpTerminalScopeDiscoveryDecision(true, false, false, true, true, true);
    EXPECT_FALSE(unrelated.allocate);
    EXPECT_FALSE(unrelated.arm);
    EXPECT_FALSE(unrelated.capture_current_draw);
}

TEST(PpTerminalScopeContent, DiscoveryFailsClosedForStaleMappingTargetOrCapacity) {
    const auto stale_mapping =
        PlanPpTerminalScopeDiscoveryDecision(true, false, true, false, true, true);
    EXPECT_EQ(stale_mapping.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(stale_mapping.loss.invalidation, 1u);
    EXPECT_FALSE(stale_mapping.allocate);
    EXPECT_FALSE(stale_mapping.capture_current_draw);

    const auto invalid_target =
        PlanPpTerminalScopeDiscoveryDecision(true, false, true, true, false, true);
    EXPECT_EQ(invalid_target.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(invalid_target.loss.invalidation, 1u);
    EXPECT_FALSE(invalid_target.allocate);

    const auto full = PlanPpTerminalScopeDiscoveryDecision(true, false, true, true, true, false);
    EXPECT_EQ(full.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(full.loss.tile_capacity, 1u);
    EXPECT_FALSE(full.allocate);
    EXPECT_FALSE(full.arm);
}

TEST(PpTerminalScopeContent, ExactFirstDrawCanRestartOnlyAnUnfrozenPoisonedGate) {
    const PpTerminalScopeContentConfig config{
        .enabled = true,
        .first = {VideoCore::ImageColorScopeDrawKind::Direct, true, 696, 1, 1, 0},
        .second = {VideoCore::ImageColorScopeDrawKind::Direct, true, 24, 1, 2, 0},
        .consumer = {VideoCore::ImageColorScopeDrawKind::Direct, true, 4, 1, 1, 0},
    };
    PpTerminalScopeContentGate gate{config};
    ASSERT_TRUE(gate.Arm(17, 90));
    const PpTerminalScopeDrawSelector unrelated{
        VideoCore::ImageColorScopeDrawKind::Direct, true, 3, 1, 1, 0};
    ASSERT_EQ(gate.ObserveDraw(17, 91, unrelated), PpTerminalScopeContentAction::ShapeLoss);
    EXPECT_TRUE(gate.CanRestartAtFirst(17, 91, config.first));
    EXPECT_FALSE(gate.CanRestartAtFirst(17, 91, config.second));
    EXPECT_FALSE(gate.CanRestartAtFirst(18, 91, config.first));

    ASSERT_TRUE(gate.Arm(17, 90));
    ASSERT_EQ(gate.ObserveDraw(17, 91, config.first), PpTerminalScopeContentAction::CaptureFirst);
    EXPECT_FALSE(gate.CanRestartAtFirst(17, 91, config.first));
    ASSERT_EQ(gate.ObserveDraw(17, 91, config.second), PpTerminalScopeContentAction::CaptureSecond);
    ASSERT_EQ(gate.ObserveConsumer(17, config.consumer),
              PpTerminalScopeConsumerAction::CaptureConsumer);
    EXPECT_FALSE(gate.CanRestartAtFirst(17, 91, config.first));
}

TEST(PpTerminalScopeContent, DiscoveryCoverageCountsOnlyBoundedPrivacySafeReasons) {
    PpTerminalScopeDiscoveryCoverage coverage{};
    coverage.Observe({.exact_candidate = true, .tracked = true});
    coverage.Observe({.exact_candidate = true, .mapping_valid = false});
    coverage.Observe({.exact_candidate = true, .mapping_valid = true, .target_valid = false});
    coverage.Observe(
        {.exact_candidate = true, .mapping_valid = true, .target_valid = true, .capacity = false});
    coverage.Observe({.exact_candidate = true,
                      .mapping_valid = true,
                      .target_valid = true,
                      .capacity = true,
                      .allocated = true});
    coverage.Observe({.exact_candidate = true, .tracked = true, .restarted = true});
    coverage.Observe({});

    EXPECT_EQ(coverage.candidates, 6u);
    EXPECT_EQ(coverage.tracked, 2u);
    EXPECT_EQ(coverage.allocated, 1u);
    EXPECT_EQ(coverage.restarted, 1u);
    EXPECT_EQ(coverage.mapping_rejected, 1u);
    EXPECT_EQ(coverage.target_rejected, 1u);
    EXPECT_EQ(coverage.capacity_rejected, 1u);
    const auto line = FormatPpTerminalScopeDiscoveryCoverage(coverage);
    EXPECT_EQ(line, "FGSCTSD c=6 t=2 a=1 r=1 m=1 v=1 z=1");
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("image"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
}

TEST(PpTerminalScopeContent, ProgressCoverageSeparatesActionsFromSuccessfulPlaneCaptures) {
    PpTerminalScopeProgressCoverage coverage{};
    coverage.Observe({.restart = true, .restart_plan_complete = true});
    coverage.Observe({.restart = true});
    coverage.Observe({.predecessor_before_action = true, .predecessor_before_captured = true});
    coverage.Observe({.predecessor_after_action = true, .predecessor_after_captured = true});
    coverage.Observe({.first_before_action = true, .first_before_captured = true});
    coverage.Observe({.first_after_action = true, .first_after_captured = true});
    coverage.Observe({.second_action = true, .second_captured = true});
    coverage.Observe({.consumer_action = true,
                      .consumer_captured = true,
                      .output_captured = true,
                      .alias_ready = true});
    coverage.Observe({});

    EXPECT_EQ(coverage.restarts, 2u);
    EXPECT_EQ(coverage.restart_plans_complete, 1u);
    EXPECT_EQ(coverage.predecessor_before_actions, 1u);
    EXPECT_EQ(coverage.predecessor_before_captures, 1u);
    EXPECT_EQ(coverage.predecessor_after_actions, 1u);
    EXPECT_EQ(coverage.predecessor_after_captures, 1u);
    EXPECT_EQ(coverage.first_before_actions, 1u);
    EXPECT_EQ(coverage.first_before_captures, 1u);
    EXPECT_EQ(coverage.first_after_actions, 1u);
    EXPECT_EQ(coverage.first_after_captures, 1u);
    EXPECT_EQ(coverage.second_actions, 1u);
    EXPECT_EQ(coverage.second_captures, 1u);
    EXPECT_EQ(coverage.consumer_actions, 1u);
    EXPECT_EQ(coverage.consumer_captures, 1u);
    EXPECT_EQ(coverage.output_captures, 1u);
    EXPECT_EQ(coverage.aliases_ready, 1u);

    const auto line = FormatPpTerminalScopeProgressCoverage(coverage);
    EXPECT_EQ(line, "FGSCTSP r=2 rp=1 ba=1 bc=1 pa=1 pc=1 fa=1 fc=1 aa=1 ac=1 sa=1 sc=1 "
                    "ca=1 cc=1 oc=1 ar=1");
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("image"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
}

TEST(PpTerminalScopeContent, DiscoveryCoverageEmitsAtFinalSequenceWithFinalizeFallback) {
    PpTerminalScopeDiscoveryCoverageEmissionGate final_sequence{};
    const FinalGuestSurfaceCaptureWindow window{.frame_start = 4000, .frame_count = 800};
    EXPECT_FALSE(final_sequence.Observe(window, 4798));
    EXPECT_TRUE(final_sequence.Observe(window, 4799));
    EXPECT_FALSE(final_sequence.Observe(window, 4799));
    EXPECT_FALSE(final_sequence.Finalize());

    PpTerminalScopeDiscoveryCoverageEmissionGate fallback{};
    EXPECT_TRUE(fallback.Finalize());
    EXPECT_FALSE(fallback.Finalize());
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

TEST(PpTerminalScopeContent, ClassifiesTheExactProducerPresentBeforeTheSelectedDraw) {
    VideoCore::ImageColorScopeProducerObservation scope{
        .draw_count = 2,
        .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
        .indexed = true,
        .element_count = 24,
        .instance_count = 1,
        .sampled_images = 2,
        .storage_writes = 0,
        .clear_at_begin = false,
        .valid = true,
    };
    const auto color = ClassifyPpTerminalScopePredecessor(
        {.classification = VideoCore::ImageProducerClass::ColorAttachment,
         .produced_since_last_observation = true},
        scope);
    EXPECT_EQ(color.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_FALSE(color.loss.Any());
    EXPECT_EQ(color.producer, VideoCore::ImageProducerClass::ColorAttachment);
    EXPECT_TRUE(color.fresh);
    EXPECT_EQ(color.draw_count, 2u);
    EXPECT_EQ(color.last_draw, VideoCore::ImageColorScopeDrawKind::Direct);
    EXPECT_TRUE(color.indexed);
    EXPECT_EQ(color.element_count, 24u);
    EXPECT_EQ(color.instance_count, 1u);
    EXPECT_EQ(color.sampled_images, 2u);
    EXPECT_EQ(color.storage_writes, 0u);
    EXPECT_FALSE(color.clear_at_begin);

    const auto upload = ClassifyPpTerminalScopePredecessor(
        {.classification = VideoCore::ImageProducerClass::CpuUpload,
         .produced_since_last_observation = false},
        {});
    EXPECT_EQ(upload.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_EQ(upload.producer, VideoCore::ImageProducerClass::CpuUpload);
    EXPECT_EQ(upload.draw_count, 0u) << "Non-color producers are named without stale scope data";
}

TEST(PpTerminalScopeContent, PreviousProducerClassificationFailsClosedAndStaysPrivacySafe) {
    const auto unknown = ClassifyPpTerminalScopePredecessor({}, {});
    EXPECT_EQ(unknown.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(unknown.loss.invalidation, 1u);

    VideoCore::ImageColorScopeProducerObservation invalid_scope{
        .draw_count = 1,
        .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
        .valid = false,
    };
    const auto invalid = ClassifyPpTerminalScopePredecessor(
        {.classification = VideoCore::ImageProducerClass::ColorAttachment}, invalid_scope);
    EXPECT_EQ(invalid.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(invalid.loss.invalidation, 1u);

    invalid_scope.valid = true;
    invalid_scope.overflow = true;
    const auto overflow = ClassifyPpTerminalScopePredecessor(
        {.classification = VideoCore::ImageProducerClass::ColorAttachment}, invalid_scope);
    EXPECT_EQ(overflow.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(overflow.loss.tile_capacity, 1u);

    invalid_scope.overflow = false;
    invalid_scope.draw_count = 0;
    const auto missing_draw = ClassifyPpTerminalScopePredecessor(
        {.classification = VideoCore::ImageProducerClass::ColorAttachment}, invalid_scope);
    EXPECT_EQ(missing_draw.status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(missing_draw.loss.gap, 1u);

    const auto line = FormatPpTerminalScopePredecessor(ClassifyPpTerminalScopePredecessor(
        {.classification = VideoCore::ImageProducerClass::Transfer,
         .produced_since_last_observation = true},
        {}));
    EXPECT_EQ(line, "pc=3 pf=1 pd=0 pk=0 pi=0 pe=0 pn=0 pr=0 pw=0 pb=0 pv=0 px=0 pt=0 pl=0");
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("image"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
}

TEST(PpTerminalScopeContent, PredecessorSampledInputsRetainBoundedOrderedProducerClasses) {
    const std::array<PpTerminalScopeSampledInput, 3> inputs{{
        {.producer = {.producer = VideoCore::ImageProducerClass::ColorAttachment,
                      .fresh = true,
                      .draw_count = 1,
                      .last_draw = VideoCore::ImageColorScopeDrawKind::Direct,
                      .indexed = true,
                      .element_count = 12,
                      .instance_count = 1,
                      .sampled_images = 1,
                      .scope_valid = true,
                      .status = FinalGuestSurfaceStatus::Complete}},
        {.producer = {.producer = VideoCore::ImageProducerClass::CpuUpload,
                      .fresh = true,
                      .status = FinalGuestSurfaceStatus::Complete},
         .aliases_output = true},
        {.producer = {.producer = VideoCore::ImageProducerClass::Transfer,
                      .status = FinalGuestSurfaceStatus::Complete}},
    }};
    const auto classified = ClassifyPpTerminalScopeSampledInputs(3, inputs);
    ASSERT_EQ(classified.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_FALSE(classified.loss.Any());
    EXPECT_EQ(classified.count, 3u);
    EXPECT_EQ(classified.alias_count, 1u);
    EXPECT_EQ(classified.inputs[0].producer.element_count, 12u);
    EXPECT_EQ(classified.inputs[1].producer.producer, VideoCore::ImageProducerClass::CpuUpload);
    EXPECT_EQ(classified.inputs[2].producer.producer, VideoCore::ImageProducerClass::Transfer);

    const auto line = FormatPpTerminalScopeSampledInputs(classified);
    EXPECT_NE(line.find("ic=3 ia=1 is=0 il=0"), std::string::npos);
    EXPECT_NE(line.find(" i0=1/1/1/1/1/12/1/1/0/0/1/0/0/0/0"), std::string::npos);
    EXPECT_NE(line.find(" i1=4/1/0/0/0/0/0/0/0/0/0/0/0/0/1"), std::string::npos);
    EXPECT_NE(line.find(" i2=3/0/0/0/0/0/0/0/0/0/0/0/0/0/0"), std::string::npos);
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("image"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
}

TEST(PpTerminalScopeContent, PredecessorSampledInputsFailClosedOnCountCapacityOrInvalidState) {
    const std::array<PpTerminalScopeSampledInput, 2> two{{
        {.producer = {.producer = VideoCore::ImageProducerClass::CpuUpload,
                      .status = FinalGuestSurfaceStatus::Complete}},
        {.producer = {.producer = VideoCore::ImageProducerClass::Transfer,
                      .status = FinalGuestSurfaceStatus::Complete}},
    }};
    const auto missing = ClassifyPpTerminalScopeSampledInputs(3, two);
    EXPECT_EQ(missing.status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(missing.loss.gap, 1u);

    std::array<PpTerminalScopeSampledInput, PpTerminalScopeSampledInputs::MaxInputs + 1> many{};
    const auto capacity = ClassifyPpTerminalScopeSampledInputs(many.size(), many);
    EXPECT_EQ(capacity.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(capacity.loss.tile_capacity, 1u);

    auto invalid = two;
    invalid[1].producer.status = FinalGuestSurfaceStatus::InvalidationLoss;
    invalid[1].producer.loss.invalidation = 1;
    const auto rejected = ClassifyPpTerminalScopeSampledInputs(2, invalid);
    EXPECT_EQ(rejected.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(rejected.loss.invalidation, 1u);
    EXPECT_EQ(rejected.count, 2u);
}

TEST(PpTerminalScopeContent, UpstreamInputRegistryJoinsOnlyTheExactPrivateProducerOutput) {
    const PpTerminalScopeDrawSelector upstream{
        .kind = VideoCore::ImageColorScopeDrawKind::Direct,
        .indexed = true,
        .element_count = 4,
        .instance_count = 1,
        .sampled_images = 6,
    };
    PpTerminalScopeUpstreamInputRegistry registry{upstream};
    PpTerminalScopeSampledInputs inputs{
        .count = 6,
        .status = FinalGuestSurfaceStatus::Complete,
    };
    for (u32 index = 0; index < inputs.count; ++index) {
        inputs.inputs[index].producer.status = FinalGuestSurfaceStatus::Complete;
        inputs.inputs[index].view.status = FinalGuestSurfaceStatus::Complete;
    }
    const VideoCore::ImageColorScopePrivateLink output{VideoCore::ImageId{81}, 8100};
    EXPECT_TRUE(registry.Observe(output, upstream, inputs).stored);
    const auto joined = registry.Resolve(output);
    ASSERT_TRUE(joined.matched);
    EXPECT_EQ(joined.inputs.count, 6u);
    EXPECT_EQ(joined.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_FALSE(joined.loss.Any());

    auto wrong_shape = upstream;
    wrong_shape.sampled_images = 5;
    EXPECT_FALSE(registry.Observe({VideoCore::ImageId{82}, 8200}, wrong_shape, inputs).stored);
    EXPECT_FALSE(registry.Resolve({VideoCore::ImageId{81}, 8101}).matched)
        << "A reused image slot must not inherit the prior producer inputs";
    registry.Reset();
    EXPECT_FALSE(registry.Resolve(output).matched);
}

TEST(PpTerminalScopeContent, FeedbackLoopClassificationRequiresExactVulkanState) {
    const auto extension = ClassifyPpTerminalScopeFeedbackLoop({
        .logical_alias = true,
        .exact_subresource = true,
        .extension_supported = true,
        .sampled_feedback_layout = true,
        .attachment_feedback_layout = true,
        .dynamic_feedback_enabled = true,
    });
    EXPECT_EQ(extension.mode, PpTerminalScopeFeedbackMode::AttachmentFeedbackLoop);
    EXPECT_TRUE(extension.exact_subresource);
    EXPECT_EQ(extension.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_FALSE(extension.loss.Any());

    const auto fallback = ClassifyPpTerminalScopeFeedbackLoop({
        .logical_alias = true,
        .exact_subresource = true,
        .sampled_general_layout = true,
        .attachment_general_layout = true,
    });
    EXPECT_EQ(fallback.mode, PpTerminalScopeFeedbackMode::GeneralFallback);
    EXPECT_EQ(fallback.status, FinalGuestSurfaceStatus::Complete);

    auto mismatched_range = PpTerminalScopeFeedbackLoopDescriptor{
        .logical_alias = true,
        .extension_supported = true,
        .sampled_feedback_layout = true,
        .attachment_feedback_layout = true,
        .dynamic_feedback_enabled = true,
    };
    const auto range_loss = ClassifyPpTerminalScopeFeedbackLoop(mismatched_range);
    EXPECT_EQ(range_loss.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(range_loss.loss.invalidation, 1u);

    mismatched_range.exact_subresource = true;
    mismatched_range.dynamic_feedback_enabled = false;
    const auto dynamic_loss = ClassifyPpTerminalScopeFeedbackLoop(mismatched_range);
    EXPECT_EQ(dynamic_loss.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(dynamic_loss.loss.invalidation, 1u);

    const auto no_alias = ClassifyPpTerminalScopeFeedbackLoop({});
    EXPECT_EQ(no_alias.mode, PpTerminalScopeFeedbackMode::None);
    EXPECT_EQ(no_alias.status, FinalGuestSurfaceStatus::AlreadyConsumed);

    const PpTerminalScopeSampledInputs logged{
        .inputs = {{{.feedback = extension, .aliases_output = true}}},
        .count = 1,
        .alias_count = 1,
        .status = FinalGuestSurfaceStatus::Complete,
    };
    const auto line = FormatPpTerminalScopeSampledInputs(logged, 'u');
    EXPECT_NE(line.find(" f0=1/1/0/0"), std::string::npos);
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("handle"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
}

TEST(PpTerminalScopeContent, SampledInputViewPreservesExactBoundCopyEligibility) {
    const auto view = ClassifyPpTerminalScopeSampledInputView({
        .width = 1920,
        .height = 1080,
        .format = FinalGuestSurfaceFormat::Bgra8,
        .samples = 1,
        .base_mip = 0,
        .mip_count = 1,
        .base_layer = 0,
        .layer_count = 1,
        .color = true,
        .uniform_state = true,
    });
    EXPECT_EQ(view.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_FALSE(view.loss.Any());
    EXPECT_TRUE(view.copy_eligible);
    EXPECT_EQ(view.width, 1920u);
    EXPECT_EQ(view.height, 1080u);
    EXPECT_EQ(view.format, FinalGuestSurfaceFormat::Bgra8);
    EXPECT_EQ(view.base_mip, 0u);
    EXPECT_EQ(view.base_layer, 0u);

    PpTerminalScopeSampledInput input{
        .producer = {.producer = VideoCore::ImageProducerClass::Transfer,
                     .fresh = true,
                     .status = FinalGuestSurfaceStatus::Complete},
        .view = view,
    };
    const auto classified = ClassifyPpTerminalScopeSampledInputs(
        1, std::span<const PpTerminalScopeSampledInput>{&input, 1});
    ASSERT_EQ(classified.status, FinalGuestSurfaceStatus::Complete);
    const auto line = FormatPpTerminalScopeSampledInputs(classified);
    EXPECT_NE(line.find(" v0=1920/1080/2/1/0/1/0/1/1/1/0/0/0"), std::string::npos);
    EXPECT_EQ(line.find("uid"), std::string::npos);
    EXPECT_EQ(line.find("handle"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
}

TEST(PpTerminalScopeContent, SampledInputViewFailsClosedForEveryUnsafeCopyClass) {
    PpTerminalScopeSampledInputViewDescriptor descriptor{
        .width = 1280,
        .height = 720,
        .format = FinalGuestSurfaceFormat::Rgba8,
        .samples = 1,
        .base_mip = 0,
        .mip_count = 1,
        .base_layer = 0,
        .layer_count = 1,
        .color = true,
        .uniform_state = true,
    };

    auto invalid = descriptor;
    invalid.width = 0;
    EXPECT_EQ(ClassifyPpTerminalScopeSampledInputView(invalid).loss.invalid_extent, 1u);

    auto format = descriptor;
    format.format = FinalGuestSurfaceFormat::Unsupported;
    EXPECT_EQ(ClassifyPpTerminalScopeSampledInputView(format).loss.unsupported_format, 1u);

    auto samples = descriptor;
    samples.samples = 4;
    EXPECT_EQ(ClassifyPpTerminalScopeSampledInputView(samples).loss.unsupported_samples, 1u);

    auto mip = descriptor;
    mip.mip_count = 2;
    EXPECT_EQ(ClassifyPpTerminalScopeSampledInputView(mip).loss.unsupported_mip, 1u);

    auto layer = descriptor;
    layer.base_layer = 2;
    EXPECT_EQ(ClassifyPpTerminalScopeSampledInputView(layer).loss.unsupported_layer, 1u);

    auto aspect = descriptor;
    aspect.color = false;
    EXPECT_EQ(ClassifyPpTerminalScopeSampledInputView(aspect).loss.unsupported_aspect, 1u);

    auto mixed = descriptor;
    mixed.uniform_state = false;
    EXPECT_EQ(ClassifyPpTerminalScopeSampledInputView(mixed).loss.invalidation, 1u);

    auto conflict = descriptor;
    conflict.view_conflict = true;
    const auto rejected = ClassifyPpTerminalScopeSampledInputView(conflict);
    EXPECT_EQ(rejected.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(rejected.loss.invalidation, 1u);
    EXPECT_FALSE(rejected.copy_eligible);
}

TEST(PpTerminalScopeContent, SampledInputContentPlansEveryEligibleNormalizedRegionTransactionally) {
    FinalGuestSurfaceWatchOrdinals selector{};
    selector.ordinals[0] = 2031;
    selector.count = 1;
    const std::array<PpTerminalScopeSampledInputView, 3> views{{
        ClassifyPpTerminalScopeSampledInputView({
            .width = 480,
            .height = 270,
            .format = FinalGuestSurfaceFormat::Unsupported,
            .samples = 1,
            .mip_count = 1,
            .layer_count = 1,
            .color = true,
            .uniform_state = true,
        }),
        ClassifyPpTerminalScopeSampledInputView({
            .width = 1920,
            .height = 1080,
            .format = FinalGuestSurfaceFormat::Rgba8,
            .samples = 1,
            .mip_count = 1,
            .layer_count = 1,
            .color = true,
            .uniform_state = true,
        }),
        ClassifyPpTerminalScopeSampledInputView({
            .width = 960,
            .height = 540,
            .format = FinalGuestSurfaceFormat::Rgba8,
            .samples = 1,
            .mip_count = 1,
            .layer_count = 1,
            .color = true,
            .uniform_state = true,
        }),
    }};
    const auto plan = PlanPpTerminalScopeSampledInputContent({
        .enabled = true,
        .logical_width = 1280,
        .logical_height = 720,
        .base_offset = 0x1000,
        .selector = selector,
        .views = views,
        .buffer_alignment = 16,
        .max_regions = 8,
        .max_bytes = 1u << 20,
    });
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_FALSE(plan.loss.Any());
    EXPECT_TRUE(plan.copy);
    EXPECT_EQ(plan.input_count, 3u);
    EXPECT_EQ(plan.capture_mask, 0b110u);
    EXPECT_EQ(plan.unavailable_mask, 0b001u);
    EXPECT_EQ(plan.copy_region_count, 2u);
    EXPECT_EQ(plan.inputs[0].loss.unsupported_format, 1u);

    ASSERT_EQ(plan.inputs[1].status, FinalGuestSurfaceStatus::Complete);
    ASSERT_EQ(plan.inputs[1].region_count, 1u);
    EXPECT_EQ(plan.inputs[1].regions[0].logical_ordinal, 2031u);
    EXPECT_EQ(plan.inputs[1].regions[0].x, 1320u);
    EXPECT_EQ(plan.inputs[1].regions[0].y, 600u);
    EXPECT_EQ(plan.inputs[1].regions[0].width, 48u);
    EXPECT_EQ(plan.inputs[1].regions[0].height, 48u);
    EXPECT_EQ(plan.inputs[1].plane_bytes, 48u * 48u * 4u);
    EXPECT_EQ(plan.inputs[1].plane_offset, 0x1000u);

    ASSERT_EQ(plan.inputs[2].status, FinalGuestSurfaceStatus::Complete);
    ASSERT_EQ(plan.inputs[2].region_count, 1u);
    EXPECT_EQ(plan.inputs[2].regions[0].x, 660u);
    EXPECT_EQ(plan.inputs[2].regions[0].y, 300u);
    EXPECT_EQ(plan.inputs[2].regions[0].width, 24u);
    EXPECT_EQ(plan.inputs[2].regions[0].height, 24u);
    EXPECT_EQ(plan.inputs[2].plane_bytes, 24u * 24u * 4u);
    EXPECT_EQ(plan.inputs[2].plane_offset, 0x3400u);
    EXPECT_EQ(plan.total_bytes, 0x3D00u);
}

TEST(PpTerminalScopeContent, SampledInputContentCapacityRejectsTheWholeCopyPlan) {
    FinalGuestSurfaceWatchOrdinals selector{};
    selector.ordinals[0] = 1;
    selector.count = 1;
    const std::array views{
        ClassifyPpTerminalScopeSampledInputView({
            .width = 1920,
            .height = 1080,
            .format = FinalGuestSurfaceFormat::Rgba8,
            .samples = 1,
            .mip_count = 1,
            .layer_count = 1,
            .color = true,
            .uniform_state = true,
        }),
    };
    const auto plan = PlanPpTerminalScopeSampledInputContent({
        .enabled = true,
        .logical_width = 1280,
        .logical_height = 720,
        .base_offset = 4096,
        .selector = selector,
        .views = views,
        .buffer_alignment = 16,
        .max_regions = 0,
        .max_bytes = 4096,
    });
    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_TRUE(plan.loss.Any());
    EXPECT_FALSE(plan.copy);
    EXPECT_EQ(plan.capture_mask, 0u);
    EXPECT_EQ(plan.copy_region_count, 0u);
    EXPECT_EQ(plan.total_bytes, 0u);
}

TEST(PpTerminalScopeContent, SampledInputCopyRequiresEveryPlannedSourceToRemainExact) {
    const auto accepted = PlanPpTerminalScopeSampledInputCopyDecision({
        .enabled = true,
        .input_count = 3,
        .capture_mask = 0b110,
        .pointer_mask = 0b110,
        .backing_mask = 0b110,
        .view_match_mask = 0b110,
    });
    EXPECT_TRUE(accepted.copy);
    EXPECT_EQ(accepted.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_FALSE(accepted.loss.Any());

    for (const u32 missing_mask : {0b100u, 0b010u, 0u}) {
        const auto rejected = PlanPpTerminalScopeSampledInputCopyDecision({
            .enabled = true,
            .input_count = 3,
            .capture_mask = 0b110,
            .pointer_mask = missing_mask,
            .backing_mask = 0b110,
            .view_match_mask = 0b110,
        });
        EXPECT_FALSE(rejected.copy);
        EXPECT_EQ(rejected.status, FinalGuestSurfaceStatus::InvalidationLoss);
        EXPECT_EQ(rejected.loss.invalidation, 1u);
    }
    const auto alias_rejected = PlanPpTerminalScopeSampledInputCopyDecision({
        .enabled = true,
        .input_count = 3,
        .capture_mask = 0b110,
        .pointer_mask = 0b110,
        .backing_mask = 0b110,
        .view_match_mask = 0b110,
        .alias_mask = 0b010,
    });
    EXPECT_FALSE(alias_rejected.copy);
    EXPECT_EQ(alias_rejected.loss.invalidation, 1u);
    EXPECT_FALSE(PlanPpTerminalScopeSampledInputCopyDecision({}).copy);
}

TEST(PpTerminalScopeContent, NamedPredecessorIsCapturedBeforeAndAfterWithoutReplacingTheLineage) {
    const PpTerminalScopeContentConfig config{
        .enabled = true,
        .capture_pre_first = true,
        .capture_predecessor = true,
        .predecessor = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                        .indexed = true,
                        .element_count = 4,
                        .instance_count = 1,
                        .sampled_images = 3},
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
        .consumer = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                     .indexed = true,
                     .element_count = 4,
                     .instance_count = 1,
                     .sampled_images = 1},
    };
    PpTerminalScopeContentGate gate{config};
    ASSERT_TRUE(gate.Arm(17, 9));
    EXPECT_EQ(gate.PreviewDraw(17, 90, config.predecessor),
              PpTerminalScopePreDrawAction::CaptureBeforePredecessor);
    EXPECT_EQ(gate.ObserveDraw(17, 90, config.predecessor),
              PpTerminalScopeContentAction::CapturePredecessor);
    EXPECT_EQ(gate.PreviewDraw(17, 90, config.first),
              PpTerminalScopePreDrawAction::CaptureBeforeFirst);
    EXPECT_EQ(gate.ObserveDraw(17, 90, config.first), PpTerminalScopeContentAction::CaptureFirst);
    EXPECT_EQ(gate.ObserveDraw(17, 90, config.second), PpTerminalScopeContentAction::CaptureSecond);
    EXPECT_EQ(gate.ObserveConsumer(17, config.consumer),
              PpTerminalScopeConsumerAction::CaptureConsumer);
    const auto take = gate.Take(17, 9);
    EXPECT_EQ(take.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_TRUE(take.predecessor_captured);
    EXPECT_EQ(take.draw_count, 2u) << "Existing first/second lineage semantics stay intact";

    ASSERT_TRUE(gate.Arm(17, 10));
    EXPECT_EQ(gate.PreviewDraw(17, 91, config.first), PpTerminalScopePreDrawAction::ShapeLoss)
        << "The selected draw cannot silently substitute for its missing predecessor";
}

TEST(PpTerminalScopeContent, NamedPredecessorAddsTwoBoundedPlanesAndExactVisualPredicates) {
    FinalGuestSurfaceWatchOrdinals selector{};
    selector.status = FinalGuestSurfaceStatus::Complete;
    selector.count = 1;
    selector.ordinals[0] = 1024;
    const auto plan = PlanPpTerminalScopeContent({
        .enabled = true,
        .armed = true,
        .capture_predecessor = true,
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
    EXPECT_EQ(plan.copy_region_count, 7u);
    EXPECT_GE(plan.predecessor_pre_plane_offset, plan.pre_first_plane_offset + plan.plane_bytes);
    EXPECT_GE(plan.predecessor_post_plane_offset,
              plan.predecessor_pre_plane_offset + plan.plane_bytes);
    EXPECT_GE(plan.total_bytes, plan.predecessor_post_plane_offset + plan.plane_bytes);
    EXPECT_LE(plan.total_bytes, PpTerminalScopeSnapshotBytes);
    EXPECT_TRUE(PlanPpTerminalScopePlaneSlot(5, false).acquire);
    EXPECT_TRUE(PlanPpTerminalScopePlaneSlot(6, true).reuse);

    constexpr u32 RegionBytes = 4;
    const PpTerminalScopeContentHistoryLayout layout{
        .region_count = 1,
        .plane_bytes = RegionBytes,
        .second_plane_offset = RegionBytes,
        .consumer_plane_offset = RegionBytes * 2,
        .output_plane_offset = RegionBytes * 3,
        .pre_first_plane_offset = RegionBytes * 4,
        .predecessor_pre_plane_offset = RegionBytes * 5,
        .predecessor_post_plane_offset = RegionBytes * 6,
        .total_bytes = RegionBytes * 7,
        .plane_mask = 0x7f,
        .regions = {{{.logical_ordinal = 1024, .buffer_offset = 0, .byte_size = RegionBytes}}},
        .format = FinalGuestSurfaceFormat::Rgba8,
    };
    const auto observation = [](std::array<u8, 4> predecessor_pre,
                                std::array<u8, 4> predecessor_post) {
        std::array<std::byte, RegionBytes * 7> bytes{};
        for (u32 index = 0; index < 4; ++index) {
            bytes[RegionBytes * 5 + index] = std::byte{predecessor_pre[index]};
            bytes[RegionBytes * 6 + index] = std::byte{predecessor_post[index]};
        }
        return bytes;
    };
    PpTerminalScopeContentReducer reducer{{.frame_start = 100, .frame_count = 3}, 8};
    reducer.ObserveContent(100, layout, observation({1, 2, 3, 255}, {4, 5, 6, 255}),
                           FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(101, layout, observation({32, 33, 34, 255}, {35, 36, 37, 255}),
                           FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(102, layout, observation({1, 2, 3, 0}, {4, 5, 6, 0}),
                           FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveCalibration({.request_ordinal = 1, .sequence = 100, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 2, .sequence = 101, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 3, .sequence = 102, .valid = true});
    const auto reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports[0].predecessor_pre_localized_visual_return_ordinals,
              (std::vector<u32>{1024}));
    EXPECT_EQ(reports[0].predecessor_post_localized_visual_return_ordinals,
              (std::vector<u32>{1024}));
    const auto line = FormatPpTerminalScopeCalibratedReport(reports[0]);
    EXPECT_NE(line.find(" yp=1024"), std::string::npos);
    EXPECT_NE(line.find(" yo=1024"), std::string::npos);
    EXPECT_EQ(line.find("address"), std::string::npos);
}

TEST(PpTerminalScopeContent, PredecessorDrawCannotEraseItsFailedPreCapture) {
    const FinalGuestSurfaceLoss busy_loss{.busy = 1};
    const auto result =
        ApplyPpTerminalScopeContentAction(FinalGuestSurfaceStatus::BusyLoss, busy_loss,
                                          PpTerminalScopeContentAction::CapturePredecessor);
    EXPECT_EQ(result.status, FinalGuestSurfaceStatus::BusyLoss);
    EXPECT_EQ(result.loss.busy, 1u);
}

TEST(PpTerminalScopeContent, PredecessorMayBeTheImmediatelyPreviousColorScopeOnly) {
    const PpTerminalScopeContentConfig config{
        .enabled = true,
        .capture_pre_first = true,
        .capture_predecessor = true,
        .predecessor = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                        .indexed = true,
                        .element_count = 4,
                        .instance_count = 1,
                        .sampled_images = 3},
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
        .consumer = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                     .indexed = true,
                     .element_count = 4,
                     .instance_count = 1,
                     .sampled_images = 1},
    };
    PpTerminalScopeContentGate gate{config};
    ASSERT_TRUE(gate.Arm(17, 9));
    EXPECT_EQ(gate.PreviewDraw(17, 90, config.predecessor),
              PpTerminalScopePreDrawAction::CaptureBeforePredecessor);
    EXPECT_EQ(gate.ObserveDraw(17, 90, config.predecessor),
              PpTerminalScopeContentAction::CapturePredecessor);
    EXPECT_EQ(gate.PreviewDraw(17, 91, config.first),
              PpTerminalScopePreDrawAction::CaptureBeforeFirst)
        << "The named predecessor is the sole draw of the immediately ended scope";
    EXPECT_EQ(gate.ObserveDraw(17, 91, config.first), PpTerminalScopeContentAction::CaptureFirst);
    EXPECT_EQ(gate.ObserveDraw(17, 91, config.second), PpTerminalScopeContentAction::CaptureSecond);

    ASSERT_TRUE(gate.Arm(17, 10));
    EXPECT_EQ(gate.PreviewDraw(17, 100, config.predecessor),
              PpTerminalScopePreDrawAction::CaptureBeforePredecessor);
    EXPECT_EQ(gate.ObserveDraw(17, 100, config.predecessor),
              PpTerminalScopeContentAction::CapturePredecessor);
    auto unrelated = config.first;
    unrelated.element_count = 12;
    EXPECT_EQ(gate.PreviewDraw(17, 101, unrelated), PpTerminalScopePreDrawAction::ShapeLoss)
        << "The predecessor must not bridge an unrelated next color scope";
}

TEST(PpTerminalScopeContent, FailedPredecessorCandidateCanRestartInALaterScope) {
    const PpTerminalScopeContentConfig config{
        .enabled = true,
        .capture_pre_first = true,
        .capture_predecessor = true,
        .predecessor = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                        .indexed = true,
                        .element_count = 4,
                        .instance_count = 1,
                        .sampled_images = 3},
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
        .consumer = {.kind = VideoCore::ImageColorScopeDrawKind::Direct,
                     .indexed = true,
                     .element_count = 4,
                     .instance_count = 1,
                     .sampled_images = 1},
    };
    PpTerminalScopeContentGate gate{config};
    ASSERT_TRUE(gate.Arm(17, 9));
    ASSERT_EQ(gate.PreviewDraw(17, 80, config.predecessor),
              PpTerminalScopePreDrawAction::CaptureBeforePredecessor);
    ASSERT_EQ(gate.ObserveDraw(17, 80, config.predecessor),
              PpTerminalScopeContentAction::CapturePredecessor);
    auto unrelated = config.first;
    unrelated.element_count = 12;
    ASSERT_EQ(gate.PreviewDraw(17, 81, unrelated), PpTerminalScopePreDrawAction::ShapeLoss);

    EXPECT_EQ(gate.ObserveConsumer(17, config.consumer), PpTerminalScopeConsumerAction::None)
        << "A poisoned discovered predecessor must not freeze on a later consumer-shaped draw";
    EXPECT_TRUE(gate.CanRestartAtFirst(17, 89, config.predecessor));
    ASSERT_TRUE(gate.Arm(17, 9));
    EXPECT_EQ(gate.PreviewDraw(17, 89, config.predecessor),
              PpTerminalScopePreDrawAction::CaptureBeforePredecessor);
    EXPECT_EQ(gate.ObserveDraw(17, 89, config.predecessor),
              PpTerminalScopeContentAction::CapturePredecessor);
    EXPECT_EQ(gate.PreviewDraw(17, 90, config.first),
              PpTerminalScopePreDrawAction::CaptureBeforeFirst);
    EXPECT_EQ(gate.ObserveDraw(17, 90, config.first), PpTerminalScopeContentAction::CaptureFirst);
    EXPECT_EQ(gate.ObserveDraw(17, 90, config.second), PpTerminalScopeContentAction::CaptureSecond);
    EXPECT_EQ(gate.ObserveConsumer(17, config.consumer),
              PpTerminalScopeConsumerAction::CaptureConsumer);

    const auto rolling_slot = PlanPpTerminalScopePlaneSlot(5, true);
    EXPECT_EQ(rolling_slot.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_TRUE(rolling_slot.reuse);
    EXPECT_TRUE(rolling_slot.requires_write_barrier)
        << "Overwriting an earlier candidate must order its pending transfer writes";
}

TEST(PpUpstreamFeedbackPrePost, PlansFourBoundedExactPrePostCandidatesInOneSlot) {
    FinalGuestSurfaceWatchOrdinals selector{};
    selector.count = 13;
    for (u32 index = 0; index < selector.count; ++index) {
        selector.ordinals[index] = index + 1;
    }

    const auto plan = PlanPpUpstreamFeedbackPrePost({
        .enabled = true,
        .logical_width = 1280,
        .logical_height = 720,
        .source_width = 1920,
        .source_height = 1080,
        .format = FinalGuestSurfaceFormat::Bgra8,
        .samples = 1,
        .base_mip = 0,
        .mip_count = 1,
        .base_layer = 0,
        .layer_count = 1,
        .color = true,
        .type_2d = true,
        .uniform_state = true,
        .selector = selector,
        .candidate_capacity = 4,
        .buffer_alignment = 16,
        .max_regions = 4 * 2 * 13,
        .max_bytes = PpTerminalScopeSnapshotBytes,
    });
    ASSERT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    ASSERT_FALSE(plan.loss.Any());
    ASSERT_TRUE(plan.copy);
    ASSERT_EQ(plan.region_count, 13u);
    ASSERT_EQ(plan.candidate_capacity, 4u);
    ASSERT_EQ(plan.regions[0].width, 48u);
    ASSERT_EQ(plan.regions[0].height, 48u);
    ASSERT_EQ(plan.regions[0].byte_size, 48u * 48u * 4u);
    EXPECT_LT(plan.total_bytes, PpTerminalScopeSnapshotBytes);

    const auto first_pre = ResolvePpUpstreamFeedbackPrePostCopy(plan, 0, false, 0);
    const auto first_post = ResolvePpUpstreamFeedbackPrePostCopy(plan, 0, true, 0);
    const auto last_post = ResolvePpUpstreamFeedbackPrePostCopy(plan, 3, true, 12);
    ASSERT_TRUE(first_pre.copy);
    ASSERT_TRUE(first_post.copy);
    ASSERT_TRUE(last_post.copy);
    EXPECT_LT(first_pre.buffer_offset, first_post.buffer_offset);
    EXPECT_LT(last_post.buffer_offset + last_post.byte_size, PpTerminalScopeSnapshotBytes + 1u);

    const auto overflow = PlanPpUpstreamFeedbackPrePost({
        .enabled = true,
        .logical_width = 1280,
        .logical_height = 720,
        .source_width = 1920,
        .source_height = 1080,
        .format = FinalGuestSurfaceFormat::Bgra8,
        .samples = 1,
        .mip_count = 1,
        .layer_count = 1,
        .color = true,
        .type_2d = true,
        .uniform_state = true,
        .selector = selector,
        .candidate_capacity = 5,
        .buffer_alignment = 16,
        .max_regions = 5 * 2 * 13,
        .max_bytes = PpTerminalScopeSnapshotBytes,
    });
    EXPECT_EQ(overflow.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(overflow.loss.byte_capacity, 1u);
    EXPECT_FALSE(overflow.copy);
}

TEST(PpUpstreamFeedbackPrePost, ResolvesOnlyOneCompleteExactPrivateCandidate) {
    const PpTerminalScopeDrawSelector selector{
        .kind = VideoCore::ImageColorScopeDrawKind::Direct,
        .indexed = true,
        .element_count = 4,
        .instance_count = 1,
        .sampled_images = 6,
    };
    PpUpstreamFeedbackPrePostRegistry registry{selector, 4};
    const VideoCore::ImageColorScopePrivateLink first{VideoCore::ImageId{81}, 8100};
    const VideoCore::ImageColorScopePrivateLink second{VideoCore::ImageId{82}, 8200};

    const auto first_pre = registry.Preview(first, selector);
    ASSERT_TRUE(first_pre.capture);
    ASSERT_EQ(first_pre.candidate_index, 0u);
    const auto first_post = registry.Complete(first, selector);
    ASSERT_TRUE(first_post.capture);
    ASSERT_EQ(first_post.candidate_index, 0u);

    ASSERT_TRUE(registry.Preview(second, selector).capture);
    ASSERT_TRUE(registry.Complete(second, selector).capture);
    const auto resolved = registry.Resolve(second);
    ASSERT_TRUE(resolved.matched);
    EXPECT_EQ(resolved.candidate_index, 1u);
    EXPECT_EQ(resolved.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_FALSE(resolved.loss.Any());

    EXPECT_EQ(registry.Complete({VideoCore::ImageId{83}, 8300}, selector).status,
              FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(registry.Resolve({VideoCore::ImageId{82}, 8201}).status,
              FinalGuestSurfaceStatus::AlreadyConsumed)
        << "A reused image slot must not inherit an earlier private capture";
    registry.Reset();
    EXPECT_FALSE(registry.Resolve(second).matched);
}

TEST(PpUpstreamFeedbackPrePost, FramePublishesOnlyTheSelectedCompletePair) {
    FinalGuestSurfaceWatchOrdinals selector{};
    selector.count = 1;
    selector.ordinals[0] = 1024;
    const auto plan = PlanPpUpstreamFeedbackPrePost({
        .enabled = true,
        .logical_width = 1280,
        .logical_height = 720,
        .source_width = 1920,
        .source_height = 1080,
        .format = FinalGuestSurfaceFormat::Bgra8,
        .samples = 1,
        .mip_count = 1,
        .layer_count = 1,
        .color = true,
        .type_2d = true,
        .uniform_state = true,
        .selector = selector,
        .candidate_capacity = 4,
        .buffer_alignment = 16,
        .max_regions = 8,
        .max_bytes = PpTerminalScopeSnapshotBytes,
    });
    ASSERT_TRUE(plan.copy);
    const PpTerminalScopeDrawSelector draw{
        .kind = VideoCore::ImageColorScopeDrawKind::Direct,
        .indexed = true,
        .element_count = 4,
        .instance_count = 1,
        .sampled_images = 6,
    };
    const VideoCore::ImageColorScopePrivateLink selected{VideoCore::ImageId{91}, 9100};
    PpUpstreamFeedbackPrePostFrame frame{draw, 4};
    ASSERT_TRUE(frame.Arm());
    ASSERT_TRUE(frame.Configure(plan));
    const auto before = frame.Preview(selected, draw);
    ASSERT_TRUE(before.capture);
    frame.MarkCopied(before.candidate_index, false);
    const auto after = frame.Complete(selected, draw);
    ASSERT_TRUE(after.capture);
    frame.MarkCopied(after.candidate_index, true);
    ASSERT_TRUE(frame.Select(selected));

    const auto take = frame.Take(true);
    EXPECT_TRUE(take.copy);
    EXPECT_EQ(take.candidate_index, 0u);
    EXPECT_EQ(take.recorded_plane_mask, 0x3u);
    EXPECT_EQ(take.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_FALSE(take.loss.Any());
    EXPECT_FALSE(take.cpu_wait);
    EXPECT_FALSE(take.finish);
    EXPECT_FALSE(take.retains_image);
    EXPECT_FALSE(take.retains_vk_image);

    ASSERT_TRUE(frame.Arm());
    ASSERT_TRUE(frame.Configure(plan));
    const auto incomplete = frame.Preview(selected, draw);
    frame.MarkCopied(incomplete.candidate_index, false);
    ASSERT_TRUE(frame.Select(selected) == false)
        << "Selection must fail closed until the matching post-draw observation exists";
    const auto failed = frame.Take(true);
    EXPECT_FALSE(failed.copy);
    EXPECT_EQ(failed.status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(failed.loss.gap, 1u);
}

TEST(PpUpstreamFeedbackPrePost, CompactsOnlyTheSelectedCandidatePlanes) {
    FinalGuestSurfaceWatchOrdinals selector{};
    selector.count = 1;
    selector.ordinals[0] = 1024;
    const auto plan = PlanPpUpstreamFeedbackPrePost({
        .enabled = true,
        .logical_width = 1280,
        .logical_height = 720,
        .source_width = 1920,
        .source_height = 1080,
        .format = FinalGuestSurfaceFormat::Bgra8,
        .samples = 1,
        .mip_count = 1,
        .layer_count = 1,
        .color = true,
        .type_2d = true,
        .uniform_state = true,
        .selector = selector,
        .candidate_capacity = 4,
        .buffer_alignment = 16,
        .max_regions = 8,
        .max_bytes = PpTerminalScopeSnapshotBytes,
    });
    ASSERT_TRUE(plan.copy);
    std::vector<std::byte> slot(plan.total_bytes);
    const auto pre = ResolvePpUpstreamFeedbackPrePostCopy(plan, 2, false, 0);
    const auto post = ResolvePpUpstreamFeedbackPrePostCopy(plan, 2, true, 0);
    ASSERT_TRUE(pre.copy);
    ASSERT_TRUE(post.copy);
    std::ranges::fill(std::span{slot}.subspan(pre.buffer_offset, pre.byte_size), std::byte{0x11});
    std::ranges::fill(std::span{slot}.subspan(post.buffer_offset, post.byte_size), std::byte{0x22});

    const auto compact = CompactPpUpstreamFeedbackPrePost(plan, 2, slot);
    ASSERT_EQ(compact.status, FinalGuestSurfaceStatus::Complete);
    ASSERT_FALSE(compact.loss.Any());
    ASSERT_EQ(compact.bytes.size(), plan.plane_bytes * 2u);
    EXPECT_EQ(compact.bytes[plan.regions[0].buffer_offset], std::byte{0x11});
    EXPECT_EQ(compact.bytes[plan.plane_bytes + plan.regions[0].buffer_offset], std::byte{0x22});

    const auto short_slot = CompactPpUpstreamFeedbackPrePost(
        plan, 2, std::span<const std::byte>{slot}.first(post.buffer_offset + post.byte_size - 1));
    EXPECT_EQ(short_slot.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(short_slot.loss.invalidation, 1u);
    EXPECT_TRUE(short_slot.bytes.empty());
}

TEST(PpUpstreamInputContent, PlansEveryBoundedNonAliasMipWithoutRetainingImages) {
    FinalGuestSurfaceWatchOrdinals selector{};
    selector.count = 2;
    selector.ordinals[0] = 1024;
    selector.ordinals[1] = 2031;
    const std::array views{
        PpUpstreamInputContentView{.width = 1920,
                                   .height = 1080,
                                   .bytes_per_texel = 8,
                                   .mip_count = 1,
                                   .color = true,
                                   .type_2d = true,
                                   .uniform_state = true},
        PpUpstreamInputContentView{.width = 960,
                                   .height = 540,
                                   .bytes_per_texel = 16,
                                   .mip_count = 1,
                                   .color = true,
                                   .type_2d = true,
                                   .uniform_state = true},
        PpUpstreamInputContentView{.width = 1920,
                                   .height = 1080,
                                   .bytes_per_texel = 4,
                                   .mip_count = 1,
                                   .color = true,
                                   .type_2d = true,
                                   .uniform_state = true},
        PpUpstreamInputContentView{.width = 480,
                                   .height = 270,
                                   .bytes_per_texel = 8,
                                   .mip_count = 7,
                                   .color = true,
                                   .type_2d = true,
                                   .uniform_state = true},
        PpUpstreamInputContentView{.width = 32,
                                   .height = 32,
                                   .bytes_per_texel = 4,
                                   .mip_count = 1,
                                   .color = true,
                                   .uniform_state = true},
        PpUpstreamInputContentView{.width = 1920,
                                   .height = 1080,
                                   .bytes_per_texel = 4,
                                   .mip_count = 1,
                                   .color = true,
                                   .type_2d = true,
                                   .uniform_state = true,
                                   .aliases_output = true},
    };
    const auto plan = PlanPpUpstreamInputContent({
        .enabled = true,
        .logical_width = 1280,
        .logical_height = 720,
        .base_offset = 1024,
        .selector = selector,
        .views = views,
        .buffer_alignment = 16,
        .max_regions = 256,
        .max_bytes = 4 * 1024 * 1024,
    });
    ASSERT_TRUE(plan.copy);
    EXPECT_EQ(plan.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_FALSE(plan.loss.Any());
    EXPECT_EQ(plan.capture_mask, 0x0fu);
    EXPECT_EQ(plan.unavailable_mask, 0x10u);
    EXPECT_EQ(plan.alias_mask, 0x20u);
    EXPECT_EQ(plan.inputs[0].region_count, 2u);
    EXPECT_EQ(plan.inputs[3].region_count, 14u);
    EXPECT_GE(plan.inputs[0].plane_offset, 1024u);
    EXPECT_GT(plan.total_bytes, 1024u);
    EXPECT_LT(plan.total_bytes, 4u * 1024 * 1024);
    EXPECT_FALSE(plan.cpu_wait);
    EXPECT_FALSE(plan.finish);
    EXPECT_FALSE(plan.retains_image);
    EXPECT_FALSE(plan.retains_vk_image);

    for (u32 input_index = 0; input_index < 4; ++input_index) {
        const auto& input = plan.inputs[input_index];
        for (u32 region_index = 0; region_index < input.region_count; ++region_index) {
            const auto& region = input.regions[region_index];
            EXPECT_GT(region.width, 0u);
            EXPECT_GT(region.height, 0u);
            EXPECT_GT(region.byte_size, 0u);
            EXPECT_LE(region.buffer_offset + region.byte_size, plan.total_bytes);
        }
    }
}

TEST(PpUpstreamInputContent, CompactsCapturedPlanesAndPreservesExplicitMasks) {
    FinalGuestSurfaceWatchOrdinals selector{};
    selector.count = 1;
    selector.ordinals[0] = 1299;
    const std::array views{
        PpUpstreamInputContentView{.width = 1920,
                                   .height = 1080,
                                   .bytes_per_texel = 8,
                                   .mip_count = 1,
                                   .color = true,
                                   .type_2d = true,
                                   .uniform_state = true},
        PpUpstreamInputContentView{.width = 32,
                                   .height = 32,
                                   .bytes_per_texel = 4,
                                   .mip_count = 1,
                                   .color = true,
                                   .uniform_state = true},
        PpUpstreamInputContentView{.width = 1920,
                                   .height = 1080,
                                   .bytes_per_texel = 4,
                                   .mip_count = 1,
                                   .color = true,
                                   .type_2d = true,
                                   .uniform_state = true,
                                   .aliases_output = true},
    };
    const auto plan = PlanPpUpstreamInputContent({
        .enabled = true,
        .logical_width = 1280,
        .logical_height = 720,
        .base_offset = 4096,
        .selector = selector,
        .views = views,
        .buffer_alignment = 16,
        .max_regions = 32,
        .max_bytes = 4 * 1024 * 1024,
    });
    ASSERT_TRUE(plan.copy);
    std::vector<std::byte> slot(plan.total_bytes);
    const auto& source = plan.inputs[0];
    std::ranges::fill(std::span{slot}.subspan(source.plane_offset, source.plane_bytes),
                      std::byte{0x5a});
    const auto compact = CompactPpUpstreamInputContent(plan, slot);
    ASSERT_EQ(compact.status, FinalGuestSurfaceStatus::Complete);
    ASSERT_FALSE(compact.loss.Any());
    EXPECT_EQ(compact.capture_mask, 0b001u);
    EXPECT_EQ(compact.unavailable_mask, 0b010u);
    EXPECT_EQ(compact.alias_mask, 0b100u);
    EXPECT_EQ(compact.bytes.size(), source.plane_bytes);
    EXPECT_EQ(compact.plane_offsets[0], 0u);
    EXPECT_EQ(compact.bytes.front(), std::byte{0x5a});
    EXPECT_TRUE(compact.bytes.back() == std::byte{0x5a});
}

TEST(PpUpstreamInputContent, RejectsUnboundedFormatsAndCapacityTransactionally) {
    FinalGuestSurfaceWatchOrdinals selector{};
    selector.count = 1;
    selector.ordinals[0] = 1297;
    const std::array invalid_views{
        PpUpstreamInputContentView{.width = 1280,
                                   .height = 720,
                                   .bytes_per_texel = 3,
                                   .mip_count = 1,
                                   .color = true,
                                   .type_2d = true,
                                   .uniform_state = true},
    };
    const auto unsupported = PlanPpUpstreamInputContent({
        .enabled = true,
        .logical_width = 1280,
        .logical_height = 720,
        .selector = selector,
        .views = invalid_views,
        .buffer_alignment = 16,
        .max_regions = 32,
        .max_bytes = 4 * 1024 * 1024,
    });
    EXPECT_FALSE(unsupported.copy);
    EXPECT_EQ(unsupported.status, FinalGuestSurfaceStatus::Unsupported);
    EXPECT_EQ(unsupported.loss.unsupported_format, 1u);

    const std::array large_views{
        PpUpstreamInputContentView{.width = 1920,
                                   .height = 1080,
                                   .bytes_per_texel = 16,
                                   .mip_count = 1,
                                   .color = true,
                                   .type_2d = true,
                                   .uniform_state = true},
    };
    const auto capacity = PlanPpUpstreamInputContent({
        .enabled = true,
        .logical_width = 1280,
        .logical_height = 720,
        .selector = selector,
        .views = large_views,
        .buffer_alignment = 16,
        .max_regions = 32,
        .max_bytes = 1024,
    });
    EXPECT_FALSE(capacity.copy);
    EXPECT_EQ(capacity.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(capacity.loss.byte_capacity, 1u);

    auto disabled = PlanPpUpstreamInputContent({});
    EXPECT_FALSE(disabled.copy);
    EXPECT_EQ(disabled.total_bytes, 0u);
}

TEST(PpUpstreamInputContent, ClassifiesExactRawTripletsWithoutPixelInterpretation) {
    const std::array a{std::byte{0x10}, std::byte{0x20}, std::byte{0x30}, std::byte{0x40}};
    const std::array b{std::byte{0x10}, std::byte{0x21}, std::byte{0x30}, std::byte{0x40}};
    const std::array c = a;
    const std::array d{std::byte{0x10}, std::byte{0x22}, std::byte{0x30}, std::byte{0x40}};
    EXPECT_EQ(ClassifyPpUpstreamInputRawTriplet(a, b, c), PpUpstreamInputRawClass::Aba);
    EXPECT_EQ(ClassifyPpUpstreamInputRawTriplet(a, a, a), PpUpstreamInputRawClass::Stable);
    EXPECT_EQ(ClassifyPpUpstreamInputRawTriplet(a, b, d), PpUpstreamInputRawClass::Ambiguous);
    EXPECT_EQ(ClassifyPpUpstreamInputRawTriplet(a, b, std::span<const std::byte>{}),
              PpUpstreamInputRawClass::Unavailable);
}

TEST(PpUpstreamInputContent, CapturesOnlyTheConfiguredPrivateCandidate) {
    PpUpstreamInputCaptureGate gate{0};
    EXPECT_TRUE(gate.Preview(0).capture);
    EXPECT_FALSE(gate.Preview(1).capture);
    EXPECT_EQ(gate.Preview(1).status, FinalGuestSurfaceStatus::AlreadyConsumed);
    EXPECT_TRUE(gate.Complete(0, true).publish);
    EXPECT_EQ(gate.Complete(1, true).status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_TRUE(gate.Preview(0).capture);
    EXPECT_EQ(gate.Complete(0, false).status, FinalGuestSurfaceStatus::InvalidationLoss);
    gate.Reset();
    EXPECT_TRUE(gate.Preview(0).capture);
}

TEST(PpUpstreamInputContent, ReducerTreatsAllMipsAsOneOpaqueLogicalRegion) {
    PpTerminalScopeContentReducer reducer{{.frame_start = 700, .frame_count = 3}, 8};
    const PpTerminalScopeContentHistoryLayout layout{
        .total_bytes = 8,
        .input_count = 1,
        .input_capture_mask = 1,
        .input_planes =
            {{{.region_count = 2,
               .plane_offset = 0,
               .plane_bytes = 8,
               .regions = {{{.logical_ordinal = 1024, .buffer_offset = 0, .byte_size = 4},
                            {.logical_ordinal = 1024, .buffer_offset = 4, .byte_size = 4}}},
               .bytes_per_texel = 2,
               .raw_compare = true,
               .status = FinalGuestSurfaceStatus::Complete}}},
        .format = FinalGuestSurfaceFormat::Rgba8,
    };
    const std::array a{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
                       std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
    auto b = a;
    b[7] = std::byte{9};
    reducer.ObserveContent(700, layout, a, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(701, layout, b, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveContent(702, layout, a, FinalGuestSurfaceStatus::Complete, {});
    reducer.ObserveCalibration({.request_ordinal = 1, .sequence = 700, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 2, .sequence = 701, .valid = true});
    reducer.ObserveCalibration({.request_ordinal = 3, .sequence = 702, .valid = true});
    const auto reports = reducer.TakeReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports[0].sampled_input_aba_ordinals[0], (std::vector<u32>{1024}));
    EXPECT_TRUE(reports[0].sampled_input_stable_ordinals[0].empty());
    EXPECT_TRUE(reports[0].sampled_input_ambiguous_ordinals[0].empty());
    EXPECT_TRUE(reports[0].sampled_input_localized_visual_return_ordinals[0].empty());
    EXPECT_FALSE(reports[0].loss.Any());
}

TEST(PpUpstreamInputContent, RestoresTheExactUniformTrackerAfterAPartialCopy) {
    struct TrackerState {
        u32 layout{};
        u32 access{};

        auto operator<=>(const TrackerState&) const = default;
    };

    const TrackerState original{.layout = 7, .access = 11};
    TrackerState current{.layout = 13, .access = 17};
    std::vector<TrackerState> transient_subresources(7, current);
    RestorePpUpstreamInputUniformTracker(current, transient_subresources, original);
    EXPECT_EQ(current, original);
    EXPECT_TRUE(transient_subresources.empty());
}

TEST(PpUpstreamInputContent, PreservesSpecificCaptureLossBeforePublicationReconciliation) {
    const auto capacity = ReconcilePpUpstreamInputPublication(
        true, false, 0, 0, FinalGuestSurfaceStatus::CapacityLoss,
        FinalGuestSurfaceLoss{.byte_capacity = 1}, false);
    EXPECT_EQ(capacity.status, FinalGuestSurfaceStatus::CapacityLoss);
    EXPECT_EQ(capacity.loss.byte_capacity, 1u);
    EXPECT_FALSE(capacity.copy);

    const auto missing = ReconcilePpUpstreamInputPublication(
        true, false, 0, 0, FinalGuestSurfaceStatus::Complete, {}, true);
    EXPECT_EQ(missing.status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(missing.loss.gap, 1u);
    EXPECT_FALSE(missing.copy);

    const auto ready = ReconcilePpUpstreamInputPublication(
        true, true, 0, 0, FinalGuestSurfaceStatus::Complete, {}, true);
    EXPECT_EQ(ready.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_TRUE(ready.copy);
}

TEST(PpUpstreamInputContent, SlotReleaseFailureMakesFinalCoverageIncomplete) {
    const auto complete = ReconcilePpUpstreamInputSlotRelease(
        true, true, FinalGuestSurfaceStatus::Complete, {}, true);
    EXPECT_EQ(complete.status, FinalGuestSurfaceStatus::Complete);
    EXPECT_TRUE(complete.copy);

    const auto failed = ReconcilePpUpstreamInputSlotRelease(
        true, false, FinalGuestSurfaceStatus::Complete, {}, true);
    EXPECT_EQ(failed.status, FinalGuestSurfaceStatus::InvalidationLoss);
    EXPECT_EQ(failed.loss.invalidation, 1u);
    EXPECT_FALSE(failed.copy);

    const auto no_slot = ReconcilePpUpstreamInputSlotRelease(
        false, false, FinalGuestSurfaceStatus::GapLoss, FinalGuestSurfaceLoss{.gap = 1}, false);
    EXPECT_EQ(no_slot.status, FinalGuestSurfaceStatus::GapLoss);
    EXPECT_EQ(no_slot.loss.gap, 1u);
}

} // namespace
