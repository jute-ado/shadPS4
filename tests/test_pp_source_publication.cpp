// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/host_passes/pp_source_publication.h"

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

} // namespace
