// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/resource.h"

TEST(SamplerLod, NonMipmappedSamplerUsesSingleEffectiveLevel) {
    AmdGpu::Sampler sampler{};
    sampler.mip_filter.Assign(AmdGpu::MipFilter::None);
    sampler.min_lod.Assign(16);
    sampler.max_lod.Assign(0);

    EXPECT_FLOAT_EQ(sampler.EffectiveMinLod(), 0.0f);
    EXPECT_FLOAT_EQ(sampler.EffectiveMaxLod(), 0.0f);
}

TEST(SamplerLod, OrderedMipmappedRangeIsPreserved) {
    AmdGpu::Sampler sampler{};
    sampler.mip_filter.Assign(AmdGpu::MipFilter::Point);
    sampler.min_lod.Assign(16);
    sampler.max_lod.Assign(64);

    EXPECT_FLOAT_EQ(sampler.EffectiveMinLod(), 0.0625f);
    EXPECT_FLOAT_EQ(sampler.EffectiveMaxLod(), 0.25f);
}

TEST(SamplerLod, InvertedMipmappedRangeCollapsesAtMinimum) {
    AmdGpu::Sampler sampler{};
    sampler.mip_filter.Assign(AmdGpu::MipFilter::Linear);
    sampler.min_lod.Assign(64);
    sampler.max_lod.Assign(16);

    EXPECT_FLOAT_EQ(sampler.EffectiveMinLod(), 0.25f);
    EXPECT_FLOAT_EQ(sampler.EffectiveMaxLod(), 0.25f);
}
