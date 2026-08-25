// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/texture_cache/image_layout_policy.h"

TEST(ImageLayoutPolicy, ClassifiesPrtThinAsMacroTiled) {
    const auto layout =
        VideoCore::ClassifyImageArrayLayout(AmdGpu::ArrayMode::ArrayPrtTiledThin1);

    EXPECT_EQ(layout.kind, VideoCore::ImageArrayLayoutKind::MacroTiled);
    EXPECT_EQ(layout.thickness, 1U);
}

TEST(ImageLayoutPolicy, ClassifiesPrtThickAndExtraThickLayouts) {
    const auto thick =
        VideoCore::ClassifyImageArrayLayout(AmdGpu::ArrayMode::ArrayPrt2DTiledThick);
    const auto extra_thick =
        VideoCore::ClassifyImageArrayLayout(AmdGpu::ArrayMode::Array3DTiledXThick);

    EXPECT_EQ(thick.kind, VideoCore::ImageArrayLayoutKind::MacroTiled);
    EXPECT_EQ(thick.thickness, 4U);
    EXPECT_EQ(extra_thick.kind, VideoCore::ImageArrayLayoutKind::MacroTiled);
    EXPECT_EQ(extra_thick.thickness, 8U);
}

TEST(ImageLayoutPolicy, LeavesLinearGeneralUnsupported) {
    const auto layout =
        VideoCore::ClassifyImageArrayLayout(AmdGpu::ArrayMode::ArrayLinearGeneral);

    EXPECT_EQ(layout.kind, VideoCore::ImageArrayLayoutKind::Invalid);
}

TEST(ImageLayoutPolicy, ClassifiesEverySupportedArrayMode) {
    using AmdGpu::ArrayMode;
    using VideoCore::ImageArrayLayoutKind;

    struct ExpectedLayout {
        ArrayMode mode;
        ImageArrayLayoutKind kind;
        u32 thickness;
    };
    constexpr std::array expected{
        ExpectedLayout{ArrayMode::ArrayLinearAligned, ImageArrayLayoutKind::Linear, 1},
        ExpectedLayout{ArrayMode::Array1DTiledThin1, ImageArrayLayoutKind::MicroTiled, 1},
        ExpectedLayout{ArrayMode::Array1DTiledThick, ImageArrayLayoutKind::MicroTiled, 4},
        ExpectedLayout{ArrayMode::Array2DTiledThin1, ImageArrayLayoutKind::MacroTiled, 1},
        ExpectedLayout{ArrayMode::ArrayPrtTiledThin1, ImageArrayLayoutKind::MacroTiled, 1},
        ExpectedLayout{ArrayMode::ArrayPrt2DTiledThin1, ImageArrayLayoutKind::MacroTiled, 1},
        ExpectedLayout{ArrayMode::Array2DTiledThick, ImageArrayLayoutKind::MacroTiled, 4},
        ExpectedLayout{ArrayMode::Array2DTiledXThick, ImageArrayLayoutKind::MacroTiled, 8},
        ExpectedLayout{ArrayMode::ArrayPrtTiledThick, ImageArrayLayoutKind::MacroTiled, 4},
        ExpectedLayout{ArrayMode::ArrayPrt2DTiledThick, ImageArrayLayoutKind::MacroTiled, 4},
        ExpectedLayout{ArrayMode::ArrayPrt3DTiledThin1, ImageArrayLayoutKind::MacroTiled, 1},
        ExpectedLayout{ArrayMode::Array3DTiledThin1, ImageArrayLayoutKind::MacroTiled, 1},
        ExpectedLayout{ArrayMode::Array3DTiledThick, ImageArrayLayoutKind::MacroTiled, 4},
        ExpectedLayout{ArrayMode::Array3DTiledXThick, ImageArrayLayoutKind::MacroTiled, 8},
        ExpectedLayout{ArrayMode::ArrayPrt3DTiledThick, ImageArrayLayoutKind::MacroTiled, 4},
    };

    for (const auto& item : expected) {
        const auto actual = VideoCore::ClassifyImageArrayLayout(item.mode);
        EXPECT_EQ(actual.kind, item.kind) << "array mode " << static_cast<u32>(item.mode);
        EXPECT_EQ(actual.thickness, item.thickness)
            << "array mode " << static_cast<u32>(item.mode);
    }
}

TEST(ImageLayoutPolicy, ConvertsBlockCompressedMipGeometryToAndFromBlockUnits) {
    const auto blocks = VideoCore::ToBlockCompressedMipGeometry(257, 129);
    EXPECT_EQ(blocks.pitch, 65U);
    EXPECT_EQ(blocks.height, 33U);

    const auto texels = VideoCore::FromBlockCompressedMipGeometry(64, 32);
    EXPECT_EQ(texels.pitch, 256U);
    EXPECT_EQ(texels.height, 128U);
}

TEST(ImageLayoutPolicy, RestoredBlockCompressedGeometryKeepsMinimumTileExtent) {
    const auto texels = VideoCore::FromBlockCompressedMipGeometry(1, 1);
    EXPECT_EQ(texels.pitch, 32U);
    EXPECT_EQ(texels.height, 32U);
}
