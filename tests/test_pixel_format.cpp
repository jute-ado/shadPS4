// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "video_core/amdgpu/pixel_format.h"

TEST(PixelFormat, SnormNzUsesDefinedComponentWidthConversions) {
    using AmdGpu::DataFormat;
    using AmdGpu::MapNumberConversion;
    using AmdGpu::NumberConversion;
    using AmdGpu::NumberFormat;

    EXPECT_EQ(MapNumberConversion(NumberFormat::SnormNz, DataFormat::Format8),
              NumberConversion::Sint8ToSnormNz);
    EXPECT_EQ(MapNumberConversion(NumberFormat::SnormNz, DataFormat::Format8_8_8_8),
              NumberConversion::Sint8ToSnormNz);
    EXPECT_EQ(MapNumberConversion(NumberFormat::SnormNz, DataFormat::Format16_16),
              NumberConversion::Sint16ToSnormNz);
}

TEST(PixelFormat, UnsupportedSnormNzPairingsDoNotAbortShaderCompilation) {
    using AmdGpu::DataFormat;
    using AmdGpu::MapNumberConversion;
    using AmdGpu::NumberConversion;
    using AmdGpu::NumberFormat;

    static_assert(MapNumberConversion(NumberFormat::SnormNz, DataFormat::Format32_32) ==
                  NumberConversion::None);
    static_assert(MapNumberConversion(NumberFormat::SnormNz, DataFormat::FormatInvalid) ==
                  NumberConversion::None);
}

TEST(PixelFormat, EveryDeclaredDataFormatHasADiagnosticName) {
    std::ifstream input{SHADPS4_PIXEL_FORMAT_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto source = contents.str();

    for (const auto name : {"FormatFmask8_S8_F1", "FormatFmask64_S16_F8", "Format4_4",
                            "Format32_As_8", "Format32_As_32_32_32_32"}) {
        EXPECT_NE(source.find(std::string{"case DataFormat::"} + name), std::string::npos) << name;
        EXPECT_NE(source.find(std::string{"return \""} + name + "\""), std::string::npos) << name;
    }
}

TEST(PixelFormat, MacroTiledBlockImagesUseBlockDimensionsWithoutAborting) {
    std::ifstream input{SHADPS4_IMAGE_INFO_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto source = contents.str();
    const auto update = source.find("void ImageInfo::UpdateSize");
    const auto end = source.find("s32 ImageInfo::MipOf", update);
    ASSERT_NE(update, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(update, end - update);
    const auto block_dims = body.find("mip_w = (mip_w + 3) / 4");
    const auto macro_size = body.find("ImageSizeMacroTiled(");
    const auto restore_dims = body.find("mip_info.pitch = std::max(mip_info.pitch * 4");
    ASSERT_NE(block_dims, std::string::npos);
    ASSERT_NE(macro_size, std::string::npos);
    ASSERT_NE(restore_dims, std::string::npos);
    EXPECT_LT(block_dims, macro_size);
    EXPECT_LT(macro_size, restore_dims);
    EXPECT_EQ(body.find("ASSERT(!props.is_block)"), std::string::npos);
}
