// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/pixel_format.h"

namespace {

TEST(PixelNumberConversion, CheckedMappingRefusesUnsupportedWidthsWithoutInventingConversion) {
    for (const auto format :
         {AmdGpu::DataFormat::FormatInvalid, AmdGpu::DataFormat::Format10_10_10_2,
          AmdGpu::DataFormat::Format2_10_10_10, AmdGpu::DataFormat::FormatBc4}) {
        EXPECT_FALSE(AmdGpu::TryMapNumberConversion(AmdGpu::NumberFormat::SnormNz, format));
    }
}

TEST(PixelNumberConversion, CheckedMappingPreservesExistingNonSnormConversions) {
    using Data = AmdGpu::DataFormat;
    using Number = AmdGpu::NumberFormat;
    using Conversion = AmdGpu::NumberConversion;
    struct Case {
        Number number;
        Data data;
        Conversion conversion;
    };
    constexpr Case cases[] = {
        {Number::Unorm, Data::Format32, Conversion::Uint32ToUnorm},
        {Number::Unorm, Data::Format8, Conversion::None},
        {Number::Srgb, Data::FormatBc4, Conversion::None},
        {Number::Srgb, Data::FormatBc5, Conversion::None},
        {Number::Srgb, Data::FormatBc6, Conversion::SrgbToNorm},
        {Number::Uscaled, Data::Format16, Conversion::UintToUscaled},
        {Number::Sscaled, Data::Format16, Conversion::SintToSscaled},
        {Number::Ubnorm, Data::Format8, Conversion::UnormToUbnorm},
        {Number::Float, Data::Format32, Conversion::None},
        {Number::Uint, Data::Format32, Conversion::None},
        {Number::Sint, Data::Format32, Conversion::None},
    };
    for (const auto& sample : cases) {
        const auto conversion = AmdGpu::TryMapNumberConversion(sample.number, sample.data);
        ASSERT_TRUE(conversion);
        EXPECT_EQ(*conversion, sample.conversion);
        EXPECT_EQ(AmdGpu::MapNumberConversion(sample.number, sample.data), *conversion);
    }
}

TEST(PixelNumberConversion, MapsEvery32BitSnormNzVectorWidth) {
    static_assert(
        AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz, AmdGpu::DataFormat::Format32) ==
        AmdGpu::NumberConversion::Sint32ToSnormNz);
    static_assert(AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz,
                                              AmdGpu::DataFormat::Format32_32) ==
                  AmdGpu::NumberConversion::Sint32ToSnormNz);
    static_assert(AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz,
                                              AmdGpu::DataFormat::Format32_32_32) ==
                  AmdGpu::NumberConversion::Sint32ToSnormNz);
    static_assert(AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz,
                                              AmdGpu::DataFormat::Format32_32_32_32) ==
                  AmdGpu::NumberConversion::Sint32ToSnormNz);
    SUCCEED();
}

TEST(PixelNumberConversion, PreservesExisting8And16BitSnormNzMappings) {
    static_assert(
        AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz, AmdGpu::DataFormat::Format8) ==
        AmdGpu::NumberConversion::Sint8ToSnormNz);
    static_assert(
        AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz, AmdGpu::DataFormat::Format8_8) ==
        AmdGpu::NumberConversion::Sint8ToSnormNz);
    static_assert(AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz,
                                              AmdGpu::DataFormat::Format8_8_8_8) ==
                  AmdGpu::NumberConversion::Sint8ToSnormNz);
    static_assert(
        AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz, AmdGpu::DataFormat::Format16) ==
        AmdGpu::NumberConversion::Sint16ToSnormNz);
    static_assert(AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz,
                                              AmdGpu::DataFormat::Format16_16) ==
                  AmdGpu::NumberConversion::Sint16ToSnormNz);
    static_assert(AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz,
                                              AmdGpu::DataFormat::Format16_16_16_16) ==
                  AmdGpu::NumberConversion::Sint16ToSnormNz);
    SUCCEED();
}

} // namespace
