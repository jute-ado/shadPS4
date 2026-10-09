// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/pixel_format.h"

namespace AmdGpu {
namespace {

TEST(NumberConversion, MapsEveryThirtyTwoBitSnormNzShape) {
    constexpr std::array formats{
        DataFormat::Format32,
        DataFormat::Format32_32,
        DataFormat::Format32_32_32,
        DataFormat::Format32_32_32_32,
    };

    for (const auto format : formats) {
        EXPECT_EQ(MapNumberConversion(NumberFormat::SnormNz, format),
                  NumberConversion::Sint32ToSnormNz);
    }
}

TEST(NumberConversion, PreservesExistingSnormNzWidths) {
    EXPECT_EQ(MapNumberConversion(NumberFormat::SnormNz, DataFormat::Format8_8_8_8),
              NumberConversion::Sint8ToSnormNz);
    EXPECT_EQ(MapNumberConversion(NumberFormat::SnormNz, DataFormat::Format16_16_16_16),
              NumberConversion::Sint16ToSnormNz);
}

} // namespace
} // namespace AmdGpu
