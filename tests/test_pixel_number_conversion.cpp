// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/pixel_format.h"

namespace {

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
