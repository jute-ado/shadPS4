// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/pixel_format.h"

TEST(PixelNumberConversion, MapsEvery32BitSnormNzVectorWidthWithoutRejectingTheFormat) {
    static_assert(AmdGpu::MapNumberConversion(AmdGpu::NumberFormat::SnormNz,
                                              AmdGpu::DataFormat::Format32) ==
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

TEST(PixelNumberConversion, RuntimeSpecializationPreserves32BitSnormNzConversion) {
    Shader::PsColorBuffer color_buffer{};
    color_buffer.num_conversion = AmdGpu::NumberConversion::Sint32ToSnormNz;

    EXPECT_EQ(color_buffer.num_conversion, AmdGpu::NumberConversion::Sint32ToSnormNz);
}
