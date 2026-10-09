// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/ir/passes/buffer_format_policy.h"
#include "video_core/amdgpu/pixel_format.h"

TEST(BufferFormatPolicy, InvalidStoresAreDiscardedBeforeVectorConstruction) {
    EXPECT_FALSE(Shader::Optimization::ShouldLowerBufferFormatStore(0));
    EXPECT_TRUE(Shader::Optimization::ShouldLowerBufferFormatStore(1));
}

TEST(BufferFormatPolicy, InvalidDescriptorsSkipNumberConversionMapping) {
    EXPECT_FALSE(Shader::Optimization::ShouldMapBufferNumberConversion(0));
    EXPECT_TRUE(Shader::Optimization::ShouldMapBufferNumberConversion(1));
}

TEST(ResourceFormatPolicy, InvalidDescriptorsHaveNoNumberConversion) {
    EXPECT_EQ(AmdGpu::MapResourceNumberConversion(AmdGpu::NumberFormat::SnormNz,
                                                  AmdGpu::DataFormat::FormatInvalid),
              AmdGpu::NumberConversion::None);
    EXPECT_EQ(AmdGpu::MapResourceNumberConversion(AmdGpu::NumberFormat::SnormNz,
                                                  AmdGpu::DataFormat::Format8),
              AmdGpu::NumberConversion::Sint8ToSnormNz);
}

TEST(BufferAddressPolicy, ThreadIdContributionIsComputeOnly) {
    EXPECT_TRUE(Shader::Optimization::ShouldApplyBufferThreadId(true, true));
    EXPECT_FALSE(Shader::Optimization::ShouldApplyBufferThreadId(true, false));
    EXPECT_FALSE(Shader::Optimization::ShouldApplyBufferThreadId(false, true));
}

TEST(BufferFormatPolicy, ReservedEncodingIsTreatedAsInvalid) {
    EXPECT_EQ(Shader::Optimization::NormalizeBufferDataFormat(15), 0u);
    EXPECT_EQ(Shader::Optimization::NormalizeBufferDataFormat(59), 0u);
    EXPECT_EQ(Shader::Optimization::NormalizeBufferDataFormat(14), 14u);
}

TEST(BufferFormatPolicy, PackedSnormNzUsesPerComponentBitWidths) {
    EXPECT_TRUE(Shader::Optimization::IsPacked10_11_11SnormNz(
        AmdGpu::DataFormat::Format11_11_10, AmdGpu::NumberFormat::SnormNz));
    EXPECT_EQ(Shader::Optimization::Packed10_11_11SnormNzMax(0), 2047u);
    EXPECT_EQ(Shader::Optimization::Packed10_11_11SnormNzMax(1), 2047u);
    EXPECT_EQ(Shader::Optimization::Packed10_11_11SnormNzMax(2), 1023u);
}

TEST(BufferFormatPolicy, DescriptorFormatParticipatesWhenInstructionIsUntyped) {
    EXPECT_EQ(Shader::Optimization::SelectBufferDataFormat(false, 0, 7), 7u);
    EXPECT_EQ(Shader::Optimization::SelectBufferNumberFormat(false, 0, 6), 6u);
    EXPECT_EQ(Shader::Optimization::SelectBufferDataFormat(true, 6, 7), 6u);
}
