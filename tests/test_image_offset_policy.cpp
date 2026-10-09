// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/backend/spirv/image_offset_policy.h"

using Shader::Backend::SPIRV::CanUseRuntimeImageOffset;
using Shader::Backend::SPIRV::ImageOffsetInstruction;
using Shader::Backend::SPIRV::NeedsImageGatherExtendedCapability;

TEST(ImageOffsetPolicy, RequiresMaintenance8ForSampleInstructions) {
    EXPECT_FALSE(CanUseRuntimeImageOffset(ImageOffsetInstruction::Sample, false, true));
    EXPECT_FALSE(CanUseRuntimeImageOffset(ImageOffsetInstruction::Sample, true, false));
    EXPECT_TRUE(CanUseRuntimeImageOffset(ImageOffsetInstruction::Sample, true, true));
}

TEST(ImageOffsetPolicy, AllowsGatherThroughImageGatherExtended) {
    EXPECT_FALSE(CanUseRuntimeImageOffset(ImageOffsetInstruction::Gather, true, false));
    EXPECT_TRUE(CanUseRuntimeImageOffset(ImageOffsetInstruction::Gather, false, true));
}

TEST(ImageOffsetPolicy, DeclaresGatherCapabilityForDynamicSampleOffsets) {
    EXPECT_FALSE(NeedsImageGatherExtendedCapability(false, false));
    EXPECT_TRUE(NeedsImageGatherExtendedCapability(true, false));
    EXPECT_TRUE(NeedsImageGatherExtendedCapability(false, true));
}
