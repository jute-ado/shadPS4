// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/frontend/translate/color_buffer_rounding.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/recompiler.h"

namespace {

using namespace Shader;
using Shader::Gcn::ApplyColorBufferRoundMode;

struct RoundingFixture {
    Info info{};
    IR::Program program{info};
    Pools pools{};
    IR::Block* block{pools.block_pool.Create(pools.inst_pool)};
    IR::IREmitter ir{*block};

    RoundingFixture() {
        program.blocks.push_back(block);
        ir.Prologue();
    }
};

TEST(ColorExportRounding, ByHalfUnorm8EmitsExactAddHalfFloorQuantization) {
    RoundingFixture fixture;
    PsColorBuffer color{};
    color.data_format = AmdGpu::DataFormat::Format8;
    color.num_format = AmdGpu::NumberFormat::Unorm;
    color.truncate = false;
    const IR::F32 source = fixture.ir.GetVectorReg<IR::F32>(IR::VectorReg(0));

    const IR::F32 result = ApplyColorBufferRoundMode(fixture.ir, source, color);

    ASSERT_EQ(result.InstRecursive()->GetOpcode(), IR::Opcode::FPDiv32);
    const auto* floor = result.InstRecursive()->Arg(0).InstRecursive();
    ASSERT_EQ(floor->GetOpcode(), IR::Opcode::FPFloor32);
    const auto* add = floor->Arg(0).InstRecursive();
    ASSERT_EQ(add->GetOpcode(), IR::Opcode::FPAdd32);
    EXPECT_EQ(add->Arg(0).InstRecursive()->GetOpcode(), IR::Opcode::FPMul32);
}

TEST(ColorExportRounding, TruncateUnorm8OmitsTheHalfStepAndUnsupportedFormatsAreIdentity) {
    RoundingFixture fixture;
    PsColorBuffer color{};
    color.data_format = AmdGpu::DataFormat::Format8_8_8_8;
    color.num_format = AmdGpu::NumberFormat::Unorm;
    color.truncate = true;
    const IR::F32 source = fixture.ir.GetVectorReg<IR::F32>(IR::VectorReg(0));

    const IR::F32 truncated = ApplyColorBufferRoundMode(fixture.ir, source, color);
    ASSERT_EQ(truncated.InstRecursive()->GetOpcode(), IR::Opcode::FPDiv32);
    const auto* floor = truncated.InstRecursive()->Arg(0).InstRecursive();
    ASSERT_EQ(floor->GetOpcode(), IR::Opcode::FPFloor32);
    EXPECT_EQ(floor->Arg(0).InstRecursive()->GetOpcode(), IR::Opcode::FPMul32);

    color.num_format = AmdGpu::NumberFormat::Float;
    EXPECT_EQ(ApplyColorBufferRoundMode(fixture.ir, source, color).InstRecursive(),
              source.InstRecursive());
    color.num_format = AmdGpu::NumberFormat::Unorm;
    color.data_format = AmdGpu::DataFormat::Format16;
    EXPECT_EQ(ApplyColorBufferRoundMode(fixture.ir, source, color).InstRecursive(),
              source.InstRecursive());
}

} // namespace
