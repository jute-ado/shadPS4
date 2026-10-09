// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "common/object_pool.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/recompiler.h"
#include "spirv/unified1/spirv.hpp"

namespace {

using namespace Shader;

std::vector<u32> EmitDescriptorStage(Info& info, Backend::Bindings& bindings) {
    IR::Program program{info};
    Pools pools{};
    IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(info.stage);
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> DescriptorBindings(const std::vector<u32>& spirv) {
    std::vector<u32> bindings;
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const u32 word_count = instruction >> 16;
        if (word_count == 0U || word_count > spirv.size() - offset) {
            ADD_FAILURE() << "Malformed SPIR-V instruction at word " << offset;
            break;
        }
        if (static_cast<spv::Op>(instruction & 0xffffU) == spv::Op::OpDecorate &&
            word_count >= 4U && spirv[offset + 2] == spv::DecorationBinding) {
            bindings.push_back(spirv[offset + 3]);
        }
        offset += word_count;
    }
    std::ranges::sort(bindings);
    return bindings;
}

void AddInlineSampler(Info& info) {
    SamplerResource sampler{};
    sampler.is_inline_sampler = true;
    info.samplers.push_back(sampler);
}

TEST(DescriptorArraySpirv, EmissionAndCachedStageAdvanceUseTheSameBindingOrigins) {
    Info array_stage{};
    array_stage.stage = Stage::Fragment;
    array_stage.l_stage = LogicalStage::Fragment;

    auto sharp = AmdGpu::Image::Null(false);
    sharp.base_level = 0;
    sharp.last_level = 2;
    array_stage.flattened_ud_buf.resize(sizeof(sharp) / sizeof(u32));
    std::memcpy(array_stage.flattened_ud_buf.data(), &sharp, sizeof(sharp));

    ImageResource image{};
    image.sharp_idx = 0;
    image.is_r128 = true;
    image.mip_fallback_mode = MipStorageFallbackMode::DynamicIndex;
    array_stage.images.push_back(image);
    AddInlineSampler(array_stage);
    ASSERT_EQ(array_stage.images.front().NumBindings(array_stage), 3U);

    Info following_stage{};
    following_stage.stage = Stage::Vertex;
    following_stage.l_stage = LogicalStage::Vertex;
    AddInlineSampler(following_stage);

    Backend::Bindings emitted{};
    const auto array_spirv = EmitDescriptorStage(array_stage, emitted);
    EXPECT_EQ(DescriptorBindings(array_spirv), (std::vector<u32>{0U, 1U}));
    EXPECT_EQ(emitted.unified, 2U);

    Backend::Bindings cached{};
    array_stage.AddBindings(cached);
    EXPECT_EQ(cached, emitted);

    const auto emitted_following_spirv = EmitDescriptorStage(following_stage, emitted);
    const auto cached_following_spirv = EmitDescriptorStage(following_stage, cached);
    EXPECT_EQ(DescriptorBindings(emitted_following_spirv), (std::vector<u32>{2U}));
    EXPECT_EQ(DescriptorBindings(cached_following_spirv), (std::vector<u32>{2U}));
    EXPECT_EQ(cached, emitted);
}

} // namespace
