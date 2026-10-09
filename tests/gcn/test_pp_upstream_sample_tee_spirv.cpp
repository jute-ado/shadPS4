// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include <spirv/unified1/spirv.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/pp_upstream_sample_tee.h"
#include "shader_recompiler/recompiler.h"

namespace {

std::vector<u32> EmitPpUpstreamSampleTeeFixture(bool instrument = true) {
    Shader::Info info{};
    info.stage = Shader::Stage::Fragment;
    info.l_stage = Shader::LogicalStage::Fragment;
    if (instrument) {
        info.loads.Set(Shader::IR::Attribute::FragCoord, 0);
        info.loads.Set(Shader::IR::Attribute::FragCoord, 1);
        info.buffers.push_back({
            .used_types = Shader::IR::Type::U32,
            .inline_cbuf = AmdGpu::Buffer::Placeholder(Shader::PpUpstreamSampleTeeBufferBytes),
            .buffer_type = Shader::BufferType::PpUpstreamSampleTee,
            .is_written = true,
        });
    }

    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());

    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    if (instrument) {
        const auto value = ir.CompositeConstruct(ir.Imm32(0.25f), ir.Imm32(0.5f),
                                                 ir.Imm32(0.75f), ir.Imm32(1.0f));
        ir.PpUpstreamSampleTee(value, 2);
    }
    ir.Epilogue();

    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.instrument_pp_upstream_sample_tee = instrument;
    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Fragment);
    Shader::Backend::Bindings bindings{};
    return Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<size_t> FindOpcodeOffsets(std::span<const u32> words, spv::Op opcode);

struct PushBlockTailLayout {
    u32 components{};
    u32 member_offset{~u32{}};
    u32 component_width{};
    u32 component_signedness{~u32{}};
};

PushBlockTailLayout InspectPushBlockTail(std::span<const u32> words) {
    u32 pointer_type{};
    for (const size_t offset : FindOpcodeOffsets(words, spv::Op::OpVariable)) {
        const u32 word_count = words[offset] >> 16;
        if (word_count >= 4 &&
            words[offset + 3] == static_cast<u32>(spv::StorageClassPushConstant)) {
            pointer_type = words[offset + 1];
            break;
        }
    }
    EXPECT_NE(pointer_type, 0u);

    u32 struct_type{};
    for (const size_t offset : FindOpcodeOffsets(words, spv::Op::OpTypePointer)) {
        const u32 word_count = words[offset] >> 16;
        if (word_count >= 4 && words[offset + 1] == pointer_type &&
            words[offset + 2] == static_cast<u32>(spv::StorageClassPushConstant)) {
            struct_type = words[offset + 3];
            break;
        }
    }
    EXPECT_NE(struct_type, 0u);

    u32 tail_type{};
    u32 tail_member{};
    for (const size_t offset : FindOpcodeOffsets(words, spv::Op::OpTypeStruct)) {
        const u32 word_count = words[offset] >> 16;
        if (word_count >= 3 && words[offset + 1] == struct_type) {
            tail_type = words[offset + word_count - 1];
            tail_member = word_count - 3;
            break;
        }
    }
    EXPECT_NE(tail_type, 0u);

    PushBlockTailLayout layout{};
    u32 component_type{};
    for (const size_t offset : FindOpcodeOffsets(words, spv::Op::OpTypeVector)) {
        const u32 word_count = words[offset] >> 16;
        if (word_count >= 4 && words[offset + 1] == tail_type) {
            component_type = words[offset + 2];
            layout.components = words[offset + 3];
            break;
        }
    }
    EXPECT_NE(component_type, 0u);

    for (const size_t offset : FindOpcodeOffsets(words, spv::Op::OpTypeInt)) {
        const u32 word_count = words[offset] >> 16;
        if (word_count >= 4 && words[offset + 1] == component_type) {
            layout.component_width = words[offset + 2];
            layout.component_signedness = words[offset + 3];
            break;
        }
    }

    for (const size_t offset : FindOpcodeOffsets(words, spv::Op::OpMemberDecorate)) {
        const u32 word_count = words[offset] >> 16;
        if (word_count >= 5 && words[offset + 1] == struct_type &&
            words[offset + 2] == tail_member &&
            words[offset + 3] == static_cast<u32>(spv::DecorationOffset)) {
            layout.member_offset = words[offset + 4];
            break;
        }
    }
    return layout;
}

u32 CountOpcode(std::span<const u32> words, spv::Op opcode) {
    u32 count{};
    for (size_t offset = 5; offset < words.size();) {
        const u32 instruction = words[offset];
        const u32 word_count = instruction >> 16;
        EXPECT_NE(word_count, 0u);
        if (word_count == 0) {
            break;
        }
        count += (instruction & 0xFFFFu) == static_cast<u32>(opcode);
        offset += word_count;
    }
    return count;
}

std::vector<size_t> FindOpcodeOffsets(std::span<const u32> words, spv::Op opcode) {
    std::vector<size_t> offsets;
    for (size_t offset = 5; offset < words.size();) {
        const u32 instruction = words[offset];
        const u32 word_count = instruction >> 16;
        EXPECT_NE(word_count, 0u);
        if (word_count == 0) {
            break;
        }
        if ((instruction & 0xFFFFu) == static_cast<u32>(opcode)) {
            offsets.push_back(offset);
        }
        offset += word_count;
    }
    return offsets;
}

TEST(PpUpstreamSampleTeeSpirv, EmitsOneGuardedExistingResultWithoutAnotherImageOperation) {
    const auto spirv = EmitPpUpstreamSampleTeeFixture();
    ASSERT_GT(spirv.size(), 5u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpSelectionMerge), 2u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpBranchConditional), 2u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpAtomicIAdd), 1u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpBitcast), 1u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpStore), 4u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageSampleImplicitLod), 0u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageSampleExplicitLod), 0u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageFetch), 0u);
}

TEST(PpUpstreamSampleTeeSpirv, CountsEveryWriterBeforeOnlyTheWinnerStoresTheValue) {
    const auto spirv = EmitPpUpstreamSampleTeeFixture();
    const auto atomics = FindOpcodeOffsets(spirv, spv::Op::OpAtomicIAdd);
    const auto stores = FindOpcodeOffsets(spirv, spv::Op::OpStore);
    ASSERT_EQ(atomics.size(), 1u);
    EXPECT_EQ(std::ranges::count_if(stores, [&](size_t offset) { return offset > atomics[0]; }),
              4u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpBranchConditional), 3u);
}

TEST(PpUpstreamSampleTeeSpirv, RejectsUnalignedOrOutOfRangeSlotsBeforeBufferAccess) {
    const auto spirv = EmitPpUpstreamSampleTeeFixture();
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpUMod), 1u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpULessThanEqual), 4u);
}

TEST(PpUpstreamSampleTeeSpirv, KeepsTheLegacyPushBlockWhenInstrumentationIsDisabled) {
    const auto ordinary = InspectPushBlockTail(EmitPpUpstreamSampleTeeFixture(false));
    EXPECT_EQ(ordinary.components, 2u);
    EXPECT_EQ(ordinary.member_offset, 112u);
    EXPECT_EQ(ordinary.component_width, 32u);
    EXPECT_EQ(ordinary.component_signedness, 0u);

    const auto instrumented = InspectPushBlockTail(EmitPpUpstreamSampleTeeFixture(true));
    EXPECT_EQ(instrumented.components, 4u);
    EXPECT_EQ(instrumented.member_offset, 112u);
    EXPECT_EQ(instrumented.component_width, 32u);
    EXPECT_EQ(instrumented.component_signedness, 0u);
}

} // namespace
