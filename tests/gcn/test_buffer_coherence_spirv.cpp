// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <map>
#include <vector>

#include <gtest/gtest.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/recompiler.h"
#include "spirv/unified1/spirv.hpp"

namespace {
using namespace Shader;

BufferResource MakeBuffer(u32 mtype, bool written, BufferType type = BufferType::Guest) {
    AmdGpu::Buffer sharp = AmdGpu::Buffer::Placeholder(256);
    // Encode the actual guest field, independently of the host bitfield declaration.
    auto words = std::bit_cast<std::array<u32, 4>>(sharp);
    words[3] |= mtype << 27U;
    return {.sharp_idx = 0,
            .used_types =
                IR::Type::U8 | IR::Type::U16 | IR::Type::U32 | IR::Type::U64 | IR::Type::F32,
            .inline_cbuf = std::bit_cast<AmdGpu::Buffer>(words),
            .buffer_type = type,
            .is_written = written};
}

std::vector<u32> EmitBuffers(const BufferResourceList& resources) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.buffers = resources;
    IR::Program program{info};
    Pools pools{};
    auto* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());
    IR::IREmitter ir{*block};
    ir.Prologue();
    ir.Epilogue();
    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Compute);
    runtime.cs_info.workgroup_size = {1, 1, 1};
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
}

struct BufferVariable {
    u32 binding{};
    bool has_binding{};
    bool storage{};
    bool coherent{};
    bool non_writable{};
};

std::map<u32, BufferVariable> Variables(const std::vector<u32>& spirv) {
    std::map<u32, BufferVariable> variables;
    for (size_t offset = 5; offset < spirv.size();) {
        const auto count = spirv[offset] >> 16U;
        if (count == 0 || count > spirv.size() - offset) {
            ADD_FAILURE() << "Malformed emitted SPIR-V";
            break;
        }
        const auto op = static_cast<spv::Op>(spirv[offset] & 0xffffU);
        if (op == spv::Op::OpVariable && count >= 4) {
            variables[spirv[offset + 2]].storage =
                spirv[offset + 3] == static_cast<u32>(spv::StorageClassStorageBuffer);
        } else if (op == spv::Op::OpDecorate && count >= 3) {
            auto& variable = variables[spirv[offset + 1]];
            const auto decoration = static_cast<spv::Decoration>(spirv[offset + 2]);
            if (decoration == spv::DecorationCoherent) {
                variable.coherent = true;
            } else if (decoration == spv::DecorationNonWritable) {
                variable.non_writable = true;
            } else if (decoration == spv::DecorationBinding && count == 4) {
                variable.binding = spirv[offset + 3];
                variable.has_binding = true;
            }
        }
        offset += count;
    }
    std::erase_if(variables, [](const auto& entry) { return !entry.second.has_binding; });
    return variables;
}

void ExpectAliases(const BufferResource& resource, bool coherent, bool non_writable) {
    const auto variables = Variables(EmitBuffers({resource}));
    ASSERT_EQ(variables.size(), 5U);
    for (const auto& [id, variable] : variables) {
        SCOPED_TRACE(id);
        EXPECT_TRUE(variable.storage);
        EXPECT_EQ(variable.binding, 0U);
        EXPECT_EQ(variable.coherent, coherent);
        EXPECT_EQ(variable.non_writable, non_writable);
    }
}

TEST(BufferCoherenceSpirv, WrittenGuestClassThreeDecoratesEveryTypedAlias) {
    ExpectAliases(MakeBuffer(3, true), true, false);
}

TEST(BufferCoherenceSpirv, LoweredSharedMemoryDoesNotDependOnGuestMtype) {
    for (u32 mtype = 0; mtype < 8; ++mtype) {
        SCOPED_TRACE(mtype);
        ExpectAliases(MakeBuffer(mtype, true, BufferType::SharedMemory), true, false);
    }
}

TEST(BufferCoherenceSpirv, OtherWrittenMtypesDoNotBecomeCoherent) {
    for (const u32 mtype : {0U, 1U, 2U, 4U, 5U, 6U, 7U}) {
        SCOPED_TRACE(mtype);
        ExpectAliases(MakeBuffer(mtype, true), false, false);
    }
}

TEST(BufferCoherenceSpirv, ReadOnlyStorageKeepsNonWritableAndNeverBecomesCoherent) {
    for (const auto type : {BufferType::Guest, BufferType::SharedMemory}) {
        for (u32 mtype = 0; mtype < 8; ++mtype) {
            SCOPED_TRACE(mtype);
            ExpectAliases(MakeBuffer(mtype, false, type), false, true);
        }
    }
}

TEST(BufferCoherenceSpirv, SharedBlockTypesDoNotLeakCoherenceAcrossBindings) {
    const auto variables = Variables(EmitBuffers({MakeBuffer(3, true), MakeBuffer(0, true)}));
    ASSERT_EQ(variables.size(), 10U);
    std::array<u32, 2> alias_counts{};
    for (const auto& [id, variable] : variables) {
        SCOPED_TRACE(id);
        ASSERT_LT(variable.binding, alias_counts.size());
        ++alias_counts[variable.binding];
        EXPECT_TRUE(variable.storage);
        EXPECT_EQ(variable.coherent, variable.binding == 0);
        EXPECT_FALSE(variable.non_writable);
    }
    EXPECT_EQ(alias_counts, (std::array<u32, 2>{5, 5}));
}
} // namespace
