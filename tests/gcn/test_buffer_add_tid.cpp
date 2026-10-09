// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#endif

#include <gtest/gtest.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "spirv/unified1/spirv.hpp"

namespace Shader::Optimization {
namespace {

AmdGpu::Buffer MakeAddTidBuffer(bool add_tid_enable) {
    AmdGpu::Buffer buffer{};
    buffer.base_address = 1;
    buffer.stride = 4;
    buffer.num_records = 64;
    buffer.data_format = static_cast<u32>(AmdGpu::DataFormat::Format8_8_8_8);
    buffer.add_tid_enable = add_tid_enable;
    return buffer;
}

bool HasBuiltIn(const std::vector<u32>& spirv, spv::BuiltIn builtin) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const u32 words = instruction >> 16;
        if (words == 0 || offset + words > spirv.size()) {
            return false;
        }
        if (static_cast<spv::Op>(instruction & 0xffffU) == spv::Op::OpDecorate && words >= 4 &&
            spirv[offset + 2] == static_cast<u32>(spv::DecorationBuiltIn) &&
            spirv[offset + 3] == static_cast<u32>(builtin)) {
            return true;
        }
        offset += words;
    }
    return false;
}

bool BuiltInHasDecoration(const std::vector<u32>& spirv, spv::BuiltIn builtin,
                          spv::Decoration decoration) {
    std::vector<u32> builtin_ids;
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const u32 words = instruction >> 16;
        if (words == 0 || offset + words > spirv.size()) {
            return false;
        }
        if (static_cast<spv::Op>(instruction & 0xffffU) == spv::Op::OpDecorate && words >= 4 &&
            spirv[offset + 2] == static_cast<u32>(spv::DecorationBuiltIn) &&
            spirv[offset + 3] == static_cast<u32>(builtin)) {
            builtin_ids.push_back(spirv[offset + 1]);
        }
        offset += words;
    }

    for (size_t offset = 5; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const u32 words = instruction >> 16;
        if (words == 0 || offset + words > spirv.size()) {
            return false;
        }
        if (static_cast<spv::Op>(instruction & 0xffffU) == spv::Op::OpDecorate && words >= 3 &&
            spirv[offset + 2] == static_cast<u32>(decoration) &&
            std::ranges::find(builtin_ids, spirv[offset + 1]) != builtin_ids.end()) {
            return true;
        }
        offset += words;
    }
    return false;
}

std::vector<u32> EmitAddTidShader(Stage stage, LogicalStage logical_stage) {
    Info info{};
    info.stage = stage;
    info.l_stage = logical_stage;

    Pools pools{};
    IR::Program program{info};
    Gcn::Block cfg_block{};
    IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    block->cfg_block = &cfg_block;
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);

    IR::IREmitter ir{*block};
    ir.Prologue();
    const auto raw = std::bit_cast<std::array<u32, 4>>(MakeAddTidBuffer(true));
    const IR::Value handle = ir.CompositeConstruct(ir.Imm32(raw[0]), ir.Imm32(raw[1]),
                                                   ir.Imm32(raw[2]), ir.Imm32(raw[3]));
    const IR::Value address = ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U));
    ir.Reference(ir.LoadBufferU32(1, handle, address, {}));
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;
    ResourceTrackingPass(program, profile);
    EXPECT_EQ(std::ranges::count_if(
                  block->Instructions(),
                  [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::LaneId; }),
              1U);
    EXPECT_EQ(std::ranges::count_if(block->Instructions(),
                                    [](const IR::Inst& inst) {
                                        return inst.GetOpcode() == IR::Opcode::IMul32 &&
                                               inst.Arg(1).IsImmediate() && inst.Arg(1).U32() == 4U;
                                    }),
              1U);

    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());
    CollectShaderInfoPass(program, profile);

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(stage);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

#ifdef SHADPS4_SPIRV_VAL_PATH
int RunSpirvValidator(const std::filesystem::path& module) {
#ifdef _WIN32
    const std::string module_path = module.string();
    return static_cast<int>(_spawnl(_P_WAIT, SHADPS4_SPIRV_VAL_PATH, SHADPS4_SPIRV_VAL_PATH,
                                    "--target-env", "vulkan1.3", module_path.c_str(), nullptr));
#else
    const std::string command = std::string{"\""} + SHADPS4_SPIRV_VAL_PATH +
                                "\" --target-env vulkan1.3 \"" + module.string() + "\"";
    return std::system(command.c_str());
#endif
}
#endif

TEST(BufferAddTid, FragmentResourceTrackingUsesTheLaneIdAndEmitsSubgroupBuiltin) {
    const auto spirv = EmitAddTidShader(Stage::Fragment, LogicalStage::Fragment);

    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
    EXPECT_TRUE(BuiltInHasDecoration(spirv, spv::BuiltInSubgroupLocalInvocationId,
                                     spv::DecorationFlat));

#ifdef SHADPS4_SPIRV_VAL_PATH
    const auto output =
        std::filesystem::temp_directory_path() / "shadps4-fragment-add-tid-test.spv";
    {
        std::ofstream file{output, std::ios::binary | std::ios::trunc};
        ASSERT_TRUE(file.good());
        file.write(reinterpret_cast<const char*>(spirv.data()),
                   static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
    }
    EXPECT_EQ(RunSpirvValidator(output), 0);
    std::error_code remove_error;
    std::filesystem::remove(output, remove_error);
#endif
}

TEST(BufferAddTid, VertexSubgroupLaneInputHasNoInterpolationDecoration) {
    const auto spirv = EmitAddTidShader(Stage::Vertex, LogicalStage::Vertex);

    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
    EXPECT_FALSE(BuiltInHasDecoration(spirv, spv::BuiltInSubgroupLocalInvocationId,
                                      spv::DecorationFlat));

#ifdef SHADPS4_SPIRV_VAL_PATH
    const auto output =
        std::filesystem::temp_directory_path() / "shadps4-vertex-add-tid-test.spv";
    {
        std::ofstream file{output, std::ios::binary | std::ios::trunc};
        ASSERT_TRUE(file.good());
        file.write(reinterpret_cast<const char*>(spirv.data()),
                   static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
    }
    EXPECT_EQ(RunSpirvValidator(output), 0);
    std::error_code remove_error;
    std::filesystem::remove(output, remove_error);
#endif
}

TEST(BufferAddTid, DescriptorBitParticipatesInBufferSpecializationIdentity) {
    Info info{};
    info.buffers.push_back(BufferResource{
        .sharp_idx = std::numeric_limits<u32>::max(),
        .used_types = IR::Type::U32,
        .inline_cbuf = MakeAddTidBuffer(true),
        .buffer_type = BufferType::Guest,
    });

    const RuntimeInfo runtime_info{};
    const Profile profile{};
    const Backend::Bindings bindings{};
    const StageSpecialization enabled{info, runtime_info, profile, bindings};

    info.buffers.front().inline_cbuf.add_tid_enable = false;
    const StageSpecialization disabled{info, runtime_info, profile, bindings};

    EXPECT_FALSE(enabled == disabled);
    EXPECT_FALSE(disabled == enabled);

    info.buffers.front().inline_cbuf = MakeAddTidBuffer(true);
    info.buffers.front().inline_cbuf.data_format = 1;
    const StageSpecialization first_data_format{info, runtime_info, profile, bindings};
    info.buffers.front().inline_cbuf.data_format = 2;
    const StageSpecialization second_data_format{info, runtime_info, profile, bindings};

    // Liverpool is Sea Islands/GFX7: DATA_FORMAT remains a format field with ADD_TID and does
    // not become high stride bits as it does on GCN3/GFX8. Raw-buffer code ignores it.
    EXPECT_TRUE(first_data_format == second_data_format);
    EXPECT_TRUE(second_data_format == first_data_format);
}

TEST(BufferSpecialization, BoundAndUnboundDescriptorsHaveOrderIndependentIdentity) {
    Info info{};
    info.buffers.push_back(BufferResource{
        .sharp_idx = 0,
        .used_types = IR::Type::U32,
        .inline_cbuf = {},
        .buffer_type = BufferType::Guest,
    });
    const auto bound_words = std::bit_cast<std::array<u32, 4>>(MakeAddTidBuffer(false));
    info.flattened_ud_buf.assign(bound_words.begin(), bound_words.end());

    const RuntimeInfo runtime_info{};
    const Profile profile{};
    const Backend::Bindings bindings{};
    const StageSpecialization bound{info, runtime_info, profile, bindings};

    std::ranges::fill(info.flattened_ud_buf, 0U);
    const StageSpecialization unbound{info, runtime_info, profile, bindings};

    EXPECT_FALSE(bound == unbound);
    EXPECT_FALSE(unbound == bound);
}

TEST(BufferSpecialization, BoundnessAloneDoesNotCreateASecondModulePermutation) {
    Info info{};
    info.buffers.push_back(BufferResource{
        .sharp_idx = 0,
        .used_types = IR::Type::U32,
        .inline_cbuf = {},
        .buffer_type = BufferType::Guest,
    });
    AmdGpu::Buffer live_default_shape{};
    live_default_shape.base_address = 1;
    live_default_shape.num_records = 1;
    const auto bound_words = std::bit_cast<std::array<u32, 4>>(live_default_shape);
    info.flattened_ud_buf.assign(bound_words.begin(), bound_words.end());

    const RuntimeInfo runtime_info{};
    const Profile profile{};
    const Backend::Bindings bindings{};
    const StageSpecialization bound{info, runtime_info, profile, bindings};
    std::ranges::fill(info.flattened_ud_buf, 0U);
    const StageSpecialization unbound{info, runtime_info, profile, bindings};

    ASSERT_NE(bound.bitset, unbound.bitset);
    EXPECT_TRUE(bound == unbound);
    EXPECT_TRUE(unbound == bound);
}

TEST(BufferAddTid, DescriptorKeepsSeaIslandsStrideWhenAddTidIsEnabled) {
    AmdGpu::Buffer buffer{};
    buffer.stride = 0x1234;
    buffer.data_format = 0xA;
    buffer.add_tid_enable = 1;

    EXPECT_EQ(buffer.GetStride(), 0x1234U);
}

} // namespace
} // namespace Shader::Optimization
