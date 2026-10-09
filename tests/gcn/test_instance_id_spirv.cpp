// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#ifdef _WIN32
#include <process.h>
#endif

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv/unified1/spirv.hpp11"

namespace {
using namespace Shader;

constexpr u32 Marker = 0x1234567U;

std::vector<u32> EmitVertexInput(IR::VectorReg input, u32 first_divisor = 1U,
                                 u32 second_divisor = 1U, DrawUserDataLocations mapping = {},
                                 std::optional<IR::ScalarReg> scalar = std::nullopt,
                                 bool mask_scalar = false) {
    Info info{};
    info.stage = Stage::Vertex;
    info.l_stage = LogicalStage::Vertex;
    Pools pools{};
    IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    auto& node = program.syntax_list.emplace_back();
    node.type = IR::AbstractSyntaxNode::Type::Block;
    node.data.block = block;
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    Profile profile{};
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Vertex);
    runtime.num_input_vgprs = 3U;
    runtime.num_user_data = NUM_USER_DATA_REGS;
    runtime.vs_info.step_rate_0 = first_divisor;
    runtime.vs_info.step_rate_1 = second_divisor;
    runtime.vs_info.draw_user_data = mapping;
    Gcn::Translator translator{info, runtime, profile};
    translator.EmitPrologue(block);
    IR::IREmitter ir{*block};
    IR::U32 value = ir.GetVectorReg<IR::U32>(input);
    if (scalar) {
        IR::U32 base = ir.GetScalarReg<IR::U32>(*scalar);
        if (mask_scalar) {
            base = ir.BitwiseAnd(base, ir.Imm32(0xffffU));
        }
        value = ir.IAdd(value, base);
    }
    ir.Reference(ir.IAdd(value, ir.Imm32(Marker)));
    ir.Epilogue();
    Optimization::SsaRewritePass(program);
    Optimization::ConstantPropagationPass(program.blocks);
    Optimization::DeadCodeEliminationPass(program);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::CollectShaderInfoPass(program, profile);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
}

// Evaluate the emitted scalar input expression, not a duplicate C++ implementation of the
// intended subtraction. This makes an absent subtraction, reversed operands, or division before
// base removal observable. Unrecognized instructions never silently become zero.
std::optional<u32> EvaluateVertexInput(const std::vector<u32>& module, u32 first_instance,
                                       u32 local_instance, u32 vertex_index = 0U,
                                       u32 base_vertex = 0U, u32 draw_index = 0U) {
    std::unordered_map<u32, spv::BuiltIn> builtins;
    std::unordered_map<u32, u32> values;
    std::optional<u32> result;
    for (size_t offset = 5U; offset < module.size();) {
        const u32 words = module[offset] >> 16U;
        if (words == 0U || words > module.size() - offset) {
            return std::nullopt;
        }
        const auto opcode = static_cast<spv::Op>(module[offset] & 0xffffU);
        const u32* const inst = module.data() + offset;
        if (opcode == spv::Op::OpDecorate && words == 4U &&
            inst[2] == static_cast<u32>(spv::Decoration::BuiltIn)) {
            builtins[inst[1]] = static_cast<spv::BuiltIn>(inst[3]);
        } else if (opcode == spv::Op::OpConstant && words == 4U) {
            values[inst[2]] = inst[3];
        } else if (opcode == spv::Op::OpLoad && words == 4U && builtins.contains(inst[3])) {
            switch (builtins.at(inst[3])) {
            case spv::BuiltIn::InstanceIndex:
                values[inst[2]] = first_instance + local_instance;
                break;
            case spv::BuiltIn::BaseInstance:
                values[inst[2]] = first_instance;
                break;
            case spv::BuiltIn::VertexIndex:
                values[inst[2]] = vertex_index;
                break;
            case spv::BuiltIn::BaseVertex:
                values[inst[2]] = base_vertex;
                break;
            case spv::BuiltIn::DrawIndex:
                values[inst[2]] = draw_index;
                break;
            default:
                break;
            }
        } else if (words == 5U && values.contains(inst[3]) && values.contains(inst[4])) {
            const u32 lhs = values.at(inst[3]);
            const u32 rhs = values.at(inst[4]);
            switch (opcode) {
            case spv::Op::OpIAdd:
                values[inst[2]] = lhs + rhs;
                if (rhs == Marker) {
                    result = lhs;
                }
                break;
            case spv::Op::OpISub:
                values[inst[2]] = lhs - rhs;
                break;
            case spv::Op::OpBitwiseAnd:
                values[inst[2]] = lhs & rhs;
                break;
            case spv::Op::OpUDiv:
                if (rhs != 0U) {
                    values[inst[2]] = lhs / rhs;
                }
                break;
            default:
                break;
            }
        }
        offset += words;
    }
    return result;
}

TEST(InstanceIdSpirv, RawGuestInstanceDoesNotIncludeHostFirstInstance) {
    const auto module = EmitVertexInput(IR::VectorReg::V3);
    for (u32 base : std::array{7U, 131374U, 0xfffffff0U}) {
        for (u32 local : std::array{0U, 1U, 5U}) {
            SCOPED_TRACE(base);
            SCOPED_TRACE(local);
            EXPECT_EQ(EvaluateVertexInput(module, base, local), std::optional<u32>{local});
        }
    }
}

TEST(InstanceIdSpirv, StepRateDivisionUsesTheRelativeInstanceBeforeAnyFetchBase) {
    for (const auto [reg, divisor] :
         std::array{std::pair{IR::VectorReg::V1, 2U}, std::pair{IR::VectorReg::V2, 3U}}) {
        const auto module = EmitVertexInput(reg, 2U, 3U);
        for (u32 local : std::array{0U, 1U, 5U, 11U}) {
            EXPECT_EQ(EvaluateVertexInput(module, 131375U, local),
                      std::optional<u32>{local / divisor});
        }
    }
}

TEST(InstanceIdSpirv, ZeroBaseAndVertexInputBehaviorAreUnchanged) {
    const auto instance_module = EmitVertexInput(IR::VectorReg::V3);
    const auto vertex_module = EmitVertexInput(IR::VectorReg::V0);
    for (u32 value : std::array{0U, 1U, 37U}) {
        EXPECT_EQ(EvaluateVertexInput(instance_module, 0U, value), std::optional<u32>{value});
        EXPECT_EQ(EvaluateVertexInput(vertex_module, 131374U, 0U, value),
                  std::optional<u32>{value});
    }
}

TEST(InstanceIdSpirv, MappedBaseVertexKeepsItsGuestVertexInputRelative) {
    const auto mapped = EmitVertexInput(IR::VectorReg::V0, 1U, 1U, {.base_vertex = 2U});
    const auto plain = EmitVertexInput(IR::VectorReg::V0);
    for (u32 base : std::array{17U, u32(-13)}) {
        for (u32 local : std::array{20U, 100U}) {
            EXPECT_EQ(EvaluateVertexInput(mapped, 0U, 0U, base + local, base),
                      std::optional<u32>{local});
            // Preserve the legacy direct/fetch convention outside the newly supported CP mode.
            EXPECT_EQ(EvaluateVertexInput(plain, 0U, 0U, base + local, base),
                      std::optional<u32>{base + local});
        }
    }
}

void ExpectValidVertexModule(const std::vector<u32>& module) {
#ifdef SHADPS4_SPIRV_VAL_PATH
    static std::atomic<u32> serial{};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
                      ("shadps4_instance_input_" + std::to_string(stamp) + "_" +
                       std::to_string(serial++) + ".spv");
    ASSERT_FALSE(std::filesystem::exists(path));
    {
        std::ofstream stream{path, std::ios::binary};
        ASSERT_TRUE(stream.good());
        stream.write(reinterpret_cast<const char*>(module.data()),
                     static_cast<std::streamsize>(module.size() * sizeof(u32)));
        ASSERT_TRUE(stream.good());
    }
#ifdef _WIN32
    const int status =
        static_cast<int>(_spawnl(_P_WAIT, SHADPS4_SPIRV_VAL_PATH, SHADPS4_SPIRV_VAL_PATH,
                                 "--target-env", "vulkan1.3", path.string().c_str(), nullptr));
#else
    const auto command = std::string{"\""} + SHADPS4_SPIRV_VAL_PATH +
                         "\" --target-env vulkan1.3 \"" + path.string() + "\"";
    const int status = std::system(command.c_str());
#endif
    EXPECT_EQ(status, 0);
    std::error_code error;
    std::filesystem::remove(path, error);
#else
    GTEST_SKIP() << "spirv-val is unavailable; relative-instance codegen tests still run";
#endif
}

TEST(InstanceIdSpirv, PackedCpStartInstanceAndRelativeInstanceRemainIndependent) {
    const auto packed =
        EmitVertexInput(IR::VectorReg::V3, 1U, 1U, {.base_instance = 2U}, IR::ScalarReg::S2, true);
    const auto complete =
        EmitVertexInput(IR::VectorReg::V3, 1U, 1U, {.base_instance = 2U}, IR::ScalarReg::S2);
    for (u32 local : std::array{0U, 1U, 5U}) {
        EXPECT_EQ(EvaluateVertexInput(packed, 131374U, local), std::optional<u32>{302U + local});
        EXPECT_EQ(EvaluateVertexInput(complete, 131374U, local),
                  std::optional<u32>{131374U + local});
    }
    ExpectValidVertexModule(packed);
}

TEST(InstanceIdSpirv, MappedVertexBaseAndDrawIndexUseTheirOwnBuiltins) {
    const auto vertex =
        EmitVertexInput(IR::VectorReg::V0, 1U, 1U, {.base_vertex = 2U}, IR::ScalarReg::S2);
    const auto draw =
        EmitVertexInput(IR::VectorReg::V3, 1U, 1U, {.draw_index = 2U}, IR::ScalarReg::S2);
    for (u32 base : std::array{17U, u32(-13)}) {
        EXPECT_EQ(EvaluateVertexInput(vertex, 0U, 0U, base + 100U, base),
                  std::optional<u32>{base + 100U});
    }
    for (u32 draw_index : std::array{0U, 3U, 17U}) {
        EXPECT_EQ(EvaluateVertexInput(draw, 131374U, 0U, 0U, 0U, draw_index),
                  std::optional<u32>{draw_index});
    }
    ExpectValidVertexModule(vertex);
    ExpectValidVertexModule(draw);
}

TEST(InstanceIdSpirv, RelativeInstanceInputIsValidVulkanSpirv) {
    ExpectValidVertexModule(EmitVertexInput(IR::VectorReg::V1, 3U));
}
} // namespace
