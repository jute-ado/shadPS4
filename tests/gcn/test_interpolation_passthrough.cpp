// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
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
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv/unified1/AMD_shader_explicit_vertex_parameter.h"
#include "spirv/unified1/spirv.hpp"

namespace {
using namespace Shader;

struct InterpolationMoveFixture {
    Info info{};
    RuntimeInfo runtime{};
    Profile profile{};
    Pools pools{};
    IR::Program program{info};
    IR::Block* block{};

    InterpolationMoveFixture(u32 selector, bool offset_bit5, bool flat, bool amd = false,
                             bool barycentric = true,
                             Info::Interpolation initial_interpolation = {Qualifier::None,
                                                                          Qualifier::None},
                             bool manual_interpolation = false) {
        info.stage = Stage::Fragment;
        info.l_stage = LogicalStage::Fragment;
        runtime.Initialize(Stage::Fragment);
        runtime.fs_info.num_inputs = 1;
        runtime.fs_info.inputs[0] = {
            .param_index = 0,
            .is_default = offset_bit5,
            .is_flat = flat,
            .default_value = 2,
        };
        info.fs_interpolation[0] = initial_interpolation;
        profile.supported_spirv = 0x00010600;
        profile.subgroup_size = 32;
        profile.supports_amd_shader_explicit_vertex_parameter = amd;
        profile.supports_fragment_shader_barycentric = !amd && barycentric;
        profile.needs_manual_interpolation = manual_interpolation;
        block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        Gcn::Translator translator{info, runtime, profile};
        translator.EmitPrologue(block);
        Gcn::GcnInst inst{};
        inst.opcode = Gcn::Opcode::V_INTERP_MOV_F32;
        inst.src[0].code = selector;
        inst.dst[0].field = Gcn::OperandField::VectorGPR;
        inst.dst[0].code = 0;
        inst.control.vintrp.attr = 0;
        inst.control.vintrp.chan = 0;
        translator.V_INTERP_MOV_F32(inst);
    }

    IR::Value Result() const {
        for (auto it = block->rbegin(); it != block->rend(); ++it) {
            if (it->GetOpcode() == IR::Opcode::SetVectorRegister) {
                return it->Arg(1).Resolve();
            }
        }
        ADD_FAILURE() << "V_INTERP_MOV did not write its destination";
        return IR::Value{0U};
    }

    size_t Count(IR::Opcode opcode) const {
        return std::ranges::count_if(
            *block, [=](const IR::Inst& inst) { return inst.GetOpcode() == opcode; });
    }
};

u32 EvaluateBits(const IR::Value& value, const std::array<u32, 3>& vertex_bits, u32 depth = 0) {
    if (depth > 16U) {
        ADD_FAILURE() << "Unexpected interpolation expression depth";
        return 0;
    }
    if (value.IsImmediate()) {
        return value.Type() == IR::Type::F32 ? std::bit_cast<u32>(value.F32()) : value.U32();
    }
    const auto* inst = value.TryInstRecursive();
    if (inst == nullptr) {
        ADD_FAILURE() << "Missing interpolation expression";
        return 0;
    }
    switch (inst->GetOpcode()) {
    case IR::Opcode::GetAttribute:
        EXPECT_EQ(inst->Arg(0).Attribute(), IR::Attribute::Param0);
        EXPECT_EQ(inst->Arg(1).U32(), 0U);
        EXPECT_LT(inst->Arg(2).U32(), vertex_bits.size());
        return vertex_bits.at(inst->Arg(2).U32());
    case IR::Opcode::BitCastU32F32:
    case IR::Opcode::BitCastF32U32:
        return EvaluateBits(inst->Arg(0), vertex_bits, depth + 1U);
    case IR::Opcode::FPSub32: {
        const auto lhs = std::bit_cast<float>(EvaluateBits(inst->Arg(0), vertex_bits, depth + 1U));
        const auto rhs = std::bit_cast<float>(EvaluateBits(inst->Arg(1), vertex_bits, depth + 1U));
        return std::bit_cast<u32>(lhs - rhs);
    }
    default:
        ADD_FAILURE() << "Unexpected interpolation opcode " << static_cast<u32>(inst->GetOpcode());
        return 0;
    }
}

TEST(InterpolationPassthrough, PackedHalfWordsAreReturnedBitExactlyForEverySelector) {
    // Two packed half components per raw vertex parameter. Subtracting these as an F32
    // before the guest unpacks either half corrupts the custom interpolation inputs.
    constexpr std::array<u32, 3> packed{0x3C003800U, 0x4000BC00U, 0x38004200U};
    constexpr std::array<u32, 3> selected_vertex{1U, 2U, 0U};
    for (bool amd : {false, true}) {
        for (u32 selector = 0; selector < 3; ++selector) {
            SCOPED_TRACE(selector);
            SCOPED_TRACE(amd);
            InterpolationMoveFixture fixture{selector, true, true, amd};
            EXPECT_EQ(EvaluateBits(fixture.Result(), packed), packed[selected_vertex[selector]]);
            EXPECT_EQ(fixture.info.fs_interpolation[0].primary, Qualifier::PerVertex);
        }
    }
}

TEST(InterpolationPassthrough, RawVertexModeDoesNotIntroduceFloatingPointArithmetic) {
    for (u32 selector = 0; selector < 3; ++selector) {
        InterpolationMoveFixture fixture{selector, true, true};
        EXPECT_EQ(fixture.Count(IR::Opcode::FPSub32), 0U);
        EXPECT_EQ(fixture.Count(IR::Opcode::GetAttribute), 1U);
    }
}

TEST(InterpolationPassthrough, ManualInterpolationProfilePreservesPackedPassthrough) {
    // NVIDIA's barycentric profile also enables the manual-interpolation flag. Keep this
    // combination covered because the raw parameter-cache mode must not depend on the
    // coefficient interpolation implementation used by P1/P2 instructions.
    constexpr std::array<u32, 3> packed{0x3C003800U, 0x4000BC00U, 0x38004200U};
    constexpr std::array<u32, 3> selected_vertex{1U, 2U, 0U};
    for (u32 selector = 0; selector < 3; ++selector) {
        SCOPED_TRACE(selector);
        InterpolationMoveFixture fixture{
            selector, true, true, false, true, {Qualifier::None, Qualifier::None}, true};
        EXPECT_EQ(EvaluateBits(fixture.Result(), packed), packed[selected_vertex[selector]]);
        EXPECT_EQ(fixture.Count(IR::Opcode::FPSub32), 0U);
    }
}

TEST(InterpolationPassthrough, OrdinaryParameterCoefficientsRetainTheirDeltas) {
    const std::array<u32, 3> vertices{std::bit_cast<u32>(2.0f), std::bit_cast<u32>(7.0f),
                                      std::bit_cast<u32>(-3.0f)};
    constexpr std::array<float, 3> expected{5.0f, -5.0f, 2.0f};
    for (u32 selector = 0; selector < 3; ++selector) {
        InterpolationMoveFixture fixture{selector, false, true};
        EXPECT_FLOAT_EQ(std::bit_cast<float>(EvaluateBits(fixture.Result(), vertices)),
                        expected[selector]);
        EXPECT_EQ(fixture.Count(IR::Opcode::FPSub32), selector == 2 ? 0U : 1U);
    }
}

TEST(InterpolationPassthrough, DefaultConstantsRemainDistinctFromPassthrough) {
    constexpr std::array<u32, 3> unused{0, 0, 0};
    for (u32 selector = 0; selector < 3; ++selector) {
        InterpolationMoveFixture fixture{selector, true, false};
        EXPECT_FLOAT_EQ(std::bit_cast<float>(EvaluateBits(fixture.Result(), unused)),
                        selector == 2 ? 1.0f : 0.0f);
        EXPECT_EQ(fixture.Count(IR::Opcode::GetAttribute), 0U);
        EXPECT_EQ(fixture.info.fs_interpolation[0].primary, Qualifier::Flat);
    }
    InterpolationMoveFixture passthrough{2, true, true};
    EXPECT_FALSE(passthrough.runtime.fs_info.inputs[0].IsDefault());
}

TEST(InterpolationPassthrough, UnsupportedHostRetainsFlatFallback) {
    // When neither AMD explicit-vertex interpolation nor KHR barycentrics is
    // available, preserve the pre-existing flat input fallback. This is a
    // deliberate fail-soft path: it must not synthesize arithmetic or claim
    // per-vertex interpolation that the target cannot express.
    constexpr std::array<u32, 3> packed{0x3C003800U, 0x4000BC00U, 0x38004200U};
    for (u32 selector = 0; selector < 3; ++selector) {
        SCOPED_TRACE(selector);
        InterpolationMoveFixture fixture{selector, true, true, false, false};
        EXPECT_EQ(fixture.info.fs_interpolation[0].primary, Qualifier::Flat);
        EXPECT_EQ(fixture.Count(IR::Opcode::FPSub32), 0U);
        EXPECT_EQ(fixture.Count(IR::Opcode::GetAttribute), 1U);
        EXPECT_EQ(EvaluateBits(fixture.Result(), packed), packed[0]);
    }
}

TEST(InterpolationPassthrough, UnsupportedHostClearsStaleAuxiliaryQualifier) {
    // A previous instruction can leave centroid/sample state on the same input. The flat
    // fallback must replace the complete qualifier, or SPIR-V emission can attach an auxiliary
    // interpolation decoration to a flat input.
    InterpolationMoveFixture fixture{2,     false, true,
                                     false, false, {Qualifier::Smooth, Qualifier::Centroid}};
    EXPECT_EQ(fixture.info.fs_interpolation[0].primary, Qualifier::Flat);
    EXPECT_EQ(fixture.info.fs_interpolation[0].auxiliary, Qualifier::None);
}

std::vector<u32> EmitInterpolationMove(u32 selector, bool passthrough, bool amd,
                                       bool barycentric = true) {
    InterpolationMoveFixture fixture{selector, passthrough, true, amd, barycentric};
    auto& node = fixture.program.syntax_list.emplace_back();
    node.type = IR::AbstractSyntaxNode::Type::Block;
    node.data.block = fixture.block;
    fixture.program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    fixture.program.post_order_blocks = IR::PostOrder(fixture.program.syntax_list.front());
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.GetVectorReg<IR::U32>(IR::VectorReg::V0));
    ir.Epilogue();
    Optimization::SsaRewritePass(fixture.program);
    Optimization::ConstantPropagationPass(fixture.program.blocks);
    Optimization::DeadCodeEliminationPass(fixture.program);
    Optimization::IdentityRemovalPass(fixture.program.blocks);
    Optimization::CollectShaderInfoPass(fixture.program, fixture.profile);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.runtime, fixture.program, bindings);
}

size_t CountSpirvOperation(const std::vector<u32>& module, spv::Op opcode) {
    size_t count{};
    for (size_t offset = 5; offset < module.size();) {
        const u32 width = module[offset] >> 16U;
        if (width == 0 || width > module.size() - offset) {
            ADD_FAILURE() << "Malformed SPIR-V instruction";
            return count;
        }
        count += (module[offset] & 0xFFFFU) == static_cast<u32>(opcode);
        offset += width;
    }
    return count;
}

std::optional<u32> FindDecoratedTarget(const std::vector<u32>& module, spv::Decoration decoration) {
    std::optional<u32> target;
    for (size_t offset = 5; offset < module.size();) {
        const u32 width = module[offset] >> 16U;
        if (width == 0 || width > module.size() - offset) {
            ADD_FAILURE() << "Malformed SPIR-V instruction";
            return std::nullopt;
        }
        if ((module[offset] & 0xFFFFU) == static_cast<u32>(spv::OpDecorate) && width >= 3 &&
            module[offset + 2] == static_cast<u32>(decoration)) {
            const u32 candidate = module[offset + 1];
            if (target && *target != candidate) {
                ADD_FAILURE() << "Interpolation decoration appears on multiple targets";
                return std::nullopt;
            }
            target = candidate;
        }
        offset += width;
    }
    return target;
}

std::unordered_map<u32, u32> ScalarConstants(const std::vector<u32>& module) {
    std::unordered_map<u32, u32> constants;
    for (size_t offset = 5; offset < module.size();) {
        const u32 width = module[offset] >> 16U;
        if (width == 0 || width > module.size() - offset) {
            ADD_FAILURE() << "Malformed SPIR-V instruction";
            return {};
        }
        if ((module[offset] & 0xFFFFU) == static_cast<u32>(spv::OpConstant) && width >= 4) {
            constants.emplace(module[offset + 2], module[offset + 3]);
        }
        offset += width;
    }
    return constants;
}

bool HasAccessChainIndex(const std::vector<u32>& module, u32 base, u32 expected_index) {
    const auto constants = ScalarConstants(module);
    for (size_t offset = 5; offset < module.size();) {
        const u32 width = module[offset] >> 16U;
        if (width == 0 || width > module.size() - offset) {
            ADD_FAILURE() << "Malformed SPIR-V instruction";
            return false;
        }
        if ((module[offset] & 0xFFFFU) == static_cast<u32>(spv::OpAccessChain) && width >= 5 &&
            module[offset + 3] == base) {
            for (u32 operand = 4; operand < width; ++operand) {
                if (const auto it = constants.find(module[offset + operand]);
                    it != constants.end() && it->second == expected_index) {
                    return true;
                }
            }
        }
        offset += width;
    }
    return false;
}

bool IdUsedOutsideInstruction(const std::vector<u32>& module, u32 id, size_t definition_offset) {
    for (size_t offset = 5; offset < module.size();) {
        const u32 width = module[offset] >> 16U;
        if (width == 0 || width > module.size() - offset) {
            ADD_FAILURE() << "Malformed SPIR-V instruction";
            return false;
        }
        const auto opcode = static_cast<spv::Op>(module[offset] & 0xFFFFU);
        if (offset != definition_offset && opcode != spv::OpName && opcode != spv::OpDecorate &&
            opcode != spv::OpMemberDecorate && opcode != spv::OpLine) {
            for (u32 operand = 1; operand < width; ++operand) {
                if (module[offset + operand] == id) {
                    return true;
                }
            }
        }
        offset += width;
    }
    return false;
}

std::vector<u32> UsedAmdInterpolationVertexIndices(const std::vector<u32>& module) {
    const auto constants = ScalarConstants(module);
    std::vector<u32> indices;
    for (size_t offset = 5; offset < module.size();) {
        const u32 width = module[offset] >> 16U;
        if (width == 0 || width > module.size() - offset) {
            ADD_FAILURE() << "Malformed SPIR-V instruction";
            return {};
        }
        if ((module[offset] & 0xFFFFU) == static_cast<u32>(spv::OpExtInst) && width >= 7 &&
            module[offset + 4] == AMD_shader_explicit_vertex_parameterInterpolateAtVertexAMD &&
            IdUsedOutsideInstruction(module, module[offset + 2], offset)) {
            if (const auto it = constants.find(module[offset + 6]); it != constants.end()) {
                indices.push_back(it->second);
            }
        }
        offset += width;
    }
    return indices;
}

void ExpectValidFragmentModule(const std::vector<u32>& module) {
#ifdef SHADPS4_SPIRV_VAL_PATH
    static std::atomic<u32> serial{};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path =
        std::filesystem::temp_directory_path() /
        ("shadps4_passthrough_" + std::to_string(stamp) + "_" + std::to_string(serial++) + ".spv");
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
    GTEST_SKIP() << "spirv-val unavailable; the translator bit-preservation tests still run";
#endif
}

TEST(InterpolationPassthroughSpirv, BothExplicitVertexBackendsKeepPackedValuesWithoutSubtracting) {
    for (bool amd : {false, true}) {
        for (u32 selector = 0; selector < 3; ++selector) {
            SCOPED_TRACE(amd);
            SCOPED_TRACE(selector);
            const auto module = EmitInterpolationMove(selector, true, amd);
            EXPECT_EQ(CountSpirvOperation(module, spv::OpFSub), 0U);
            ExpectValidFragmentModule(module);
        }
    }
}

TEST(InterpolationPassthroughSpirv, CoefficientModeStillEmitsItsSubtraction) {
    for (bool amd : {false, true}) {
        const auto module = EmitInterpolationMove(0, false, amd);
        EXPECT_EQ(CountSpirvOperation(module, spv::OpFSub), 1U);
        ExpectValidFragmentModule(module);
    }
}
TEST(InterpolationPassthroughSpirv, KhrInputUsesPerVertexDecorationAndSelectedIndex) {
    for (u32 selector = 0; selector < 3; ++selector) {
        SCOPED_TRACE(selector);
        const auto module = EmitInterpolationMove(selector, true, false);
        const auto target = FindDecoratedTarget(module, spv::DecorationPerVertexKHR);
        ASSERT_TRUE(target.has_value());
        EXPECT_TRUE(HasAccessChainIndex(module, *target, (selector + 1U) % 3U));
        ExpectValidFragmentModule(module);
    }
}

TEST(InterpolationPassthroughSpirv, AmdInputUsesSelectedExplicitVertexInstruction) {
    for (u32 selector = 0; selector < 3; ++selector) {
        SCOPED_TRACE(selector);
        const auto module = EmitInterpolationMove(selector, true, true);
        const auto target = FindDecoratedTarget(module, spv::DecorationExplicitInterpAMD);
        ASSERT_TRUE(target.has_value());
        const auto indices = UsedAmdInterpolationVertexIndices(module);
        ASSERT_EQ(indices.size(), 1U);
        EXPECT_EQ(indices.front(), (selector + 1U) % 3U);
        ExpectValidFragmentModule(module);
    }
}

TEST(InterpolationPassthroughSpirv, UnsupportedHostUsesFlatInputFallback) {
    for (u32 selector = 0; selector < 3; ++selector) {
        SCOPED_TRACE(selector);
        const auto module = EmitInterpolationMove(selector, true, false, false);
        EXPECT_EQ(CountSpirvOperation(module, spv::OpFSub), 0U);
        EXPECT_TRUE(FindDecoratedTarget(module, spv::DecorationFlat).has_value());
        EXPECT_FALSE(FindDecoratedTarget(module, spv::DecorationPerVertexKHR).has_value());
        EXPECT_FALSE(FindDecoratedTarget(module, spv::DecorationExplicitInterpAMD).has_value());
        ExpectValidFragmentModule(module);
    }
}

} // namespace
