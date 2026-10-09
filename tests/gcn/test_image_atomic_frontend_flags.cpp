// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>

#include <gtest/gtest.h>

#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace {

using namespace Shader;

struct AtomicObservation {
    u32 count{};
    IR::TextureInstInfo flags{};
    u32 descriptor_dwords{};
};

AtomicObservation TranslateAtomic(Gcn::OpcodeMIMG operation, IR::Opcode expected_ir, bool is_array,
                                  bool r128, bool return_previous) {
    // Synthetic image atomic v8, v[4:7], s[8:11/15], with independent GCN DA/R128/GLC bits.
    // Decode the instruction and call the production frontend; no manually prepared image IR
    // or resource-tracking stub can supply the flags that this regression is checking.
    const std::array<u32, 2> words{
        0xF0000100U | (static_cast<u32>(operation) << 18U) |
            (static_cast<u32>(return_previous) << 13U) | (static_cast<u32>(is_array) << 14U) |
            (static_cast<u32>(r128) << 15U),
        (2U << 16U) | (8U << 8U) | 4U,
    };
    Gcn::GcnCodeSlice slice{words.data(), words.data() + words.size()};
    Gcn::GcnDecodeContext decoder;
    const auto decoded = decoder.decodeInstruction(slice);
    EXPECT_EQ(static_cast<bool>(decoded.control.mimg.da), is_array);
    EXPECT_EQ(static_cast<bool>(decoded.control.mimg.r128), r128);

    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Compute);
    Profile profile{};
    Pools pools{};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    Gcn::Translator translator{info, runtime, profile};
    translator.Translate(block, 0U, std::span{&decoded, 1U});

    AtomicObservation observation{};
    for (const auto& inst : block->Instructions()) {
        if (inst.GetOpcode() == expected_ir) {
            ++observation.count;
            observation.flags = inst.Flags<IR::TextureInstInfo>();
            observation.descriptor_dwords = Optimization::GetResourceHandleDwordCount(inst, 0U);
        }
    }
    return observation;
}

void ExpectAtomicFlags(Gcn::OpcodeMIMG operation, IR::Opcode expected_ir) {
    for (const bool is_array : {false, true}) {
        for (const bool r128 : {false, true}) {
            for (const bool return_previous : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "array=" << is_array << " r128=" << r128
                                                  << " return_previous=" << return_previous);
                const auto result =
                    TranslateAtomic(operation, expected_ir, is_array, r128, return_previous);
                ASSERT_EQ(result.count, 1U);
                EXPECT_EQ(result.flags.is_array.Value(), is_array);
                EXPECT_EQ(result.flags.is_r128.Value(), r128);
                EXPECT_EQ(result.descriptor_dwords, r128 ? 4U : 8U);
            }
        }
    }
}

bool TranslateValidTwoDimensionalAtomic() {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;

    auto image = AmdGpu::Image::Null(false);
    image.base_address = 0x1000U;
    image.type = static_cast<u64>(AmdGpu::ImageType::Color2D);
    image.width = 15U;
    image.height = 15U;
    info.flattened_ud_buf.resize(sizeof(image) / sizeof(u32));
    std::memcpy(info.flattened_ud_buf.data(), &image, sizeof(image));

    IR::Program program{info};
    Pools pools{};
    Gcn::Block cfg_block{};
    IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    block->cfg_block = &cfg_block;
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    IR::IREmitter ir{*block};
    ir.Prologue();
    // Atomics already carry their final integer coordinate as argument 1. They do not use the
    // sampler-style body that contains coordinate words followed by LOD/sample operands.
    const auto coords = ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U));
    const auto result = ir.ImageAtomicIAdd(ir.GetUserData(IR::ScalarReg::S0), coords,
                                           ir.Imm32(1U), {});
    ir.Reference(result);
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;
    profile.supports_storage_image_multisample = true;
    Optimization::ResourceTrackingPass(program, profile, false);
    return !info.translation_failed;
}

TEST(ResourceTrackingImageAtomic, ValidTwoDimensionalCoordinateIsNotParsedAsSamplerBody) {
    // RED before the atomic fast path: PatchImageArgs interpreted the two-component coordinate
    // as a body and required a third LOD/sample argument, rejecting valid atomic IR.
    EXPECT_TRUE(TranslateValidTwoDimensionalAtomic());
}

TEST(ImageAtomicFrontend, ExchangePreservesArrayAndDescriptorWidthFlags) {
    ExpectAtomicFlags(Gcn::OpcodeMIMG::IMAGE_ATOMIC_SWAP, IR::Opcode::ImageAtomicExchange32);
}

TEST(ImageAtomicFrontend, CompareSwapRetainsArrayAndDescriptorWidthFlags) {
    ExpectAtomicFlags(Gcn::OpcodeMIMG::IMAGE_ATOMIC_CMPSWAP, IR::Opcode::ImageAtomicCmpSwap32);
}

TEST(ImageAtomicFrontend, AddRetainsArrayAndDescriptorWidthFlags) {
    ExpectAtomicFlags(Gcn::OpcodeMIMG::IMAGE_ATOMIC_ADD, IR::Opcode::ImageAtomicIAdd32);
}

} // namespace
