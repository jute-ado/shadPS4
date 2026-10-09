// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <gtest/gtest.h>

#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace {
using namespace Shader;

struct DrawUserDataFixture {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    RuntimeInfo runtime{};
    Profile profile{};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    IR::Block* block{};

    DrawUserDataFixture() {
        info.stage = Stage::Vertex;
        info.l_stage = LogicalStage::Vertex;
        info.user_data = user_data;
        runtime.Initialize(Stage::Vertex);
        runtime.num_user_data = NUM_USER_DATA_REGS;
        runtime.vs_info.draw_user_data = {};
        block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        auto& node = program.syntax_list.emplace_back();
        node.type = IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
        program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
        program.post_order_blocks = IR::PostOrder(program.syntax_list.front());
    }

    void Prologue() {
        Gcn::Translator translator{info, runtime, profile};
        translator.EmitPrologue(block);
    }

    void Cleanup() {
        Optimization::SsaRewritePass(program);
        Optimization::ConstantPropagationPass(program.blocks);
        Optimization::IdentityRemovalPass(program.blocks);
        Optimization::DeadCodeEliminationPass(program);
    }

    IR::Value FirstReference() const {
        for (const auto& inst : *block) {
            if (inst.GetOpcode() == IR::Opcode::Reference) {
                return inst.Arg(0).Resolve();
            }
        }
        return {};
    }
};

TEST(DrawUserData, RuntimeInitializationDisablesAllDynamicLocations) {
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Vertex);
    EXPECT_EQ(runtime.vs_info.draw_user_data.base_vertex, DrawUserDataLocations::Disabled);
    EXPECT_EQ(runtime.vs_info.draw_user_data.base_instance, DrawUserDataLocations::Disabled);
    EXPECT_EQ(runtime.vs_info.draw_user_data.draw_index, DrawUserDataLocations::Disabled);
}

TEST(DrawUserData, MappedScalarRegistersAreDynamicBeforeHostResourceAnalysis) {
    for (const auto attribute :
         {IR::Attribute::BaseVertex, IR::Attribute::BaseInstance, IR::Attribute::DrawIndex}) {
        DrawUserDataFixture fixture;
        auto& mapping = fixture.runtime.vs_info.draw_user_data;
        if (attribute == IR::Attribute::BaseVertex) {
            mapping.base_vertex = 2U;
        } else if (attribute == IR::Attribute::BaseInstance) {
            mapping.base_instance = 2U;
        } else {
            mapping.draw_index = 2U;
        }
        fixture.Prologue();
        IR::IREmitter ir{*fixture.block};
        ir.Reference(ir.GetScalarReg<IR::U32>(IR::ScalarReg::S2));
        fixture.Cleanup();
        const auto* const source = fixture.FirstReference().TryInstRecursive();
        ASSERT_NE(source, nullptr);
        ASSERT_EQ(source->GetOpcode(), IR::Opcode::GetAttributeU32);
        EXPECT_EQ(source->Arg(0).Attribute(), attribute);
    }
}

TEST(DrawUserData, UnmappedUserDataKeepsItsOrdinaryPushConstantSource) {
    DrawUserDataFixture fixture;
    fixture.runtime.vs_info.draw_user_data.base_instance = 2U;
    fixture.Prologue();
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.GetScalarReg<IR::U32>(IR::ScalarReg::S3));
    fixture.Cleanup();
    const auto* const source = fixture.FirstReference().TryInstRecursive();
    ASSERT_NE(source, nullptr);
    EXPECT_EQ(source->GetOpcode(), IR::Opcode::GetUserData);
    EXPECT_EQ(source->Arg(0).ScalarReg(), IR::ScalarReg::S3);
}

TEST(DrawUserData, InvalidOrCollidingDestinationsFailBeforeAnyRegisterIsSubstituted) {
    for (const auto mapping :
         {DrawUserDataLocations{.base_instance = 16U}, DrawUserDataLocations{.base_vertex = 254U},
          DrawUserDataLocations{.base_vertex = 2U, .base_instance = 2U}}) {
        DrawUserDataFixture fixture;
        fixture.runtime.vs_info.draw_user_data = mapping;
        fixture.Prologue();
        EXPECT_TRUE(fixture.info.translation_failed);
    }
}

TEST(DrawUserData, PhysicalNonVertexStageDoesNotReadTheVertexMappingUnion) {
    DrawUserDataFixture fixture;
    fixture.info.stage = Stage::Local;
    fixture.runtime.vs_info.draw_user_data.base_instance = 2U;
    fixture.Prologue();
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.GetScalarReg<IR::U32>(IR::ScalarReg::S2));
    fixture.Cleanup();
    const auto* const source = fixture.FirstReference().TryInstRecursive();
    ASSERT_NE(source, nullptr);
    EXPECT_EQ(source->GetOpcode(), IR::Opcode::GetUserData);
    EXPECT_FALSE(fixture.info.translation_failed);
}

TEST(DrawUserData, ImplicitDescriptorWordsCannotCaptureDynamicDrawUserData) {
    // Image instructions carry only the first descriptor word. S2 is therefore not an IR
    // dependency after DCE, but it is still a word of the implicit eight-dword image descriptor.
    for (u8 destination : {u8{0}, u8{2}, u8{7}}) {
        DrawUserDataFixture fixture;
        fixture.runtime.vs_info.draw_user_data.base_instance = destination;
        fixture.Prologue();
        IR::IREmitter ir{*fixture.block};
        const auto value = ir.ImageRead(ir.GetScalarReg<IR::U32>(IR::ScalarReg::S0),
                                        ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U)),
                                        ir.Imm32(0U), IR::U32{}, {});
        ir.Reference(value);
        fixture.Cleanup();
        EXPECT_FALSE(
            Optimization::FlattenExtendedUserdataPass(fixture.program, {}, &fixture.runtime));
        EXPECT_TRUE(fixture.info.translation_failed);
    }
}

TEST(DrawUserData, DynamicRegisterOutsideDescriptorSpanDoesNotRejectTheShader) {
    DrawUserDataFixture fixture;
    fixture.runtime.vs_info.draw_user_data.base_instance = 8U;
    fixture.Prologue();
    IR::IREmitter ir{*fixture.block};
    ir.Reference(ir.ImageRead(ir.GetScalarReg<IR::U32>(IR::ScalarReg::S0),
                              ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U)), ir.Imm32(0U),
                              IR::U32{}, {}));
    fixture.Cleanup();
    EXPECT_TRUE(Optimization::FlattenExtendedUserdataPass(fixture.program, {}, &fixture.runtime));
    EXPECT_FALSE(fixture.info.translation_failed);
}
} // namespace
