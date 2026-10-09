// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp>

#include <cmath>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/pp_upstream_sample_tee.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"

namespace {

constexpr u64 TargetPgmHash = 0x1234abcdu;
constexpr u64 OtherPgmHash = 0x5678abcdu;

std::vector<u32> EmitPpUpstreamSampleTeeFixture(bool instrument = true,
                                               bool profile_enabled = true,
                                               u64 pgm_hash = TargetPgmHash,
                                               u32 cube_metadata =
                                                   Shader::PackPpUpstreamSampleTeeCubeMetadata({
                                                       .classification =
                                                           Shader::PpUpstreamSampleTeeCubeFaceClass::
                                                               NonCube,
                                                       .observed = true,
                                                   })) {
    Shader::Info info{};
    info.stage = Shader::Stage::Fragment;
    info.l_stage = Shader::LogicalStage::Fragment;
    info.pgm_hash = pgm_hash;
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
        ir.PpUpstreamSampleTee(value, 2, ir.Imm32(cube_metadata), ir.Imm32(0x81234567u));
    }
    ir.Epilogue();

    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.instrument_pp_upstream_sample_tee = profile_enabled;
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

bool IrValueDependsOn(const Shader::IR::Value& value, const Shader::IR::Inst* source,
                      std::unordered_set<const Shader::IR::Inst*>& visited) {
    if (value.IsImmediate()) {
        return false;
    }
    const auto* const inst = value.InstRecursive();
    if (inst == source) {
        return true;
    }
    if (!visited.insert(inst).second) {
        return false;
    }
    for (size_t index = 0; index < inst->NumArgs(); ++index) {
        if (IrValueDependsOn(inst->Arg(index), source, visited)) {
            return true;
        }
    }
    return false;
}

bool IrValueDependsOn(const Shader::IR::Value& value, const Shader::IR::Inst* source) {
    std::unordered_set<const Shader::IR::Inst*> visited;
    return IrValueDependsOn(value, source, visited);
}

struct TrackedSampleFixture {
    std::vector<u32> spirv{};
    Shader::PpUpstreamSampleTeeCubeBindingFact binding{};
    u32 image_operation_count{};
    u32 image_query_count{};
    u32 raw_image_operation_count{};
    u32 tee_count{};
    u32 tee_argument_count{};
    u32 tee_resource_count{};
    float sample_face{};
    u32 tee_metadata{};
    bool sample_face_is_immediate{};
    bool tee_metadata_is_immediate{};
    bool sample_coords_depend_on_raw_face{};
    bool tee_value_is_existing_sample_ssa{};
    bool tee_value_is_fragment_export{};
    bool tee_metadata_depends_on_raw_face{};
};

TrackedSampleFixture EmitResourceTrackedSampleFixture(
    bool cube, std::optional<float> raw_face_value, bool target = true,
    u32 cube_count = 3, u32 base_layer = 0, u32 last_layer = 17,
    bool force_dynamic_classifier = false, bool capture_fragment_export = false) {
    Shader::Info info{};
    info.stage = Shader::Stage::Fragment;
    info.l_stage = Shader::LogicalStage::Fragment;
    info.pgm_hash = target ? TargetPgmHash : OtherPgmHash;
    info.is_pp_upstream_sample_tee_target = target;
    info.pp_upstream_sample_tee_capture_fragment_export = capture_fragment_export;
    info.flattened_ud_buf.resize(sizeof(AmdGpu::Image) / sizeof(u32));

    auto sharp = AmdGpu::Image::Null(false);
    sharp.type = static_cast<u64>(cube ? AmdGpu::ImageType::Cube
                                      : AmdGpu::ImageType::Color2DArray);
    sharp.depth = cube ? cube_count - 1 : 17;
    sharp.base_array = base_layer;
    sharp.last_array = last_layer;
    std::memcpy(info.flattened_ud_buf.data(), &sharp, sizeof(sharp));

    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    Shader::Gcn::Block cfg_block{};
    block->cfg_block = &cfg_block;
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = Shader::IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = Shader::IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());

    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    const auto image_handle = ir.GetUserData(Shader::IR::ScalarReg::S0);
    const auto sampler_handle = ir.CompositeConstruct(
        ir.Imm32(0u), ir.Imm32(0u), ir.Imm32(0u), ir.Imm32(0u));
    const auto raw_face = raw_face_value
                              ? force_dynamic_classifier
                                    ? Shader::IR::F32{ir.FPAdd(ir.Imm32(*raw_face_value),
                                                              ir.Imm32(0.0f))}
                                    : ir.Imm32(*raw_face_value)
                              : ir.GetAttribute(Shader::IR::Attribute::FragCoord, 2);
    const auto* const raw_face_inst = raw_face.TryInstRecursive();
    const auto address1 = ir.CompositeConstruct(
        ir.Imm32(1.25f), ir.Imm32(1.5f), raw_face, ir.Imm32(0.0f));
    const auto address2 = ir.CompositeConstruct(
        ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f));
    const auto address3 = ir.CompositeConstruct(
        ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f));
    Shader::IR::TextureInstInfo texture_info{};
    texture_info.is_array.Assign(1);
    const auto raw_sample = ir.ImageSampleRaw(image_handle, sampler_handle, address1, address2,
                                              address3, ir.Imm32(0.0f), texture_info);
    if (!target || capture_fragment_export) {
        for (u32 component = 0; component < 4; ++component) {
            const u32 sampled_component = capture_fragment_export ? 0u : component;
            ir.SetAttribute(Shader::IR::Attribute::RenderTarget0,
                            ir.FPAdd(Shader::IR::F32{ir.CompositeExtract(raw_sample,
                                                                         sampled_component)},
                                     ir.Imm32(0.125f * static_cast<float>(component + 1))),
                            component);
        }
        info.mrt_mask |= 1;
    }
    ir.Epilogue();

    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.instrument_pp_upstream_sample_tee = true;
    Shader::Optimization::ResourceTrackingPass(program, profile);
    if (force_dynamic_classifier) {
        Shader::Optimization::ConstantPropagationPass(program.blocks);
    }
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::CollectShaderInfoPass(program, profile);

    TrackedSampleFixture result{};
    const Shader::IR::Inst* sample{};
    const Shader::IR::Inst* tee{};
    for (const auto& inst : block->Instructions()) {
        switch (inst.GetOpcode()) {
        case Shader::IR::Opcode::ImageSampleImplicitLod:
        case Shader::IR::Opcode::ImageSampleExplicitLod:
        case Shader::IR::Opcode::ImageSampleDrefImplicitLod:
        case Shader::IR::Opcode::ImageSampleDrefExplicitLod:
        case Shader::IR::Opcode::ImageGather:
        case Shader::IR::Opcode::ImageGatherDref:
        case Shader::IR::Opcode::ImageGradient:
            ++result.image_operation_count;
            sample = &inst;
            break;
        case Shader::IR::Opcode::ImageQueryDimensions:
        case Shader::IR::Opcode::ImageQueryLod:
            ++result.image_query_count;
            break;
        case Shader::IR::Opcode::ImageRead:
        case Shader::IR::Opcode::ImageSampleRaw:
            ++result.raw_image_operation_count;
            break;
        case Shader::IR::Opcode::PpUpstreamSampleTee:
            ++result.tee_count;
            tee = &inst;
            break;
        default:
            break;
        }
    }
    if (sample != nullptr) {
        if (raw_face_inst != nullptr) {
            result.sample_coords_depend_on_raw_face =
                IrValueDependsOn(sample->Arg(1), raw_face_inst);
        }
        const auto* const coords = sample->Arg(1).TryInstRecursive();
        if (coords != nullptr &&
            coords->GetOpcode() == Shader::IR::Opcode::CompositeConstructF32x3 &&
            coords->NumArgs() >= 3 && coords->Arg(2).IsImmediate()) {
            result.sample_face_is_immediate = true;
            result.sample_face = coords->Arg(2).F32();
        }
    }
    if (tee != nullptr) {
        result.tee_argument_count = static_cast<u32>(tee->NumArgs());
        result.tee_value_is_existing_sample_ssa =
            sample != nullptr && tee->Arg(0).InstRecursive() == sample;
        const auto* const exported = tee->Arg(0).TryInstRecursive();
        std::array<const Shader::IR::Inst*, 4> output_values{};
        for (const auto& inst : block->Instructions()) {
            if (inst.GetOpcode() != Shader::IR::Opcode::SetAttribute ||
                inst.Arg(0).Attribute() != Shader::IR::Attribute::RenderTarget0) {
                continue;
            }
            const u32 component = inst.Arg(2).U32();
            if (component < output_values.size()) {
                output_values[component] = inst.Arg(1).InstRecursive();
            }
        }
        result.tee_value_is_fragment_export =
            exported != nullptr &&
            exported->GetOpcode() == Shader::IR::Opcode::CompositeConstructF32x4 &&
            std::ranges::all_of(std::views::iota(0u, 4u), [&](u32 component) {
                const auto* const value = exported->Arg(component).TryInstRecursive();
                return value != nullptr && value == output_values[component];
            });
        if (tee->NumArgs() >= 3) {
            if (raw_face_inst != nullptr) {
                result.tee_metadata_depends_on_raw_face =
                    IrValueDependsOn(tee->Arg(2), raw_face_inst);
            }
            if (tee->Arg(2).IsImmediate()) {
                result.tee_metadata_is_immediate = true;
                result.tee_metadata = tee->Arg(2).U32();
            }
        }
    }
    if (info.images.size() != 1u) {
        ADD_FAILURE() << "production ResourceTrackingPass did not retain one image binding";
        return result;
    }
    result.binding = Shader::ObservePpUpstreamSampleTeeCubeBinding(
        info, target, info.l_stage, 0, 0);
    result.tee_resource_count = static_cast<u32>(std::ranges::count_if(
        info.buffers, [](const Shader::BufferResource& resource) {
            return resource.buffer_type == Shader::BufferType::PpUpstreamSampleTee;
        }));

    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::Stage::Fragment);
    Shader::Backend::Bindings bindings{};
    result.spirv =
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    return result;
}

std::vector<std::vector<u32>> BuildSpirvIdDependencies(std::span<const u32> words) {
    std::vector<std::vector<u32>> dependencies(words.size() > 3 ? words[3] : 0);
    std::vector<size_t> definitions(dependencies.size());
    for (size_t offset = 5; offset < words.size();) {
        const u32 instruction = words[offset];
        const u32 word_count = instruction >> 16;
        if (word_count == 0 || offset + word_count > words.size()) {
            break;
        }
        const auto opcode = static_cast<spv::Op>(instruction & 0xFFFFu);
        bool has_result{};
        bool has_type{};
        spv::HasResultAndType(opcode, &has_result, &has_type);
        if (has_result) {
            const size_t result_offset = offset + 1 + static_cast<size_t>(has_type);
            const u32 result_id = words[result_offset];
            if (result_id < definitions.size()) {
                definitions[result_id] = offset;
            }
        }
        offset += word_count;
    }
    const auto add_defined = [&](std::vector<u32>& result, u32 operand) {
        if (operand < definitions.size() && definitions[operand] != 0) {
            result.push_back(operand);
        }
    };
    for (size_t offset = 5; offset < words.size();) {
        const u32 instruction = words[offset];
        const u32 word_count = instruction >> 16;
        if (word_count == 0 || offset + word_count > words.size()) {
            break;
        }
        const auto opcode = static_cast<spv::Op>(instruction & 0xFFFFu);
        bool has_result{};
        bool has_type{};
        spv::HasResultAndType(opcode, &has_result, &has_type);
        if (!has_result) {
            offset += word_count;
            continue;
        }
        const size_t result_offset = offset + 1 + static_cast<size_t>(has_type);
        const u32 result_id = words[result_offset];
        if (result_id >= dependencies.size()) {
            offset += word_count;
            continue;
        }
        const size_t first = result_offset + 1;
        const size_t last = offset + word_count;
        auto& result = dependencies[result_id];
        const auto add_range = [&](size_t begin, size_t end) {
            for (size_t index = begin; index < std::min(end, last); ++index) {
                add_defined(result, words[index]);
            }
        };
        switch (opcode) {
        case spv::Op::OpUndef:
        case spv::Op::OpConstantTrue:
        case spv::Op::OpConstantFalse:
        case spv::Op::OpConstant:
        case spv::Op::OpConstantNull:
        case spv::Op::OpSpecConstantTrue:
        case spv::Op::OpSpecConstantFalse:
        case spv::Op::OpSpecConstant:
        case spv::Op::OpFunction:
        case spv::Op::OpFunctionParameter:
        case spv::Op::OpLabel:
            break;
        case spv::Op::OpConstantComposite:
        case spv::Op::OpSpecConstantComposite:
        case spv::Op::OpCompositeConstruct:
        case spv::Op::OpAccessChain:
        case spv::Op::OpInBoundsAccessChain:
        case spv::Op::OpPtrAccessChain:
            add_range(first, last);
            break;
        case spv::Op::OpSpecConstantOp:
            add_range(first + 1, last); // first operand is an opcode literal
            break;
        case spv::Op::OpCompositeExtract:
            add_range(first, first + 1); // trailing component indices are literals
            break;
        case spv::Op::OpCompositeInsert:
        case spv::Op::OpVectorShuffle:
            add_range(first, first + 2); // trailing component indices are literals
            break;
        case spv::Op::OpExtInst:
            add_range(first + 2, last); // set id + instruction literal precede arguments
            break;
        case spv::Op::OpLoad:
            add_range(first, first + 1); // optional memory-access words are literals
            break;
        case spv::Op::OpVariable:
            add_range(first + 1, first + 2); // storage class is a literal
            break;
        case spv::Op::OpFunctionCall:
            add_range(first + 1, last); // function id is not a value dependency
            break;
        case spv::Op::OpImageSampleImplicitLod:
        case spv::Op::OpImageSampleExplicitLod:
        case spv::Op::OpImageSampleDrefImplicitLod:
        case spv::Op::OpImageSampleDrefExplicitLod:
            add_range(first, first + 2); // ignore optional image-operands mask/literals
            break;
        default:
            // Arithmetic, conversions, comparisons, selects and phis in the
            // metadata DAG have only id operands. Requiring a real definition
            // keeps integer literals from masquerading as ids.
            add_range(first, last);
            break;
        }
        offset += word_count;
    }
    return dependencies;
}

bool SpirvIdDependsOn(u32 value_id, u32 source_id,
                      const std::vector<std::vector<u32>>& dependencies,
                      std::unordered_set<u32>& visited) {
    if (value_id == source_id) {
        return true;
    }
    if (value_id >= dependencies.size() || !visited.insert(value_id).second) {
        return false;
    }
    return std::ranges::any_of(dependencies[value_id], [&](u32 operand) {
        return SpirvIdDependsOn(operand, source_id, dependencies, visited);
    });
}

bool SpirvIdDependsOn(u32 value_id, u32 source_id,
                      const std::vector<std::vector<u32>>& dependencies) {
    std::unordered_set<u32> visited;
    return SpirvIdDependsOn(value_id, source_id, dependencies, visited);
}

struct SpirvInstructionIndex {
    std::vector<size_t> definition_offsets{};
    std::unordered_map<u32, u32> constants{};
};

SpirvInstructionIndex BuildSpirvInstructionIndex(std::span<const u32> words) {
    SpirvInstructionIndex result{};
    result.definition_offsets.resize(words.size() > 3 ? words[3] : 0);
    for (size_t offset = 5; offset < words.size();) {
        const u32 word_count = words[offset] >> 16;
        if (word_count == 0 || offset + word_count > words.size()) {
            break;
        }
        const auto opcode = static_cast<spv::Op>(words[offset] & 0xFFFFu);
        bool has_result{};
        bool has_type{};
        spv::HasResultAndType(opcode, &has_result, &has_type);
        if (has_result) {
            const size_t result_offset = offset + 1 + static_cast<size_t>(has_type);
            const u32 result_id = words[result_offset];
            if (result_id < result.definition_offsets.size()) {
                result.definition_offsets[result_id] = offset;
            }
            if (opcode == spv::Op::OpConstant && word_count == 4) {
                result.constants.emplace(result_id, words[offset + 3]);
            }
        }
        offset += word_count;
    }
    return result;
}

std::optional<std::pair<u32, u32>> ResolveSpirvBasePlusConstant(
    std::span<const u32> words, const SpirvInstructionIndex& index, u32 value_id) {
    if (value_id >= index.definition_offsets.size()) {
        return std::nullopt;
    }
    const size_t offset = index.definition_offsets[value_id];
    if (offset == 0 || static_cast<spv::Op>(words[offset] & 0xFFFFu) != spv::Op::OpIAdd ||
        (words[offset] >> 16) != 5) {
        return std::nullopt;
    }
    const u32 lhs = words[offset + 3];
    const u32 rhs = words[offset + 4];
    if (const auto it = index.constants.find(lhs); it != index.constants.end()) {
        return std::pair{rhs, it->second};
    }
    if (const auto it = index.constants.find(rhs); it != index.constants.end()) {
        return std::pair{lhs, it->second};
    }
    return std::nullopt;
}

std::optional<u32> ResolveSpirvBufferDword(std::span<const u32> words,
                                          const SpirvInstructionIndex& index,
                                          u32 pointer_id) {
    if (pointer_id >= index.definition_offsets.size()) {
        return std::nullopt;
    }
    const size_t offset = index.definition_offsets[pointer_id];
    if (offset == 0 ||
        static_cast<spv::Op>(words[offset] & 0xFFFFu) != spv::Op::OpAccessChain ||
        (words[offset] >> 16) < 6) {
        return std::nullopt;
    }
    return words[offset + (words[offset] >> 16) - 1];
}

struct TeeSpirvRecordLayout {
    u32 record_base{};
    std::array<bool, 6> winner_offsets{};
    bool writer_offset_six{};
    bool record_stride_seven{};
    u32 cube_metadata_value_id{};
    u32 dependency_metadata_value_id{};
};

TeeSpirvRecordLayout InspectTeeSpirvRecordLayout(std::span<const u32> words) {
    TeeSpirvRecordLayout result{};
    const auto index = BuildSpirvInstructionIndex(words);
    const auto atomics = FindOpcodeOffsets(words, spv::Op::OpAtomicIAdd);
    if (atomics.size() != 1) {
        ADD_FAILURE() << "expected exactly one tee writer atomic";
        return result;
    }
    const u32 writer_pointer = words[atomics[0] + 3];
    const auto writer_dword = ResolveSpirvBufferDword(words, index, writer_pointer);
    if (!writer_dword) {
        ADD_FAILURE() << "atomic pointer does not resolve to the tee dword access chain";
        return result;
    }
    const auto writer = ResolveSpirvBasePlusConstant(words, index, *writer_dword);
    if (!writer) {
        ADD_FAILURE() << "writer dword is not record base plus a literal offset";
        return result;
    }
    result.record_base = writer->first;
    result.writer_offset_six = writer->second == 6u;

    for (const size_t offset : FindOpcodeOffsets(words, spv::Op::OpStore)) {
        if (offset <= atomics[0]) {
            continue;
        }
        const auto dword = ResolveSpirvBufferDword(words, index, words[offset + 1]);
        if (!dword) {
            continue;
        }
        const auto address = ResolveSpirvBasePlusConstant(words, index, *dword);
        if (!address || address->first != result.record_base || address->second >= 6u) {
            continue;
        }
        result.winner_offsets[address->second] = true;
        if (address->second == 4u) {
            result.cube_metadata_value_id = words[offset + 2];
        } else if (address->second == 5u) {
            result.dependency_metadata_value_id = words[offset + 2];
        }
    }
    if (result.record_base < index.definition_offsets.size()) {
        const size_t record_offset = index.definition_offsets[result.record_base];
        if (record_offset != 0 &&
            static_cast<spv::Op>(words[record_offset] & 0xFFFFu) == spv::Op::OpIAdd) {
            for (size_t operand = record_offset + 3;
                 operand < record_offset + (words[record_offset] >> 16); ++operand) {
                const u32 operand_id = words[operand];
                if (operand_id >= index.definition_offsets.size()) {
                    continue;
                }
                const size_t multiply_offset = index.definition_offsets[operand_id];
                if (multiply_offset == 0 ||
                    static_cast<spv::Op>(words[multiply_offset] & 0xFFFFu) != spv::Op::OpIMul) {
                    continue;
                }
                const u32 lhs = words[multiply_offset + 3];
                const u32 rhs = words[multiply_offset + 4];
                result.record_stride_seven =
                    (index.constants.contains(lhs) && index.constants.at(lhs) == 7u) ||
                    (index.constants.contains(rhs) && index.constants.at(rhs) == 7u);
            }
        }
    }
    return result;
}

TEST(PpUpstreamSampleTeeSpirv, EmitsOneGuardedExistingResultWithoutAnotherImageOperation) {
    const auto spirv = EmitPpUpstreamSampleTeeFixture();
    ASSERT_GT(spirv.size(), 5u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpSelectionMerge), 2u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpBranchConditional), 2u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpAtomicIAdd), 1u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpBitcast), 1u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpStore), 6u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageSampleImplicitLod), 0u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageSampleExplicitLod), 0u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageFetch), 0u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageRead), 0u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageQuerySize), 0u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageQuerySizeLod), 0u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageQueryLevels), 0u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageQueryLod), 0u);
}

TEST(PpUpstreamSampleTeeSpirv, CountsEveryWriterBeforeOnlyTheWinnerStoresTheValue) {
    const auto spirv = EmitPpUpstreamSampleTeeFixture();
    const auto atomics = FindOpcodeOffsets(spirv, spv::Op::OpAtomicIAdd);
    const auto stores = FindOpcodeOffsets(spirv, spv::Op::OpStore);
    ASSERT_EQ(atomics.size(), 1u);
    EXPECT_EQ(std::ranges::count_if(stores, [&](size_t offset) { return offset > atomics[0]; }),
              6u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpBranchConditional), 3u);
}

TEST(PpUpstreamSampleTeeSpirv, PreBlendExportModeTeesTheFragmentOutputWithoutAnotherImageOperation) {
    const auto fixture = EmitResourceTrackedSampleFixture(false, 0.0f, true, 1, 0, 0, false,
                                                          true);
    EXPECT_EQ(fixture.image_operation_count, 1u);
    EXPECT_EQ(fixture.image_query_count, 0u);
    EXPECT_EQ(fixture.raw_image_operation_count, 0u);
    EXPECT_EQ(fixture.tee_count, 1u);
    EXPECT_EQ(fixture.tee_argument_count, 4u);
    EXPECT_FALSE(fixture.tee_value_is_existing_sample_ssa);
    EXPECT_TRUE(fixture.tee_value_is_fragment_export);
    EXPECT_EQ(fixture.tee_resource_count, 1u);
    EXPECT_EQ(CountOpcode(fixture.spirv, spv::Op::OpImageSampleImplicitLod), 1u);
    EXPECT_EQ(CountOpcode(fixture.spirv, spv::Op::OpAtomicIAdd), 1u);
    const auto samples = FindOpcodeOffsets(fixture.spirv, spv::Op::OpImageSampleImplicitLod);
    ASSERT_EQ(samples.size(), 1u);
    const auto layout = InspectTeeSpirvRecordLayout(fixture.spirv);
    ASSERT_NE(layout.dependency_metadata_value_id, 0u);
    const auto dependencies = BuildSpirvIdDependencies(fixture.spirv);
    EXPECT_TRUE(SpirvIdDependsOn(layout.dependency_metadata_value_id,
                                fixture.spirv[samples[0] + 2], dependencies))
        << "the existing dependency metadata dword must carry the sole sampled scalar used by the export";
    const u32 sample_coordinate_id = fixture.spirv[samples[0] + 4];
    EXPECT_TRUE(SpirvIdDependsOn(layout.dependency_metadata_value_id, sample_coordinate_id,
                                dependencies))
        << "the same metadata dword must carry the exact coordinates used by the sole sample";
}

TEST(PpUpstreamSampleTeeSpirv,
     CubeClassUsesTheFixedSevenDwordRecordWithoutAnyAdditionalImageOperation) {
    const auto non_cube = EmitPpUpstreamSampleTeeFixture();
    const u32 stride_compressed = Shader::PackPpUpstreamSampleTeeCubeMetadata({
        .classification = Shader::PpUpstreamSampleTeeCubeFaceClass::StrideCompressed,
        .observed = true,
    });
    const auto cube = EmitPpUpstreamSampleTeeFixture(
        true, true, TargetPgmHash, stride_compressed);

    for (const auto opcode : {spv::Op::OpImageSampleImplicitLod,
                              spv::Op::OpImageSampleExplicitLod,
                              spv::Op::OpImageSampleDrefImplicitLod,
                              spv::Op::OpImageSampleDrefExplicitLod,
                              spv::Op::OpImageFetch, spv::Op::OpImageRead,
                              spv::Op::OpImageQuerySize,
                              spv::Op::OpImageQuerySizeLod,
                              spv::Op::OpImageQueryLevels,
                              spv::Op::OpImageQueryLod}) {
        EXPECT_EQ(CountOpcode(non_cube, opcode), 0u) << static_cast<u32>(opcode);
        EXPECT_EQ(CountOpcode(cube, opcode), 0u) << static_cast<u32>(opcode);
    }
    EXPECT_EQ(CountOpcode(cube, spv::Op::OpAtomicIAdd), 1u);
    EXPECT_EQ(CountOpcode(cube, spv::Op::OpStore),
              CountOpcode(non_cube, spv::Op::OpStore))
        << "cube classification reuses the one metadata dword in the fixed tee ABI";
    const auto atomics = FindOpcodeOffsets(cube, spv::Op::OpAtomicIAdd);
    const auto stores = FindOpcodeOffsets(cube, spv::Op::OpStore);
    ASSERT_EQ(atomics.size(), 1u);
    EXPECT_EQ(std::ranges::count_if(stores,
                                    [&](size_t offset) { return offset > atomics[0]; }),
              6u)
        << "the winner stores four result dwords plus cube and dependency metadata; writers stay atomic";
    const auto layout = InspectTeeSpirvRecordLayout(cube);
    EXPECT_EQ(layout.winner_offsets, (std::array{true, true, true, true, true, true}))
        << "winner stores must address RGBA at +0..3, cube metadata at +4, and dependency metadata at +5";
    EXPECT_TRUE(layout.writer_offset_six)
        << "the independent writer atomic must address record dword +6";
    EXPECT_TRUE(layout.record_stride_seven);
}

TEST(PpUpstreamSampleTeeSpirv,
     ProductionResourceTrackingKeepsRawFacesUnchangedAndClassifiesThatSameSsa) {
    using Class = Shader::PpUpstreamSampleTeeCubeFaceClass;
    const std::array faces{5.0f, 8.0f, 13.0f, 6.0f, -1.0f, 0.5f};
    const std::array expected_classes{
        Class::Identity, Class::StrideCompressed, Class::StrideCompressed,
        Class::Reserved, Class::Invalid, Class::Invalid};
    for (u32 index = 0; index < faces.size(); ++index) {
        const auto cube = EmitResourceTrackedSampleFixture(true, faces[index]);
        ASSERT_TRUE(cube.sample_face_is_immediate);
        EXPECT_EQ(cube.sample_face, faces[index])
            << "observation cannot apply or hide a cube face-coordinate behavior change";
        ASSERT_TRUE(cube.tee_metadata_is_immediate);
        const auto decoded = Shader::DecodePpUpstreamSampleTeeCubeMetadata(
            cube.tee_metadata);
        EXPECT_TRUE(decoded.observed);
        EXPECT_FALSE(decoded.malformed);
        EXPECT_EQ(decoded.classification, expected_classes[index]);
        EXPECT_EQ(cube.image_operation_count, 1u);
        EXPECT_EQ(cube.image_query_count, 0u);
        EXPECT_EQ(cube.raw_image_operation_count, 0u);
        EXPECT_EQ(cube.tee_count, 1u);
        EXPECT_EQ(cube.tee_argument_count, 4u);
        EXPECT_TRUE(cube.tee_value_is_existing_sample_ssa);
        EXPECT_EQ(cube.tee_resource_count, 1u);
    }
}

TEST(PpUpstreamSampleTeeSpirv,
     DynamicClassifierDagMatchesAllClassesAndIgnoresCubeCountAndLiveLayerRange) {
    using Class = Shader::PpUpstreamSampleTeeCubeFaceClass;
    const std::array faces{0.0f,
                           5.0f,
                           6.0f,
                           8.0f,
                           13.0f,
                           14.0f,
                           -1.0f,
                           0.5f,
                           std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::infinity()};
    const std::array expected{Class::Identity,
                              Class::Identity,
                              Class::Reserved,
                              Class::StrideCompressed,
                              Class::StrideCompressed,
                              Class::Reserved,
                              Class::Invalid,
                              Class::Invalid,
                              Class::Invalid,
                              Class::Invalid};
    for (u32 index = 0; index < faces.size(); ++index) {
        const auto fixture = EmitResourceTrackedSampleFixture(
            true, faces[index], true, 3, 0, 17, true);
        ASSERT_TRUE(fixture.tee_metadata_is_immediate) << index;
        const auto decoded = Shader::DecodePpUpstreamSampleTeeCubeMetadata(
            fixture.tee_metadata);
        EXPECT_TRUE(decoded.observed) << index;
        EXPECT_FALSE(decoded.malformed) << index;
        EXPECT_EQ(decoded.classification, expected[index]) << index;
        EXPECT_EQ(fixture.image_operation_count, 1u);
        EXPECT_EQ(fixture.tee_count, 1u);
        EXPECT_EQ(fixture.tee_argument_count, 4u);
        EXPECT_TRUE(fixture.tee_value_is_existing_sample_ssa);
    }

    const auto single = EmitResourceTrackedSampleFixture(
        true, 13.0f, true, 1, 0, 5, true);
    const auto multi = EmitResourceTrackedSampleFixture(
        true, 13.0f, true, 3, 0, 17, true);
    const auto aligned_offset = EmitResourceTrackedSampleFixture(
        true, 13.0f, true, 3, 6, 11, true);
    ASSERT_TRUE(single.tee_metadata_is_immediate);
    ASSERT_TRUE(multi.tee_metadata_is_immediate);
    ASSERT_TRUE(aligned_offset.tee_metadata_is_immediate);
    EXPECT_EQ(single.tee_metadata, multi.tee_metadata);
    EXPECT_EQ(single.tee_metadata, aligned_offset.tee_metadata)
        << "cube count/base/last are static host facts and cannot enter cached shader metadata";
    EXPECT_EQ(single.spirv, multi.spirv)
        << "same cube permutation/raw face must lower identically across live cube counts";
    EXPECT_EQ(single.spirv, aligned_offset.spirv)
        << "live layer ranges are not specialization keys and cannot affect the classifier DAG";

    Shader::ImageSpecialization cube_specialization{};
    cube_specialization.type = AmdGpu::ImageType::Color2DArray;
    cube_specialization.is_cube = true;
    const auto cube_count_variant = cube_specialization;
    EXPECT_EQ(cube_specialization, cube_count_variant);
    auto ordinary_specialization = cube_specialization;
    ordinary_specialization.is_cube = false;
    EXPECT_NE(cube_specialization, ordinary_specialization)
        << "cube-vs-ordinary remains a real cache key while cube count/range do not";
}

TEST(PpUpstreamSampleTeeSpirv,
     ProductionResourceTrackingUsesOneExistingSampleSsaAndTheSameDynamicRawCubeFace) {
    const auto cube = EmitResourceTrackedSampleFixture(true, std::nullopt);
    const auto ordinary_array = EmitResourceTrackedSampleFixture(false, std::nullopt);

    EXPECT_TRUE(cube.binding.observed);
    EXPECT_TRUE(cube.binding.valid);
    EXPECT_EQ(cube.binding.logical_stage, Shader::LogicalStage::Fragment);
    EXPECT_EQ(cube.binding.image_resource_index, 0u);
    EXPECT_EQ(cube.binding.array_element, 0u);
    EXPECT_EQ(cube.binding.shape, Shader::PpUpstreamSampleTeeCubeShape::CubeArray);
    EXPECT_EQ(ordinary_array.binding.shape,
              Shader::PpUpstreamSampleTeeCubeShape::NonCube)
        << "a flattened Color2DArray view is not evidence that the live sharp was a cube";

    EXPECT_EQ(cube.image_operation_count, 1u);
    EXPECT_EQ(cube.image_query_count, 0u);
    EXPECT_EQ(cube.raw_image_operation_count, 0u);
    EXPECT_EQ(cube.tee_count, 1u);
    EXPECT_EQ(cube.tee_argument_count, 4u);
    EXPECT_TRUE(cube.sample_coords_depend_on_raw_face);
    EXPECT_TRUE(cube.tee_value_is_existing_sample_ssa)
        << "the tee must consume the result SSA of the one sample already performed";
    EXPECT_TRUE(cube.tee_metadata_depends_on_raw_face)
        << "classification must consume the same pre-FixCubeCoords face SSA as the sample";

    EXPECT_EQ(ordinary_array.image_operation_count, 1u);
    EXPECT_EQ(ordinary_array.image_query_count, 0u);
    EXPECT_EQ(ordinary_array.raw_image_operation_count, 0u);
    EXPECT_EQ(ordinary_array.tee_count, 1u);
    EXPECT_TRUE(ordinary_array.sample_coords_depend_on_raw_face);
    EXPECT_TRUE(ordinary_array.tee_value_is_existing_sample_ssa);
    EXPECT_FALSE(ordinary_array.tee_metadata_depends_on_raw_face)
        << "ordinary arrays emit the NonCube class without inspecting their layer coordinate";
    ASSERT_TRUE(ordinary_array.tee_metadata_is_immediate);
    const auto ordinary_metadata = Shader::DecodePpUpstreamSampleTeeCubeMetadata(
        ordinary_array.tee_metadata);
    EXPECT_TRUE(ordinary_metadata.observed);
    EXPECT_FALSE(ordinary_metadata.malformed);
    EXPECT_EQ(ordinary_metadata.classification,
              Shader::PpUpstreamSampleTeeCubeFaceClass::NonCube);

    for (const auto* fixture : {&cube, &ordinary_array}) {
        const auto& spirv = fixture->spirv;
        const auto samples =
            FindOpcodeOffsets(spirv, spv::Op::OpImageSampleImplicitLod);
        ASSERT_EQ(samples.size(), 1u);
        EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageSampleExplicitLod), 0u);
        EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageSampleDrefImplicitLod), 0u);
        EXPECT_EQ(CountOpcode(spirv, spv::Op::OpImageSampleDrefExplicitLod), 0u);
        for (const auto opcode : {spv::Op::OpImageFetch, spv::Op::OpImageRead,
                                  spv::Op::OpImageQuerySize,
                                  spv::Op::OpImageQuerySizeLod,
                                  spv::Op::OpImageQueryLevels,
                                  spv::Op::OpImageQueryLod}) {
            EXPECT_EQ(CountOpcode(spirv, opcode), 0u) << static_cast<u32>(opcode);
        }
        const auto atomics = FindOpcodeOffsets(spirv, spv::Op::OpAtomicIAdd);
        const auto stores = FindOpcodeOffsets(spirv, spv::Op::OpStore);
        ASSERT_EQ(atomics.size(), 1u);
        const std::vector<size_t> winner_stores = [&] {
            std::vector<size_t> result;
            std::ranges::copy_if(stores, std::back_inserter(result),
                                 [&](size_t offset) { return offset > atomics[0]; });
            return result;
        }();
        ASSERT_EQ(winner_stores.size(), 6u)
            << "four result dwords plus cube and dependency metadata precede no separate writer store";

        const u32 sample_id = spirv[samples[0] + 2];
        const auto dependencies = BuildSpirvIdDependencies(spirv);
        const u32 sample_payload_stores =
            static_cast<u32>(std::ranges::count_if(winner_stores, [&](size_t offset) {
                const u32 stored_value_id = spirv[offset + 2];
                return SpirvIdDependsOn(stored_value_id, sample_id, dependencies);
            }));
        EXPECT_EQ(sample_payload_stores, 4u)
            << "exactly RGBA must derive from the sole existing image sample SSA";
        const auto layout = InspectTeeSpirvRecordLayout(spirv);
        EXPECT_EQ(layout.winner_offsets,
                  (std::array{true, true, true, true, true, true}));
        EXPECT_TRUE(layout.writer_offset_six);
        EXPECT_TRUE(layout.record_stride_seven);
        ASSERT_NE(layout.cube_metadata_value_id, 0u);
        ASSERT_NE(layout.dependency_metadata_value_id, 0u);
        if (fixture == &cube) {
            const u32 sample_coordinate_id = spirv[samples[0] + 4];
            const auto instruction_index = BuildSpirvInstructionIndex(spirv);
            ASSERT_LT(sample_coordinate_id,
                      instruction_index.definition_offsets.size());
            const size_t coordinate_definition =
                instruction_index.definition_offsets[sample_coordinate_id];
            ASSERT_NE(coordinate_definition, 0u);
            ASSERT_EQ(static_cast<spv::Op>(spirv[coordinate_definition] & 0xFFFFu),
                      spv::Op::OpCompositeConstruct)
                << "the sole sample's coordinate vector must expose its exact layer scalar";
            ASSERT_EQ(spirv[coordinate_definition] >> 16, 6u)
                << "cube/2D-array sample coordinates are exactly three components";
            const u32 sampled_face_scalar_id = spirv[coordinate_definition + 5];
            ASSERT_NE(sampled_face_scalar_id, 0u);
            EXPECT_TRUE(SpirvIdDependsOn(layout.cube_metadata_value_id,
                                         sampled_face_scalar_id, dependencies))
                << "the metadata store must derive from component 2 of the sole sample coordinate, not any unrelated component-2 extract";
        }
    }

    EXPECT_EQ(CountOpcode(cube.spirv, spv::Op::OpImageSampleImplicitLod),
              CountOpcode(ordinary_array.spirv,
                          spv::Op::OpImageSampleImplicitLod));
    EXPECT_EQ(CountOpcode(cube.spirv, spv::Op::OpStore),
              CountOpcode(ordinary_array.spirv, spv::Op::OpStore))
        << "cube classification fits the target tee record and adds no second store path";
}

TEST(PpUpstreamSampleTeeSpirv,
     RealNonTargetCubeRetainsOneOrdinarySampleAndLegacyPushWithoutTeeWork) {
    const auto disabled = EmitResourceTrackedSampleFixture(true, 13.0f, false);
    EXPECT_EQ(disabled.image_operation_count, 1u);
    EXPECT_EQ(disabled.image_query_count, 0u);
    EXPECT_EQ(disabled.raw_image_operation_count, 0u);
    EXPECT_EQ(disabled.tee_count, 0u);
    EXPECT_EQ(disabled.tee_resource_count, 0u);
    EXPECT_EQ(CountOpcode(disabled.spirv, spv::Op::OpAtomicIAdd), 0u);
    EXPECT_EQ(CountOpcode(disabled.spirv, spv::Op::OpImageSampleImplicitLod), 1u);
    EXPECT_EQ(CountOpcode(disabled.spirv, spv::Op::OpImageSampleExplicitLod), 0u);
    for (const auto opcode : {spv::Op::OpImageFetch, spv::Op::OpImageRead,
                              spv::Op::OpImageQuerySize,
                              spv::Op::OpImageQuerySizeLod,
                              spv::Op::OpImageQueryLevels,
                              spv::Op::OpImageQueryLod}) {
        EXPECT_EQ(CountOpcode(disabled.spirv, opcode), 0u)
            << static_cast<u32>(opcode);
    }
    const auto legacy = InspectPushBlockTail(disabled.spirv);
    EXPECT_EQ(legacy.components, 2u);
    EXPECT_EQ(legacy.member_offset, 112u);
    EXPECT_EQ(legacy.component_width, 32u);
    EXPECT_EQ(legacy.component_signedness, 0u);
}

TEST(PpUpstreamSampleTeeSpirv, RejectsUnalignedOrOutOfRangeSlotsBeforeBufferAccess) {
    const auto spirv = EmitPpUpstreamSampleTeeFixture();
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpUMod), 1u);
    EXPECT_GE(CountOpcode(spirv, spv::Op::OpULessThanEqual), 4u);
}

TEST(PpUpstreamSampleTeeSpirv, KeepsTheLegacyPushBlockWhenInstrumentationIsDisabled) {
    const auto ordinary = InspectPushBlockTail(EmitPpUpstreamSampleTeeFixture(false, false));
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

TEST(PpUpstreamSampleTeeSpirv, KeepsTheLegacyPushBlockForANonTargetShader) {
    const auto spirv = EmitPpUpstreamSampleTeeFixture(false, true, OtherPgmHash);
    const auto ordinary = InspectPushBlockTail(spirv);
    EXPECT_EQ(ordinary.components, 2u);
    EXPECT_EQ(ordinary.member_offset, 112u);
    EXPECT_EQ(ordinary.component_width, 32u);
    EXPECT_EQ(ordinary.component_signedness, 0u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpAtomicIAdd), 0u);
}

TEST(PpUpstreamSampleTeeSpirv, KeepsTheLegacyPushBlockWhenTheTargetHasNoTeeResource) {
    const auto spirv = EmitPpUpstreamSampleTeeFixture(false, true, TargetPgmHash);
    const auto ordinary = InspectPushBlockTail(spirv);
    EXPECT_EQ(ordinary.components, 2u);
    EXPECT_EQ(ordinary.member_offset, 112u);
    EXPECT_EQ(ordinary.component_width, 32u);
    EXPECT_EQ(ordinary.component_signedness, 0u);
    EXPECT_EQ(CountOpcode(spirv, spv::Op::OpAtomicIAdd), 0u);
}

TEST(PpUpstreamSampleTeeSpirv, ClearsNonTargetInfoWithoutAddingResourcesOrFragCoord) {
    Shader::Info info{};
    info.stage = Shader::Stage::Fragment;
    info.l_stage = Shader::LogicalStage::Fragment;
    info.pgm_hash = OtherPgmHash;
    info.is_pp_upstream_sample_tee_target = false;
    info.pp_upstream_sample_tee_results[2] = 1;
    Shader::IR::Program program{info};

    Shader::Profile profile{};
    profile.instrument_pp_upstream_sample_tee = true;
    Shader::Optimization::CollectShaderInfoPass(program, profile);

    EXPECT_TRUE(std::ranges::all_of(program.info.pp_upstream_sample_tee_results,
                                    [](u32 count) { return count == 0; }));
    EXPECT_TRUE(program.info.buffers.empty());
    EXPECT_FALSE(program.info.loads.Get(Shader::IR::Attribute::FragCoord, 0));
    EXPECT_FALSE(program.info.loads.Get(Shader::IR::Attribute::FragCoord, 1));
}

TEST(PpUpstreamFragmentExportDependencies,
     TracesTheActualRenderTargetValueToImageBufferVaryingAndUserDataRoots) {
    Shader::Info info{};
    info.stage = Shader::Stage::Fragment;
    info.l_stage = Shader::LogicalStage::Fragment;
    info.is_pp_upstream_sample_tee_target = true;
    info.pp_upstream_sample_tee_capture_fragment_export = true;
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    Shader::Gcn::Block cfg_block{};
    block->cfg_block = &cfg_block;
    program.blocks.push_back(block);

    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    const auto image = ir.ImageSampleRaw(
        ir.Imm32(3u), ir.Imm32(0u),
        ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f)),
        ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f)),
        ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f)),
        ir.Imm32(0.0f), {});
    const auto buffer = ir.LoadBufferF32(1, ir.Imm32(2u), ir.Imm32(16u), {});
    const auto varying = ir.GetAttribute(Shader::IR::Attribute::Param4, 1);
    const auto user_data = ir.BitCast<Shader::IR::F32>(ir.GetUserData(Shader::IR::ScalarReg::S5));
    ir.SetAttribute(Shader::IR::Attribute::RenderTarget0,
                    ir.FPAdd(Shader::IR::F32{ir.CompositeExtract(image, 0)},
                             Shader::IR::F32{buffer}),
                    0);
    ir.SetAttribute(Shader::IR::Attribute::RenderTarget0,
                    ir.FPAdd(varying, user_data), 1);
    ir.Epilogue();

    Shader::Optimization::AnalyzePpUpstreamFragmentExportDependencies(program);
    const auto& fact = info.pp_upstream_fragment_export_dependencies;
    EXPECT_TRUE(fact.observed);
    EXPECT_TRUE(fact.valid);
    EXPECT_EQ(fact.component_mask, 0x3u);
    EXPECT_EQ(fact.image_binding_mask, 1ull << 3);
    EXPECT_EQ(fact.buffer_binding_mask, 1ull << 2);
    EXPECT_EQ(fact.parameter_mask, 1u << 4);
    EXPECT_EQ(fact.user_data_mask, 1u << 5);
    EXPECT_EQ(fact.image_component_mask, 0x1u);
    EXPECT_EQ(fact.buffer_component_mask, 0x1u);
    EXPECT_EQ(fact.parameter_component_mask, 0x2u);
    EXPECT_EQ(fact.user_data_component_mask, 0x2u);
    EXPECT_EQ(fact.undef_component_mask, 0u);
    EXPECT_EQ(fact.unknown_component_mask, 0u);
    EXPECT_FALSE(fact.frag_coord);
    EXPECT_FALSE(fact.front_face);
    EXPECT_FALSE(fact.unknown);
    EXPECT_FALSE(fact.overflowed);
}

TEST(PpUpstreamFragmentExportDependencies,
     IgnoresNonTargetOutputsAndFailsClosedForMalformedResourceBindings) {
    auto analyze = [](bool target, u32 image_binding) {
        Shader::Info info{};
        info.stage = Shader::Stage::Fragment;
        info.l_stage = Shader::LogicalStage::Fragment;
        info.is_pp_upstream_sample_tee_target = target;
        info.pp_upstream_sample_tee_capture_fragment_export = target;
        Shader::Pools pools{};
        Shader::IR::Program program{info};
        auto* const block = pools.block_pool.Create(pools.inst_pool);
        Shader::Gcn::Block cfg_block{};
        block->cfg_block = &cfg_block;
        program.blocks.push_back(block);
        Shader::IR::IREmitter ir{*block};
        ir.Prologue();
        const auto sample = ir.ImageSampleRaw(
            ir.Imm32(image_binding), ir.Imm32(0u),
            ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f),
                                  ir.Imm32(0.0f)),
            ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f),
                                  ir.Imm32(0.0f)),
            ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f),
                                  ir.Imm32(0.0f)),
            ir.Imm32(0.0f), {});
        ir.SetAttribute(Shader::IR::Attribute::RenderTarget0,
                        Shader::IR::F32{ir.CompositeExtract(sample, 0)}, 0);
        ir.Epilogue();
        Shader::Optimization::AnalyzePpUpstreamFragmentExportDependencies(program);
        return info.pp_upstream_fragment_export_dependencies;
    };

    const auto disabled = analyze(false, 0);
    EXPECT_FALSE(disabled.observed);
    EXPECT_FALSE(disabled.valid);

    const auto malformed = analyze(true, Shader::NUM_IMAGES);
    EXPECT_TRUE(malformed.observed);
    EXPECT_FALSE(malformed.valid);
    EXPECT_TRUE(malformed.overflowed);
    EXPECT_EQ(malformed.image_binding_mask, 0u);
}

TEST(PpUpstreamFragmentExportDependencies,
     CompositeExtractFollowsOnlyTheSelectedLaneInsteadOfEveryVectorLane) {
    Shader::Info info{};
    info.stage = Shader::Stage::Fragment;
    info.l_stage = Shader::LogicalStage::Fragment;
    info.is_pp_upstream_sample_tee_target = true;
    info.pp_upstream_sample_tee_capture_fragment_export = true;
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    Shader::Gcn::Block cfg_block{};
    block->cfg_block = &cfg_block;
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    const auto undef_it = block->PrependNewInst(block->end(), Shader::IR::Opcode::UndefU32);
    const Shader::IR::U32 undef{Shader::IR::Value{&*undef_it}};
    const auto vector = ir.CompositeConstruct(
        ir.GetAttribute(Shader::IR::Attribute::Param2, 0),
        Shader::IR::F32{ir.LoadBufferF32(1, ir.Imm32(1u), ir.Imm32(0u), {})},
        Shader::IR::F32{ir.BitCast<Shader::IR::F32>(ir.GetUserData(Shader::IR::ScalarReg::S3))},
        Shader::IR::F32{ir.BitCast<Shader::IR::F32>(undef)});
    ir.SetAttribute(Shader::IR::Attribute::RenderTarget0,
                    Shader::IR::F32{ir.CompositeExtract(vector, 0)}, 0);
    ir.Epilogue();

    Shader::Optimization::AnalyzePpUpstreamFragmentExportDependencies(program);
    const auto& fact = info.pp_upstream_fragment_export_dependencies;
    EXPECT_TRUE(fact.valid);
    EXPECT_EQ(fact.parameter_mask, 1u << 2);
    EXPECT_EQ(fact.parameter_component_mask, 1u);
    EXPECT_EQ(fact.buffer_binding_mask, 0u);
    EXPECT_EQ(fact.user_data_mask, 0u);
    EXPECT_EQ(fact.undef_component_mask, 0u);
}

TEST(PpUpstreamFragmentExportDependencies,
     TracesTheLoweredImageSampleSeenAtTheFinalIrBoundary) {
    Shader::Info info{};
    info.stage = Shader::Stage::Fragment;
    info.l_stage = Shader::LogicalStage::Fragment;
    info.is_pp_upstream_sample_tee_target = true;
    info.pp_upstream_sample_tee_capture_fragment_export = true;
    Shader::Pools pools{};
    Shader::IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    Shader::Gcn::Block cfg_block{};
    block->cfg_block = &cfg_block;
    program.blocks.push_back(block);
    Shader::IR::IREmitter ir{*block};
    ir.Prologue();
    const auto sample = ir.ImageSampleExplicitLod(
        ir.Imm32(4u), ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f)),
        ir.Imm32(0.0f), {}, {});
    ir.SetAttribute(Shader::IR::Attribute::RenderTarget0,
                    Shader::IR::F32{ir.CompositeExtract(sample, 0)}, 0);
    ir.Epilogue();

    Shader::Optimization::AnalyzePpUpstreamFragmentExportDependencies(program);
    const auto& fact = info.pp_upstream_fragment_export_dependencies;
    EXPECT_TRUE(fact.valid);
    EXPECT_EQ(fact.image_binding_mask, 1ull << 4);
    EXPECT_EQ(fact.image_component_mask, 1u);
}

TEST(PpUpstreamFragmentExportDependencies,
     TargetIrDumpIsNarrowlyEnabledWithoutChangingTheGlobalDumpSetting) {
    Shader::Info info{};
    info.stage = Shader::Stage::Fragment;
    info.l_stage = Shader::LogicalStage::Fragment;
    EXPECT_FALSE(Shader::IR::ShouldDumpProgram(false, info));

    info.is_pp_upstream_sample_tee_target = true;
    EXPECT_FALSE(Shader::IR::ShouldDumpProgram(false, info));

    info.pp_upstream_sample_tee_capture_fragment_export = true;
    EXPECT_TRUE(Shader::IR::ShouldDumpProgram(false, info));

    info.l_stage = Shader::LogicalStage::Vertex;
    EXPECT_FALSE(Shader::IR::ShouldDumpProgram(false, info));
    EXPECT_TRUE(Shader::IR::ShouldDumpProgram(true, info));

    Shader::Info depth_writer{};
    depth_writer.stage = Shader::Stage::Vertex;
    depth_writer.l_stage = Shader::LogicalStage::Vertex;
    depth_writer.pgm_hash = 0xf8139f04;
    EXPECT_TRUE(Shader::IR::ShouldDumpProgram(false, depth_writer));

    depth_writer.stage = Shader::Stage::Fragment;
    depth_writer.l_stage = Shader::LogicalStage::Fragment;
    depth_writer.pgm_hash = 0x14bd455a;
    EXPECT_TRUE(Shader::IR::ShouldDumpProgram(false, depth_writer));
    depth_writer.pgm_hash = 0x144934aa;
    EXPECT_TRUE(Shader::IR::ShouldDumpProgram(false, depth_writer));
}

} // namespace
