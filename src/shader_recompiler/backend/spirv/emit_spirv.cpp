// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <magic_enum/magic_enum.hpp>

#include "common/assert.h"
#include "common/func_traits.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Backend::SPIRV {
namespace {

static constexpr spv::ExecutionMode GetInputPrimitiveType(AmdGpu::PrimitiveType type) {
    switch (type) {
    case AmdGpu::PrimitiveType::PointList:
        return spv::ExecutionMode::InputPoints;
    case AmdGpu::PrimitiveType::LineList:
    case AmdGpu::PrimitiveType::LineStrip:
        return spv::ExecutionMode::InputLines;
    case AmdGpu::PrimitiveType::TriangleList:
    case AmdGpu::PrimitiveType::TriangleStrip:
    case AmdGpu::PrimitiveType::RectList:
        return spv::ExecutionMode::Triangles;
    case AmdGpu::PrimitiveType::AdjTriangleList:
        return spv::ExecutionMode::InputTrianglesAdjacency;
    case AmdGpu::PrimitiveType::AdjLineList:
        return spv::ExecutionMode::InputLinesAdjacency;
    default:
        UNREACHABLE_MSG("Unknown input primitive type {}", u32(type));
    }
}

static constexpr spv::ExecutionMode GetOutputPrimitiveType(AmdGpu::GsOutputPrimitiveType type) {
    switch (type) {
    case AmdGpu::GsOutputPrimitiveType::PointList:
        return spv::ExecutionMode::OutputVertices;
    case AmdGpu::GsOutputPrimitiveType::LineStrip:
        return spv::ExecutionMode::OutputLineStrip;
    case AmdGpu::GsOutputPrimitiveType::TriangleStrip:
        return spv::ExecutionMode::OutputTriangleStrip;
    default:
        UNREACHABLE_MSG("Unknown output primitive type {}", u32(type));
    }
}

template <auto func, typename... Args>
void SetDefinition(EmitContext& ctx, IR::Inst* inst, Args... args) {
    inst->SetDefinition<Id>(func(ctx, std::forward<Args>(args)...));
}

template <typename ArgType>
ArgType Arg(EmitContext& ctx, const IR::Value& arg) {
    if constexpr (std::is_same_v<ArgType, Id>) {
        return ctx.Def(arg);
    } else if constexpr (std::is_same_v<ArgType, const IR::Value&>) {
        return arg;
    } else if constexpr (std::is_same_v<ArgType, u32>) {
        return arg.U32();
    } else if constexpr (std::is_same_v<ArgType, u64>) {
        return arg.U64();
    } else if constexpr (std::is_same_v<ArgType, bool>) {
        return arg.U1();
    } else if constexpr (std::is_same_v<ArgType, IR::Attribute>) {
        return arg.Attribute();
    } else if constexpr (std::is_same_v<ArgType, IR::ScalarReg>) {
        return arg.ScalarReg();
    } else if constexpr (std::is_same_v<ArgType, IR::VectorReg>) {
        return arg.VectorReg();
    } else if constexpr (std::is_same_v<ArgType, IR::VirtualReg>) {
        return arg.VirtualReg();
    } else if constexpr (std::is_same_v<ArgType, const char*>) {
        return arg.StringLiteral();
    } else if constexpr (std::is_same_v<ArgType, IR::Patch>) {
        return arg.Patch();
    }
    UNREACHABLE();
}

template <auto func, bool is_first_arg_inst, size_t... I>
void Invoke(EmitContext& ctx, IR::Inst* inst, std::index_sequence<I...>) {
    using Traits = Common::FuncTraits<decltype(func)>;
    if constexpr (std::is_same_v<typename Traits::ReturnType, Id>) {
        if constexpr (is_first_arg_inst) {
            SetDefinition<func>(
                ctx, inst, inst,
                Arg<typename Traits::template ArgType<I + 2>>(ctx, inst->Arg(I))...);
        } else {
            SetDefinition<func>(
                ctx, inst, Arg<typename Traits::template ArgType<I + 1>>(ctx, inst->Arg(I))...);
        }
    } else {
        if constexpr (is_first_arg_inst) {
            func(ctx, inst, Arg<typename Traits::template ArgType<I + 2>>(ctx, inst->Arg(I))...);
        } else {
            func(ctx, Arg<typename Traits::template ArgType<I + 1>>(ctx, inst->Arg(I))...);
        }
    }
}

template <auto func>
void Invoke(EmitContext& ctx, IR::Inst* inst) {
    using Traits = Common::FuncTraits<decltype(func)>;
    static_assert(Traits::NUM_ARGS >= 1, "Insufficient arguments");
    if constexpr (Traits::NUM_ARGS == 1) {
        Invoke<func, false>(ctx, inst, std::make_index_sequence<0>{});
    } else {
        using FirstArgType = typename Traits::template ArgType<1>;
        static constexpr bool is_first_arg_inst = std::is_same_v<FirstArgType, IR::Inst*>;
        using Indices = std::make_index_sequence<Traits::NUM_ARGS - (is_first_arg_inst ? 2 : 1)>;
        Invoke<func, is_first_arg_inst>(ctx, inst, Indices{});
    }
}

void EmitInst(EmitContext& ctx, IR::Inst* inst) {
    switch (inst->GetOpcode()) {
#define OPCODE(name, result_type, ...)                                                             \
    case IR::Opcode::name:                                                                         \
        return Invoke<&Emit##name>(ctx, inst);
#include "shader_recompiler/ir/opcodes.inc"
#undef OPCODE
    }
    UNREACHABLE_MSG("Invalid opcode {}", inst->GetOpcode());
}

Id TypeId(const EmitContext& ctx, IR::Type type) {
    switch (type) {
    case IR::Type::U1:
        return ctx.U1[1];
    case IR::Type::U32:
        return ctx.U32[1];
    default:
        UNREACHABLE_MSG("Phi node type {}", type);
    }
}

void Traverse(EmitContext& ctx, const IR::Program& program) {
    struct ActiveLoop {
        IR::Block* header{};
        bool materializes_dynamic_scalar_mask{};
    };
    std::vector<ActiveLoop> active_loops;
    std::vector<IR::Block*> partial_unroll_headers;
    IR::Block* scan_current_block{};
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        switch (node.type) {
        case IR::AbstractSyntaxNode::Type::Block:
            scan_current_block = node.data.block;
            if (!active_loops.empty() &&
                std::ranges::any_of(scan_current_block->Instructions(), [](const IR::Inst& inst) {
                    return inst.GetOpcode() == IR::Opcode::MaterializeScalarMaskWord;
                })) {
                active_loops.back().materializes_dynamic_scalar_mask = true;
            }
            break;
        case IR::AbstractSyntaxNode::Type::Loop:
            active_loops.push_back({.header = scan_current_block});
            scan_current_block = nullptr;
            break;
        case IR::AbstractSyntaxNode::Type::Repeat:
            if (!active_loops.empty()) {
                if (active_loops.back().materializes_dynamic_scalar_mask) {
                    partial_unroll_headers.push_back(active_loops.back().header);
                }
                active_loops.pop_back();
            }
            scan_current_block = nullptr;
            break;
        default:
            scan_current_block = nullptr;
            break;
        }
    }
    program.info.has_dynamic_scalar_mask_loop = !partial_unroll_headers.empty();

    IR::Block* current_block{};
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        switch (node.type) {
        case IR::AbstractSyntaxNode::Type::Block: {
            const Id label{node.data.block->Definition<Id>()};
            if (current_block) {
                ctx.OpBranch(label);
            }
            current_block = node.data.block;
            ctx.AddLabel(label);
            for (IR::Inst& inst : node.data.block->Instructions()) {
                EmitInst(ctx, &inst);
            }
            ctx.first_to_last_label_map[label.value] = ctx.last_label;
            break;
        }
        case IR::AbstractSyntaxNode::Type::If: {
            const Id if_label{node.data.if_node.body->Definition<Id>()};
            const Id endif_label{node.data.if_node.merge->Definition<Id>()};
            ctx.OpSelectionMerge(endif_label, spv::SelectionControlMask::MaskNone);
            ctx.OpBranchConditional(ctx.Def(node.data.if_node.cond), if_label, endif_label);
            break;
        }
        case IR::AbstractSyntaxNode::Type::Loop: {
            const Id body_label{node.data.loop.body->Definition<Id>()};
            const Id continue_label{node.data.loop.continue_block->Definition<Id>()};
            const Id endloop_label{node.data.loop.merge->Definition<Id>()};

            // Scalar-mask walks carry guest execution-state through loop backedges. Keep their
            // structured control flow exact: PartialCount is only an optimization hint, but
            // drivers may transform these ballot-heavy loops in ways that change their result.
            // The classification above remains useful for dispatch tiling.
            ctx.OpLoopMerge(endloop_label, continue_label, spv::LoopControlMask::MaskNone);
            ctx.OpBranch(body_label);
            break;
        }
        case IR::AbstractSyntaxNode::Type::Break: {
            const Id break_label{node.data.break_node.merge->Definition<Id>()};
            const Id skip_label{node.data.break_node.skip->Definition<Id>()};
            ctx.OpBranchConditional(ctx.Def(node.data.break_node.cond), break_label, skip_label);
            break;
        }
        case IR::AbstractSyntaxNode::Type::EndIf:
            if (current_block) {
                ctx.OpBranch(node.data.end_if.merge->Definition<Id>());
            }
            break;
        case IR::AbstractSyntaxNode::Type::Repeat: {
            Id cond{ctx.Def(node.data.repeat.cond)};
            const Id loop_header_label{node.data.repeat.loop_header->Definition<Id>()};
            const Id merge_label{node.data.repeat.merge->Definition<Id>()};
            ctx.OpBranchConditional(cond, loop_header_label, merge_label);
            break;
        }
        case IR::AbstractSyntaxNode::Type::Return:
            ctx.OpReturn();
            break;
        case IR::AbstractSyntaxNode::Type::Unreachable:
            ctx.OpUnreachable();
            break;
        }
        if (node.type != IR::AbstractSyntaxNode::Type::Block) {
            current_block = nullptr;
        }
    }
}

void TraverseWave64Lockstep(EmitContext& ctx, const IR::Program& program) {
    constexpr u32 ExitState = std::numeric_limits<u32>::max();
    enum class TransitionKind { Unset, Direct, Conditional, Exit };
    struct Transition {
        TransitionKind kind{TransitionKind::Unset};
        IR::U1 condition{};
        u32 taken{ExitState};
        u32 fallthrough{ExitState};
    };

    std::unordered_map<IR::Block*, u32> block_indices;
    block_indices.reserve(program.blocks.size());
    for (u32 index = 0; index < program.blocks.size(); ++index) {
        block_indices.emplace(program.blocks[index], index);
    }
    const auto index_of = [&](IR::Block* block) { return block_indices.at(block); };

    std::vector<Transition> transitions(program.blocks.size());
    IR::Block* current_block{};
    const auto direct = [&](IR::Block* from, IR::Block* to) {
        transitions[index_of(from)] = {
            .kind = TransitionKind::Direct, .taken = index_of(to)};
    };
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        using Type = IR::AbstractSyntaxNode::Type;
        switch (node.type) {
        case Type::Block:
            if (current_block && transitions[index_of(current_block)].kind == TransitionKind::Unset) {
                direct(current_block, node.data.block);
            }
            current_block = node.data.block;
            break;
        case Type::If:
            transitions[index_of(current_block)] = {
                .kind = TransitionKind::Conditional,
                .condition = node.data.if_node.cond,
                .taken = index_of(node.data.if_node.body),
                .fallthrough = index_of(node.data.if_node.merge),
            };
            current_block = nullptr;
            break;
        case Type::Loop:
            direct(current_block, node.data.loop.body);
            current_block = nullptr;
            break;
        case Type::Break:
            transitions[index_of(current_block)] = {
                .kind = TransitionKind::Conditional,
                .condition = node.data.break_node.cond,
                .taken = index_of(node.data.break_node.merge),
                .fallthrough = index_of(node.data.break_node.skip),
            };
            current_block = nullptr;
            break;
        case Type::EndIf:
            if (current_block) {
                direct(current_block, node.data.end_if.merge);
            }
            current_block = nullptr;
            break;
        case Type::Repeat:
            transitions[index_of(current_block)] = {
                .kind = TransitionKind::Conditional,
                .condition = node.data.repeat.cond,
                .taken = index_of(node.data.repeat.loop_header),
                .fallthrough = index_of(node.data.repeat.merge),
            };
            current_block = nullptr;
            break;
        case Type::Return:
        case Type::Unreachable:
            if (current_block) {
                transitions[index_of(current_block)].kind = TransitionKind::Exit;
            }
            current_block = nullptr;
            break;
        }
    }
    if (current_block && transitions[index_of(current_block)].kind == TransitionKind::Unset) {
        transitions[index_of(current_block)].kind = TransitionKind::Exit;
    }

    const Id entry_label = ctx.OpLabel();
    const Id loop_header = ctx.OpLabel();
    const Id scan_blocks = ctx.OpLabel();
    const Id continue_label = ctx.OpLabel();
    const Id merge_label = ctx.OpLabel();
    ctx.AddLabel(entry_label);
    const Id state = ctx.DefineVar<false>(ctx.U32[1], spv::StorageClass::Function);
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if ((inst.GetOpcode() == IR::Opcode::GetVirtualRegister ||
                 inst.GetOpcode() == IR::Opcode::SetVirtualRegister)) {
                const IR::VirtualReg reg = inst.Arg(0).VirtualReg();
                if (!ctx.virtual_registers.contains(reg.Key())) {
                    ctx.virtual_registers.emplace(
                        reg.Key(), ctx.DefineVar<false>(ctx.ValueType(reg.type),
                                                        spv::StorageClass::Function));
                }
            }
            if (inst.Type() == IR::Type::Void || inst.Type() == IR::Type::Opaque) {
                continue;
            }
            // A dispatcher destroys the host CFG dominance relationships assumed by SSA. Spill
            // every value rather than trying to infer only cross-case values: backend helpers can
            // introduce internal SPIR-V blocks which are deliberately absent from the guest IR.
            ctx.cross_block_values.emplace(
                &inst, ctx.DefineVar<false>(ctx.ValueType(inst.Type()),
                                            spv::StorageClass::Function));
        }
    }
    ctx.spill_all_values = true;
    // Prologue emission defines backend context values (for example packed buffer offsets) that
    // are consumed by instructions in multiple guest blocks. Those definitions are shader-entry
    // state, not guest control-flow state, so they must dominate the dispatcher loop.
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::Prologue) {
                EmitInst(ctx, &inst);
            }
        }
    }
    ctx.OpStore(state, ctx.u32_zero_value);
    ctx.OpBranch(loop_header);

    ctx.AddLabel(loop_header);
    // Reset and repopulate one workgroup-wide active flag. Every invocation performs both atomics,
    // so the barriers and loop decision remain workgroup-uniform even after guest lanes retire.
    ASSERT(ctx.runtime_info.cs_info.shared_memory_size >= 2 * sizeof(u32));
    const u32 active_word = (ctx.runtime_info.cs_info.shared_memory_size - 2 * sizeof(u32)) /
                            sizeof(u32);
    const Id active_ptr = ctx.EmitSharedMemoryAccess(
        ctx.shared_u32, ctx.shared_memory_u32, ctx.ConstU32(active_word));
    const Id workgroup_scope = ctx.ConstU32(static_cast<u32>(spv::Scope::Workgroup));
    const Id workgroup_semantics = ctx.ConstU32(static_cast<u32>(
        spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::WorkgroupMemory));
    ctx.OpAtomicAnd(ctx.U32[1], active_ptr, workgroup_scope, workgroup_semantics,
                    ctx.u32_zero_value);
    EmitBarrier(ctx);
    const Id current_state = ctx.OpLoad(ctx.U32[1], state);
    const Id lane_active =
        ctx.OpINotEqual(ctx.U1[1], current_state, ctx.ConstU32(ExitState));
    ctx.OpAtomicOr(ctx.U32[1], active_ptr, workgroup_scope, workgroup_semantics,
                   ctx.OpSelect(ctx.U32[1], lane_active, ctx.u32_one_value,
                                ctx.u32_zero_value));
    EmitBarrier(ctx);
    const Id any_active = ctx.OpLoad(ctx.U32[1], active_ptr);
    const Id is_exit = ctx.OpIEqual(ctx.U1[1], any_active, ctx.u32_zero_value);
    ctx.OpLoopMerge(merge_label, continue_label, spv::LoopControlMask::MaskNone);
    ctx.OpBranchConditional(is_exit, merge_label, scan_blocks);

    ctx.AddLabel(scan_blocks);
    for (u32 index = 0; index < program.blocks.size(); ++index) {
        IR::Block* const block = program.blocks[index];
        ctx.current_block = block;
        const Id block_state = ctx.OpLoad(ctx.U32[1], state);
        const Id block_active =
            ctx.OpIEqual(ctx.U1[1], block_state, ctx.ConstU32(index));
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::Prologue) {
                continue;
            }
            const IR::Opcode opcode = inst.GetOpcode();
            const bool is_barrier = opcode == IR::Opcode::Barrier ||
                                    opcode == IR::Opcode::WorkgroupMemoryBarrier ||
                                    opcode == IR::Opcode::DeviceMemoryBarrier;
            const bool is_metadata = opcode == IR::Opcode::ConditionRef ||
                                     opcode == IR::Opcode::Reference ||
                                     opcode == IR::Opcode::PhiMove ||
                                     opcode == IR::Opcode::Epilogue;
            const bool predicate_effect = !is_barrier && !is_metadata &&
                                          (opcode == IR::Opcode::SetVirtualRegister ||
                                           inst.MayHaveSideEffects());
            Id effect_merge{};
            if (predicate_effect) {
                const Id effect_body = ctx.OpLabel();
                effect_merge = ctx.OpLabel();
                ctx.OpSelectionMerge(effect_merge, spv::SelectionControlMask::MaskNone);
                ctx.OpBranchConditional(block_active, effect_body, effect_merge);
                ctx.AddLabel(effect_body);
            }
            EmitInst(ctx, &inst);
            if (const auto spill = ctx.cross_block_values.find(&inst);
                spill != ctx.cross_block_values.end()) {
                const Id prior = ctx.OpLoad(ctx.ValueType(inst.Type()), spill->second);
                ctx.OpStore(spill->second,
                            ctx.OpSelect(ctx.ValueType(inst.Type()), block_active,
                                         inst.Definition<Id>(), prior));
            }
            if (predicate_effect) {
                ctx.OpBranch(effect_merge);
                ctx.AddLabel(effect_merge);
            }
        }
        const Transition& transition = transitions[index];
        Id next_state{};
        switch (transition.kind) {
        case TransitionKind::Direct:
            next_state = ctx.ConstU32(transition.taken);
            break;
        case TransitionKind::Conditional:
            next_state = ctx.OpSelect(ctx.U32[1], ctx.Def(transition.condition),
                                      ctx.ConstU32(transition.taken),
                                      ctx.ConstU32(transition.fallthrough));
            break;
        case TransitionKind::Unset:
        case TransitionKind::Exit:
            next_state = ctx.ConstU32(ExitState);
            break;
        }
        const Id prior_state = ctx.OpLoad(ctx.U32[1], state);
        ctx.OpStore(state,
                    ctx.OpSelect(ctx.U32[1], block_active, next_state, prior_state));
    }
    ctx.OpBranch(continue_label);
    ctx.AddLabel(continue_label);
    ctx.OpBranch(loop_header);
    ctx.AddLabel(merge_label);
    ctx.current_block = nullptr;
    ctx.OpReturn();
}

Id DefineMain(EmitContext& ctx, const IR::Program& program) {
    const Id void_function{ctx.TypeFunction(ctx.void_id)};
    const Id main{ctx.OpFunction(ctx.void_id, spv::FunctionControlMask::MaskNone, void_function)};
    for (IR::Block* const block : program.blocks) {
        block->SetDefinition(ctx.OpLabel());
    }
    if (program.info.emulate_wave64_control_flow) {
        TraverseWave64Lockstep(ctx, program);
    } else {
        Traverse(ctx, program);
    }
    ctx.OpFunctionEnd();
    return main;
}

spv::ExecutionMode ExecutionMode(AmdGpu::TessellationType primitive) {
    switch (primitive) {
    case AmdGpu::TessellationType::Isoline:
        return spv::ExecutionMode::Isolines;
    case AmdGpu::TessellationType::Triangle:
        return spv::ExecutionMode::Triangles;
    case AmdGpu::TessellationType::Quad:
        return spv::ExecutionMode::Quads;
    }
    UNREACHABLE_MSG("Tessellation primitive {}", magic_enum::enum_name(primitive));
}

spv::ExecutionMode ExecutionMode(AmdGpu::TessellationPartitioning spacing) {
    switch (spacing) {
    case AmdGpu::TessellationPartitioning::Integer:
        return spv::ExecutionMode::SpacingEqual;
    case AmdGpu::TessellationPartitioning::FracOdd:
        return spv::ExecutionMode::SpacingFractionalOdd;
    case AmdGpu::TessellationPartitioning::FracEven:
        return spv::ExecutionMode::SpacingFractionalEven;
    case AmdGpu::TessellationPartitioning::Pow2:
        // Pow2 rounds tessellation factors to the nearest power of 2, which has no
        // direct Vulkan equivalent. SpacingEqual (integer) is the closest match.
        LOG_WARNING(Render_Vulkan, "Tessellation partitioning Pow2 has no Vulkan equivalent, "
                                   "falling back to SpacingEqual");
        return spv::ExecutionMode::SpacingEqual;
    default:
        break;
    }
    UNREACHABLE_MSG("Tessellation spacing {}", magic_enum::enum_name(spacing));
}

void SetupCapabilities(const Info& info, const Profile& profile, const RuntimeInfo& runtime_info,
                       EmitContext& ctx) {
    ctx.AddCapability(spv::Capability::Image1D);
    ctx.AddCapability(spv::Capability::Sampled1D);
    ctx.AddCapability(spv::Capability::ImageQuery);
    ctx.AddCapability(spv::Capability::Int8);
    ctx.AddCapability(spv::Capability::Int16);
    ctx.AddCapability(spv::Capability::Int64);
    ctx.AddCapability(spv::Capability::UniformAndStorageBuffer8BitAccess);
    ctx.AddCapability(spv::Capability::UniformAndStorageBuffer16BitAccess);
    if (info.uses_fp16) {
        ctx.AddCapability(spv::Capability::Float16);
    }
    if (info.uses_fp64) {
        ctx.AddCapability(spv::Capability::Float64);
    }
    if (info.has_storage_images) {
        ctx.AddCapability(spv::Capability::StorageImageExtendedFormats);
        ctx.AddCapability(spv::Capability::StorageImageReadWithoutFormat);
        ctx.AddCapability(spv::Capability::StorageImageWriteWithoutFormat);
        if (profile.supports_image_load_store_lod) {
            ctx.AddExtension("SPV_AMD_shader_image_load_store_lod");
            ctx.AddCapability(spv::Capability::ImageReadWriteLodAMD);
        }
    }
    if (info.has_image_gather) {
        ctx.AddCapability(spv::Capability::ImageGatherExtended);
    }
    if (info.has_image_query) {
        ctx.AddCapability(spv::Capability::ImageQuery);
    }
    if ((info.uses_image_atomic_float_min_max && profile.supports_image_fp32_atomic_min_max) ||
        (info.uses_buffer_atomic_float_min_max && profile.supports_buffer_fp32_atomic_min_max)) {
        ctx.AddExtension("SPV_EXT_shader_atomic_float_min_max");
        ctx.AddCapability(spv::Capability::AtomicFloat32MinMaxEXT);
    }
    if (info.uses_lane_id || info.emulate_wave64) {
        ctx.AddCapability(spv::Capability::GroupNonUniform);
    }
    if (info.uses_group_quad) {
        ctx.AddCapability(spv::Capability::GroupNonUniformQuad);
    }
    if (info.uses_group_ballot) {
        ctx.AddCapability(spv::Capability::GroupNonUniformBallot);
    }
    if (info.uses_group_vote) {
        ctx.AddCapability(spv::Capability::GroupNonUniformVote);
    }
    if (info.uses_group_shuffle) {
        ctx.AddCapability(spv::Capability::GroupNonUniformShuffle);
    }
    const auto stage = info.l_stage;
    if (stage == LogicalStage::Vertex) {
        ctx.AddExtension("SPV_KHR_shader_draw_parameters");
        ctx.AddCapability(spv::Capability::DrawParameters);
    }
    if (stage == LogicalStage::Geometry) {
        ctx.AddCapability(spv::Capability::Geometry);
    }
    if (info.stage == Stage::Fragment) {
        if (profile.supports_amd_shader_explicit_vertex_parameter) {
            ctx.AddExtension("SPV_AMD_shader_explicit_vertex_parameter");
        } else if (profile.supports_fragment_shader_barycentric) {
            ctx.AddExtension("SPV_KHR_fragment_shader_barycentric");
            ctx.AddCapability(spv::Capability::FragmentBarycentricKHR);
            ctx.AddCapability(spv::Capability::InterpolationFunction);
        }
        if (info.loads.Get(IR::Attribute::SampleIndex) ||
            runtime_info.fs_info.addr_flags.linear_sample_ena ||
            runtime_info.fs_info.addr_flags.persp_sample_ena ||
            (!profile.supports_amd_shader_explicit_vertex_parameter &&
             profile.supports_fragment_shader_barycentric &&
             info.loads.Get(IR::Attribute::BaryCoordSmoothSample))) {
            ctx.AddCapability(spv::Capability::SampleRateShading);
        }
        if (info.loads.GetAny(IR::Attribute::RenderTargetIndex)) {
            ctx.AddCapability(spv::Capability::Geometry);
        }
        if (info.stores.Get(IR::Attribute::StencilRef) && profile.supports_shader_stencil_export) {
            ctx.AddExtension("SPV_EXT_shader_stencil_export");
            ctx.AddCapability(spv::Capability::StencilExportEXT);
        }
    }
    if (stage == LogicalStage::TessellationControl || stage == LogicalStage::TessellationEval) {
        ctx.AddCapability(spv::Capability::Tessellation);
    }
    if (stage == LogicalStage::Vertex || stage == LogicalStage::TessellationControl ||
        stage == LogicalStage::TessellationEval) {
        if (info.stores.GetAny(IR::Attribute::RenderTargetIndex)) {
            ctx.AddCapability(spv::Capability::ShaderLayer);
        }
        if (info.stores.GetAny(IR::Attribute::ViewportIndex)) {
            ctx.AddCapability(spv::Capability::ShaderViewportIndex);
        }
    } else if (stage == LogicalStage::Geometry &&
               info.stores.GetAny(IR::Attribute::ViewportIndex)) {
        ctx.AddCapability(spv::Capability::MultiViewport);
    }
    if (info.uses_dma) {
        ctx.AddCapability(spv::Capability::PhysicalStorageBufferAddresses);
        ctx.AddExtension("SPV_KHR_physical_storage_buffer");
    }
    const auto shared_type_count = std::popcount(static_cast<u32>(info.shared_types));
    if (shared_type_count > 1 && profile.supports_workgroup_explicit_memory_layout) {
        ctx.AddExtension("SPV_KHR_workgroup_memory_explicit_layout");
        ctx.AddCapability(spv::Capability::WorkgroupMemoryExplicitLayoutKHR);
        ctx.AddCapability(spv::Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR);
    }
    if (info.uses_buffer_int64_atomics || info.uses_shared_int64_atomics) {
        if (info.uses_buffer_int64_atomics) {
            ASSERT_MSG(ctx.profile.supports_buffer_int64_atomics,
                       "Shader requires support for atomic Int64 buffer operations that your "
                       "Vulkan instance does not advertise");
        }
        if (info.uses_shared_int64_atomics) {
            ASSERT_MSG(ctx.profile.supports_shared_int64_atomics,
                       "Shader requires support for atomic Int64 shared memory operations that "
                       "your Vulkan instance does not advertise");
        }
        ctx.AddCapability(spv::Capability::Int64Atomics);
    }
}

void DefineEntryPoint(const Info& info, EmitContext& ctx, Id main) {
    const std::span interfaces(ctx.interfaces.data(), ctx.interfaces.size());
    spv::ExecutionModel execution_model{};
    switch (info.l_stage) {
    case LogicalStage::Compute: {
        const std::array<u32, 3> workgroup_size{ctx.runtime_info.cs_info.workgroup_size};
        execution_model = spv::ExecutionModel::GLCompute;
        ctx.AddExecutionMode(main, spv::ExecutionMode::LocalSize, workgroup_size[0],
                             workgroup_size[1], workgroup_size[2]);
        if (ctx.profile.supports_subgroup_uniform_control_flow &&
            (info.uses_lane_id || info.uses_group_quad || info.uses_group_ballot ||
             info.uses_group_vote ||
             info.uses_group_shuffle || info.requires_subgroup_uniform_control_flow)) {
            ctx.AddExtension("SPV_KHR_subgroup_uniform_control_flow");
            // SPV_KHR_subgroup_uniform_control_flow assigns execution-mode value 4421. The
            // vendored SPIR-V enum predates the symbolic name.
            ctx.AddExecutionMode(main, static_cast<spv::ExecutionMode>(4421));
        }
        break;
    }
    case LogicalStage::Vertex:
        execution_model = spv::ExecutionModel::Vertex;
        break;
    case LogicalStage::TessellationControl:
        execution_model = spv::ExecutionModel::TessellationControl;
        ctx.AddCapability(spv::Capability::Tessellation);
        ctx.AddExecutionMode(main, spv::ExecutionMode::OutputVertices,
                             ctx.runtime_info.hs_info.NumOutputControlPoints());
        break;
    case LogicalStage::TessellationEval: {
        execution_model = spv::ExecutionModel::TessellationEvaluation;
        ctx.AddExecutionMode(main, ExecutionMode(ctx.runtime_info.es_vs_info.tess_type));
        ctx.AddExecutionMode(main, ExecutionMode(ctx.runtime_info.es_vs_info.tess_partitioning));
        ctx.AddExecutionMode(main, ctx.runtime_info.es_vs_info.tess_topology ==
                                           AmdGpu::TessellationTopology::TriangleCcw
                                       ? spv::ExecutionMode::VertexOrderCcw
                                       : spv::ExecutionMode::VertexOrderCw);
        break;
    }
    case LogicalStage::Fragment:
        execution_model = spv::ExecutionModel::Fragment;
        if (ctx.profile.lower_left_origin_mode) {
            ctx.AddExecutionMode(main, spv::ExecutionMode::OriginLowerLeft);
        } else {
            ctx.AddExecutionMode(main, spv::ExecutionMode::OriginUpperLeft);
        }
        if (info.has_discard) {
            ctx.AddCapability(spv::Capability::DemoteToHelperInvocation);
        }
        if (info.stores.GetAny(IR::Attribute::Depth)) {
            ctx.AddExecutionMode(main, spv::ExecutionMode::DepthReplacing);
        }
        break;
    case LogicalStage::Geometry:
        execution_model = spv::ExecutionModel::Geometry;
        ctx.AddExecutionMode(main, GetInputPrimitiveType(ctx.runtime_info.gs_info.in_primitive));
        ctx.AddExecutionMode(main,
                             GetOutputPrimitiveType(ctx.runtime_info.gs_info.out_primitive[0]));
        ctx.AddExecutionMode(main, spv::ExecutionMode::OutputVertices,
                             ctx.runtime_info.gs_info.output_vertices);
        ctx.AddExecutionMode(main, spv::ExecutionMode::Invocations,
                             ctx.runtime_info.gs_info.num_invocations);
        break;
    default:
        UNREACHABLE_MSG("Stage {}", u32(info.stage));
    }
    ctx.AddEntryPoint(execution_model, main, "main", interfaces);
}

void SetupDenormFlushMode(EmitContext& ctx, const Profile& profile, const RuntimeInfo& runtime_info,
                          Id main_func) {
    const auto fp32_denorm_mode = runtime_info.fp_denorm_mode32;
    if (fp32_denorm_mode == AmdGpu::FpDenormMode::InOutFlush) {
        if (profile.support_fp32_denorm_flush) {
            ctx.AddCapability(spv::Capability::DenormFlushToZero);
            ctx.AddExecutionMode(main_func, spv::ExecutionMode::DenormFlushToZero, 32U);
        } else {
            static std::once_flag logged;
            std::call_once(logged, [] {
                LOG_WARNING(Render_Vulkan, "Float32 denorm flushing is not supported by the GPU");
            });
        }
    } else {
        LOG_WARNING(Render_Vulkan, "Unknown FP denorm mode {}", u32(fp32_denorm_mode));
    }

    if (!ctx.info.uses_fp16 && !ctx.info.uses_fp64) {
        return;
    }

    const auto fp16_64_denorm_mode = runtime_info.fp_denorm_mode16_64;
    const auto same_denorm_mode = fp16_64_denorm_mode == fp32_denorm_mode;
    if (!same_denorm_mode && !profile.supports_denorm_behavior_independence) {
        static std::once_flag logged;
        std::call_once(logged, [] {
            LOG_WARNING(Render_Vulkan, "Float32 and Float16/64 denorm modes are different but the "
                                       "GPU does not support that");
        });
        return;
    }

    if (fp16_64_denorm_mode == AmdGpu::FpDenormMode::InOutFlush) {
        if (profile.support_fp16_denorm_flush) {
            ctx.AddCapability(spv::Capability::DenormFlushToZero);
            ctx.AddExecutionMode(main_func, spv::ExecutionMode::DenormFlushToZero, 16U);
        } else {
            static std::once_flag logged;
            std::call_once(logged, [] {
                LOG_WARNING(Render_Vulkan, "Float16 denorm flushing is not supported by the GPU");
            });
        }
        if (profile.support_fp64_denorm_flush) {
            ctx.AddCapability(spv::Capability::DenormFlushToZero);
            ctx.AddExecutionMode(main_func, spv::ExecutionMode::DenormFlushToZero, 64U);
        } else {
            static std::once_flag logged;
            std::call_once(logged, [] {
                LOG_WARNING(Render_Vulkan, "Float64 denorm flushing is not supported by the GPU");
            });
        }
    } else if (fp16_64_denorm_mode == AmdGpu::FpDenormMode::InOutAllow) {
        if (profile.support_fp16_denorm_preserve) {
            ctx.AddCapability(spv::Capability::DenormPreserve);
            ctx.AddExecutionMode(main_func, spv::ExecutionMode::DenormPreserve, 16U);
        } else {
            static std::once_flag logged;
            std::call_once(logged, [] {
                LOG_WARNING(Render_Vulkan, "Float16 denorm preserving is not supported by the GPU");
            });
        }
        if (profile.support_fp64_denorm_preserve) {
            ctx.AddCapability(spv::Capability::DenormPreserve);
            ctx.AddExecutionMode(main_func, spv::ExecutionMode::DenormPreserve, 64U);
        } else {
            static std::once_flag logged;
            std::call_once(logged, [] {
                LOG_WARNING(Render_Vulkan, "Float64 denorm preserving is not supported by the GPU");
            });
        }
    } else {
        LOG_WARNING(Render_Vulkan, "Unknown Float16/64 denorm mode {}", u32(fp16_64_denorm_mode));
    }
}

void SetupRoundingMode(EmitContext& ctx, const Profile& profile, const RuntimeInfo& runtime_info,
                       Id main_func) {
    const auto fp_round_mode = runtime_info.fp_round_mode32;
    if (fp_round_mode == AmdGpu::FpRoundMode::ToZero) {
        if (profile.support_fp32_round_to_zero) {
            ctx.AddCapability(spv::Capability::RoundingModeRTZ);
            ctx.AddExecutionMode(main_func, spv::ExecutionMode::RoundingModeRTZ, 32U);
        } else {
            static std::once_flag logged;
            std::call_once(logged, [] {
                LOG_WARNING(Render_Vulkan, "Float32 rounding to zero is not supported by the GPU");
            });
        }
    } else if (fp_round_mode != AmdGpu::FpRoundMode::NearestEven) {
        LOG_WARNING(Render_Vulkan, "Unknown FP rounding mode {}", u32(fp_round_mode));
    }

    if (ctx.info.uses_fp16 || ctx.info.uses_fp64) {
        const auto fp16_64_round_mode = runtime_info.fp_round_mode16_64;
        const auto same_round_mode = fp16_64_round_mode == fp_round_mode;
        if (!same_round_mode && !profile.supports_rounding_mode_independence) {
            static std::once_flag logged;
            std::call_once(logged, [] {
                LOG_WARNING(Render_Vulkan, "Float32 and Float16/64 rounding modes are different "
                                           "but the GPU does not support that");
            });
            return;
        }

        if (fp16_64_round_mode == AmdGpu::FpRoundMode::ToZero) {
            if (ctx.info.uses_fp16) {
                if (profile.support_fp16_round_to_zero) {
                    ctx.AddCapability(spv::Capability::RoundingModeRTZ);
                    ctx.AddExecutionMode(main_func, spv::ExecutionMode::RoundingModeRTZ, 16U);
                } else {
                    static std::once_flag logged;
                    std::call_once(logged, [] {
                        LOG_WARNING(Render_Vulkan,
                                    "Float16 rounding to zero is not supported by the GPU");
                    });
                }
            }
            if (ctx.info.uses_fp64) {
                if (profile.support_fp64_round_to_zero) {
                    ctx.AddCapability(spv::Capability::RoundingModeRTZ);
                    ctx.AddExecutionMode(main_func, spv::ExecutionMode::RoundingModeRTZ, 64U);
                } else {
                    static std::once_flag logged;
                    std::call_once(logged, [] {
                        LOG_WARNING(Render_Vulkan,
                                    "Float64 rounding to zero is not supported by the GPU");
                    });
                }
            }
        } else if (fp16_64_round_mode != AmdGpu::FpRoundMode::NearestEven) {
            LOG_WARNING(Render_Vulkan, "Unknown Float16/64 rounding mode {}",
                        u32(fp16_64_round_mode));
        }
    }
}

void SetupInfNanPreserveMode(EmitContext& ctx, const Profile& profile,
                             const RuntimeInfo& runtime_info, Id main_func) {
    if (profile.support_fp16_signed_zero_inf_nan_preserve ||
        profile.support_fp32_signed_zero_inf_nan_preserve ||
        profile.support_fp64_signed_zero_inf_nan_preserve) {
        ctx.AddCapability(spv::Capability::SignedZeroInfNanPreserve);
    }

    if (profile.support_fp32_signed_zero_inf_nan_preserve) {
        ctx.AddExecutionMode(main_func, spv::ExecutionMode::SignedZeroInfNanPreserve, 32U);
    } else {
        static std::once_flag logged;
        std::call_once(logged, [] {
            LOG_WARNING(Render_Vulkan,
                        "Float32 signed zero/inf/nan preserve mode is not supported by the GPU");
        });
    }
    if (ctx.info.uses_fp16) {
        if (profile.support_fp16_signed_zero_inf_nan_preserve) {
            ctx.AddExecutionMode(main_func, spv::ExecutionMode::SignedZeroInfNanPreserve, 16U);
        } else {
            static std::once_flag logged;
            std::call_once(logged, [] {
                LOG_WARNING(
                    Render_Vulkan,
                    "Float16 signed zero/inf/nan preserve mode is not supported by the GPU");
            });
        }
    }
    if (ctx.info.uses_fp64) {
        if (profile.support_fp64_signed_zero_inf_nan_preserve) {
            ctx.AddExecutionMode(main_func, spv::ExecutionMode::SignedZeroInfNanPreserve, 64U);
        } else {
            static std::once_flag logged;
            std::call_once(logged, [] {
                LOG_WARNING(
                    Render_Vulkan,
                    "Float64 signed zero/inf/nan preserve mode is not supported by the GPU");
            });
        }
    }
}

void SetupFloatMode(EmitContext& ctx, const Profile& profile, const RuntimeInfo& runtime_info,
                    Id main_func) {
    ctx.AddExtension("SPV_KHR_float_controls");
    SetupDenormFlushMode(ctx, profile, runtime_info, main_func);
    SetupRoundingMode(ctx, profile, runtime_info, main_func);
    SetupInfNanPreserveMode(ctx, profile, runtime_info, main_func);
}

void PatchPhiNodes(const IR::Program& program, EmitContext& ctx) {
    auto inst{program.blocks.front()->begin()};
    size_t block_index{0};
    ctx.PatchDeferredPhi([&](u32 phi_arg, Id first_parent) {
        if (phi_arg == 0) {
            ++inst;
            if (inst == program.blocks[block_index]->end() ||
                inst->GetOpcode() != IR::Opcode::Phi) {
                do {
                    ++block_index;
                    inst = program.blocks[block_index]->begin();
                } while (inst->GetOpcode() != IR::Opcode::Phi);
            }
        }
        const Id arg = ctx.Def(inst->Arg(phi_arg));
        const Id parent = ctx.first_to_last_label_map[first_parent.value];
        return std::make_pair(arg, parent);
    });
}
} // Anonymous namespace

std::vector<u32> EmitSPIRV(const Profile& profile, const RuntimeInfo& runtime_info,
                           const IR::Program& program, Bindings& binding) {
    EmitContext ctx{profile, runtime_info, program.info, binding};
    const Id main{DefineMain(ctx, program)};
    DefineEntryPoint(program.info, ctx, main);
    SetupCapabilities(program.info, profile, runtime_info, ctx);
    SetupFloatMode(ctx, profile, runtime_info, main);
    PatchPhiNodes(program, ctx);
    binding.user_data += program.info.ud_mask.NumRegs();
    return ctx.Assemble();
}

Id EmitPhi(EmitContext& ctx, IR::Inst* inst) {
    const size_t num_args{inst->NumArgs()};
    boost::container::small_vector<Id, 32> blocks;
    blocks.reserve(num_args);
    for (size_t index = 0; index < num_args; ++index) {
        blocks.push_back(inst->PhiBlock(index)->Definition<Id>());
    }
    // The type of a phi instruction is stored in its flags
    const Id result_type{TypeId(ctx, inst->Flags<IR::Type>())};
    return ctx.DeferredOpPhi(result_type, std::span(blocks.data(), blocks.size()));
}

void EmitVoid(EmitContext&) {}

Id EmitIdentity(EmitContext& ctx, const IR::Value& value) {
    UNREACHABLE_MSG("Forward identity declaration");
}

Id EmitConditionRef(EmitContext& ctx, const IR::Value& value) {
    const Id id{ctx.Def(value)};
    ASSERT_MSG(Sirit::ValidId(id), "Forward identity declaration");
    return id;
}

void EmitReference(EmitContext&) {}

void EmitPhiMove(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetScc(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetExec(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetVcc(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetSccLo(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetVccLo(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetVccHi(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetM0(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetScc(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetExec(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetVcc(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetSccLo(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetVccLo(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetVccHi(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetM0(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

} // namespace Shader::Backend::SPIRV
