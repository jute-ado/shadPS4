// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/structured_control_flow.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace Shader {

IR::BlockList GenerateBlocks(const IR::AbstractSyntaxList& syntax_list) {
    size_t num_syntax_blocks{};
    for (const auto& [_, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            ++num_syntax_blocks;
        }
    }
    IR::BlockList blocks{};
    blocks.reserve(num_syntax_blocks);
    for (const auto& [data, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            blocks.push_back(data.block);
        }
    }
    return blocks;
}

IR::Program TranslateProgram(const std::span<const u32>& code, Pools& pools, Info& info,
                             RuntimeInfo& runtime_info, const Profile& profile,
                             SrtMemoryReader reader) {
    // Ensure first instruction is expected.
    constexpr u32 token_mov_vcchi = 0xBEEB03FF;
    if (code[0] != token_mov_vcchi) {
        LOG_WARNING(Render_Recompiler, "First instruction is not s_mov_b32 vcc_hi, #imm");
    }

    Gcn::GcnCodeSlice slice(code.data(), code.data() + code.size());
    Gcn::GcnDecodeContext decoder;

    // Decode and save instructions
    IR::Program program{info};
    program.ins_list.reserve(code.size());
    while (!slice.atEnd()) {
        program.ins_list.emplace_back(decoder.decodeInstruction(slice));
    }

    // Clear any previous pooled data.
    pools.ReleaseContents();

    // Create control flow graph
    Common::ObjectPool<Gcn::Block> gcn_block_pool{64};
    Gcn::CFG cfg{gcn_block_pool, program.ins_list};

    // Structurize control flow graph and create program.
    program.syntax_list =
        Shader::Gcn::BuildASL(pools.inst_pool, pools.block_pool, cfg, info, runtime_info, profile);
    program.blocks = GenerateBlocks(program.syntax_list);
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());

    // On NVIDIA GPUs HW interpolation of clip distance values seems broken, and we need to emulate
    // it with expensive discard in PS.
    Shader::InjectClipDistanceAttributes(program, runtime_info);

    // Run optimization passes
    if (!profile.support_float64) {
        Shader::Optimization::LowerFp64ToFp32(program);
    }
    const u64 compute_thread_count =
        info.stage == Stage::Compute
            ? static_cast<u64>(runtime_info.cs_info.workgroup_size[0]) *
                  runtime_info.cs_info.workgroup_size[1] * runtime_info.cs_info.workgroup_size[2]
            : 0;
    // The synthetic whole-wave dispatcher now emits valid SPIR-V, but its guest state-machine
    // semantics are not yet visually correct for complex nested EXEC control. Keep production
    // fail-closed until the Shadow foreground oracle is coherent; the lowering remains directly
    // exercised by focused tests.
    constexpr bool EnableWave64Dispatcher = false;
    // Keep exact mask capture independent from scalar branch synchronization. Capturing saved
    // EXEC/VCC words is a larger state transformation; the smaller wave64 vote + exec-aware
    // ReadFirstLane path can be validated on its own.
    constexpr bool CaptureSplitWave64Masks = false;
    // Production remains fail-closed. Candidate runs may temporarily add an exact-shader oracle
    // here, but no title/hash selector is promoted.
    constexpr bool EnableSplitWave64ScalarControl = false;
    constexpr u64 ShadowForegroundWave64Oracle = 0x91c6c95eULL;
    const bool enable_split_wave64_scalar_control =
        EnableSplitWave64ScalarControl && info.pgm_hash == ShadowForegroundWave64Oracle &&
        info.stage == Stage::Compute && profile.subgroup_size == 32 &&
        compute_thread_count == 64;
    constexpr bool EnableWave64ReadFirstLaneOracle = false;
    const bool enable_wave64_read_first_lane =
        EnableWave64ReadFirstLaneOracle && info.pgm_hash == ShadowForegroundWave64Oracle &&
        info.stage == Stage::Compute && profile.subgroup_size == 32 &&
        compute_thread_count == 64;
    const bool predicate_vector_by_exec =
        EnableWave64Dispatcher && enable_split_wave64_scalar_control;
    Shader::Optimization::SsaRewritePass(program, predicate_vector_by_exec,
                                         CaptureSplitWave64Masks);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::IdentityRemovalPass(program.blocks);
    if (info.l_stage == LogicalStage::TessellationControl) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::HullShaderTransform(program, runtime_info);
    } else if (info.l_stage == LogicalStage::TessellationEval) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::DomainShaderTransform(program, runtime_info);
    }
    Shader::Optimization::RingAccessElimination(program, runtime_info);
    Shader::Optimization::ReadLaneEliminationPass(program);
    Shader::Optimization::FlattenExtendedUserdataPass(program, reader);
    Shader::Optimization::ResourceTrackingPass(program, profile);
    // Resolve descriptor provenance before wave64 exchanges turn cross-lane VGPR values into
    // shared-memory loads, which intentionally no longer retain their original sharp source.
    Shader::Optimization::Wave64SubgroupLoweringPass(program, runtime_info, profile,
                                                      predicate_vector_by_exec,
                                                      enable_split_wave64_scalar_control,
                                                      enable_wave64_read_first_lane);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::Wave64ProtocolCleanupPass(program);
    Shader::Optimization::LowerBufferFormatToRaw(program);
    Shader::Optimization::SharedMemorySimplifyPass(program, profile);
    Shader::Optimization::SharedMemoryToStoragePass(program, runtime_info, profile);
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);
    Shader::Optimization::IdentityRemovalPass(program.blocks);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::Wave64ProtocolCleanupPass(program);
    Shader::Optimization::BufferAccessRangePass(program);
    Shader::Optimization::LowerUserClipPlanes(program, runtime_info);
    if (program.info.emulate_wave64_control_flow) {
        // The dispatcher backend virtualizes every value because its synthetic switch erases the
        // original CFG dominance. Keep SSA destruction here limited to phis; performing a second
        // cross-block spill creates redundant path-dependent register state.
        Shader::Optimization::SsaDestroyPass(program);
    }
    Shader::Optimization::CollectShaderInfoPass(program, profile);

    Shader::IR::DumpProgram(program, info);

    return program;
}

} // namespace Shader
