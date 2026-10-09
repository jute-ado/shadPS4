// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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
                             RuntimeInfo& runtime_info, const Profile& profile) {
    const bool trace_passes = std::getenv("SHADPS4_DIAGNOSTIC_SHADER_PASSES") != nullptr;
    const auto trace_pass = [&](const char* pass) {
        if (trace_passes) {
            LOG_INFO(Render_Recompiler, "TEMP shader pass hash={:#x} stage={} pass={}",
                     info.pgm_hash, info.stage, pass);
        }
    };
    // Ensure first instruction is expected.
    constexpr u32 token_mov_vcchi = 0xBEEB03FF;
    if (code[0] != token_mov_vcchi) {
        LOG_WARNING(Render_Recompiler, "First instruction is not s_mov_b32 vcc_hi, #imm");
    }

    trace_pass("decode");
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
    trace_pass("build-asl");
    program.syntax_list =
        Shader::Gcn::BuildASL(pools.inst_pool, pools.block_pool, cfg, info, runtime_info, profile);
    program.blocks = GenerateBlocks(program.syntax_list);
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());

    // On NVIDIA GPUs HW interpolation of clip distance values seems broken, and we need to emulate
    // it with expensive discard in PS.
    trace_pass("inject-clip-distance");
    Shader::InjectClipDistanceAttributes(program, runtime_info);

    // Run optimization passes
    if (!profile.support_float64) {
        trace_pass("lower-fp64");
        Shader::Optimization::LowerFp64ToFp32(program);
    }
    trace_pass("ssa-rewrite");
    Shader::Optimization::SsaRewritePass(program.post_order_blocks);
    trace_pass("constant-propagation-1");
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    trace_pass("identity-removal-1");
    Shader::Optimization::IdentityRemovalPass(program.blocks);
    if (info.l_stage == LogicalStage::TessellationControl) {
        trace_pass("tessellation-preprocess-control");
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        trace_pass("hull-transform");
        Shader::Optimization::HullShaderTransform(program, runtime_info);
    } else if (info.l_stage == LogicalStage::TessellationEval) {
        trace_pass("tessellation-preprocess-eval");
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        trace_pass("domain-transform");
        Shader::Optimization::DomainShaderTransform(program, runtime_info);
    }
    trace_pass("ring-access-elimination");
    Shader::Optimization::RingAccessElimination(program, runtime_info);
    trace_pass("read-lane-elimination");
    Shader::Optimization::ReadLaneEliminationPass(program);
    trace_pass("flatten-extended-userdata");
    Shader::Optimization::FlattenExtendedUserdataPass(program);
    if (info.pgm_hash == 0x862e4da8) {
        Shader::IR::DumpProgram(program, info, "after-flatten.");
    }
    trace_pass("resource-tracking");
    Shader::Optimization::ResourceTrackingPass(program, profile);
    trace_pass("lower-buffer-format");
    Shader::Optimization::LowerBufferFormatToRaw(program);
    trace_pass("shared-memory-simplify");
    Shader::Optimization::SharedMemorySimplifyPass(program, profile);
    trace_pass("shared-memory-to-storage");
    Shader::Optimization::SharedMemoryToStoragePass(program, runtime_info, profile);
    trace_pass("read-lane-workgroup");
    Shader::Optimization::ReadLaneWorkgroupPass(program, runtime_info, profile);
    trace_pass("shared-memory-barrier");
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);
    trace_pass("identity-removal-2");
    Shader::Optimization::IdentityRemovalPass(program.blocks);
    trace_pass("dead-code-elimination");
    Shader::Optimization::DeadCodeEliminationPass(program);
    trace_pass("constant-propagation-2");
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    trace_pass("buffer-access-range");
    Shader::Optimization::BufferAccessRangePass(program);
    trace_pass("lower-user-clip-planes");
    Shader::Optimization::LowerUserClipPlanes(program, runtime_info);
    trace_pass("collect-shader-info");
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    trace_pass("complete");

    Shader::IR::DumpProgram(program, info);

    return program;
}

} // namespace Shader
