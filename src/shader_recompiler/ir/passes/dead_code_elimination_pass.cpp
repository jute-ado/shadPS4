// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_set>
#include <vector>
#include "shader_recompiler/ir/program.h"

namespace Shader::Optimization {

void DeadCodeEliminationPass(IR::Program& program, bool keep_exec_markers) {
    // Use counts cannot distinguish observable values from mutually used dead Phi cycles.
    // Mark dependencies from observable roots instead, including control stored outside IR uses.
    std::unordered_set<IR::Inst*> live;
    std::vector<IR::Inst*> pending;
    const auto mark_inst = [&](IR::Inst* inst) {
        if (live.insert(inst).second) {
            pending.push_back(inst);
        }
    };
    const auto mark_value = [&](const IR::Value& value) {
        // IsImmediate unwraps Identity. Preserve the exact node even when it resolves to a
        // constant: consumers can still hold that Identity pointer without a tracked use.
        if (value.IsIdentity() || !value.IsImmediate()) {
            mark_inst(value.Inst());
        }
    };
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : *block) {
            const bool is_exec_marker =
                inst.GetOpcode() == IR::Opcode::GetExec || inst.GetOpcode() == IR::Opcode::SetExec;
            if (inst.MayHaveSideEffects() || (keep_exec_markers && is_exec_marker)) {
                mark_inst(&inst);
            }
        }
    }
    for (const auto& node : program.syntax_list) {
        switch (node.type) {
        case IR::AbstractSyntaxNode::Type::If:
            mark_value(node.data.if_node.cond);
            break;
        case IR::AbstractSyntaxNode::Type::Repeat:
            mark_value(node.data.repeat.cond);
            break;
        case IR::AbstractSyntaxNode::Type::Break:
            mark_value(node.data.break_node.cond);
            break;
        default:
            break;
        }
    }
    while (!pending.empty()) {
        IR::Inst* const inst = pending.back();
        pending.pop_back();
        for (size_t index = 0; index < inst->NumArgs(); ++index) {
            mark_value(inst->Arg(index));
        }
    }

    // IsImmediate recursively unwraps Identity, so these edges may never have had tracked uses.
    // Resolve dead-node operands (including Phi operands) while every Identity is intact. Otherwise
    // invalidating a constant Identity first makes its consumer attempt an untracked UndoUse.
    // Live Identity nodes and externally held ASL conditions remain unchanged.
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : *block) {
            if (!live.contains(&inst)) {
                for (size_t index = 0; index < inst.NumArgs(); ++index) {
                    const auto arg = inst.Arg(index);
                    if (arg.IsIdentity()) {
                        inst.SetArg(index, arg.Resolve());
                    }
                }
            }
        }
    }

    // Detach all dead operands while every instruction is still in its block. Only then
    // unlink dead nodes, so cyclic uses and references across blocks are decremented safely.
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : *block) {
            if (!live.contains(&inst)) {
                inst.Invalidate();
            }
        }
    }
    for (IR::Block* const block : program.blocks) {
        auto it = block->begin();
        while (it != block->end()) {
            if (!live.contains(&*it)) {
                it = block->Instructions().erase(it);
            } else {
                ++it;
            }
        }
    }
}

} // namespace Shader::Optimization
