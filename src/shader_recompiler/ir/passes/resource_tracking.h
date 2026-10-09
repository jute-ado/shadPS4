// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "shader_recompiler/ir/basic_block.h"
namespace Shader::Optimization::Detail {
inline const IR::Inst* ReconstructedScalarSource(const IR::Inst* inst) {
    if (inst->GetOpcode() == IR::Opcode::BitwiseOr32) {
        const IR::Inst* low = inst->Arg(0).TryInstRecursive();
        const IR::Inst* high = inst->Arg(1).TryInstRecursive();
        const IR::Inst* low_source = low != nullptr ? ReconstructedScalarSource(low) : nullptr;
        const IR::Inst* high_source = high != nullptr ? ReconstructedScalarSource(high) : nullptr;
        return low_source == high_source ? low_source : nullptr;
    }
    if (inst->GetOpcode() != IR::Opcode::CompositeExtractU32x4)
        return nullptr;
    const IR::Inst* ballot = inst->Arg(0).TryInstRecursive();
    if (!ballot || ballot->GetOpcode() != IR::Opcode::Ballot)
        return nullptr;
    const IR::Inst* predicate = ballot->Arg(0).TryInstRecursive();
    if (predicate && predicate->GetOpcode() == IR::Opcode::SelectU1)
        predicate = predicate->Arg(2).TryInstRecursive();
    if (!predicate || predicate->GetOpcode() != IR::Opcode::INotEqual32)
        return nullptr;
    const IR::Inst* masked = predicate->Arg(0).TryInstRecursive();
    if (!masked || masked->GetOpcode() != IR::Opcode::BitwiseAnd32)
        return nullptr;
    const IR::Inst* shifted = masked->Arg(0).TryInstRecursive();
    if (!shifted || shifted->GetOpcode() != IR::Opcode::ShiftRightLogical32)
        return nullptr;
    return shifted->Arg(0).TryInstRecursive();
}
} // namespace Shader::Optimization::Detail
