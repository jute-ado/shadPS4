// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>

#include "shader_recompiler/ir/opcodes.h"

namespace Shader::Optimization {

struct SharpSourceOperandRange {
    size_t begin{};
    size_t end{};
};

// Descriptor SSA can pass through ordinary value selection or through a conditional conversion
// from the flattened per-invocation mask representation. Predicate/control operands cannot carry
// a descriptor, so follow only the value operands that preserve the original sharp lineage.
constexpr SharpSourceOperandRange SharpSourceTraversalRange(IR::Opcode opcode, size_t num_args) {
    switch (opcode) {
    case IR::Opcode::Phi:
        return {0, num_args};
    case IR::Opcode::SelectU32:
        return {1, 3};
    case IR::Opcode::MaterializeScalarMaskWord:
        return {2, 3};
    default:
        return {};
    }
}

} // namespace Shader::Optimization
