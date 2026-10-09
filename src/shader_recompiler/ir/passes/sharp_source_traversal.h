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

// Descriptor SSA can merge through a Phi or through a conditional scalar-register restore.
// The Select condition is control metadata, so only its two value operands can contain sharps.
constexpr SharpSourceOperandRange SharpSourceTraversalRange(IR::Opcode opcode,
                                                            size_t num_args) {
    switch (opcode) {
    case IR::Opcode::Phi:
        return {0, num_args};
    case IR::Opcode::SelectU32:
        return {1, 3};
    case IR::Opcode::MaterializeScalarMaskWord:
        // Only the scalar fallback can carry a descriptor. The predicate, thread bit, and mask
        // word index are representation metadata.
        return {2, 3};
    default:
        return {};
    }
}

} // namespace Shader::Optimization
