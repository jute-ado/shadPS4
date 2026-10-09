// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/ir/value.h"

namespace Shader {

[[nodiscard]] inline bool IsContractingAddSub(IR::Opcode opcode) noexcept {
    return opcode == IR::Opcode::FPAdd32 || opcode == IR::Opcode::FPAdd64 ||
           opcode == IR::Opcode::FPSub32;
}

[[nodiscard]] inline bool RequiresSeparateMultiplyRounding(const IR::Inst* multiply) noexcept {
    for (const auto& use : multiply->Uses()) {
        const auto opcode = use.user->GetOpcode();
        if (IsContractingAddSub(opcode)) {
            return true;
        }
        if (opcode == IR::Opcode::FPNeg32 || opcode == IR::Opcode::FPNeg64) {
            for (const auto& negate_use : use.user->Uses()) {
                if (IsContractingAddSub(negate_use.user->GetOpcode())) {
                    return true;
                }
            }
        }
    }
    return false;
}

} // namespace Shader
