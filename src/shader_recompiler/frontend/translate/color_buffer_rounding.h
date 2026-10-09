// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Gcn {

[[nodiscard]] inline IR::F32 ApplyColorBufferRoundMode(IR::IREmitter& ir,
                                                       const IR::F32& value,
                                                       const PsColorBuffer& color_buffer) {
    const bool unorm8 = color_buffer.num_format == AmdGpu::NumberFormat::Unorm &&
                        color_buffer.num_conversion == AmdGpu::NumberConversion::None &&
                        (color_buffer.data_format == AmdGpu::DataFormat::Format8 ||
                         color_buffer.data_format == AmdGpu::DataFormat::Format8_8 ||
                         color_buffer.data_format == AmdGpu::DataFormat::Format8_8_8_8);
    if (!unorm8) {
        return value;
    }

    const IR::F32 scale = ir.Imm32(255.0f);
    IR::F32 scaled = ir.FPMul(ir.FPClamp(value, ir.Imm32(0.0f), ir.Imm32(1.0f)), scale);
    if (!color_buffer.truncate) {
        scaled = ir.FPAdd(scaled, ir.Imm32(0.5f));
    }
    return ir.FPDiv(ir.FPFloor(scaled), scale);
}

} // namespace Shader::Gcn
