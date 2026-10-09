// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/amdgpu/pixel_format.h"

namespace Shader::Optimization {

[[nodiscard]] constexpr u32 SelectBufferDataFormat(bool is_instruction_typed,
                                                    u32 instruction_format,
                                                    u32 descriptor_format) {
    return is_instruction_typed ? instruction_format : descriptor_format;
}

[[nodiscard]] constexpr u32 SelectBufferNumberFormat(bool is_instruction_typed,
                                                      u32 instruction_format,
                                                      u32 descriptor_format) {
    return is_instruction_typed ? instruction_format : descriptor_format;
}

[[nodiscard]] constexpr u32 NormalizeBufferDataFormat(u32 data_format) {
    // The raw-buffer lowering pass supports the contiguous typed-buffer formats
    // 0..14. Encoding 15 is reserved, while 16..63 are image/resource formats
    // without a corresponding MTBUF component layout here.
    return data_format < 15 ? data_format : 0;
}

[[nodiscard]] constexpr bool IsPacked10_11_11SnormNz(AmdGpu::DataFormat data_format,
                                                      AmdGpu::NumberFormat number_format) {
    return AmdGpu::RemapDataFormat(data_format) == AmdGpu::DataFormat::Format10_11_11 &&
           number_format == AmdGpu::NumberFormat::SnormNz;
}

[[nodiscard]] constexpr u32 Packed10_11_11SnormNzMax(u32 component) {
    return component < 2 ? 2047u : 1023u;
}

[[nodiscard]] constexpr bool ShouldLowerBufferFormatStore(u32 data_format) {
    return data_format != 0;
}

[[nodiscard]] constexpr bool ShouldMapBufferNumberConversion(u32 data_format) {
    return data_format != 0;
}

[[nodiscard]] constexpr bool ShouldApplyBufferThreadId(bool add_thread_id,
                                                       bool is_compute_stage) {
    return add_thread_id && is_compute_stage;
}

} // namespace Shader::Optimization
