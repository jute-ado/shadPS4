// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include "video_core/quad_list_indices.h"

namespace VideoCore {
namespace {

template <typename T>
void ConvertQuadListTriangleIndicesImpl(std::span<T> output, std::span<const T> input) {
    const size_t quad_count = input.size() / 4;
    if (quad_count > output.size() / 6) {
        return;
    }

    for (size_t quad = 0; quad < quad_count; ++quad) {
        const size_t input_base = quad * 4;
        const size_t output_base = quad * 6;
        output[output_base + 0] = input[input_base + 0];
        output[output_base + 1] = input[input_base + 1];
        output[output_base + 2] = input[input_base + 2];
        output[output_base + 3] = input[input_base + 0];
        output[output_base + 4] = input[input_base + 2];
        output[output_base + 5] = input[input_base + 3];
    }
}

} // Anonymous namespace

u32 QuadListTriangleIndexCount(u32 input_count) {
    const u64 output_count = static_cast<u64>(input_count / 4) * 6;
    return output_count <= std::numeric_limits<u32>::max() ? static_cast<u32>(output_count) : 0;
}

bool ShouldConvertQuadListToTriangles(
    AmdGpu::PrimitiveType primitive_type, AmdGpu::ShaderStageEnable::VgtStages shader_stages,
    bool primitive_restart_enabled) {
    return primitive_type == AmdGpu::PrimitiveType::QuadList &&
           shader_stages == AmdGpu::ShaderStageEnable::Vs && !primitive_restart_enabled;
}

QuadListDrawPlan PlanQuadListDraw(AmdGpu::PrimitiveType primitive_type,
                                  AmdGpu::ShaderStageEnable::VgtStages shader_stages,
                                  bool primitive_restart_enabled, bool indexed, u32 index_count) {
    const bool convert = ShouldConvertQuadListToTriangles(
        primitive_type, shader_stages, primitive_restart_enabled);
    return {
        .convert = convert,
        .indexed = indexed || convert,
        .primitive_type = convert ? AmdGpu::PrimitiveType::TriangleList : primitive_type,
        .index_count = convert ? QuadListTriangleIndexCount(index_count) : index_count,
    };
}

void GenerateQuadListTriangleIndices(std::span<u32> output, u32 input_count) {
    const u32 output_count = QuadListTriangleIndexCount(input_count);
    if (output.size() < output_count) {
        return;
    }

    for (u32 input_base = 0, output_base = 0; output_base < output_count;
         input_base += 4, output_base += 6) {
        output[output_base + 0] = input_base + 0;
        output[output_base + 1] = input_base + 1;
        output[output_base + 2] = input_base + 2;
        output[output_base + 3] = input_base + 0;
        output[output_base + 4] = input_base + 2;
        output[output_base + 5] = input_base + 3;
    }
}

void ConvertQuadListTriangleIndices(std::span<u16> output, std::span<const u16> input) {
    ConvertQuadListTriangleIndicesImpl(output, input);
}

void ConvertQuadListTriangleIndices(std::span<u32> output, std::span<const u32> input) {
    ConvertQuadListTriangleIndicesImpl(output, input);
}

} // namespace VideoCore
