// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "common/types.h"
#include "video_core/amdgpu/regs_vertex.h"

namespace VideoCore {

[[nodiscard]] u32 QuadListTriangleIndexCount(u32 input_count);

struct QuadListDrawPlan {
    bool convert{};
    bool indexed{};
    AmdGpu::PrimitiveType primitive_type{};
    u32 index_count{};
};

[[nodiscard]] bool ShouldConvertQuadListToTriangles(
    AmdGpu::PrimitiveType primitive_type, AmdGpu::ShaderStageEnable::VgtStages shader_stages,
    bool primitive_restart_enabled);

[[nodiscard]] QuadListDrawPlan PlanQuadListDraw(
    AmdGpu::PrimitiveType primitive_type, AmdGpu::ShaderStageEnable::VgtStages shader_stages,
    bool primitive_restart_enabled, bool indexed, u32 index_count);

void GenerateQuadListTriangleIndices(std::span<u32> output, u32 input_count);

void ConvertQuadListTriangleIndices(std::span<u16> output, std::span<const u16> input);

void ConvertQuadListTriangleIndices(std::span<u32> output, std::span<const u32> input);

} // namespace VideoCore
