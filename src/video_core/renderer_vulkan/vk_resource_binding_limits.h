// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"

namespace Vulkan::ResourceBindingLimits {

constexpr u32 MaxGraphicsStages = Shader::MaxStageTypes - 1;
constexpr u32 MaxPipelineImageBindings = MaxGraphicsStages * Shader::NUM_IMAGE_BINDINGS;
constexpr u32 MaxPipelineImageInfos =
    MaxGraphicsStages * (Shader::NUM_IMAGE_BINDINGS + Shader::NUM_SAMPLERS);

} // namespace Vulkan::ResourceBindingLimits
