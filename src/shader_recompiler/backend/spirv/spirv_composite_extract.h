// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "shader_recompiler/ir/type.h"

namespace Shader::Backend::SPIRV {

/// Returns whether a statically selected vector component can be emitted safely.
[[nodiscard]] constexpr bool IsValidCompositeExtractIndex(u32 vector_components,
                                                           u32 index) noexcept {
    return index < vector_components;
}

[[nodiscard]] constexpr u32 GetVectorComponentCount(IR::Type type) noexcept {
    switch (type) {
    case IR::Type::U32x2:
    case IR::Type::F16x2:
    case IR::Type::F32x2:
    case IR::Type::F64x2:
        return 2;
    case IR::Type::U32x3:
    case IR::Type::F16x3:
    case IR::Type::F32x3:
    case IR::Type::F64x3:
        return 3;
    case IR::Type::U32x4:
    case IR::Type::F16x4:
    case IR::Type::F32x4:
    case IR::Type::F64x4:
        return 4;
    default:
        return 0;
    }
}

} // namespace Shader::Backend::SPIRV
