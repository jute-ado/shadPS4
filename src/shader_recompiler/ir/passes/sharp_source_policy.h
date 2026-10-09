// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Shader::Optimization {

enum class SharpSourceKind : u8 {
    UserData,
    Flattened,
    Dynamic,
};

struct SharpSource {
    SharpSourceKind kind{};
    u32 location{};

    bool operator==(const SharpSource&) const = default;
};

[[nodiscard]] constexpr SharpSource ClassifySharpSource(const bool is_user_data,
                                                        const u32 location) {
    if (is_user_data) {
        return {SharpSourceKind::UserData, location};
    }
    if (location == 0) {
        return {SharpSourceKind::Dynamic, 0};
    }
    return {SharpSourceKind::Flattened, location};
}

} // namespace Shader::Optimization
