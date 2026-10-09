// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Shader::Backend::SPIRV {

[[nodiscard]] constexpr bool ShouldUseZeroForInactiveReadLane(
    const u64 shader_hash, const bool diagnostic_enabled) noexcept {
    return diagnostic_enabled && shader_hash == 0x87cf9b8a;
}

} // namespace Shader::Backend::SPIRV
