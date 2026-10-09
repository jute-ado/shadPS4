// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Shader::Backend::SPIRV {

[[nodiscard]] constexpr u32 LoopIterationLimit(u64 shader_hash) noexcept {
    // These shaders can enter data-dependent loops that exceed the Windows GPU watchdog. The
    // 0x4defcf7f loop consumes 16-byte records from a 0x160-byte constant buffer after a 16-byte
    // header, so 32 iterations remain above the maximum 21 complete records in a valid binding.
    return shader_hash == 0x4defcf7f ? 32 : 0;
}

[[nodiscard]] constexpr bool ShouldGuardDiagnosticLoop(u64 shader_hash, u32 loop_index,
                                                       u32 selected_loops) noexcept {
    constexpr u32 ShadowLoopCount = 7;
    return shader_hash == 0x87cf9b8a && loop_index < ShadowLoopCount &&
           (selected_loops & (u32{1} << loop_index)) != 0;
}

[[nodiscard]] constexpr bool ShouldDisableDiagnosticLoopUnrolling(
    u64 shader_hash, u32 loop_index, u32 selected_loops) noexcept {
    return ShouldGuardDiagnosticLoop(shader_hash, loop_index, selected_loops);
}

[[nodiscard]] constexpr bool ShouldDisableLoopUnrollingForDynamicScalarMask(
    bool materializes_dynamic_scalar_mask) noexcept {
    return materializes_dynamic_scalar_mask;
}

[[nodiscard]] constexpr u32 LoopPartialCountForDynamicScalarMask(
    bool materializes_dynamic_scalar_mask) noexcept {
    // Full unrolling of loops that repeatedly switch between scalar and ballot-backed EXEC
    // representations can make host shader compilation effectively unbounded. A small partial
    // count preserves a compact loop while still exposing enough consecutive iterations for the
    // driver to schedule the mask walk efficiently.
    return materializes_dynamic_scalar_mask ? 16 : 0;
}

} // namespace Shader::Backend::SPIRV
