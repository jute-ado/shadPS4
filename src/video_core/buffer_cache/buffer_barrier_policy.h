// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

enum class BufferShaderAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
};

[[nodiscard]] constexpr BufferShaderAccess GetBufferShaderAccess(bool is_written) noexcept {
    return is_written ? BufferShaderAccess::ReadWrite : BufferShaderAccess::ReadOnly;
}

[[nodiscard]] constexpr bool NeedsBufferBarrier(bool same_access, bool same_stage,
                                                bool source_access_writes) noexcept {
    return !same_access || !same_stage || source_access_writes;
}

[[nodiscard]] constexpr bool NeedsGpuModifiedReadBarrier(bool gpu_modified, bool shader_writes,
                                                         bool write_dependency_covered) noexcept {
    return gpu_modified && !shader_writes && !write_dependency_covered;
}

[[nodiscard]] constexpr bool ShouldSnapshotGpuModifiedComputeControl(
    bool gpu_modified, bool is_compute, bool shader_writes, bool is_formatted, u64 size) noexcept {
    constexpr u64 MaxSnapshotSize = 512;
    return gpu_modified && is_compute && !shader_writes && !is_formatted &&
           size <= MaxSnapshotSize;
}

} // namespace VideoCore
