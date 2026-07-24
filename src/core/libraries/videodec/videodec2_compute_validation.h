// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/error_codes.h"
#include "core/libraries/videodec/videodec2.h"
#include "core/libraries/videodec/videodec_error.h"

namespace Libraries::Videodec2 {

inline constexpr u64 Videodec2ComputeMemorySize = 16_MB;

[[nodiscard]] constexpr s32 ValidateComputeQueueArguments(
    const OrbisVideodec2ComputeConfigInfo& config, const OrbisVideodec2ComputeMemoryInfo& memory) {
    if (config.thisSize != sizeof(OrbisVideodec2ComputeConfigInfo) ||
        memory.thisSize != sizeof(OrbisVideodec2ComputeMemoryInfo)) {
        return ORBIS_VIDEODEC2_ERROR_STRUCT_SIZE;
    }
    if (config.reserved0 != 0 || config.reserved1 != 0) {
        return ORBIS_VIDEODEC2_ERROR_CONFIG_INFO;
    }
    if (config.computePipeId > 4) {
        return ORBIS_VIDEODEC2_ERROR_COMPUTE_PIPE_ID;
    }
    if (config.computeQueueId > 7) {
        return ORBIS_VIDEODEC2_ERROR_COMPUTE_QUEUE_ID;
    }
    if (!memory.cpuGpuMemory) {
        return ORBIS_VIDEODEC2_ERROR_MEMORY_POINTER;
    }
    if (memory.cpuGpuMemorySize < Videodec2ComputeMemorySize) {
        return ORBIS_VIDEODEC2_ERROR_MEMORY_SIZE;
    }
    return ORBIS_OK;
}

} // namespace Libraries::Videodec2
