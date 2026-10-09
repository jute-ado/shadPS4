// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

#include "common/types.h"

namespace Vulkan {

enum class DeviceLossSource {
    Presentation,
    Submission,
};

constexpr bool ShouldEnableDiagnosticCheckpoints(bool native_checkpoints_requested,
                                                 bool extension_available) {
    return native_checkpoints_requested && extension_available;
}

constexpr bool ShouldRecordPipelineBindDiagnostics(bool crash_diagnostics_requested,
                                                   bool native_checkpoints_requested) {
    return crash_diagnostics_requested || native_checkpoints_requested;
}

constexpr bool ShouldReportDiagnosticCheckpoints(DeviceLossSource /*source*/,
                                                 bool checkpoints_supported) {
    return checkpoints_supported;
}

constexpr bool ShouldEnableCrashDiagnosticLayer(bool crash_diagnostics_requested,
                                                bool native_checkpoints_requested) {
    return crash_diagnostics_requested && !native_checkpoints_requested;
}

inline const void* EncodeDiagnosticCheckpoint(u64 marker) {
    static_assert(sizeof(uintptr_t) >= sizeof(marker));
    return reinterpret_cast<const void*>(static_cast<uintptr_t>(marker));
}

inline u64 DecodeDiagnosticCheckpoint(const void* marker) {
    return static_cast<u64>(reinterpret_cast<uintptr_t>(marker));
}

// The marker is an opaque value to Vulkan. Native checkpoint diagnostics use it as a bounded
// command identity that can be resolved against PipelineBindHistory when a device is lost.
inline const void* EncodeDiagnosticCommandId(u64 command_id) {
    return EncodeDiagnosticCheckpoint(command_id);
}

inline u64 DecodeDiagnosticCommandId(const void* marker) {
    return DecodeDiagnosticCheckpoint(marker);
}

} // namespace Vulkan
