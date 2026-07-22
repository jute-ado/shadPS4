// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

#include "common/types.h"

namespace Vulkan {

constexpr bool ShouldEnableDiagnosticCheckpoints(bool native_checkpoints_requested,
                                                 bool extension_available) {
    return native_checkpoints_requested && extension_available;
}

constexpr bool ShouldEnableCrashDiagnosticLayer(bool crash_diagnostics_requested,
                                                bool native_checkpoints_requested) {
    return crash_diagnostics_requested && !native_checkpoints_requested;
}

inline const void* EncodeDiagnosticCheckpoint(u64 pipeline_hash) {
    static_assert(sizeof(uintptr_t) >= sizeof(pipeline_hash));
    return reinterpret_cast<const void*>(static_cast<uintptr_t>(pipeline_hash));
}

inline u64 DecodeDiagnosticCheckpoint(const void* marker) {
    return static_cast<u64>(reinterpret_cast<uintptr_t>(marker));
}

} // namespace Vulkan
