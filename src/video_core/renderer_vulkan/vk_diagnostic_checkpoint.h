// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <map>
#include <mutex>
#include <tuple>

#include "common/types.h"

namespace Vulkan {

enum class DiagnosticCheckpointKind : u32 {
    Compute,
    Graphics,
};

enum class DiagnosticCheckpointPhase : u32 {
    Before,
    After,
};

struct DiagnosticCheckpointMarker {
    static constexpr u32 Magic = 0x53484443;

    u32 magic{Magic};
    DiagnosticCheckpointKind kind{};
    DiagnosticCheckpointPhase phase{};
    u64 first_hash{};
    u64 second_hash{};
    std::array<u32, 3> runtime{};
};

inline const DiagnosticCheckpointMarker* MakeDiagnosticCheckpoint(
    const DiagnosticCheckpointKind kind, const u64 first_hash, const u64 second_hash = 0,
    const DiagnosticCheckpointPhase phase = DiagnosticCheckpointPhase::Before,
    const std::array<u32, 3>& runtime = {}) {
    static std::mutex mutex;
    static std::map<std::tuple<u32, u32, u64, u64, u32, u32, u32>, DiagnosticCheckpointMarker>
        markers;

    const std::scoped_lock lock{mutex};
    const auto key = std::tuple{static_cast<u32>(kind), static_cast<u32>(phase), first_hash,
                                second_hash, runtime[0], runtime[1], runtime[2]};
    const auto [it, inserted] = markers.try_emplace(
        key, DiagnosticCheckpointMarker{.kind = kind,
                                        .phase = phase,
                                        .first_hash = first_hash,
                                        .second_hash = second_hash,
                                        .runtime = runtime});
    return &it->second;
}

} // namespace Vulkan
