// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include <mutex>
#include <tuple>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;

enum class DiagnosticCheckpointKind : u32 {
    Compute,
    Graphics,
};

struct DiagnosticCheckpointMarker {
    static constexpr u32 Magic = 0x53484443;

    u32 magic{Magic};
    DiagnosticCheckpointKind kind{};
    u64 first_hash{};
    u64 second_hash{};
    std::array<u32, 8> runtime{};
};

inline const DiagnosticCheckpointMarker* MakeDiagnosticCheckpoint(
    DiagnosticCheckpointKind kind, u64 first_hash, u64 second_hash = 0) {
    static std::mutex mutex;
    static std::map<std::tuple<u32, u64, u64>, DiagnosticCheckpointMarker> markers;

    const std::scoped_lock lock{mutex};
    const auto key = std::tuple{static_cast<u32>(kind), first_hash, second_hash};
    const auto [it, inserted] = markers.try_emplace(
        key, DiagnosticCheckpointMarker{.kind = kind,
                                        .first_hash = first_hash,
                                        .second_hash = second_hash});
    return &it->second;
}

void LogDiagnosticCheckpoints(const Instance& instance);
void RecordDiagnosticCommand(const DiagnosticCheckpointMarker& marker);
void InitializeDiagnosticGpuMarkers(const Instance& instance);
void RecordDiagnosticGpuMarker(vk::CommandBuffer cmdbuf,
                               const DiagnosticCheckpointMarker& marker);
void SealDiagnosticSubmission(u64 signal_value);
void LogDiagnosticCheckpoints(const Instance& instance, u64 last_known_tick);

} // namespace Vulkan
