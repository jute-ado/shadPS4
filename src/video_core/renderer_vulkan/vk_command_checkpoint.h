// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <optional>

#include "common/types.h"

namespace Vulkan {

enum class CommandCheckpointType : u64 {
    Draw = 1,
    DrawIndexed,
    DrawIndirect,
    DrawIndexedIndirect,
    Dispatch,
    DispatchIndirect,
};

struct CommandCheckpoint {
    u64 sequence{};
    CommandCheckpointType type{};
    u64 pipeline_hash{};
    std::array<u64, 6> shader_hashes{};
    std::array<u64, 6> arguments{};
};

namespace CommandCheckpointDetail {

constexpr size_t HistorySize = 4096;

struct AtomicCommandCheckpoint {
    std::atomic<u64> sequence{};
    std::atomic<u64> type{};
    std::atomic<u64> pipeline_hash{};
    std::array<std::atomic<u64>, 6> shader_hashes{};
    std::array<std::atomic<u64>, 6> arguments{};
};

inline std::atomic<u64> next_sequence{1};
inline std::array<AtomicCommandCheckpoint, HistorySize> history{};

} // namespace CommandCheckpointDetail

inline const void* RecordCommandCheckpoint(CommandCheckpointType type, u64 pipeline_hash,
                                           std::array<u64, 6> shader_hashes,
                                           std::array<u64, 6> arguments) {
    using namespace CommandCheckpointDetail;
    const u64 sequence = next_sequence.fetch_add(1, std::memory_order_relaxed);
    auto& slot = history[sequence % HistorySize];
    slot.type.store(static_cast<u64>(type), std::memory_order_relaxed);
    slot.pipeline_hash.store(pipeline_hash, std::memory_order_relaxed);
    for (size_t index = 0; index < shader_hashes.size(); ++index) {
        slot.shader_hashes[index].store(shader_hashes[index], std::memory_order_relaxed);
    }
    for (size_t index = 0; index < arguments.size(); ++index) {
        slot.arguments[index].store(arguments[index], std::memory_order_relaxed);
    }
    slot.sequence.store(sequence, std::memory_order_release);
    return reinterpret_cast<const void*>(static_cast<uintptr_t>(sequence));
}

inline std::optional<CommandCheckpoint> FindCommandCheckpoint(const void* marker) {
    using namespace CommandCheckpointDetail;
    const u64 sequence = static_cast<u64>(reinterpret_cast<uintptr_t>(marker));
    if (sequence == 0) {
        return std::nullopt;
    }
    auto& slot = history[sequence % HistorySize];
    if (slot.sequence.load(std::memory_order_acquire) != sequence) {
        return std::nullopt;
    }
    CommandCheckpoint result{
        .sequence = sequence,
        .type = static_cast<CommandCheckpointType>(slot.type.load(std::memory_order_relaxed)),
        .pipeline_hash = slot.pipeline_hash.load(std::memory_order_relaxed),
    };
    for (size_t index = 0; index < result.shader_hashes.size(); ++index) {
        result.shader_hashes[index] =
            slot.shader_hashes[index].load(std::memory_order_relaxed);
    }
    for (size_t index = 0; index < result.arguments.size(); ++index) {
        result.arguments[index] = slot.arguments[index].load(std::memory_order_relaxed);
    }
    if (slot.sequence.load(std::memory_order_acquire) != sequence) {
        return std::nullopt;
    }
    return result;
}

} // namespace Vulkan
