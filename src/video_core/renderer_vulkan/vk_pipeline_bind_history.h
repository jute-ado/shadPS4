// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <mutex>
#include <span>
#include <string_view>
#include <vector>

#include "common/types.h"

namespace Vulkan {

enum class PipelineBindType : u8 {
    Graphics,
    Compute,
};

enum class PipelineCommandType : u8 {
    Unknown,
    Draw,
    DrawIndexed,
    DrawIndirect,
    DrawIndexedIndirect,
    Dispatch,
    DispatchIndirect,
};

struct PipelineCommandInfo {
    PipelineCommandType type{};
    std::array<u64, 3> arguments{};

    bool operator==(const PipelineCommandInfo&) const = default;
};

constexpr std::string_view PipelineCommandName(PipelineCommandType type) {
    switch (type) {
    case PipelineCommandType::Draw:
        return "draw";
    case PipelineCommandType::DrawIndexed:
        return "draw-indexed";
    case PipelineCommandType::DrawIndirect:
        return "draw-indirect";
    case PipelineCommandType::DrawIndexedIndirect:
        return "draw-indexed-indirect";
    case PipelineCommandType::Dispatch:
        return "dispatch";
    case PipelineCommandType::DispatchIndirect:
        return "dispatch-indirect";
    default:
        return "unknown";
    }
}

struct PipelineBindRecord {
    static constexpr size_t MaxShaderHashes = 6;

    PipelineBindType type{};
    u64 pipeline_hash{};
    std::array<u64, MaxShaderHashes> shader_hashes{};
    PipelineCommandInfo command{};

    bool operator==(const PipelineBindRecord&) const = default;
};

inline PipelineBindRecord MakePipelineBindRecord(PipelineBindType type, u64 pipeline_hash,
                                                 std::span<const u64> shader_hashes,
                                                 PipelineCommandInfo command = {}) {
    PipelineBindRecord record{
        .type = type,
        .pipeline_hash = pipeline_hash,
        .command = command,
    };
    std::ranges::copy(
        shader_hashes.first(std::min(shader_hashes.size(), record.shader_hashes.size())),
        record.shader_hashes.begin());
    return record;
}

class PipelineBindHistory {
public:
    static constexpr size_t Capacity = 64;

    void Record(const PipelineBindRecord& record) {
        std::scoped_lock lock{mutex};
        records[next] = record;
        next = (next + 1) % Capacity;
        count = std::min(count + 1, Capacity);
    }

    [[nodiscard]] std::vector<PipelineBindRecord> RecentUnique() const {
        std::scoped_lock lock{mutex};
        std::vector<PipelineBindRecord> result;
        result.reserve(count);
        for (size_t offset = 0; offset < count; ++offset) {
            const size_t index = (next + Capacity - 1 - offset) % Capacity;
            const auto& candidate = records[index];
            const bool already_present = std::ranges::any_of(result, [&](const auto& existing) {
                return existing.type == candidate.type &&
                       existing.pipeline_hash == candidate.pipeline_hash;
            });
            if (!already_present) {
                result.push_back(candidate);
            }
        }
        return result;
    }

private:
    mutable std::mutex mutex;
    std::array<PipelineBindRecord, Capacity> records{};
    size_t next{};
    size_t count{};
};

inline PipelineBindHistory& GetPipelineBindHistory() {
    static PipelineBindHistory history;
    return history;
}

} // namespace Vulkan
