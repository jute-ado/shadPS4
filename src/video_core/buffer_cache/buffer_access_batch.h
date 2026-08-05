// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <compare>
#include <span>
#include <vector>

#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

constexpr vk::AccessFlags2 ShaderBufferAccess(bool is_written) noexcept {
    vk::AccessFlags2 access = vk::AccessFlagBits2::eShaderRead;
    if (is_written) {
        access |= vk::AccessFlagBits2::eShaderWrite;
    }
    return access;
}

constexpr bool HasBufferWriteAccess(vk::AccessFlags2 access) noexcept {
    constexpr vk::AccessFlags2 WriteAccesses = vk::AccessFlagBits2::eShaderWrite |
                                               vk::AccessFlagBits2::eTransferWrite |
                                               vk::AccessFlagBits2::eMemoryWrite;
    return bool(access & WriteAccesses);
}

constexpr bool NeedsBufferBarrier(vk::AccessFlags2 source_access,
                                  vk::PipelineStageFlags2 source_stages,
                                  vk::AccessFlags2 destination_access,
                                  vk::PipelineStageFlags2 destination_stages) noexcept {
    return source_access != destination_access || source_stages != destination_stages ||
           HasBufferWriteAccess(source_access) || HasBufferWriteAccess(destination_access);
}

template <typename Key>
class BasicBufferAccessBatch {
public:
    struct Entry {
        Key key;
        vk::AccessFlags2 access;
        vk::PipelineStageFlags2 stages;

        auto operator<=>(const Entry&) const = default;
    };

    void Add(Key key, vk::AccessFlags2 access, vk::PipelineStageFlags2 stages) {
        const auto entry = std::ranges::find(entries, key, &Entry::key);
        if (entry == entries.end()) {
            entries.push_back(Entry{
                .key = key,
                .access = access,
                .stages = stages,
            });
            return;
        }
        entry->access |= access;
        entry->stages |= stages;
    }

    [[nodiscard]] std::span<const Entry> Entries() const noexcept {
        return entries;
    }

    void Clear() noexcept {
        entries.clear();
    }

private:
    std::vector<Entry> entries;
};

class Buffer;
using BufferAccessBatch = BasicBufferAccessBatch<Buffer*>;

} // namespace VideoCore
