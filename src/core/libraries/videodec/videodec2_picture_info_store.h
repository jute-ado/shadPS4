// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <optional>
#include <unordered_map>

#include "videodec2_avc.h"

namespace Libraries::Videodec2 {

class PictureInfoStore {
public:
    void Set(const void* owner, const void* frame_buffer,
             const OrbisVideodec2AvcPictureInfo& picture) {
        std::scoped_lock lock{m_mutex};
        m_pictures[frame_buffer] = {
            .owner = owner,
            .picture = picture,
        };
    }

    [[nodiscard]] std::optional<OrbisVideodec2AvcPictureInfo> Get(
        const void* frame_buffer) const {
        std::scoped_lock lock{m_mutex};
        const auto entry = m_pictures.find(frame_buffer);
        if (entry == m_pictures.end()) {
            return std::nullopt;
        }
        return entry->second.picture;
    }

    void Clear(const void* owner) {
        std::scoped_lock lock{m_mutex};
        for (auto entry = m_pictures.begin(); entry != m_pictures.end();) {
            if (entry->second.owner == owner) {
                entry = m_pictures.erase(entry);
            } else {
                ++entry;
            }
        }
    }

private:
    struct Entry {
        const void* owner;
        OrbisVideodec2AvcPictureInfo picture;
    };

    mutable std::mutex m_mutex;
    std::unordered_map<const void*, Entry> m_pictures;
};

} // namespace Libraries::Videodec2
