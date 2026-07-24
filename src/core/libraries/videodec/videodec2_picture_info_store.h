// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <optional>

#include "videodec2_avc.h"

namespace Libraries::Videodec2 {

class PictureInfoStore {
public:
    void Set(const OrbisVideodec2AvcPictureInfo& picture) {
        std::scoped_lock lock{m_mutex};
        m_latest = picture;
    }

    [[nodiscard]] std::optional<OrbisVideodec2AvcPictureInfo> Get() const {
        std::scoped_lock lock{m_mutex};
        return m_latest;
    }

    void Clear() {
        std::scoped_lock lock{m_mutex};
        m_latest.reset();
    }

private:
    mutable std::mutex m_mutex;
    std::optional<OrbisVideodec2AvcPictureInfo> m_latest;
};

} // namespace Libraries::Videodec2
