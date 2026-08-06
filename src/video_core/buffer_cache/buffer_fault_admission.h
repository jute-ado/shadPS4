// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

namespace VideoCore {

[[nodiscard]] constexpr bool IsBufferFaultOwned(bool is_registered, bool owned_before_lock,
                                                bool owned_after_lock) noexcept {
    return is_registered || owned_before_lock || owned_after_lock;
}

template <typename Invalidate, typename MarkDirty>
[[nodiscard]] bool ProcessTrackedBufferFault(Invalidate&& invalidate, MarkDirty&& mark_dirty) {
    if (!std::invoke(std::forward<Invalidate>(invalidate))) {
        return false;
    }
    std::invoke(std::forward<MarkDirty>(mark_dirty));
    return true;
}

} // namespace VideoCore
