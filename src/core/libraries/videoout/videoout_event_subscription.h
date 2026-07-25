// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <vector>

#include "common/types.h"
#include "core/libraries/videoout/videoout_event_id.h"

namespace Libraries::VideoOut {

[[nodiscard]] constexpr u64 GetVideoOutEventIdent(OrbisVideoOutInternalEventId event_id) {
    return static_cast<u64>(event_id);
}

inline void AddVideoOutEventSubscription(std::vector<s64>& subscriptions, s64 equeue) {
    if (!std::ranges::contains(subscriptions, equeue)) {
        subscriptions.push_back(equeue);
    }
}

[[nodiscard]] inline bool RemoveVideoOutEventSubscription(std::vector<s64>& subscriptions,
                                                          s64 equeue) {
    return std::erase(subscriptions, equeue) != 0;
}

} // namespace Libraries::VideoOut
