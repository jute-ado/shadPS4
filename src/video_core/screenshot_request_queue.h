// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/types.h"

namespace VideoCore {

enum class ScreenshotRequest : u32 {
    None = 0,
    GameOnly = 1,
    WithOverlays = 2,
};

enum class ScreenshotRequestOrigin : u8 {
    User,
    Automation,
};

struct ScreenshotRequestBatch {
    u32 notifying_count{};
    u32 silent_count{};

    [[nodiscard]] u32 Total() const {
        return notifying_count + silent_count;
    }
};

class ScreenshotRequestQueue {
public:
    void Push(const ScreenshotRequest request, const ScreenshotRequestOrigin origin) {
        const bool silent = origin == ScreenshotRequestOrigin::Automation;
        switch (request) {
        case ScreenshotRequest::GameOnly:
            (silent ? game_only_silent : game_only_notifying)
                .fetch_add(1, std::memory_order_relaxed);
            break;
        case ScreenshotRequest::WithOverlays:
            (silent ? with_overlays_silent : with_overlays_notifying)
                .fetch_add(1, std::memory_order_relaxed);
            break;
        case ScreenshotRequest::None:
        default:
            break;
        }
    }

    [[nodiscard]] ScreenshotRequestBatch ConsumeGameOnly() {
        return Consume(game_only_notifying, game_only_silent);
    }

    [[nodiscard]] ScreenshotRequestBatch ConsumeWithOverlays() {
        return Consume(with_overlays_notifying, with_overlays_silent);
    }

private:
    static ScreenshotRequestBatch Consume(std::atomic<u32>& notifying, std::atomic<u32>& silent) {
        return {
            .notifying_count = notifying.exchange(0, std::memory_order_acq_rel),
            .silent_count = silent.exchange(0, std::memory_order_acq_rel),
        };
    }

    std::atomic<u32> game_only_notifying{0};
    std::atomic<u32> game_only_silent{0};
    std::atomic<u32> with_overlays_notifying{0};
    std::atomic<u32> with_overlays_silent{0};
};

} // namespace VideoCore
