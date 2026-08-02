// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "common/types.h"
#include "common/unique_function.h"

namespace AmdGpu {

enum class FlipEopPosition {
    Preceding,
    Following,
};

[[nodiscard]] constexpr FlipEopPosition DecodeFlipEopPosition(u32 nop_count) noexcept {
    return nop_count == 0x33 ? FlipEopPosition::Following : FlipEopPosition::Preceding;
}

class EopFlipTracker {
public:
    [[nodiscard]] Common::UniqueFunction<void> BeginEop() {
        auto eop = std::make_shared<Eop>();
        for (auto& flip : pending_flips) {
            eop->Attach(std::move(flip));
        }
        pending_flips.clear();
        last_eop = eop;
        return [eop = std::move(eop)] { eop->Complete(); };
    }

    [[nodiscard]] bool QueueFlip(FlipEopPosition position,
                                 Common::UniqueFunction<void>&& callback) {
        if (position == FlipEopPosition::Following) {
            pending_flips.emplace_back(std::move(callback));
            return true;
        }
        if (!last_eop) {
            return false;
        }
        last_eop->Attach(std::move(callback));
        return true;
    }

private:
    class Eop {
    public:
        void Attach(Common::UniqueFunction<void>&& callback) {
            Common::UniqueFunction<void> ready_callback;
            {
                std::scoped_lock lock{mutex};
                if (completed) {
                    ready_callback = std::move(callback);
                } else {
                    flips.emplace_back(std::move(callback));
                }
            }
            if (ready_callback) {
                ready_callback();
            }
        }

        void Complete() {
            std::vector<Common::UniqueFunction<void>> ready_callbacks;
            {
                std::scoped_lock lock{mutex};
                if (completed) {
                    return;
                }
                completed = true;
                ready_callbacks = std::move(flips);
            }
            for (auto& callback : ready_callbacks) {
                callback();
            }
        }

    private:
        std::mutex mutex;
        bool completed{};
        std::vector<Common::UniqueFunction<void>> flips;
    };

    std::shared_ptr<Eop> last_eop;
    std::vector<Common::UniqueFunction<void>> pending_flips;
};

} // namespace AmdGpu
