// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <utility>

#include "common/types.h"
#include "common/unique_function.h"

namespace AmdGpu {

enum class FlipEopPosition {
    Preceding,
    Following,
};

[[nodiscard]] constexpr FlipEopPosition DecodeFlipEopPosition(u32 nop_count) {
    return nop_count == 0x33 ? FlipEopPosition::Following : FlipEopPosition::Preceding;
}

class EopFlipTracker {
public:
    [[nodiscard]] Common::UniqueFunction<void> BeginEop(bool signals_interrupt = false) {
        auto eop =
            std::make_shared<Eop>(signals_interrupt, non_interrupting_flip_associations);
        if (pending_flip) {
            (void)eop->Attach(std::move(pending_flip));
        }
        last_eop = eop;
        return [eop = std::move(eop)] { eop->Complete(); };
    }

    [[nodiscard]] bool QueueFlip(FlipEopPosition position,
                                 Common::UniqueFunction<void>&& callback) {
        if (position == FlipEopPosition::Following) {
            if (pending_flip) {
                return false;
            }
            pending_flip = std::move(callback);
            return true;
        }
        return last_eop && last_eop->Attach(std::move(callback));
    }

    [[nodiscard]] u64 ConsumeNonInterruptingFlipAssociations() {
        return non_interrupting_flip_associations.exchange(0, std::memory_order_relaxed);
    }

private:
    class Eop {
    public:
        explicit Eop(bool signals_interrupt_, std::atomic<u64>& association_counter_)
            : signals_interrupt{signals_interrupt_}, association_counter{association_counter_} {}

        [[nodiscard]] bool Attach(Common::UniqueFunction<void>&& callback) {
            if (!signals_interrupt) {
                association_counter.fetch_add(1, std::memory_order_relaxed);
            }
            Common::UniqueFunction<void> ready_callback;
            {
                std::scoped_lock lock{mutex};
                if (flip) {
                    return false;
                }
                if (completed) {
                    ready_callback = std::move(callback);
                } else {
                    flip = std::move(callback);
                }
            }
            if (ready_callback) {
                ready_callback();
            }
            return true;
        }

        void Complete() {
            Common::UniqueFunction<void> ready_callback;
            {
                std::scoped_lock lock{mutex};
                if (completed) {
                    return;
                }
                completed = true;
                ready_callback = std::move(flip);
            }
            if (ready_callback) {
                ready_callback();
            }
        }

    private:
        std::mutex mutex;
        bool signals_interrupt{};
        std::atomic<u64>& association_counter;
        bool completed{};
        Common::UniqueFunction<void> flip;
    };

    std::atomic<u64> non_interrupting_flip_associations{};
    std::shared_ptr<Eop> last_eop;
    Common::UniqueFunction<void> pending_flip;
};

} // namespace AmdGpu
