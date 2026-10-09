// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <queue>

#include "common/types.h"
#include "video_core/amdgpu/eop_completion.h"

namespace AmdGpu {

// Only the command-processor owner appends/flushes this batch. Adjacent EOPs
// share a submission, but must be flushed before any other packet, host command,
// coroutine yield or the end of a command buffer can overtake them.
class EopSubmissionBatch {
public:
    ~EopSubmissionBatch() {
        Cancel();
    }

    [[nodiscard]] bool HasPending() const noexcept {
        return !pending.empty();
    }

    template <typename Packet, typename WriteMemory, typename SignalInterrupt,
              typename NotifyCompletion>
    void Add(Packet packet, WriteMemory&& write_memory, SignalInterrupt&& signal_interrupt,
             NotifyCompletion&& notify_completion) {
        DeferEopUntilGpuCompletion(
            std::move(packet),
            [this](Common::UniqueFunction<void>&& completion) {
                pending.emplace_back(std::move(completion));
            },
            std::forward<WriteMemory>(write_memory),
            std::forward<SignalInterrupt>(signal_interrupt),
            std::forward<NotifyCompletion>(notify_completion));
    }

    template <typename Prepare, typename Submit>
    void Flush(Prepare&& prepare, Submit&& submit) {
        Flush(std::forward<Prepare>(prepare), std::forward<Submit>(submit), [] {});
    }

    template <typename Prepare, typename Submit, typename WakeOwner>
    void Flush(Prepare&& prepare, Submit&& submit, WakeOwner&& wake_owner) {
        if (pending.empty()) {
            return;
        }
        std::forward<Prepare>(prepare)();
        std::forward<Submit>(submit)(
            Common::UniqueFunction<void, u64>{
                [this, completions = std::exchange(pending, {})](u64 tick) mutable {
                    // Called synchronously by the owner, before enabling the
                    // worker wakeup for this exact successful submission.
                    ready.emplace(tick, std::move(completions));
                }},
            Common::UniqueFunction<void>{
                [lifetime = lifetime, wake_owner = std::forward<WakeOwner>(wake_owner)]() mutable {
                    std::scoped_lock lock{lifetime->mutex};
                    if (lifetime->active) {
                        // Wake only. Never write guest memory or signal an IRQ
                        // on the scheduler worker, including page-fault paths.
                        wake_owner();
                    }
                }});
    }

    template <typename IsComplete>
    void DrainCompleted(IsComplete&& is_complete) {
        if (draining || !lifetime->active) {
            return;
        }
        draining = true;
        struct ResetDraining {
            bool& draining;
            ~ResetDraining() {
                draining = false;
            }
        } reset{draining};
        while (lifetime->active && !ready.empty() && is_complete(ready.front().tick)) {
            // A fence write may fault and synchronously re-enter owner work.
            // Remove the batch first and suppress nested drains until its whole
            // fence/IRQ/flip sequence has finished.
            auto completions = std::move(ready.front().completions);
            ready.pop();
            for (auto& completion : completions) {
                if (!lifetime->active) {
                    break;
                }
                completion();
            }
        }
    }

    // Disarm worker-to-owner handoffs before requesting owner stop. Owner join
    // then retires an already-running publication; Cancel itself does not wait
    // for GPU completion or hold this mutex across guest memory/IRQ callbacks.
    void Cancel() {
        std::scoped_lock lock{lifetime->mutex};
        lifetime->active = false;
    }

private:
    struct Lifetime {
        std::mutex mutex;
        std::atomic_bool active = true;
    };
    struct ReadyBatch {
        u64 tick;
        std::vector<Common::UniqueFunction<void>> completions;
    };

    std::shared_ptr<Lifetime> lifetime = std::make_shared<Lifetime>();
    std::vector<Common::UniqueFunction<void>> pending;
    std::queue<ReadyBatch> ready;
    bool draining = false;
};

} // namespace AmdGpu
