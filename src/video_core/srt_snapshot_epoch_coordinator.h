// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/types.h"

namespace VideoCore {

/// Coordinates one short-lived coherent SRT snapshot with CPU writes and guest-memory unmaps.
/// Watcher callbacks run without the coordinator mutex because changing host page protection may
/// enter operating-system fault handling paths.
class SrtSnapshotEpochCoordinator {
public:
    [[nodiscard]] u64 BeginEpoch() {
        std::scoped_lock lock{mutex};
        if (state) {
            return 0;
        }
        const u64 id = next_id++;
        if (next_id == 0) {
            ++next_id;
        }
        state.emplace(State{.id = id, .owner = std::this_thread::get_id()});
        return id;
    }

    template <typename ArmWatcher>
    [[nodiscard]] bool BeginPageCopy(u64 id, VAddr address, u64 size, ArmWatcher&& arm_watcher) {
        {
            std::unique_lock lock{mutex};
            if (!state || state->id != id || state->invalid || state->closing) {
                return false;
            }
            auto page = state->pages.find(address);
            if (page != state->pages.end()) {
                Require(page->second.size == size && !page->second.arming &&
                        !page->second.copying && page->second.watcher_owned);
                page->second.copying = true;
                return true;
            }
            state->pages.emplace(address, Page{.size = size, .arming = true});
        }

        std::forward<ArmWatcher>(arm_watcher)(address, size);

        std::unique_lock lock{mutex};
        Require(state && state->id == id);
        auto page = state->pages.find(address);
        Require(page != state->pages.end() && page->second.arming);
        page->second.arming = false;
        page->second.watcher_owned = true;
        if (!state->invalid && !state->closing) {
            page->second.copying = true;
        }
        cv.notify_all();
        return page->second.copying;
    }

    [[nodiscard]] bool EndPageCopy(u64 id, VAddr address, u64 size, bool copied) {
        std::scoped_lock lock{mutex};
        Require(state && state->id == id);
        auto page = state->pages.find(address);
        Require(page != state->pages.end() && page->second.size == size && page->second.copying);
        page->second.copying = false;
        cv.notify_all();
        return copied && !state->invalid && !state->closing;
    }

    [[nodiscard]] bool IsEpochValid(u64 id) const {
        std::scoped_lock lock{mutex};
        return state && state->id == id && !state->invalid;
    }

    template <typename ReleaseWatcher>
    [[nodiscard]] bool SealEpoch(u64 id, ReleaseWatcher&& release_watcher) {
        std::vector<Range> watchers;
        bool valid{};
        {
            std::unique_lock lock{mutex};
            if (!state || state->id != id) {
                return false;
            }
            Require(state->owner == std::this_thread::get_id());
            cv.wait(lock, [&] { return !state->releasing_watchers; });
            Require(NoActiveCopiesLocked());
            state->closing = true;
            state->releasing_watchers = true;
            valid = !state->invalid;
            watchers = ClaimWatchersLocked();
        }
        ReleaseWatchers(watchers, std::forward<ReleaseWatcher>(release_watcher));
        {
            std::scoped_lock lock{mutex};
            Require(state && state->id == id);
            state.reset();
        }
        cv.notify_all();
        return valid;
    }

    template <typename ReleaseWatcher>
    void InvalidateRangeAndWait(VAddr address, u64 size, ReleaseWatcher&& release_watcher) {
        std::vector<Range> watchers;
        u64 epoch_id{};
        {
            std::unique_lock lock{mutex};
            if (!state || !OverlapsCapturedPageLocked(address, size)) {
                return;
            }
            epoch_id = state->id;
            if (state->closing) {
                cv.wait(lock, [&] { return !state || state->id != epoch_id; });
                return;
            }
            state->invalid = true;
            Require(state->owner != std::this_thread::get_id() || NoActiveCopiesLocked());
            cv.wait(lock, [&] {
                return !state || state->id != epoch_id ||
                       (!state->releasing_watchers && NoActiveCopiesLocked());
            });
            if (!state || state->id != epoch_id) {
                return;
            }
            state->releasing_watchers = true;
            watchers = ClaimWatchersLocked();
        }
        ReleaseWatchers(watchers, std::forward<ReleaseWatcher>(release_watcher));
        FinishWatcherRelease(epoch_id);
    }

private:
    using Range = std::pair<VAddr, u64>;

    struct Page {
        u64 size{};
        bool arming{};
        bool copying{};
        bool watcher_owned{};
    };

    struct State {
        u64 id{};
        std::thread::id owner{};
        bool invalid{};
        bool closing{};
        bool releasing_watchers{};
        std::unordered_map<VAddr, Page> pages;
    };

    static void Require(bool condition) {
        if (!condition) [[unlikely]] {
            std::terminate();
        }
    }

    [[nodiscard]] static bool RangesOverlap(VAddr lhs_address, u64 lhs_size, VAddr rhs_address,
                                            u64 rhs_size) {
        if (lhs_size == 0 || rhs_size == 0) {
            return false;
        }
        return lhs_address <= rhs_address ? rhs_address - lhs_address < lhs_size
                                          : lhs_address - rhs_address < rhs_size;
    }

    [[nodiscard]] bool NoActiveCopiesLocked() const {
        return std::ranges::none_of(state->pages, [](const auto& entry) {
            return entry.second.arming || entry.second.copying;
        });
    }

    [[nodiscard]] bool OverlapsCapturedPageLocked(VAddr address, u64 size) const {
        return std::ranges::any_of(state->pages, [&](const auto& entry) {
            return RangesOverlap(address, size, entry.first, entry.second.size);
        });
    }

    [[nodiscard]] std::vector<Range> ClaimWatchersLocked() {
        std::vector<Range> watchers;
        watchers.reserve(state->pages.size());
        for (auto& [address, page] : state->pages) {
            if (page.watcher_owned) {
                page.watcher_owned = false;
                watchers.emplace_back(address, page.size);
            }
        }
        return watchers;
    }

    template <typename ReleaseWatcher>
    static void ReleaseWatchers(const std::vector<Range>& watchers,
                                ReleaseWatcher&& release_watcher) {
        for (const auto& [address, size] : watchers) {
            release_watcher(address, size);
        }
    }

    void FinishWatcherRelease(u64 id) {
        {
            std::scoped_lock lock{mutex};
            if (state && state->id == id) {
                state->releasing_watchers = false;
            }
        }
        cv.notify_all();
    }

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::optional<State> state;
    u64 next_id{1};
};

} // namespace VideoCore
