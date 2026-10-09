// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bit>
#include <limits>

#include "common/types.h"

namespace VideoCore {

constexpr u64 ReadbackPrefetchWindowSize = 512_KB;
constexpr u32 MaxReadbackPrefetchHotWindows = 64;
constexpr u32 MaxReadbackPrefetchPlansPerSubmit = 16;

template <typename Func>
[[nodiscard]] constexpr bool ForEachReadbackPrefetchCacheSlice(VAddr address, u64 size,
                                                                u64 cache_page_size,
                                                                Func&& func) noexcept {
    if (size == 0 || cache_page_size == 0 || !std::has_single_bit(cache_page_size) ||
        address > std::numeric_limits<VAddr>::max() - size) {
        return false;
    }

    const VAddr end = address + size;
    while (address < end) {
        const VAddr page_end = (address | (cache_page_size - 1)) + 1;
        const u64 slice_size = std::min<u64>(end - address, page_end - address);
        func(address, slice_size);
        address += slice_size;
    }
    return true;
}

struct ReadbackPrefetchPlan {
    VAddr address{};
    u64 size{};
    u64 generation{};
    u64 token{};

    auto operator<=>(const ReadbackPrefetchPlan&) const = default;
};

struct ReadbackPrefetchBatch {
    std::array<ReadbackPrefetchPlan, MaxReadbackPrefetchPlansPerSubmit> plans{};
    u32 count{};
};

enum class ReadbackPrefetchFaultDisposition : u8 {
    None,
    PendingExact,
    Stale,
};

struct ReadbackPrefetchFaultObservation {
    ReadbackPrefetchFaultDisposition disposition{};
    u64 token{};
    u64 submission_tick{};
};

enum class ReadbackPrefetchCompletionDisposition : u8 {
    Unknown,
    Publish,
    Stale,
};

/**
 * Tracks a small LRU of windows which previously required a synchronous Relaxed readback.
 *
 * This class owns no guest data and submits no work. It only joins a future GPU write to the
 * exact generation of an asynchronously submitted readback. Callers remain responsible for
 * validating buffer bounds and for publishing bytes only after Complete returns Publish.
 */
class RelaxedReadbackPrefetchTracker {
public:
    void Reset() noexcept {
        entries = {};
        use_clock = 0;
    }

    void ObserveSynchronousFault(VAddr address, u64 size) noexcept {
        const auto window = WindowFor(address, size);
        if (!window.valid) {
            return;
        }
        ObserveSynchronousWindow(window.address, ReadbackPrefetchWindowSize);
    }

    void ObserveSynchronousWindow(VAddr address, u64 size) noexcept {
        if (!IsValidRange(address, size) || size > ReadbackPrefetchWindowSize) {
            return;
        }
        if (Entry* entry = Find(address, size)) {
            entry->last_use = NextUse();
            return;
        }

        Entry* replacement = nullptr;
        for (auto& entry : entries) {
            if (!entry.occupied) {
                replacement = &entry;
                break;
            }
            if (!entry.pending && !entry.planned &&
                (replacement == nullptr || entry.last_use < replacement->last_use)) {
                replacement = &entry;
            }
        }
        if (replacement == nullptr) {
            return;
        }
        *replacement = Entry{
            .occupied = true,
            .address = address,
            .size = size,
            .last_use = NextUse(),
        };
    }

    void ObserveGpuWrite(VAddr address, u64 size) noexcept {
        if (!IsValidRange(address, size)) {
            return;
        }
        const VAddr end = address + size;
        for (auto& entry : entries) {
            if (!entry.occupied || end <= entry.address || address >= entry.address + entry.size) {
                continue;
            }
            if (entry.generation == std::numeric_limits<u64>::max()) {
                entry.generation = 1;
                entry.pending = false;
                entry.planned = false;
                entry.token = 0;
                entry.pending_generation = 0;
                entry.submission_tick = 0;
            } else {
                ++entry.generation;
            }
            entry.dirty = true;
            entry.last_use = NextUse();
        }
    }

    [[nodiscard]] ReadbackPrefetchBatch PlanSubmit() noexcept {
        ReadbackPrefetchBatch batch{};
        for (auto& entry : entries) {
            if (!entry.occupied || !entry.dirty || entry.pending || entry.planned) {
                continue;
            }
            entry.planned = true;
            entry.token = NextToken();
            entry.pending_generation = entry.generation;
            batch.plans[batch.count++] = ReadbackPrefetchPlan{
                .address = entry.address,
                .size = entry.size,
                .generation = entry.generation,
                .token = entry.token,
            };
            if (batch.count == MaxReadbackPrefetchPlansPerSubmit) {
                break;
            }
        }
        return batch;
    }

    bool CommitSubmitted(const ReadbackPrefetchPlan& plan, u64 submission_tick) noexcept {
        Entry* entry = Find(plan.address, plan.size);
        if (entry == nullptr || !entry->planned || entry->pending || plan.token == 0 ||
            submission_tick == 0 ||
            entry->token != plan.token || entry->pending_generation != plan.generation) {
            return false;
        }
        entry->planned = false;
        entry->pending = true;
        entry->submission_tick = submission_tick;
        return true;
    }

    void Abandon(const ReadbackPrefetchPlan& plan) noexcept {
        Entry* entry = Find(plan.address, plan.size);
        if (entry == nullptr || !entry->planned || entry->token != plan.token) {
            return;
        }
        entry->planned = false;
        entry->token = 0;
        entry->pending_generation = 0;
    }

    [[nodiscard]] bool CompleteNoWork(const ReadbackPrefetchPlan& plan) noexcept {
        Entry* entry = Find(plan.address, plan.size);
        if (entry == nullptr || !entry->planned || entry->pending || plan.token == 0 ||
            entry->token != plan.token || entry->pending_generation != plan.generation ||
            entry->generation != plan.generation) {
            return false;
        }
        entry->dirty = false;
        entry->planned = false;
        entry->token = 0;
        entry->pending_generation = 0;
        entry->last_use = NextUse();
        return true;
    }

    [[nodiscard]] ReadbackPrefetchFaultObservation ClassifyFault(VAddr address,
                                                                  u64 size) const noexcept {
        if (!IsValidRange(address, size)) {
            return {};
        }
        const VAddr end = address + size;
        bool stale = false;
        for (const auto& entry : entries) {
            if (!entry.occupied || address < entry.address || end > entry.address + entry.size ||
                (!entry.dirty && !entry.pending)) {
                continue;
            }
            if (entry.pending && entry.generation == entry.pending_generation) {
                return {
                    .disposition = ReadbackPrefetchFaultDisposition::PendingExact,
                    .token = entry.token,
                    .submission_tick = entry.submission_tick,
                };
            }
            stale = true;
        }
        return {.disposition = stale ? ReadbackPrefetchFaultDisposition::Stale
                                     : ReadbackPrefetchFaultDisposition::None};
    }

    [[nodiscard]] bool CanPublish(u64 token, u64 generation) const noexcept {
        if (token == 0) {
            return false;
        }
        for (const auto& entry : entries) {
            if (entry.occupied && entry.pending && entry.token == token) {
                return entry.pending_generation == generation && entry.generation == generation;
            }
        }
        return false;
    }

    ReadbackPrefetchCompletionDisposition Complete(u64 token, u64 generation,
                                                    bool publication_succeeded = true) noexcept {
        if (token == 0) {
            return ReadbackPrefetchCompletionDisposition::Unknown;
        }
        for (auto& entry : entries) {
            if (!entry.occupied || !entry.pending || entry.token != token) {
                continue;
            }
            entry.pending = false;
            entry.token = 0;
            entry.submission_tick = 0;
            if (!publication_succeeded || entry.pending_generation != generation ||
                entry.generation != generation) {
                entry.pending_generation = 0;
                return ReadbackPrefetchCompletionDisposition::Stale;
            }
            entry.pending_generation = 0;
            entry.dirty = false;
            entry.last_use = NextUse();
            return ReadbackPrefetchCompletionDisposition::Publish;
        }
        return ReadbackPrefetchCompletionDisposition::Unknown;
    }

    [[nodiscard]] u32 HotWindowCount() const noexcept {
        u32 count = 0;
        for (const auto& entry : entries) {
            count += entry.occupied ? 1U : 0U;
        }
        return count;
    }

private:
    struct Window {
        VAddr address{};
        bool valid{};
    };

    struct Entry {
        bool occupied{};
        bool dirty{};
        bool planned{};
        bool pending{};
        VAddr address{};
        u64 size{};
        u64 generation{};
        u64 pending_generation{};
        u64 token{};
        u64 submission_tick{};
        u64 last_use{};
    };

    static bool IsValidRange(VAddr address, u64 size) noexcept {
        return address != 0 && size != 0 && address <= std::numeric_limits<VAddr>::max() - size;
    }

    static Window WindowFor(VAddr address, u64 size) noexcept {
        if (!IsValidRange(address, size) || size > ReadbackPrefetchWindowSize) {
            return {};
        }
        const VAddr window_address = address & ~(ReadbackPrefetchWindowSize - 1ULL);
        if (address + size > window_address + ReadbackPrefetchWindowSize) {
            return {};
        }
        return {.address = window_address, .valid = true};
    }

    Entry* Find(VAddr address, u64 size) noexcept {
        for (auto& entry : entries) {
            if (entry.occupied && entry.address == address && entry.size == size) {
                return &entry;
            }
        }
        return nullptr;
    }

    const Entry* Find(VAddr address, u64 size) const noexcept {
        for (const auto& entry : entries) {
            if (entry.occupied && entry.address == address && entry.size == size) {
                return &entry;
            }
        }
        return nullptr;
    }

    u64 NextUse() noexcept {
        if (use_clock == std::numeric_limits<u64>::max()) {
            u64 ordinal = 1;
            for (auto& entry : entries) {
                if (entry.occupied) {
                    entry.last_use = ordinal++;
                }
            }
            use_clock = ordinal;
        }
        return ++use_clock;
    }

    u64 NextToken() noexcept {
        if (++token_clock == 0) {
            ++token_clock;
        }
        return token_clock;
    }

    std::array<Entry, MaxReadbackPrefetchHotWindows> entries{};
    u64 use_clock{};
    u64 token_clock{};
};

} // namespace VideoCore
