// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <utility>

#include "video_core/buffer_cache/range_set.h"

namespace VideoCore {

// CPU invalidation queries share this index with owner-thread registration changes. This
// protects only index operations, not Buffer lifetimes or later GPU/readback work.
template <typename Id, typename Mutex = std::mutex>
class BufferRegistrationIndex {
public:
    void Add(VAddr address, size_t size, const Id& id) {
        const std::scoped_lock lock{mutex};
        ranges.Add(address, size, id);
    }

    void Subtract(VAddr address, size_t size) {
        const std::scoped_lock lock{mutex};
        ranges.Subtract(address, size);
    }

    [[nodiscard]] bool Intersects(VAddr address, size_t size) const {
        const std::scoped_lock lock{mutex};
        return ranges.Intersects(address, size);
    }

    // Only the GPU owner may enumerate: all writers are on that same thread, and concurrent
    // CPU queries are read-only. Preserve the existing owner iteration contract; never retain
    // our index lock across a callback that can perform readback or further index operations.
    template <typename Func>
    void OwnerForEachInRange(VAddr address, size_t size, Func&& func) const {
        ranges.ForEachInRange(address, size, std::forward<Func>(func));
    }

private:
    friend struct BufferRegistrationIndexTestAccess;
    mutable Mutex mutex;
    SplitRangeMap<Id> ranges;
};

} // namespace VideoCore
