// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <iterator>
#include <map>

#include "common/types.h"

namespace VideoCore {

class BufferWriteRangeTracker {
public:
    [[nodiscard]] bool Transition(bool is_write, u64 offset, u64 size) noexcept {
        const bool has_pending_write = Intersects(offset, size);
        if (is_write) {
            Add(offset, size);
        } else if (has_pending_write) {
            Subtract(offset, size);
        }
        return has_pending_write;
    }

    void Add(u64 offset, u64 size) noexcept {
        if (size == 0 || offset > ~u64{0} - size) {
            return;
        }
        u64 begin = offset;
        u64 end = offset + size;
        auto it = ranges.lower_bound(begin);
        if (it != ranges.begin()) {
            auto previous = std::prev(it);
            if (previous->second >= begin) {
                begin = previous->first;
                end = std::max(end, previous->second);
                it = ranges.erase(previous);
            }
        }
        while (it != ranges.end() && it->first <= end) {
            end = std::max(end, it->second);
            it = ranges.erase(it);
        }
        ranges.emplace_hint(it, begin, end);
    }

    void Subtract(u64 offset, u64 size) noexcept {
        if (size == 0 || offset > ~u64{0} - size) {
            return;
        }
        const u64 end = offset + size;
        auto it = ranges.upper_bound(offset);
        if (it != ranges.begin()) {
            --it;
        }
        while (it != ranges.end() && it->first < end) {
            const u64 range_begin = it->first;
            const u64 range_end = it->second;
            if (range_end <= offset) {
                ++it;
            } else if (offset <= range_begin && end >= range_end) {
                it = ranges.erase(it);
            } else if (offset <= range_begin) {
                ranges.erase(it);
                ranges.emplace(end, range_end);
                break;
            } else if (end >= range_end) {
                it->second = offset;
                ++it;
            } else {
                it->second = offset;
                ranges.emplace(end, range_end);
                break;
            }
        }
    }

    [[nodiscard]] bool Intersects(u64 offset, u64 size) const noexcept {
        if (size == 0 || offset > ~u64{0} - size) {
            return false;
        }
        const u64 end = offset + size;
        auto it = ranges.upper_bound(offset);
        if (it != ranges.begin()) {
            const auto previous = std::prev(it);
            if (previous->second > offset) {
                return true;
            }
        }
        return it != ranges.end() && it->first < end;
    }

private:
    std::map<u64, u64> ranges;
};

} // namespace VideoCore
