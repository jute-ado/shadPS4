// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace VideoCore {

// Collects all buffer uses in one draw or dispatch. Entries for a buffer are disjoint: overlapping
// uses are split at exact range boundaries and unioned, so binding order cannot turn simultaneous
// command accesses into synthetic producer/consumer transitions.
template <typename Key, typename Access, typename Stages = Access>
class BasicBufferAccessRangeBatch {
public:
    struct Entry {
        Key key{};
        std::uint64_t offset{};
        std::uint64_t size{};
        Access access{};
        Stages stages{};

        bool operator==(const Entry&) const = default;
    };

    void Add(Key key, std::uint64_t offset, std::uint64_t size, Access access, Stages stages) {
        if (size == 0 || size > std::numeric_limits<std::uint64_t>::max() - offset) {
            return;
        }

        source.swap(entries);
        entries.clear();
        same_key.clear();
        rebuilt.clear();
        entries.reserve(source.size() + 3);
        same_key.reserve(source.size());
        rebuilt.reserve(source.size() + 3);
        for (const auto& entry : source) {
            if (entry.key == key) {
                same_key.push_back(entry);
            } else {
                entries.push_back(entry);
            }
        }

        const auto end = offset + size;
        auto cursor = offset;
        for (const auto& entry : same_key) {
            const auto entry_end = entry.offset + entry.size;
            if (entry_end <= offset || entry.offset >= end) {
                rebuilt.push_back(entry);
                continue;
            }
            if (entry.offset < offset) {
                rebuilt.push_back(Entry{
                    .key = key,
                    .offset = entry.offset,
                    .size = offset - entry.offset,
                    .access = entry.access,
                    .stages = entry.stages,
                });
            }
            const auto overlap_begin = std::max(entry.offset, offset);
            const auto overlap_end = std::min(entry_end, end);
            if (cursor < overlap_begin) {
                rebuilt.push_back(Entry{
                    .key = key,
                    .offset = cursor,
                    .size = overlap_begin - cursor,
                    .access = access,
                    .stages = stages,
                });
            }
            rebuilt.push_back(Entry{
                .key = key,
                .offset = overlap_begin,
                .size = overlap_end - overlap_begin,
                .access = entry.access | access,
                .stages = entry.stages | stages,
            });
            cursor = std::max(cursor, overlap_end);
            if (entry_end > end) {
                rebuilt.push_back(Entry{
                    .key = key,
                    .offset = end,
                    .size = entry_end - end,
                    .access = entry.access,
                    .stages = entry.stages,
                });
            }
        }
        if (cursor < end) {
            rebuilt.push_back(Entry{
                .key = key,
                .offset = cursor,
                .size = end - cursor,
                .access = access,
                .stages = stages,
            });
        }

        std::ranges::sort(rebuilt, {}, &Entry::offset);
        for (const auto& entry : rebuilt) {
            if (!entries.empty()) {
                auto& previous = entries.back();
                if (previous.key == entry.key && previous.offset + previous.size == entry.offset &&
                    previous.access == entry.access && previous.stages == entry.stages) {
                    previous.size += entry.size;
                    continue;
                }
            }
            entries.push_back(entry);
        }
        std::ranges::sort(entries, [](const Entry& left, const Entry& right) {
            const auto less = std::less<Key>{};
            if (less(left.key, right.key)) {
                return true;
            }
            if (less(right.key, left.key)) {
                return false;
            }
            return left.offset < right.offset;
        });
    }

    [[nodiscard]] std::span<const Entry> Entries() const noexcept {
        return entries;
    }

    void Clear() noexcept {
        entries.clear();
    }

private:
    std::vector<Entry> entries;
    std::vector<Entry> source;
    std::vector<Entry> same_key;
    std::vector<Entry> rebuilt;
};

} // namespace VideoCore
