// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <vector>

namespace Core {

// Private backing identities must not change guest-visible VMA coalescing. The caller serializes
// this index with its allocator lock; validity compares only the captured address range.
class MemoryRangeIdentity {
public:
    struct Part {
        std::uint64_t address{};
        std::uint64_t size{};
        std::uint64_t identity{};
        bool operator==(const Part&) const = default;
    };

    struct Snapshot {
        std::uint64_t address{};
        std::uint64_t size{};
        std::vector<Part> parts;
        bool operator==(const Snapshot&) const = default;
    };

    [[nodiscard]] bool Stamp(std::uint64_t address, std::uint64_t size) {
        if (!ValidRange(address, size)) {
            return false;
        }
        if (size == 0) {
            return true;
        }
        if (next_identity == std::numeric_limits<std::uint64_t>::max()) {
            return false;
        }
        Assign(address, size, next_identity + 1);
        ++next_identity;
        return true;
    }

    [[nodiscard]] std::optional<Snapshot> Capture(std::uint64_t address,
                                                  std::uint64_t size) const {
        if (!ValidRange(address, size)) {
            return std::nullopt;
        }
        Snapshot snapshot{address, size, {}};
        const auto end = address + size;
        auto cursor = address;
        auto next = boundaries.upper_bound(cursor);
        auto identity = next == boundaries.begin() ? 0 : std::prev(next)->second;
        while (cursor < end) {
            const auto part_end = next == boundaries.end() ? end : std::min(end, next->first);
            snapshot.parts.push_back({cursor, part_end - cursor, identity});
            cursor = part_end;
            if (next != boundaries.end() && cursor == next->first) {
                identity = next->second;
                ++next;
            }
        }
        return snapshot;
    }

    [[nodiscard]] bool Matches(const Snapshot& snapshot) const {
        const auto current = Capture(snapshot.address, snapshot.size);
        return current && *current == snapshot;
    }

    [[nodiscard]] bool Erase(std::uint64_t address, std::uint64_t size) {
        if (!ValidRange(address, size)) {
            return false;
        }
        if (size != 0) {
            Assign(address, size, 0);
        }
        return true;
    }

private:
    [[nodiscard]] static constexpr bool ValidRange(std::uint64_t address, std::uint64_t size) {
        return size <= std::numeric_limits<std::uint64_t>::max() - address;
    }

    [[nodiscard]] std::uint64_t IdentityAt(std::uint64_t address) const {
        const auto next = boundaries.upper_bound(address);
        return next == boundaries.begin() ? 0 : std::prev(next)->second;
    }

    void Assign(std::uint64_t address, std::uint64_t size, std::uint64_t identity) {
        const auto end = address + size;
        const auto old_end_identity = IdentityAt(end);
        const auto [end_position, end_inserted] = boundaries.try_emplace(end, old_end_identity);
        decltype(boundaries)::iterator start;
        try {
            start = boundaries.insert_or_assign(address, identity).first;
        } catch (...) {
            // A failed start-node allocation must not leave a redundant boundary which changes
            // snapshot equality. This index rollback is not a whole-VMM OOM transaction.
            if (end_inserted) {
                boundaries.erase(end_position);
            }
            throw;
        }
        boundaries.erase(std::next(start), end_position);
        RemoveRedundantBoundary(address);
        RemoveRedundantBoundary(end);
    }

    void RemoveRedundantBoundary(std::uint64_t address) {
        const auto position = boundaries.find(address);
        if (position == boundaries.end()) {
            return;
        }
        const auto before = position == boundaries.begin() ? 0 : std::prev(position)->second;
        if (before == position->second) {
            boundaries.erase(position);
        }
    }

    std::map<std::uint64_t, std::uint64_t> boundaries;
    std::uint64_t next_identity{};
    friend struct MemoryRangeIdentityInspector;
};

} // namespace Core
