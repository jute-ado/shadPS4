// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace VideoCore {

struct SrtOverlayReuseKey {
    std::uint64_t snapshot_address{};
    std::uint64_t snapshot_size{};
    constexpr bool operator==(const SrtOverlayReuseKey&) const = default;
};

struct SrtOverlayReuseSlice {
    std::uint64_t source_address{};
    std::uint64_t source_size{};
    std::uint64_t source_offset{};
    std::uint64_t output_offset{};
    std::uint64_t download_offset{};
    std::uint64_t size{};
    std::uint64_t owner_generation{};
    std::uint32_t buffer_id{};
    std::uint32_t mapped_buffer_id{};
    constexpr bool operator==(const SrtOverlayReuseSlice&) const = default;
};

// Owner-thread-only raw bytes, never CPU-composed SRT results or borrowed allocation pointers.
// Reuse is valid only in a caller-proven no-write selection scope. Generation alone is not enough.
class SrtOverlayReuseCache {
public:
    static constexpr std::size_t MaximumRetainedBytes = 1024 * 1024;
    static constexpr std::size_t MaximumEntries = 256;
    static constexpr std::size_t MaximumSlices = 64;

    explicit SrtOverlayReuseCache(bool enabled = false) : enabled{enabled} {}
    SrtOverlayReuseCache(const SrtOverlayReuseCache&) = delete;
    SrtOverlayReuseCache& operator=(const SrtOverlayReuseCache&) = delete;

    class SelectionScope {
    public:
        explicit SelectionScope(SrtOverlayReuseCache& owner) : owner{owner} {
            owner.Enter();
        }
        SelectionScope(const SelectionScope&) = delete;
        SelectionScope& operator=(const SelectionScope&) = delete;
        ~SelectionScope() {
            owner.Leave();
        }

    private:
        SrtOverlayReuseCache& owner;
    };

    [[nodiscard]] SelectionScope BeginSelection() {
        return SelectionScope{*this};
    }

    [[nodiscard]] std::uint64_t CaptureToken() const noexcept {
        return enabled && depth == 1 && !invalidated ? revision : 0;
    }

    void InvalidateSelection() noexcept {
        if (depth != 0) {
            invalidated = true;
            Clear();
        }
    }

    [[nodiscard]] std::size_t RetainedBytes() const noexcept {
        return retained_bytes;
    }
    [[nodiscard]] std::size_t EntryCount() const noexcept {
        return entries.size();
    }

    [[nodiscard]] bool TryRead(std::uint64_t token, SrtOverlayReuseKey key,
                               std::span<const SrtOverlayReuseSlice> slices,
                               std::span<std::uint8_t> output) const noexcept {
        if (token == 0 || token != CaptureToken()) {
            return false;
        }
        const auto found = std::ranges::find_if(entries, [&](const Entry& entry) {
            return entry.key == key && entry.bytes.size() == output.size() &&
                   std::ranges::equal(entry.slices, slices);
        });
        if (found == entries.end()) {
            return false;
        }
        std::memcpy(output.data(), found->bytes.data(), output.size());
        return true;
    }

    [[nodiscard]] bool Store(std::uint64_t token, SrtOverlayReuseKey key,
                             std::span<const SrtOverlayReuseSlice> slices,
                             std::span<const std::uint8_t> bytes) noexcept {
        if (token == 0 || token != CaptureToken() || entries.size() >= MaximumEntries ||
            !ValidFootprint(key, slices, bytes.size())) {
            return false;
        }
        const std::size_t metadata = sizeof(Entry) + slices.size() * sizeof(SrtOverlayReuseSlice);
        if (metadata > MaximumRetainedBytes - retained_bytes ||
            bytes.size() > MaximumRetainedBytes - retained_bytes - metadata) {
            return false;
        }
        try {
            Entry entry{key, {slices.begin(), slices.end()}, {bytes.begin(), bytes.end()}};
            entries.push_back(std::move(entry));
            retained_bytes += metadata + bytes.size();
            return true;
        } catch (const std::bad_alloc&) {
            return false;
        }
    }

private:
    struct Entry {
        SrtOverlayReuseKey key;
        std::vector<SrtOverlayReuseSlice> slices;
        std::vector<std::uint8_t> bytes;
    };

    static bool ValidFootprint(SrtOverlayReuseKey key, std::span<const SrtOverlayReuseSlice> slices,
                               std::size_t byte_size) noexcept {
        if (key.snapshot_size == 0 || byte_size == 0 ||
            key.snapshot_size > std::numeric_limits<std::uint64_t>::max() - key.snapshot_address ||
            slices.empty() || slices.size() > MaximumSlices) {
            return false;
        }
        std::uint64_t previous_output_end{};
        std::uint64_t previous_download_end{};
        for (const auto& slice : slices) {
            if (slice.size == 0 || slice.owner_generation == 0 ||
                slice.buffer_id != slice.mapped_buffer_id ||
                slice.source_size >
                    std::numeric_limits<std::uint64_t>::max() - slice.source_address ||
                slice.source_offset > slice.source_size ||
                slice.size > slice.source_size - slice.source_offset ||
                slice.output_offset > key.snapshot_size ||
                slice.size > key.snapshot_size - slice.output_offset ||
                slice.download_offset > byte_size ||
                slice.size > byte_size - slice.download_offset ||
                slice.output_offset < previous_output_end ||
                slice.download_offset < previous_download_end ||
                slice.source_address + slice.source_offset !=
                    key.snapshot_address + slice.output_offset) {
                return false;
            }
            previous_output_end = slice.output_offset + slice.size;
            previous_download_end = slice.download_offset + slice.size;
        }
        return true;
    }

    void Enter() noexcept {
        if (depth == 0) {
            Clear();
            invalidated = false;
            if (revision == std::numeric_limits<std::uint64_t>::max()) {
                enabled = false;
            } else {
                ++revision;
            }
        } else {
            InvalidateSelection();
        }
        ++depth;
    }

    void Leave() noexcept {
        if (--depth == 0) {
            Clear();
        }
    }

    void Clear() noexcept {
        entries.clear();
        retained_bytes = 0;
    }

    bool enabled{};
    bool invalidated{};
    std::uint32_t depth{};
    std::uint64_t revision{};
    std::size_t retained_bytes{};
    std::vector<Entry> entries;
};

} // namespace VideoCore
