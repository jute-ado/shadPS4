// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>

#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"

namespace VideoCore {

struct CpuPageUploadRange {
    size_t offset;
    size_t size;

    auto operator<=>(const CpuPageUploadRange&) const = default;
};

class CpuPageWriteSnapshot {
public:
    static constexpr size_t UploadAlignment = sizeof(u32);
    static constexpr size_t WordsPerPage = TRACKER_BYTES_PER_PAGE / UploadAlignment;

    explicit CpuPageWriteSnapshot(std::span<const u8, TRACKER_BYTES_PER_PAGE> page,
                                  size_t write_offset, size_t write_size) {
        std::ranges::copy(page, before.begin());
        MarkWrite(write_offset, write_size);
    }

    void MarkWrite(size_t write_offset, size_t write_size) noexcept {
        const size_t begin = std::min(write_offset, TRACKER_BYTES_PER_PAGE);
        const size_t length = std::min(write_size, TRACKER_BYTES_PER_PAGE - begin);
        const size_t end = begin + length;
        const size_t word_begin = begin / UploadAlignment;
        const size_t word_end = (end + UploadAlignment - 1) / UploadAlignment;
        for (size_t word = word_begin; word < word_end; ++word) {
            forced_words.set(word);
        }
    }

    template <typename Func>
    void ForEachUploadRange(std::span<const u8, TRACKER_BYTES_PER_PAGE> current,
                            size_t range_offset, size_t range_size, Func&& func) const {
        const size_t begin = std::min(range_offset, TRACKER_BYTES_PER_PAGE);
        const size_t length = std::min(range_size, TRACKER_BYTES_PER_PAGE - begin);
        const size_t end = begin + length;
        const size_t word_begin = begin / UploadAlignment;
        const size_t word_end = (end + UploadAlignment - 1) / UploadAlignment;

        size_t dirty_begin = WordsPerPage;
        const auto emit_dirty_range = [&](size_t dirty_end) {
            if (dirty_begin == WordsPerPage) {
                return;
            }
            func(CpuPageUploadRange{
                .offset = dirty_begin * UploadAlignment,
                .size = (dirty_end - dirty_begin) * UploadAlignment,
            });
            dirty_begin = WordsPerPage;
        };

        for (size_t word = word_begin; word < word_end; ++word) {
            const size_t offset = word * UploadAlignment;
            const bool is_dirty =
                forced_words.test(word) ||
                std::memcmp(before.data() + offset, current.data() + offset, UploadAlignment) != 0;
            if (is_dirty) {
                if (dirty_begin == WordsPerPage) {
                    dirty_begin = word;
                }
            } else {
                emit_dirty_range(word);
            }
        }
        emit_dirty_range(word_end);
    }

private:
    std::array<u8, TRACKER_BYTES_PER_PAGE> before{};
    std::bitset<WordsPerPage> forced_words{};
};

class CpuPageWriteTracker {
public:
    static constexpr size_t MaximumSnapshots = 4096;

    [[nodiscard]] bool Capture(VAddr page_addr, std::span<const u8, TRACKER_BYTES_PER_PAGE> page,
                               size_t write_offset, size_t write_size) noexcept {
        try {
            std::scoped_lock lock{mutex};
            const auto snapshot = snapshots.find(page_addr);
            if (snapshot != snapshots.end()) {
                snapshot->second.MarkWrite(write_offset, write_size);
                return true;
            }
            if (snapshots.size() >= MaximumSnapshots) {
                return false;
            }
            snapshots.emplace(page_addr, CpuPageWriteSnapshot{page, write_offset, write_size});
            return true;
        } catch (...) {
            return false;
        }
    }

    template <typename Func>
    [[nodiscard]] bool Consume(VAddr page_addr, std::span<const u8, TRACKER_BYTES_PER_PAGE> current,
                               size_t range_offset, size_t range_size, Func&& func) {
        std::scoped_lock lock{mutex};
        const auto snapshot = snapshots.find(page_addr);
        if (snapshot == snapshots.end()) {
            return false;
        }
        snapshot->second.ForEachUploadRange(current, range_offset, range_size,
                                            std::forward<Func>(func));
        snapshots.erase(snapshot);
        return true;
    }

    void Discard(VAddr addr, size_t size) noexcept {
        if (size == 0) {
            return;
        }
        const VAddr first_page = addr & ~(TRACKER_BYTES_PER_PAGE - 1);
        const VAddr last_byte =
            addr + std::min<u64>(size - 1, std::numeric_limits<VAddr>::max() - addr);
        const VAddr last_page = last_byte & ~(TRACKER_BYTES_PER_PAGE - 1);
        std::scoped_lock lock{mutex};
        for (VAddr page = first_page;; page += TRACKER_BYTES_PER_PAGE) {
            snapshots.erase(page);
            if (page == last_page) {
                break;
            }
        }
    }

private:
    std::mutex mutex;
    std::unordered_map<VAddr, CpuPageWriteSnapshot> snapshots;
};

} // namespace VideoCore
