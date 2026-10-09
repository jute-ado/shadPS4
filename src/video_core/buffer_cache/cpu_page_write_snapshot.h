// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>

#include "common/types.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/buffer_cache/region_definitions.h"

namespace VideoCore {

struct CpuPageUploadRange {
    size_t offset;
    size_t size;

    auto operator<=>(const CpuPageUploadRange&) const = default;
};

struct CpuSnapshotCaptureRange {
    VAddr address;
    u64 size;
};

[[nodiscard]] constexpr std::optional<CpuSnapshotCaptureRange> PlanCpuSnapshotCapture(VAddr address,
                                                                                      u64 size) {
    constexpr u64 WordMask = sizeof(u32) - 1;
    if (size == 0 || size > std::numeric_limits<VAddr>::max() - address ||
        address + size > std::numeric_limits<VAddr>::max() - WordMask) {
        return std::nullopt;
    }
    const VAddr begin = address & ~WordMask;
    const VAddr end = (address + size + WordMask) & ~WordMask;
    return CpuSnapshotCaptureRange{begin, end - begin};
}

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

        ForEachCapturedUploadRange(current.subspan(word_begin * UploadAlignment,
                                                   (word_end - word_begin) * UploadAlignment),
                                   word_begin * UploadAlignment, std::forward<Func>(func));
    }

    // Inspect an immutable, word-aligned CPU capture without reading live guest memory.
    // Partial output ranges still need their whole boundary words for the same dirty-word
    // decision that the eventual GPU upload will make.
    template <typename Func>
    bool ForEachCapturedUploadRange(std::span<const u8> current, size_t page_offset,
                                    Func&& func) const {
        if (page_offset > TRACKER_BYTES_PER_PAGE ||
            current.size() > TRACKER_BYTES_PER_PAGE - page_offset ||
            page_offset % UploadAlignment != 0 || current.size() % UploadAlignment != 0) {
            return false;
        }
        const size_t word_begin = page_offset / UploadAlignment;
        const size_t word_end = (page_offset + current.size()) / UploadAlignment;

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
                std::memcmp(before.data() + offset, current.data() + (offset - page_offset),
                            UploadAlignment) != 0;
            if (is_dirty) {
                if (dirty_begin == WordsPerPage) {
                    dirty_begin = word;
                }
            } else {
                emit_dirty_range(word);
            }
        }
        emit_dirty_range(word_end);
        return true;
    }

private:
    std::array<u8, TRACKER_BYTES_PER_PAGE> before{};
    std::bitset<WordsPerPage> forced_words{};
};

class CpuPageWriteTracker {
public:
    static constexpr size_t MaximumSnapshots = 4096;

    // Record existing whole-page CPU upload intent without touching disappearing backing.
    // Metadata allocation failure must fail-stop through noexcept, never silently retain
    // old GPU authority or fall back to reading an unmapped guest page.
    void ForceMappingUpload(VAddr address, u64 size) noexcept {
        if (size == 0) {
            return;
        }
        constexpr u64 Mask = TRACKER_BYTES_PER_PAGE - 1;
        if (size > std::numeric_limits<VAddr>::max() - address ||
            address + size > std::numeric_limits<VAddr>::max() - Mask) {
            std::terminate();
        }
        const VAddr begin = address & ~Mask;
        const VAddr end = (address + size + Mask) & ~Mask;
        std::scoped_lock lock{mutex};
        forced_uploads.Add(begin, end - begin);
    }

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
            const size_t begin = std::min(range_offset, TRACKER_BYTES_PER_PAGE);
            const size_t length = std::min(range_size, TRACKER_BYTES_PER_PAGE - begin);
            forced_uploads.Subtract(page_addr + begin, length);
            // Preserve the caller's existing full-query upload fallback, not just the
            // marker intersection. Other CPU words may lack a before-image too.
            return false;
        }
        const size_t begin = std::min(range_offset, TRACKER_BYTES_PER_PAGE);
        const size_t length = std::min(range_size, TRACKER_BYTES_PER_PAGE - begin);
        constexpr VAddr Mask = CpuPageWriteSnapshot::UploadAlignment - 1;
        const VAddr word_begin = (page_addr + begin) & ~Mask;
        const VAddr word_end = (page_addr + begin + length + Mask) & ~Mask;
        if (!forced_uploads.Intersects(word_begin, word_end - word_begin)) {
            snapshot->second.ForEachUploadRange(current, range_offset, range_size,
                                                std::forward<Func>(func));
            snapshots.erase(snapshot);
            return true;
        }
        RangeSet uploads;
        snapshot->second.ForEachUploadRange(current, range_offset, range_size,
                                            [&](CpuPageUploadRange range) {
                                                uploads.Add(page_addr + range.offset, range.size);
                                            });
        forced_uploads.ForEachInRange(word_begin, word_end - word_begin, [&](VAddr start, VAddr end) {
            // Vulkan uploads already use whole words, including partial boundary words.
            const VAddr aligned_start = start & ~Mask;
            const VAddr aligned_end = (end + Mask) & ~Mask;
            uploads.Add(aligned_start, aligned_end - aligned_start);
        });
        uploads.ForEach([&](VAddr start, VAddr end) {
            func(CpuPageUploadRange{static_cast<size_t>(start - page_addr),
                                   static_cast<size_t>(end - start)});
            forced_uploads.Subtract(start, end - start);
        });
        snapshots.erase(snapshot);
        return true;
    }

    // Reapply newer CPU words to a private GPU/CPU snapshot. Neither consume the before-image
    // nor change GPU ownership: later binding still needs to upload precisely these words.
    [[nodiscard]] bool ApplyPendingWrites(VAddr captured_address, std::span<const u8> captured,
                                          VAddr output_address, std::span<u8> output) {
        constexpr size_t Alignment = CpuPageWriteSnapshot::UploadAlignment;
        if (captured_address % Alignment != 0 || captured.size() % Alignment != 0 ||
            captured.size() > std::numeric_limits<VAddr>::max() - captured_address ||
            output_address < captured_address ||
            output_address - captured_address > captured.size() ||
            output.size() > captured.size() - (output_address - captured_address)) {
            return false;
        }
        const VAddr output_end = output_address + output.size();
        std::scoped_lock lock{mutex};
        size_t consumed = 0;
        while (consumed < captured.size()) {
            const VAddr address = captured_address + consumed;
            const VAddr page_address = address & ~(TRACKER_BYTES_PER_PAGE - 1);
            const size_t page_offset = address - page_address;
            const size_t length =
                std::min(captured.size() - consumed, TRACKER_BYTES_PER_PAGE - page_offset);
            if (const auto snapshot = snapshots.find(page_address); snapshot != snapshots.end()) {
                snapshot->second.ForEachCapturedUploadRange(
                    captured.subspan(consumed, length), page_offset, [&](CpuPageUploadRange range) {
                        const VAddr begin = std::max(page_address + range.offset, output_address);
                        const VAddr end =
                            std::min(page_address + range.offset + range.size, output_end);
                        if (begin < end) {
                            std::memcpy(output.data() + (begin - output_address),
                                        captured.data() + (begin - captured_address), end - begin);
                        }
                    });
            }
            consumed += length;
        }
        forced_uploads.ForEachInRange(output_address, output.size(), [&](VAddr begin, VAddr end) {
            std::memcpy(output.data() + (begin - output_address),
                        captured.data() + (begin - captured_address), end - begin);
        });
        return true;
    }

    void Discard(VAddr addr, size_t size) noexcept {
        if (size == 0) {
            return;
        }
        if (size > std::numeric_limits<VAddr>::max() - addr) {
            std::terminate();
        }
        const VAddr first_page = addr & ~(TRACKER_BYTES_PER_PAGE - 1);
        const VAddr last_byte = addr + size - 1;
        const VAddr last_page = last_byte & ~(TRACKER_BYTES_PER_PAGE - 1);
        std::scoped_lock lock{mutex};
        // At most MaximumSnapshots entries, even for a very large sparse unmap.
        for (auto it = snapshots.begin(); it != snapshots.end();) {
            if (it->first >= first_page && it->first <= last_page) {
                it = snapshots.erase(it);
            } else {
                ++it;
            }
        }
        // Readback publications can reconcile a sub-page range. Do not erase markers
        // outside the bytes actually reconciled merely because snapshots are page-based.
        forced_uploads.Subtract(addr, size);
    }

private:
    std::mutex mutex;
    std::unordered_map<VAddr, CpuPageWriteSnapshot> snapshots;
    RangeSet forced_uploads;
};

} // namespace VideoCore
