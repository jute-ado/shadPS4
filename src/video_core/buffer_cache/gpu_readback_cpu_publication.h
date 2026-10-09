// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <vector>

#include "video_core/buffer_cache/cpu_page_write_snapshot.h"

namespace VideoCore {

struct GpuReadbackPublicationRange {
    VAddr address;
    u64 size;
    u64 source_offset;
    size_t owner_index;
};

[[nodiscard]] inline bool IsValidGpuReadbackPublicationRange(
    const GpuReadbackPublicationRange& range, u64 source_size) {
    return range.size != 0 && range.size <= std::numeric_limits<VAddr>::max() - range.address &&
           range.source_offset <= source_size && range.size <= source_size - range.source_offset;
}

// Prepare the entire batch before publishing any bytes or retiring any ownership. Pending CPU
// words are omitted, rather than copied from a capture over a potentially newer CPU store. This
// does not detect stores that the page tracker could not observe. Readers run outside its mutex.
template <typename ReadCpu>
[[nodiscard]] std::optional<std::vector<GpuReadbackPublicationRange>> PrepareGpuReadbackPublication(
    CpuPageWriteTracker& tracker, std::span<const GpuReadbackPublicationRange> ranges,
    u64 source_size, ReadCpu&& read_cpu) {
    for (const auto& range : ranges) {
        if (!IsValidGpuReadbackPublicationRange(range, source_size)) {
            return std::nullopt;
        }
    }
    try {
        std::vector<GpuReadbackPublicationRange> prepared;
        std::array<u8, TRACKER_BYTES_PER_PAGE> mask{};
        for (const auto& range : ranges) {
            u64 consumed = 0;
            while (consumed < range.size) {
                const VAddr address = range.address + consumed;
                const u64 length =
                    std::min<u64>(range.size - consumed,
                                  TRACKER_BYTES_PER_PAGE - address % TRACKER_BYTES_PER_PAGE);
                if (!tracker.CapturePendingWriteMask(address, std::span{mask}.first(length),
                                                     read_cpu)) {
                    return std::nullopt;
                }
                size_t cursor = 0;
                while (cursor < length) {
                    if (mask[cursor] != 0) {
                        ++cursor;
                        continue;
                    }
                    const size_t begin = cursor;
                    while (cursor < length && mask[cursor] == 0) {
                        ++cursor;
                    }
                    const GpuReadbackPublicationRange safe{address + begin, cursor - begin,
                                                           range.source_offset + consumed + begin,
                                                           range.owner_index};
                    if (!prepared.empty() && prepared.back().owner_index == safe.owner_index &&
                        prepared.back().address + prepared.back().size == safe.address &&
                        prepared.back().source_offset + prepared.back().size ==
                            safe.source_offset) {
                        prepared.back().size += safe.size;
                    } else {
                        prepared.push_back(safe);
                    }
                }
                consumed += length;
            }
        }
        return prepared;
    } catch (const std::bad_alloc&) {
        return std::nullopt;
    }
}

// A failed write is never acknowledged. Earlier successful writes may have been published;
// preparation is all-or-nothing, but the caller's backing writer need not support batch rollback.
template <typename Publish, typename Reconcile>
[[nodiscard]] bool PublishPreparedGpuReadback(std::span<const GpuReadbackPublicationRange> prepared,
                                              std::span<const u8> source, Publish&& publish,
                                              Reconcile&& reconcile) {
    for (const auto& range : prepared) {
        if (!IsValidGpuReadbackPublicationRange(range, source.size())) {
            return false;
        }
    }
    for (const auto& range : prepared) {
        if (!std::invoke(publish, range, source.subspan(range.source_offset, range.size))) {
            return false;
        }
        std::invoke(reconcile, range);
    }
    return true;
}

} // namespace VideoCore
