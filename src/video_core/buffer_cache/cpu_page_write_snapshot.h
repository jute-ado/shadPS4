// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

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
    explicit CpuPageWriteSnapshot(std::span<const u8, TRACKER_BYTES_PER_PAGE> page,
                                  size_t write_offset, size_t write_size) {
        std::ranges::copy(page, before.begin());
        MarkWrite(write_offset, write_size);
    }

    void MarkWrite(size_t write_offset, size_t write_size) noexcept {}

    template <typename Func>
    void ForEachUploadRange(std::span<const u8, TRACKER_BYTES_PER_PAGE> current,
                            size_t range_offset, size_t range_size, Func&& func) const {
        const size_t begin = std::min(range_offset, TRACKER_BYTES_PER_PAGE);
        const size_t end =
            std::min(TRACKER_BYTES_PER_PAGE, range_offset + std::min(range_size,
                                                                     TRACKER_BYTES_PER_PAGE));
        if (begin < end) {
            func(CpuPageUploadRange{.offset = begin, .size = end - begin});
        }
    }

private:
    std::array<u8, TRACKER_BYTES_PER_PAGE> before{};
};

} // namespace VideoCore
