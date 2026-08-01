// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "common/types.h"

namespace VideoCore {

struct DmaDirtyRange {
    VAddr address;
    u64 size;
};

class DmaDirtyRangeTracker {
public:
    void Mark(VAddr address, u64 size) {
        if (size == 0) {
            return;
        }

        const std::scoped_lock lock{mutex};
        ranges.push_back({address, size});
    }

    [[nodiscard]] std::vector<DmaDirtyRange> Take() {
        std::vector<DmaDirtyRange> dirty_ranges;
        {
            const std::scoped_lock lock{mutex};
            dirty_ranges.swap(ranges);
        }
        if (dirty_ranges.empty()) {
            return dirty_ranges;
        }

        std::ranges::sort(dirty_ranges, {}, &DmaDirtyRange::address);
        size_t output_index = 0;
        for (size_t input_index = 1; input_index < dirty_ranges.size(); ++input_index) {
            auto& output = dirty_ranges[output_index];
            const auto& input = dirty_ranges[input_index];
            const VAddr output_end = output.address + output.size;
            if (input.address <= output_end) {
                const VAddr input_end = input.address + input.size;
                output.size = std::max(output_end, input_end) - output.address;
            } else {
                dirty_ranges[++output_index] = input;
            }
        }
        dirty_ranges.resize(output_index + 1);
        return dirty_ranges;
    }

private:
    std::mutex mutex;
    std::vector<DmaDirtyRange> ranges;
};

} // namespace VideoCore
