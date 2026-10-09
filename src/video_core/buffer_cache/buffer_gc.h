// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>

namespace VideoCore {

template <typename Cache, typename Tick, typename Cleanup>
std::size_t CollectLruGarbage(Cache& cache, Tick cutoff, std::size_t deletion_limit,
                              Cleanup&& cleanup) {
    std::size_t deletion_count{};
    if (deletion_limit == 0) {
        return deletion_count;
    }
    cache.ForEachItemBelow(cutoff, [&](auto object) -> bool {
        cleanup(object);
        return ++deletion_count >= deletion_limit;
    });
    return deletion_count;
}

} // namespace VideoCore
