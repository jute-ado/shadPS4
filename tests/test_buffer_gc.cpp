// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "common/lru_cache.h"
#include "video_core/buffer_cache/buffer_gc.h"

TEST(BufferGarbageCollection, ReclaimsOnlyItemsAtOrBelowTheAgeCutoff) {
    Common::LeastRecentlyUsedCache<int, unsigned long long> cache;
    cache.Insert(10, 0);
    cache.Insert(20, 39);
    cache.Insert(30, 40);
    cache.Insert(40, 41);

    std::vector<int> reclaimed;
    const auto count =
        VideoCore::CollectLruGarbage(cache, 40ULL, 10, [&](int value) {
            reclaimed.push_back(value);
        });

    EXPECT_EQ(count, 3);
    EXPECT_EQ(reclaimed, (std::vector{10, 20, 30}));
}

TEST(BufferGarbageCollection, StopsAtThePerPassDeletionLimit) {
    Common::LeastRecentlyUsedCache<int, unsigned long long> cache;
    cache.Insert(10, 0);
    cache.Insert(20, 1);
    cache.Insert(30, 2);

    std::vector<int> reclaimed;
    const auto count =
        VideoCore::CollectLruGarbage(cache, 10ULL, 2, [&](int value) {
            reclaimed.push_back(value);
        });

    EXPECT_EQ(count, 2);
    EXPECT_EQ(reclaimed, (std::vector{10, 20}));
}

TEST(BufferGarbageCollection, AllowsCleanupToRemoveTheCurrentLruItem) {
    Common::LeastRecentlyUsedCache<int, unsigned long long> cache;
    std::array<std::size_t, 4> lru_ids{};
    lru_ids[1] = cache.Insert(1, 0);
    lru_ids[2] = cache.Insert(2, 0);
    lru_ids[3] = cache.Insert(3, 0);

    std::vector<int> reclaimed;
    const auto count =
        VideoCore::CollectLruGarbage(cache, 0ULL, 3, [&](int value) {
            reclaimed.push_back(value);
            cache.Free(lru_ids[value]);
        });

    EXPECT_EQ(count, 3);
    EXPECT_EQ(reclaimed, (std::vector{1, 2, 3}));
}
