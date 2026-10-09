// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "common/lru_cache.h"
#include "video_core/texture_cache/image_gc_policy.h"

namespace {
using Cache = Common::LeastRecentlyUsedCache<u32, u64>;
using VideoCore::CollectImageForGc;
using VideoCore::ImageGcResult;

TEST(LruTraversal, BooleanTrueStopsAtCurrentItem) {
    Cache cache;
    for (u32 i = 0; i < 4; ++i)
        cache.Insert(i, 0);
    std::vector<u32> seen;
    cache.ForEachItemBelow(0, [&](u32 value) {
        seen.push_back(value);
        return value == 1;
    });
    EXPECT_EQ(seen, (std::vector<u32>{0, 1}));
}

TEST(LruTraversal, BooleanFalseAndVoidVisitAllEligibleItems) {
    Cache cache;
    cache.Insert(1, 1);
    cache.Insert(2, 2);
    cache.Insert(3, 3);
    std::vector<u32> seen;
    cache.ForEachItemBelow(2, [&](u32 value) {
        seen.push_back(value);
        return false;
    });
    EXPECT_EQ(seen, (std::vector<u32>{1, 2}));
    seen.clear();
    cache.ForEachItemBelow(2, [&](u32 value) { seen.push_back(value); });
    EXPECT_EQ(seen, (std::vector<u32>{1, 2}));
}

TEST(LruTraversal, CallbackIsInvokedAsLvalueWithStoredObjectLvalue) {
    Cache cache;
    cache.Insert(1, 0);
    cache.Insert(2, 0);
    struct Callback {
        u32 calls{};
        bool operator()(u32& value) & {
            ++calls;
            value += 10;
            return true;
        }
    } callback;
    cache.ForEachItemBelow(0, callback);
    EXPECT_EQ(callback.calls, 1U);
    std::vector<u32> seen;
    cache.ForEachItemBelow(0, [&](u32 value) { seen.push_back(value); });
    EXPECT_EQ(seen, (std::vector<u32>{11, 2}));
}

TEST(LruTraversal, TemporaryLvalueQualifiedCallbackStillStops) {
    Cache cache;
    cache.Insert(1, 0);
    cache.Insert(2, 0);
    u32 calls{};
    struct Callback {
        u32& calls;
        bool operator()(u32&) & {
            ++calls;
            return true;
        }
    };
    cache.ForEachItemBelow(0, Callback{calls});
    EXPECT_EQ(calls, 1U);
}

TEST(LruTraversal, FreeCurrentPreservesNextAndStop) {
    Cache cache;
    std::array<size_t, 3> ids{};
    for (u32 i = 0; i < ids.size(); ++i)
        ids[i] = cache.Insert(i, 0);
    std::vector<u32> seen;
    cache.ForEachItemBelow(0, [&](u32 value) {
        seen.push_back(value);
        cache.Free(ids[value]);
        return value == 1;
    });
    EXPECT_EQ(seen, (std::vector<u32>{0, 1}));
    seen.clear();
    cache.ForEachItemBelow(0, [&](u32 value) { seen.push_back(value); });
    EXPECT_EQ(seen, (std::vector<u32>{2}));
}

TEST(LruTraversal, EmptyAndTooNewNeverInvokeCallback) {
    Cache cache;
    u32 calls{};
    cache.ForEachItemBelow(0, [&](u32) {
        ++calls;
        return true;
    });
    cache.Insert(1, 2);
    cache.ForEachItemBelow(1, [&](u32) {
        ++calls;
        return true;
    });
    EXPECT_EQ(calls, 0U);
}

TEST(ImageGcAllowance, RetainedTiledWritesDoNotConsumeDeletionAllowance) {
    std::size_t remaining = 2;
    u32 downloads{}, frees{};
    EXPECT_EQ(CollectImageForGc(
                  remaining, true, true, true, [&] { ++downloads; }, [&] { ++frees; }),
              ImageGcResult::Retained);
    EXPECT_EQ(remaining, 2U);
    EXPECT_EQ(downloads, 0U);
    EXPECT_EQ(frees, 0U);
}

TEST(ImageGcAllowance, UnpressuredDownloadRetentionDoesNotConsumeAllowance) {
    std::size_t remaining = 2;
    EXPECT_EQ(CollectImageForGc(
                  remaining, true, false, false, [] { FAIL(); }, [] { FAIL(); }),
              ImageGcResult::Retained);
    EXPECT_EQ(remaining, 2U);
}

TEST(ImageGcAllowance, OnlyEligibleCollectionDownloadsThenFreesThenSpends) {
    std::size_t remaining = 2;
    std::vector<int> seen;
    EXPECT_EQ(CollectImageForGc(
                  remaining, true, false, true,
                  [&] {
                      EXPECT_EQ(remaining, 2U);
                      seen.push_back(1);
                  },
                  [&] {
                      EXPECT_EQ(remaining, 2U);
                      seen.push_back(2);
                  }),
              ImageGcResult::Deleted);
    EXPECT_EQ(seen, (std::vector<int>{1, 2}));
    EXPECT_EQ(remaining, 1U);
}

TEST(ImageGcAllowance, CleanTiledImageCanBeCollectedWithoutDownload) {
    std::size_t remaining = 1;
    u32 frees{};
    EXPECT_EQ(CollectImageForGc(
                  remaining, false, true, false, [] { FAIL(); }, [&] { ++frees; }),
              ImageGcResult::Deleted);
    EXPECT_EQ(remaining, 0U);
    EXPECT_EQ(frees, 1U);
}

TEST(ImageGcAllowance, ZeroAllowanceInvokesNeitherCallback) {
    std::size_t remaining = 0;
    EXPECT_EQ(CollectImageForGc(
                  remaining, false, false, true, [] { FAIL(); }, [] { FAIL(); }),
              ImageGcResult::Stop);
    EXPECT_EQ(remaining, 0U);
}

TEST(ImageGcAllowance, FailedCollectionDoesNotSpendAllowanceOrSwallowException) {
    std::size_t remaining = 2;
    EXPECT_THROW(CollectImageForGc(
                     remaining, true, false, true, [] { throw std::runtime_error("download"); },
                     [] { FAIL(); }),
                 std::runtime_error);
    EXPECT_EQ(remaining, 2U);
    EXPECT_THROW(
        CollectImageForGc(
            remaining, false, false, true, [] {}, [] { throw std::runtime_error("free"); }),
        std::runtime_error);
    EXPECT_EQ(remaining, 2U);
}

TEST(ImageGcAllowance, ActualLruTraversesRetainedPrefixToReachCleanImage) {
    Cache cache;
    std::array<size_t, 42> ids{};
    for (u32 i = 0; i < ids.size(); ++i)
        ids[i] = cache.Insert(i, 0);
    std::size_t remaining = 1;
    std::vector<u32> freed;
    cache.ForEachItemBelow(0, [&](u32 value) {
        const auto result = CollectImageForGc(
            remaining, value < 40, true, true, [] { FAIL(); },
            [&] {
                freed.push_back(value);
                cache.Free(ids[value]);
            });
        return result == ImageGcResult::Stop;
    });
    EXPECT_EQ(freed, (std::vector<u32>{40}));
    EXPECT_EQ(remaining, 0U);
}

TEST(ImageGcAllowance, EntirelyRetainedTraversalTerminatesWithoutSpending) {
    Cache cache;
    for (u32 i = 0; i < 64; ++i)
        cache.Insert(i, 0);
    std::size_t remaining = 3;
    u32 visited{};
    cache.ForEachItemBelow(0, [&](u32) {
        ++visited;
        return CollectImageForGc(
                   remaining, true, true, true, [] { FAIL(); }, [] { FAIL(); }) ==
               ImageGcResult::Stop;
    });
    EXPECT_EQ(visited, 64U);
    EXPECT_EQ(remaining, 3U);
}

TEST(ImageGcAllowance, ProductionCleanupUsesTestedEligibilityAndCallbacks) {
    std::ifstream file{std::string{IMAGE_GC_SOURCE_ROOT} +
                       "/src/video_core/texture_cache/texture_cache.cpp"};
    const std::string source{std::istreambuf_iterator<char>{file}, {}};
    const auto begin = source.find("void TextureCache::GarbageCollectImages()");
    const auto end = source.find("void TextureCache::GarbageCollectSamplers()", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(body.find("CollectImageForGc("), std::string::npos);
    EXPECT_NE(body.find("image.SafeToDownload()"), std::string::npos);
    EXPECT_NE(body.find("image.info.IsTiled()"), std::string::npos);
    EXPECT_NE(body.find("[&] { DownloadImageMemory(image_id); }"), std::string::npos);
    EXPECT_NE(body.find("[&] { FreeImage(image_id); }"), std::string::npos);
    EXPECT_EQ(body.find("--num_deletions"), std::string::npos);
}
} // namespace
