// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace {

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string FunctionBody(const std::string& source, std::string_view begin_marker,
                         std::string_view end_marker) {
    const auto function_begin = source.find(begin_marker);
    const auto function_end = source.find(end_marker, function_begin);
    EXPECT_NE(function_begin, std::string::npos);
    EXPECT_NE(function_end, std::string::npos);
    if (function_begin == std::string::npos || function_end == std::string::npos) {
        return {};
    }
    return source.substr(function_begin, function_end - function_begin);
}

// A small CPU-only model of the page-manager watcher contract. Snapshot arming requires a
// currently logical mapping; release does not, because an unmap can retire the mapping before a
// claimed release callback runs. Both releases and already-removed watchers are idempotent.
class SyntheticWatcherState {
public:
    bool logical_mapping = true;
    std::vector<std::uint8_t> counts{0, 0};

    bool Update(bool track, bool require_gpu_mapping) {
        if (require_gpu_mapping && !logical_mapping) {
            return false;
        }
        for (auto& count : counts) {
            if (track) {
                ++count;
            } else if (count != 0) {
                --count;
            }
        }
        return true;
    }
};

} // namespace

TEST(PageManagerWatcherLifecycle, RetiredLogicalMappingStillDrainsClaimedRelease) {
    const std::string source = ReadSource(SHADPS4_PAGE_MANAGER_SOURCE_PATH);
    const std::string update = FunctionBody(
        source, "template <bool track, bool is_read, bool require_gpu_mapping = false>",
        "void UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask)");

    // The admission check is compile-time strict only for snapshot arming. Release uses the
    // default false policy, so it must still reach the decrement loop after logical retirement.
    const auto strict_policy = update.find(
        "template <bool track, bool is_read, bool require_gpu_mapping = false>");
    const auto strict_guard = update.find("if constexpr (require_gpu_mapping)", strict_policy);
    const auto mapping_check = update.find("if (!logical_mappings.Contains(", strict_guard);
    const auto mapping_reject = update.find("return false;", mapping_check);
    const auto page_loop = update.find("for (; page != page_end; ++page)");
    const auto release_guard = update.find("if constexpr (!track)", page_loop);
    const auto delta = update.find("state.AddDelta<", release_guard);

    ASSERT_NE(strict_policy, std::string::npos);
    ASSERT_NE(strict_guard, std::string::npos);
    ASSERT_NE(mapping_check, std::string::npos);
    ASSERT_NE(mapping_reject, std::string::npos);
    ASSERT_NE(page_loop, std::string::npos);
    ASSERT_NE(release_guard, std::string::npos);
    ASSERT_NE(delta, std::string::npos);
    EXPECT_LT(strict_policy, strict_guard);
    EXPECT_LT(strict_guard, mapping_check);
    EXPECT_LT(mapping_check, mapping_reject);
    EXPECT_LT(mapping_reject, page_loop);
    EXPECT_LT(page_loop, release_guard);
    EXPECT_LT(release_guard, delta);

    // These call sites are the lifecycle distinction under test: only arming asks for strict
    // mapping admission; sealing and invalidation release through the generic path.
    EXPECT_NE(source.find("UpdatePageWatchers<true, false, true>(watcher_address, watcher_size)"),
              std::string::npos);
    EXPECT_NE(source.find("UpdatePageWatchers<false, false>(address, size)"),
              std::string::npos);

    SyntheticWatcherState state;
    ASSERT_TRUE(state.Update(true, true));
    state.logical_mapping = false;

    // The mapping is retired before the coordinator's claimed watcher is released. Counts must
    // still drain, and a repeated stale release must remain a no-op instead of underflowing.
    ASSERT_TRUE(state.Update(false, false));
    EXPECT_EQ(state.counts, (std::vector<std::uint8_t>{0, 0}));
    ASSERT_TRUE(state.Update(false, false));
    EXPECT_EQ(state.counts, (std::vector<std::uint8_t>{0, 0}));
}
