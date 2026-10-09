// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <optional>
#include <sstream>
#include <string_view>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "common/slot_vector.h"
#include "video_core/buffer_cache/download_readback_policy.h"
#include "video_core/buffer_cache/gpu_modified_range_owner.h"
#include "video_core/buffer_cache/range_set.h"

namespace VideoCore {
namespace {

using BufferId = Common::SlotId;

std::string ReadText(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string FunctionBody(const std::string& source, std::string_view signature) {
    const auto begin = source.find(signature);
    EXPECT_NE(begin, std::string::npos);
    if (begin == std::string::npos) {
        return {};
    }
    const auto open = source.find('{', begin);
    EXPECT_NE(open, std::string::npos);
    if (open == std::string::npos) {
        return {};
    }
    unsigned depth{};
    for (std::size_t i = open; i < source.size(); ++i) {
        if (source[i] == '{') {
            ++depth;
        } else if (source[i] == '}' && --depth == 0) {
            return source.substr(begin, i - begin + 1);
        }
    }
    ADD_FAILURE() << "Unbalanced function body for " << signature;
    return {};
}

TEST(GpuModifiedRangeOwnership, ReplacementTransfersRangesToTheNewBuffer) {
    RangeMap<BufferId> ranges;
    constexpr VAddr address = 0x20000;
    constexpr u64 size = 0x1000;
    const BufferId old_buffer{7};
    const BufferId replacement{9};

    ranges.Add(address, size, old_buffer);
    std::vector<std::tuple<VAddr, VAddr, BufferId>> moved;
    ranges.ForEachInRange(address, size, [&](VAddr begin, VAddr end, BufferId owner) {
        moved.emplace_back(begin, end, owner);
    });
    ASSERT_EQ(moved.size(), 1U);
    EXPECT_EQ(std::get<2>(moved.front()), old_buffer);

    ranges.Subtract(address, size);
    for (const auto& [begin, end, owner] : moved) {
        ranges.Add(begin, end - begin, replacement);
    }

    moved.clear();
    ranges.ForEachInRange(address, size, [&](VAddr begin, VAddr end, BufferId owner) {
        moved.emplace_back(begin, end, owner);
    });
    ASSERT_EQ(moved.size(), 1U);
    EXPECT_EQ(std::get<2>(moved.front()), replacement);
}

TEST(GpuModifiedRangeOwnership, UnrelatedReplacementDoesNotInheritStaleDirtyBytes) {
    RangeMap<BufferId> ranges;
    constexpr VAddr address = 0x30000;
    constexpr u64 size = 0x2000;
    ranges.Add(address, size, BufferId{4});
    ranges.Subtract(address, size);

    EXPECT_FALSE(ranges.Intersects(address, size));
    std::vector<BufferId> owners;
    ranges.ForEachInRange(address, size,
                          [&](VAddr, VAddr, BufferId owner) { owners.push_back(owner); });
    EXPECT_TRUE(owners.empty());
}

TEST(GpuModifiedRangeOwnership, OverlappingWriteReplacesPreviousOwner) {
    RangeMap<BufferId> ranges;
    constexpr VAddr address = 0x40000;
    constexpr u64 size = 0x100;
    const BufferId old_buffer{11};
    const BufferId replacement{12};

    ranges.Add(address, size, old_buffer);
    // A replacement allocation may publish a write that overlaps only part of the old
    // allocation.  The overlapping bytes must be attributed to the newest allocation.
    ranges.Assign(address + 0x80, size, replacement);

    std::vector<std::tuple<VAddr, VAddr, BufferId>> actual;
    ranges.ForEachInRange(address, size + 0x80, [&](VAddr begin, VAddr end, BufferId owner) {
        actual.emplace_back(begin, end, owner);
    });
    ASSERT_EQ(actual.size(), 2U);
    EXPECT_EQ(std::get<0>(actual[0]), address);
    EXPECT_EQ(std::get<1>(actual[0]), address + 0x80);
    EXPECT_EQ(std::get<2>(actual[0]), old_buffer);
    EXPECT_EQ(std::get<0>(actual[1]), address + 0x80);
    EXPECT_EQ(std::get<1>(actual[1]), address + 0x180);
    EXPECT_EQ(std::get<2>(actual[1]), replacement);
}

TEST(GpuModifiedRangeOwnership, GenerationDistinguishesAReusedBufferSlot) {
    constexpr VAddr address = 0x50000;
    constexpr u64 size = 0x100;
    constexpr u64 page_size = 0x1000;
    const GpuModifiedRangeOwner original{BufferId{7}, 11};
    const GpuModifiedRangeOwner replacement{BufferId{7}, 12};
    RangeMap<GpuModifiedRangeOwner> ranges;
    ranges.Add(address, size, original);
    ranges.Assign(address, size, replacement);

    const std::array stale_readback{
        OwnedDownloadRange<GpuModifiedRangeOwner>{original, address, size},
    };
    std::vector<VAddr> unmarked_pages;
    ASSERT_TRUE(ReconcilePublishedGpuOwnerRanges(
        ranges, std::span<const OwnedDownloadRange<GpuModifiedRangeOwner>>{stale_readback},
        page_size, [&](VAddr page) { unmarked_pages.push_back(page); }));
    EXPECT_TRUE(ranges.Intersects(address, size));
    EXPECT_TRUE(unmarked_pages.empty());

    const std::array current_readback{
        OwnedDownloadRange<GpuModifiedRangeOwner>{replacement, address, size},
    };
    ASSERT_TRUE(ReconcilePublishedGpuOwnerRanges(
        ranges, std::span<const OwnedDownloadRange<GpuModifiedRangeOwner>>{current_readback},
        page_size, [&](VAddr page) { unmarked_pages.push_back(page); }));
    EXPECT_FALSE(ranges.Intersects(address, size));
    EXPECT_EQ(unmarked_pages, std::vector<VAddr>{address & ~(page_size - 1)});
}

TEST(GpuModifiedRangeOwnership, PlannedReadbackRejectsOwnerReplacementBeforePublication) {
    using Owner = GpuModifiedRangeOwner;
    constexpr VAddr address = 0x60000;
    constexpr u64 size = 0x80;
    constexpr u64 max_staging_size = 0x1000;
    const Owner original{BufferId{7}, 21};
    const Owner replacement{BufferId{9}, 22};

    // This is the SRT/readback handoff: planning resolves the source while the original
    // allocation owns the dirty range, then a same-address replacement arrives before the
    // asynchronous completion callback is allowed to publish its bytes.
    RangeMap<Owner> dirty_ranges;
    dirty_ranges.Add(address, size, original);
    std::vector<PlannedDownloadRange<Owner>> planned;
    u64 total_size{};
    ASSERT_TRUE(PlanOwnedDownloadRanges<Owner>(
        [&](auto&& callback) { dirty_ranges.ForEachInRange(address, size, callback); },
        [&](const Owner& owner) -> std::optional<DownloadRangeOwner<Owner>> {
            if (owner != original) {
                return std::nullopt;
            }
            return DownloadRangeOwner<Owner>{owner, address, size};
        },
        [&](const auto& range) { planned.push_back(range); }, total_size, max_staging_size));
    ASSERT_EQ(planned.size(), 1U);
    EXPECT_EQ(planned.front().owner, original);
    EXPECT_EQ(planned.front().address, address);
    EXPECT_EQ(planned.front().size, size);

    dirty_ranges.Assign(address, size, replacement);

    // The old source must not publish through the replacement's address. The range remains
    // dirty so a later readback from the replacement can still become authoritative.
    EXPECT_FALSE(AreGpuReadbackOwnersCompatible(
        dirty_ranges, std::span<const PlannedDownloadRange<Owner>>{planned}));
    EXPECT_TRUE(dirty_ranges.Intersects(address, size));
}

TEST(GpuModifiedRangeOwnership, SynchronousSnapshotRejectsReplacedOwnerGeneration) {
    using Owner = GpuModifiedRangeOwner;
    constexpr VAddr address = 0x68000;
    constexpr u64 size = 0x80;
    const Owner captured_owner{BufferId{7}, 31};
    const Owner replacement_owner{BufferId{7}, 32};

    RangeMap<Owner> dirty_ranges;
    dirty_ranges.Add(address, size, replacement_owner);
    const std::array captured{
        OwnedDownloadRange<Owner>{captured_owner, address, size},
    };

    // The source bytes were copied from generation 31, but the same slot now contains generation
    // 32. A synchronous snapshot must reject the stale overlay rather than publish it.
    const bool compatible = AreGpuReadbackOwnersCompatible(
        dirty_ranges, std::span<const OwnedDownloadRange<Owner>>{captured});
    EXPECT_FALSE(compatible);
    EXPECT_TRUE(dirty_ranges.Intersects(address, size));
}

TEST(GpuModifiedRangeOwnership, SynchronousSnapshotChecksOwnerBeforeOverlayPublication) {
    const auto source = ReadText(SHADPS4_BUFFER_CACHE_SOURCE_PATH);
    const auto body = FunctionBody(source, "bool BufferCache::DownloadBufferMemorySnapshot(");
    ASSERT_FALSE(body.empty());

    const auto captured_owner = body.find("captured_owners");
    const auto compatibility = body.find("AreGpuReadbackOwnersCompatible");
    const auto overlay = body.find("ApplyMemorySnapshotOverlays");
    ASSERT_NE(captured_owner, std::string::npos);
    ASSERT_NE(compatibility, std::string::npos);
    ASSERT_NE(overlay, std::string::npos);
    EXPECT_LT(captured_owner, compatibility);
    EXPECT_LT(compatibility, overlay);
}

TEST(GpuModifiedRangeOwnership, ProductionTracksTheOwningBufferForEveryDirtyRange) {
    const auto header = ReadText(SHADPS4_BUFFER_CACHE_HEADER_PATH);
    const auto source = ReadText(SHADPS4_BUFFER_CACHE_SOURCE_PATH);

    EXPECT_NE(header.find("RangeMap<GpuModifiedRangeOwner> gpu_modified_ranges;"),
              std::string::npos);
    EXPECT_NE(source.find("AssignGpuModifiedRange(device_addr, size, buffer_id)"),
              std::string::npos);
    EXPECT_NE(source.find("AssignGpuModifiedRange(device_addr, size, buffer_id)"),
              std::string::npos);
    EXPECT_NE(source.find("gpu_modified_ranges.ForEachInRange"), std::string::npos);
    EXPECT_NE(source.find("AssignGpuModifiedRange(begin, size, new_buffer_id)"), std::string::npos);
}

} // namespace
} // namespace VideoCore
