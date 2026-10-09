// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_copy_endpoints.h"

namespace {

struct FakeBuffer {
    u64 handle;
    VAddr base;

    u64 Handle() const { return handle; }
    u64 Offset(VAddr address) const { return address - base; }
};

TEST(BufferCopyEndpoints, ReacquiresSourceAfterDestinationRelocatesCacheStorage) {
    FakeBuffer source{71, 0x4000};
    FakeBuffer destination{92, 0x9000};
    FakeBuffer relocated_source{};
    // Keep the moved-from object alive so the regression is deterministic and does
    // not dereference freed storage. The production cache also frees that object.
    const auto [src, dst] = VideoCore::PrepareBufferCopyEndpoints(
        0x4080, 0x9020, [&]() -> const FakeBuffer& { return source; },
        [&]() -> const FakeBuffer& {
            relocated_source = source;
            source.handle = 0;
            source.base = 0;
            return destination;
        }, [&]() -> const auto& { return relocated_source; },
        [&]() -> const auto& { return destination; });
    EXPECT_EQ(relocated_source.handle, 71);
    EXPECT_EQ(src.handle, 71);
    EXPECT_EQ(src.offset, 0x80);
    EXPECT_EQ(dst.handle, 92);
    EXPECT_EQ(dst.offset, 0x20);
}

TEST(BufferCopyEndpoints, UsesReplacementAfterDestinationRetiresAnOverlappingBuffer) {
    FakeBuffer original{71, 0x4000};
    FakeBuffer replacement{93, 0x2000};
    const auto [src, dst] = VideoCore::PrepareBufferCopyEndpoints(
        0x4080, 0x4020, [&]() -> const FakeBuffer& { return original; },
        [&]() -> const FakeBuffer& {
            original.handle = 0;
            return replacement;
        }, [&]() -> const auto& { return replacement; },
        [&]() -> const auto& { return replacement; });
    EXPECT_EQ(src.handle, 93);
    EXPECT_EQ(src.offset, 0x2080);
    EXPECT_EQ(dst.handle, 93);
    EXPECT_EQ(dst.offset, 0x2020);
}

TEST(BufferCopyEndpoints, RetainsSynchronizationOrderAndCapturesValuesExactlyOnce) {
    std::vector<std::string> events;
    struct TracedBuffer {
        std::vector<std::string>& events;
        std::string name;
        u64 Handle() const { events.push_back(name + ".handle"); return 7; }
        u64 Offset(VAddr address) const { events.push_back(name + ".offset"); return address; }
    };
    const TracedBuffer source{events, "source"};
    const TracedBuffer destination{events, "destination"};
    const auto endpoints = VideoCore::PrepareBufferCopyEndpoints(
        17, 19,
        [&]() -> const TracedBuffer& { events.push_back("prepare_source"); return source; },
        [&]() -> const TracedBuffer& {
            events.push_back("prepare_destination"); return destination;
        }, [&]() -> const TracedBuffer& {
            events.push_back("resolve_source"); return source;
        }, [&]() -> const TracedBuffer& {
            events.push_back("resolve_destination"); return destination;
        });
    EXPECT_EQ(endpoints.first.offset, 17);
    EXPECT_EQ(endpoints.second.offset, 19);
    EXPECT_EQ(events, (std::vector<std::string>{"prepare_source", "prepare_destination",
                                              "resolve_source", "source.handle", "source.offset",
                                              "resolve_destination", "destination.handle",
                                              "destination.offset"}));
}

TEST(BufferCopyEndpoints, PreservesExistingIndependentBuffers) {
    const FakeBuffer source{71, 0x100000000ULL};
    const FakeBuffer destination{92, 0x300000000ULL};
    const auto [src, dst] = VideoCore::PrepareBufferCopyEndpoints(
        0x100000080ULL, 0x300000100ULL, [&]() -> const auto& { return source; },
        [&]() -> const auto& { return destination; }, [&]() -> const auto& { return source; },
        [&]() -> const auto& { return destination; });
    EXPECT_EQ(src.handle, 71);
    EXPECT_EQ(src.offset, 0x80);
    EXPECT_EQ(dst.handle, 92);
    EXPECT_EQ(dst.offset, 0x100);
}

TEST(BufferCopyEndpoints, SupportsSameBackingBufferWithDifferentOffsets) {
    const FakeBuffer buffer{42, 0x1000};
    const auto [src, dst] = VideoCore::PrepareBufferCopyEndpoints(
        0x1020, 0x1060, [&]() -> const auto& { return buffer; },
        [&]() -> const auto& { return buffer; }, [&]() -> const auto& { return buffer; },
        [&]() -> const auto& { return buffer; });
    EXPECT_EQ(src.handle, dst.handle);
    EXPECT_EQ(src.offset, 0x20);
    EXPECT_EQ(dst.offset, 0x60);
}

TEST(BufferCopyEndpoints, SupportsGlobalDataShareEndpointsWithoutCacheLookup) {
    const FakeBuffer gds{101, 0};
    const FakeBuffer cached{103, 0x2000};
    for (bool source_gds : {false, true}) {
        for (bool destination_gds : {false, true}) {
            const auto [src, dst] = VideoCore::PrepareBufferCopyEndpoints(
                source_gds ? 12 : 0x200c, destination_gds ? 32 : 0x2020,
                [&]() -> const auto& { return source_gds ? gds : cached; },
                [&]() -> const auto& { return destination_gds ? gds : cached; },
                [&]() -> const auto& { return source_gds ? gds : cached; },
                [&]() -> const auto& { return destination_gds ? gds : cached; });
            EXPECT_EQ(src.handle, source_gds ? 101 : 103);
            EXPECT_EQ(dst.handle, destination_gds ? 101 : 103);
            EXPECT_EQ(src.offset, 12);
            EXPECT_EQ(dst.offset, 32);
        }
    }
}

TEST(BufferCopyEndpoints, PublishesEachBatchBeforePreparingTheNextSource) {
    std::vector<std::string> events;
    for (u32 batch = 0; batch < 2; ++batch) {
        const bool published = VideoCore::PrepareBufferCopyBatchThenPublish(
            [&] {
                events.push_back("destination" + std::to_string(batch));
                return batch + 10;
            },
            [&] { events.push_back("settle" + std::to_string(batch)); },
            [&] {
                events.push_back("source" + std::to_string(batch));
                return batch + 20;
            },
            [&](u32 source, u32 destination) {
                events.push_back("publish" + std::to_string(batch));
                EXPECT_EQ(source, batch + 20);
                EXPECT_EQ(destination, batch + 10);
                return true;
            });
        ASSERT_TRUE(published);
    }
    EXPECT_EQ(events,
              (std::vector<std::string>{"destination0", "settle0", "source0", "publish0",
                                        "destination1", "settle1", "source1", "publish1"}));
}

TEST(BufferCopyEndpoints, DetectsOnlyPhysicalOverlapOnTheSameBuffer) {
    const VideoCore::BufferCopyEndpoint<u64> source{17, 100};
    EXPECT_TRUE(VideoCore::BufferCopyRangesOverlap(source, 20, {17, 110}, 20));
    EXPECT_FALSE(VideoCore::BufferCopyRangesOverlap(source, 20, {17, 120}, 20));
    EXPECT_FALSE(VideoCore::BufferCopyRangesOverlap(source, 20, {18, 110}, 20));
}

TEST(BufferCopyEndpoints, HlePublishesSourceLastBeforeTheNextWaitCapablePreparation) {
    std::ifstream input{SHADPS4_SHADER_HLE_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string source = contents.str();
    const auto begin = source.find("for (const auto& batch : batches)");
    const auto end = source.find("static constexpr vk::MemoryBarrier2 WRITE_BARRIER", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const std::string body = source.substr(begin, end - begin);

    const auto destination = body.find("PrepareBufferBinding(destination_address");
    const auto settle = body.find("buffer_cache.DrainReadyFaults()", destination);
    const auto source_last = body.find("PrepareBufferBinding(source_address", settle);
    const auto finalize = body.find("FinalizeBufferBinding(", source_last);
    const auto copy = body.find("copyBuffer(", finalize);
    ASSERT_NE(destination, std::string::npos);
    ASSERT_NE(settle, std::string::npos);
    ASSERT_NE(source_last, std::string::npos);
    ASSERT_NE(finalize, std::string::npos);
    ASSERT_NE(copy, std::string::npos);
    EXPECT_LT(destination, settle);
    EXPECT_LT(settle, source_last);
    EXPECT_LT(source_last, finalize);
    EXPECT_LT(finalize, copy);
    EXPECT_EQ(body.find("ObtainBuffer("), std::string::npos);
    EXPECT_EQ(body.find("PrepareBufferBinding(", source_last + 1), std::string::npos)
        << "No wait-capable acquisition may follow a materialized stream source";
}

} // namespace
