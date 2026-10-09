// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <future>
#include <shared_mutex>
#include <vector>

#include <gtest/gtest.h>

#include "core/backing_memory_write.h"

namespace {

constexpr VAddr Base = 0x1000;
constexpr std::array<u8, 8> Payload{1, 2, 3, 4, 5, 6, 7, 8};

struct Fixture {
    std::map<VAddr, Core::VirtualMemoryArea> areas;
    Core::MemoryRangeIdentity incarnations;
    std::shared_mutex mutex;
    std::array<u8, 64> backing;
    unsigned publications{};
    std::vector<Core::ReadbackWriteRange> published_ranges;
    std::function<bool()> pending;

    Fixture() {
        backing.fill(0xcc);
    }

    void Map(VAddr address, u64 size, PAddr physical) {
        areas.insert_or_assign(
            address, Core::VirtualMemoryArea{.base = address,
                                             .size = size,
                                             .phys_areas = {{0, {.base = physical, .size = size}}},
                                             .type = Core::VMAType::Direct});
        ASSERT_TRUE(incarnations.Stamp(address, size));
    }

    bool Schedule(std::vector<Core::ReadbackWriteRange> destinations) {
        auto token =
            Core::CaptureDeferredBackingWrite(areas, incarnations, destinations, Payload.size());
        if (!token)
            return false;
        pending = [this, token = std::move(*token)] {
            return Core::PublishDeferredBackingWrite(
                mutex, areas, incarnations, backing.data(), token, Payload, [&](auto ranges) {
                    ++publications;
                    published_ranges.assign(ranges.begin(), ranges.end());
                });
        };
        return true;
    }
};

TEST(DeferredReadbackPublication, UnchangedBackingPublishesCompletePayload) {
    Fixture f;
    f.Map(Base, 8, 0);
    ASSERT_TRUE(f.Schedule({{Base, 8, 0}}));
    ASSERT_TRUE(f.pending());
    EXPECT_TRUE(std::equal(Payload.begin(), Payload.end(), f.backing.begin()));
    EXPECT_EQ(f.publications, 1u);
}

TEST(DeferredReadbackPublication, SameAddressReplacementCannotRetargetOldBytes) {
    Fixture f;
    f.Map(Base, 8, 0);
    ASSERT_TRUE(f.Schedule({{Base, 8, 0}}));
    f.Map(Base, 8, 16);
    const auto before = f.backing;
    EXPECT_FALSE(f.pending());
    EXPECT_EQ(f.backing, before);
    EXPECT_EQ(f.publications, 0u);
}

TEST(DeferredReadbackPublication, IdenticalAddressAndPhysicalReuseStillRejectsOldIncarnation) {
    Fixture f;
    f.Map(Base, 8, 0);
    ASSERT_TRUE(f.Schedule({{Base, 8, 0}}));
    f.areas.erase(Base);
    ASSERT_TRUE(f.incarnations.Erase(Base, 8));
    f.Map(Base, 8, 0);
    const auto before = f.backing;
    EXPECT_FALSE(f.pending());
    EXPECT_EQ(f.backing, before);
    EXPECT_EQ(f.publications, 0u);
}

TEST(DeferredReadbackPublication, PartialReplacementPreservesUnchangedSingleCopyPrefix) {
    Fixture f;
    f.Map(Base, 8, 0);
    ASSERT_TRUE(f.Schedule({{Base, 8, 0}}));
    f.areas.at(Base).size = 4;
    f.areas.at(Base).phys_areas.at(0).size = 4;
    f.Map(Base + 4, 4, 16);
    auto expected = f.backing;
    std::copy_n(Payload.begin(), 4, expected.begin());
    EXPECT_FALSE(f.pending());
    EXPECT_EQ(f.backing, expected);
    EXPECT_EQ(f.publications, 1u);
    EXPECT_EQ(f.published_ranges, (std::vector<Core::ReadbackWriteRange>{{Base, 4, 0}}));
}

TEST(DeferredReadbackPublication, UnmappedSuffixDoesNotLoseStillOwnedPrefix) {
    Fixture f;
    f.Map(Base, 4, 0);
    f.Map(Base + 4, 4, 4);
    ASSERT_TRUE(f.Schedule({{Base, 8, 0}}));
    f.areas.erase(Base + 4);
    ASSERT_TRUE(f.incarnations.Erase(Base + 4, 4));
    auto expected = f.backing;
    std::copy_n(Payload.begin(), 4, expected.begin());
    EXPECT_FALSE(f.pending());
    EXPECT_EQ(f.backing, expected);
    EXPECT_EQ(f.publications, 1u);
    EXPECT_EQ(f.published_ranges, (std::vector<Core::ReadbackWriteRange>{{Base, 4, 0}}));
}

TEST(DeferredReadbackPublication, LaterInvalidDestinationDoesNotLoseEarlierCopyOrCleanup) {
    Fixture f;
    f.Map(Base, 4, 0);
    f.Map(Base + 32, 4, 16);
    ASSERT_TRUE(f.Schedule({{Base, 4, 0}, {Base + 32, 4, 4}}));
    f.Map(Base + 32, 4, 32);
    auto expected = f.backing;
    std::copy_n(Payload.begin(), 4, expected.begin());
    EXPECT_FALSE(f.pending());
    EXPECT_EQ(f.backing, expected);
    EXPECT_EQ(f.publications, 1u);
    EXPECT_EQ(f.published_ranges, (std::vector<Core::ReadbackWriteRange>{{Base, 4, 0}}));
}

TEST(DeferredReadbackPublication, ReplacementInsideOneCopyPreservesBothUnchangedSides) {
    Fixture f;
    f.Map(Base, 8, 0);
    ASSERT_TRUE(f.Schedule({{Base, 8, 0}}));
    auto right = f.areas.at(Base);
    right.base = Base + 6;
    right.size = 2;
    right.phys_areas = {{0, {.base = 6, .size = 2}}};
    f.areas.emplace(Base + 6, std::move(right));
    f.areas.at(Base).size = 2;
    f.areas.at(Base).phys_areas.at(0).size = 2;
    f.Map(Base + 2, 4, 16);
    auto expected = f.backing;
    std::copy_n(Payload.begin(), 2, expected.begin());
    std::copy_n(Payload.begin() + 6, 2, expected.begin() + 6);
    EXPECT_FALSE(f.pending());
    EXPECT_EQ(f.backing, expected);
    EXPECT_EQ(f.publications, 1u);
    EXPECT_EQ(f.published_ranges,
              (std::vector<Core::ReadbackWriteRange>{{Base, 2, 0}, {Base + 6, 2, 6}}));
}

TEST(DeferredReadbackPublication, EveryReplacementSubrangeKeepsExactlyItsUnchangedBytes) {
    for (u64 begin = 0; begin < Payload.size(); ++begin) {
        for (u64 end = begin + 1; end <= Payload.size(); ++end) {
            SCOPED_TRACE(::testing::Message{} << "replacement [" << begin << ", " << end << ")");
            Fixture f;
            f.Map(Base, Payload.size(), 0);
            ASSERT_TRUE(f.Schedule({{Base, Payload.size(), 0}}));
            const auto original = f.areas.at(Base);
            f.areas.clear();
            auto expected = f.backing;
            std::vector<Core::ReadbackWriteRange> expected_ranges;
            const auto preserve = [&](u64 offset, u64 size) {
                if (size == 0) {
                    return;
                }
                auto part = original;
                part.base += offset;
                part.size = size;
                part.phys_areas = {{0, {.base = offset, .size = size}}};
                f.areas.emplace(part.base, std::move(part));
                std::copy_n(Payload.begin() + offset, size, expected.begin() + offset);
                expected_ranges.push_back({Base + offset, size, offset});
            };
            preserve(0, begin);
            preserve(end, Payload.size() - end);
            f.Map(Base + begin, end - begin, 16);
            EXPECT_FALSE(f.pending());
            EXPECT_EQ(f.backing, expected);
            EXPECT_EQ(f.published_ranges, expected_ranges);
            EXPECT_EQ(f.publications, expected_ranges.empty() ? 0u : 1u);
        }
    }
}

TEST(DeferredReadbackPublication, CapturePartitionsIdentityChangesInsideOnePhysicalFragment) {
    Fixture f;
    f.Map(Base, Payload.size(), 0);
    ASSERT_TRUE(f.incarnations.Stamp(Base + 2, 4));
    ASSERT_TRUE(f.Schedule({{Base, Payload.size(), 0}}));
    ASSERT_TRUE(f.pending());
    EXPECT_TRUE(std::equal(Payload.begin(), Payload.end(), f.backing.begin()));
    EXPECT_EQ(f.published_ranges,
              (std::vector<Core::ReadbackWriteRange>{{Base, Payload.size(), 0}}));
}

TEST(DeferredReadbackPublication, NewPhysicalAliasDoesNotInvalidateOriginalDestination) {
    Fixture f;
    f.Map(Base, 8, 0);
    ASSERT_TRUE(f.Schedule({{Base, 8, 0}}));
    f.Map(Base + 32, 8, 0);
    ASSERT_TRUE(f.pending());
    EXPECT_TRUE(std::equal(Payload.begin(), Payload.end(), f.backing.begin()));
    EXPECT_EQ(f.publications, 1u);
}

TEST(DeferredReadbackPublication, NonoverlappingReplacementDoesNotInvalidateDestination) {
    Fixture f;
    f.Map(Base, 8, 0);
    f.Map(Base + 32, 8, 16);
    ASSERT_TRUE(f.Schedule({{Base, 8, 0}}));
    f.Map(Base + 32, 8, 32);
    ASSERT_TRUE(f.pending());
    EXPECT_TRUE(std::equal(Payload.begin(), Payload.end(), f.backing.begin()));
}

TEST(DeferredReadbackPublication, ProtectionOnlyChangeKeepsPhysicalDestination) {
    Fixture f;
    f.Map(Base, 8, 0);
    ASSERT_TRUE(f.Schedule({{Base, 8, 0}}));
    f.areas.at(Base).prot = Core::MemoryProt::CpuRead;
    ASSERT_TRUE(f.pending());
    EXPECT_TRUE(std::equal(Payload.begin(), Payload.end(), f.backing.begin()));
}

TEST(DeferredReadbackPublication, ExistingAliasCannotRetargetAfterItsOriginalBindingIsFreed) {
    Fixture f;
    f.Map(Base, 8, 0);
    f.Map(Base + 32, 8, 0);
    ASSERT_TRUE(f.Schedule({{Base, 8, 0}}));
    f.areas.erase(Base);
    ASSERT_TRUE(f.incarnations.Erase(Base, 8));
    // Another alias still refers to the original physical bytes, but it is not the destination
    // that this deferred publication owns. Do not silently retarget through that alias.
    const auto before = f.backing;
    EXPECT_FALSE(f.pending());
    EXPECT_EQ(f.backing, before);
}

TEST(DeferredReadbackPublication, InvalidCaptureDoesNotQueuePartialWriteOrClearOwnership) {
    Fixture f;
    f.Map(Base, 4, 0);
    EXPECT_FALSE(f.Schedule({{Base, 8, 0}}));
    EXPECT_FALSE(f.pending);
    EXPECT_EQ(f.publications, 0u);
    EXPECT_TRUE(std::ranges::all_of(f.backing, [](u8 value) { return value == 0xcc; }));
}

TEST(DeferredReadbackPublication, SourceOffsetsAndGuestOverflowAreValidatedBeforeEnqueue) {
    Fixture f;
    f.Map(Base, 8, 0);
    EXPECT_FALSE(f.Schedule({{Base, 4, 6}}));
    EXPECT_FALSE(f.Schedule({{Base, 1, std::numeric_limits<u64>::max()}}));
    EXPECT_FALSE(f.Schedule({{std::numeric_limits<VAddr>::max() - 3, 8, 0}}));
    EXPECT_FALSE(f.pending);
}

TEST(DeferredReadbackPublication, SourceSizeAndAllHostDestinationsValidateBeforeAnyCopy) {
    Fixture f;
    f.Map(Base, 4, 0);
    f.Map(Base + 32, 4, 16);
    const std::array ranges{Core::ReadbackWriteRange{Base, 4, 0},
                            Core::ReadbackWriteRange{Base + 32, 4, 4}};
    const auto token = Core::CaptureDeferredBackingWrite(f.areas, f.incarnations, ranges, 8);
    ASSERT_TRUE(token);
    const auto before = f.backing;
    EXPECT_FALSE(Core::PublishDeferredBackingWrite(
        f.mutex, f.areas, f.incarnations, f.backing.data(), *token,
        std::span<const u8>{Payload}.first(4), [&](auto) { ++f.publications; }));
    EXPECT_FALSE(Core::PublishDeferredBackingWrite(
        f.mutex, f.areas, f.incarnations,
        reinterpret_cast<u8*>(std::numeric_limits<uintptr_t>::max() - 8), *token, Payload,
        [&](auto) { ++f.publications; }));
    EXPECT_EQ(f.backing, before);
    EXPECT_EQ(f.publications, 0u);
}

TEST(DeferredReadbackPublication, StructuralSplittingPreservesDestinationIdentity) {
    Fixture f;
    f.Map(Base, 8, 0);
    ASSERT_TRUE(f.Schedule({{Base, 8, 0}}));
    auto right = f.areas.at(Base);
    right.base = Base + 4;
    right.size = 4;
    right.phys_areas = {{0, {.base = 4, .size = 4}}};
    f.areas.at(Base).size = 4;
    f.areas.at(Base).phys_areas.at(0).size = 4;
    f.areas.emplace(Base + 4, std::move(right));
    ASSERT_TRUE(f.pending());
    EXPECT_TRUE(std::equal(Payload.begin(), Payload.end(), f.backing.begin()));
}

TEST(DeferredReadbackPublication, PublicationAndSuccessCleanupRetainTheMappingLock) {
    using namespace std::chrono_literals;
    Fixture f;
    f.Map(Base, 8, 0);
    const std::array ranges{Core::ReadbackWriteRange{Base, 8, 0}};
    const auto token = Core::CaptureDeferredBackingWrite(f.areas, f.incarnations, ranges, 8);
    ASSERT_TRUE(token);
    std::promise<void> entered;
    std::promise<void> finish_cleanup;
    auto release = finish_cleanup.get_future().share();
    auto publisher = std::async(std::launch::async, [&] {
        return Core::PublishDeferredBackingWrite(f.mutex, f.areas, f.incarnations, f.backing.data(),
                                                 *token, Payload, [&](auto) {
                                                     entered.set_value();
                                                     release.wait();
                                                     ++f.publications;
                                                 });
    });
    const auto started = entered.get_future().wait_for(1s);
    EXPECT_EQ(started, std::future_status::ready);
    const bool admitted_while_publishing = f.mutex.try_lock();
    if (admitted_while_publishing) {
        f.mutex.unlock();
    }
    finish_cleanup.set_value(); // Always release, even after a failed timing assertion.
    EXPECT_FALSE(admitted_while_publishing);
    EXPECT_TRUE(publisher.get());
    {
        std::unique_lock lock{f.mutex};
        f.Map(Base, 8, 16);
    }
    EXPECT_EQ(f.publications, 1u);
    EXPECT_TRUE(std::equal(Payload.begin(), Payload.end(), f.backing.begin()));
    EXPECT_TRUE(std::ranges::all_of(std::span{f.backing}.subspan(16, 8),
                                    [](u8 value) { return value == 0xcc; }));
}

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input.good());
    std::string source{std::istreambuf_iterator<char>{input}, {}};
    std::erase(source, '\r');
    return source;
}

std::string Function(const std::string& source, const std::string& signature) {
    const auto begin = source.find(signature);
    EXPECT_NE(begin, std::string::npos);
    if (begin == std::string::npos)
        return {};
    return source.substr(begin, source.find("\n}\n", begin) - begin);
}

TEST(DeferredReadbackWiring, MemoryManagerUsesTheLockingPublisherAndIncarnationOnlyIndex) {
    const auto source = ReadSource(SHADPS4_MEMORY_SOURCE_PATH);
    const auto capture = Function(source, "MemoryManager::CaptureReadbackDestinations(");
    EXPECT_LT(capture.find("std::shared_lock lock{mutex}"),
              capture.find("CaptureDeferredBackingWrite(vma_map, virtual_identity"));
    const auto publish = Function(source, "MemoryManager::TryPublishReadback(");
    EXPECT_NE(publish.find("PublishDeferredBackingWrite(mutex, vma_map, virtual_identity"),
              std::string::npos);
    EXPECT_EQ(publish.find("TryWriteBacking("), std::string::npos);
    const auto protect = Function(source, "MemoryManager::ProtectBytes(");
    EXPECT_EQ(protect.find("virtual_identity.Stamp"), std::string::npos);
    for (const char* operation : {"CreateArea", "PoolCommit"}) {
        EXPECT_NE(Function(source, "MemoryManager::" + std::string{operation} + "(")
                      .find("virtual_identity.Stamp"),
                  std::string::npos);
    }
    for (const char* operation : {"UnmapBytesFromEntry", "PoolDecommit"}) {
        EXPECT_NE(Function(source, "MemoryManager::" + std::string{operation} + "(")
                      .find("virtual_identity.Erase"),
                  std::string::npos);
    }
}

TEST(DeferredReadbackWiring, BufferCapturePrecedesDirtyConsumptionAndGpuCopyRecording) {
    const auto body = Function(ReadSource(SHADPS4_BUFFER_CACHE_SOURCE_PATH),
                               "void BufferCache::DownloadBufferMemory(");
    const auto capture = body.find("CaptureReadbackDestinations");
    ASSERT_NE(capture, std::string::npos);
    EXPECT_LT(capture, body.find("gpu_modified_ranges.Subtract"));
    EXPECT_LT(capture, body.find("download_buffer.Map"));
    EXPECT_LT(capture, body.find("cmdbuf.copyBuffer"));
    const auto callback = body.substr(body.find("scheduler.DeferOperation"));
    EXPECT_NE(callback.find("token = std::move(*publication)"), std::string::npos);
    EXPECT_NE(callback.find("TryPublishReadback("), std::string::npos);
    EXPECT_NE(callback.find("token, {download, total_size_bytes}"), std::string::npos);
    EXPECT_NE(callback.find("AreGpuReadbackOwnersCompatible"), std::string::npos);
    EXPECT_LT(callback.find("AreGpuReadbackOwnersCompatible"),
              callback.find("TryPublishReadback("));
    EXPECT_LT(callback.find("TryPublishReadback("),
              callback.find("cpu_page_write_tracker.Discard"));
    EXPECT_NE(callback.find("for (const auto& range : ranges)"), std::string::npos);
    EXPECT_EQ(callback.find("for (const auto& range : token.ranges)"), std::string::npos);
}

TEST(DeferredReadbackWiring, AsyncPublicationDoesNotDiscardWholeCpuSnapshotPages) {
    const auto body = Function(ReadSource(SHADPS4_BUFFER_CACHE_SOURCE_PATH),
                               "void BufferCache::DownloadBufferMemory(");
    const auto async_begin = body.rfind("if constexpr (async)");
    ASSERT_NE(async_begin, std::string::npos);
    const auto async = body.substr(async_begin);
    EXPECT_NE(async.find("cpu_page_write_tracker.DiscardPublished"), std::string::npos);
    const auto async_end = async.find("} else {");
    ASSERT_NE(async_end, std::string::npos);
    EXPECT_EQ(async.substr(0, async_end).find("cpu_page_write_tracker.Discard("),
              std::string::npos);
}

TEST(DeferredReadbackWiring, AsyncPublicationReconcilesOnlyPublishedGpuRanges) {
    const auto body = Function(ReadSource(SHADPS4_BUFFER_CACHE_SOURCE_PATH),
                               "void BufferCache::DownloadBufferMemory(");
    const auto async_begin = body.rfind("if constexpr (async)");
    ASSERT_NE(async_begin, std::string::npos);
    const auto async = body.substr(async_begin);
    const auto async_end = async.find("} else {");
    ASSERT_NE(async_end, std::string::npos);
    const auto callback = async.substr(0, async_end);
    EXPECT_NE(callback.find("ReconcilePublishedGpuOwnerRanges"), std::string::npos);
    EXPECT_NE(callback.find("memory_tracker->UnmarkRegionAsGpuModified"), std::string::npos);
}

TEST(DeferredReadbackWiring, ImageCapturePrecedesCopyAndDeferredCallbackOwnsOnlyToken) {
    const auto body = Function(ReadSource(SHADPS4_TEXTURE_CACHE_SOURCE_PATH),
                               "void TextureCache::DownloadImageMemory(");
    const auto capture = body.find("CaptureReadbackDestinations");
    ASSERT_NE(capture, std::string::npos);
    EXPECT_LT(capture, body.find("download_buffer.Map"));
    EXPECT_LT(capture, body.find("cmdbuf.copyImageToBuffer"));
    const auto callback = body.substr(body.find("scheduler.DeferPriorityOperation"));
    EXPECT_NE(callback.find("token = std::move(*publication)"), std::string::npos);
    EXPECT_NE(callback.find("TryPublishReadback(token"), std::string::npos);
    EXPECT_EQ(callback.find("TryWriteBacking("), std::string::npos);
    // Synchronous publication deliberately retains its existing physical-prefix write policy.
    EXPECT_NE(body.find("TryWriteBacking("), std::string::npos);
}

TEST(DeferredReadbackWiring, AsyncImagePublicationInvalidatesTheMappedDownloadRange) {
    const auto body = Function(ReadSource(SHADPS4_TEXTURE_CACHE_SOURCE_PATH),
                               "void TextureCache::DownloadImageMemory(");
    const auto callback = body.substr(body.find("scheduler.DeferPriorityOperation"));
    const auto invalidate = callback.find("InvalidateMappedRange");
    const auto publish = callback.find("TryPublishReadback");
    ASSERT_NE(invalidate, std::string::npos);
    ASSERT_NE(publish, std::string::npos);
    EXPECT_LT(invalidate, publish);
}

} // namespace
