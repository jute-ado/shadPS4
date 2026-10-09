// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <iterator>
#include <shared_mutex>

#include "core/backing_memory_write.h"

namespace {

using Areas = std::map<VAddr, Core::VirtualMemoryArea>;
constexpr VAddr Base = 0x1000;
constexpr std::array<u8, 8> Payload{1, 2, 3, 4, 5, 6, 7, 8};

Core::VirtualMemoryArea PhysicalArea(VAddr base, u64 size, PAddr physical_base = 0) {
    return {.base = base,
            .size = size,
            .phys_areas = {{0, {.base = physical_base, .size = size}}},
            .type = Core::VMAType::Direct};
}

Core::VirtualMemoryArea NonPhysicalArea(VAddr base, u64 size) {
    return {.base = base, .size = size, .type = Core::VMAType::File};
}

} // namespace

TEST(BackingMemoryWrite, AdvancesSourceAcrossNoncontiguousPhysicalFragments) {
    Areas areas{{Base, PhysicalArea(Base, 8)}, {Base + 8, NonPhysicalArea(Base + 8, 8)}};
    areas.at(Base).phys_areas = {{0, {.base = 0, .size = 4}}, {4, {.base = 16, .size = 4}}};
    std::array<u8, 32> backing{};

    ASSERT_TRUE(Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 8));
    EXPECT_EQ((std::array<u8, 4>{backing[0], backing[1], backing[2], backing[3]}),
              (std::array<u8, 4>{1, 2, 3, 4}));
    EXPECT_EQ((std::array<u8, 4>{backing[16], backing[17], backing[18], backing[19]}),
              (std::array<u8, 4>{5, 6, 7, 8}));
}

TEST(BackingMemoryWrite, NonPhysicalSuffixReturnsIncompleteAfterPublishingValidPrefix) {
    const Areas areas{{Base, PhysicalArea(Base, 4)}, {Base + 4, NonPhysicalArea(Base + 4, 4)}};
    std::array<u8, 32> backing{};
    backing.fill(0xcc);
    auto expected = backing;
    std::copy_n(Payload.begin(), 4, expected.begin());

    EXPECT_FALSE(Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 8));
    EXPECT_EQ(backing, expected);
}

TEST(BackingMemoryWrite, CompleteSingleFragmentCopiesRequestedBytes) {
    const Areas areas{{Base, PhysicalArea(Base, 8)}, {Base + 8, NonPhysicalArea(Base + 8, 8)}};
    std::array<u8, 32> backing{};

    ASSERT_TRUE(Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 8));
    EXPECT_TRUE(std::equal(Payload.begin(), Payload.end(), backing.begin()));
}

TEST(BackingMemoryWrite, AdvancesSourceAcrossAdjacentVirtualAreas) {
    const Areas areas{{Base, PhysicalArea(Base, 4, 16)},
                      {Base + 4, PhysicalArea(Base + 4, 4, 0)},
                      {Base + 8, NonPhysicalArea(Base + 8, 8)}};
    std::array<u8, 32> backing{};

    ASSERT_TRUE(Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 8));
    EXPECT_EQ((std::array<u8, 4>{backing[16], backing[17], backing[18], backing[19]}),
              (std::array<u8, 4>{1, 2, 3, 4}));
    EXPECT_EQ((std::array<u8, 4>{backing[0], backing[1], backing[2], backing[3]}),
              (std::array<u8, 4>{5, 6, 7, 8}));
}

TEST(BackingMemoryWrite, PhysicalGapStopsAtTheValidPrefix) {
    Areas areas{{Base, PhysicalArea(Base, 8)}, {Base + 8, NonPhysicalArea(Base + 8, 8)}};
    areas.at(Base).phys_areas = {{0, {.base = 0, .size = 3}}, {4, {.base = 16, .size = 4}}};
    std::array<u8, 32> backing{};
    auto expected = backing;
    std::copy_n(Payload.begin(), 3, expected.begin());

    EXPECT_FALSE(Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 8));
    EXPECT_EQ(backing, expected);
}

TEST(BackingMemoryWrite, PhysicalFragmentCannotExtendPastItsVirtualArea) {
    Areas areas{{Base, PhysicalArea(Base, 4)}, {Base + 4, NonPhysicalArea(Base + 4, 4)}};
    areas.at(Base).phys_areas.at(0).size = 8;
    std::array<u8, 32> backing{};
    const auto original = backing;

    EXPECT_FALSE(Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 8));
    EXPECT_EQ(backing, original);
}

TEST(BackingMemoryWrite, EmptyWriteNeedsNoBackingOrSource) {
    const Areas areas{{Base, PhysicalArea(Base, 8)}};
    EXPECT_TRUE(Core::TryWriteBackingMemory(areas, nullptr, Base, nullptr, 0));
    EXPECT_TRUE(Core::TryWriteBackingMemory({}, nullptr, 0, nullptr, 0));
}

TEST(BackingMemoryWrite, EverySubrangeAcrossEveryPhysicalSplitCopiesExactlyOnce) {
    for (u64 split = 1; split < Payload.size(); ++split) {
        Areas areas{{Base, PhysicalArea(Base, Payload.size())}};
        areas.at(Base).phys_areas = {{0, {.base = 0, .size = split}},
                                     {split, {.base = 16, .size = Payload.size() - split}}};
        for (u64 start = 0; start < Payload.size(); ++start) {
            for (u64 size = 1; size <= Payload.size() - start; ++size) {
                SCOPED_TRACE(::testing::Message() << split << '/' << start << '/' << size);
                std::array<u8, 32> backing{};
                std::array<u8, 32> expected{};
                for (u64 i = 0; i < size; ++i) {
                    const auto virtual_offset = start + i;
                    const auto physical_offset =
                        virtual_offset < split ? virtual_offset : 16 + virtual_offset - split;
                    expected[physical_offset] = Payload[i];
                }
                ASSERT_TRUE(Core::TryWriteBackingMemory(areas, backing.data(), Base + start,
                                                        Payload.data(), size));
                EXPECT_EQ(backing, expected);
            }
        }
    }
}

TEST(BackingMemoryWrite, ClipsFirstAndLastVirtualAreaOffsets) {
    const Areas areas{{Base, PhysicalArea(Base, 4, 0)}, {Base + 4, PhysicalArea(Base + 4, 4, 16)}};
    std::array<u8, 32> backing{};
    std::array<u8, 32> expected{};
    expected[2] = 1;
    expected[3] = 2;
    expected[16] = 3;
    expected[17] = 4;

    ASSERT_TRUE(Core::TryWriteBackingMemory(areas, backing.data(), Base + 2, Payload.data(), 4));
    EXPECT_EQ(backing, expected);
}

TEST(BackingMemoryWrite, PhysicalBackingTypesDoNotRequireReadableGuestAlias) {
    for (const auto type :
         {Core::VMAType::Direct, Core::VMAType::Flexible, Core::VMAType::Pooled}) {
        Areas areas{{Base, PhysicalArea(Base, 8)}};
        areas.at(Base).type = type;
        areas.at(Base).prot = Core::MemoryProt::NoAccess;
        std::array<u8, 8> backing{};

        ASSERT_TRUE(Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 8));
        EXPECT_EQ(backing, Payload);
    }
}

TEST(BackingMemoryWrite, MissingVirtualCoveragePreservesOnlyTheAvailablePrefix) {
    const std::array<Areas, 4> missing{
        Areas{},
        Areas{{Base + 4, PhysicalArea(Base + 4, 4)}},
        Areas{{Base, PhysicalArea(Base, 4)}},
        Areas{{Base, PhysicalArea(Base, 4)}, {Base + 6, PhysicalArea(Base + 6, 2, 16)}},
    };
    for (size_t i = 0; i < missing.size(); ++i) {
        const auto& areas = missing[i];
        std::array<u8, 32> backing{};
        auto expected = backing;
        if (i >= 2) {
            std::copy_n(Payload.begin(), 4, expected.begin());
        }
        EXPECT_FALSE(Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 8));
        EXPECT_EQ(backing, expected);
    }
}

TEST(BackingMemoryWrite, RangeStartingAfterLastVirtualAreaFails) {
    const Areas areas{{Base, PhysicalArea(Base, 8)}};
    std::array<u8, 32> backing{};
    const auto original = backing;
    EXPECT_FALSE(Core::TryWriteBackingMemory(areas, backing.data(), Base + 8, Payload.data(), 1));
    EXPECT_EQ(backing, original);
}

TEST(BackingMemoryWrite, MissingPhysicalCoveragePreservesOnlyTheAvailablePrefix) {
    const std::array<std::map<uintptr_t, Core::PhysicalMemoryArea>, 5> fragments{
        std::map<uintptr_t, Core::PhysicalMemoryArea>{},
        {{4, {.base = 0, .size = 4}}},
        {{0, {.base = 0, .size = 0}}},
        {{0, {.base = 0, .size = 4}}},
        {{0, {.base = 0, .size = 4}}, {4, {.base = 16, .size = 0}}},
    };
    for (size_t i = 0; i < fragments.size(); ++i) {
        Areas areas{{Base, PhysicalArea(Base, 8)}};
        areas.at(Base).phys_areas = fragments[i];
        std::array<u8, 32> backing{};
        auto expected = backing;
        if (i >= 3) {
            std::copy_n(Payload.begin(), 4, expected.begin());
        }
        EXPECT_FALSE(Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 8));
        EXPECT_EQ(backing, expected);
    }
}

TEST(BackingMemoryWrite, RejectsOverflowingGuestAndVirtualAreaRanges) {
    constexpr auto Max = std::numeric_limits<VAddr>::max();
    std::array<u8, 32> backing{};
    const auto original = backing;
    const Areas normal{{Base, PhysicalArea(Base, 8)}};
    const Areas overflowing{{Max - 7, PhysicalArea(Max - 7, 8)}};
    EXPECT_FALSE(Core::TryWriteBackingMemory(normal, backing.data(), Max - 3, Payload.data(), 8));
    EXPECT_FALSE(
        Core::TryWriteBackingMemory(overflowing, backing.data(), Max - 7, Payload.data(), 1));
    EXPECT_EQ(backing, original);
}

TEST(BackingMemoryWrite, RejectsOverflowingPhysicalMetadataBeforeCopy) {
    std::array<u8, 32> backing{};
    const auto original = backing;
    Areas areas{{Base, PhysicalArea(Base, 8)}};
    areas.at(Base).phys_areas.at(0).base = std::numeric_limits<PAddr>::max() - 3;
    EXPECT_FALSE(Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 1));
    areas.at(Base).phys_areas.at(0).base = 0;
    areas.at(Base).phys_areas.at(0).size = std::numeric_limits<u64>::max();
    EXPECT_FALSE(Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 1));
    EXPECT_EQ(backing, original);
}

TEST(BackingMemoryWrite, RejectsWrappedHostAddressesWithoutDereferencing) {
    constexpr auto Max = std::numeric_limits<uintptr_t>::max();
    const Areas areas{{Base, PhysicalArea(Base, 8, 16)}};
    std::array<u8, 32> backing{};
    const auto original = backing;
    EXPECT_FALSE(Core::TryWriteBackingMemory(areas, backing.data(), Base,
                                             reinterpret_cast<const void*>(Max - 3), 8));
    EXPECT_FALSE(Core::TryWriteBackingMemory(areas, reinterpret_cast<u8*>(Max - 3), Base,
                                             Payload.data(), 8));
    EXPECT_EQ(backing, original);
}

TEST(BackingMemoryWrite, RejectsNullNonemptySourceOrBacking) {
    const Areas areas{{Base, PhysicalArea(Base, 8)}};
    std::array<u8, 32> backing{};
    const auto original = backing;
    EXPECT_FALSE(Core::TryWriteBackingMemory(areas, backing.data(), Base, nullptr, 8));
    EXPECT_FALSE(Core::TryWriteBackingMemory(areas, nullptr, Base, Payload.data(), 8));
    EXPECT_EQ(backing, original);
}

TEST(BackingMemoryWrite, IncompleteFastPathWholeWriteFallbackRewritesIdenticalPrefix) {
    const Areas areas{{Base, PhysicalArea(Base, 4)}, {Base + 4, NonPhysicalArea(Base + 4, 4)}};
    std::array<u8, 32> backing{};
    auto expected_prefix = backing;
    std::copy_n(Payload.begin(), 4, expected_prefix.begin());
    auto expected_complete = backing;
    std::copy(Payload.begin(), Payload.end(), expected_complete.begin());
    bool fallback_called = false;
    if (!Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 8)) {
        EXPECT_EQ(backing, expected_prefix);
        // This array models the guest alias of the prefix and its separately backed suffix.
        // The existing EOP/EOS fallback retries the complete write, not only the missing suffix.
        std::memcpy(backing.data(), Payload.data(), Payload.size());
        fallback_called = true;
    }
    EXPECT_TRUE(fallback_called);
    EXPECT_EQ(backing, expected_complete);
}

TEST(BackingMemoryWrite, IgnoredReadbackResultStillPublishesThePhysicalPrefix) {
    const Areas areas{{Base, PhysicalArea(Base, 4)}, {Base + 4, NonPhysicalArea(Base + 4, 4)}};
    std::array<u8, 32> backing{};
    auto expected = backing;
    std::copy_n(Payload.begin(), 4, expected.begin());
    bool gpu_owned = true;
    const auto existing_download_caller = [&] {
        // Some existing download paths ignore the return and then clear GPU ownership.
        (void)Core::TryWriteBackingMemory(areas, backing.data(), Base, Payload.data(), 8);
        gpu_owned = false;
    };
    existing_download_caller();
    EXPECT_FALSE(gpu_owned);
    EXPECT_EQ(backing, expected);
}

TEST(BackingMemoryWrite, ExactPublicationRefusesChangedIncarnationWithoutPartialWrite) {
    const Areas areas{{Base, PhysicalArea(Base, 8)}};
    Core::MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(Base, 8));
    const Core::ReadbackWriteRange range{Base, 8, 0};
    const auto token = Core::CaptureDeferredBackingWrite(
        areas, identities, std::span<const Core::ReadbackWriteRange>{&range, 1}, Payload.size());
    ASSERT_TRUE(token);

    std::shared_mutex mutex;
    std::array<u8, 32> backing{};
    backing.fill(0xcc);
    ASSERT_TRUE(Core::PublishDeferredBackingWriteExact(
        mutex, areas, identities, backing.data(), *token, std::span<const u8>{Payload}));
    EXPECT_EQ(std::equal(Payload.begin(), Payload.end(), backing.begin()), true);

    ASSERT_TRUE(identities.Erase(Base + 4, 4));
    EXPECT_FALSE(Core::PublishDeferredBackingWriteExact(
        mutex, areas, identities, backing.data(), *token, std::span<const u8>{Payload}));
    EXPECT_EQ(std::count(backing.begin(), backing.end(), u8{0xcc}), 24);
}

TEST(BackingMemoryWrite, ExactPublicationAcrossPhysicalFragmentsIsAllOrNothing) {
    Areas areas{{Base, PhysicalArea(Base, 8)}};
    areas.at(Base).phys_areas = {{0, {.base = 0, .size = 4}},
                                 {4, {.base = 16, .size = 4}}};
    Core::MemoryRangeIdentity identities;
    ASSERT_TRUE(identities.Stamp(Base, 8));
    const std::array ranges{Core::ReadbackWriteRange{Base, 8, 0}};
    const auto token = Core::CaptureDeferredBackingWrite(areas, identities, ranges, Payload.size());
    ASSERT_TRUE(token);
    ASSERT_EQ(token->fragments.size(), 2u);

    std::shared_mutex mutex;
    std::array<u8, 32> backing{};
    backing.fill(0xcc);
    const auto before = backing;

    // The second physical fragment is both replaced and assigned a new incarnation after the
    // GPU work was queued.  Exact fence publication must not expose the still-valid first half.
    areas.at(Base).phys_areas.at(4).base = 24;
    ASSERT_TRUE(identities.Erase(Base + 4, 4));
    ASSERT_TRUE(identities.Stamp(Base + 4, 4));

    EXPECT_FALSE(Core::PublishDeferredBackingWriteExact(
        mutex, areas, identities, backing.data(), *token, std::span<const u8>{Payload}));
    EXPECT_EQ(backing, before);
}

TEST(BackingMemoryWriteWiring, MemoryManagerUsesTestedHelperInsideMappingLock) {
    std::ifstream stream{SHADPS4_MEMORY_SOURCE_PATH};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream}, {}};
    const auto begin = source.find("bool MemoryManager::TryWriteBacking(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = source.find("\n}", begin);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto lock = body.find("std::shared_lock lk{mutex}");
    const auto call = body.find(
        "return TryWriteBackingMemory(vma_map, impl.BackingBase(), virtual_addr, data, size)");
    ASSERT_NE(lock, std::string::npos);
    ASSERT_NE(call, std::string::npos);
    EXPECT_LT(lock, call);
    EXPECT_EQ(body.find("memcpy"), std::string::npos);
    EXPECT_EQ(body.find("IsValidMapping("), std::string::npos);
}
