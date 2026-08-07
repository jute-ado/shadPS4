// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <optional>
#include <span>
#include <unordered_map>

#include <gtest/gtest.h>

#include "core/physical_backing_provenance.h"
#include "video_core/buffer_cache/physical_backing_publication_coordinator.h"

namespace {

using VideoCore::PhysicalBackingBdaDelta;
using VideoCore::PhysicalBackingDeviceAddress;
using VideoCore::PhysicalBackingPublicationCoordinator;

constexpr u64 PageSize = 16_KB;
constexpr u64 ImportedDeviceBase = 0x4'0000'0000ULL;
constexpr VAddr GuestA = 0x1000'0000ULL;
constexpr VAddr GuestB = 0x2000'0000ULL;
constexpr u64 PhysicalPage = 3 * PageSize;
constexpr u64 SentinelOffset = 64;
constexpr u32 Sentinel = 0x4E415448;

class FakeDirectMemory {
public:
    void StoreSentinel() {
        std::memcpy(backing.data() + PhysicalPage + SentinelOffset, &Sentinel, sizeof(Sentinel));
    }

    void Apply(std::span<const PhysicalBackingBdaDelta> deltas) {
        for (const auto& delta : deltas) {
            page_table[delta.guest_page] = delta.device_address.value;
        }
    }

    [[nodiscard]] std::optional<u32> Read(VAddr guest_address) const {
        const VAddr guest_page = guest_address & ~(PageSize - 1);
        const auto publication = page_table.find(guest_page);
        if (publication == page_table.end() || publication->second == 0 ||
            publication->second < ImportedDeviceBase) {
            return std::nullopt;
        }
        const u64 physical_offset = publication->second - ImportedDeviceBase;
        const u64 page_offset = guest_address & (PageSize - 1);
        if (physical_offset + page_offset > backing.size() - sizeof(u32)) {
            return std::nullopt;
        }
        u32 value{};
        std::memcpy(&value, backing.data() + physical_offset + page_offset, sizeof(value));
        return value;
    }

private:
    std::array<std::byte, 8 * PageSize> backing{};
    std::unordered_map<VAddr, u64> page_table;
};

} // namespace

TEST(PristinePhysicalBackingDma, DynamicReadBeforeCacheDiscoveryUsesOriginalSentinel) {
    PhysicalBackingPublicationCoordinator coordinator{
        PhysicalBackingDeviceAddress{ImportedDeviceBase}, 8 * PageSize};
    constexpr std::array spans{
        Core::PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
    };
    FakeDirectMemory memory;
    memory.StoreSentinel();

    const auto publications = coordinator.MapSpans(spans);

    ASSERT_TRUE(publications.has_value());
    memory.Apply(*publications);
    ASSERT_TRUE(memory.Read(GuestA + SentinelOffset).has_value());
    EXPECT_EQ(*memory.Read(GuestA + SentinelOffset), Sentinel);
}

TEST(PristinePhysicalBackingDma, GpuWriteSuppressesEveryAliasBeforeTheWriter) {
    PhysicalBackingPublicationCoordinator coordinator{
        PhysicalBackingDeviceAddress{ImportedDeviceBase}, 8 * PageSize};
    constexpr std::array spans{
        Core::PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        Core::PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    FakeDirectMemory memory;
    memory.StoreSentinel();
    const auto publications = coordinator.MapSpans(spans);
    ASSERT_TRUE(publications.has_value());
    memory.Apply(*publications);
    ASSERT_EQ(memory.Read(GuestA + SentinelOffset), Sentinel);
    ASSERT_EQ(memory.Read(GuestB + SentinelOffset), Sentinel);

    const auto suppression =
        coordinator.SuppressGuestRangeForGpuWrite(GuestA + SentinelOffset, sizeof(Sentinel));

    ASSERT_TRUE(suppression.has_value());
    memory.Apply(*suppression);
    EXPECT_FALSE(memory.Read(GuestA + SentinelOffset).has_value());
    EXPECT_FALSE(memory.Read(GuestB + SentinelOffset).has_value());
}
