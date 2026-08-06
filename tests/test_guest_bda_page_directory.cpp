// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <limits>

#include "video_core/buffer_cache/guest_bda_page_directory.h"

namespace {

using VideoCore::GuestBdaBackingKind;
using VideoCore::GuestBdaCacheCoherence;
using VideoCore::GuestBdaDeviceAddress;
using VideoCore::GuestBdaFallbackRange;
using VideoCore::GuestBdaPageDirectory;
using VideoCore::GuestBdaPageProvenance;
using VideoCore::GuestBdaPageRange;
using VideoCore::GuestBdaPhysicalRange;

constexpr u64 PageSize = GuestBdaPageDirectory::PageSize;

GuestBdaFallbackRange MakeFallback(VAddr guest_address, u64 size, u64 backing_id,
                                   u64 backing_offset, u64 device_address) {
    return {
        .guest = {.address = guest_address, .size = size},
        .physical = {.backing_id = backing_id, .offset = backing_offset},
        .device_address = GuestBdaDeviceAddress{device_address},
        .kind = GuestBdaBackingKind::SharedPhysical,
        .coherent = true,
        .gpu_dirty = false,
    };
}

} // namespace

TEST(GuestBdaPageDirectory, MapsPhysicalFallbackAtPageGranularity) {
    GuestBdaPageDirectory directory;
    const auto mapping =
        directory.MapFallback(MakeFallback(0x20'0000, 2 * PageSize, 7, 0x80'0000, 0x1'0000'0000));
    ASSERT_TRUE(mapping.has_value());

    const auto first = directory.ResolvePage(0x20'0000);
    const auto second = directory.ResolvePage(0x20'0000 + PageSize);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(first->device_address, GuestBdaDeviceAddress{0x1'0000'0000});
    EXPECT_EQ(first->physical.backing_id, u64{7});
    EXPECT_EQ(first->physical.offset, u64{0x80'0000});
    EXPECT_EQ(first->provenance, GuestBdaPageProvenance::PhysicalFallback);
    EXPECT_EQ(second->device_address, GuestBdaDeviceAddress{0x1'0000'0000 + PageSize});
    EXPECT_EQ(second->physical.offset, 0x80'0000 + PageSize);
}

TEST(GuestBdaPageDirectory, CacheOverrideRestoresOnlyCoherentFallback) {
    GuestBdaPageDirectory directory;
    const auto mapping =
        directory.MapFallback(MakeFallback(0x40'0000, 2 * PageSize, 11, 0x100'0000, 0x2'0000'0000));
    ASSERT_TRUE(mapping.has_value());

    const auto first_override =
        directory.RegisterCache(*mapping, GuestBdaDeviceAddress{0x3'0000'0000});
    ASSERT_TRUE(first_override.has_value());
    const auto cached = directory.ResolvePage(0x40'0000 + PageSize);
    ASSERT_TRUE(cached.has_value());
    EXPECT_EQ(cached->device_address, GuestBdaDeviceAddress{0x3'0000'0000 + PageSize});
    EXPECT_EQ(cached->provenance, GuestBdaPageProvenance::CacheOverride);

    EXPECT_TRUE(
        directory.UnregisterCache(*first_override, GuestBdaCacheCoherence::CoherentWithBacking));
    const auto restored = directory.ResolvePage(0x40'0000 + PageSize);
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(restored->device_address, GuestBdaDeviceAddress{0x2'0000'0000 + PageSize});
    EXPECT_EQ(restored->provenance, GuestBdaPageProvenance::PhysicalFallback);

    const auto dirty_override =
        directory.RegisterCache(*mapping, GuestBdaDeviceAddress{0x4'0000'0000});
    ASSERT_TRUE(dirty_override.has_value());
    EXPECT_TRUE(directory.UnregisterCache(*dirty_override, GuestBdaCacheCoherence::GpuDirty));
    EXPECT_FALSE(directory.ResolvePage(0x40'0000).has_value());
    EXPECT_FALSE(directory.ResolvePage(0x40'0000 + PageSize).has_value());
}

TEST(GuestBdaPageDirectory, UnsupportedAndDirtyFallbackMappingsFailClosed) {
    GuestBdaPageDirectory directory;
    auto unsupported = MakeFallback(0x60'0000, PageSize, 1, 0, 0x5'0000'0000);

    for (const auto kind : {GuestBdaBackingKind::Unsupported, GuestBdaBackingKind::Private,
                            GuestBdaBackingKind::File}) {
        unsupported.kind = kind;
        EXPECT_FALSE(directory.MapFallback(unsupported).has_value());
        EXPECT_FALSE(directory.ResolvePage(unsupported.guest.address).has_value());
        unsupported.guest.address += PageSize;
    }

    auto incoherent = MakeFallback(0x70'0000, PageSize, 2, 0, 0x6'0000'0000);
    incoherent.coherent = false;
    EXPECT_FALSE(directory.MapFallback(incoherent).has_value());
    EXPECT_FALSE(directory.ResolvePage(incoherent.guest.address).has_value());

    auto dirty =
        MakeFallback(0x70'0000 + PageSize, PageSize, 2, PageSize, 0x6'0000'0000 + PageSize);
    dirty.gpu_dirty = true;
    EXPECT_FALSE(directory.MapFallback(dirty).has_value());
    EXPECT_FALSE(directory.ResolvePage(dirty.guest.address).has_value());
}

TEST(GuestBdaPageDirectory, UnmapAndRemapRejectStaleOverrides) {
    GuestBdaPageDirectory directory;
    const auto old_mapping =
        directory.MapFallback(MakeFallback(0x80'0000, PageSize, 3, 0, 0x7'0000'0000));
    ASSERT_TRUE(old_mapping.has_value());
    const auto old_override =
        directory.RegisterCache(*old_mapping, GuestBdaDeviceAddress{0x8'0000'0000});
    ASSERT_TRUE(old_override.has_value());
    EXPECT_TRUE(directory.Unmap(*old_mapping));

    const auto new_mapping =
        directory.MapFallback(MakeFallback(0x80'0000, PageSize, 4, 0x20'0000, 0x9'0000'0000));
    ASSERT_TRUE(new_mapping.has_value());
    EXPECT_NE(old_mapping->generation, new_mapping->generation);
    EXPECT_FALSE(
        directory.RegisterCache(*old_mapping, GuestBdaDeviceAddress{0xA'0000'0000}).has_value());
    EXPECT_FALSE(
        directory.UnregisterCache(*old_override, GuestBdaCacheCoherence::CoherentWithBacking));

    const auto current = directory.ResolvePage(0x80'0000);
    ASSERT_TRUE(current.has_value());
    EXPECT_EQ(current->device_address, GuestBdaDeviceAddress{0x9'0000'0000});
    EXPECT_EQ(current->physical.backing_id, u64{4});
    EXPECT_EQ(current->provenance, GuestBdaPageProvenance::PhysicalFallback);
}

TEST(GuestBdaPageDirectory, PhysicalAliasesResolveToSameBackingOffset) {
    GuestBdaPageDirectory directory;
    const auto first =
        directory.MapFallback(MakeFallback(0xA0'0000, PageSize, 17, 0x400'0000, 0xB'0000'0000));
    const auto alias =
        directory.MapFallback(MakeFallback(0xC0'0000, PageSize, 17, 0x400'0000, 0xB'0000'0000));
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(alias.has_value());

    const auto first_page = directory.ResolvePage(0xA0'0000);
    const auto alias_page = directory.ResolvePage(0xC0'0000);
    ASSERT_TRUE(first_page.has_value());
    ASSERT_TRUE(alias_page.has_value());
    EXPECT_EQ(first_page->physical, alias_page->physical);
    EXPECT_EQ(first_page->device_address, alias_page->device_address);
}

TEST(GuestBdaPageDirectory, RejectsUnalignedWraparoundAndOutOfRangeGuestRanges) {
    GuestBdaPageDirectory directory;
    constexpr VAddr AddressLimit = GuestBdaPageDirectory::AddressLimit;

    EXPECT_FALSE(directory.MapFallback(MakeFallback(0x1001, PageSize, 1, 0, 0x1000)).has_value());
    EXPECT_FALSE(
        directory.MapFallback(MakeFallback(0x4000, PageSize - 1, 1, 0, 0x4000)).has_value());
    EXPECT_FALSE(
        directory.MapFallback(MakeFallback(AddressLimit, PageSize, 1, 0, 0x8000)).has_value());
    EXPECT_FALSE(
        directory.MapFallback(MakeFallback(AddressLimit - PageSize, 2 * PageSize, 1, 0, 0xC000))
            .has_value());
    EXPECT_FALSE(directory
                     .MapFallback(MakeFallback(std::numeric_limits<VAddr>::max() - PageSize + 1,
                                               2 * PageSize, 1, 0, 0x10'000))
                     .has_value());

    const auto final_page =
        directory.MapFallback(MakeFallback(AddressLimit - PageSize, PageSize, 1, 0, 0x20'000));
    ASSERT_TRUE(final_page.has_value());
    EXPECT_TRUE(directory.ResolvePage(AddressLimit - PageSize).has_value());
    EXPECT_FALSE(directory.ResolvePage(AddressLimit).has_value());
}
