// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <limits>

#include "video_core/buffer_cache/physical_backing_publication_state.h"

namespace {

using VideoCore::PhysicalBackingDeviceAddress;
using VideoCore::PhysicalBackingPublicationState;

constexpr u64 PageSize = PhysicalBackingPublicationState::PageSize;
constexpr u64 ImportedBase = 0x1'0000'0000;
constexpr u64 OverrideBase = 0x2'0000'0000;
constexpr VAddr GuestA = 0x1000'0000;
constexpr VAddr GuestB = GuestA + PageSize;
constexpr u64 PhysicalPage = 3 * PageSize;

PhysicalBackingPublicationState MakeState() {
    return {PhysicalBackingDeviceAddress{ImportedBase}, 16 * PageSize};
}

} // namespace

TEST(PhysicalBackingPublicationState, MapsEligibleGuestPageToImportedPhysicalBacking) {
    auto state = MakeState();
    const auto mapping = state.MapGuestPage(GuestA, PhysicalPage, 1, 1);

    ASSERT_TRUE(mapping.has_value());
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationState, ResolvesPhysicalAliasesThroughOneSharedState) {
    auto state = MakeState();
    const auto first = state.MapGuestPage(GuestA, PhysicalPage, 1, 1);
    const auto second = state.MapGuestPage(GuestB, PhysicalPage, 2, 1);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());

    const auto override = state.ActivateOverride(
        PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase + 5 * PageSize}, 7);
    ASSERT_TRUE(override.has_value());
    EXPECT_EQ(state.Resolve(GuestA).value, OverrideBase + 5 * PageSize);
    EXPECT_EQ(state.Resolve(GuestB).value, OverrideBase + 5 * PageSize);

    ASSERT_TRUE(state.RetireClean(*override));
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
    EXPECT_EQ(state.Resolve(GuestB).value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationState, DirtyRetirementSuppressesAliasesUntilOrderedWriteback) {
    auto state = MakeState();
    ASSERT_TRUE(state.MapGuestPage(GuestA, PhysicalPage, 1, 1));
    ASSERT_TRUE(state.MapGuestPage(GuestB, PhysicalPage, 2, 1));
    const auto override =
        state.ActivateOverride(PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, 11);
    ASSERT_TRUE(override.has_value());

    const auto writeback = state.RetireGpuDirty(*override);
    ASSERT_TRUE(writeback.has_value());
    EXPECT_EQ(state.Resolve(GuestA).value, u64{0});
    EXPECT_EQ(state.Resolve(GuestB).value, u64{0});

    ASSERT_TRUE(state.CompleteOrderedWriteback(*writeback));
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
    EXPECT_EQ(state.Resolve(GuestB).value, ImportedBase + PhysicalPage);
    EXPECT_FALSE(state.CompleteOrderedWriteback(*writeback));
}

TEST(PhysicalBackingPublicationState, StaleMappingGenerationCannotRestoreDirtyPage) {
    auto state = MakeState();
    const auto old_mapping = state.MapGuestPage(GuestA, PhysicalPage, 1, 1);
    ASSERT_TRUE(old_mapping.has_value());
    const auto override =
        state.ActivateOverride(PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, 1);
    ASSERT_TRUE(override.has_value());
    const auto stale_writeback = state.RetireGpuDirty(*override);
    ASSERT_TRUE(stale_writeback.has_value());

    ASSERT_TRUE(state.UnmapGuestPage(*old_mapping));
    ASSERT_TRUE(state.MapGuestPage(GuestA, PhysicalPage, 2, 1));
    EXPECT_FALSE(state.CompleteOrderedWriteback(*stale_writeback));
    EXPECT_EQ(state.Resolve(GuestA).value, u64{0});

    const auto replacement = state.ActivateOverride(
        PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase + PageSize}, 2);
    ASSERT_TRUE(replacement.has_value());
    EXPECT_EQ(state.Resolve(GuestA).value, OverrideBase + PageSize);
    const auto current_writeback = state.RetireGpuDirty(*replacement);
    ASSERT_TRUE(current_writeback.has_value());
    EXPECT_TRUE(state.CompleteOrderedWriteback(*current_writeback));
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationState, StaleOwnerCompletionCannotReplaceNewerOwner) {
    auto state = MakeState();
    ASSERT_TRUE(state.MapGuestPage(GuestA, PhysicalPage, 1, 1));
    const auto first =
        state.ActivateOverride(PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, 1);
    ASSERT_TRUE(first.has_value());
    const auto stale_writeback = state.RetireGpuDirty(*first);
    ASSERT_TRUE(stale_writeback.has_value());
    ASSERT_TRUE(state.CompleteOrderedWriteback(*stale_writeback));

    const auto second = state.ActivateOverride(
        PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase + PageSize}, 2);
    ASSERT_TRUE(second.has_value());
    const auto current_writeback = state.RetireGpuDirty(*second);
    ASSERT_TRUE(current_writeback.has_value());

    EXPECT_FALSE(state.CompleteOrderedWriteback(*stale_writeback));
    EXPECT_EQ(state.Resolve(GuestA).value, u64{0});
    EXPECT_TRUE(state.CompleteOrderedWriteback(*current_writeback));
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationState, UnmappingOneAliasDoesNotAffectAnother) {
    auto state = MakeState();
    const auto first = state.MapGuestPage(GuestA, PhysicalPage, 1, 1);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(state.MapGuestPage(GuestB, PhysicalPage, 2, 1));

    ASSERT_TRUE(state.UnmapGuestPage(*first));
    EXPECT_EQ(state.Resolve(GuestA).value, u64{0});
    EXPECT_EQ(state.Resolve(GuestB).value, ImportedBase + PhysicalPage);
    EXPECT_FALSE(state.UnmapGuestPage(*first));
}

TEST(PhysicalBackingPublicationState, GuestMappingGenerationAdvancesAcrossRemap) {
    auto state = MakeState();
    const auto old_mapping = state.MapGuestPage(GuestA, PhysicalPage, 5, 1);
    ASSERT_TRUE(old_mapping.has_value());
    ASSERT_TRUE(state.UnmapGuestPage(*old_mapping));

    EXPECT_FALSE(state.MapGuestPage(GuestA, PhysicalPage, 5, 1));
    EXPECT_FALSE(state.MapGuestPage(GuestA, PhysicalPage, 4, 1));
    const auto current_mapping = state.MapGuestPage(GuestA, PhysicalPage, 6, 1);
    ASSERT_TRUE(current_mapping.has_value());
    EXPECT_FALSE(state.UnmapGuestPage(*old_mapping));
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
    EXPECT_TRUE(state.UnmapGuestPage(*current_mapping));
}

TEST(PhysicalBackingPublicationState, InvalidAndConflictingMappingsFailClosed) {
    auto state = MakeState();
    const auto original = state.MapGuestPage(GuestA, PhysicalPage, 1, 1);
    ASSERT_TRUE(original.has_value());

    EXPECT_FALSE(state.MapGuestPage(GuestA, PhysicalPage, 1, 1));
    EXPECT_FALSE(state.MapGuestPage(GuestA, PhysicalPage + PageSize, 2, 1));
    EXPECT_FALSE(state.MapGuestPage(GuestA + 1, PhysicalPage, 3, 1));
    EXPECT_FALSE(state.MapGuestPage(GuestB, PhysicalPage + 1, 4, 1));
    EXPECT_FALSE(state.MapGuestPage(GuestB, 16 * PageSize, 5, 1));
    EXPECT_FALSE(state.MapGuestPage(GuestB, PhysicalPage, 0, 1));
    EXPECT_FALSE(state.MapGuestPage(GuestB, PhysicalPage, 5, 0));
    EXPECT_FALSE(state.MapGuestPage(GuestB, PhysicalPage, 5, 2));
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
    EXPECT_EQ(state.Resolve(GuestB).value, u64{0});
}

TEST(PhysicalBackingPublicationState, AddressAndBackingOverflowsFailClosed) {
    PhysicalBackingPublicationState bda_overflow{
        PhysicalBackingDeviceAddress{std::numeric_limits<u64>::max() - PageSize + 1}, 2 * PageSize};
    EXPECT_FALSE(bda_overflow.MapGuestPage(GuestA, PageSize, 1, 1));

    auto state = MakeState();
    EXPECT_FALSE(
        state.MapGuestPage(std::numeric_limits<VAddr>::max() - PageSize + 1, PhysicalPage, 1, 1));

    PhysicalBackingPublicationState unaligned_backing{PhysicalBackingDeviceAddress{ImportedBase},
                                                      PageSize + 1};
    EXPECT_FALSE(unaligned_backing.MapGuestPage(GuestA, 0, 1, 1));
}

TEST(PhysicalBackingPublicationState, DuplicateAndStaleOverridesFailClosed) {
    auto state = MakeState();
    ASSERT_TRUE(state.MapGuestPage(GuestA, PhysicalPage, 1, 1));
    const auto first =
        state.ActivateOverride(PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, 9);
    ASSERT_TRUE(first.has_value());

    EXPECT_FALSE(
        state.ActivateOverride(PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, 9));
    EXPECT_FALSE(state.ActivateOverride(PhysicalPage,
                                        PhysicalBackingDeviceAddress{OverrideBase + PageSize}, 10));
    EXPECT_FALSE(state.RetireClean(
        {.physical_offset = PhysicalPage, .owner_generation = 8, .state_generation = 1}));
    EXPECT_EQ(state.Resolve(GuestA).value, OverrideBase);
    EXPECT_TRUE(state.RetireClean(*first));
    EXPECT_FALSE(state.RetireClean(*first));
}

TEST(PhysicalBackingPublicationState, ReallocatedPhysicalPageRejectsOldWriteback) {
    auto state = MakeState();
    const auto old_mapping = state.MapGuestPage(GuestA, PhysicalPage, 1, 40);
    ASSERT_TRUE(old_mapping.has_value());
    const auto old_override =
        state.ActivateOverride(PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, 1);
    ASSERT_TRUE(old_override.has_value());
    const auto old_writeback = state.RetireGpuDirty(*old_override);
    ASSERT_TRUE(old_writeback.has_value());
    ASSERT_TRUE(state.UnmapGuestPage(*old_mapping));

    ASSERT_TRUE(state.ReallocatePhysicalPage(PhysicalPage, 40, 41));
    EXPECT_FALSE(state.MapGuestPage(GuestA, PhysicalPage, 2, 40));
    ASSERT_TRUE(state.MapGuestPage(GuestA, PhysicalPage, 2, 41));
    EXPECT_FALSE(state.CompleteOrderedWriteback(*old_writeback));
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
}
