// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "video_core/buffer_cache/physical_backing_publication_state.h"

namespace {

using VideoCore::PhysicalBackingDeviceAddress;
using VideoCore::PhysicalBackingPublicationState;

constexpr u64 PageSize = PhysicalBackingPublicationState::PageSize;
constexpr u64 ImportedBase = 0x1'0000'0000;
constexpr u64 OverrideBase = 0x2'0000'0000;
constexpr VAddr GuestA = 0x1000'0000;
constexpr VAddr GuestB = GuestA + PageSize;
constexpr VAddr GuestC = GuestB + PageSize;
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

    u32 commit_count = 0;
    bool observed_suppressed_publication = false;
    ASSERT_TRUE(state.CommitOrderedWriteback(*writeback, [&] {
        ++commit_count;
        observed_suppressed_publication =
            state.Resolve(GuestA).value == 0 && state.Resolve(GuestB).value == 0;
        return true;
    }));
    EXPECT_EQ(commit_count, 1u);
    EXPECT_TRUE(observed_suppressed_publication);
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
    EXPECT_EQ(state.Resolve(GuestB).value, ImportedBase + PhysicalPage);
    EXPECT_FALSE(state.CommitOrderedWriteback(*writeback, [&] {
        ++commit_count;
        return true;
    }));
    EXPECT_EQ(commit_count, 1u);
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
    bool stale_commit_called = false;
    EXPECT_FALSE(state.CommitOrderedWriteback(*stale_writeback, [&] {
        stale_commit_called = true;
        return true;
    }));
    EXPECT_FALSE(stale_commit_called);
    EXPECT_EQ(state.Resolve(GuestA).value, u64{0});

    const auto replacement = state.ActivateOverride(
        PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase + PageSize}, 2);
    ASSERT_TRUE(replacement.has_value());
    EXPECT_EQ(state.Resolve(GuestA).value, OverrideBase + PageSize);
    const auto current_writeback = state.RetireGpuDirty(*replacement);
    ASSERT_TRUE(current_writeback.has_value());
    EXPECT_TRUE(state.CommitOrderedWriteback(*current_writeback, [] { return true; }));
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
    ASSERT_TRUE(state.CommitOrderedWriteback(*stale_writeback, [] { return true; }));

    const auto second = state.ActivateOverride(
        PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase + PageSize}, 2);
    ASSERT_TRUE(second.has_value());
    const auto current_writeback = state.RetireGpuDirty(*second);
    ASSERT_TRUE(current_writeback.has_value());

    bool stale_commit_called = false;
    EXPECT_FALSE(state.CommitOrderedWriteback(*stale_writeback, [&] {
        stale_commit_called = true;
        return true;
    }));
    EXPECT_FALSE(stale_commit_called);
    EXPECT_EQ(state.Resolve(GuestA).value, u64{0});
    EXPECT_TRUE(state.CommitOrderedWriteback(*current_writeback, [] { return true; }));
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationState, FailedWritebackRemainsSuppressedAndCanRetry) {
    auto state = MakeState();
    ASSERT_TRUE(state.MapGuestPage(GuestA, PhysicalPage, 1, 1));
    const auto override =
        state.ActivateOverride(PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, 1);
    ASSERT_TRUE(override.has_value());
    const auto writeback = state.RetireGpuDirty(*override);
    ASSERT_TRUE(writeback.has_value());

    u32 commit_count = 0;
    EXPECT_FALSE(state.CommitOrderedWriteback(*writeback, [&] {
        ++commit_count;
        EXPECT_EQ(state.Resolve(GuestA).value, u64{0});
        return false;
    }));
    EXPECT_EQ(commit_count, 1u);
    EXPECT_EQ(state.Resolve(GuestA).value, u64{0});

    EXPECT_TRUE(state.CommitOrderedWriteback(*writeback, [&] {
        ++commit_count;
        EXPECT_EQ(state.Resolve(GuestA).value, u64{0});
        return true;
    }));
    EXPECT_EQ(commit_count, 2u);
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationState, WritebackCallbackRejectsSamePhysicalPageMutation) {
    auto state = MakeState();
    const auto first = state.MapGuestPage(GuestA, PhysicalPage, 1, 1);
    const auto second = state.MapGuestPage(GuestB, PhysicalPage, 2, 1);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    const auto override =
        state.ActivateOverride(PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, 1);
    ASSERT_TRUE(override.has_value());
    const auto writeback = state.RetireGpuDirty(*override);
    ASSERT_TRUE(writeback.has_value());

    u32 commit_count = 0;
    EXPECT_FALSE(state.CommitOrderedWriteback(*writeback, [&] {
        ++commit_count;
        EXPECT_FALSE(state.UnmapGuestPage(*first));
        EXPECT_FALSE(state.MapGuestPage(GuestC, PhysicalPage, 3, 1));
        EXPECT_FALSE(state.ActivateOverride(
            PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase + PageSize}, 2));
        EXPECT_FALSE(state.ReallocatePhysicalPage(PhysicalPage, 1, 2));
        EXPECT_EQ(state.Resolve(GuestA).value, u64{0});
        EXPECT_EQ(state.Resolve(GuestB).value, u64{0});
        return false;
    }));
    EXPECT_EQ(commit_count, 1u);
    EXPECT_EQ(state.Resolve(GuestA).value, u64{0});
    EXPECT_EQ(state.Resolve(GuestB).value, u64{0});
    EXPECT_EQ(state.Resolve(GuestC).value, u64{0});

    EXPECT_TRUE(state.CommitOrderedWriteback(*writeback, [&] {
        ++commit_count;
        return true;
    }));
    EXPECT_EQ(commit_count, 2u);
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
    EXPECT_EQ(state.Resolve(GuestB).value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationState, NestedCommitIsRejectedWithoutInvokingNestedCallback) {
    auto state = MakeState();
    ASSERT_TRUE(state.MapGuestPage(GuestA, PhysicalPage, 1, 1));
    const auto override =
        state.ActivateOverride(PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, 1);
    ASSERT_TRUE(override.has_value());
    const auto writeback = state.RetireGpuDirty(*override);
    ASSERT_TRUE(writeback.has_value());

    u32 outer_count = 0;
    u32 nested_count = 0;
    EXPECT_FALSE(state.CommitOrderedWriteback(*writeback, [&] {
        ++outer_count;
        EXPECT_FALSE(state.CommitOrderedWriteback(*writeback, [&] {
            ++nested_count;
            return true;
        }));
        return false;
    }));
    EXPECT_EQ(outer_count, 1u);
    EXPECT_EQ(nested_count, 0u);
    EXPECT_EQ(state.Resolve(GuestA).value, u64{0});

    EXPECT_TRUE(state.CommitOrderedWriteback(*writeback, [] { return true; }));
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationState, ThrowingWritebackCallbackRestoresRetryablePendingState) {
    auto state = MakeState();
    ASSERT_TRUE(state.MapGuestPage(GuestA, PhysicalPage, 1, 1));
    const auto override =
        state.ActivateOverride(PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, 1);
    ASSERT_TRUE(override.has_value());
    const auto writeback = state.RetireGpuDirty(*override);
    ASSERT_TRUE(writeback.has_value());

    u32 commit_count = 0;
    EXPECT_THROW((void)state.CommitOrderedWriteback(
                     *writeback,
                     [&]() -> bool {
                         ++commit_count;
                         EXPECT_EQ(state.Resolve(GuestA).value, u64{0});
                         throw std::runtime_error{"synthetic writeback failure"};
                     }),
                 std::runtime_error);
    EXPECT_EQ(commit_count, 1u);
    EXPECT_EQ(state.Resolve(GuestA).value, u64{0});

    EXPECT_TRUE(state.CommitOrderedWriteback(*writeback, [&] {
        ++commit_count;
        return true;
    }));
    EXPECT_EQ(commit_count, 2u);
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationState, AliasFreeWritebackRejectsReallocationUntilCommitEnds) {
    auto state = MakeState();
    const auto mapping = state.MapGuestPage(GuestA, PhysicalPage, 1, 1);
    ASSERT_TRUE(mapping.has_value());
    const auto override =
        state.ActivateOverride(PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, 1);
    ASSERT_TRUE(override.has_value());
    ASSERT_TRUE(state.UnmapGuestPage(*mapping));
    const auto writeback = state.RetireGpuDirty(*override);
    ASSERT_TRUE(writeback.has_value());

    u32 commit_count = 0;
    EXPECT_FALSE(state.CommitOrderedWriteback(*writeback, [&] {
        ++commit_count;
        EXPECT_FALSE(state.ReallocatePhysicalPage(PhysicalPage, 1, 2));
        return false;
    }));
    EXPECT_EQ(commit_count, 1u);

    EXPECT_THROW((void)state.CommitOrderedWriteback(
                     *writeback,
                     [&]() -> bool {
                         ++commit_count;
                         EXPECT_FALSE(state.ReallocatePhysicalPage(PhysicalPage, 1, 2));
                         throw std::runtime_error{"synthetic alias-free writeback failure"};
                     }),
                 std::runtime_error);
    EXPECT_EQ(commit_count, 2u);

    EXPECT_TRUE(state.CommitOrderedWriteback(*writeback, [&] {
        ++commit_count;
        return true;
    }));
    EXPECT_EQ(commit_count, 3u);
    ASSERT_TRUE(state.MapGuestPage(GuestA, PhysicalPage, 2, 1));
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationState, UnrelatedPhysicalMutationDoesNotInvalidateCommitGuard) {
    auto state = MakeState();
    ASSERT_TRUE(state.MapGuestPage(GuestA, PhysicalPage, 1, 1));
    const auto override =
        state.ActivateOverride(PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, 1);
    ASSERT_TRUE(override.has_value());
    const auto writeback = state.RetireGpuDirty(*override);
    ASSERT_TRUE(writeback.has_value());

    u64 mapping_generation = 2;
    u64 guest_page_index = 1;
    EXPECT_FALSE(state.CommitOrderedWriteback(*writeback, [&] {
        for (u64 physical_page_index = 0; physical_page_index < 16; ++physical_page_index) {
            const u64 physical_offset = physical_page_index * PageSize;
            if (physical_offset == PhysicalPage) {
                continue;
            }
            EXPECT_TRUE(state.MapGuestPage(GuestA + guest_page_index * PageSize, physical_offset,
                                           mapping_generation, 1));
            ++mapping_generation;
            ++guest_page_index;
        }

        const auto unrelated_override =
            state.ActivateOverride(0, PhysicalBackingDeviceAddress{OverrideBase + 8 * PageSize}, 1);
        EXPECT_TRUE(unrelated_override.has_value());
        if (unrelated_override) {
            EXPECT_TRUE(state.RetireClean(*unrelated_override));
        }
        EXPECT_EQ(state.Resolve(GuestA).value, u64{0});
        return false;
    }));

    EXPECT_EQ(state.Resolve(GuestA).value, u64{0});
    EXPECT_EQ(state.Resolve(GuestB).value, ImportedBase);
    EXPECT_TRUE(state.CommitOrderedWriteback(*writeback, [] { return true; }));
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

TEST(PhysicalBackingPublicationState, MappingGenerationIsGlobalAndMonotonicAcrossGuestPages) {
    auto state = MakeState();
    const auto old_mapping = state.MapGuestPage(GuestA, PhysicalPage, 5, 1);
    ASSERT_TRUE(old_mapping.has_value());

    EXPECT_FALSE(state.MapGuestPage(GuestB, PhysicalPage + PageSize, 5, 1));
    EXPECT_FALSE(state.MapGuestPage(GuestB, PhysicalPage + PageSize, 4, 1));
    ASSERT_TRUE(state.MapGuestPage(GuestB, PhysicalPage + PageSize, 6, 1));
    ASSERT_TRUE(state.UnmapGuestPage(*old_mapping));
    EXPECT_FALSE(state.MapGuestPage(GuestA, PhysicalPage, 6, 1));
    const auto current_mapping = state.MapGuestPage(GuestA, PhysicalPage, 7, 1);
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
    PhysicalBackingPublicationState one_page_bda_overflow{
        PhysicalBackingDeviceAddress{std::numeric_limits<u64>::max()}, PageSize};
    EXPECT_FALSE(one_page_bda_overflow.MapGuestPage(GuestA, 0, 1, 1));

    PhysicalBackingPublicationState bda_overflow{
        PhysicalBackingDeviceAddress{std::numeric_limits<u64>::max() - PageSize + 1}, 2 * PageSize};
    EXPECT_FALSE(bda_overflow.MapGuestPage(GuestA, PageSize, 1, 1));

    const VAddr last_complete_guest_page = std::numeric_limits<VAddr>::max() - (PageSize - 1);
    auto boundary_state = MakeState();
    EXPECT_TRUE(boundary_state.MapGuestPage(last_complete_guest_page, PhysicalPage, 1, 1));

    PhysicalBackingPublicationState last_device_page{
        PhysicalBackingDeviceAddress{std::numeric_limits<u64>::max() - (PageSize - 1)}, PageSize};
    ASSERT_TRUE(last_device_page.MapGuestPage(GuestA, 0, 1, 1));
    EXPECT_EQ(last_device_page.Resolve(GuestA).value,
              std::numeric_limits<u64>::max() - (PageSize - 1));

    PhysicalBackingPublicationState unaligned_backing{PhysicalBackingDeviceAddress{ImportedBase},
                                                      PageSize + 1};
    EXPECT_FALSE(unaligned_backing.MapGuestPage(GuestA, 0, 1, 1));

    auto override_state = MakeState();
    ASSERT_TRUE(override_state.MapGuestPage(GuestA, PhysicalPage, 1, 1));
    EXPECT_FALSE(override_state.ActivateOverride(
        PhysicalPage, PhysicalBackingDeviceAddress{std::numeric_limits<u64>::max()}, 1));
    EXPECT_TRUE(override_state.ActivateOverride(
        PhysicalPage,
        PhysicalBackingDeviceAddress{std::numeric_limits<u64>::max() - (PageSize - 1)}, 1));
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
    bool stale_commit_called = false;
    EXPECT_FALSE(state.CommitOrderedWriteback(*old_writeback, [&] {
        stale_commit_called = true;
        return true;
    }));
    EXPECT_FALSE(stale_commit_called);
    EXPECT_EQ(state.Resolve(GuestA).value, ImportedBase + PhysicalPage);
}
