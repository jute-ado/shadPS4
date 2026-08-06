// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/exact_address_space_host_import_probe.h"

namespace Vulkan {
namespace {

TEST(ExactAddressSpaceHostImportProbe, ComputesExactProductionBackingSizeWithoutOverflow) {
    static_assert(ExactAddressSpaceBaseBackingSize == 8448ull * 1024 * 1024);
    EXPECT_EQ(CalculateExactAddressSpaceBackingSize(0), ExactAddressSpaceBaseBackingSize);
    EXPECT_EQ(CalculateExactAddressSpaceBackingSize(20000),
              ExactAddressSpaceBaseBackingSize + 20000ull * 1024 * 1024);
    EXPECT_FALSE(CalculateExactAddressSpaceBackingSize(20001).has_value());
    EXPECT_FALSE(CalculateExactAddressSpaceBackingSize(std::numeric_limits<std::uint64_t>::max())
                     .has_value());
}

TEST(ExactAddressSpaceHostImportProbe, RequiresBackingSizeToFitWindowsSizeTExactly) {
    EXPECT_EQ(ValidateExactBackingSizeForSizeT(ExactAddressSpaceBaseBackingSize,
                                               std::numeric_limits<std::uint64_t>::max()),
              ExactBackingSizeValidation::Valid);
    EXPECT_EQ(ValidateExactBackingSizeForSizeT(ExactAddressSpaceBaseBackingSize, 0xffffffffull),
              ExactBackingSizeValidation::ExceedsHostSizeT);
    EXPECT_EQ(ValidateExactBackingSizeForSizeT(0, std::numeric_limits<std::uint64_t>::max()),
              ExactBackingSizeValidation::Empty);
}

struct FakeWindowsBackingAdapter {
    using Mapping = std::uint32_t;
    using Reservation = std::uint32_t;

    std::vector<std::string_view> calls;
    ExactWindowsBackingRecipe create_recipe{};
    ExactWindowsBackingRecipe reserve_recipe{};
    ExactWindowsBackingRecipe map_recipe{};
    bool create_succeeds{true};
    bool reserve_succeeds{true};
    bool map_succeeds{true};
    bool unmap_succeeds{true};
    bool release_succeeds{true};
    bool close_succeeds{true};

    Mapping CreatePageFileMapping(std::uint64_t, ExactWindowsBackingRecipe recipe) {
        calls.push_back("create_mapping");
        create_recipe = recipe;
        return create_succeeds ? 1 : 0;
    }
    Reservation ReservePlaceholder(std::uint64_t, ExactWindowsBackingRecipe recipe) {
        calls.push_back("reserve_placeholder");
        reserve_recipe = recipe;
        return reserve_succeeds ? 2 : 0;
    }
    void* MapReplacingPlaceholder(Mapping, Reservation, std::uint64_t,
                                  ExactWindowsBackingRecipe recipe) {
        calls.push_back("map_replace_placeholder");
        map_recipe = recipe;
        return map_succeeds ? reinterpret_cast<void*>(0x10000) : nullptr;
    }
    bool ViewMatchesReservation(Reservation, void* pointer) {
        return pointer == reinterpret_cast<void*>(0x10000);
    }
    bool UnmapPreservingPlaceholder(void*) {
        calls.push_back("unmap_preserve_placeholder");
        return unmap_succeeds;
    }
    bool ReleasePlaceholder(Reservation) {
        calls.push_back("release_placeholder");
        return release_succeeds;
    }
    bool CloseMapping(Mapping) {
        calls.push_back("close_mapping");
        return close_succeeds;
    }
};

TEST(ExactAddressSpaceHostImportProbe, UsesProductionWindowsBackingRecipeAndCleanupOrder) {
    FakeWindowsBackingAdapter adapter;
    auto acquisition = AcquireExactWindowsAddressSpaceBacking(
        adapter, ExactAddressSpaceBaseBackingSize);
    ASSERT_EQ(acquisition.failure, ExactWindowsBackingFailure::None);
    EXPECT_EQ(adapter.calls, (std::vector<std::string_view>{
                                 "create_mapping", "reserve_placeholder",
                                 "map_replace_placeholder"}));
    EXPECT_TRUE(adapter.create_recipe.create_file_mapping2);
    EXPECT_TRUE(adapter.create_recipe.invalid_page_file);
    EXPECT_TRUE(adapter.create_recipe.file_map_all_access);
    EXPECT_TRUE(adapter.create_recipe.page_execute_readwrite);
    EXPECT_TRUE(adapter.create_recipe.sec_commit);
    EXPECT_TRUE(adapter.reserve_recipe.virtual_alloc2);
    EXPECT_TRUE(adapter.reserve_recipe.mem_reserve);
    EXPECT_TRUE(adapter.reserve_recipe.mem_reserve_placeholder);
    EXPECT_TRUE(adapter.reserve_recipe.page_noaccess);
    EXPECT_TRUE(adapter.map_recipe.map_view_of_file3);
    EXPECT_TRUE(adapter.map_recipe.mem_replace_placeholder);
    EXPECT_TRUE(adapter.map_recipe.page_execute_readwrite);

    EXPECT_EQ(ReleaseExactWindowsAddressSpaceBacking(adapter, acquisition.backing, false).failure,
              ExactWindowsBackingFailure::VulkanResourcesStillOwned);
    EXPECT_EQ(adapter.calls.size(), 3u);

    const auto cleanup =
        ReleaseExactWindowsAddressSpaceBacking(adapter, acquisition.backing, true);
    EXPECT_EQ(cleanup.failure, ExactWindowsBackingFailure::None);
    EXPECT_EQ(adapter.calls, (std::vector<std::string_view>{
                                 "create_mapping", "reserve_placeholder",
                                 "map_replace_placeholder", "unmap_preserve_placeholder",
                                 "release_placeholder", "close_mapping"}));
}

TEST(ExactAddressSpaceHostImportProbe, RollsBackEveryPartiallyAcquiredWindowsResource) {
    FakeWindowsBackingAdapter reserve_failure;
    reserve_failure.reserve_succeeds = false;
    auto reserve = AcquireExactWindowsAddressSpaceBacking(
        reserve_failure, ExactAddressSpaceBaseBackingSize);
    EXPECT_EQ(reserve.failure, ExactWindowsBackingFailure::PlaceholderReservationFailed);
    EXPECT_EQ(reserve_failure.calls, (std::vector<std::string_view>{
                                         "create_mapping", "reserve_placeholder",
                                         "close_mapping"}));

    FakeWindowsBackingAdapter map_failure;
    map_failure.map_succeeds = false;
    auto map = AcquireExactWindowsAddressSpaceBacking(map_failure, ExactAddressSpaceBaseBackingSize);
    EXPECT_EQ(map.failure, ExactWindowsBackingFailure::ViewMappingFailed);
    EXPECT_EQ(map_failure.calls, (std::vector<std::string_view>{
                                     "create_mapping", "reserve_placeholder",
                                     "map_replace_placeholder", "release_placeholder",
                                     "close_mapping"}));
}

TEST(ExactAddressSpaceHostImportProbe, RequiresImmutableChecksAndRequirementsBeforeBacking) {
    ExactHostImportProtocol protocol;
    EXPECT_TRUE(protocol.Complete(ExactHostImportStage::Capability));
    EXPECT_TRUE(protocol.Complete(ExactHostImportStage::DeviceLimits));
    EXPECT_TRUE(protocol.Complete(ExactHostImportStage::ExternalBufferProperties));
    EXPECT_TRUE(protocol.Complete(ExactHostImportStage::BufferCreation));
    EXPECT_TRUE(protocol.Complete(ExactHostImportStage::MemoryRequirements));
    EXPECT_TRUE(protocol.CanAllocateLargeBacking());
    EXPECT_TRUE(protocol.Complete(ExactHostImportStage::Backing));

    ExactHostImportProtocol skipped;
    EXPECT_TRUE(skipped.Complete(ExactHostImportStage::Capability));
    EXPECT_FALSE(skipped.Complete(ExactHostImportStage::Backing));
    EXPECT_FALSE(skipped.CanAllocateLargeBacking());
}

TEST(ExactAddressSpaceHostImportProbe, ClassifiesOnlyImmutableIncompatibilityAsUnsupported) {
    EXPECT_EQ(ClassifyExactHostImportFailure(ExactHostImportFailure::ExtensionUnavailable),
              ExactHostImportDisposition::Unsupported);
    EXPECT_EQ(ClassifyExactHostImportFailure(ExactHostImportFailure::HandleNotImportable),
              ExactHostImportDisposition::Unsupported);
    EXPECT_EQ(ClassifyExactHostImportFailure(ExactHostImportFailure::NoCoherentMemoryType),
              ExactHostImportDisposition::Unsupported);
    EXPECT_EQ(ClassifyExactHostImportFailure(ExactHostImportFailure::RequirementExceedsBacking),
              ExactHostImportDisposition::ExactDesignIncompatible);
    EXPECT_EQ(ClassifyExactHostImportFailure(ExactHostImportFailure::RequirementAlignmentMismatch),
              ExactHostImportDisposition::ExactDesignIncompatible);
    EXPECT_EQ(ClassifyExactHostImportFailure(ExactHostImportFailure::BackingCommitFailed),
              ExactHostImportDisposition::ResourceLimited);
    EXPECT_EQ(ClassifyExactHostImportFailure(ExactHostImportFailure::VulkanOutOfMemory),
              ExactHostImportDisposition::ResourceLimited);
    EXPECT_EQ(ClassifyExactHostImportFailure(ExactHostImportFailure::CleanupFailed),
              ExactHostImportDisposition::Error);
}

} // namespace
} // namespace Vulkan
