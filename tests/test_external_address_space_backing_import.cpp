// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/external_address_space_backing_import.h"

namespace Vulkan {
namespace {

constexpr std::uint32_t HostCoherent = 1u << 2;

ExternalAddressSpaceImportRequest ValidRequest() {
    return {
        .extension_available = true,
        .buffer_device_address_available = true,
        .host_allocation_importable = true,
        .dedicated_allocation_required = false,
        .lease_pointer = 0x10000,
        .lease_size = 0x8000,
        .import_pointer = 0x10000,
        .import_size = 0x8000,
        .minimum_imported_pointer_alignment = 0x1000,
        .memory_requirement_size = 0x8000,
        .memory_requirement_alignment = 0x1000,
        .maximum_buffer_size = 0x10000,
        .maximum_memory_allocation_size = 0x10000,
        .buffer_memory_type_bits = 0b1110,
        .host_pointer_memory_type_bits = 0b1100,
        .memory_type_count = 4,
        .host_coherent_property = HostCoherent,
    };
}

TEST(ExternalAddressSpaceBackingImport, ChecksApplicableDeviceLimitsButNotStorageRange) {
    const std::array properties{0u, 0u, HostCoherent, HostCoherent};
    auto request = ValidRequest();
    request.maximum_buffer_size = request.lease_size - 1;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::DeviceLimitExceeded);

    request = ValidRequest();
    request.maximum_memory_allocation_size = request.lease_size - 1;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::DeviceLimitExceeded);

    // maxStorageBufferRange is intentionally absent: BDA access is not descriptor-range limited.
    request = ValidRequest();
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::None);
}

TEST(ExternalAddressSpaceBackingImport, TreatsTheExtensionAsAnOptionalCapability) {
    const std::array properties{0u, 0u, HostCoherent, HostCoherent};
    auto request = ValidRequest();
    request.extension_available = false;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::ExtensionUnavailable);

    request = ValidRequest();
    request.buffer_device_address_available = false;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::BufferDeviceAddressUnavailable);

    request = ValidRequest();
    request.host_allocation_importable = false;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::HostAllocationNotImportable);
}

TEST(ExternalAddressSpaceBackingImport, RequiresTheExactLeasedPointerAndSize) {
    const std::array properties{0u, 0u, HostCoherent, HostCoherent};
    auto request = ValidRequest();
    request.import_pointer += 0x1000;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::BackingMismatch);

    request = ValidRequest();
    request.import_size -= 0x1000;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::BackingMismatch);

    request = ValidRequest();
    request.lease_pointer++;
    request.import_pointer = request.lease_pointer;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::PointerMisaligned);

    request = ValidRequest();
    request.minimum_imported_pointer_alignment = 0;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::InvalidRequirement);

    request = ValidRequest();
    request.memory_requirement_size = request.lease_size + 1;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::RequirementExceedsBacking);
}

TEST(ExternalAddressSpaceBackingImport, IntersectsBufferAndHostBitsAndSelectsCoherentMemory) {
    const std::array properties{0u, HostCoherent, 0u, HostCoherent};
    auto request = ValidRequest();
    request.buffer_memory_type_bits = 0b1110;
    request.host_pointer_memory_type_bits = 0b1100;
    const auto plan = PlanExternalAddressSpaceBackingImport(request, properties);
    ASSERT_EQ(plan.failure, ExternalAddressSpaceImportFailure::None);
    EXPECT_EQ(plan.compatible_memory_type_bits, 0b1100u);
    EXPECT_EQ(plan.memory_type_index, 3u);
    EXPECT_TRUE(plan.use_dedicated_allocation);

    request = ValidRequest();
    request.dedicated_allocation_required = true;
    const auto dedicated = PlanExternalAddressSpaceBackingImport(request, properties);
    ASSERT_EQ(dedicated.failure, ExternalAddressSpaceImportFailure::None);
    EXPECT_TRUE(dedicated.use_dedicated_allocation);

    request.host_pointer_memory_type_bits = 0b0001;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::NoCompatibleMemoryType);

    request = ValidRequest();
    request.buffer_memory_type_bits = 0b0100;
    request.host_pointer_memory_type_bits = 0b0100;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::NoCoherentMemoryType);
}

TEST(ExternalAddressSpaceBackingImport, RejectsOverflowAndIncompleteMemoryPropertyEvidence) {
    const std::array properties{HostCoherent};
    auto request = ValidRequest();
    request.buffer_memory_type_bits = 1u << 5;
    request.host_pointer_memory_type_bits = 1u << 5;
    request.memory_type_count = 6;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::MemoryPropertyEvidenceMissing);

    request = ValidRequest();
    request.buffer_memory_type_bits = 1u << 5;
    request.host_pointer_memory_type_bits = 1u << 5;
    request.memory_type_count = 4;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::MemoryTypeOutOfRange);

    request = ValidRequest();
    request.lease_pointer = std::numeric_limits<std::uintptr_t>::max() - 0xfff;
    request.import_pointer = request.lease_pointer;
    EXPECT_EQ(PlanExternalAddressSpaceBackingImport(request, properties).failure,
              ExternalAddressSpaceImportFailure::BackingRangeOverflow);
}

TEST(ExternalAddressSpaceBackingImport, RetainsResourcesInDependencyOrderWithoutPublishingPages) {
    ExternalAddressSpaceImportOwnership ownership;
    EXPECT_TRUE(ownership.RetainLease());
    EXPECT_TRUE(ownership.OwnMemory());
    EXPECT_TRUE(ownership.OwnBuffer());
    EXPECT_TRUE(ownership.RetainDeviceAddress(0x12340000));
    EXPECT_TRUE(ownership.IsReady());
    EXPECT_EQ(ownership.GuestPagePublicationCount(), 0u);

    EXPECT_FALSE(ownership.ReleaseMemory());
    EXPECT_FALSE(ownership.ReleaseLease());
    EXPECT_TRUE(ownership.ReleaseBuffer());
    EXPECT_TRUE(ownership.ReleaseMemory());
    EXPECT_TRUE(ownership.ReleaseLease());
    EXPECT_TRUE(ownership.IsEmpty());
    EXPECT_EQ(ownership.GuestPagePublicationCount(), 0u);
}

struct LoggedResource {
    std::vector<std::string_view>* log{};
    std::string_view name;

    LoggedResource(std::vector<std::string_view>& log_, std::string_view name_)
        : log{&log_}, name{name_} {}
    LoggedResource(const LoggedResource&) = delete;
    LoggedResource& operator=(const LoggedResource&) = delete;
    LoggedResource(LoggedResource&& other) noexcept : log{other.log}, name{other.name} {
        other.log = nullptr;
    }
    ~LoggedResource() {
        if (log != nullptr) {
            log->push_back(name);
        }
    }
};

TEST(ExternalAddressSpaceBackingImport, ResourceOwnerDestroysBufferThenMemoryThenLease) {
    std::vector<std::string_view> log;
    {
        ExternalAddressSpaceImportResources<LoggedResource, LoggedResource, LoggedResource> owner{
            LoggedResource{log, "lease"}, LoggedResource{log, "memory"},
            LoggedResource{log, "buffer"}};
        EXPECT_TRUE(log.empty());
    }
    EXPECT_EQ(log, (std::vector<std::string_view>{"buffer", "memory", "lease"}));
}

TEST(ExternalAddressSpaceBackingImport, StagingDestroysBoundBufferBeforeImportedMemoryOnFailure) {
    std::vector<std::string_view> log;
    {
        ExternalAddressSpaceImportStaging<LoggedResource, LoggedResource> staging;
        staging.buffer.emplace(log, "buffer");
        staging.memory.emplace(log, "memory");
    }
    EXPECT_EQ(log, (std::vector<std::string_view>{"buffer", "memory"}));

    log.clear();
    {
        ExternalAddressSpaceImportStaging<LoggedResource, LoggedResource> staging;
        staging.buffer.emplace(log, "buffer");
    }
    EXPECT_EQ(log, (std::vector<std::string_view>{"buffer"}));
}

TEST(ExternalAddressSpaceBackingImport, FailsClosedOnZeroDeviceAddressAndNeverPublishes) {
    ExternalAddressSpaceImportOwnership ownership;
    EXPECT_TRUE(ownership.RetainLease());
    EXPECT_TRUE(ownership.OwnMemory());
    EXPECT_TRUE(ownership.OwnBuffer());
    EXPECT_FALSE(ownership.RetainDeviceAddress(0));
    EXPECT_FALSE(ownership.IsReady());
    EXPECT_EQ(ownership.GuestPagePublicationCount(), 0u);
}

} // namespace
} // namespace Vulkan
