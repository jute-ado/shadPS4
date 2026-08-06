// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/external_host_memory_probe.h"

namespace Vulkan {
namespace {

TEST(ExternalHostMemoryProbe, RequiresExtensionAndValidImportedPointerAlignment) {
    EXPECT_EQ(ValidateExternalHostMemoryProbeCapability(false, 0x1000),
              ExternalHostMemoryProbeCapability::ExtensionUnavailable);
    EXPECT_EQ(ValidateExternalHostMemoryProbeCapability(true, 0),
              ExternalHostMemoryProbeCapability::InvalidPointerAlignment);
    EXPECT_EQ(ValidateExternalHostMemoryProbeCapability(true, 24),
              ExternalHostMemoryProbeCapability::InvalidPointerAlignment);
    EXPECT_EQ(ValidateExternalHostMemoryProbeCapability(true, 0x1000),
              ExternalHostMemoryProbeCapability::Available);
}

TEST(ExternalHostMemoryProbe, ValidatesImportedHostAllocationBoundsAndAlignment) {
    constexpr std::uintptr_t aligned_address = 0x10000;
    constexpr std::size_t alignment = 0x1000;
    constexpr std::size_t backing_size = 0x4000;

    EXPECT_EQ(ValidateImportedHostAllocation(reinterpret_cast<const void*>(aligned_address),
                                             backing_size, alignment, backing_size),
              ImportedHostAllocationValidation::Valid);
    EXPECT_EQ(ValidateImportedHostAllocation(nullptr, backing_size, alignment, backing_size),
              ImportedHostAllocationValidation::NullPointer);
    EXPECT_EQ(ValidateImportedHostAllocation(reinterpret_cast<const void*>(aligned_address), 0,
                                             alignment, 1),
              ImportedHostAllocationValidation::EmptyBacking);
    EXPECT_EQ(ValidateImportedHostAllocation(reinterpret_cast<const void*>(aligned_address),
                                             backing_size, alignment, 0),
              ImportedHostAllocationValidation::EmptyRequirement);
    EXPECT_EQ(ValidateImportedHostAllocation(reinterpret_cast<const void*>(aligned_address),
                                             backing_size, 0, backing_size),
              ImportedHostAllocationValidation::InvalidAlignment);
    EXPECT_EQ(ValidateImportedHostAllocation(reinterpret_cast<const void*>(aligned_address),
                                             backing_size, 24, backing_size),
              ImportedHostAllocationValidation::InvalidAlignment);
    EXPECT_EQ(ValidateImportedHostAllocation(reinterpret_cast<const void*>(aligned_address + 1),
                                             backing_size, alignment, backing_size),
              ImportedHostAllocationValidation::MisalignedPointer);
    EXPECT_EQ(ValidateImportedHostAllocation(reinterpret_cast<const void*>(aligned_address),
                                             backing_size - 1, alignment, backing_size - 1),
              ImportedHostAllocationValidation::MisalignedSize);
    EXPECT_EQ(ValidateImportedHostAllocation(reinterpret_cast<const void*>(aligned_address),
                                             backing_size, alignment, backing_size + 1),
              ImportedHostAllocationValidation::RequirementExceedsBacking);

    constexpr auto overflowing_address = std::numeric_limits<std::uintptr_t>::max() - 0x7ff;
    EXPECT_EQ(ValidateImportedHostAllocation(reinterpret_cast<const void*>(overflowing_address),
                                             alignment, alignment, alignment),
              ImportedHostAllocationValidation::AddressOverflow);
}

TEST(ExternalHostMemoryProbe, SelectsOnlyIntersectingImportedHostMemoryTypes) {
    constexpr std::array properties{
        ImportedHostMemoryTypeProperties{.host_coherent = false},
        ImportedHostMemoryTypeProperties{.host_coherent = true},
        ImportedHostMemoryTypeProperties{.host_coherent = true},
    };

    const auto selected = SelectImportedHostMemoryType(0b110, 0b011, properties, true);
    ASSERT_EQ(selected.failure, ImportedHostMemoryTypeSelectionFailure::None);
    ASSERT_TRUE(selected.index.has_value());
    EXPECT_EQ(*selected.index, 1u);
    EXPECT_EQ(selected.compatible_bits, 0b010u);

    const auto no_intersection = SelectImportedHostMemoryType(0b100, 0b011, properties, false);
    EXPECT_EQ(no_intersection.failure,
              ImportedHostMemoryTypeSelectionFailure::NoCompatibleMemoryType);
    EXPECT_FALSE(no_intersection.index.has_value());
    EXPECT_EQ(no_intersection.compatible_bits, 0u);
}

TEST(ExternalHostMemoryProbe, FailsClosedWhenCoherentImportedMemoryIsUnavailable) {
    constexpr std::array properties{
        ImportedHostMemoryTypeProperties{.host_coherent = false},
        ImportedHostMemoryTypeProperties{.host_coherent = false},
    };

    const auto coherent = SelectImportedHostMemoryType(0b11, 0b11, properties, true);
    EXPECT_EQ(coherent.failure, ImportedHostMemoryTypeSelectionFailure::NoCoherentMemoryType);
    EXPECT_FALSE(coherent.index.has_value());
    EXPECT_EQ(coherent.compatible_bits, 0b11u);

    const auto noncoherent = SelectImportedHostMemoryType(0b11, 0b11, properties, false);
    ASSERT_EQ(noncoherent.failure, ImportedHostMemoryTypeSelectionFailure::None);
    ASSERT_TRUE(noncoherent.index.has_value());
    EXPECT_EQ(*noncoherent.index, 0u);
}

TEST(ExternalHostMemoryProbe, IgnoresCompatibilityBitsBeyondProvidedMemoryProperties) {
    constexpr std::array properties{
        ImportedHostMemoryTypeProperties{.host_coherent = true},
    };

    const auto selected = SelectImportedHostMemoryType(0x80000001u, 0x80000001u, properties, true);
    ASSERT_EQ(selected.failure, ImportedHostMemoryTypeSelectionFailure::None);
    ASSERT_TRUE(selected.index.has_value());
    EXPECT_EQ(*selected.index, 0u);
    EXPECT_EQ(selected.compatible_bits, 0x1u);

    const auto absent = SelectImportedHostMemoryType(0x80000000u, 0x80000000u, properties, true);
    EXPECT_EQ(absent.failure, ImportedHostMemoryTypeSelectionFailure::NoCompatibleMemoryType);
    EXPECT_FALSE(absent.index.has_value());
    EXPECT_EQ(absent.compatible_bits, 0u);
}

TEST(ExternalHostMemoryProbe, RecordsOrderedStagesAndFailsClosedOnProtocolViolations) {
    ExternalHostMemoryProbeProgress progress;

    EXPECT_TRUE(progress.Complete(ExternalHostMemoryProbeStage::Capability));
    EXPECT_TRUE(progress.Complete(ExternalHostMemoryProbeStage::Backing));
    EXPECT_FALSE(progress.Complete(ExternalHostMemoryProbeStage::BufferCreation));

    const auto& result = progress.Result();
    EXPECT_EQ(result.completed_stage, ExternalHostMemoryProbeStage::Backing);
    EXPECT_EQ(result.failure_stage, ExternalHostMemoryProbeStage::BufferCreation);
    EXPECT_EQ(result.failure, ExternalHostMemoryProbeFailure::UnexpectedStage);
    EXPECT_FALSE(result.Succeeded());

    progress.Fail(ExternalHostMemoryProbeStage::HostPointerProperties,
                  ExternalHostMemoryProbeFailure::HostPointerQueryFailed);
    EXPECT_EQ(progress.Result().failure, ExternalHostMemoryProbeFailure::UnexpectedStage);
}

TEST(ExternalHostMemoryProbe, SucceedsOnlyAfterNonzeroDeviceAddressIsRetained) {
    ExternalHostMemoryProbeProgress progress;
    for (const auto stage : {ExternalHostMemoryProbeStage::Capability,
                             ExternalHostMemoryProbeStage::Backing,
                             ExternalHostMemoryProbeStage::ExternalBufferProperties,
                             ExternalHostMemoryProbeStage::BufferCreation,
                             ExternalHostMemoryProbeStage::MemoryRequirements,
                             ExternalHostMemoryProbeStage::HostPointerProperties,
                             ExternalHostMemoryProbeStage::MemoryTypeSelection,
                             ExternalHostMemoryProbeStage::MemoryAllocation,
                             ExternalHostMemoryProbeStage::MemoryBinding,
                             ExternalHostMemoryProbeStage::DeviceAddress}) {
        ASSERT_TRUE(progress.Complete(stage));
    }
    EXPECT_FALSE(progress.Result().Succeeded());

    progress.Fail(ExternalHostMemoryProbeStage::Retained,
                  ExternalHostMemoryProbeFailure::ZeroDeviceAddress);
    EXPECT_FALSE(progress.Result().Succeeded());

    ExternalHostMemoryProbeProgress success;
    for (const auto stage : {ExternalHostMemoryProbeStage::Capability,
                             ExternalHostMemoryProbeStage::Backing,
                             ExternalHostMemoryProbeStage::ExternalBufferProperties,
                             ExternalHostMemoryProbeStage::BufferCreation,
                             ExternalHostMemoryProbeStage::MemoryRequirements,
                             ExternalHostMemoryProbeStage::HostPointerProperties,
                             ExternalHostMemoryProbeStage::MemoryTypeSelection,
                             ExternalHostMemoryProbeStage::MemoryAllocation,
                             ExternalHostMemoryProbeStage::MemoryBinding,
                             ExternalHostMemoryProbeStage::DeviceAddress,
                             ExternalHostMemoryProbeStage::Retained}) {
        ASSERT_TRUE(success.Complete(stage));
    }
    EXPECT_TRUE(success.Result().Succeeded());
}

struct TrackedProbeHandle {
    std::vector<std::string>* destruction_order{};
    std::string name;

    TrackedProbeHandle() = default;

    TrackedProbeHandle(std::vector<std::string>& order, std::string name_)
        : destruction_order{&order}, name{std::move(name_)} {}

    TrackedProbeHandle(TrackedProbeHandle&& other) noexcept
        : destruction_order{std::exchange(other.destruction_order, nullptr)},
          name{std::move(other.name)} {}

    TrackedProbeHandle& operator=(TrackedProbeHandle&&) = delete;
    TrackedProbeHandle(const TrackedProbeHandle&) = delete;
    TrackedProbeHandle& operator=(const TrackedProbeHandle&) = delete;

    ~TrackedProbeHandle() {
        if (destruction_order != nullptr) {
            destruction_order->push_back(name);
        }
    }

    explicit operator bool() const noexcept {
        return destruction_order != nullptr;
    }
};

TEST(ExternalHostMemoryProbe, RawImportOwnerDestroysBufferBeforeImportedMemory) {
    std::vector<std::string> destruction_order;
    {
        ExternalHostMemoryImportOwner<TrackedProbeHandle, TrackedProbeHandle> owner{
            TrackedProbeHandle{destruction_order, "memory"},
            TrackedProbeHandle{destruction_order, "buffer"}, 0x12340000};
        EXPECT_TRUE(owner.IsRetained());
        EXPECT_EQ(owner.DeviceAddress(), 0x12340000u);
    }

    ASSERT_EQ(destruction_order.size(), 2u);
    EXPECT_EQ(destruction_order[0], "buffer");
    EXPECT_EQ(destruction_order[1], "memory");
}

TEST(ExternalHostMemoryProbe, RawImportOwnerRejectsPartialOrZeroAddressOwnership) {
    std::vector<std::string> destruction_order;
    ExternalHostMemoryImportOwner<TrackedProbeHandle, TrackedProbeHandle> zero_address{
        TrackedProbeHandle{destruction_order, "memory-a"},
        TrackedProbeHandle{destruction_order, "buffer-a"}, 0};
    EXPECT_FALSE(zero_address.IsRetained());

    ExternalHostMemoryImportOwner<TrackedProbeHandle, TrackedProbeHandle> missing_buffer{
        TrackedProbeHandle{destruction_order, "memory-b"}, TrackedProbeHandle{}, 0x1000};
    EXPECT_FALSE(missing_buffer.IsRetained());
}

TEST(ExternalHostMemoryProbe, LimitsHandleTypesToBackingProvenance) {
    EXPECT_EQ(AllowedExternalHostHandleTypes(ExternalHostBackingProvenance::PageFileMapping),
              ExternalHostHandleClass::HostAllocation);
    EXPECT_EQ(AllowedExternalHostHandleTypes(ExternalHostBackingProvenance::ForeignMapped),
              ExternalHostHandleClass::HostMappedForeignMemory);
    EXPECT_EQ(AllowedExternalHostHandleTypes(ExternalHostBackingProvenance::Unknown),
              ExternalHostHandleClass::None);
}

TEST(ExternalHostMemoryProbe, ClassifiesOnlyCapabilityOrInvalidHandleAsUnsupported) {
    using enum ExternalHostMemoryProbeFailure;
    using enum ExternalHostProbeVkResultClass;

    EXPECT_EQ(ClassifyExternalHostProbeAttempt(false, HandleTypeNotImportable, NotCalled),
              ExternalHostProbeDisposition::Unsupported);
    EXPECT_EQ(ClassifyExternalHostProbeAttempt(false, HostPointerQueryFailed,
                                               ErrorInvalidExternalHandle),
              ExternalHostProbeDisposition::Unsupported);
    EXPECT_EQ(ClassifyExternalHostProbeAttempt(false, NoCompatibleMemoryType, Success),
              ExternalHostProbeDisposition::Unsupported);

    for (const auto [failure, vk_result] : {
             std::pair{InvalidPointerAlignment, NotCalled},
             std::pair{BackingAllocationFailed, ErrorOutOfMemory},
             std::pair{BackingValidationFailed, NotCalled},
             std::pair{MemoryRequirementsInvalid, Success},
             std::pair{HostPointerQueryFailed, ErrorUnknown},
             std::pair{MemoryAllocationFailed, ErrorOutOfMemory},
             std::pair{MemoryAllocationFailed, ErrorUnknown},
             std::pair{ZeroDeviceAddress, Success},
             std::pair{UnexpectedStage, NotCalled},
         }) {
        EXPECT_EQ(ClassifyExternalHostProbeAttempt(false, failure, vk_result),
                  ExternalHostProbeDisposition::Error);
    }
    EXPECT_EQ(ClassifyExternalHostProbeAttempt(true, None, Success),
              ExternalHostProbeDisposition::Pass);
}

TEST(ExternalHostMemoryProbe, AggregatesTopLevelDispositionWithoutHidingErrors) {
    constexpr std::array unsupported{
        ExternalHostProbeDisposition::Unsupported,
        ExternalHostProbeDisposition::Unsupported,
    };
    EXPECT_EQ(ClassifyExternalHostProbeResult(unsupported),
              ExternalHostProbeDisposition::Unsupported);

    constexpr std::array mixed{
        ExternalHostProbeDisposition::Unsupported,
        ExternalHostProbeDisposition::Error,
    };
    EXPECT_EQ(ClassifyExternalHostProbeResult(mixed), ExternalHostProbeDisposition::Error);

    constexpr std::array pass{
        ExternalHostProbeDisposition::Error,
        ExternalHostProbeDisposition::Pass,
    };
    EXPECT_EQ(ClassifyExternalHostProbeResult(pass), ExternalHostProbeDisposition::Pass);
}

TEST(ExternalHostMemoryProbe, ClaimsCleanupOnlyFromExplicitResourceResults) {
    EXPECT_TRUE(ExternalHostProbeCleanupResult{}.Complete());
    EXPECT_FALSE(ExternalHostProbeCleanupResult{.unmap_attempted = true}.Complete());
    EXPECT_FALSE((ExternalHostProbeCleanupResult{
        .unmap_attempted = true,
        .unmap_succeeded = true,
        .close_attempted = true,
    }.Complete()));
    EXPECT_TRUE((ExternalHostProbeCleanupResult{
        .unmap_attempted = true,
        .unmap_succeeded = true,
        .close_attempted = true,
        .close_succeeded = true,
    }.Complete()));
}

} // namespace
} // namespace Vulkan
