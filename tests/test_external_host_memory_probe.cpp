// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstdint>
#include <limits>

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

} // namespace
} // namespace Vulkan
