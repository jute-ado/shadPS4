// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace Vulkan {

enum class ExternalHostMemoryProbeCapability {
    Available,
    ExtensionUnavailable,
    InvalidPointerAlignment,
};

[[nodiscard]] constexpr ExternalHostMemoryProbeCapability ValidateExternalHostMemoryProbeCapability(
    bool extension_enabled, std::size_t min_pointer_alignment) noexcept {
    if (!extension_enabled) {
        return ExternalHostMemoryProbeCapability::ExtensionUnavailable;
    }
    if (!std::has_single_bit(min_pointer_alignment)) {
        return ExternalHostMemoryProbeCapability::InvalidPointerAlignment;
    }
    return ExternalHostMemoryProbeCapability::Available;
}

enum class ImportedHostAllocationValidation {
    Valid,
    NullPointer,
    EmptyBacking,
    EmptyRequirement,
    InvalidAlignment,
    MisalignedPointer,
    MisalignedSize,
    AddressOverflow,
    RequirementExceedsBacking,
};

[[nodiscard]] constexpr ImportedHostAllocationValidation ValidateImportedHostAllocation(
    const void* pointer, std::size_t backing_size, std::size_t min_alignment,
    std::size_t requirement_size) noexcept {
    if (pointer == nullptr) {
        return ImportedHostAllocationValidation::NullPointer;
    }
    if (backing_size == 0) {
        return ImportedHostAllocationValidation::EmptyBacking;
    }
    if (requirement_size == 0) {
        return ImportedHostAllocationValidation::EmptyRequirement;
    }
    if (!std::has_single_bit(min_alignment)) {
        return ImportedHostAllocationValidation::InvalidAlignment;
    }

    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (backing_size > std::numeric_limits<std::uintptr_t>::max() - address) {
        return ImportedHostAllocationValidation::AddressOverflow;
    }
    if ((address & (min_alignment - 1)) != 0) {
        return ImportedHostAllocationValidation::MisalignedPointer;
    }
    if ((backing_size & (min_alignment - 1)) != 0) {
        return ImportedHostAllocationValidation::MisalignedSize;
    }
    if (requirement_size > backing_size) {
        return ImportedHostAllocationValidation::RequirementExceedsBacking;
    }
    return ImportedHostAllocationValidation::Valid;
}

struct ImportedHostMemoryTypeProperties {
    bool host_coherent{};
};

enum class ImportedHostMemoryTypeSelectionFailure {
    None,
    NoCompatibleMemoryType,
    NoCoherentMemoryType,
};

struct ImportedHostMemoryTypeSelection {
    std::optional<std::uint32_t> index;
    ImportedHostMemoryTypeSelectionFailure failure{
        ImportedHostMemoryTypeSelectionFailure::NoCompatibleMemoryType};
    std::uint32_t compatible_bits{};
};

[[nodiscard]] constexpr ImportedHostMemoryTypeSelection SelectImportedHostMemoryType(
    std::uint32_t requirement_bits, std::uint32_t host_bits,
    std::span<const ImportedHostMemoryTypeProperties> memory_properties,
    bool require_coherent) noexcept {
    const auto memory_type_count = std::min<std::size_t>(memory_properties.size(), 32);
    const std::uint32_t provided_bits =
        memory_type_count == 32
            ? std::numeric_limits<std::uint32_t>::max()
            : (std::uint32_t{1} << static_cast<std::uint32_t>(memory_type_count)) - 1;
    const std::uint32_t compatible_bits = requirement_bits & host_bits & provided_bits;
    if (compatible_bits == 0) {
        return {.failure = ImportedHostMemoryTypeSelectionFailure::NoCompatibleMemoryType,
                .compatible_bits = 0};
    }

    for (std::uint32_t index = 0; index < memory_type_count; ++index) {
        if ((compatible_bits & (std::uint32_t{1} << index)) == 0) {
            continue;
        }
        if (require_coherent && !memory_properties[index].host_coherent) {
            continue;
        }
        return {.index = index,
                .failure = ImportedHostMemoryTypeSelectionFailure::None,
                .compatible_bits = compatible_bits};
    }

    return {.failure = ImportedHostMemoryTypeSelectionFailure::NoCoherentMemoryType,
            .compatible_bits = compatible_bits};
}

} // namespace Vulkan
