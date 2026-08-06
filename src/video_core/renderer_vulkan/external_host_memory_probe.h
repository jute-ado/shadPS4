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
#include <utility>

namespace Vulkan {

enum class ExternalHostBackingProvenance : std::uint8_t {
    Unknown,
    PageFileMapping,
    ForeignMapped,
};

enum class ExternalHostHandleClass : std::uint8_t {
    None,
    HostAllocation,
    HostMappedForeignMemory,
};

[[nodiscard]] constexpr ExternalHostHandleClass AllowedExternalHostHandleTypes(
    ExternalHostBackingProvenance provenance) noexcept {
    switch (provenance) {
    case ExternalHostBackingProvenance::PageFileMapping:
        return ExternalHostHandleClass::HostAllocation;
    case ExternalHostBackingProvenance::ForeignMapped:
        return ExternalHostHandleClass::HostMappedForeignMemory;
    case ExternalHostBackingProvenance::Unknown:
        return ExternalHostHandleClass::None;
    }
    return ExternalHostHandleClass::None;
}

enum class ExternalHostMemoryProbeStage : std::uint8_t {
    NotStarted,
    Capability,
    Backing,
    ExternalBufferProperties,
    BufferCreation,
    MemoryRequirements,
    HostPointerProperties,
    MemoryTypeSelection,
    MemoryAllocation,
    MemoryBinding,
    DeviceAddress,
    Retained,
};

enum class ExternalHostMemoryProbeFailure : std::uint8_t {
    None,
    UnexpectedStage,
    ExtensionUnavailable,
    InvalidPointerAlignment,
    BackingAllocationFailed,
    BackingValidationFailed,
    ExternalBufferQueryFailed,
    HandleTypeNotImportable,
    BufferCreationFailed,
    MemoryRequirementsInvalid,
    HostPointerQueryFailed,
    NoCompatibleMemoryType,
    NoCoherentMemoryType,
    MemoryAllocationFailed,
    MemoryBindingFailed,
    ZeroDeviceAddress,
};

enum class ExternalHostProbeVkResultClass : std::uint8_t {
    NotCalled,
    Success,
    ErrorInvalidExternalHandle,
    ErrorOutOfMemory,
    ErrorUnknown,
};

enum class ExternalHostProbeDisposition : std::uint8_t {
    Pass,
    Unsupported,
    Error,
};

[[nodiscard]] constexpr ExternalHostProbeDisposition ClassifyExternalHostProbeAttempt(
    bool succeeded, ExternalHostMemoryProbeFailure failure,
    ExternalHostProbeVkResultClass vk_result) noexcept {
    if (succeeded) {
        return ExternalHostProbeDisposition::Pass;
    }
    if (failure == ExternalHostMemoryProbeFailure::ExtensionUnavailable ||
        failure == ExternalHostMemoryProbeFailure::HandleTypeNotImportable ||
        failure == ExternalHostMemoryProbeFailure::NoCompatibleMemoryType ||
        failure == ExternalHostMemoryProbeFailure::NoCoherentMemoryType ||
        vk_result == ExternalHostProbeVkResultClass::ErrorInvalidExternalHandle) {
        return ExternalHostProbeDisposition::Unsupported;
    }
    return ExternalHostProbeDisposition::Error;
}

[[nodiscard]] constexpr ExternalHostProbeDisposition ClassifyExternalHostProbeResult(
    std::span<const ExternalHostProbeDisposition> attempts) noexcept {
    if (attempts.empty()) {
        return ExternalHostProbeDisposition::Error;
    }
    bool all_unsupported = true;
    for (const auto attempt : attempts) {
        if (attempt == ExternalHostProbeDisposition::Pass) {
            return ExternalHostProbeDisposition::Pass;
        }
        all_unsupported &= attempt == ExternalHostProbeDisposition::Unsupported;
    }
    return all_unsupported ? ExternalHostProbeDisposition::Unsupported
                           : ExternalHostProbeDisposition::Error;
}

struct ExternalHostProbeCleanupResult {
    bool unmap_attempted{};
    bool unmap_succeeded{};
    bool close_attempted{};
    bool close_succeeded{};

    [[nodiscard]] constexpr bool Complete() const noexcept {
        return (!unmap_attempted || unmap_succeeded) && (!close_attempted || close_succeeded);
    }
};

struct ExternalHostMemoryProbeResult {
    ExternalHostMemoryProbeStage completed_stage{ExternalHostMemoryProbeStage::NotStarted};
    ExternalHostMemoryProbeStage failure_stage{ExternalHostMemoryProbeStage::NotStarted};
    ExternalHostMemoryProbeFailure failure{ExternalHostMemoryProbeFailure::None};

    [[nodiscard]] constexpr bool Succeeded() const noexcept {
        return failure == ExternalHostMemoryProbeFailure::None &&
               completed_stage == ExternalHostMemoryProbeStage::Retained;
    }
};

class ExternalHostMemoryProbeProgress {
public:
    [[nodiscard]] constexpr bool Complete(ExternalHostMemoryProbeStage stage) noexcept {
        if (result.failure != ExternalHostMemoryProbeFailure::None) {
            return false;
        }
        if (stage != NextStage()) {
            SetFailure(stage, ExternalHostMemoryProbeFailure::UnexpectedStage);
            return false;
        }
        result.completed_stage = stage;
        return true;
    }

    constexpr void Fail(ExternalHostMemoryProbeStage stage,
                        ExternalHostMemoryProbeFailure failure) noexcept {
        if (result.failure != ExternalHostMemoryProbeFailure::None) {
            return;
        }
        if (stage != NextStage() || failure == ExternalHostMemoryProbeFailure::None) {
            SetFailure(stage, ExternalHostMemoryProbeFailure::UnexpectedStage);
            return;
        }
        SetFailure(stage, failure);
    }

    [[nodiscard]] constexpr const ExternalHostMemoryProbeResult& Result() const noexcept {
        return result;
    }

private:
    [[nodiscard]] constexpr ExternalHostMemoryProbeStage NextStage() const noexcept {
        return static_cast<ExternalHostMemoryProbeStage>(
            static_cast<std::uint8_t>(result.completed_stage) + 1);
    }

    constexpr void SetFailure(ExternalHostMemoryProbeStage stage,
                              ExternalHostMemoryProbeFailure failure) noexcept {
        result.failure_stage = stage;
        result.failure = failure;
    }

private:
    ExternalHostMemoryProbeResult result{};
};

template <typename MemoryHandle, typename BufferHandle>
class ExternalHostMemoryImportOwner {
public:
    ExternalHostMemoryImportOwner(MemoryHandle memory, BufferHandle buffer,
                                  std::uint64_t device_address) noexcept
        : memory{std::move(memory)}, buffer{std::move(buffer)}, device_address{device_address} {}

    ExternalHostMemoryImportOwner(ExternalHostMemoryImportOwner&&) noexcept = default;
    ExternalHostMemoryImportOwner& operator=(ExternalHostMemoryImportOwner&&) = delete;
    ExternalHostMemoryImportOwner(const ExternalHostMemoryImportOwner&) = delete;
    ExternalHostMemoryImportOwner& operator=(const ExternalHostMemoryImportOwner&) = delete;

    [[nodiscard]] bool IsRetained() const noexcept {
        return static_cast<bool>(memory) && static_cast<bool>(buffer) && device_address != 0;
    }

    [[nodiscard]] std::uint64_t DeviceAddress() const noexcept {
        return device_address;
    }

private:
    // Declaration order is intentional: reverse member destruction releases the buffer before
    // the imported device memory that backs it.
    MemoryHandle memory;
    BufferHandle buffer;
    std::uint64_t device_address{};
};

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
