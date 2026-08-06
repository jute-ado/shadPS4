// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include "core/address_space_backing_size.h"

namespace Vulkan {

inline constexpr std::uint64_t ExactAddressSpaceBaseBackingSize = Core::AddressSpaceBaseBackingSize;
inline constexpr std::uint64_t ExactAddressSpaceMaxExtraDmemMbytes = 20000;

[[nodiscard]] constexpr std::optional<std::uint64_t> CalculateExactAddressSpaceBackingSize(
    std::uint64_t extra_dmem_mbytes) noexcept {
    if (extra_dmem_mbytes > ExactAddressSpaceMaxExtraDmemMbytes ||
        extra_dmem_mbytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return std::nullopt;
    }
    return Core::CalculateAddressSpaceBackingSize(static_cast<std::int64_t>(extra_dmem_mbytes));
}

enum class ExactBackingSizeValidation : std::uint8_t {
    Valid,
    Empty,
    ExceedsHostSizeT,
};

[[nodiscard]] constexpr ExactBackingSizeValidation ValidateExactBackingSizeForSizeT(
    std::uint64_t size, std::uint64_t size_t_max) noexcept {
    if (size == 0) {
        return ExactBackingSizeValidation::Empty;
    }
    if (size > size_t_max) {
        return ExactBackingSizeValidation::ExceedsHostSizeT;
    }
    return ExactBackingSizeValidation::Valid;
}

[[nodiscard]] constexpr bool IsExactImportedPointerAligned(std::uintptr_t pointer,
                                                           std::uint64_t alignment) noexcept {
    return pointer != 0 && std::has_single_bit(alignment) && pointer % alignment == 0;
}

[[nodiscard]] constexpr bool IsExactHostImportAlignmentValid(std::uint64_t allocation_size,
                                                             std::uintptr_t pointer,
                                                             std::uint64_t alignment) noexcept {
    return allocation_size != 0 && IsExactImportedPointerAligned(pointer, alignment) &&
           allocation_size % alignment == 0;
}

// Semantic mirror of the production Windows AddressSpace allocation recipe. Keeping the recipe
// explicit makes accidental changes to a Win32 flag observable in focused tests.
struct ExactWindowsBackingRecipe {
    bool create_file_mapping2{};
    bool invalid_page_file{};
    bool file_map_all_access{};
    bool page_execute_readwrite{};
    bool sec_commit{};
    bool virtual_alloc2{};
    bool mem_reserve{};
    bool mem_reserve_placeholder{};
    bool page_noaccess{};
    bool map_view_of_file3{};
    bool mem_replace_placeholder{};
};

inline constexpr ExactWindowsBackingRecipe ExactCreateMappingRecipe{
    .create_file_mapping2 = true,
    .invalid_page_file = true,
    .file_map_all_access = true,
    .page_execute_readwrite = true,
    .sec_commit = true,
};
inline constexpr ExactWindowsBackingRecipe ExactReservePlaceholderRecipe{
    .virtual_alloc2 = true,
    .mem_reserve = true,
    .mem_reserve_placeholder = true,
    .page_noaccess = true,
};
inline constexpr ExactWindowsBackingRecipe ExactMapViewRecipe{
    .file_map_all_access = true,
    .page_execute_readwrite = true,
    .map_view_of_file3 = true,
    .mem_replace_placeholder = true,
};

enum class ExactWindowsBackingFailure : std::uint8_t {
    None,
    MappingCreationFailed,
    PlaceholderReservationFailed,
    ViewMappingFailed,
    VulkanResourcesStillOwned,
    CleanupFailed,
};

template <typename Adapter>
struct ExactWindowsAddressSpaceBacking {
    typename Adapter::Mapping mapping{};
    typename Adapter::Reservation reservation{};
    void* pointer{};
    std::uint64_t size{};
};

struct ExactWindowsBackingRollback {
    bool unmap_attempted{};
    bool unmap_succeeded{};
    bool release_attempted{};
    bool release_succeeded{};
    bool close_attempted{};
    bool close_succeeded{};
};

template <typename Adapter>
struct ExactWindowsBackingAcquisition {
    ExactWindowsAddressSpaceBacking<Adapter> backing{};
    ExactWindowsBackingFailure failure{ExactWindowsBackingFailure::None};
    ExactWindowsBackingRollback rollback{};
    bool rollback_complete{true};
};

template <typename Adapter>
[[nodiscard]] ExactWindowsBackingAcquisition<Adapter> AcquireExactWindowsAddressSpaceBacking(
    Adapter& adapter, std::uint64_t size) {
    ExactWindowsBackingAcquisition<Adapter> result;
    auto& backing = result.backing;
    backing.size = size;
    backing.mapping = adapter.CreatePageFileMapping(size, ExactCreateMappingRecipe);
    if (!backing.mapping) {
        result.failure = ExactWindowsBackingFailure::MappingCreationFailed;
        return result;
    }
    backing.reservation = adapter.ReservePlaceholder(size, ExactReservePlaceholderRecipe);
    if (!backing.reservation) {
        result.failure = ExactWindowsBackingFailure::PlaceholderReservationFailed;
        result.rollback.close_attempted = true;
        result.rollback.close_succeeded = adapter.CloseMapping(backing.mapping);
        result.rollback_complete = result.rollback.close_succeeded;
        backing.mapping = {};
        return result;
    }
    backing.pointer = adapter.MapReplacingPlaceholder(backing.mapping, backing.reservation, size,
                                                      ExactMapViewRecipe);
    if (backing.pointer == nullptr ||
        !adapter.ViewMatchesReservation(backing.reservation, backing.pointer)) {
        result.failure = ExactWindowsBackingFailure::ViewMappingFailed;
        if (backing.pointer != nullptr) {
            result.rollback.unmap_attempted = true;
            result.rollback.unmap_succeeded = adapter.UnmapPreservingPlaceholder(backing.pointer);
        }
        result.rollback.release_attempted = true;
        result.rollback.release_succeeded = adapter.ReleasePlaceholder(backing.reservation);
        result.rollback.close_attempted = true;
        result.rollback.close_succeeded = adapter.CloseMapping(backing.mapping);
        result.rollback_complete =
            (!result.rollback.unmap_attempted || result.rollback.unmap_succeeded) &&
            result.rollback.release_succeeded && result.rollback.close_succeeded;
        backing.reservation = {};
        backing.mapping = {};
        return result;
    }
    return result;
}

struct ExactWindowsBackingCleanup {
    ExactWindowsBackingFailure failure{ExactWindowsBackingFailure::None};
    bool unmap_succeeded{};
    bool release_succeeded{};
    bool close_succeeded{};
};

template <typename Adapter>
[[nodiscard]] ExactWindowsBackingCleanup ReleaseExactWindowsAddressSpaceBacking(
    Adapter& adapter, ExactWindowsAddressSpaceBacking<Adapter>& backing,
    bool vulkan_resources_released) {
    ExactWindowsBackingCleanup result;
    if (!vulkan_resources_released) {
        result.failure = ExactWindowsBackingFailure::VulkanResourcesStillOwned;
        return result;
    }
    result.unmap_succeeded =
        backing.pointer == nullptr || adapter.UnmapPreservingPlaceholder(backing.pointer);
    result.release_succeeded =
        !backing.reservation || adapter.ReleasePlaceholder(backing.reservation);
    result.close_succeeded = !backing.mapping || adapter.CloseMapping(backing.mapping);
    if (!result.unmap_succeeded || !result.release_succeeded || !result.close_succeeded) {
        result.failure = ExactWindowsBackingFailure::CleanupFailed;
    }
    backing.pointer = nullptr;
    backing.reservation = {};
    backing.mapping = {};
    backing.size = 0;
    return result;
}

enum class ExactHostImportStage : std::uint8_t {
    NotStarted,
    Capability,
    DeviceLimits,
    ExternalBufferProperties,
    BufferCreation,
    MemoryRequirements,
    Backing,
    HostPointerProperties,
    MemoryTypeSelection,
    MemoryAllocation,
    MemoryBinding,
    DeviceAddress,
    DataVerification,
    Retained,
};

class ExactHostImportProtocol {
public:
    [[nodiscard]] constexpr bool Complete(ExactHostImportStage stage) noexcept {
        if (failed || stage != NextStage()) {
            failed = true;
            return false;
        }
        completed = stage;
        return true;
    }

    [[nodiscard]] constexpr bool CanAllocateLargeBacking() const noexcept {
        return !failed && completed == ExactHostImportStage::MemoryRequirements;
    }

    [[nodiscard]] constexpr ExactHostImportStage CompletedStage() const noexcept {
        return completed;
    }

private:
    [[nodiscard]] constexpr ExactHostImportStage NextStage() const noexcept {
        return static_cast<ExactHostImportStage>(static_cast<std::uint8_t>(completed) + 1);
    }

private:
    ExactHostImportStage completed{ExactHostImportStage::NotStarted};
    bool failed{};
};

enum class ExactHostImportFailure : std::uint8_t {
    None,
    ExtensionUnavailable,
    BufferDeviceAddressUnavailable,
    HandleNotImportable,
    NoCompatibleMemoryType,
    NoCoherentMemoryType,
    RequirementExceedsBacking,
    RequirementAlignmentMismatch,
    BackingCommitFailed,
    VulkanOutOfMemory,
    InvalidArguments,
    DeviceLimitExceeded,
    Win32LifecycleFailed,
    VulkanCallFailed,
    ZeroDeviceAddress,
    DataVerificationFailed,
    CleanupFailed,
    ProtocolViolation,
};

enum class ExactHostImportDisposition : std::uint8_t {
    Pass,
    Unsupported,
    ResourceLimited,
    ExactDesignIncompatible,
    Error,
};

enum class ExactWindowsBackingErrorClass : std::uint8_t {
    ResourceLimited,
    Other,
};

[[nodiscard]] constexpr ExactWindowsBackingErrorClass ClassifyExactWindowsBackingErrorCode(
    std::uint32_t error_code) noexcept {
    // Stable Win32 values from winerror.h: ERROR_NOT_ENOUGH_MEMORY, ERROR_OUTOFMEMORY, and
    // ERROR_COMMITMENT_LIMIT. Everything else fails closed as a lifecycle error.
    switch (error_code) {
    case 8:
    case 14:
    case 1455:
        return ExactWindowsBackingErrorClass::ResourceLimited;
    default:
        return ExactWindowsBackingErrorClass::Other;
    }
}

enum class ExactVulkanFailureClass : std::uint8_t {
    InvalidExternalHandle,
    OutOfMemory,
    Other,
};

struct ExactSelectedMemoryTypeEvidence {
    std::uint32_t property_flags{};
    bool host_coherent{};
};

[[nodiscard]] constexpr bool HasRequiredExactHostVisibilityEvidence(
    bool canonical_view_verified, bool guest_alias_verified) noexcept {
    return canonical_view_verified && guest_alias_verified;
}

[[nodiscard]] constexpr bool HasRequiredExactGuestPublicationEvidence(
    bool canonical_view_verified, bool guest_alias_verified,
    bool guest_page_table_verified, bool imported_before_verified,
    bool cache_override_verified, bool imported_after_restore_verified,
    bool direct_import_control_verified) noexcept {
    return HasRequiredExactHostVisibilityEvidence(canonical_view_verified, guest_alias_verified) &&
           guest_page_table_verified && imported_before_verified && cache_override_verified &&
           imported_after_restore_verified && direct_import_control_verified;
}

[[nodiscard]] constexpr ExactSelectedMemoryTypeEvidence MakeExactSelectedMemoryTypeEvidence(
    std::uint32_t property_flags, std::uint32_t host_coherent_flag) noexcept {
    return {.property_flags = property_flags,
            .host_coherent = static_cast<bool>(property_flags & host_coherent_flag)};
}

[[nodiscard]] constexpr ExactHostImportFailure ClassifyExactBackingAcquisitionFailure(
    ExactWindowsBackingErrorClass error_class, bool rollback_complete) noexcept {
    if (!rollback_complete || error_class == ExactWindowsBackingErrorClass::Other) {
        return ExactHostImportFailure::Win32LifecycleFailed;
    }
    return ExactHostImportFailure::BackingCommitFailed;
}

[[nodiscard]] constexpr ExactHostImportFailure ClassifyExactVulkanFailure(
    ExactVulkanFailureClass failure_class) noexcept {
    switch (failure_class) {
    case ExactVulkanFailureClass::InvalidExternalHandle:
        return ExactHostImportFailure::HandleNotImportable;
    case ExactVulkanFailureClass::OutOfMemory:
        return ExactHostImportFailure::VulkanOutOfMemory;
    case ExactVulkanFailureClass::Other:
        return ExactHostImportFailure::VulkanCallFailed;
    }
    return ExactHostImportFailure::VulkanCallFailed;
}

[[nodiscard]] constexpr ExactHostImportDisposition ClassifyExactHostImportFailure(
    ExactHostImportFailure failure) noexcept {
    switch (failure) {
    case ExactHostImportFailure::None:
        return ExactHostImportDisposition::Pass;
    case ExactHostImportFailure::ExtensionUnavailable:
    case ExactHostImportFailure::BufferDeviceAddressUnavailable:
    case ExactHostImportFailure::HandleNotImportable:
    case ExactHostImportFailure::NoCompatibleMemoryType:
    case ExactHostImportFailure::NoCoherentMemoryType:
    case ExactHostImportFailure::DeviceLimitExceeded:
        return ExactHostImportDisposition::Unsupported;
    case ExactHostImportFailure::RequirementExceedsBacking:
    case ExactHostImportFailure::RequirementAlignmentMismatch:
        return ExactHostImportDisposition::ExactDesignIncompatible;
    case ExactHostImportFailure::BackingCommitFailed:
    case ExactHostImportFailure::VulkanOutOfMemory:
        return ExactHostImportDisposition::ResourceLimited;
    case ExactHostImportFailure::InvalidArguments:
    case ExactHostImportFailure::Win32LifecycleFailed:
    case ExactHostImportFailure::VulkanCallFailed:
    case ExactHostImportFailure::ZeroDeviceAddress:
    case ExactHostImportFailure::DataVerificationFailed:
    case ExactHostImportFailure::CleanupFailed:
    case ExactHostImportFailure::ProtocolViolation:
        return ExactHostImportDisposition::Error;
    }
    return ExactHostImportDisposition::Error;
}

} // namespace Vulkan
