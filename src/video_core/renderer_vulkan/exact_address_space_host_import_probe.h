// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include "core/address_space_backing_size.h"

namespace Vulkan {

inline constexpr std::uint64_t ExactAddressSpaceBaseBackingSize =
    Core::AddressSpaceBaseBackingSize;
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

template <typename Adapter>
struct ExactWindowsBackingAcquisition {
    ExactWindowsAddressSpaceBacking<Adapter> backing{};
    ExactWindowsBackingFailure failure{ExactWindowsBackingFailure::None};
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
        result.rollback_complete = adapter.CloseMapping(backing.mapping);
        backing.mapping = {};
        return result;
    }
    backing.pointer = adapter.MapReplacingPlaceholder(backing.mapping, backing.reservation, size,
                                                      ExactMapViewRecipe);
    if (backing.pointer == nullptr) {
        result.failure = ExactWindowsBackingFailure::ViewMappingFailed;
        const bool released = adapter.ReleasePlaceholder(backing.reservation);
        const bool closed = adapter.CloseMapping(backing.mapping);
        result.rollback_complete = released && closed;
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
    result.release_succeeded = !backing.reservation || adapter.ReleasePlaceholder(backing.reservation);
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
        return ExactHostImportDisposition::Unsupported;
    case ExactHostImportFailure::RequirementExceedsBacking:
    case ExactHostImportFailure::RequirementAlignmentMismatch:
    case ExactHostImportFailure::DeviceLimitExceeded:
        return ExactHostImportDisposition::ExactDesignIncompatible;
    case ExactHostImportFailure::BackingCommitFailed:
    case ExactHostImportFailure::VulkanOutOfMemory:
        return ExactHostImportDisposition::ResourceLimited;
    case ExactHostImportFailure::InvalidArguments:
    case ExactHostImportFailure::Win32LifecycleFailed:
    case ExactHostImportFailure::VulkanCallFailed:
    case ExactHostImportFailure::ZeroDeviceAddress:
    case ExactHostImportFailure::CleanupFailed:
    case ExactHostImportFailure::ProtocolViolation:
        return ExactHostImportDisposition::Error;
    }
    return ExactHostImportDisposition::Error;
}

} // namespace Vulkan
