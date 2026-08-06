// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bit>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>

namespace Vulkan {

enum class ExternalAddressSpaceImportFailure : std::uint8_t {
    None,
    ExtensionUnavailable,
    BufferDeviceAddressUnavailable,
    HostAllocationNotImportable,
    BackingMismatch,
    PointerMisaligned,
    RequirementExceedsBacking,
    DeviceLimitExceeded,
    BackingRangeOverflow,
    NoCompatibleMemoryType,
    NoCoherentMemoryType,
    MemoryPropertyEvidenceMissing,
    MemoryTypeOutOfRange,
    InvalidRequirement,
};

struct ExternalAddressSpaceImportRequest {
    bool extension_available{};
    bool buffer_device_address_available{};
    bool host_allocation_importable{};
    bool dedicated_allocation_required{};
    std::uintptr_t lease_pointer{};
    std::uint64_t lease_size{};
    std::uintptr_t import_pointer{};
    std::uint64_t import_size{};
    std::uint64_t minimum_imported_pointer_alignment{};
    std::uint64_t memory_requirement_size{};
    std::uint64_t memory_requirement_alignment{};
    std::uint64_t maximum_buffer_size{};
    std::uint64_t maximum_memory_allocation_size{};
    std::uint32_t buffer_memory_type_bits{};
    std::uint32_t host_pointer_memory_type_bits{};
    std::uint32_t memory_type_count{};
    std::uint32_t host_coherent_property{};
};

struct ExternalAddressSpaceImportPlan {
    ExternalAddressSpaceImportFailure failure{ExternalAddressSpaceImportFailure::None};
    std::uint32_t compatible_memory_type_bits{};
    std::uint32_t memory_type_index{};
    bool use_dedicated_allocation{};
};

[[nodiscard]] constexpr bool IsPowerOfTwo(std::uint64_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

[[nodiscard]] constexpr bool IsExternalAddressSpacePhysicalWriteRangeValid(
    std::uint64_t backing_size, std::uint64_t physical_offset,
    std::uint64_t write_size) noexcept {
    return backing_size != 0 && write_size != 0 && physical_offset < backing_size &&
           write_size <= backing_size - physical_offset;
}

[[nodiscard]] constexpr ExternalAddressSpaceImportPlan PlanExternalAddressSpaceBackingImport(
    const ExternalAddressSpaceImportRequest& request,
    std::span<const std::uint32_t> memory_property_flags) noexcept {
    if (!request.extension_available) {
        return {.failure = ExternalAddressSpaceImportFailure::ExtensionUnavailable};
    }
    if (!request.buffer_device_address_available) {
        return {.failure = ExternalAddressSpaceImportFailure::BufferDeviceAddressUnavailable};
    }
    if (!request.host_allocation_importable) {
        return {.failure = ExternalAddressSpaceImportFailure::HostAllocationNotImportable};
    }
    if (request.lease_pointer == 0 || request.lease_size == 0 ||
        request.import_pointer != request.lease_pointer ||
        request.import_size != request.lease_size) {
        return {.failure = ExternalAddressSpaceImportFailure::BackingMismatch};
    }
    if (!IsPowerOfTwo(request.minimum_imported_pointer_alignment) ||
        !IsPowerOfTwo(request.memory_requirement_alignment) ||
        request.memory_requirement_size == 0 || request.maximum_buffer_size == 0 ||
        request.maximum_memory_allocation_size == 0 || request.memory_type_count == 0 ||
        request.memory_type_count > 32 || request.host_coherent_property == 0) {
        return {.failure = ExternalAddressSpaceImportFailure::InvalidRequirement};
    }
    if (request.lease_pointer % request.minimum_imported_pointer_alignment != 0 ||
        request.lease_size % request.minimum_imported_pointer_alignment != 0) {
        return {.failure = ExternalAddressSpaceImportFailure::PointerMisaligned};
    }
    if (request.memory_requirement_size > request.lease_size) {
        return {.failure = ExternalAddressSpaceImportFailure::RequirementExceedsBacking};
    }
    if (request.lease_size > request.maximum_buffer_size ||
        request.lease_size > request.maximum_memory_allocation_size) {
        return {.failure = ExternalAddressSpaceImportFailure::DeviceLimitExceeded};
    }
    if (request.lease_size > std::numeric_limits<std::uintptr_t>::max() - request.lease_pointer) {
        return {.failure = ExternalAddressSpaceImportFailure::BackingRangeOverflow};
    }

    const auto compatible_bits =
        request.buffer_memory_type_bits & request.host_pointer_memory_type_bits;
    if (compatible_bits == 0) {
        return {.failure = ExternalAddressSpaceImportFailure::NoCompatibleMemoryType};
    }

    for (std::uint32_t index = 0; index < 32; ++index) {
        const auto bit = std::uint32_t{1} << index;
        if ((compatible_bits & bit) == 0) {
            continue;
        }
        if (index >= request.memory_type_count) {
            return {.failure = ExternalAddressSpaceImportFailure::MemoryTypeOutOfRange,
                    .compatible_memory_type_bits = compatible_bits};
        }
        if (index >= memory_property_flags.size()) {
            return {.failure = ExternalAddressSpaceImportFailure::MemoryPropertyEvidenceMissing,
                    .compatible_memory_type_bits = compatible_bits};
        }
        if ((memory_property_flags[index] & request.host_coherent_property) != 0) {
            return {.failure = ExternalAddressSpaceImportFailure::None,
                    .compatible_memory_type_bits = compatible_bits,
                    .memory_type_index = index,
                    .use_dedicated_allocation = true};
        }
    }
    return {.failure = ExternalAddressSpaceImportFailure::NoCoherentMemoryType,
            .compatible_memory_type_bits = compatible_bits};
}

/** Owns an imported buffer in dependency order; C++ destroys buffer, memory, then lease. */
template <typename Lease, typename Memory, typename Buffer>
struct ExternalAddressSpaceImportResources {
    ExternalAddressSpaceImportResources(Lease lease_, Memory memory_, Buffer buffer_)
        : lease{std::move(lease_)}, memory{std::move(memory_)}, buffer{std::move(buffer_)} {}

    Lease lease;
    Memory memory;
    Buffer buffer;
};

/** Keeps partial construction failures in Vulkan's required buffer-before-memory order. */
template <typename Memory, typename Buffer>
struct ExternalAddressSpaceImportStaging {
    std::optional<Memory> memory;
    std::optional<Buffer> buffer;
};

class ExternalAddressSpaceImportOwnership {
public:
    [[nodiscard]] constexpr bool RetainLease() noexcept {
        if (lease || memory || buffer) {
            return false;
        }
        lease = true;
        return true;
    }

    [[nodiscard]] constexpr bool OwnMemory() noexcept {
        if (!lease || memory || buffer) {
            return false;
        }
        memory = true;
        return true;
    }

    [[nodiscard]] constexpr bool OwnBuffer() noexcept {
        if (!lease || !memory || buffer) {
            return false;
        }
        buffer = true;
        return true;
    }

    [[nodiscard]] constexpr bool RetainDeviceAddress(std::uint64_t address) noexcept {
        if (!buffer || address == 0) {
            return false;
        }
        device_address = address;
        return true;
    }

    [[nodiscard]] constexpr bool ReleaseBuffer() noexcept {
        if (!buffer) {
            return false;
        }
        device_address = 0;
        buffer = false;
        return true;
    }

    [[nodiscard]] constexpr bool ReleaseMemory() noexcept {
        if (buffer || !memory) {
            return false;
        }
        memory = false;
        return true;
    }

    [[nodiscard]] constexpr bool ReleaseLease() noexcept {
        if (buffer || memory || !lease) {
            return false;
        }
        lease = false;
        return true;
    }

    [[nodiscard]] constexpr bool IsReady() const noexcept {
        return lease && memory && buffer && device_address != 0;
    }

    [[nodiscard]] constexpr bool IsEmpty() const noexcept {
        return !lease && !memory && !buffer && device_address == 0;
    }

    [[nodiscard]] static constexpr std::uint64_t GuestPagePublicationCount() noexcept {
        return 0;
    }

private:
    bool lease{};
    bool memory{};
    bool buffer{};
    std::uint64_t device_address{};
};

} // namespace Vulkan
