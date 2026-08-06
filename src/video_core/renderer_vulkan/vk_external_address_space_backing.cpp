// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <cstring>
#include <limits>

#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_external_address_space_backing.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {
namespace {

constexpr auto HostAllocationHandle = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
constexpr auto ImportedBufferUsage =
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
    vk::BufferUsageFlagBits::eTransferDst;

const char* FailureName(ExternalAddressSpaceImportFailure failure) noexcept {
    switch (failure) {
    case ExternalAddressSpaceImportFailure::None:
        return "none";
    case ExternalAddressSpaceImportFailure::ExtensionUnavailable:
        return "extension unavailable";
    case ExternalAddressSpaceImportFailure::BufferDeviceAddressUnavailable:
        return "buffer device address unavailable";
    case ExternalAddressSpaceImportFailure::HostAllocationNotImportable:
        return "host allocation not importable";
    case ExternalAddressSpaceImportFailure::BackingMismatch:
        return "backing mismatch";
    case ExternalAddressSpaceImportFailure::PointerMisaligned:
        return "host pointer or allocation size misaligned";
    case ExternalAddressSpaceImportFailure::RequirementExceedsBacking:
        return "memory requirement exceeds backing";
    case ExternalAddressSpaceImportFailure::DeviceLimitExceeded:
        return "device size limit exceeded";
    case ExternalAddressSpaceImportFailure::BackingRangeOverflow:
        return "backing range overflow";
    case ExternalAddressSpaceImportFailure::NoCompatibleMemoryType:
        return "no compatible memory type";
    case ExternalAddressSpaceImportFailure::NoCoherentMemoryType:
        return "no coherent memory type";
    case ExternalAddressSpaceImportFailure::MemoryPropertyEvidenceMissing:
        return "memory property evidence missing";
    case ExternalAddressSpaceImportFailure::MemoryTypeOutOfRange:
        return "memory type out of range";
    case ExternalAddressSpaceImportFailure::InvalidRequirement:
        return "invalid requirement";
    }
    return "unknown";
}

void LogUnavailable(const char* reason) {
    LOG_INFO(Render_Vulkan, "External address-space backing import disabled: {}", reason);
}

} // namespace

ExternalAddressSpaceBacking::ExternalAddressSpaceBacking(const Instance& instance,
                                                         Core::AddressSpaceBackingLease lease) {
    if (!lease) {
        LogUnavailable("canonical backing lease unavailable");
        return;
    }
    if (!instance.IsExternalMemoryHostSupported()) {
        // Instance::CreateDevice already reports unavailable optional extensions once.
        return;
    }
    if (!instance.GetVk12Features().bufferDeviceAddress) {
        LogUnavailable("buffer device address unavailable");
        return;
    }
    const auto lease_pointer = reinterpret_cast<std::uintptr_t>(lease.Base());
    if (lease_pointer == 0 || lease.Size() == 0 ||
        lease.Size() > std::numeric_limits<std::uintptr_t>::max() - lease_pointer) {
        LogUnavailable("canonical backing range is invalid");
        return;
    }

    const auto device = instance.GetDevice();
    const auto physical_device = instance.GetPhysicalDevice();
    const auto external_properties = physical_device.getExternalBufferProperties({
        .flags = {},
        .usage = ImportedBufferUsage,
        .handleType = HostAllocationHandle,
    });
    const auto external_features =
        external_properties.externalMemoryProperties.externalMemoryFeatures;
    const bool importable =
        bool(external_features & vk::ExternalMemoryFeatureFlagBits::eImportable) &&
        bool(external_properties.externalMemoryProperties.compatibleHandleTypes &
             HostAllocationHandle);
    const bool dedicated =
        bool(external_features & vk::ExternalMemoryFeatureFlagBits::eDedicatedOnly);
    if (!importable) {
        LogUnavailable("host allocation handle is not importable");
        return;
    }
    if (instance.MinImportedHostPointerAlignment() == 0 ||
        lease_pointer % instance.MinImportedHostPointerAlignment() != 0 ||
        lease.Size() % instance.MinImportedHostPointerAlignment() != 0) {
        LogUnavailable("canonical backing does not satisfy host import alignment");
        return;
    }
    if (lease.Size() > instance.MaxBufferSize() ||
        lease.Size() > instance.MaxMemoryAllocationSize()) {
        LogUnavailable("canonical backing exceeds device size limits");
        return;
    }

    const vk::ExternalMemoryBufferCreateInfo external_buffer_info{
        .handleTypes = HostAllocationHandle,
    };
    const vk::BufferCreateInfo buffer_info{
        .pNext = &external_buffer_info,
        .size = lease.Size(),
        .usage = ImportedBufferUsage,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    ExternalAddressSpaceImportStaging<vk::UniqueDeviceMemory, vk::UniqueBuffer> candidates;
    auto [buffer_result, created_buffer] = device.createBufferUnique(buffer_info);
    if (buffer_result != vk::Result::eSuccess) {
        LogUnavailable("buffer creation failed");
        return;
    }
    candidates.buffer.emplace(std::move(created_buffer));

    const auto memory_requirements = device.getBufferMemoryRequirements(**candidates.buffer);
    const auto [host_pointer_result, host_pointer_properties] =
        device.getMemoryHostPointerPropertiesEXT(HostAllocationHandle, lease.Base());
    if (host_pointer_result != vk::Result::eSuccess) {
        LogUnavailable("host pointer property query failed");
        return;
    }

    const auto& memory_properties = instance.GetMemoryProperties();
    std::array<u32, VK_MAX_MEMORY_TYPES> memory_property_flags{};
    for (u32 index = 0; index < memory_properties.memoryTypeCount; ++index) {
        memory_property_flags[index] =
            static_cast<u32>(memory_properties.memoryTypes[index].propertyFlags);
    }

    const ExternalAddressSpaceImportRequest request{
        .extension_available = instance.IsExternalMemoryHostSupported(),
        .buffer_device_address_available =
            static_cast<bool>(instance.GetVk12Features().bufferDeviceAddress),
        .host_allocation_importable = importable,
        .dedicated_allocation_required = dedicated,
        .lease_pointer = lease_pointer,
        .lease_size = lease.Size(),
        .import_pointer = lease_pointer,
        .import_size = lease.Size(),
        .minimum_imported_pointer_alignment = instance.MinImportedHostPointerAlignment(),
        .memory_requirement_size = memory_requirements.size,
        .memory_requirement_alignment = memory_requirements.alignment,
        .maximum_buffer_size = instance.MaxBufferSize(),
        .maximum_memory_allocation_size = instance.MaxMemoryAllocationSize(),
        .buffer_memory_type_bits = memory_requirements.memoryTypeBits,
        .host_pointer_memory_type_bits = host_pointer_properties.memoryTypeBits,
        .memory_type_count = memory_properties.memoryTypeCount,
        .host_coherent_property = static_cast<u32>(vk::MemoryPropertyFlagBits::eHostCoherent),
    };
    const auto plan = PlanExternalAddressSpaceBackingImport(
        request,
        std::span<const u32>{memory_property_flags.data(), memory_properties.memoryTypeCount});
    if (plan.failure != ExternalAddressSpaceImportFailure::None) {
        LogUnavailable(FailureName(plan.failure));
        return;
    }

    const vk::ImportMemoryHostPointerInfoEXT import_info{
        .handleType = HostAllocationHandle,
        .pHostPointer = lease.Base(),
    };
    const vk::MemoryDedicatedAllocateInfo dedicated_info{
        .buffer = **candidates.buffer,
    };
    const vk::MemoryAllocateFlagsInfo allocation_flags{
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    const vk::StructureChain allocation_chain{
        vk::MemoryAllocateInfo{
            .allocationSize = lease.Size(),
            .memoryTypeIndex = plan.memory_type_index,
        },
        import_info,
        dedicated_info,
        allocation_flags,
    };
    auto [memory_result, created_memory] =
        device.allocateMemoryUnique(allocation_chain.get<vk::MemoryAllocateInfo>());
    if (memory_result != vk::Result::eSuccess) {
        LogUnavailable("host memory import failed");
        return;
    }
    candidates.memory.emplace(std::move(created_memory));
    if (device.bindBufferMemory(**candidates.buffer, **candidates.memory, 0) !=
        vk::Result::eSuccess) {
        LogUnavailable("imported memory bind failed");
        return;
    }

    const vk::DeviceAddress address = device.getBufferAddress({.buffer = **candidates.buffer});
    if (address == 0) {
        LogUnavailable("buffer device address is zero");
        return;
    }

    resources.emplace(std::move(lease), std::move(*candidates.memory),
                      std::move(*candidates.buffer));
    device_address = address;
    LOG_INFO(
        Render_Vulkan,
        "Retained external address-space backing import ({} bytes); guest page publications: 0",
        request.lease_size);
}

bool ExternalAddressSpaceBacking::TryWritePhysical(u64 physical_offset,
                                                   std::span<const u8> bytes) noexcept {
    if (!resources ||
        !IsExternalAddressSpacePhysicalWriteRangeValid(resources->lease.Size(), physical_offset,
                                                       bytes.size())) {
        return false;
    }
    std::memcpy(resources->lease.Base() + physical_offset, bytes.data(), bytes.size());
    std::atomic_thread_fence(std::memory_order_release);
    return true;
}

} // namespace Vulkan
