// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include <nlohmann/json.hpp>

#include "video_core/renderer_vulkan/external_host_memory_probe.h"
#include "video_core/renderer_vulkan/vk_common.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

using Json = nlohmann::ordered_json;

constexpr std::uint64_t MaxBackingSize = 64ull * 1024 * 1024;
constexpr std::size_t MaxDiagnosticText = 256;

struct Arguments {
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint64_t backing_size{};
};

[[nodiscard]] constexpr std::string_view StageName(Vulkan::ExternalHostMemoryProbeStage stage) {
    using enum Vulkan::ExternalHostMemoryProbeStage;
    switch (stage) {
    case NotStarted:
        return "not_started";
    case Capability:
        return "capability";
    case Backing:
        return "backing";
    case ExternalBufferProperties:
        return "external_buffer_properties";
    case BufferCreation:
        return "buffer_creation";
    case MemoryRequirements:
        return "memory_requirements";
    case HostPointerProperties:
        return "host_pointer_properties";
    case MemoryTypeSelection:
        return "memory_type_selection";
    case MemoryAllocation:
        return "memory_allocation";
    case MemoryBinding:
        return "memory_binding";
    case DeviceAddress:
        return "device_address";
    case Retained:
        return "retained";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view FailureName(
    Vulkan::ExternalHostMemoryProbeFailure failure) {
    using enum Vulkan::ExternalHostMemoryProbeFailure;
    switch (failure) {
    case None:
        return "none";
    case UnexpectedStage:
        return "unexpected_stage";
    case ExtensionUnavailable:
        return "extension_unavailable";
    case InvalidPointerAlignment:
        return "invalid_pointer_alignment";
    case BackingAllocationFailed:
        return "backing_allocation_failed";
    case BackingValidationFailed:
        return "backing_validation_failed";
    case ExternalBufferQueryFailed:
        return "external_buffer_query_failed";
    case HandleTypeNotImportable:
        return "handle_type_not_importable";
    case BufferCreationFailed:
        return "buffer_creation_failed";
    case MemoryRequirementsInvalid:
        return "memory_requirements_invalid";
    case HostPointerQueryFailed:
        return "host_pointer_query_failed";
    case NoCompatibleMemoryType:
        return "no_compatible_memory_type";
    case NoCoherentMemoryType:
        return "no_coherent_memory_type";
    case MemoryAllocationFailed:
        return "memory_allocation_failed";
    case MemoryBindingFailed:
        return "memory_binding_failed";
    case ZeroDeviceAddress:
        return "zero_device_address";
    }
    return "unknown";
}

[[nodiscard]] std::optional<std::uint64_t> ParseUnsigned(std::string_view value) {
    int base = 10;
    if (value.starts_with("0x") || value.starts_with("0X")) {
        value.remove_prefix(2);
        base = 16;
    }
    if (value.empty()) {
        return std::nullopt;
    }
    std::uint64_t parsed{};
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), parsed, base);
    if (error != std::errc{} || end != value.data() + value.size()) {
        return std::nullopt;
    }
    return parsed;
}

[[nodiscard]] std::string BoundedDiagnosticText(std::string_view value) {
    return std::string{value.substr(0, MaxDiagnosticText)};
}

[[nodiscard]] std::optional<Arguments> ParseArguments(int argc, char** argv, std::string& error) {
    std::optional<std::uint64_t> vendor;
    std::optional<std::uint64_t> device;
    std::optional<std::uint64_t> size;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            error = "every option requires a value";
            return std::nullopt;
        }
        const std::string_view option{argv[index]};
        const auto value = ParseUnsigned(argv[index + 1]);
        if (!value) {
            error = "invalid unsigned option value";
            return std::nullopt;
        }
        if (option == "--vendor-id") {
            vendor = value;
        } else if (option == "--device-id") {
            device = value;
        } else if (option == "--size-bytes") {
            size = value;
        } else {
            error = "unknown option";
            return std::nullopt;
        }
    }
    if (!vendor || !device || !size || *vendor > std::numeric_limits<std::uint32_t>::max() ||
        *device > std::numeric_limits<std::uint32_t>::max() || *size == 0 ||
        *size > MaxBackingSize) {
        error = "vendor, device, and a 1..67108864 byte backing are required";
        return std::nullopt;
    }
    return Arguments{.vendor_id = static_cast<std::uint32_t>(*vendor),
                     .device_id = static_cast<std::uint32_t>(*device),
                     .backing_size = *size};
}

[[nodiscard]] Json BaseReport() {
    return Json{
        {"schema", "shadps4.external-host-memory-import-probe.v1"},
        {"backingScope", "bounded_synthetic"},
        {"backingSizeBytes", 0},
        {"mechanismEvidenceOnly", true},
        {"publicationAttempted", false},
        {"guestPagesPublished", 0},
        {"externalHostExtensionAvailable", false},
        {"externalHostEnabled", false},
        {"bufferDeviceAddressFeature", false},
        {"bufferDeviceAddressEnabled", false},
        {"cleanupComplete", false},
        {"durationMs", 0},
        {"status", "error"},
        {"reason", "not_started"},
        {"attempts", Json::array()},
    };
}

[[nodiscard]] std::string HexBytes(std::span<const std::uint8_t> bytes) {
    constexpr std::array digits{'0', '1', '2', '3', '4', '5', '6', '7',
                                '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    std::string result(bytes.size() * 2, '0');
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        result[index * 2] = digits[bytes[index] >> 4];
        result[index * 2 + 1] = digits[bytes[index] & 0xf];
    }
    return result;
}

struct PagefileBacking {
#ifdef _WIN32
    HANDLE mapping{};
#endif
    void* pointer{};
    std::uint64_t size{};

    PagefileBacking() = default;
    PagefileBacking(const PagefileBacking&) = delete;
    PagefileBacking& operator=(const PagefileBacking&) = delete;

    PagefileBacking(PagefileBacking&& other) noexcept
#ifdef _WIN32
        : mapping{std::exchange(other.mapping, nullptr)},
#endif
          pointer{std::exchange(other.pointer, nullptr)}, size{std::exchange(other.size, 0)} {
    }

    PagefileBacking& operator=(PagefileBacking&&) = delete;

    ~PagefileBacking() {
#ifdef _WIN32
        if (pointer != nullptr) {
            UnmapViewOfFile(pointer);
        }
        if (mapping != nullptr) {
            CloseHandle(mapping);
        }
#endif
    }

    [[nodiscard]] explicit operator bool() const noexcept {
#ifdef _WIN32
        return mapping != nullptr && pointer != nullptr;
#else
        return false;
#endif
    }
};

[[nodiscard]] PagefileBacking AllocatePagefileBacking(std::uint64_t size) {
    PagefileBacking backing;
#ifdef _WIN32
    backing.mapping =
        CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT,
                           static_cast<DWORD>(size >> 32), static_cast<DWORD>(size), nullptr);
    if (backing.mapping == nullptr) {
        return backing;
    }
    backing.pointer = MapViewOfFile(backing.mapping, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (backing.pointer != nullptr) {
        backing.size = size;
    }
#else
    static_cast<void>(size);
#endif
    return backing;
}

[[nodiscard]] Json InitialAttempt(std::string_view handle_name) {
    return Json{
        {"handleType", handle_name},
        {"completedStage", "not_started"},
        {"failureStage", "not_started"},
        {"failure", "none"},
        {"succeeded", false},
        {"importable", false},
        {"exportable", false},
        {"dedicatedOnly", false},
        {"compatibleHandleTypes", 0},
        {"pointerModuloAlignment", nullptr},
        {"backingSizeBytes", 0},
        {"bufferMemoryTypeBits", 0},
        {"hostPointerMemoryTypeBits", 0},
        {"compatibleMemoryTypeBits", 0},
        {"selectedMemoryTypeIndex", nullptr},
        {"selectedMemoryTypeHostCoherent", false},
        {"selectedMemoryPropertyFlags", 0},
        {"memoryRequirementBytes", 0},
        {"memoryRequirementAlignment", 0},
        {"bufferCreateVkResult", "not_called"},
        {"hostPointerQueryVkResult", "not_called"},
        {"memoryAllocateVkResult", "not_called"},
        {"bindVkResult", "not_called"},
        {"deviceAddressNonzero", false},
        {"deviceAddressModuloAlignment", nullptr},
        {"cleanupComplete", false},
        {"durationMs", 0},
    };
}

void RecordProgress(Json& attempt, const Vulkan::ExternalHostMemoryProbeProgress& progress) {
    const auto& result = progress.Result();
    attempt["completedStage"] = StageName(result.completed_stage);
    attempt["failureStage"] = StageName(result.failure_stage);
    attempt["failure"] = FailureName(result.failure);
    attempt["succeeded"] = result.Succeeded();
}

[[nodiscard]] Json ProbeHandle(vk::PhysicalDevice physical_device, vk::Device device,
                               vk::ExternalMemoryHandleTypeFlagBits handle_type,
                               std::string_view handle_name, std::uint64_t requested_backing_size,
                               std::size_t min_pointer_alignment) {
    using namespace Vulkan;

    const auto started = std::chrono::steady_clock::now();
    Json attempt = InitialAttempt(handle_name);
    [&] {
        ExternalHostMemoryProbeProgress progress;
        const auto complete_stage = [&](ExternalHostMemoryProbeStage stage) {
            if (progress.Complete(stage)) {
                return true;
            }
            RecordProgress(attempt, progress);
            return false;
        };
        if (!complete_stage(ExternalHostMemoryProbeStage::Capability)) {
            return;
        }

        if (!std::has_single_bit(min_pointer_alignment) || min_pointer_alignment > MaxBackingSize) {
            progress.Fail(ExternalHostMemoryProbeStage::Backing,
                          ExternalHostMemoryProbeFailure::InvalidPointerAlignment);
            RecordProgress(attempt, progress);
            return;
        }
        const auto remainder = requested_backing_size & (min_pointer_alignment - 1);
        const auto aligned_size = remainder == 0
                                      ? requested_backing_size
                                      : requested_backing_size + min_pointer_alignment - remainder;
        if (aligned_size < requested_backing_size || aligned_size > MaxBackingSize) {
            progress.Fail(ExternalHostMemoryProbeStage::Backing,
                          ExternalHostMemoryProbeFailure::BackingValidationFailed);
            RecordProgress(attempt, progress);
            return;
        }

        auto backing = AllocatePagefileBacking(aligned_size);
        if (!backing) {
            progress.Fail(ExternalHostMemoryProbeStage::Backing,
                          ExternalHostMemoryProbeFailure::BackingAllocationFailed);
            RecordProgress(attempt, progress);
            return;
        }
        attempt["backingSizeBytes"] = backing.size;
        attempt["pointerModuloAlignment"] =
            reinterpret_cast<std::uintptr_t>(backing.pointer) % min_pointer_alignment;
        const auto backing_validation = ValidateImportedHostAllocation(
            backing.pointer, backing.size, min_pointer_alignment, backing.size);
        if (backing_validation != ImportedHostAllocationValidation::Valid) {
            progress.Fail(ExternalHostMemoryProbeStage::Backing,
                          ExternalHostMemoryProbeFailure::BackingValidationFailed);
            RecordProgress(attempt, progress);
            return;
        }
        if (!complete_stage(ExternalHostMemoryProbeStage::Backing)) {
            return;
        }

        constexpr vk::BufferUsageFlags usage =
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
        const vk::PhysicalDeviceExternalBufferInfo external_info{
            .usage = usage,
            .handleType = handle_type,
        };
        const auto external_properties = physical_device.getExternalBufferProperties(external_info);
        const auto& external_memory = external_properties.externalMemoryProperties;
        const auto external_features = external_memory.externalMemoryFeatures;
        attempt["compatibleHandleTypes"] =
            static_cast<std::uint32_t>(external_memory.compatibleHandleTypes);
        attempt["importable"] =
            static_cast<bool>(external_features & vk::ExternalMemoryFeatureFlagBits::eImportable);
        attempt["exportable"] =
            static_cast<bool>(external_features & vk::ExternalMemoryFeatureFlagBits::eExportable);
        attempt["dedicatedOnly"] = static_cast<bool>(
            external_features & vk::ExternalMemoryFeatureFlagBits::eDedicatedOnly);
        const bool compatible =
            static_cast<bool>(external_memory.compatibleHandleTypes & handle_type);
        if (!attempt["importable"].get<bool>() || !compatible) {
            progress.Fail(ExternalHostMemoryProbeStage::ExternalBufferProperties,
                          ExternalHostMemoryProbeFailure::HandleTypeNotImportable);
            RecordProgress(attempt, progress);
            return;
        }
        if (!complete_stage(ExternalHostMemoryProbeStage::ExternalBufferProperties)) {
            return;
        }

        const vk::ExternalMemoryBufferCreateInfo external_buffer_info{.handleTypes = handle_type};
        const vk::BufferCreateInfo buffer_info{
            .pNext = &external_buffer_info,
            .size = backing.size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };
        auto [buffer_result, buffer] = device.createBufferUnique(buffer_info);
        attempt["bufferCreateVkResult"] = vk::to_string(buffer_result);
        if (buffer_result != vk::Result::eSuccess) {
            progress.Fail(ExternalHostMemoryProbeStage::BufferCreation,
                          ExternalHostMemoryProbeFailure::BufferCreationFailed);
            RecordProgress(attempt, progress);
            return;
        }
        if (!complete_stage(ExternalHostMemoryProbeStage::BufferCreation)) {
            return;
        }

        const auto requirements = device.getBufferMemoryRequirements(*buffer);
        attempt["memoryRequirementBytes"] = requirements.size;
        attempt["memoryRequirementAlignment"] = requirements.alignment;
        attempt["bufferMemoryTypeBits"] = requirements.memoryTypeBits;
        const auto requirement_validation = ValidateImportedHostAllocation(
            backing.pointer, backing.size, min_pointer_alignment, requirements.size);
        if (requirement_validation != ImportedHostAllocationValidation::Valid) {
            progress.Fail(ExternalHostMemoryProbeStage::MemoryRequirements,
                          ExternalHostMemoryProbeFailure::MemoryRequirementsInvalid);
            RecordProgress(attempt, progress);
            return;
        }
        if (!complete_stage(ExternalHostMemoryProbeStage::MemoryRequirements)) {
            return;
        }

        auto [host_result, host_properties] =
            device.getMemoryHostPointerPropertiesEXT(handle_type, backing.pointer);
        attempt["hostPointerQueryVkResult"] = vk::to_string(host_result);
        if (host_result != vk::Result::eSuccess) {
            progress.Fail(ExternalHostMemoryProbeStage::HostPointerProperties,
                          ExternalHostMemoryProbeFailure::HostPointerQueryFailed);
            RecordProgress(attempt, progress);
            return;
        }
        attempt["hostPointerMemoryTypeBits"] = host_properties.memoryTypeBits;
        if (!complete_stage(ExternalHostMemoryProbeStage::HostPointerProperties)) {
            return;
        }

        const auto physical_memory = physical_device.getMemoryProperties();
        std::vector<ImportedHostMemoryTypeProperties> memory_properties;
        memory_properties.reserve(physical_memory.memoryTypeCount);
        for (std::uint32_t index = 0; index < physical_memory.memoryTypeCount; ++index) {
            memory_properties.push_back({.host_coherent = static_cast<bool>(
                                             physical_memory.memoryTypes[index].propertyFlags &
                                             vk::MemoryPropertyFlagBits::eHostCoherent)});
        }
        const auto selection = SelectImportedHostMemoryType(
            requirements.memoryTypeBits, host_properties.memoryTypeBits, memory_properties, true);
        attempt["compatibleMemoryTypeBits"] = selection.compatible_bits;
        if (!selection.index) {
            progress.Fail(ExternalHostMemoryProbeStage::MemoryTypeSelection,
                          selection.failure ==
                                  ImportedHostMemoryTypeSelectionFailure::NoCoherentMemoryType
                              ? ExternalHostMemoryProbeFailure::NoCoherentMemoryType
                              : ExternalHostMemoryProbeFailure::NoCompatibleMemoryType);
            RecordProgress(attempt, progress);
            return;
        }
        const auto selected_flags = physical_memory.memoryTypes[*selection.index].propertyFlags;
        attempt["selectedMemoryTypeIndex"] = *selection.index;
        attempt["selectedMemoryTypeHostCoherent"] =
            static_cast<bool>(selected_flags & vk::MemoryPropertyFlagBits::eHostCoherent);
        attempt["selectedMemoryPropertyFlags"] = static_cast<std::uint32_t>(selected_flags);
        if (!complete_stage(ExternalHostMemoryProbeStage::MemoryTypeSelection)) {
            return;
        }

        const vk::MemoryAllocateFlagsInfo allocation_flags{
            .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
        };
        const vk::ImportMemoryHostPointerInfoEXT import_info{
            .pNext = &allocation_flags,
            .handleType = handle_type,
            .pHostPointer = backing.pointer,
        };
        const vk::MemoryAllocateInfo allocation_info{
            .pNext = &import_info,
            .allocationSize = requirements.size,
            .memoryTypeIndex = *selection.index,
        };
        auto [memory_result, memory] = device.allocateMemoryUnique(allocation_info);
        attempt["memoryAllocateVkResult"] = vk::to_string(memory_result);
        if (memory_result != vk::Result::eSuccess) {
            progress.Fail(ExternalHostMemoryProbeStage::MemoryAllocation,
                          ExternalHostMemoryProbeFailure::MemoryAllocationFailed);
            RecordProgress(attempt, progress);
            return;
        }
        if (!complete_stage(ExternalHostMemoryProbeStage::MemoryAllocation)) {
            return;
        }

        const auto bind_result = device.bindBufferMemory(*buffer, *memory, 0);
        attempt["bindVkResult"] = vk::to_string(bind_result);
        if (bind_result != vk::Result::eSuccess) {
            progress.Fail(ExternalHostMemoryProbeStage::MemoryBinding,
                          ExternalHostMemoryProbeFailure::MemoryBindingFailed);
            RecordProgress(attempt, progress);
            return;
        }
        if (!complete_stage(ExternalHostMemoryProbeStage::MemoryBinding)) {
            return;
        }

        const auto address =
            device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = *buffer});
        attempt["deviceAddressNonzero"] = address != 0;
        attempt["deviceAddressModuloAlignment"] =
            requirements.alignment == 0 ? 0 : address % requirements.alignment;
        if (address == 0) {
            progress.Fail(ExternalHostMemoryProbeStage::DeviceAddress,
                          ExternalHostMemoryProbeFailure::ZeroDeviceAddress);
            RecordProgress(attempt, progress);
            return;
        }
        if (!complete_stage(ExternalHostMemoryProbeStage::DeviceAddress)) {
            return;
        }

        ExternalHostMemoryImportOwner owner{std::move(memory), std::move(buffer), address};
        if (!owner.IsRetained()) {
            progress.Fail(ExternalHostMemoryProbeStage::Retained,
                          ExternalHostMemoryProbeFailure::ZeroDeviceAddress);
            RecordProgress(attempt, progress);
            return;
        }
        if (!complete_stage(ExternalHostMemoryProbeStage::Retained)) {
            return;
        }
        RecordProgress(attempt, progress);
    }();
    attempt["cleanupComplete"] = true;
    attempt["durationMs"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count();
    return attempt;
}

} // namespace

int main(int argc, char** argv) {
    const auto started = std::chrono::steady_clock::now();
    Json report = BaseReport();
    int exit_code = 70;
    try {
        exit_code = [&]() -> int {
            std::string argument_error;
            const auto arguments = ParseArguments(argc, argv, argument_error);
            if (!arguments) {
                report["reason"] = argument_error;
                std::cerr
                    << "Usage: external-host-memory-import-probe --vendor-id ID --device-id ID "
                       "--size-bytes 1..67108864\n";
                return 64;
            }
            report["backingSizeBytes"] = arguments->backing_size;
            report["requestedVendorId"] = arguments->vendor_id;
            report["requestedDeviceId"] = arguments->device_id;

            VULKAN_HPP_DEFAULT_DISPATCHER.init();
            const vk::ApplicationInfo application_info{
                .pApplicationName = "shadps4_external_host_memory_import_probe",
                .applicationVersion = 1,
                .pEngineName = "shadps4_probe",
                .engineVersion = 1,
                .apiVersion = vk::ApiVersion13,
            };
            auto [instance_result, instance] = vk::createInstanceUnique(
                vk::InstanceCreateInfo{.pApplicationInfo = &application_info});
            if (instance_result != vk::Result::eSuccess) {
                report["reason"] = "instance_creation_failed";
                report["vkResult"] = vk::to_string(instance_result);
                return 2;
            }
            VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

            auto [enumerate_result, devices] = instance->enumeratePhysicalDevices();
            if (enumerate_result != vk::Result::eSuccess) {
                report["reason"] = "device_enumeration_failed";
                report["vkResult"] = vk::to_string(enumerate_result);
                return 2;
            }
            std::vector<vk::PhysicalDevice> matches;
            for (const auto physical_device : devices) {
                const auto properties = physical_device.getProperties();
                if (properties.vendorID == arguments->vendor_id &&
                    properties.deviceID == arguments->device_id) {
                    matches.push_back(physical_device);
                }
            }
            if (matches.size() != 1) {
                report["matchingDeviceCount"] = matches.size();
                if (matches.empty()) {
                    report["status"] = "unsupported";
                    report["reason"] = "target_device_not_found";
                } else {
                    report["reason"] = "device_identity_ambiguous";
                }
                return 2;
            }
            const auto physical_device = matches.front();

            auto [extensions_result, extensions] =
                physical_device.enumerateDeviceExtensionProperties();
            if (extensions_result != vk::Result::eSuccess) {
                report["reason"] = "extension_enumeration_failed";
                report["vkResult"] = vk::to_string(extensions_result);
                return 2;
            }
            const bool has_external_host =
                std::ranges::any_of(extensions, [](const auto& extension) {
                    return std::string_view{extension.extensionName} ==
                           VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
                });
            report["externalHostExtensionAvailable"] = has_external_host;

            const auto identity_chain =
                physical_device
                    .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties,
                                    vk::PhysicalDeviceDriverProperties>();
            const auto& properties = identity_chain.get().properties;
            const auto& identity = identity_chain.get<vk::PhysicalDeviceIDProperties>();
            const auto& driver = identity_chain.get<vk::PhysicalDeviceDriverProperties>();
            report["device"] = {
                {"vendorId", properties.vendorID},
                {"deviceId", properties.deviceID},
                {"name", std::string{properties.deviceName.data()}},
                {"deviceUuid", HexBytes(identity.deviceUUID)},
                {"driverId", static_cast<std::uint32_t>(driver.driverID)},
                {"driverName", std::string{driver.driverName.data()}},
                {"driverInfo", std::string{driver.driverInfo.data()}},
                {"driverVersion", properties.driverVersion},
                {"apiVersion", properties.apiVersion},
            };
            if (!has_external_host) {
                report["status"] = "unsupported";
                report["reason"] = "external_host_extension_unavailable";
                return 2;
            }

            vk::PhysicalDeviceExternalMemoryHostPropertiesEXT host_properties{};
            vk::PhysicalDeviceProperties2 property_query{.pNext = &host_properties};
            physical_device.getProperties2(&property_query);
            const auto min_alignment =
                static_cast<std::size_t>(host_properties.minImportedHostPointerAlignment);
            report["minImportedHostPointerAlignment"] = min_alignment;

            const auto feature_chain =
                physical_device.getFeatures2<vk::PhysicalDeviceFeatures2,
                                             vk::PhysicalDeviceVulkan12Features>();
            const auto& features12 = feature_chain.get<vk::PhysicalDeviceVulkan12Features>();
            report["bufferDeviceAddressFeature"] = features12.bufferDeviceAddress == VK_TRUE;
            if (!features12.bufferDeviceAddress) {
                report["status"] = "unsupported";
                report["reason"] = "buffer_device_address_unavailable";
                return 2;
            }

            const auto queue_families = physical_device.getQueueFamilyProperties();
            const auto queue_it = std::ranges::find_if(queue_families, [](const auto& family) {
                return static_cast<bool>(family.queueFlags & (vk::QueueFlagBits::eGraphics |
                                                              vk::QueueFlagBits::eCompute));
            });
            if (queue_it == queue_families.end()) {
                report["status"] = "unsupported";
                report["reason"] = "queue_unavailable";
                return 2;
            }
            const auto queue_index =
                static_cast<std::uint32_t>(std::distance(queue_families.begin(), queue_it));
            constexpr float queue_priority = 1.0f;
            const vk::DeviceQueueCreateInfo queue_info{
                .queueFamilyIndex = queue_index,
                .queueCount = 1,
                .pQueuePriorities = &queue_priority,
            };
            const vk::PhysicalDeviceVulkan12Features enabled_features{
                .bufferDeviceAddress = VK_TRUE,
            };
            constexpr std::array enabled_extensions{VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME};
            const vk::DeviceCreateInfo device_info{
                .pNext = &enabled_features,
                .queueCreateInfoCount = 1,
                .pQueueCreateInfos = &queue_info,
                .enabledExtensionCount = static_cast<std::uint32_t>(enabled_extensions.size()),
                .ppEnabledExtensionNames = enabled_extensions.data(),
            };
            auto [device_result, device] = physical_device.createDeviceUnique(device_info);
            if (device_result != vk::Result::eSuccess) {
                report["reason"] = "logical_device_creation_failed";
                report["vkResult"] = vk::to_string(device_result);
                return 2;
            }
            VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
            report["externalHostEnabled"] = true;
            report["bufferDeviceAddressEnabled"] = true;

            report["attempts"].push_back(ProbeHandle(
                physical_device, *device, vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
                "host_allocation_ext", arguments->backing_size, min_alignment));
            report["attempts"].push_back(ProbeHandle(
                physical_device, *device,
                vk::ExternalMemoryHandleTypeFlagBits::eHostMappedForeignMemoryEXT,
                "host_mapped_foreign_memory_ext", arguments->backing_size, min_alignment));

            const bool any_success =
                std::ranges::any_of(report["attempts"], [](const auto& attempt) {
                    return attempt.at("succeeded").template get<bool>();
                });
            if (any_success) {
                report["status"] = "pass";
                report["reason"] = "at_least_one_handle_type_retained";
                return 0;
            }
            constexpr std::array unsupported_failures{
                "invalid_pointer_alignment",   "backing_validation_failed",
                "memory_requirements_invalid", "handle_type_not_importable",
                "host_pointer_query_failed",   "no_compatible_memory_type",
                "no_coherent_memory_type",     "memory_allocation_failed",
                "zero_device_address"};
            const bool all_unsupported =
                std::ranges::all_of(report["attempts"], [&](const auto& attempt) {
                    const auto failure = attempt.at("failure").template get<std::string>();
                    return std::ranges::find(unsupported_failures, failure) !=
                           unsupported_failures.end();
                });
            report["status"] = all_unsupported ? "unsupported" : "error";
            report["reason"] =
                all_unsupported ? "no_supported_host_import_path" : "probe_operation_failed";
            return 2;
        }();
    } catch (const std::exception& exception) {
        report["status"] = "error";
        report["reason"] = "unexpected_exception";
        report["exception"] = BoundedDiagnosticText(exception.what());
        exit_code = 70;
    }
    report["cleanupComplete"] = true;
    report["durationMs"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    try {
        std::cout << report.dump() << '\n';
    } catch (const std::exception& exception) {
        std::cerr << "Failed to serialize bounded probe result: " << exception.what() << '\n';
        return 70;
    }
    return exit_code;
}
