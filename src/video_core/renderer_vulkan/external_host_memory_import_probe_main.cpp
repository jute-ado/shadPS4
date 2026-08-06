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

#include "video_core/renderer_vulkan/exact_address_space_host_import_probe.h"
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
    std::uint64_t extra_dmem_mbytes{};
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

[[nodiscard]] constexpr std::string_view DispositionName(
    Vulkan::ExternalHostProbeDisposition disposition) {
    using enum Vulkan::ExternalHostProbeDisposition;
    switch (disposition) {
    case Pass:
        return "pass";
    case Unsupported:
        return "unsupported";
    case Error:
        return "error";
    }
    return "error";
}

[[nodiscard]] constexpr Vulkan::ExternalHostProbeVkResultClass ClassifyVkResult(vk::Result result) {
    using enum Vulkan::ExternalHostProbeVkResultClass;
    switch (result) {
    case vk::Result::eSuccess:
        return Success;
    case vk::Result::eErrorInvalidExternalHandle:
        return ErrorInvalidExternalHandle;
    case vk::Result::eErrorOutOfHostMemory:
    case vk::Result::eErrorOutOfDeviceMemory:
        return ErrorOutOfMemory;
    default:
        return ErrorUnknown;
    }
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
    std::uint64_t extra_dmem_mbytes{};
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
#ifdef SHADPS4_EXACT_ADDRESS_SPACE_PROBE
            error = "--size-bytes is not valid for the exact backing probe";
            return std::nullopt;
#else
            size = value;
#endif
        } else if (option == "--extra-dmem-mb") {
#ifdef SHADPS4_EXACT_ADDRESS_SPACE_PROBE
            extra_dmem_mbytes = *value;
#else
            error = "--extra-dmem-mb is only valid for the exact backing probe";
            return std::nullopt;
#endif
        } else {
            error = "unknown option";
            return std::nullopt;
        }
    }
#ifdef SHADPS4_EXACT_ADDRESS_SPACE_PROBE
    const auto exact_size = Vulkan::CalculateExactAddressSpaceBackingSize(extra_dmem_mbytes);
    if (!vendor || !device || !exact_size ||
        *vendor > std::numeric_limits<std::uint32_t>::max() ||
        *device > std::numeric_limits<std::uint32_t>::max() ||
        Vulkan::ValidateExactBackingSizeForSizeT(
            *exact_size, static_cast<std::uint64_t>(std::numeric_limits<SIZE_T>::max())) !=
            Vulkan::ExactBackingSizeValidation::Valid) {
        error = "vendor, device, and an optional 0..20000 MiB extra dmem are required";
        return std::nullopt;
    }
    return Arguments{.vendor_id = static_cast<std::uint32_t>(*vendor),
                     .device_id = static_cast<std::uint32_t>(*device),
                     .backing_size = *exact_size,
                     .extra_dmem_mbytes = extra_dmem_mbytes};
#else
    if (!vendor || !device || !size || *vendor > std::numeric_limits<std::uint32_t>::max() ||
        *device > std::numeric_limits<std::uint32_t>::max() || *size == 0 ||
        *size > MaxBackingSize) {
        error = "vendor, device, and a 1..67108864 byte backing are required";
        return std::nullopt;
    }
    return Arguments{.vendor_id = static_cast<std::uint32_t>(*vendor),
                     .device_id = static_cast<std::uint32_t>(*device),
                     .backing_size = *size};
#endif
}

[[nodiscard]] Json BaseReport() {
    return Json{
        {"schema", "shadps4.external-host-memory-import-probe.v1"},
#ifdef SHADPS4_EXACT_ADDRESS_SPACE_PROBE
        {"backingScope", "exact_windows_address_space_backing"},
#else
        {"backingScope", "bounded_synthetic"},
#endif
        {"backingSizeBytes", 0},
        {"mechanismEvidenceOnly", true},
        {"publicationAttempted", false},
        {"guestPagesPublished", 0},
        {"externalHostExtensionAvailable", false},
        {"externalHostEnabled", false},
        {"bufferDeviceAddressFeature", false},
        {"bufferDeviceAddressEnabled", false},
        {"cleanupAttempted", false},
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

struct PagefileCleanupState {
    Vulkan::ExternalHostProbeCleanupResult result{};
#ifdef _WIN32
    DWORD unmap_error{};
    DWORD close_error{};
#endif
};

struct PagefileBacking {
#ifdef _WIN32
    HANDLE mapping{};
#endif
    void* pointer{};
    std::uint64_t size{};
    PagefileCleanupState* cleanup{};

    PagefileBacking() = default;
    PagefileBacking(const PagefileBacking&) = delete;
    PagefileBacking& operator=(const PagefileBacking&) = delete;

    PagefileBacking(PagefileBacking&& other) noexcept
#ifdef _WIN32
        : mapping{std::exchange(other.mapping, nullptr)},
#endif
          pointer{std::exchange(other.pointer, nullptr)}, size{std::exchange(other.size, 0)},
          cleanup{std::exchange(other.cleanup, nullptr)} {
    }

    PagefileBacking& operator=(PagefileBacking&&) = delete;

    ~PagefileBacking() {
        Finalize();
    }

    void Finalize() noexcept {
#ifdef _WIN32
        if (pointer != nullptr) {
            cleanup->result.unmap_attempted = true;
            cleanup->result.unmap_succeeded = UnmapViewOfFile(pointer) != FALSE;
            if (!cleanup->result.unmap_succeeded) {
                cleanup->unmap_error = GetLastError();
            }
            pointer = nullptr;
        }
        if (mapping != nullptr) {
            cleanup->result.close_attempted = true;
            cleanup->result.close_succeeded = CloseHandle(mapping) != FALSE;
            if (!cleanup->result.close_succeeded) {
                cleanup->close_error = GetLastError();
            }
            mapping = nullptr;
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

[[nodiscard]] PagefileBacking AllocatePagefileBacking(std::uint64_t size,
                                                      PagefileCleanupState& cleanup) {
    PagefileBacking backing;
    backing.cleanup = &cleanup;
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
        {"backingProvenance", "windows_page_file_mapping"},
        {"handleType", handle_name},
        {"disposition", "error"},
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
        {"cleanupAttempted", false},
        {"unmapAttempted", false},
        {"unmapSucceeded", false},
        {"unmapWin32Error", 0},
        {"closeAttempted", false},
        {"closeSucceeded", false},
        {"closeWin32Error", 0},
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

struct HandleProbeOutput {
    Json report;
    Vulkan::ExternalHostProbeDisposition disposition;
};

[[nodiscard]] HandleProbeOutput ProbeHostAllocation(vk::PhysicalDevice physical_device,
                                                    vk::Device device,
                                                    std::uint64_t requested_backing_size,
                                                    std::size_t min_pointer_alignment) {
    using namespace Vulkan;

    constexpr auto provenance = ExternalHostBackingProvenance::PageFileMapping;
    static_assert(AllowedExternalHostHandleTypes(provenance) ==
                  ExternalHostHandleClass::HostAllocation);
    constexpr auto handle_type = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;

    const auto started = std::chrono::steady_clock::now();
    Json attempt = InitialAttempt("host_allocation_ext");
    PagefileCleanupState cleanup;
    ExternalHostMemoryProbeFailure final_failure = ExternalHostMemoryProbeFailure::UnexpectedStage;
    ExternalHostProbeVkResultClass relevant_vk_result = ExternalHostProbeVkResultClass::NotCalled;
    [&] {
        ExternalHostMemoryProbeProgress progress;
        const auto record_progress = [&] {
            RecordProgress(attempt, progress);
            final_failure = progress.Result().failure;
        };
        const auto complete_stage = [&](ExternalHostMemoryProbeStage stage) {
            if (progress.Complete(stage)) {
                return true;
            }
            record_progress();
            return false;
        };
        if (!complete_stage(ExternalHostMemoryProbeStage::Capability)) {
            return;
        }

        if (!std::has_single_bit(min_pointer_alignment) || min_pointer_alignment > MaxBackingSize) {
            progress.Fail(ExternalHostMemoryProbeStage::Backing,
                          ExternalHostMemoryProbeFailure::InvalidPointerAlignment);
            record_progress();
            return;
        }
        const auto remainder = requested_backing_size & (min_pointer_alignment - 1);
        const auto aligned_size = remainder == 0
                                      ? requested_backing_size
                                      : requested_backing_size + min_pointer_alignment - remainder;
        if (aligned_size < requested_backing_size || aligned_size > MaxBackingSize) {
            progress.Fail(ExternalHostMemoryProbeStage::Backing,
                          ExternalHostMemoryProbeFailure::BackingValidationFailed);
            record_progress();
            return;
        }

        auto backing = AllocatePagefileBacking(aligned_size, cleanup);
        if (!backing) {
            progress.Fail(ExternalHostMemoryProbeStage::Backing,
                          ExternalHostMemoryProbeFailure::BackingAllocationFailed);
            record_progress();
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
            record_progress();
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
            record_progress();
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
        relevant_vk_result = ClassifyVkResult(buffer_result);
        if (buffer_result != vk::Result::eSuccess) {
            progress.Fail(ExternalHostMemoryProbeStage::BufferCreation,
                          ExternalHostMemoryProbeFailure::BufferCreationFailed);
            record_progress();
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
            record_progress();
            return;
        }
        if (!complete_stage(ExternalHostMemoryProbeStage::MemoryRequirements)) {
            return;
        }

        auto [host_result, host_properties] =
            device.getMemoryHostPointerPropertiesEXT(handle_type, backing.pointer);
        attempt["hostPointerQueryVkResult"] = vk::to_string(host_result);
        relevant_vk_result = ClassifyVkResult(host_result);
        if (host_result != vk::Result::eSuccess) {
            progress.Fail(ExternalHostMemoryProbeStage::HostPointerProperties,
                          ExternalHostMemoryProbeFailure::HostPointerQueryFailed);
            record_progress();
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
            record_progress();
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
        relevant_vk_result = ClassifyVkResult(memory_result);
        if (memory_result != vk::Result::eSuccess) {
            progress.Fail(ExternalHostMemoryProbeStage::MemoryAllocation,
                          ExternalHostMemoryProbeFailure::MemoryAllocationFailed);
            record_progress();
            return;
        }
        if (!complete_stage(ExternalHostMemoryProbeStage::MemoryAllocation)) {
            return;
        }

        const auto bind_result = device.bindBufferMemory(*buffer, *memory, 0);
        attempt["bindVkResult"] = vk::to_string(bind_result);
        relevant_vk_result = ClassifyVkResult(bind_result);
        if (bind_result != vk::Result::eSuccess) {
            progress.Fail(ExternalHostMemoryProbeStage::MemoryBinding,
                          ExternalHostMemoryProbeFailure::MemoryBindingFailed);
            record_progress();
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
            record_progress();
            return;
        }
        if (!complete_stage(ExternalHostMemoryProbeStage::DeviceAddress)) {
            return;
        }

        ExternalHostMemoryImportOwner owner{std::move(memory), std::move(buffer), address};
        if (!owner.IsRetained()) {
            progress.Fail(ExternalHostMemoryProbeStage::Retained,
                          ExternalHostMemoryProbeFailure::ZeroDeviceAddress);
            record_progress();
            return;
        }
        if (!complete_stage(ExternalHostMemoryProbeStage::Retained)) {
            return;
        }
        record_progress();
    }();
    const auto disposition = ClassifyExternalHostProbeAttempt(attempt["succeeded"].get<bool>(),
                                                              final_failure, relevant_vk_result);
    attempt["disposition"] = DispositionName(disposition);
    attempt["cleanupAttempted"] = cleanup.result.unmap_attempted || cleanup.result.close_attempted;
    attempt["unmapAttempted"] = cleanup.result.unmap_attempted;
    attempt["unmapSucceeded"] = cleanup.result.unmap_succeeded;
    attempt["unmapWin32Error"] = cleanup.unmap_error;
    attempt["closeAttempted"] = cleanup.result.close_attempted;
    attempt["closeSucceeded"] = cleanup.result.close_succeeded;
    attempt["closeWin32Error"] = cleanup.close_error;
    attempt["cleanupComplete"] = cleanup.result.Complete();
    attempt["durationMs"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count();
    return {.report = std::move(attempt), .disposition = disposition};
}

#ifdef SHADPS4_EXACT_ADDRESS_SPACE_PROBE
struct ExactWindowsBackingAdapter {
    using Mapping = HANDLE;
    using Reservation = void*;

    HANDLE process{GetCurrentProcess()};
    DWORD last_error{};

    Mapping CreatePageFileMapping(std::uint64_t size,
                                  Vulkan::ExactWindowsBackingRecipe recipe) noexcept {
        if (!(recipe.create_file_mapping2 && recipe.invalid_page_file &&
              recipe.file_map_all_access && recipe.page_execute_readwrite && recipe.sec_commit)) {
            return nullptr;
        }
        auto mapping = CreateFileMapping2(INVALID_HANDLE_VALUE, nullptr, FILE_MAP_ALL_ACCESS,
                                          PAGE_EXECUTE_READWRITE, SEC_COMMIT, size, nullptr, nullptr,
                                          0);
        if (mapping == nullptr) {
            last_error = GetLastError();
        }
        return mapping;
    }

    Reservation ReservePlaceholder(std::uint64_t size,
                                   Vulkan::ExactWindowsBackingRecipe recipe) noexcept {
        if (!(recipe.virtual_alloc2 && recipe.mem_reserve && recipe.mem_reserve_placeholder &&
              recipe.page_noaccess)) {
            return nullptr;
        }
        auto reservation = VirtualAlloc2(process, nullptr, static_cast<SIZE_T>(size),
                                         MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS,
                                         nullptr, 0);
        if (reservation == nullptr) {
            last_error = GetLastError();
        }
        return reservation;
    }

    void* MapReplacingPlaceholder(Mapping mapping, Reservation reservation, std::uint64_t size,
                                  Vulkan::ExactWindowsBackingRecipe recipe) noexcept {
        if (!(recipe.map_view_of_file3 && recipe.file_map_all_access &&
              recipe.mem_replace_placeholder && recipe.page_execute_readwrite)) {
            return nullptr;
        }
        auto pointer = MapViewOfFile3(mapping, process, reservation, 0, static_cast<SIZE_T>(size),
                                      MEM_REPLACE_PLACEHOLDER, PAGE_EXECUTE_READWRITE, nullptr, 0);
        if (pointer == nullptr) {
            last_error = GetLastError();
        }
        return pointer;
    }

    bool ViewMatchesReservation(Reservation reservation, void* pointer) const noexcept {
        return pointer == reservation;
    }

    bool UnmapPreservingPlaceholder(void* pointer) noexcept {
        const bool succeeded =
            UnmapViewOfFile2(process, pointer, MEM_PRESERVE_PLACEHOLDER) != FALSE;
        if (!succeeded) {
            last_error = GetLastError();
        }
        return succeeded;
    }

    bool ReleasePlaceholder(Reservation reservation) noexcept {
        const bool succeeded = VirtualFreeEx(process, reservation, 0, MEM_RELEASE) != FALSE;
        if (!succeeded) {
            last_error = GetLastError();
        }
        return succeeded;
    }

    bool CloseMapping(Mapping mapping) noexcept {
        const bool succeeded = CloseHandle(mapping) != FALSE;
        if (!succeeded) {
            last_error = GetLastError();
        }
        return succeeded;
    }
};

[[nodiscard]] constexpr std::string_view ExactDispositionName(
    Vulkan::ExactHostImportDisposition disposition) {
    using enum Vulkan::ExactHostImportDisposition;
    switch (disposition) {
    case Pass:
        return "pass";
    case Unsupported:
        return "unsupported";
    case ResourceLimited:
        return "resource_limited";
    case ExactDesignIncompatible:
        return "exact_design_incompatible";
    case Error:
        return "error";
    }
    return "error";
}

struct ExactProbeOutput {
    Json report;
    Vulkan::ExactHostImportDisposition disposition;
};

[[nodiscard]] ExactProbeOutput ProbeExactAddressSpaceHostAllocation(
    vk::PhysicalDevice physical_device, vk::Device device, std::uint64_t backing_size,
    std::size_t min_pointer_alignment) {
    using namespace Vulkan;
    constexpr auto handle_type = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
    constexpr vk::BufferUsageFlags usage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

    Json attempt = InitialAttempt("host_allocation_ext");
    attempt["backingProvenance"] = "windows_page_file_mapping_exact_address_space_recipe";
    attempt["backingSizeBytes"] = backing_size;
    attempt["windowsBackingRecipe"] = {
        {"createFileMapping2",
         "INVALID_HANDLE_VALUE,FILE_MAP_ALL_ACCESS,PAGE_EXECUTE_READWRITE,SEC_COMMIT"},
        {"virtualAlloc2", "MEM_RESERVE|MEM_RESERVE_PLACEHOLDER,PAGE_NOACCESS"},
        {"mapViewOfFile3", "offset=0,MEM_REPLACE_PLACEHOLDER,PAGE_EXECUTE_READWRITE"},
        {"cleanup",
         "destroy_buffer,destroy_memory,UnmapViewOfFile2(MEM_PRESERVE_PLACEHOLDER),"
         "VirtualFreeEx(MEM_RELEASE),CloseHandle"},
    };
    ExactHostImportFailure failure = ExactHostImportFailure::ProtocolViolation;
    ExactHostImportProtocol protocol;
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    ExactWindowsBackingAdapter adapter;
    ExactWindowsBackingAcquisition<ExactWindowsBackingAdapter> acquisition;
    bool backing_acquired{};
    std::uint64_t address{};

    const auto fail = [&](ExactHostImportFailure value, std::string_view reason) {
        failure = value;
        attempt["failure"] = reason;
    };
    [&] {
        if (!protocol.Complete(ExactHostImportStage::Capability)) {
            return;
        }

        const auto property_chain =
            physical_device.getProperties2<vk::PhysicalDeviceProperties2,
                                           vk::PhysicalDeviceMaintenance3Properties,
                                           vk::PhysicalDeviceMaintenance4Properties>();
        const auto& properties = property_chain.get().properties;
        const auto max_allocation =
            property_chain.get<vk::PhysicalDeviceMaintenance3Properties>().maxMemoryAllocationSize;
        const auto max_buffer =
            property_chain.get<vk::PhysicalDeviceMaintenance4Properties>().maxBufferSize;
        attempt["maxBufferSize"] = max_buffer;
        attempt["maxMemoryAllocationSize"] = max_allocation;
        attempt["maxStorageBufferRange"] = properties.limits.maxStorageBufferRange;
        if (backing_size > max_buffer || backing_size > max_allocation) {
            fail(ExactHostImportFailure::DeviceLimitExceeded, "exact_backing_exceeds_device_limit");
            return;
        }
        if (!protocol.Complete(ExactHostImportStage::DeviceLimits)) {
            return;
        }

        const vk::PhysicalDeviceExternalBufferInfo external_info{
            .usage = usage,
            .handleType = handle_type,
        };
        const auto external = physical_device.getExternalBufferProperties(external_info)
                                  .externalMemoryProperties;
        attempt["compatibleHandleTypes"] =
            static_cast<std::uint32_t>(external.compatibleHandleTypes);
        attempt["importable"] = static_cast<bool>(
            external.externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eImportable);
        if (!attempt["importable"].get<bool>() ||
            !(external.compatibleHandleTypes & handle_type)) {
            fail(ExactHostImportFailure::HandleNotImportable, "handle_type_not_importable");
            return;
        }
        if (!protocol.Complete(ExactHostImportStage::ExternalBufferProperties)) {
            return;
        }

        const vk::ExternalMemoryBufferCreateInfo external_buffer{.handleTypes = handle_type};
        const vk::BufferCreateInfo buffer_info{
            .pNext = &external_buffer,
            .size = backing_size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };
        auto [buffer_result, created_buffer] = device.createBufferUnique(buffer_info);
        attempt["bufferCreateVkResult"] = vk::to_string(buffer_result);
        if (buffer_result != vk::Result::eSuccess) {
            fail(buffer_result == vk::Result::eErrorOutOfHostMemory ||
                         buffer_result == vk::Result::eErrorOutOfDeviceMemory
                     ? ExactHostImportFailure::VulkanOutOfMemory
                     : ExactHostImportFailure::VulkanCallFailed,
                 "buffer_creation_failed");
            return;
        }
        buffer = std::move(created_buffer);
        if (!protocol.Complete(ExactHostImportStage::BufferCreation)) {
            return;
        }

        const auto requirements = device.getBufferMemoryRequirements(*buffer);
        attempt["memoryRequirementBytes"] = requirements.size;
        attempt["memoryRequirementAlignment"] = requirements.alignment;
        attempt["bufferMemoryTypeBits"] = requirements.memoryTypeBits;
        if (requirements.size > backing_size) {
            fail(ExactHostImportFailure::RequirementExceedsBacking,
                 "memory_requirement_exceeds_exact_backing");
            return;
        }
        if (requirements.alignment == 0 || backing_size % requirements.alignment != 0 ||
            !std::has_single_bit(min_pointer_alignment) ||
            backing_size % min_pointer_alignment != 0) {
            fail(ExactHostImportFailure::RequirementAlignmentMismatch,
                 "exact_backing_alignment_mismatch");
            return;
        }
        if (!protocol.Complete(ExactHostImportStage::MemoryRequirements) ||
            !protocol.CanAllocateLargeBacking()) {
            return;
        }

        acquisition = AcquireExactWindowsAddressSpaceBacking(adapter, backing_size);
        attempt["win32Error"] = adapter.last_error;
        attempt["rollbackComplete"] = acquisition.rollback_complete;
        if (acquisition.failure != ExactWindowsBackingFailure::None) {
            fail(ExactHostImportFailure::BackingCommitFailed, "exact_backing_allocation_failed");
            return;
        }
        backing_acquired = true;
        attempt["pointerModuloAlignment"] =
            reinterpret_cast<std::uintptr_t>(acquisition.backing.pointer) % min_pointer_alignment;
        if (!protocol.Complete(ExactHostImportStage::Backing)) {
            return;
        }

        auto [host_result, host_properties] =
            device.getMemoryHostPointerPropertiesEXT(handle_type, acquisition.backing.pointer);
        attempt["hostPointerQueryVkResult"] = vk::to_string(host_result);
        if (host_result != vk::Result::eSuccess) {
            fail(host_result == vk::Result::eErrorInvalidExternalHandle
                     ? ExactHostImportFailure::HandleNotImportable
                     : ExactHostImportFailure::VulkanCallFailed,
                 "host_pointer_query_failed");
            return;
        }
        attempt["hostPointerMemoryTypeBits"] = host_properties.memoryTypeBits;
        if (!protocol.Complete(ExactHostImportStage::HostPointerProperties)) {
            return;
        }

        const auto physical_memory = physical_device.getMemoryProperties();
        std::vector<ImportedHostMemoryTypeProperties> memory_properties;
        for (std::uint32_t index = 0; index < physical_memory.memoryTypeCount; ++index) {
            memory_properties.push_back({.host_coherent = static_cast<bool>(
                                             physical_memory.memoryTypes[index].propertyFlags &
                                             vk::MemoryPropertyFlagBits::eHostCoherent)});
        }
        const auto selection = SelectImportedHostMemoryType(
            requirements.memoryTypeBits, host_properties.memoryTypeBits, memory_properties, true);
        attempt["compatibleMemoryTypeBits"] = selection.compatible_bits;
        if (!selection.index) {
            fail(selection.failure == ImportedHostMemoryTypeSelectionFailure::NoCoherentMemoryType
                     ? ExactHostImportFailure::NoCoherentMemoryType
                     : ExactHostImportFailure::NoCompatibleMemoryType,
                 "no_compatible_coherent_memory_type");
            return;
        }
        attempt["selectedMemoryTypeIndex"] = *selection.index;
        if (!protocol.Complete(ExactHostImportStage::MemoryTypeSelection)) {
            return;
        }

        const vk::MemoryAllocateFlagsInfo allocation_flags{
            .flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
        const vk::ImportMemoryHostPointerInfoEXT import_info{
            .pNext = &allocation_flags,
            .handleType = handle_type,
            .pHostPointer = acquisition.backing.pointer,
        };
        const vk::MemoryAllocateInfo allocation_info{
            .pNext = &import_info,
            .allocationSize = requirements.size,
            .memoryTypeIndex = *selection.index,
        };
        auto [memory_result, allocated_memory] = device.allocateMemoryUnique(allocation_info);
        attempt["memoryAllocateVkResult"] = vk::to_string(memory_result);
        if (memory_result != vk::Result::eSuccess) {
            fail(memory_result == vk::Result::eErrorOutOfHostMemory ||
                         memory_result == vk::Result::eErrorOutOfDeviceMemory
                     ? ExactHostImportFailure::VulkanOutOfMemory
                     : ExactHostImportFailure::VulkanCallFailed,
                 "memory_allocation_failed");
            return;
        }
        memory = std::move(allocated_memory);
        if (!protocol.Complete(ExactHostImportStage::MemoryAllocation)) {
            return;
        }

        const auto bind_result = device.bindBufferMemory(*buffer, *memory, 0);
        attempt["bindVkResult"] = vk::to_string(bind_result);
        if (bind_result != vk::Result::eSuccess) {
            fail(ExactHostImportFailure::VulkanCallFailed, "memory_binding_failed");
            return;
        }
        if (!protocol.Complete(ExactHostImportStage::MemoryBinding)) {
            return;
        }
        address = device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = *buffer});
        attempt["deviceAddressNonzero"] = address != 0;
        attempt["deviceAddressModuloAlignment"] =
            requirements.alignment == 0 ? 0 : address % requirements.alignment;
        if (address == 0) {
            fail(ExactHostImportFailure::ZeroDeviceAddress, "zero_device_address");
            return;
        }
        if (!protocol.Complete(ExactHostImportStage::DeviceAddress) ||
            !protocol.Complete(ExactHostImportStage::Retained)) {
            return;
        }
        failure = ExactHostImportFailure::None;
    }();

    // Vulkan objects must stop referring to the host pointer before its mapped view is released.
    buffer.reset();
    memory.reset();
    if (backing_acquired) {
        const auto cleanup =
            ReleaseExactWindowsAddressSpaceBacking(adapter, acquisition.backing, true);
        attempt["cleanupAttempted"] = true;
        attempt["unmapAttempted"] = true;
        attempt["unmapSucceeded"] = cleanup.unmap_succeeded;
        attempt["closeAttempted"] = true;
        attempt["closeSucceeded"] = cleanup.close_succeeded;
        attempt["placeholderReleaseSucceeded"] = cleanup.release_succeeded;
        attempt["cleanupComplete"] = cleanup.failure == ExactWindowsBackingFailure::None;
        attempt["win32CleanupError"] = adapter.last_error;
        if (cleanup.failure != ExactWindowsBackingFailure::None) {
            failure = ExactHostImportFailure::CleanupFailed;
            attempt["failure"] = "cleanup_failed";
        }
    } else {
        attempt["cleanupComplete"] = acquisition.rollback_complete;
    }
    const auto disposition = ClassifyExactHostImportFailure(failure);
    attempt["disposition"] = ExactDispositionName(disposition);
    attempt["succeeded"] = disposition == ExactHostImportDisposition::Pass;
    attempt["completedStage"] = static_cast<std::uint32_t>(protocol.CompletedStage());
    return {.report = std::move(attempt), .disposition = disposition};
}
#endif

} // namespace

int main(int argc, char** argv) {
    const auto started = std::chrono::steady_clock::now();
    Json report = BaseReport();
    int exit_code = 70;
    bool normal_scope_exit = false;
    try {
        exit_code = [&]() -> int {
            std::string argument_error;
            const auto arguments = ParseArguments(argc, argv, argument_error);
            if (!arguments) {
                report["reason"] = argument_error;
#ifdef SHADPS4_EXACT_ADDRESS_SPACE_PROBE
                std::cerr << "Usage: exact-address-space-host-import-probe --vendor-id ID "
                             "--device-id ID [--extra-dmem-mb 0..20000]\n";
#else
                std::cerr
                    << "Usage: external-host-memory-import-probe --vendor-id ID --device-id ID "
                       "--size-bytes 1..67108864\n";
#endif
                return 64;
            }
            report["backingSizeBytes"] = arguments->backing_size;
#ifdef SHADPS4_EXACT_ADDRESS_SPACE_PROBE
            report["extraDmemMbytes"] = arguments->extra_dmem_mbytes;
            report["exactProductionBackingRecipe"] = true;
#endif
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

#ifdef SHADPS4_EXACT_ADDRESS_SPACE_PROBE
            auto host_allocation = ProbeExactAddressSpaceHostAllocation(
                physical_device, *device, arguments->backing_size, min_alignment);
            report["attempts"].push_back(std::move(host_allocation.report));
            report["status"] = ExactDispositionName(host_allocation.disposition);
            if (host_allocation.disposition == Vulkan::ExactHostImportDisposition::Pass) {
                report["reason"] = "exact_address_space_host_allocation_retained";
                return 0;
            }
            report["reason"] = "exact_address_space_host_allocation_not_retained";
            return 2;
#else
            auto host_allocation = ProbeHostAllocation(physical_device, *device,
                                                       arguments->backing_size, min_alignment);
            report["attempts"].push_back(std::move(host_allocation.report));
            const std::array dispositions{host_allocation.disposition};
            const auto disposition = ClassifyExternalHostProbeResult(dispositions);
            report["status"] = DispositionName(disposition);
            if (disposition == Vulkan::ExternalHostProbeDisposition::Pass) {
                report["reason"] = "host_allocation_retained";
                return 0;
            }
            report["reason"] = disposition == Vulkan::ExternalHostProbeDisposition::Unsupported
                                   ? "host_allocation_incompatible"
                                   : "host_allocation_probe_error";
            return 2;
#endif
        }();
        normal_scope_exit = true;
    } catch (const std::exception& exception) {
        report["status"] = "error";
        report["reason"] = "unexpected_exception";
        report["exception"] = BoundedDiagnosticText(exception.what());
        exit_code = 70;
    }
    bool cleanup_attempted = false;
    bool cleanup_complete = normal_scope_exit;
    for (const auto& attempt : report["attempts"]) {
        cleanup_attempted |= attempt.at("cleanupAttempted").get<bool>();
        cleanup_complete &= attempt.at("cleanupComplete").get<bool>();
    }
    report["cleanupAttempted"] = cleanup_attempted;
    report["cleanupComplete"] = cleanup_complete;
    if (!cleanup_complete) {
        report["status"] = "error";
        report["reason"] = "cleanup_failed";
        exit_code = 70;
    }
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
