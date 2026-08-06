// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

#ifdef SHADPS4_EXACT_ADDRESS_SPACE_PROBE
#include <sirit/sirit.h>
#endif

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
    if (!vendor || !device || !exact_size || *vendor > std::numeric_limits<std::uint32_t>::max() ||
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
        {"dataVerificationAttempted", false},
        {"dataVerificationPassed", false},
        {"dataVerificationFailure", "not_started"},
        {"dataVerificationTrials", Json::array()},
        {"cleanupAttempted", false},
        {"unmapAttempted", false},
        {"unmapSucceeded", false},
        {"unmapWin32Error", 0},
        {"closeAttempted", false},
        {"closeSucceeded", false},
        {"closeWin32Error", 0},
        {"placeholderReleaseAttempted", false},
        {"placeholderReleaseSucceeded", false},
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
        auto mapping =
            CreateFileMapping2(INVALID_HANDLE_VALUE, nullptr, FILE_MAP_ALL_ACCESS,
                               PAGE_EXECUTE_READWRITE, SEC_COMMIT, size, nullptr, nullptr, 0);
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
        auto reservation =
            VirtualAlloc2(process, nullptr, static_cast<SIZE_T>(size),
                          MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
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

static_assert(ERROR_NOT_ENOUGH_MEMORY == 8);
static_assert(ERROR_OUTOFMEMORY == 14);
static_assert(ERROR_COMMITMENT_LIMIT == 1455);

[[nodiscard]] constexpr Vulkan::ExactVulkanFailureClass ClassifyExactVulkanResult(
    vk::Result result) noexcept {
    switch (result) {
    case vk::Result::eErrorInvalidExternalHandle:
        return Vulkan::ExactVulkanFailureClass::InvalidExternalHandle;
    case vk::Result::eErrorOutOfHostMemory:
    case vk::Result::eErrorOutOfDeviceMemory:
        return Vulkan::ExactVulkanFailureClass::OutOfMemory;
    default:
        return Vulkan::ExactVulkanFailureClass::Other;
    }
}

constexpr std::size_t ExactVisibilityWordCount = 64;
constexpr vk::DeviceSize ExactVisibilityByteCount =
    ExactVisibilityWordCount * sizeof(std::uint32_t);
constexpr vk::DeviceSize ExactVisibilityOutputByteCount = ExactVisibilityByteCount * 3;
constexpr std::size_t ExactVisibilityPageTableEntryCount = 8;
constexpr std::uint64_t ExactVisibilityGuestPage = 3;
constexpr std::uint64_t ExactVisibilityInPageOffset = 0x100;

class ExactHostVisibilityShader final : public Sirit::Module {
public:
    ExactHostVisibilityShader() : Sirit::Module{0x00010500} {
        AddCapability(spv::Capability::Shader);
        AddCapability(spv::Capability::Int64);
        AddCapability(spv::Capability::PhysicalStorageBufferAddresses);
        AddExtension("SPV_KHR_physical_storage_buffer");
        SetMemoryModel(spv::AddressingModel::PhysicalStorageBuffer64, spv::MemoryModel::GLSL450);

        const auto void_type = TypeVoid();
        const auto u32_type = TypeUInt(32);
        const auto u64_type = TypeUInt(64);
        const auto u32x3_type = TypeVector(u32_type, 3);
        const auto zero = Constant(u32_type, 0u);
        const auto one = Constant(u32_type, 1u);
        const auto bda_output_base = Constant(u32_type, ExactVisibilityWordCount);
        const auto page_table_output_base = Constant(u32_type, ExactVisibilityWordCount * 2);

        const auto storage_array_type = TypeRuntimeArray(u32_type);
        Decorate(storage_array_type, spv::Decoration::ArrayStride, sizeof(std::uint32_t));
        const auto storage_block_type = TypeStruct(storage_array_type);
        Decorate(storage_block_type, spv::Decoration::Block);
        MemberDecorate(storage_block_type, 0u, spv::Decoration::Offset, 0u);
        const auto storage_block_pointer =
            TypePointer(spv::StorageClass::StorageBuffer, storage_block_type);
        const auto storage_u32_pointer = TypePointer(spv::StorageClass::StorageBuffer, u32_type);

        const auto descriptor_input =
            AddGlobalVariable(storage_block_pointer, spv::StorageClass::StorageBuffer);
        Decorate(descriptor_input, spv::Decoration::DescriptorSet, 0u);
        Decorate(descriptor_input, spv::Decoration::Binding, 0u);
        const auto output =
            AddGlobalVariable(storage_block_pointer, spv::StorageClass::StorageBuffer);
        Decorate(output, spv::Decoration::DescriptorSet, 0u);
        Decorate(output, spv::Decoration::Binding, 1u);

        const auto page_table_array_type = TypeRuntimeArray(u64_type);
        Decorate(page_table_array_type, spv::Decoration::ArrayStride, sizeof(std::uint64_t));
        const auto page_table_block_type = TypeStruct(page_table_array_type);
        Decorate(page_table_block_type, spv::Decoration::Block);
        MemberDecorate(page_table_block_type, 0u, spv::Decoration::Offset, 0u);
        const auto page_table_block_pointer =
            TypePointer(spv::StorageClass::StorageBuffer, page_table_block_type);
        const auto storage_u64_pointer = TypePointer(spv::StorageClass::StorageBuffer, u64_type);
        const auto page_table =
            AddGlobalVariable(page_table_block_pointer, spv::StorageClass::StorageBuffer);
        Decorate(page_table, spv::Decoration::DescriptorSet, 0u);
        Decorate(page_table, spv::Decoration::Binding, 2u);

        const auto push_block_type = TypeStruct(u64_type, u64_type);
        Decorate(push_block_type, spv::Decoration::Block);
        MemberDecorate(push_block_type, 0u, spv::Decoration::Offset, 0u);
        MemberDecorate(push_block_type, 1u, spv::Decoration::Offset, sizeof(std::uint64_t));
        const auto push_block_pointer =
            TypePointer(spv::StorageClass::PushConstant, push_block_type);
        const auto push_u64_pointer = TypePointer(spv::StorageClass::PushConstant, u64_type);
        const auto push = AddGlobalVariable(push_block_pointer, spv::StorageClass::PushConstant);

        const auto input_u32x3_pointer = TypePointer(spv::StorageClass::Input, u32x3_type);
        const auto global_invocation_id =
            AddGlobalVariable(input_u32x3_pointer, spv::StorageClass::Input);
        Decorate(global_invocation_id, spv::Decoration::BuiltIn, spv::BuiltIn::GlobalInvocationId);

        const auto physical_block_pointer =
            TypePointer(spv::StorageClass::PhysicalStorageBuffer, storage_block_type);
        const auto physical_u32_pointer =
            TypePointer(spv::StorageClass::PhysicalStorageBuffer, u32_type);

        const auto main_function =
            OpFunction(void_type, spv::FunctionControlMask::MaskNone, TypeFunction(void_type));
        AddLabel();
        const auto invocation =
            OpCompositeExtract(u32_type, OpLoad(u32x3_type, global_invocation_id), 0u);
        const auto descriptor_pointer =
            OpAccessChain(storage_u32_pointer, descriptor_input, zero, invocation);
        const auto descriptor_value = OpLoad(u32_type, descriptor_pointer);
        const auto descriptor_output_pointer =
            OpAccessChain(storage_u32_pointer, output, zero, invocation);
        OpStore(descriptor_output_pointer, descriptor_value);

        const auto address_pointer = OpAccessChain(push_u64_pointer, push, zero);
        const auto address = OpLoad(u64_type, address_pointer);
        const auto physical_block = OpConvertUToPtr(physical_block_pointer, address);
        const auto physical_pointer =
            OpAccessChain(physical_u32_pointer, physical_block, zero, invocation);
        const auto physical_value =
            OpLoad(u32_type, physical_pointer, spv::MemoryAccessMask::Aligned, 4u);
        const auto physical_output_index = OpIAdd(u32_type, bda_output_base, invocation);
        const auto physical_output_pointer =
            OpAccessChain(storage_u32_pointer, output, zero, physical_output_index);
        OpStore(physical_output_pointer, physical_value);

        const auto guest_address_pointer = OpAccessChain(push_u64_pointer, push, one);
        const auto guest_address = OpLoad(u64_type, guest_address_pointer);
        const auto guest_page =
            OpShiftRightLogical(u64_type, guest_address, Constant(u64_type, 14ull));
        const auto guest_page32 = OpUConvert(u32_type, guest_page);
        const auto page_table_pointer =
            OpAccessChain(storage_u64_pointer, page_table, zero, guest_page32);
        const auto page_base = OpLoad(u64_type, page_table_pointer);
        const auto guest_page_offset =
            OpBitwiseAnd(u64_type, guest_address, Constant(u64_type, 0x3fffull));
        const auto resolved_address = OpIAdd(u64_type, page_base, guest_page_offset);
        const auto resolved_block = OpConvertUToPtr(physical_block_pointer, resolved_address);
        const auto resolved_pointer =
            OpAccessChain(physical_u32_pointer, resolved_block, zero, invocation);
        const auto resolved_value =
            OpLoad(u32_type, resolved_pointer, spv::MemoryAccessMask::Aligned, 4u);
        const auto resolved_output_index = OpIAdd(u32_type, page_table_output_base, invocation);
        const auto resolved_output_pointer =
            OpAccessChain(storage_u32_pointer, output, zero, resolved_output_index);
        OpStore(resolved_output_pointer, resolved_value);
        OpReturn();
        OpFunctionEnd();

        AddExecutionMode(main_function, spv::ExecutionMode::LocalSize,
                         static_cast<std::uint32_t>(ExactVisibilityWordCount), 1u, 1u);
        AddEntryPoint(spv::ExecutionModel::GLCompute, main_function, "main", global_invocation_id,
                      descriptor_input, output, page_table, push);
    }
};

struct ExactVisibilityProbeOutput {
    bool succeeded{};
    std::string failure{"not_started"};
    Json trials = Json::array();
};

struct ExactMappedMemoryGuard {
    vk::Device device;
    vk::DeviceMemory memory;
    void* pointer{};

    ~ExactMappedMemoryGuard() {
        if (pointer != nullptr) {
            device.unmapMemory(memory);
        }
    }
};

struct ExactGuestAliasView {
    HANDLE process{GetCurrentProcess()};
    void* base{};
    std::byte* write_pointer{};

    ExactGuestAliasView() = default;
    ExactGuestAliasView(const ExactGuestAliasView&) = delete;
    ExactGuestAliasView& operator=(const ExactGuestAliasView&) = delete;
    ExactGuestAliasView(ExactGuestAliasView&& other) noexcept
        : process{other.process}, base{std::exchange(other.base, nullptr)},
          write_pointer{std::exchange(other.write_pointer, nullptr)} {}
    ExactGuestAliasView& operator=(ExactGuestAliasView&&) = delete;

    ~ExactGuestAliasView() {
        if (base != nullptr) {
            UnmapViewOfFile2(process, base, MEM_PRESERVE_PLACEHOLDER);
            VirtualFreeEx(process, base, 0, MEM_RELEASE);
        }
    }
};

[[nodiscard]] std::optional<ExactGuestAliasView> MapExactGuestAlias(HANDLE mapping,
                                                                    std::uint64_t offset,
                                                                    std::size_t byte_count) {
    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const auto granularity = static_cast<std::uint64_t>(system_info.dwAllocationGranularity);
    if (granularity == 0 || offset > std::numeric_limits<std::uint64_t>::max() - byte_count) {
        return std::nullopt;
    }
    const auto view_offset = offset - offset % granularity;
    const auto delta = offset - view_offset;
    const auto required = delta + byte_count;
    const auto view_size = ((required + granularity - 1) / granularity) * granularity;
    auto reservation =
        VirtualAlloc2(GetCurrentProcess(), nullptr, view_size,
                      MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
    if (reservation == nullptr) {
        return std::nullopt;
    }
    auto mapped = MapViewOfFile3(mapping, GetCurrentProcess(), reservation, view_offset, view_size,
                                 MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0);
    if (mapped != reservation) {
        if (mapped != nullptr) {
            UnmapViewOfFile2(GetCurrentProcess(), mapped, MEM_PRESERVE_PLACEHOLDER);
        }
        VirtualFreeEx(GetCurrentProcess(), reservation, 0, MEM_RELEASE);
        return std::nullopt;
    }
    ExactGuestAliasView view;
    view.base = mapped;
    view.write_pointer = static_cast<std::byte*>(mapped) + delta;
    return view;
}

[[nodiscard]] std::optional<std::uint32_t> FindExactVisibilityMemoryType(
    vk::PhysicalDevice physical_device, std::uint32_t allowed, vk::MemoryPropertyFlags required) {
    const auto properties = physical_device.getMemoryProperties();
    for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if ((allowed & (std::uint32_t{1} << index)) != 0 &&
            (properties.memoryTypes[index].propertyFlags & required) == required) {
            return index;
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::uint64_t ExactVisibilityHash(std::span<const std::uint32_t> words) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto word : words) {
        for (std::uint32_t shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(word >> shift);
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

[[nodiscard]] ExactVisibilityProbeOutput VerifyExactHostVisibility(
    vk::PhysicalDevice physical_device, vk::Device device, std::uint32_t queue_family_index,
    vk::Buffer imported_buffer, vk::DeviceAddress imported_address, void* host_pointer,
    HANDLE backing_mapping, std::uint64_t backing_size) {
    ExactVisibilityProbeOutput result;
    const auto fail = [&](std::string_view reason) {
        result.failure = reason;
        return result;
    };
    if (backing_size < (4ull << 30) + 0x4000 + ExactVisibilityByteCount) {
        return fail("backing_too_small_for_high_offset_verification");
    }

    const vk::BufferCreateInfo output_buffer_info{
        .size = ExactVisibilityOutputByteCount,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    auto [output_buffer_result, created_output_buffer] =
        device.createBufferUnique(output_buffer_info);
    if (output_buffer_result != vk::Result::eSuccess) {
        return fail("output_buffer_creation_failed");
    }
    vk::UniqueDeviceMemory output_memory;
    vk::UniqueBuffer output_buffer = std::move(created_output_buffer);
    const auto output_requirements = device.getBufferMemoryRequirements(*output_buffer);
    const auto output_memory_type = FindExactVisibilityMemoryType(
        physical_device, output_requirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!output_memory_type) {
        return fail("coherent_output_memory_unavailable");
    }
    auto [output_memory_result, allocated_output_memory] = device.allocateMemoryUnique({
        .allocationSize = output_requirements.size,
        .memoryTypeIndex = *output_memory_type,
    });
    if (output_memory_result != vk::Result::eSuccess) {
        return fail("output_memory_allocation_failed");
    }
    output_memory = std::move(allocated_output_memory);
    if (device.bindBufferMemory(*output_buffer, *output_memory, 0) != vk::Result::eSuccess) {
        return fail("output_memory_binding_failed");
    }
    auto [map_result, mapped] = device.mapMemory(*output_memory, 0, ExactVisibilityOutputByteCount);
    if (map_result != vk::Result::eSuccess || mapped == nullptr) {
        return fail("output_memory_mapping_failed");
    }
    ExactMappedMemoryGuard mapped_guard{device, *output_memory, mapped};

    constexpr vk::DeviceSize page_table_bytes =
        ExactVisibilityPageTableEntryCount * sizeof(vk::DeviceAddress);
    auto [page_table_buffer_result, created_page_table_buffer] = device.createBufferUnique({
        .size = page_table_bytes,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
        .sharingMode = vk::SharingMode::eExclusive,
    });
    if (page_table_buffer_result != vk::Result::eSuccess) {
        return fail("page_table_buffer_creation_failed");
    }
    vk::UniqueDeviceMemory page_table_memory;
    vk::UniqueBuffer page_table_buffer = std::move(created_page_table_buffer);
    const auto page_table_requirements = device.getBufferMemoryRequirements(*page_table_buffer);
    const auto page_table_memory_type = FindExactVisibilityMemoryType(
        physical_device, page_table_requirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!page_table_memory_type) {
        return fail("coherent_page_table_memory_unavailable");
    }
    auto [page_table_memory_result, allocated_page_table_memory] = device.allocateMemoryUnique({
        .allocationSize = page_table_requirements.size,
        .memoryTypeIndex = *page_table_memory_type,
    });
    if (page_table_memory_result != vk::Result::eSuccess) {
        return fail("page_table_memory_allocation_failed");
    }
    page_table_memory = std::move(allocated_page_table_memory);
    if (device.bindBufferMemory(*page_table_buffer, *page_table_memory, 0) !=
        vk::Result::eSuccess) {
        return fail("page_table_memory_binding_failed");
    }
    auto [page_table_map_result, page_table_mapped] =
        device.mapMemory(*page_table_memory, 0, page_table_bytes);
    if (page_table_map_result != vk::Result::eSuccess || page_table_mapped == nullptr) {
        return fail("page_table_memory_mapping_failed");
    }
    ExactMappedMemoryGuard page_table_mapped_guard{device, *page_table_memory, page_table_mapped};

    constexpr vk::DeviceSize cache_page_bytes = 0x4000;
    auto [cache_buffer_result, created_cache_buffer] = device.createBufferUnique({
        .size = cache_page_bytes,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                 vk::BufferUsageFlagBits::eShaderDeviceAddress |
                 vk::BufferUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
    });
    if (cache_buffer_result != vk::Result::eSuccess) {
        return fail("cache_buffer_creation_failed");
    }
    vk::UniqueDeviceMemory cache_memory;
    vk::UniqueBuffer cache_buffer = std::move(created_cache_buffer);
    const auto cache_requirements = device.getBufferMemoryRequirements(*cache_buffer);
    const auto cache_memory_type =
        FindExactVisibilityMemoryType(physical_device, cache_requirements.memoryTypeBits,
                                      vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (!cache_memory_type) {
        return fail("device_local_cache_memory_unavailable");
    }
    const vk::MemoryAllocateFlagsInfo cache_allocation_flags{
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    auto [cache_memory_result, allocated_cache_memory] = device.allocateMemoryUnique({
        .pNext = &cache_allocation_flags,
        .allocationSize = cache_requirements.size,
        .memoryTypeIndex = *cache_memory_type,
    });
    if (cache_memory_result != vk::Result::eSuccess) {
        return fail("cache_memory_allocation_failed");
    }
    cache_memory = std::move(allocated_cache_memory);
    if (device.bindBufferMemory(*cache_buffer, *cache_memory, 0) != vk::Result::eSuccess) {
        return fail("cache_memory_binding_failed");
    }
    const auto cache_address =
        device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = *cache_buffer});
    if (cache_address == 0) {
        return fail("cache_device_address_zero");
    }

    ExactHostVisibilityShader shader;
    const auto shader_code = shader.Assemble();
    auto [shader_result, shader_module] = device.createShaderModuleUnique({
        .codeSize = shader_code.size() * sizeof(std::uint32_t),
        .pCode = shader_code.data(),
    });
    if (shader_result != vk::Result::eSuccess) {
        return fail("shader_module_creation_failed");
    }

    const std::array layout_bindings{
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    };
    auto [layout_result, descriptor_layout] = device.createDescriptorSetLayoutUnique({
        .bindingCount = static_cast<std::uint32_t>(layout_bindings.size()),
        .pBindings = layout_bindings.data(),
    });
    if (layout_result != vk::Result::eSuccess) {
        return fail("descriptor_layout_creation_failed");
    }
    const vk::DescriptorSetLayout descriptor_layout_handle = *descriptor_layout;
    const vk::PushConstantRange push_range{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(vk::DeviceAddress) * 2,
    };
    auto [pipeline_layout_result, pipeline_layout] = device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_layout_handle,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    });
    if (pipeline_layout_result != vk::Result::eSuccess) {
        return fail("pipeline_layout_creation_failed");
    }
    const vk::PipelineShaderStageCreateInfo stage{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = *shader_module,
        .pName = "main",
    };
    auto [pipeline_result, pipeline] =
        device.createComputePipelineUnique({}, {.stage = stage, .layout = *pipeline_layout});
    if (pipeline_result != vk::Result::eSuccess) {
        return fail("compute_pipeline_creation_failed");
    }

    const vk::DescriptorPoolSize pool_size{
        .type = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 3,
    };
    auto [pool_result, descriptor_pool] = device.createDescriptorPoolUnique({
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    });
    if (pool_result != vk::Result::eSuccess) {
        return fail("descriptor_pool_creation_failed");
    }
    auto [set_result, descriptor_sets] = device.allocateDescriptorSetsUnique({
        .descriptorPool = *descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_layout_handle,
    });
    if (set_result != vk::Result::eSuccess || descriptor_sets.size() != 1) {
        return fail("descriptor_set_allocation_failed");
    }

    auto [command_pool_result, command_pool] = device.createCommandPoolUnique({
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queue_family_index,
    });
    if (command_pool_result != vk::Result::eSuccess) {
        return fail("command_pool_creation_failed");
    }
    auto [command_result, command_buffers] = device.allocateCommandBuffersUnique({
        .commandPool = *command_pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    });
    if (command_result != vk::Result::eSuccess || command_buffers.size() != 1) {
        return fail("command_buffer_allocation_failed");
    }
    auto [fence_result, fence] = device.createFenceUnique({});
    if (fence_result != vk::Result::eSuccess) {
        return fail("fence_creation_failed");
    }
    const auto queue = device.getQueue(queue_family_index, 0);

    const std::array<std::uint64_t, 3> offsets{
        0x4000ull,
        (4ull << 30) + 0x4000ull,
        backing_size - 0x4000ull,
    };
    bool canonical_view_verified{};
    bool guest_alias_verified{};
    bool guest_page_table_verified{};
    for (const bool use_guest_alias : {false, true}) {
        for (std::size_t offset_index = 0; offset_index < offsets.size(); ++offset_index) {
            const auto page_offset = offsets[offset_index];
            const auto source_offset = page_offset + ExactVisibilityInPageOffset;
            constexpr std::uint64_t guest_address =
                (ExactVisibilityGuestPage << 14) + ExactVisibilityInPageOffset;
            const auto trial_index = offset_index + (use_guest_alias ? offsets.size() : 0);
            std::array<std::uint32_t, ExactVisibilityWordCount> expected{};
            for (std::size_t index = 0; index < expected.size(); ++index) {
                expected[index] = 0xa5c30000u ^
                                  (static_cast<std::uint32_t>(trial_index) * 0x01010101u) ^
                                  (static_cast<std::uint32_t>(index) * 0x9e3779b9u);
            }
            std::optional<ExactGuestAliasView> guest_alias;
            std::byte* write_pointer = static_cast<std::byte*>(host_pointer) + source_offset;
            if (use_guest_alias) {
                auto mapped_alias =
                    MapExactGuestAlias(backing_mapping, source_offset, ExactVisibilityByteCount);
                if (!mapped_alias) {
                    return fail("guest_alias_mapping_failed");
                }
                guest_alias.emplace(std::move(*mapped_alias));
                write_pointer = guest_alias->write_pointer;
            }
            std::memcpy(write_pointer, expected.data(), ExactVisibilityByteCount);
            std::memset(mapped, 0, ExactVisibilityOutputByteCount);
            std::memset(page_table_mapped, 0, page_table_bytes);
            static_cast<vk::DeviceAddress*>(page_table_mapped)[ExactVisibilityGuestPage] =
                imported_address + page_offset;

            const vk::DescriptorBufferInfo input_info{
                .buffer = imported_buffer,
                .offset = source_offset,
                .range = ExactVisibilityByteCount,
            };
            const vk::DescriptorBufferInfo output_info{
                .buffer = *output_buffer,
                .offset = 0,
                .range = ExactVisibilityOutputByteCount,
            };
            const vk::DescriptorBufferInfo page_table_info{
                .buffer = *page_table_buffer,
                .offset = 0,
                .range = page_table_bytes,
            };
            const std::array writes{
                vk::WriteDescriptorSet{
                    .dstSet = *descriptor_sets.front(),
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &input_info,
                },
                vk::WriteDescriptorSet{
                    .dstSet = *descriptor_sets.front(),
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &output_info,
                },
                vk::WriteDescriptorSet{
                    .dstSet = *descriptor_sets.front(),
                    .dstBinding = 2,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &page_table_info,
                },
            };
            device.updateDescriptorSets(writes, {});

            auto& command_buffer = command_buffers.front();
            if (command_buffer->reset({}) != vk::Result::eSuccess ||
                command_buffer->begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}) !=
                    vk::Result::eSuccess) {
                return fail("command_buffer_begin_failed");
            }
            const vk::MemoryBarrier host_to_shader{
                .srcAccessMask = vk::AccessFlagBits::eHostWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            };
            command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                                            vk::PipelineStageFlagBits::eComputeShader, {},
                                            host_to_shader, {}, {});
            command_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
            command_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipeline_layout, 0,
                                               *descriptor_sets.front(), {});
            const std::array<vk::DeviceAddress, 2> push_data{
                imported_address + source_offset,
                guest_address,
            };
            command_buffer->pushConstants(*pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                          sizeof(push_data), push_data.data());
            command_buffer->dispatch(1, 1, 1);
            const vk::MemoryBarrier shader_to_host{
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                .dstAccessMask = vk::AccessFlagBits::eHostRead,
            };
            command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                            vk::PipelineStageFlagBits::eHost, {}, shader_to_host,
                                            {}, {});
            if (command_buffer->end() != vk::Result::eSuccess) {
                return fail("command_buffer_end_failed");
            }
            const vk::CommandBuffer submitted = *command_buffer;
            const vk::SubmitInfo submit{
                .commandBufferCount = 1,
                .pCommandBuffers = &submitted,
            };
            if (device.resetFences(*fence) != vk::Result::eSuccess ||
                queue.submit(submit, *fence) != vk::Result::eSuccess) {
                return fail("queue_submission_failed");
            }
            const auto wait_result = device.waitForFences(*fence, true, 10'000'000'000ull);
            if (wait_result != vk::Result::eSuccess) {
                if (Vulkan::RequiresExactProbeIdleBeforeResourceCleanup(
                        false, wait_result == vk::Result::eErrorDeviceLost)) {
                    const auto idle_result = device.waitIdle();
                    if (!Vulkan::CanUseOrdinaryExactProbeResourceCleanup(
                            idle_result == vk::Result::eSuccess,
                            idle_result == vk::Result::eErrorDeviceLost)) {
                        std::cerr << "{\"schema\":\"shadps4.exact-probe-fatal.v1\","
                                     "\"status\":\"fatal\","
                                     "\"reason\":\"unsafe_resource_cleanup_prevented\"}\n";
                        std::cerr.flush();
                        std::_Exit(EXIT_FAILURE);
                    }
                }
                return fail("queue_fence_wait_failed");
            }

            const auto actual =
                std::span{static_cast<const std::uint32_t*>(mapped), ExactVisibilityWordCount * 3};
            std::size_t descriptor_mismatches{};
            std::size_t bda_mismatches{};
            std::size_t page_table_mismatches{};
            std::optional<std::size_t> descriptor_first_mismatch;
            std::optional<std::size_t> bda_first_mismatch;
            std::optional<std::size_t> page_table_first_mismatch;
            for (std::size_t index = 0; index < expected.size(); ++index) {
                if (actual[index] != expected[index]) {
                    descriptor_first_mismatch = descriptor_first_mismatch.value_or(index);
                    ++descriptor_mismatches;
                }
                if (actual[ExactVisibilityWordCount + index] != expected[index]) {
                    bda_first_mismatch = bda_first_mismatch.value_or(index);
                    ++bda_mismatches;
                }
                if (actual[ExactVisibilityWordCount * 2 + index] != expected[index]) {
                    page_table_first_mismatch = page_table_first_mismatch.value_or(index);
                    ++page_table_mismatches;
                }
            }
            result.trials.push_back({
                {"physicalPageOffset", page_offset},
                {"sourceOffset", source_offset},
                {"guestAddress", guest_address},
                {"writeSource", use_guest_alias ? "guest_alias_view" : "canonical_view"},
                {"expectedHash", ExactVisibilityHash(expected)},
                {"descriptorHash", ExactVisibilityHash(actual.first(ExactVisibilityWordCount))},
                {"bdaHash", ExactVisibilityHash(actual.subspan(ExactVisibilityWordCount,
                                                               ExactVisibilityWordCount))},
                {"pageTableHash", ExactVisibilityHash(actual.subspan(ExactVisibilityWordCount * 2,
                                                                     ExactVisibilityWordCount))},
                {"descriptorMismatchCount", descriptor_mismatches},
                {"bdaMismatchCount", bda_mismatches},
                {"pageTableMismatchCount", page_table_mismatches},
                {"descriptorFirstMismatch", descriptor_first_mismatch},
                {"bdaFirstMismatch", bda_first_mismatch},
                {"pageTableFirstMismatch", page_table_first_mismatch},
            });
            if (descriptor_mismatches != 0 || bda_mismatches != 0 || page_table_mismatches != 0) {
                result.failure = page_table_mismatches != 0 ? "guest_page_table_data_mismatch"
                                 : descriptor_mismatches != 0 && bda_mismatches != 0
                                     ? "descriptor_and_bda_data_mismatch"
                                 : descriptor_mismatches != 0 ? "descriptor_data_mismatch"
                                                              : "bda_data_mismatch";
                return result;
            }
            canonical_view_verified |= !use_guest_alias;
            guest_alias_verified |= use_guest_alias;
            guest_page_table_verified = true;
        }
    }

    const auto switch_page_offset = offsets.front();
    const auto switch_source_offset = switch_page_offset + ExactVisibilityInPageOffset;
    constexpr std::uint64_t switch_guest_address =
        (ExactVisibilityGuestPage << 14) + ExactVisibilityInPageOffset;
    std::array<std::uint32_t, ExactVisibilityWordCount> imported_expected{};
    std::array<std::uint32_t, ExactVisibilityWordCount> cache_expected{};
    for (std::size_t index = 0; index < imported_expected.size(); ++index) {
        imported_expected[index] =
            0x1a2b0000u ^ (static_cast<std::uint32_t>(index) * 0x9e3779b9u);
        cache_expected[index] =
            0xc4d50000u ^ (static_cast<std::uint32_t>(index) * 0x7f4a7c15u);
    }
    if (ExactVisibilityHash(imported_expected) == ExactVisibilityHash(cache_expected)) {
        return fail("publication_switch_patterns_not_distinct");
    }
    std::memcpy(static_cast<std::byte*>(host_pointer) + switch_source_offset,
                imported_expected.data(), ExactVisibilityByteCount);

    const vk::DescriptorBufferInfo switch_input_info{
        .buffer = imported_buffer,
        .offset = switch_source_offset,
        .range = ExactVisibilityByteCount,
    };
    const vk::DescriptorBufferInfo switch_output_info{
        .buffer = *output_buffer,
        .offset = 0,
        .range = ExactVisibilityOutputByteCount,
    };
    const vk::DescriptorBufferInfo switch_page_table_info{
        .buffer = *page_table_buffer,
        .offset = 0,
        .range = page_table_bytes,
    };
    const std::array switch_writes{
        vk::WriteDescriptorSet{
            .dstSet = *descriptor_sets.front(),
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &switch_input_info,
        },
        vk::WriteDescriptorSet{
            .dstSet = *descriptor_sets.front(),
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &switch_output_info,
        },
        vk::WriteDescriptorSet{
            .dstSet = *descriptor_sets.front(),
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &switch_page_table_info,
        },
    };
    device.updateDescriptorSets(switch_writes, {});

    constexpr std::array<std::string_view, 3> phase_names{
        "imported_before", "cache_override", "imported_restored"};
    bool imported_before_verified{};
    bool cache_override_verified{};
    bool imported_after_restore_verified{};
    bool direct_import_control_verified{true};
    for (std::size_t phase = 0; phase < phase_names.size(); ++phase) {
        const bool use_cache = phase == 1;
        const auto expected_page = use_cache ? std::span{cache_expected}
                                             : std::span{imported_expected};
        std::memset(mapped, 0, ExactVisibilityOutputByteCount);
        std::memset(page_table_mapped, 0, page_table_bytes);
        static_cast<vk::DeviceAddress*>(page_table_mapped)[ExactVisibilityGuestPage] =
            use_cache ? cache_address : imported_address + switch_page_offset;

        auto& command_buffer = command_buffers.front();
        if (command_buffer->reset({}) != vk::Result::eSuccess ||
            command_buffer->begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}) !=
                vk::Result::eSuccess) {
            return fail("publication_switch_command_buffer_begin_failed");
        }
        if (use_cache) {
            command_buffer->updateBuffer(*cache_buffer, ExactVisibilityInPageOffset,
                                         ExactVisibilityByteCount, cache_expected.data());
            const vk::BufferMemoryBarrier cache_to_shader{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = *cache_buffer,
                .offset = ExactVisibilityInPageOffset,
                .size = ExactVisibilityByteCount,
            };
            command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                            vk::PipelineStageFlagBits::eComputeShader, {}, {},
                                            cache_to_shader, {});
        }
        const vk::MemoryBarrier host_to_shader{
            .srcAccessMask = vk::AccessFlagBits::eHostWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        };
        command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                                        vk::PipelineStageFlagBits::eComputeShader, {},
                                        host_to_shader, {}, {});
        command_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        command_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipeline_layout, 0,
                                           *descriptor_sets.front(), {});
        const std::array<vk::DeviceAddress, 2> switch_push_data{
            imported_address + switch_source_offset,
            switch_guest_address,
        };
        command_buffer->pushConstants(*pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                      sizeof(switch_push_data), switch_push_data.data());
        command_buffer->dispatch(1, 1, 1);
        const vk::MemoryBarrier shader_to_host{
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eHostRead,
        };
        command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                        vk::PipelineStageFlagBits::eHost, {}, shader_to_host, {},
                                        {});
        if (command_buffer->end() != vk::Result::eSuccess) {
            return fail("publication_switch_command_buffer_end_failed");
        }
        const vk::CommandBuffer submitted = *command_buffer;
        const vk::SubmitInfo submit{
            .commandBufferCount = 1,
            .pCommandBuffers = &submitted,
        };
        if (device.resetFences(*fence) != vk::Result::eSuccess ||
            queue.submit(submit, *fence) != vk::Result::eSuccess) {
            return fail("publication_switch_queue_submission_failed");
        }
        const auto wait_result = device.waitForFences(*fence, true, 10'000'000'000ull);
        if (wait_result != vk::Result::eSuccess) {
            if (Vulkan::RequiresExactProbeIdleBeforeResourceCleanup(
                    false, wait_result == vk::Result::eErrorDeviceLost)) {
                const auto idle_result = device.waitIdle();
                if (!Vulkan::CanUseOrdinaryExactProbeResourceCleanup(
                        idle_result == vk::Result::eSuccess,
                        idle_result == vk::Result::eErrorDeviceLost)) {
                    std::cerr << "{\"schema\":\"shadps4.exact-probe-fatal.v1\","
                                 "\"status\":\"fatal\","
                                 "\"reason\":\"unsafe_resource_cleanup_prevented\"}\n";
                    std::cerr.flush();
                    std::_Exit(EXIT_FAILURE);
                }
            }
            return fail("publication_switch_queue_fence_wait_failed");
        }

        const auto actual =
            std::span{static_cast<const std::uint32_t*>(mapped), ExactVisibilityWordCount * 3};
        const auto descriptor_actual = actual.first(ExactVisibilityWordCount);
        const auto direct_actual =
            actual.subspan(ExactVisibilityWordCount, ExactVisibilityWordCount);
        const auto page_table_actual =
            actual.subspan(ExactVisibilityWordCount * 2, ExactVisibilityWordCount);
        std::size_t descriptor_mismatch_count{};
        std::size_t direct_mismatch_count{};
        std::size_t page_table_mismatch_count{};
        for (std::size_t index = 0; index < imported_expected.size(); ++index) {
            descriptor_mismatch_count += descriptor_actual[index] != imported_expected[index];
            direct_mismatch_count += direct_actual[index] != imported_expected[index];
            page_table_mismatch_count += page_table_actual[index] != expected_page[index];
        }
        result.trials.push_back({
            {"trialKind", "guest_page_table_publication_switch"},
            {"phase", phase_names[phase]},
            {"pageTableSource", use_cache ? "device_local_cache" : "imported_backing"},
            {"expectedImportedHash", ExactVisibilityHash(imported_expected)},
            {"expectedPageTableHash", ExactVisibilityHash(expected_page)},
            {"descriptorHash", ExactVisibilityHash(descriptor_actual)},
            {"bdaHash", ExactVisibilityHash(direct_actual)},
            {"pageTableHash", ExactVisibilityHash(page_table_actual)},
            {"descriptorMismatchCount", descriptor_mismatch_count},
            {"bdaMismatchCount", direct_mismatch_count},
            {"pageTableMismatchCount", page_table_mismatch_count},
        });
        direct_import_control_verified &=
            descriptor_mismatch_count == 0 && direct_mismatch_count == 0;
        const bool phase_verified = descriptor_mismatch_count == 0 &&
                                    direct_mismatch_count == 0 &&
                                    page_table_mismatch_count == 0;
        imported_before_verified |= phase == 0 && phase_verified;
        cache_override_verified |= phase == 1 && phase_verified;
        imported_after_restore_verified |= phase == 2 && phase_verified;
        if (!phase_verified) {
            return fail("guest_page_table_publication_switch_data_mismatch");
        }
    }
    result.succeeded = Vulkan::HasRequiredExactGuestPublicationEvidence(
        canonical_view_verified, guest_alias_verified, guest_page_table_verified,
        imported_before_verified, cache_override_verified, imported_after_restore_verified,
        direct_import_control_verified);
    if (!result.succeeded) {
        return fail("required_visibility_evidence_incomplete");
    }
    result.failure = "none";
    return result;
}

[[nodiscard]] ExactProbeOutput ProbeExactAddressSpaceHostAllocation(
    vk::PhysicalDevice physical_device, vk::Device device, std::uint64_t backing_size,
    std::size_t min_pointer_alignment, std::uint32_t queue_family_index) {
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
        {"cleanup", "destroy_buffer,destroy_memory,UnmapViewOfFile2(MEM_PRESERVE_PLACEHOLDER),"
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
        const auto external =
            physical_device.getExternalBufferProperties(external_info).externalMemoryProperties;
        attempt["compatibleHandleTypes"] =
            static_cast<std::uint32_t>(external.compatibleHandleTypes);
        attempt["importable"] = static_cast<bool>(external.externalMemoryFeatures &
                                                  vk::ExternalMemoryFeatureFlagBits::eImportable);
        if (!attempt["importable"].get<bool>() || !(external.compatibleHandleTypes & handle_type)) {
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
            fail(ClassifyExactVulkanFailure(ClassifyExactVulkanResult(buffer_result)),
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
        if (requirements.alignment == 0 || !std::has_single_bit(min_pointer_alignment) ||
            requirements.size == 0 || requirements.size % min_pointer_alignment != 0) {
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
            const auto& rollback = acquisition.rollback;
            attempt["cleanupAttempted"] =
                rollback.unmap_attempted || rollback.release_attempted || rollback.close_attempted;
            attempt["unmapAttempted"] = rollback.unmap_attempted;
            attempt["unmapSucceeded"] = rollback.unmap_succeeded;
            attempt["placeholderReleaseAttempted"] = rollback.release_attempted;
            attempt["placeholderReleaseSucceeded"] = rollback.release_succeeded;
            attempt["closeAttempted"] = rollback.close_attempted;
            attempt["closeSucceeded"] = rollback.close_succeeded;
            attempt["cleanupComplete"] = acquisition.rollback_complete;
            const auto win32_error_class = ClassifyExactWindowsBackingErrorCode(adapter.last_error);
            const auto acquisition_failure = ClassifyExactBackingAcquisitionFailure(
                win32_error_class, acquisition.rollback_complete);
            fail(acquisition_failure,
                 acquisition_failure == ExactHostImportFailure::BackingCommitFailed
                     ? "exact_backing_resource_limited"
                     : "exact_backing_win32_lifecycle_failed");
            return;
        }
        backing_acquired = true;
        attempt["pointerModuloAlignment"] =
            reinterpret_cast<std::uintptr_t>(acquisition.backing.pointer) % min_pointer_alignment;
        if (!IsExactHostImportAlignmentValid(
                requirements.size, reinterpret_cast<std::uintptr_t>(acquisition.backing.pointer),
                min_pointer_alignment)) {
            fail(ExactHostImportFailure::RequirementAlignmentMismatch,
                 "exact_import_pointer_or_allocation_size_misaligned");
            return;
        }
        if (!protocol.Complete(ExactHostImportStage::Backing)) {
            return;
        }

        auto [host_result, host_properties] =
            device.getMemoryHostPointerPropertiesEXT(handle_type, acquisition.backing.pointer);
        attempt["hostPointerQueryVkResult"] = vk::to_string(host_result);
        if (host_result != vk::Result::eSuccess) {
            fail(ClassifyExactVulkanFailure(ClassifyExactVulkanResult(host_result)),
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
        const auto selected_flags =
            static_cast<std::uint32_t>(physical_memory.memoryTypes[*selection.index].propertyFlags);
        const auto selected_evidence = MakeExactSelectedMemoryTypeEvidence(
            selected_flags, static_cast<std::uint32_t>(vk::MemoryPropertyFlagBits::eHostCoherent));
        attempt["selectedMemoryPropertyFlags"] = selected_evidence.property_flags;
        attempt["selectedMemoryTypeHostCoherent"] = selected_evidence.host_coherent;
        if (!protocol.Complete(ExactHostImportStage::MemoryTypeSelection)) {
            return;
        }

        const vk::MemoryAllocateFlagsInfo allocation_flags{
            .flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
        const vk::MemoryDedicatedAllocateInfo dedicated_info{
            .pNext = &allocation_flags,
            .buffer = *buffer,
        };
        const vk::ImportMemoryHostPointerInfoEXT import_info{
            .pNext = &dedicated_info,
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
            fail(ClassifyExactVulkanFailure(ClassifyExactVulkanResult(memory_result)),
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
        if (!protocol.Complete(ExactHostImportStage::DeviceAddress)) {
            return;
        }
        attempt["dataVerificationAttempted"] = true;
        auto visibility = VerifyExactHostVisibility(physical_device, device, queue_family_index,
                                                    *buffer, address, acquisition.backing.pointer,
                                                    acquisition.backing.mapping, backing_size);
        attempt["dataVerificationPassed"] = visibility.succeeded;
        attempt["dataVerificationFailure"] = visibility.failure;
        attempt["dataVerificationTrials"] = std::move(visibility.trials);
        if (!visibility.succeeded) {
            fail(ExactHostImportFailure::DataVerificationFailed,
                 attempt["dataVerificationFailure"].get<std::string>());
            return;
        }
        if (!protocol.Complete(ExactHostImportStage::DataVerification) ||
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
        attempt["placeholderReleaseAttempted"] = true;
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
            report["shaderInt64Feature"] = feature_chain.get().features.shaderInt64 == VK_TRUE;
            if (!features12.bufferDeviceAddress) {
                report["status"] = "unsupported";
                report["reason"] = "buffer_device_address_unavailable";
                return 2;
            }
#ifdef SHADPS4_EXACT_ADDRESS_SPACE_PROBE
            if (!feature_chain.get().features.shaderInt64) {
                report["status"] = "unsupported";
                report["reason"] = "shader_int64_unavailable";
                return 2;
            }
#endif

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
#ifdef SHADPS4_EXACT_ADDRESS_SPACE_PROBE
            const vk::PhysicalDeviceFeatures enabled_core_features{
                .shaderInt64 = VK_TRUE,
            };
#endif
            constexpr std::array enabled_extensions{VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME};
            const vk::DeviceCreateInfo device_info{
                .pNext = &enabled_features,
                .queueCreateInfoCount = 1,
                .pQueueCreateInfos = &queue_info,
                .enabledExtensionCount = static_cast<std::uint32_t>(enabled_extensions.size()),
                .ppEnabledExtensionNames = enabled_extensions.data(),
#ifdef SHADPS4_EXACT_ADDRESS_SPACE_PROBE
                .pEnabledFeatures = &enabled_core_features,
#endif
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
            report["mechanismEvidenceOnly"] = false;
            report["dataVerificationRequired"] = true;
            auto host_allocation = ProbeExactAddressSpaceHostAllocation(
                physical_device, *device, arguments->backing_size, min_alignment, queue_index);
            report["attempts"].push_back(std::move(host_allocation.report));
            report["status"] = ExactDispositionName(host_allocation.disposition);
            if (host_allocation.disposition == Vulkan::ExactHostImportDisposition::Pass) {
                report["reason"] = "exact_address_space_host_data_verified";
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
