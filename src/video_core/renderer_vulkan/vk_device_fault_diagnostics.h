// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <string_view>
#include <thread>

#include <vulkan/vulkan_core.h>

namespace Vulkan {

constexpr bool DeviceFaultDiagnosticsRequested(std::string_view value) noexcept {
    return value == "1";
}

constexpr bool ShouldEnableDeviceFaultDiagnostics(std::string_view value, bool advertised,
                                                  bool feature_supported) noexcept {
    return DeviceFaultDiagnosticsRequested(value) && advertised && feature_supported;
}

inline constexpr uint32_t DeviceFaultAddressLimit = 16;
inline constexpr uint32_t DeviceFaultVendorLimit = 8;

struct BoundedDeviceFault {
    VkResult count_result = VK_NOT_READY;
    VkResult payload_result = VK_NOT_READY;
    uint32_t advertised_address_count{};
    uint32_t advertised_vendor_count{};
    uint32_t address_count{};
    uint32_t vendor_count{};
    bool payload_available{};
    bool truncated{};
    char description[VK_MAX_DESCRIPTION_SIZE]{};
    std::array<VkDeviceFaultAddressInfoEXT, DeviceFaultAddressLimit> addresses{};
    std::array<VkDeviceFaultVendorInfoEXT, DeviceFaultVendorLimit> vendors{};
};

template <size_t Size>
std::string_view BoundedDeviceFaultText(const char (&text)[Size]) noexcept {
    return {text, static_cast<size_t>(std::find(text, text + Size, '\0') - text)};
}

// Query performs at most two calls. Never allocate from a driver-provided count, request a vendor
// binary, or retry an incomplete result: the bounded prefix is sufficient for text diagnostics.
template <typename Query>
BoundedDeviceFault QueryBoundedDeviceFault(Query&& query) {
    BoundedDeviceFault report;
    VkDeviceFaultCountsEXT counts{.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    report.count_result = query(&counts, nullptr);
    if (report.count_result != VK_SUCCESS && report.count_result != VK_INCOMPLETE) {
        return report;
    }
    report.advertised_address_count = counts.addressInfoCount;
    report.advertised_vendor_count = counts.vendorInfoCount;
    counts.addressInfoCount = std::min(counts.addressInfoCount, DeviceFaultAddressLimit);
    counts.vendorInfoCount = std::min(counts.vendorInfoCount, DeviceFaultVendorLimit);
    counts.vendorBinarySize = 0;
    const auto address_capacity = counts.addressInfoCount;
    const auto vendor_capacity = counts.vendorInfoCount;
    VkDeviceFaultInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
        .pAddressInfos = report.addresses.data(),
        .pVendorInfos = report.vendors.data(),
        .pVendorBinaryData = nullptr,
    };
    report.payload_result = query(&counts, &info);
    if (report.payload_result != VK_SUCCESS && report.payload_result != VK_INCOMPLETE) {
        return report;
    }
    report.payload_available = true;
    report.address_count = std::min(counts.addressInfoCount, address_capacity);
    report.vendor_count = std::min(counts.vendorInfoCount, vendor_capacity);
    report.truncated = report.payload_result == VK_INCOMPLETE ||
                       report.advertised_address_count > report.address_count ||
                       report.advertised_vendor_count > report.vendor_count;
    std::copy(std::begin(info.description), std::end(info.description),
              std::begin(report.description));
    return report;
}

enum class DeviceFaultReportStatus { Skipped, Reported, AlreadyReported, WaitTimedOut };

class DeviceFaultReportOnce {
public:
    template <typename Report>
    DeviceFaultReportStatus Run(bool enabled, VkResult result, Report&& report,
                                std::chrono::milliseconds wait_budget = std::chrono::seconds{2}) {
        if (!enabled || result != VK_ERROR_DEVICE_LOST) {
            return DeviceFaultReportStatus::Skipped;
        }
        State expected = State::Idle;
        if (!state.compare_exchange_strong(expected, State::Reporting, std::memory_order_acq_rel)) {
            // Other failure sites may be about to assert. Let the reporting caller finish first,
            // but never wait indefinitely on a driver that stopped answering diagnostic queries.
            const auto deadline = std::chrono::steady_clock::now() + wait_budget;
            while (state.load(std::memory_order_acquire) == State::Reporting) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    return DeviceFaultReportStatus::WaitTimedOut;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
            }
            return DeviceFaultReportStatus::AlreadyReported;
        }
        struct CompleteOnExit {
            std::atomic<State>& state;
            ~CompleteOnExit() {
                state.store(State::Complete, std::memory_order_release);
            }
        } completion{state};
        report();
        return DeviceFaultReportStatus::Reported;
    }

private:
    enum class State { Idle, Reporting, Complete };
    std::atomic<State> state{State::Idle};
};

} // namespace Vulkan
