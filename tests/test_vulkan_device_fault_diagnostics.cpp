// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iterator>
#include <latch>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_device_fault_diagnostics.h"

namespace {

std::string ReadFaultSource(const char* path) {
    std::ifstream source{path, std::ios::binary};
    EXPECT_TRUE(source.is_open());
    std::string text{std::istreambuf_iterator<char>{source}, std::istreambuf_iterator<char>{}};
    std::erase(text, '\r');
    return text;
}

TEST(VulkanDeviceFaultWiring, OptInFeatureIsQueriedEnabledAndUnlinked) {
    const auto source = ReadFaultSource(SHADPS4_VK_INSTANCE_SOURCE_PATH);
    EXPECT_NE(source.find("SHADPS4_DEVICE_FAULT_DIAGNOSTICS"), std::string::npos);
    EXPECT_NE(source.find("VK_EXT_DEVICE_FAULT_EXTENSION_NAME"), std::string::npos);
    EXPECT_NE(source.find("get<vk::PhysicalDeviceFaultFeaturesEXT>()"), std::string::npos);
    EXPECT_NE(source.find("device_chain.unlink<vk::PhysicalDeviceFaultFeaturesEXT>()"),
              std::string::npos);
    EXPECT_NE(source.find(".deviceFaultVendorBinary = false"), std::string::npos);
}

TEST(VulkanDeviceFaultWiring, FaultReportPrecedesOptionalBudgetGate) {
    const auto source = ReadFaultSource(SHADPS4_VK_INSTANCE_SOURCE_PATH);
    const auto start = source.find("void Instance::LogDeviceMemoryBudget(");
    ASSERT_NE(start, std::string::npos);
    const auto fault = source.find("LogDeviceFault(reason, result);", start);
    const auto budget = source.find("if (!VulkanMemoryDiagnosticsEnabled())", start);
    ASSERT_NE(fault, std::string::npos);
    ASSERT_NE(budget, std::string::npos);
    EXPECT_LT(fault, budget);
}

TEST(VulkanDeviceFaultWiring, LostReportUsesTheBoundedOnceQuery) {
    const auto source = ReadFaultSource(SHADPS4_VK_INSTANCE_SOURCE_PATH);
    const auto header = ReadFaultSource(SHADPS4_VK_INSTANCE_HEADER_PATH);
    EXPECT_NE(header.find("DeviceFaultReportOnce"), std::string::npos);
    EXPECT_NE(source.find("vkGetDeviceFaultInfoEXT"), std::string::npos);
    EXPECT_NE(source.find("device_fault_report.Run("), std::string::npos);
    EXPECT_NE(source.find("QueryBoundedDeviceFault("), std::string::npos);
}

TEST(VulkanDeviceFaultDiagnostics, ExactOptInAndAdvertisedFeatureAreRequired) {
    for (const auto value : {"", "0", "true", "yes", "01", " 1", "1 "}) {
        EXPECT_FALSE(Vulkan::ShouldEnableDeviceFaultDiagnostics(value, true, true));
    }
    EXPECT_FALSE(Vulkan::ShouldEnableDeviceFaultDiagnostics("1", false, true));
    EXPECT_FALSE(Vulkan::ShouldEnableDeviceFaultDiagnostics("1", true, false));
    EXPECT_TRUE(Vulkan::ShouldEnableDeviceFaultDiagnostics("1", true, true));
}

TEST(VulkanDeviceFaultDiagnostics, DisabledUnsupportedAndNonLostNeverQueryOrConsumeOnce) {
    Vulkan::DeviceFaultReportOnce gate;
    unsigned calls = 0;
    const auto report = [&] { ++calls; };
    for (const auto value : {"", "0", "1"}) {
        EXPECT_EQ(gate.Run(Vulkan::ShouldEnableDeviceFaultDiagnostics(value, false, false),
                           VK_ERROR_DEVICE_LOST, report),
                  Vulkan::DeviceFaultReportStatus::Skipped);
    }
    for (const auto result : {VK_SUCCESS, VK_TIMEOUT, VK_ERROR_OUT_OF_HOST_MEMORY,
                              VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_UNKNOWN}) {
        EXPECT_EQ(gate.Run(true, result, report), Vulkan::DeviceFaultReportStatus::Skipped);
    }
    EXPECT_EQ(calls, 0u);
    EXPECT_EQ(gate.Run(true, VK_ERROR_DEVICE_LOST, report),
              Vulkan::DeviceFaultReportStatus::Reported);
    EXPECT_EQ(gate.Run(true, VK_ERROR_DEVICE_LOST, report),
              Vulkan::DeviceFaultReportStatus::AlreadyReported);
    EXPECT_EQ(calls, 1u);
}

TEST(VulkanDeviceFaultDiagnostics, HugeDriverCountsAreClampedAndIncompleteTextIsRetained) {
    unsigned calls = 0;
    const auto report = Vulkan::QueryBoundedDeviceFault(
        [&](VkDeviceFaultCountsEXT* counts, VkDeviceFaultInfoEXT* info) -> VkResult {
            ++calls;
            EXPECT_EQ(counts->sType, VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT);
            if (info == nullptr) {
                counts->addressInfoCount = std::numeric_limits<uint32_t>::max();
                counts->vendorInfoCount = std::numeric_limits<uint32_t>::max();
                counts->vendorBinarySize = std::numeric_limits<VkDeviceSize>::max();
                return VK_SUCCESS;
            }
            EXPECT_EQ(counts->addressInfoCount, 16u);
            EXPECT_EQ(counts->vendorInfoCount, 8u);
            EXPECT_EQ(counts->vendorBinarySize, 0u);
            EXPECT_EQ(info->pVendorBinaryData, nullptr);
            EXPECT_EQ(info->sType, VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT);
            info->description[0] = 'X';
            info->pAddressInfos[15].reportedAddress = 0x1234;
            info->pVendorInfos[7].vendorFaultCode = 0x5678;
            return VK_INCOMPLETE;
        });
    EXPECT_EQ(calls, 2u);
    EXPECT_TRUE(report.payload_available);
    EXPECT_TRUE(report.truncated);
    EXPECT_EQ(report.address_count, 16u);
    EXPECT_EQ(report.vendor_count, 8u);
    EXPECT_EQ(report.addresses[15].reportedAddress, 0x1234u);
    EXPECT_EQ(report.vendors[7].vendorFaultCode, 0x5678u);
    EXPECT_EQ(Vulkan::BoundedDeviceFaultText(report.description), "X");
}

TEST(VulkanDeviceFaultDiagnostics, ExactCountsAndDescriptionOnlyReportsAreSupported) {
    for (const uint32_t count : {0u, 1u}) {
        const auto report = Vulkan::QueryBoundedDeviceFault(
            [&](VkDeviceFaultCountsEXT* counts, VkDeviceFaultInfoEXT* info) -> VkResult {
                if (info == nullptr) {
                    counts->addressInfoCount = count;
                    counts->vendorInfoCount = count;
                } else {
                    info->description[0] = 'D';
                    EXPECT_EQ(counts->addressInfoCount, count);
                    EXPECT_EQ(counts->vendorInfoCount, count);
                }
                return VK_SUCCESS;
            });
        EXPECT_TRUE(report.payload_available);
        EXPECT_FALSE(report.truncated);
        EXPECT_EQ(report.address_count, count);
        EXPECT_EQ(report.vendor_count, count);
        EXPECT_EQ(Vulkan::BoundedDeviceFaultText(report.description), "D");
    }
}

TEST(VulkanDeviceFaultDiagnostics, QueryFailureDoesNotPublishPartialDriverRows) {
    for (const bool fail_count : {true, false}) {
        unsigned calls = 0;
        const auto report = Vulkan::QueryBoundedDeviceFault(
            [&](VkDeviceFaultCountsEXT* counts, VkDeviceFaultInfoEXT* info) -> VkResult {
                ++calls;
                counts->addressInfoCount = 1;
                if (fail_count || info != nullptr) {
                    return VK_ERROR_OUT_OF_HOST_MEMORY;
                }
                return VK_SUCCESS;
            });
        EXPECT_EQ(calls, fail_count ? 1u : 2u);
        EXPECT_FALSE(report.payload_available);
        EXPECT_EQ(report.address_count, 0u);
        EXPECT_EQ(report.vendor_count, 0u);
    }
}

TEST(VulkanDeviceFaultDiagnostics, ReturnedCountsCannotExposeBeyondSuppliedCapacity) {
    const auto report = Vulkan::QueryBoundedDeviceFault(
        [](VkDeviceFaultCountsEXT* counts, VkDeviceFaultInfoEXT* info) -> VkResult {
            // Defensively tolerate an inconsistent driver count without indexing past our arrays.
            counts->addressInfoCount = info ? 100u : 1u;
            counts->vendorInfoCount = info ? 100u : 2u;
            return VK_SUCCESS;
        });
    EXPECT_EQ(report.address_count, 1u);
    EXPECT_EQ(report.vendor_count, 2u);
}

TEST(VulkanDeviceFaultDiagnostics, UnterminatedDriverTextHasABoundedView) {
    char description[VK_MAX_DESCRIPTION_SIZE];
    std::fill(std::begin(description), std::end(description), 'X');
    EXPECT_EQ(Vulkan::BoundedDeviceFaultText(description).size(), VK_MAX_DESCRIPTION_SIZE);
    description[7] = '\0';
    EXPECT_EQ(Vulkan::BoundedDeviceFaultText(description).size(), 7u);
}

TEST(VulkanDeviceFaultDiagnostics, RacingLostCallersPerformOnlyOneBoundedQuery) {
    Vulkan::DeviceFaultReportOnce gate;
    std::atomic<unsigned> calls{};
    std::atomic<unsigned> reports{};
    std::latch start{8};
    std::array<std::jthread, 8> threads;
    for (auto& thread : threads) {
        thread = std::jthread([&] {
            start.count_down();
            start.wait();
            const auto status = gate.Run(true, VK_ERROR_DEVICE_LOST, [&] {
                const auto report = Vulkan::QueryBoundedDeviceFault(
                    [&](VkDeviceFaultCountsEXT*, VkDeviceFaultInfoEXT*) {
                        ++calls;
                        return VK_SUCCESS;
                    });
                EXPECT_TRUE(report.payload_available);
                ++reports;
            });
            EXPECT_NE(status, Vulkan::DeviceFaultReportStatus::Skipped);
            EXPECT_NE(status, Vulkan::DeviceFaultReportStatus::WaitTimedOut);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(calls, 2u);
    EXPECT_EQ(reports, 1u);
}

TEST(VulkanDeviceFaultDiagnostics, RacingWaitIsBoundedWhenReportingDriverStalls) {
    Vulkan::DeviceFaultReportOnce gate;
    std::latch entered{1};
    std::latch release{1};
    std::jthread owner([&] {
        (void)gate.Run(true, VK_ERROR_DEVICE_LOST, [&] {
            entered.count_down();
            release.wait();
        });
    });
    entered.wait();
    EXPECT_EQ(gate.Run(
                  true, VK_ERROR_DEVICE_LOST, [] { ADD_FAILURE(); }, std::chrono::milliseconds{0}),
              Vulkan::DeviceFaultReportStatus::WaitTimedOut);
    release.count_down();
    owner.join();
    EXPECT_EQ(gate.Run(true, VK_ERROR_DEVICE_LOST, [] { ADD_FAILURE(); }),
              Vulkan::DeviceFaultReportStatus::AlreadyReported);
}

TEST(VulkanDeviceFaultDiagnostics, ReportingExceptionStillReleasesOnceState) {
    Vulkan::DeviceFaultReportOnce gate;
    EXPECT_THROW((void)gate.Run(true, VK_ERROR_DEVICE_LOST,
                                [] { throw std::runtime_error{"diagnostic failure"}; }),
                 std::runtime_error);
    EXPECT_EQ(gate.Run(true, VK_ERROR_DEVICE_LOST, [] { ADD_FAILURE(); }),
              Vulkan::DeviceFaultReportStatus::AlreadyReported);
}

} // namespace
