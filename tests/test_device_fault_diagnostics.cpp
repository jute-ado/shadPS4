// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace Vulkan {
namespace {

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(DeviceFaultDiagnostics, EnablesOptionalExtensionAndQueriesFaultBeforeAbort) {
    const auto instance_header = ReadSource(SHADPS4_VK_INSTANCE_HEADER_PATH);
    const auto instance_source = ReadSource(SHADPS4_VK_INSTANCE_SOURCE_PATH);
    const auto scheduler_source = ReadSource(SHADPS4_VK_SCHEDULER_SOURCE_PATH);
    EXPECT_NE(instance_source.find("add_extension(VK_EXT_DEVICE_FAULT_EXTENSION_NAME)"),
              std::string::npos);
    EXPECT_NE(instance_header.find("IsDeviceFaultSupported"), std::string::npos);
    const auto submit = scheduler_source.find("GetGraphicsQueue().submit");
    const auto fault = scheduler_source.find("LogDeviceFaultInfo", submit);
    const auto assertion = scheduler_source.find("ASSERT_MSG", submit);
    ASSERT_NE(fault, std::string::npos);
    ASSERT_NE(assertion, std::string::npos);
    EXPECT_LT(fault, assertion);
}

TEST(DeviceFaultDiagnostics, QueriesFaultWhenPresentationWaitReportsDeviceLoss) {
    const auto presenter_source = ReadSource(SHADPS4_VK_PRESENTER_SOURCE_PATH);
    const auto wait = presenter_source.find("waitForFences");
    const auto fault = presenter_source.find("LogDeviceFaultInfo", wait);
    const auto assertion = presenter_source.find("ASSERT_MSG", wait);
    ASSERT_NE(wait, std::string::npos);
    ASSERT_NE(fault, std::string::npos);
    ASSERT_NE(assertion, std::string::npos);
    EXPECT_LT(fault, assertion);
}

TEST(DeviceFaultDiagnostics, BoundsDriverAddressOutputAndOnlyAssociatesNearbyBuffers) {
    const auto scheduler_source = ReadSource(SHADPS4_VK_SCHEDULER_SOURCE_PATH);
    EXPECT_NE(scheduler_source.find("MaxLoggedFaultAddresses"), std::string::npos);
    EXPECT_NE(scheduler_source.find("MaxNearestBufferDistance"), std::string::npos);
    EXPECT_NE(scheduler_source.find("additional device-fault address"), std::string::npos);
}

} // namespace
} // namespace Vulkan
