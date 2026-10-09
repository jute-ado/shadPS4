// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace {

std::string ReadText(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(VulkanFeatureEnablement, EnablesSupportedIndirectFirstInstance) {
    const auto source = ReadText(SHADPS4_VULKAN_INSTANCE_SOURCE_PATH);
    const auto features_begin = source.find("vk::PhysicalDeviceFeatures2{");
    const auto features_end = source.find("vk::PhysicalDeviceVulkan11Features{", features_begin);
    ASSERT_NE(features_begin, std::string::npos);
    ASSERT_NE(features_end, std::string::npos);

    const auto enabled_features = source.substr(features_begin, features_end - features_begin);
    EXPECT_NE(enabled_features.find(
                  ".drawIndirectFirstInstance = features.drawIndirectFirstInstance"),
              std::string::npos);
}

TEST(VulkanFeatureEnablement, DeviceFaultDiagnosticsAreFeatureGatedAndQueriedBeforeAssertion) {
    const auto instance_source = ReadText(SHADPS4_VULKAN_INSTANCE_SOURCE_PATH);
    EXPECT_NE(instance_source.find("VK_EXT_DEVICE_FAULT_EXTENSION_NAME"), std::string::npos);
    EXPECT_NE(instance_source.find("vk::PhysicalDeviceFaultFeaturesEXT"), std::string::npos);
    EXPECT_NE(instance_source.find("device_chain.unlink<vk::PhysicalDeviceFaultFeaturesEXT>()"),
              std::string::npos);

    const auto scheduler_source = ReadText(SHADPS4_VULKAN_SCHEDULER_SOURCE_PATH);
    const auto submit = scheduler_source.find("auto submit_result =");
    const auto query = scheduler_source.find("LogDeviceLossDiagnostics()", submit);
    const auto assertion = scheduler_source.find("ASSERT_MSG(submit_result", submit);
    ASSERT_NE(submit, std::string::npos);
    ASSERT_NE(query, std::string::npos);
    ASSERT_NE(assertion, std::string::npos);
    EXPECT_LT(query, assertion);
}

TEST(VulkanFeatureEnablement, NvidiaDiagnosticCheckpointsIdentifyTheFaultingShader) {
    const auto instance_source = ReadText(SHADPS4_VULKAN_INSTANCE_SOURCE_PATH);
    EXPECT_NE(instance_source.find("VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME"),
              std::string::npos);

    const auto rasterizer_source = ReadText(SHADPS4_VULKAN_RASTERIZER_SOURCE_PATH);
    EXPECT_NE(rasterizer_source.find("MakeDiagnosticCheckpoint"), std::string::npos);
    EXPECT_NE(rasterizer_source.find("vkCmdSetCheckpointNV"), std::string::npos);

    const auto scheduler_source = ReadText(SHADPS4_VULKAN_SCHEDULER_SOURCE_PATH);
    const auto submit = scheduler_source.find("auto submit_result =");
    const auto query = scheduler_source.find("LogDeviceLossDiagnostics()", submit);
    const auto assertion = scheduler_source.find("ASSERT_MSG(submit_result", submit);
    ASSERT_NE(submit, std::string::npos);
    ASSERT_NE(query, std::string::npos);
    ASSERT_NE(assertion, std::string::npos);
    EXPECT_NE(scheduler_source.find("vkGetQueueCheckpointDataNV"), std::string::npos);
    EXPECT_LT(query, assertion);
}

TEST(VulkanFeatureEnablement, PresenterDeviceLossAlsoReportsQueueDiagnostics) {
    const auto presenter_source = ReadText(SHADPS4_VULKAN_PRESENTER_SOURCE_PATH);
    const auto wait = presenter_source.find("waitForFences");
    const auto diagnostics = presenter_source.find("LogDeviceLossDiagnostics", wait);
    const auto assertion = presenter_source.find("ASSERT_MSG(result", wait);
    ASSERT_NE(wait, std::string::npos);
    ASSERT_NE(diagnostics, std::string::npos);
    ASSERT_NE(assertion, std::string::npos);
    EXPECT_LT(diagnostics, assertion);
}

} // namespace
