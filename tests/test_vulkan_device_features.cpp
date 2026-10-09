// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

namespace {

std::string ReadText(const char* path) {
    std::ifstream source{path, std::ios::binary};
    EXPECT_TRUE(source.is_open());
    std::string text{std::istreambuf_iterator<char>{source}, std::istreambuf_iterator<char>{}};
    std::erase(text, '\r');
    return text;
}

TEST(VulkanDeviceFeatures, EnablesSupportedIndirectFirstInstance) {
    const std::string text = ReadText(SHADPS4_VK_INSTANCE_SOURCE_PATH);

    EXPECT_NE(text.find(".drawIndirectFirstInstance = features.drawIndirectFirstInstance"),
              std::string::npos);
}

TEST(VulkanDeviceFeatures, EnablesSupportedSampledImageNonUniformIndexing) {
    const std::string text = ReadText(SHADPS4_VK_INSTANCE_SOURCE_PATH);

    EXPECT_NE(text.find(".shaderSampledImageArrayNonUniformIndexing =\n"
                        "                vk12_features.shaderSampledImageArrayNonUniformIndexing"),
              std::string::npos);
}

TEST(VulkanDeviceFeatures, ExposesRealSampledImageLimitsAndFeature) {
    const std::string text = ReadText(SHADPS4_VK_INSTANCE_HEADER_PATH);

    EXPECT_NE(text.find("return properties.limits.maxPerStageDescriptorSampledImages;"),
              std::string::npos);
    EXPECT_NE(text.find("return properties.limits.maxDescriptorSetSampledImages;"),
              std::string::npos);
    EXPECT_NE(text.find("return properties.limits.maxPerStageResources;"), std::string::npos);
    EXPECT_NE(text.find("return vk12_features.shaderSampledImageArrayNonUniformIndexing;"),
              std::string::npos);
}

TEST(VulkanDeviceFeatures, PopulatesDynamicSampledImageBudgetFromDeviceAndHeap) {
    const std::string text = ReadText(SHADPS4_VK_PIPELINE_CACHE_SOURCE_PATH);

    EXPECT_NE(text.find("DescriptorHeapSampledImages"), std::string::npos);
    EXPECT_NE(text.find(".max_sampled_images_per_stage = "
                        "instance.MaxPerStageDescriptorSampledImages()"),
              std::string::npos);
    EXPECT_NE(text.find(".max_sampled_images_per_set = instance.MaxDescriptorSetSampledImages()"),
              std::string::npos);
    EXPECT_NE(text.find(".max_resources_per_stage = instance.MaxPerStageResources()"),
              std::string::npos);
    EXPECT_NE(
        text.find(".max_sampled_images_per_descriptor_heap_set =\n"
                  "            DescriptorHeapSampledImages / DescriptorHeap::DescriptorSetBatch"),
        std::string::npos);
    EXPECT_NE(text.find(".supports_sampled_image_array_nonuniform_indexing =\n"
                        "            instance_.IsSampledImageArrayNonUniformIndexingSupported()"),
              std::string::npos);
}

TEST(VulkanDeviceFeatures, ProfileOwnsDynamicSampledImageBudgetInputs) {
    const std::string text = ReadText(SHADPS4_PROFILE_HEADER_PATH);

    EXPECT_NE(text.find("u32 max_sampled_images_per_stage{};"), std::string::npos);
    EXPECT_NE(text.find("u32 max_sampled_images_per_set{};"), std::string::npos);
    EXPECT_NE(text.find("u32 max_resources_per_stage{};"), std::string::npos);
    EXPECT_NE(text.find("u32 max_sampled_images_per_descriptor_heap_set{};"), std::string::npos);
    EXPECT_NE(text.find("bool supports_sampled_image_array_nonuniform_indexing{};"),
              std::string::npos);
}

} // namespace
