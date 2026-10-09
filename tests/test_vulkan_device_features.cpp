// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

namespace {

TEST(VulkanDeviceFeatures, EnablesSupportedIndirectFirstInstance) {
    std::ifstream source{SHADPS4_VK_INSTANCE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(source.is_open());
    const std::string text{std::istreambuf_iterator<char>{source},
                           std::istreambuf_iterator<char>{}};

    EXPECT_NE(text.find(".drawIndirectFirstInstance = features.drawIndirectFirstInstance"),
              std::string::npos);
}

TEST(VulkanDeviceFeatures, EnablesSupportedSampledImageArrayNonUniformIndexing) {
    std::ifstream source{SHADPS4_VK_INSTANCE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(source.is_open());
    const std::string text{std::istreambuf_iterator<char>{source},
                           std::istreambuf_iterator<char>{}};

    const auto shader_int8 = text.find(".shaderInt8 = vk12_features.shaderInt8");
    const auto nonuniform = text.find(
        ".shaderSampledImageArrayNonUniformIndexing =\n"
        "                vk12_features.shaderSampledImageArrayNonUniformIndexing");
    ASSERT_NE(shader_int8, std::string::npos);
    ASSERT_NE(nonuniform, std::string::npos);
    EXPECT_LT(shader_int8, nonuniform);
}

TEST(DynamicImageDescriptorTable, DescriptorArraysMayExceedTheGuestResourceRecordCount) {
    std::ifstream source{SHADPS4_RASTERIZER_HEADER_PATH, std::ios::binary};
    ASSERT_TRUE(source.is_open());
    const std::string text{std::istreambuf_iterator<char>{source},
                           std::istreambuf_iterator<char>{}};

    EXPECT_NE(text.find(
                  "boost::container::small_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES>"),
              std::string::npos);
    EXPECT_NE(text.find(
                  "boost::container::small_vector<DescriptorImageLayoutBinding, Shader::NUM_IMAGES>"),
              std::string::npos);
    EXPECT_NE(text.find(
                  "boost::container::small_vector<VideoCore::ImageId, Shader::NUM_IMAGES>"),
              std::string::npos);
    EXPECT_NE(text.find(
                  "boost::container::small_vector<ImageBindingInfo, Shader::NUM_IMAGES>"),
              std::string::npos);
}

TEST(DynamicImageDescriptorTable, ShaderPermutationsDistinguishSignedAndUnsignedImages) {
    std::ifstream source{SHADPS4_SPECIALIZATION_HEADER_PATH, std::ios::binary};
    ASSERT_TRUE(source.is_open());
    const std::string text{std::istreambuf_iterator<char>{source},
                           std::istreambuf_iterator<char>{}};

    EXPECT_NE(text.find("AmdGpu::NumberClass number_class"), std::string::npos);
    EXPECT_NE(text.find("spec.number_class = AmdGpu::GetNumberClass(sharp.GetNumberFmt())"),
              std::string::npos);
    EXPECT_EQ(text.find("bool is_integer = false"), std::string::npos);
}

TEST(DynamicImageDescriptorTable, EmitsTheArrayAccessOnlyInsideTheBoundsBranch) {
    std::ifstream source{SHADPS4_SPIRV_IMAGE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(source.is_open());
    const std::string text{std::istreambuf_iterator<char>{source},
                           std::istreambuf_iterator<char>{}};
    const auto function_begin = text.find("Id EmitImageSampleExplicitLodDynamic(");
    ASSERT_NE(function_begin, std::string::npos);
    const auto function_end = text.find("Id EmitImageSampleDrefImplicitLod(", function_begin);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = text.substr(function_begin, function_end - function_begin);

    const auto compare = body.find("OpULessThan");
    const auto branch = body.find("OpBranchConditional");
    const auto true_label = body.find("AddLabel(in_bounds_label)");
    const auto access = body.find("OpAccessChain", true_label);
    const auto zero = body.find("const Id zero");
    const auto merge = body.find("OpPhi", zero);
    EXPECT_LT(compare, branch);
    EXPECT_LT(branch, true_label);
    EXPECT_LT(true_label, access);
    EXPECT_LT(access, zero);
    EXPECT_LT(zero, merge);
}

} // namespace
