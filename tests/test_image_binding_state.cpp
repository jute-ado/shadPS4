// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include "video_core/texture_cache/image_binding_state.h"

namespace {

TEST(ImageBindingState, StorageThenSampledRetainsGeneralLayoutRequirement) {
    VideoCore::ImageBindingState binding{};
    binding.RecordShaderAccess(true);
    binding.RecordShaderAccess(false);
    EXPECT_TRUE(binding.is_bound);
    EXPECT_TRUE(binding.force_general);
}

TEST(ImageBindingState, SampledThenStorageRequiresGeneralLayout) {
    VideoCore::ImageBindingState binding{};
    binding.RecordShaderAccess(false);
    binding.RecordShaderAccess(true);
    EXPECT_TRUE(binding.is_bound);
    EXPECT_TRUE(binding.force_general);
}

TEST(ImageBindingState, SingleStorageBindingRecordsItsStorageRequirement) {
    VideoCore::ImageBindingState binding{};
    binding.RecordShaderAccess(true);
    EXPECT_TRUE(binding.is_bound);
    EXPECT_TRUE(binding.force_general);
}

TEST(ImageBindingState, SampledOnlyBindingsDoNotForceGeneralLayout) {
    VideoCore::ImageBindingState binding{};
    binding.RecordShaderAccess(false);
    binding.RecordShaderAccess(false);
    EXPECT_TRUE(binding.is_bound);
    EXPECT_FALSE(binding.force_general);
}

TEST(ImageBindingState, ShaderAccessDoesNotEraseTargetOrRebindFlags) {
    VideoCore::ImageBindingState binding{.is_target = 1U, .needs_rebind = 1U};
    binding.RecordShaderAccess(false);
    binding.RecordShaderAccess(true);
    EXPECT_TRUE(binding.is_target);
    EXPECT_TRUE(binding.needs_rebind);
    EXPECT_TRUE(binding.force_general);
}

TEST(ImageBindingState, GeneralStorageAccessIncludesShaderWritesWithoutAttachmentAccess) {
    const VideoCore::ImageBindingState binding{.force_general = 1U};
    EXPECT_EQ(binding.GeneralLayoutAccessMask(false),
              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}

TEST(ImageBindingState, TargetOnlyAccessRetainsAttachmentDependenciesWithoutShaderWrite) {
    const VideoCore::ImageBindingState binding{.is_target = 1U};
    EXPECT_EQ(binding.GeneralLayoutAccessMask(false),
              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eColorAttachmentRead |
                  vk::AccessFlagBits2::eColorAttachmentWrite);
}

TEST(ImageBindingState, StorageTargetAccessRetainsShaderAndAttachmentDependencies) {
    const VideoCore::ImageBindingState binding{.is_target = 1U, .force_general = 1U};
    EXPECT_EQ(binding.GeneralLayoutAccessMask(false),
              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
                  vk::AccessFlagBits2::eColorAttachmentRead |
                  vk::AccessFlagBits2::eColorAttachmentWrite);
}

TEST(ImageBindingState, DepthTargetAccessKeepsItsExistingDepthDependency) {
    const VideoCore::ImageBindingState binding{.is_target = 1U};
    EXPECT_EQ(binding.GeneralLayoutAccessMask(true),
              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
}

TEST(ImageBindingState, InitialAndLateBindingUseTheSamePolicy) {
    std::ifstream input{SHADPS4_DESCRIPTOR_RASTERIZER_SOURCE_PATH};
    std::ostringstream stream;
    stream << input.rdbuf();
    const auto source = stream.str();
    const auto initial_begin = source.find("void Rasterizer::BindTextures(");
    const auto late_begin = source.find("bool Rasterizer::RebindPendingImageBindings(");
    const auto late_end = source.find("void Rasterizer::ObserveDescriptorImageDrawDrop(");
    ASSERT_NE(initial_begin, std::string::npos);
    ASSERT_NE(late_begin, std::string::npos);
    ASSERT_NE(late_end, std::string::npos);
    const auto initial = source.substr(initial_begin, late_begin - initial_begin);
    const auto late = source.substr(late_begin, late_end - late_begin);
    EXPECT_NE(initial.find("binding.RecordShaderAccess(image_desc.is_written)"), std::string::npos);
    EXPECT_NE(late.find("binding.RecordShaderAccess(is_storage)"), std::string::npos);
    EXPECT_NE(initial.find("binding.GeneralLayoutAccessMask(image.info.props.is_depth)"),
              std::string::npos);
    EXPECT_NE(late.find("binding.GeneralLayoutAccessMask(image->info.props.is_depth)"),
              std::string::npos);
    EXPECT_EQ(initial.find("binding.force_general |="), std::string::npos);
    EXPECT_EQ(late.find("binding.force_general |="), std::string::npos);
}

} // namespace
