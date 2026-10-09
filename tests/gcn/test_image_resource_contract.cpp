// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

namespace {

std::string ReadText(const char* path) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input.is_open()) << path;
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

TEST(ImageResourceContract, ReadUdSharpBoundsChecksTheWholeDescriptorBeforeCopying) {
    const std::string source = ReadText(SHADPS4_INFO_HEADER_PATH);
    const auto begin = source.find("inline T ReadUdSharp");
    ASSERT_NE(begin, std::string::npos);
    const auto body = source.substr(begin, 700);

    // A descriptor index is a dword offset, while T can span multiple dwords. The source
    // contract deliberately requires a byte copy after the complete range check so malformed
    // T#/S# indices cannot rely on vector operator[] or an unaligned typed load.
    EXPECT_NE(body.find("flattened_ud_buf.size()"), std::string::npos);
    EXPECT_NE(body.find("sizeof(T) / sizeof(u32)"), std::string::npos);
    EXPECT_NE(body.find("std::memcpy(&value"), std::string::npos);
    EXPECT_EQ(body.find("reinterpret_cast<const T*>"), std::string::npos);
}

TEST(ImageResourceContract, SpirvImageOperationsUseCheckedHandleLookup) {
    const std::string image_source = ReadText(SHADPS4_SPIRV_IMAGE_SOURCE_PATH);
    const std::string atomic_source = ReadText(SHADPS4_SPIRV_ATOMIC_SOURCE_PATH);

    EXPECT_EQ(image_source.find("ctx.images["), std::string::npos);
    EXPECT_EQ(image_source.find("ctx.samplers["), std::string::npos);
    EXPECT_EQ(image_source.find("ctx.info.images["), std::string::npos);
    EXPECT_NE(image_source.find("TryGetImage(handle)"), std::string::npos);
    EXPECT_NE(image_source.find("TryGetSampler(handle)"), std::string::npos);

    EXPECT_EQ(atomic_source.find("ctx.images["), std::string::npos);
    EXPECT_NE(atomic_source.find("TryGetImage(handle)"), std::string::npos);
}

TEST(ImageResourceContract, SpirvHandleHelpersRejectIndicesOutsideTheirTables) {
    const std::string source = ReadText(SHADPS4_SPIRV_CONTEXT_SOURCE_PATH);
    const auto image_helper = source.find("EmitContext::TryGetImage");
    const auto sampler_helper = source.find("EmitContext::TryGetSampler");
    ASSERT_NE(image_helper, std::string::npos);
    ASSERT_NE(sampler_helper, std::string::npos);

    EXPECT_NE(source.substr(image_helper, 500).find("index >= images.size()"),
              std::string::npos);
    EXPECT_NE(source.substr(sampler_helper, 500).find("index >= samplers.size()"),
              std::string::npos);
}

} // namespace
