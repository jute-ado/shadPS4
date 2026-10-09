// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace Shader {
namespace {

std::string ReadSource(const char* path) {
    std::ifstream file{path, std::ios::binary};
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

TEST(SpirvImageContractSource, FmaskSubstitutionPreservesFourComponentImageResult) {
    const auto source = ReadSource(SHADPS4_RESOURCE_TRACKING_SOURCE_PATH);
    EXPECT_NE(source.find("ir.CompositeConstruct(fmaskx, fmasky, ir.Imm32(0.0f), "
                          "ir.Imm32(0.0f))"),
              std::string::npos);
}

TEST(SpirvImageContractSource, MultisampledStorageImagesDeclareRequiredCapability) {
    const auto source = ReadSource(SHADPS4_SPIRV_EMIT_SOURCE_PATH);
    EXPECT_NE(source.find("spv::Capability::StorageImageMultisample"), std::string::npos);
}

} // namespace
} // namespace Shader
