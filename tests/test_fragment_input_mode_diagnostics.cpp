// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <gtest/gtest.h>

#if __has_include("video_core/renderer_vulkan/fragment_input_mode_diagnostics.h")
#include "video_core/renderer_vulkan/fragment_input_mode_diagnostics.h"
#define HAS_FRAGMENT_INPUT_MODE_DIAGNOSTICS 1
#else
#define HAS_FRAGMENT_INPUT_MODE_DIAGNOSTICS 0
#endif

namespace {

#if HAS_FRAGMENT_INPUT_MODE_DIAGNOSTICS
struct Input {
    bool is_default{};
    bool is_flat{};
};

TEST(FragmentInputModeDiagnostics, RawDefaultAndFlatBitsRemainIndependent) {
    const std::array inputs{Input{true, false}, Input{false, true}, Input{true, true},
                            Input{false, false}};
    const auto row = Vulkan::SummarizeFragmentInputModes(4, std::span{inputs});
    EXPECT_EQ(row.declared_count, 4U);
    EXPECT_EQ(row.captured_count, 4U);
    EXPECT_EQ(row.default_mask, 0b0101U);
    EXPECT_EQ(row.flat_mask, 0b0110U);
}

TEST(FragmentInputModeDiagnostics, InputThirtyOneUsesTheUnsignedHighBit) {
    std::array<Input, 32> inputs{};
    inputs[31] = {true, true};
    const auto row = Vulkan::SummarizeFragmentInputModes(32, std::span{inputs});
    EXPECT_EQ(row.default_mask, 0x80000000U);
    EXPECT_EQ(row.flat_mask, 0x80000000U);
    EXPECT_EQ(row.captured_count, 32U);
}

TEST(FragmentInputModeDiagnostics, HugeCountClampsToThirtyTwoWithoutReadingLaterInputs) {
    std::array<Input, 40> inputs{};
    inputs[0] = {true, false};
    inputs[31] = {false, true};
    inputs[32] = {true, true};
    const auto row = Vulkan::SummarizeFragmentInputModes(std::numeric_limits<std::uint32_t>::max(),
                                                         std::span{inputs});
    EXPECT_EQ(row.declared_count, std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(row.captured_count, 32U);
    EXPECT_EQ(row.default_mask, 1U);
    EXPECT_EQ(row.flat_mask, 0x80000000U);
}

TEST(FragmentInputModeDiagnostics, ShortSpanAndDeclaredCountBothBoundCapture) {
    const std::array inputs{Input{true, true}, Input{true, true}};
    const auto short_span = Vulkan::SummarizeFragmentInputModes(32, std::span{inputs});
    EXPECT_EQ(short_span.captured_count, 2U);
    EXPECT_EQ(short_span.default_mask, 3U);
    const auto one = Vulkan::SummarizeFragmentInputModes(1, std::span{inputs});
    EXPECT_EQ(one.captured_count, 1U);
    EXPECT_EQ(one.flat_mask, 1U);
    const auto empty = Vulkan::SummarizeFragmentInputModes(0, std::span{inputs});
    EXPECT_EQ(empty.captured_count, 0U);
    EXPECT_EQ(empty.default_mask, 0U);
    EXPECT_EQ(empty.flat_mask, 0U);
    const auto missing = Vulkan::SummarizeFragmentInputModes(32, std::span<const Input>{});
    EXPECT_EQ(missing.captured_count, 0U);
    EXPECT_EQ(missing.default_mask, 0U);
    EXPECT_EQ(missing.flat_mask, 0U);
}
#else
TEST(FragmentInputModeDiagnostics, BoundedProductionHelperExists) {
    FAIL() << "Fragment input-mode diagnostic helper is absent";
}
#endif

std::string PipelineSource() {
    std::ifstream file{SHADPS4_FRAGMENT_INPUT_DIAGNOSTICS_PIPELINE_SOURCE_PATH};
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

TEST(FragmentInputModeDiagnostics, CompileReceiptIsDumpAndFragmentGated) {
    const auto source = PipelineSource();
    const auto begin = source.find("vk::ShaderModule PipelineCache::CompileModule(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = source.find("PipelineCache::Result PipelineCache::GetProgram(", begin);
    ASSERT_NE(end, std::string::npos);
    const auto function = source.substr(begin, end - begin);
    const auto gate = function.find("EmulatorSettings.IsDumpShaders() &&");
    ASSERT_NE(gate, std::string::npos);
    const auto fragment = function.find("runtime_info.stage == Shader::Stage::Fragment", gate);
    ASSERT_NE(fragment, std::string::npos);
    const auto summarize = function.find("SummarizeFragmentInputModes(", fragment);
    ASSERT_NE(summarize, std::string::npos);
    const auto receipt = function.find("ShaderFragmentInputsV1", summarize);
    ASSERT_NE(receipt, std::string::npos);
    EXPECT_EQ(function.find("ShaderFragmentInputsV1", receipt + 1), std::string::npos);
    EXPECT_EQ(source.find("ShaderFragmentInputsV1", begin + receipt + 1), std::string::npos);
}

TEST(FragmentInputModeDiagnostics, BothCompilePathsSupplyTheirEmittedRuntime) {
    const auto source = PipelineSource();
    EXPECT_NE(source.find("CompileModule(program->info, runtime_info, params.code, 0, spv)"),
              std::string::npos);
    EXPECT_NE(source.find("CompileModule(new_info, runtime_info, params.code, perm_idx, spv)"),
              std::string::npos);
}

} // namespace
