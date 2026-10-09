// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/opcodes.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace Shader {
namespace {

constexpr auto UnsupportedFormat = AmdGpu::DataFormat::Format10_10_10_2;
constexpr auto SupportedFormat = AmdGpu::DataFormat::Format16;

struct ConversionTranslation {
    Info info{};
    RuntimeInfo runtime{};
    Profile profile{};
    Pools pools{};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};

    explicit ConversionTranslation(Stage stage) {
        EmulatorSettings.SetDirectMemoryAccessEnabled(false);
        EmulatorSettings.SetDumpShaders(false);
        info.stage = stage;
        info.l_stage = stage == Stage::Vertex ? LogicalStage::Vertex : LogicalStage::Compute;
        info.user_data = user_data;
        runtime.Initialize(stage);
        runtime.num_user_data = NUM_USER_DATA_REGS;
        if (stage == Stage::Vertex) {
            runtime.vs_info.num_outputs = 1U;
        } else {
            runtime.cs_info.workgroup_size = {1U, 1U, 1U};
        }
        profile.supported_spirv = 0x00010600U;
        profile.subgroup_size = 32U;
    }
};

AmdGpu::Image MakeImage(AmdGpu::DataFormat format) {
    auto image = AmdGpu::Image::Null(false);
    image.base_address = 0x1000U;
    image.width = 15U;
    image.height = 15U;
    image.data_format = static_cast<u32>(format);
    image.num_format = static_cast<u32>(AmdGpu::NumberFormat::SnormNz);
    return image;
}

int TranslateImage(Gcn::OpcodeMIMG operation, AmdGpu::DataFormat format, bool rejected) {
    ConversionTranslation fixture{Stage::Compute};
    const auto image_words = std::bit_cast<std::array<u32, 8>>(MakeImage(format));
    std::copy(image_words.begin(), image_words.end(), fixture.user_data.begin());
    AmdGpu::Buffer output{};
    output.base_address = 0x200000U;
    output.stride = 4U;
    output.num_records = 16U;
    output.data_format = static_cast<u32>(AmdGpu::DataFormat::Format32);
    output.num_format = static_cast<u32>(AmdGpu::NumberFormat::Uint);
    const auto output_words = std::bit_cast<std::array<u32, 4>>(output);
    std::copy(output_words.begin(), output_words.end(), fixture.user_data.begin() + 12U);
    // Synthetic image operation, followed by a raw output store so DCE retains its result.
    // Image=s[0:7], sampler=s[8:11], output buffer=s[12:15], coordinates=v[0:1].
    const std::vector<u32> code{
        0xBEEB03FFU, 0U,
        0x7E000280U, 0x7E020280U, 0x7E080280U,
        0xF0000100U | (static_cast<u32>(operation) << 18U),
        (2U << 21U) | (4U << 8U),
        0xBF8C0F70U,
        0xE0700000U, 0x80030400U,
        0xBF810000U,
    };
    const auto program = TranslateProgram(code, fixture.pools, fixture.info, fixture.runtime,
                                          fixture.profile);
    return fixture.info.translation_failed == rejected && !fixture.info.images.empty() ? 0 : 20;
}

struct FetchConversionTranslation : ConversionTranslation {
    static constexpr VAddr CodeAddress = 0x100000U;
    static constexpr VAddr DescriptorAddress = 0x200000U;
    std::vector<u32> fetch;
    AmdGpu::Buffer descriptor{};
    bool failed{};

    FetchConversionTranslation(AmdGpu::DataFormat format, bool instruction_format)
        : ConversionTranslation{Stage::Vertex} {
        runtime.num_user_data = 4U;
        user_data[0] = CodeAddress;
        user_data[2] = DescriptorAddress;
        descriptor.base_address = 0x300000U;
        descriptor.stride = 16U;
        descriptor.num_records = 4U;
        descriptor.data_format = static_cast<u32>(format);
        descriptor.num_format = static_cast<u32>(AmdGpu::NumberFormat::SnormNz);
        descriptor.dst_sel_x = static_cast<u32>(AmdGpu::CompSwizzle::Red);
        descriptor.dst_sel_y = static_cast<u32>(AmdGpu::CompSwizzle::Green);
        descriptor.dst_sel_z = static_cast<u32>(AmdGpu::CompSwizzle::Blue);
        descriptor.dst_sel_w = static_cast<u32>(AmdGpu::CompSwizzle::Alpha);
        const u32 load = instruction_format
                             ? 0xE8000000U |
                                   (static_cast<u32>(Gcn::OpcodeMTBUF::TBUFFER_LOAD_FORMAT_XYZW)
                                    << 16U) |
                                   (static_cast<u32>(AmdGpu::DataFormat::Format32_32_32_32) << 19U) |
                                   (static_cast<u32>(AmdGpu::NumberFormat::Float) << 23U)
                             : 0xE0000000U |
                                   (static_cast<u32>(Gcn::OpcodeMUBUF::BUFFER_LOAD_FORMAT_XYZW)
                                    << 18U);
        fetch = {
            // s_load_dwordx4 s[8:11], s[2:3], 0; load v[4:7]; return s[30:31].
            0xC0000000U | (static_cast<u32>(Gcn::OpcodeSMRD::S_LOAD_DWORDX4) << 22U) |
                (8U << 15U) | (1U << 9U) | (1U << 8U),
            0xBF8C007FU,
            load | (1U << 13U), 0x80020400U,
            0xBF8C0F70U,
            0xBE800000U | (static_cast<u32>(Gcn::OpcodeSOP1::S_SETPC_B64) << 8U) | 30U,
        };
    }

    static bool PS4_SYSV_ABI Read(void* opaque, VAddr address, void* output, u64 size) {
        auto& self = *static_cast<FetchConversionTranslation*>(opaque);
        const auto copy = [&](VAddr base, const void* source, u64 length) {
            if (address < base || address - base > length || size > length - (address - base)) {
                return false;
            }
            std::memcpy(output, static_cast<const u8*>(source) + (address - base), size);
            return true;
        };
        if (copy(CodeAddress, self.fetch.data(), self.fetch.size() * sizeof(u32)) ||
            copy(DescriptorAddress, &self.descriptor, sizeof(self.descriptor))) {
            return true;
        }
        self.failed = true;
        return false;
    }
};

int TranslateFetch(AmdGpu::DataFormat format, bool instruction_format, bool rejected) {
    FetchConversionTranslation fixture{format, instruction_format};
    const std::array<u32, 6> code{
        0xBEEB03FFU, 0U,
        0xBE800000U | (30U << 16U) |
            (static_cast<u32>(Gcn::OpcodeSOP1::S_SWAPPC_B64) << 8U),
        0xF80008CFU, 0x07060504U, // export position from fetched v[4:7]
        0xBF810000U,
    };
    const auto program = TranslateProgram(
        code, fixture.pools, fixture.info, fixture.runtime, fixture.profile,
        SrtMemoryReader{FetchConversionTranslation::Read, &fixture, &fixture.failed});
    return !fixture.failed && fixture.info.has_fetch_shader &&
                   fixture.info.translation_failed == rejected &&
                   (!rejected || program.blocks.empty())
               ? 0
               : 21;
}

TEST(ResourceNumberConversion, UnsupportedSampleIsRejectedBeforeConversion) {
    EXPECT_EXIT(std::_Exit(TranslateImage(Gcn::OpcodeMIMG::IMAGE_SAMPLE_LZ, UnsupportedFormat, true)),
                ::testing::ExitedWithCode(0), "");
}

TEST(ResourceNumberConversion, UnsupportedImageReadIsRejectedBeforeConversion) {
    EXPECT_EXIT(std::_Exit(TranslateImage(Gcn::OpcodeMIMG::IMAGE_LOAD, UnsupportedFormat, true)),
                ::testing::ExitedWithCode(0), "");
}

TEST(ResourceNumberConversion, UnsupportedImageWriteIsRejectedBeforeConversion) {
    EXPECT_EXIT(std::_Exit(TranslateImage(Gcn::OpcodeMIMG::IMAGE_STORE, UnsupportedFormat, true)),
                ::testing::ExitedWithCode(0), "");
}

TEST(ResourceNumberConversion, SupportedImageConsumersStillTranslate) {
    for (const auto operation : {Gcn::OpcodeMIMG::IMAGE_SAMPLE_LZ, Gcn::OpcodeMIMG::IMAGE_LOAD,
                                 Gcn::OpcodeMIMG::IMAGE_STORE}) {
        EXPECT_EXIT(std::_Exit(TranslateImage(operation, SupportedFormat, false)),
                    ::testing::ExitedWithCode(0), "");
    }
}

TEST(ResourceNumberConversion, UnsupportedFetchConversionIsRejected) {
    EXPECT_EXIT(std::_Exit(TranslateFetch(UnsupportedFormat, false, true)),
                ::testing::ExitedWithCode(0), "");
}

TEST(ResourceNumberConversion, SupportedFetchConversionStillTranslates) {
    EXPECT_EXIT(std::_Exit(TranslateFetch(SupportedFormat, false, false)),
                ::testing::ExitedWithCode(0), "");
}

TEST(ResourceNumberConversion, FetchInstructionFormatOverridesUnsupportedDescriptorBits) {
    EXPECT_EXIT(std::_Exit(TranslateFetch(UnsupportedFormat, true, false)),
                ::testing::ExitedWithCode(0), "");
}

int CheckDynamicCompatibility(bool unsupported_model, bool unsupported_candidate) {
    Info info{};
    info.flattened_ud_buf.resize(16U);
    const auto model = MakeImage(unsupported_model ? UnsupportedFormat : SupportedFormat);
    const auto candidate = MakeImage(unsupported_candidate ? UnsupportedFormat : SupportedFormat);
    std::memcpy(info.flattened_ud_buf.data(), &model, sizeof(model));
    std::memcpy(info.flattened_ud_buf.data() + 8U, &candidate, sizeof(candidate));
    const ImageResource resource{.sharp_idx = 0U,
                                 .dynamic_descriptor_count = 2U,
                                 .dynamic_descriptor_stride_dw = 8U};
    if (IsSupportedDynamicImageModel(model, false) != !unsupported_model) {
        return 22;
    }
    return resource.IsTableEntryCompatible(info, 1U) ==
                   !(unsupported_model || unsupported_candidate)
               ? 0
               : 23;
}

TEST(ResourceNumberConversion, UnsupportedDynamicModelIsRefused) {
    EXPECT_EXIT(std::_Exit(CheckDynamicCompatibility(true, false)),
                ::testing::ExitedWithCode(0), "");
}

TEST(ResourceNumberConversion, UnsupportedDynamicEntryIsRefusedWithoutAssertion) {
    EXPECT_EXIT(std::_Exit(CheckDynamicCompatibility(false, true)),
                ::testing::ExitedWithCode(0), "");
}

TEST(ResourceNumberConversion, SupportedDynamicConversionsRemainCompatible) {
    EXPECT_EXIT(std::_Exit(CheckDynamicCompatibility(false, false)),
                ::testing::ExitedWithCode(0), "");
}

} // namespace
} // namespace Shader
