// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/opcodes.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"

namespace Shader {
namespace {

enum class FormatSource { Raw, Instruction, Descriptor };

// Only synthetic instructions and owned user-data descriptors are used here. The load result is
// stored through the same descriptor so production DCE must retain the operation and resource
// tracking must combine the load with raw-store access to that resource.
int TranslateAndSpecialize(FormatSource source, AmdGpu::DataFormat data_format,
                           AmdGpu::NumberFormat number_format,
                           AmdGpu::NumberConversion expected_conversion,
                           bool mixed_descriptor_access = false, bool descriptor_first = false) {
    EmulatorSettings.SetDirectMemoryAccessEnabled(false);
    EmulatorSettings.SetDumpShaders(false);
    AmdGpu::Buffer descriptor{};
    descriptor.base_address = 0x100000U;
    descriptor.stride = 4U;
    descriptor.num_records = 16U;
    descriptor.data_format = static_cast<u32>(data_format);
    descriptor.num_format = static_cast<u32>(number_format);
    descriptor.dst_sel_x = static_cast<u32>(AmdGpu::CompSwizzle::Red);
    descriptor.dst_sel_y = static_cast<u32>(AmdGpu::CompSwizzle::Green);
    descriptor.dst_sel_z = static_cast<u32>(AmdGpu::CompSwizzle::Blue);
    descriptor.dst_sel_w = static_cast<u32>(AmdGpu::CompSwizzle::Alpha);
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    const auto words = std::bit_cast<std::array<u32, 4>>(descriptor);
    std::copy(words.begin(), words.end(), user_data.begin());

    u32 load{};
    switch (source) {
    case FormatSource::Raw:
        load = 0xE0000000U | (static_cast<u32>(Gcn::OpcodeMUBUF::BUFFER_LOAD_DWORD) << 18U);
        break;
    case FormatSource::Instruction:
        // tbuffer_load_format_x v0, v0, s[0:3], 0, format32/float.
        load = 0xE8000000U | (static_cast<u32>(Gcn::OpcodeMTBUF::TBUFFER_LOAD_FORMAT_X) << 16U) |
               (static_cast<u32>(AmdGpu::DataFormat::Format32) << 19U) |
               (static_cast<u32>(AmdGpu::NumberFormat::Float) << 23U);
        break;
    case FormatSource::Descriptor:
        load = 0xE0000000U | (static_cast<u32>(Gcn::OpcodeMUBUF::BUFFER_LOAD_FORMAT_X) << 18U);
        break;
    }
    std::vector<u32> code{
        0xBEEB03FFU, 0U, // s_mov_b32 vcc_hi, literal(0)
        load,        0x80000000U,
        0xBF8C0F70U,              // s_waitcnt vmcnt(0)
        0xE0700000U, 0x80000000U, // buffer_store_dword v0, v0, s[0:3], 0
        0xBF810000U,              // s_endpgm
    };
    if (mixed_descriptor_access) {
        const std::array<u32, 5> descriptor_access{
            0xE0000000U | (static_cast<u32>(Gcn::OpcodeMUBUF::BUFFER_LOAD_FORMAT_X) << 18U),
            0x80000000U,
            0xBF8C0F70U,
            0xE0700000U,
            0x80000000U,
        };
        const auto position = descriptor_first ? code.begin() + 2 : code.end() - 1;
        code.insert(position, descriptor_access.begin(), descriptor_access.end());
    }
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.user_data = user_data;
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Compute);
    runtime.num_user_data = 4U;
    runtime.cs_info.workgroup_size = {1U, 1U, 1U};
    Profile profile{};
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;
    Pools pools{};
    const auto program = TranslateProgram(code, pools, info, runtime, profile);
    if (info.translation_failed || info.buffers.size() != 1U || !info.buffers[0].is_written ||
        info.buffers[0].is_formatted != (source != FormatSource::Raw)) {
        return 10;
    }
    const auto specialization = TryCreateStageSpecialization(info, runtime, profile, {});
    if (!specialization || specialization->buffers.size() != 1U ||
        specialization->buffers[0].num_conversion != expected_conversion) {
        return 11;
    }
    return 0;
}

int SpecializeDescriptorSnapshot(bool image, FormatSource source, AmdGpu::DataFormat data_format,
                                 bool expected_accept) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.flattened_ud_buf.resize(NUM_USER_DATA_REGS);
    if (image) {
        auto descriptor = AmdGpu::Image::Null(false);
        descriptor.base_address = 0x1000U;
        descriptor.data_format = static_cast<u32>(data_format);
        descriptor.num_format = static_cast<u32>(AmdGpu::NumberFormat::SnormNz);
        const auto words = std::bit_cast<std::array<u32, 8>>(descriptor);
        std::copy(words.begin(), words.end(), info.flattened_ud_buf.begin());
        info.images.push_back({.sharp_idx = 0});
    } else {
        AmdGpu::Buffer descriptor{};
        descriptor.base_address = 0x100000U;
        descriptor.num_records = 16U;
        descriptor.data_format = static_cast<u32>(data_format);
        descriptor.num_format = static_cast<u32>(AmdGpu::NumberFormat::SnormNz);
        const auto words = std::bit_cast<std::array<u32, 4>>(descriptor);
        std::copy(words.begin(), words.end(), info.flattened_ud_buf.begin());
        info.buffers.push_back(
            {.sharp_idx = 0,
             .used_types = IR::Type::U32,
             .buffer_type = BufferType::Guest,
             .is_formatted = source != FormatSource::Raw,
             .is_instruction_formatted_only = source == FormatSource::Instruction});
    }
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Compute);
    const auto specialization = TryCreateStageSpecialization(info, runtime, {}, {});
    return specialization.has_value() == expected_accept ? 0 : 12;
}

TEST(SpecializationNumberConversion, RawAccessDoesNotConsumeUnusedFormatBits) {
    EXPECT_EXIT(std::_Exit(TranslateAndSpecialize(
                    FormatSource::Raw, AmdGpu::DataFormat::Format10_10_10_2,
                    AmdGpu::NumberFormat::SnormNz, AmdGpu::NumberConversion::None)),
                ::testing::ExitedWithCode(0), "");
}

TEST(SpecializationNumberConversion, TypedInstructionOverridesUnusedDescriptorConversion) {
    EXPECT_EXIT(std::_Exit(TranslateAndSpecialize(
                    FormatSource::Instruction, AmdGpu::DataFormat::Format10_10_10_2,
                    AmdGpu::NumberFormat::SnormNz, AmdGpu::NumberConversion::None)),
                ::testing::ExitedWithCode(0), "");
}

TEST(SpecializationNumberConversion, InvalidFormattedResourceRetainsNullLowering) {
    EXPECT_EXIT(std::_Exit(TranslateAndSpecialize(
                    FormatSource::Descriptor, AmdGpu::DataFormat::FormatInvalid,
                    AmdGpu::NumberFormat::SnormNz, AmdGpu::NumberConversion::None)),
                ::testing::ExitedWithCode(0), "");
}

TEST(SpecializationNumberConversion, SupportedDescriptorConversionIsPreserved) {
    EXPECT_EXIT(std::_Exit(TranslateAndSpecialize(
                    FormatSource::Descriptor, AmdGpu::DataFormat::Format8,
                    AmdGpu::NumberFormat::SnormNz, AmdGpu::NumberConversion::Sint8ToSnormNz)),
                ::testing::ExitedWithCode(0), "");
}

TEST(SpecializationNumberConversion, TypedInstructionIgnoresSupportedDescriptorConversion) {
    EXPECT_EXIT(std::_Exit(TranslateAndSpecialize(
                    FormatSource::Instruction, AmdGpu::DataFormat::Format8,
                    AmdGpu::NumberFormat::SnormNz, AmdGpu::NumberConversion::None)),
                ::testing::ExitedWithCode(0), "");
}

TEST(SpecializationNumberConversion, DescriptorThenTypedAliasPreservesRequiredConversion) {
    EXPECT_EXIT(
        std::_Exit(TranslateAndSpecialize(FormatSource::Instruction, AmdGpu::DataFormat::Format8,
                                          AmdGpu::NumberFormat::SnormNz,
                                          AmdGpu::NumberConversion::Sint8ToSnormNz, true, true)),
        ::testing::ExitedWithCode(0), "");
}

TEST(SpecializationNumberConversion, TypedThenDescriptorAliasPreservesRequiredConversion) {
    EXPECT_EXIT(
        std::_Exit(TranslateAndSpecialize(FormatSource::Instruction, AmdGpu::DataFormat::Format8,
                                          AmdGpu::NumberFormat::SnormNz,
                                          AmdGpu::NumberConversion::Sint8ToSnormNz, true, false)),
        ::testing::ExitedWithCode(0), "");
}

TEST(SpecializationNumberConversion, ProbeRefusesUnsupportedConsumedBufferConversion) {
    EXPECT_EXIT(std::_Exit(SpecializeDescriptorSnapshot(
                    false, FormatSource::Descriptor, AmdGpu::DataFormat::Format10_10_10_2, false)),
                ::testing::ExitedWithCode(0), "");
}

TEST(SpecializationNumberConversion, ProbeRefusesUnsupportedImageConversion) {
    EXPECT_EXIT(std::_Exit(SpecializeDescriptorSnapshot(
                    true, FormatSource::Descriptor, AmdGpu::DataFormat::Format10_10_10_2, false)),
                ::testing::ExitedWithCode(0), "");
}

TEST(SpecializationNumberConversion, ProbePreservesTypedAndRawUnusedFormatControls) {
    EXPECT_EXIT(std::_Exit(SpecializeDescriptorSnapshot(
                    false, FormatSource::Instruction, AmdGpu::DataFormat::Format10_10_10_2, true)),
                ::testing::ExitedWithCode(0), "");
    EXPECT_EXIT(std::_Exit(SpecializeDescriptorSnapshot(
                    false, FormatSource::Raw, AmdGpu::DataFormat::Format10_10_10_2, true)),
                ::testing::ExitedWithCode(0), "");
}

TEST(SpecializationNumberConversion, ProbePreservesSupportedBufferAndImageConversions) {
    EXPECT_EXIT(std::_Exit(SpecializeDescriptorSnapshot(false, FormatSource::Descriptor,
                                                        AmdGpu::DataFormat::Format16, true)),
                ::testing::ExitedWithCode(0), "");
    EXPECT_EXIT(std::_Exit(SpecializeDescriptorSnapshot(true, FormatSource::Descriptor,
                                                        AmdGpu::DataFormat::Format32, true)),
                ::testing::ExitedWithCode(0), "");
}

} // namespace
} // namespace Shader
