// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "instructions.hpp"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"

namespace {
using namespace Shader;

struct FetchMappingFixture {
    static constexpr VAddr CodeAddress = 0x18000;
    static constexpr VAddr TableAddress = 0x19000;
    Info info{};
    RuntimeInfo runtime{};
    Profile profile{};
    Pools pools{};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    std::vector<u32> code;
    AmdGpu::Buffer descriptor{};
    u32 code_reads{};
    u32 descriptor_reads{};
    bool read_failed{};

    explicit FetchMappingFixture(bool attribute = false, bool direct_descriptor = false) {
        info.stage = Stage::Vertex;
        info.l_stage = LogicalStage::Vertex;
        info.user_data = user_data;
        runtime.Initialize(Stage::Vertex);
        runtime.num_user_data = NUM_USER_DATA_REGS;
        user_data[0] = static_cast<u32>(CodeAddress);
        user_data[4] = static_cast<u32>(TableAddress);
        descriptor.base_address = 0x100000;
        descriptor.stride = 16;
        descriptor.num_records = 16;
        descriptor.data_format = static_cast<u32>(AmdGpu::DataFormat::Format32_32_32_32);
        descriptor.num_format = static_cast<u32>(AmdGpu::NumberFormat::Float);
        if (attribute) {
            // Decode a small fetch program using the real parser: one four-word V# load, one
            // formatted attribute load, then return. The direct variant characterizes the
            // existing ReadUdReg sentinel contract for direct four-word userdata descriptors.
            const u32 base = direct_descriptor ? IR::NumScalarRegs : 4U;
            const u32 offset = direct_descriptor ? 4U : 0U;
            code.push_back(0xc0000000U | (2U << 22) | (8U << 15) | ((base / 2) << 9) | (1U << 8) |
                           offset);
            code.push_back(0xe0000000U | (3U << 18) | (1U << 13));
            code.push_back((4U << 8) | (2U << 16) | (128U << 24));
            if (direct_descriptor) {
                std::memcpy(&user_data[4], &descriptor, sizeof(descriptor));
            }
        }
        code.push_back(SOP1(OpcodeSOP1::S_SETPC_B64, SOperand7::S0, SOperand8::S0).Get());
    }

    static bool PS4_SYSV_ABI Read(void* context, VAddr address, void* output, u64 size) {
        auto& self = *static_cast<FetchMappingFixture*>(context);
        const auto code_bytes = std::as_bytes(std::span{self.code});
        if (address >= CodeAddress && address - CodeAddress <= code_bytes.size() &&
            size <= code_bytes.size() - (address - CodeAddress)) {
            ++self.code_reads;
            std::memcpy(output, code_bytes.data() + (address - CodeAddress), size);
            return true;
        }
        if (address == TableAddress && size == sizeof(self.descriptor)) {
            ++self.descriptor_reads;
            std::memcpy(output, &self.descriptor, size);
            return true;
        }
        self.read_failed = true;
        return false;
    }

    void Emit() {
        const SrtMemoryReader reader{Read, this, &read_failed};
        auto* block = pools.block_pool.Create(pools.inst_pool);
        Gcn::Translator translator{info, runtime, profile, reader};
        translator.EmitPrologue(block);
        Gcn::GcnInst call{};
        call.src[0].code = 0;
        translator.EmitFetch(call);
    }
};

TEST(DrawUserDataFetch, EitherDynamicCodePointerHalfDeclinesBeforeAnyGuestRead) {
    for (u8 location : {u8{0}, u8{1}}) {
        FetchMappingFixture fixture;
        fixture.runtime.vs_info.draw_user_data.base_instance = location;
        fixture.Emit();
        EXPECT_TRUE(fixture.info.translation_failed);
        EXPECT_EQ(fixture.code_reads, 0U);
        EXPECT_EQ(fixture.descriptor_reads, 0U);
        EXPECT_FALSE(fixture.read_failed);
    }
}

TEST(DrawUserDataFetch, EitherDynamicTablePointerHalfDeclinesBeforeDescriptorRead) {
    for (u8 location : {u8{4}, u8{5}}) {
        FetchMappingFixture fixture{true};
        fixture.runtime.vs_info.draw_user_data.base_instance = location;
        fixture.Emit();
        EXPECT_TRUE(fixture.info.translation_failed);
        EXPECT_GT(fixture.code_reads, 0U);
        EXPECT_EQ(fixture.descriptor_reads, 0U);
        EXPECT_FALSE(fixture.read_failed);
    }
}

TEST(DrawUserDataFetch, EveryDynamicDirectDescriptorWordDeclinesBeforeBinding) {
    for (u8 location : {u8{4}, u8{5}, u8{6}, u8{7}}) {
        FetchMappingFixture fixture{true, true};
        fixture.runtime.vs_info.draw_user_data.base_instance = location;
        fixture.Emit();
        EXPECT_TRUE(fixture.info.translation_failed);
        EXPECT_GT(fixture.code_reads, 0U);
        EXPECT_EQ(fixture.descriptor_reads, 0U);
        EXPECT_FALSE(fixture.read_failed);
    }
}

TEST(DrawUserDataFetch, MappingOutsideSourceSpansPreservesOptimizedFetch) {
    for (u8 location : {u8{2}, u8{8}, DrawUserDataLocations::Disabled}) {
        FetchMappingFixture fixture{true};
        fixture.runtime.vs_info.draw_user_data.base_instance = location;
        fixture.Emit();
        EXPECT_FALSE(fixture.info.translation_failed);
        EXPECT_GT(fixture.code_reads, 0U);
        EXPECT_EQ(fixture.descriptor_reads, 1U);
        EXPECT_FALSE(fixture.read_failed);
    }
}

TEST(DrawUserDataFetch, UnmappedDirectDescriptorRemainsHostReadable) {
    FetchMappingFixture fixture{true, true};
    fixture.Emit();
    EXPECT_FALSE(fixture.info.translation_failed);
    EXPECT_GT(fixture.code_reads, 0U);
    EXPECT_EQ(fixture.descriptor_reads, 0U);
    EXPECT_FALSE(fixture.read_failed);
}

TEST(DrawUserDataFetchProbe, CachedProbeRejectsDynamicCodePointerBeforeHostRead) {
    FetchMappingFixture fixture{true};
    fixture.info.has_fetch_shader = true;
    fixture.info.fetch_shader_sgpr_base = 0;
    fixture.runtime.vs_info.draw_user_data.base_instance = 1;
    const SrtMemoryReader reader{FetchMappingFixture::Read, &fixture, &fixture.read_failed};
    const auto spec =
        TryCreateStageSpecialization(fixture.info, fixture.runtime, fixture.profile, {}, reader);
    EXPECT_FALSE(spec.has_value());
    EXPECT_EQ(fixture.code_reads, 0U);
    EXPECT_EQ(fixture.descriptor_reads, 0U);
}

TEST(DrawUserDataFetchProbe, ChangedFetchTablePointerCannotBypassFreshTranslationAdmission) {
    // This is the same factory/constructor called for the cached aggregate probe before a fresh
    // TranslateModule. The previous main-program translation does not certify changed fetch code.
    FetchMappingFixture fixture{true};
    fixture.info.has_fetch_shader = true;
    fixture.info.fetch_shader_sgpr_base = 0;
    fixture.runtime.vs_info.draw_user_data.base_instance = 4;
    const SrtMemoryReader reader{FetchMappingFixture::Read, &fixture, &fixture.read_failed};
    const auto spec =
        TryCreateStageSpecialization(fixture.info, fixture.runtime, fixture.profile, {}, reader);
    EXPECT_FALSE(spec.has_value());
    EXPECT_GT(fixture.code_reads, 0U);
    EXPECT_EQ(fixture.descriptor_reads, 0U);
}

TEST(DrawUserDataFetchProbe, ChangedDirectDescriptorIncludesAllFourUserDataWords) {
    FetchMappingFixture fixture{true, true};
    fixture.info.has_fetch_shader = true;
    fixture.info.fetch_shader_sgpr_base = 0;
    fixture.runtime.vs_info.draw_user_data.base_instance = 7;
    const SrtMemoryReader reader{FetchMappingFixture::Read, &fixture, &fixture.read_failed};
    const auto spec =
        TryCreateStageSpecialization(fixture.info, fixture.runtime, fixture.profile, {}, reader);
    EXPECT_FALSE(spec.has_value());
    EXPECT_GT(fixture.code_reads, 0U);
    EXPECT_EQ(fixture.descriptor_reads, 0U);
}

TEST(DrawUserDataFetchProbe, NonoverlappingMappedAndDisabledProbesKeepCapturedAttributes) {
    for (u8 location : {u8{2}, u8{8}, DrawUserDataLocations::Disabled}) {
        FetchMappingFixture fixture{true};
        fixture.info.has_fetch_shader = true;
        fixture.info.fetch_shader_sgpr_base = 0;
        fixture.runtime.vs_info.draw_user_data.base_instance = location;
        const SrtMemoryReader reader{FetchMappingFixture::Read, &fixture, &fixture.read_failed};
        const auto spec = TryCreateStageSpecialization(fixture.info, fixture.runtime,
                                                       fixture.profile, {}, reader);
        ASSERT_TRUE(spec.has_value());
        ASSERT_TRUE(spec->fetch_shader_data.has_value());
        EXPECT_EQ(spec->vs_attribs.size(), 1U);
        EXPECT_EQ(fixture.code_reads, 4U);
        EXPECT_EQ(fixture.descriptor_reads, 1U);
        EXPECT_FALSE(fixture.read_failed);
    }
}

} // namespace
