// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "common/serdes.h"
#include "shader_recompiler/specialization.h"
#include "shader_recompiler/specialization_identity.h"
#include "shader_recompiler/specialization_serialization.h"

namespace Shader {
namespace {

Info MakeFmaskInfo(bool is_r128, u32 snapshot_words) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.pgm_hash = 0x1234U;
    auto image = AmdGpu::Image::Null(false);
    image.base_address = 0x1000U;
    image.data_format = static_cast<u32>(AmdGpu::DataFormat::FormatFmask8_S2_F1);
    image.num_format = static_cast<u32>(AmdGpu::NumberFormat::Uint);
    image.type = static_cast<u64>(AmdGpu::ImageType::Color2DMsaa);
    image.width = 63U;
    image.height = 31U;
    info.flattened_ud_buf.resize(NUM_USER_DATA_REGS + snapshot_words);
    info.srt_info.flattened_bufsize_dw = static_cast<u32>(info.flattened_ud_buf.size());
    std::memcpy(info.flattened_ud_buf.data() + NUM_USER_DATA_REGS, &image,
                snapshot_words * sizeof(u32));
    info.fmasks.push_back({.sharp_idx = NUM_USER_DATA_REGS, .is_r128 = is_r128});
    return info;
}

StageSpecialization MakeFmaskSpecialization(const Info& info) {
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Compute);
    runtime.cs_info.workgroup_size = {1U, 1U, 1U};
    return StageSpecialization{info, runtime, Profile{}, Backend::Bindings{}};
}

std::vector<u8> EncodeFmaskMetadata(const Info& info, const StageSpecialization& spec) {
    Serialization::Archive encoded;
    EXPECT_TRUE(SerializeShaderCacheMetadata(
        encoded, info, spec, ComputeShaderPermutationIdentity(info.pgm_hash, info.l_stage, spec)));
    return encoded.TakeOff();
}

TEST(FMaskDescriptorMetadata, CompilerAndPersistentDescriptorWidthsInvalidateOldCaches) {
    EXPECT_GE(ShaderBinaryVersion, 12U);
    EXPECT_GE(ShaderMetaVersion, 12U);
}

TEST(FMaskDescriptorMetadata, R128DescriptorAtSnapshotEndValidatesAndRoundTrips) {
    const auto info = MakeFmaskInfo(true, 4U);
    const auto spec = MakeFmaskSpecialization(info);
    ASSERT_TRUE(ValidateShaderCacheMetadata(info, spec));

    Serialization::Archive input{EncodeFmaskMetadata(info, spec)};
    Info decoded_info{};
    StageSpecialization decoded_spec{};
    std::optional<Gcn::FetchShaderData> fetch;
    u64 identity{};
    ASSERT_TRUE(DeserializeShaderCacheMetadata(input, decoded_info, fetch, decoded_spec, identity));
    ASSERT_EQ(decoded_info.fmasks.size(), 1U);
    EXPECT_TRUE(decoded_info.fmasks[0].is_r128);
    EXPECT_EQ(decoded_info.flattened_ud_buf.size(), NUM_USER_DATA_REGS + 4U);
    EXPECT_EQ(decoded_spec, MakeFmaskSpecialization(decoded_info));
    EXPECT_EQ(decoded_spec.fmasks[0].width, 63U);
    EXPECT_EQ(decoded_spec.fmasks[0].height, 31U);
}

TEST(FMaskDescriptorMetadata, FullDescriptorStillRequiresAllEightWords) {
    auto info = MakeFmaskInfo(false, 8U);
    const auto spec = MakeFmaskSpecialization(info);
    EXPECT_TRUE(ValidateShaderCacheMetadata(info, spec));
    info.flattened_ud_buf.resize(NUM_USER_DATA_REGS + 4U);
    info.srt_info.flattened_bufsize_dw = static_cast<u32>(info.flattened_ud_buf.size());
    EXPECT_FALSE(ValidateShaderCacheMetadata(info, spec));
}

TEST(FMaskDescriptorMetadata, TruncatedR128DescriptorFailsBeforePublishing) {
    auto info = MakeFmaskInfo(true, 4U);
    const auto spec = MakeFmaskSpecialization(info);
    info.flattened_ud_buf.pop_back();
    --info.srt_info.flattened_bufsize_dw;
    EXPECT_FALSE(ValidateShaderCacheMetadata(info, spec));
}

TEST(FMaskDescriptorMetadata, WidthFlagIsPersistedAndNonBooleanValueIsRejected) {
    auto info = MakeFmaskInfo(false, 8U);
    const auto spec = MakeFmaskSpecialization(info);
    const auto full = EncodeFmaskMetadata(info, spec);
    info.fmasks[0].is_r128 = true;
    auto short_descriptor = EncodeFmaskMetadata(info, spec);
    ASSERT_EQ(full.size(), short_descriptor.size());
    std::vector<size_t> differences;
    for (size_t index = 0; index < full.size(); ++index) {
        if (full[index] != short_descriptor[index]) {
            differences.push_back(index);
        }
    }
    ASSERT_EQ(differences.size(), 1U);
    EXPECT_EQ(full[differences[0]], 0U);
    EXPECT_EQ(short_descriptor[differences[0]], 1U);
    short_descriptor[differences[0]] = 2U;

    Serialization::Archive invalid{std::move(short_descriptor)};
    Info decoded_info{};
    decoded_info.pgm_hash = 0xdeadbeefU;
    StageSpecialization decoded_spec{};
    std::optional<Gcn::FetchShaderData> fetch;
    u64 identity = 0xabcdefU;
    EXPECT_FALSE(
        DeserializeShaderCacheMetadata(invalid, decoded_info, fetch, decoded_spec, identity));
    EXPECT_EQ(decoded_info.pgm_hash, 0xdeadbeefU);
    EXPECT_EQ(identity, 0xabcdefU);
    EXPECT_TRUE(decoded_info.fmasks.empty());
}

} // namespace
} // namespace Shader
