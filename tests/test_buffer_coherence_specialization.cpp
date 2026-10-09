// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "common/serdes.h"
#include "shader_recompiler/specialization_identity.h"
#include "shader_recompiler/specialization_serialization.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

namespace {
using namespace Shader;

Info MakeInfo(u32 mtype, bool written = true, BufferType type = BufferType::Guest,
              bool inline_descriptor = true) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.pgm_hash = 0x1234;
    info.flattened_ud_buf.resize(NUM_USER_DATA_REGS);
    const auto sharp = AmdGpu::Buffer::Placeholder(256);
    auto words = std::bit_cast<std::array<u32, 4>>(sharp);
    words[3] |= mtype << 27U;
    std::memcpy(info.flattened_ud_buf.data(), words.data(), sizeof(words));
    info.buffers.push_back(
        {.sharp_idx = 0,
         .used_types = IR::Type::U32,
         .inline_cbuf = inline_descriptor ? std::bit_cast<AmdGpu::Buffer>(words) : AmdGpu::Buffer{},
         .buffer_type = type,
         .is_written = written});
    return info;
}

StageSpecialization MakeSpec(const Info& info) {
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Compute);
    runtime.cs_info.workgroup_size = {1, 1, 1};
    return StageSpecialization{info, runtime, Profile{}, Backend::Bindings{}};
}

u64 Identity(const StageSpecialization& spec) {
    return ComputeShaderPermutationIdentity(0x1234, LogicalStage::Compute, spec);
}

std::vector<u8> Encode(const StageSpecialization& spec) {
    Serialization::Archive archive;
    spec.Serialize(archive);
    return archive.TakeOff();
}

TEST(BufferCoherenceSpecialization, ConsumedRuntimeMtypeChangesEqualityIdentityAndPreloadChoice) {
    auto old_info = MakeInfo(0, true, BufferType::Guest, false);
    auto old_spec = MakeSpec(old_info);
    const auto old_id = Identity(old_spec);
    auto new_info = MakeInfo(3, true, BufferType::Guest, false);
    const auto new_spec = MakeSpec(new_info);
    EXPECT_NE(old_spec, new_spec);
    EXPECT_NE(old_id, Identity(new_spec));
    Vulkan::Program program;
    program.AddPermut({}, std::move(old_spec), std::move(old_info), old_id);
    EXPECT_EQ(program.InspectPreloadPermutation(new_spec, Identity(new_spec)).action,
              Vulkan::Program::PreloadAction::Insert);
    EXPECT_EQ(program.InspectPreloadPermutation(new_spec, old_id).action,
              Vulkan::Program::PreloadAction::Conflict);
}

TEST(BufferCoherenceSpecialization, UnconsumedMtypesCollapseForReadOnlyAndSharedMemory) {
    for (const auto mode :
         {std::pair{false, BufferType::Guest}, std::pair{true, BufferType::SharedMemory}}) {
        const auto baseline_info = MakeInfo(0, mode.first, mode.second);
        const auto baseline = MakeSpec(baseline_info);
        for (u32 mtype = 0; mtype < 8; ++mtype) {
            const auto info = MakeInfo(mtype, mode.first, mode.second);
            const auto spec = MakeSpec(info);
            EXPECT_EQ(baseline, spec);
            EXPECT_EQ(Identity(baseline), Identity(spec));
            EXPECT_EQ(Encode(baseline), Encode(spec));
        }
    }
}

TEST(BufferCoherenceSpecialization, OtherWrittenMtypesCollapseToOneNoncoherentPermutation) {
    const auto baseline_info = MakeInfo(0);
    const auto baseline = MakeSpec(baseline_info);
    for (const u32 mtype : {1U, 2U, 4U, 5U, 6U, 7U}) {
        const auto info = MakeInfo(mtype);
        const auto spec = MakeSpec(info);
        EXPECT_EQ(baseline, spec);
        EXPECT_EQ(Identity(baseline), Identity(spec));
        EXPECT_EQ(Encode(baseline), Encode(spec));
    }
}

TEST(BufferCoherenceSpecialization, NullRuntimeBindingDoesNotEraseEmittedCoherence) {
    auto normal_info = MakeInfo(0, true, BufferType::Guest, false);
    auto coherent_info = MakeInfo(3, true, BufferType::Guest, false);
    normal_info.flattened_ud_buf[2] = 0; // NUM_RECORDS, not the MTYPE dword.
    coherent_info.flattened_ud_buf[2] = 0;
    const auto normal = MakeSpec(normal_info);
    const auto coherent = MakeSpec(coherent_info);
    EXPECT_FALSE(normal.bitset.test(0));
    EXPECT_FALSE(coherent.bitset.test(0));
    EXPECT_FALSE(normal.buffers[0].is_coherent);
    EXPECT_TRUE(coherent.buffers[0].is_coherent);
    EXPECT_NE(normal, coherent);
    EXPECT_NE(Identity(normal), Identity(coherent));
}

TEST(BufferCoherenceSpecialization, UniformRecordCannotClaimStorageCoherence) {
    const auto info = MakeInfo(3);
    auto spec = MakeSpec(info);
    ASSERT_TRUE(spec.buffers[0].is_coherent);
    spec.buffers[0].is_storage = false;
    Serialization::Archive input{Encode(spec)};
    StageSpecialization decoded;
    decoded.user_data_mask = 0x5a;
    EXPECT_FALSE(decoded.Deserialize(input));
    EXPECT_EQ(decoded.user_data_mask, 0x5a);
    EXPECT_TRUE(decoded.buffers.empty());
}

TEST(BufferCoherenceSpecialization, WireFormatRetainsEffectiveFlagAndRejectsNonBoolean) {
    const auto normal_info = MakeInfo(0);
    const auto coherent_info = MakeInfo(3);
    const auto normal = MakeSpec(normal_info);
    const auto coherent = MakeSpec(coherent_info);
    auto normal_bytes = Encode(normal);
    auto coherent_bytes = Encode(coherent);
    ASSERT_EQ(normal_bytes.size(), coherent_bytes.size());
    std::vector<size_t> differences;
    for (size_t i = 0; i < normal_bytes.size(); ++i) {
        if (normal_bytes[i] != coherent_bytes[i]) {
            differences.push_back(i);
        }
    }
    ASSERT_EQ(differences.size(), 1U);
    EXPECT_EQ(normal_bytes[differences[0]], 0);
    EXPECT_EQ(coherent_bytes[differences[0]], 1);
    StageSpecialization decoded;
    Serialization::Archive valid{std::move(coherent_bytes)};
    ASSERT_TRUE(decoded.Deserialize(valid));
    EXPECT_EQ(decoded.buffers, coherent.buffers);
    EXPECT_EQ(Identity(decoded), Identity(coherent));
    normal_bytes[differences[0]] = 2;
    Serialization::Archive invalid{std::move(normal_bytes)};
    const auto old_buffers = decoded.buffers;
    EXPECT_FALSE(decoded.Deserialize(invalid));
    EXPECT_EQ(decoded.buffers, old_buffers);
}

TEST(BufferCoherenceSpecialization, MetadataRoundTripsRawDescriptorAndEffectiveSpecialization) {
    for (const auto type : {BufferType::Guest, BufferType::SharedMemory}) {
        auto info = MakeInfo(type == BufferType::Guest ? 3 : 0, true, type);
        auto spec = MakeSpec(info);
        ASSERT_TRUE(spec.buffers[0].is_coherent);
        Serialization::Archive encoded;
        ASSERT_TRUE(SerializeShaderCacheMetadata(encoded, info, spec, Identity(spec)));
        Serialization::Archive input{encoded.TakeOff()};
        Info decoded_info{};
        StageSpecialization decoded_spec{};
        std::optional<Gcn::FetchShaderData> fetch;
        u64 identity{};
        ASSERT_TRUE(
            DeserializeShaderCacheMetadata(input, decoded_info, fetch, decoded_spec, identity));
        EXPECT_EQ(decoded_info.buffers[0].inline_cbuf, info.buffers[0].inline_cbuf);
        EXPECT_EQ(decoded_spec, MakeSpec(decoded_info));
        EXPECT_EQ(identity, Identity(spec));
        EXPECT_EQ(decoded_spec.info, &decoded_info);
    }
}

TEST(BufferCoherenceSpecialization, MismatchedMetadataRefusesBeforeAppendingOrPublishing) {
    auto info = MakeInfo(0);
    const auto spec = MakeSpec(info);
    info.buffers[0].inline_cbuf = MakeInfo(3).buffers[0].inline_cbuf;
    EXPECT_FALSE(ValidateShaderCacheMetadata(info, spec));
    Serialization::Archive encoded;
    EXPECT_FALSE(SerializeShaderCacheMetadata(encoded, info, spec, Identity(spec)));
    EXPECT_TRUE(encoded.TakeOff().empty());
    // Bypass encoder validation to test the independently untrusted decoder boundary.
    Serialization::Archive malformed;
    Serialization::Writer writer{malformed};
    writer.Write(ShaderMetaVersion);
    writer.Write(ShaderBinaryVersion);
    writer.Write(Identity(spec));
    spec.Serialize(malformed);
    info.Serialize(malformed);
    Serialization::Archive input{malformed.TakeOff()};
    Info output_info{};
    output_info.pgm_hash = 0xdead;
    StageSpecialization output_spec{};
    output_spec.info = &output_info;
    std::optional<Gcn::FetchShaderData> fetch;
    u64 identity = 0xbeef;
    EXPECT_FALSE(DeserializeShaderCacheMetadata(input, output_info, fetch, output_spec, identity));
    EXPECT_EQ(output_info.pgm_hash, 0xdead);
    EXPECT_TRUE(output_info.buffers.empty());
    EXPECT_EQ(output_spec.info, &output_info);
    EXPECT_EQ(identity, 0xbeef);
}

TEST(BufferCoherenceSpecialization, PreviousBinaryAndMetadataVersionsCannotPreload) {
    const auto info = MakeInfo(3);
    const auto spec = MakeSpec(info);
    for (const size_t header_word : {0U, 1U}) {
        Serialization::Archive encoded;
        ASSERT_TRUE(SerializeShaderCacheMetadata(encoded, info, spec, Identity(spec)));
        auto bytes = encoded.TakeOff();
        const u32 previous_version = 9;
        std::memcpy(bytes.data() + header_word * sizeof(u32), &previous_version,
                    sizeof(previous_version));
        Serialization::Archive input{std::move(bytes)};
        Info output_info{};
        output_info.pgm_hash = 0xdead;
        StageSpecialization output_spec{};
        output_spec.info = &output_info;
        std::optional<Gcn::FetchShaderData> fetch;
        u64 identity = 0xbeef;
        EXPECT_FALSE(
            DeserializeShaderCacheMetadata(input, output_info, fetch, output_spec, identity));
        EXPECT_EQ(output_info.pgm_hash, 0xdead);
        EXPECT_EQ(output_spec.info, &output_info);
        EXPECT_EQ(identity, 0xbeef);
    }
}
} // namespace
