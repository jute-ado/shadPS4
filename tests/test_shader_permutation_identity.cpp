// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "common/serdes.h"
#include "shader_recompiler/specialization_identity.h"

namespace {

constexpr u64 ProgramHash = 0x73b8c213ULL;

u64 Identity(const Shader::StageSpecialization& spec) {
    return Shader::ComputeShaderPermutationIdentity(ProgramHash, Shader::LogicalStage::Compute,
                                                    spec);
}

u64 Identity(const Shader::StageSpecialization& spec, u64 program_hash,
             Shader::LogicalStage logical_stage) {
    return Shader::ComputeShaderPermutationIdentity(program_hash, logical_stage, spec);
}

AmdGpu::CompMapping RgbaMapping() {
    return AmdGpu::CompMapping{
        .r = AmdGpu::CompSwizzle::Red,
        .g = AmdGpu::CompSwizzle::Green,
        .b = AmdGpu::CompSwizzle::Blue,
        .a = AmdGpu::CompSwizzle::Alpha,
    };
}

AmdGpu::CompMapping RgbZeroMapping() {
    return AmdGpu::CompMapping{
        .r = AmdGpu::CompSwizzle::Red,
        .g = AmdGpu::CompSwizzle::Green,
        .b = AmdGpu::CompSwizzle::Blue,
        .a = AmdGpu::CompSwizzle::Zero,
    };
}

Shader::StageSpecialization MakeImageSpecialization(AmdGpu::NumberClass number_class,
                                                    AmdGpu::CompMapping dst_select,
                                                    bool has_supported_dynamic_model = true) {
    Shader::StageSpecialization spec{};
    spec.runtime_info.Initialize(Shader::Stage::Compute);
    spec.runtime_info.num_user_data = 4;
    spec.runtime_info.cs_info.shared_memory_size = 1024;
    spec.runtime_info.cs_info.workgroup_size = {8, 8, 1};
    spec.runtime_info.cs_info.tgid_enable = {true, true, false};
    spec.start = {.unified = 3, .buffer = 1, .user_data = 2};
    spec.images.push_back({
        .type = AmdGpu::ImageType::Color2D,
        .number_class = number_class,
        .is_storage = true,
        .is_cube = false,
        .is_srgb = false,
        .has_supported_dynamic_model = has_supported_dynamic_model,
        .dst_select = dst_select,
        .num_conversion = AmdGpu::NumberConversion::None,
        .num_bindings = 1,
    });
    return spec;
}

void ExpectSameSemantic(Shader::StageSpecialization first, Shader::StageSpecialization second) {
    Shader::Info first_info{};
    Shader::Info second_info{};
    first.info = &first_info;
    second.info = &second_info;
    EXPECT_EQ(first, second);
    EXPECT_EQ(second, first);
    EXPECT_EQ(Identity(first), Identity(second));
}

void ExpectDifferentSemantic(Shader::StageSpecialization first,
                             Shader::StageSpecialization second) {
    Shader::Info first_info{};
    Shader::Info second_info{};
    first.info = &first_info;
    second.info = &second_info;
    EXPECT_NE(first, second);
    EXPECT_NE(second, first);
    EXPECT_NE(Identity(first), Identity(second));
}

std::string ReadTextFile(const char* path) {
    std::ifstream file{path};
    std::ostringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

TEST(ShaderPermutationIdentity, EncounterOrderDoesNotChangePersistedIds) {
    const auto float_rgba = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    const auto uint_rgba = MakeImageSpecialization(AmdGpu::NumberClass::Uint, RgbaMapping());

    const std::array ab{Identity(float_rgba), Identity(uint_rgba)};
    const std::array ba{Identity(uint_rgba), Identity(float_rgba)};

    EXPECT_EQ(ab[0], ba[1]);
    EXPECT_EQ(ab[1], ba[0]);
    EXPECT_NE(ab[0], ab[1]);
}

TEST(StageSpecializationSerialization, GeometryCopyShaderSurvivesSourceDestruction) {
    constexpr std::array<u32, 6> CopyShader{0xBE802000U, 0xBF8C0F70U, 0x7E000200U,
                                             0xD1FF0000U, 0x00000100U, 0xBF810000U};
    Serialization::Archive output;
    u64 original_identity{};
    {
        std::vector<u32> temporary_copy{CopyShader.begin(), CopyShader.end()};
        Shader::StageSpecialization source{};
        source.runtime_info.Initialize(Shader::Stage::Geometry);
        source.runtime_info.gs_info.vs_copy = temporary_copy;
        source.runtime_info.gs_info.vs_copy_hash = 0xA11CE5EEDULL;
        original_identity = Identity(source, ProgramHash, Shader::LogicalStage::Geometry);
        source.Serialize(output);
    }

    // Reuse freed allocations so a restored raw span cannot accidentally appear valid.
    std::vector<std::vector<u32>> heap_churn(256, std::vector<u32>(CopyShader.size(), 0xDEADBEEFU));
    ASSERT_FALSE(heap_churn.empty());

    Serialization::Archive input{output.TakeOff()};
    Shader::StageSpecialization retained{};
    {
        Shader::StageSpecialization restored{};
        ASSERT_TRUE(restored.Deserialize(input));
        retained = restored;
    }
    heap_churn.emplace_back(CopyShader.size(), 0xBAD0C0DEU);
    ASSERT_EQ(retained.runtime_info.stage, Shader::Stage::Geometry);
    EXPECT_TRUE(std::ranges::equal(retained.runtime_info.gs_info.vs_copy, CopyShader));
    EXPECT_EQ(Identity(retained, ProgramHash, Shader::LogicalStage::Geometry), original_identity);
    EXPECT_TRUE(input.IsEoS());
}

TEST(ShaderPermutationIdentity, GeometryCopyContentsParticipateBeyondCachedHash) {
    std::array<u32, 2> first_copy{0xAAAAAAAAU, 0xBBBBBBBBU};
    std::array<u32, 2> second_copy{0xAAAAAAAAU, 0xCCCCCCCCU};
    Shader::StageSpecialization first{};
    first.runtime_info.Initialize(Shader::Stage::Geometry);
    first.runtime_info.gs_info.vs_copy = first_copy;
    first.runtime_info.gs_info.vs_copy_hash = 0x1234U;
    auto second = first;
    second.runtime_info.gs_info.vs_copy = second_copy;

    EXPECT_NE(Identity(first, ProgramHash, Shader::LogicalStage::Geometry),
              Identity(second, ProgramHash, Shader::LogicalStage::Geometry));
    EXPECT_NE(first.runtime_info, second.runtime_info);
}

TEST(ShaderPermutationIdentity, ProgramAndLogicalStageParticipate) {
    const auto spec = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());

    EXPECT_NE(Identity(spec, ProgramHash, Shader::LogicalStage::Compute),
              Identity(spec, ProgramHash + 1, Shader::LogicalStage::Compute));
    EXPECT_NE(Identity(spec, ProgramHash, Shader::LogicalStage::Compute),
              Identity(spec, ProgramHash, Shader::LogicalStage::Vertex));
}

TEST(ShaderPermutationIdentity, ProgramCacheKeySeparatesPhysicalAndLogicalStages) {
    const Shader::ProgramCacheKey compute{
        .program_hash = ProgramHash,
        .stage = Shader::Stage::Compute,
        .logical_stage = Shader::LogicalStage::Compute,
    };
    const Shader::ProgramCacheKey vertex{
        .program_hash = ProgramHash,
        .stage = Shader::Stage::Vertex,
        .logical_stage = Shader::LogicalStage::Vertex,
    };
    const Shader::ProgramCacheKey export_vertex{
        .program_hash = ProgramHash,
        .stage = Shader::Stage::Export,
        .logical_stage = Shader::LogicalStage::Vertex,
    };
    const Shader::ProgramCacheKey vertex_tessellation_eval{
        .program_hash = ProgramHash,
        .stage = Shader::Stage::Vertex,
        .logical_stage = Shader::LogicalStage::TessellationEval,
    };

    std::unordered_map<Shader::ProgramCacheKey, int, Shader::ProgramCacheKeyHash> programs;
    programs.emplace(compute, 1);
    programs.emplace(vertex, 2);
    programs.emplace(export_vertex, 3);
    programs.emplace(vertex_tessellation_eval, 4);

    ASSERT_EQ(programs.size(), 4U);
    EXPECT_EQ(programs.at(compute), 1);
    EXPECT_EQ(programs.at(vertex), 2);
    EXPECT_EQ(programs.at(export_vertex), 3);
    EXPECT_EQ(programs.at(vertex_tessellation_eval), 4);
}

TEST(ShaderPermutationIdentity, PhysicalAndLogicalStagePairsMatchGraphicsBindingRules) {
    using Shader::IsPhysicalStageCompatible;
    EXPECT_TRUE(IsPhysicalStageCompatible(Shader::Stage::Fragment,
                                          Shader::LogicalStage::Fragment));
    EXPECT_TRUE(IsPhysicalStageCompatible(Shader::Stage::Hull,
                                          Shader::LogicalStage::TessellationControl));
    EXPECT_TRUE(IsPhysicalStageCompatible(Shader::Stage::Vertex,
                                          Shader::LogicalStage::TessellationEval));
    EXPECT_TRUE(IsPhysicalStageCompatible(Shader::Stage::Export,
                                          Shader::LogicalStage::TessellationEval));
    EXPECT_TRUE(IsPhysicalStageCompatible(Shader::Stage::Vertex,
                                          Shader::LogicalStage::Vertex));
    EXPECT_TRUE(IsPhysicalStageCompatible(Shader::Stage::Export,
                                          Shader::LogicalStage::Vertex));
    EXPECT_TRUE(IsPhysicalStageCompatible(Shader::Stage::Local,
                                          Shader::LogicalStage::Vertex));
    EXPECT_TRUE(IsPhysicalStageCompatible(Shader::Stage::Geometry,
                                          Shader::LogicalStage::Geometry));
    EXPECT_TRUE(IsPhysicalStageCompatible(Shader::Stage::Compute,
                                          Shader::LogicalStage::Compute));

    EXPECT_FALSE(IsPhysicalStageCompatible(Shader::Stage::Compute,
                                           Shader::LogicalStage::Vertex));
    EXPECT_FALSE(IsPhysicalStageCompatible(Shader::Stage::Vertex,
                                           Shader::LogicalStage::Fragment));
    EXPECT_FALSE(IsPhysicalStageCompatible(Shader::Stage::Fragment,
                                           Shader::LogicalStage::Compute));
}

TEST(ShaderPermutationIdentity, ShadowBufferFieldsAreSemantic) {
    Shader::StageSpecialization baseline{};
    baseline.runtime_info.Initialize(Shader::Stage::Compute);
    baseline.buffers.push_back({.stride = 16});

    auto add_tid = baseline;
    add_tid.buffers[0].add_tid_enable = true;
    ExpectDifferentSemantic(baseline, add_tid);
}

TEST(ShaderPermutationIdentity, ShadowImageFieldsAreSemantic) {
    const auto float_image = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    const auto uint_image = MakeImageSpecialization(AmdGpu::NumberClass::Uint, RgbaMapping());
    ExpectDifferentSemantic(float_image, uint_image);

    const auto unsupported =
        MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping(), false);
    ExpectDifferentSemantic(float_image, unsupported);

    const auto rgb_zero = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbZeroMapping());
    ExpectDifferentSemantic(float_image, rgb_zero);
}

TEST(ShaderPermutationIdentity, UserDataMaskIsSemantic) {
    Shader::StageSpecialization baseline{};
    baseline.runtime_info.Initialize(Shader::Stage::Compute);
    auto changed = baseline;
    changed.user_data_mask = 1u << 7;

    ExpectDifferentSemantic(baseline, changed);
}

TEST(ShaderPermutationIdentity, InactiveBufferFieldsRemainCoherentWithEquality) {
    Shader::StageSpecialization baseline{};
    baseline.runtime_info.Initialize(Shader::Stage::Compute);
    baseline.buffers.emplace_back();
    auto ignored_fields = baseline;
    ignored_fields.buffers[0].data_format = 13;
    ignored_fields.buffers[0].num_format = 7;
    ignored_fields.buffers[0].index_stride = 2;
    ignored_fields.buffers[0].element_size = 3;
    ignored_fields.buffers[0].dst_select = RgbZeroMapping();
    ignored_fields.buffers[0].num_conversion = AmdGpu::NumberConversion::Uint32ToUnorm;

    ASSERT_EQ(baseline.buffers[0], ignored_fields.buffers[0]);
    ExpectSameSemantic(baseline, ignored_fields);
}

TEST(ShaderPermutationIdentity, FetchShaderOverridesParticipate) {
    Shader::StageSpecialization baseline{};
    baseline.runtime_info.Initialize(Shader::Stage::Vertex);
    baseline.fetch_shader_data.emplace();
    baseline.fetch_shader_data->attributes.push_back({
        .semantic = 1,
        .dest_vgpr = 2,
        .num_elements = 4,
        .sgpr_base = 4,
        .dword_offset = 8,
        .instance_data = Shader::Gcn::VertexAttribute::None,
        .inst_offset = 0,
        .data_format = 0,
        .num_format = 0,
    });

    auto changed_offset = baseline;
    changed_offset.fetch_shader_data->attributes[0].inst_offset = 16;
    ExpectDifferentSemantic(baseline, changed_offset);

    auto changed_format = baseline;
    changed_format.fetch_shader_data->attributes[0].data_format = 10;
    changed_format.fetch_shader_data->attributes[0].num_format = 4;
    ExpectDifferentSemantic(baseline, changed_format);
}

TEST(RuntimeInfoSemanticEquality, StageAndCommonTranslationStateParticipate) {
    Shader::RuntimeInfo compute{};
    Shader::RuntimeInfo fragment{};
    compute.Initialize(Shader::Stage::Compute);
    fragment.Initialize(Shader::Stage::Fragment);
    EXPECT_NE(compute, fragment);

    auto changed_fp_mode = compute;
    changed_fp_mode.fp_round_mode32 = AmdGpu::FpRoundMode::ToZero;
    EXPECT_NE(compute, changed_fp_mode);

    auto changed_vgprs = compute;
    changed_vgprs.num_allocated_vgprs = 12;
    EXPECT_NE(compute, changed_vgprs);
}

TEST(RuntimeInfoSemanticEquality, ComputeSharedMemoryParticipates) {
    Shader::RuntimeInfo first{};
    first.Initialize(Shader::Stage::Compute);
    first.cs_info.shared_memory_size = 1024;
    first.cs_info.workgroup_size = {8, 8, 1};
    first.cs_info.tgid_enable = {true, true, false};
    auto second = first;
    second.cs_info.shared_memory_size = 2048;

    EXPECT_NE(first, second);
}

TEST(RuntimeInfoSemanticEquality, GeometryStateIsReflexiveAndExact) {
    Shader::RuntimeInfo baseline{};
    baseline.Initialize(Shader::Stage::Geometry);
    EXPECT_EQ(baseline, baseline);

    auto invocations = baseline;
    invocations.gs_info.num_invocations = 2;
    EXPECT_NE(baseline, invocations);

    auto input_size = baseline;
    input_size.gs_info.in_vertex_data_size = 64;
    EXPECT_NE(baseline, input_size);

    auto output_size = baseline;
    output_size.gs_info.out_vertex_data_size = 128;
    EXPECT_NE(baseline, output_size);

    auto mode = baseline;
    mode.gs_info.mode = AmdGpu::GsScenario::ScenarioA;
    EXPECT_NE(baseline, mode);

    const std::array<u32, 1> one_word{};
    auto copy_size = baseline;
    copy_size.gs_info.vs_copy = one_word;
    EXPECT_NE(baseline, copy_size);
}

TEST(StageSpecializationSemanticEquality, InfoAddressAndBoundnessAreNotSemantic) {
    auto first = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    auto second = first;
    first.bitset.reset();
    second.bitset.set(0);

    ExpectSameSemantic(first, second);
}

TEST(StageSpecializationSemanticEquality, StartOriginsAreConditionallySemantic) {
    Shader::StageSpecialization empty{};
    empty.runtime_info.Initialize(Shader::Stage::Compute);
    auto changed_empty = empty;
    changed_empty.start = {.unified = 9, .buffer = 8, .user_data = 7};
    ExpectSameSemantic(empty, changed_empty);

    Shader::StageSpecialization sampler{};
    sampler.runtime_info.Initialize(Shader::Stage::Compute);
    sampler.samplers.emplace_back();
    auto sampler_buffer_origin = sampler;
    sampler_buffer_origin.start.buffer = 4;
    ExpectSameSemantic(sampler, sampler_buffer_origin);
    auto sampler_unified_origin = sampler;
    sampler_unified_origin.start.unified = 4;
    ExpectDifferentSemantic(sampler, sampler_unified_origin);

    Shader::StageSpecialization buffer{};
    buffer.runtime_info.Initialize(Shader::Stage::Compute);
    buffer.buffers.emplace_back();
    auto buffer_origin = buffer;
    buffer_origin.start.buffer = 4;
    ExpectDifferentSemantic(buffer, buffer_origin);

    Shader::StageSpecialization user_data{};
    user_data.runtime_info.Initialize(Shader::Stage::Compute);
    user_data.user_data_mask = 1;
    auto user_data_origin = user_data;
    user_data_origin.start.user_data = 4;
    ExpectDifferentSemantic(user_data, user_data_origin);
}

TEST(ShaderPermutationIdentity, CacheStoresAndValidatesSemanticIdentityWithoutOrdinalPersistence) {
    const std::string header = ReadTextFile(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const std::string source = ReadTextFile(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const std::string serialization = ReadTextFile(SHADPS4_PIPELINE_SERIALIZATION_SOURCE_PATH);
    const std::string serialization_header =
        ReadTextFile(SHADPS4_PIPELINE_SERIALIZATION_HEADER_PATH);

    EXPECT_NE(header.find("std::unique_ptr<Shader::Info> info"), std::string::npos);
    EXPECT_NE(header.find("u64 permutation_id"), std::string::npos);
    EXPECT_NE(header.find("spec.info = info.get()"), std::string::npos);
    EXPECT_NE(header.find("candidate.permutation_id == permutation_id"), std::string::npos);
    EXPECT_NE(source.find("ComputeShaderPermutationIdentity"), std::string::npos);
    EXPECT_EQ(source.find("HashCombine(params.hash"), std::string::npos);
    EXPECT_NE(source.find("RegisterShaderBinary(std::move(spv), permutation_id)"),
              std::string::npos);
    EXPECT_NE(serialization.find("expected_permutation_id"), std::string::npos);
    EXPECT_NE(serialization.find("permutation_id != expected_permutation_id"), std::string::npos);
    EXPECT_NE(serialization.find("InspectPreloadPermutation(spec, permutation_id)"),
              std::string::npos);
    EXPECT_NE(serialization.find("expected_pipeline_permutation_id"), std::string::npos);
    EXPECT_NE(serialization.find("expected_logical_stage"), std::string::npos);
    EXPECT_NE(serialization.find("ResetPipelineLoadScratch()"), std::string::npos);
    EXPECT_NE(serialization.find("stage_fetch_shader"), std::string::npos);
    EXPECT_NE(serialization.find("if (stage_fetch_shader)"), std::string::npos);
    EXPECT_NE(serialization.find("runtime_infos.fill"), std::string::npos);
    EXPECT_NE(serialization.find("runtime_infos[output_slot] = selected->spec.runtime_info"),
              std::string::npos);
    EXPECT_EQ(serialization.find("const auto stage_runtime_info = spec.runtime_info"),
              std::string::npos);
    EXPECT_NE(serialization.find("spec.runtime_info.stage != program->info.stage"),
              std::string::npos);
    EXPECT_NE(serialization.find("IsPhysicalStageCompatible"), std::string::npos);
    EXPECT_NE(serialization.find("LogicalStage::Compute)] != 0"), std::string::npos);
    EXPECT_NE(source.find("ProgramCacheKey cache_key"), std::string::npos);
    EXPECT_EQ(serialization.find("perm_idx"), std::string::npos);
    EXPECT_EQ(serialization_header.find("perm_idx"), std::string::npos);
    EXPECT_EQ(serialization.find("fmt::format(\"{:#018x}_{}\""), std::string::npos);
}

TEST(ShaderPermutationIdentity, ShaderArtifactsUseSemanticIdentityInsteadOfEncounterOrder) {
    const auto source = ReadTextFile(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto header = ReadTextFile(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const auto compile_begin = source.find("vk::ShaderModule PipelineCache::CompileModule");
    const auto compile_end = source.find("PipelineCache::Result PipelineCache::GetProgram", compile_begin);
    ASSERT_NE(compile_begin, std::string::npos);
    ASSERT_NE(compile_end, std::string::npos);
    const auto compile = source.substr(compile_begin, compile_end - compile_begin);

    EXPECT_EQ(compile.find("perm_idx"), std::string::npos);
    EXPECT_NE(compile.find("DumpShader(spv, info.pgm_hash, info.stage, \"spv\", permutation_id)"),
              std::string::npos);
    EXPECT_NE(compile.find("GetShaderPatch(info.pgm_hash, info.stage, permutation_id, \"spv\")"),
              std::string::npos);
    EXPECT_NE(compile.find("GetShaderName(info.stage, info.pgm_hash, permutation_id)"),
              std::string::npos);
    EXPECT_NE(header.find("std::optional<u64> permutation_id"), std::string::npos);
}

} // namespace
