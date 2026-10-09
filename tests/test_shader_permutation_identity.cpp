// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

#include <gtest/gtest.h>

#include "common/serdes.h"
#include "shader_recompiler/specialization_identity.h"
#include "shader_recompiler/specialization_serialization.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

namespace Shader {
void ResetWalkerRegistrationStatsForTesting();
size_t WalkerRegistrationCountForTesting();
size_t WalkerRegisteredBytesForTesting();
} // namespace Shader

namespace {

constexpr u64 ProgramHash = 0x73b8c213ULL;

void PS4_SYSV_ABI TestPersistedWalker(const u32*, u32*, Shader::PFN_SrtRangeRead, void*) {}

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
                                                    AmdGpu::CompMapping dst_select) {
    Shader::StageSpecialization spec{};
    spec.runtime_info.Initialize(Shader::Stage::Compute);
    spec.runtime_info.num_user_data = 4;
    spec.runtime_info.cs_info.shared_memory_size = 1024;
    spec.runtime_info.cs_info.workgroup_size = {8, 8, 1};
    spec.runtime_info.cs_info.tgid_enable = {true, true, false};
    spec.start = {.unified = 3, .buffer = 1, .user_data = 2};
    spec.bitset.set(0);
    spec.images.push_back({
        .type = AmdGpu::ImageType::Color2D,
        .number_class = number_class,
        .is_storage = true,
        .is_cube = false,
        .is_srgb = false,
        .dst_select = dst_select,
        .num_conversion = AmdGpu::NumberConversion::None,
        .num_bindings = 1,
    });
    return spec;
}

Shader::StageSpecialization MakeCanonicalGoldenSpecialization() {
    Shader::StageSpecialization spec{};
    spec.runtime_info.Initialize(Shader::Stage::Vertex);
    spec.runtime_info.num_user_data = 7;
    spec.runtime_info.num_input_vgprs = 3;
    spec.runtime_info.num_allocated_vgprs = 12;
    spec.runtime_info.vs_info.num_outputs = 1;
    spec.runtime_info.vs_info.outputs[0][0] = Shader::Output::PointSize;
    spec.runtime_info.vs_info.step_rate_0 = 2;
    spec.start = {.unified = 9, .buffer = 4, .user_data = 6};
    spec.user_data_mask = (1U << 1) | (1U << 6);
    spec.fetch_shader_data.emplace();
    spec.fetch_shader_data->vertex_offset_sgpr = 2;
    spec.fetch_shader_data->instance_offset_sgpr = 3;
    spec.fetch_shader_data->attributes.push_back({
        .semantic = 1,
        .dest_vgpr = 2,
        .num_elements = 4,
        .sgpr_base = 4,
        .dword_offset = 8,
        .instance_data = Shader::Gcn::VertexAttribute::Plain,
        .inst_offset = 16,
        .data_format = 10,
        .num_format = 4,
    });
    spec.vs_attribs.push_back({
        .divisor = 2,
        .num_class = AmdGpu::NumberClass::Float,
        .dst_select = RgbaMapping(),
    });
    spec.buffers.push_back({
        .stride = 16,
        .is_storage = true,
        .is_formatted = true,
        .swizzle_enable = true,
        .data_format = 10,
        .num_format = 4,
        .index_stride = 1,
        .element_size = 2,
        .add_tid_enable = true,
        .dst_select = RgbaMapping(),
        .num_conversion = AmdGpu::NumberConversion::None,
    });
    spec.images.push_back({
        .type = AmdGpu::ImageType::Color2D,
        .number_class = AmdGpu::NumberClass::Float,
        .is_storage = true,
        .is_cube = false,
        .is_srgb = false,
        .dst_select = RgbaMapping(),
        .num_conversion = AmdGpu::NumberConversion::None,
        .atomic_data_format = AmdGpu::DataFormat::Format32,
        .atomic_number_format = AmdGpu::NumberFormat::Float,
        .num_bindings = 2,
    });
    spec.fmasks.push_back({.width = 1920, .height = 1080});
    spec.samplers.push_back({.force_unnormalized = false, .force_degamma = true});
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

struct MetadataFixture {
    Shader::Info info{};
    Shader::StageSpecialization spec{};
    u64 permutation_id{};

    MetadataFixture() {
        info.stage = Shader::Stage::Vertex;
        info.l_stage = Shader::LogicalStage::Vertex;
        info.pgm_hash = ProgramHash;
        info.has_fetch_shader = true;
        info.flattened_ud_buf.resize(Shader::NUM_USER_DATA_REGS);

        spec.info = &info;
        spec.runtime_info.Initialize(Shader::Stage::Vertex);
        spec.runtime_info.num_user_data = 4;
        spec.fetch_shader_data.emplace();
        spec.fetch_shader_data->size = 8;
        spec.fetch_shader_data->vertex_offset_sgpr = 2;
        spec.fetch_shader_data->instance_offset_sgpr = 3;
        spec.fetch_shader_data->attributes.push_back({
            .semantic = 1,
            .dest_vgpr = 2,
            .num_elements = 4,
            .sgpr_base = 4,
            .dword_offset = 8,
            .instance_data = Shader::Gcn::VertexAttribute::Plain,
        });
        spec.vs_attribs.emplace_back();
        permutation_id =
            Shader::ComputeShaderPermutationIdentity(info.pgm_hash, info.l_stage, spec);
    }

    std::vector<u8> Serialize() const {
        Serialization::Archive ar;
        EXPECT_TRUE(Shader::SerializeShaderCacheMetadata(ar, info, spec, permutation_id));
        return ar.TakeOff();
    }
};

std::vector<size_t> DifferingByteOffsets(std::span<const u8> first, std::span<const u8> second) {
    if (first.size() != second.size()) {
        return {};
    }
    std::vector<size_t> offsets;
    for (size_t offset = 0; offset < first.size(); ++offset) {
        if (first[offset] != second[offset]) {
            offsets.push_back(offset);
        }
    }
    return offsets;
}

TEST(ShaderPermutationIdentity, EncounterOrderDoesNotChangePersistedIds) {
    const auto float_rgba = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    const auto uint_rgba = MakeImageSpecialization(AmdGpu::NumberClass::Uint, RgbaMapping());

    const std::array ab{
        Identity(float_rgba),
        Identity(uint_rgba),
    };
    const std::array ba{
        Identity(uint_rgba),
        Identity(float_rgba),
    };

    EXPECT_EQ(ab[0], ba[1]);
    EXPECT_EQ(ab[1], ba[0]);
    EXPECT_NE(ab[0], ab[1]);
}

TEST(ShaderPermutationIdentity, StableCanonicalGoldenVector) {
    const auto spec = MakeCanonicalGoldenSpecialization();

    EXPECT_EQ(Identity(spec, 0x0123456789ABCDEFULL, Shader::LogicalStage::Vertex),
              0x405A19E17871A28EULL);
}

TEST(ShaderPermutationIdentity, CompleteResourceBindingAndUserDataShapeParticipates) {
    const auto baseline = MakeCanonicalGoldenSpecialization();
    const auto baseline_id = Identity(baseline, ProgramHash, Shader::LogicalStage::Vertex);
    const auto expect_change = [&](const char* field, auto mutate) {
        SCOPED_TRACE(field);
        auto changed = baseline;
        mutate(changed);
        EXPECT_NE(Identity(changed, ProgramHash, Shader::LogicalStage::Vertex), baseline_id);
    };

    expect_change("physical stage",
                  [](auto& spec) { spec.runtime_info.stage = Shader::Stage::Export; });
    expect_change("unified binding origin", [](auto& spec) { ++spec.start.unified; });
    expect_change("buffer binding origin", [](auto& spec) { ++spec.start.buffer; });
    expect_change("user-data binding origin", [](auto& spec) { ++spec.start.user_data; });
    expect_change("user-data shape", [](auto& spec) { spec.user_data_mask ^= 1U << 4; });

    expect_change("vertex attribute count", [](auto& spec) { spec.vs_attribs.emplace_back(); });
    expect_change("vertex divisor", [](auto& spec) { ++spec.vs_attribs[0].divisor; });
    expect_change("vertex number class",
                  [](auto& spec) { spec.vs_attribs[0].num_class = AmdGpu::NumberClass::Uint; });
    expect_change("vertex destination mapping",
                  [](auto& spec) { spec.vs_attribs[0].dst_select = RgbZeroMapping(); });

    expect_change("buffer count", [](auto& spec) { spec.buffers.emplace_back(); });
    expect_change("buffer stride", [](auto& spec) { ++spec.buffers[0].stride; });
    expect_change("buffer storage", [](auto& spec) { spec.buffers[0].is_storage = false; });
    expect_change("buffer formatted", [](auto& spec) { spec.buffers[0].is_formatted = false; });
    expect_change("buffer swizzle", [](auto& spec) { spec.buffers[0].swizzle_enable = false; });
    expect_change("buffer thread index",
                  [](auto& spec) { spec.buffers[0].add_tid_enable = false; });
    expect_change("buffer data format", [](auto& spec) { ++spec.buffers[0].data_format; });
    expect_change("buffer number format", [](auto& spec) { ++spec.buffers[0].num_format; });
    expect_change("buffer index stride", [](auto& spec) { ++spec.buffers[0].index_stride; });
    expect_change("buffer element size", [](auto& spec) { ++spec.buffers[0].element_size; });
    expect_change("buffer destination mapping",
                  [](auto& spec) { spec.buffers[0].dst_select = RgbZeroMapping(); });
    expect_change("buffer number conversion", [](auto& spec) {
        spec.buffers[0].num_conversion = AmdGpu::NumberConversion::UintToUscaled;
    });

    expect_change("image count", [](auto& spec) { spec.images.emplace_back(); });
    expect_change("image type",
                  [](auto& spec) { spec.images[0].type = AmdGpu::ImageType::Color3D; });
    expect_change("image number class",
                  [](auto& spec) { spec.images[0].number_class = AmdGpu::NumberClass::Sint; });
    expect_change("image storage", [](auto& spec) { spec.images[0].is_storage = false; });
    expect_change("image cube", [](auto& spec) { spec.images[0].is_cube = true; });
    expect_change("image srgb", [](auto& spec) { spec.images[0].is_srgb = true; });
    expect_change("image destination mapping",
                  [](auto& spec) { spec.images[0].dst_select = RgbZeroMapping(); });
    expect_change("image number conversion", [](auto& spec) {
        spec.images[0].num_conversion = AmdGpu::NumberConversion::SrgbToNorm;
    });
    expect_change("image descriptor count", [](auto& spec) { ++spec.images[0].num_bindings; });

    expect_change("fmask count", [](auto& spec) { spec.fmasks.emplace_back(); });
    expect_change("fmask width", [](auto& spec) { ++spec.fmasks[0].width; });
    expect_change("fmask height", [](auto& spec) { ++spec.fmasks[0].height; });
    expect_change("sampler count", [](auto& spec) { spec.samplers.emplace_back(); });
    expect_change("sampler coordinates",
                  [](auto& spec) { spec.samplers[0].force_unnormalized = true; });
    expect_change("sampler degamma", [](auto& spec) { spec.samplers[0].force_degamma = false; });
}

TEST(ShaderPermutationIdentity, StorageDestinationMappingIsSemantic) {
    const auto rgba = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    const auto rgb_zero = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbZeroMapping());

    EXPECT_NE(Identity(rgba), Identity(rgb_zero));
}

TEST(ShaderPermutationIdentity, ExactAtomicImageFormatIsSemanticWithinOneNumberClass) {
    auto r32_float = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    r32_float.images[0].atomic_data_format = AmdGpu::DataFormat::Format32;
    r32_float.images[0].atomic_number_format = AmdGpu::NumberFormat::Float;
    auto r16_float = r32_float;
    r16_float.images[0].atomic_data_format = AmdGpu::DataFormat::Format16;

    EXPECT_EQ(r32_float.images[0].number_class, r16_float.images[0].number_class);
    ExpectDifferentSemantic(r32_float, r16_float);
}

TEST(ShaderPermutationIdentity, IntegerNumberClassAndBufferThreadIndexAreSemantic) {
    const auto signed_image = MakeImageSpecialization(AmdGpu::NumberClass::Sint, RgbaMapping());
    const auto unsigned_image = MakeImageSpecialization(AmdGpu::NumberClass::Uint, RgbaMapping());
    EXPECT_NE(Identity(signed_image), Identity(unsigned_image));

    Shader::StageSpecialization baseline{};
    baseline.runtime_info.Initialize(Shader::Stage::Compute);
    baseline.buffers.push_back({.stride = 16});
    auto add_thread_id = baseline;
    add_thread_id.buffers[0].add_tid_enable = true;
    ExpectDifferentSemantic(baseline, add_thread_id);
}

TEST(ShaderPermutationIdentity, ComputeSharedMemorySizeIsSemantic) {
    auto small = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    auto large = small;
    large.runtime_info.cs_info.shared_memory_size *= 2;

    EXPECT_NE(Identity(small), Identity(large));
}

TEST(ShaderPermutationIdentity, InfoObjectAddressIsNotSemantic) {
    Shader::Info first_info{};
    Shader::Info second_info{};
    auto first = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    auto second = first;
    first.info = &first_info;
    second.info = &second_info;

    EXPECT_EQ(Identity(first), Identity(second));
}

TEST(ShaderPermutationIdentity, ProgramAndLogicalStageParticipate) {
    const auto spec = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());

    EXPECT_NE(Identity(spec, ProgramHash, Shader::LogicalStage::Compute),
              Identity(spec, ProgramHash + 1, Shader::LogicalStage::Compute));
    EXPECT_NE(Identity(spec, ProgramHash, Shader::LogicalStage::Compute),
              Identity(spec, ProgramHash, Shader::LogicalStage::Vertex));
}

TEST(ShaderPermutationIdentity, InactiveBufferFieldsDoNotParticipate) {
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

    EXPECT_EQ(baseline.buffers[0], ignored_fields.buffers[0]);
    EXPECT_EQ(Identity(baseline), Identity(ignored_fields));
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
    EXPECT_NE(*baseline.fetch_shader_data, *changed_offset.fetch_shader_data);
    EXPECT_NE(Identity(baseline), Identity(changed_offset));

    auto changed_format = baseline;
    changed_format.fetch_shader_data->attributes[0].data_format = 10;
    changed_format.fetch_shader_data->attributes[0].num_format = 4;
    EXPECT_NE(*baseline.fetch_shader_data, *changed_format.fetch_shader_data);
    EXPECT_NE(Identity(baseline), Identity(changed_format));
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

TEST(ShaderPermutationIdentity, EveryPhysicalStageRuntimeShapeParticipates) {
    const auto expect_change = [](Shader::StageSpecialization baseline,
                                  Shader::StageSpecialization changed,
                                  Shader::LogicalStage logical_stage) {
        EXPECT_NE(Identity(baseline, ProgramHash, logical_stage),
                  Identity(changed, ProgramHash, logical_stage));
    };

    Shader::StageSpecialization local{};
    local.runtime_info.Initialize(Shader::Stage::Local);
    auto changed_local = local;
    changed_local.runtime_info.ls_info.ls_stride = 16;
    expect_change(local, changed_local, Shader::LogicalStage::Vertex);

    Shader::StageSpecialization export_stage{};
    export_stage.runtime_info.Initialize(Shader::Stage::Export);
    auto changed_export = export_stage;
    changed_export.runtime_info.es_info.vertex_data_size = 32;
    expect_change(export_stage, changed_export, Shader::LogicalStage::Vertex);

    Shader::StageSpecialization vertex{};
    vertex.runtime_info.Initialize(Shader::Stage::Vertex);
    auto changed_vertex = vertex;
    changed_vertex.runtime_info.vs_info.num_outputs = 1;
    expect_change(vertex, changed_vertex, Shader::LogicalStage::Vertex);

    Shader::StageSpecialization hull{};
    hull.runtime_info.Initialize(Shader::Stage::Hull);
    auto changed_hull = hull;
    changed_hull.runtime_info.hs_info.num_threads = 4;
    expect_change(hull, changed_hull, Shader::LogicalStage::TessellationControl);

    Shader::StageSpecialization geometry{};
    geometry.runtime_info.Initialize(Shader::Stage::Geometry);
    auto changed_geometry = geometry;
    changed_geometry.runtime_info.gs_info.output_vertices = 3;
    expect_change(geometry, changed_geometry, Shader::LogicalStage::Geometry);

    Shader::StageSpecialization fragment{};
    fragment.runtime_info.Initialize(Shader::Stage::Fragment);
    fragment.runtime_info.fs_info.num_inputs = 1;
    auto changed_fragment = fragment;
    changed_fragment.runtime_info.fs_info.inputs[0].param_index = 2;
    expect_change(fragment, changed_fragment, Shader::LogicalStage::Fragment);

    Shader::StageSpecialization compute{};
    compute.runtime_info.Initialize(Shader::Stage::Compute);
    auto changed_compute = compute;
    changed_compute.runtime_info.cs_info.shared_memory_size = 256;
    expect_change(compute, changed_compute, Shader::LogicalStage::Compute);
}

TEST(RuntimeInfoSemanticEquality, GeometryInvocationCountIsReflexiveAndExact) {
    Shader::RuntimeInfo zero{};
    zero.Initialize(Shader::Stage::Geometry);
    EXPECT_EQ(zero, zero);

    Shader::RuntimeInfo one{};
    one.Initialize(Shader::Stage::Geometry);
    one.gs_info.num_invocations = 1;
    auto two = one;
    two.gs_info.num_invocations = 2;
    EXPECT_NE(one, two);
}

TEST(RuntimeInfoSemanticEquality, GeometryRingShapeAndModeParticipate) {
    Shader::RuntimeInfo baseline{};
    baseline.Initialize(Shader::Stage::Geometry);
    baseline.gs_info.num_invocations = 1;

    auto changed_input_size = baseline;
    changed_input_size.gs_info.in_vertex_data_size = 64;
    EXPECT_NE(baseline, changed_input_size);

    auto changed_output_size = baseline;
    changed_output_size.gs_info.out_vertex_data_size = 128;
    EXPECT_NE(baseline, changed_output_size);

    auto changed_mode = baseline;
    changed_mode.gs_info.mode = AmdGpu::GsScenario::ScenarioA;
    EXPECT_NE(baseline, changed_mode);

    const std::array<u32, 1> one_word{};
    auto changed_copy_size = baseline;
    changed_copy_size.gs_info.vs_copy = one_word;
    EXPECT_NE(baseline, changed_copy_size);
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

TEST(StageSpecializationSerialization, EveryRuntimeAndResourceRecordFamilyRoundTrips) {
    const auto expect_runtime_round_trip = [](Shader::StageSpecialization original) {
        Serialization::Archive output;
        original.Serialize(output);
        Serialization::Archive input{output.TakeOff()};
        Shader::StageSpecialization restored{};
        ASSERT_TRUE(restored.Deserialize(input));
        EXPECT_TRUE(input.IsEoS());
        EXPECT_EQ(restored.runtime_info, original.runtime_info);
    };

    Shader::StageSpecialization local{};
    local.runtime_info.Initialize(Shader::Stage::Local);
    local.runtime_info.ls_info.ls_stride = 16;
    expect_runtime_round_trip(local);

    Shader::StageSpecialization export_stage{};
    export_stage.runtime_info.Initialize(Shader::Stage::Export);
    export_stage.runtime_info.hs_es_vs_info.hs_output_cp_stride = 12;
    export_stage.runtime_info.es_vs_info.tess_type = AmdGpu::TessellationType::Quad;
    export_stage.runtime_info.es_vs_info.tess_topology = AmdGpu::TessellationTopology::TriangleCcw;
    export_stage.runtime_info.es_vs_info.tess_partitioning =
        AmdGpu::TessellationPartitioning::FracEven;
    export_stage.runtime_info.es_info.vertex_data_size = 32;
    expect_runtime_round_trip(export_stage);

    Shader::StageSpecialization vertex = MakeCanonicalGoldenSpecialization();
    vertex.runtime_info.vs_info.tess_emulated_primitive = true;
    vertex.runtime_info.vs_info.emulate_depth_negative_one_to_one = true;
    vertex.runtime_info.vs_info.clip_disable = true;
    vertex.runtime_info.vs_info.user_clip_plane_mask = 3;
    Serialization::Archive vertex_output;
    vertex.Serialize(vertex_output);
    Serialization::Archive vertex_input{vertex_output.TakeOff()};
    Shader::StageSpecialization restored_vertex{};
    ASSERT_TRUE(restored_vertex.Deserialize(vertex_input));
    EXPECT_EQ(restored_vertex.runtime_info, vertex.runtime_info);
    EXPECT_EQ(restored_vertex.fetch_shader_data, vertex.fetch_shader_data);
    EXPECT_EQ(restored_vertex.vs_attribs, vertex.vs_attribs);
    EXPECT_EQ(restored_vertex.buffers, vertex.buffers);
    EXPECT_EQ(restored_vertex.images, vertex.images);
    EXPECT_EQ(restored_vertex.fmasks, vertex.fmasks);
    EXPECT_EQ(restored_vertex.samplers, vertex.samplers);
    EXPECT_EQ(restored_vertex.start, vertex.start);
    EXPECT_EQ(restored_vertex.user_data_mask, vertex.user_data_mask);
    EXPECT_EQ(restored_vertex.bitset, vertex.bitset);
    EXPECT_TRUE(vertex_input.IsEoS());

    Shader::StageSpecialization hull{};
    hull.runtime_info.Initialize(Shader::Stage::Hull);
    hull.runtime_info.hs_es_vs_info.hs_output_cp_stride = 8;
    hull.runtime_info.hs_info.num_input_control_points = 3;
    hull.runtime_info.hs_info.num_threads = 4;
    hull.runtime_info.hs_info.tess_type = AmdGpu::TessellationType::Triangle;
    hull.runtime_info.hs_info.offchip_lds_enable = true;
    hull.runtime_info.hs_info.ls_stride = 24;
    hull.runtime_info.hs_info.hs_output_base = 64;
    expect_runtime_round_trip(hull);

    Shader::StageSpecialization fragment{};
    fragment.runtime_info.Initialize(Shader::Stage::Fragment);
    fragment.runtime_info.fs_info.en_flags.front_face_ena = 1;
    fragment.runtime_info.fs_info.addr_flags.persp_center_ena = 1;
    fragment.runtime_info.fs_info.num_inputs = 1;
    fragment.runtime_info.fs_info.inputs[0] = {
        .param_index = 2,
        .is_default = true,
        .is_flat = false,
        .default_value = 3,
    };
    fragment.runtime_info.fs_info.color_buffers[0] = {
        .data_format = AmdGpu::DataFormat::Format8_8_8_8,
        .num_format = AmdGpu::NumberFormat::Unorm,
        .num_conversion = AmdGpu::NumberConversion::None,
        .export_format = AmdGpu::ShaderExportFormat::ABGR_32,
        .swizzle = RgbaMapping(),
    };
    fragment.runtime_info.fs_info.z_export_format = AmdGpu::ShaderExportFormat::R_32;
    fragment.runtime_info.fs_info.mrtz_mask = 3;
    fragment.runtime_info.fs_info.dual_source_blending = true;
    fragment.runtime_info.fs_info.clip_distance_emulation = true;
    expect_runtime_round_trip(fragment);

    Shader::StageSpecialization compute{};
    compute.runtime_info.Initialize(Shader::Stage::Compute);
    compute.runtime_info.cs_info.shared_memory_size = 1024;
    compute.runtime_info.cs_info.workgroup_size = {8, 8, 1};
    compute.runtime_info.cs_info.tgid_enable = {true, false, true};
    expect_runtime_round_trip(compute);
}

TEST(StageSpecializationSerialization, RoundTrips32BitSnormNzConversion) {
    auto original = MakeImageSpecialization(AmdGpu::NumberClass::Sint, RgbaMapping());
    original.images[0].num_conversion = AmdGpu::NumberConversion::Sint32ToSnormNz;
    original.buffers.push_back({
        .stride = 12,
        .is_storage = false,
        .is_formatted = true,
        .data_format = static_cast<u32>(AmdGpu::DataFormat::Format32_32_32),
        .num_format = static_cast<u32>(AmdGpu::NumberFormat::SnormNz),
        .dst_select = RgbaMapping(),
        .num_conversion = AmdGpu::NumberConversion::Sint32ToSnormNz,
    });

    Serialization::Archive output;
    original.Serialize(output);
    Serialization::Archive input{output.TakeOff()};
    Shader::StageSpecialization restored{};
    ASSERT_TRUE(restored.Deserialize(input));
    ASSERT_EQ(restored.images.size(), 1U);
    ASSERT_EQ(restored.buffers.size(), 1U);
    EXPECT_EQ(restored.images[0].num_conversion, AmdGpu::NumberConversion::Sint32ToSnormNz);
    EXPECT_EQ(restored.buffers[0].num_conversion, AmdGpu::NumberConversion::Sint32ToSnormNz);
    EXPECT_TRUE(input.IsEoS());
}

TEST(StageSpecializationSerialization, RejectsResourceOnlyConversionInColorRuntimeState) {
    Shader::StageSpecialization none{};
    none.runtime_info.Initialize(Shader::Stage::Fragment);
    none.runtime_info.fs_info.color_buffers[0].num_conversion = AmdGpu::NumberConversion::None;
    auto srgb = none;
    srgb.runtime_info.fs_info.color_buffers[0].num_conversion =
        AmdGpu::NumberConversion::SrgbToNorm;

    Serialization::Archive none_archive;
    none.Serialize(none_archive);
    auto bytes = none_archive.TakeOff();
    Serialization::Archive srgb_archive;
    srgb.Serialize(srgb_archive);
    const auto srgb_bytes = srgb_archive.TakeOff();
    const auto differences = DifferingByteOffsets(bytes, srgb_bytes);
    ASSERT_EQ(differences.size(), 1U);

    bytes[differences.front()] = static_cast<u8>(AmdGpu::NumberConversion::Sint32ToSnormNz);
    Serialization::Archive malformed{std::move(bytes)};
    Shader::StageSpecialization decoded{};
    EXPECT_FALSE(decoded.Deserialize(malformed));
}

TEST(StageSpecializationSerialization, RejectsNonCanonicalRuntimeBooleanWireValue) {
    Shader::StageSpecialization disabled{};
    disabled.runtime_info.Initialize(Shader::Stage::Compute);
    disabled.runtime_info.cs_info.workgroup_size = {8, 8, 1};
    auto enabled = disabled;
    enabled.runtime_info.cs_info.tgid_enable[1] = true;

    Serialization::Archive disabled_archive;
    disabled.Serialize(disabled_archive);
    auto disabled_bytes = disabled_archive.TakeOff();
    Serialization::Archive enabled_archive;
    enabled.Serialize(enabled_archive);
    const auto enabled_bytes = enabled_archive.TakeOff();
    const auto differences = DifferingByteOffsets(disabled_bytes, enabled_bytes);
    ASSERT_EQ(differences.size(), 1U);
    ASSERT_EQ(disabled_bytes[differences.front()], 0U);
    ASSERT_EQ(enabled_bytes[differences.front()], 1U);

    disabled_bytes[differences.front()] = 2U;
    Serialization::Archive malformed{std::move(disabled_bytes)};
    Shader::StageSpecialization decoded{};
    EXPECT_FALSE(decoded.Deserialize(malformed));
}

TEST(StageSpecializationSerialization, RejectsInvalidImageEnumWireValue) {
    Shader::StageSpecialization float_image{};
    float_image.runtime_info.Initialize(Shader::Stage::Compute);
    float_image.images.push_back({
        .type = AmdGpu::ImageType::Color2D,
        .number_class = AmdGpu::NumberClass::Float,
        .num_bindings = 1,
    });
    auto sint_image = float_image;
    sint_image.images[0].number_class = AmdGpu::NumberClass::Sint;

    Serialization::Archive float_archive;
    float_image.Serialize(float_archive);
    auto float_bytes = float_archive.TakeOff();
    Serialization::Archive sint_archive;
    sint_image.Serialize(sint_archive);
    const auto sint_bytes = sint_archive.TakeOff();
    const auto differences = DifferingByteOffsets(float_bytes, sint_bytes);
    ASSERT_EQ(differences.size(), 1U);

    float_bytes[differences.front()] = 0x7FU;
    Serialization::Archive malformed{std::move(float_bytes)};
    Shader::StageSpecialization decoded{};
    EXPECT_FALSE(decoded.Deserialize(malformed));
}

TEST(StageSpecializationSerialization, RejectsNonCanonicalBitfieldBooleanWireValues) {
    const auto expect_rejected = [](Shader::StageSpecialization disabled,
                                    Shader::StageSpecialization enabled) {
        Serialization::Archive disabled_archive;
        disabled.Serialize(disabled_archive);
        auto disabled_bytes = disabled_archive.TakeOff();
        Serialization::Archive enabled_archive;
        enabled.Serialize(enabled_archive);
        const auto enabled_bytes = enabled_archive.TakeOff();
        const auto differences = DifferingByteOffsets(disabled_bytes, enabled_bytes);
        ASSERT_EQ(differences.size(), 1U);
        ASSERT_EQ(disabled_bytes[differences.front()], 0U);
        ASSERT_EQ(enabled_bytes[differences.front()], 1U);

        disabled_bytes[differences.front()] = 2U;
        Serialization::Archive malformed{std::move(disabled_bytes)};
        Shader::StageSpecialization decoded{};
        decoded.runtime_info.Initialize(Shader::Stage::Vertex);
        decoded.user_data_mask = 0x5AU;
        EXPECT_FALSE(decoded.Deserialize(malformed));
        EXPECT_EQ(decoded.runtime_info.stage, Shader::Stage::Vertex);
        EXPECT_EQ(decoded.user_data_mask, 0x5AU);
    };

    Shader::StageSpecialization buffer_disabled{};
    buffer_disabled.runtime_info.Initialize(Shader::Stage::Compute);
    buffer_disabled.buffers.push_back({.stride = 16, .is_storage = false});
    auto buffer_enabled = buffer_disabled;
    buffer_enabled.buffers[0].is_storage = true;
    expect_rejected(buffer_disabled, buffer_enabled);

    Shader::StageSpecialization sampler_disabled{};
    sampler_disabled.runtime_info.Initialize(Shader::Stage::Compute);
    sampler_disabled.samplers.push_back({.force_unnormalized = false, .force_degamma = false});
    auto sampler_enabled = sampler_disabled;
    sampler_enabled.samplers[0].force_degamma = true;
    expect_rejected(sampler_disabled, sampler_enabled);
}

TEST(InfoSerialization, EveryResourceRecordFamilyRoundTripsExplicitState) {
    Shader::Info source{};
    source.stage = Shader::Stage::Compute;
    source.l_stage = Shader::LogicalStage::Compute;
    source.flattened_ud_buf.resize(Shader::NUM_USER_DATA_REGS);

    Shader::BufferResource buffer{
        .sharp_idx = 0,
        .used_types = Shader::IR::Type::U32 | Shader::IR::Type::F32,
        .buffer_type = Shader::BufferType::Guest,
        .instance_attrib = 3,
        .is_written = true,
        .is_formatted = true,
    };
    const Shader::BufferAddressLayout layout{
        .stride = 16,
        .data_format = 10,
        .element_size = 2,
        .index_stride = 1,
        .swizzle_enable = true,
        .add_tid_enable = true,
    };
    buffer.access_range.RecordAddressLayout(layout);
    buffer.access_range.Add(32, 64);
    source.buffers.push_back(buffer);
    source.images.push_back({
        .sharp_idx = 4,
        .is_depth = true,
        .is_array = true,
        .is_written = true,
        .queries_dimensions = true,
        .mip_fallback_mode = Shader::MipStorageFallbackMode::ConstantIndex,
        .constant_mip_index = 2,
    });
    source.samplers.push_back({
        .sharp_idx = 12,
        .inline_sampler = {.raw0 = 0x1234U, .raw1 = 0x5678U},
        .is_inline_sampler = true,
        .associated_image = 0,
        .disable_aniso = true,
    });
    source.fmasks.push_back({.sharp_idx = 4});
    source.addr64_pointer_roots[0] = {.buffer_resource_index = 0, .pointer_lo_flat_index = 0};
    source.num_addr64_pointer_roots = 1;

    Serialization::Archive output;
    source.Serialize(output);
    Serialization::Archive input{output.TakeOff()};
    Shader::Info restored{};
    ASSERT_TRUE(restored.Deserialize(input));
    EXPECT_TRUE(input.IsEoS());
    ASSERT_EQ(restored.buffers.size(), 1U);
    EXPECT_EQ(restored.buffers[0].sharp_idx, source.buffers[0].sharp_idx);
    EXPECT_EQ(restored.buffers[0].used_types, source.buffers[0].used_types);
    EXPECT_EQ(restored.buffers[0].buffer_type, source.buffers[0].buffer_type);
    EXPECT_EQ(restored.buffers[0].instance_attrib, source.buffers[0].instance_attrib);
    EXPECT_EQ(restored.buffers[0].is_written, source.buffers[0].is_written);
    EXPECT_EQ(restored.buffers[0].is_formatted, source.buffers[0].is_formatted);
    EXPECT_EQ(restored.buffers[0].access_range.UpperBound(), 96U);
    EXPECT_TRUE(restored.buffers[0].access_range.IsBounded());
    EXPECT_TRUE(restored.buffers[0].access_range.HasAddressLayout());
    EXPECT_EQ(restored.buffers[0].access_range.AddressLayout(), layout);
    ASSERT_EQ(restored.images.size(), 1U);
    EXPECT_TRUE(restored.images[0].is_depth);
    EXPECT_TRUE(restored.images[0].is_array);
    EXPECT_EQ(restored.images[0].mip_fallback_mode, Shader::MipStorageFallbackMode::ConstantIndex);
    EXPECT_EQ(restored.images[0].constant_mip_index, 2U);
    ASSERT_EQ(restored.samplers.size(), 1U);
    EXPECT_EQ(restored.samplers[0].inline_sampler.raw0, 0x1234U);
    EXPECT_EQ(restored.samplers[0].inline_sampler.raw1, 0x5678U);
    EXPECT_TRUE(restored.samplers[0].is_inline_sampler);
    EXPECT_TRUE(restored.samplers[0].disable_aniso);
    ASSERT_EQ(restored.fmasks.size(), 1U);
    EXPECT_EQ(restored.fmasks[0].sharp_idx, 4U);
    ASSERT_EQ(restored.num_addr64_pointer_roots, 1U);
    EXPECT_EQ(restored.addr64_pointer_roots[0], source.addr64_pointer_roots[0]);
}

TEST(InfoSerialization, RejectsNonCanonicalResourceBooleanWireValue) {
    Shader::Info color{};
    color.stage = Shader::Stage::Compute;
    color.l_stage = Shader::LogicalStage::Compute;
    color.flattened_ud_buf.resize(Shader::NUM_USER_DATA_REGS);
    color.images.push_back({.sharp_idx = 0, .is_depth = false});
    auto depth = color;
    depth.images[0].is_depth = true;

    Serialization::Archive color_archive;
    color.Serialize(color_archive);
    auto color_bytes = color_archive.TakeOff();
    Serialization::Archive depth_archive;
    depth.Serialize(depth_archive);
    const auto depth_bytes = depth_archive.TakeOff();
    const auto differences = DifferingByteOffsets(color_bytes, depth_bytes);
    ASSERT_EQ(differences.size(), 1U);
    ASSERT_EQ(color_bytes[differences.front()], 0U);
    ASSERT_EQ(depth_bytes[differences.front()], 1U);

    color_bytes[differences.front()] = 2U;
    Serialization::Archive malformed{std::move(color_bytes)};
    Shader::Info decoded{};
    EXPECT_FALSE(decoded.Deserialize(malformed));
}

TEST(ShaderPermutationIdentity, DynamicSampledTableLayoutChangesIdentity) {
    auto baseline = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    baseline.images[0].num_bindings = Shader::NUM_IMAGES;
    baseline.images[0].dynamic_descriptor_count = Shader::NUM_IMAGES;
    baseline.images[0].dynamic_descriptor_stride_dw = 8;
    baseline.images[0].requires_dynamic_oob_guard = true;

    auto different_stride = baseline;
    different_stride.images[0].dynamic_descriptor_stride_dw = 12;
    ExpectDifferentSemantic(baseline, different_stride);

    auto different_guard = baseline;
    different_guard.images[0].requires_dynamic_oob_guard = false;
    ExpectDifferentSemantic(baseline, different_guard);
}

TEST(InfoSerialization, DynamicSampledImageTableSemanticsRoundTrip) {
    Shader::Info source{};
    source.stage = Shader::Stage::Compute;
    source.l_stage = Shader::LogicalStage::Compute;
    constexpr u32 TableEntries = Shader::NUM_IMAGES;
    constexpr u32 TableStrideDwords = 8;
    constexpr u32 TableSpanDwords = (TableEntries - 1U) * TableStrideDwords + 8U;
    source.srt_info.flattened_bufsize_dw = Shader::NUM_USER_DATA_REGS + TableSpanDwords;
    source.srt_info.range_reservations.push_back({
        .pointer_lo_flat_index = 0,
        .source_offset_dw = 3620,
        .destination_flat_offset_dw = Shader::NUM_USER_DATA_REGS,
        .span_dw = TableSpanDwords,
    });
    source.flattened_ud_buf.resize(source.srt_info.flattened_bufsize_dw);
    source.images.push_back({
        .sharp_idx = Shader::NUM_USER_DATA_REGS,
        .dynamic_descriptor_count = TableEntries,
        .dynamic_descriptor_stride_dw = TableStrideDwords,
        .requires_dynamic_oob_guard = true,
    });

    Serialization::Archive encoded;
    source.Serialize(encoded);
    Serialization::Archive input{encoded.TakeOff()};
    Shader::Info restored{};
    ASSERT_TRUE(restored.Deserialize(input));
    ASSERT_TRUE(input.IsEoS());
    ASSERT_EQ(restored.images.size(), 1U);
    EXPECT_EQ(restored.images[0].dynamic_descriptor_count, TableEntries);
    EXPECT_EQ(restored.images[0].dynamic_descriptor_stride_dw, TableStrideDwords);
    EXPECT_TRUE(restored.images[0].requires_dynamic_oob_guard);
    EXPECT_EQ(restored.srt_info.range_reservations, source.srt_info.range_reservations);
    EXPECT_EQ(restored.srt_info.flattened_bufsize_dw, source.srt_info.flattened_bufsize_dw);
    EXPECT_EQ(restored.flattened_ud_buf.size(), source.flattened_ud_buf.size());
}

TEST(InfoSerialization, SixBitSamplerImageAssociationRoundTrips) {
    Shader::Info source{};
    source.stage = Shader::Stage::Compute;
    source.l_stage = Shader::LogicalStage::Compute;
    source.flattened_ud_buf.resize(Shader::NUM_USER_DATA_REGS);
    for (u32 index = 0; index < Shader::NUM_IMAGES; ++index) {
        source.images.push_back({.sharp_idx = 0});
    }
    source.samplers.push_back({
        .sharp_idx = 0,
        .is_inline_sampler = true,
        .associated_image = Shader::NUM_IMAGES - 1,
        .disable_aniso = true,
    });

    Serialization::Archive encoded;
    source.Serialize(encoded);
    Serialization::Archive input{encoded.TakeOff()};
    Shader::Info restored{};
    ASSERT_TRUE(restored.Deserialize(input));
    ASSERT_EQ(restored.samplers.size(), 1U);
    EXPECT_EQ(restored.samplers[0].associated_image, Shader::NUM_IMAGES - 1);
}

TEST(InfoSerialization, RejectsInvalidResourceEnumWireValue) {
    Shader::Info no_fallback{};
    no_fallback.stage = Shader::Stage::Compute;
    no_fallback.l_stage = Shader::LogicalStage::Compute;
    no_fallback.flattened_ud_buf.resize(Shader::NUM_USER_DATA_REGS);
    no_fallback.images.push_back(
        {.sharp_idx = 0, .mip_fallback_mode = Shader::MipStorageFallbackMode::None});
    auto dynamic_fallback = no_fallback;
    dynamic_fallback.images[0].mip_fallback_mode = Shader::MipStorageFallbackMode::DynamicIndex;

    Serialization::Archive no_fallback_archive;
    no_fallback.Serialize(no_fallback_archive);
    auto no_fallback_bytes = no_fallback_archive.TakeOff();
    Serialization::Archive dynamic_fallback_archive;
    dynamic_fallback.Serialize(dynamic_fallback_archive);
    const auto dynamic_fallback_bytes = dynamic_fallback_archive.TakeOff();
    const auto differences = DifferingByteOffsets(no_fallback_bytes, dynamic_fallback_bytes);
    ASSERT_EQ(differences.size(), 1U);

    no_fallback_bytes[differences.front()] = 0x7FU;
    Serialization::Archive malformed{std::move(no_fallback_bytes)};
    Shader::Info decoded{};
    decoded.stage = Shader::Stage::Vertex;
    decoded.pgm_hash = 0xBADCAFEULL;
    EXPECT_FALSE(decoded.Deserialize(malformed));
    EXPECT_EQ(decoded.stage, Shader::Stage::Vertex);
    EXPECT_EQ(decoded.pgm_hash, 0xBADCAFEULL);

    Shader::Info one_x_anisotropy{};
    one_x_anisotropy.stage = Shader::Stage::Compute;
    one_x_anisotropy.l_stage = Shader::LogicalStage::Compute;
    one_x_anisotropy.flattened_ud_buf.resize(Shader::NUM_USER_DATA_REGS);
    one_x_anisotropy.samplers.push_back({.sharp_idx = 0, .is_inline_sampler = true});
    auto two_x_anisotropy = one_x_anisotropy;
    two_x_anisotropy.samplers[0].inline_sampler.raw0 = 1ULL << 9U;

    Serialization::Archive one_x_archive;
    one_x_anisotropy.Serialize(one_x_archive);
    auto one_x_bytes = one_x_archive.TakeOff();
    Serialization::Archive two_x_archive;
    two_x_anisotropy.Serialize(two_x_archive);
    const auto two_x_bytes = two_x_archive.TakeOff();
    const auto sampler_differences = DifferingByteOffsets(one_x_bytes, two_x_bytes);
    ASSERT_EQ(sampler_differences.size(), 1U);

    // Bits 9..11 encode AnisoRatio. Seven is outside the declared hardware enum domain.
    one_x_bytes[sampler_differences.front()] = 0x0EU;
    Serialization::Archive invalid_sampler{std::move(one_x_bytes)};
    EXPECT_FALSE(decoded.Deserialize(invalid_sampler));
    EXPECT_EQ(decoded.stage, Shader::Stage::Vertex);
    EXPECT_EQ(decoded.pgm_hash, 0xBADCAFEULL);
}

TEST(FetchShaderSerialization, RejectsInvalidAttributeDomainTransactionally) {
    Shader::Gcn::FetchShaderData valid{};
    valid.vertex_offset_sgpr = 0;
    valid.instance_offset_sgpr = 1;
    valid.attributes.push_back({
        .semantic = 1,
        .dest_vgpr = 2,
        .num_elements = 4,
        .sgpr_base = 2,
        .dword_offset = 0,
        .instance_data = Shader::Gcn::VertexAttribute::None,
    });
    auto instanced = valid;
    instanced.attributes[0].instance_data = Shader::Gcn::VertexAttribute::Plain;

    Serialization::Archive valid_archive;
    valid.Serialize(valid_archive);
    auto valid_bytes = valid_archive.TakeOff();
    Serialization::Archive instanced_archive;
    instanced.Serialize(instanced_archive);
    const auto instanced_bytes = instanced_archive.TakeOff();
    const auto differences = DifferingByteOffsets(valid_bytes, instanced_bytes);
    ASSERT_EQ(differences.size(), 1U);

    valid_bytes[differences.front()] = 0xFFU;
    Serialization::Archive malformed{std::move(valid_bytes)};
    Shader::Gcn::FetchShaderData decoded{};
    decoded.vertex_offset_sgpr = 7;
    EXPECT_FALSE(decoded.Deserialize(malformed));
    EXPECT_EQ(decoded.vertex_offset_sgpr, 7);
    EXPECT_TRUE(decoded.attributes.empty());
}

TEST(CheckedSerialization, ValidVectorAndStringRoundTripUnchanged) {
    const std::vector<u32> expected_values{3, 5, 8, 13};
    const std::string expected_text{"semantic-cache"};
    Serialization::Archive output;
    Serialization::Writer writer{output};
    writer.Write(expected_values);
    writer.Write(expected_text);

    Serialization::Archive input{output.TakeOff()};
    Serialization::Reader reader{input};
    std::vector<u32> values;
    std::string text;
    ASSERT_TRUE(reader.Read(values, 16));
    ASSERT_TRUE(reader.Read(text, 64));
    EXPECT_TRUE(reader.IsValid());
    EXPECT_TRUE(input.IsEoS());
    EXPECT_EQ(values, expected_values);
    EXPECT_EQ(text, expected_text);
}

TEST(CheckedSerialization, ValidNestedContainersRoundTripUnchanged) {
    const std::vector<std::vector<u32>> expected{{}, {1}, {2, 3, 5}, {}};
    Serialization::Archive output;
    Serialization::Writer writer{output};
    writer.Write(expected);

    Serialization::Archive input{output.TakeOff()};
    Serialization::Reader reader{input};
    std::vector<std::vector<u32>> actual;
    ASSERT_TRUE(reader.Read(actual, 8));
    EXPECT_TRUE(reader.IsValid());
    EXPECT_TRUE(input.IsEoS());
    EXPECT_EQ(actual, expected);
}

TEST(CheckedSerialization, OversizedAndTruncatedContainersFailWithoutEscaping) {
    Serialization::Archive oversized_vector;
    Serialization::Writer vector_writer{oversized_vector};
    vector_writer.Write(size_t{33});
    Serialization::Archive vector_input{oversized_vector.TakeOff()};
    Serialization::Reader vector_reader{vector_input};
    std::vector<u32> values;
    EXPECT_FALSE(vector_reader.Read(values, 32));
    EXPECT_FALSE(vector_reader.IsValid());
    EXPECT_TRUE(values.empty());

    Serialization::Archive oversized_string;
    Serialization::Writer string_writer{oversized_string};
    string_writer.Write(size_t{129});
    Serialization::Archive string_input{oversized_string.TakeOff()};
    Serialization::Reader string_reader{string_input};
    std::string text;
    EXPECT_FALSE(string_reader.Read(text, 128));
    EXPECT_FALSE(string_reader.IsValid());
    EXPECT_TRUE(text.empty());

    Serialization::Archive truncated;
    Serialization::Writer truncated_writer{truncated};
    truncated_writer.Write(size_t{2});
    truncated_writer.Write(u32{7});
    Serialization::Archive truncated_input{truncated.TakeOff()};
    Serialization::Reader truncated_reader{truncated_input};
    EXPECT_FALSE(truncated_reader.Read(values, 2));
    EXPECT_FALSE(truncated_reader.IsValid());
    EXPECT_TRUE(values.empty());
}

TEST(ShaderCacheMetadata, RejectsPreInlineAccessMetadataWithoutPublishingOutputs) {
    MetadataFixture fixture;
    AmdGpu::Buffer inline_buffer{};
    inline_buffer.base_address = 1;
    inline_buffer.stride = 4;
    inline_buffer.num_records = 64;
    inline_buffer.data_format = static_cast<u32>(AmdGpu::DataFormat::Format32);
    inline_buffer.num_format = static_cast<u32>(AmdGpu::NumberFormat::Uint);
    fixture.info.buffers.push_back({
        .sharp_idx = std::numeric_limits<u32>::max(),
        .used_types = Shader::IR::Type::U32,
        .inline_cbuf = inline_buffer,
        .buffer_type = Shader::BufferType::Guest,
        .is_written = false,
        .is_formatted = false,
    });
    fixture.spec.buffers.push_back({.stride = 4, .is_storage = 1});
    fixture.permutation_id = Shader::ComputeShaderPermutationIdentity(
        fixture.info.pgm_hash, fixture.info.l_stage, fixture.spec);

    // This is valid metadata, including a valid semantic identity. The persisted false access
    // flags cannot distinguish a real read-only buffer from an old misclassified inline store.
    // Loading cached Info does not rerun resource tracking to repair those flags.
    Serialization::Archive current_input{fixture.Serialize()};
    Shader::Info current_info{};
    Shader::StageSpecialization current_spec{};
    std::optional<Shader::Gcn::FetchShaderData> current_fetch;
    u64 current_id{};
    ASSERT_TRUE(Shader::DeserializeShaderCacheMetadata(current_input, current_info, current_fetch,
                                                       current_spec, current_id));
    ASSERT_EQ(current_info.buffers.size(), 1U);
    EXPECT_FALSE(current_info.buffers.front().is_written);
    EXPECT_FALSE(current_info.buffers.front().is_formatted);
    EXPECT_EQ(current_id, fixture.permutation_id);
    EXPECT_EQ(current_id, Shader::ComputeShaderPermutationIdentity(
                              current_info.pgm_hash, current_info.l_stage, current_spec));

    auto bytes = fixture.Serialize();
    ASSERT_GE(bytes.size(), sizeof(u32));
    constexpr u32 stale_access_metadata_version = 7U;
    std::memcpy(bytes.data(), &stale_access_metadata_version, sizeof(u32));
    Serialization::Archive old_input{std::move(bytes)};
    Shader::Info info{};
    info.pgm_hash = 0xBADCAFEULL;
    Shader::StageSpecialization spec{};
    spec.info = &info;
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    u64 permutation_id = 0xDEADBEEFULL;

    EXPECT_FALSE(
        Shader::DeserializeShaderCacheMetadata(old_input, info, fetch, spec, permutation_id));
    EXPECT_EQ(info.pgm_hash, 0xBADCAFEULL);
    EXPECT_TRUE(info.buffers.empty());
    EXPECT_EQ(spec.info, &info);
    EXPECT_FALSE(fetch.has_value());
    EXPECT_EQ(permutation_id, 0xDEADBEEFULL);
}

TEST(ShaderCacheMetadata, RejectsPreBoundsGuardSpirvVersionWithoutPublishingOutputs) {
    const MetadataFixture fixture;
    Serialization::Archive current_input{fixture.Serialize()};
    Shader::Info current_info{};
    Shader::StageSpecialization current_spec{};
    std::optional<Shader::Gcn::FetchShaderData> current_fetch;
    u64 current_id{};
    ASSERT_TRUE(Shader::DeserializeShaderCacheMetadata(current_input, current_info, current_fetch,
                                                       current_spec, current_id));
    auto bytes = fixture.Serialize();
    ASSERT_GE(bytes.size(), 2 * sizeof(u32));
    // The second header word is the backend binary version, independent of the
    // metadata layout version. Version 6 modules lack the guest BDA bounds guard.
    constexpr u32 unguarded_binary_version = 6U;
    std::memcpy(bytes.data() + sizeof(u32), &unguarded_binary_version, sizeof(u32));
    Serialization::Archive input{std::move(bytes)};
    Shader::Info info{};
    info.pgm_hash = 0xBADCAFEULL;
    Shader::StageSpecialization spec{};
    spec.info = &info;
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    u64 permutation_id = 0xDEADBEEFULL;

    EXPECT_FALSE(Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id));
    EXPECT_EQ(info.pgm_hash, 0xBADCAFEULL);
    EXPECT_EQ(spec.info, &info);
    EXPECT_FALSE(fetch.has_value());
    EXPECT_EQ(permutation_id, 0xDEADBEEFULL);
}

TEST(ShaderCacheMetadata, RejectsNonAtomicFaultBitmapSpirvWithoutPublishingOutputs) {
    const MetadataFixture fixture;
    Serialization::Archive current_input{fixture.Serialize()};
    Shader::Info current_info{};
    Shader::StageSpecialization current_spec{};
    std::optional<Shader::Gcn::FetchShaderData> current_fetch;
    u64 current_id{};
    ASSERT_TRUE(Shader::DeserializeShaderCacheMetadata(current_input, current_info, current_fetch,
                                                       current_spec, current_id));
    auto bytes = fixture.Serialize();
    ASSERT_GE(bytes.size(), 2 * sizeof(u32));
    // Metadata layout 8 is unchanged; backend binary 7 still has a non-atomic bitmap update.
    constexpr u32 non_atomic_binary_version = 7U;
    std::memcpy(bytes.data() + sizeof(u32), &non_atomic_binary_version, sizeof(u32));
    Serialization::Archive input{std::move(bytes)};
    Shader::Info info{};
    info.pgm_hash = 0xBADCAFEULL;
    Shader::StageSpecialization spec{};
    spec.info = &info;
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    u64 permutation_id = 0xDEADBEEFULL;

    EXPECT_FALSE(Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id));
    EXPECT_EQ(info.pgm_hash, 0xBADCAFEULL);
    EXPECT_TRUE(info.buffers.empty());
    EXPECT_EQ(spec.info, &info);
    EXPECT_FALSE(fetch.has_value());
    EXPECT_EQ(permutation_id, 0xDEADBEEFULL);
}

TEST(ShaderCacheMetadata, EveryTruncatedPrefixFailsClosed) {
    const MetadataFixture fixture;
    const auto complete = fixture.Serialize();
    ASSERT_GT(complete.size(), 1U);
    for (size_t length = 0; length < complete.size(); ++length) {
        std::vector<u8> prefix{complete.begin(), complete.begin() + length};
        Serialization::Archive input{std::move(prefix)};
        Shader::Info info{};
        Shader::StageSpecialization spec{};
        std::optional<Shader::Gcn::FetchShaderData> fetch;
        u64 permutation_id{};
        EXPECT_FALSE(
            Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id))
            << "accepted truncated metadata length " << length << " of " << complete.size();
    }
}

TEST(ShaderCacheMetadata, FailedDecodeDoesNotPublishPartialOutputs) {
    const MetadataFixture fixture;
    auto malformed = fixture.Serialize();
    malformed.push_back(0xA5);

    Serialization::Archive input{std::move(malformed)};
    Shader::Info info{};
    info.stage = Shader::Stage::Compute;
    info.l_stage = Shader::LogicalStage::Compute;
    info.pgm_hash = 0xBADCAFEULL;
    Shader::StageSpecialization spec{};
    spec.info = &info;
    spec.runtime_info.Initialize(Shader::Stage::Compute);
    spec.user_data_mask = 0x5A;
    std::optional<Shader::Gcn::FetchShaderData> fetch{std::in_place};
    fetch->vertex_offset_sgpr = 7;
    u64 permutation_id = 0xDEADBEEFCAFEBABEULL;

    EXPECT_FALSE(Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id));
    EXPECT_EQ(info.stage, Shader::Stage::Compute);
    EXPECT_EQ(info.l_stage, Shader::LogicalStage::Compute);
    EXPECT_EQ(info.pgm_hash, 0xBADCAFEULL);
    EXPECT_EQ(spec.info, &info);
    EXPECT_EQ(spec.runtime_info.stage, Shader::Stage::Compute);
    EXPECT_EQ(spec.user_data_mask, 0x5AU);
    ASSERT_TRUE(fetch.has_value());
    EXPECT_EQ(fetch->vertex_offset_sgpr, 7);
    EXPECT_EQ(permutation_id, 0xDEADBEEFCAFEBABEULL);
}

TEST(ShaderCacheMetadata, OversizedSpecializationVectorsFailClosed) {
    MetadataFixture fixture;
    fixture.spec.vs_attribs.resize(33);
    fixture.permutation_id = Shader::ComputeShaderPermutationIdentity(
        fixture.info.pgm_hash, fixture.info.l_stage, fixture.spec);
    auto encoded = fixture.Serialize();

    Serialization::Archive input{std::move(encoded)};
    Shader::Info info{};
    Shader::StageSpecialization spec{};
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    u64 permutation_id{};
    EXPECT_FALSE(Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id));
}

TEST(ShaderCacheMetadata, InvalidStagesAndFragmentInputCountFailBeforeIdentity) {
    const auto expect_rejected = [](MetadataFixture fixture) {
        auto encoded = fixture.Serialize();
        Serialization::Archive input{std::move(encoded)};
        Shader::Info info{};
        Shader::StageSpecialization spec{};
        std::optional<Shader::Gcn::FetchShaderData> fetch;
        u64 permutation_id{};
        EXPECT_FALSE(
            Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id));
    };

    MetadataFixture invalid_runtime_stage;
    invalid_runtime_stage.spec.runtime_info.stage = static_cast<Shader::Stage>(99);
    expect_rejected(std::move(invalid_runtime_stage));

    MetadataFixture invalid_info_stage;
    invalid_info_stage.info.stage = static_cast<Shader::Stage>(99);
    expect_rejected(std::move(invalid_info_stage));

    MetadataFixture invalid_logical_stage;
    invalid_logical_stage.info.l_stage = static_cast<Shader::LogicalStage>(99);
    expect_rejected(std::move(invalid_logical_stage));

    MetadataFixture too_many_fragment_inputs;
    too_many_fragment_inputs.info.stage = Shader::Stage::Fragment;
    too_many_fragment_inputs.info.l_stage = Shader::LogicalStage::Fragment;
    too_many_fragment_inputs.spec.runtime_info.Initialize(Shader::Stage::Fragment);
    too_many_fragment_inputs.spec.runtime_info.fs_info.num_inputs = 33;
    expect_rejected(std::move(too_many_fragment_inputs));
}

TEST(ShaderCacheMetadata, RejectsFetchShaderInjectionAndFlagMismatch) {
    const auto expect_rejected = [](MetadataFixture fixture) {
        fixture.permutation_id = Shader::ComputeShaderPermutationIdentity(
            fixture.info.pgm_hash, fixture.info.l_stage, fixture.spec);
        auto encoded = fixture.Serialize();
        Serialization::Archive input{std::move(encoded)};
        Shader::Info info{};
        Shader::StageSpecialization spec{};
        std::optional<Shader::Gcn::FetchShaderData> fetch;
        u64 permutation_id{};
        EXPECT_FALSE(
            Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id));
    };

    MetadataFixture injected_fragment_fetch;
    injected_fragment_fetch.info.stage = Shader::Stage::Fragment;
    injected_fragment_fetch.info.l_stage = Shader::LogicalStage::Fragment;
    injected_fragment_fetch.spec.runtime_info.Initialize(Shader::Stage::Fragment);
    expect_rejected(std::move(injected_fragment_fetch));

    MetadataFixture missing_fetch_payload;
    missing_fetch_payload.spec.fetch_shader_data.reset();
    expect_rejected(std::move(missing_fetch_payload));

    MetadataFixture unexpected_fetch_payload;
    unexpected_fetch_payload.info.has_fetch_shader = false;
    expect_rejected(std::move(unexpected_fetch_payload));
}

TEST(ShaderCacheMetadata, OversizedSrtWalkerIsRejectedBeforeCodeRegistration) {
    Shader::ResetWalkerRegistrationStatsForTesting();
    Serialization::Archive encoded;
    Serialization::Writer writer{encoded};
    writer.Write(u32{(1U << 20) + 1U});
    writer.Write(u32{Shader::NUM_USER_DATA_REGS});

    Serialization::Archive input{encoded.TakeOff()};
    Shader::PersistentSrtInfo srt{};
    EXPECT_FALSE(srt.Deserialize(input));
    EXPECT_EQ(srt.walker_func, nullptr);
    EXPECT_EQ(Shader::WalkerRegistrationCountForTesting(), 0U);
    EXPECT_EQ(Shader::WalkerRegisteredBytesForTesting(), 0U);
}

TEST(ShaderCacheMetadata, NonzeroSrtWalkerIsRejectedWithoutExecutableAllocation) {
    Shader::ResetWalkerRegistrationStatsForTesting();
    constexpr std::array<u8, 5> PersistedCodeAndJunk{0x90, 0x90, 0xC3, 0xCC, 0xA5};
    Serialization::Archive encoded;
    Serialization::Writer writer{encoded};
    writer.Write(static_cast<u32>(PersistedCodeAndJunk.size() - 1));
    writer.Write(u32{Shader::NUM_USER_DATA_REGS});
    writer.Write(PersistedCodeAndJunk.data(), PersistedCodeAndJunk.size());

    Serialization::Archive input{encoded.TakeOff()};
    Shader::PersistentSrtInfo srt{};
    srt.walker_func = &TestPersistedWalker;
    srt.walker_func_size = 99;
    srt.flattened_bufsize_dw = 77;
    EXPECT_FALSE(srt.Deserialize(input));
    EXPECT_EQ(srt.walker_func, &TestPersistedWalker);
    EXPECT_EQ(srt.walker_func_size, 99U);
    EXPECT_EQ(srt.flattened_bufsize_dw, 77U);
    EXPECT_EQ(Shader::WalkerRegistrationCountForTesting(), 0U);
    EXPECT_EQ(Shader::WalkerRegisteredBytesForTesting(), 0U);
    EXPECT_EQ(input.RemainingBytes(), PersistedCodeAndJunk.size());
}

TEST(ShaderCacheMetadata, RuntimeSrtWalkerBytesAreNeverPersisted) {
    Shader::PersistentSrtInfo srt{};
    srt.walker_func = &TestPersistedWalker;
    srt.walker_func_size = 4;
    srt.flattened_bufsize_dw = 23;

    Serialization::Archive encoded;
    srt.Serialize(encoded);
    Serialization::Archive input{encoded.TakeOff()};
    Serialization::Reader reader{input};
    u32 persisted_code_size = 99;
    u32 flattened_size{};
    u32 reservation_count = 99;
    ASSERT_TRUE(reader.Read(persisted_code_size));
    ASSERT_TRUE(reader.Read(flattened_size));
    ASSERT_TRUE(reader.Read(reservation_count));
    EXPECT_EQ(persisted_code_size, 0U);
    EXPECT_EQ(flattened_size, 23U);
    EXPECT_EQ(reservation_count, 0U);
    EXPECT_TRUE(input.IsEoS());
}

TEST(ShaderCacheMetadata, RuntimeSrtWalkerMetadataIsNotPublished) {
    MetadataFixture fixture;
    fixture.info.srt_info.walker_func = &TestPersistedWalker;
    fixture.info.srt_info.walker_func_size = 4;
    fixture.info.srt_info.flattened_bufsize_dw = Shader::NUM_USER_DATA_REGS + 4;
    fixture.info.flattened_ud_buf.resize(fixture.info.srt_info.flattened_bufsize_dw);
    // The shape is otherwise valid: rejection must be the unpersisted executable pointer
    // program, not an incidental mismatch in flattened-buffer lengths.
    ASSERT_TRUE(Shader::ValidateShaderCacheMetadata(fixture.info, fixture.spec));

    Serialization::Archive encoded;
    EXPECT_FALSE(Shader::SerializeShaderCacheMetadata(encoded, fixture.info, fixture.spec,
                                                      fixture.permutation_id));
    EXPECT_EQ(encoded.SizeBytes(), 0U);
}

TEST(ShaderCacheMetadata, DeclarativeDynamicTableReservationRoundTripsWithoutNativeWalker) {
    MetadataFixture fixture;
    fixture.info.stage = Shader::Stage::Compute;
    fixture.info.l_stage = Shader::LogicalStage::Compute;
    fixture.info.has_fetch_shader = false;
    fixture.spec.runtime_info.Initialize(Shader::Stage::Compute);
    fixture.spec.fetch_shader_data.reset();
    fixture.spec.vs_attribs.clear();

    constexpr u32 TableEntries = Shader::NUM_IMAGES;
    constexpr u32 TableStrideDwords = 8;
    constexpr u32 TableSpanDwords = (TableEntries - 1U) * TableStrideDwords + 8U;
    fixture.info.srt_info.flattened_bufsize_dw = Shader::NUM_USER_DATA_REGS + TableSpanDwords;
    fixture.info.srt_info.range_reservations.push_back({
        .pointer_lo_flat_index = 0,
        .source_offset_dw = 3620,
        .destination_flat_offset_dw = Shader::NUM_USER_DATA_REGS,
        .span_dw = TableSpanDwords,
    });
    fixture.info.flattened_ud_buf.resize(fixture.info.srt_info.flattened_bufsize_dw);
    fixture.info.images.push_back({
        .sharp_idx = Shader::NUM_USER_DATA_REGS,
        .dynamic_descriptor_count = TableEntries,
        .dynamic_descriptor_stride_dw = TableStrideDwords,
        .requires_dynamic_oob_guard = true,
    });
    fixture.spec.images.push_back({
        .type = AmdGpu::ImageType::Color2D,
        .number_class = AmdGpu::NumberClass::Float,
        .num_bindings = TableEntries,
        .dynamic_descriptor_count = TableEntries,
        .dynamic_descriptor_stride_dw = TableStrideDwords,
        .requires_dynamic_oob_guard = true,
    });
    fixture.permutation_id = Shader::ComputeShaderPermutationIdentity(
        fixture.info.pgm_hash, fixture.info.l_stage, fixture.spec);

    Serialization::Archive encoded;
    ASSERT_TRUE(Shader::ValidateShaderCacheMetadata(fixture.info, fixture.spec));
    ASSERT_TRUE(Shader::SerializeShaderCacheMetadata(encoded, fixture.info, fixture.spec,
                                                     fixture.permutation_id));
    const auto bytes = encoded.TakeOff();
    Serialization::Archive component_input{std::vector<u8>{bytes}};
    Serialization::Reader header{component_input};
    u32 meta_version{};
    u32 binary_version{};
    u64 component_permutation_id{};
    ASSERT_TRUE(header.Read(meta_version));
    ASSERT_TRUE(header.Read(binary_version));
    ASSERT_TRUE(header.Read(component_permutation_id));
    Shader::Info component_info{};
    Shader::StageSpecialization component_spec{};
    component_spec.info = &component_info;
    ASSERT_TRUE(component_spec.Deserialize(component_input));
    ASSERT_TRUE(component_info.Deserialize(component_input));
    ASSERT_TRUE(Shader::ValidateShaderCacheMetadata(component_info, component_spec));

    Serialization::Archive input{std::vector<u8>{bytes}};
    Shader::Info info{};
    Shader::StageSpecialization spec{};
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    u64 permutation_id{};
    ASSERT_TRUE(Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id));
    ASSERT_EQ(info.images.size(), 1U);
    EXPECT_EQ(info.images[0].dynamic_descriptor_count, TableEntries);
    EXPECT_EQ(info.images[0].dynamic_descriptor_stride_dw, TableStrideDwords);
    EXPECT_TRUE(info.images[0].requires_dynamic_oob_guard);
    EXPECT_EQ(info.srt_info.range_reservations, fixture.info.srt_info.range_reservations);
    EXPECT_EQ(spec.images[0].num_bindings, TableEntries);
    EXPECT_EQ(permutation_id, fixture.permutation_id);
}

TEST(ShaderCacheMetadata, ValidRoundTripRestoresRuntimeFetchAndExactReuse) {
    const MetadataFixture fixture;
    auto encoded = fixture.Serialize();
    Serialization::Archive input{std::move(encoded)};
    Shader::Info info{};
    Shader::StageSpecialization spec{};
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    u64 permutation_id{};
    ASSERT_TRUE(Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id));
    EXPECT_TRUE(input.IsEoS());
    EXPECT_EQ(permutation_id, fixture.permutation_id);
    EXPECT_EQ(spec.runtime_info, fixture.spec.runtime_info);
    EXPECT_EQ(fetch, fixture.spec.fetch_shader_data);
    EXPECT_EQ(spec.info, &info);

    Vulkan::Program program{};
    auto stored_spec = spec;
    auto stored_info = info;
    ASSERT_TRUE(program.InsertPreloadPermutation({}, std::move(stored_spec), std::move(stored_info),
                                                 permutation_id));
    ASSERT_EQ(program.modules.size(), 1U);
    const auto reuse = program.InspectPreloadPermutation(spec, permutation_id);
    EXPECT_EQ(reuse.action, Vulkan::Program::PreloadAction::Reuse);
    EXPECT_EQ(program.modules.size(), 1U);

    auto conflicting = spec;
    conflicting.runtime_info.num_user_data++;
    EXPECT_EQ(program.InspectPreloadPermutation(conflicting, permutation_id).action,
              Vulkan::Program::PreloadAction::Conflict);
    EXPECT_EQ(program.modules.size(), 1U);
    EXPECT_EQ(program.modules.front().spec.info, program.modules.front().info.get());
}

TEST(ShaderCacheMetadata, ExactAtomicImageFormatRoundTrips) {
    MetadataFixture fixture;
    fixture.info.stage = Shader::Stage::Compute;
    fixture.info.l_stage = Shader::LogicalStage::Compute;
    fixture.info.has_fetch_shader = false;
    fixture.info.images.push_back({.sharp_idx = 0, .is_atomic = true, .is_written = true});
    fixture.spec.runtime_info.Initialize(Shader::Stage::Compute);
    fixture.spec.fetch_shader_data.reset();
    fixture.spec.images.push_back({
        .type = AmdGpu::ImageType::Color2D,
        .number_class = AmdGpu::NumberClass::Float,
        .is_storage = true,
        .atomic_data_format = AmdGpu::DataFormat::Format16_16,
        .atomic_number_format = AmdGpu::NumberFormat::Float,
        .num_bindings = 1,
    });
    fixture.permutation_id = Shader::ComputeShaderPermutationIdentity(
        fixture.info.pgm_hash, fixture.info.l_stage, fixture.spec);

    Serialization::Archive input{fixture.Serialize()};
    Shader::Info info{};
    Shader::StageSpecialization spec{};
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    u64 permutation_id{};
    ASSERT_TRUE(Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id));
    ASSERT_EQ(spec.images.size(), 1U);
    EXPECT_EQ(spec.images[0].atomic_data_format, AmdGpu::DataFormat::Format16_16);
    EXPECT_EQ(spec.images[0].atomic_number_format, AmdGpu::NumberFormat::Float);
    EXPECT_EQ(permutation_id, fixture.permutation_id);
}

TEST(ShaderCacheMetadata, RejectsAtomicResourceAndExactFormatMismatch) {
    const auto expect_rejected = [](bool resource_is_atomic, AmdGpu::DataFormat data_format,
                                    AmdGpu::NumberFormat number_format,
                                    AmdGpu::NumberClass number_class = AmdGpu::NumberClass::Sint) {
        MetadataFixture fixture;
        fixture.info.stage = Shader::Stage::Compute;
        fixture.info.l_stage = Shader::LogicalStage::Compute;
        fixture.info.has_fetch_shader = false;
        fixture.info.images.push_back(
            {.sharp_idx = 0, .is_atomic = resource_is_atomic, .is_written = true});
        fixture.spec.runtime_info.Initialize(Shader::Stage::Compute);
        fixture.spec.fetch_shader_data.reset();
        fixture.spec.images.push_back({
            .type = AmdGpu::ImageType::Color2D,
            .number_class = number_class,
            .is_storage = true,
            .atomic_data_format = data_format,
            .atomic_number_format = number_format,
            .num_bindings = 1,
        });
        fixture.permutation_id = Shader::ComputeShaderPermutationIdentity(
            fixture.info.pgm_hash, fixture.info.l_stage, fixture.spec);

        Serialization::Archive input{fixture.Serialize()};
        Shader::Info info{};
        Shader::StageSpecialization spec{};
        std::optional<Shader::Gcn::FetchShaderData> fetch;
        u64 permutation_id{};
        EXPECT_FALSE(
            Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id));
    };

    expect_rejected(true, AmdGpu::DataFormat::FormatInvalid, AmdGpu::NumberFormat::Unorm);
    expect_rejected(false, AmdGpu::DataFormat::Format32, AmdGpu::NumberFormat::Sint);
    expect_rejected(true, AmdGpu::DataFormat::Format8, AmdGpu::NumberFormat::Sint);
    expect_rejected(true, AmdGpu::DataFormat::Format32, AmdGpu::NumberFormat::Uint,
                    AmdGpu::NumberClass::Float);
}

TEST(ShaderCacheMetadata, RejectsOutOfBoundsSharpsAndDescriptorCounts) {
    const auto expect_rejected = [](u32 sharp_idx, u32 num_bindings) {
        MetadataFixture fixture;
        fixture.info.stage = Shader::Stage::Compute;
        fixture.info.l_stage = Shader::LogicalStage::Compute;
        fixture.info.has_fetch_shader = false;
        fixture.info.images.push_back({.sharp_idx = sharp_idx, .is_written = true});
        fixture.spec.runtime_info.Initialize(Shader::Stage::Compute);
        fixture.spec.fetch_shader_data.reset();
        fixture.spec.images.push_back({
            .type = AmdGpu::ImageType::Color2D,
            .number_class = AmdGpu::NumberClass::Float,
            .is_storage = true,
            .num_bindings = num_bindings,
        });
        fixture.permutation_id = Shader::ComputeShaderPermutationIdentity(
            fixture.info.pgm_hash, fixture.info.l_stage, fixture.spec);

        Serialization::Archive input{fixture.Serialize()};
        Shader::Info info{};
        Shader::StageSpecialization spec{};
        std::optional<Shader::Gcn::FetchShaderData> fetch;
        u64 permutation_id{};
        EXPECT_FALSE(
            Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id));
    };

    expect_rejected(Shader::NUM_USER_DATA_REGS - 4, 1);
    expect_rejected(0, Shader::NUM_IMAGES + 1);
}

TEST(ShaderCacheMetadata, RejectsUnsafeFetchAndInconsistentUserDataShapes) {
    const auto expect_rejected = [](const char* field, auto mutate) {
        SCOPED_TRACE(field);
        MetadataFixture fixture;
        mutate(fixture);
        fixture.permutation_id = Shader::ComputeShaderPermutationIdentity(
            fixture.info.pgm_hash, fixture.info.l_stage, fixture.spec);

        Serialization::Archive input{fixture.Serialize()};
        Shader::Info info{};
        Shader::StageSpecialization spec{};
        std::optional<Shader::Gcn::FetchShaderData> fetch;
        u64 permutation_id{};
        EXPECT_FALSE(
            Shader::DeserializeShaderCacheMetadata(input, info, fetch, spec, permutation_id));
    };

    expect_rejected("fetch shader pointer", [](auto& fixture) {
        fixture.info.fetch_shader_sgpr_base = Shader::NUM_USER_DATA_REGS - 1;
    });
    expect_rejected("vertex offset", [](auto& fixture) {
        fixture.spec.fetch_shader_data->vertex_offset_sgpr = Shader::NUM_USER_DATA_REGS;
    });
    expect_rejected("negative instance offset", [](auto& fixture) {
        fixture.spec.fetch_shader_data->instance_offset_sgpr = -2;
    });
    expect_rejected("attribute pointer", [](auto& fixture) {
        fixture.spec.fetch_shader_data->attributes[0].sgpr_base = Shader::NUM_USER_DATA_REGS - 1;
    });
    expect_rejected("attribute semantic", [](auto& fixture) {
        fixture.spec.fetch_shader_data->attributes[0].semantic = Shader::IR::NumParams;
    });
    expect_rejected("attribute shape", [](auto& fixture) {
        fixture.spec.fetch_shader_data->attributes[0].num_elements = 0;
    });
    expect_rejected("specialized attribute count",
                    [](auto& fixture) { fixture.spec.vs_attribs.clear(); });
    expect_rejected("user-data mask", [](auto& fixture) { fixture.spec.user_data_mask = 1; });
    expect_rejected("tessellation pointer", [](auto& fixture) {
        fixture.info.stage = Shader::Stage::Hull;
        fixture.info.l_stage = Shader::LogicalStage::TessellationControl;
        fixture.info.has_fetch_shader = false;
        fixture.info.tess_consts_dword_offset = 0;
        fixture.info.tess_consts_ptr_base = Shader::IR::ScalarReg::S15;
        fixture.spec.runtime_info.Initialize(Shader::Stage::Hull);
        fixture.spec.fetch_shader_data.reset();
        fixture.spec.vs_attribs.clear();
    });
}

TEST(StageSpecializationOwnership, GeometryCopyShaderSurvivesMoveAndSourceDestruction) {
    constexpr std::array<u32, 4> CopyShader{0xBE802000U, 0xBF8C0F70U, 0x7E000200U, 0xBF810000U};
    Shader::StageSpecialization retained{};
    {
        std::vector<u32> temporary_copy{CopyShader.begin(), CopyShader.end()};
        Shader::StageSpecialization source{};
        source.runtime_info.Initialize(Shader::Stage::Geometry);
        source.runtime_info.gs_info.vs_copy = temporary_copy;
        retained = std::move(source);
    }

    std::vector<std::vector<u32>> heap_churn(256, std::vector<u32>(CopyShader.size(), 0xDEADBEEFU));
    ASSERT_FALSE(heap_churn.empty());
    EXPECT_TRUE(std::ranges::equal(retained.runtime_info.gs_info.vs_copy, CopyShader));
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

TEST(ShaderPermutationIdentity, PhysicalAndLogicalStagePairsMatchPipelineRules) {
    constexpr std::array physical_stages{
        Shader::Stage::Fragment, Shader::Stage::Vertex, Shader::Stage::Geometry,
        Shader::Stage::Export,   Shader::Stage::Hull,   Shader::Stage::Local,
        Shader::Stage::Compute,
    };
    constexpr std::array logical_stages{
        Shader::LogicalStage::Fragment,         Shader::LogicalStage::TessellationControl,
        Shader::LogicalStage::TessellationEval, Shader::LogicalStage::Vertex,
        Shader::LogicalStage::Geometry,         Shader::LogicalStage::Compute,
        Shader::LogicalStage::NumLogicalStages,
    };
    const auto expected = [](Shader::Stage physical, Shader::LogicalStage logical) {
        return (physical == Shader::Stage::Fragment && logical == Shader::LogicalStage::Fragment) ||
               (physical == Shader::Stage::Hull &&
                logical == Shader::LogicalStage::TessellationControl) ||
               ((physical == Shader::Stage::Vertex || physical == Shader::Stage::Export) &&
                logical == Shader::LogicalStage::TessellationEval) ||
               ((physical == Shader::Stage::Vertex || physical == Shader::Stage::Export ||
                 physical == Shader::Stage::Local) &&
                logical == Shader::LogicalStage::Vertex) ||
               (physical == Shader::Stage::Geometry && logical == Shader::LogicalStage::Geometry) ||
               (physical == Shader::Stage::Compute && logical == Shader::LogicalStage::Compute);
    };

    for (const auto physical : physical_stages) {
        for (const auto logical : logical_stages) {
            EXPECT_EQ(Shader::IsPhysicalStageCompatible(physical, logical),
                      expected(physical, logical))
                << "physical=" << static_cast<u32>(physical)
                << " logical=" << static_cast<u32>(logical);
        }
    }
    for (const auto logical : logical_stages) {
        EXPECT_FALSE(Shader::IsPhysicalStageCompatible(static_cast<Shader::Stage>(99), logical));
    }
}

TEST(StageSpecializationSemanticEquality, BoundnessIsRuntimeStateNotCodeShape) {
    auto unbound = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    auto bound = unbound;
    unbound.bitset.reset();
    bound.bitset.set(0);

    ExpectSameSemantic(unbound, bound);
}

TEST(StageSpecializationSemanticEquality, BindingOriginsAreConditionallySemantic) {
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
    user_data.user_data_mask = 1U << 7;
    auto user_data_origin = user_data;
    user_data_origin.start.user_data = 4;
    ExpectDifferentSemantic(user_data, user_data_origin);

    auto changed_mask = user_data;
    changed_mask.user_data_mask |= 1U << 3;
    ExpectDifferentSemantic(user_data, changed_mask);
}

TEST(ShaderPermutationPreload, ReusesExactMatchesAndRejectsEitherKindOfConflict) {
    Vulkan::Program program{};
    auto stored = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    const u64 stored_identity = Identity(stored);
    Shader::Info stored_info{};
    ASSERT_TRUE(program.InsertPreloadPermutation({}, std::move(stored), std::move(stored_info),
                                                 stored_identity));
    ASSERT_EQ(program.modules.size(), 1U);
    ASSERT_NE(program.modules.front().info, nullptr);
    EXPECT_EQ(program.modules.front().spec.info, program.modules.front().info.get());

    auto exact_match = program.modules.front().spec;
    const auto reuse = program.InspectPreloadPermutation(exact_match, stored_identity);
    EXPECT_EQ(reuse.action, Vulkan::Program::PreloadAction::Reuse);
    EXPECT_EQ(reuse.module, &program.modules.front());

    auto identity_collision = exact_match;
    identity_collision.images.front().number_class = AmdGpu::NumberClass::Uint;
    EXPECT_EQ(program.InspectPreloadPermutation(identity_collision, stored_identity).action,
              Vulkan::Program::PreloadAction::Conflict);

    EXPECT_EQ(program.InspectPreloadPermutation(exact_match, stored_identity + 1).action,
              Vulkan::Program::PreloadAction::Conflict);
}

TEST(ShaderPermutationPreload, ReverseEncounterOrderKeepsTheSameSemanticIds) {
    Shader::Info first_info{};
    Shader::Info second_info{};
    auto first = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    auto second = MakeImageSpecialization(AmdGpu::NumberClass::Uint, RgbZeroMapping());
    first.info = &first_info;
    second.info = &second_info;
    const u64 first_id = Identity(first);
    const u64 second_id = Identity(second);

    const auto populate =
        [](Vulkan::Program& program,
           std::initializer_list<std::pair<Shader::StageSpecialization, u64>> permutations) {
            for (auto [spec, id] : permutations) {
                Shader::Info info{};
                ASSERT_TRUE(
                    program.InsertPreloadPermutation({}, std::move(spec), std::move(info), id));
            }
        };

    Vulkan::Program forward{};
    populate(forward, {{first, first_id}, {second, second_id}});
    Vulkan::Program reverse{};
    populate(reverse, {{second, second_id}, {first, first_id}});

    ASSERT_EQ(forward.modules.size(), 2U);
    ASSERT_EQ(reverse.modules.size(), 2U);
    EXPECT_EQ(forward.modules[0].permutation_id, reverse.modules[1].permutation_id);
    EXPECT_EQ(forward.modules[1].permutation_id, reverse.modules[0].permutation_id);
    EXPECT_EQ(forward.InspectPreloadPermutation(first, first_id).action,
              Vulkan::Program::PreloadAction::Reuse);
    EXPECT_EQ(reverse.InspectPreloadPermutation(first, first_id).action,
              Vulkan::Program::PreloadAction::Reuse);
    EXPECT_EQ(forward.InspectPreloadPermutation(second, second_id).action,
              Vulkan::Program::PreloadAction::Reuse);
    EXPECT_EQ(reverse.InspectPreloadPermutation(second, second_id).action,
              Vulkan::Program::PreloadAction::Reuse);
}

TEST(ShaderPermutationPreload, AuthoritativePostTranslationSpecIsRecheckedAfterProbeMiss) {
    Shader::Info stored_info{};
    auto stored = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
    const u64 stored_id = Identity(stored);
    Vulkan::Program program{};
    ASSERT_TRUE(
        program.InsertPreloadPermutation({}, std::move(stored), std::move(stored_info), stored_id));

    auto stale_probe = program.modules.front().spec;
    stale_probe.images.clear();
    EXPECT_EQ(program.InspectPreloadPermutation(stale_probe, Identity(stale_probe)).action,
              Vulkan::Program::PreloadAction::Insert);

    const auto authoritative = program.modules.front().spec;
    const auto retry = program.InspectPreloadPermutation(authoritative, stored_id);
    EXPECT_EQ(retry.action, Vulkan::Program::PreloadAction::Reuse);
    EXPECT_EQ(retry.module, &program.modules.front());
    EXPECT_EQ(program.modules.size(), 1U);
}

TEST(ShaderPermutationPreload, BatchConflictPublishesNothing) {
    const Shader::ProgramCacheKey existing_key{
        .program_hash = 0x100,
        .stage = Shader::Stage::Compute,
        .logical_stage = Shader::LogicalStage::Compute,
    };
    const Shader::ProgramCacheKey conflicting_key{
        .program_hash = 0x200,
        .stage = Shader::Stage::Compute,
        .logical_stage = Shader::LogicalStage::Compute,
    };
    Vulkan::ProgramCacheMap programs;
    auto existing_program = std::make_unique<Vulkan::Program>();
    existing_program->info.pgm_hash = existing_key.program_hash;
    existing_program->info.stage = existing_key.stage;
    existing_program->info.l_stage = existing_key.logical_stage;
    programs.try_emplace(existing_key, std::move(existing_program));
    auto conflicting_program = std::make_unique<Vulkan::Program>();
    conflicting_program->info.pgm_hash = conflicting_key.program_hash;
    conflicting_program->info.stage = conflicting_key.stage;
    conflicting_program->info.l_stage = conflicting_key.logical_stage;
    programs.try_emplace(conflicting_key, std::move(conflicting_program));

    auto make_pending = [](const Shader::ProgramCacheKey& key, Vulkan::Program* existing,
                           bool new_program, u64 permutation_id) {
        Shader::Info info{};
        info.pgm_hash = key.program_hash;
        info.stage = key.stage;
        info.l_stage = key.logical_stage;
        Shader::StageSpecialization spec{};
        spec.runtime_info.Initialize(key.stage);
        spec.info = &info;
        return Vulkan::PendingProgramPermutation{
            .cache_key = key,
            .existing_program = existing,
            .new_program = new_program ? std::make_unique<Vulkan::Program>() : nullptr,
            .module = Vulkan::Program::Module{{}, std::move(spec), std::move(info), permutation_id},
        };
    };

    std::vector<Vulkan::PendingProgramPermutation> pending;
    pending.emplace_back(make_pending(existing_key, programs.at(existing_key).get(), false, 1));
    pending.emplace_back(make_pending(conflicting_key, nullptr, true, 2));
    EXPECT_FALSE(Vulkan::PublishPreloadedPrograms(programs, pending));
    EXPECT_TRUE(programs.at(existing_key)->modules.empty());
    EXPECT_TRUE(programs.at(conflicting_key)->modules.empty());
    EXPECT_EQ(programs.size(), 2U);
}

TEST(ShaderPermutationPreload, BatchPublicationPreservesOwnedInfoAddresses) {
    const Shader::ProgramCacheKey existing_key{
        .program_hash = 0x300,
        .stage = Shader::Stage::Compute,
        .logical_stage = Shader::LogicalStage::Compute,
    };
    const Shader::ProgramCacheKey new_key{
        .program_hash = 0x400,
        .stage = Shader::Stage::Vertex,
        .logical_stage = Shader::LogicalStage::Vertex,
    };
    Vulkan::ProgramCacheMap programs;
    auto existing_program = std::make_unique<Vulkan::Program>();
    existing_program->info.pgm_hash = existing_key.program_hash;
    existing_program->info.stage = existing_key.stage;
    existing_program->info.l_stage = existing_key.logical_stage;
    programs.try_emplace(existing_key, std::move(existing_program));

    const auto make_pending = [](const Shader::ProgramCacheKey& key, Vulkan::Program* existing,
                                 bool new_program, u64 permutation_id) {
        Shader::Info info{};
        info.pgm_hash = key.program_hash;
        info.stage = key.stage;
        info.l_stage = key.logical_stage;
        Shader::StageSpecialization spec{};
        spec.runtime_info.Initialize(key.stage);
        spec.info = &info;
        auto program = new_program ? std::make_unique<Vulkan::Program>() : nullptr;
        if (program) {
            program->info = info;
        }
        return Vulkan::PendingProgramPermutation{
            .cache_key = key,
            .existing_program = existing,
            .new_program = std::move(program),
            .module = Vulkan::Program::Module{{}, std::move(spec), std::move(info), permutation_id},
        };
    };

    std::vector<Vulkan::PendingProgramPermutation> pending;
    pending.emplace_back(make_pending(existing_key, programs.at(existing_key).get(), false, 3));
    pending.emplace_back(make_pending(new_key, nullptr, true, 4));
    const Shader::Info* existing_info = pending[0].module.info.get();
    const Shader::Info* new_info = pending[1].module.info.get();

    ASSERT_TRUE(Vulkan::PublishPreloadedPrograms(programs, pending));
    ASSERT_EQ(programs.at(existing_key)->modules.size(), 1U);
    ASSERT_EQ(programs.at(new_key)->modules.size(), 1U);
    EXPECT_EQ(programs.at(existing_key)->modules.front().info.get(), existing_info);
    EXPECT_EQ(programs.at(existing_key)->modules.front().spec.info, existing_info);
    EXPECT_EQ(programs.at(new_key)->modules.front().info.get(), new_info);
    EXPECT_EQ(programs.at(new_key)->modules.front().spec.info, new_info);
}

TEST(StageSpecializationOwnership, ProgramRelocationKeepsEveryModuleInfoRebound) {
    Vulkan::Program program{};
    constexpr size_t ModuleCount = Vulkan::Program::MaxPermutations + 3;
    for (size_t index = 0; index < ModuleCount; ++index) {
        auto spec = MakeImageSpecialization(AmdGpu::NumberClass::Float, RgbaMapping());
        spec.images.front().num_bindings = static_cast<u32>(index + 1);
        Shader::Info info{};
        spec.info = &info;
        const u64 permutation_id = Identity(spec);
        ASSERT_TRUE(
            program.InsertPreloadPermutation({}, std::move(spec), std::move(info), permutation_id));
    }
    ASSERT_GT(program.modules.size(), Vulkan::Program::MaxPermutations);
    for (const auto& module : program.modules) {
        ASSERT_NE(module.info, nullptr);
        EXPECT_EQ(module.spec.info, module.info.get());
    }

    Vulkan::Program moved = std::move(program);
    ASSERT_EQ(moved.modules.size(), ModuleCount);
    for (const auto& module : moved.modules) {
        ASSERT_NE(module.info, nullptr);
        EXPECT_EQ(module.spec.info, module.info.get());
    }
}

} // namespace
