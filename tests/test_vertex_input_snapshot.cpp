// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/specialization.h"

namespace Shader::Gcn {
namespace {

AmdGpu::Buffer MakeBuffer(u64 address, u32 stride, u32 records, u32 format) {
    AmdGpu::Buffer buffer{};
    buffer.base_address = address;
    buffer.stride = stride;
    buffer.num_records = records;
    buffer.data_format = format;
    buffer.num_format = 1;
    buffer.dst_sel_x = 0;
    buffer.dst_sel_y = 1;
    buffer.dst_sel_z = 2;
    buffer.dst_sel_w = 3;
    return buffer;
}

TEST(VertexInputSnapshot, CapturesEachDescriptorOnceForAllDrawPhases) {
    FetchShaderData fetch{};
    fetch.attributes.push_back(VertexAttribute{.semantic = 7});
    fetch.attributes.push_back(VertexAttribute{.semantic = 2});

    const std::array generation_a{
        MakeBuffer(0x1000, 16, 64, 4),
        MakeBuffer(0x2000, 32, 128, 11),
    };
    const std::array generation_b{
        MakeBuffer(0x9000, 4, 8, 1),
        MakeBuffer(0xa000, 8, 16, 2),
    };
    auto live = generation_a;
    size_t reads = 0;

    const auto snapshot = VertexInputSnapshot::Capture(fetch, [&](const VertexAttribute&) {
        const auto index = reads++;
        const auto captured = live[index];
        live[index] = generation_b[index];
        return captured;
    });

    ASSERT_TRUE(snapshot.valid);
    ASSERT_EQ(reads, generation_a.size());
    ASSERT_EQ(snapshot.buffers.size(), generation_a.size());
    for (size_t phase = 0; phase < 3; ++phase) {
        EXPECT_EQ(snapshot.buffers[0], generation_a[0]);
        EXPECT_EQ(snapshot.buffers[1], generation_a[1]);
    }
    EXPECT_EQ(live, generation_b);
}

TEST(VertexInputSnapshot, RejectsUnsupportedAttributeCountsWithoutReadingOrThrowing) {
    FetchShaderData fetch{};
    fetch.attributes.resize(MaxVertexBufferCount + 1);
    size_t reads = 0;

    EXPECT_NO_THROW({
        const auto snapshot = VertexInputSnapshot::Capture(
            fetch, [&](const VertexAttribute&) {
                ++reads;
                return AmdGpu::Buffer{};
            });
        EXPECT_FALSE(snapshot.valid);
        EXPECT_TRUE(snapshot.buffers.empty());
    });
    EXPECT_EQ(reads, 0u);
}

TEST(StageResourceSnapshot, CapturesEveryResourceOnceForSpecializationAndBinding) {
    const std::array buffer_a{
        MakeBuffer(0x1100, 16, 64, 4),
        MakeBuffer(0x2200, 32, 128, 11),
    };
    const std::array buffer_b{
        MakeBuffer(0x9100, 4, 8, 1),
        MakeBuffer(0xa200, 8, 16, 2),
    };
    std::array<AmdGpu::Image, 2> image_a{};
    image_a[0].base_address = 0x330;
    image_a[1].base_address = 0x440;
    std::array<AmdGpu::Image, 2> image_b{};
    image_b[0].base_address = 0xb30;
    image_b[1].base_address = 0xc40;
    std::array<AmdGpu::Sampler, 2> sampler_a{};
    sampler_a[0].raw0 = 0x1234;
    sampler_a[1].raw0 = 0x5678;
    std::array<AmdGpu::Sampler, 2> sampler_b{};
    sampler_b[0].raw0 = 0x9abc;
    sampler_b[1].raw0 = 0xdef0;
    std::array<AmdGpu::Image, 1> fmask_a{};
    fmask_a[0].base_address = 0x550;
    std::array<AmdGpu::Image, 1> fmask_b{};
    fmask_b[0].base_address = 0xd50;

    auto live_buffers = buffer_a;
    auto live_images = image_a;
    auto live_samplers = sampler_a;
    auto live_fmasks = fmask_a;
    size_t buffer_reads{};
    size_t image_reads{};
    size_t sampler_reads{};
    size_t fmask_reads{};
    const auto snapshot = StageResourceSnapshot::Capture(
        buffer_a.size(), image_a.size(), sampler_a.size(), fmask_a.size(),
        [&](size_t index) {
            ++buffer_reads;
            const auto value = live_buffers[index];
            live_buffers[index] = buffer_b[index];
            return value;
        },
        [&](size_t index) {
            ++image_reads;
            const auto value = live_images[index];
            live_images[index] = image_b[index];
            return value;
        },
        [&](size_t index) {
            ++sampler_reads;
            const auto value = live_samplers[index];
            live_samplers[index] = sampler_b[index];
            return value;
        },
        [&](size_t index) {
            ++fmask_reads;
            const auto value = live_fmasks[index];
            live_fmasks[index] = fmask_b[index];
            return value;
        });

    ASSERT_TRUE(snapshot.valid);
    EXPECT_EQ(buffer_reads, buffer_a.size());
    EXPECT_EQ(image_reads, image_a.size());
    EXPECT_EQ(sampler_reads, sampler_a.size());
    EXPECT_EQ(fmask_reads, fmask_a.size());
    EXPECT_TRUE(std::ranges::equal(snapshot.buffers, buffer_a));
    EXPECT_TRUE(std::ranges::equal(snapshot.images, image_a));
    EXPECT_TRUE(std::ranges::equal(snapshot.samplers, sampler_a));
    EXPECT_TRUE(std::ranges::equal(snapshot.fmasks, fmask_a));

    // Both pipeline specialization and descriptor binding must consume these captured values,
    // even after every live descriptor has advanced to another guest generation.
    for (size_t phase = 0; phase < 2; ++phase) {
        EXPECT_EQ(snapshot.buffers[0], buffer_a[0]);
        EXPECT_EQ(snapshot.images[1], image_a[1]);
        EXPECT_EQ(snapshot.samplers[0], sampler_a[0]);
        EXPECT_EQ(snapshot.fmasks[0], fmask_a[0]);
    }
    EXPECT_EQ(live_buffers, buffer_b);
    EXPECT_EQ(live_images, image_b);
    EXPECT_EQ(live_samplers, sampler_b);
    EXPECT_EQ(live_fmasks, fmask_b);
}

TEST(StageResourceSnapshot, RejectsUnsupportedCountsWithoutPartialReads) {
    size_t reads{};
    const auto reader = [&](size_t) {
        ++reads;
        return AmdGpu::Buffer{};
    };
    const auto image_reader = [&](size_t) {
        ++reads;
        return AmdGpu::Image{};
    };
    const auto sampler_reader = [&](size_t) {
        ++reads;
        return AmdGpu::Sampler{};
    };
    const auto fmask_reader = [&](size_t) {
        ++reads;
        return AmdGpu::Image{};
    };

    const auto snapshot = StageResourceSnapshot::Capture(
        StageResourceSnapshot::MaxBuffers + 1, 0, 0, 0, reader, image_reader, sampler_reader,
        fmask_reader);
    EXPECT_FALSE(snapshot.valid);
    EXPECT_TRUE(snapshot.buffers.empty());
    EXPECT_TRUE(snapshot.images.empty());
    EXPECT_TRUE(snapshot.samplers.empty());
    EXPECT_TRUE(snapshot.fmasks.empty());
    EXPECT_EQ(reads, 0u);
}

TEST(StageResourceSnapshot, ProductionPipelineAndBindingsShareTheSameCapture) {
    const auto read_source = [](const char* path) {
        std::ifstream stream(path, std::ios::binary);
        EXPECT_TRUE(stream.is_open());
        return std::string(std::istreambuf_iterator<char>{stream},
                           std::istreambuf_iterator<char>{});
    };
    const std::string pipeline = read_source(STAGE_RESOURCE_PIPELINE_SOURCE_PATH);
    const std::string rasterizer = read_source(STAGE_RESOURCE_RASTERIZER_SOURCE_PATH);

    EXPECT_NE(pipeline.find("StageResourceSnapshot::Capture(program->info)"), std::string::npos);
    EXPECT_NE(pipeline.find("StageResourceSnapshot::Capture(info)"), std::string::npos);
    EXPECT_NE(pipeline.find("StageSpecialization(program->info, runtime_info, profile, start,\n"
                            "                                                &vertex_inputs, &resources)"),
              std::string::npos);
    EXPECT_NE(pipeline.find("StageSpecialization(info, runtime_info, profile, binding, &vertex_inputs,\n"
                            "                                            &resources)"),
              std::string::npos);

    size_t graphics_calls{};
    for (size_t pos{}; (pos = rasterizer.find("BindResources(pipeline, &pipeline_result.resources)",
                                               pos)) != std::string::npos;
         ++graphics_calls, ++pos) {
    }
    EXPECT_EQ(graphics_calls, 2u);
    EXPECT_NE(rasterizer.find("resources.buffers[index]"), std::string::npos);
    EXPECT_NE(rasterizer.find("resources.images[image_resource_index]"), std::string::npos);
    EXPECT_NE(rasterizer.find("resources.samplers[sampler_index]"), std::string::npos);

    const auto bind_buffers = rasterizer.find("bool Rasterizer::BindBuffers(");
    const auto bind_textures = rasterizer.find("void Rasterizer::BindTextures(");
    const auto next_function = rasterizer.find("void Rasterizer::UpdateDynamicState(", bind_textures);
    ASSERT_NE(bind_buffers, std::string::npos);
    ASSERT_NE(bind_textures, std::string::npos);
    ASSERT_NE(next_function, std::string::npos);
    const auto buffers_body = rasterizer.substr(bind_buffers, bind_textures - bind_buffers);
    const auto textures_body = rasterizer.substr(bind_textures, next_function - bind_textures);
    EXPECT_EQ(buffers_body.find("GetSharp(stage)"), std::string::npos);
    EXPECT_EQ(textures_body.find("GetSharp(stage)"), std::string::npos);
}

TEST(CompileResourceSnapshot, ReplaysTheCapturedFlattenedDescriptorsWithoutLiveRereads) {
    Info info{};
    std::array<u32, NUM_USER_DATA_REGS> live{};
    for (u32 index = 0; index < live.size(); ++index) {
        live[index] = 0x1000u + index;
    }
    info.srt_info.flattened_bufsize_dw = static_cast<u32>(live.size());
    info.flattened_ud_buf.assign(live.begin(), live.end());
    const auto captured = CompileResourceSnapshot::Capture(info);
    ASSERT_TRUE(captured.valid);
    ASSERT_EQ(captured.flattened_user_data.size(), live.size());

    for (u32 index = 0; index < live.size(); ++index) {
        live[index] = 0x9000u + index;
    }
    info.flattened_ud_buf.assign(live.begin(), live.end());
    info.UseCompileResourceSnapshot(captured);
    ASSERT_TRUE(info.ReplayCompileResourceSnapshot());

    EXPECT_TRUE(std::ranges::equal(info.flattened_ud_buf, captured.flattened_user_data));
    EXPECT_FALSE(std::ranges::equal(info.flattened_ud_buf, live));
}

TEST(CompileResourceSnapshot, ReplaysTheCapturedFetchShaderWithoutLiveCodeRereads) {
    Info info{};
    info.has_fetch_shader = true;
    info.fetch_shader_sgpr_base = 0;

    FetchShaderData captured{};
    captured.size = 24;
    captured.vertex_offset_sgpr = 6;
    captured.instance_offset_sgpr = 8;
    captured.attributes.push_back(VertexAttribute{
        .semantic = 3,
        .dest_vgpr = 5,
        .num_elements = 4,
        .sgpr_base = 2,
        .dword_offset = 12,
        .instance_data = VertexAttribute::Plain,
        .inst_offset = 16,
        .data_format = 11,
        .num_format = 4,
    });

    info.UseCompileFetchShaderSnapshot(captured);
    const auto* replayed = info.CompileFetchShaderSnapshot();
    ASSERT_NE(replayed, nullptr);
    EXPECT_EQ(replayed, &captured);
    EXPECT_EQ(*replayed, captured);
    ASSERT_EQ(replayed->attributes.size(), 1u);
    EXPECT_EQ(replayed->attributes[0].inst_offset, 16u);
    EXPECT_EQ(replayed->attributes[0].data_format, 11u);
    EXPECT_EQ(replayed->attributes[0].num_format, 4u);

    std::ifstream stream{FETCH_SHADER_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string source{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};
    const auto parse = source.find("std::optional<FetchShaderData> ParseFetchShader(");
    const auto replay = source.find("if (info.compile_fetch_shader_snapshot)", parse);
    const auto live = source.find("GetFetchShaderCode(info", parse);
    ASSERT_NE(parse, std::string::npos);
    ASSERT_NE(replay, std::string::npos);
    ASSERT_NE(live, std::string::npos);
    EXPECT_LT(replay, live);
    EXPECT_NE(source.substr(replay, live - replay).find("return *info.compile_fetch_shader_snapshot"),
              std::string::npos);

}

TEST(CompileResourceSnapshot, CachedPermutationCompilationConsumesTheExactCapturedImage) {
    std::ifstream stream{STAGE_RESOURCE_PIPELINE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const std::string pipeline{std::istreambuf_iterator<char>{stream},
                               std::istreambuf_iterator<char>{}};

    const auto refresh = pipeline.find("info.RefreshFlatBuf();");
    const auto capture = pipeline.find("CompileResourceSnapshot::Capture(info)", refresh);
    const auto resources = pipeline.find("StageResourceSnapshot::Capture(info)", capture);
    const auto compile = pipeline.find("module = CompileModule(new_info", resources);
    ASSERT_NE(refresh, std::string::npos);
    ASSERT_NE(capture, std::string::npos);
    ASSERT_NE(resources, std::string::npos);
    ASSERT_NE(compile, std::string::npos);
    EXPECT_LT(refresh, capture);
    EXPECT_LT(capture, resources);
    EXPECT_LT(resources, compile);
    EXPECT_NE(pipeline.substr(compile, 240).find("&compile_resources"), std::string::npos);

    const auto compile_function = pipeline.find("vk::ShaderModule PipelineCache::CompileModule(");
    const auto translate = pipeline.find("Shader::TranslateProgram(", compile_function);
    ASSERT_NE(compile_function, std::string::npos);
    ASSERT_NE(translate, std::string::npos);
    const auto use_snapshot =
        pipeline.find("info.UseCompileResourceSnapshot(*compile_resources)", compile_function);
    ASSERT_NE(use_snapshot, std::string::npos);
    EXPECT_LT(use_snapshot, translate);

    const auto fetch_capture =
        pipeline.find("const auto* compile_fetch_shader =", resources);
    const auto fetch_replay =
        pipeline.find("info.UseCompileFetchShaderSnapshot(*compile_fetch_shader)",
                      compile_function);
    ASSERT_NE(fetch_capture, std::string::npos);
    ASSERT_NE(fetch_replay, std::string::npos);
    EXPECT_LT(fetch_capture, compile);
    EXPECT_LT(fetch_replay, translate);
    EXPECT_NE(pipeline.substr(compile, 320).find("compile_fetch_shader"), std::string::npos);
}

} // namespace
} // namespace Shader::Gcn
