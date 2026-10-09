// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace VideoCore {
namespace {

std::string ReadText(const char* path) {
    std::ifstream file{path, std::ios::binary};
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

TEST(SrtMemorySyncSource, PipelineSelectionUsesOneCoherentReaderAndSealsBeforePublication) {
    const auto pipeline = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto rasterizer = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);

    EXPECT_NE(pipeline.find("SrtReadSession srt_read_session{page_manager, buffer_cache}"),
              std::string::npos);
    EXPECT_NE(pipeline.find("MakeSrtMemoryReader(srt_read_session)"), std::string::npos);
    EXPECT_NE(pipeline.find("Shader::TranslateProgram(code, pools, info, runtime_info, profile,"),
              std::string::npos);
    EXPECT_NE(pipeline.find("info.RefreshFlatBuf(reader)"), std::string::npos);
    EXPECT_NE(pipeline.find("srt_read_session.Seal()"), std::string::npos);
    EXPECT_NE(pipeline.find("EmitSPIRV(profile, runtime_info, *ir_program,"), std::string::npos);
    EXPECT_EQ(pipeline.find("compile_guest_memory_reader"), std::string::npos);
    EXPECT_NE(pipeline.find("auto aggregate_snapshot = program->info"), std::string::npos);
    EXPECT_NE(pipeline.find("auto selected_info = *candidate.info"), std::string::npos);
    EXPECT_EQ(pipeline.find("auto& selected_info = *candidate.info"), std::string::npos);

    const auto cached_probe = pipeline.find("if (const auto probe =");
    const auto cached_seal = pipeline.find("if (!seal_snapshot())", cached_probe);
    const auto cached_publish =
        pipeline.find("*probe->info = std::move(*selected_info)", cached_probe);
    ASSERT_NE(cached_probe, std::string::npos);
    ASSERT_NE(cached_seal, std::string::npos);
    ASSERT_NE(cached_publish, std::string::npos);
    EXPECT_LT(cached_seal, cached_publish);
    EXPECT_NE(rasterizer.find(
                  "pipeline_cache{instance, scheduler, liverpool, page_manager, buffer_cache}"),
              std::string::npos);
}

TEST(SrtMemorySyncSource, CpuWritesAndUnmapsInvalidateSnapshotsBeforeMutatingMappings) {
    const auto rasterizer = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto invalidate = rasterizer.find("bool Rasterizer::InvalidateMemory");
    const auto notify = rasterizer.find("page_manager.NotifyCpuWrite(addr, size)", invalidate);
    const auto buffer_invalidate =
        rasterizer.find("buffer_cache.InvalidateMemory(addr, size)", invalidate);
    ASSERT_NE(notify, std::string::npos);
    ASSERT_NE(buffer_invalidate, std::string::npos);
    EXPECT_LT(notify, buffer_invalidate);

    const auto unmap = rasterizer.find("void Rasterizer::UnmapMemory");
    const auto prepare = rasterizer.find("page_manager.PrepareForGpuUnmap(addr, size)", unmap);
    const auto unregister = rasterizer.find("page_manager.OnGpuUnmap(addr, size", unmap);
    ASSERT_NE(prepare, std::string::npos);
    ASSERT_NE(unregister, std::string::npos);
    EXPECT_LT(prepare, unregister);
}

TEST(SrtMemorySyncSource, VertexInputsNeverReadGuestDescriptorsAfterSnapshotSeal) {
    const auto pipeline = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto graphics = ReadText(SHADPS4_GRAPHICS_PIPELINE_SOURCE_PATH);

    EXPECT_NE(pipeline.find("CaptureVertexBuffers"), std::string::npos);
    EXPECT_NE(pipeline.find("captured_vertex_buffers"), std::string::npos);
    EXPECT_EQ(pipeline.find("attrib.GetSharp(*vs_info)"), std::string::npos);
    EXPECT_EQ(graphics.find("attrib.GetSharp(vs_info)"), std::string::npos);
    EXPECT_NE(graphics.find("captured_vertex_buffers"), std::string::npos);
    EXPECT_NE(pipeline.find("job.info_copies[stage] = *infos[stage]"), std::string::npos);
    EXPECT_NE(pipeline.find("!capture_vertex_buffers(program->info, spec.fetch_shader_data) || "
                            "!seal_snapshot()"),
              std::string::npos);
    EXPECT_NE(
        pipeline.find("!capture_vertex_buffers(new_info, authoritative_spec.fetch_shader_data) ||"),
        std::string::npos);
}

} // namespace
} // namespace VideoCore
