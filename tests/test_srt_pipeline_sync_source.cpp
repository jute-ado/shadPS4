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

    EXPECT_NE(pipeline.find("SrtReadSession srt_read_session{"),
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
    for (const auto* capture :
         {"!capture_vertex_buffers(program->info, spec.fetch_shader_data)",
          "!capture_vertex_buffers(new_info, authoritative_spec.fetch_shader_data)"}) {
        const auto capture_at = pipeline.find(capture);
        const auto seal_at = pipeline.find("if (!seal_snapshot())", capture_at);
        const auto compile_at = pipeline.find("const auto module = CompileModule(", capture_at);
        ASSERT_NE(capture_at, std::string::npos);
        ASSERT_NE(seal_at, std::string::npos);
        ASSERT_NE(compile_at, std::string::npos);
        EXPECT_LT(capture_at, seal_at);
        EXPECT_LT(seal_at, compile_at);
    }
}

TEST(SrtFailureDiagnostics, WarningSeparatesPhaseReaderAndTranslationWithoutAddresses) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin = source.find("const auto fail_snapshot =");
    const auto end = source.find("const auto seal_snapshot =", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(body.find("phase={} read={} translationRejected={}"), std::string::npos);
    EXPECT_NE(body.find("srt_read_session.LastFailure()"), std::string::npos);
    EXPECT_NE(body.find("SrtMemoryReadFailureName"), std::string::npos);
    const auto warning_begin = body.find("LOG_WARNING(");
    ASSERT_NE(warning_begin, std::string::npos);
    const auto warning_end = body.find(';', warning_begin);
    ASSERT_NE(warning_end, std::string::npos);
    const auto default_warning = body.substr(warning_begin, warning_end - warning_begin);
    EXPECT_EQ(default_warning.find("address"), std::string::npos);
    EXPECT_EQ(body.find("ReadMemory"), std::string::npos);
}

TEST(SrtFailureDiagnostics, ActualSelectionPathsHaveDistinctFailureLabels) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto end = source.find("PipelineCache::ReplaceShader(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_EQ(body.find("return fail_snapshot();"), std::string::npos);
    for (const auto* phase :
         {"initial_translation", "initial_vertex_capture", "initial_seal", "aggregate_refresh",
          "probe_specialization", "selected_refresh", "selected_specialization",
          "selected_vertex_capture", "cached_seal", "permutation_translation",
          "permutation_identity", "reuse_seal", "permutation_vertex_capture", "permutation_seal"}) {
        EXPECT_NE(body.find(phase), std::string::npos) << phase;
    }
    EXPECT_NE(body.find("program->info.translation_failed"), std::string::npos);
    EXPECT_NE(body.find("new_info.translation_failed"), std::string::npos);
}

TEST(SrtFailureDiagnostics, OptionalFirstRequestLogIsBoundedAndNeverReadsGuestData) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto session = source.find("SrtReadSession srt_read_session{");
    ASSERT_NE(session, std::string::npos);
    const auto construct = source.substr(session, source.find(';', session) - session);
    EXPECT_NE(construct.find("page_manager"), std::string::npos);
    EXPECT_NE(construct.find("buffer_cache"), std::string::npos);
    EXPECT_NE(construct.find("GetSrtReadFailureDiagnostics().Enabled()"), std::string::npos);
    const auto begin = source.find("const auto fail_snapshot =");
    const auto end = source.find("const auto seal_snapshot =", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(body.find("GetSrtReadFailureDiagnostics().Emit("), std::string::npos);
    EXPECT_NE(body.find("srt_read_session.FirstFailureRequest()"), std::string::npos);
    EXPECT_NE(body.find("SrtReadFailureV1"), std::string::npos);
    EXPECT_NE(body.find("request_known={}"), std::string::npos);
    EXPECT_NE(body.find("output_null={} invalid_reasons={}"), std::string::npos);
    EXPECT_EQ(body.find("ReadMemory"), std::string::npos);
    EXPECT_EQ(body.find("steady_clock"), std::string::npos);
    EXPECT_EQ(body.find("FailureFlag()"), std::string::npos);
}

TEST(SrtMemorySyncSource, AllSpecializationAdmissionsRefuseUnsupportedConsumedConversions) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto end = source.find("PipelineCache::ReplaceShader(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_EQ(body.find("Shader::StageSpecialization("), std::string::npos);
    std::size_t admissions{};
    for (std::size_t at{};
         (at = body.find("Shader::TryCreateStageSpecialization(", at)) != std::string::npos; ++at) {
        ++admissions;
    }
    EXPECT_EQ(admissions, 4U);
    for (const auto* phase : {"initial_specialization", "permutation_specialization"}) {
        EXPECT_NE(body.find(phase), std::string::npos) << phase;
    }
    // An invalid aggregate is only a cache miss. Current IR must be allowed to prove a different
    // format source; never return solely because a stale probe could not be specialized.
    EXPECT_NE(body.find("probe_spec ? std::ranges::find"), std::string::npos);
    EXPECT_EQ(body.find("if (!probe_spec)"), std::string::npos);
    const auto selected_guard = body.find("if (!selected_spec || !reader.Valid())");
    const auto selected_identity = body.find("const auto selected_id =");
    ASSERT_NE(selected_guard, std::string::npos);
    ASSERT_NE(selected_identity, std::string::npos);
    EXPECT_LT(selected_guard, selected_identity);
}

} // namespace
} // namespace VideoCore
