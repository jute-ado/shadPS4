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

TEST(SrtMemorySyncSource, PipelineCompilationAndRefreshUseBoundedPrivateSnapshotSessions) {
    const auto pipeline = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto rasterizer = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);

    EXPECT_NE(pipeline.find("SrtReadSession srt_read_session"), std::string::npos);
    EXPECT_NE(pipeline.find("MakeSrtMemoryReader(srt_read_session)"), std::string::npos);
    EXPECT_NE(pipeline.find("Shader::CompileResourceCapture compile_capture"),
              std::string::npos);
    EXPECT_NE(pipeline.find("Shader::TranslateProgram("), std::string::npos);
    EXPECT_NE(pipeline.find("info.RefreshFlatBuf(capture_reader)"),
              std::string::npos);
    EXPECT_EQ(pipeline.find("return {&SynchronizeSrtRange, this};"), std::string::npos);
    EXPECT_NE(rasterizer.find(
                  "pipeline_cache{instance, scheduler, liverpool, page_manager, buffer_cache}"),
              std::string::npos);
}

TEST(SrtMemorySyncSource, EveryShaderPermutationOwnsItsMatchingRefreshedResourceInfo) {
    const auto pipeline = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    EXPECT_NE(pipeline.find("const auto compile_resources = compile_capture.Finish(info, epoch_valid)"),
              std::string::npos);
    EXPECT_NE(pipeline.find("program->AddPermut(module, std::move(spec), std::move(new_info))"),
              std::string::npos);
    EXPECT_NE(pipeline.find("auto& selected_info = program->GetPermutInfo(perm_idx)"),
              std::string::npos);
    EXPECT_NE(pipeline.find("selected_info.UseCompileResourceSnapshot(compile_resources)"),
              std::string::npos);
    EXPECT_NE(pipeline.find("selected_info.RefreshFlatBuf()"),
              std::string::npos);
    EXPECT_NE(pipeline.find("return std::make_tuple(&selected_info, module"),
              std::string::npos);
    EXPECT_EQ(pipeline.find("return std::make_tuple(&program->info, module"),
              std::string::npos);
}

TEST(SrtMemorySyncSource, RefusedCaptureCannotPublishAProgramOrCreateAComputePipeline) {
    const auto pipeline = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto attempt_begin =
        pipeline.find("PipelineCache::Result PipelineCache::GetProgramAttempt(");
    const auto attempt_end = pipeline.find("PipelineCache::ReplaceShader", attempt_begin);
    ASSERT_NE(attempt_begin, std::string::npos);
    ASSERT_NE(attempt_end, std::string::npos);
    const auto attempt = pipeline.substr(attempt_begin, attempt_end - attempt_begin);

    const auto first_module = attempt.find("const auto module = CompileModule(");
    const auto first_refusal = attempt.find("if (!module)", first_module);
    const auto first_return = attempt.find("return {};", first_refusal);
    const auto first_metadata = attempt.find("RegisterShaderMeta(program->info", first_refusal);
    const auto first_permutation = attempt.find("program->AddPermut", first_refusal);
    ASSERT_NE(first_module, std::string::npos);
    ASSERT_NE(first_refusal, std::string::npos);
    ASSERT_NE(first_return, std::string::npos);
    ASSERT_NE(first_metadata, std::string::npos);
    ASSERT_NE(first_permutation, std::string::npos);
    EXPECT_LT(first_refusal, first_return);
    EXPECT_LT(first_return, first_metadata);
    EXPECT_LT(first_return, first_permutation);

    const auto repeated_capture = attempt.find("compile_capture.Finish(info, epoch_valid)");
    const auto repeated_refusal = attempt.find("if (!compile_resources.valid", repeated_capture);
    const auto repeated_return = attempt.find("return {};", repeated_refusal);
    const auto repeated_metadata = attempt.find("RegisterShaderMeta(new_info", repeated_refusal);
    const auto repeated_permutation = attempt.find("program->AddPermut", repeated_refusal);
    ASSERT_NE(repeated_capture, std::string::npos);
    ASSERT_NE(repeated_refusal, std::string::npos);
    ASSERT_NE(repeated_return, std::string::npos);
    ASSERT_NE(repeated_metadata, std::string::npos);
    ASSERT_NE(repeated_permutation, std::string::npos);
    EXPECT_LT(repeated_refusal, repeated_return);
    EXPECT_LT(repeated_return, repeated_metadata);
    EXPECT_LT(repeated_return, repeated_permutation);

    const auto compute_begin =
        pipeline.find("const ComputePipeline* PipelineCache::GetComputePipeline()");
    const auto compute_end =
        pipeline.find("bool PipelineCache::RefreshGraphicsKey()", compute_begin);
    ASSERT_NE(compute_begin, std::string::npos);
    ASSERT_NE(compute_end, std::string::npos);
    const auto compute = pipeline.substr(compute_begin, compute_end - compute_begin);
    const auto selection_gate = compute.find("if (!RefreshComputeKey())");
    const auto refused_pipeline = compute.find("return nullptr;", selection_gate);
    const auto pipeline_publication = compute.find("compute_pipelines.try_emplace", selection_gate);
    ASSERT_NE(selection_gate, std::string::npos);
    ASSERT_NE(refused_pipeline, std::string::npos);
    ASSERT_NE(pipeline_publication, std::string::npos);
    EXPECT_LT(selection_gate, refused_pipeline);
    EXPECT_LT(refused_pipeline, pipeline_publication);
}

TEST(SrtMemorySyncSource, FreshRetryReturnsAfterItsFirstAcceptedSelection) {
    const auto pipeline = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto get_program = pipeline.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto get_attempt =
        pipeline.find("PipelineCache::Result PipelineCache::GetProgramAttempt(", get_program);
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(get_attempt, std::string::npos);
    const auto body = pipeline.substr(get_program, get_attempt - get_program);

    const auto epoch = body.find("SrtReadSession srt_read_session");
    const auto selection = body.find(
        "GetProgramAttempt(stage, l_stage, params, binding, srt_read_session, attempt_pending)");
    const auto accepted = body.find(
        "std::get<0>(result) != nullptr && static_cast<bool>(std::get<1>(result))", selection);
    const auto accepted_return = body.find("return result;", accepted);
    const auto invalid_epoch = body.find("if (!srt_read_session.Valid())", accepted_return);
    const auto exhausted_return = body.find("return {};", accepted_return);
    ASSERT_NE(epoch, std::string::npos);
    ASSERT_NE(selection, std::string::npos);
    ASSERT_NE(accepted, std::string::npos);
    ASSERT_NE(accepted_return, std::string::npos);
    ASSERT_NE(invalid_epoch, std::string::npos);
    ASSERT_NE(exhausted_return, std::string::npos);
    EXPECT_LT(epoch, selection);
    EXPECT_LT(selection, accepted);
    EXPECT_LT(accepted, accepted_return);
    EXPECT_LT(accepted_return, invalid_epoch);
    EXPECT_LT(invalid_epoch, exhausted_return);
    EXPECT_LT(accepted_return, exhausted_return);
}

TEST(SrtMemorySyncSource, CaptureProvenanceIsGatedAndRecordsAcceptedOrRefusedPrograms) {
    const auto pipeline = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto get_program = pipeline.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto get_attempt =
        pipeline.find("PipelineCache::Result PipelineCache::GetProgramAttempt(", get_program);
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(get_attempt, std::string::npos);
    const auto body = pipeline.substr(get_program, get_attempt - get_program);

    const auto debug_gate =
        body.find("const bool collect_debug_provenance = EmulatorSettings.IsShaderCollect()");
    const auto selection_gate =
        body.find("const bool collect_selection_provenance = shader_selection_trace.Enabled()");
    const auto session = body.find(
        "collect_debug_provenance || collect_selection_provenance");
    const auto publish = body.find("DebugState.CollectShaderCaptureProvenance(");
    const auto observable = body.find("LOG_INFO(Render_Vulkan, \"{}\", *payload)", publish);
    const auto success = body.find("record_provenance(true)", observable);
    const auto accepted_return = body.find("return result;", success);
    const auto refusal = body.find("record_provenance(false)", accepted_return);
    const auto refused_return = body.find("return {};", refusal);

    ASSERT_NE(debug_gate, std::string::npos);
    ASSERT_NE(selection_gate, std::string::npos);
    ASSERT_NE(session, std::string::npos);
    ASSERT_NE(publish, std::string::npos);
    ASSERT_NE(observable, std::string::npos);
    ASSERT_NE(success, std::string::npos);
    ASSERT_NE(accepted_return, std::string::npos);
    ASSERT_NE(refusal, std::string::npos);
    ASSERT_NE(refused_return, std::string::npos);
    EXPECT_LT(debug_gate, selection_gate);
    EXPECT_LT(selection_gate, session);
    EXPECT_LT(session, publish);
    EXPECT_LT(publish, observable);
    EXPECT_LT(observable, success);
    EXPECT_LT(success, accepted_return);
    EXPECT_LT(accepted_return, refusal);
    EXPECT_LT(refusal, refused_return);
    EXPECT_EQ(body.find("params.hash =="), std::string::npos);
    EXPECT_EQ(body.find("GetGameSerial"), std::string::npos);
}

TEST(SrtMemorySyncSource, NewProgramSealsCaptureBeforeSpirvAndShaderModulePublication) {
    const auto pipeline = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto attempt_begin =
        pipeline.find("PipelineCache::Result PipelineCache::GetProgramAttempt(");
    const auto first_program_value =
        pipeline.find("auto& program = it_pgm.value();", attempt_begin);
    const auto cached_begin =
        pipeline.find("auto& program = it_pgm.value();", first_program_value + 1);
    ASSERT_NE(attempt_begin, std::string::npos);
    ASSERT_NE(cached_begin, std::string::npos);
    const auto first_program = pipeline.substr(attempt_begin, cached_begin - attempt_begin);

    const auto translate = first_program.find("TranslateModule(");
    const auto specialize = first_program.find("Shader::StageSpecialization(", translate);
    const auto seal = first_program.find("srt_read_session.Seal()", specialize);
    const auto finish = first_program.find("compile_capture.Finish(program->info", seal);
    const auto compile = first_program.find("CompileModule(program->info", finish);
    const auto publish = first_program.find("RegisterShaderMeta(program->info", compile);

    ASSERT_NE(translate, std::string::npos);
    ASSERT_NE(specialize, std::string::npos);
    ASSERT_NE(seal, std::string::npos);
    ASSERT_NE(finish, std::string::npos);
    ASSERT_NE(compile, std::string::npos);
    ASSERT_NE(publish, std::string::npos);
    EXPECT_LT(translate, specialize);
    EXPECT_LT(specialize, seal);
    EXPECT_LT(seal, finish);
    EXPECT_LT(finish, compile);
    EXPECT_LT(compile, publish);
}

TEST(SrtMemorySyncSource, CpuFaultInvalidatesSnapshotBeforeCacheInvalidation) {
    const auto rasterizer = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto begin = rasterizer.find("bool Rasterizer::InvalidateMemory(");
    const auto end = rasterizer.find("bool Rasterizer::ReadMemory(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = rasterizer.substr(begin, end - begin);

    const auto epoch = body.find("page_manager.NotifyCpuWrite(addr, size)");
    const auto buffer = body.find("buffer_cache.InvalidateMemory(addr, size)");
    const auto texture = body.find("texture_cache.InvalidateMemory(addr, size)");
    ASSERT_NE(epoch, std::string::npos);
    ASSERT_NE(buffer, std::string::npos);
    ASSERT_NE(texture, std::string::npos);
    EXPECT_LT(epoch, buffer);
    EXPECT_LT(epoch, texture);
}

TEST(SrtMemorySyncSource, GuestBackingWriteBridgeOnlyInvalidatesOverlappingSnapshots) {
    const auto rasterizer = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto begin = rasterizer.find("void Rasterizer::NotifyGuestBackingWrite(");
    const auto end = rasterizer.find("bool Rasterizer::InvalidateMemory(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = rasterizer.substr(begin, end - begin);

    EXPECT_NE(body.find("page_manager.NotifyCpuWrite(addr, size)"), std::string::npos);
    EXPECT_EQ(body.find("buffer_cache."), std::string::npos);
    EXPECT_EQ(body.find("texture_cache."), std::string::npos);
}

} // namespace
} // namespace VideoCore
