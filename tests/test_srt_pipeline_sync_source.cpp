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

std::string FunctionBody(const std::string& source, std::string_view signature) {
    const auto begin = source.find(signature);
    if (begin == std::string::npos) {
        return {};
    }
    const auto open = source.find('{', begin);
    if (open == std::string::npos) {
        return {};
    }
    int depth{};
    for (std::size_t index = open; index < source.size(); ++index) {
        if (source[index] == '{') {
            ++depth;
        } else if (source[index] == '}' && --depth == 0) {
            return source.substr(begin, index - begin + 1);
        }
    }
    return {};
}

TEST(SrtMemorySyncSource, PipelineSelectionUsesOneCoherentReaderAndSealsBeforePublication) {
    const auto pipeline = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto rasterizer = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);

    EXPECT_NE(pipeline.find("SrtReadSession srt_read_session{"), std::string::npos);
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
          "initial_spirv", "probe_specialization", "selected_refresh", "selected_specialization",
          "selected_vertex_capture", "cached_seal", "permutation_translation", "permutation_spirv",
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

TEST(SrtFailureDiagnostics, AggregateFailureCanCaptureOwnedFlatPrefixWithoutAnotherGuestRead) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin = source.find("const auto fail_snapshot =");
    const auto end = source.find("const auto seal_snapshot =", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(body.find("const Shader::Info* failed_info"), std::string::npos);
    const auto capture = body.find("CaptureSrtFailureFlatSnapshot(");
    ASSERT_NE(capture, std::string::npos);
    EXPECT_NE(body.substr(capture, 120).find("failed_info->flattened_ud_buf"), std::string::npos);
    EXPECT_NE(body.find("captured_flat=["), std::string::npos);
    EXPECT_EQ(body.find("ReadGuestMemory"), std::string::npos);
    EXPECT_EQ(body.find("ReadMemory"), std::string::npos);

    const auto aggregate = source.find("fail_snapshot(\"aggregate_refresh\"");
    ASSERT_NE(aggregate, std::string::npos);
    EXPECT_NE(source.substr(aggregate, 160).find("&aggregate_snapshot"), std::string::npos);
}

TEST(SrtFailureDiagnostics, ExactFlatSelectorCannotBeStarvedByBroadFailureBudget) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin = source.find("const auto fail_snapshot =");
    const auto end = source.find("const auto seal_snapshot =", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(body.find("GetSrtFailureFlatDiagnostics().EmitFailure("), std::string::npos);
    EXPECT_NE(body.find("SrtFailureFlatTargetV1"), std::string::npos);
    EXPECT_NE(body.find("srt_read_session.FirstFailureRequest()"), std::string::npos);
    EXPECT_NE(body.find("CaptureSrtFailureFlatSnapshot("), std::string::npos);
    EXPECT_EQ(body.find("ReadGuestMemory"), std::string::npos);
    EXPECT_EQ(body.find("ReadMemory"), std::string::npos);
}

TEST(SrtFailureDiagnostics, InitialTranslationPassesFailedShaderInfoToExactCapture) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto call = source.find("fail_snapshot(\"initial_translation\"");
    ASSERT_NE(call, std::string::npos);
    const auto end = source.find(");", call);
    ASSERT_NE(end, std::string::npos);
    const auto invocation = source.substr(call, end - call);
    EXPECT_NE(invocation.find("&program->info"), std::string::npos);
}

TEST(SrtFailureDiagnostics, ExactSelectorRetainsRequestMetadataWithoutBroadLogging) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto session = source.find("SrtReadSession srt_read_session{");
    ASSERT_NE(session, std::string::npos);
    const auto construct = source.substr(session, source.find(';', session) - session);
    EXPECT_NE(construct.find("GetSrtReadFailureDiagnostics().Enabled()"), std::string::npos);
    EXPECT_NE(construct.find("capture_exact_flat_target"), std::string::npos);
    EXPECT_NE(construct.find("||"), std::string::npos);
    const auto exact_match =
        source.rfind("GetSrtFailureFlatDiagnostics().Matches(params.hash)", session);
    ASSERT_NE(exact_match, std::string::npos);
    EXPECT_LT(exact_match, session);
}

TEST(SrtFailureDiagnostics, ExactSelectorArmsOnlyCachedAggregateRootCompositionReceipt) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto get_program = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto initial = source.find("if (new_program)", get_program);
    const auto cached = source.find("auto aggregate_snapshot = program->info", initial);
    const auto refresh = source.find("aggregate_snapshot.RefreshFlatBuf(reader)", cached);
    const auto arm = source.rfind("srt_read_session.ArmCompositionReceipt()", refresh);
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(initial, std::string::npos);
    ASSERT_NE(cached, std::string::npos);
    ASSERT_NE(refresh, std::string::npos);
    ASSERT_NE(arm, std::string::npos);
    EXPECT_GT(arm, cached);
    EXPECT_LT(arm, refresh);
    EXPECT_EQ(source.substr(initial, cached - initial).find("ArmCompositionReceipt"),
              std::string::npos);
    EXPECT_NE(source.substr(cached, refresh - cached).find("if (capture_exact_flat_target)"),
              std::string::npos);
}

TEST(SrtFailureDiagnostics, ExactFlatAndCompositionReceiptsShareOneTargetRowWithoutGuestRead) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin = source.find("GetSrtFailureFlatDiagnostics().EmitFailure(");
    const auto end = source.find("});", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(body.find("SrtFailureFlatTargetV1"), std::string::npos);
    EXPECT_NE(body.find("SrtRootCompositionTargetV1"), std::string::npos);
    EXPECT_NE(body.find("srt_read_session.CompositionReceipt()"), std::string::npos);
    EXPECT_EQ(body.find("ReadMemory"), std::string::npos);
    EXPECT_EQ(body.find("ReadGuestMemory"), std::string::npos);
    EXPECT_EQ(body.find("GetSrtFailureFlatDiagnostics().Emit("), std::string::npos);
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
         (at = body.find("Shader::TryCreateStageSpecializationDetailed(", at)) !=
         std::string::npos;
         ++at) {
        ++admissions;
    }
    EXPECT_EQ(admissions, 4U);
    EXPECT_NE(body.find("StageSpecializationFailureReasonName"), std::string::npos);
    EXPECT_NE(body.find("specializationFailure={}"), std::string::npos);
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

TEST(SrtFailureDiagnostics, FailedSpirvTranslationCannotReachModuleCompilation) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto initial = source.find("if (new_program)");
    const auto permutation = source.find("const size_t perm_idx =", initial);
    ASSERT_NE(initial, std::string::npos);
    ASSERT_NE(permutation, std::string::npos);

    const auto initial_emit = source.find("Shader::Backend::SPIRV::EmitSPIRV(", initial);
    const auto initial_guard = source.find(
        "if (program->info.translation_failed || !reader.Valid())", initial_emit);
    const auto initial_compile = source.find("const auto module = CompileModule(", initial_emit);
    ASSERT_NE(initial_emit, std::string::npos);
    ASSERT_NE(initial_guard, std::string::npos);
    ASSERT_NE(initial_compile, std::string::npos);
    EXPECT_LT(initial_emit, initial_guard);
    EXPECT_LT(initial_guard, initial_compile);
    const auto initial_failure = source.find("\"initial_spirv\"", initial_guard);
    const auto initial_erase = source.find("program_cache.erase(cache_key)", initial_failure);
    ASSERT_NE(initial_failure, std::string::npos);
    ASSERT_NE(initial_erase, std::string::npos);
    EXPECT_LT(initial_failure, initial_erase);
    EXPECT_LT(initial_erase, initial_compile);

    const auto permutation_emit =
        source.find("Shader::Backend::SPIRV::EmitSPIRV(", permutation);
    const auto permutation_guard = source.find(
        "if (new_info.translation_failed || !reader.Valid())", permutation_emit);
    const auto permutation_compile = source.find("const auto module = CompileModule(", permutation_emit);
    ASSERT_NE(permutation_emit, std::string::npos);
    ASSERT_NE(permutation_guard, std::string::npos);
    ASSERT_NE(permutation_compile, std::string::npos);
    EXPECT_LT(permutation_emit, permutation_guard);
    EXPECT_LT(permutation_guard, permutation_compile);
    EXPECT_NE(source.substr(permutation_guard, permutation_compile - permutation_guard)
                  .find("fail_snapshot(\"permutation_spirv\""),
              std::string::npos);
}

TEST(SrtFailureDiagnostics, GuestModuleCreationUsesFailClosedVulkanResultHandling) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto module_begin = source.find("vk::ShaderModule PipelineCache::CompileModule(");
    const auto module_end = source.find("PipelineCache::Result PipelineCache::GetProgram(",
                                       module_begin);
    ASSERT_NE(module_begin, std::string::npos);
    ASSERT_NE(module_end, std::string::npos);
    const auto module = source.substr(module_begin, module_end - module_begin);

    // Guest-generated and patched SPIR-V is not an internal invariant: a backend or a user patch
    // can produce a structurally invalid module, and vkCreateShaderModule returns a failure
    // result for that input. This path must not call the assert-on-failure CompileSPV helper.
    EXPECT_NE(module.find("TryCompileSPV("), std::string::npos);
    EXPECT_EQ(module.find(" CompileSPV("), std::string::npos);
    EXPECT_NE(module.find("if (!compiled_module)"), std::string::npos);
    EXPECT_NE(module.find("CompilationOutcome::Rejected"), std::string::npos);
}

TEST(SrtFailureDiagnostics, GuestModuleFailureCannotPublishCachePermutation) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto get_program = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto replace_shader = source.find("std::optional<vk::ShaderModule> PipelineCache::ReplaceShader",
                                           get_program);
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(replace_shader, std::string::npos);
    const auto body = source.substr(get_program, replace_shader - get_program);

    // Both initial and permutation compilation must reject the null handle before metadata,
    // program-cache, or binding publication. The labels also make the failure observable in the
    // existing bounded SRT diagnostic stream.
    std::size_t compile_count{};
    for (std::size_t at{};
         (at = body.find("const auto module = CompileModule(", at)) != std::string::npos;
         at += 1) {
        ++compile_count;
    }
    ASSERT_EQ(compile_count, 2U);

    std::size_t search{};
    for (const auto* label : {"initial_module", "permutation_module"}) {
        const auto compile = body.find("const auto module = CompileModule(", search);
        ASSERT_NE(compile, std::string::npos);
        const auto failure = body.find(std::string{"fail_snapshot(\""} + label + "\"", compile);
        const auto metadata = body.find("RegisterShaderMeta(", compile);
        const auto publish = body.find("AddPermut(", compile);
        ASSERT_NE(failure, std::string::npos) << label;
        ASSERT_NE(metadata, std::string::npos) << label;
        ASSERT_NE(publish, std::string::npos) << label;
        EXPECT_LT(failure, metadata) << label;
        EXPECT_LT(failure, publish) << label;
        const auto guard = body.rfind("if (!module)", failure);
        ASSERT_NE(guard, std::string::npos) << label;
        EXPECT_LT(compile, guard) << label;
        search = metadata;
    }
}

TEST(SrtFailureDiagnostics, EveryImageCoordinateConsumerChecksViewArity) {
    const auto source = ReadText(SHADPS4_SPIRV_IMAGE_SOURCE_PATH);
    const auto sampled_guard = FunctionBody(source, "bool ValidateImageSampleOperation(");
    ASSERT_FALSE(sampled_guard.empty());
    // Sampled operations add a multisample-operation check before delegating to the same
    // coordinate validator. Require that delegation instead of demanding an inlined call.
    EXPECT_NE(sampled_guard.find("IsImageSampleOperationAllowed(texture.view_type)"),
              std::string::npos);
    EXPECT_NE(sampled_guard.find(
                  "return ValidateImageCoordinates(ctx, inst, texture, coordinate_arg, false);"),
              std::string::npos);
    struct Consumer {
        const char* signature;
        const char* validator;
        const char* coordinate_argument;
    };
    for (const auto& consumer : {
             Consumer{"Id EmitImageSampleImplicitLod(", "ValidateImageSampleOperation", "1U"},
             Consumer{"Id EmitImageSampleExplicitLod(", "ValidateImageSampleOperation", "1U"},
             Consumer{"Id EmitImageSampleExplicitLodDynamic(", "ValidateImageSampleOperation",
                      "2U"},
             Consumer{"Id EmitImageSampleDrefImplicitLod(", "ValidateImageSampleOperation", "1U"},
             Consumer{"Id EmitImageSampleDrefExplicitLod(", "ValidateImageSampleOperation", "1U"},
             Consumer{"Id EmitImageGather(", "ValidateImageSampleOperation", "1U"},
             Consumer{"Id EmitImageGatherDref(", "ValidateImageSampleOperation", "1U"},
             Consumer{"Id EmitImageQueryLod(", "ValidateImageSampleOperation", "1U"},
             Consumer{"Id EmitImageGradient(", "ValidateImageSampleOperation", "1U"},
             Consumer{"Id EmitImageRead(", "ValidateImageCoordinates", "1U, true"},
             Consumer{"void EmitImageWrite(", "ValidateImageCoordinates", "1U, true"},
         }) {
        const auto body = FunctionBody(source, consumer.signature);
        ASSERT_FALSE(body.empty()) << consumer.signature;
        const auto guard =
            body.find(std::string{"if (!"} + consumer.validator + "(ctx, *inst, texture, " +
                      consumer.coordinate_argument + "))");
        ASSERT_NE(guard, std::string::npos) << consumer.signature;
        const auto first_emission = body.find("ctx.Op");
        ASSERT_NE(first_emission, std::string::npos) << consumer.signature;
        EXPECT_LT(guard, first_emission) << consumer.signature;
        EXPECT_NE(body.substr(guard, first_emission - guard).find("return"), std::string::npos)
            << consumer.signature;
    }
}

TEST(SrtFailureDiagnostics, EveryImageAtomicConsumerChecksIntegerCoordinates) {
    const auto source = ReadText(SHADPS4_SPIRV_ATOMIC_SOURCE_PATH);
    const auto validator = FunctionBody(source, "bool ValidateImageAtomicCoordinates(");
    ASSERT_FALSE(validator.empty());
    EXPECT_NE(validator.find("const IR::Type coordinate_type = inst.Arg(1).Type();"),
              std::string::npos);
    EXPECT_NE(
        validator.find("IsImageCoordinateTypeValid(texture.view_type, coordinate_type, true)"),
        std::string::npos);

    for (const auto* signature : {"Id ImageAtomicU32(", "Id ImageAtomicF32(",
                                  "Id ImageAtomicU32CmpSwap("}) {
        const auto body = FunctionBody(source, signature);
        ASSERT_FALSE(body.empty()) << signature;
        const auto guard =
            body.find("if (!ValidateImageAtomicCoordinates(ctx, *inst, texture))");
        ASSERT_NE(guard, std::string::npos) << signature;
        const auto emission = body.find("ctx.OpImageTexelPointer");
        ASSERT_NE(emission, std::string::npos) << signature;
        EXPECT_LT(guard, emission) << signature;
        EXPECT_NE(body.substr(guard, emission - guard).find("return"), std::string::npos)
            << signature;
    }
}

} // namespace
} // namespace VideoCore
