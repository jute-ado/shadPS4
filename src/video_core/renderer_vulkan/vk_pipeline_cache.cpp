// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <ranges>
#include <string_view>

#include "common/hash.h"
#include "common/io_file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/scope_exit.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/specialization_identity.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/srt_memory_sync.h"
#include "video_core/buffer_cache/srt_preparation_retry.h"
#include "video_core/cache_storage.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/async_compilation_completion.h"
#include "video_core/renderer_vulkan/compilation_latency_timer.h"
#include "video_core/renderer_vulkan/host_compilation_pause.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

using SrtReadSession =
    VideoCore::SrtMemoryReadSession<VideoCore::PageManager, VideoCore::BufferCache>;

static Shader::SrtMemoryReader MakeSrtMemoryReader(SrtReadSession& session) {
    return {&SrtReadSession::ReadCallback, &session, session.FailureFlag()};
}

using Shader::LogicalStage;
using Shader::Output;
using Shader::Stage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;
constexpr static u32 DescriptorHeapSampledImages = 8192U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, DescriptorHeapSampledImages},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(Stage stage, LogicalStage l_stage,
                                                           Shader::SrtMemoryReader reader) {
    auto& info = runtime_infos[u32(l_stage)];
    const auto& regs = liverpool->regs;
    const auto BuildCommon = [&](const auto& program) {
        info.num_user_data = program.settings.num_user_regs;
        info.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.num_allocated_vgprs = program.NumVgprs();
        info.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.fp_round_mode32 = program.settings.fp_round_mode32;
        info.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    info.Initialize(stage);
    switch (stage) {
    case Stage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        const auto* hull_info = infos[u32(Shader::LogicalStage::TessellationControl)];
        hull_info->ReadTessConstantBuffer(tess_constants, reader);
        info.ls_info.ls_stride = tess_constants.ls_stride;
        break;
    }
    case Stage::Hull: {
        BuildCommon(regs.hs_program);
        info.hs_info.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.hs_info.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.hs_info.tess_type = regs.tess_config.type;
        info.hs_info.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;

        // We need to initialize most hs_info fields after finding the V# with tess constants
        break;
    }
    case Stage::Export: {
        BuildCommon(regs.es_program);
        info.es_info.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Vertex: {
        BuildCommon(regs.vs_program);
        info.vs_info.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        info.vs_info.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.vs_info.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.vs_info.num_outputs = MapOutputs(info.vs_info.outputs, regs.vs_output_control);
        info.vs_info.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.vs_info.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
        info.vs_info.clip_disable = regs.IsClipDisabled();
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Geometry: {
        BuildCommon(regs.gs_program);
        auto& gs_info = info.gs_info;
        gs_info.num_outputs = MapOutputs(gs_info.outputs, regs.vs_output_control);
        gs_info.output_vertices = regs.vgt_gs_max_vert_out;
        gs_info.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            gs_info.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            gs_info.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            gs_info.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        gs_info.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        gs_info.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        gs_info.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        gs_info.vs_copy = params_vc.code;
        gs_info.vs_copy_hash = params_vc.hash;
        DumpShader(gs_info.vs_copy, gs_info.vs_copy_hash, Shader::Stage::Vertex, 0, "copy.bin");
        break;
    }
    case Stage::Fragment: {
        BuildCommon(regs.ps_program);
        info.fs_info.en_flags = regs.ps_input_ena;
        info.fs_info.addr_flags = regs.ps_input_addr;
        info.fs_info.num_inputs = regs.num_interp;
        info.fs_info.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.fs_info.mrtz_mask = regs.depth_shader_control.z_export_enable |
                                 (stencil_ref_export_enable << 1) |
                                 (regs.depth_shader_control.mask_export_enable << 2) |
                                 (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.fs_info.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.fs_info.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.fs_info.dual_source_blending = false;
        }
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.fs_info.inputs[i] = {
                .param_index = u8(ps_inputs[i].input_offset),
                .is_default = bool(ps_inputs[i].use_default),
                .is_flat = bool(ps_inputs[i].flat_shade),
                .default_value = u8(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.fs_info.color_buffers[i] = graphics_key.color_buffers[i];
        }
        // Lowered user clip planes ride the same emulation path as guest-exported distances, so
        // the fragment side arms whenever the hardware vertex stage lowers them, keeping its input
        // locations in sync with the shifted vertex outputs.
        const bool lowers_user_clip_planes =
            regs.clipper_control.user_clip_plane_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Geometry));
        info.fs_info.clip_distance_emulation =
            ((regs.vs_output_control.clip_distance_enable &&
              !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Local))) ||
             lowers_user_clip_planes) &&
            profile.needs_clip_distance_emulation;
        break;
    }
    case Stage::Compute: {
        const auto& cs_pgm = liverpool->GetCsRegs();
        info.num_user_data = cs_pgm.settings.num_user_regs;
        info.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.cs_info.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                       cs_pgm.num_thread_z.full};
        info.cs_info.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                    cs_pgm.IsTgidEnabled(2)};
        info.cs_info.shared_memory_size = cs_pgm.SharedMemSize();
        break;
    }
    default:
        break;
    }
    return info;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_, VideoCore::PageManager& page_manager_,
                             VideoCore::BufferCache& buffer_cache_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      page_manager{page_manager_}, buffer_cache{buffer_cache_},
      desc_heap{instance, scheduler.GetMasterSemaphore(), DescriptorHeapSizes} {
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        // When binding a UBO, we calculate its size considering the offset in the larger buffer
        // cache underlying resource. In some cases, it may produce sizes exceeding the system
        // maximum allowed UBO range, so we need to reduce the threshold to prevent issues.
        .max_ubo_size = instance.UniformMaxSize() - instance.UniformMinAlignment(),
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .max_sampled_images_per_stage = instance.MaxPerStageDescriptorSampledImages(),
        .max_sampled_images_per_set = instance.MaxDescriptorSetSampledImages(),
        .max_resources_per_stage = instance.MaxPerStageResources(),
        .max_sampled_images_per_descriptor_heap_set =
            DescriptorHeapSampledImages / DescriptorHeap::DescriptorSetBatch,
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_sampled_image_array_nonuniform_indexing =
            instance_.IsSampledImageArrayNonUniformIndexingSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
    };
    WarmUp();

    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);

    if (EmulatorSettings.IsAsyncGraphicsPipelineCompilation()) {
        async_graphics_compiler =
            std::make_unique<AsyncGraphicsPipelineQueue>([this](AsyncGraphicsPipelineJob job) {
                return CompileGraphicsPipelineAsync(std::move(job));
            });
    }
}

PipelineCache::~PipelineCache() {
    async_graphics_compiler.reset();
}

void AsyncGraphicsPipelineJob::RebindCompileInfo() noexcept {
    for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
        if (!retained_infos[stage]) {
            compile_infos[stage] = nullptr;
            continue;
        }
        info_copies[stage].user_data =
            std::span{user_data_copies[stage]}.first(user_data_sizes[stage]);
        compile_infos[stage] = &info_copies[stage];
    }
}

AsyncGraphicsPipelineJob PipelineCache::MakeAsyncGraphicsPipelineJob() const {
    AsyncGraphicsPipelineJob job{
        .key = graphics_key,
        .retained_infos = infos,
        .runtime_infos = runtime_infos,
        .fetch_shader = fetch_shader,
        .modules = modules,
    };
    for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
        if (!infos[stage]) {
            continue;
        }
        job.info_copies[stage] = *infos[stage];
        job.user_data_sizes[stage] = static_cast<u32>(infos[stage]->user_data.size());
        ASSERT(job.user_data_sizes[stage] <= Shader::NUM_USER_DATA_REGS);
        std::ranges::copy(infos[stage]->user_data, job.user_data_copies[stage].begin());
    }
    return job;
}

static bool IsCompilationTimingEnabled() {
    const auto logger = Common::Log::ALL_LOGGERS.find(Common::Log::Class::Render_Compilation);
    return logger != Common::Log::ALL_LOGGERS.end() && logger->second &&
           logger->second->should_log(spdlog::level::debug);
}

AsyncGraphicsPipelineResult PipelineCache::CompileGraphicsPipelineAsync(
    AsyncGraphicsPipelineJob job) {
    job.RebindCompileInfo();
    GraphicsPipeline::SerializationSupport serialization{};
    CompilationLatencyTimer<> timing{IsCompilationTimingEnabled()};
    const auto pipeline_hash = timing.IsEnabled() ? std::hash<GraphicsPipelineKey>{}(job.key) : 0;
    if (timing.IsEnabled()) {
        LOG_DEBUG(Render_Compilation, "async_pipeline_worker_start key={:#x}", pipeline_hash);
    }
    const char* outcome = "exception";
    SCOPE_EXIT {
        if (timing.IsEnabled()) {
            LOG_DEBUG(Render_Compilation,
                      "async_pipeline_worker_finish key={:#x} elapsedMs={:.3f} outcome={}",
                      pipeline_hash, timing.ElapsedMilliseconds(), outcome);
        }
    };
    auto pipeline = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, job.key, vk::PipelineCache{}, job.compile_infos,
        job.runtime_infos, job.fetch_shader, job.modules, serialization, false);
    RequireValidCompiledPipeline(pipeline);
    outcome = "complete";
    return {
        .pipeline = std::move(pipeline),
        .serialization = std::move(serialization),
        .modules = job.modules,
        .retained_infos = job.retained_infos,
    };
}

void PipelineCache::PublishAsyncGraphicsPipeline(
    AsyncGraphicsPipelineQueue::Completion completion) {
    if (!completion.result || !completion.result->pipeline ||
        !completion.result->pipeline->IsValid()) {
        LOG_ERROR(Render_Vulkan, "Asynchronous graphics pipeline compilation failed");
        return;
    }

    auto result = std::move(*completion.result);
    const auto [it, is_new] = graphics_pipelines.try_emplace(completion.key);
    if (!is_new) {
        return;
    }
    result.pipeline->RebindStages(result.retained_infos);
    it.value() = std::move(result.pipeline);
    const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(completion.key);
    RegisterPipelineData(completion.key, pipeline_hash, result.serialization);
    ++num_new_pipelines;

    if (EmulatorSettings.IsShaderCollect()) {
        for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
            if (completion.key.stage_hashes[stage]) {
                module_related_pipelines[result.modules[stage]].emplace_back(completion.key);
            }
        }
    }
    LOG_INFO(Render_Vulkan, "Published asynchronous graphics pipeline {:#x}", pipeline_hash);
}

bool PipelineCache::PublishAsyncGraphicsPipelines(const GraphicsPipelineKey* requested_key) {
    if (!async_graphics_compiler) {
        return false;
    }
    return DrainAsyncCompilationResults(
        *async_graphics_compiler, requested_key, [this](auto completion) {
            PublishAsyncGraphicsPipeline(std::move(completion));
        });
}

const GraphicsPipeline* PipelineCache::WaitForAsyncGraphicsPipeline(
    const GraphicsPipelineKey& key) {
    CompilationLatencyTimer<> timing{IsCompilationTimingEnabled()};
    const auto pipeline_hash = timing.IsEnabled() ? std::hash<GraphicsPipelineKey>{}(key) : 0;
    if (timing.IsEnabled()) {
        LOG_DEBUG(Render_Compilation, "async_pipeline_wait_start key={:#x}", pipeline_hash);
    }
    const char* outcome = "exception";
    SCOPE_EXIT {
        if (timing.IsEnabled()) {
            LOG_DEBUG(Render_Compilation,
                      "async_pipeline_wait_finish key={:#x} elapsedMs={:.3f} outcome={}",
                      pipeline_hash, timing.ElapsedMilliseconds(), outcome);
        }
    };
    for (;;) {
        if (PublishAsyncGraphicsPipelines(&key)) {
            outcome = "failed";
            return nullptr;
        }
        if (const auto it = graphics_pipelines.find(key); it != graphics_pipelines.end()) {
            outcome = "ready";
            return it->second.get();
        }

        auto completion = async_graphics_compiler->WaitTake();
        if (!completion) {
            outcome = "stopped";
            return nullptr;
        }
        const bool requested_pipeline_failed = completion->key == key && !completion->result;
        PublishAsyncGraphicsPipeline(std::move(*completion));
        if (requested_pipeline_failed) {
            outcome = "failed";
            return nullptr;
        }
    }
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline() {
    PublishAsyncGraphicsPipelines();
    if (!RefreshGraphicsKey()) {
        return nullptr;
    }
    if (const auto it = graphics_pipelines.find(graphics_key); it != graphics_pipelines.end()) {
        return it->second.get();
    }

    if (async_graphics_compiler) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        if (async_graphics_compiler->Submit(graphics_key, MakeAsyncGraphicsPipelineJob())) {
            LOG_INFO(Render_Vulkan, "Queued graphics pipeline {:#x} for asynchronous compilation",
                     pipeline_hash);
        }
        fetch_shader.reset();
        return WaitForAsyncGraphicsPipeline(graphics_key);
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);
        ScopedHostCompilationGuestPause pause{DebugState};

        GraphicsPipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, false);

        if (!it.value()->IsValid()) {
            graphics_pipelines.erase(it);
            fetch_shader.reset();
            return nullptr;
        }

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
        fetch_shader.reset();
    }
    return it->second.get();
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);
        // The dispatch waits for its pipeline, but async mode must not stop guest threads.
        ScopedHostCompilationGuestPause pause{
            DebugState, !EmulatorSettings.IsAsyncGraphicsPipelineCompilation()};

        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey() {
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();
    }

    // Compile and bind shader stages
    if (!RefreshGraphicsStages()) {
        return false;
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;
    fetch_shader = std::nullopt;

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](Shader::Stage stage_in, Shader::LogicalStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader_;
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_,
                 key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        if (infos[stage_out_idx] == nullptr || !modules[stage_out_idx]) {
            key.stage_hashes[stage_out_idx] = 0;
            return false;
        }
        if (fetch_shader_) {
            fetch_shader = fetch_shader_;
        }
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);

    if (regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Fragment)) &&
        !bind_stage(Stage::Fragment, LogicalStage::Fragment)) {
        return false;
    }

    const auto* fs_info = infos[static_cast<u32>(LogicalStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Vertex, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        if (!bind_stage(Stage::Vertex, LogicalStage::Vertex)) {
            return false;
        }
        break;
    default:
        UNREACHABLE_MSG("unhandled stage_en: {}", (u32)regs.stage_enable.raw);
    }

    const auto* vs_info = infos[static_cast<u32>(Shader::LogicalStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        if (vs_info->captured_vertex_buffers.size() != fetch_shader->attributes.size()) {
            LOG_ERROR(Render_Vulkan,
                      "Vertex descriptor snapshot count does not match fetch shader attributes");
            return false;
        }
        u32 vertex_binding = 0;
        for (const auto& buffer : vs_info->captured_vertex_buffers) {
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    std::tie(infos[0], modules[0], fetch_shader, compute_key.value) =
        GetProgram(Shader::Stage::Compute, LogicalStage::Compute, cs_params, binding);
    return infos[0] != nullptr && static_cast<bool>(modules[0]);
}

std::optional<Shader::IR::Program> PipelineCache::TranslateModule(
    Shader::Info& info, Shader::RuntimeInfo& runtime_info, const std::span<const u32>& code,
    Shader::SrtMemoryReader reader) {
    auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile, reader);
    if (!reader.Valid() || info.translation_failed) {
        return std::nullopt;
    }
    return ir_program;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              const std::vector<u32>& spv) {
    ScopedHostCompilationGuestPause pause{DebugState,
                                          !EmulatorSettings.IsAsyncGraphicsPipelineCompilation()};
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.stage, perm_idx, "bin");
    DumpShader(spv, info.pgm_hash, info.stage, perm_idx, "spv");

    vk::ShaderModule module;
    auto patch = GetShaderPatch(info.pgm_hash, info.stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    const auto name = GetShaderName(info.stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(Stage stage, LogicalStage l_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    return VideoCore::RetrySrtPreparation(
        binding,
        [&](Shader::Backend::Bindings& binding, VideoCore::SrtMemoryReadFailure& attempt_failure,
            bool& attempt_translation_rejected) -> Result {
            SrtReadSession srt_read_session{page_manager, buffer_cache};
            const auto reader = MakeSrtMemoryReader(srt_read_session);
            auto runtime_info = BuildRuntimeInfo(stage, l_stage, reader);
            const Shader::ProgramCacheKey cache_key{
                .program_hash = params.hash,
                .stage = stage,
                .logical_stage = l_stage,
            };

            const auto fail_snapshot = [&](std::string_view phase,
                                           bool translation_rejected = false) {
                attempt_failure = srt_read_session.LastFailure();
                attempt_translation_rejected = translation_rejected;
                LOG_WARNING(Render_Vulkan,
                            "Preparing {} shader {:#x}: resource preparation failed "
                            "phase={} read={} translationRejected={}",
                            stage, params.hash, phase,
                            VideoCore::SrtMemoryReadFailureName(srt_read_session.LastFailure()),
                            translation_rejected);
                return Result{};
            };
            const auto seal_snapshot = [&] { return reader.Valid() && srt_read_session.Seal(); };
            const auto capture_vertex_buffers = [&](Shader::Info& info,
                                                    const auto& fetch_shader_data) {
                if (!fetch_shader_data) {
                    info.captured_vertex_buffers.clear();
                    return reader.Valid();
                }
                return Shader::Gcn::CaptureVertexBuffers(info, *fetch_shader_data, reader);
            };

            auto [it_pgm, new_program] = program_cache.try_emplace(cache_key);
            if (new_program) {
                it_pgm.value() = std::make_unique<Program>(stage, l_stage, params);
                auto& program = it_pgm.value();
                const auto start = binding;
                auto ir_program = TranslateModule(program->info, runtime_info, params.code, reader);
                if (!ir_program) {
                    const bool translation_rejected = program->info.translation_failed;
                    program_cache.erase(cache_key);
                    return fail_snapshot("initial_translation", translation_rejected);
                }

                auto admitted_spec = Shader::TryCreateStageSpecialization(
                    program->info, runtime_info, profile, start, reader);
                if (!admitted_spec || !reader.Valid()) {
                    program_cache.erase(cache_key);
                    return fail_snapshot("initial_specialization", !admitted_spec);
                }
                auto& spec = *admitted_spec;
                const auto permutation_id =
                    Shader::ComputeShaderPermutationIdentity(params.hash, l_stage, spec);
                auto compiled_binding = binding;
                auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, *ir_program,
                                                             compiled_binding, reader);
                if (!capture_vertex_buffers(program->info, spec.fetch_shader_data)) {
                    program_cache.erase(cache_key);
                    return fail_snapshot("initial_vertex_capture");
                }
                if (!seal_snapshot()) {
                    program_cache.erase(cache_key);
                    return fail_snapshot("initial_seal");
                }

                const auto module = CompileModule(program->info, params.code, 0, spv);
                if (RegisterShaderMeta(program->info, spec, permutation_id)) {
                    RegisterShaderBinary(std::move(spv), permutation_id);
                }
                auto module_info = program->info;
                program->AddPermut(module, std::move(spec), std::move(module_info), permutation_id);
                binding = compiled_binding;
                return std::make_tuple(program->modules[0].info.get(), module,
                                       program->modules[0].spec.fetch_shader_data, permutation_id);
            }

            auto& program = it_pgm.value();
            auto aggregate_snapshot = program->info;
            aggregate_snapshot.pgm_base = params.Base();
            aggregate_snapshot.user_data = params.user_data;
            if (!aggregate_snapshot.RefreshFlatBuf(reader)) {
                return fail_snapshot("aggregate_refresh");
            }
            const auto start = binding;
            const auto probe_spec = Shader::TryCreateStageSpecialization(
                aggregate_snapshot, runtime_info, profile, start, reader);
            if (!reader.Valid()) {
                return fail_snapshot("probe_specialization");
            }

            std::string_view selected_failure = "selected_missing_info";
            const auto validate_selected =
                [&](const Program::Module& candidate) -> std::optional<Shader::Info> {
                if (!candidate.info) {
                    selected_failure = "selected_missing_info";
                    return std::nullopt;
                }
                auto selected_info = *candidate.info;
                selected_info.pgm_base = params.Base();
                selected_info.user_data = params.user_data;
                if (!selected_info.RefreshFlatBuf(reader)) {
                    selected_failure = "selected_refresh";
                    return std::nullopt;
                }
                const auto selected_spec = Shader::TryCreateStageSpecialization(
                    selected_info, runtime_info, profile, start, reader);
                if (!selected_spec || !reader.Valid()) {
                    selected_failure = "selected_specialization";
                    return std::nullopt;
                }
                const auto selected_id =
                    Shader::ComputeShaderPermutationIdentity(params.hash, l_stage, *selected_spec);
                if (!reader.Valid() || selected_id != candidate.permutation_id ||
                    *selected_spec != candidate.spec) {
                    selected_failure = "selected_specialization";
                    return std::nullopt;
                }
                if (!capture_vertex_buffers(selected_info, candidate.spec.fetch_shader_data)) {
                    selected_failure = "selected_vertex_capture";
                    return std::nullopt;
                }
                return selected_info;
            };

            // An aggregate describes previous translations, not necessarily the current one.
            // Refusing its conversion makes it a cache miss; fresh IR below remains authoritative.
            if (const auto probe =
                    probe_spec ? std::ranges::find(program->modules, *probe_spec,
                                                  &Program::Module::spec)
                               : program->modules.end();
                probe != program->modules.end()) {
                auto selected_info = validate_selected(*probe);
                if (selected_info) {
                    if (!seal_snapshot()) {
                        return fail_snapshot("cached_seal");
                    }
                    *probe->info = std::move(*selected_info);
                    probe->info->AddBindings(binding);
                    return std::make_tuple(probe->info.get(), probe->module,
                                           probe->spec.fetch_shader_data, probe->permutation_id);
                }
            }
            if (!reader.Valid()) {
                return fail_snapshot(selected_failure);
            }

            const size_t perm_idx = program->modules.size();
            auto new_info = Shader::Info(stage, l_stage, params);
            auto ir_program = TranslateModule(new_info, runtime_info, params.code, reader);
            if (!ir_program) {
                return fail_snapshot("permutation_translation", new_info.translation_failed);
            }
            auto admitted_spec = Shader::TryCreateStageSpecialization(
                new_info, runtime_info, profile, start, reader);
            if (!admitted_spec || !reader.Valid()) {
                return fail_snapshot("permutation_specialization", !admitted_spec);
            }
            auto& authoritative_spec = *admitted_spec;
            const u64 permutation_id =
                Shader::ComputeShaderPermutationIdentity(params.hash, l_stage, authoritative_spec);
            const auto authoritative =
                program->InspectPreloadPermutation(authoritative_spec, permutation_id);
            if (authoritative.action == Program::PreloadAction::Conflict || !reader.Valid()) {
                return fail_snapshot("permutation_identity");
            }

            if (authoritative.action == Program::PreloadAction::Reuse) {
                if (!authoritative.module) {
                    return fail_snapshot("reuse_missing_module");
                }
                auto selected_info = validate_selected(*authoritative.module);
                if (!selected_info) {
                    return fail_snapshot(selected_failure);
                }
                if (!seal_snapshot()) {
                    return fail_snapshot("reuse_seal");
                }
                *authoritative.module->info = std::move(*selected_info);
                authoritative.module->info->AddBindings(binding);
                return std::make_tuple(authoritative.module->info.get(),
                                       authoritative.module->module,
                                       authoritative.module->spec.fetch_shader_data,
                                       authoritative.module->permutation_id);
            }

            auto compiled_binding = binding;
            auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, *ir_program,
                                                         compiled_binding, reader);
            if (!capture_vertex_buffers(new_info, authoritative_spec.fetch_shader_data)) {
                return fail_snapshot("permutation_vertex_capture");
            }
            if (!seal_snapshot()) {
                return fail_snapshot("permutation_seal");
            }

            const auto module = CompileModule(new_info, params.code, perm_idx, spv);
            if (RegisterShaderMeta(new_info, authoritative_spec, permutation_id)) {
                RegisterShaderBinary(std::move(spv), permutation_id);
            }
            program->AddPermut(module, std::move(authoritative_spec), std::move(new_info),
                               permutation_id);
            binding = compiled_binding;
            const auto& selected = program->modules.back();
            return std::make_tuple(selected.info.get(), selected.module,
                                   selected.spec.fetch_shader_data, selected.permutation_id);
        },
        [](const Result& result) { return std::get<0>(result) && std::get<1>(result); });
}
std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                new_module = m.module;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::Stage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::Stage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
} // namespace Vulkan
