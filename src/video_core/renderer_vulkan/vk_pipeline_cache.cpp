// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include <ranges>
#include <stdexcept>

#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
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
#include "video_core/cache_storage.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/host_compilation_pause.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

constexpr size_t MaxOutstandingAsyncShaderModules = 16;

using Shader::LogicalStage;
using Shader::Output;
using Shader::Stage;

static Shader::SrtMemoryReader MakeSrtMemoryReader(SrtReadSession& session) {
    return {&SrtReadSession::Callback, &session};
}

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

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(Stage stage, LogicalStage l_stage) {
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
        hull_info->ReadTessConstantBuffer(tess_constants);
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
        DumpShader(gs_info.vs_copy, gs_info.vs_copy_hash, Shader::Stage::Vertex, "copy.bin");
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
        async_graphics_compiler = std::make_unique<AsyncGraphicsPipelineQueue>(
            [this](AsyncGraphicsPipelineJob job) {
                return CompileGraphicsPipelineAsync(std::move(job));
            },
            std::numeric_limits<size_t>::max());
    }
}

PipelineCache::~PipelineCache() {
    async_graphics_compiler.reset();
    async_shader_compiler.reset();
    pending_shader_modules.clear();
    const auto device = instance.GetDevice();
    for (const auto module : retired_shader_modules) {
        device.destroyShaderModule(module);
    }
}

AsyncGraphicsPipelineJob PipelineCache::MakeAsyncGraphicsPipelineJob() const {
    AsyncGraphicsPipelineJob job{
        .key = graphics_key,
        .retained_infos = infos,
        .fetch_shader = fetch_shader,
        .modules = modules,
        .module_generation = module_generation,
    };
    job.CaptureRuntimeInfos(runtime_infos);
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

AsyncGraphicsPipelineResult PipelineCache::CompileGraphicsPipelineAsync(
    AsyncGraphicsPipelineJob job) {
    job.RebindCompileInfo();
    GraphicsPipeline::SerializationSupport serialization{};
    auto pipeline = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, job.key, vk::PipelineCache{}, job.compile_infos,
        job.runtime_infos, job.fetch_shader, job.modules, serialization, false);
    return {
        .pipeline = std::move(pipeline),
        .serialization = std::move(serialization),
        .modules = job.modules,
        .retained_infos = job.retained_infos,
        .module_generation = job.module_generation,
    };
}

void PipelineCache::PublishAsyncGraphicsPipeline(
    AsyncGraphicsPipelineQueue::Completion completion) {
    if (completion.key.module_generation != module_generation) {
        LOG_INFO(Render_Vulkan,
                 "Discarded stale asynchronous graphics pipeline completion after shader "
                 "replacement");
        return;
    }
    if (!completion.result) {
        LOG_ERROR(Render_Vulkan, "Asynchronous graphics pipeline compilation failed");
        failed_graphics_pipelines.insert(completion.key.pipeline);
        return;
    }

    auto result = std::move(*completion.result);
    if (result.module_generation != module_generation) {
        LOG_INFO(Render_Vulkan,
                 "Discarded stale asynchronous graphics pipeline after shader replacement");
        return;
    }
    const auto [it, is_new] = graphics_pipelines.try_emplace(completion.key.pipeline);
    if (!is_new) {
        return;
    }
    result.pipeline->RebindStages(result.retained_infos);
    it.value() = std::move(result.pipeline);
    const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(completion.key.pipeline);
    RegisterPipelineData(completion.key.pipeline, pipeline_hash, result.serialization);
    ++num_new_pipelines;

    if (EmulatorSettings.IsShaderCollect()) {
        for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
            if (completion.key.pipeline.stage_hashes[stage]) {
                module_related_pipelines[result.modules[stage]].emplace_back(
                    completion.key.pipeline);
            }
        }
    }
    LOG_INFO(Render_Vulkan, "Published asynchronous graphics pipeline {:#x}", pipeline_hash);
}

void PipelineCache::PublishAsyncGraphicsPipelines() {
    if (!async_graphics_compiler) {
        return;
    }
    while (auto completion = async_graphics_compiler->TryTake()) {
        PublishAsyncGraphicsPipeline(std::move(*completion));
    }
}

AsyncShaderModuleResult PipelineCache::CompileShaderModuleAsync(
    std::unique_ptr<AsyncShaderModuleJob> job) {
    ASSERT(job && job->ir_program.has_value());
    job->RebindOwnedViews();
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, job->runtime_info, *job->ir_program,
                                                 job->binding);
    job->ir_program.reset();
    job->pools.reset();
    const std::span<const u32> module_code = job->is_patched ? std::span<const u32>{*job->patch}
                                                             : std::span<const u32>{spv};
    const vk::ShaderModuleCreateInfo shader_info{
        .codeSize = module_code.size_bytes(),
        .pCode = module_code.data(),
    };
    auto [module_result, module] = instance.GetDevice().createShaderModuleUnique(shader_info);
    if (module_result != vk::Result::eSuccess) {
        throw std::runtime_error{
            fmt::format("Failed to compile asynchronous SPIR-V shader: {}",
                        vk::to_string(module_result))};
    }
    return {
        .job = std::move(job),
        .spv = std::move(spv),
        .module = std::move(module),
    };
}

bool PipelineCache::SubmitAsyncShaderModule(std::unique_ptr<AsyncShaderModuleJob> job) {
    ASSERT(async_shader_compiler && job && job->ir_program.has_value());
    const auto key = job->key;
    if (pending_shader_modules.contains(key) ||
        !async_shader_compiler->Submit(key, std::move(job))) {
        return false;
    }
    pending_shader_modules.insert(key);
    LOG_INFO(Render_Vulkan, "Queued {} shader permutation {:#x} for asynchronous backend emission",
             key.program.stage, key.permutation_id);
    return true;
}

void PipelineCache::PublishAsyncShaderModule(AsyncShaderModuleQueue::Completion completion) {
    pending_shader_modules.erase(completion.key);
    if (!completion.result) {
        LOG_ERROR(Render_Vulkan,
                  "Asynchronous shader-module emission failed for permutation {:#x}",
                  completion.key.permutation_id);
        return;
    }

    auto result = std::move(*completion.result);
    ASSERT(result.job);
    auto& job = *result.job;
    job.RebindOwnedViews();
    if (job.key != completion.key) {
        LOG_ERROR(Render_Vulkan, "Discarded asynchronous shader result with mismatched key");
        return;
    }

    const auto program_it = program_cache.find(job.key.program);
    if (program_it == program_cache.end() ||
        program_it->second->generation != job.key.program_generation) {
        LOG_INFO(Render_Vulkan,
                 "Discarded stale asynchronous shader permutation {:#x}",
                 job.key.permutation_id);
        return;
    }

    auto& program = *program_it->second;
    const auto decision = program.InspectPreloadPermutation(job.spec, job.key.permutation_id);
    if (decision.action == Program::PreloadAction::Reuse) {
        LOG_INFO(Render_Vulkan,
                 "Discarded duplicate asynchronous shader permutation {:#x}",
                 job.key.permutation_id);
        return;
    }
    if (decision.action == Program::PreloadAction::Conflict) {
        LOG_ERROR(Render_Vulkan,
                  "Discarded conflicting asynchronous shader permutation {:#x}",
                  job.key.permutation_id);
        return;
    }

    const auto module = result.module.get();
    DumpShader(result.spv, job.info.pgm_hash, job.info.stage, "spv", job.key.permutation_id);
    if (job.is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", job.info.stage, job.info.pgm_hash);
    }
    const auto name = GetShaderName(job.info.stage, job.info.pgm_hash, job.key.permutation_id);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, job.info.l_stage, module, result.spv, job.code,
                                 job.patch ? std::span<const u32>{*job.patch}
                                           : std::span<const u32>{},
                                 job.is_patched);
    }
    RegisterShaderMeta(job.info, job.spec.fetch_shader_data, job.spec, job.key.permutation_id);
    RegisterShaderBinary(std::move(result.spv), job.key.permutation_id);

    job.info.user_data = {};
    job.info.ClearCompileCaptureReader();
    job.info.ClearCompileFetchShaderSnapshot();
    program.AddPermut(module, std::move(job.spec), std::move(job.info), job.key.permutation_id);
    result.module.release();
    RecordShaderSelection(job.key.program.stage, job.key.program.logical_stage,
                          job.key.program.program_hash, job.provenance, job.key.permutation_id,
                          VideoCore::ShaderSelectionCacheAction::New, program.modules.size(),
                          job.key.permutation_id);
    LOG_INFO(Render_Vulkan, "Published asynchronous {} shader permutation {:#x}",
             job.key.program.stage, job.key.permutation_id);
}

void PipelineCache::PublishAsyncShaderModules() {
    if (!async_shader_compiler) {
        return;
    }
    while (auto completion = async_shader_compiler->TryTake()) {
        PublishAsyncShaderModule(std::move(*completion));
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

    if (failed_graphics_pipelines.contains(graphics_key)) {
        fetch_shader.reset();
        return nullptr;
    }

    if (async_graphics_compiler) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        const AsyncGraphicsPipelineKey async_key{
            .pipeline = graphics_key,
            .module_generation = module_generation,
        };
        if (async_graphics_compiler->Submit(async_key, MakeAsyncGraphicsPipelineJob())) {
            LOG_INFO(Render_Vulkan, "Queued graphics pipeline {:#x} for asynchronous compilation",
                     pipeline_hash);
        }
        fetch_shader.reset();
        return nullptr;
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
        ScopedHostCompilationGuestPause pause{DebugState};

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

void PipelineCache::BeginComputeDispatchTrace() noexcept {
    if (!shader_selection_trace.Enabled()) {
        return;
    }
    compute_dispatch_sequence.fetch_add(1, std::memory_order_relaxed);
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
    bool stage_selection_failed = false;

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
            stage_selection_failed = true;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader_;
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_,
                 key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        if (infos[stage_out_idx] == nullptr || !modules[stage_out_idx]) {
            stage_selection_failed = true;
            return false;
        }
        if (fetch_shader_) {
            fetch_shader = fetch_shader_;
        }
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);

    bind_stage(Stage::Fragment, LogicalStage::Fragment);
    if (stage_selection_failed) {
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
        bind_stage(Stage::Vertex, LogicalStage::Vertex);
        break;
    default:
        UNREACHABLE_MSG("unhandled stage_en: {}", (u32)regs.stage_enable.raw);
    }

    const auto* vs_info = infos[static_cast<u32>(Shader::LogicalStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return !stage_selection_failed;
}

bool PipelineCache::RefreshComputeKey() {
    pending_compute_selection_trace.reset();
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    std::tie(infos[0], modules[0], fetch_shader, compute_key.value) =
        GetProgram(Shader::Stage::Compute, LogicalStage::Compute, cs_params, binding);
    const bool selected = infos[0] != nullptr && static_cast<bool>(modules[0]);
    if (selected && pending_compute_selection_trace) {
        pending_compute_selection_trace->compute_pipeline_id = compute_key.value;
        shader_selection_trace.Record(*pending_compute_selection_trace);
    }
    pending_compute_selection_trace.reset();
    return selected;
}

std::optional<Shader::IR::Program> PipelineCache::TranslateModule(
    Shader::Pools& translation_pools, Shader::Info& info, Shader::RuntimeInfo& runtime_info,
    const std::span<const u32>& code, size_t perm_idx,
    const Shader::CompileResourceSnapshot* compile_resources,
    const Shader::Gcn::FetchShaderData* compile_fetch_shader,
    Shader::SrtMemoryReader compile_reader) {
    if (compile_resources) {
        ASSERT(compile_resources->valid);
        info.UseCompileResourceSnapshot(*compile_resources);
    }
    if (compile_fetch_shader) {
        info.UseCompileFetchShaderSnapshot(*compile_fetch_shader);
    }
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.stage, "bin");

    auto ir_program =
        Shader::TranslateProgram(code, translation_pools, info, runtime_info, profile,
                                 compile_reader);
    if (!compile_reader.Valid()) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping {} shader {:#x}: its compile-time guest-memory snapshot changed "
                    "or could not be read",
                    info.stage, info.pgm_hash);
        return {};
    }
    return ir_program;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code,
                                              const Shader::IR::Program& ir_program,
                                              Shader::Backend::Bindings& binding,
                                              u64 permutation_id) {
    ScopedHostCompilationGuestPause pause{DebugState,
                                          !EmulatorSettings.IsAsyncGraphicsPipelineCompilation()};
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    DumpShader(spv, info.pgm_hash, info.stage, "spv", permutation_id);

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.stage, permutation_id, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    const auto name = GetShaderName(info.stage, info.pgm_hash, permutation_id);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    RegisterShaderBinary(std::move(spv), permutation_id);
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(Stage stage, LogicalStage l_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    PublishAsyncShaderModules();
    constexpr u32 MaxCaptureAttempts = 2;
    const auto initial_binding = binding;
    const bool collect_debug_provenance = EmulatorSettings.IsShaderCollect();
    const bool collect_selection_provenance = shader_selection_trace.Enabled();
    SrtReadSession srt_read_session{page_manager, buffer_cache,
                                    collect_debug_provenance || collect_selection_provenance};
    const auto record_provenance = [&](bool succeeded) {
        if (collect_debug_provenance) {
            const auto payload = DebugState.CollectShaderCaptureProvenance(
                params.hash, static_cast<u32>(stage), succeeded, srt_read_session.Provenance());
            if (payload) {
                LOG_INFO(Render_Vulkan, "{}", *payload);
            }
        }
    };
    bool pending = false;
    for (u32 attempt = 0; attempt < MaxCaptureAttempts; ++attempt) {
        binding = initial_binding;
        bool attempt_pending = false;
        auto result =
            GetProgramAttempt(stage, l_stage, params, binding, srt_read_session, attempt_pending);
        if (std::get<0>(result) != nullptr && static_cast<bool>(std::get<1>(result))) {
            record_provenance(true);
            return result;
        }
        if (attempt_pending) {
            pending = true;
            break;
        }
        if (!srt_read_session.Valid()) {
            break;
        }
    }
    binding = initial_binding;
    if (!pending) {
        record_provenance(false);
    }
    return {};
}

void PipelineCache::RecordShaderSelection(
    Stage stage, LogicalStage l_stage, u64 program_hash,
    const VideoCore::SrtCaptureProvenance& provenance, u64 semantic_identity,
    VideoCore::ShaderSelectionCacheAction cache_action, size_t module_count,
    u64 selected_permutation_id) noexcept {
    if (!shader_selection_trace.Enabled()) {
        return;
    }
    VideoCore::ShaderSelectionTraceRecord record{
        .dispatch_sequence = l_stage == LogicalStage::Compute
                                 ? compute_dispatch_sequence.load(std::memory_order_relaxed)
                                 : 0,
        .physical_stage = stage,
        .logical_stage = l_stage,
        .program_hash = program_hash,
        .srt_source_digest = VideoCore::DigestShaderSelectionSrtSources(provenance),
        .srt_content_digest = VideoCore::DigestShaderSelectionSrtContents(provenance),
        .semantic_identity = semantic_identity,
        .cache_action = cache_action,
        .module_count = static_cast<u32>(module_count),
        .selected_permutation_id = selected_permutation_id,
        .compute_pipeline_id = 0,
    };
    if (l_stage == LogicalStage::Compute &&
        cache_action != VideoCore::ShaderSelectionCacheAction::Reject) {
        pending_compute_selection_trace = record;
    } else {
        shader_selection_trace.Record(record);
    }
}

PipelineCache::Result PipelineCache::GetProgramAttempt(Stage stage, LogicalStage l_stage,
                                                       const Shader::ShaderParams& params,
                                                       Shader::Backend::Bindings& binding,
                                                       SrtReadSession& srt_read_session,
                                                       bool& pending) {
    pending = false;
    auto runtime_info = BuildRuntimeInfo(stage, l_stage);
    const Shader::ProgramCacheKey cache_key{
        .program_hash = params.hash,
        .stage = stage,
        .logical_stage = l_stage,
    };
    if (async_shader_compiler && l_stage != LogicalStage::Compute &&
        !program_cache.contains(cache_key) && !async_shader_compiler->CanAccept()) {
        pending = true;
        return {};
    }
    auto [it_pgm, new_program] = program_cache.try_emplace(cache_key);
    if (new_program) {
        const u64 program_generation = next_program_generation++;
        it_pgm.value() =
            std::make_unique<Program>(stage, l_stage, params, program_generation);
        auto& program = it_pgm.value();
        auto start = binding;
        Shader::CompileResourceCapture compile_capture{MakeSrtMemoryReader(srt_read_session)};
        const auto capture_reader = compile_capture.Reader();

        if (async_shader_compiler && l_stage != LogicalStage::Compute) {
            auto job = std::make_unique<AsyncShaderModuleJob>();
            job->CaptureOwnedInputs(stage, l_stage, params, runtime_info, start);
            job->info.UseCompileCaptureReader(capture_reader);
            auto translated = TranslateModule(*job->pools, job->info, job->runtime_info, job->code,
                                              0, nullptr, nullptr, capture_reader);
            if (!translated) {
                program_cache.erase(cache_key);
                return {};
            }
            job->ir_program.emplace(std::move(*translated));

            auto spec =
                Shader::StageSpecialization(job->info, job->runtime_info, profile, start);
            const auto permutation_id =
                Shader::ComputeShaderPermutationIdentity(params.hash, l_stage, spec);
            if (!capture_reader.Valid()) {
                RecordShaderSelection(stage, l_stage, params.hash, srt_read_session.Provenance(),
                                      permutation_id,
                                      VideoCore::ShaderSelectionCacheAction::Reject,
                                      program->modules.size(), 0);
                program_cache.erase(cache_key);
                return {};
            }
            const bool epoch_valid = srt_read_session.Seal();
            auto compile_resources = compile_capture.Finish(job->info, epoch_valid);
            if (!compile_resources.valid) {
                RecordShaderSelection(stage, l_stage, params.hash, srt_read_session.Provenance(),
                                      permutation_id,
                                      VideoCore::ShaderSelectionCacheAction::Reject,
                                      program->modules.size(), 0);
                program_cache.erase(cache_key);
                return {};
            }
            job->info.UseCompileResourceSnapshot(std::move(compile_resources));
            program->info = job->info;
            program->info.pgm_base = params.Base();
            program->info.user_data = params.user_data;
            program->info.ClearCompileResourceAccess();
            program->info.ClearCompileFetchShaderSnapshot();

            job->spec = std::move(spec);
            job->key = {
                .program = cache_key,
                .program_generation = program_generation,
                .permutation_id = permutation_id,
            };
            job->provenance = srt_read_session.Provenance();
            job->patch = GetShaderPatch(job->info.pgm_hash, job->info.stage, permutation_id, "spv");
            job->is_patched = job->patch.has_value() && EmulatorSettings.IsPatchShaders();
            job->RebindOwnedViews();
            if (!SubmitAsyncShaderModule(std::move(job))) {
                program_cache.erase(cache_key);
                return {};
            }
            pending = true;
            return {};
        }

        program->info.UseCompileCaptureReader(capture_reader);
        auto ir_program = TranslateModule(pools, program->info, runtime_info, params.code, 0,
                                          nullptr, nullptr, capture_reader);
        if (!ir_program) {
            program_cache.erase(cache_key);
            return {};
        }
        auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, start);
        const auto permutation_id =
            Shader::ComputeShaderPermutationIdentity(params.hash, l_stage, spec);
        if (!capture_reader.Valid()) {
            RecordShaderSelection(stage, l_stage, params.hash, srt_read_session.Provenance(),
                                  permutation_id,
                                  VideoCore::ShaderSelectionCacheAction::Reject,
                                  program->modules.size(), 0);
            program_cache.erase(cache_key);
            return {};
        }
        const bool epoch_valid = srt_read_session.Seal();
        const auto compile_resources = compile_capture.Finish(program->info, epoch_valid);
        if (!compile_resources.valid) {
            RecordShaderSelection(stage, l_stage, params.hash, srt_read_session.Provenance(),
                                  permutation_id,
                                  VideoCore::ShaderSelectionCacheAction::Reject,
                                  program->modules.size(), 0);
            program_cache.erase(cache_key);
            return {};
        }
        program->info.UseCompileResourceSnapshot(compile_resources);
        const auto module = CompileModule(program->info, runtime_info, params.code, *ir_program,
                                          binding, permutation_id);
        if (!module) {
            RecordShaderSelection(stage, l_stage, params.hash, srt_read_session.Provenance(),
                                  permutation_id,
                                  VideoCore::ShaderSelectionCacheAction::Reject,
                                  program->modules.size(), 0);
            program_cache.erase(cache_key);
            return {};
        }
        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, permutation_id);
        auto module_info = program->info;
        program->AddPermut(module, std::move(spec), std::move(module_info), permutation_id);
        RecordShaderSelection(stage, l_stage, params.hash, srt_read_session.Provenance(),
                              permutation_id, VideoCore::ShaderSelectionCacheAction::New,
                              program->modules.size(), permutation_id);
        auto& selected_info = program->GetPermutInfo(0);
        return std::make_tuple(&selected_info, module,
                               program->modules[0].spec.fetch_shader_data, permutation_id);
    }

    auto& program = it_pgm.value();
    auto& info = program->info;
    info.ClearCompileResourceAccess();
    info.ClearCompileFetchShaderSnapshot();
    info.pgm_base = params.Base(); // Needs to be actualized for inline cbuffer address fixup
    info.user_data = params.user_data;
    Shader::CompileResourceCapture compile_capture{MakeSrtMemoryReader(srt_read_session)};
    const auto capture_reader = compile_capture.Reader();
    info.UseCompileCaptureReader(capture_reader);
    info.RefreshFlatBuf(capture_reader);
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);
    const auto semantic_identity =
        Shader::ComputeShaderPermutationIdentity(params.hash, l_stage, spec);
    const auto* compile_fetch_shader = spec.fetch_shader_data ? &*spec.fetch_shader_data : nullptr;
    const AsyncShaderModuleKey async_shader_key{
        .program = cache_key,
        .program_generation = program->generation,
        .permutation_id = semantic_identity,
    };
    if (async_shader_compiler && l_stage != LogicalStage::Compute &&
        pending_shader_modules.contains(async_shader_key)) {
        info.ClearCompileResourceAccess();
        pending = true;
        return {};
    }
    if (!capture_reader.Valid()) {
        RecordShaderSelection(stage, l_stage, params.hash, srt_read_session.Provenance(),
                              semantic_identity, VideoCore::ShaderSelectionCacheAction::Reject,
                              program->modules.size(), 0);
        info.ClearCompileResourceAccess();
        return {};
    }
    const bool epoch_valid = srt_read_session.Seal();
    const auto compile_resources = compile_capture.Finish(info, epoch_valid);
    info.ClearCompileResourceAccess();
    if (!compile_resources.valid) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping {} shader {:#x}: its specialization snapshot changed or could "
                    "not be read",
                     info.stage, info.pgm_hash);
        RecordShaderSelection(stage, l_stage, params.hash, srt_read_session.Provenance(),
                              semantic_identity, VideoCore::ShaderSelectionCacheAction::Reject,
                              program->modules.size(), 0);
        return {};
    }

    size_t perm_idx = program->modules.size();
    u64 permutation_id{};

    vk::ShaderModule module{};

    const auto it = program->FindPermut(spec);
    const bool new_permutation = it == program->modules.end();
    if (new_permutation) {
        permutation_id = semantic_identity;
        if (program->InspectPreloadPermutation(spec, permutation_id).action !=
            Program::PreloadAction::Insert) {
            LOG_ERROR(Render_Vulkan,
                      "Shader permutation identity {:#x} of {}_{:x} collides with an existing "
                      "specialization, rejecting the new permutation",
                      permutation_id, stage, params.hash);
            RecordShaderSelection(stage, l_stage, params.hash, srt_read_session.Provenance(),
                                  semantic_identity,
                                  VideoCore::ShaderSelectionCacheAction::Reject,
                                  program->modules.size(), 0);
            return {};
        }
        if (async_shader_compiler && l_stage != LogicalStage::Compute) {
            if (!async_shader_compiler->CanAccept()) {
                pending = true;
                return {};
            }
            auto job = std::make_unique<AsyncShaderModuleJob>();
            job->key = async_shader_key;
            job->CaptureOwnedInputs(stage, l_stage, params, runtime_info, binding);
            auto translated = TranslateModule(*job->pools, job->info, job->runtime_info, job->code,
                                              perm_idx, &compile_resources,
                                              compile_fetch_shader);
            if (!translated) {
                RecordShaderSelection(stage, l_stage, params.hash,
                                      srt_read_session.Provenance(), semantic_identity,
                                      VideoCore::ShaderSelectionCacheAction::Reject,
                                      program->modules.size(), 0);
                return {};
            }
            job->ir_program.emplace(std::move(*translated));
            job->spec = std::move(spec);
            job->provenance = srt_read_session.Provenance();
            job->patch = GetShaderPatch(job->info.pgm_hash, job->info.stage, permutation_id, "spv");
            job->is_patched = job->patch.has_value() && EmulatorSettings.IsPatchShaders();
            job->RebindOwnedViews();
            if (!SubmitAsyncShaderModule(std::move(job))) {
                return {};
            }
            pending = true;
            return {};
        } else {
            auto new_info = Shader::Info(stage, l_stage, params);
            auto ir_program = TranslateModule(pools, new_info, runtime_info, params.code, perm_idx,
                                              &compile_resources, compile_fetch_shader);
            if (!ir_program) {
                RecordShaderSelection(stage, l_stage, params.hash,
                                      srt_read_session.Provenance(), semantic_identity,
                                      VideoCore::ShaderSelectionCacheAction::Reject,
                                      program->modules.size(), 0);
                return {};
            }
            module = CompileModule(new_info, runtime_info, params.code, *ir_program, binding,
                                   permutation_id);
            if (!module) {
                RecordShaderSelection(stage, l_stage, params.hash,
                                      srt_read_session.Provenance(), semantic_identity,
                                      VideoCore::ShaderSelectionCacheAction::Reject,
                                      program->modules.size(), 0);
                return {};
            }

            RegisterShaderMeta(new_info, spec.fetch_shader_data, spec, permutation_id);
            new_info.ClearCompileCaptureReader();
            new_info.ClearCompileFetchShaderSnapshot();
            program->AddPermut(module, std::move(spec), std::move(new_info));
            program->modules.back().permutation_id = permutation_id;
            RecordShaderSelection(stage, l_stage, params.hash, srt_read_session.Provenance(),
                                  semantic_identity, VideoCore::ShaderSelectionCacheAction::New,
                                  program->modules.size(), permutation_id);
        }
    } else {
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        permutation_id = it->permutation_id;
        RecordShaderSelection(stage, l_stage, params.hash, srt_read_session.Provenance(),
                              semantic_identity, VideoCore::ShaderSelectionCacheAction::Reuse,
                              program->modules.size(), permutation_id);
    }

    auto& selected_info = program->GetPermutInfo(perm_idx);
    selected_info.pgm_base = params.Base();
    selected_info.user_data = params.user_data;
    selected_info.UseCompileResourceSnapshot(compile_resources);
    selected_info.ClearCompileFetchShaderSnapshot();
    selected_info.RefreshFlatBuf();
    if (!new_permutation) {
        selected_info.AddBindings(binding);
    }
    return std::make_tuple(&selected_info, module,
                           program->modules[perm_idx].spec.fetch_shader_data, permutation_id);
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    std::optional<vk::ShaderModule> replacement;
    liverpool->SendCommand<true>([this, module, spv_code, &replacement] {
        replacement = ReplaceShaderOnGpuOwner(module, spv_code);
    });
    return replacement;
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShaderOnGpuOwner(
    vk::ShaderModule module, std::span<const u32> spv_code) {
    bool found = false;
    for (const auto& [_, program] : program_cache) {
        for (const auto& m : program->modules) {
            if (m.module == module) {
                found = true;
            }
        }
    }
    if (!found) {
        return {};
    }

    const auto device = instance.GetDevice();
    const vk::ShaderModuleCreateInfo shader_info{
        .codeSize = spv_code.size_bytes(),
        .pCode = spv_code.data(),
    };
    auto [creation_result, replacement] = device.createShaderModuleUnique(shader_info);
    if (creation_result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Failed to replace SPIR-V shader: {}",
                  vk::to_string(creation_result));
        return {};
    }

    const vk::ShaderModule new_module = replacement.get();
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                std::exchange(m.module, new_module);
            }
        }
    }
    replacement.release();
    ++module_generation;
    failed_graphics_pipelines.clear();

    if (async_graphics_compiler) {
        retired_shader_modules.push_back(module);
    } else {
        device.destroyShaderModule(module);
    }

    if (const auto related_it = module_related_pipelines.find(module);
        related_it != module_related_pipelines.end()) {
        auto pipeline_keys = std::move(related_it->second);
        module_related_pipelines.erase(module);
        for (const auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                const auto& stale_key = std::get<GraphicsPipelineKey>(key);
                if (auto pipeline_it = graphics_pipelines.find(stale_key);
                    pipeline_it != graphics_pipelines.end()) {
                    auto stale_pipeline = std::move(pipeline_it.value());
                    graphics_pipelines.erase(stale_key);
                    scheduler.DeferOperation(
                        [pipeline = std::move(stale_pipeline)]() mutable { pipeline.reset(); });
                }
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                const auto& stale_key = std::get<ComputePipelineKey>(key);
                if (auto pipeline_it = compute_pipelines.find(stale_key);
                    pipeline_it != compute_pipelines.end()) {
                    auto stale_pipeline = std::move(pipeline_it.value());
                    compute_pipelines.erase(stale_key);
                    scheduler.DeferOperation(
                        [pipeline = std::move(stale_pipeline)]() mutable { pipeline.reset(); });
                }
            }
        }
        module_related_pipelines[new_module] = std::move(pipeline_keys);
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::Stage stage, u64 hash,
                                         std::optional<u64> permutation_id) {
    if (permutation_id) {
        return fmt::format("{}_{:#018x}_{:#018x}", stage, hash, *permutation_id);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage,
                               std::string_view ext, std::optional<u64> permutation_id) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, permutation_id), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::Stage stage,
                                                              u64 permutation_id,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, permutation_id), ext);
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
