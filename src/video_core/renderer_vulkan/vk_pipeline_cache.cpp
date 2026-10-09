// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <chrono>
#include <iterator>
#include <bit>
#include <cstdlib>
#include <ranges>
#include <string_view>

#include "common/hash.h"
#include "common/io_file.h"
#include "common/logging/compilation_timing.h"
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
#include "video_core/buffer_cache/srt_failure_flat_diagnostics.h"
#include "video_core/buffer_cache/srt_failure_flat_provenance.h"
#include "video_core/buffer_cache/srt_failure_flat_snapshot.h"
#include "video_core/buffer_cache/srt_memory_sync.h"
#include "video_core/buffer_cache/srt_session_diagnostics.h"
#include "video_core/buffer_cache/srt_preparation_retry.h"
#include "video_core/buffer_cache/srt_read_failure_diagnostics.h"
#include "video_core/cache_storage.h"
#include "video_core/indirect_draw_user_data.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/async_compilation_completion.h"
#include "video_core/renderer_vulkan/async_graphics_cache_policy.h"
#include "video_core/renderer_vulkan/async_graphics_optimization_diagnostic.h"
#include "video_core/renderer_vulkan/compilation_latency_timer.h"
#include "video_core/renderer_vulkan/fragment_input_mode_diagnostics.h"
#include "video_core/renderer_vulkan/host_compilation_pause.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/scaled_min_max_blend.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

static_assert(VideoCore::SrtSessionDiagnosticWindow::StageCount ==
              static_cast<u32>(Shader::Stage::Compute) + 1);

static VideoCore::SrtSessionDiagnostics& GetSrtSessionDiagnostics() {
    // GetProgram is owner-side; thread-local storage also prevents accidental cross-owner races.
    static thread_local VideoCore::SrtSessionDiagnostics diagnostics{[] {
        const char* enabled = std::getenv("SHADPS4_SRT_SNAPSHOT_TIMING");
        const char* start = std::getenv("SHADPS4_SRT_SNAPSHOT_TIMING_START_FRAME");
        return VideoCore::SrtSnapshotTimingConfig::Parse(enabled != nullptr ? enabled : "",
                                                        start != nullptr ? start : "");
    }()};
    return diagnostics;
}

static u64 SrtSessionMonotonicMicroseconds() noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static void LogSrtSessionDiagnosticWindow(const VideoCore::SrtSessionDiagnosticWindow& row) {
    if (row.truncated) {
        LOG_INFO(Render_Vulkan, "SrtSessionCountsTruncatedV1 rows={} next_presented_frame={}",
                 VideoCore::SrtSessionDiagnostics::MaximumRows, row.presented_frame);
        return;
    }
    static constexpr std::array names{"fs", "vs", "gs", "es", "hs", "ls", "cs"};
    std::string stage_counts;
    for (size_t index = 0; index < row.stages.size(); ++index) {
        const auto& stage = row.stages[index];
        const auto name = names[index];
        fmt::format_to(
            std::back_inserter(stage_counts),
            "{}_sessions={} {}_selections={} {}_retries={} {}_reads={} {}_pages={} {}_hits={} "
            "{}_bytes={} {}_max_pages={} {}_total_us={} ",
            name, stage.sessions, name, stage.selections, name, stage.retries, name, stage.read_calls,
            name, stage.new_pages, name, stage.cache_hits, name, stage.requested_bytes,
            name, stage.max_pages_per_session, name, stage.total_us);
    }
    const auto& total = row.totals;
    LOG_INFO(Render_Vulkan,
             "SrtSessionCountsV1 presented_frame={} begin_us={} end_us={} sessions={} "
             "selections={} retries={} invalid_sessions={} read_calls={} new_pages={} "
             "cache_hits={} requested_bytes={} max_pages_per_session={} total_us={} "
             "max_session_us={} saturated={} {}",
             row.presented_frame, row.begin_us, row.end_us, total.sessions, total.selections,
             total.retries, total.invalid_sessions, total.read_calls, total.new_pages,
             total.cache_hits, total.requested_bytes, total.max_pages_per_session, total.total_us,
             total.max_session_us, row.saturated, stage_counts);
}

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

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(
    Stage stage, LogicalStage l_stage, Shader::SrtMemoryReader reader,
    Shader::DrawUserDataLocations draw_user_data) {
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
        info.vs_info.draw_user_data = l_stage == LogicalStage::Vertex
                                         ? draw_user_data
                                         : Shader::DrawUserDataLocations{};
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
        if (const auto params_vc = AmdGpu::GetParams(regs.vs_program)) {
            gs_info.vs_copy = params_vc->code;
            gs_info.vs_copy_hash = params_vc->hash;
            DumpShader(gs_info.vs_copy, gs_info.vs_copy_hash, Shader::Stage::Vertex, 0,
                       "copy.bin");
        } else {
            gs_info.vs_copy = {};
            gs_info.vs_copy_hash = 0;
        }
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
        .supports_storage_image_multisample =
            instance_.IsShaderStorageImageMultisampleSupported(),
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
    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);

    // WarmUp can materialize serialized graphics/compute pipelines and passes this handle to
    // their Vulkan constructors. Create and publish the driver cache before loading those records;
    // otherwise a cache-enabled startup dereferences the still-null unique handle.
    WarmUp();

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
        .generation = async_generation,
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

static vk::PipelineCreateFlags AsyncGraphicsPipelineCreationFlags() {
    // Launch-time diagnostic, captured once at the first async compile. Never persisted.
    static const auto flags = [] {
        const char* value = std::getenv("SHADPS4_ASYNC_GRAPHICS_DISABLE_OPTIMIZATION");
        const bool enabled = ParseAsyncGraphicsDisableOptimization(value ? value : "");
        const auto creation_flags = AsyncGraphicsDiagnosticFlags(true, enabled);
        if (creation_flags != 0) {
            LOG_WARNING(Render_Vulkan,
                        "Diagnostic: async graphics pipeline optimization disabled; "
                        "creation speedup is not guaranteed and GPU performance may decrease");
        }
        return vk::PipelineCreateFlags{creation_flags};
    }();
    return flags;
}

static bool AsyncGraphicsWorkerNoCacheEnabled() {
    // Launch-time diagnostic, captured once at the first async compile. Passing a null cache to
    // the worker keeps the owner-created cache available for synchronous work while avoiding
    // concurrent access to that cache on drivers that do not implement its synchronization well.
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_ASYNC_GRAPHICS_WORKER_NO_CACHE");
        const bool no_cache = ParseAsyncGraphicsWorkerNoCache(value ? value : "");
        if (no_cache) {
            LOG_WARNING(Render_Vulkan,
                        "Diagnostic: async graphics pipeline worker bypasses the driver pipeline "
                        "cache");
        }
        return no_cache;
    }();
    return enabled;
}

AsyncGraphicsPipelineResult PipelineCache::CompileGraphicsPipelineAsync(
    AsyncGraphicsPipelineJob job) {
    job.RebindCompileInfo();
    GraphicsPipeline::SerializationSupport serialization{};
    auto timing = Common::Log::TraceCompilation([&] {
        return Common::CompilationContext{.phase = Common::CompilationPhase::GraphicsWorker,
                                          .key = std::hash<GraphicsPipelineKey>{}(job.key)};
    });
    const auto worker_pipeline_cache =
        AsyncGraphicsWorkerNoCacheEnabled() ? vk::PipelineCache{} : *pipeline_cache;
    // When selected, the shared cache is created before the worker and remains alive until its
    // queue is joined. The opt-out above passes a null cache to avoid concurrent driver-cache
    // access on an A/B diagnostic run.
    auto pipeline = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, job.key, worker_pipeline_cache, job.compile_infos,
        job.runtime_infos, job.fetch_shader, job.modules, serialization, false,
        AsyncGraphicsPipelineCreationFlags());
    RequireValidCompiledPipeline(pipeline);
    timing.SetOutcome(Common::CompilationOutcome::Complete);
    return {
        .pipeline = std::move(pipeline),
        .generation = job.generation,
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
    if (result.generation != async_generation) {
        LOG_DEBUG(Render_Vulkan,
                  "Discarding stale asynchronous graphics pipeline {:#x} from generation {}",
                  std::hash<GraphicsPipelineKey>{}(completion.key), result.generation);
        return;
    }
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
        *async_graphics_compiler, requested_key,
        [this](auto completion) { PublishAsyncGraphicsPipeline(std::move(completion)); });
}

const GraphicsPipeline* PipelineCache::WaitForAsyncGraphicsPipeline(
    const GraphicsPipelineKey& key) {
    auto timing = Common::Log::TraceCompilation([&] {
        return Common::CompilationContext{.phase = Common::CompilationPhase::GraphicsWait,
                                          .key = std::hash<GraphicsPipelineKey>{}(key)};
    });
    for (;;) {
        if (PublishAsyncGraphicsPipelines(&key)) {
            timing.SetOutcome(Common::CompilationOutcome::Failed);
            return nullptr;
        }
        if (const auto it = graphics_pipelines.find(key); it != graphics_pipelines.end()) {
            timing.SetOutcome(Common::CompilationOutcome::Ready);
            return it->second.get();
        }

        auto completion = async_graphics_compiler->WaitTake();
        if (!completion) {
            timing.SetOutcome(Common::CompilationOutcome::Stopped);
            return nullptr;
        }
        const bool requested_pipeline_failed = completion->key == key && !completion->result;
        PublishAsyncGraphicsPipeline(std::move(*completion));
        if (requested_pipeline_failed) {
            timing.SetOutcome(Common::CompilationOutcome::Failed);
            return nullptr;
        }
    }
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(
    VideoCore::IndirectDrawRegisterLocations register_locations) {
    Shader::DrawUserDataLocations draw_user_data{};
    // Only the conventional physical VS is supported by the new mapping. Preserve the
    // previous disabled behavior for ES/LS/tessellation topologies until separately modeled.
    if (liverpool->regs.stage_enable.raw == AmdGpu::ShaderStageEnable::VgtStages::Vs) {
        const auto resolved = VideoCore::ResolveVertexDrawUserData(register_locations);
        if (!resolved) {
            LOG_ERROR(Render_Vulkan, "Unsupported indirect vertex user-data destinations");
            return nullptr;
        }
        draw_user_data = *resolved;
    }
    PublishAsyncGraphicsPipelines();
    if (!RefreshGraphicsKey(draw_user_data)) {
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
        {
            auto timing = Common::Log::TraceCompilation([&] {
                return Common::CompilationContext{.phase = Common::CompilationPhase::OwnerCompute,
                                                  .key = pipeline_hash,
                                                  .stage = static_cast<int>(Shader::Stage::Compute),
                                                  .logical =
                                                      static_cast<int>(LogicalStage::Compute)};
            });
            it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                           *pipeline_cache, compute_key, *infos[0],
                                                           modules[0], sdata, false);
            timing.SetOutcome(Common::CompilationOutcome::Complete);
        }
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey(Shader::DrawUserDataLocations draw_user_data) {
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
        color_buffer.blend_self_scale = IsSelfScaledMinMaxBlend(
            regs.blend_control[cb], col_buf.info.blend_bypass);
    }

    // Compile and bind shader stages
    if (!RefreshGraphicsStages(draw_user_data)) {
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

bool PipelineCache::RefreshGraphicsStages(Shader::DrawUserDataLocations draw_user_data) {
    auto snapshot_reuse_scope = buffer_cache.BeginSrtOverlayReuseSelection();
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
        if (!params) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting {} shader stage without valid binary metadata", stage_in);
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader_;
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_,
                 key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, *params, binding, draw_user_data);
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
    if (!cs_params) {
        infos[0] = nullptr;
        modules[0] = nullptr;
        fetch_shader.reset();
        return false;
    }
    std::tie(infos[0], modules[0], fetch_shader, compute_key.value) =
        GetProgram(Shader::Stage::Compute, LogicalStage::Compute, *cs_params, binding);
    return infos[0] != nullptr && static_cast<bool>(modules[0]);
}

std::optional<Shader::IR::Program> PipelineCache::TranslateModule(Shader::Info& info,
                                                                  Shader::RuntimeInfo& runtime_info,
                                                                  const std::span<const u32>& code,
                                                                  Shader::SrtMemoryReader reader) {
    auto timing = Common::Log::TraceCompilation([&] {
        return Common::CompilationContext{.phase = Common::CompilationPhase::OwnerTranslate,
                                          .key = info.pgm_hash,
                                          .stage = static_cast<int>(info.stage),
                                          .logical = static_cast<int>(info.l_stage)};
    });
    auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile, reader);
    if (!reader.Valid() || info.translation_failed) {
        timing.SetOutcome(Common::CompilationOutcome::Rejected);
        return std::nullopt;
    }
    timing.SetOutcome(Common::CompilationOutcome::Complete);
    return ir_program;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info,
                                              const Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code,
                                              size_t perm_idx, const std::vector<u32>& spv) {
    auto timing = Common::Log::TraceCompilation([&] {
        return Common::CompilationContext{.phase = Common::CompilationPhase::OwnerModule,
                                          .key = info.pgm_hash,
                                          .stage = static_cast<int>(info.stage),
                                          .logical = static_cast<int>(info.l_stage),
                                          .perm = static_cast<std::int64_t>(perm_idx)};
    });
    ScopedHostCompilationGuestPause pause{DebugState,
                                          !EmulatorSettings.IsAsyncGraphicsPipelineCompilation()};
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.stage, perm_idx, "bin");
    DumpShader(spv, info.pgm_hash, info.stage, perm_idx, "spv");

    if (EmulatorSettings.IsDumpShaders() && runtime_info.stage == Shader::Stage::Fragment) {
        const auto modes = SummarizeFragmentInputModes(
            runtime_info.fs_info.num_inputs, std::span{runtime_info.fs_info.inputs});
        LOG_INFO(Render_Vulkan,
                 "ShaderFragmentInputsV1 shader={:#x} permutation={} num_inputs={} "
                 "captured_inputs={} use_default_mask={:#010x} flat_mask={:#010x}",
                 info.pgm_hash, perm_idx, modes.declared_count, modes.captured_count,
                 modes.default_mask, modes.flat_mask);
    }

    std::optional<vk::ShaderModule> compiled_module;
    auto patch = GetShaderPatch(info.pgm_hash, info.stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.stage, info.pgm_hash);
        compiled_module = TryCompileSPV(*patch, instance.GetDevice());
    } else {
        // Guest-generated SPIR-V is an external result of the recompiler. A malformed module
        // must be rejected as a cache miss instead of entering CompileSPV's assert-on-result path.
        compiled_module = TryCompileSPV(spv, instance.GetDevice());
    }
    if (!compiled_module) {
        LOG_WARNING(Render_Vulkan,
                    "Rejecting {} shader {:#x}: Vulkan did not admit the generated SPIR-V {}",
                    info.stage, info.pgm_hash, perm_idx != 0 ? "permutation" : "module");
        timing.SetOutcome(Common::CompilationOutcome::Rejected);
        return {};
    }
    const vk::ShaderModule module = *compiled_module;

    const auto name = GetShaderName(info.stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    timing.SetOutcome(Common::CompilationOutcome::Complete);
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(Stage stage, LogicalStage l_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding,
                                                Shader::DrawUserDataLocations draw_user_data) {
    const bool capture_exact_flat_target =
        VideoCore::GetSrtFailureFlatDiagnostics().Matches(params.hash);
    auto& session_diagnostics = GetSrtSessionDiagnostics();
    u32 diagnostic_attempt{};
    return VideoCore::RetrySrtPreparation(
        binding,
        [&](Shader::Backend::Bindings& binding, VideoCore::SrtMemoryReadFailure& attempt_failure,
            bool& attempt_translation_rejected) -> Result {
            if (diagnostic_attempt != 0) {
                buffer_cache.InvalidateSrtOverlayReuse();
            }
            auto session_diagnostic = session_diagnostics.BeginAttempt(
                DebugState.GetFrameNum(), static_cast<u32>(stage), ++diagnostic_attempt,
                SrtSessionMonotonicMicroseconds, LogSrtSessionDiagnosticWindow);
            SrtReadSession srt_read_session{page_manager, buffer_cache,
                                            VideoCore::GetSrtReadFailureDiagnostics().Enabled() ||
                                                capture_exact_flat_target,
                                            session_diagnostic.has_value()};
            SCOPE_EXIT {
                if (session_diagnostic) {
                    session_diagnostics.CompleteSession(
                        *session_diagnostic, srt_read_session.DiagnosticCounts(),
                        !srt_read_session.Valid(), SrtSessionMonotonicMicroseconds);
                }
            };
            const auto reader = MakeSrtMemoryReader(srt_read_session);
            auto runtime_info = BuildRuntimeInfo(stage, l_stage, reader, draw_user_data);
            const auto cache_draw_user_data =
                stage == Stage::Vertex && l_stage == LogicalStage::Vertex
                    ? runtime_info.vs_info.draw_user_data
                    : Shader::DrawUserDataLocations{};
            const auto cache_key =
                Shader::MakeProgramCacheKey(params.hash, stage, l_stage, cache_draw_user_data);

            const auto emit_source_read_records = [&](unsigned row, std::string_view phase) {
                const auto source_reads = srt_read_session.SourceReads();
                for (u32 source_index = 0;
                     source_index < source_reads.size(); ++source_index) {
                    const auto& source = source_reads[source_index];
                    const auto& bytes = source.first_bytes;
                    LOG_WARNING(
                        Render_Vulkan,
                        "SrtSourceReadV1 row={} source_index={} stage={} shader={:#x} "
                        "phase={} request_known={} request_address={:#x} "
                        "request_size={} output_null={} failure={} route={} "
                        "snapshot_address={:#x} snapshot_size={} source_bytes_known={} "
                        "diagnostic_snapshot_id={} bytes_captured={} first_bytes=[{:02x} {:02x} {:02x} {:02x} "
                        "{:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} "
                        "{:02x} {:02x} {:02x} {:02x}]",
                        row, source_index, stage, params.hash, phase, source.request_known,
                        source.request_address, source.request_size, source.output_null,
                        VideoCore::SrtMemoryReadFailureName(source.failure),
                        VideoCore::SrtSnapshotCompositionRouteName(source.route),
                        source.snapshot_address, source.snapshot_size, source.source_bytes_known,
                        source.diagnostic_snapshot_id, source.bytes_captured, bytes[0], bytes[1], bytes[2], bytes[3], bytes[4],
                        bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11],
                        bytes[12], bytes[13], bytes[14], bytes[15]);
                }
            };

            const auto fail_snapshot = [&](
                                           std::string_view phase,
                                           bool translation_rejected = false,
                                           const Shader::Info* failed_info = nullptr,
                                           Shader::StageSpecializationFailureReason
                                               specialization_failure =
                                                   Shader::StageSpecializationFailureReason::None) {
                buffer_cache.InvalidateSrtOverlayReuse();
                attempt_failure = srt_read_session.LastFailure();
                attempt_translation_rejected = translation_rejected;
                LOG_WARNING(Render_Vulkan,
                            "Preparing {} shader {:#x}: resource preparation failed "
                            "phase={} read={} translationRejected={} specializationFailure={}",
                            stage, params.hash, phase,
                            VideoCore::SrtMemoryReadFailureName(srt_read_session.LastFailure()),
                            translation_rejected,
                            Shader::StageSpecializationFailureReasonName(specialization_failure));
                VideoCore::GetSrtReadFailureDiagnostics().Emit([&](unsigned row) {
                    const auto request = srt_read_session.FirstFailureRequest();
                    LOG_WARNING(Render_Vulkan,
                                "SrtReadFailureV1 row={} stage={} shader={:#x} phase={} read={} "
                                 "translationRejected={} specializationFailure={} request_known={} "
                                 "address={:#x} size={} output_null={} invalid_reasons={}",
                                 row, stage, params.hash, phase,
                                 VideoCore::SrtMemoryReadFailureName(attempt_failure),
                                 translation_rejected,
                                 Shader::StageSpecializationFailureReasonName(specialization_failure),
                                 request.known, request.address, request.size, request.output_null,
                                 request.invalid_reasons);
                    if (failed_info) {
                        const auto flat =
                            VideoCore::CaptureSrtFailureFlatSnapshot(failed_info->flattened_ud_buf);
                        const auto& words = flat.words;
                        LOG_WARNING(
                            Render_Vulkan,
                            "SrtFailureFlatV1 row={} stage={} shader={:#x} phase={} "
                            "flat_total={} flat_copied={} full={} captured_flat=[{:08x} {:08x} "
                            "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} "
                            "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} "
                            "{:08x} {:08x}]",
                            row, stage, params.hash, phase, flat.total_dwords, flat.copied_dwords,
                            flat.has_first_walker_prefix, words[0], words[1], words[2], words[3],
                            words[4], words[5], words[6], words[7], words[8], words[9], words[10],
                            words[11], words[12], words[13], words[14], words[15], words[16],
                            words[17], words[18], words[19], words[20], words[21]);
                    }
                });
                if (failed_info) {
                    VideoCore::GetSrtFailureFlatDiagnostics().EmitFailure(
                        params.hash, [&](unsigned row) {
                        const auto request = srt_read_session.FirstFailureRequest();
                        const auto flat =
                            VideoCore::CaptureSrtFailureFlatSnapshot(failed_info->flattened_ud_buf);
                        const auto& words = flat.words;
                        LOG_WARNING(
                            Render_Vulkan,
                            "SrtFailureFlatTargetV1 row={} stage={} shader={:#x} phase={} "
                            "request_known={} address={:#x} size={} invalid_reasons={} "
                            "flat_total={} flat_copied={} full={} captured_flat=[{:08x} {:08x} "
                            "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} "
                            "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} "
                            "{:08x} {:08x}]",
                            row, stage, params.hash, phase, request.known, request.address,
                            request.size, request.invalid_reasons, flat.total_dwords,
                            flat.copied_dwords, flat.has_first_walker_prefix, words[0], words[1],
                            words[2], words[3], words[4], words[5], words[6], words[7], words[8],
                            words[9], words[10], words[11], words[12], words[13], words[14],
                            words[15], words[16], words[17], words[18], words[19], words[20],
                            words[21]);
                        if (const auto provenance = VideoCore::CaptureSrtFailureFlatProvenance(
                                params.hash, phase, flat, request, attempt_failure)) {
                            LOG_WARNING(
                                Render_Vulkan,
                                "SrtFailureFlatProvenanceV1 row={} stage={} shader={:#x} "
                                "phase={} low_flat_index={} high_flat_index={} raw_low={:08x} "
                                "raw_high={:08x} raw_address={:#x} masked_address={:#x} "
                                "pointer_size={} invalid_reasons={} pointer_failure={}",
                                row, stage, provenance->shader_hash, provenance->phase,
                                provenance->low_flat_index, provenance->high_flat_index,
                                provenance->raw_low, provenance->raw_high, provenance->raw_address,
                                provenance->masked_address, provenance->pointer_size,
                                provenance->invalid_reasons,
                                 VideoCore::SrtMemoryReadFailureName(provenance->pointer_failure));
                        }
                        emit_source_read_records(row, phase);
                        const auto& composition = srt_read_session.CompositionReceipt();
                        const VideoCore::SrtSnapshotCompositionReceipt missing_receipt{};
                        const auto& receipt = composition ? *composition : missing_receipt;
                        const auto cpu_words =
                            std::bit_cast<std::array<u64, 4>>(receipt.cpu_sample);
                        const auto composed_words =
                            std::bit_cast<std::array<u64, 4>>(receipt.composed_sample);
                        LOG_WARNING(
                            Render_Vulkan,
                            "SrtRootCompositionTargetV1 row={} stage={} shader={:#x} phase={} "
                            "receipt_known={} available={} route={} snapshot={:#x} "
                            "snapshot_size={} focus={:#x} focus_size={} sample_size={} "
                            "diagnostic_snapshot_id={} "
                            "sample_diff_bytes={} first_sample_diff_known={} "
                            "first_sample_diff={} last_sample_diff_known={} last_sample_diff={} "
                            "overlay_count={} overlay_bytes={} cpu64=[{:016x} {:016x} {:016x} "
                            "{:016x}] composed64=[{:016x} {:016x} {:016x} {:016x}]",
                            row, stage, params.hash, phase, composition.has_value(),
                            composition && receipt.route !=
                                               VideoCore::SrtSnapshotCompositionRoute::Unavailable,
                            VideoCore::SrtSnapshotCompositionRouteName(receipt.route),
                            receipt.snapshot_address, receipt.snapshot_size, receipt.focus_address,
                            receipt.focus_size, receipt.sample_size, receipt.diagnostic_snapshot_id,
                            receipt.sample_differing_byte_count,
                            receipt.first_differing_focus_offset.has_value(),
                            receipt.first_differing_focus_offset.value_or(0),
                            receipt.last_differing_focus_offset.has_value(),
                            receipt.last_differing_focus_offset.value_or(0), receipt.overlay_count,
                            receipt.overlay_bytes, cpu_words[0], cpu_words[1], cpu_words[2],
                            cpu_words[3], composed_words[0], composed_words[1], composed_words[2],
                            composed_words[3]);
                        const auto& cpu_focus = receipt.cpu_write_focus;
                        LOG_WARNING(
                            Render_Vulkan,
                            "SrtCpuWriteFocusV1 row={} stage={} shader={:#x} phase={} available={} "
                            "word_address={:#x} word_count={} snapshot_mask={:#x} forced_mask={:#x} "
                            "changed_mask={:#x} mapping_overlap_mask={:#x} "
                            "before=[{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}] "
                            "captured=[{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}]",
                            row, stage, params.hash, phase, cpu_focus.available,
                            cpu_focus.word_address, cpu_focus.word_count, cpu_focus.snapshot_mask,
                            cpu_focus.forced_word_mask, cpu_focus.changed_word_mask,
                            cpu_focus.mapping_overlap_mask, cpu_focus.before[0], cpu_focus.before[1],
                            cpu_focus.before[2], cpu_focus.before[3], cpu_focus.before[4],
                            cpu_focus.before[5], cpu_focus.before[6], cpu_focus.before[7],
                            cpu_focus.captured[0], cpu_focus.captured[1], cpu_focus.captured[2],
                            cpu_focus.captured[3], cpu_focus.captured[4], cpu_focus.captured[5],
                            cpu_focus.captured[6], cpu_focus.captured[7]);
                        for (u32 index = 0; index < receipt.focus_owner_count &&
                                            index < receipt.focus_owners.size(); ++index) {
                            const auto& owner = receipt.focus_owners[index];
                            LOG_WARNING(
                                Render_Vulkan,
                                "SrtFocusOwnerV1 row={} owner_index={} owner_total={} stage={} "
                                "shader={:#x} phase={} source={:#x}:{:#x} source_offset={:#x} "
                                "output_offset={:#x} size={} buffer_id={} mapped_buffer_id={} "
                                "owner_generation={}",
                                row, index, receipt.focus_owner_total, stage, params.hash, phase,
                                owner.source_address, owner.source_size, owner.source_offset,
                                owner.output_offset, owner.size, owner.source_buffer_id,
                                owner.mapped_buffer_id, owner.owner_generation);
                        }
                        });
                }
                return Result{};
            };
            const auto emit_success_source_reads = [&](std::string_view phase) {
                if (!capture_exact_flat_target || srt_read_session.SourceReads().empty()) {
                    return;
                }
                VideoCore::GetSrtFailureFlatDiagnostics().Emit(params.hash, [&](unsigned row) {
                    emit_source_read_records(row, phase);
                });
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
                    const auto result =
                        fail_snapshot("initial_translation", translation_rejected, &program->info);
                    program_cache.erase(cache_key);
                    return result;
                }

                auto admitted = Shader::TryCreateStageSpecializationDetailed(
                    program->info, runtime_info, profile, start, reader);
                auto admitted_spec = std::move(admitted.specialization);
                if (!admitted_spec || !reader.Valid()) {
                    const auto failure = admitted_spec
                                             ? (reader.Valid()
                                                    ? Shader::StageSpecializationFailureReason::None
                                                    : Shader::StageSpecializationFailureReason::
                                                          ReaderInvalid)
                                             : admitted.failure_reason;
                    program_cache.erase(cache_key);
                    return fail_snapshot("initial_specialization", !admitted_spec, nullptr,
                                         failure);
                }
                auto& spec = *admitted_spec;
                const auto permutation_id =
                    Shader::ComputeShaderPermutationIdentity(params.hash, l_stage, spec);
                auto compiled_binding = binding;
                auto spv = [&] {
                    auto timing = Common::Log::TraceCompilation([&] {
                        return Common::CompilationContext{.phase =
                                                              Common::CompilationPhase::OwnerSpirv,
                                                          .key = params.hash,
                                                          .stage = static_cast<int>(stage),
                                                          .logical = static_cast<int>(l_stage),
                                                          .perm = 0};
                     });
                     auto output = Shader::Backend::SPIRV::EmitSPIRV(
                         profile, runtime_info, *ir_program, compiled_binding, reader);
                     timing.SetOutcome(reader.Valid() && !program->info.translation_failed
                                           ? Common::CompilationOutcome::Complete
                                           : Common::CompilationOutcome::Rejected);
                     return output;
                 }();
                 if (program->info.translation_failed || !reader.Valid()) {
                     const bool translation_rejected = program->info.translation_failed;
                     const auto result = fail_snapshot(
                         "initial_spirv", translation_rejected,
                         translation_rejected ? &program->info : nullptr);
                     program_cache.erase(cache_key);
                     return result;
                 }
                 if (!capture_vertex_buffers(program->info, spec.fetch_shader_data)) {
                     program_cache.erase(cache_key);
                     return fail_snapshot("initial_vertex_capture");
                }
                if (!seal_snapshot()) {
                    program_cache.erase(cache_key);
                    return fail_snapshot("initial_seal");
                }

                const auto module = CompileModule(program->info, runtime_info, params.code, 0, spv);
                if (!module) {
                    program_cache.erase(cache_key);
                    return fail_snapshot("initial_module");
                }
                if (RegisterShaderMeta(program->info, spec, permutation_id)) {
                    RegisterShaderBinary(std::move(spv), permutation_id);
                }
                auto module_info = program->info;
                program->AddPermut(module, std::move(spec), std::move(module_info), permutation_id);
                binding = compiled_binding;
                emit_success_source_reads("initial_success");
                return std::make_tuple(program->modules[0].info.get(), module,
                                       program->modules[0].spec.fetch_shader_data, permutation_id);
            }

            auto& program = it_pgm.value();
            auto aggregate_snapshot = program->info;
            aggregate_snapshot.pgm_base = params.Base();
            aggregate_snapshot.user_data = params.user_data;
            if (capture_exact_flat_target) {
                srt_read_session.ArmCompositionReceipt();
            }
            if (!aggregate_snapshot.RefreshFlatBuf(reader)) {
                return fail_snapshot("aggregate_refresh", false, &aggregate_snapshot);
            }
            const auto start = binding;
            auto probe_admission = Shader::TryCreateStageSpecializationDetailed(
                aggregate_snapshot, runtime_info, profile, start, reader);
            auto probe_spec = std::move(probe_admission.specialization);
            if (!reader.Valid()) {
                return fail_snapshot(
                    "probe_specialization", false, &aggregate_snapshot,
                    probe_spec ? Shader::StageSpecializationFailureReason::ReaderInvalid
                               : probe_admission.failure_reason);
            }

            std::string_view selected_failure = "selected_missing_info";
            Shader::StageSpecializationFailureReason selected_specialization_failure =
                Shader::StageSpecializationFailureReason::None;
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
                auto selected_admission = Shader::TryCreateStageSpecializationDetailed(
                    selected_info, runtime_info, profile, start, reader);
                auto selected_spec = std::move(selected_admission.specialization);
                if (!selected_spec || !reader.Valid()) {
                    selected_failure = "selected_specialization";
                    selected_specialization_failure =
                        selected_spec
                            ? Shader::StageSpecializationFailureReason::ReaderInvalid
                            : selected_admission.failure_reason;
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
            if (const auto probe = probe_spec ? std::ranges::find(program->modules, *probe_spec,
                                                                  &Program::Module::spec)
                                              : program->modules.end();
                probe != program->modules.end()) {
                auto selected_info = validate_selected(*probe);
                if (selected_info) {
                    auto selected_binding = binding;
                    if (!selected_info->AddBindings(selected_binding)) {
                        return fail_snapshot("cached_bindings", false, &*selected_info);
                    }
                    if (!seal_snapshot()) {
                        return fail_snapshot("cached_seal");
                    }
                    *probe->info = std::move(*selected_info);
                    binding = selected_binding;
                    emit_success_source_reads("cached_success");
                    return std::make_tuple(probe->info.get(), probe->module,
                                           probe->spec.fetch_shader_data, probe->permutation_id);
                }
            }
            if (!reader.Valid()) {
                return fail_snapshot(selected_failure, false, nullptr,
                                     selected_specialization_failure);
            }

            const size_t perm_idx = program->modules.size();
            auto new_info = Shader::Info(stage, l_stage, params);
            auto ir_program = TranslateModule(new_info, runtime_info, params.code, reader);
            if (!ir_program) {
                return fail_snapshot("permutation_translation", new_info.translation_failed);
            }
            auto admitted = Shader::TryCreateStageSpecializationDetailed(
                new_info, runtime_info, profile, start, reader);
            auto admitted_spec = std::move(admitted.specialization);
            if (!admitted_spec || !reader.Valid()) {
                const auto failure = admitted_spec
                                         ? Shader::StageSpecializationFailureReason::ReaderInvalid
                                         : admitted.failure_reason;
                return fail_snapshot("permutation_specialization", !admitted_spec, nullptr,
                                     failure);
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
                    return fail_snapshot(selected_failure, false, nullptr,
                                         selected_specialization_failure);
                }
                auto selected_binding = binding;
                if (!selected_info->AddBindings(selected_binding)) {
                    return fail_snapshot("reuse_bindings", false, &*selected_info);
                }
                if (!seal_snapshot()) {
                    return fail_snapshot("reuse_seal");
                }
                *authoritative.module->info = std::move(*selected_info);
                binding = selected_binding;
                emit_success_source_reads("reuse_success");
                return std::make_tuple(authoritative.module->info.get(),
                                       authoritative.module->module,
                                       authoritative.module->spec.fetch_shader_data,
                                       authoritative.module->permutation_id);
            }

            auto compiled_binding = binding;
            auto spv = [&] {
                auto timing = Common::Log::TraceCompilation([&] {
                    return Common::CompilationContext{.phase = Common::CompilationPhase::OwnerSpirv,
                                                      .key = params.hash,
                                                      .stage = static_cast<int>(stage),
                                                      .logical = static_cast<int>(l_stage),
                                                      .perm = static_cast<std::int64_t>(perm_idx)};
                 });
                 auto output = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, *ir_program,
                                                                 compiled_binding, reader);
                 timing.SetOutcome(reader.Valid() && !new_info.translation_failed
                                       ? Common::CompilationOutcome::Complete
                                       : Common::CompilationOutcome::Rejected);
                 return output;
             }();
             if (new_info.translation_failed || !reader.Valid()) {
                 const bool translation_rejected = new_info.translation_failed;
                 return fail_snapshot("permutation_spirv", translation_rejected,
                                      translation_rejected ? &new_info : nullptr);
             }
             if (!capture_vertex_buffers(new_info, authoritative_spec.fetch_shader_data)) {
                 return fail_snapshot("permutation_vertex_capture");
             }
            if (!seal_snapshot()) {
                return fail_snapshot("permutation_seal");
            }

            const auto module = CompileModule(new_info, runtime_info, params.code, perm_idx, spv);
            if (!module) {
                return fail_snapshot("permutation_module");
            }
            if (RegisterShaderMeta(new_info, authoritative_spec, permutation_id)) {
                RegisterShaderBinary(std::move(spv), permutation_id);
            }
            program->AddPermut(module, std::move(authoritative_spec), std::move(new_info),
                               permutation_id);
            binding = compiled_binding;
            const auto& selected = program->modules.back();
            emit_success_source_reads("permutation_success");
            return std::make_tuple(selected.info.get(), selected.module,
                                   selected.spec.fetch_shader_data, selected.permutation_id);
        },
        [](const Result& result) { return std::get<0>(result) && std::get<1>(result); });
}
std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    if (async_graphics_compiler) {
        async_graphics_compiler->WaitIdle();
    }
    ++async_generation;
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
                if (auto it = graphics_pipelines.find(graphics_key);
                    it != graphics_pipelines.end()) {
                    // A recorded command buffer may still reference this pipeline. Remove it
                    // from the lookup map immediately so a replacement can be built, but keep
                    // the Vulkan owner alive until the scheduler's current GPU tick retires.
                    auto retired = std::move(it.value());
                    graphics_pipelines.erase(it);
                    scheduler.DeferOperation([retired = std::move(retired)]() mutable {});
                }
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                if (auto it = compute_pipelines.find(compute_key);
                    it != compute_pipelines.end()) {
                    auto retired = std::move(it.value());
                    compute_pipelines.erase(it);
                    scheduler.DeferOperation([retired = std::move(retired)]() mutable {});
                }
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
