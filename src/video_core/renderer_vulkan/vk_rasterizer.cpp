// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <cstring>
#include <optional>

#include "common/debug.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/addr64_pointer_residency.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_barrier_policy.h"
#include "video_core/buffer_cache/buffer_residency.h"
#include "video_core/buffer_cache/fault_range.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderer_vulkan/depth_stencil_policy.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_diagnostic_checkpoint.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_compute_dispatch_policy.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/depth_association.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/image_usage_policy.h"
#include "video_core/texture_cache/sampler_usage_policy.h"
#include "video_core/texture_cache/texture_cache.h"

#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace Vulkan {

// Per-command GPU markers are intentionally disabled outside narrowly scoped diagnostics. Shadow
// can execute millions of draws during its intro, making unconditional markers materially alter
// memory pressure and GPU scheduling. Device-fault reporting remains available in the scheduler.
static constexpr bool EnablePerCommandGpuDiagnostics = false;

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, page_manager{this},
      buffer_cache{instance, scheduler, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, liverpool_, buffer_cache, page_manager},
      occlusion_queries{instance, scheduler},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool} {
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
}

Rasterizer::~Rasterizer() = default;

void Rasterizer::CpSync() {
    scheduler.EndRendering();
    auto cmdbuf = scheduler.CommandBuffer();

    const vk::MemoryBarrier ib_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eDrawIndirect,
                           vk::DependencyFlagBits::eByRegion, ib_barrier, {}, {});
}

bool Rasterizer::FilterDraw() {
    const auto& regs = liverpool->regs;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->regs;
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    boost::container::static_vector<VideoCore::ImageId, AmdGpu::NUM_COLOR_BUFFERS + 1>
        target_images;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->last_cb_extent[cb];
        std::construct_at(&desc, col_buf, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
        image.usage.render_target = 1u;
        target_images.push_back(image_id);
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->last_db_extent;
        auto& [image_id, desc] = db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
        image.usage.depth_target = 1u;
        target_images.push_back(image_id);
    } else {
        db_desc.first = {};
    }

    std::ranges::sort(target_images, {}, &VideoCore::ImageId::index);
    const auto unique = std::ranges::unique(target_images);
    target_images.erase(unique.begin(), unique.end());
    const u32 scale_percent = EmulatorSettings.GetInternalResolutionScale();
    const u32 display_width = EmulatorSettings.GetInternalScreenWidth();
    const u32 display_height = EmulatorSettings.GetInternalScreenHeight();
    std::array<bool, AmdGpu::NUM_COLOR_BUFFERS + 1> eligible{};
    for (u32 index = 0; index < target_images.size(); ++index) {
        eligible[index] = texture_cache.GetImage(target_images[index])
                              .CanHostScale(scale_percent, display_width, display_height);
    }
    prepared_host_scale = VideoCore::ResolveHostAttachmentScale(
        scale_percent, std::span<const bool>{eligible.data(), target_images.size()});
    for (const auto image_id : target_images) {
        ASSERT(texture_cache.GetImage(image_id).SetHostScale(prepared_host_scale * 100,
                                                             display_width, display_height));
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::Info& info,
    const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->last_cb_extent[0]);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    image.Clear(clear_value, desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const auto& regs = liverpool->regs;
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }
    const auto& vs = pipeline->GetStage(Shader::LogicalStage::Vertex);
    const auto* fs = pipeline->TryGetStage(Shader::LogicalStage::Fragment);
    const u64 diagnostic_index_address =
        is_indexed ? regs.index_base_address.Address<VAddr>() +
                         index_offset * (regs.index_buffer_type.index_type ==
                                                 AmdGpu::IndexType::Index16
                                             ? sizeof(u16)
                                             : sizeof(u32))
                   : 0;
    DiagnosticCheckpointMarker draw_marker{
        .kind = DiagnosticCheckpointKind::Graphics,
        .first_hash = vs.pgm_hash,
        .second_hash = fs ? fs->pgm_hash : 0,
        .runtime = {regs.num_indices, regs.num_instances.NumInstances(), is_indexed, 0,
                    static_cast<u32>(diagnostic_index_address),
                    static_cast<u32>(diagnostic_index_address >> 32)},
    };
    const DiagnosticCheckpointMarker* checkpoint_marker{};
    if (EnablePerCommandGpuDiagnostics && instance.SupportsDiagnosticCheckpoints()) {
        checkpoint_marker = MakeDiagnosticCheckpoint(DiagnosticCheckpointKind::Graphics,
                                                     draw_marker.first_hash,
                                                     draw_marker.second_hash);
        RecordDiagnosticCommand(draw_marker);
    }
    const auto record_draw_phase = [&](u32 phase) {
        if (!checkpoint_marker) {
            return;
        }
        draw_marker.runtime[3] = phase;
        const auto diagnostic_cmdbuf = scheduler.CommandBuffer();
        VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdSetCheckpointNV(
            static_cast<VkCommandBuffer>(diagnostic_cmdbuf), checkpoint_marker);
        RecordDiagnosticGpuMarker(diagnostic_cmdbuf, draw_marker);
    };
    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    VideoCore::PrepareDrawBuffersThenBindDynamicVertexState(
        [] {},
        [&] {
            if (is_indexed) {
                buffer_cache.BindIndexBuffer(index_offset, buffer_barriers);
            }
        },
        [&] { buffer_cache.BindVertexBuffers(*pipeline, buffer_barriers); });

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    const auto zpass_query = occlusion_queries.PrepareDrawQuery();
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Graphics, pipeline->Handle());
    RecordSparseDiagnosticCheckpoint(false, vs.pgm_hash, fs ? fs->pgm_hash : 0);
    occlusion_queries.BeginDraw(cmdbuf, zpass_query);
    record_draw_phase(0xB3);
    if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    occlusion_queries.EndDraw(cmdbuf, zpass_query);
    record_draw_phase(0xA3);
    DebugState.IncDrawCall();

    ResetBindings();
    RecordGpuWork();
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }
    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    VideoCore::Buffer* buffer{};
    u32 base{};
    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    VideoCore::PrepareDrawBuffersThenBindDynamicVertexState(
        [&] {
            std::tie(buffer, base) =
                buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);
            if (count_address != 0) {
                std::tie(count_buffer, count_base) =
                    buffer_cache.ObtainBuffer(count_address, 4, false);
            }
            if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                                  vk::PipelineStageFlagBits2::eDrawIndirect)) {
                buffer_barriers.emplace_back(*barrier);
            }
            if (count_buffer) {
                if (auto barrier = count_buffer->GetBarrier(
                        vk::AccessFlagBits2::eIndirectCommandRead,
                        vk::PipelineStageFlagBits2::eDrawIndirect)) {
                    buffer_barriers.emplace_back(*barrier);
                }
            }
        },
        [&] {
            if (is_indexed) {
                buffer_cache.BindIndexBuffer(0, buffer_barriers);
            }
        },
        [&] { buffer_cache.BindVertexBuffers(*pipeline, buffer_barriers); });

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    const auto zpass_query = occlusion_queries.PrepareDrawQuery();
    scheduler.BeginRendering(state);

    // We can safely ignore both SGPR UD indices and results of fetch shader parsing, as vertex and
    // instance offsets will be automatically applied by Vulkan from indirect args buffer.

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Graphics, pipeline->Handle());
    occlusion_queries.BeginDraw(cmdbuf, zpass_query);
    const auto& vs = pipeline->GetStage(Shader::LogicalStage::Vertex);
    const auto* fs = pipeline->TryGetStage(Shader::LogicalStage::Fragment);
    RecordSparseDiagnosticCheckpoint(false, vs.pgm_hash, fs ? fs->pgm_hash : 0);
    DiagnosticCheckpointMarker draw_marker{
        .kind = DiagnosticCheckpointKind::Graphics,
        .first_hash = vs.pgm_hash,
        .second_hash = fs ? fs->pgm_hash : 0,
        .runtime = {max_count, stride, is_indexed, 0xB3},
    };
    const DiagnosticCheckpointMarker* checkpoint_marker{};
    if (EnablePerCommandGpuDiagnostics && instance.SupportsDiagnosticCheckpoints()) {
        checkpoint_marker = MakeDiagnosticCheckpoint(DiagnosticCheckpointKind::Graphics,
                                                     draw_marker.first_hash,
                                                     draw_marker.second_hash);
        RecordDiagnosticCommand(draw_marker);
        VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdSetCheckpointNV(
            static_cast<VkCommandBuffer>(cmdbuf), checkpoint_marker);
        RecordDiagnosticGpuMarker(cmdbuf, draw_marker);
    }
    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_base, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_base,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    }
    if (checkpoint_marker) {
        draw_marker.runtime[3] = 0xA3;
        RecordDiagnosticGpuMarker(cmdbuf, draw_marker);
    }
    occlusion_queries.EndDraw(cmdbuf, zpass_query);
    ResetBindings();
    RecordGpuWork();
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;

    const auto& cs_program = liverpool->GetCsRegs();
    const auto& max_group_count = instance.GetPhysicalDeviceLimits().maxComputeWorkGroupCount;
    if (!IsComputeDispatchWithinHostLimits(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z,
                                           max_group_count)) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping invalid direct compute dispatch {}x{}x{}; host limit is {}x{}x{}",
                    cs_program.dim_x, cs_program.dim_y, cs_program.dim_z, max_group_count[0],
                    max_group_count[1], max_group_count[2]);
        return;
    }

    scheduler.PopPendingOperations();

    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    // TEMP Shadow diagnostic: capture first-use pointer metadata for scene-list shaders.
    static bool logged_scene_list_build = false;
    static bool logged_scene_list_reduce = false;
    static bool logged_shadow_lighting_a = false;
    static bool logged_shadow_lighting_b = false;
    static std::array<bool, 7> logged_shadow_intro_stall{};
    constexpr std::array<u64, 7> shadow_intro_stall_hashes{
        0x73b8c213, 0x8ad8f333, 0xd035685b, 0x99288157, 0xbd1957a9, 0x181538e7,
        0x87cf9b8a};
    bool* intro_stall_log{};
    for (u32 index = 0; index < shadow_intro_stall_hashes.size(); ++index) {
        if (cs.pgm_hash == shadow_intro_stall_hashes[index]) {
            intro_stall_log = &logged_shadow_intro_stall[index];
            break;
        }
    }
    bool* logged = cs.pgm_hash == 0x79704cba ? &logged_scene_list_build
                   : cs.pgm_hash == 0x729c541e ? &logged_scene_list_reduce
                   : cs.pgm_hash == 0xa402ddac ? &logged_shadow_lighting_a
                   : cs.pgm_hash == 0x0654a5a1 ? &logged_shadow_lighting_b
                                              : intro_stall_log;
    if (logged && !*logged) {
        *logged = true;
        LOG_INFO(Render_Vulkan,
                 "TEMP Shadow compute {:#x}: dma={} readconst={} roots={} overflow={} "
                 "read_roots={} read_overflow={} flat_dw={}",
                 cs.pgm_hash, cs.uses_dma, static_cast<u32>(cs.readconst_types),
                 cs.num_addr64_pointer_roots, cs.addr64_pointer_roots_overflow,
                 cs.num_readconst_pointer_roots, cs.readconst_pointer_roots_overflow,
                 cs.flattened_ud_buf.size());
        LOG_INFO(Render_Vulkan,
                 "TEMP Shadow compute {:#x}: groups={}x{}x{} threads={}x{}x{} buffers={} "
                 "images={}",
                 cs.pgm_hash, cs_program.dim_x, cs_program.dim_y, cs_program.dim_z,
                 cs_program.num_thread_x.full, cs_program.num_thread_y.full,
                 cs_program.num_thread_z.full, cs.buffers.size(), cs.images.size());
        for (u32 index = 0; index < cs.flattened_ud_buf.size(); ++index) {
            LOG_INFO(Render_Vulkan, "TEMP Shadow compute {:#x}: flat[{}]={:#x}", cs.pgm_hash,
                     index, cs.flattened_ud_buf[index]);
        }
        u32 buffer_ordinal{};
        for (const auto& desc : cs.buffers) {
            const auto sharp = desc.GetSharp(cs);
            LOG_INFO(Render_Vulkan,
                     "TEMP Shadow compute {:#x}: buffer[{}] sharp_dw={} addr={:#x} stride={} "
                     "records={} size={} written={} special={}",
                     cs.pgm_hash, buffer_ordinal++, desc.sharp_idx, sharp.base_address,
                     sharp.stride, sharp.num_records, sharp.GetSize(), desc.is_written,
                     desc.IsSpecial());
        }
    }
    static u32 logged_shadow_b33_dispatches = 0;
    static u32 logged_shadow_875_dispatches = 0;
    const bool log_shadow_dispatch =
        (cs.pgm_hash == 0xb33e9db6 && logged_shadow_b33_dispatches < 8) ||
        (cs.pgm_hash == 0x875e4f39 && logged_shadow_875_dispatches < 8);
    if (log_shadow_dispatch) {
        const u32 set = cs.pgm_hash == 0xb33e9db6 ? logged_shadow_b33_dispatches
                                                   : logged_shadow_875_dispatches;
        LOG_INFO(Render_Vulkan,
                 "TEMP Shadow dispatch shader={:#x} set={}: groups={}x{}x{} threads={}x{}x{}",
                 cs.pgm_hash, set, cs_program.dim_x, cs_program.dim_y,
                 cs_program.dim_z, cs_program.num_thread_x.full, cs_program.num_thread_y.full,
                 cs_program.num_thread_z.full);
        if (cs.pgm_hash == 0xb33e9db6) {
            ++logged_shadow_b33_dispatches;
        } else {
            ++logged_shadow_875_dispatches;
        }
    }
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) {
        return;
    }
    if (!BindResources(pipeline)) {
        return;
    }

    scheduler.EndRendering();
    if (ComputeDispatchRequiresIdleBeforeChunks(cs.pgm_hash)) {
        scheduler.Finish();
    }
    const u32 max_chunk_x = ComputeDispatchMaxChunkX(cs.pgm_hash, cs_program.dim_x);
    const u32 max_chunk_y = ComputeDispatchMaxChunkY(cs.pgm_hash, cs_program.dim_y);
    const DiagnosticCheckpointMarker* diagnostic_marker{};
    DiagnosticCheckpointMarker recorded_marker{};
    if (EnablePerCommandGpuDiagnostics && instance.SupportsDiagnosticCheckpoints()) {
        diagnostic_marker =
            MakeDiagnosticCheckpoint(DiagnosticCheckpointKind::Compute, cs.pgm_hash);
        recorded_marker = *diagnostic_marker;
        recorded_marker.runtime[0] = cs_program.dim_x;
        recorded_marker.runtime[1] = cs_program.dim_y;
        recorded_marker.runtime[2] = cs_program.dim_z;
        RecordDiagnosticCommand(recorded_marker);
    }
    const u32 chunk_count = ComputeDispatchChunk2DCount(cs_program.dim_x, cs_program.dim_y,
                                                        max_chunk_x, max_chunk_y);
    const bool submit_chunks = ComputeDispatchRequiresChunkSubmissions(cs.pgm_hash);
    RecordSparseDiagnosticCheckpoint(true, cs.pgm_hash);
    if (diagnostic_marker) {
        const auto cmdbuf = scheduler.CommandBuffer();
        VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdSetCheckpointNV(
            static_cast<VkCommandBuffer>(cmdbuf), diagnostic_marker);
        recorded_marker.runtime[3] = 0xB4;
        RecordDiagnosticGpuMarker(cmdbuf, recorded_marker);
    }
    bool needs_binding = true;
    for (u32 chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        if (needs_binding) {
            pipeline->BindResources(set_writes, buffer_barriers, push_data);
            needs_binding = false;
        }
        const auto cmdbuf = scheduler.CommandBuffer();
        scheduler.BindPipeline(PipelineBindPoint::Compute, pipeline->Handle());
        const auto chunk = ComputeDispatchChunk2DAt(cs_program.dim_x, cs_program.dim_y,
                                                    max_chunk_x, max_chunk_y, chunk_index);
        if (!ComputeDispatchChunkNeedsBase(chunk)) {
            cmdbuf.dispatch(chunk.group_count_x, chunk.group_count_y, cs_program.dim_z);
        } else {
            cmdbuf.dispatchBase(chunk.base_x, chunk.base_y, 0, chunk.group_count_x,
                               chunk.group_count_y, cs_program.dim_z);
        }
        if (submit_chunks && chunk_index + 1 < chunk_count) {
            scheduler.Flush();
            needs_binding = true;
        }
    }
    if (diagnostic_marker) {
        recorded_marker.runtime[3] = 0xA4;
        RecordDiagnosticGpuMarker(scheduler.CommandBuffer(), recorded_marker);
    }
    DebugState.IncDispatch();

    ResetBindings();
    RecordGpuWork();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Compute, pipeline->Handle());
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    RecordSparseDiagnosticCheckpoint(true, info.pgm_hash);
    cmdbuf.dispatchIndirect(buffer->Handle(), base);
    DebugState.IncDispatch();

    ResetBindings();
    RecordGpuWork();
}

void Rasterizer::RecordGpuWork() {
    if (submission_chunk_policy.RecordGpuWork()) {
        scheduler.Flush();
    }
}

void Rasterizer::RecordSparseDiagnosticCheckpoint(bool is_compute, u64 first_hash,
                                                   u64 second_hash) {
    if (!instance.SupportsDiagnosticCheckpoints()) {
        return;
    }
    const auto kind = is_compute ? DiagnosticCheckpointKind::Compute
                                 : DiagnosticCheckpointKind::Graphics;
    const u32 kind_value = static_cast<u32>(kind);
    if (last_sparse_diagnostic_kind == kind_value &&
        last_sparse_diagnostic_first_hash == first_hash &&
        last_sparse_diagnostic_second_hash == second_hash) {
        return;
    }
    last_sparse_diagnostic_kind = kind_value;
    last_sparse_diagnostic_first_hash = first_hash;
    last_sparse_diagnostic_second_hash = second_hash;
    const auto* marker = MakeDiagnosticCheckpoint(kind, first_hash, second_hash);
    RecordDiagnosticCommand(*marker);
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdSetCheckpointNV(
        static_cast<VkCommandBuffer>(scheduler.CommandBuffer()), marker);
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    buffer_cache.RunGarbageCollector();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

    set_write_index = 0;
    set_writes.clear();
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        if (stage->pgm_hash == 0x57bd9f27) {
            u64 signature = 1469598103934665603ULL;
            for (const u32 value : stage->flattened_ud_buf) {
                signature = (signature ^ value) * 1099511628211ULL;
            }
            static u64 last_signature{};
            static u32 logged_signatures{};
            if (signature != last_signature && logged_signatures++ < 128) {
                last_signature = signature;
                const auto& cs = liverpool->GetCsRegs();
                LOG_INFO(Render_Vulkan,
                         "TEMP Shadow 57bd resources signature={:#x} dispatch={}x{}x{} "
                         "buffers={} images={} flat_dw={}",
                         signature, cs.dim_x, cs.dim_y, cs.dim_z, stage->buffers.size(),
                         stage->images.size(), stage->flattened_ud_buf.size());
                u32 ordinal{};
                for (const auto& desc : stage->buffers) {
                    const auto sharp = desc.GetSharp(*stage);
                    LOG_INFO(Render_Vulkan,
                             "TEMP Shadow 57bd buffer[{}] sharp_dw={} addr={:#x} stride={} "
                             "records={} size={} written={} special={}",
                             ordinal++, desc.sharp_idx, sharp.base_address, sharp.stride,
                             sharp.num_records, sharp.GetSize(), desc.is_written,
                             desc.IsSpecial());
                }
                ordinal = 0;
                for (const auto& desc : stage->images) {
                    const auto sharp = desc.GetSharp(*stage);
                    LOG_INFO(Render_Vulkan,
                             "TEMP Shadow 57bd image[{}] sharp_dw={} addr={:#x} pitch={} "
                             "extent={}x{}x{} levels={} type={} tile={} format={} num={} "
                             "written={}",
                             ordinal++, desc.sharp_idx, sharp.Address(), sharp.pitch + 1U,
                             sharp.width + 1U, sharp.height + 1U, sharp.depth + 1U,
                             sharp.last_level + 1U, static_cast<u32>(sharp.GetType()),
                             static_cast<u32>(sharp.GetTileMode()),
                             static_cast<u32>(sharp.GetDataFmt()),
                             static_cast<u32>(sharp.GetNumberFmt()), desc.is_written);
                }
            }
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        stage->PushUd(binding, push_data);
        Shader::PrepareReadConstPointersThenRefresh(
            stage->num_readconst_pointer_roots + 1U,
            [&] {
                std::array<u64, Shader::MaxReadConstPointerRoots> pointers{};
                for (u32 ordinal = 0; ordinal < stage->num_readconst_pointer_roots; ++ordinal) {
                    const u32 index = stage->readconst_pointer_roots[ordinal].pointer_lo_flat_index;
                    if (index + 1U < stage->flattened_ud_buf.size()) {
                        pointers[ordinal] = u64{stage->flattened_ud_buf[index]} |
                                            (u64{stage->flattened_ud_buf[index + 1U]} << 32U);
                    }
                }
                return pointers;
            },
            [&] { PrepareAddr64PointerResidency(*stage); }, [&] { stage->RefreshFlatBuf(); });
        BindBuffers(*stage, binding, push_data);
        BindTextures(*stage, binding);
        uses_dma |= stage->uses_dma;
    }

    if (uses_dma) {
        // We only use fault buffer for DMA right now.
        buffer_cache.SynchronizeDmaBuffers();
        fault_process_pending = true;
    }

    return true;
}

void Rasterizer::PrepareAddr64PointerResidency(const Shader::Info& stage) {
    // TEMP Shadow diagnostic: inspect the main deferred-lighting shader's pointer roots.
    static bool logged_shadow_deferred_lighting = false;
    static bool logged_shadow_scene_light = false;
    static bool logged_shadow_corrupt_texture = false;
    static bool logged_shadow_missing_terrain = false;
    static bool logged_shadow_predicated_scene = false;
    bool* logged = stage.pgm_hash == 0x291dbcd0 ? &logged_shadow_deferred_lighting
                   : stage.pgm_hash == 0x617e09a3 ? &logged_shadow_scene_light
                   : stage.pgm_hash == 0xc61ae8f9 ? &logged_shadow_corrupt_texture
                   : stage.pgm_hash == 0xef6bb4ef ? &logged_shadow_missing_terrain
                   : stage.pgm_hash == 0xa9027daa ? &logged_shadow_predicated_scene
                                                   : nullptr;
    if (logged && !*logged) {
        *logged = true;
        LOG_INFO(Render_Vulkan,
                 "TEMP Shadow lighting {:#x}: dma={} readconst={} roots={} overflow={} "
                 "read_roots={} read_overflow={} flat_dw={}",
                 stage.pgm_hash, stage.uses_dma, static_cast<u32>(stage.readconst_types),
                 stage.num_addr64_pointer_roots, stage.addr64_pointer_roots_overflow,
                 stage.num_readconst_pointer_roots, stage.readconst_pointer_roots_overflow,
                 stage.flattened_ud_buf.size());
        for (u32 ordinal = 0; ordinal < stage.num_readconst_pointer_roots; ++ordinal) {
            const auto& root = stage.readconst_pointer_roots[ordinal];
            const u32 lo = root.pointer_lo_flat_index < stage.flattened_ud_buf.size()
                               ? stage.flattened_ud_buf[root.pointer_lo_flat_index]
                               : 0;
            const u32 hi = root.pointer_lo_flat_index + 1U < stage.flattened_ud_buf.size()
                               ? stage.flattened_ud_buf[root.pointer_lo_flat_index + 1U]
                               : 0;
            LOG_INFO(Render_Vulkan,
                     "TEMP Shadow lighting {:#x} root[{}]: flat={} ptr={:#x}:{:#x} "
                     "offset_dw={}..{} dynamic={}",
                     stage.pgm_hash, ordinal, root.pointer_lo_flat_index, hi, lo,
                     root.minimum_offset_dwords,
                     root.maximum_offset_dwords, root.dynamic_offset);
        }
    }
    if (stage.addr64_pointer_roots_overflow) {
        return;
    }

    for (u32 ordinal = 0; ordinal < stage.num_addr64_pointer_roots; ++ordinal) {
        const auto& root = stage.addr64_pointer_roots[ordinal];
        if (root.buffer_resource_index >= stage.buffers.size() ||
            root.pointer_lo_flat_index + 1U >= stage.flattened_ud_buf.size()) {
            continue;
        }

        const u32 pointer_lo = stage.flattened_ud_buf[root.pointer_lo_flat_index];
        const u32 pointer_hi = stage.flattened_ud_buf[root.pointer_lo_flat_index + 1U];
        const auto plan = Shader::PlanAddr64PointerResidency(
            pointer_lo, pointer_hi, 1ULL << VideoCore::MemoryTracker::MAX_CPU_PAGE_BITS);
        if (!plan.valid || !VideoCore::IsProcessableDmaFaultRange(
                               plan.address, plan.address + plan.size,
                               1ULL << VideoCore::MemoryTracker::MAX_CPU_PAGE_BITS,
                               [this](VAddr address, u64 size) {
                                   return page_manager.IsGpuMapped(address, size);
                               })) {
            continue;
        }
        VideoCore::MakeDmaFaultRangeResident(buffer_cache, plan.address, plan.size);
    }

    if (stage.readconst_pointer_roots_overflow) {
        return;
    }
    constexpr u64 page_size = VideoCore::BufferCache::CACHING_PAGESIZE;
    constexpr u64 address_space_size = page_size * VideoCore::BufferCache::CACHING_NUMPAGES;
    static bool logged_shadow_scene_readconst_ownership = false;
    static bool logged_shadow_corrupt_texture_readconst_ownership = false;
    static bool logged_shadow_missing_terrain_readconst_ownership = false;
    static bool logged_shadow_predicated_scene_readconst_ownership = false;
    const bool log_shadow_scene_readconst_ownership =
        (stage.pgm_hash == 0x617e09a3 && !logged_shadow_scene_readconst_ownership) ||
        (stage.pgm_hash == 0xc61ae8f9 &&
         !logged_shadow_corrupt_texture_readconst_ownership) ||
        (stage.pgm_hash == 0xef6bb4ef &&
         !logged_shadow_missing_terrain_readconst_ownership) ||
        (stage.pgm_hash == 0xa9027daa &&
         !logged_shadow_predicated_scene_readconst_ownership);
    for (u32 ordinal = 0; ordinal < stage.num_readconst_pointer_roots; ++ordinal) {
        const auto& root = stage.readconst_pointer_roots[ordinal];
        if (root.pointer_lo_flat_index + 1U >= stage.flattened_ud_buf.size()) {
            continue;
        }
        const auto plan = Shader::PlanReadConstPointerResidency(
            stage.flattened_ud_buf[root.pointer_lo_flat_index],
            stage.flattened_ud_buf[root.pointer_lo_flat_index + 1U],
            root.minimum_offset_dwords, root.maximum_offset_dwords, root.dynamic_offset,
            page_size, address_space_size);
        if (!plan.valid || !VideoCore::IsProcessableDmaFaultRange(
                               plan.address, plan.address + plan.size, address_space_size,
                               [this](VAddr address, u64 size) {
                                   return page_manager.IsGpuMapped(address, size);
                               })) {
            continue;
        }
        if (log_shadow_scene_readconst_ownership) {
            LOG_INFO(Render_Vulkan,
                     "TEMP Shadow scene readconst root[{}]: guest={:#x}:{:#x} tracked_gpu={} "
                     "exact_gpu={}",
                     ordinal, plan.address, plan.size,
                     buffer_cache.IsRegionGpuModified(plan.address, plan.size),
                     buffer_cache.IsRegionGpuModifiedExact(plan.address, plan.size));
        }
        VideoCore::MakeDmaFaultRangeCpuVisible(buffer_cache, plan.address, plan.size);
    }
    if (log_shadow_scene_readconst_ownership) {
        if (stage.pgm_hash == 0x617e09a3) {
            logged_shadow_scene_readconst_ownership = true;
        } else if (stage.pgm_hash == 0xc61ae8f9) {
            logged_shadow_corrupt_texture_readconst_ownership = true;
        } else if (stage.pgm_hash == 0xef6bb4ef) {
            logged_shadow_missing_terrain_readconst_ownership = true;
        } else {
            logged_shadow_predicated_scene_readconst_ownership = true;
        }
    }
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    const bool requires_buffer_copy = VideoCore::RequiresBufferCopyForDepthFormatMismatch(
        src_image.info.pixel_format, dst_image.info.pixel_format, src_image.info.props.is_depth,
        dst_image.info.props.is_depth);
    if ((instance.IsMaintenance8Supported() ||
         src_image.info.props.is_depth == dst_image.info.props.is_depth) &&
        !requires_buffer_copy) {
        dst_image.CopyImage(src_image);
    } else {
        const auto& copy_buffer =
            buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::DeviceLocal);
        dst_image.CopyImageWithBuffer(src_image, copy_buffer.Handle(), 0);
    }
    dst_image.flags |= VideoCore::ImageFlagBits::GpuModified;
    dst_image.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    image1.Clear(clear, range);
    image1.flags |= VideoCore::ImageFlagBits::GpuModified;
    image1.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    buffer_bindings.clear();
    static bool logged_shadow_scene_buffers = false;
    static u32 logged_shadow_terrain_buffer_sets = 0;
    static u32 logged_shadow_terrain_indices = 0;
    static u32 logged_shadow_b33_buffer_sets = 0;
    static u32 logged_shadow_87_buffer_states = 0;
    const bool log_shadow_terrain_buffers =
        stage.pgm_hash == 0xef6bb4ef && liverpool->regs.num_indices >= 10000 &&
        liverpool->regs.num_indices != logged_shadow_terrain_indices &&
        logged_shadow_terrain_buffer_sets < 32;
    const bool log_shadow_scene_buffers =
        (stage.pgm_hash == 0x617e09a3 && !logged_shadow_scene_buffers) ||
        log_shadow_terrain_buffers ||
        (stage.pgm_hash == 0xb33e9db6 && logged_shadow_b33_buffer_sets < 8);
    u32 shadow_scene_buffer_ordinal = 0;

    struct GpuModifiedReadBarrier {
        VideoCore::Buffer* buffer;
        bool needs_fallback{};
        bool write_dependency_covered{};
    };
    boost::container::small_vector<GpuModifiedReadBarrier, Shader::NUM_BUFFERS> read_barriers;

    for (const auto& desc : stage.buffers) {
        const auto vsharp = desc.GetSharp(stage);
        if (log_shadow_scene_buffers ||
            (stage.pgm_hash == 0xef6bb4ef && !vsharp.Valid())) {
            LOG_INFO(Render_Vulkan,
                     "TEMP Shadow buffer shader={:#x} draw_indices={} set={} ordinal={} "
                     "sharp_dw={} special={} type={} valid={} addr={:#x} stride={} records={} "
                     "size={} written={} clip_cntl={:#010x} clip_space={}",
                     stage.pgm_hash, liverpool->regs.num_indices,
                     logged_shadow_terrain_buffer_sets, shadow_scene_buffer_ordinal,
                     desc.sharp_idx, desc.IsSpecial(),
                     static_cast<u32>(desc.buffer_type), vsharp.Valid(), vsharp.base_address,
                     vsharp.stride, vsharp.num_records, vsharp.GetSize(), desc.is_written,
                     std::bit_cast<u32>(liverpool->regs.clipper_control),
                     static_cast<u32>(liverpool->regs.clipper_control.clip_space));
        }
        ++shadow_scene_buffer_ordinal;
        const bool has_gpu_mapping = !desc.IsSpecial() && vsharp.base_address != 0 &&
                                     memory->IsValidGpuMapping(vsharp.base_address, 0);
        const bool has_cpu_mapping = !desc.IsSpecial() && vsharp.base_address != 0 &&
                                     memory->IsMappedMemory(vsharp.base_address, 1);
        if (VideoCore::IsGuestBufferDescriptorBindable(
                desc.IsSpecial(), vsharp.base_address, vsharp.GetSize(), has_gpu_mapping,
                has_cpu_mapping)) {
            const u64 binding_size = desc.GetBindingSize(vsharp);
            const u64 size = memory->ClampRangeSize(vsharp.base_address, binding_size);
            if (size > 64_MB) {
                LOG_ERROR(Render_Vulkan,
                          "TEMP oversized buffer binding shader={:#x} stage={} sharp_dw={} "
                          "addr={:#x} stride={} records={} raw_size={} binding_size={} "
                          "clamped_size={} written={} formatted={}",
                          stage.pgm_hash, stage.stage, desc.sharp_idx, vsharp.base_address,
                          vsharp.stride, vsharp.num_records, vsharp.GetSize(), binding_size, size,
                          desc.is_written, desc.is_formatted);
            }
            const auto buffer_id = buffer_cache.FindBuffer(vsharp.base_address, size);
            buffer_bindings.emplace_back(buffer_id, vsharp, size);
            if (stage.pgm_hash == 0xef6bb4ef && logged_shadow_terrain_buffer_sets < 8) {
                LOG_INFO(Render_Vulkan,
                         "TEMP Shadow bound buffer shader={:#x} ordinal={} sharp_dw={} "
                         "gpu_mapping={} cpu_mapping={} binding_size={} clamped_size={} "
                         "buffer_id={}",
                         stage.pgm_hash, shadow_scene_buffer_ordinal - 1, desc.sharp_idx,
                         has_gpu_mapping, has_cpu_mapping, binding_size, size,
                         buffer_id.index);
            }
        } else {
            buffer_bindings.emplace_back(VideoCore::BufferId{}, vsharp, 0);
            if (stage.pgm_hash == 0xef6bb4ef && logged_shadow_terrain_buffer_sets < 8) {
                LOG_INFO(Render_Vulkan,
                         "TEMP Shadow unbound buffer shader={:#x} ordinal={} sharp_dw={} "
                         "special={} addr_nonzero={} size_nonzero={} gpu_mapping={} "
                         "cpu_mapping={}",
                         stage.pgm_hash, shadow_scene_buffer_ordinal - 1, desc.sharp_idx,
                         desc.IsSpecial(), vsharp.base_address != 0, vsharp.GetSize() > 0,
                         has_gpu_mapping, has_cpu_mapping);
            }
        }
    }
    if (log_shadow_scene_buffers) {
        if (stage.pgm_hash == 0x617e09a3) {
            logged_shadow_scene_buffers = true;
        } else {
            logged_shadow_terrain_indices = liverpool->regs.num_indices;
            ++logged_shadow_terrain_buffer_sets;
        }
        if (stage.pgm_hash == 0xb33e9db6) {
            ++logged_shadow_b33_buffer_sets;
        }
    }

    // Second pass to re-bind buffers that were updated after binding
    for (u32 i = 0; i < buffer_bindings.size(); i++) {
        const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
        const auto& desc = stage.buffers[i];
        const bool is_storage = desc.IsStorage(vsharp);
        const u32 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
        // Buffer is not from the cache, either a special buffer or unbound.
        if (!buffer_id) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);

                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->regs.clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->regs.clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const auto& cs_program = liverpool->GetCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            }
        } else {
            const bool was_gpu_modified = !desc.is_written &&
                                          buffer_cache.IsRegionGpuModifiedExact(
                                              vsharp.base_address, size);
            auto [vk_buffer, offset] = buffer_cache.ObtainBuffer(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id);
            bool gpu_snapshot = false;
            if (VideoCore::ShouldSnapshotGpuModifiedComputeControl(
                    was_gpu_modified, stage.stage == Shader::Stage::Compute, desc.is_written,
                    desc.is_formatted, size)) {
                auto& snapshot = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const auto [snapshot_data, snapshot_offset] = snapshot.Map(size, alignment);
                ASSERT(snapshot_data != nullptr);
                snapshot.Commit();
                scheduler.EndRendering();
                const auto cmdbuf = scheduler.CommandBuffer();
                const std::array pre_barriers{
                    vk::BufferMemoryBarrier2{
                        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
                        .buffer = vk_buffer->Handle(),
                        .offset = offset,
                        .size = size,
                    },
                    vk::BufferMemoryBarrier2{
                        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
                        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                        .buffer = snapshot.Handle(),
                        .offset = snapshot_offset,
                        .size = size,
                    },
                };
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
                    .pBufferMemoryBarriers = pre_barriers.data(),
                });
                cmdbuf.copyBuffer(vk_buffer->Handle(), snapshot.Handle(),
                                  vk::BufferCopy{offset, snapshot_offset, size});
                const vk::BufferMemoryBarrier2 post_barrier{
                    .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .buffer = snapshot.Handle(),
                    .offset = snapshot_offset,
                    .size = size,
                };
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .bufferMemoryBarrierCount = 1,
                    .pBufferMemoryBarriers = &post_barrier,
                });
                vk_buffer = &snapshot;
                offset = static_cast<u32>(snapshot_offset);
                gpu_snapshot = true;
            }
            const u32 offset_aligned = Common::AlignDown(offset, alignment);
            const u32 adjust = offset - offset_aligned;
            if (adjust % 4 != 0) {
                LOG_WARNING(Render_Vulkan, "Buffer binding {} in shader {:#x} isn't dword aligned",
                            i, stage.pgm_hash);
            }
            push_data.AddOffset(binding.buffer, adjust);
            buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
            const auto shader_access = VideoCore::GetBufferShaderAccess(desc.is_written);
            const vk::AccessFlags2 access_mask =
                shader_access == VideoCore::BufferShaderAccess::ReadWrite
                    ? vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
                    : vk::AccessFlagBits2::eShaderRead;
            bool write_dependency_covered = gpu_snapshot;
            if (!gpu_snapshot) {
                if (auto barrier = vk_buffer->GetBarrier(
                        access_mask, vk::PipelineStageFlagBits2::eAllCommands, offset_aligned,
                        size + adjust)) {
                constexpr auto write_flags = vk::AccessFlagBits2::eTransferWrite |
                                             vk::AccessFlagBits2::eShaderWrite |
                                             vk::AccessFlagBits2::eMemoryWrite;
                write_dependency_covered = static_cast<bool>(barrier->srcAccessMask & write_flags);
                buffer_barriers.emplace_back(*barrier);
                }
            }

            auto candidate = std::ranges::find(read_barriers, vk_buffer,
                                               &GpuModifiedReadBarrier::buffer);
            if (candidate == read_barriers.end()) {
                candidate = read_barriers.emplace(read_barriers.end(), vk_buffer);
            }
            candidate->write_dependency_covered |= write_dependency_covered;
            candidate->needs_fallback |= VideoCore::NeedsGpuModifiedReadBarrier(
                was_gpu_modified, desc.is_written, write_dependency_covered);
            if (stage.pgm_hash == 0x87cf9b8a && logged_shadow_87_buffer_states < 32) {
                LOG_INFO(Render_Vulkan,
                         "TEMP Shadow 87 buffer state ordinal={} addr={:#x} size={} "
                         "written={} formatted={} gpu_modified={} snapshot={} "
                         "write_dependency={} fallback={}",
                         i, vsharp.base_address, size, desc.is_written, desc.is_formatted,
                         was_gpu_modified, gpu_snapshot, write_dependency_covered,
                         candidate->needs_fallback);
                ++logged_shadow_87_buffer_states;
            }
            if (desc.is_written && desc.is_formatted) {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }
        }

        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageBuffer : vk::DescriptorType::eUniformBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        ++binding.buffer;
    }

    for (const auto& candidate : read_barriers) {
        if (!candidate.needs_fallback || candidate.write_dependency_covered) {
            continue;
        }
        buffer_barriers.emplace_back(vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .buffer = candidate.buffer->Handle(),
            .offset = 0,
            .size = candidate.buffer->SizeBytes(),
        });
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    // For loading/storing to explicit mip levels, when no native instruction support, bind an array
    // of descriptors consecutively, 1 for each mip level. The shader can index this with LOD
    // operand.
    // This array holds the size of each consecutive array with the number of bindings consumed.
    // This is currently always 1 for anything other than mip fallback arrays.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;
    boost::container::small_vector<std::pair<bool, bool>, 8> image_native_extent_requirements;
    const auto append_invalid_image_binding = [&](bool is_storage) {
        auto& binding_entry = image_bindings.emplace_back(
            std::piecewise_construct, std::tuple{}, std::tuple{});
        binding_entry.second.type = is_storage ? VideoCore::TextureCache::BindingType::Storage
                                               : VideoCore::TextureCache::BindingType::Texture;
        image_descriptor_array_sizes.push_back(1);
        image_native_extent_requirements.emplace_back(false, false);
    };

    static bool logged_shadow_scene_images = false;
    static bool logged_shadow_corrupt_texture_images = false;
    static bool logged_shadow_predicated_scene_images = false;
    static u32 logged_shadow_b33_image_sets = 0;
    static u32 logged_shadow_875_image_sets = 0;
    const bool log_shadow_scene_images =
        (stage.pgm_hash == 0x617e09a3 && !logged_shadow_scene_images) ||
        (stage.pgm_hash == 0xc61ae8f9 && !logged_shadow_corrupt_texture_images) ||
        (stage.pgm_hash == 0xa9027daa && !logged_shadow_predicated_scene_images) ||
        (stage.pgm_hash == 0xb33e9db6 && logged_shadow_b33_image_sets < 8) ||
        (stage.pgm_hash == 0x875e4f39 && logged_shadow_875_image_sets < 8);
    u32 shadow_scene_image_ordinal = 0;
    for (const auto& image_desc : stage.images) {
        if (log_shadow_scene_images) {
            const auto raw = stage.ReadUdSharp<AmdGpu::Image>(image_desc.sharp_idx);
            LOG_INFO(Render_Vulkan,
                     "TEMP Shadow scene image[{}]: sharp_dw={} valid={} addr={:#x} "
                     "pitch={} width={} height={} depth={} type={} levels={} data_format={} "
                     "num_format={} is_depth={} written={}",
                     shadow_scene_image_ordinal, image_desc.sharp_idx, raw.Valid(), raw.Address(),
                     raw.pitch, raw.width, raw.height, raw.depth, static_cast<u32>(raw.GetType()),
                     raw.last_level + 1U, static_cast<u32>(raw.GetDataFmt()),
                     static_cast<u32>(raw.GetNumberFmt()), image_desc.is_depth,
                     image_desc.is_written);
        }
        ++shadow_scene_image_ordinal;
        const auto tsharp = image_desc.GetSharp(stage);
        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
            append_invalid_image_binding(image_desc.is_written);
            continue;
        }

        if (!memory->IsValidGpuMapping(tsharp.Address(), 0) || tsharp.pitch < tsharp.width ||
            !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt) ||
            !VideoCore::IsImageComponentMappingUsable(tsharp.dst_sel_x, tsharp.dst_sel_y,
                                                      tsharp.dst_sel_z, tsharp.dst_sel_w) ||
            !VideoCore::IsImageDescriptorFormatUsable(
                data_fmt, num_fmt, LiverpoolToVK::IsSurfaceFormatSupported)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# shader={:#x} stage={} sharp_dw={} address={:#x}, "
                        "pitch={}, width={}, height={}, depth={}, data_format={}, num_format={}, "
                        "type={}, tile_mode={}, read_roots={}, read_overflow={}, flat_dw={}",
                        stage.pgm_hash, stage.stage, image_desc.sharp_idx, tsharp.Address(),
                        tsharp.pitch, tsharp.width, tsharp.height, tsharp.depth,
                        static_cast<u32>(data_fmt), static_cast<u32>(num_fmt),
                        static_cast<u32>(tsharp.GetType()), static_cast<u32>(tsharp.GetTileMode()),
                        stage.num_readconst_pointer_roots, stage.readconst_pointer_roots_overflow,
                        stage.flattened_ud_buf.size());
            static std::array<u64, 16> logged_invalid_shadow_texture_shaders{};
            static u32 logged_invalid_shadow_texture_count = 0;
            static bool logged_shadow_862e_invalid_texture = false;
            bool log_invalid_shadow_texture = false;
            if (stage.pgm_hash == 0x862e4da8 && !logged_shadow_862e_invalid_texture) {
                logged_shadow_862e_invalid_texture = true;
                log_invalid_shadow_texture = true;
            }
            {
                bool already_logged = false;
                for (u32 index = 0; index < logged_invalid_shadow_texture_count; ++index) {
                    already_logged |= logged_invalid_shadow_texture_shaders[index] == stage.pgm_hash;
                }
                if (!already_logged && logged_invalid_shadow_texture_count <
                                           logged_invalid_shadow_texture_shaders.size()) {
                    logged_invalid_shadow_texture_shaders[logged_invalid_shadow_texture_count++] =
                        stage.pgm_hash;
                    log_invalid_shadow_texture = true;
                }
            }
            if (log_invalid_shadow_texture) {
                for (u32 offset = 0; offset < 8; ++offset) {
                    const u32 index = image_desc.sharp_idx + offset;
                    const u32 value = index < stage.flattened_ud_buf.size()
                                          ? stage.flattened_ud_buf[index]
                                          : 0;
                    LOG_INFO(Render_Vulkan,
                             "TEMP Shadow invalid T# shader={:#x} flat[{}]={:#x}",
                             stage.pgm_hash, index, value);
                }
                for (u32 ordinal = 0; ordinal < stage.num_readconst_pointer_roots; ++ordinal) {
                    const auto& root = stage.readconst_pointer_roots[ordinal];
                    const u32 lo = root.pointer_lo_flat_index < stage.flattened_ud_buf.size()
                                       ? stage.flattened_ud_buf[root.pointer_lo_flat_index]
                                       : 0;
                    const u32 hi = root.pointer_lo_flat_index + 1U < stage.flattened_ud_buf.size()
                                       ? stage.flattened_ud_buf[root.pointer_lo_flat_index + 1U]
                                       : 0;
                    const auto plan = Shader::PlanReadConstPointerResidency(
                        lo, hi, root.minimum_offset_dwords, root.maximum_offset_dwords,
                        root.dynamic_offset, VideoCore::BufferCache::CACHING_PAGESIZE,
                        VideoCore::BufferCache::CACHING_PAGESIZE *
                            VideoCore::BufferCache::CACHING_NUMPAGES);
                    LOG_INFO(Render_Vulkan,
                             "TEMP Shadow invalid T# shader={:#x} root[{}] flat={} "
                             "ptr={:#x}:{:#x} offset_dw={}..{} dynamic={} plan={:#x}:{:#x} "
                             "tracked_gpu={} exact_gpu={}",
                             stage.pgm_hash, ordinal, root.pointer_lo_flat_index,
                             hi, lo, root.minimum_offset_dwords, root.maximum_offset_dwords,
                             root.dynamic_offset, plan.address, plan.size,
                             plan.valid && buffer_cache.IsRegionGpuModified(plan.address, plan.size),
                             plan.valid &&
                                 buffer_cache.IsRegionGpuModifiedExact(plan.address, plan.size));
                }
            }
            append_invalid_image_binding(image_desc.is_written);
            continue;
        }

        const auto& device_limits = instance.GetPhysicalDeviceLimits();
        const VideoCore::HostImageResourceLimits host_image_limits{
            .max_1d_dimension = device_limits.maxImageDimension1D,
            .max_2d_dimension = device_limits.maxImageDimension2D,
            .max_3d_dimension = device_limits.maxImageDimension3D,
            .max_array_layers = device_limits.maxImageArrayLayers,
            .max_mip_levels = 32,
        };
        const auto descriptor_type = tsharp.GetType();
        const auto base_type = tsharp.GetBaseType();
        const VideoCore::HostImageResourceExtent host_extent{
            .width = static_cast<u32>(tsharp.width + 1U),
            .height = static_cast<u32>(tsharp.height + 1U),
            .depth = base_type == AmdGpu::ImageType::Color3D
                         ? static_cast<u32>(tsharp.depth + 1U)
                         : 1U,
        };
        const VideoCore::HostImageSubresourceExtent host_resources{
            .levels = tsharp.NumLevels(),
            .layers = tsharp.NumLayers(),
        };
        if (!VideoCore::IsImageResourceWithinHostLimits(descriptor_type, host_extent,
                                                        host_resources, host_image_limits)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting T# outside host image limits shader={:#x} stage={} "
                        "sharp_dw={} address={:#x} extent={}x{}x{} levels={} layers={} type={}",
                        stage.pgm_hash, stage.stage, image_desc.sharp_idx, tsharp.Address(),
                        host_extent.width, host_extent.height, host_extent.depth,
                        host_resources.levels, host_resources.layers,
                        static_cast<u32>(descriptor_type));
            append_invalid_image_binding(image_desc.is_written);
            continue;
        }
        if (!VideoCore::IsImageEstimatedAllocationWithinLimit(
                host_extent, host_resources, AmdGpu::NumBitsPerBlock(data_fmt),
                AmdGpu::IsBlockCoded(data_fmt), tsharp.NumSamples(),
                instance.GetTotalMemoryBudget())) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting T# whose minimum allocation exceeds the device budget "
                        "shader={:#x} stage={} sharp_dw={} address={:#x} extent={}x{}x{} "
                        "levels={} layers={} bits={} block={}",
                        stage.pgm_hash, stage.stage, image_desc.sharp_idx, tsharp.Address(),
                        host_extent.width, host_extent.height, host_extent.depth,
                        host_resources.levels, host_resources.layers,
                        AmdGpu::NumBitsPerBlock(data_fmt), AmdGpu::IsBlockCoded(data_fmt));
            append_invalid_image_binding(image_desc.is_written);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(stage);
        const auto array_mode = AmdGpu::GetArrayMode(tsharp.GetTileMode());
        const VideoCore::ImageInfo guest_info{tsharp, image_desc};
        const bool fully_mapped = guest_info.guest_size != 0 &&
                                  memory->IsMappedMemory(guest_info.guest_address,
                                                         guest_info.guest_size);
        if (tsharp.lod_hw_cnt_en) {
            static u32 lod_counter_binding_logs{};
            if (lod_counter_binding_logs++ < 128) {
                LOG_WARNING(Render_Vulkan,
                            "TEMP PRT LOD counter shader={:#x} stage={} sharp_dw={} bank={} "
                            "min_lod_warn={:#x} base_level={} last_level={} tile_mode={} "
                            "array_mode={} address={:#x} size={:#x} fully_mapped={} written={}",
                            stage.pgm_hash, stage.stage, image_desc.sharp_idx,
                            tsharp.counter_bank_id, tsharp.min_lod_warn, tsharp.base_level,
                            tsharp.last_level, static_cast<u32>(tsharp.GetTileMode()),
                            static_cast<u32>(array_mode), guest_info.guest_address,
                            guest_info.guest_size, fully_mapped, image_desc.is_written);
            }
        }
        static const bool diagnostic_allow_incomplete_non_sparse =
            std::getenv("SHADPS4_DIAGNOSTIC_ALLOW_INCOMPLETE_NON_SPARSE_IMAGES") != nullptr;
        lod_feedback_images.Observe(guest_info.guest_address, guest_info.guest_size,
                                    tsharp.lod_hw_cnt_en);
        const bool allow_incomplete_non_sparse = lod_feedback_images.CanUseIncompleteBacking(
            guest_info.guest_address, guest_info.guest_size, image_desc.is_written,
            diagnostic_allow_incomplete_non_sparse);
        if (!VideoCore::IsImageGuestBackingUsable(array_mode, fully_mapped,
                                                  allow_incomplete_non_sparse)) {
            static u32 rejected_unmapped_image_logs = 0;
            if (rejected_unmapped_image_logs++ < 32) {
                LOG_WARNING(Render_Vulkan,
                            "Rejecting non-sparse T# without complete guest backing "
                            "shader={:#x} stage={} sharp_dw={} address={:#x} size={:#x} "
                            "tile_mode={} lod_feedback={} written={} diagnostic_enabled={} "
                            "diagnostic_allowed={}",
                            stage.pgm_hash, stage.stage, image_desc.sharp_idx,
                            guest_info.guest_address, guest_info.guest_size,
                            static_cast<u32>(tsharp.GetTileMode()), tsharp.lod_hw_cnt_en,
                            image_desc.is_written, diagnostic_allow_incomplete_non_sparse,
                            allow_incomplete_non_sparse);
            }
            append_invalid_image_binding(image_desc.is_written);
            continue;
        }
        if (array_mode == AmdGpu::ArrayMode::ArrayPrt3DTiledThick) {
            LOG_ERROR(Render_Vulkan,
                      "TEMP Shadow PRT3D thick shader={:#x} stage={} sharp_dw={} addr={:#x} "
                      "pitch={} width={} height={} depth={} levels={} data_format={} num_format={} "
                      "tile_mode={} type={} base_type={} layers={} samples={} pow2pad={}",
                      stage.pgm_hash, stage.stage, image_desc.sharp_idx, tsharp.Address(),
                      tsharp.pitch, tsharp.width, tsharp.height, tsharp.depth,
                      tsharp.last_level + 1U, static_cast<u32>(data_fmt),
                      static_cast<u32>(num_fmt), static_cast<u32>(tsharp.GetTileMode()),
                      static_cast<u32>(tsharp.GetType()), static_cast<u32>(tsharp.GetBaseType()),
                      tsharp.NumLayers(), tsharp.NumSamples(), tsharp.pow2pad);
        }

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = image_bindings.emplace_back(
                std::piecewise_construct, std::tuple{}, std::tuple{tsharp, image_desc});
            if (array_mode == AmdGpu::ArrayMode::ArrayPrt3DTiledThick) {
                LOG_ERROR(Render_Vulkan,
                          "TEMP Shadow PRT3D info shader={:#x} sharp_dw={} addr={:#x} "
                          "type={} volume={} block={} extent={}x{}x{} levels={} layers={} "
                          "bits={} guest_size={:#x} mip0_size={:#x} mip0_pitch={} "
                          "mip0_height={}",
                          stage.pgm_hash, image_desc.sharp_idx, desc.info.guest_address,
                          static_cast<u32>(desc.info.type), bool(desc.info.props.is_volume),
                          bool(desc.info.props.is_block), desc.info.size.width, desc.info.size.height,
                          desc.info.size.depth, desc.info.resources.levels,
                          desc.info.resources.layers, desc.info.num_bits, desc.info.guest_size,
                          desc.info.mips_layout[0].size, desc.info.mips_layout[0].pitch,
                          desc.info.mips_layout[0].height);
            }
            image_native_extent_requirements.emplace_back(image_desc.uses_integer_coordinates,
                                                          image_desc.queries_dimensions);

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            image_id = texture_cache.FindImage(desc);
            auto* image = &texture_cache.GetImage(image_id);
            if (VideoCore::ShouldUseAssociatedDepthForView(desc.view_info.format)) {
                if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                    // If this image has an associated depth image, it's a stencil attachment.
                    // Redirect the access to the actual depth-stencil buffer.
                    image_id = depth_image_id;
                    image = &texture_cache.GetImage(image_id);
                }
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }
    if (stage.pgm_hash == 0xb33e9db6 && log_shadow_scene_images) {
        ++logged_shadow_b33_image_sets;
    } else if (stage.pgm_hash == 0x875e4f39 && log_shadow_scene_images) {
        ++logged_shadow_875_image_sets;
    }
    if (log_shadow_scene_images) {
        if (stage.pgm_hash == 0x617e09a3) {
            logged_shadow_scene_images = true;
        } else if (stage.pgm_hash == 0xc61ae8f9) {
            logged_shadow_corrupt_texture_images = true;
        } else {
            logged_shadow_predicated_scene_images = true;
        }
    }

    // Second pass to re-bind images that were updated after binding
    ASSERT(image_native_extent_requirements.size() == image_bindings.size());
    for (u32 binding_index = 0; binding_index < image_bindings.size(); ++binding_index) {
        auto& [image_id, desc] = image_bindings[binding_index];
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            const auto [uses_integer_coordinates, queries_dimensions] =
                image_native_extent_requirements[binding_index];
            if (VideoCore::ShouldForceNativeHostImageAccess(is_storage, uses_integer_coordinates,
                                                            queries_dimensions)) {
                ASSERT(image.SetHostScale(100));
            }
            auto& image_view = texture_cache.FindTexture(image_id, desc);

            // The image is either bound as storage in a separate descriptor or bound as a render
            // target in a feedback loop. The feedback-loop layout applies to color and depth.
            if (image.binding.force_general ||
                (image.binding.is_target &&
                 instance.IsAttachmentFeedbackLoopLayoutSupported())) {
                image.Transit(instance.IsAttachmentFeedbackLoopLayoutSupported() &&
                                      image.binding.is_target
                                  ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                  : vk::ImageLayout::eGeneral,
                              vk::AccessFlagBits2::eShaderRead |
                                  (image.info.props.is_depth
                                       ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                       : vk::AccessFlagBits2::eColorAttachmentWrite |
                                             vk::AccessFlagBits2::eColorAttachmentRead),
                              {});
            } else {
                if (is_storage) {
                    image.Transit(vk::ImageLayout::eGeneral,
                                  vk::AccessFlagBits2::eShaderRead |
                                      vk::AccessFlagBits2::eShaderWrite,
                                  desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead,
                                  desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        const auto normalized_anisotropy = VideoCore::NormalizeSamplerAnisotropy(
            static_cast<AmdGpu::AnisoRatio>(ssharp.max_aniso.Value()));
        if (normalized_anisotropy != ssharp.max_aniso.Value()) {
            LOG_WARNING(Render_Vulkan,
                        "Reserved sampler anisotropy encoding {}; falling back to 1x",
                        static_cast<u32>(ssharp.max_aniso.Value()));
            ssharp.max_aniso.Assign(normalized_anisotropy);
        }
        const auto normalized_mip_filter = VideoCore::NormalizeSamplerMipFilter(
            static_cast<AmdGpu::MipFilter>(ssharp.mip_filter.Value()));
        if (normalized_mip_filter != ssharp.mip_filter.Value()) {
            LOG_WARNING(Render_Vulkan,
                        "Reserved sampler mip-filter encoding {}; falling back to none",
                        static_cast<u32>(ssharp.mip_filter.Value()));
            ssharp.mip_filter.Assign(normalized_mip_filter);
        }
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(stage);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        const auto vk_sampler = texture_cache.GetSampler(ssharp, liverpool->regs.ta_bc_base);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop_aspects = {};
    const auto& regs = liverpool->regs;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    current_host_scale = prepared_host_scale;
    for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
        if (cb_descs[cb].first &&
            texture_cache.GetImage(cb_descs[cb].first).GetHostScale() != prepared_host_scale) {
            current_host_scale = 1;
        }
    }
    if (db_desc.first &&
        texture_cache.GetImage(db_desc.first).GetHostScale() != prepared_host_scale) {
        current_host_scale = 1;
    }
    const auto observe_scale = [this](const VideoCore::Image& image) {
        const u32 scale = image.GetHostScale();
        current_host_scale = current_host_scale == 0       ? scale
                             : current_host_scale == scale ? scale
                                                           : 1;
    };
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        image->SetBackingSamples(key.color_samples[cb]);
        static_cast<void>(texture_cache.FindRenderTarget(image_id, desc));
        ASSERT(image->SetHostScale(current_host_scale * 100,
                                   EmulatorSettings.GetInternalScreenWidth(),
                                   EmulatorSettings.GetInternalScreenHeight()));
        const auto& image_view = image->FindView(desc.view_info, false);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            image->Transit(instance.IsAttachmentFeedbackLoopLayoutSupported()
                               ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                               : vk::ImageLayout::eGeneral,
                           vk::AccessFlagBits2::eColorAttachmentWrite, {});
            attachment_feedback_loop_aspects |= vk::ImageAspectFlagBits::eColor;
        } else {
            image->Transit(vk::ImageLayout::eColorAttachmentOptimal,
                           vk::AccessFlagBits2::eColorAttachmentWrite |
                               vk::AccessFlagBits2::eColorAttachmentRead,
                           desc.view_info.range);
        }

        const auto host_extent = image->GetHostExtent();
        state.width = std::min<u32>(state.width, std::max(host_extent.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(host_extent.height >> mip, 1u));
        observe_scale(*image);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        static_cast<void>(texture_cache.FindDepthTarget(image_id, desc));
        auto& image = texture_cache.GetImage(image_id);
        ASSERT(image.SetHostScale(current_host_scale * 100,
                                  EmulatorSettings.GetInternalScreenWidth(),
                                  EmulatorSettings.GetInternalScreenHeight()));
        const auto& image_view = image.FindView(desc.view_info, false);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear = VideoCore::DepthStencilPolicy::ShouldClearDepth(
            regs.depth_render_control.depth_clear_enable, regs.depth_control.depth_enable,
            regs.depth_control.depth_write_enable,
            texture_cache.IsMetaCleared(htile_address, slice));
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const bool feedback_loop = image.binding.is_bound &&
                                   instance.IsAttachmentFeedbackLoopLayoutSupported();
        const auto new_layout = feedback_loop
                                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                : desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        image.Transit(new_layout,
                      vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                          vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                      feedback_loop ? std::optional<VideoCore::SubresourceRange>{}
                                    : std::optional{desc.view_info.range});
        if (feedback_loop) {
            attachment_feedback_loop_aspects |= VideoCore::AttachmentFeedbackLoopAspects(
                true, has_stencil);
        }

        const auto host_extent = image.GetHostExtent();
        state.width = std::min<u32>(state.width, host_extent.width);
        state.height = std::min<u32>(state.height, host_extent.height);
        observe_scale(image);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }
    if (current_host_scale == 0) {
        current_host_scale = 1;
    }
    push_data.host_inv_scale = 1.0f / static_cast<float>(current_host_scale);

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->last_cb_extent[0];
    const auto& mrt1_hint = liverpool->last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->regs.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->regs.color_buffers[0].Address(),
                                 liverpool->regs.color_buffers[1].Address()));
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->regs;

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->regs.depth_view.slice_start;
    sub_range.extent.layers = liverpool->regs.depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    read_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                       sub_range);
    write_image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        sub_range);

    auto aspect_mask = vk::ImageAspectFlags(0);
    if (is_depth) {
        aspect_mask |= vk::ImageAspectFlagBits::eDepth;
    }
    if (is_stencil) {
        aspect_mask |= vk::ImageAspectFlagBits::eStencil;
    }

    vk::ImageCopy region = {
        .srcSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .srcOffset = {0, 0, 0},
        .dstSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .dstOffset = {0, 0, 0},
        .extent = write_image.GetHostExtent(),
    };
    scheduler.CommandBuffer().copyImage(read_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                        write_image.GetImage(),
                                        vk::ImageLayout::eTransferDstOptimal, region);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    buffer_cache.FillBuffer(address, num_bytes, value, is_gds);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    buffer_cache.CopyBuffer(dst, src, num_bytes, dst_gds, src_gds);
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size);
    return true;
}

bool Rasterizer::ArmDiagnosticReadWatch(const VAddr addr, const u64 size) {
    return page_manager.ArmDiagnosticReadWatch(addr, size);
}

bool Rasterizer::HandleDiagnosticReadWatchFault(void* context, const VAddr addr,
                                                const bool is_write) {
    return page_manager.HandleDiagnosticReadWatchFault(context, addr, is_write);
}

void Rasterizer::ProcessDownloadImages() {
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    lod_feedback_images.Invalidate(addr, size);
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.Commit(instance, scheduler.CommandBuffer());
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->regs;

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;

            if (current_host_scale != 1) {
                viewport.x *= static_cast<float>(current_host_scale);
                viewport.y *= static_cast<float>(current_host_scale);
                viewport.width *= static_cast<float>(current_host_scale);
                viewport.height *= static_cast<float>(current_host_scale);
            }
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x * static_cast<s32>(current_host_scale),
                       vp_scsr.top_left_y * static_cast<s32>(current_host_scale)},
            .extent = {vp_scsr.GetWidth() * current_host_scale,
                       vp_scsr.GetHeight() * current_host_scale},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        // GCN REPLACE_OP writes DB_STENCILREFMASK.STENCILOPVAL, so a face whose stencil ops
        // include ReplaceOp takes its Vulkan reference from op_val.
        const auto& sc = regs.stencil_control;
        const bool front_op = VideoCore::DepthStencilPolicy::UsesStencilOpValue(
            sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op =
            regs.depth_control.backface_enable
                ? VideoCore::DepthStencilPolicy::UsesStencilOpValue(
                      sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                : front_op;
        const auto ref_conflict = [](AmdGpu::CompareFunc func, const AmdGpu::StencilRefMask& ref) {
            return func != AmdGpu::CompareFunc::Always && func != AmdGpu::CompareFunc::Never &&
                   ref.stencil_test_val != ref.stencil_op_val;
        };
        if ((front_op && ref_conflict(regs.depth_control.stencil_ref_func, front)) ||
            (back_op && regs.depth_control.backface_enable &&
             ref_conflict(regs.depth_control.stencil_bf_func, back))) {
            LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                       "op_val; the stencil test will use op_val");
        }
        dynamic_state.SetStencilReferences(
            VideoCore::DepthStencilPolicy::SelectStencilReference(front_op, front.stencil_op_val,
                                                                  front.stencil_test_val),
            VideoCore::DepthStencilPolicy::SelectStencilReference(back_op, back.stencil_op_val,
                                                                  back.stencil_test_val));
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopAspects(attachment_feedback_loop_aspects);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

} // namespace Vulkan
