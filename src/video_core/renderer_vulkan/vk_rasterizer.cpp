// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <cstring>

#include "common/debug.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/addr64_pointer_residency.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_residency.h"
#include "video_core/buffer_cache/fault_range.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderdoc.h"
#include "video_core/renderdoc_shader_selector.h"
#include "video_core/renderer_vulkan/compute_dependency_probe.h"
#include "video_core/renderer_vulkan/depth_stencil_policy.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/sampler_rejection_diagnostics.h"
#include "video_core/renderer_vulkan/texture_rejection_diagnostics.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/depth_association.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace Vulkan {

static void LogColorLedgerEnd(const VideoCore::ColorLedgerSummary& end) {
    const auto reason = end.reason == VideoCore::ColorLedgerEnd::Window     ? "window"
                        : end.reason == VideoCore::ColorLedgerEnd::Capacity ? "capacity"
                                                                            : "teardown";
    LOG_WARNING(Render_Vulkan,
                "ColorCommandLedgerV1 end reason={} renderer_ms={} observed={} duplicates={} "
                "unique={} truncated={}",
                reason, end.elapsed_ms, end.observed, end.duplicates, end.unique,
                end.reason != VideoCore::ColorLedgerEnd::Window);
}

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
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, page_manager, buffer_cache} {
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    const auto environment = [](const char* name) -> std::optional<std::string_view> {
        const auto* value = std::getenv(name);
        return value ? std::optional<std::string_view>{value} : std::nullopt;
    };
    const auto enable = environment("SHADPS4_COLOR_COMMAND_LEDGER");
    if (!enable || *enable == "0") {
        return;
    }
    const auto policy = VideoCore::ColorCommandLedgerPolicy::Parse(
        enable, environment("SHADPS4_COLOR_COMMAND_LEDGER_START_MS"),
        environment("SHADPS4_COLOR_COMMAND_LEDGER_DURATION_MS"));
    try {
        if (!policy.valid) {
            LOG_WARNING(Render_Vulkan, "ColorCommandLedgerV1 refused invalid launch policy");
            return;
        }
        color_command_ledger = VideoCore::EmittedColorCommandLedger{policy};
        color_ledger_origin = std::chrono::steady_clock::now();
        LOG_WARNING(Render_Vulkan,
                    "ColorCommandLedgerV1 begin origin=renderer_construction start_ms={} "
                    "duration_ms={} capacity=32 counts=first_command_exemplars "
                    "format=requested_view",
                    policy.start_ms, policy.duration_ms);
    } catch (...) {
        color_command_ledger.Disable();
    }
}

Rasterizer::~Rasterizer() {
    if (color_command_ledger.Enabled()) {
        try {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - color_ledger_origin)
                                     .count();
            color_command_ledger.Close(static_cast<u64>(elapsed), LogColorLedgerEnd);
        } catch (...) {
            // Logging is best-effort and cannot make owner teardown throw.
        }
    }
}

void Rasterizer::RecordEmittedColorCommand(const GraphicsPipeline& pipeline,
                                           const RenderState& state,
                                           VideoCore::EmittedColorCommand command) noexcept {
    if (!color_command_ledger.Enabled()) {
        return;
    }
    try {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - color_ledger_origin)
                                 .count();
        color_command_ledger.Observe(
            static_cast<u64>(elapsed),
            [&]() -> std::optional<VideoCore::EmittedColorCommand> {
                if (!VideoCore::HasBoundRenderDocColorAttachment(state.color_attachments,
                                                                 state.num_color_attachments)) {
                    return std::nullopt;
                }
                command.key.vertex_hash = pipeline.GetStage(Shader::LogicalStage::Vertex).pgm_hash;
                const auto* fragment = pipeline.GetStages()[u32(Shader::LogicalStage::Fragment)];
                command.key.has_fragment = fragment != nullptr;
                command.key.fragment_hash = fragment ? fragment->pgm_hash : 0;
                command.key.pipeline_key_hash =
                    std::hash<GraphicsPipelineKey>{}(pipeline.GetGraphicsKey());
                for (u32 index = 0; index < state.num_color_attachments; ++index) {
                    if (!state.color_attachments[index].image_view) {
                        continue;
                    }
                    const auto& [image_id, desc] = cb_descs[index];
                    if (!image_id) {
                        return std::nullopt;
                    }
                    const auto& image = texture_cache.GetImage(image_id);
                    const auto extent = image.GetHostExtent();
                    const auto mip = desc.view_info.range.base.level;
                    if (mip >= 32) {
                        return std::nullopt;
                    }
                    // This is the existing bound-view request, not a guest-register
                    // reinterpretation or proof of a driver-supported format substitution.
                    command.key.colors[index] = {
                        true, u32(desc.view_info.format), std::max(extent.width >> mip, 1u),
                        std::max(extent.height >> mip, 1u), desc.view_info.range.extent.layers,
                        image.image_uid};
                    command.key.has_presentable_color |= image.usage.vo_surface;
                }
                if (db_desc.first && state.depth_stencil_attachment.image_view) {
                    const auto& desc = db_desc.second;
                    const auto& image = texture_cache.GetImage(db_desc.first);
                    const auto extent = image.GetHostExtent();
                    const auto mip = desc.view_info.range.base.level;
                    if (mip >= 32) {
                        return std::nullopt;
                    }
                    const auto& depth = state.depth_stencil_attachment;
                    command.key.depth = {
                        true,
                        image.image_uid,
                        u32(desc.view_info.format),
                        std::max(extent.width >> mip, 1u),
                        std::max(extent.height >> mip, 1u),
                        desc.view_info.range.extent.layers,
                        depth.has_depth,
                        depth.has_stencil,
                        depth.depth_clear,
                        depth.stencil_clear,
                    };
                }
                return command;
            },
            [](std::size_t ordinal, u64 time, const VideoCore::EmittedColorCommand& row) {
                std::string colors;
                for (std::size_t index = 0; index < row.key.colors.size(); ++index) {
                    const auto& target = row.key.colors[index];
                    if (target.bound) {
                        colors += fmt::format("{}:{}:{}x{}x{}#{};", index, target.format,
                                              target.width, target.height, target.layers,
                                              target.resource_id);
                    }
                }
                std::string depth;
                if (row.key.depth.bound) {
                    const auto& target = row.key.depth;
                    depth = fmt::format("{}:{}x{}x{}#{} d{} s{} dc{} sc{}", target.format,
                                        target.width, target.height, target.layers, target.resource_id,
                                        target.has_depth, target.has_stencil, target.depth_clear,
                                        target.stencil_clear);
                }
                const auto kind = row.key.kind;
                const bool direct = kind == VideoCore::ColorCommandKind::Direct ||
                                    kind == VideoCore::ColorCommandKind::Indexed;
                LOG_WARNING(Render_Vulkan,
                            "ColorCommandLedgerV1 row={} renderer_ms={} vs={:016x} fs={:016x} "
                            "pipeline_key={:016x} fs_present={} kind={} direct_counts_known={} "
                            "elements={} instances={} "
                            "api_command_count={} api_max_count={} presentable={} colors=[{}] "
                            "depth=[{}]",
                            ordinal, time, row.key.vertex_hash, row.key.fragment_hash,
                            row.key.pipeline_key_hash, row.key.has_fragment, u32(kind), direct,
                            row.elements, row.instances, row.command_count, row.max_command_count,
                            row.stride, row.key.has_presentable_color, colors, depth);
            },
            LogColorLedgerEnd);
    } catch (...) {
        color_command_ledger.Disable();
    }
}

bool Rasterizer::HasPresentableRenderDocColorAttachment(const RenderState& state) {
    for (u32 index = 0; index < state.num_color_attachments; ++index) {
        if (!state.color_attachments[index].image_view || !cb_descs[index].first) {
            continue;
        }
        if (texture_cache.GetImage(cb_descs[index].first).usage.vo_surface) {
            return true;
        }
    }
    return false;
}

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
    buffer_cache.DrainReadyFaults();

    if (!FilterDraw()) {
        return;
    }

    const auto& regs = liverpool->regs;
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);
    RefreshImageDescriptorLayouts();

    VideoCore::PrepareDrawBuffersThenBindCommandState(
        [&] { return buffer_cache.PrepareVertexBuffers(*pipeline, buffer_barriers); },
        [&]() -> std::optional<VideoCore::BufferCache::PreparedIndexBuffer> {
            if (is_indexed) {
                return buffer_cache.PrepareIndexBuffer(index_offset, buffer_barriers);
            }
            return std::nullopt;
        },
        [] {},
        [&](const auto& prepared_vertex) { buffer_cache.BindVertexBuffers(prepared_vertex); },
        [&](const auto& prepared_index) {
            if (prepared_index) {
                buffer_cache.BindIndexBuffer(*prepared_index);
            }
        });

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Graphics, pipeline->Handle());

    if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    scheduler.MarkGuestWork(std::nullopt, pipeline->GetStage(Shader::LogicalStage::Vertex).pgm_hash,
                            VideoCore::GuestWorkCaptureRequiresColorAttachment() &&
                                VideoCore::HasBoundRenderDocColorAttachment(
                                    state.color_attachments, state.num_color_attachments),
                            VideoCore::GuestWorkCaptureRequiresColorAttachment() &&
                                HasPresentableRenderDocColorAttachment(state));
    if (color_command_ledger.Enabled()) {
        RecordEmittedColorCommand(
            *pipeline, state,
            {.key = {.kind = is_indexed ? VideoCore::ColorCommandKind::Indexed
                                        : VideoCore::ColorCommandKind::Direct},
             .elements = regs.num_indices,
             .instances = regs.num_instances.NumInstances()});
    }
    DebugState.IncDrawCall();

    ResetBindings();
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();
    buffer_cache.DrainReadyFaults();

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
    RefreshImageDescriptorLayouts();

    VideoCore::Buffer* buffer{};
    u32 base{};
    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    VideoCore::PrepareDrawBuffersThenBindCommandState(
        [&] { return buffer_cache.PrepareVertexBuffers(*pipeline, buffer_barriers); },
        [&]() -> std::optional<VideoCore::BufferCache::PreparedIndexBuffer> {
            if (is_indexed) {
                return buffer_cache.PrepareIndexBuffer(0, buffer_barriers);
            }
            return std::nullopt;
        },
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
                if (auto barrier =
                        count_buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                                 vk::PipelineStageFlagBits2::eDrawIndirect)) {
                    buffer_barriers.emplace_back(*barrier);
                }
            }
        },
        [&](const auto& prepared_vertex) { buffer_cache.BindVertexBuffers(prepared_vertex); },
        [&](const auto& prepared_index) {
            if (prepared_index) {
                buffer_cache.BindIndexBuffer(*prepared_index);
            }
        });

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    // We can safely ignore both SGPR UD indices and results of fetch shader parsing, as vertex and
    // instance offsets will be automatically applied by Vulkan from indirect args buffer.

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Graphics, pipeline->Handle());

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

    scheduler.MarkGuestWork(std::nullopt, pipeline->GetStage(Shader::LogicalStage::Vertex).pgm_hash,
                            VideoCore::GuestWorkCaptureRequiresColorAttachment() &&
                                VideoCore::HasBoundRenderDocColorAttachment(
                                    state.color_attachments, state.num_color_attachments),
                            VideoCore::GuestWorkCaptureRequiresColorAttachment() &&
                                HasPresentableRenderDocColorAttachment(state));
    if (color_command_ledger.Enabled()) {
        const auto kind = count_address != 0
                              ? is_indexed ? VideoCore::ColorCommandKind::IndexedIndirectCount
                                           : VideoCore::ColorCommandKind::IndirectCount
                          : is_indexed ? VideoCore::ColorCommandKind::IndexedIndirect
                                       : VideoCore::ColorCommandKind::Indirect;
        RecordEmittedColorCommand(*pipeline, state,
                                  {.key = {.kind = kind},
                                   .command_count = count_address == 0 ? max_count : 0,
                                   .max_command_count = count_address != 0 ? max_count : 0,
                                   .stride = stride});
    }
    ResetBindings();
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();
    buffer_cache.DrainReadyFaults();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Compute, pipeline->Handle());
    GetComputeDependencyProbe().Dispatch(
        [&] { cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z); },
        [&](const vk::MemoryBarrier2& barrier) {
            cmdbuf.pipelineBarrier2(
                vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
        },
        [] {
            LOG_WARNING(Render_Vulkan,
                        "DIAGNOSTIC compute dependency probe enabled: global same-queue barriers "
                        "after emitted compute commands");
        });
    scheduler.MarkGuestWork(cs.pgm_hash);
    DebugState.IncDispatch();

    ResetBindings();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();
    buffer_cache.DrainReadyFaults();

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
    GetComputeDependencyProbe().Dispatch(
        [&] { cmdbuf.dispatchIndirect(buffer->Handle(), base); },
        [&](const vk::MemoryBarrier2& barrier) {
            cmdbuf.pipelineBarrier2(
                vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
        },
        [] {
            LOG_WARNING(Render_Vulkan,
                        "DIAGNOSTIC compute dependency probe enabled: global same-queue barriers "
                        "after emitted compute commands");
        });
    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    scheduler.MarkGuestWork(cs.pgm_hash);
    DebugState.IncDispatch();

    ResetBindings();
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::FlushWithGpuCompletion(Common::UniqueFunction<void, u64>&& on_submitted,
                                       Common::UniqueFunction<void>&& wake_owner) {
    scheduler.FlushWithGpuCompletion(std::move(on_submitted), std::move(wake_owner));
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    buffer_cache.DrainReadyFaults();
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
    descriptor_image_layout_bindings.clear();

    size_t required_buffer_infos{};
    size_t required_image_infos{};
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        required_buffer_infos += stage->buffers.size();
        required_image_infos += stage->samplers.size();
        for (const auto& image : stage->images) {
            required_image_infos += image.NumBindings(*stage);
        }
    }
    // Descriptor writes retain pointers into these arrays until BindResources returns. Reserve
    // the complete pipeline requirement before any stage publishes those pointers.
    buffer_infos.reserve(required_buffer_infos);
    image_infos.reserve(required_image_infos);

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        stage->PushUd(binding, push_data);
        PrepareAddr64PointerResidency(*stage);
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
    if (instance.IsMaintenance8Supported() ||
        src_image.info.props.is_depth == dst_image.info.props.is_depth) {
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

    for (const auto& desc : stage.buffers) {
        const auto vsharp = desc.GetSharp(stage);
        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0 &&
            memory->IsValidMapping(vsharp.base_address)) {
            const u64 binding_size = desc.GetBindingSize(vsharp);
            const u64 bounded_size = memory->ClampRangeSize(vsharp.base_address, binding_size);
            // The descriptor base may be a sparse hole with valid storage later in the range.
            // Keep that storage at its original offset, but do not allocate an unmapped tail or
            // an entirely unmapped descriptor merely because it lies inside the guest VMA space.
            // Retain the existing address-space bound first: a later unrelated mapped allocation
            // must not expand a deliberately oversized descriptor into a multi-gigabyte buffer.
            const u64 size = page_manager.GetGpuMappedExtent(vsharp.base_address, bounded_size);
            if (size != 0) {
                const auto buffer_id = buffer_cache.FindBuffer(vsharp.base_address, size);
                buffer_bindings.emplace_back(buffer_id, vsharp, size);
                continue;
            }
        }
        buffer_bindings.emplace_back(VideoCore::BufferId{}, vsharp, 0);
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
            const auto [vk_buffer, offset] = buffer_cache.ObtainBuffer(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id);
            const u32 offset_aligned = Common::AlignDown(offset, alignment);
            const u32 adjust = offset - offset_aligned;
            if (adjust % 4 != 0) {
                LOG_WARNING(Render_Vulkan, "Buffer binding {} in shader {:#x} isn't dword aligned",
                            i, stage.pgm_hash);
            }
            push_data.AddOffset(binding.buffer, adjust);
            buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
            if (auto barrier =
                    vk_buffer->GetBarrier(VideoCore::ShaderBufferAccess(desc.is_written),
                                          vk::PipelineStageFlagBits2::eAllCommands)) {
                buffer_barriers.emplace_back(*barrier);
            }
            if (desc.is_written && desc.is_formatted) {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }
        }

        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        const auto descriptor_binding = binding.TryAddBufferBinding();
        ASSERT_MSG(descriptor_binding.has_value(), "Descriptor binding index overflow");
        set_write.dstBinding = *descriptor_binding;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageBuffer : vk::DescriptorType::eUniformBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    // For loading/storing to explicit mip levels, when no native instruction support, bind an array
    // of descriptors consecutively, 1 for each mip level. The shader can index this with LOD
    // operand.
    // This array holds the descriptor count of each consecutive descriptor array.
    // This is currently always 1 for anything other than mip fallback arrays.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;
    boost::container::small_vector<std::pair<bool, bool>, 8> image_native_extent_requirements;
    const auto append_invalid_image_binding = [&](bool is_storage) {
        auto& [image_id, desc] =
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
        desc.type = is_storage ? VideoCore::TextureCache::BindingType::Storage
                               : VideoCore::TextureCache::BindingType::Texture;
        image_native_extent_requirements.emplace_back(false, false);
    };

    for (const auto& image_desc : stage.images) {
        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(stage);
        ASSERT(num_bindings != 0);
        const bool is_dynamic_table = image_desc.dynamic_descriptor_count != 0;

        for (u32 i = 0; i < num_bindings; ++i) {
            if (is_dynamic_table && !image_desc.IsTableEntryCompatible(stage, i)) {
                append_invalid_image_binding(image_desc.is_written);
                continue;
            }

            const auto tsharp = image_desc.GetSharp(stage, i);
            if (texture_cache.IsMeta(tsharp.Address())) {
                LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
            }

            const auto data_fmt = tsharp.GetDataFmt();
            const auto num_fmt = tsharp.GetNumberFmt();
            if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
                append_invalid_image_binding(image_desc.is_written);
                continue;
            }

            // IsValidGpuMapping checks address width, not backing residency.
            const bool address_40bit = memory->IsValidGpuMapping(tsharp.Address(), 0);
            const auto rejection_reasons = TextureRejectionReasons(
                address_40bit, magic_enum::enum_contains(data_fmt),
                magic_enum::enum_contains(num_fmt),
                // Storage views never consume DST_SEL. Sampled views must reject reserved
                // selectors before Binding constructs ImageViewInfo/ComponentMapping.
                image_desc.is_written ||
                    TextureComponentSelectorsValid(
                        {static_cast<u32>(tsharp.dst_sel_x), static_cast<u32>(tsharp.dst_sel_y),
                         static_cast<u32>(tsharp.dst_sel_z), static_cast<u32>(tsharp.dst_sel_w)}));
            const auto build_texture_observation = [&] {
                TextureRejectionObservation observation{
                    .shader_hash = stage.pgm_hash,
                    .stage = static_cast<u32>(stage.stage),
                    .logical_stage = static_cast<u32>(stage.l_stage),
                    .sharp_index = image_desc.sharp_idx,
                    .array_index = i,
                    .descriptor_count = num_bindings,
                    .descriptor_stride_dw = image_desc.dynamic_descriptor_stride_dw,
                    .reasons = rejection_reasons,
                    .is_r128 = image_desc.is_r128,
                    // Logical GPU mapping includes sparse placeholders, not proof of backing.
                    .base_byte_logical_gpu_mapped =
                        address_40bit && page_manager.IsGpuMapped(tsharp.Address(), 1),
                };
                static_assert(sizeof(observation.words) == sizeof(tsharp));
                std::memcpy(observation.words.data(), &tsharp, sizeof(tsharp));
                observation.roots =
                    CaptureSrtRootProvenance(stage.flattened_ud_buf, stage.user_data.size());
                return observation;
            };
            if (rejection_reasons != 0) {
                LOG_WARNING(Render_Vulkan,
                            "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                            "data_format={}, num_format={}",
                            tsharp.Address(), tsharp.pitch, tsharp.width,
                            static_cast<u32>(data_fmt), static_cast<u32>(num_fmt));
                static TextureRejectionDiagnostics rejection_diagnostics{
                    TextureRejectionDiagnosticsEnabled()};
                if (const auto rejected =
                        rejection_diagnostics.Observe(build_texture_observation)) {
                    const auto& words = rejected->words;
                    const auto& roots = rejected->roots;
                    const auto& ud = roots.words;
                    LOG_WARNING(
                        Render_Vulkan,
                        "T# diagnostic stage={} logical_stage={} shader={:#x} sharp={} entry={} "
                        "count={} stride_dw={} r128={} reasons={:#x} address_40bit={} "
                        "base_byte_logical_gpu_mapped={} type={} pow2={} tile={} "
                        "base_mip={} last_mip={} height_raw={} depth_raw={} "
                        "base_array={} last_array={} raw=[{:08x} {:08x} {:08x} {:08x} "
                        "{:08x} {:08x} {:08x} {:08x}] "
                        "ud_declared={} flat_dwords={} ud_copied={} ud_full_prefix={} "
                        "captured_ud=[{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} "
                        "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}]",
                        rejected->stage, rejected->logical_stage, rejected->shader_hash,
                        rejected->sharp_index, rejected->array_index, rejected->descriptor_count,
                        rejected->descriptor_stride_dw, rejected->is_r128, rejected->reasons,
                        address_40bit, rejected->base_byte_logical_gpu_mapped,
                        static_cast<u32>(tsharp.type), static_cast<u32>(tsharp.pow2pad),
                        static_cast<u32>(tsharp.tiling_index), static_cast<u32>(tsharp.base_level),
                        static_cast<u32>(tsharp.last_level), static_cast<u32>(tsharp.height),
                        static_cast<u32>(tsharp.depth), static_cast<u32>(tsharp.base_array),
                        static_cast<u32>(tsharp.last_array), words[0], words[1], words[2], words[3],
                        words[4], words[5], words[6], words[7], roots.declared_ud_dwords,
                        roots.flat_dwords, roots.copied_dwords, roots.has_full_prefix,
                        ud[0], ud[1], ud[2], ud[3], ud[4], ud[5], ud[6], ud[7],
                        ud[8], ud[9], ud[10], ud[11], ud[12], ud[13], ud[14], ud[15]);
                }
                append_invalid_image_binding(image_desc.is_written);
                continue;
            }

            auto& [image_id, desc] = image_bindings.emplace_back(
                std::piecewise_construct, std::tuple{}, std::tuple{tsharp, image_desc});
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

            // Only failure logging consumes this context; accepted entries are not logged.
            ScopedTextureDiagnosticContext diagnostic_context{
                TextureRejectionDiagnosticsEnabled(), build_texture_observation};
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

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((image.binding.force_general || image.binding.is_target) &&
                !image.info.props.is_depth) {
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
            descriptor_image_layout_bindings.emplace_back(static_cast<u32>(image_infos.size() - 1),
                                                          image_id.index);
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        const auto descriptor_binding = binding.TryAddImageBinding(array_size);
        ASSERT_MSG(descriptor_binding.has_value(),
                   "Invalid descriptor count or binding index overflow");
        set_write.dstBinding = *descriptor_binding;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
    }

    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(stage);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        if (TextureRejectionDiagnosticsEnabled()) {
            const auto reasons = SamplerRejectionReasons(ssharp.raw0, ssharp.raw1);
            if (reasons != 0) {
                static SamplerRejectionDiagnostics sampler_diagnostics{true};
                if (const auto observed = sampler_diagnostics.Observe([&] {
                        SamplerRejectionObservation observation{
                            .shader_hash = stage.pgm_hash,
                            .stage = static_cast<u32>(stage.stage),
                            .logical_stage = static_cast<u32>(stage.l_stage),
                            .binding = binding.unified,
                            .sharp_index = sampler.sharp_idx,
                            .reasons = reasons,
                            .is_inline = sampler.is_inline_sampler != 0,
                            .disable_aniso = sampler.disable_aniso != 0,
                            .host_anisotropy_supported = instance.IsAnisotropicFilteringSupported(),
                            .associated_image = sampler.associated_image,
                        };
                        // Both S# copies and the optional T# come from owned stage data.
                        const auto raw_sampler = sampler.GetSharp(stage);
                        std::memcpy(observation.words.data(), &raw_sampler, sizeof(raw_sampler));
                        std::memcpy(observation.effective_words.data(), &ssharp, sizeof(ssharp));
                        observation.roots =
                            CaptureSrtRootProvenance(stage.flattened_ud_buf, stage.user_data.size());
                        if (!sampler.is_inline_sampler &&
                            sampler.associated_image < stage.images.size()) {
                            const auto& image_desc = stage.images[sampler.associated_image];
                            observation.associated_sharp_index = image_desc.sharp_idx;
                            const u32 dwords = image_desc.is_r128 ? 4U : 8U;
                            if (image_desc.sharp_idx <= stage.flattened_ud_buf.size() &&
                                dwords <= stage.flattened_ud_buf.size() - image_desc.sharp_idx) {
                                observation.associated_descriptor_dwords = dwords;
                                std::memcpy(observation.associated_words.data(),
                                            stage.flattened_ud_buf.data() + image_desc.sharp_idx,
                                            dwords * sizeof(u32));
                            }
                        }
                        return observation;
                    })) {
                    const auto& words = observed->words;
                    const auto& effective = observed->effective_words;
                    const auto& image = observed->associated_words;
                    const auto& roots = observed->roots;
                    const auto& ud = roots.words;
                    LOG_WARNING(
                        Render_Vulkan,
                        "S# diagnostic stage={} logical_stage={} shader={:#x} binding={} "
                        "sharp={} inline={} disable_aniso={} host_anisotropy={} reasons={:#x} "
                        "associated_image={} associated_sharp={} associated_dwords={} "
                        "raw=[{:08x} {:08x} {:08x} {:08x}] "
                        "effective=[{:08x} {:08x} {:08x} {:08x}] "
                        "associated_raw=[{:08x} {:08x} {:08x} {:08x} "
                        "{:08x} {:08x} {:08x} {:08x}] "
                        "ud_declared={} flat_dwords={} ud_copied={} ud_full_prefix={} "
                        "captured_ud=[{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} "
                        "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}]",
                        observed->stage, observed->logical_stage, observed->shader_hash,
                        observed->binding, observed->sharp_index, observed->is_inline,
                        observed->disable_aniso, observed->host_anisotropy_supported,
                        observed->reasons, observed->associated_image,
                        observed->associated_sharp_index, observed->associated_descriptor_dwords,
                        words[0], words[1], words[2], words[3], effective[0], effective[1],
                        effective[2], effective[3], image[0], image[1], image[2], image[3],
                        image[4], image[5], image[6], image[7], roots.declared_ud_dwords,
                        roots.flat_dwords, roots.copied_dwords, roots.has_full_prefix,
                        ud[0], ud[1], ud[2], ud[3], ud[4], ud[5], ud[6], ud[7],
                        ud[8], ud[9], ud[10], ud[11], ud[12], ud[13], ud[14], ud[15]);
                }
            }
        }
        const auto vk_sampler = texture_cache.GetSampler(ssharp, liverpool->regs.ta_bc_base);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        const auto descriptor_binding = binding.TryAddSamplerBinding();
        ASSERT_MSG(descriptor_binding.has_value(), "Descriptor binding index overflow");
        set_write.dstBinding = *descriptor_binding;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

void Rasterizer::RefreshImageDescriptorLayouts() {
    RefreshDescriptorImageLayouts(
        image_infos, descriptor_image_layout_bindings, [this](const u32 image_index) {
            return texture_cache.GetImage(VideoCore::ImageId{image_index}).backing->state.layout;
        });
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
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
            attachment_feedback_loop = true;
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
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        image.Transit(new_layout,
                      vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                          vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                      desc.view_info.range);

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
    page_manager.NotifyCpuWrite(addr, size);
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

void Rasterizer::ProcessDownloadImages() {
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    return page_manager.IsGpuMapped(addr, size);
}

void Rasterizer::MapMemory(VAddr addr, u64 size, VideoCore::GpuMappingKind kind) {
    page_manager.OnGpuMap(addr, size, kind);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size, VideoCore::GpuMappingKind kind) {
    // A coherent shader-resource snapshot may currently own page watchers for this mapping.
    // Invalidate it and wait for any in-flight private page copy before the VMM can remove bytes.
    page_manager.PrepareForGpuUnmap(addr, size);
    // Keep configured readbacks before backing removal, but do not take CPU-write
    // before-images: this range may already be a free/reserved or sparse mapping.
    buffer_cache.InvalidateMemory(addr, size, VideoCore::MemoryInvalidationReason::MappingChange);

    // Images may still be referenced by commands being recorded on the GPU thread. Queue their
    // eviction in the same command stream so it cannot race resource lookup or recording. This
    // must remain non-blocking: guest VMM threads are allowed to unmap while GPU work runs.
    liverpool->SendCommand<false>([this, addr, size] { texture_cache.UnmapMemory(addr, size); });
    page_manager.OnGpuUnmap(addr, size, kind);
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
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
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
