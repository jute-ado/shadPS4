// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "common/alignment.h"
#include "common/debug.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/addr64_pointer_residency.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/index_draw_state.h"
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
#include "video_core/renderer_vulkan/utility_buffer_reservation.h"
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

static_assert(
    VideoCore::IndirectUserDataBankOffsets ==
    std::array<u32, 6>{
        offsetof(AmdGpu::Regs, ps_program.user_data) / sizeof(u32) - AmdGpu::Regs::ShRegWordOffset,
        offsetof(AmdGpu::Regs, vs_program.user_data) / sizeof(u32) - AmdGpu::Regs::ShRegWordOffset,
        offsetof(AmdGpu::Regs, gs_program.user_data) / sizeof(u32) - AmdGpu::Regs::ShRegWordOffset,
        offsetof(AmdGpu::Regs, es_program.user_data) / sizeof(u32) - AmdGpu::Regs::ShRegWordOffset,
        offsetof(AmdGpu::Regs, hs_program.user_data) / sizeof(u32) - AmdGpu::Regs::ShRegWordOffset,
        offsetof(AmdGpu::Regs, ls_program.user_data) / sizeof(u32) - AmdGpu::Regs::ShRegWordOffset,
    });
static_assert(AmdGpu::NUM_USER_DATA == 16);

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

static void LogColorAttachmentProvenanceEnd(
    const VideoCore::ColorAttachmentProvenanceSummary& end) {
    const auto reason = end.reason == VideoCore::ColorAttachmentProvenanceEnd::Window ? "window"
                        : end.reason == VideoCore::ColorAttachmentProvenanceEnd::Capacity
                            ? "capacity"
                            : "teardown";
    LOG_WARNING(Render_Vulkan,
                "ColorAttachmentProvenanceV1 end reason={} renderer_ms={} matched={} completed={}",
                reason, end.elapsed_ms, end.matched_commands, end.completed_commands);
}

static std::string FormatColorAttachmentProvenanceBytes(const u8* data, u32 size) {
    std::string result{"["};
    for (u32 index = 0; index < size; ++index) {
        if (index != 0) {
            result.push_back(' ');
        }
        result += fmt::format("{:02x}", static_cast<u32>(data[index]));
    }
    result.push_back(']');
    return result;
}

static DescriptorImageCommandContext MakeGraphicsDescriptorImageContext(
    const GraphicsPipeline& pipeline, DescriptorImageCommandKind kind) {
    const auto* fragment = pipeline.GetStages()[u32(Shader::LogicalStage::Fragment)];
    return {
        .kind = kind,
        .vertex_hash = pipeline.GetStage(Shader::LogicalStage::Vertex).pgm_hash,
        .fragment_hash = fragment ? fragment->pgm_hash : 0,
    };
}

static void LogDrawDropDiagnosticsEnd(const DrawDropDiagnosticsSummary& summary,
                                     const u64 elapsed_ms) {
    LOG_WARNING(Render_Vulkan,
                "DrawDropDiagnosticsV1 end renderer_ms={} total={} max_events={} saturated={}",
                elapsed_ms, summary.total, DrawDropDiagnostics::MaxEvents, summary.saturated);
    for (u32 reason_index = 0; reason_index < DrawDropDiagnosticsSummary::ReasonCount;
         ++reason_index) {
        const auto reason = static_cast<DrawDropReason>(reason_index);
        for (u32 command_index = 0; command_index < DrawDropDiagnosticsSummary::CommandCount;
             ++command_index) {
            const auto command = static_cast<DescriptorImageCommandKind>(command_index);
            const auto count = summary.Count(reason, command);
            if (count == 0) {
                continue;
            }
            LOG_WARNING(Render_Vulkan,
                        "DrawDropDiagnosticsV1 count reason={} command={} count={} "
                        "behavior=observation_only",
                        DrawDropReasonName(reason), DescriptorImageCommandKindName(command),
                        count);
        }
    }
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
    const auto descriptor_finalization_policy = DescriptorImageFinalizationReceiptPolicy::Parse(
        environment("SHADPS4_DESCRIPTOR_IMAGE_FINALIZATION_RECEIPT"),
        environment("SHADPS4_DESCRIPTOR_IMAGE_FINALIZATION_RECEIPT_START_MS"),
        environment("SHADPS4_DESCRIPTOR_IMAGE_FINALIZATION_RECEIPT_DURATION_MS"));
    if (!descriptor_finalization_policy.valid) {
        LOG_WARNING(Render_Vulkan,
                    "DescriptorImageFinalizationReceiptV1 refused invalid launch policy");
    } else if (descriptor_finalization_policy.enabled) {
        descriptor_image_finalization_receipt =
            DescriptorImageFinalizationReceipt{descriptor_finalization_policy};
        descriptor_finalization_origin = std::chrono::steady_clock::now();
        LOG_WARNING(Render_Vulkan,
                    "DescriptorImageFinalizationReceiptV1 armed start_ms={} duration_ms={} "
                    "capacity=32 clock=renderer_construction behavior=observation_only",
                    descriptor_finalization_policy.start_ms,
                    descriptor_finalization_policy.duration_ms);
    }
    const auto draw_drop_policy = DrawDropDiagnosticsPolicy::Parse(
        environment("SHADPS4_DRAW_DROP_DIAGNOSTICS"),
        environment("SHADPS4_DRAW_DROP_DIAGNOSTICS_START_MS"),
        environment("SHADPS4_DRAW_DROP_DIAGNOSTICS_DURATION_MS"));
    if (!draw_drop_policy.valid) {
        LOG_WARNING(Render_Vulkan, "DrawDropDiagnosticsV1 refused invalid launch policy");
    } else if (draw_drop_policy.enabled) {
        draw_drop_diagnostics = DrawDropDiagnostics{draw_drop_policy};
        draw_drop_diagnostics_origin = std::chrono::steady_clock::now();
        LOG_WARNING(Render_Vulkan,
                    "DrawDropDiagnosticsV1 armed start_ms={} duration_ms={} max_events={} "
                    "clock=renderer_construction behavior=observation_only",
                    draw_drop_policy.start_ms, draw_drop_policy.duration_ms,
                    DrawDropDiagnostics::MaxEvents);
    }
    color_attachment_provenance_policy = VideoCore::ColorAttachmentProvenancePolicy::Parse(
        environment("SHADPS4_COLOR_ATTACHMENT_PROVENANCE"),
        environment("SHADPS4_COLOR_ATTACHMENT_PROVENANCE_START_MS"),
        environment("SHADPS4_COLOR_ATTACHMENT_PROVENANCE_DURATION_MS"),
        environment("SHADPS4_COLOR_ATTACHMENT_PROVENANCE_POINTS"),
        environment("SHADPS4_COLOR_ATTACHMENT_PROVENANCE_SELECTORS"),
        environment("SHADPS4_COLOR_ATTACHMENT_PROVENANCE_CAPACITY"));
    if (!color_attachment_provenance_policy.valid) {
        LOG_WARNING(Render_Vulkan, "ColorAttachmentProvenanceV1 refused invalid launch policy");
        color_attachment_provenance_policy = {};
    } else if (color_attachment_provenance_policy.enabled) {
        try {
            constexpr u64 max_targets = AmdGpu::NUM_COLOR_BUFFERS;
            constexpr u64 max_samples = VideoCore::ColorAttachmentProvenancePolicy::MaxPoints;
            constexpr u64 max_commands =
                VideoCore::ColorAttachmentProvenancePolicy::MaxCommandCapacity;
            const u64 atom_size = std::max<u64>(instance.NonCoherentAtomSize(), 4);
            const u64 buffer_size = Common::AlignUp(
                max_commands * max_targets * max_samples * 2 * atom_size, atom_size);
            color_attachment_provenance_receipt =
                VideoCore::ColorAttachmentProvenanceReceipt{color_attachment_provenance_policy};
            color_attachment_provenance_slots =
                VideoCore::ColorAttachmentProvenanceSlotAllocator{buffer_size, atom_size};
            color_attachment_provenance_buffer = std::make_shared<VideoCore::Buffer>(
                instance, scheduler, VideoCore::MemoryUsage::Download, 0,
                vk::BufferUsageFlagBits::eTransferDst, buffer_size);
            if (color_attachment_provenance_buffer->mapped_data.empty()) {
                LOG_WARNING(
                    Render_Vulkan,
                    "ColorAttachmentProvenanceV1 disabled: download storage was not mapped");
                color_attachment_provenance_buffer.reset();
                color_attachment_provenance_policy = {};
                color_attachment_provenance_receipt = VideoCore::ColorAttachmentProvenanceReceipt{
                    VideoCore::ColorAttachmentProvenancePolicy{}};
            } else {
                color_attachment_provenance_origin = std::chrono::steady_clock::now();
                LOG_WARNING(
                    Render_Vulkan,
                    "ColorAttachmentProvenanceV1 armed start_ms={} duration_ms={} capacity={} "
                    "points={} selectors={} atom_size={} behavior=observation_only",
                    color_attachment_provenance_policy.start_ms,
                    color_attachment_provenance_policy.duration_ms,
                    color_attachment_provenance_policy.command_capacity,
                    color_attachment_provenance_policy.point_count,
                    color_attachment_provenance_policy.selector_count, atom_size);
            }
        } catch (...) {
            LOG_WARNING(Render_Vulkan,
                        "ColorAttachmentProvenanceV1 disabled after storage allocation failure");
            color_attachment_provenance_buffer.reset();
            color_attachment_provenance_policy = {};
            color_attachment_provenance_receipt = VideoCore::ColorAttachmentProvenanceReceipt{
                VideoCore::ColorAttachmentProvenancePolicy{}};
        }
    }
    emitted_color_vertex_receipt = VideoCore::EmittedColorVertexReceipt::Parse(
        environment("SHADPS4_EMITTED_COLOR_VERTEX_HASH"));
    if (!emitted_color_vertex_receipt.Valid()) {
        LOG_WARNING(Render_Vulkan, "EmittedColorVertexReceiptV1 refused invalid exact vertex hash");
    } else if (emitted_color_vertex_receipt.Enabled()) {
        LOG_WARNING(Render_Vulkan, "EmittedColorVertexReceiptV1 armed command_emission_only=true");
    }
    const auto index_provenance_policy = VideoCore::IndexBindingProvenancePolicy::Parse(
        environment("SHADPS4_INDEX_BINDING_PROVENANCE"),
        environment("SHADPS4_INDEX_BINDING_PROVENANCE_START_FRAME"),
        environment("SHADPS4_INDEX_BINDING_PROVENANCE_VERTEX_HASH"));
    index_binding_provenance = VideoCore::IndexBindingProvenanceReceipt{index_provenance_policy};
    if (!index_provenance_policy.valid) {
        LOG_WARNING(Render_Vulkan, "IndexBindingProvenanceV1 refused invalid launch policy");
    } else if (index_provenance_policy.enabled) {
        LOG_WARNING(Render_Vulkan,
                    "IndexBindingProvenanceV1 armed start_frame={} capacity=16 clock=guest_flip "
                    "exact_vertex_filter={} behavior=register_observation_only",
                    index_provenance_policy.start_frame,
                    index_provenance_policy.vertex_hash.has_value());
    }
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
    if (draw_drop_diagnostics.Armed()) {
        try {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - draw_drop_diagnostics_origin)
                                     .count();
            LogDrawDropDiagnosticsEnd(
                draw_drop_diagnostics.Close(static_cast<u64>(elapsed)), static_cast<u64>(elapsed));
        } catch (...) {
            // Logging is best-effort and cannot make owner teardown throw.
        }
    }
    if (color_attachment_provenance_receipt.Enabled()) {
        try {
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - color_attachment_provenance_origin)
                    .count();
            if (const auto end =
                    color_attachment_provenance_receipt.Close(static_cast<u64>(elapsed))) {
                LogColorAttachmentProvenanceEnd(*end);
            }
        } catch (...) {
            // Logging is best-effort and cannot make owner teardown throw.
        }
    }
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

void Rasterizer::ObserveDrawDrop(
    const DrawDropReason reason, const DescriptorImageCommandContext& context) noexcept {
    if (!draw_drop_diagnostics.Enabled()) {
        return;
    }
    try {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - draw_drop_diagnostics_origin)
                                 .count();
        static_cast<void>(draw_drop_diagnostics.Observe(static_cast<u64>(elapsed), reason, context));
    } catch (...) {
        // Diagnostics must not change draw admission or make the owner path throw.
    }
}

void Rasterizer::RecordIndexBindingProvenance(
    const GraphicsPipeline& pipeline, bool indirect, u32 index_offset,
    VideoCore::IndirectDrawRegisterLocations register_locations) noexcept {
    if (!index_binding_provenance.Enabled()) {
        return;
    }
    try {
        const auto frame = DebugState.GetFrameNum();
        const auto vertex_hash = pipeline.GetStage(Shader::LogicalStage::Vertex).pgm_hash;
        const auto row = index_binding_provenance.Observe(frame, vertex_hash);
        if (!row) {
            return;
        }
        const auto& regs = liverpool->regs;
        const u32 index_size = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16
                                   ? sizeof(u16)
                                   : sizeof(u32);
        const auto preparation_count =
            indirect ? AmdGpu::CurrentIndexPreparationCount(regs) : regs.num_indices;
        const auto span =
            VideoCore::DescribeIndexBindingPreparation(preparation_count, index_size);
        const bool userdata_applicable = indirect && register_locations.present;
        const auto read_cpu_register = [&](u32 stage, u32 sgpr) {
            return regs.ProgramForStage(stage)->user_data[sgpr];
        };
        const auto base_vertex = VideoCore::ObserveIndirectUserDataRegister(
            register_locations.base_vtx_loc, userdata_applicable, read_cpu_register);
        const auto start_instance = VideoCore::ObserveIndirectUserDataRegister(
            register_locations.start_inst_loc, userdata_applicable, read_cpu_register);
        const auto draw_index = VideoCore::ObserveIndirectUserDataRegister(
            register_locations.draw_index_loc,
            userdata_applicable && register_locations.draw_index_present &&
                register_locations.draw_index_enabled,
            read_cpu_register);
        LOG_WARNING(
            Render_Vulkan,
            "IndexBindingProvenanceV1 row={} frame={} clock=guest_flip phase=before_prepare "
            "vertex={:#x} indirect={} base={:#x} index_type_raw={:#x} index_size={} "
            "index_offset={} current_index_buffer_size={} current_max_index_size={} "
            "current_num_indices={} "
            "requested_bytes_wide={} prepared_bytes_u32={} span_truncated={} "
            "indirect_first_index_known=false register_count_is_executed_count=false "
            "indirect_userdata_applicable={} base_vtx_loc={:#x} base_vtx_cpu_known={} "
            "base_vtx_stage={} base_vtx_sgpr={} base_vtx_cpu_value={:#x} "
            "start_inst_loc={:#x} start_inst_cpu_known={} start_inst_stage={} "
            "start_inst_sgpr={} start_inst_cpu_value={:#x} "
            "draw_index_loc_present={} draw_index_enabled={} draw_index_loc={:#x} "
            "draw_index_cpu_known={} draw_index_stage={} draw_index_sgpr={} "
            "draw_index_cpu_value={:#x} cpu_value_is_gpu_indirect_argument=false",
            *row, frame, vertex_hash, indirect, regs.index_base_address.Address<VAddr>(),
            regs.index_buffer_type.raw, index_size, index_offset, regs.index_size,
            regs.max_index_size, regs.num_indices, span.wide_bytes, span.prepared_bytes,
            span.truncated,
            userdata_applicable, register_locations.base_vtx_loc, base_vertex.value_known,
            base_vertex.stage, base_vertex.sgpr, base_vertex.value,
            register_locations.start_inst_loc, start_instance.value_known, start_instance.stage,
            start_instance.sgpr, start_instance.value, register_locations.draw_index_present,
            register_locations.draw_index_enabled, register_locations.draw_index_loc,
            draw_index.value_known, draw_index.stage, draw_index.sgpr, draw_index.value);
    } catch (...) {
        index_binding_provenance.Disable();
    }
}

void Rasterizer::RecordEmittedColorCommand(const GraphicsPipeline& pipeline,
                                           const RenderState& state,
                                           VideoCore::EmittedColorCommand command) noexcept {
    if (!color_command_ledger.Enabled() && !emitted_color_vertex_receipt.Enabled()) {
        return;
    }
    try {
        const u64 vertex_hash = pipeline.GetStage(Shader::LogicalStage::Vertex).pgm_hash;
        const bool has_color = VideoCore::HasBoundRenderDocColorAttachment(
            state.color_attachments, state.num_color_attachments);
        if (emitted_color_vertex_receipt.Observe(vertex_hash, has_color)) {
            LOG_WARNING(Render_Vulkan,
                        "EmittedColorVertexReceiptV1 matched vertex={:#x} "
                        "color_attachment=true command_emission_only=true",
                        vertex_hash);
        }
        if (!color_command_ledger.Enabled()) {
            return;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - color_ledger_origin)
                                 .count();
        color_command_ledger.Observe(
            static_cast<u64>(elapsed),
            [&]() -> std::optional<VideoCore::EmittedColorCommand> {
                if (!has_color) {
                    return std::nullopt;
                }
                command.key.vertex_hash = vertex_hash;
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
                        std::max(extent.height >> mip, 1u), desc.view_info.range.extent.layers};
                }
                return command;
            },
            [](std::size_t ordinal, u64 time, const VideoCore::EmittedColorCommand& row) {
                std::string colors;
                for (std::size_t index = 0; index < row.key.colors.size(); ++index) {
                    const auto& target = row.key.colors[index];
                    if (target.bound) {
                        colors += fmt::format("{}:{}:{}x{}x{};", index, target.format, target.width,
                                              target.height, target.layers);
                    }
                }
                const auto kind = row.key.kind;
                const bool direct = kind == VideoCore::ColorCommandKind::Direct ||
                                    kind == VideoCore::ColorCommandKind::Indexed;
                LOG_WARNING(Render_Vulkan,
                            "ColorCommandLedgerV1 row={} renderer_ms={} vs={:016x} fs={:016x} "
                            "pipeline_key={:016x} fs_present={} kind={} direct_counts_known={} "
                            "elements={} instances={} "
                            "api_command_count={} api_max_count={} stride={} colors=[{}]",
                            ordinal, time, row.key.vertex_hash, row.key.fragment_hash,
                            row.key.pipeline_key_hash, row.key.has_fragment, u32(kind), direct,
                            row.elements, row.instances, row.command_count, row.max_command_count,
                            row.stride, colors);
            },
            LogColorLedgerEnd);
    } catch (...) {
        color_command_ledger.Disable();
    }
}

std::optional<Rasterizer::ColorAttachmentProvenanceCommand>
Rasterizer::PrepareColorAttachmentProvenance(const GraphicsPipeline& pipeline,
                                             const RenderState& state) {
    if (!color_attachment_provenance_receipt.Enabled() || !color_attachment_provenance_buffer) {
        return std::nullopt;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - color_attachment_provenance_origin)
                             .count();
    const u64 pipeline_key_hash = std::hash<GraphicsPipelineKey>{}(pipeline.GetGraphicsKey());
    const auto begun =
        color_attachment_provenance_receipt.Begin(static_cast<u64>(elapsed), pipeline_key_hash);
    if (begun.end) {
        LogColorAttachmentProvenanceEnd(*begun.end);
        return std::nullopt;
    }
    if (!begun.capture) {
        return std::nullopt;
    }

    ColorAttachmentProvenanceCommand command{
        .token = *begun.capture,
        .vertex_hash = pipeline.GetStage(Shader::LogicalStage::Vertex).pgm_hash,
        .fragment_hash = pipeline.GetStages()[u32(Shader::LogicalStage::Fragment)]
                             ? pipeline.GetStages()[u32(Shader::LogicalStage::Fragment)]->pgm_hash
                             : 0,
        .pipeline_key_hash = pipeline_key_hash,
    };

    struct Candidate {
        VideoCore::ColorAttachmentProvenanceTarget target{};
        VideoCore::ImageId image_id{};
        u64 image_uid{};
        u32 attachment{};
        vk::Format base_format{vk::Format::eUndefined};
        vk::Format view_format{vk::Format::eUndefined};
    };
    std::array<Candidate, AmdGpu::NUM_COLOR_BUFFERS> candidates{};
    std::array<VideoCore::ColorAttachmentProvenanceTarget, AmdGpu::NUM_COLOR_BUFFERS> targets{};
    std::size_t candidate_count{};
    for (u32 attachment = 0; attachment < state.num_color_attachments; ++attachment) {
        if (!state.color_attachments[attachment].image_view ||
            candidate_count >= candidates.size()) {
            continue;
        }
        const auto [image_id, desc] = cb_descs[attachment];
        if (!image_id) {
            continue;
        }
        auto* image =
            texture_cache.TryGetImage(image_id, texture_cache.GetImage(image_id).image_uid);
        if (!image || !image->backing) {
            continue;
        }
        const auto& image_ci = image->backing->image.image_ci;
        const auto host_extent = image->GetHostExtent();
        if (!(image_ci.usage & vk::ImageUsageFlagBits::eTransferSrc)) {
            continue;
        }
        auto& target = targets[candidate_count];
        target.image_uid = image->image_uid;
        target.format = static_cast<u32>(desc.view_info.format);
        target.width = host_extent.width;
        target.height = host_extent.height;
        target.levels = image_ci.mipLevels;
        target.layers = image_ci.arrayLayers;
        target.mip = desc.view_info.range.base.level;
        target.layer = desc.view_info.range.base.layer;
        target.bits_per_texel = image->info.num_bits;
        target.samples = static_cast<u32>(image_ci.samples);
        target.compressed = image->info.props.is_block;
        target.depth = image->info.props.is_depth;
        target.is_3d = image->info.props.is_volume || image_ci.imageType == vk::ImageType::e3D;
        candidates[candidate_count] = {
            .target = target,
            .image_id = image_id,
            .image_uid = image->image_uid,
            .attachment = attachment,
            .base_format = image_ci.format,
            .view_format = desc.view_info.format,
        };
        ++candidate_count;
    }

    const auto unique = VideoCore::UniqueColorAttachmentProvenanceTargets(
        std::span<const VideoCore::ColorAttachmentProvenanceTarget>{targets.data(),
                                                                    candidate_count});
    for (const auto candidate_index : unique) {
        const auto& candidate = candidates[candidate_index];
        for (std::size_t point_index = 0;
             point_index < color_attachment_provenance_policy.point_count; ++point_index) {
            const auto planned = VideoCore::PlanColorAttachmentProvenanceSample(
                candidate.target, color_attachment_provenance_policy.points[point_index],
                color_attachment_provenance_slots);
            if (!planned.plan) {
                continue;
            }
            command.samples.push_back({
                .image_id = candidate.image_id,
                .image_uid = candidate.image_uid,
                .attachment = candidate.attachment,
                .base_format = candidate.base_format,
                .view_format = candidate.view_format,
                .plan = *planned.plan,
            });
        }
    }

    if (command.samples.empty()) {
        if (const auto end = color_attachment_provenance_receipt.Finish(
                command.token, static_cast<u64>(elapsed))) {
            LogColorAttachmentProvenanceEnd(*end);
        }
        return std::nullopt;
    }

    // BeginRendering may have materialized a clear. End it before sampling so the pre-value is
    // the actual attachment contents, then resume with load operations only.
    scheduler.EndRendering();
    RecordColorAttachmentProvenanceCopies(command, false);
    auto load_state = state;
    for (u32 attachment = 0; attachment < load_state.num_color_attachments; ++attachment) {
        load_state.color_attachments[attachment].is_clear = 0;
    }
    load_state.depth_stencil_attachment.depth_clear = false;
    load_state.depth_stencil_attachment.stencil_clear = false;
    scheduler.BeginRendering(load_state);
    return command;
}

void Rasterizer::CompleteColorAttachmentProvenance(ColorAttachmentProvenanceCommand command,
                                                   const RenderState& state) {
    if (!color_attachment_provenance_buffer || command.samples.empty()) {
        return;
    }
    scheduler.EndRendering();
    RecordColorAttachmentProvenanceCopies(command, true);
    auto load_state = state;
    for (u32 attachment = 0; attachment < load_state.num_color_attachments; ++attachment) {
        load_state.color_attachments[attachment].is_clear = 0;
    }
    load_state.depth_stencil_attachment.depth_clear = false;
    load_state.depth_stencil_attachment.stencil_clear = false;
    scheduler.BeginRendering(load_state);

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - color_attachment_provenance_origin)
                             .count();
    if (const auto end =
            color_attachment_provenance_receipt.Finish(command.token, static_cast<u64>(elapsed))) {
        LogColorAttachmentProvenanceEnd(*end);
    }

    u64 invalidate_begin = std::numeric_limits<u64>::max();
    u64 invalidate_end{};
    for (const auto& sample : command.samples) {
        invalidate_begin = std::min(invalidate_begin, sample.plan.pre_offset);
        invalidate_begin = std::min(invalidate_begin, sample.plan.post_offset);
        invalidate_end =
            std::max(invalidate_end, std::max(sample.plan.pre_offset, sample.plan.post_offset) +
                                         sample.plan.payload_bytes);
    }
    if (invalidate_begin == std::numeric_limits<u64>::max() || invalidate_end <= invalidate_begin) {
        return;
    }
    auto storage = color_attachment_provenance_buffer;
    scheduler.DeferOperation([storage = std::move(storage), command = std::move(command),
                              invalidate_begin, invalidate_end,
                              elapsed = static_cast<u64>(elapsed)] {
        if (!storage->InvalidateMappedRange(invalidate_begin, invalidate_end - invalidate_begin)) {
            LOG_WARNING(Render_Vulkan, "ColorAttachmentProvenanceV1 readback invalidation failed");
            return;
        }
        for (const auto& sample : command.samples) {
            const auto read = [&](const u64 offset) {
                if (offset > storage->mapped_data.size() ||
                    sample.plan.payload_bytes > storage->mapped_data.size() - offset) {
                    return std::string{"[oob]"};
                }
                return FormatColorAttachmentProvenanceBytes(storage->mapped_data.data() + offset,
                                                            sample.plan.payload_bytes);
            };
            const auto pre = read(sample.plan.pre_offset);
            const auto post = read(sample.plan.post_offset);
            bool changed{};
            if (sample.plan.pre_offset <= storage->mapped_data.size() &&
                sample.plan.post_offset <= storage->mapped_data.size() &&
                sample.plan.payload_bytes <= storage->mapped_data.size() - sample.plan.pre_offset &&
                sample.plan.payload_bytes <=
                    storage->mapped_data.size() - sample.plan.post_offset) {
                changed = std::memcmp(storage->mapped_data.data() + sample.plan.pre_offset,
                                      storage->mapped_data.data() + sample.plan.post_offset,
                                      sample.plan.payload_bytes) != 0;
            }
            LOG_WARNING(
                Render_Vulkan,
                "ColorAttachmentProvenanceV1 sample ordinal={} renderer_ms={} attachment={} "
                "image_uid={:016x} mip={} layer={} x={} y={} width={} height={} "
                "base_format={} view_format={} payload_bytes={} pre={} post={} changed={}",
                command.token.ordinal, elapsed, sample.attachment, sample.image_uid,
                sample.plan.mip, sample.plan.layer, sample.plan.x, sample.plan.y, sample.plan.width,
                sample.plan.height, vk::to_string(sample.base_format),
                vk::to_string(sample.view_format), sample.plan.payload_bytes, pre, post, changed);
        }
    });
}

void Rasterizer::RecordColorAttachmentProvenanceCopies(
    const ColorAttachmentProvenanceCommand& command, const bool post) {
    if (!color_attachment_provenance_buffer || command.samples.empty()) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    boost::container::small_vector<vk::BufferMemoryBarrier2, 32> buffer_barriers;
    for (const auto& sample : command.samples) {
        auto* image = texture_cache.TryGetImage(sample.image_id, sample.image_uid);
        if (!image || !image->backing) {
            continue;
        }
        const auto& image_ci = image->backing->image.image_ci;
        if (!(image_ci.usage & vk::ImageUsageFlagBits::eTransferSrc)) {
            continue;
        }
        const u32 subresource_index =
            sample.plan.mip * image->info.resources.layers + sample.plan.layer;
        auto previous = image->backing->state;
        if (!image->backing->subresource_states.empty() &&
            subresource_index < image->backing->subresource_states.size()) {
            previous = image->backing->subresource_states[subresource_index];
        }
        if (previous.layout == vk::ImageLayout::eUndefined) {
            continue;
        }
        const vk::ImageSubresourceRange range{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = sample.plan.mip,
            .levelCount = 1,
            .baseArrayLayer = sample.plan.layer,
            .layerCount = 1,
        };
        const vk::ImageMemoryBarrier2 to_copy{
            .srcStageMask = previous.pl_stage,
            .srcAccessMask = previous.access_mask,
            .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .oldLayout = previous.layout,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image->GetImage(),
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier2(
            vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_copy});
        const auto offset = post ? sample.plan.post_offset : sample.plan.pre_offset;
        const vk::BufferImageCopy copy{
            .bufferOffset = offset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = sample.plan.mip,
                .baseArrayLayer = sample.plan.layer,
                .layerCount = 1,
            },
            .imageOffset = {static_cast<s32>(sample.plan.x), static_cast<s32>(sample.plan.y), 0},
            .imageExtent = {1, 1, 1},
        };
        cmdbuf.copyImageToBuffer(image->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                 color_attachment_provenance_buffer->Handle(), copy);
        const vk::ImageMemoryBarrier2 restore{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = previous.pl_stage,
            .dstAccessMask = previous.access_mask,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = previous.layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image->GetImage(),
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier2(
            vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &restore});
        buffer_barriers.emplace_back(vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = color_attachment_provenance_buffer->Handle(),
            .offset = offset,
            .size = sample.plan.payload_bytes,
        });
    }
    if (!buffer_barriers.empty()) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = static_cast<u32>(buffer_barriers.size()),
            .pBufferMemoryBarriers = buffer_barriers.data(),
        });
    }
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

bool Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
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
        image_id = texture_cache.FindImage(desc);
        if (!image_id) {
            LOG_WARNING(Render_Vulkan,
                        "Dropping draw after unsupported color-target image admission cb={}", cb);
            return false;
        }
        TrackBoundImage(image_id);
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
        image_id = texture_cache.FindImage(desc);
        if (!image_id) {
            LOG_WARNING(Render_Vulkan,
                        "Dropping draw after unsupported depth-target image admission");
            return false;
        }
        TrackBoundImage(image_id);
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
    return true;
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
    if (!image_id) {
        LOG_WARNING(Render_Vulkan, "Skipping fast-clear elimination for unsupported image");
        return;
    }
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
    const auto command_kind = is_indexed ? DescriptorImageCommandKind::IndexedDraw
                                         : DescriptorImageCommandKind::DirectDraw;
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        ObserveDrawDrop(DrawDropReason::NullPipeline, {.kind = command_kind});
        return;
    }
    const auto descriptor_context = MakeGraphicsDescriptorImageContext(*pipeline, command_kind);

    if (!PrepareRenderState(pipeline)) {
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }
    if (!BindResources(pipeline)) {
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }
    RenderState state{};

    VideoCore::BufferCache::BufferBindingBatch binding_batch;
    bool index_buffer_prepare_failed = false;
    const bool draw_buffers_ready = VideoCore::PrepareDrawBuffersThenBindCommandState(
        [&] { return buffer_cache.PrepareVertexBuffers(*pipeline, buffer_barriers); },
        [&]() -> std::optional<VideoCore::BufferCache::PreparedIndexBuffer> {
            if (is_indexed) {
                RecordIndexBindingProvenance(*pipeline, false, index_offset);
                auto prepared =
                    buffer_cache.PrepareIndexBuffer(index_offset, buffer_barriers, false);
                index_buffer_prepare_failed = !prepared;
                return prepared;
            }
            return std::nullopt;
        },
        [] {}, [&] { buffer_cache.DrainReadyFaults(); },
        [&](auto& prepared_vertex, auto& prepared_index) {
            if (index_buffer_prepare_failed) {
                return false;
            }
            if (!FinalizeUtilityBufferBindings()) {
                return false;
            }
            if (!buffer_cache.FinalizeVertexBuffers(prepared_vertex, binding_batch)) {
                return false;
            }
            if (prepared_index &&
                !buffer_cache.FinalizeIndexBuffer(*prepared_index, binding_batch)) {
                return false;
            }
            if (!FinalizeBufferBindings(binding_batch)) {
                return false;
            }
            // Buffer preparation may wait for the stream buffer and execute deferred image
            // retirement. Resolve attachments only after that mutation window, but before image
            // descriptors are finalized so feedback-loop layouts remain coherent.
            if (!BeginRendering(pipeline, state) || !RebindPendingImageBindings() ||
                !FinalizeImageDescriptors(descriptor_context)) {
                ObserveDescriptorImageDrawDrop(descriptor_context);
                return false;
            }
            buffer_cache.CommitBufferBindings(binding_batch, buffer_barriers);
            return true;
        },
        [&](const auto& prepared_vertex) { buffer_cache.BindVertexBuffers(prepared_vertex); },
        [&](const auto& prepared_index) {
            if (prepared_index) {
                buffer_cache.BindIndexBuffer(*prepared_index);
            }
        });
    if (!draw_buffers_ready) {
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Graphics, pipeline->Handle());
    auto color_provenance = PrepareColorAttachmentProvenance(*pipeline, state);

    if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    if (color_provenance) {
        CompleteColorAttachmentProvenance(std::move(*color_provenance), state);
    }
    scheduler.MarkGuestWork(std::nullopt, pipeline->GetStage(Shader::LogicalStage::Vertex).pgm_hash,
                            VideoCore::GuestWorkCaptureRequiresColorAttachment() &&
                                VideoCore::HasBoundRenderDocColorAttachment(
                                    state.color_attachments, state.num_color_attachments));
    if (color_command_ledger.Enabled() || emitted_color_vertex_receipt.Enabled()) {
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
                              u32 max_count, VAddr count_address,
                              VideoCore::IndirectDrawRegisterLocations register_locations) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();
    buffer_cache.DrainReadyFaults();

    if (!FilterDraw()) {
        return;
    }

    const auto command_kind = count_address != 0
                                  ? is_indexed ? DescriptorImageCommandKind::IndexedIndirectCountDraw
                                               : DescriptorImageCommandKind::IndirectCountDraw
                              : is_indexed ? DescriptorImageCommandKind::IndexedIndirectDraw
                                           : DescriptorImageCommandKind::IndirectDraw;
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(register_locations);
    if (!pipeline) {
        ObserveDrawDrop(DrawDropReason::NullPipeline, {.kind = command_kind});
        return;
    }
    const auto descriptor_context = MakeGraphicsDescriptorImageContext(*pipeline, command_kind);

    if (!PrepareRenderState(pipeline)) {
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }
    if (!BindResources(pipeline)) {
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }
    const auto arg_range = AmdGpu::TryGetIndirectBufferRange(
        arg_address, offset, static_cast<u64>(stride) * max_count);
    const auto count_range = count_address != 0
                                 ? AmdGpu::TryGetIndirectBufferRange(count_address, 0, sizeof(u32))
                                 : std::optional<AmdGpu::IndirectBufferRange>{};
    if (!arg_range || (count_address != 0 && !count_range)) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping indirect {} draw with overflowing argument range: "
                    "address={:#x} offset={} stride={} max_count={} count_address={:#x}",
                    is_indexed ? "indexed" : "non-indexed", arg_address, offset, stride,
                    max_count, count_address);
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }
    RenderState state{};

    std::optional<VideoCore::BufferCache::PreparedBufferBinding> prepared_buffer;
    std::optional<VideoCore::BufferCache::PreparedBufferBinding> prepared_count_buffer;
    std::optional<VideoCore::BufferCache::BufferBindingEndpoint> buffer;
    std::optional<VideoCore::BufferCache::BufferBindingEndpoint> count_buffer;
    VideoCore::BufferCache::BufferBindingBatch binding_batch;
    bool index_buffer_prepare_failed = false;
    const bool draw_buffers_ready = VideoCore::PrepareDrawBuffersThenBindCommandState(
        [&] { return buffer_cache.PrepareVertexBuffers(*pipeline, buffer_barriers); },
        [&]() -> std::optional<VideoCore::BufferCache::PreparedIndexBuffer> {
            if (is_indexed) {
                RecordIndexBindingProvenance(*pipeline, true, 0, register_locations);
                auto prepared = buffer_cache.PrepareIndexBuffer(0, buffer_barriers, true);
                index_buffer_prepare_failed = !prepared;
                return prepared;
            }
            return std::nullopt;
        },
        [&] {
            prepared_buffer =
                buffer_cache.PrepareBufferBinding(arg_range->address, arg_range->size, false);
            if (count_address != 0) {
                prepared_count_buffer =
                    buffer_cache.PrepareBufferBinding(count_range->address, count_range->size,
                                                      false);
            }
        },
        [&] { buffer_cache.DrainReadyFaults(); },
        [&](auto& prepared_vertex, auto& prepared_index) {
            if (index_buffer_prepare_failed) {
                return false;
            }
            if (!FinalizeUtilityBufferBindings()) {
                return false;
            }
            if (!buffer_cache.FinalizeVertexBuffers(prepared_vertex, binding_batch)) {
                return false;
            }
            if (prepared_index &&
                !buffer_cache.FinalizeIndexBuffer(*prepared_index, binding_batch)) {
                return false;
            }
            if (!FinalizeBufferBindings(binding_batch)) {
                return false;
            }
            ASSERT(prepared_buffer.has_value());
            buffer = buffer_cache.FinalizeBufferBinding(
                *prepared_buffer, vk::AccessFlagBits2::eIndirectCommandRead,
                vk::PipelineStageFlagBits2::eDrawIndirect, binding_batch);
            count_buffer =
                prepared_count_buffer
                    ? buffer_cache.FinalizeBufferBinding(
                          *prepared_count_buffer, vk::AccessFlagBits2::eIndirectCommandRead,
                          vk::PipelineStageFlagBits2::eDrawIndirect, binding_batch)
                    : std::nullopt;
            if (!buffer || (prepared_count_buffer && !count_buffer)) {
                return false;
            }
            // Buffer preparation may wait for the stream buffer and execute deferred image
            // retirement. Resolve attachments only after that mutation window, but before image
            // descriptors are finalized so feedback-loop layouts remain coherent.
            if (!BeginRendering(pipeline, state) || !RebindPendingImageBindings() ||
                !FinalizeImageDescriptors(descriptor_context)) {
                ObserveDescriptorImageDrawDrop(descriptor_context);
                return false;
            }
            buffer_cache.CommitBufferBindings(binding_batch, buffer_barriers);
            return true;
        },
        [&](const auto& prepared_vertex) { buffer_cache.BindVertexBuffers(prepared_vertex); },
        [&](const auto& prepared_index) {
            if (prepared_index) {
                buffer_cache.BindIndexBuffer(*prepared_index);
            }
        });
    if (!draw_buffers_ready) {
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    // Vulkan supplies GPU indirect BaseVertex/BaseInstance; the immutable pipeline context
    // maps enabled conventional-VS CP destinations to those builtins in the shader.

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Graphics, pipeline->Handle());
    auto color_provenance = PrepareColorAttachmentProvenance(*pipeline, state);

    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->buffer, buffer->offset, count_buffer->buffer,
                                            count_buffer->offset, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->buffer, buffer->offset, max_count, stride);
        }
        DebugState.IncDrawCall();
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->buffer, buffer->offset, count_buffer->buffer,
                                     count_buffer->offset, max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->buffer, buffer->offset, max_count, stride);
        }
        DebugState.IncDrawCall();
    }
    if (color_provenance) {
        CompleteColorAttachmentProvenance(std::move(*color_provenance), state);
    }

    scheduler.MarkGuestWork(std::nullopt, pipeline->GetStage(Shader::LogicalStage::Vertex).pgm_hash,
                            VideoCore::GuestWorkCaptureRequiresColorAttachment() &&
                                VideoCore::HasBoundRenderDocColorAttachment(
                                    state.color_attachments, state.num_color_attachments));
    if (color_command_ledger.Enabled() || emitted_color_vertex_receipt.Enabled()) {
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
    const DescriptorImageCommandContext pipeline_context{
        .kind = DescriptorImageCommandKind::DirectDispatch,
    };
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        ObserveDrawDrop(DrawDropReason::NullPipeline, pipeline_context);
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    const DescriptorImageCommandContext descriptor_context{
        .kind = DescriptorImageCommandKind::DirectDispatch,
        .compute_hash = cs.pgm_hash,
    };
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }
    buffer_cache.DrainReadyFaults();
    VideoCore::BufferCache::BufferBindingBatch binding_batch;
    if (!FinalizeUtilityBufferBindings() || !FinalizeBufferBindings(binding_batch)) {
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }
    if (!RebindPendingImageBindings() || !FinalizeImageDescriptors(descriptor_context)) {
        ObserveDescriptorImageDrawDrop(descriptor_context);
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }
    buffer_cache.CommitBufferBindings(binding_batch, buffer_barriers);

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
    const DescriptorImageCommandContext pipeline_context{
        .kind = DescriptorImageCommandKind::IndirectDispatch,
    };
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        ObserveDrawDrop(DrawDropReason::NullPipeline, pipeline_context);
        return;
    }
    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    const DescriptorImageCommandContext descriptor_context{
        .kind = DescriptorImageCommandKind::IndirectDispatch,
        .compute_hash = cs.pgm_hash,
    };

    if (!BindResources(pipeline)) {
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }

    const auto indirect_range =
        AmdGpu::TryGetIndirectBufferRange(address, offset, static_cast<u64>(size));
    if (!indirect_range) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping indirect dispatch with overflowing argument range: "
                    "address={:#x} offset={} size={}",
                    address, offset, size);
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }
    const auto prepared_buffer = buffer_cache.PrepareBufferBinding(
        indirect_range->address, indirect_range->size, false);
    buffer_cache.DrainReadyFaults();
    VideoCore::BufferCache::BufferBindingBatch binding_batch;
    if (!FinalizeUtilityBufferBindings() || !FinalizeBufferBindings(binding_batch)) {
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }
    const auto buffer = buffer_cache.FinalizeBufferBinding(
        prepared_buffer, vk::AccessFlagBits2::eIndirectCommandRead,
        vk::PipelineStageFlagBits2::eDrawIndirect, binding_batch);
    if (!buffer) {
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }
    if (!RebindPendingImageBindings() || !FinalizeImageDescriptors(descriptor_context)) {
        ObserveDescriptorImageDrawDrop(descriptor_context);
        ObserveDrawDrop(DrawDropReason::ResourceBinding, descriptor_context);
        ResetBindings();
        return;
    }
    buffer_cache.CommitBufferBindings(binding_batch, buffer_barriers);

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Compute, pipeline->Handle());
    GetComputeDependencyProbe().Dispatch(
        [&] { cmdbuf.dispatchIndirect(buffer->buffer, buffer->offset); },
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
    pending_buffer_bindings.clear();
    utility_buffer_bindings_failed = false;
    image_infos.clear();
    descriptor_image_layout_bindings.clear();
    pending_image_bindings.clear();
    pending_utility_buffer_bindings.clear();
    resource_bindings_failed = false;

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
    pending_image_bindings.reserve(required_image_infos);

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
        if (utility_buffer_bindings_failed || resource_bindings_failed) {
            return false;
        }
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
        dst_image.CopyImageWithBuffer(src_image, copy_buffer.Handle(), 0,
                                      copy_buffer.SizeBytes());
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
        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            const u64 binding_size = desc.GetBindingSize(vsharp);
            if (!memory->IsValidMapping(vsharp.base_address, binding_size)) {
                LOG_WARNING(Render_Vulkan,
                            "Rejecting buffer binding with invalid guest address: "
                            "shader={:#x} address={:#x} size={:#x}",
                            stage.pgm_hash, vsharp.base_address, binding_size);
                resource_bindings_failed = true;
                return;
            }
            const u64 size = memory->ClampRangeSize(vsharp.base_address, binding_size);
            const auto buffer_id = buffer_cache.FindBuffer(vsharp.base_address, size);
            buffer_bindings.emplace_back(buffer_id, vsharp, size);
        } else {
            buffer_bindings.emplace_back(VideoCore::BufferId{}, vsharp, 0);
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
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                PendingUtilityBufferBinding pending{
                    .buffer_info_index = static_cast<u32>(buffer_infos.size()),
                    .alignment = alignment,
                    .bytes = std::vector<u8>(ubo_size),
                };
                if (ubo_size != 0) {
                    std::memcpy(pending.bytes.data(), stage.flattened_ud_buf.data(), ubo_size);
                }
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, ubo_size);
                pending_utility_buffer_bindings.push_back(std::move(pending));
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->regs.clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->regs.clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    PendingUtilityBufferBinding pending{
                        .buffer_info_index = static_cast<u32>(buffer_infos.size()),
                        .alignment = alignment,
                        .bytes = std::vector<u8>(ubo_size),
                    };
                    std::memcpy(pending.bytes.data(), planes.data(), ubo_size);
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, ubo_size);
                    pending_utility_buffer_bindings.push_back(std::move(pending));
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                const auto& cs_program = liverpool->GetCsRegs();
                const auto lds_size = UtilityBuffer::CheckedSharedMemorySize(
                    cs_program.SharedMemSize(), cs_program.dim_x, cs_program.dim_y,
                    cs_program.dim_z);
                const auto& stream_buffer =
                    buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                if (!lds_size || *lds_size > std::numeric_limits<std::size_t>::max() ||
                    *lds_size > static_cast<u64>(stream_buffer.SizeBytes())) {
                    LOG_WARNING(Render_Vulkan,
                                "Rejecting shared-memory binding with invalid size: "
                                "shared_bytes={} dimensions=({}, {}, {})",
                                cs_program.SharedMemSize(), cs_program.dim_x, cs_program.dim_y,
                                cs_program.dim_z);
                    utility_buffer_bindings_failed = true;
                    return;
                }
                PendingUtilityBufferBinding pending{
                    .buffer_info_index = static_cast<u32>(buffer_infos.size()),
                    .alignment = alignment,
                    .bytes = std::vector<u8>(static_cast<std::size_t>(*lds_size)),
                };
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, *lds_size);
                pending_utility_buffer_bindings.push_back(std::move(pending));
            } else {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            }
        } else {
            const auto prepared = buffer_cache.PrepareBufferBinding(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id);
            const u32 buffer_info_index = static_cast<u32>(buffer_infos.size());
            buffer_infos.emplace_back(VK_NULL_HANDLE, 0, size);
            pending_buffer_bindings.push_back({
                .prepared = prepared,
                .buffer_info_index = buffer_info_index,
                .push_binding = binding.buffer,
                .alignment = alignment,
                .size = static_cast<u32>(size),
                .access = VideoCore::ShaderBufferAccess(desc.is_written),
            });
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

bool Rasterizer::FinalizeUtilityBufferBindings() {
    auto& stream_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
    if (pending_utility_buffer_bindings.empty()) {
        return true;
    }

    // Reserve one contiguous range. Mapping each descriptor separately could wrap the ring in
    // the middle of a command and invalidate the first descriptor before Vulkan consumes it.
    const auto reservation = UtilityBuffer::PlanReservation(
        pending_utility_buffer_bindings, static_cast<u64>(stream_buffer.SizeBytes()));
    if (!reservation) {
        return false;
    }
    const auto [data, base_offset] =
        stream_buffer.Map(reservation->total_size, reservation->base_alignment);
    if (reservation->total_size != 0 && data == nullptr) {
        return false;
    }
    u64 cursor{};
    for (const auto& pending : pending_utility_buffer_bindings) {
        const auto aligned_cursor =
            UtilityBuffer::CheckedAlignUp(cursor, static_cast<u64>(pending.alignment));
        if (!aligned_cursor) {
            return false;
        }
        cursor = *aligned_cursor;
        if (!pending.bytes.empty()) {
            std::memcpy(data + cursor, pending.bytes.data(), pending.bytes.size());
        }
        buffer_infos[pending.buffer_info_index] = vk::DescriptorBufferInfo{
            stream_buffer.Handle(), base_offset + cursor, pending.bytes.size()};
        const auto next_cursor =
            UtilityBuffer::CheckedAdd(cursor, static_cast<u64>(pending.bytes.size()));
        if (!next_cursor) {
            return false;
        }
        cursor = *next_cursor;
    }
    stream_buffer.Commit();
    return true;
}

bool Rasterizer::FinalizeBufferBindings(VideoCore::BufferCache::BufferBindingBatch& batch) {
    for (const auto& pending : pending_buffer_bindings) {
        const auto endpoint = buffer_cache.FinalizeBufferBinding(
            pending.prepared, pending.access, vk::PipelineStageFlagBits2::eAllCommands, batch);
        if (!endpoint) {
            return false;
        }
        const u64 offset_aligned = Common::AlignDown(endpoint->offset, pending.alignment);
        const u32 adjust = static_cast<u32>(endpoint->offset - offset_aligned);
        if (adjust % 4 != 0) {
            LOG_WARNING(Render_Vulkan, "Buffer binding {} isn't dword aligned",
                        pending.push_binding);
        }
        push_data.AddOffset(pending.push_binding, adjust);
        buffer_infos[pending.buffer_info_index] =
            vk::DescriptorBufferInfo{endpoint->buffer, offset_aligned, pending.size + adjust};
    }
    return true;
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
        if (num_bindings == 0) {
            // A malformed guest mip window must never become a wrapped descriptor-array count.
            // Keep the runtime descriptor stream structurally valid with one null binding; fresh
            // translation rejects the same resource before pipeline creation.
            LOG_WARNING(Render_Vulkan,
                        "Dropping image binding with malformed mip range for shader {:#x}",
                        stage.pgm_hash);
            append_invalid_image_binding(image_desc.is_written);
            image_descriptor_array_sizes.push_back(1);
            continue;
        }
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

            // The guest descriptor's array and mip fields are inclusive windows.  Validate the
            // raw window before ImageDesc constructs ImageInfo/ImageViewInfo: ImageViewInfo keeps
            // a defensive layer clamp for legacy callers, but clamping here would silently bind a
            // different layer for malformed Color2DArray/cube descriptors.
            if (!tsharp.HasValidSubresourceRange()) {
                LOG_WARNING(Render_Vulkan,
                            "Rejecting invalid T# subresource range address={:#x} type={} "
                            "mip={}..{} array={}..{}",
                            tsharp.Address(), static_cast<u32>(tsharp.type),
                            static_cast<u32>(tsharp.base_level),
                            static_cast<u32>(tsharp.last_level),
                            static_cast<u32>(tsharp.base_array),
                            static_cast<u32>(tsharp.last_array));
                append_invalid_image_binding(image_desc.is_written);
                continue;
            }

            // IsValidGpuMapping checks address width, not backing residency.
            const bool address_40bit = memory->IsValidGpuMapping(tsharp.Address(), 0);
            const auto surface_format =
                LiverpoolToVK::TrySurfaceFormat(data_fmt, num_fmt);
            const bool valid_sample_count = tsharp.HasValidMsaaSampleExponent();
            const auto rejection_reasons = TextureRejectionReasons(
                address_40bit, magic_enum::enum_contains(data_fmt),
                magic_enum::enum_contains(num_fmt),
                // Storage views never consume DST_SEL. Sampled views must reject reserved
                // selectors before Binding constructs ImageViewInfo/ComponentMapping.
                image_desc.is_written ||
                    TextureComponentSelectorsValid(
                        {static_cast<u32>(tsharp.dst_sel_x), static_cast<u32>(tsharp.dst_sel_y),
                         static_cast<u32>(tsharp.dst_sel_z), static_cast<u32>(tsharp.dst_sel_w)}),
                surface_format.has_value(), valid_sample_count);
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
                        roots.flat_dwords, roots.copied_dwords, roots.has_full_prefix, ud[0], ud[1],
                        ud[2], ud[3], ud[4], ud[5], ud[6], ud[7], ud[8], ud[9], ud[10], ud[11],
                        ud[12], ud[13], ud[14], ud[15]);
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
            ScopedTextureDiagnosticContext diagnostic_context{TextureRejectionDiagnosticsEnabled(),
                                                              build_texture_observation};
            image_id = texture_cache.FindImage(desc);
            if (!image_id) {
                // TextureCache rejects unsupported host image capabilities without publishing a
                // backing-less cache entry. Keep this descriptor as the existing null binding so
                // one malformed T# cannot reach Vulkan with an unsupported image create-info.
                LOG_WARNING(Render_Vulkan,
                            "Dropping texture binding after unsupported host image admission "
                            "address={:#x} format={} type={}",
                            tsharp.Address(), vk::to_string(surface_format.value_or(vk::Format::eUndefined)),
                            static_cast<u32>(tsharp.type));
                continue;
            }
            auto* image = &texture_cache.GetImage(image_id);
            if (VideoCore::ShouldUseAssociatedDepthForView(desc.view_info.format)) {
                if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                    // If this image has an associated depth image, it's a stencil attachment.
                    // Redirect the access to the actual depth-stencil buffer.
                    image_id = depth_image_id;
                    image = &texture_cache.GetImage(image_id);
                }
            }
            image->binding.RecordShaderAccess(image_desc.is_written);
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
                if (!image_id) {
                    LOG_WARNING(Render_Vulkan,
                                "Dropping texture binding after image rebind failed");
                    image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE,
                                             vk::ImageLayout::eGeneral);
                    continue;
                }
            }

            TrackBoundImage(image_id);

            auto& image = texture_cache.GetImage(image_id);
            const auto [uses_integer_coordinates, queries_dimensions] =
                image_native_extent_requirements[binding_index];
            if (VideoCore::ShouldForceNativeHostImageAccess(is_storage, uses_integer_coordinates,
                                                            queries_dimensions)) {
                ASSERT(image.SetHostScale(100));
            }
            auto& image_view = texture_cache.FindTexture(image_id, desc);
            static_cast<void>(image_view);

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((image.binding.force_general || image.binding.is_target) &&
                !image.info.props.is_depth) {
                image.Transit(instance.IsAttachmentFeedbackLoopLayoutSupported() &&
                                      image.binding.is_target
                                  ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                  : vk::ImageLayout::eGeneral,
                              image.binding.GeneralLayoutAccessMask(image.info.props.is_depth),
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

            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, image.backing->state.layout);
            const u32 pending_index = static_cast<u32>(pending_image_bindings.size());
            pending_image_bindings.push_back(
                {image_id, image.image_uid, desc.view_info, desc,
                 image_native_extent_requirements[binding_index].first,
                 image_native_extent_requirements[binding_index].second});
            descriptor_image_layout_bindings.emplace_back(static_cast<u32>(image_infos.size() - 1),
                                                          pending_index);
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
                        observation.roots = CaptureSrtRootProvenance(stage.flattened_ud_buf,
                                                                     stage.user_data.size());
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
                        roots.flat_dwords, roots.copied_dwords, roots.has_full_prefix, ud[0], ud[1],
                        ud[2], ud[3], ud[4], ud[5], ud[6], ud[7], ud[8], ud[9], ud[10], ud[11],
                        ud[12], ud[13], ud[14], ud[15]);
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

bool Rasterizer::RebindPendingImageBindings() {
    // Keep the reconciliation opt-in while the route is being characterized. The default path
    // remains byte-for-byte equivalent to the established non-repairing resolver until a strict
    // Shadow and Uncharted promotion batch proves this generic handoff safe.
    const char* enabled = std::getenv("SHADPS4_LATE_IMAGE_BINDING_REBIND");
    if (!enabled || std::strcmp(enabled, "1") != 0) {
        return true;
    }

    bool all_resolved = true;
    for (auto& pending : pending_image_bindings) {
        auto* image = texture_cache.TryGetImage(pending.image_id, pending.image_uid);
        bool needs_rebind = image == nullptr;
        if (image && !needs_rebind) {
            needs_rebind = image->binding.needs_rebind != 0;
            if (!needs_rebind) {
                needs_rebind = !VideoCore::IsValidSubresourceRange(pending.view_info.range,
                                                                    image->info.resources);
            }
            if (!needs_rebind) {
                // FindView(false) is intentionally non-waiting. It only validates that the
                // existing backing still owns a live Vulkan view after late cache maintenance.
                const auto& image_view = image->FindView(pending.view_info, false);
                needs_rebind = !image_view.image_view;
            }
        }
        if (!needs_rebind) {
            continue;
        }

        // The descriptor key was captured before the wait-capable buffer preparation phase. Find
        // the current cache generation only here, at the enclosing draw boundary where mutation
        // and view creation are allowed. FinalizeImageDescriptors remains side-effect free.
        if (image && image->binding.needs_rebind) {
            image->binding = {};
        }
        auto desc = pending.desc;
        auto image_id = texture_cache.FindImage(desc);
        if (VideoCore::ShouldUseAssociatedDepthForView(desc.view_info.format)) {
            if (image_id) {
                auto& resolved = texture_cache.GetImage(image_id);
                if (auto depth_image_id = texture_cache.GetAssociatedDepth(resolved)) {
                    image_id = depth_image_id;
                }
            }
        }
        if (!image_id) {
            all_resolved = false;
            continue;
        }

        image = &texture_cache.GetImage(image_id);
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (VideoCore::ShouldForceNativeHostImageAccess(
                is_storage, pending.uses_integer_coordinates, pending.queries_dimensions)) {
            if (!image->SetHostScale(100)) {
                all_resolved = false;
                continue;
            }
        }

        auto& image_view = texture_cache.FindTexture(image_id, desc);
        if (!image_view.image_view) {
            all_resolved = false;
            continue;
        }

        image->binding.RecordShaderAccess(is_storage);

        // Re-emit the same layout transition performed during the initial bind. A replacement
        // image starts with its own backing state, so merely changing the id/uid is insufficient.
        if ((image->binding.force_general || image->binding.is_target) &&
            !image->info.props.is_depth) {
            image->Transit(
                instance.IsAttachmentFeedbackLoopLayoutSupported() && image->binding.is_target
                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                    : vk::ImageLayout::eGeneral,
                image->binding.GeneralLayoutAccessMask(image->info.props.is_depth),
                {});
        } else if (is_storage) {
            image->Transit(vk::ImageLayout::eGeneral,
                           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                           desc.view_info.range);
        } else {
            const auto new_layout = image->info.props.is_depth
                                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                        : vk::ImageLayout::eShaderReadOnlyOptimal;
            image->Transit(new_layout, vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
        }
        image->usage.storage |= is_storage;
        image->usage.texture |= !is_storage;

        pending.image_id = image_id;
        pending.image_uid = image->image_uid;
        pending.view_info = desc.view_info;
        pending.desc = desc;
        TrackBoundImage(image_id);
    }
    return all_resolved;
}

void Rasterizer::ObserveDescriptorImageDrawDrop(const DescriptorImageCommandContext& context) {
    if (!descriptor_image_finalization_receipt.Enabled()) {
        return;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - descriptor_finalization_origin)
                             .count();
    const auto observation =
        descriptor_image_finalization_receipt.Observe(static_cast<u64>(elapsed), [&] {
            return DescriptorImageFinalizationObservation{
                .failure = DescriptorImageFinalizationFailure::DrawDropped,
                .command = context,
            };
        });
    if (observation) {
        LOG_WARNING(Render_Vulkan,
                    "DescriptorImageFinalizationReceiptV1 failure={} command={} "
                    "renderer_ms={} vs={:016x} fs={:016x} cs={:016x} occurrences={} "
                    "behavior=observation_only",
                    DescriptorImageFinalizationFailureName(observation->failure),
                    DescriptorImageCommandKindName(observation->command.kind), elapsed,
                    observation->command.vertex_hash, observation->command.fragment_hash,
                    observation->command.compute_hash,
                    descriptor_image_finalization_receipt.FailureOccurrences(
                        DescriptorImageFinalizationFailure::DrawDropped));
    }
}

bool Rasterizer::FinalizeImageDescriptors(const DescriptorImageCommandContext& context) {
    std::size_t finalization_ordinal{};
    return FinalizeDescriptorImageInfos(
        image_infos, descriptor_image_layout_bindings,
        [this, &context,
         &finalization_ordinal](const u32 pending_index) -> std::optional<ResolvedDescriptorImage> {
            const u32 descriptor_index =
                finalization_ordinal < descriptor_image_layout_bindings.size()
                    ? descriptor_image_layout_bindings[finalization_ordinal].descriptor_index
                    : DescriptorImageFinalizationObservation::InvalidIndex;
            ++finalization_ordinal;
            const auto observe_failure = [&](DescriptorImageFinalizationFailure failure,
                                             u32 image_id, u64 image_uid) {
                if (!descriptor_image_finalization_receipt.Enabled()) {
                    return;
                }
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - descriptor_finalization_origin)
                        .count();
                const auto observation =
                    descriptor_image_finalization_receipt.Observe(static_cast<u64>(elapsed), [&] {
                        return DescriptorImageFinalizationObservation{
                            .failure = failure,
                            .command = context,
                            .descriptor_index = descriptor_index,
                            .pending_index = pending_index,
                            .image_id = image_id,
                            .image_uid = image_uid,
                        };
                    });
                if (observation) {
                    LOG_WARNING(
                        Render_Vulkan,
                        "DescriptorImageFinalizationReceiptV1 failure={} command={} "
                        "renderer_ms={} vs={:016x} fs={:016x} cs={:016x} descriptor_index={} "
                        "pending_index={} image_id={} image_uid={:016x} occurrences={} "
                        "behavior=observation_only",
                        DescriptorImageFinalizationFailureName(observation->failure),
                        DescriptorImageCommandKindName(observation->command.kind), elapsed,
                        observation->command.vertex_hash, observation->command.fragment_hash,
                        observation->command.compute_hash, observation->descriptor_index,
                        observation->pending_index, observation->image_id, observation->image_uid,
                        descriptor_image_finalization_receipt.FailureOccurrences(failure));
                }
            };

            if (pending_index >= pending_image_bindings.size()) {
                observe_failure(DescriptorImageFinalizationFailure::PendingIndexOutOfBounds,
                                DescriptorImageFinalizationObservation::InvalidIndex, 0);
                return std::nullopt;
            }
            const auto& pending = pending_image_bindings[pending_index];
            auto* image = texture_cache.TryGetImage(pending.image_id, pending.image_uid);
            if (!image) {
                observe_failure(DescriptorImageFinalizationFailure::StaleGeneration,
                                pending.image_id.index, pending.image_uid);
                return std::nullopt;
            }
            if (image->binding.needs_rebind) {
                observe_failure(DescriptorImageFinalizationFailure::NeedsRebind,
                                pending.image_id.index, pending.image_uid);
                return std::nullopt;
            }
            if (!VideoCore::IsValidSubresourceRange(pending.view_info.range,
                                                    image->info.resources)) {
                LOG_WARNING(Render_Vulkan,
                            "Rejecting stale image view range during descriptor finalization: "
                            "base={}+{} extent={}x{} available={}x{} address={:#x}",
                            pending.view_info.range.base.level, pending.view_info.range.base.layer,
                            pending.view_info.range.extent.levels,
                            pending.view_info.range.extent.layers, image->info.resources.levels,
                            image->info.resources.layers, image->info.guest_address);
                observe_failure(DescriptorImageFinalizationFailure::ViewMissing,
                                pending.image_id.index, pending.image_uid);
                return std::nullopt;
            }
            const auto& image_view = image->FindView(pending.view_info, false);
            if (!image_view.image_view) {
                observe_failure(DescriptorImageFinalizationFailure::ViewMissing,
                                pending.image_id.index, pending.image_uid);
                // A view can disappear between the late backing preparation and descriptor
                // finalization. Do not dereference the empty optional: abort this publication and
                // let the caller discard the draw, just as for other stale finalization inputs.
                return std::nullopt;
            }
            const auto layout = image->GetSubresourceLayout(pending.view_info.range);
            if (!layout) {
                LOG_WARNING(Render_Vulkan,
                            "Rejecting mixed-layout descriptor view: base={}+{} extent={}x{}",
                            pending.view_info.range.base.level, pending.view_info.range.base.layer,
                            pending.view_info.range.extent.levels,
                            pending.view_info.range.extent.layers);
                observe_failure(DescriptorImageFinalizationFailure::ViewMissing,
                                pending.image_id.index, pending.image_uid);
                return std::nullopt;
            }
            return ResolvedDescriptorImage{*image_view.image_view, *layout};
        });
}

void Rasterizer::TrackBoundImage(VideoCore::ImageId image_id) {
    const auto& image = texture_cache.GetImage(image_id);
    bound_images.push_back({image_id, image.image_uid});
}

bool Rasterizer::BeginRendering(const GraphicsPipeline* pipeline, RenderState& state) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->regs;
    const auto& key = pipeline->GetGraphicsKey();
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
            image_id = texture_cache.FindImage(desc);
            if (!image_id) {
                LOG_WARNING(Render_Vulkan,
                            "Dropping draw after color attachment image rebind failed cb={}", cb);
                state.color_attachments[cb] = {};
                return false;
            }
            TrackBoundImage(image_id);
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

    return true;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->last_cb_extent[0];
    const auto& mrt1_hint = liverpool->last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->regs.color_buffers[1], mrt1_hint};
    const auto mrt0_id = texture_cache.FindImage(mrt0_desc, true);
    const auto mrt1_id = texture_cache.FindImage(mrt1_desc, true);
    if (!mrt0_id || !mrt1_id) {
        LOG_WARNING(Render_Vulkan, "Skipping resolve after unsupported color image admission");
        return;
    }
    auto& mrt0_image = texture_cache.GetImage(mrt0_id);
    auto& mrt1_image = texture_cache.GetImage(mrt1_id);

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

    const auto read_image_id = texture_cache.FindImage(read_desc);
    const auto write_image_id = texture_cache.FindImage(write_desc);
    if (!read_image_id || !write_image_id) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping depth/stencil copy after unsupported image admission");
        return;
    }
    auto& read_image = texture_cache.GetImage(read_image_id);
    auto& write_image = texture_cache.GetImage(write_image_id);

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->regs.depth_view.slice_start;
    sub_range.extent.layers = liverpool->regs.depth_view.NumSlices() - sub_range.base.layer;
    if (!VideoCore::IsValidSubresourceRange(sub_range, read_image.info.resources) ||
        !VideoCore::IsValidSubresourceRange(sub_range, write_image.info.resources)) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping depth/stencil copy with out-of-range subresource");
        return;
    }

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

bool Rasterizer::HasGpuMappedSubrange(VAddr addr, u64 size) {
    return !page_manager.GetGpuMappedSubranges(addr, size).empty();
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

void Rasterizer::UnmapMappedMemory(VAddr addr, u64 size, VideoCore::GpuMappingKind kind) {
    // Snapshot the visible ranges before revoking any of them. Each call below updates the page
    // manager, so iterating its live interval map would otherwise skip or revisit a range.
    const auto mapped_subranges = page_manager.GetGpuMappedSubranges(addr, size, kind);
    for (const auto& [mapped_addr, mapped_size] : mapped_subranges) {
        UnmapMemory(mapped_addr, mapped_size, kind);
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
