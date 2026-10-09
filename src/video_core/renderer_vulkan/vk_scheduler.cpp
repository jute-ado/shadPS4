// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/debug.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/diagnostic_gpu_marker_policy.h"
#include "video_core/renderer_vulkan/vk_diagnostic_checkpoint.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_submission_chunk_policy.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

namespace {

struct DiagnosticSubmission {
    u64 signal_value{};
    size_t gpu_marker_begin{};
    size_t gpu_marker_end{};
    std::vector<DiagnosticCheckpointMarker> commands;
};

std::mutex diagnostic_mutex;
std::deque<DiagnosticCheckpointMarker> diagnostic_commands;
std::deque<DiagnosticSubmission> diagnostic_submissions;
std::atomic_flag diagnostic_loss_logged;
constexpr size_t DiagnosticGpuMarkerCapacity = 8 * 1024 * 1024;
VkBuffer diagnostic_gpu_buffer{};
VmaAllocation diagnostic_gpu_allocation{};
u32* diagnostic_gpu_mapped{};
VmaAllocator diagnostic_gpu_allocator{};
std::vector<DiagnosticCheckpointMarker> diagnostic_gpu_commands;
size_t diagnostic_sealed_gpu_marker_end{};

} // Anonymous namespace

void InitializeDiagnosticGpuMarkers(const Instance& instance) {
    const bool diagnostic_opt_in =
        std::getenv("SHADPS4_DIAGNOSTIC_GPU_MARKERS") != nullptr;
    if (!ShouldEnableDiagnosticGpuMarkers(instance.SupportsBufferMarker(), diagnostic_opt_in) ||
        diagnostic_gpu_buffer) {
        return;
    }
    const VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = DiagnosticGpuMarkerCapacity * sizeof(u32),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation_info{
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        .preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };
    VmaAllocationInfo created_info{};
    diagnostic_gpu_allocator = instance.GetAllocator();
    const VkResult result = vmaCreateBuffer(diagnostic_gpu_allocator, &buffer_info,
                                            &allocation_info, &diagnostic_gpu_buffer,
                                            &diagnostic_gpu_allocation, &created_info);
    ASSERT_MSG(result == VK_SUCCESS, "Failed to create diagnostic marker buffer: {}",
               vk::to_string(vk::Result{result}));
    diagnostic_gpu_mapped = static_cast<u32*>(created_info.pMappedData);
    std::memset(diagnostic_gpu_mapped, 0, buffer_info.size);
    diagnostic_gpu_commands.reserve(DiagnosticGpuMarkerCapacity);
}

void RecordDiagnosticGpuMarker(vk::CommandBuffer cmdbuf,
                               const DiagnosticCheckpointMarker& marker) {
    if (!diagnostic_gpu_buffer ||
        diagnostic_gpu_commands.size() == DiagnosticGpuMarkerCapacity) {
        return;
    }
    const u32 marker_index = static_cast<u32>(diagnostic_gpu_commands.size());
    diagnostic_gpu_commands.emplace_back(marker);
    const VkPipelineStageFlagBits marker_stage =
        marker.kind == DiagnosticCheckpointKind::Compute
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            : VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteBufferMarkerAMD(
        static_cast<VkCommandBuffer>(cmdbuf), marker_stage, diagnostic_gpu_buffer,
        static_cast<VkDeviceSize>(marker_index) * sizeof(u32), marker_index + 1);
}

void RecordDiagnosticCommand(const DiagnosticCheckpointMarker& marker) {
    if (diagnostic_gpu_mapped) {
        vmaInvalidateAllocation(diagnostic_gpu_allocator, diagnostic_gpu_allocation, 0,
                                VK_WHOLE_SIZE);
    }
    const std::scoped_lock lock{diagnostic_mutex};
    if (!diagnostic_commands.empty()) {
        const auto& previous = diagnostic_commands.back();
        if (previous.kind == marker.kind && previous.first_hash == marker.first_hash &&
            previous.second_hash == marker.second_hash) {
            return;
        }
    }
    static constexpr size_t MaxCommands = 8192;
    if (diagnostic_commands.size() == MaxCommands) {
        diagnostic_commands.pop_front();
    }
    diagnostic_commands.emplace_back(marker);
}

void SealDiagnosticSubmission(u64 signal_value) {
    const std::scoped_lock lock{diagnostic_mutex};
    DiagnosticSubmission submission{.signal_value = signal_value,
                                    .gpu_marker_begin = diagnostic_sealed_gpu_marker_end,
                                    .gpu_marker_end = diagnostic_gpu_commands.size()};
    diagnostic_sealed_gpu_marker_end = submission.gpu_marker_end;
    submission.commands.assign(diagnostic_commands.begin(), diagnostic_commands.end());
    diagnostic_commands.clear();
    diagnostic_submissions.emplace_back(std::move(submission));
    static constexpr size_t MaxSubmissions = 64;
    if (diagnostic_submissions.size() > MaxSubmissions) {
        diagnostic_submissions.pop_front();
    }
}

void LogDiagnosticCheckpoints(const Instance& instance) {
    LogDiagnosticCheckpoints(instance, 0);
}

void LogDiagnosticCheckpoints(const Instance& instance, u64 last_known_tick) {
    if (!instance.SupportsDiagnosticCheckpoints()) {
        return;
    }
    if (diagnostic_loss_logged.test_and_set()) {
        return;
    }

    u32 checkpoint_count{};
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(
        static_cast<VkQueue>(instance.GetGraphicsQueue()), &checkpoint_count, nullptr);
    std::vector<VkCheckpointDataNV> checkpoints(
        checkpoint_count, VkCheckpointDataNV{VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(
        static_cast<VkQueue>(instance.GetGraphicsQueue()), &checkpoint_count, checkpoints.data());
    LOG_ERROR(Render_Vulkan, "Device-lost checkpoint count={}", checkpoint_count);
    for (const auto& checkpoint : checkpoints) {
        const auto* marker =
            static_cast<const DiagnosticCheckpointMarker*>(checkpoint.pCheckpointMarker);
        if (!marker || marker->magic != DiagnosticCheckpointMarker::Magic) {
            LOG_ERROR(Render_Vulkan, "Device-lost checkpoint stage={:#x} marker={}",
                      static_cast<u32>(checkpoint.stage), checkpoint.pCheckpointMarker);
        } else if (marker->kind == DiagnosticCheckpointKind::Graphics) {
            LOG_ERROR(Render_Vulkan,
                      "Device-lost checkpoint stage={:#x} vertex_shader={:#x} fragment_shader={:#x}",
                      static_cast<u32>(checkpoint.stage), marker->first_hash,
                      marker->second_hash);
        } else {
            LOG_ERROR(Render_Vulkan, "Device-lost checkpoint stage={:#x} compute_shader={:#x}",
                      static_cast<u32>(checkpoint.stage), marker->first_hash);
        }
    }

    const std::scoped_lock lock{diagnostic_mutex};
    LOG_ERROR(Render_Vulkan,
              "Device-lost recorded submissions={} last_known_gpu_tick={}",
              diagnostic_submissions.size(), last_known_tick);
    for (const auto& submission : diagnostic_submissions) {
        if (submission.signal_value <= last_known_tick) {
            continue;
        }
        LOG_ERROR(Render_Vulkan, "Device-lost submission tick={} commands={}",
                  submission.signal_value, submission.commands.size());
        for (size_t index = 0; index < submission.commands.size(); ++index) {
            const auto& marker = submission.commands[index];
            if (marker.kind == DiagnosticCheckpointKind::Graphics) {
                LOG_ERROR(Render_Vulkan,
                          "Device-lost command={} graphics vertex_shader={:#x} fragment_shader={:#x} "
                          "draw={} instances={} indexed={} phase={:#x} index_address={:#x}",
                          index, marker.first_hash, marker.second_hash, marker.runtime[0],
                          marker.runtime[1], marker.runtime[2], marker.runtime[3],
                          u64{marker.runtime[4]} | (u64{marker.runtime[5]} << 32));
            } else {
                LOG_ERROR(Render_Vulkan, "Device-lost command={} compute_shader={:#x}", index,
                          marker.first_hash);
            }
        }
        if (diagnostic_gpu_mapped) {
            LOG_ERROR(Render_Vulkan, "Device-lost submission GPU marker range=[{}, {})",
                      submission.gpu_marker_begin, submission.gpu_marker_end);
            for (size_t index = submission.gpu_marker_begin;
                 index < submission.gpu_marker_end; ++index) {
                const auto& marker = diagnostic_gpu_commands[index];
                const bool did_complete = diagnostic_gpu_mapped[index] == index + 1;
                if (marker.kind == DiagnosticCheckpointKind::Graphics) {
                    LOG_ERROR(Render_Vulkan,
                              "Device-lost submission marker={} completed={} graphics "
                              "vertex_shader={:#x} fragment_shader={:#x} draw={} instances={} "
                              "indexed={} phase={:#x} index_address={:#x}",
                              index, did_complete, marker.first_hash, marker.second_hash,
                              marker.runtime[0], marker.runtime[1], marker.runtime[2],
                              marker.runtime[3],
                              u64{marker.runtime[4]} | (u64{marker.runtime[5]} << 32));
                } else {
                    LOG_ERROR(Render_Vulkan,
                              "Device-lost submission marker={} completed={} compute_shader={:#x} dispatch={}x{}x{} phase={:#x}",
                              index, did_complete, marker.first_hash, marker.runtime[0],
                              marker.runtime[1], marker.runtime[2], marker.runtime[3]);
                }
            }
        }
    }

    if (diagnostic_gpu_mapped) {
        size_t completed{};
        while (completed < diagnostic_gpu_commands.size() &&
               diagnostic_gpu_mapped[completed] == completed + 1) {
            ++completed;
        }
        LOG_ERROR(Render_Vulkan, "Device-lost GPU markers completed={} recorded={}", completed,
                  diagnostic_gpu_commands.size());
        const size_t first = completed > 16 ? completed - 16 : 0;
        const size_t last = std::min(diagnostic_gpu_commands.size(), completed + 16);
        for (size_t index = first; index < last; ++index) {
            const auto& marker = diagnostic_gpu_commands[index];
            const bool did_complete = index < completed;
            if (marker.kind == DiagnosticCheckpointKind::Graphics) {
                LOG_ERROR(Render_Vulkan,
                          "Device-lost GPU marker={} completed={} graphics vertex_shader={:#x} "
                          "fragment_shader={:#x} draw={} instances={} indexed={} phase={:#x} "
                          "index_address={:#x}",
                          index, did_complete, marker.first_hash, marker.second_hash,
                          marker.runtime[0], marker.runtime[1], marker.runtime[2],
                          marker.runtime[3],
                          u64{marker.runtime[4]} | (u64{marker.runtime[5]} << 32));
            } else {
                LOG_ERROR(Render_Vulkan,
                          "Device-lost GPU marker={} completed={} compute_shader={:#x} dispatch={}x{}x{} phase={:#x}",
                          index, did_complete, marker.first_hash, marker.runtime[0],
                          marker.runtime[1], marker.runtime[2], marker.runtime[3]);
            }
        }
    }
}

void LogDeviceFault(const Instance& instance) {
    if (!instance.SupportsDeviceFault()) {
        return;
    }
    VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    VkResult result = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT(
        static_cast<VkDevice>(instance.GetDevice()), &counts, nullptr);
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        LOG_ERROR(Render_Vulkan, "TEMP device fault count query failed: {}",
                  vk::to_string(vk::Result{result}));
        return;
    }
    std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
    std::vector<std::byte> binary(counts.vendorBinarySize);
    VkDeviceFaultInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
        .pAddressInfos = addresses.data(),
        .pVendorInfos = vendors.data(),
        .pVendorBinaryData = binary.data(),
    };
    result = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT(
        static_cast<VkDevice>(instance.GetDevice()), &counts, &info);
    LOG_ERROR(Render_Vulkan,
              "TEMP device fault result={} description='{}' addresses={} vendors={} binary={}",
              vk::to_string(vk::Result{result}), info.description, counts.addressInfoCount,
              counts.vendorInfoCount, counts.vendorBinarySize);
    for (const auto& address : addresses) {
        LOG_ERROR(Render_Vulkan, "TEMP device fault address type={} address={:#x} precision={:#x}",
                  vk::to_string(vk::DeviceFaultAddressTypeEXT{address.addressType}),
                  address.reportedAddress, address.addressPrecision);
    }
    for (const auto& vendor : vendors) {
        LOG_ERROR(Render_Vulkan,
                  "TEMP device fault vendor description='{}' code={:#x} data={:#x}",
                  vendor.description, vendor.vendorFaultCode, vendor.vendorFaultData);
    }
}

std::mutex Scheduler::submit_mutex;

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, master_semaphore{instance}, command_pool{instance, &master_semaphore} {
    InitializeDiagnosticGpuMarkers(instance);
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    AllocateWorkerCommandBuffers();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    is_rendering = true;
    render_state = new_state;

    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
        const auto& cb = render_state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = render_state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue =
            vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = std::bit_cast<float>(
                                                                          db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };

    current_cmdbuf.beginRendering(rendering_info);
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    current_cmdbuf.endRendering();
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish() {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick);
}

void Scheduler::Wait(u64 tick) {
    if (tick >= master_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    master_semaphore.Wait(tick);
}

void Scheduler::PopPendingOperations() {
    std::unique_lock lk(pending_ops_mutex);
    master_semaphore.Refresh();
    while (!pending_ops.empty() && master_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    current_cmdbuf = command_pool.Commit();
    Check(current_cmdbuf.begin(begin_info));

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();
    pipeline_bind_tracker.Reset();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    std::scoped_lock lk{submit_mutex};
    const u64 signal_value = master_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    EndRendering();
    Check(current_cmdbuf.end());
    if (instance.SupportsDiagnosticCheckpoints()) {
        SealDiagnosticSubmission(signal_value);
    }

    const vk::Semaphore timeline = master_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    static constexpr std::array<vk::PipelineStageFlags, 2> wait_stage_masks = {
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = 1U,
        .pCommandBuffers = &current_cmdbuf,
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    if (submit_result == vk::Result::eErrorDeviceLost) {
        LogDeviceFault(instance);
        if (instance.SupportsDiagnosticCheckpoints()) {
            LogDiagnosticCheckpoints(instance);
        }
    }
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    master_semaphore.Refresh();
    static constexpr u64 MaxGpuSubmissionLead = 8;
    if (const auto wait_tick = SubmissionBackpressureWaitTick(
            signal_value, master_semaphore.KnownGpuTick(), MaxGpuSubmissionLead)) {
        master_semaphore.Wait(*wait_tick);
    }
    AllocateWorkerCommandBuffers();

    // Apply pending operations
    PopPendingOperations();
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        master_semaphore.Wait(op.gpu_tick);
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
    }
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_aspects);
    }
}

} // namespace Vulkan
