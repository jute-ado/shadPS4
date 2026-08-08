// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/div_ceil.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/fault_download.h"
#include "video_core/buffer_cache/fault_frame_correlation.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/fault_range.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include "video_core/host_shaders/fault_buffer_process_comp.h"

namespace VideoCore {

static constexpr size_t PageFaultAreaSize = FaultDownloadSlotCount * sizeof(u64);

static constexpr std::string_view StatusName(FaultFrameCorrelationStatus status) {
    switch (status) {
    case FaultFrameCorrelationStatus::Complete:
        return "complete";
    case FaultFrameCorrelationStatus::NoBatches:
        return "no-batches";
    case FaultFrameCorrelationStatus::Incomplete:
        return "incomplete";
    case FaultFrameCorrelationStatus::Gap:
        return "gap";
    }
    return "unknown";
}

FaultManager::FaultManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler_,
                           BufferCache& buffer_cache_, PageManager& page_manager_,
                           u32 caching_pagebits, u64 caching_num_pages_)
    : scheduler{scheduler_}, buffer_cache{buffer_cache_},
      page_manager{page_manager_},
      caching_pagesize{1ULL << caching_pagebits}, caching_num_pages{caching_num_pages_},
      fault_buffer_size{caching_num_pages_ / 8},
      fault_buffer{instance, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, fault_buffer_size},
      download_buffer{instance, scheduler, MemoryUsage::Download,
                      0,        AllFlags,  MaxPendingFaults * PageFaultAreaSize} {
    const auto device = instance.GetDevice();
    Vulkan::SetObjectName(device, fault_buffer.Handle(), "Fault Buffer");

    const std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }};
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = 2,
        .pBindings = bindings.data(),
    };
    fault_process_desc_layout =
        Vulkan::Check(device.createDescriptorSetLayoutUnique(desc_layout_ci));

    std::vector<std::string> defines{{fmt::format("CACHING_PAGEBITS={}", caching_pagebits),
                                      fmt::format("MAX_PAGE_FAULTS={}", FaultDownloadSlotCount)}};
    const auto module = Vulkan::Compile(HostShaders::FAULT_BUFFER_PROCESS_COMP,
                                        vk::ShaderStageFlagBits::eCompute, device, defines);
    Vulkan::SetObjectName(device, module, "Fault Buffer Parser");

    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };

    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &(*fault_process_desc_layout),
    };
    fault_process_pipeline_layout = Vulkan::Check(device.createPipelineLayoutUnique(layout_info));

    const vk::ComputePipelineCreateInfo pipeline_info = {
        .stage = shader_ci,
        .layout = *fault_process_pipeline_layout,
    };
    fault_process_pipeline = Vulkan::Check(device.createComputePipelineUnique({}, pipeline_info));
    Vulkan::SetObjectName(device, *fault_process_pipeline, "Fault Buffer Parser Pipeline");

    device.destroyShaderModule(module);
}

void FaultManager::ReportFaultFrameCorrelation(
    std::span<const FaultFrameCorrelationObservation> observations) {
    auto& diagnostic = GetFaultFrameCorrelationRuntime();
    const auto config = diagnostic.GetConfiguration();
    if (!config.enabled) {
        return;
    }
    for (const auto& observation : observations) {
        LOG_INFO(Render_Vulkan,
                 "Fault frame correlation: frame={} process_time_us={} page_count={} "
                 "batch_count={} stable={} change={} exact_aba={} status={} loss={}",
                 observation.frame_sequence, observation.process_time_us, observation.page_count,
                 observation.batch_count, observation.stable, observation.changed,
                 observation.exact_aba, StatusName(observation.status),
                 static_cast<u32>(observation.loss));
    }
    const auto coverage = diagnostic.GetCoverage();
    LOG_INFO(Render_Vulkan,
             "Fault frame correlation coverage: first_frame={} frame_count={} page_cap={} "
             "selected_frames={} emitted_frames={} complete_frames={} no_batch_frames={} "
             "incomplete_frames={} gap_frames={} stable_frames={} changed_frames={} "
             "exact_aba_frames={} total_batches={} total_unique_pages={} dropped_pages={}",
             config.first_frame, config.frame_count, config.page_cap, coverage.selected_frames,
             observations.size(), coverage.complete_frames, coverage.no_batch_frames,
             coverage.incomplete_frames, coverage.gap_frames, coverage.stable_frames,
             coverage.changed_frames, coverage.exact_aba_frames, coverage.total_batches,
             coverage.total_unique_pages, coverage.dropped_pages);
}

void FaultManager::ProcessFaultBuffer() {
    // Capture the parsed-flip attribution before scheduling any work. The deferred callback may
    // execute after later flips have already been parsed.
    const auto correlation_stamp = GetFaultFrameCorrelationRuntime().CaptureStamp();
    if (u64 wait_tick = fault_areas[current_area]) {
        scheduler.Wait(wait_tick);
        scheduler.PopPendingOperations();
    }

    const u32 offset = current_area * PageFaultAreaSize;
    u8* mapped = download_buffer.mapped_data.data() + offset;
    std::memset(mapped, 0, PageFaultAreaSize);

    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .buffer = fault_buffer.Handle(),
        .offset = 0,
        .size = fault_buffer_size,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .buffer = fault_buffer.Handle(),
        .offset = 0,
        .size = fault_buffer_size,
    };
    const vk::DescriptorBufferInfo fault_buffer_info = {
        .buffer = fault_buffer.Handle(),
        .offset = 0,
        .range = fault_buffer_size,
    };
    const vk::DescriptorBufferInfo download_info = {
        .buffer = download_buffer.Handle(),
        .offset = offset,
        .range = PageFaultAreaSize,
    };
    const std::array<vk::WriteDescriptorSet, 2> writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &fault_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &download_info,
        },
    }};
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    scheduler.BindPipeline(Vulkan::PipelineBindPoint::Compute, *fault_process_pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *fault_process_pipeline_layout, 0,
                                writes);
    // 1 bit per page, 32 pages per workgroup
    const u32 num_threads = caching_num_pages / 32;
    const u32 num_workgroups = Common::DivCeil(num_threads, 64u);
    cmdbuf.dispatch(num_workgroups, 1, 1);

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });

    scheduler.DeferOperation([this, mapped, area = current_area, correlation_stamp] {
        fault_ranges.Clear();
        const u64* fault_buf = std::bit_cast<const u64*>(mapped);
        const u32 reported_fault_count = fault_buf[0];
        const auto fault_count =
            BoundFaultDownloadCount(reported_fault_count, FaultDownloadSlotCount);
        if (fault_count.overflowed) {
            LOG_WARNING(Render_Vulkan,
                        "GPU fault download overflow: reported {} page(s), retained first {} "
                        "address slot(s)",
                        reported_fault_count, fault_count.address_count);
        }
        if (correlation_stamp.selected) {
            std::vector<u64> private_page_ids;
            private_page_ids.reserve(fault_count.address_count);
            for (size_t i = 1; i <= fault_count.address_count; ++i) {
                private_page_ids.push_back(fault_buf[i] / caching_pagesize);
            }
            GetFaultFrameCorrelationRuntime().ObserveBatch(
                correlation_stamp, private_page_ids, fault_count.overflowed);
        }
        const VAddr address_space_size = caching_num_pages * caching_pagesize;
        u32 invalid_fault_count = 0;
        VAddr first_invalid_fault = 0;
        for (size_t i = 1; i <= fault_count.address_count; ++i) {
            const VAddr start = fault_buf[i];
            const VAddr end = start + caching_pagesize;
            const bool is_processable = IsProcessableDmaFaultRange(
                start, end, address_space_size,
                [this](VAddr address, u64 size) { return page_manager.IsGpuMapped(address, size); });
            if (!is_processable) {
                if (invalid_fault_count++ == 0) {
                    first_invalid_fault = start;
                }
                continue;
            }
            fault_ranges.Add(start, caching_pagesize);
            LOG_INFO(Render_Vulkan, "Accessed non-GPU cached memory at {:#x}", start);
        }
        if (invalid_fault_count != 0) {
            LOG_WARNING(Render_Vulkan,
                        "Ignored {} invalid or unmapped GPU fault page(s), first at {:#x}",
                        invalid_fault_count, first_invalid_fault);
        }
        fault_ranges.ForEach([&](VAddr start, VAddr end) {
            const bool is_processable = IsProcessableDmaFaultRange(
                start, end, address_space_size,
                [this](VAddr address, u64 size) { return page_manager.IsGpuMapped(address, size); });
            if (!is_processable) {
                LOG_WARNING(Render_Vulkan,
                            "Ignoring invalid or unmapped merged GPU fault range {:#x}-{:#x}", start,
                            end);
                return;
            }
            MakeDmaFaultRangeResident(buffer_cache, start, static_cast<u32>(end - start));
        });
        fault_areas[area] = 0;
    });

    fault_areas[current_area++] = scheduler.CurrentTick();
    current_area %= MaxPendingFaults;
}

} // namespace VideoCore
