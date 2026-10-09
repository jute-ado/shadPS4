// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/div_ceil.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/fault_range.h"
#include "video_core/buffer_cache/fault_record_snapshot.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include "video_core/host_shaders/fault_buffer_process_comp.h"

namespace VideoCore {

static constexpr size_t MaxPageFaults = MaxGpuFaultRecords + 1;
static constexpr size_t PageFaultAreaSize = GpuFaultSlotBytes;

FaultManager::FaultManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler_,
                           BufferCache& buffer_cache_, PageManager& page_manager_,
                           u32 caching_pagebits, u64 caching_num_pages_)
    : scheduler{scheduler_}, buffer_cache{buffer_cache_}, page_manager{page_manager_},
      caching_pagesize{1ULL << caching_pagebits}, caching_num_pages{caching_num_pages_},
      fault_buffer_size{caching_num_pages_ / 8},
      fault_buffer{instance, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, fault_buffer_size},
      download_buffer{instance, scheduler, MemoryUsage::Download,
                      0,        AllFlags,  MaxPendingFaults * PageFaultAreaSize},
      completed_faults{
          [this](std::span<const u64> records) { MakeCompletedFaultsResident(records); }} {
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
                                      fmt::format("MAX_PAGE_FAULTS={}", MaxPageFaults)}};
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

void FaultManager::ProcessFaultBuffer() {
    if (u64 wait_tick = fault_areas[current_area]) {
        scheduler.Wait(wait_tick);
        scheduler.PopPendingOperations();
    }

    const u32 offset = current_area * PageFaultAreaSize;

    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        .buffer = fault_buffer.Handle(),
        .offset = 0,
        .size = fault_buffer_size,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        .buffer = fault_buffer.Handle(),
        .offset = 0,
        .size = fault_buffer_size,
    };
    const vk::BufferMemoryBarrier2 reset_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        .buffer = download_buffer.Handle(),
        .offset = offset,
        .size = PageFaultAreaSize,
    };
    const vk::BufferMemoryBarrier2 readback_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .buffer = download_buffer.Handle(),
        .offset = offset,
        .size = PageFaultAreaSize,
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
    // Slot reuse already waits for its previous GPU tick and copies the old records. Clear on
    // the GPU so an atomic counter never depends on unflushed writes to noncoherent host memory.
    cmdbuf.fillBuffer(download_buffer.Handle(), offset, PageFaultAreaSize, 0);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &reset_barrier,
    });
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
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &readback_barrier,
    });

    scheduler.DeferOperation([this, area = current_area] { CollectCompletedFaults(area); });

    fault_areas[current_area++] = scheduler.CurrentTick();
    current_area %= MaxPendingFaults;
}

void FaultManager::CollectCompletedFaults(u32 area) {
    const u32 offset = area * PageFaultAreaSize;
    const auto slot =
        std::span<const u8>{download_buffer.mapped_data}.subspan(offset, PageFaultAreaSize);
    const auto snapshot = CaptureGpuFaultRecords(
        slot, [&] { return download_buffer.InvalidateMappedRange(offset, PageFaultAreaSize); });
    if (!snapshot) {
        LOG_ERROR(Render_Vulkan, "Could not invalidate completed GPU fault readback slot {}", area);
        fault_areas[area] = 0;
        return;
    }
    if (snapshot->plan.truncated) {
        LOG_WARNING(Render_Vulkan, "GPU fault slot reported {} records; only {} fit in the slot",
                    snapshot->plan.reported_count, snapshot->plan.read_count);
    }
    // This completion can run inside a stream upload or an SRT snapshot's Finish. Only copy
    // records here: creating/merging cached buffers would invalidate the suspended operation.
    completed_faults.Enqueue(snapshot->Records());
    fault_areas[area] = 0;
}

void FaultManager::DrainReadyFaults() {
    completed_faults.Drain();
}

void FaultManager::MakeCompletedFaultsResident(std::span<const u64> records) {
    RangeSet fault_ranges;
    const VAddr address_space_size = caching_num_pages * caching_pagesize;
    u32 invalid_fault_count = 0;
    VAddr first_invalid_fault = 0;
    for (const VAddr start : records) {
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
}

} // namespace VideoCore
