// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/div_ceil.h"
#include "common/logging/log.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/fault_range.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include "video_core/host_shaders/fault_buffer_process_comp.h"

#include <algorithm>
#include <bit>
#include <iterator>
#include <limits>
#include <vector>

#include <fmt/format.h>

namespace VideoCore {

static constexpr size_t MaxPageFaults = 1024;
static constexpr size_t PageFaultAreaSize = MaxPageFaults * sizeof(u64);

FaultManager::FaultManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler_,
                           BufferCache& buffer_cache_, PageManager& page_manager_,
                           u32 caching_pagebits, u64 caching_num_pages_)
    : scheduler{scheduler_}, buffer_cache{buffer_cache_}, page_manager{page_manager_},
      caching_pagesize{1ULL << caching_pagebits}, caching_num_pages{caching_num_pages_},
      fault_page_buffer_size{caching_num_pages_ / 8},
      bda_fallback_requested{BdaFallbackConsumptionDiagnosticEnabled()},
      bda_fallback_config{ReadBdaFallbackWindowConfig()},
      bda_fallback_enabled{bda_fallback_requested && IsBdaFallbackConfigValid(bda_fallback_config)},
      bda_fallback_word_count{
          bda_fallback_requested
              ? std::max<size_t>(1, BdaFallbackWindowWordCount(bda_fallback_config))
              : 0},
      fault_buffer_size{fault_page_buffer_size + bda_fallback_word_count * sizeof(u32)},
      fault_buffer{instance, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, fault_buffer_size},
      download_buffer{
          instance,
          scheduler,
          MemoryUsage::Download,
          0,
          AllFlags,
          MaxPendingFaults * PageFaultAreaSize + bda_fallback_word_count * sizeof(u32)} {
    const auto device = instance.GetDevice();
    Vulkan::SetObjectName(device, fault_buffer.Handle(), "Fault Buffer");
    if (bda_fallback_requested) {
        fault_buffer.Fill(fault_page_buffer_size,
                          static_cast<u32>(bda_fallback_word_count * sizeof(u32)), 0);
        if (bda_fallback_enabled) {
            LOG_INFO(Render_Vulkan,
                     "BDA fallback consumption diagnostic enabled: first_frame={} frames={} "
                     "max_operations={} words={}",
                     bda_fallback_config.first_frame, bda_fallback_config.frame_count,
                     bda_fallback_config.max_operations_per_frame, bda_fallback_word_count);
        } else {
            LOG_ERROR(Render_Vulkan,
                      "BDA fallback consumption diagnostic unavailable: invalid bounded window");
        }
    }

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

std::pair<u32, u32> FaultManager::BdaFallbackTokens(u64 frame, u32 operation) noexcept {
    const auto plan =
        PlanBdaFallbackMark(bda_fallback_config, bda_fallback_enabled, frame, operation, 0);
    if (plan.status == BdaFallbackMarkStatus::OperationOverflow &&
        frame >= bda_fallback_config.first_frame) {
        const u64 frame_index = frame - bda_fallback_config.first_frame;
        if (frame_index < bda_fallback_config.frame_count) {
            bda_fallback_operation_overflow[frame_index].store(true, std::memory_order_relaxed);
        }
    }
    if (plan.status != BdaFallbackMarkStatus::Valid) {
        return {0, 0};
    }
    return {static_cast<u32>(plan.bit_index), 1};
}

void FaultManager::ObserveBdaFallbackFrameBoundary(u64 frame, u64 process_time_us) noexcept {
    if (!bda_fallback_enabled || frame < bda_fallback_config.first_frame) {
        return;
    }
    const u64 frame_index = frame - bda_fallback_config.first_frame;
    if (frame_index < bda_fallback_config.frame_count) {
        bda_fallback_process_time_us[frame_index].store(process_time_us, std::memory_order_release);
    }
}

void FaultManager::ProcessFaultBuffer() {
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
        .size = fault_page_buffer_size,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .buffer = fault_buffer.Handle(),
        .offset = 0,
        .size = fault_page_buffer_size,
    };
    const vk::DescriptorBufferInfo fault_buffer_info = {
        .buffer = fault_buffer.Handle(),
        .offset = 0,
        .range = fault_page_buffer_size,
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

    const u64 current_frame = GetBdaFallbackParsedFrameSequence().Read().sequence;
    const u64 window_end = bda_fallback_config.first_frame + bda_fallback_config.frame_count;
    if (bda_fallback_enabled && !bda_fallback_readback_requested && current_frame >= window_end) {
        bda_fallback_readback_requested = true;
        const u64 diagnostic_size = bda_fallback_word_count * sizeof(u32);
        const u64 download_offset = MaxPendingFaults * PageFaultAreaSize;
        u8* const diagnostic_mapped = download_buffer.mapped_data.data() + download_offset;
        std::memset(diagnostic_mapped, 0, diagnostic_size);

        const vk::BufferMemoryBarrier2 diagnostic_pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = fault_buffer.Handle(),
            .offset = fault_page_buffer_size,
            .size = diagnostic_size,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &diagnostic_pre_barrier,
        });
        const vk::BufferCopy diagnostic_copy = {
            .srcOffset = fault_page_buffer_size,
            .dstOffset = download_offset,
            .size = diagnostic_size,
        };
        cmdbuf.copyBuffer(fault_buffer.Handle(), download_buffer.Handle(), diagnostic_copy);
        const vk::BufferMemoryBarrier2 diagnostic_post_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
            .buffer = download_buffer.Handle(),
            .offset = download_offset,
            .size = diagnostic_size,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &diagnostic_post_barrier,
        });

        scheduler.DeferOperation([this, diagnostic_mapped, download_offset, diagnostic_size] {
            download_buffer.InvalidateMappedRange(download_offset, diagnostic_size);
            const auto source =
                std::span{std::bit_cast<const u32*>(diagnostic_mapped), bda_fallback_word_count};
            const u64 bits_per_frame =
                static_cast<u64>(bda_fallback_config.max_operations_per_frame) *
                NumBdaFallbackLogicalStages;
            const size_t words_per_frame = BdaFallbackFrameWordCount(bda_fallback_config);
            BdaFallbackFrameReducer reducer{bda_fallback_config};
            u64 complete{};
            u64 stable{};
            u64 changed{};
            u64 aba{};
            u64 overflow{};
            u64 incomplete{};
            u64 marked_bits{};
            u64 anomaly_reports{};
            u64 anomaly_truncated{};
            u64 detail_truncated{};
            std::vector<u32> previous_words;
            for (u32 frame_index = 0; frame_index < bda_fallback_config.frame_count;
                 ++frame_index) {
                std::vector<u32> frame_words(words_per_frame);
                const u64 source_bit = static_cast<u64>(frame_index) * bits_per_frame;
                const size_t source_word = static_cast<size_t>(source_bit / 32);
                const u32 source_shift = static_cast<u32>(source_bit % 32);
                for (size_t word = 0; word < words_per_frame; ++word) {
                    const size_t index = source_word + word;
                    u64 combined = index < source.size() ? source[index] : 0;
                    if (source_shift != 0 && index + 1 < source.size()) {
                        combined |= static_cast<u64>(source[index + 1]) << 32;
                    }
                    frame_words[word] = static_cast<u32>(combined >> source_shift);
                }
                if (const u32 tail_bits = static_cast<u32>(bits_per_frame % 32); tail_bits != 0) {
                    frame_words.back() &= (1U << tail_bits) - 1;
                }
                u64 frame_marked_bits{};
                for (const u32 word : frame_words) {
                    frame_marked_bits += std::popcount(word);
                }
                marked_bits += frame_marked_bits;
                const bool operation_overflow =
                    bda_fallback_operation_overflow[frame_index].load(std::memory_order_relaxed);
                const u64 frame = bda_fallback_config.first_frame + frame_index;
                const u64 process_time =
                    bda_fallback_process_time_us[frame_index].load(std::memory_order_acquire);
                const auto observation =
                    reducer.Observe(frame, frame_words,
                                    process_time == 0 ? BdaFallbackFrameAvailability::Incomplete
                                                      : BdaFallbackFrameAvailability::Complete,
                                    operation_overflow);
                LOG_INFO(Render_Vulkan,
                         "BDA fallback consumption frame: frame={} flip_process_time_us={} "
                         "status={} marked_stage_bits={} changed={} exact_aba={} "
                         "aba_middle_frame={}",
                         frame, process_time, static_cast<u32>(observation.status),
                         frame_marked_bits, observation.changed_from_previous,
                         observation.exact_aba_return, observation.aba_middle_frame);
                if (observation.status == BdaFallbackFrameStatus::Incomplete) {
                    ++incomplete;
                    previous_words.clear();
                    continue;
                }
                if (observation.status == BdaFallbackFrameStatus::OperationOverflow) {
                    ++overflow;
                    previous_words.clear();
                    continue;
                }
                ++complete;
                if (observation.has_previous) {
                    observation.changed_from_previous ? ++changed : ++stable;
                }
                aba += observation.exact_aba_return;
                if (observation.changed_from_previous) {
                    u64 changed_stage_bits{};
                    std::array<u64, 8> sample_bits{};
                    size_t sample_count{};
                    for (size_t word = 0; word < frame_words.size(); ++word) {
                        u32 different = frame_words[word] ^ previous_words[word];
                        while (different != 0) {
                            const u32 bit = std::countr_zero(different);
                            const u64 bit_index = static_cast<u64>(word) * 32 + bit;
                            if (sample_count < sample_bits.size()) {
                                sample_bits[sample_count++] = bit_index;
                            }
                            ++changed_stage_bits;
                            different &= different - 1;
                        }
                    }
                    detail_truncated += changed_stage_bits - sample_count;
                    if (anomaly_reports < MaxBdaFallbackWindowFrames) {
                        std::string samples;
                        for (size_t sample = 0; sample < sample_count; ++sample) {
                            const u64 operation = sample_bits[sample] / NumBdaFallbackLogicalStages;
                            const u64 stage = sample_bits[sample] % NumBdaFallbackLogicalStages;
                            fmt::format_to(std::back_inserter(samples), "{}:{}{}", operation, stage,
                                           sample + 1 == sample_count ? "" : ",");
                        }
                        LOG_WARNING(
                            Render_Vulkan,
                            "BDA fallback consumption event: frame={} process_time_us={} "
                            "previous_frame={} exact_aba={} aba_middle_frame={} "
                            "changed_stage_bits={} operation_stage_samples={} sample_loss={}",
                            observation.frame, process_time, observation.previous_frame,
                            observation.exact_aba_return, observation.aba_middle_frame,
                            changed_stage_bits, samples, changed_stage_bits - sample_count);
                        ++anomaly_reports;
                    } else {
                        ++anomaly_truncated;
                    }
                }
                previous_words = std::move(frame_words);
            }
            LOG_INFO(Render_Vulkan,
                     "BDA fallback consumption diagnostic: complete={} stable={} changed={} "
                     "exact_aba={} marked_stage_bits={}",
                     complete, stable, changed, aba, marked_bits);
            LOG_WARNING(Render_Vulkan,
                        "BDA fallback consumption coverage: unavailable=0 incomplete={} "
                        "capacity_loss=0 operation_overflow={} readbacks=1 anomaly_reports={} "
                        "anomaly_truncated={} operation_stage_detail_truncated={}",
                        incomplete, overflow, anomaly_reports, anomaly_truncated, detail_truncated);
        });
    }

    scheduler.DeferOperation([this, mapped, area = current_area] {
        fault_ranges.Clear();
        const u64* fault_buf = std::bit_cast<const u64*>(mapped);
        const u32 fault_count = fault_buf[0];
        const VAddr address_space_size = caching_num_pages * caching_pagesize;
        u32 invalid_fault_count = 0;
        VAddr first_invalid_fault = 0;
        for (u32 i = 1; i <= fault_count; ++i) {
            const VAddr start = fault_buf[i];
            const VAddr end = start + caching_pagesize;
            const bool is_processable = IsProcessableDmaFaultRange(
                start, end, address_space_size, [this](VAddr address, u64 size) {
                    return page_manager.IsGpuMapped(address, size);
                });
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
                start, end, address_space_size, [this](VAddr address, u64 size) {
                    return page_manager.IsGpuMapped(address, size);
                });
            if (!is_processable) {
                LOG_WARNING(Render_Vulkan,
                            "Ignoring invalid or unmapped merged GPU fault range {:#x}-{:#x}",
                            start, end);
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
