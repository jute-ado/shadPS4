// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/dma_publication_gate.h"
#include "video_core/buffer_cache/fault_download.h"
#include "video_core/buffer_cache/range_set.h"

namespace VideoCore {

class BufferCache;
class PageManager;

class FaultManager {
    static constexpr size_t MaxPendingFaults = 8;

public:
    explicit FaultManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                          BufferCache& buffer_cache, PageManager& page_manager,
                          u32 caching_pagebits, u64 caching_num_pages);

    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return &fault_buffer;
    }

    void ProcessFaultBuffer();

    [[nodiscard]] DmaFaultEpoch ProcessFaultBufferSynchronous();

private:
    Vulkan::Scheduler& scheduler;
    BufferCache& buffer_cache;
    PageManager& page_manager;
    RangeSet fault_ranges;
    u64 caching_pagesize;
    u64 caching_num_pages;
    u64 fault_buffer_size;
    Buffer fault_buffer;
    Buffer download_buffer;
    std::array<u64, MaxPendingFaults> fault_areas{};
    std::array<DmaFaultEpochCompletion, MaxPendingFaults> fault_epochs{};
    u32 current_area{};
    vk::UniqueDescriptorSetLayout fault_process_desc_layout;
    vk::UniquePipeline fault_process_pipeline;
    vk::UniquePipelineLayout fault_process_pipeline_layout;
};

} // namespace VideoCore
