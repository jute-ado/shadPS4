// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "video_core/buffer_cache/bda_fallback_consumption.h"
#include "video_core/buffer_cache/buffer.h"
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

    [[nodiscard]] std::pair<u32, u32> BdaFallbackTokens(u64 frame, u32 operation) noexcept;
    void ObserveBdaFallbackFrameBoundary(u64 frame, u64 process_time_us) noexcept;
    [[nodiscard]] bool NeedsBdaFallbackReadback(u64 frame) const noexcept {
        return bda_fallback_enabled && !bda_fallback_readback_requested &&
               frame >= bda_fallback_config.first_frame + bda_fallback_config.frame_count;
    }

private:
    Vulkan::Scheduler& scheduler;
    BufferCache& buffer_cache;
    PageManager& page_manager;
    RangeSet fault_ranges;
    u64 caching_pagesize;
    u64 caching_num_pages;
    u64 fault_page_buffer_size;
    bool bda_fallback_requested{};
    BdaFallbackWindowConfig bda_fallback_config{};
    bool bda_fallback_enabled{};
    size_t bda_fallback_word_count{};
    u64 fault_buffer_size;
    bool bda_fallback_readback_requested{};
    std::array<std::atomic_bool, MaxBdaFallbackWindowFrames> bda_fallback_operation_overflow{};
    std::array<std::atomic<u64>, MaxBdaFallbackWindowFrames> bda_fallback_process_time_us{};
    Buffer fault_buffer;
    Buffer download_buffer;
    std::array<u64, MaxPendingFaults> fault_areas{};
    u32 current_area{};
    vk::UniqueDescriptorSetLayout fault_process_desc_layout;
    vk::UniquePipeline fault_process_pipeline;
    vk::UniquePipelineLayout fault_process_pipeline_layout;
};

} // namespace VideoCore
