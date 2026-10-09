// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_result_retry.h"
#include "video_core/renderer_vulkan/submit_wait_diagnostic.h"

#include "common/assert.h"

namespace Vulkan {

constexpr u64 WAIT_TIMEOUT = std::numeric_limits<u64>::max();

MasterSemaphore::MasterSemaphore(const Instance& instance_) : instance{instance_} {
    const vk::StructureChain semaphore_chain = {
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        },
    };
    auto [semaphore_result, sem] =
        instance.GetDevice().createSemaphoreUnique(semaphore_chain.get());
    ASSERT_MSG(semaphore_result == vk::Result::eSuccess, "Failed to create master semaphore: {}",
               vk::to_string(semaphore_result));
    semaphore = std::move(sem);
}

MasterSemaphore::~MasterSemaphore() = default;

void MasterSemaphore::Refresh() {
    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        if (counter_result != vk::Result::eSuccess) {
            instance.LogDeviceMemoryBudget("semaphore_counter", counter_result);
        }
        ASSERT_MSG(counter_result == vk::Result::eSuccess,
                   "Failed to get master semaphore value: {}", vk::to_string(counter_result));
        counter = cntr;
        if (counter < this_tick) {
            return;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
}

void MasterSemaphore::Wait(u64 tick) {
    const bool trace_wait = SubmitWaitDiagnosticEnabled();
    const u64 wait_id = trace_wait ? NextWaitDiagnosticId() : 0;
    const u64 known_gpu_tick_before = KnownGpuTick();
    if (trace_wait) {
        LOG_INFO(Render_Vulkan,
                 "submit_wait wait_begin id={} target_tick={} logical_tick={} "
                 "known_gpu_tick={} fast_path={}",
                 wait_id, tick, CurrentTick(), known_gpu_tick_before,
                 known_gpu_tick_before >= tick);
    }
    // No need to wait if the GPU is ahead of the tick
    if (IsFree(tick)) {
        if (trace_wait) {
            LOG_INFO(Render_Vulkan,
                     "submit_wait wait_end id={} target_tick={} result=already_free "
                     "logical_tick={} known_gpu_tick={}",
                     wait_id, tick, CurrentTick(), KnownGpuTick());
        }
        return;
    }
    // Update the GPU tick and try again
    Refresh();
    if (IsFree(tick)) {
        if (trace_wait) {
            LOG_INFO(Render_Vulkan,
                     "submit_wait wait_end id={} target_tick={} result=free_after_refresh "
                     "logical_tick={} known_gpu_tick={}",
                     wait_id, tick, CurrentTick(), KnownGpuTick());
        }
        return;
    }

    if (trace_wait) {
        LOG_INFO(Render_Vulkan,
                 "submit_wait wait_block id={} target_tick={} logical_tick={} "
                 "known_gpu_tick={}",
                 wait_id, tick, CurrentTick(), KnownGpuTick());
    }

    // If none of the above is hit, fallback to a regular wait
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };

    const auto result = RetryOnTimeout([&] {
        return instance.GetDevice().waitSemaphores(&wait_info, WAIT_TIMEOUT);
    });
    if (trace_wait) {
        LOG_INFO(Render_Vulkan,
                 "submit_wait wait_result id={} target_tick={} result={} "
                 "logical_tick={} known_gpu_tick_before_refresh={}",
                 wait_id, tick, vk::to_string(result), CurrentTick(), KnownGpuTick());
    }
    if (result != vk::Result::eSuccess) {
        instance.LogDeviceMemoryBudget("semaphore_wait", result);
    }
    Check<"wait for master semaphore">(result);
    Refresh();
    if (trace_wait) {
        LOG_INFO(Render_Vulkan,
                 "submit_wait wait_end id={} target_tick={} result=completed "
                 "logical_tick={} known_gpu_tick={}",
                 wait_id, tick, CurrentTick(), KnownGpuTick());
    }
}

} // namespace Vulkan
