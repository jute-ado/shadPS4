// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_occlusion_query.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

OcclusionQueryManager::OcclusionQueryManager(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    const vk::QueryPoolCreateInfo create_info{
        .queryType = vk::QueryType::eOcclusion,
        .queryCount = QueryPoolSize,
    };
    query_pool = Check(instance.GetDevice().createQueryPoolUnique(create_info));
    SetObjectName(instance.GetDevice(), *query_pool, "Guest ZPASS Query Pool");
    if (instance.IsHostQueryResetSupported()) {
        instance.GetDevice().resetQueryPool(*query_pool, 0, QueryPoolSize);
    }
}

void OcclusionQueryManager::ToggleCounting() {
    counting_enabled = !counting_enabled;
}

void OcclusionQueryManager::ResetCounting() {
    if (!active_queries.empty()) {
        ResolveActiveQueries();
    }
    counter.Reset();
    query_window_valid = true;
}

std::optional<u32> OcclusionQueryManager::PrepareDrawQuery() {
    if (!counting_enabled) {
        return std::nullopt;
    }
    for (u32 attempt = 0; attempt < QueryPoolSize; ++attempt) {
        const u32 slot = (query_cursor + attempt) % QueryPoolSize;
        if (query_busy[slot]) {
            continue;
        }
        query_busy[slot] = true;
        query_cursor = (slot + 1) % QueryPoolSize;
        if (!instance.IsHostQueryResetSupported()) {
            scheduler.EndRendering();
            scheduler.CommandBuffer().resetQueryPool(*query_pool, slot, 1);
        }
        return slot;
    }
    query_window_valid = false;
    LOG_WARNING(Render_Vulkan, "ZPASS query pool exhausted; predication window will fail open");
    return std::nullopt;
}

void OcclusionQueryManager::BeginDraw(vk::CommandBuffer cmdbuf, std::optional<u32> query) {
    if (query) {
        cmdbuf.beginQuery(*query_pool, *query, {});
    }
}

void OcclusionQueryManager::EndDraw(vk::CommandBuffer cmdbuf, std::optional<u32> query) {
    if (query) {
        cmdbuf.endQuery(*query_pool, *query);
        active_queries.push_back(*query);
    }
}

void OcclusionQueryManager::ResolveActiveQueries() {
    if (active_queries.empty()) {
        return;
    }

    scheduler.EndRendering();
    scheduler.Finish();

    u64 samples{};
    for (const u32 slot : active_queries) {
        u64 result{};
        const auto status = instance.GetDevice().getQueryPoolResults(
            *query_pool, slot, 1, sizeof(result), &result, sizeof(result),
            vk::QueryResultFlagBits::e64);
        if (status == vk::Result::eSuccess) {
            samples += result;
        } else {
            query_window_valid = false;
            LOG_WARNING(Render_Vulkan, "ZPASS query {} could not be read: {}", slot,
                        vk::to_string(status));
        }
        query_busy[slot] = false;
        if (instance.IsHostQueryResetSupported()) {
            instance.GetDevice().resetQueryPool(*query_pool, slot, 1);
        }
    }
    active_queries.clear();
    if (query_window_valid) {
        (void)counter.AddSamples(samples);
    }
    static u32 logged_windows{};
    if (logged_windows++ < 200) {
        LOG_INFO(Render_Vulkan, "TEMP ZPASS resolved samples={} valid={} counter={:#x}", samples,
                 query_window_valid, counter.Snapshot());
    }
}

void OcclusionQueryManager::DumpCounters(VAddr address, u32 num_counter_pairs) {
    if (address == 0 || num_counter_pairs == 0) {
        return;
    }
    ResolveActiveQueries();
    if (query_window_valid) {
        WriteGuestCounters(address, num_counter_pairs);
    }
    query_window_valid = true;
}

void OcclusionQueryManager::WriteGuestCounters(VAddr address, u32 num_counter_pairs) const {
    auto* memory = Core::Memory::Instance();
    const u64 result_size = u64(num_counter_pairs) * sizeof(u64) * 2;
    if (!memory->IsValidMapping(address, result_size)) {
        LOG_WARNING(Render_Vulkan, "Pixel-pipe stats writeback address is invalid: {:#x}", address);
        return;
    }
    const u64 value = counter.Snapshot();
    for (u32 pair = 0; pair < num_counter_pairs; ++pair) {
        void* const destination = reinterpret_cast<void*>(address + pair * sizeof(u64) * 2);
        if (!memory->TryWriteBacking(destination, &value, sizeof(value))) {
            std::memcpy(destination, &value, sizeof(value));
        }
    }
}

} // namespace Vulkan
