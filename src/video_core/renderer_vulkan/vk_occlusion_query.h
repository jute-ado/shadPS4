// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <optional>
#include <vector>

#include "common/types.h"
#include "video_core/amdgpu/predication_policy.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

// Accuracy-first ZPASS implementation. Query windows are resolved when the guest requests a
// pixel-pipe statistics dump, so IT_SET_PREDICATION observes real rasterized sample counts.
// This deliberately uses a CPU wait as a first correctness step; the interface keeps the query
// lifetime separate so it can later be moved to GPU-side conditional rendering.
class OcclusionQueryManager {
public:
    OcclusionQueryManager(const Instance& instance, Scheduler& scheduler);

    void ToggleCounting();
    void ResetCounting();
    void DumpCounters(VAddr address, u32 num_counter_pairs);

    [[nodiscard]] std::optional<u32> PrepareDrawQuery();
    void BeginDraw(vk::CommandBuffer cmdbuf, std::optional<u32> query);
    void EndDraw(vk::CommandBuffer cmdbuf, std::optional<u32> query);

private:
    void ResolveActiveQueries();
    void WriteGuestCounters(VAddr address, u32 num_counter_pairs) const;

    static constexpr u32 QueryPoolSize = 4096;

    const Instance& instance;
    Scheduler& scheduler;
    vk::UniqueQueryPool query_pool;
    std::array<bool, QueryPoolSize> query_busy{};
    std::vector<u32> active_queries;
    AmdGpu::ZpassCounterState counter;
    u32 query_cursor{};
    bool counting_enabled{};
    bool query_window_valid{true};
};

} // namespace Vulkan
