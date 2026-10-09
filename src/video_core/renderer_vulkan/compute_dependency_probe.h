// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <string_view>
#include <utility>

#include <vulkan/vulkan.hpp>

namespace Vulkan {

inline bool IsComputeDependencyProbeEnabled(const char* value) noexcept {
    return value && std::string_view{value} == "1";
}

class ComputeDependencyProbe {
public:
    explicit ComputeDependencyProbe(bool enabled) : enabled{enabled} {}

    template <typename DispatchCommand, typename Dependency, typename Log>
    void Dispatch(DispatchCommand&& dispatch, Dependency&& dependency, Log&& log) {
        std::forward<DispatchCommand>(dispatch)();
        if (!enabled) {
            return;
        }
        // Diagnostic only. Callers supply a literal Vulkan dispatch, already outside rendering.
        // This does not change resource tracking or wait for host-visible GPU completion.
        vk::MemoryBarrier2 barrier{};
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
        std::forward<Dependency>(dependency)(barrier);
        if (!logged.test_and_set(std::memory_order_relaxed)) {
            try {
                std::forward<Log>(log)();
            } catch (...) {
                // A diagnostic sink must not interrupt an already recorded command/dependency.
                // Keep the attempt spent; dispatch and dependency exceptions still propagate.
            }
        }
    }

private:
    const bool enabled;
    std::atomic_flag logged = ATOMIC_FLAG_INIT;
};

inline ComputeDependencyProbe& GetComputeDependencyProbe() {
    static ComputeDependencyProbe probe{
        IsComputeDependencyProbeEnabled(std::getenv("SHADPS4_COMPUTE_DEPENDENCY_PROBE"))};
    return probe;
}

} // namespace Vulkan
