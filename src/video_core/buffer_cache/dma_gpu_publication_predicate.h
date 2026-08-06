// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace VideoCore {

struct GpuDmaPredicateSupport {
    bool extension_available{};
    bool conditional_rendering_feature{};
};

struct GpuDmaPredicateSlot {
    std::uint32_t index{};
    std::uint64_t generation{};
};

template <std::size_t SlotCount>
class GpuDmaPredicateSlotPool {
    static_assert(SlotCount > 0);
    static_assert(SlotCount <= std::numeric_limits<std::uint32_t>::max());

public:
    [[nodiscard]] constexpr std::optional<GpuDmaPredicateSlot> Acquire(
        std::uint64_t completed_gpu_tick) {
        for (std::size_t i = 0; i < slots.size(); ++i) {
            auto& slot = slots[i];
            if (slot.owned || slot.retire_tick > completed_gpu_tick) {
                continue;
            }
            slot.owned = true;
            if (++slot.generation == 0) {
                ++slot.generation;
            }
            return GpuDmaPredicateSlot{.index = static_cast<std::uint32_t>(i),
                                       .generation = slot.generation};
        }
        return std::nullopt;
    }

    [[nodiscard]] constexpr bool Owns(GpuDmaPredicateSlot lease) const {
        return lease.index < slots.size() && slots[lease.index].owned &&
               slots[lease.index].generation == lease.generation;
    }

    constexpr bool Retire(GpuDmaPredicateSlot lease, std::uint64_t submission_tick) {
        if (!Owns(lease) || submission_tick == 0) {
            return false;
        }
        auto& slot = slots[lease.index];
        slot.owned = false;
        slot.retire_tick = submission_tick;
        return true;
    }

    constexpr bool ReleaseUnsubmitted(GpuDmaPredicateSlot lease) {
        if (!Owns(lease)) {
            return false;
        }
        slots[lease.index].owned = false;
        return true;
    }

private:
    struct SlotState {
        std::uint64_t generation{};
        std::uint64_t retire_tick{};
        bool owned{};
    };
    std::array<SlotState, SlotCount> slots{};
};

enum class GpuDmaPredicateCommand {
    ResetClean,
    TransferWriteToShaderReadWriteBarrier,
    RasterDiscardDiscovery,
    ShaderWriteToConditionalReadBarrier,
    BeginConditionalPublication,
    PublicationDraw,
    EndConditionalPublication,
};

struct GpuDmaPublicationPlan {
    std::array<GpuDmaPredicateCommand, 7> commands;
    GpuDmaPredicateSlot slot;
    std::uint32_t predicate_value_for_clean{1};
    std::uint32_t predicate_value_for_fault{0};
    bool conditional_inverted{};
    bool requires_cpu_wait{};
};

template <std::size_t SlotCount>
[[nodiscard]] constexpr std::optional<GpuDmaPublicationPlan> BuildGpuDmaPublicationPlan(
    GpuDmaPredicateSupport support, std::optional<GpuDmaPredicateSlot> slot,
    const GpuDmaPredicateSlotPool<SlotCount>& pool) {
    if (!support.extension_available || !support.conditional_rendering_feature || !slot ||
        !pool.Owns(*slot)) {
        return std::nullopt;
    }
    return GpuDmaPublicationPlan{
        .commands = {GpuDmaPredicateCommand::ResetClean,
                     GpuDmaPredicateCommand::TransferWriteToShaderReadWriteBarrier,
                     GpuDmaPredicateCommand::RasterDiscardDiscovery,
                     GpuDmaPredicateCommand::ShaderWriteToConditionalReadBarrier,
                     GpuDmaPredicateCommand::BeginConditionalPublication,
                     GpuDmaPredicateCommand::PublicationDraw,
                     GpuDmaPredicateCommand::EndConditionalPublication},
        .slot = *slot,
    };
}

} // namespace VideoCore
