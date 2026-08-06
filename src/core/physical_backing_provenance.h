// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "common/types.h"

namespace Core {

enum class PhysicalMemoryType : u32 {
    Free = 0,
    Allocated = 1,
    Mapped = 2,
    Pooled = 3,
    Committed = 4,
    Flexible = 5,
};

struct PhysicalMemoryArea {
    PAddr base = 0;
    u64 size = 0;
    s32 memory_type = 0;
    PhysicalMemoryType dma_type = PhysicalMemoryType::Free;
    u64 allocation_generation = 0;

    [[nodiscard]] PAddr GetEnd() const {
        return base + size;
    }

    [[nodiscard]] bool CanMergeWith(const PhysicalMemoryArea& next) const {
        constexpr u64 Max = std::numeric_limits<u64>::max();
        if (size == 0 || next.size == 0 || base > Max - size || next.base > Max - next.size ||
            size > Max - next.size) {
            return false;
        }
        return base + size == next.base && memory_type == next.memory_type &&
               dma_type == next.dma_type && allocation_generation == next.allocation_generation;
    }
};

/// One bounded global source for physical ownership generations. Zero is reserved
/// for allocations whose provenance cannot safely be published.
class PhysicalBackingGenerationSource {
public:
    constexpr explicit PhysicalBackingGenerationSource(u64 last_generation = 0) noexcept
        : last_generation{last_generation} {}

    [[nodiscard]] constexpr u64 Acquire() noexcept {
        if (last_generation == std::numeric_limits<u64>::max()) {
            return 0;
        }
        return ++last_generation;
    }

private:
    u64 last_generation{};
};

inline void AcquirePhysicalBacking(PhysicalMemoryArea& area,
                                   PhysicalBackingGenerationSource& generations) noexcept {
    area.allocation_generation = generations.Acquire();
}

struct PhysicalBackingRetirement {
    u64 physical_offset{};
    u64 size{};
    u64 allocation_generation{};

    auto operator<=>(const PhysicalBackingRetirement&) const = default;
};

[[nodiscard]] inline std::optional<PhysicalBackingRetirement> RetirePhysicalBacking(
    PhysicalMemoryArea& area) noexcept {
    if (area.allocation_generation == 0) {
        return std::nullopt;
    }
    const PhysicalBackingRetirement retirement{
        .physical_offset = area.base,
        .size = area.size,
        .allocation_generation = area.allocation_generation,
    };
    area.allocation_generation = 0;
    return retirement;
}

[[nodiscard]] inline std::pair<PhysicalMemoryArea, PhysicalMemoryArea> SplitPhysicalBackingArea(
    const PhysicalMemoryArea& area, u64 offset) noexcept {
    PhysicalMemoryArea first = area;
    PhysicalMemoryArea second = area;
    first.size = offset;
    second.base += offset;
    second.size -= offset;
    return {first, second};
}

enum class PhysicalBackingMappingClass {
    Unsupported,
    Direct,
    Pooled,
    Flexible,
    File,
};

struct PhysicalBackingSpan {
    VAddr guest_base{};
    u64 physical_offset{};
    u64 size{};
    u64 allocation_generation{};

    auto operator<=>(const PhysicalBackingSpan&) const = default;
};

/// Collect exact canonical physical spans for one complete guest mapping. Any
/// unsupported or incomplete provenance rejects the whole collection.
template <typename PhysicalAreas>
[[nodiscard]] std::optional<std::vector<PhysicalBackingSpan>> CollectPhysicalBackingSpans(
    VAddr guest_base, u64 size, PhysicalBackingMappingClass mapping_class, bool eligible,
    const PhysicalAreas& areas) {
    constexpr u64 PageSize = 16_KB;
    constexpr u64 PageMask = PageSize - 1;
    if (!eligible ||
        (mapping_class != PhysicalBackingMappingClass::Direct &&
         mapping_class != PhysicalBackingMappingClass::Pooled) ||
        size == 0 || (guest_base & PageMask) != 0 || (size & PageMask) != 0 ||
        guest_base > std::numeric_limits<VAddr>::max() - (size - 1) || areas.empty()) {
        return std::nullopt;
    }

    std::vector<PhysicalBackingSpan> spans;
    const auto required_dma_type = mapping_class == PhysicalBackingMappingClass::Direct
                                       ? PhysicalMemoryType::Mapped
                                       : PhysicalMemoryType::Committed;
    u64 mapping_offset = 0;
    u64 remaining_size = size;
    auto area_it = areas.begin();
    while (remaining_size != 0) {
        if (area_it == areas.end() || area_it->first != mapping_offset) {
            return std::nullopt;
        }
        const auto& area = area_it->second;
        if (area.size == 0 || (area_it->first & PageMask) != 0 || (area.base & PageMask) != 0 ||
            (area.size & PageMask) != 0 || area.dma_type != required_dma_type ||
            area.allocation_generation == 0 || area.size > remaining_size ||
            area.base > std::numeric_limits<u64>::max() - (area.size - 1)) {
            return std::nullopt;
        }

        spans.push_back({
            .guest_base = guest_base + mapping_offset,
            .physical_offset = area.base,
            .size = area.size,
            .allocation_generation = area.allocation_generation,
        });
        mapping_offset += area.size;
        remaining_size -= area.size;
        ++area_it;
    }
    if (area_it != areas.end()) {
        return std::nullopt;
    }
    return spans;
}

} // namespace Core
