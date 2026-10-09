// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <span>
#include <utility>

#include "common/types.h"

namespace Shader {

constexpr u32 MaxAddr64PointerRoots = 8;
constexpr u32 MaxAddr64ReadConstDependencies = 32;
constexpr u32 MaxAddr64PointerTraversalNodes = 256;
constexpr u32 MaxReadConstPointerRoots = 32;
constexpr u32 Addr64PointerTableDwords = 64;
constexpr u32 Addr64PointerTableBytes = Addr64PointerTableDwords * sizeof(u32);

struct Addr64PointerRoot {
    u32 buffer_resource_index{};
    u32 pointer_lo_flat_index{};

    auto operator<=>(const Addr64PointerRoot&) const = default;
};

struct ReadConstPointerRoot {
    u32 pointer_lo_flat_index{};
    u32 minimum_offset_dwords{};
    u32 maximum_offset_dwords{};
    bool dynamic_offset{};

    auto operator<=>(const ReadConstPointerRoot&) const = default;
};

struct Addr64PointerRootSelection {
    std::array<Addr64PointerRoot, MaxAddr64PointerRoots> values{};
    u32 count{};
    bool overflow{};
};

enum class Addr64PointerTraversalInsert : u32 {
    Inserted,
    Duplicate,
    Overflow,
};

class Addr64PointerTraversalSet {
public:
    constexpr Addr64PointerTraversalInsert TryInsert(u64 node) noexcept {
        if (std::ranges::find(nodes.begin(), nodes.begin() + count, node) !=
            nodes.begin() + count) {
            return Addr64PointerTraversalInsert::Duplicate;
        }
        if (count == nodes.size()) {
            did_overflow = true;
            return Addr64PointerTraversalInsert::Overflow;
        }
        nodes[count++] = node;
        return Addr64PointerTraversalInsert::Inserted;
    }

    constexpr u32 size() const noexcept {
        return count;
    }

    constexpr bool overflow() const noexcept {
        return did_overflow;
    }

private:
    std::array<u64, MaxAddr64PointerTraversalNodes> nodes{};
    u32 count{};
    bool did_overflow{};
};

struct Addr64PointerResidencyPlan {
    u64 address{};
    u32 size{};
    bool valid{};
};

template <typename SnapshotPointers, typename PreparePointers, typename RefreshFlattenedUserData>
constexpr void PrepareReadConstPointersThenRefresh(
    u32 maximum_iterations, SnapshotPointers&& snapshot_pointers,
    PreparePointers&& prepare_pointers, RefreshFlattenedUserData&& refresh_flattened_user_data) {
    if (maximum_iterations == 0) {
        return;
    }
    auto previous = snapshot_pointers();
    for (u32 iteration = 0; iteration < maximum_iterations; ++iteration) {
        prepare_pointers();
        refresh_flattened_user_data();
        auto current = snapshot_pointers();
        if (current == previous) {
            return;
        }
        previous = std::move(current);
    }
}

constexpr Addr64PointerResidencyPlan PlanAddr64PointerResidency(
    u32 pointer_lo, u32 pointer_hi, u64 address_space_size) noexcept {
    const u64 address = u64{pointer_lo} | (u64{pointer_hi} << 32U);
    return {
        .address = address,
        .size = Addr64PointerTableBytes,
        .valid = address != 0 && address < address_space_size &&
                 Addr64PointerTableBytes <= address_space_size - address,
    };
}

constexpr Addr64PointerResidencyPlan PlanReadConstPointerResidency(
    u32 pointer_lo, u32 pointer_hi, u32 minimum_offset_dwords, u32 maximum_offset_dwords,
    bool dynamic_offset, u64 page_size, u64 address_space_size) noexcept {
    const u64 base = u64{pointer_lo} | (u64{pointer_hi} << 32U);
    if (base == 0 || base >= address_space_size || page_size == 0 ||
        (page_size & (page_size - 1U)) != 0) {
        return {};
    }
    const u64 first = dynamic_offset ? base : base + u64{minimum_offset_dwords} * sizeof(u32);
    const u64 last = dynamic_offset ? base : base + u64{maximum_offset_dwords} * sizeof(u32);
    if (first < base || last < first || last >= address_space_size) {
        return {};
    }
    const u64 start = first & ~(page_size - 1U);
    const u64 end = std::min(address_space_size, (last | (page_size - 1U)) + 1U);
    if (end <= start || end - start > std::numeric_limits<u32>::max()) {
        return {};
    }
    return {.address = start, .size = static_cast<u32>(end - start), .valid = true};
}

constexpr Addr64PointerRootSelection SelectAddr64PointerRoots(
    u32 buffer_resource_index, std::span<const u32> flattened_dependencies) noexcept {
    Addr64PointerRootSelection selection{};
    if (flattened_dependencies.size() > MaxAddr64ReadConstDependencies) {
        selection.overflow = true;
        return selection;
    }

    std::array<u32, MaxAddr64ReadConstDependencies> sorted{};
    u32 unique_count = 0;
    for (const u32 dependency : flattened_dependencies) {
        if (std::ranges::find(sorted.begin(), sorted.begin() + unique_count, dependency) ==
            sorted.begin() + unique_count) {
            sorted[unique_count++] = dependency;
        }
    }
    std::sort(sorted.begin(), sorted.begin() + unique_count);

    for (u32 i = 0; i + 1 < unique_count;) {
        if (sorted[i] + 1U != sorted[i + 1]) {
            ++i;
            continue;
        }
        if (selection.count == selection.values.size()) {
            return {.overflow = true};
        }
        selection.values[selection.count++] = {
            .buffer_resource_index = buffer_resource_index,
            .pointer_lo_flat_index = sorted[i],
        };
        i += 2;
    }
    return selection;
}

} // namespace Shader
