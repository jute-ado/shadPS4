// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <span>

#include "common/types.h"

namespace Shader {

constexpr u32 MaxAddr64PointerRoots = 8;
constexpr u32 MaxAddr64ReadConstDependencies = 32;
constexpr u32 MaxAddr64PointerTraversalNodes = 256;
constexpr u32 Addr64PointerTableDwords = 64;
constexpr u32 Addr64PointerTableBytes = Addr64PointerTableDwords * sizeof(u32);

struct Addr64PointerRoot {
    u32 buffer_resource_index{};
    u32 pointer_lo_flat_index{};

    auto operator<=>(const Addr64PointerRoot&) const = default;
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
