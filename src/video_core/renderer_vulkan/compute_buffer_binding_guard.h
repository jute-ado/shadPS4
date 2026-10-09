// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <utility>

#include "common/types.h"

namespace Vulkan {

struct ComputeBufferBindingRange {
    VAddr address;
    u64 size;
};

struct ComputeBufferBindingSelection {
    VAddr address;
    u64 size;
    u64 mapping_token;
};

template <typename RangeSizeProvider>
[[nodiscard]] ComputeBufferBindingRange ClampComputeBufferBindingRange(
    VAddr address, u64 declared_size, RangeSizeProvider&& clamp_range_size) {
    return {
        .address = address,
        .size = std::forward<RangeSizeProvider>(clamp_range_size)(address, declared_size),
    };
}

template <typename MappingTokenProvider>
[[nodiscard]] std::optional<ComputeBufferBindingSelection> SelectComputeBufferBinding(
    VAddr address, u64 size, MappingTokenProvider&& get_mapping_token) {
    const u64 mapping_token = std::forward<MappingTokenProvider>(get_mapping_token)(address, size);
    if (mapping_token == 0) {
        return std::nullopt;
    }
    return ComputeBufferBindingSelection{address, size, mapping_token};
}

template <typename MappingTokenProvider>
[[nodiscard]] std::optional<ComputeBufferBindingSelection> SelectComputeBufferBinding(
    const ComputeBufferBindingRange& range, MappingTokenProvider&& get_mapping_token) {
    return SelectComputeBufferBinding(range.address, range.size,
                                      std::forward<MappingTokenProvider>(get_mapping_token));
}

template <typename MappingTokenProvider>
[[nodiscard]] bool IsComputeBufferBindingCurrent(
    const ComputeBufferBindingSelection& selection, MappingTokenProvider&& get_mapping_token) {
    return selection.mapping_token != 0 &&
           std::forward<MappingTokenProvider>(get_mapping_token)(selection.address,
                                                                 selection.size) ==
               selection.mapping_token;
}

template <typename MappingTokenProvider>
[[nodiscard]] bool AreComputeBufferBindingsCurrent(
    std::span<const ComputeBufferBindingSelection> selections,
    MappingTokenProvider&& get_mapping_token) {
    return std::ranges::all_of(selections, [&](const ComputeBufferBindingSelection& selection) {
        return IsComputeBufferBindingCurrent(selection, get_mapping_token);
    });
}

} // namespace Vulkan
