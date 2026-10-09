// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>

#include "common/types.h"

namespace Vulkan {

constexpr u32 ComputeDiagnosticPreviewDwords = 24;
constexpr u32 LargeComputeWorkgroupThreads = 1024;
constexpr u64 ComputeDiagnosticSynchronizationLimit = 4096;

constexpr bool IsLargeComputeWorkgroup(u32 thread_x, u32 thread_y, u32 thread_z) {
    return static_cast<u64>(thread_x) * thread_y * thread_z >= LargeComputeWorkgroupThreads;
}

constexpr bool ShouldSynchronizeDiagnosticPreview(bool is_written, bool is_special,
                                                  bool is_gpu_modified, u64 size) {
    return !is_written && !is_special && is_gpu_modified && size > 0 &&
           size <= ComputeDiagnosticSynchronizationLimit;
}

struct DwordPreview {
    std::array<u32, ComputeDiagnosticPreviewDwords> words{};
    u32 size{};
};

inline DwordPreview CopyDwordPreview(const void* data, u64 byte_size) {
    DwordPreview result{};
    if (data == nullptr) {
        return result;
    }

    result.size =
        static_cast<u32>(std::min<u64>(byte_size / sizeof(u32), ComputeDiagnosticPreviewDwords));
    std::memcpy(result.words.data(), data, result.size * sizeof(u32));
    return result;
}

} // namespace Vulkan
