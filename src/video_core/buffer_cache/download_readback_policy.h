// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <utility>

#include "common/alignment.h"
#include "common/types.h"

namespace VideoCore {

struct DownloadReservation {
    u64 size;
    u64 alignment;
};

struct DownloadRange {
    VAddr address;
    u64 size;
};

struct MemorySnapshotOverlay {
    u64 output_offset;
    u64 download_offset;
    u64 size;
};

[[nodiscard]] constexpr std::optional<DownloadRange> ClipDownloadRange(VAddr tracked_address,
                                                                       u64 tracked_size,
                                                                       VAddr requested_address,
                                                                       u64 requested_size) {
    if (tracked_size > std::numeric_limits<VAddr>::max() - tracked_address ||
        requested_size > std::numeric_limits<VAddr>::max() - requested_address) {
        return std::nullopt;
    }
    const VAddr begin = std::max(tracked_address, requested_address);
    const VAddr end = std::min(tracked_address + tracked_size, requested_address + requested_size);
    if (begin >= end) {
        return std::nullopt;
    }
    return DownloadRange{begin, end - begin};
}

[[nodiscard]] constexpr DownloadReservation PlanDownloadReservation(u64 size, u64 alignment,
                                                                    u64 non_coherent_atom_size,
                                                                    bool is_coherent) {
    if (is_coherent) {
        return {size, alignment};
    }
    return {
        .size = Common::AlignUp(size, non_coherent_atom_size),
        .alignment = std::max(alignment, non_coherent_atom_size),
    };
}

template <typename Finish, typename Invalidate, typename Consume>
bool CompleteDownloadReadback(Finish&& finish, Invalidate&& invalidate, Consume&& consume) {
    std::forward<Finish>(finish)();
    if (!std::forward<Invalidate>(invalidate)()) {
        return false;
    }
    std::forward<Consume>(consume)();
    return true;
}

template <typename Finish, typename Invalidate, typename ReadCpuBacking, typename ApplyGpuOverlay>
bool CompletePrivateSnapshotReadback(Finish&& finish, Invalidate&& invalidate,
                                     ReadCpuBacking&& read_cpu_backing,
                                     ApplyGpuOverlay&& apply_gpu_overlay) {
    std::forward<Finish>(finish)();
    if (!std::forward<Invalidate>(invalidate)()) {
        return false;
    }
    if (!std::forward<ReadCpuBacking>(read_cpu_backing)()) {
        return false;
    }
    return std::forward<ApplyGpuOverlay>(apply_gpu_overlay)();
}

[[nodiscard]] inline bool ApplyMemorySnapshotOverlays(
    std::span<u8> output, std::span<const u8> download,
    std::span<const MemorySnapshotOverlay> overlays) {
    for (const auto& overlay : overlays) {
        if (overlay.output_offset > output.size() ||
            overlay.size > output.size() - overlay.output_offset ||
            overlay.download_offset > download.size() ||
            overlay.size > download.size() - overlay.download_offset) {
            return false;
        }
    }
    for (const auto& overlay : overlays) {
        std::memcpy(output.data() + overlay.output_offset,
                    download.data() + overlay.download_offset, overlay.size);
    }
    return true;
}

template <typename ReadCpuBacking, typename ReadGpuBuffer>
[[nodiscard]] bool ReadAuthoritativeMemorySnapshot(bool has_exact_gpu_modifications,
                                                   bool is_gpu_page_owned,
                                                   ReadCpuBacking&& read_cpu_backing,
                                                   ReadGpuBuffer&& read_gpu_buffer) {
    if (has_exact_gpu_modifications) {
        return std::forward<ReadGpuBuffer>(read_gpu_buffer)();
    }
    // An asynchronous download removes exact ownership before its completion callback publishes
    // bytes into guest backing. Coarse GPU ownership remains set during that window, so the CPU
    // copy is not yet authoritative and must not be sampled.
    if (is_gpu_page_owned) {
        return false;
    }
    return std::forward<ReadCpuBacking>(read_cpu_backing)();
}

} // namespace VideoCore
