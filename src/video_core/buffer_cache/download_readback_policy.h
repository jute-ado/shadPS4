// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

#include "common/alignment.h"
#include "common/types.h"
#include "video_core/shader_capture_provenance.h"

namespace VideoCore {

struct DownloadReservation {
    u64 size;
    u64 alignment;
};

struct DownloadRange {
    VAddr address;
    u64 size;
};

[[nodiscard]] constexpr std::optional<DownloadRange> ClipDownloadRange(
    VAddr tracked_address, u64 tracked_size, VAddr requested_address, u64 requested_size) {
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

template <typename ReadCpuBacking, typename ReadGpuBuffer>
[[nodiscard]] SnapshotReadProvenance ReadAuthoritativeMemorySnapshotWithProvenance(
    bool has_exact_gpu_modifications, bool is_gpu_page_owned, ReadCpuBacking&& read_cpu_backing,
    ReadGpuBuffer&& read_gpu_buffer) {
    if (has_exact_gpu_modifications) {
        const bool completed = std::forward<ReadGpuBuffer>(read_gpu_buffer)();
        return {
            .completed = completed,
            .exact_gpu_modified = true,
            .coarse_gpu_owned = is_gpu_page_owned,
            .source = completed ? ShaderCaptureSource::GpuBuffer : ShaderCaptureSource::Refused,
        };
    }
    // Async readback transfers exact ownership before its completion callback publishes the
    // downloaded bytes. Coarse GPU ownership remains set during that window, so guest backing is
    // not yet authoritative even though there is no longer an exact GPU-modified range.
    if (is_gpu_page_owned) {
        return {
            .completed = false,
            .exact_gpu_modified = false,
            .coarse_gpu_owned = true,
            .source = ShaderCaptureSource::Refused,
        };
    }
    const bool completed = std::forward<ReadCpuBacking>(read_cpu_backing)();
    return {
        .completed = completed,
        .exact_gpu_modified = false,
        .coarse_gpu_owned = false,
        .source = completed ? ShaderCaptureSource::CpuBacking : ShaderCaptureSource::Refused,
    };
}

template <typename ReadCpuBacking, typename ReadGpuBuffer>
[[nodiscard]] bool ReadAuthoritativeMemorySnapshot(bool has_exact_gpu_modifications,
                                                   bool is_gpu_page_owned,
                                                   ReadCpuBacking&& read_cpu_backing,
                                                   ReadGpuBuffer&& read_gpu_buffer) {
    return ReadAuthoritativeMemorySnapshotWithProvenance(
               has_exact_gpu_modifications, is_gpu_page_owned,
               std::forward<ReadCpuBacking>(read_cpu_backing),
               std::forward<ReadGpuBuffer>(read_gpu_buffer))
        .completed;
}

} // namespace VideoCore

