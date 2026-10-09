// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <new>
#include <span>
#include <utility>
#include <vector>

#include "core/libraries/ngs2/ngs2.h"
#include "core/memory.h"

namespace Libraries::Ngs2::GuestIo {

constexpr size_t MaxRenderDescriptorBytes = 1024 * 1024;
constexpr size_t MaxRenderOutputBytes = 64 * 1024 * 1024;

enum class SnapshotError {
    None,
    InvalidDescriptorList,
    InvalidOutputAddress,
    InvalidOutputSize,
    AllocationFailed,
};

struct RenderBufferSnapshot {
    std::vector<OrbisNgs2RenderBufferInfo> descriptors;
    std::vector<u8> staging;
    std::vector<Core::GuestMemoryRange> output_ranges;
    std::vector<Core::GuestMemoryWrite> output_writes;
};

struct RenderPublicationResult {
    s32 result{};
    bool attempted{};
    bool succeeded{};
};

// Synchronous public-export orchestration. The renderer controls the publication boundary;
// the writer receives every snapshotted output together, not one callback per buffer.
template <typename Render, typename WriteGuest>
RenderPublicationResult RenderPreparedBuffers(RenderBufferSnapshot& snapshot, Render&& render,
                                              WriteGuest&& write_guest) {
    RenderPublicationResult result;
    auto publish = [&] {
        result.attempted = true;
        result.succeeded =
            write_guest(std::span<const Core::GuestMemoryWrite>{snapshot.output_writes});
        return result.succeeded;
    };
    result.result = render(
        [](void* context) { return (*static_cast<decltype(publish)*>(context))(); }, &publish);
    return result;
}

template <typename ReadGuest, typename CanWriteGuest>
SnapshotError SnapshotRenderBuffers(const OrbisNgs2RenderBufferInfo* guest_descriptors,
                                    u32 descriptor_count, ReadGuest&& read_guest,
                                    CanWriteGuest&& can_write_guest,
                                    RenderBufferSnapshot* out_snapshot) {
    if (!out_snapshot) {
        return SnapshotError::InvalidDescriptorList;
    }
    *out_snapshot = {};
    if (descriptor_count == 0) {
        return SnapshotError::None;
    }
    if (!guest_descriptors ||
        descriptor_count > MaxRenderDescriptorBytes / sizeof(OrbisNgs2RenderBufferInfo)) {
        return SnapshotError::InvalidDescriptorList;
    }

    RenderBufferSnapshot candidate;
    try {
        candidate.descriptors.resize(descriptor_count);
        candidate.output_ranges.reserve(descriptor_count);
        candidate.output_writes.reserve(descriptor_count);
    } catch (const std::bad_alloc&) {
        return SnapshotError::AllocationFailed;
    }

    const size_t descriptor_bytes =
        static_cast<size_t>(descriptor_count) * sizeof(OrbisNgs2RenderBufferInfo);
    if (!std::forward<ReadGuest>(read_guest)(guest_descriptors, candidate.descriptors.data(),
                                             descriptor_bytes)) {
        return SnapshotError::InvalidDescriptorList;
    }

    size_t total_output_bytes = 0;
    for (const auto& descriptor : candidate.descriptors) {
        if (descriptor.bufferSize == 0) {
            continue;
        }
        if (!descriptor.buffer) {
            return SnapshotError::InvalidOutputAddress;
        }
        if (descriptor.bufferSize > MaxRenderOutputBytes - total_output_bytes) {
            return SnapshotError::InvalidOutputSize;
        }
        total_output_bytes += descriptor.bufferSize;
        candidate.output_ranges.push_back(
            {reinterpret_cast<VAddr>(descriptor.buffer), descriptor.bufferSize});
    }

    if (!std::forward<CanWriteGuest>(can_write_guest)(
            std::span<const Core::GuestMemoryRange>{candidate.output_ranges})) {
        return SnapshotError::InvalidOutputAddress;
    }

    try {
        candidate.staging.resize(total_output_bytes);
    } catch (const std::bad_alloc&) {
        return SnapshotError::AllocationFailed;
    }

    size_t output_index = 0;
    size_t staging_offset = 0;
    for (auto& descriptor : candidate.descriptors) {
        if (descriptor.bufferSize == 0) {
            descriptor.buffer = nullptr;
            continue;
        }
        const auto& output_range = candidate.output_ranges[output_index++];
        descriptor.buffer = candidate.staging.data() + staging_offset;
        candidate.output_writes.push_back(
            {output_range.address, descriptor.buffer, output_range.size});
        staging_offset += descriptor.bufferSize;
    }

    *out_snapshot = std::move(candidate);
    return SnapshotError::None;
}

} // namespace Libraries::Ngs2::GuestIo
