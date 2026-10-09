// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

#include "common/types.h"
#include "video_core/buffer_cache/cpu_write_focus_diagnostics.h"

namespace VideoCore {

enum class SrtSnapshotCompositionRoute : u8 {
    Unavailable,
    CpuOnly,
    GpuComposed,
};

[[nodiscard]] constexpr std::string_view SrtSnapshotCompositionRouteName(
    SrtSnapshotCompositionRoute route) noexcept {
    switch (route) {
    case SrtSnapshotCompositionRoute::Unavailable:
        return "unavailable";
    case SrtSnapshotCompositionRoute::CpuOnly:
        return "cpu_only";
    case SrtSnapshotCompositionRoute::GpuComposed:
        return "gpu_composed";
    default:
        return "invalid";
    }
}

struct SrtSnapshotCompositionReceiptConfig {
    VAddr snapshot_address{};
    u64 snapshot_size{};
    VAddr focus_address{};
    u64 focus_size{};
    SrtSnapshotCompositionRoute route{SrtSnapshotCompositionRoute::Unavailable};
    u32 overlay_count{};
    u64 overlay_bytes{};
    // Diagnostic-only correlation id assigned to one private snapshot attempt. Zero means that
    // overlay diagnostics were not armed for this receipt.
    u64 diagnostic_snapshot_id{};
};

// An immutable diagnostic value. It deliberately owns the sampled bytes so callers can log it
// after the private readback buffers and their guest mappings have gone out of scope.
struct SrtSnapshotFocusOwner {
    VAddr source_address{};
    u64 source_size{};
    u64 source_offset{};
    u64 output_offset{};
    u64 size{};
    u32 source_buffer_id{};
    u32 mapped_buffer_id{};
    u64 owner_generation{};
};

struct SrtSnapshotCompositionReceipt {
    static constexpr size_t MaximumSampleBytes = 32;
    static constexpr size_t MaximumFocusOwners = 4;

    VAddr snapshot_address{};
    u64 snapshot_size{};
    VAddr focus_address{};
    u64 focus_size{};
    SrtSnapshotCompositionRoute route{SrtSnapshotCompositionRoute::Unavailable};
    u32 sample_size{};
    std::array<u8, MaximumSampleBytes> cpu_sample{};
    std::array<u8, MaximumSampleBytes> composed_sample{};
    u32 sample_differing_byte_count{};
    std::optional<u32> first_differing_focus_offset{};
    std::optional<u32> last_differing_focus_offset{};
    u32 overlay_count{};
    u64 overlay_bytes{};
    u64 diagnostic_snapshot_id{};
    CpuWriteFocusDiagnostic cpu_write_focus{};
    std::array<SrtSnapshotFocusOwner, MaximumFocusOwners> focus_owners{};
    u32 focus_owner_count{};
    u32 focus_owner_total{};
};

// Per-read, synchronous request carrier. Input focus metadata remains separate from the owned
// result so the diagnostic cannot mistake partially initialized output for valid evidence.
struct SrtSnapshotCompositionReceiptRequest {
    VAddr focus_address{};
    u64 focus_size{};
    std::optional<SrtSnapshotCompositionReceipt> result{};
};

[[nodiscard]] inline std::optional<SrtSnapshotCompositionReceipt>
CaptureSrtSnapshotCompositionReceipt(const SrtSnapshotCompositionReceiptConfig& config,
                                     std::span<const u8> cpu_snapshot,
                                     std::span<const u8> composed_snapshot) noexcept {
    constexpr VAddr MaximumAddress = std::numeric_limits<VAddr>::max();
    if (config.snapshot_size == 0 || config.focus_size == 0 ||
        config.snapshot_size > MaximumAddress - config.snapshot_address ||
        config.focus_size > MaximumAddress - config.focus_address ||
        config.focus_address < config.snapshot_address) {
        return std::nullopt;
    }

    const VAddr focus_offset = config.focus_address - config.snapshot_address;
    if (focus_offset > config.snapshot_size ||
        config.focus_size > config.snapshot_size - focus_offset ||
        config.snapshot_size > std::numeric_limits<size_t>::max()) {
        return std::nullopt;
    }

    const bool overlays_are_empty = config.overlay_count == 0 && config.overlay_bytes == 0;
    const bool overlays_are_present = config.overlay_count != 0 && config.overlay_bytes != 0;
    switch (config.route) {
    case SrtSnapshotCompositionRoute::Unavailable:
        if (!overlays_are_empty || !cpu_snapshot.empty() || !composed_snapshot.empty()) {
            return std::nullopt;
        }
        return SrtSnapshotCompositionReceipt{
            .snapshot_address = config.snapshot_address,
            .snapshot_size = config.snapshot_size,
            .focus_address = config.focus_address,
            .focus_size = config.focus_size,
            .route = config.route,
            .diagnostic_snapshot_id = config.diagnostic_snapshot_id,
        };
    case SrtSnapshotCompositionRoute::CpuOnly:
        if (!overlays_are_empty || cpu_snapshot.size() != config.snapshot_size ||
            !composed_snapshot.empty()) {
            return std::nullopt;
        }
        break;
    case SrtSnapshotCompositionRoute::GpuComposed:
        if (!overlays_are_present || cpu_snapshot.size() != config.snapshot_size ||
            composed_snapshot.size() != config.snapshot_size) {
            return std::nullopt;
        }
        break;
    default:
        return std::nullopt;
    }

    SrtSnapshotCompositionReceipt receipt{
        .snapshot_address = config.snapshot_address,
        .snapshot_size = config.snapshot_size,
        .focus_address = config.focus_address,
        .focus_size = config.focus_size,
        .route = config.route,
        .sample_size = static_cast<u32>(
            std::min<u64>(config.focus_size, SrtSnapshotCompositionReceipt::MaximumSampleBytes)),
        .overlay_count = config.overlay_count,
        .overlay_bytes = config.overlay_bytes,
        .diagnostic_snapshot_id = config.diagnostic_snapshot_id,
    };
    const size_t sample_offset = static_cast<size_t>(focus_offset);
    const size_t sample_size = receipt.sample_size;
    std::ranges::copy(cpu_snapshot.subspan(sample_offset, sample_size), receipt.cpu_sample.begin());
    if (composed_snapshot.empty()) {
        std::ranges::copy(receipt.cpu_sample, receipt.composed_sample.begin());
    } else {
        std::ranges::copy(composed_snapshot.subspan(sample_offset, sample_size),
                          receipt.composed_sample.begin());
    }

    for (u32 offset = 0; offset < receipt.sample_size; ++offset) {
        if (receipt.cpu_sample[offset] == receipt.composed_sample[offset]) {
            continue;
        }
        if (!receipt.first_differing_focus_offset) {
            receipt.first_differing_focus_offset = offset;
        }
        receipt.last_differing_focus_offset = offset;
        ++receipt.sample_differing_byte_count;
    }
    return receipt;
}

} // namespace VideoCore
