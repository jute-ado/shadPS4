// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

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

template <typename Owner>
struct DownloadRangeOwner {
    Owner owner;
    VAddr base_address;
    u64 size;
};

template <typename Owner>
struct PlannedDownloadRange {
    Owner owner;
    VAddr address;
    u64 size;
    u64 source_offset;
    u64 staging_offset;
};

// Resolve each dirty interval against the allocation that produced it before recording a
// readback. A guest address may have been rebound since the original GPU write; using the caller's
// containing buffer in that case copies unrelated bytes. The resolver returns the owning
// allocation and this policy rejects any interval that falls outside it or would exceed the
// staging capacity. Emit is called only for validated intervals.
template <typename Owner, typename Enumerate, typename ResolveOwner, typename Emit>
[[nodiscard]] bool PlanOwnedDownloadRanges(Enumerate&& enumerate, ResolveOwner&& resolve_owner,
                                           Emit&& emit, u64& total_size, u64 max_staging_size) {
    constexpr u64 alignment = 64ULL;
    bool valid = true;
    std::invoke(
        std::forward<Enumerate>(enumerate), [&](VAddr start, VAddr end, const auto& owner_key) {
            if (!valid || start >= end) {
                valid = false;
                return;
            }
            const auto owner = std::invoke(std::forward<ResolveOwner>(resolve_owner), owner_key);
            if (!owner || owner->size > std::numeric_limits<VAddr>::max() - owner->base_address) {
                valid = false;
                return;
            }
            const VAddr owner_end = owner->base_address + owner->size;
            if (start < owner->base_address || end > owner_end) {
                valid = false;
                return;
            }
            const u64 range_size = end - start;
            if (range_size > std::numeric_limits<u64>::max() - (alignment - 1)) {
                valid = false;
                return;
            }
            const u64 aligned_size = (range_size + alignment - 1) & ~(alignment - 1);
            if (total_size > max_staging_size || aligned_size > max_staging_size - total_size) {
                valid = false;
                return;
            }
            std::invoke(std::forward<Emit>(emit),
                        PlannedDownloadRange<Owner>{owner->owner, start, range_size,
                                                    start - owner->base_address, total_size});
            total_size += aligned_size;
        });
    return valid;
}

// Validate an interval against the page table at every caching-page boundary. Checking only the
// first byte is insufficient after a guest rebind: a stale dirty interval can span a page whose
// current allocation is different from the interval's recorded owner.
template <typename Owner, typename PageOwner>
[[nodiscard]] bool AllPagesOwnedBy(const Owner& expected, VAddr address, u64 size, u64 page_size,
                                   PageOwner&& page_owner) {
    if (size == 0 || page_size == 0 || size > std::numeric_limits<VAddr>::max() - address) {
        return false;
    }
    const VAddr end = address + size;
    VAddr cursor = address;
    while (cursor < end) {
        if (std::invoke(std::forward<PageOwner>(page_owner), cursor) != expected) {
            return false;
        }
        const u64 page_offset = cursor % page_size;
        const u64 bytes_to_boundary = page_size - page_offset;
        const u64 step = std::min<u64>(bytes_to_boundary, end - cursor);
        if (step == 0) {
            return false;
        }
        cursor += step;
    }
    return true;
}

struct MemorySnapshotOverlay {
    u64 output_offset;
    u64 download_offset;
    u64 size;
};

// Diagnostic overlays that intersect the caller's focus are always emitted.  A large snapshot
// can contain hundreds of unrelated dirty fragments; spending the process-wide bounded budget on
// those fragments first would otherwise hide the one range needed to explain a failing read.
// Non-focused entries remain bounded and preserve their existing enumeration order.
template <typename Diagnostics, typename Emit, typename TakeSlot>
void ForEachSnapshotOverlayDiagnostic(Diagnostics&& diagnostics, Emit&& emit,
                                      TakeSlot&& take_slot) {
    for (const auto& diagnostic : diagnostics) {
        if (diagnostic.focus_intersects) {
            std::invoke(emit, diagnostic);
        }
    }
    for (const auto& diagnostic : diagnostics) {
        if (diagnostic.focus_intersects) {
            continue;
        }
        if (!std::invoke(take_slot)) {
            break;
        }
        std::invoke(emit, diagnostic);
    }
}

// Return whether an overlay's output range intersects the focus range used by a diagnostic
// snapshot. Treat wrapped ranges as invalid rather than allowing a diagnostic-only probe to
// accidentally classify an unrelated overlay as the failing source.
[[nodiscard]] constexpr bool SnapshotOverlayIntersectsFocus(u64 overlay_offset, u64 overlay_size,
                                                            u64 focus_offset, u64 focus_size) noexcept {
    if (overlay_size == 0 || focus_size == 0 ||
        overlay_offset > std::numeric_limits<u64>::max() - overlay_size ||
        focus_offset > std::numeric_limits<u64>::max() - focus_size) {
        return false;
    }
    const u64 overlay_end = overlay_offset + overlay_size;
    const u64 focus_end = focus_offset + focus_size;
    return overlay_offset < focus_end && focus_offset < overlay_end;
}

template <typename Owner>
struct OwnedDownloadRange {
    Owner owner;
    VAddr address;
    u64 size;
};

// Before copying an asynchronous readback into guest backing, reject the whole publication if a
// currently dirty interval is owned by a different GPU allocation/version. This conservative
// check prevents stale bytes from an older source from overwriting a same-address replacement;
// ranges that are no longer GPU-dirty remain eligible because the captured readback is still the
// last known GPU write for that mapping.
template <typename DirtyRanges, typename CapturedRange>
[[nodiscard]] bool AreGpuReadbackOwnersCompatible(const DirtyRanges& dirty_ranges,
                                                  std::span<const CapturedRange> captured_ranges) {
    for (const auto& captured : captured_ranges) {
        if (captured.size == 0 ||
            captured.size > std::numeric_limits<VAddr>::max() - captured.address) {
            return false;
        }
        bool mismatch = false;
        dirty_ranges.ForEachInRange(captured.address, captured.size,
                                    [&](VAddr, VAddr, const auto& current_owner) {
                                        mismatch |= current_owner != captured.owner;
                                    });
        if (mismatch) {
            return false;
        }
    }
    return true;
}

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

// Commit GPU ownership only for ranges that a deferred readback actually published. A pending
// readback can be split when a mapping changes while it is in flight; clearing an entire tracker
// page for one surviving fragment would otherwise hide an unrelated dirty fragment in that page.
// The caller supplies the page-level unmark operation because only the buffer cache owns the
// memory tracker. DirtyRanges must provide Subtract() and Intersects().
template <typename DirtyRanges, typename PublishedRange, typename UnmarkPage>
[[nodiscard]] bool ReconcilePublishedGpuRanges(DirtyRanges& dirty_ranges,
                                               std::span<const PublishedRange> published_ranges,
                                               u64 page_size, UnmarkPage&& unmark_page) {
    if (page_size == 0) {
        return false;
    }
    bool valid = true;
    for (const auto& range : published_ranges) {
        if (range.size == 0) {
            continue;
        }
        if (range.size > std::numeric_limits<VAddr>::max() - range.address) {
            valid = false;
            continue;
        }
        dirty_ranges.Subtract(range.address, range.size);
        const VAddr end = range.address + range.size;
        VAddr page = range.address - (range.address % page_size);
        while (true) {
            // A full page query is intentional: a dirty neighbour outside the published
            // fragment keeps the page GPU-owned until that neighbour is read back too.
            if (!dirty_ranges.Intersects(page, page_size)) {
                std::invoke(unmark_page, page);
            }
            if (end - page <= page_size) {
                break;
            }
            page += page_size;
        }
    }
    return valid;
}

// A deferred readback also captures the GPU allocation/version that produced each source range.
// At completion, intersect it with the current dirty map and reconcile only intervals whose
// owner still matches. A same-address replacement must retain GPU ownership until it receives a
// readback from its own allocation; address-only cleanup would clear the replacement with stale
// bytes from the predecessor.
template <typename DirtyRanges, typename PublishedRange, typename UnmarkPage>
[[nodiscard]] bool ReconcilePublishedGpuOwnerRanges(
    DirtyRanges& dirty_ranges, std::span<const PublishedRange> published_ranges, u64 page_size,
    UnmarkPage&& unmark_page) {
    if (page_size == 0) {
        return false;
    }
    bool valid = true;
    std::vector<DownloadRange> still_owned;
    for (const auto& range : published_ranges) {
        if (range.size == 0) {
            continue;
        }
        if (range.size > std::numeric_limits<VAddr>::max() - range.address) {
            valid = false;
            continue;
        }
        dirty_ranges.ForEachInRange(
            range.address, range.size,
            [&](VAddr current_begin, VAddr current_end, const auto& current_owner) {
                if (current_owner != range.owner) {
                    return;
                }
                const VAddr begin = std::max(current_begin, range.address);
                const VAddr end = std::min(current_end, range.address + range.size);
                if (begin < end) {
                    still_owned.push_back({begin, end - begin});
                }
            });
    }
    const bool reconciled =
        ReconcilePublishedGpuRanges(dirty_ranges, std::span<const DownloadRange>{still_owned},
                                    page_size, std::forward<UnmarkPage>(unmark_page));
    return valid && reconciled;
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
