// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/buffer_cache/srt_snapshot_composition_receipt.h"
#include "video_core/buffer_cache/srt_session_diagnostics.h"

namespace VideoCore {

// Logical GPU mapping admission retains the guest's 16KiB granularity. Private captures and
// write watchers use the existing 4KiB host tracker, so an unused sibling host page cannot
// repeatedly invalidate an otherwise unchanged shader-resource snapshot.
inline constexpr u64 SrtGpuMappingPageSize = 16_KB;
inline constexpr u64 SrtSnapshotPageSize = TRACKER_BYTES_PER_PAGE;
// Keep one private readback bounded well below BufferCache's 32 MiB download ring. A bounded
// group amortizes scheduler.Finish() across adjacent host pages without allowing one malformed
// or unusually large SRT request to monopolize staging memory.
inline constexpr u64 SrtSnapshotBatchPageCount = 16;
inline constexpr u64 SrtSnapshotBatchBytes = SrtSnapshotPageSize * SrtSnapshotBatchPageCount;
inline constexpr VAddr MaxSrtGpuAddress = 1ULL << 40;

// Diagnostic classification only. This does not change the existing bool failure contract,
// retry policy, page protection, or which bytes may be published from a snapshot.
enum class SrtMemoryReadFailure {
    None,
    InvalidRequest,
    UnmappedRange,
    ReadbackUnavailable,
    PageCopyRejected,
    EpochInvalidated,
    NewPageAfterSeal,
    MissingCapturedPage,
};

struct SrtReadFailureRequest {
    bool known{};
    VAddr address{};
    u64 size{};
    bool output_null{};
    u32 invalid_reasons{};
    bool operator==(const SrtReadFailureRequest&) const = default;
};

// Bounded evidence for one SRT source-range read. This is diagnostic data only: source bytes are
// copied from the caller's already-materialized output, and the optional composition route is
// retained only when the request is inside the snapshot range that produced that receipt.
struct SrtSourceReadRecord {
    static constexpr size_t MaximumBytes = 16;

    bool request_known{};
    VAddr request_address{};
    u64 request_size{};
    bool output_null{};
    SrtMemoryReadFailure failure{SrtMemoryReadFailure::None};
    SrtSnapshotCompositionRoute route{SrtSnapshotCompositionRoute::Unavailable};
    VAddr snapshot_address{};
    u64 snapshot_size{};
    bool source_bytes_known{};
    u32 bytes_captured{};
    // Identifies the private snapshot attempt that supplied the optional composition route.
    // Zero means that the read was not associated with an armed snapshot receipt.
    u64 diagnostic_snapshot_id{};
    std::array<u8, MaximumBytes> first_bytes{};
};

[[nodiscard]] inline SrtSourceReadRecord CaptureSrtSourceReadRecord(
    VAddr request_address, u64 request_size, SrtMemoryReadFailure failure, bool output_null,
    std::span<const u8> source_bytes,
    const std::optional<SrtSnapshotCompositionReceipt>& composition) noexcept {
    SrtSourceReadRecord record{
        .request_known = true,
        .request_address = request_address,
        .request_size = request_size,
        .output_null = output_null,
        .failure = failure,
    };

    if (failure == SrtMemoryReadFailure::None && composition) {
        const auto& receipt = *composition;
        const bool address_in_snapshot =
            request_address >= receipt.snapshot_address &&
            request_address - receipt.snapshot_address <= receipt.snapshot_size;
        const u64 offset = address_in_snapshot ? request_address - receipt.snapshot_address : 0;
        const bool range_in_snapshot =
            address_in_snapshot && request_size <= receipt.snapshot_size - offset;
        if (range_in_snapshot && receipt.route != SrtSnapshotCompositionRoute::Unavailable) {
            record.route = receipt.route;
            record.snapshot_address = receipt.snapshot_address;
            record.snapshot_size = receipt.snapshot_size;
            record.diagnostic_snapshot_id = receipt.diagnostic_snapshot_id;
        }
    }

    if (failure != SrtMemoryReadFailure::None || output_null) {
        return record;
    }

    const size_t copy_size =
        std::min<size_t>(source_bytes.size(), std::min<u64>(request_size, record.MaximumBytes));
    std::ranges::copy(source_bytes.first(copy_size), record.first_bytes.begin());
    record.bytes_captured = static_cast<u32>(copy_size);
    record.source_bytes_known =
        copy_size == std::min<u64>(request_size, record.MaximumBytes);
    return record;
}

[[nodiscard]] constexpr u32 SrtInvalidReadReasons(VAddr address, u64 size,
                                                  bool output_null) noexcept {
    // Independent clauses may overlap. Never subtract an out-of-range address.
    return (output_null ? 1U : 0U) | (address == 0 ? 2U : 0U) | (size == 0 ? 4U : 0U) |
           (size > std::numeric_limits<u32>::max() ? 8U : 0U) |
           (address >= MaxSrtGpuAddress ? 16U : 0U) |
           (address < MaxSrtGpuAddress && size > MaxSrtGpuAddress - address ? 32U : 0U);
}

[[nodiscard]] constexpr std::string_view SrtMemoryReadFailureName(SrtMemoryReadFailure failure) {
    switch (failure) {
    case SrtMemoryReadFailure::None:
        return "none";
    case SrtMemoryReadFailure::InvalidRequest:
        return "invalid_request";
    case SrtMemoryReadFailure::UnmappedRange:
        return "unmapped_range";
    case SrtMemoryReadFailure::ReadbackUnavailable:
        return "readback_unavailable";
    case SrtMemoryReadFailure::PageCopyRejected:
        return "page_copy_rejected";
    case SrtMemoryReadFailure::EpochInvalidated:
        return "epoch_invalidated";
    case SrtMemoryReadFailure::NewPageAfterSeal:
        return "new_page_after_seal";
    case SrtMemoryReadFailure::MissingCapturedPage:
        return "missing_captured_page";
    }
    return "unknown";
}

// Transitional compatibility for callers that have not yet moved to private snapshot reads.
// Remove this once the pipeline and generated walker switch atomically to SrtMemoryReader.
template <typename PageManager, typename BufferCache>
bool SynchronizeSrtMemoryRange(PageManager& page_manager, BufferCache& buffer_cache, VAddr address,
                               u64 size) {
    if (address == 0 || size == 0 || size > std::numeric_limits<u32>::max() ||
        address >= MaxSrtGpuAddress || size > MaxSrtGpuAddress - address) {
        return false;
    }
    const VAddr mapped_begin = Common::AlignDown(address, SrtGpuMappingPageSize);
    const VAddr mapped_end = Common::AlignUp(address + size, SrtGpuMappingPageSize);
    if (!page_manager.IsGpuMapped(mapped_begin, mapped_end - mapped_begin)) {
        return false;
    }
    if (!buffer_cache.IsRegionGpuModifiedExact(address, size)) {
        return true;
    }
    return buffer_cache.ReadMemoryExact(mapped_begin, mapped_end - mapped_begin);
}

template <typename PageManager, typename BufferCache>
bool ReadSrtMemoryRange(PageManager& page_manager, BufferCache& buffer_cache, VAddr address,
                        void* output, u64 size, SrtMemoryReadFailure* failure = nullptr,
                        SrtSnapshotCompositionReceiptRequest* receipt_request = nullptr) {
    const auto fail = [&](SrtMemoryReadFailure reason) {
        if (failure && *failure == SrtMemoryReadFailure::None) {
            *failure = reason;
        }
        return false;
    };
    if (output == nullptr || address == 0 || size == 0 || size > std::numeric_limits<u32>::max() ||
        address >= MaxSrtGpuAddress || size > MaxSrtGpuAddress - address) {
        return fail(SrtMemoryReadFailure::InvalidRequest);
    }
    const VAddr mapped_begin = Common::AlignDown(address, SrtGpuMappingPageSize);
    const VAddr mapped_end = Common::AlignUp(address + size, SrtGpuMappingPageSize);
    if (!page_manager.IsGpuMapped(mapped_begin, mapped_end - mapped_begin)) {
        return fail(SrtMemoryReadFailure::UnmappedRange);
    }
    bool readback_completed{};
    if (receipt_request != nullptr) {
        if constexpr (requires {
                          buffer_cache.ReadMemorySnapshot(address, output, size, receipt_request);
                      }) {
            readback_completed =
                buffer_cache.ReadMemorySnapshot(address, output, size, receipt_request);
        } else {
            readback_completed = buffer_cache.ReadMemorySnapshot(address, output, size);
        }
    } else {
        readback_completed = buffer_cache.ReadMemorySnapshot(address, output, size);
    }
    if (!readback_completed) {
        return fail(SrtMemoryReadFailure::ReadbackUnavailable);
    }
    return true;
}

/// Captures every host tracking page used by one shader-selection attempt into private storage.
/// Every copy still requires complete 16KiB logical GPU mapping coverage. The active
/// PageManager epoch prevents overlapping CPU writes or unmaps from being accepted between page
/// copies. Once sealed, already captured pages remain available for deterministic replay, while
/// attempts to discover new guest pages fail closed.
template <typename PageManager, typename BufferCache>
class SrtMemoryReadSession {
public:
    static constexpr u32 MaxSourceReadRecords = 16;

    SrtMemoryReadSession(PageManager& page_manager_, BufferCache& buffer_cache_,
                         bool capture_failure_details_ = false,
                         bool capture_diagnostic_counts_ = false)
        : page_manager{page_manager_}, buffer_cache{buffer_cache_},
          snapshot_epoch{page_manager.BeginSnapshotEpoch()},
          capture_failure_details{capture_failure_details_},
          diagnostic_counts{capture_diagnostic_counts_} {}

    SrtMemoryReadSession(const SrtMemoryReadSession&) = delete;
    SrtMemoryReadSession& operator=(const SrtMemoryReadSession&) = delete;

    void ArmCompositionReceipt() noexcept {
        composition_receipt.reset();
        composition_receipt_armed = true;
    }

    [[nodiscard]] const std::optional<SrtSnapshotCompositionReceipt>& CompositionReceipt()
        const noexcept {
        return composition_receipt;
    }

    [[nodiscard]] bool Read(VAddr address, void* output, u64 size) {
        diagnostic_counts.Read(size);
        if (failed) {
            return false;
        }
        // A composition receipt describes the snapshot decision for this Read invocation only.
        // Do not reuse the session's first receipt for later cached reads: those reads may have
        // a different authority route (or no newly composed snapshot at all).
        std::optional<SrtSnapshotCompositionReceipt> read_composition_receipt;
        const auto fail = [&](SrtMemoryReadFailure reason) {
            if (capture_failure_details && !failure_request_latched) {
                first_failure_request = {true, address, size, output == nullptr,
                                         SrtInvalidReadReasons(address, size, output == nullptr)};
                failure_request_latched = true;
            }
            CaptureSourceRead(address, output, size, reason, read_composition_receipt);
            return Fail(reason);
        };
        if (output == nullptr || address == 0 || size == 0 ||
            size > std::numeric_limits<u32>::max() || address >= MaxSrtGpuAddress ||
            size > MaxSrtGpuAddress - address) {
            return fail(SrtMemoryReadFailure::InvalidRequest);
        }

        const VAddr page_begin = Common::AlignDown(address, SrtSnapshotPageSize);
        const VAddr page_end = Common::AlignUp(address + size, SrtSnapshotPageSize);
        std::optional<SrtSnapshotCompositionReceiptRequest> composition_request;
        if (composition_receipt_armed) {
            composition_receipt_armed = false;
            const u64 focus_size =
                std::min<u64>(size, SrtSnapshotPageSize - (address - page_begin));
            SrtSnapshotCompositionReceiptRequest request{
                .focus_address = address,
                .focus_size = focus_size,
            };
            request.result = CaptureSrtSnapshotCompositionReceipt(
                SrtSnapshotCompositionReceiptConfig{
                    .snapshot_address = page_begin,
                    .snapshot_size = SrtSnapshotPageSize,
                    .focus_address = address,
                    .focus_size = focus_size,
                    .route = SrtSnapshotCompositionRoute::Unavailable,
                },
                {}, {});
            if (pages.contains(page_begin)) {
                composition_receipt = std::move(request.result);
                read_composition_receipt = composition_receipt;
            } else {
                composition_request = std::move(request);
            }
        }
        for (VAddr page_address = page_begin; page_address < page_end;) {
            if (pages.contains(page_address)) {
                diagnostic_counts.CacheHit();
                page_address += SrtSnapshotPageSize;
                continue;
            }
            if (sealed) {
                return fail(SrtMemoryReadFailure::NewPageAfterSeal);
            }

            // Keep an armed composition receipt scoped to the first host page. The receipt's
            // focus and source-byte evidence intentionally describe that page, even when the
            // following uncaptured pages are coalesced into a single readback.
            const bool has_composition_request =
                composition_request && page_address == page_begin;
            const VAddr batch_begin = page_address;
            VAddr batch_end = page_address + SrtSnapshotPageSize;
            if (!has_composition_request) {
                while (batch_end < page_end && !pages.contains(batch_end) &&
                       batch_end - batch_begin < SrtSnapshotBatchBytes) {
                    batch_end += SrtSnapshotPageSize;
                }
            }

            const u64 batch_size = batch_end - batch_begin;
            std::vector<u8> batch_bytes(static_cast<size_t>(batch_size));
            std::array<VAddr, SrtSnapshotBatchPageCount> begun_pages{};
            u64 begun_page_count{};
            const auto release_begun_pages = [&] {
                for (u64 index = 0; index < begun_page_count; ++index) {
                    // The result is intentionally ignored: this is cleanup after a failed
                    // batch admission/copy, and every begun page must leave the active-copy
                    // state before the epoch can be sealed.
                    static_cast<void>(snapshot_epoch.EndPageCopy(
                        begun_pages[index], SrtSnapshotPageSize, false));
                }
            };
            for (VAddr begun_address = batch_begin; begun_address < batch_end;
                 begun_address += SrtSnapshotPageSize) {
                if (!snapshot_epoch.BeginPageCopy(begun_address, SrtSnapshotPageSize)) {
                    release_begun_pages();
                    return fail(snapshot_epoch.Valid() ? SrtMemoryReadFailure::PageCopyRejected
                                                       : SrtMemoryReadFailure::EpochInvalidated);
                }
                begun_pages[begun_page_count++] = begun_address;
            }

            auto* receipt_request =
                has_composition_request ? &*composition_request : nullptr;
            const bool copied =
                ReadSrtMemoryRange(page_manager, buffer_cache, batch_begin, batch_bytes.data(),
                                   batch_size, &failure, receipt_request);
            bool accepted = true;
            for (u64 index = 0; index < begun_page_count; ++index) {
                // Do not short-circuit: all pages acquired above must leave the copying state,
                // even if the first EndPageCopy observes an invalidated epoch.
                accepted = snapshot_epoch.EndPageCopy(begun_pages[index], SrtSnapshotPageSize,
                                                      copied) &&
                           accepted;
            }
            if (!accepted) {
                // Preserve the lower-level read classification in the per-request diagnostic.
                // EndPageCopy still supplies the session-level fallback when the read itself did
                // not produce a reason (for example, an epoch rejection after a successful copy).
                const auto copy_failure =
                    snapshot_epoch.Valid() ? SrtMemoryReadFailure::PageCopyRejected
                                           : SrtMemoryReadFailure::EpochInvalidated;
                return fail(failure != SrtMemoryReadFailure::None ? failure : copy_failure);
            }
            if (receipt_request != nullptr) {
                composition_receipt = std::move(receipt_request->result);
                read_composition_receipt = composition_receipt;
            }
            for (u64 index = 0; index < begun_page_count; ++index) {
                CachedPage page{};
                const auto page_offset = static_cast<size_t>(index * SrtSnapshotPageSize);
                std::memcpy(page.bytes.data(), batch_bytes.data() + page_offset,
                            SrtSnapshotPageSize);
                pages.emplace(begun_pages[index], std::move(page));
                diagnostic_counts.NewPage();
            }
            page_address = batch_end;
        }

        if (!Valid()) {
            return fail(SrtMemoryReadFailure::EpochInvalidated);
        }

        auto* destination = static_cast<u8*>(output);
        VAddr cursor = address;
        u64 remaining = size;
        while (remaining != 0) {
            const VAddr page_address = Common::AlignDown(cursor, SrtSnapshotPageSize);
            const auto page = pages.find(page_address);
            if (page == pages.end()) {
                return fail(SrtMemoryReadFailure::MissingCapturedPage);
            }
            const u64 page_offset = cursor - page_address;
            const u64 copy_size = std::min<u64>(remaining, SrtSnapshotPageSize - page_offset);
            std::memcpy(destination, page->second.bytes.data() + page_offset, copy_size);
            destination += copy_size;
            cursor += copy_size;
            remaining -= copy_size;
        }
        CaptureSourceRead(address, output, size, SrtMemoryReadFailure::None,
                          read_composition_receipt);
        return true;
    }

    static bool PS4_SYSV_ABI ReadCallback(void* context, VAddr address, void* output, u64 size) {
        return static_cast<SrtMemoryReadSession*>(context)->Read(address, output, size);
    }

    [[nodiscard]] bool Valid() const {
        return !failed && (sealed ? seal_result : snapshot_epoch.Valid());
    }

    [[nodiscard]] bool Seal() {
        if (!sealed) {
            seal_result = snapshot_epoch.Seal();
            sealed = true;
        }
        const bool result = !failed && seal_result;
        if (!result && capture_failure_details && !failure_request_latched) {
            // Seal can fail without any failed Read. Retain that unknown origin even if a
            // later Read is invalid, without changing the legacy failure flag or reason.
            failure_request_latched = true;
        }
        return result;
    }

    [[nodiscard]] const bool* FailureFlag() const noexcept {
        return &failed;
    }

    [[nodiscard]] SrtReadFailureRequest FirstFailureRequest() const noexcept {
        return first_failure_request;
    }

    [[nodiscard]] SrtMemoryReadFailure LastFailure() const {
        if (failure != SrtMemoryReadFailure::None) {
            return failure;
        }
        // A CPU write/unmap can invalidate between the last read and Seal without touching
        // the reader's existing failure flag. Report that state without changing the flag.
        return Valid() ? SrtMemoryReadFailure::None : SrtMemoryReadFailure::EpochInvalidated;
    }

    [[nodiscard]] const SrtSessionReadCounts& DiagnosticCounts() const noexcept {
        return diagnostic_counts.Counts();
    }

    [[nodiscard]] std::span<const SrtSourceReadRecord> SourceReads() const noexcept {
        return std::span<const SrtSourceReadRecord>{source_reads}.first(source_read_count);
    }

private:
    void CaptureSourceRead(VAddr address, void* output, u64 size,
                           SrtMemoryReadFailure read_failure,
                           const std::optional<SrtSnapshotCompositionReceipt>& composition) noexcept {
        if (!capture_failure_details ||
            (source_read_count >= MaxSourceReadRecords &&
             read_failure == SrtMemoryReadFailure::None) ||
            (source_failure_recorded && read_failure != SrtMemoryReadFailure::None)) {
            return;
        }
        const size_t sample_size =
            read_failure == SrtMemoryReadFailure::None
                ? static_cast<size_t>(std::min<u64>(size, SrtSourceReadRecord::MaximumBytes))
                : 0;
        const std::span<const u8> source_bytes{
            output == nullptr ? nullptr : static_cast<const u8*>(output), sample_size};
        const auto record = CaptureSrtSourceReadRecord(
            address, size, read_failure, output == nullptr, source_bytes, composition);
        if (source_read_count < MaxSourceReadRecords) {
            source_reads[source_read_count++] = record;
        } else {
            source_reads[MaxSourceReadRecords - 1] = record;
        }
        if (read_failure != SrtMemoryReadFailure::None) {
            source_failure_recorded = true;
        }
    }

    bool Fail(SrtMemoryReadFailure reason) {
        if (failure == SrtMemoryReadFailure::None) {
            failure = reason;
        }
        failed = true;
        return false;
    }

    struct CachedPage {
        std::array<u8, SrtSnapshotPageSize> bytes{};
    };

    PageManager& page_manager;
    BufferCache& buffer_cache;
    decltype(std::declval<PageManager&>().BeginSnapshotEpoch()) snapshot_epoch;
    std::unordered_map<VAddr, CachedPage> pages;
    bool failed{};
    bool sealed{};
    bool seal_result{};
    SrtMemoryReadFailure failure{};
    const bool capture_failure_details{};
    SrtSessionReadCounter diagnostic_counts;
    bool failure_request_latched{};
    SrtReadFailureRequest first_failure_request{};
    std::array<SrtSourceReadRecord, MaxSourceReadRecords> source_reads{};
    u32 source_read_count{};
    bool source_failure_recorded{};
    bool composition_receipt_armed{};
    std::optional<SrtSnapshotCompositionReceipt> composition_receipt{};
};

} // namespace VideoCore
