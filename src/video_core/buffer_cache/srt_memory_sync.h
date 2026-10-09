// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "common/alignment.h"
#include "common/types.h"

namespace VideoCore {

inline constexpr u64 SrtTrackingPageSize = 16_KB;
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
    const VAddr mapped_begin = Common::AlignDown(address, SrtTrackingPageSize);
    const VAddr mapped_end = Common::AlignUp(address + size, SrtTrackingPageSize);
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
                        void* output, u64 size, SrtMemoryReadFailure* failure = nullptr) {
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
    const VAddr mapped_begin = Common::AlignDown(address, SrtTrackingPageSize);
    const VAddr mapped_end = Common::AlignUp(address + size, SrtTrackingPageSize);
    if (!page_manager.IsGpuMapped(mapped_begin, mapped_end - mapped_begin)) {
        return fail(SrtMemoryReadFailure::UnmappedRange);
    }
    if (!buffer_cache.ReadMemorySnapshot(address, output, size)) {
        return fail(SrtMemoryReadFailure::ReadbackUnavailable);
    }
    return true;
}

/// Captures every SRT page used by one shader-selection attempt into private storage. The active
/// PageManager epoch prevents overlapping CPU writes or unmaps from being accepted between page
/// copies. Once sealed, already captured pages remain available for deterministic replay, while
/// attempts to discover new guest pages fail closed.
template <typename PageManager, typename BufferCache>
class SrtMemoryReadSession {
public:
    SrtMemoryReadSession(PageManager& page_manager_, BufferCache& buffer_cache_)
        : page_manager{page_manager_}, buffer_cache{buffer_cache_},
          snapshot_epoch{page_manager.BeginSnapshotEpoch()} {}

    SrtMemoryReadSession(const SrtMemoryReadSession&) = delete;
    SrtMemoryReadSession& operator=(const SrtMemoryReadSession&) = delete;

    [[nodiscard]] bool Read(VAddr address, void* output, u64 size) {
        if (failed) {
            return false;
        }
        if (output == nullptr || address == 0 || size == 0 ||
            size > std::numeric_limits<u32>::max() || address >= MaxSrtGpuAddress ||
            size > MaxSrtGpuAddress - address) {
            return Fail(SrtMemoryReadFailure::InvalidRequest);
        }

        const VAddr page_begin = Common::AlignDown(address, SrtTrackingPageSize);
        const VAddr page_end = Common::AlignUp(address + size, SrtTrackingPageSize);
        for (VAddr page_address = page_begin; page_address < page_end;
             page_address += SrtTrackingPageSize) {
            if (pages.contains(page_address)) {
                continue;
            }
            if (sealed) {
                return Fail(SrtMemoryReadFailure::NewPageAfterSeal);
            }
            if (!snapshot_epoch.BeginPageCopy(page_address, SrtTrackingPageSize)) {
                return Fail(snapshot_epoch.Valid() ? SrtMemoryReadFailure::PageCopyRejected
                                                   : SrtMemoryReadFailure::EpochInvalidated);
            }

            CachedPage page{};
            const bool copied = ReadSrtMemoryRange(page_manager, buffer_cache, page_address,
                                                   page.bytes.data(), page.bytes.size(), &failure);
            const bool accepted =
                snapshot_epoch.EndPageCopy(page_address, SrtTrackingPageSize, copied);
            if (!accepted) {
                return Fail(snapshot_epoch.Valid() ? SrtMemoryReadFailure::PageCopyRejected
                                                   : SrtMemoryReadFailure::EpochInvalidated);
            }
            pages.emplace(page_address, std::move(page));
        }

        if (!Valid()) {
            return Fail(SrtMemoryReadFailure::EpochInvalidated);
        }

        auto* destination = static_cast<u8*>(output);
        VAddr cursor = address;
        u64 remaining = size;
        while (remaining != 0) {
            const VAddr page_address = Common::AlignDown(cursor, SrtTrackingPageSize);
            const auto page = pages.find(page_address);
            if (page == pages.end()) {
                return Fail(SrtMemoryReadFailure::MissingCapturedPage);
            }
            const u64 page_offset = cursor - page_address;
            const u64 copy_size = std::min<u64>(remaining, SrtTrackingPageSize - page_offset);
            std::memcpy(destination, page->second.bytes.data() + page_offset, copy_size);
            destination += copy_size;
            cursor += copy_size;
            remaining -= copy_size;
        }
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
        return !failed && seal_result;
    }

    [[nodiscard]] const bool* FailureFlag() const noexcept {
        return &failed;
    }

    [[nodiscard]] SrtMemoryReadFailure LastFailure() const {
        if (failure != SrtMemoryReadFailure::None) {
            return failure;
        }
        // A CPU write/unmap can invalidate between the last read and Seal without touching
        // the reader's existing failure flag. Report that state without changing the flag.
        return Valid() ? SrtMemoryReadFailure::None : SrtMemoryReadFailure::EpochInvalidated;
    }

private:
    bool Fail(SrtMemoryReadFailure reason) {
        if (failure == SrtMemoryReadFailure::None) {
            failure = reason;
        }
        failed = true;
        return false;
    }

    struct CachedPage {
        std::array<u8, SrtTrackingPageSize> bytes{};
    };

    PageManager& page_manager;
    BufferCache& buffer_cache;
    decltype(std::declval<PageManager&>().BeginSnapshotEpoch()) snapshot_epoch;
    std::unordered_map<VAddr, CachedPage> pages;
    bool failed{};
    bool sealed{};
    bool seal_result{};
    SrtMemoryReadFailure failure{};
};

} // namespace VideoCore
