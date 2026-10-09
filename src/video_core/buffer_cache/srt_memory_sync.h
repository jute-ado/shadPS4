// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/types.h"
#include "video_core/shader_capture_provenance.h"

namespace VideoCore {

template <typename PageManager, typename BufferCache>
bool ReadSrtMemoryRange(PageManager& page_manager, BufferCache& buffer_cache, VAddr address,
                        void* output, u64 size) {
    constexpr VAddr MaxGpuAddress = 1ULL << 40;
    if (address == 0 || output == nullptr || size == 0 ||
        size > std::numeric_limits<u32>::max() || address >= MaxGpuAddress ||
        size > MaxGpuAddress - address) {
        return false;
    }
    constexpr u64 BufferCachePageSize = 16_KB;
    const VAddr mapped_begin = Common::AlignDown(address, BufferCachePageSize);
    const VAddr mapped_end = Common::AlignUp(address + size, BufferCachePageSize);
    if (!page_manager.IsGpuMapped(mapped_begin, mapped_end - mapped_begin)) {
        return false;
    }

    struct PageToken {
        VAddr address;
        u64 token;
    };
    std::vector<PageToken> page_tokens;
    for (VAddr page = mapped_begin; page < mapped_end; page += BufferCachePageSize) {
        const u64 token = page_manager.GetGpuMappingToken(page, BufferCachePageSize);
        if (token == 0) {
            return false;
        }
        page_tokens.push_back({page, token});
    }

    if (!buffer_cache.ReadMemorySnapshot(address, output, size)) {
        return false;
    }
    return std::ranges::all_of(page_tokens, [&](const PageToken& page) {
        return page_manager.GetGpuMappingToken(page.address, BufferCachePageSize) == page.token;
    });
}

template <typename PageManager, typename BufferCache>
class SrtMemoryReadSession {
public:
    SrtMemoryReadSession(PageManager& page_manager_, BufferCache& buffer_cache_,
                         bool collect_provenance_ = false)
        : page_manager{page_manager_}, buffer_cache{buffer_cache_},
          snapshot_epoch{page_manager.BeginSnapshotEpoch()},
          provenance{collect_provenance_ ? std::make_unique<SrtCaptureProvenance>() : nullptr} {}

    [[nodiscard]] bool Read(VAddr address, void* output, u64 size) {
        if (!Valid()) {
            return false;
        }
        return ReadSrtMemoryRange(page_manager, *this, address, output, size) && Valid();
    }

    [[nodiscard]] bool Valid() {
        if (!snapshot_epoch.Valid()) {
            return false;
        }
        if (!CachedPagesAreCurrent()) {
            snapshot_epoch.Invalidate();
            return false;
        }
        return true;
    }

    [[nodiscard]] bool Seal() {
        if (!CachedPagesAreCurrent()) {
            snapshot_epoch.Invalidate();
        }
        return snapshot_epoch.Seal();
    }

    [[nodiscard]] const SrtCaptureProvenance& Provenance() const noexcept {
        static const SrtCaptureProvenance EmptyProvenance;
        return provenance ? *provenance : EmptyProvenance;
    }

    static bool PS4_SYSV_ABI Callback(void* context, VAddr address, void* output, u64 size) {
        return static_cast<SrtMemoryReadSession*>(context)->Read(address, output, size);
    }

    [[nodiscard]] bool ReadMemorySnapshot(VAddr address, void* output, u64 size) {
        constexpr u64 TrackingPageSize = 16_KB;
        if (!Valid()) {
            return false;
        }

        auto* destination = static_cast<u8*>(output);
        VAddr current_address = address;
        u64 remaining = size;
        while (remaining != 0) {
            const VAddr page_address = Common::AlignDown(current_address, TrackingPageSize);
            const u64 page_offset = current_address - page_address;
            const u64 copy_size = std::min(remaining, TrackingPageSize - page_offset);

            auto page = pages.find(page_address);
            if (page == pages.end()) {
                const u64 mapping_token =
                    page_manager.GetGpuMappingToken(page_address, TrackingPageSize);
                if (mapping_token == 0) {
                    return false;
                }
                std::vector<u8> bytes(TrackingPageSize);
                if (!snapshot_epoch.BeginPageCopy(page_address, TrackingPageSize)) {
                    return false;
                }
                SnapshotReadProvenance read_provenance{};
                const bool copied = provenance
                                        ? buffer_cache.ReadMemorySnapshot(
                                              page_address, bytes.data(), bytes.size(),
                                              &read_provenance)
                                        : buffer_cache.ReadMemorySnapshot(
                                              page_address, bytes.data(), bytes.size());
                const bool copy_valid =
                    snapshot_epoch.EndPageCopy(page_address, TrackingPageSize, copied);
                const bool mapping_current =
                    page_manager.GetGpuMappingToken(page_address, TrackingPageSize) == mapping_token;
                if (!mapping_current) {
                    snapshot_epoch.Invalidate();
                }
                if (provenance) {
                    const bool accepted = copy_valid && mapping_current;
                    const auto potential_writer =
                        accepted ? buffer_cache.LatestPotentialShaderWrite(
                                       page_address, TrackingPageSize, mapping_token)
                                 : std::nullopt;
                    if (!provenance->CanRecord()) {
                        provenance->MarkTruncated();
                    } else {
                        provenance->Record({
                            .address = page_address,
                            .size = TrackingPageSize,
                            .exact_gpu_modified = read_provenance.exact_gpu_modified,
                            .coarse_gpu_owned = read_provenance.coarse_gpu_owned,
                            .source = accepted ? read_provenance.source
                                               : ShaderCaptureSource::Refused,
                            .mapping_token = mapping_token,
                            .content_digest = accepted ? DigestShaderCaptureContent(bytes) : 0,
                            .potential_writer = potential_writer,
                        });
                    }
                }
                if (!copy_valid || !mapping_current) {
                    return false;
                }
                page = pages.emplace(page_address, CachedPage{std::move(bytes), mapping_token}).first;
            }
            std::memcpy(destination, page->second.bytes.data() + page_offset, copy_size);
            destination += copy_size;
            current_address += copy_size;
            remaining -= copy_size;
        }
        return Valid();
    }

private:
    [[nodiscard]] bool CachedPagesAreCurrent() const {
        constexpr u64 TrackingPageSize = 16_KB;
        return std::ranges::all_of(pages, [&](const auto& entry) {
            return page_manager.GetGpuMappingToken(entry.first, TrackingPageSize) ==
                   entry.second.mapping_token;
        });
    }

    struct CachedPage {
        std::vector<u8> bytes;
        u64 mapping_token;
    };

    PageManager& page_manager;
    BufferCache& buffer_cache;
    decltype(std::declval<PageManager&>().BeginSnapshotEpoch()) snapshot_epoch;
    std::unordered_map<VAddr, CachedPage> pages;
    std::unique_ptr<SrtCaptureProvenance> provenance;
};

} // namespace VideoCore
