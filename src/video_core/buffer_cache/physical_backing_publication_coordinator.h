// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/physical_backing_provenance.h"
#include "video_core/buffer_cache/physical_backing_publication_state.h"

namespace VideoCore {

struct PhysicalBackingBdaDelta {
    VAddr guest_page{};
    PhysicalBackingDeviceAddress device_address{};

    auto operator<=>(const PhysicalBackingBdaDelta&) const = default;
};

struct PhysicalBackingCachePageToken {
    PhysicalBackingOverride publication{};

    auto operator<=>(const PhysicalBackingCachePageToken&) const = default;
};

struct PhysicalBackingCachePagePublication {
    PhysicalBackingCachePageToken token{};
    std::vector<PhysicalBackingBdaDelta> deltas;
};

struct PhysicalBackingDirtyCachePagePublication {
    PhysicalBackingWriteback writeback{};
    std::vector<PhysicalBackingBdaDelta> deltas;
};

/// GPU-thread-owned coordinator for physical mapping provenance and BDA publications.
/// It is deliberately unsynchronized; callers must marshal every operation to the GPU thread.
class PhysicalBackingPublicationCoordinator {
public:
    static constexpr u64 PageSize = PhysicalBackingPublicationState::PageSize;
    static constexpr u64 AddressSpaceSize = 1ULL << 40;

    PhysicalBackingPublicationCoordinator(PhysicalBackingDeviceAddress imported_base,
                                          u64 backing_size) noexcept
        : state{imported_base, backing_size} {}

    /// Atomically maps complete physical spans and returns the exact page-table deltas.
    /// No delta is returned unless every page is accepted by the publication policy.
    [[nodiscard]] std::optional<std::vector<PhysicalBackingBdaDelta>> MapSpans(
        std::span<const Core::PhysicalBackingSpan> spans) {
        size_t page_count = 0;
        std::unordered_set<VAddr> batch_guest_pages;
        for (const Core::PhysicalBackingSpan& span : spans) {
            if (!ValidateSpan(span)) {
                return std::nullopt;
            }
            const u64 span_pages = span.size / PageSize;
            if (span_pages > std::numeric_limits<size_t>::max() - page_count) {
                return std::nullopt;
            }
            page_count += static_cast<size_t>(span_pages);
            for (u64 offset = 0; offset < span.size; offset += PageSize) {
                const VAddr guest_page = span.guest_base + offset;
                if (mapping_tokens.contains(guest_page) ||
                    !batch_guest_pages.emplace(guest_page).second) {
                    return std::nullopt;
                }
            }
        }
        if (page_count == 0) {
            return std::nullopt;
        }

        std::vector<PhysicalBackingMapping> mapped;
        std::vector<PhysicalBackingBdaDelta> deltas;
        mapped.reserve(page_count);
        deltas.reserve(page_count);
        for (const Core::PhysicalBackingSpan& span : spans) {
            for (u64 offset = 0; offset < span.size; offset += PageSize) {
                const auto mapping_generation = AcquireMappingGeneration();
                if (!mapping_generation) {
                    RollbackMappings(mapped);
                    return std::nullopt;
                }
                auto mapping = state.MapGuestPage(span.guest_base + offset,
                                                  span.physical_offset + offset,
                                                  *mapping_generation,
                                                  span.allocation_generation);
                if (!mapping) {
                    RollbackMappings(mapped);
                    return std::nullopt;
                }
                const auto address = state.Resolve(mapping->guest_page);
                if (address.value == 0) {
                    static_cast<void>(state.UnmapGuestPage(*mapping));
                    RollbackMappings(mapped);
                    return std::nullopt;
                }
                mapping_tokens.emplace(mapping->guest_page, *mapping);
                physical_aliases[mapping->physical_offset].emplace(mapping->guest_page);
                mapped.push_back(*mapping);
                deltas.push_back({mapping->guest_page, address});
            }
        }
        return deltas;
    }

    /// Atomically removes a complete guest range and returns zero page-table deltas.
    [[nodiscard]] std::optional<std::vector<PhysicalBackingBdaDelta>> UnmapRange(
        VAddr guest_base, u64 size) {
        if (size == 0 || !IsPageAligned(guest_base) || !IsPageAligned(size) ||
            guest_base >= AddressSpaceSize || size > AddressSpaceSize - guest_base) {
            return std::nullopt;
        }

        const size_t page_count = static_cast<size_t>(size / PageSize);
        std::vector<PhysicalBackingMapping> mappings;
        mappings.reserve(page_count);
        for (u64 offset = 0; offset < size; offset += PageSize) {
            const auto mapping_it = mapping_tokens.find(guest_base + offset);
            if (mapping_it == mapping_tokens.end() ||
                !state.CanUnmapGuestPage(mapping_it->second)) {
                return std::nullopt;
            }
            mappings.push_back(mapping_it->second);
        }

        std::vector<PhysicalBackingBdaDelta> deltas;
        deltas.reserve(page_count);
        for (const PhysicalBackingMapping& mapping : mappings) {
            if (!state.UnmapGuestPage(mapping)) {
                return std::nullopt;
            }
            mapping_tokens.erase(mapping.guest_page);
            ErasePhysicalAlias(mapping.physical_offset, mapping.guest_page);
            deltas.push_back({mapping.guest_page, {}});
        }
        return deltas;
    }

    /// Publishes one buffer-cache page for every guest alias of its physical page.
    /// Texture overlap fails closed until texture ownership participates in this policy.
    [[nodiscard]] std::optional<PhysicalBackingCachePagePublication> ActivateCachePage(
        u64 physical_offset, PhysicalBackingDeviceAddress override_page_address,
        bool has_texture_overlap) {
        const auto aliases_it = physical_aliases.find(physical_offset);
        if (has_texture_overlap || aliases_it == physical_aliases.end() ||
            aliases_it->second.empty()) {
            return std::nullopt;
        }
        const auto owner_generation = AcquireOwnerGeneration();
        if (!owner_generation) {
            return std::nullopt;
        }
        const auto publication =
            state.ActivateOverride(physical_offset, override_page_address, *owner_generation);
        if (!publication) {
            return std::nullopt;
        }
        return PhysicalBackingCachePagePublication{
            .token = {*publication},
            .deltas = MakeAliasDeltas(physical_offset),
        };
    }

    [[nodiscard]] std::optional<std::vector<PhysicalBackingBdaDelta>> RetireCachePageClean(
        const PhysicalBackingCachePageToken& token) {
        if (!physical_aliases.contains(token.publication.physical_offset) ||
            !state.RetireClean(token.publication)) {
            return std::nullopt;
        }
        return MakeAliasDeltas(token.publication.physical_offset);
    }

    [[nodiscard]] std::optional<PhysicalBackingDirtyCachePagePublication>
    RetireCachePageGpuDirty(const PhysicalBackingCachePageToken& token) {
        if (!physical_aliases.contains(token.publication.physical_offset)) {
            return std::nullopt;
        }
        const auto writeback = state.RetireGpuDirty(token.publication);
        if (!writeback) {
            return std::nullopt;
        }
        return PhysicalBackingDirtyCachePagePublication{
            .writeback = *writeback,
            .deltas = MakeAliasDeltas(token.publication.physical_offset),
        };
    }

    template <typename Commit>
    [[nodiscard]] std::optional<std::vector<PhysicalBackingBdaDelta>>
    CommitCachePageWriteback(const PhysicalBackingWriteback& writeback, Commit&& commit) {
        if (!physical_aliases.contains(writeback.physical_offset) ||
            !state.CommitOrderedWriteback(writeback, std::forward<Commit>(commit))) {
            return std::nullopt;
        }
        return MakeAliasDeltas(writeback.physical_offset);
    }

private:
    [[nodiscard]] static constexpr bool IsPageAligned(u64 value) noexcept {
        return (value & (PageSize - 1)) == 0;
    }

    [[nodiscard]] static constexpr bool ValidateSpan(
        const Core::PhysicalBackingSpan& span) noexcept {
        return span.size != 0 && span.allocation_generation != 0 &&
               IsPageAligned(span.guest_base) && IsPageAligned(span.physical_offset) &&
               IsPageAligned(span.size) && span.guest_base < AddressSpaceSize &&
               span.size <= AddressSpaceSize - span.guest_base &&
               span.physical_offset <= std::numeric_limits<u64>::max() - (span.size - 1);
    }

    [[nodiscard]] std::optional<u64> AcquireMappingGeneration() noexcept {
        if (last_mapping_generation == std::numeric_limits<u64>::max()) {
            return std::nullopt;
        }
        return ++last_mapping_generation;
    }

    [[nodiscard]] std::optional<u64> AcquireOwnerGeneration() noexcept {
        if (last_owner_generation == std::numeric_limits<u64>::max()) {
            return std::nullopt;
        }
        return ++last_owner_generation;
    }

    [[nodiscard]] std::vector<PhysicalBackingBdaDelta> MakeAliasDeltas(
        u64 physical_offset) const {
        std::vector<PhysicalBackingBdaDelta> deltas;
        const auto aliases_it = physical_aliases.find(physical_offset);
        if (aliases_it == physical_aliases.end()) {
            return deltas;
        }
        deltas.reserve(aliases_it->second.size());
        for (const VAddr guest_page : aliases_it->second) {
            deltas.push_back({guest_page, state.Resolve(guest_page)});
        }
        return deltas;
    }

    void ErasePhysicalAlias(u64 physical_offset, VAddr guest_page) {
        const auto aliases_it = physical_aliases.find(physical_offset);
        if (aliases_it == physical_aliases.end()) {
            return;
        }
        aliases_it->second.erase(guest_page);
        if (aliases_it->second.empty()) {
            physical_aliases.erase(aliases_it);
        }
    }

    void RollbackMappings(std::span<const PhysicalBackingMapping> mappings) {
        for (auto mapping_it = mappings.rbegin(); mapping_it != mappings.rend(); ++mapping_it) {
            static_cast<void>(state.UnmapGuestPage(*mapping_it));
            mapping_tokens.erase(mapping_it->guest_page);
            ErasePhysicalAlias(mapping_it->physical_offset, mapping_it->guest_page);
        }
    }

    PhysicalBackingPublicationState state;
    u64 last_mapping_generation{};
    u64 last_owner_generation{};
    std::unordered_map<VAddr, PhysicalBackingMapping> mapping_tokens;
    std::unordered_map<u64, std::unordered_set<VAddr>> physical_aliases;
};

} // namespace VideoCore
