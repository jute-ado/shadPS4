// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
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

struct PhysicalBackingCachePageRequest {
    VAddr guest_page{};
    PhysicalBackingDeviceAddress override_page_address{};
};

struct PhysicalBackingCachePublicationBatch {
    std::vector<PhysicalBackingCachePageToken> tokens;
    std::vector<PhysicalBackingBdaDelta> deltas;
};

struct PhysicalBackingDirtySlice {
    u32 offset{};
    u32 size{};

    auto operator<=>(const PhysicalBackingDirtySlice&) const = default;
};

struct PhysicalBackingDirtyCachePagePublication {
    PhysicalBackingWriteback writeback{};
    std::vector<PhysicalBackingBdaDelta> deltas;
    std::vector<PhysicalBackingDirtySlice> dirty_slices;
};

struct PhysicalBackingOwnerRetirement {
    std::vector<PhysicalBackingBdaDelta> deltas;
    std::optional<PhysicalBackingWriteback> writeback;
    std::vector<PhysicalBackingDirtySlice> dirty_slices;
};

struct PhysicalBackingTextureToken {
    u64 generation{};
    std::vector<u64> physical_pages;

    auto operator<=>(const PhysicalBackingTextureToken&) const = default;
};

struct PhysicalBackingTexturePublication {
    PhysicalBackingTextureToken token{};
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
                const auto address = PublishedAddress(mapping->guest_page, mapping->physical_offset);
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
                active_cache_owners.contains(mapping_it->second.physical_offset) ||
                pending_writebacks.contains(mapping_it->second.physical_offset) ||
                texture_block_generations.contains(mapping_it->second.physical_offset) ||
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
            aliases_it->second.empty() || active_cache_owners.contains(physical_offset) ||
            pending_writebacks.contains(physical_offset) ||
            texture_block_generations.contains(physical_offset)) {
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
        PhysicalBackingCachePagePublication result{
            .token = {*publication},
            .deltas = MakeAliasDeltas(physical_offset),
        };
        active_cache_owners.emplace(physical_offset, ActiveCacheOwner{.token = result.token});
        return result;
    }

    [[nodiscard]] std::optional<PhysicalBackingCachePagePublication> ActivateCachePageForGuest(
        VAddr guest_page, PhysicalBackingDeviceAddress override_page_address,
        bool has_texture_overlap) {
        if (!IsPageAligned(guest_page)) {
            return std::nullopt;
        }
        const auto mapping_it = mapping_tokens.find(guest_page);
        if (mapping_it == mapping_tokens.end()) {
            return std::nullopt;
        }
        return ActivateCachePage(mapping_it->second.physical_offset, override_page_address,
                                 has_texture_overlap);
    }

    /// Atomically activates all distinct physical pages backing one synchronized cache buffer.
    [[nodiscard]] std::optional<PhysicalBackingCachePublicationBatch>
    ActivateCachePagesForGuests(std::span<const PhysicalBackingCachePageRequest> requests) {
        if (requests.empty()) {
            return std::nullopt;
        }

        struct PendingOwner {
            u64 physical_offset{};
            PhysicalBackingDeviceAddress override_page_address{};
        };
        std::vector<PendingOwner> pending;
        pending.reserve(requests.size());
        std::unordered_set<VAddr> guest_pages;
        std::unordered_set<u64> physical_pages;
        for (const auto& request : requests) {
            if (!IsPageAligned(request.guest_page) || request.override_page_address.value == 0 ||
                !guest_pages.emplace(request.guest_page).second) {
                return std::nullopt;
            }
            const auto mapping_it = mapping_tokens.find(request.guest_page);
            if (mapping_it == mapping_tokens.end()) {
                return std::nullopt;
            }
            const u64 physical_offset = mapping_it->second.physical_offset;
            if (!physical_pages.emplace(physical_offset).second) {
                continue;
            }
            if (!physical_aliases.contains(physical_offset) ||
                active_cache_owners.contains(physical_offset) ||
                pending_writebacks.contains(physical_offset) ||
                texture_block_generations.contains(physical_offset)) {
                return std::nullopt;
            }
            pending.push_back({physical_offset, request.override_page_address});
        }
        if (pending.empty() ||
            pending.size() > std::numeric_limits<u64>::max() - last_owner_generation) {
            return std::nullopt;
        }

        std::vector<PhysicalBackingCachePageToken> tokens;
        tokens.reserve(pending.size());
        for (const auto& owner : pending) {
            const u64 owner_generation = ++last_owner_generation;
            const auto publication =
                state.ActivateOverride(owner.physical_offset, owner.override_page_address,
                                       owner_generation);
            if (!publication) {
                for (auto token_it = tokens.rbegin(); token_it != tokens.rend(); ++token_it) {
                    static_cast<void>(state.RetireClean(token_it->publication));
                }
                return std::nullopt;
            }
            tokens.push_back({*publication});
        }

        PhysicalBackingCachePublicationBatch result{.tokens = std::move(tokens)};
        for (const auto& token : result.tokens) {
            active_cache_owners.emplace(token.publication.physical_offset,
                                        ActiveCacheOwner{.token = token});
            auto deltas = MakeAliasDeltas(token.publication.physical_offset);
            result.deltas.insert(result.deltas.end(), deltas.begin(), deltas.end());
        }
        std::ranges::sort(result.deltas, {}, &PhysicalBackingBdaDelta::guest_page);
        return result;
    }

    [[nodiscard]] std::optional<PhysicalBackingDeviceAddress> ResolveGuestPagePublication(
        VAddr guest_page) const {
        if (!IsPageAligned(guest_page)) {
            return std::nullopt;
        }
        const auto mapping_it = mapping_tokens.find(guest_page);
        if (mapping_it == mapping_tokens.end()) {
            return std::nullopt;
        }
        return PublishedAddress(guest_page, mapping_it->second.physical_offset);
    }

    [[nodiscard]] std::optional<std::vector<PhysicalBackingBdaDelta>> RetireCachePageClean(
        const PhysicalBackingCachePageToken& token) {
        const auto owner_it = FindActiveCacheOwner(token);
        if (owner_it == active_cache_owners.end() ||
            !physical_aliases.contains(token.publication.physical_offset) ||
            !state.RetireClean(token.publication)) {
            return std::nullopt;
        }
        active_cache_owners.erase(owner_it);
        return MakeAliasDeltas(token.publication.physical_offset);
    }

    [[nodiscard]] std::optional<PhysicalBackingDirtyCachePagePublication>
    RetireCachePageGpuDirty(const PhysicalBackingCachePageToken& token) {
        const auto owner_it = FindActiveCacheOwner(token);
        if (owner_it == active_cache_owners.end() ||
            !physical_aliases.contains(token.publication.physical_offset)) {
            return std::nullopt;
        }
        const auto writeback = state.RetireGpuDirty(token.publication);
        if (!writeback) {
            return std::nullopt;
        }
        auto dirty_slices = std::move(owner_it->second.dirty_slices);
        pending_writebacks.emplace(writeback->physical_offset, *writeback);
        active_cache_owners.erase(owner_it);
        return PhysicalBackingDirtyCachePagePublication{
            .writeback = *writeback,
            .deltas = MakeAliasDeltas(token.publication.physical_offset),
            .dirty_slices = std::move(dirty_slices),
        };
    }

    [[nodiscard]] bool MarkCachePageGpuDirty(const PhysicalBackingCachePageToken& token,
                                             u32 offset, u32 size) {
        const auto owner_it = FindActiveCacheOwner(token);
        if (owner_it == active_cache_owners.end() || size == 0 || offset >= PageSize ||
            size > PageSize - offset) {
            return false;
        }
        auto& slices = owner_it->second.dirty_slices;
        slices.push_back({offset, size});
        NormalizeDirtySlices(slices);
        return true;
    }

    [[nodiscard]] std::optional<PhysicalBackingOwnerRetirement> RetireOwnerForCpuWrite(
        VAddr guest_page) {
        const auto mapping_it = mapping_tokens.find(guest_page);
        if (mapping_it == mapping_tokens.end()) {
            return std::nullopt;
        }
        const u64 physical_offset = mapping_it->second.physical_offset;
        const auto owner_it = active_cache_owners.find(physical_offset);
        if (owner_it == active_cache_owners.end()) {
            return std::nullopt;
        }

        PhysicalBackingOwnerRetirement retirement{
            .dirty_slices = owner_it->second.dirty_slices,
        };
        if (retirement.dirty_slices.empty()) {
            if (!state.RetireClean(owner_it->second.token.publication)) {
                return std::nullopt;
            }
        } else {
            const auto writeback = state.RetireGpuDirty(owner_it->second.token.publication);
            if (!writeback) {
                return std::nullopt;
            }
            retirement.writeback = *writeback;
            pending_writebacks.emplace(physical_offset, *writeback);
        }
        active_cache_owners.erase(owner_it);
        retirement.deltas = MakeAliasDeltas(physical_offset);
        return retirement;
    }

    template <typename Commit>
    [[nodiscard]] std::optional<std::vector<PhysicalBackingBdaDelta>>
    CommitCachePageWriteback(const PhysicalBackingWriteback& writeback, Commit&& commit) {
        const auto pending_it = pending_writebacks.find(writeback.physical_offset);
        if (pending_it == pending_writebacks.end() || pending_it->second != writeback ||
            !physical_aliases.contains(writeback.physical_offset) ||
            !state.CommitOrderedWriteback(writeback, std::forward<Commit>(commit))) {
            return std::nullopt;
        }
        pending_writebacks.erase(pending_it);
        return MakeAliasDeltas(writeback.physical_offset);
    }

    /// Suppresses buffer BDA publication for every physical alias while a texture may own it.
    [[nodiscard]] std::optional<PhysicalBackingTexturePublication> BeginTextureOverlap(
        VAddr guest_base, u64 size) {
        if (size == 0 || !IsPageAligned(guest_base) || !IsPageAligned(size) ||
            guest_base >= AddressSpaceSize || size > AddressSpaceSize - guest_base) {
            return std::nullopt;
        }

        std::vector<u64> physical_pages;
        physical_pages.reserve(static_cast<size_t>(size / PageSize));
        for (u64 offset = 0; offset < size; offset += PageSize) {
            const auto mapping_it = mapping_tokens.find(guest_base + offset);
            if (mapping_it == mapping_tokens.end()) {
                return std::nullopt;
            }
            const u64 physical_offset = mapping_it->second.physical_offset;
            if (active_cache_owners.contains(physical_offset) ||
                pending_writebacks.contains(physical_offset)) {
                return std::nullopt;
            }
            physical_pages.push_back(physical_offset);
        }
        std::ranges::sort(physical_pages);
        physical_pages.erase(std::unique(physical_pages.begin(), physical_pages.end()),
                             physical_pages.end());

        const auto generation = AcquireTextureGeneration();
        if (!generation) {
            return std::nullopt;
        }
        for (const u64 physical_offset : physical_pages) {
            texture_block_generations[physical_offset].emplace(*generation);
        }
        PhysicalBackingTextureToken token{
            .generation = *generation,
            .physical_pages = std::move(physical_pages),
        };
        return PhysicalBackingTexturePublication{
            .token = token,
            .deltas = MakePhysicalDeltas(token.physical_pages),
        };
    }

    [[nodiscard]] std::optional<std::vector<PhysicalBackingBdaDelta>> EndTextureOverlap(
        const PhysicalBackingTextureToken& token) {
        if (token.generation == 0 || token.physical_pages.empty()) {
            return std::nullopt;
        }
        for (size_t index = 0; index < token.physical_pages.size(); ++index) {
            const u64 physical_offset = token.physical_pages[index];
            if ((index != 0 && physical_offset <= token.physical_pages[index - 1]) ||
                !IsPageAligned(physical_offset)) {
                return std::nullopt;
            }
            const auto block_it = texture_block_generations.find(physical_offset);
            if (block_it == texture_block_generations.end() ||
                !block_it->second.contains(token.generation) ||
                active_cache_owners.contains(physical_offset) ||
                pending_writebacks.contains(physical_offset)) {
                return std::nullopt;
            }
        }
        for (const u64 physical_offset : token.physical_pages) {
            auto block_it = texture_block_generations.find(physical_offset);
            block_it->second.erase(token.generation);
            if (block_it->second.empty()) {
                texture_block_generations.erase(block_it);
            }
        }
        return MakePhysicalDeltas(token.physical_pages);
    }

private:
    struct ActiveCacheOwner {
        PhysicalBackingCachePageToken token{};
        std::vector<PhysicalBackingDirtySlice> dirty_slices;
    };

    using ActiveCacheOwners = std::unordered_map<u64, ActiveCacheOwner>;

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

    [[nodiscard]] std::optional<u64> AcquireTextureGeneration() noexcept {
        if (last_texture_generation == std::numeric_limits<u64>::max()) {
            return std::nullopt;
        }
        return ++last_texture_generation;
    }

    [[nodiscard]] PhysicalBackingDeviceAddress PublishedAddress(VAddr guest_page,
                                                                u64 physical_offset) const {
        if (texture_block_generations.contains(physical_offset)) {
            return {};
        }
        return state.Resolve(guest_page);
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
            deltas.push_back({guest_page, PublishedAddress(guest_page, physical_offset)});
        }
        std::ranges::sort(deltas, {}, &PhysicalBackingBdaDelta::guest_page);
        return deltas;
    }

    [[nodiscard]] std::vector<PhysicalBackingBdaDelta> MakePhysicalDeltas(
        std::span<const u64> physical_pages) const {
        std::vector<PhysicalBackingBdaDelta> deltas;
        for (const u64 physical_offset : physical_pages) {
            auto physical_deltas = MakeAliasDeltas(physical_offset);
            deltas.insert(deltas.end(), physical_deltas.begin(), physical_deltas.end());
        }
        std::ranges::sort(deltas, {}, &PhysicalBackingBdaDelta::guest_page);
        return deltas;
    }

    [[nodiscard]] ActiveCacheOwners::iterator FindActiveCacheOwner(
        const PhysicalBackingCachePageToken& token) {
        const auto owner_it = active_cache_owners.find(token.publication.physical_offset);
        if (owner_it == active_cache_owners.end() || owner_it->second.token != token) {
            return active_cache_owners.end();
        }
        return owner_it;
    }

    static void NormalizeDirtySlices(std::vector<PhysicalBackingDirtySlice>& slices) {
        std::ranges::sort(slices, {}, &PhysicalBackingDirtySlice::offset);
        size_t output = 0;
        for (const PhysicalBackingDirtySlice slice : slices) {
            if (output != 0) {
                auto& previous = slices[output - 1];
                const u32 previous_end = previous.offset + previous.size;
                if (slice.offset <= previous_end) {
                    previous.size = std::max(previous_end, slice.offset + slice.size) -
                                    previous.offset;
                    continue;
                }
            }
            slices[output++] = slice;
        }
        slices.resize(output);
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
    u64 last_texture_generation{};
    std::unordered_map<VAddr, PhysicalBackingMapping> mapping_tokens;
    std::unordered_map<u64, std::unordered_set<VAddr>> physical_aliases;
    ActiveCacheOwners active_cache_owners;
    std::unordered_map<u64, PhysicalBackingWriteback> pending_writebacks;
    std::unordered_map<u64, std::unordered_set<u64>> texture_block_generations;
};

} // namespace VideoCore
