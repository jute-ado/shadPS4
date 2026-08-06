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

[[nodiscard]] constexpr bool ShouldAcquirePhysicalBackingTextureOwnership(
    bool is_already_gpu_modified, bool will_gpu_write) noexcept {
    return will_gpu_write && !is_already_gpu_modified;
}

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

struct PhysicalBackingCachePageMigration {
    PhysicalBackingCachePageToken previous_token{};
    PhysicalBackingCachePageToken token{};
    std::vector<PhysicalBackingBdaDelta> deltas;
};

struct PhysicalBackingCachePageMigrationBatch {
    std::vector<PhysicalBackingCachePageMigration> migrations;
    std::vector<PhysicalBackingBdaDelta> deltas;
};

struct PhysicalBackingCachePageRequest {
    VAddr guest_page{};
    PhysicalBackingDeviceAddress override_page_address{};
};

struct PhysicalBackingCachePageActivation {
    VAddr guest_page{};
    PhysicalBackingCachePageToken token{};
};

struct PhysicalBackingCachePublicationBatch {
    std::vector<PhysicalBackingCachePageActivation> owners;
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
        std::vector<VAddr> guest_pages;
        guest_pages.reserve(page_count);
        for (u64 offset = 0; offset < size; offset += PageSize) {
            guest_pages.push_back(guest_base + offset);
        }
        return UnmapPages(guest_pages);
    }

    /// Atomically removes an ordered subset of mapped guest pages.
    [[nodiscard]] std::optional<std::vector<PhysicalBackingBdaDelta>> UnmapPages(
        std::span<const VAddr> guest_pages) {
        if (guest_pages.empty()) {
            return std::nullopt;
        }
        std::vector<PhysicalBackingMapping> mappings;
        mappings.reserve(guest_pages.size());
        VAddr previous_guest_page = 0;
        bool first = true;
        for (const VAddr guest_page : guest_pages) {
            const auto mapping_it = mapping_tokens.find(guest_page);
            if (!IsPageAligned(guest_page) || (!first && guest_page <= previous_guest_page) ||
                mapping_it == mapping_tokens.end() ||
                active_cache_owners.contains(mapping_it->second.physical_offset) ||
                pending_writebacks.contains(mapping_it->second.physical_offset) ||
                texture_block_generations.contains(mapping_it->second.physical_offset) ||
                !state.CanUnmapGuestPage(mapping_it->second)) {
                return std::nullopt;
            }
            mappings.push_back(mapping_it->second);
            previous_guest_page = guest_page;
            first = false;
        }

        std::vector<PhysicalBackingBdaDelta> deltas;
        deltas.reserve(guest_pages.size());
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

    [[nodiscard]] bool RetirePhysicalAllocations(
        std::span<const Core::PhysicalBackingRetirement> retirements) {
        std::vector<std::pair<u64, u64>> tracked_pages;
        std::unordered_set<u64> batch_pages;
        for (const auto& retirement : retirements) {
            if (retirement.size == 0 || retirement.allocation_generation == 0 ||
                !IsPageAligned(retirement.physical_offset) ||
                !IsPageAligned(retirement.size) ||
                retirement.physical_offset >
                    std::numeric_limits<u64>::max() - (retirement.size - 1)) {
                return false;
            }
            for (u64 offset = 0; offset < retirement.size; offset += PageSize) {
                const u64 physical_offset = retirement.physical_offset + offset;
                if (!batch_pages.emplace(physical_offset).second) {
                    return false;
                }
                if (!state.HasPhysicalPage(physical_offset)) {
                    continue;
                }
                if (physical_aliases.contains(physical_offset) ||
                    active_cache_owners.contains(physical_offset) ||
                    texture_block_generations.contains(physical_offset) ||
                    !state.CanRetirePhysicalPage(physical_offset,
                                                 retirement.allocation_generation)) {
                    return false;
                }
                tracked_pages.emplace_back(physical_offset, retirement.allocation_generation);
            }
        }
        for (const auto& [physical_offset, allocation_generation] : tracked_pages) {
            if (!state.RetirePhysicalPage(physical_offset, allocation_generation)) {
                return false;
            }
            pending_writebacks.erase(physical_offset);
        }
        return true;
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
            VAddr guest_page{};
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
            pending.push_back({request.guest_page, physical_offset,
                               request.override_page_address});
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

        PhysicalBackingCachePublicationBatch result;
        result.owners.reserve(tokens.size());
        for (size_t index = 0; index < tokens.size(); ++index) {
            const auto& token = tokens[index];
            result.owners.push_back({pending[index].guest_page, token});
            active_cache_owners.emplace(token.publication.physical_offset,
                                        ActiveCacheOwner{.token = token});
            auto deltas = MakeAliasDeltas(token.publication.physical_offset);
            result.deltas.insert(result.deltas.end(), deltas.begin(), deltas.end());
        }
        std::ranges::sort(result.deltas, {}, &PhysicalBackingBdaDelta::guest_page);
        return result;
    }

    /// Acquires unowned pages atomically while retaining any already-valid physical owner.
    /// Existing owners remain owned by their original cache buffer and must migrate before a
    /// different alias receives GPU writes.
    [[nodiscard]] std::optional<PhysicalBackingCachePublicationBatch>
    AcquireCachePagesForGuests(std::span<const PhysicalBackingCachePageRequest> requests) {
        if (requests.empty()) {
            return std::nullopt;
        }
        std::vector<PhysicalBackingCachePageRequest> unowned_requests;
        std::vector<u64> existing_physical_pages;
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
            if (active_cache_owners.contains(physical_offset)) {
                existing_physical_pages.push_back(physical_offset);
            } else {
                unowned_requests.push_back(request);
            }
        }

        PhysicalBackingCachePublicationBatch result;
        if (!unowned_requests.empty()) {
            auto acquired = ActivateCachePagesForGuests(unowned_requests);
            if (!acquired) {
                return std::nullopt;
            }
            result = std::move(*acquired);
        }
        for (const u64 physical_offset : existing_physical_pages) {
            auto deltas = MakeAliasDeltas(physical_offset);
            result.deltas.insert(result.deltas.end(), deltas.begin(), deltas.end());
        }
        std::ranges::sort(result.deltas, {}, &PhysicalBackingBdaDelta::guest_page);
        return result;
    }

    /// Atomically replaces complete texture ownership with full-page dirty cache mirrors.
    /// No imported-backing delta is exposed between the two authoritative owners.
    [[nodiscard]] std::optional<PhysicalBackingCachePublicationBatch>
    TransitionTexturePagesToDirtyCachePages(
        std::span<const PhysicalBackingTextureToken> texture_tokens,
        std::span<const PhysicalBackingCachePageRequest> requests) {
        if (texture_tokens.empty() || requests.empty()) {
            return std::nullopt;
        }

        std::unordered_map<u64, std::unordered_set<u64>> release_generations;
        std::unordered_set<u64> texture_pages;
        for (const auto& token : texture_tokens) {
            if (token.generation == 0 || token.physical_pages.empty()) {
                return std::nullopt;
            }
            for (size_t index = 0; index < token.physical_pages.size(); ++index) {
                const u64 physical_offset = token.physical_pages[index];
                if (!IsPageAligned(physical_offset) ||
                    (index != 0 && physical_offset <= token.physical_pages[index - 1])) {
                    return std::nullopt;
                }
                const auto block_it = texture_block_generations.find(physical_offset);
                if (block_it == texture_block_generations.end() ||
                    !block_it->second.contains(token.generation) ||
                    !release_generations[physical_offset].emplace(token.generation).second) {
                    return std::nullopt;
                }
                texture_pages.emplace(physical_offset);
            }
        }
        for (const auto& [physical_offset, generations] : release_generations) {
            const auto block_it = texture_block_generations.find(physical_offset);
            if (block_it == texture_block_generations.end() || block_it->second != generations) {
                return std::nullopt;
            }
        }

        struct PendingOwner {
            VAddr guest_page{};
            u64 physical_offset{};
            PhysicalBackingDeviceAddress override_page_address{};
            std::optional<PhysicalBackingCachePageToken> existing_token;
            bool requires_migration{};
        };
        std::vector<PendingOwner> pending;
        pending.reserve(requests.size());
        std::unordered_set<VAddr> guest_pages;
        std::unordered_set<u64> requested_physical_pages;
        size_t alias_count = 0;
        for (const auto& request : requests) {
            if (!IsPageAligned(request.guest_page) || request.override_page_address.value == 0 ||
                request.override_page_address.value >
                    std::numeric_limits<u64>::max() - (PageSize - 1) ||
                !guest_pages.emplace(request.guest_page).second) {
                return std::nullopt;
            }
            const auto mapping_it = mapping_tokens.find(request.guest_page);
            if (mapping_it == mapping_tokens.end()) {
                return std::nullopt;
            }
            const u64 physical_offset = mapping_it->second.physical_offset;
            const auto aliases_it = physical_aliases.find(physical_offset);
            const auto owner_it = active_cache_owners.find(physical_offset);
            if (!requested_physical_pages.emplace(physical_offset).second ||
                aliases_it == physical_aliases.end() || aliases_it->second.empty() ||
                pending_writebacks.contains(physical_offset)) {
                return std::nullopt;
            }
            std::optional<PhysicalBackingCachePageToken> existing_token;
            bool requires_migration = false;
            if (owner_it != active_cache_owners.end()) {
                existing_token = owner_it->second.token;
                requires_migration =
                    state.Resolve(request.guest_page) != request.override_page_address;
            }
            if (aliases_it->second.size() > std::numeric_limits<size_t>::max() - alias_count) {
                return std::nullopt;
            }
            alias_count += aliases_it->second.size();
            pending.push_back({request.guest_page, physical_offset, request.override_page_address,
                               existing_token, requires_migration});
        }
        const auto new_owner_count = static_cast<u64>(std::ranges::count_if(
            pending, [](const PendingOwner& owner) {
                return !owner.existing_token.has_value() || owner.requires_migration;
            }));
        if (requested_physical_pages != texture_pages ||
            new_owner_count > std::numeric_limits<u64>::max() - last_owner_generation) {
            return std::nullopt;
        }

        std::vector<u64> staged_owner_generations(pending.size());
        u64 next_owner_index = 0;
        for (size_t index = 0; index < pending.size(); ++index) {
            const auto& owner = pending[index];
            if (owner.existing_token && !owner.requires_migration) {
                continue;
            }
            const u64 owner_generation = last_owner_generation + ++next_owner_index;
            staged_owner_generations[index] = owner_generation;
            if (owner.requires_migration &&
                !state.CanMigrateOverride(owner.existing_token->publication,
                                          owner.override_page_address, owner_generation)) {
                return std::nullopt;
            }
        }

        PhysicalBackingCachePublicationBatch result;
        result.owners.reserve(pending.size());
        result.deltas.reserve(alias_count);
        active_cache_owners.reserve(active_cache_owners.size() + pending.size());
        std::vector<std::optional<PhysicalBackingCachePageToken>> staged_tokens(pending.size());
        std::vector<PhysicalBackingCachePageToken> newly_staged_tokens;
        newly_staged_tokens.reserve(static_cast<size_t>(new_owner_count));
        for (size_t index = 0; index < pending.size(); ++index) {
            const auto& owner = pending[index];
            if (owner.existing_token && !owner.requires_migration) {
                staged_tokens[index] = owner.existing_token;
                continue;
            }
            if (owner.requires_migration) {
                continue;
            }
            const auto publication = state.ActivateOverride(
                owner.physical_offset, owner.override_page_address,
                staged_owner_generations[index]);
            if (!publication) {
                for (auto token_it = newly_staged_tokens.rbegin();
                     token_it != newly_staged_tokens.rend();
                     ++token_it) {
                    static_cast<void>(state.RetireClean(token_it->publication));
                }
                return std::nullopt;
            }
            staged_tokens[index] = PhysicalBackingCachePageToken{*publication};
            newly_staged_tokens.push_back(*staged_tokens[index]);
        }
        for (size_t index = 0; index < pending.size(); ++index) {
            const auto& owner = pending[index];
            if (!owner.requires_migration) {
                continue;
            }
            const auto publication = state.MigrateOverride(
                owner.existing_token->publication, owner.override_page_address,
                staged_owner_generations[index]);
            staged_tokens[index] = PhysicalBackingCachePageToken{publication.value()};
        }

        for (const u64 physical_offset : texture_pages) {
            texture_block_generations.erase(physical_offset);
        }
        last_owner_generation += new_owner_count;
        for (size_t index = 0; index < pending.size(); ++index) {
            const auto& owner = pending[index];
            const auto& token = *staged_tokens[index];
            if (owner.existing_token) {
                auto& active_owner = active_cache_owners.find(owner.physical_offset)->second;
                active_owner.token = token;
                active_owner.dirty_slices = {{0, static_cast<u32>(PageSize)}};
            } else {
                active_cache_owners.emplace(
                    owner.physical_offset,
                    ActiveCacheOwner{
                        .token = token,
                        .dirty_slices = {{0, static_cast<u32>(PageSize)}},
                    });
            }
            result.owners.push_back({owner.guest_page, token});
            for (const VAddr guest_page : physical_aliases.find(owner.physical_offset)->second) {
                result.deltas.push_back(
                    {guest_page, PublishedAddress(guest_page, owner.physical_offset)});
            }
        }
        std::ranges::sort(result.deltas, {}, &PhysicalBackingBdaDelta::guest_page);
        return result;
    }

    [[nodiscard]] std::optional<u64> ResolvePhysicalPageForGuest(VAddr guest_page) const {
        if (!IsPageAligned(guest_page)) {
            return std::nullopt;
        }
        const auto mapping_it = mapping_tokens.find(guest_page);
        if (mapping_it == mapping_tokens.end()) {
            return std::nullopt;
        }
        return mapping_it->second.physical_offset;
    }

    [[nodiscard]] std::optional<std::vector<u64>> ResolvePhysicalPagesForDeltas(
        std::span<const PhysicalBackingBdaDelta> deltas) const {
        std::vector<u64> physical_pages;
        physical_pages.reserve(deltas.size());
        for (const auto& delta : deltas) {
            const auto physical_page = ResolvePhysicalPageForGuest(delta.guest_page);
            if (!physical_page) {
                return std::nullopt;
            }
            physical_pages.push_back(*physical_page);
        }
        return physical_pages;
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

    [[nodiscard]] std::optional<PhysicalBackingCachePageToken>
    ResolveActiveCachePageForGuest(VAddr guest_page) const {
        if (!IsPageAligned(guest_page)) {
            return std::nullopt;
        }
        const auto mapping_it = mapping_tokens.find(guest_page);
        if (mapping_it == mapping_tokens.end()) {
            return std::nullopt;
        }
        const auto owner_it = active_cache_owners.find(mapping_it->second.physical_offset);
        if (owner_it == active_cache_owners.end()) {
            return std::nullopt;
        }
        return owner_it->second.token;
    }

    [[nodiscard]] std::vector<PhysicalBackingBdaDelta> SelectActiveCacheOwnerDeltas(
        std::span<const PhysicalBackingBdaDelta> deltas) const {
        std::vector<PhysicalBackingBdaDelta> owned_deltas;
        owned_deltas.reserve(deltas.size());
        for (const auto& delta : deltas) {
            if (ResolveActiveCachePageForGuest(delta.guest_page)) {
                owned_deltas.push_back(delta);
            }
        }
        return owned_deltas;
    }

    [[nodiscard]] std::optional<PhysicalBackingCachePageMigration> MigrateCachePageForGuest(
        VAddr guest_page, PhysicalBackingDeviceAddress override_page_address) {
        if (!IsPageAligned(guest_page) || override_page_address.value == 0) {
            return std::nullopt;
        }
        const auto mapping_it = mapping_tokens.find(guest_page);
        if (mapping_it == mapping_tokens.end()) {
            return std::nullopt;
        }
        const u64 physical_offset = mapping_it->second.physical_offset;
        const auto owner_it = active_cache_owners.find(physical_offset);
        if (owner_it == active_cache_owners.end() || pending_writebacks.contains(physical_offset) ||
            texture_block_generations.contains(physical_offset)) {
            return std::nullopt;
        }
        const auto owner_generation = AcquireOwnerGeneration();
        if (!owner_generation) {
            return std::nullopt;
        }
        const PhysicalBackingCachePageToken previous_token = owner_it->second.token;
        const auto migrated = state.MigrateOverride(previous_token.publication,
                                                    override_page_address, *owner_generation);
        if (!migrated) {
            return std::nullopt;
        }
        owner_it->second.token = {*migrated};
        return PhysicalBackingCachePageMigration{
            .previous_token = previous_token,
            .token = owner_it->second.token,
            .deltas = MakeAliasDeltas(physical_offset),
        };
    }

    /// Atomically transfers distinct live cache owners into replacement buffer storage.
    [[nodiscard]] std::optional<PhysicalBackingCachePageMigrationBatch>
    MigrateCachePagesForGuests(std::span<const PhysicalBackingCachePageRequest> requests) {
        if (requests.empty() ||
            requests.size() > std::numeric_limits<u64>::max() - last_owner_generation) {
            return std::nullopt;
        }

        struct PendingMigration {
            VAddr guest_page{};
            u64 physical_offset{};
            PhysicalBackingDeviceAddress override_page_address{};
            PhysicalBackingCachePageToken previous_token{};
        };
        std::vector<PendingMigration> pending;
        pending.reserve(requests.size());
        std::unordered_set<VAddr> guest_pages;
        std::unordered_set<u64> physical_pages;
        for (const auto& request : requests) {
            if (!IsPageAligned(request.guest_page) || request.override_page_address.value == 0 ||
                !guest_pages.emplace(request.guest_page).second) {
                return std::nullopt;
            }
            const auto mapping_it = mapping_tokens.find(request.guest_page);
            if (mapping_it == mapping_tokens.end() ||
                !physical_pages.emplace(mapping_it->second.physical_offset).second) {
                return std::nullopt;
            }
            const u64 physical_offset = mapping_it->second.physical_offset;
            const auto owner_it = active_cache_owners.find(physical_offset);
            if (owner_it == active_cache_owners.end() ||
                pending_writebacks.contains(physical_offset) ||
                texture_block_generations.contains(physical_offset)) {
                return std::nullopt;
            }
            pending.push_back({request.guest_page, physical_offset,
                               request.override_page_address, owner_it->second.token});
        }

        std::vector<PhysicalBackingOverrideMigrationRequest> state_requests;
        state_requests.reserve(pending.size());
        for (size_t index = 0; index < pending.size(); ++index) {
            state_requests.push_back({
                .current = pending[index].previous_token.publication,
                .override_page_address = pending[index].override_page_address,
                .owner_generation = last_owner_generation + index + 1,
            });
        }

        PhysicalBackingCachePageMigrationBatch result;
        result.migrations.reserve(pending.size());
        std::vector<std::vector<PhysicalBackingBdaDelta>> migration_deltas;
        migration_deltas.reserve(pending.size());
        size_t delta_count = 0;
        for (const auto& migration : pending) {
            auto deltas = MakeAliasDeltas(migration.physical_offset);
            if (deltas.size() > std::numeric_limits<size_t>::max() - delta_count) {
                return std::nullopt;
            }
            delta_count += deltas.size();
            migration_deltas.push_back(std::move(deltas));
        }
        result.deltas.reserve(delta_count);

        const auto migrated_overrides = state.MigrateOverrides(state_requests);
        if (!migrated_overrides) {
            return std::nullopt;
        }
        last_owner_generation += pending.size();

        for (size_t index = 0; index < pending.size(); ++index) {
            auto owner_it = active_cache_owners.find(pending[index].physical_offset);
            owner_it->second.token = {(*migrated_overrides)[index]};
            result.migrations.push_back({
                .previous_token = pending[index].previous_token,
                .token = owner_it->second.token,
                .deltas = std::move(migration_deltas[index]),
            });
            const auto& deltas = result.migrations.back().deltas;
            result.deltas.insert(result.deltas.end(), deltas.begin(), deltas.end());
        }
        std::ranges::sort(result.deltas, {}, &PhysicalBackingBdaDelta::guest_page);
        return result;
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

    [[nodiscard]] std::optional<std::vector<PhysicalBackingDirtySlice>>
    ResolveCachePageDirtySlices(const PhysicalBackingCachePageToken& token) const {
        const auto owner_it = active_cache_owners.find(token.publication.physical_offset);
        if (owner_it == active_cache_owners.end() || owner_it->second.token != token) {
            return std::nullopt;
        }
        return owner_it->second.dirty_slices;
    }

    [[nodiscard]] bool MarkCachePageGpuDirtyForGuest(VAddr guest_page, u32 offset, u32 size) {
        if (!IsPageAligned(guest_page)) {
            return false;
        }
        const auto mapping_it = mapping_tokens.find(guest_page);
        if (mapping_it == mapping_tokens.end()) {
            return false;
        }
        const auto owner_it = active_cache_owners.find(mapping_it->second.physical_offset);
        if (owner_it == active_cache_owners.end()) {
            return false;
        }
        return MarkCachePageGpuDirty(owner_it->second.token, offset, size);
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

    /// Publishes a prevalidated batch whose GPU copies into canonical backing have already been
    /// encoded in command order. Host visibility remains guarded by the writeback timeline tracker.
    [[nodiscard]] std::optional<std::vector<PhysicalBackingBdaDelta>>
    PublishSubmittedCachePageGpuWritebacks(
        std::span<const PhysicalBackingWriteback> writebacks) {
        if (writebacks.empty()) {
            return std::nullopt;
        }
        std::vector<u64> physical_pages;
        physical_pages.reserve(writebacks.size());
        u64 previous_physical_offset = 0;
        bool first = true;
        for (const auto& writeback : writebacks) {
            const auto pending = pending_writebacks.find(writeback.physical_offset);
            if ((!first && writeback.physical_offset <= previous_physical_offset) ||
                pending == pending_writebacks.end() || pending->second != writeback ||
                !physical_aliases.contains(writeback.physical_offset)) {
                return std::nullopt;
            }
            physical_pages.push_back(writeback.physical_offset);
            previous_physical_offset = writeback.physical_offset;
            first = false;
        }
        if (!state.PublishSubmittedWritebacks(writebacks)) {
            return std::nullopt;
        }
        for (const u64 physical_page : physical_pages) {
            pending_writebacks.erase(physical_page);
        }
        return MakePhysicalDeltas(physical_pages);
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
            if (texture_block_generations.contains(physical_offset) ||
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
