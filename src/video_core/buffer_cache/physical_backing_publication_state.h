// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <optional>
#include <unordered_map>

#include "common/types.h"

namespace VideoCore {

/// A GPU device address. This deliberately cannot be constructed from a host pointer.
struct PhysicalBackingDeviceAddress {
    u64 value{};

    auto operator<=>(const PhysicalBackingDeviceAddress&) const = default;
};

/// Identifies one generation of a guest-to-physical page mapping.
struct PhysicalBackingMapping {
    VAddr guest_page{};
    u64 physical_offset{};
    u64 mapping_generation{};
    u64 allocation_generation{};

    auto operator<=>(const PhysicalBackingMapping&) const = default;
};

/// Identifies one owner generation of a physical-page cache override.
struct PhysicalBackingOverride {
    u64 physical_offset{};
    u64 owner_generation{};
    u64 state_generation{};

    auto operator<=>(const PhysicalBackingOverride&) const = default;
};

/// One ordered writeback whose completion may restore imported backing publication.
struct PhysicalBackingWriteback {
    u64 physical_offset{};
    u64 allocation_generation{};
    u64 owner_generation{};
    u64 state_generation{};
    u64 mapping_epoch{};
    u64 writeback_generation{};

    auto operator<=>(const PhysicalBackingWriteback&) const = default;
};

/// Pure publication policy for a page-file-backed GPU import and cache overrides.
///
/// Guest pages refer indirectly to physical pages. Resolve therefore performs two
/// bounded hash lookups and an override changes every alias without rewriting any
/// guest entry. Mutations are generation checked so delayed cache retirement or
/// writeback work cannot publish a newer mapping accidentally.
class PhysicalBackingPublicationState {
public:
    static constexpr u64 PageSize = 16_KB;

    PhysicalBackingPublicationState(PhysicalBackingDeviceAddress imported_base,
                                    u64 backing_size) noexcept
        : imported_base{imported_base}, backing_size{backing_size},
          eligible{IsBackingEligible(imported_base, backing_size)} {}

    [[nodiscard]] std::optional<PhysicalBackingMapping> MapGuestPage(VAddr guest_page,
                                                                     u64 physical_offset,
                                                                     u64 mapping_generation,
                                                                     u64 allocation_generation) {
        if (!eligible || mapping_generation == 0 || allocation_generation == 0 ||
            !IsPageAligned(guest_page) || !IsPhysicalPageEligible(physical_offset) ||
            !HasCompletePage(guest_page) || guest_mappings.contains(guest_page)) {
            return std::nullopt;
        }
        const auto generation_it = guest_generations.find(guest_page);
        if (generation_it != guest_generations.end() &&
            mapping_generation <= generation_it->second) {
            return std::nullopt;
        }

        auto [physical_it, inserted] = physical_pages.try_emplace(physical_offset);
        PhysicalPageState& physical = physical_it->second;
        if (!inserted && physical.allocation_generation != allocation_generation) {
            return std::nullopt;
        }
        const auto next_epoch = NextGeneration(physical.mapping_epoch);
        if (!next_epoch || physical.alias_count == std::numeric_limits<u64>::max()) {
            if (inserted) {
                physical_pages.erase(physical_it);
            }
            return std::nullopt;
        }

        const PhysicalBackingMapping mapping{
            .guest_page = guest_page,
            .physical_offset = physical_offset,
            .mapping_generation = mapping_generation,
            .allocation_generation = allocation_generation,
        };
        const auto [guest_it, guest_inserted] = guest_mappings.emplace(guest_page, mapping);
        if (!guest_inserted) {
            if (inserted) {
                physical_pages.erase(physical_it);
            }
            return std::nullopt;
        }

        if (inserted) {
            physical.allocation_generation = allocation_generation;
        }
        physical.mapping_epoch = *next_epoch;
        ++physical.alias_count;
        guest_generations.insert_or_assign(guest_page, mapping_generation);
        return guest_it->second;
    }

    [[nodiscard]] bool UnmapGuestPage(const PhysicalBackingMapping& mapping) {
        const auto guest_it = guest_mappings.find(mapping.guest_page);
        if (guest_it == guest_mappings.end() || guest_it->second != mapping) {
            return false;
        }

        const auto physical_it = physical_pages.find(mapping.physical_offset);
        if (physical_it == physical_pages.end() || physical_it->second.alias_count == 0) {
            return false;
        }
        PhysicalPageState& physical = physical_it->second;
        const auto next_epoch = NextGeneration(physical.mapping_epoch);
        if (!next_epoch) {
            return false;
        }

        physical.mapping_epoch = *next_epoch;
        --physical.alias_count;
        guest_mappings.erase(guest_it);
        return true;
    }

    [[nodiscard]] PhysicalBackingDeviceAddress Resolve(VAddr guest_page) const noexcept {
        if (!IsPageAligned(guest_page)) {
            return {};
        }
        const auto guest_it = guest_mappings.find(guest_page);
        if (guest_it == guest_mappings.end()) {
            return {};
        }
        const auto physical_it = physical_pages.find(guest_it->second.physical_offset);
        if (physical_it == physical_pages.end()) {
            return {};
        }

        const PhysicalPageState& physical = physical_it->second;
        switch (physical.publication) {
        case Publication::ImportedBacking:
            return {imported_base.value + guest_it->second.physical_offset};
        case Publication::CacheOverride:
            return physical.override_address;
        case Publication::AwaitingWriteback:
            return {};
        }
        return {};
    }

    [[nodiscard]] std::optional<PhysicalBackingOverride> ActivateOverride(
        u64 physical_offset, PhysicalBackingDeviceAddress override_page_address,
        u64 owner_generation) {
        if (!eligible || owner_generation == 0 || override_page_address.value == 0 ||
            !IsPhysicalPageEligible(physical_offset)) {
            return std::nullopt;
        }
        const auto physical_it = physical_pages.find(physical_offset);
        if (physical_it == physical_pages.end() || physical_it->second.alias_count == 0) {
            return std::nullopt;
        }

        PhysicalPageState& physical = physical_it->second;
        const bool replaces_stale_writeback =
            physical.publication == Publication::AwaitingWriteback &&
            owner_generation > physical.owner_generation;
        if (physical.publication != Publication::ImportedBacking && !replaces_stale_writeback) {
            return std::nullopt;
        }
        const auto next_state = NextGeneration(physical.state_generation);
        if (!next_state) {
            return std::nullopt;
        }

        physical.publication = Publication::CacheOverride;
        physical.owner_generation = owner_generation;
        physical.state_generation = *next_state;
        physical.override_address = override_page_address;
        return PhysicalBackingOverride{
            .physical_offset = physical_offset,
            .owner_generation = owner_generation,
            .state_generation = physical.state_generation,
        };
    }

    [[nodiscard]] bool RetireClean(const PhysicalBackingOverride& override) {
        const auto physical_it = FindActiveOverride(override);
        if (physical_it == physical_pages.end()) {
            return false;
        }
        RestoreImportedBacking(physical_it->second);
        return true;
    }

    [[nodiscard]] std::optional<PhysicalBackingWriteback> RetireGpuDirty(
        const PhysicalBackingOverride& override) {
        const auto physical_it = FindActiveOverride(override);
        if (physical_it == physical_pages.end()) {
            return std::nullopt;
        }

        PhysicalPageState& physical = physical_it->second;
        const auto next_writeback = NextGeneration(physical.writeback_generation);
        if (!next_writeback) {
            return std::nullopt;
        }
        physical.publication = Publication::AwaitingWriteback;
        physical.override_address = {};
        physical.writeback_generation = *next_writeback;
        return PhysicalBackingWriteback{
            .physical_offset = override.physical_offset,
            .allocation_generation = physical.allocation_generation,
            .owner_generation = override.owner_generation,
            .state_generation = override.state_generation,
            .mapping_epoch = physical.mapping_epoch,
            .writeback_generation = physical.writeback_generation,
        };
    }

    [[nodiscard]] bool CompleteOrderedWriteback(const PhysicalBackingWriteback& writeback) {
        const auto physical_it = physical_pages.find(writeback.physical_offset);
        if (physical_it == physical_pages.end()) {
            return false;
        }

        PhysicalPageState& physical = physical_it->second;
        if (physical.publication != Publication::AwaitingWriteback ||
            physical.allocation_generation != writeback.allocation_generation ||
            physical.owner_generation != writeback.owner_generation ||
            physical.state_generation != writeback.state_generation ||
            physical.mapping_epoch != writeback.mapping_epoch ||
            physical.writeback_generation != writeback.writeback_generation) {
            return false;
        }

        RestoreImportedBacking(physical);
        return true;
    }

    /// Marks a freed physical page as a new allocation generation. This is the
    /// explicit recovery path when an old allocation was retired while a dirty
    /// writeback was still pending. No guest alias may remain at this boundary.
    [[nodiscard]] bool ReallocatePhysicalPage(u64 physical_offset,
                                              u64 previous_allocation_generation,
                                              u64 new_allocation_generation) {
        if (new_allocation_generation == 0 ||
            new_allocation_generation <= previous_allocation_generation) {
            return false;
        }
        const auto physical_it = physical_pages.find(physical_offset);
        if (physical_it == physical_pages.end()) {
            return false;
        }
        PhysicalPageState& physical = physical_it->second;
        if (physical.alias_count != 0 ||
            physical.allocation_generation != previous_allocation_generation) {
            return false;
        }
        const auto next_state = NextGeneration(physical.state_generation);
        const auto next_epoch = NextGeneration(physical.mapping_epoch);
        if (!next_state || !next_epoch) {
            return false;
        }

        RestoreImportedBacking(physical);
        physical.allocation_generation = new_allocation_generation;
        physical.state_generation = *next_state;
        physical.mapping_epoch = *next_epoch;
        return true;
    }

private:
    enum class Publication {
        ImportedBacking,
        CacheOverride,
        AwaitingWriteback,
    };

    struct PhysicalPageState {
        Publication publication{Publication::ImportedBacking};
        PhysicalBackingDeviceAddress override_address{};
        u64 owner_generation{};
        u64 allocation_generation{};
        u64 state_generation{};
        u64 mapping_epoch{};
        u64 writeback_generation{};
        u64 alias_count{};
    };

    using PhysicalPages = std::unordered_map<u64, PhysicalPageState>;

    [[nodiscard]] static constexpr bool IsPageAligned(u64 value) noexcept {
        return (value & (PageSize - 1)) == 0;
    }

    [[nodiscard]] static constexpr bool HasCompletePage(u64 address) noexcept {
        return address <= std::numeric_limits<u64>::max() - PageSize;
    }

    [[nodiscard]] static constexpr std::optional<u64> NextGeneration(u64 generation) noexcept {
        if (generation == std::numeric_limits<u64>::max()) {
            return std::nullopt;
        }
        return generation + 1;
    }

    [[nodiscard]] static constexpr bool IsBackingEligible(PhysicalBackingDeviceAddress base,
                                                          u64 size) noexcept {
        return base.value != 0 && size >= PageSize && IsPageAligned(size) &&
               base.value <= std::numeric_limits<u64>::max() - (size - PageSize);
    }

    [[nodiscard]] bool IsPhysicalPageEligible(u64 physical_offset) const noexcept {
        return IsPageAligned(physical_offset) && physical_offset <= backing_size - PageSize &&
               imported_base.value <= std::numeric_limits<u64>::max() - physical_offset;
    }

    [[nodiscard]] PhysicalPages::iterator FindActiveOverride(
        const PhysicalBackingOverride& override) {
        const auto physical_it = physical_pages.find(override.physical_offset);
        if (physical_it == physical_pages.end()) {
            return physical_it;
        }
        const PhysicalPageState& physical = physical_it->second;
        if (physical.publication != Publication::CacheOverride ||
            physical.owner_generation != override.owner_generation ||
            physical.state_generation != override.state_generation) {
            return physical_pages.end();
        }
        return physical_it;
    }

    static void RestoreImportedBacking(PhysicalPageState& physical) noexcept {
        physical.publication = Publication::ImportedBacking;
        physical.override_address = {};
        physical.owner_generation = 0;
    }

    PhysicalBackingDeviceAddress imported_base{};
    u64 backing_size{};
    bool eligible{};
    std::unordered_map<VAddr, PhysicalBackingMapping> guest_mappings;
    std::unordered_map<VAddr, u64> guest_generations;
    PhysicalPages physical_pages;
};

} // namespace VideoCore
