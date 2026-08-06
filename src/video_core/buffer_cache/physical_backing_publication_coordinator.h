// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/physical_backing_provenance.h"
#include "video_core/buffer_cache/physical_backing_publication_state.h"

namespace VideoCore {

struct PhysicalBackingBdaDelta {
    VAddr guest_page{};
    PhysicalBackingDeviceAddress device_address{};

    auto operator<=>(const PhysicalBackingBdaDelta&) const = default;
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
                mapped.push_back(*mapping);
                deltas.push_back({mapping->guest_page, address});
            }
        }
        return deltas;
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

    void RollbackMappings(std::span<const PhysicalBackingMapping> mappings) {
        for (auto mapping_it = mappings.rbegin(); mapping_it != mappings.rend(); ++mapping_it) {
            static_cast<void>(state.UnmapGuestPage(*mapping_it));
            mapping_tokens.erase(mapping_it->guest_page);
        }
    }

    PhysicalBackingPublicationState state;
    u64 last_mapping_generation{};
    std::unordered_map<VAddr, PhysicalBackingMapping> mapping_tokens;
};

} // namespace VideoCore
