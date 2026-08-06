// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <compare>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "common/types.h"

namespace VideoCore {

struct GuestBdaDeviceAddress {
    u64 value{};

    auto operator<=>(const GuestBdaDeviceAddress&) const = default;
};

struct GuestBdaPhysicalPage {
    u64 backing_id{};
    u64 offset{};

    auto operator<=>(const GuestBdaPhysicalPage&) const = default;
};

struct GuestBdaPhysicalRange {
    u64 backing_id{};
    u64 offset{};
};

struct GuestBdaPageRange {
    VAddr address{};
    u64 size{};
};

enum class GuestBdaBackingKind {
    SharedPhysical,
    Private,
    File,
    Unsupported,
};

enum class GuestBdaPageProvenance {
    PhysicalFallback,
    CacheOverride,
};

enum class GuestBdaCacheCoherence {
    CoherentWithBacking,
    GpuDirty,
};

struct GuestBdaFallbackRange {
    GuestBdaPageRange guest;
    GuestBdaPhysicalRange physical;
    GuestBdaDeviceAddress device_address;
    GuestBdaBackingKind kind{GuestBdaBackingKind::Unsupported};
    bool coherent{};
    bool gpu_dirty{};
};

struct GuestBdaMapping {
    GuestBdaPageRange guest;
    u64 generation{};
};

struct GuestBdaCacheRegistration {
    GuestBdaPageRange guest;
    u64 mapping_generation{};
    u64 registration_generation{};
};

struct GuestBdaPageValue {
    GuestBdaDeviceAddress device_address;
    GuestBdaPhysicalPage physical;
    GuestBdaPageProvenance provenance{GuestBdaPageProvenance::PhysicalFallback};
    u64 mapping_generation{};
};

class GuestBdaPageDirectory {
public:
    static constexpr u32 PageBits = 14;
    static constexpr u64 PageSize = u64{1} << PageBits;
    static constexpr VAddr AddressLimit = u64{1} << 40;

    [[nodiscard]] std::optional<GuestBdaMapping> MapFallback(const GuestBdaFallbackRange& mapping) {
        if (!IsGuestRangeValid(mapping.guest)) {
            return std::nullopt;
        }

        const std::scoped_lock lock{mutex};
        if (!IsFallbackValid(mapping)) {
            EraseRange(mapping.guest);
            return std::nullopt;
        }

        const u64 generation = NextGeneration();
        for (u64 offset = 0; offset < mapping.guest.size; offset += PageSize) {
            const GuestBdaPageValue fallback{
                .device_address = GuestBdaDeviceAddress{mapping.device_address.value + offset},
                .physical =
                    {
                        .backing_id = mapping.physical.backing_id,
                        .offset = mapping.physical.offset + offset,
                    },
                .provenance = GuestBdaPageProvenance::PhysicalFallback,
                .mapping_generation = generation,
            };
            pages[PageIndex(mapping.guest.address + offset)] = PageState{
                .fallback = fallback,
                .current = fallback,
                .mapping_generation = generation,
            };
        }
        return GuestBdaMapping{.guest = mapping.guest, .generation = generation};
    }

    [[nodiscard]] std::optional<GuestBdaCacheRegistration> RegisterCache(
        const GuestBdaMapping& mapping, GuestBdaDeviceAddress device_address) {
        if (!IsGuestRangeValid(mapping.guest) || !IsPageAligned(device_address.value) ||
            AddOverflows(device_address.value, mapping.guest.size)) {
            return std::nullopt;
        }

        const std::scoped_lock lock{mutex};
        if (!EveryPageMatches(mapping, 0)) {
            return std::nullopt;
        }

        const u64 registration_generation = NextGeneration();
        for (u64 offset = 0; offset < mapping.guest.size; offset += PageSize) {
            auto& state = pages.at(PageIndex(mapping.guest.address + offset));
            state.current.device_address = GuestBdaDeviceAddress{device_address.value + offset};
            state.current.provenance = GuestBdaPageProvenance::CacheOverride;
            state.registration_generation = registration_generation;
        }
        return GuestBdaCacheRegistration{
            .guest = mapping.guest,
            .mapping_generation = mapping.generation,
            .registration_generation = registration_generation,
        };
    }

    [[nodiscard]] bool UnregisterCache(const GuestBdaCacheRegistration& registration,
                                       GuestBdaCacheCoherence coherence) {
        if (!IsGuestRangeValid(registration.guest) || registration.registration_generation == 0) {
            return false;
        }

        const std::scoped_lock lock{mutex};
        const GuestBdaMapping mapping{
            .guest = registration.guest,
            .generation = registration.mapping_generation,
        };
        if (!EveryPageMatches(mapping, registration.registration_generation)) {
            return false;
        }

        if (coherence == GuestBdaCacheCoherence::GpuDirty) {
            EraseRange(registration.guest);
            return true;
        }

        for (u64 offset = 0; offset < registration.guest.size; offset += PageSize) {
            auto& state = pages.at(PageIndex(registration.guest.address + offset));
            state.current = state.fallback;
            state.registration_generation = 0;
        }
        return true;
    }

    [[nodiscard]] bool Unmap(const GuestBdaMapping& mapping) {
        if (!IsGuestRangeValid(mapping.guest)) {
            return false;
        }

        const std::scoped_lock lock{mutex};
        if (!EveryPageHasMapping(mapping)) {
            return false;
        }
        EraseRange(mapping.guest);
        return true;
    }

    [[nodiscard]] std::optional<GuestBdaPageValue> ResolvePage(VAddr page_address) const {
        if (page_address >= AddressLimit || !IsPageAligned(page_address)) {
            return std::nullopt;
        }

        const std::scoped_lock lock{mutex};
        const auto it = pages.find(PageIndex(page_address));
        return it == pages.end() ? std::nullopt
                                 : std::optional<GuestBdaPageValue>{it->second.current};
    }

private:
    struct PageState {
        GuestBdaPageValue fallback;
        GuestBdaPageValue current;
        u64 mapping_generation{};
        u64 registration_generation{};
    };

    [[nodiscard]] static constexpr bool IsPageAligned(u64 value) {
        return (value & (PageSize - 1)) == 0;
    }

    [[nodiscard]] static constexpr bool AddOverflows(u64 value, u64 size) {
        return size > std::numeric_limits<u64>::max() - value;
    }

    [[nodiscard]] static constexpr bool IsGuestRangeValid(const GuestBdaPageRange& range) {
        return range.size != 0 && IsPageAligned(range.address) && IsPageAligned(range.size) &&
               range.address < AddressLimit && range.size <= AddressLimit - range.address;
    }

    [[nodiscard]] static constexpr bool IsFallbackValid(const GuestBdaFallbackRange& mapping) {
        return mapping.kind == GuestBdaBackingKind::SharedPhysical && mapping.coherent &&
               !mapping.gpu_dirty && IsPageAligned(mapping.physical.offset) &&
               IsPageAligned(mapping.device_address.value) &&
               !AddOverflows(mapping.physical.offset, mapping.guest.size) &&
               !AddOverflows(mapping.device_address.value, mapping.guest.size);
    }

    [[nodiscard]] static constexpr u64 PageIndex(VAddr address) {
        return address >> PageBits;
    }

    [[nodiscard]] bool EveryPageHasMapping(const GuestBdaMapping& mapping) const {
        for (u64 offset = 0; offset < mapping.guest.size; offset += PageSize) {
            const auto it = pages.find(PageIndex(mapping.guest.address + offset));
            if (it == pages.end() || it->second.mapping_generation != mapping.generation) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool EveryPageMatches(const GuestBdaMapping& mapping,
                                        u64 registration_generation) const {
        for (u64 offset = 0; offset < mapping.guest.size; offset += PageSize) {
            const auto it = pages.find(PageIndex(mapping.guest.address + offset));
            if (it == pages.end() || it->second.mapping_generation != mapping.generation ||
                it->second.registration_generation != registration_generation) {
                return false;
            }
        }
        return true;
    }

    void EraseRange(const GuestBdaPageRange& range) {
        for (u64 offset = 0; offset < range.size; offset += PageSize) {
            pages.erase(PageIndex(range.address + offset));
        }
    }

    [[nodiscard]] u64 NextGeneration() {
        ++generation;
        if (generation == 0) {
            ++generation;
        }
        return generation;
    }

    mutable std::mutex mutex;
    std::unordered_map<u64, PageState> pages;
    u64 generation{};
};

} // namespace VideoCore
