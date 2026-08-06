// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/types.h"

namespace VideoCore {

enum class PhysicalBackingTextureProducer {
    Copy,
    ComputeShader,
};

[[nodiscard]] constexpr PhysicalBackingTextureProducer PhysicalBackingTextureMirrorProducer(
    bool is_tiled) noexcept {
    return is_tiled ? PhysicalBackingTextureProducer::ComputeShader
                    : PhysicalBackingTextureProducer::Copy;
}

[[nodiscard]] constexpr bool ShouldAcquirePhysicalBackingBufferOwnership(
    bool will_gpu_write) noexcept {
    return will_gpu_write;
}

class PhysicalBackingTextureWriteOrderTracker {
public:
    [[nodiscard]] bool Acquire(u32 image_index) {
        return write_orders.try_emplace(image_index, 0).second;
    }

    [[nodiscard]] bool MarkGpuWrites(std::span<const u32> image_indices) {
        if (image_indices.empty() || last_write_order == std::numeric_limits<u64>::max()) {
            return false;
        }
        for (const u32 image_index : image_indices) {
            if (!write_orders.contains(image_index)) {
                return false;
            }
        }
        const u64 write_order = ++last_write_order;
        for (const u32 image_index : image_indices) {
            write_orders[image_index] = write_order;
        }
        return true;
    }

    [[nodiscard]] u64 Get(u32 image_index) const {
        const auto order = write_orders.find(image_index);
        return order == write_orders.end() ? 0 : order->second;
    }

    [[nodiscard]] bool Release(u32 image_index) {
        return write_orders.erase(image_index) == 1;
    }

private:
    std::unordered_map<u32, u64> write_orders;
    u64 last_write_order{};
};

[[nodiscard]] constexpr bool ShouldInvalidateTextureCacheBeforeGpuBufferFill(
    bool is_gds, bool requires_gpu_fill) noexcept {
    return !is_gds && requires_gpu_fill;
}

struct PhysicalBackingAliasMigrationCopy {
    u64 source_offset{};
    u64 destination_offset{};
    u64 size{};
};

struct PhysicalBackingTextureBufferTransition {
    VAddr base{};
    u32 size{};
};

struct PhysicalBackingTextureOwnershipRecord {
    u32 image_index{};
    VAddr guest_base{};
    u64 guest_size{};
    u64 gpu_write_order{};
    std::vector<u64> physical_pages;
};

struct PhysicalBackingTextureOwnershipImage {
    u32 image_index{};
    u64 gpu_write_order{};

    auto operator<=>(const PhysicalBackingTextureOwnershipImage&) const = default;
};

struct PhysicalBackingTextureOwnershipComponent {
    std::vector<PhysicalBackingTextureOwnershipImage> oldest_to_newest_images;
    PhysicalBackingTextureBufferTransition ownership_span{};
    std::vector<u64> physical_pages;
};

struct PhysicalBackingTexturePageSource {
    u64 physical_page{};
    VAddr guest_page{};
    u64 gpu_write_order{};
    u32 image_index{};

    auto operator<=>(const PhysicalBackingTexturePageSource&) const = default;
};

[[nodiscard]] inline std::optional<std::vector<PhysicalBackingTexturePageSource>>
PlanPhysicalBackingTexturePageSources(
    std::span<const PhysicalBackingTexturePageSource> oldest_to_newest_candidates) {
    constexpr u64 PageMask = 16_KB - 1;
    if (oldest_to_newest_candidates.empty()) {
        return std::nullopt;
    }
    std::unordered_map<u64, PhysicalBackingTexturePageSource> newest_sources;
    newest_sources.reserve(oldest_to_newest_candidates.size());
    for (const auto& candidate : oldest_to_newest_candidates) {
        if ((candidate.physical_page & PageMask) != 0 || (candidate.guest_page & PageMask) != 0) {
            return std::nullopt;
        }
        const auto existing = newest_sources.find(candidate.physical_page);
        if (existing == newest_sources.end()) {
            newest_sources.emplace(candidate.physical_page, candidate);
        } else if (candidate.gpu_write_order > existing->second.gpu_write_order) {
            existing->second = candidate;
        } else if (candidate.gpu_write_order == existing->second.gpu_write_order &&
                   (candidate.guest_page != existing->second.guest_page ||
                    candidate.image_index != existing->second.image_index)) {
            return std::nullopt;
        }
    }
    std::vector<PhysicalBackingTexturePageSource> sources;
    sources.reserve(newest_sources.size());
    for (const auto& [_, source] : newest_sources) {
        sources.push_back(source);
    }
    std::ranges::sort(sources, {}, &PhysicalBackingTexturePageSource::physical_page);
    return sources;
}

[[nodiscard]] constexpr std::optional<PhysicalBackingTextureBufferTransition>
PlanPhysicalBackingTextureOwnershipSpan(VAddr image_base, u64 image_size) noexcept {
    constexpr u64 PageSize = 16_KB;
    constexpr u64 PageMask = PageSize - 1;
    constexpr u64 Max = std::numeric_limits<u64>::max();
    if (image_size == 0 || image_base > Max - image_size) {
        return std::nullopt;
    }
    const VAddr aligned_base = image_base & ~PageMask;
    const VAddr image_end = image_base + image_size;
    if (image_end > Max - PageMask) {
        return std::nullopt;
    }
    const VAddr aligned_end = (image_end + PageMask) & ~PageMask;
    const u64 aligned_size = aligned_end - aligned_base;
    if (aligned_size == 0 || aligned_size > std::numeric_limits<u32>::max()) {
        return std::nullopt;
    }
    return PhysicalBackingTextureBufferTransition{
        .base = aligned_base,
        .size = static_cast<u32>(aligned_size),
    };
}

[[nodiscard]] inline std::optional<PhysicalBackingTextureOwnershipComponent>
PlanPhysicalBackingTextureOwnershipComponent(
    std::span<const PhysicalBackingTextureOwnershipRecord> records,
    std::span<const u64> seed_physical_pages) {
    constexpr u64 PageMask = 16_KB - 1;
    if (records.empty() || seed_physical_pages.empty()) {
        return std::nullopt;
    }

    std::unordered_set<u64> reachable_pages;
    reachable_pages.reserve(seed_physical_pages.size());
    for (const u64 physical_page : seed_physical_pages) {
        if ((physical_page & PageMask) != 0) {
            return std::nullopt;
        }
        reachable_pages.emplace(physical_page);
    }

    std::vector<const PhysicalBackingTextureOwnershipRecord*> selected;
    std::vector<bool> is_selected(records.size());
    std::unordered_set<u32> image_indices;
    for (const auto& record : records) {
        if (record.image_index == std::numeric_limits<u32>::max() ||
            !image_indices.emplace(record.image_index).second ||
            !PlanPhysicalBackingTextureOwnershipSpan(record.guest_base, record.guest_size)) {
            return std::nullopt;
        }
        for (const u64 physical_page : record.physical_pages) {
            if ((physical_page & PageMask) != 0) {
                return std::nullopt;
            }
        }
    }

    bool grew = true;
    while (grew) {
        grew = false;
        for (size_t index = 0; index < records.size(); ++index) {
            if (is_selected[index]) {
                continue;
            }
            const auto& record = records[index];
            if (!std::ranges::any_of(record.physical_pages, [&](u64 physical_page) {
                    return reachable_pages.contains(physical_page);
                })) {
                continue;
            }
            is_selected[index] = true;
            selected.push_back(&record);
            reachable_pages.insert(record.physical_pages.begin(), record.physical_pages.end());
            grew = true;
        }
    }
    if (selected.empty()) {
        return std::nullopt;
    }

    std::ranges::sort(selected, [](const auto* left, const auto* right) {
        return std::tie(left->gpu_write_order, left->image_index) <
               std::tie(right->gpu_write_order, right->image_index);
    });
    VAddr component_base = std::numeric_limits<VAddr>::max();
    VAddr component_end = 0;
    std::unordered_set<u64> component_physical_pages;
    PhysicalBackingTextureOwnershipComponent result;
    result.oldest_to_newest_images.reserve(selected.size());
    for (const auto* record : selected) {
        const auto span =
            PlanPhysicalBackingTextureOwnershipSpan(record->guest_base, record->guest_size);
        component_base = std::min(component_base, span->base);
        component_end = std::max(component_end, span->base + span->size);
        component_physical_pages.insert(record->physical_pages.begin(),
                                        record->physical_pages.end());
        result.oldest_to_newest_images.push_back({record->image_index, record->gpu_write_order});
    }
    const u64 component_size = component_end - component_base;
    if (component_size == 0 || component_size > std::numeric_limits<u32>::max()) {
        return std::nullopt;
    }
    result.ownership_span = {
        .base = component_base,
        .size = static_cast<u32>(component_size),
    };
    result.physical_pages.assign(component_physical_pages.begin(), component_physical_pages.end());
    std::ranges::sort(result.physical_pages);
    return result;
}

[[nodiscard]] constexpr std::optional<PhysicalBackingTextureBufferTransition>
PlanPhysicalBackingTextureBufferTransition(VAddr image_base, u64 image_size, VAddr write_address,
                                           u64 write_size) noexcept {
    constexpr u64 Max = std::numeric_limits<u64>::max();
    if (image_size == 0 || image_size > std::numeric_limits<u32>::max() || write_size == 0 ||
        image_base > Max - image_size || write_address < image_base ||
        write_address > Max - write_size || write_address + write_size > image_base + image_size) {
        return std::nullopt;
    }
    return PhysicalBackingTextureBufferTransition{
        .base = image_base,
        .size = static_cast<u32>(image_size),
    };
}

template <typename Left, typename Right>
[[nodiscard]] constexpr bool PhysicalBackingPagesIntersect(const Left& left,
                                                           const Right& right) noexcept {
    return std::ranges::any_of(
        left, [&](const auto& page) { return std::ranges::contains(right, page); });
}

[[nodiscard]] constexpr std::optional<PhysicalBackingAliasMigrationCopy>
PlanPhysicalBackingAliasMigrationCopy(VAddr source_base, u64 source_size, VAddr source_page,
                                      VAddr destination_base, u64 destination_size,
                                      VAddr destination_page) noexcept {
    constexpr u64 PageSize = 16_KB;
    constexpr u64 PageMask = PageSize - 1;
    constexpr u64 Max = std::numeric_limits<u64>::max();
    if ((source_page & PageMask) != 0 || (destination_page & PageMask) != 0 ||
        source_size < PageSize || destination_size < PageSize || source_page < source_base ||
        destination_page < destination_base || source_base > Max - (source_size - 1) ||
        destination_base > Max - (destination_size - 1)) {
        return std::nullopt;
    }
    const u64 source_offset = source_page - source_base;
    const u64 destination_offset = destination_page - destination_base;
    if (source_offset > source_size - PageSize ||
        destination_offset > destination_size - PageSize) {
        return std::nullopt;
    }
    return PhysicalBackingAliasMigrationCopy{
        .source_offset = source_offset,
        .destination_offset = destination_offset,
        .size = PageSize,
    };
}

enum class PhysicalBackingTextureConsumer {
    ComputeShaderRead,
    TransferRead,
};

[[nodiscard]] constexpr PhysicalBackingTextureConsumer PhysicalBackingTextureUploadConsumer(
    bool is_tiled) {
    return is_tiled ? PhysicalBackingTextureConsumer::ComputeShaderRead
                    : PhysicalBackingTextureConsumer::TransferRead;
}

enum class PhysicalBackingCommandResourceKind {
    Buffer,
    Texture,
};

struct PhysicalBackingCommandResource {
    PhysicalBackingCommandResourceKind kind{};
    u32 index{};

    auto operator<=>(const PhysicalBackingCommandResource&) const = default;
};

struct PhysicalBackingCommandAccess {
    PhysicalBackingCommandResource resource{};
    bool is_written{};
    std::vector<u64> physical_pages;
};

struct PhysicalBackingCommandPagePlan {
    u64 physical_page{};
    std::vector<PhysicalBackingCommandResource> readers;
    std::optional<PhysicalBackingCommandResource> writer;
};

struct PhysicalBackingCommandAliasPlan {
    std::vector<PhysicalBackingCommandResource> read_snapshot_order;
    std::vector<PhysicalBackingCommandResource> writer_prepare_order;
    std::vector<PhysicalBackingCommandResource> writer_finalize_order;
    std::vector<PhysicalBackingCommandPagePlan> pages;
};

struct PhysicalBackingCachePageOwnerLocation {
    u32 buffer_index{};
    u32 owner_index{};
    u64 physical_page{};

    auto operator<=>(const PhysicalBackingCachePageOwnerLocation&) const = default;
};

[[nodiscard]] inline std::optional<std::vector<PhysicalBackingCachePageOwnerLocation>>
PlanPhysicalBackingCachePageRetirements(
    std::span<const u64> requested_physical_pages,
    std::span<const PhysicalBackingCachePageOwnerLocation> owners) {
    std::vector<PhysicalBackingCachePageOwnerLocation> retirements(owners.begin(), owners.end());
    return retirements;
}

[[nodiscard]] inline std::optional<PhysicalBackingCommandAliasPlan>
PlanPhysicalBackingGpuCommandAliases(std::span<const PhysicalBackingCommandAccess> accesses) {
    constexpr u64 PageMask = 16_KB - 1;
    struct AggregatedAccess {
        PhysicalBackingCommandResource resource{};
        bool is_read{};
        bool is_written{};
        std::vector<u64> physical_pages;
    };

    std::vector<AggregatedAccess> aggregated;
    aggregated.reserve(accesses.size());
    for (const auto& access : accesses) {
        if (access.resource.index == std::numeric_limits<u32>::max() ||
            std::ranges::any_of(access.physical_pages,
                                [](u64 page) { return (page & PageMask) != 0; })) {
            return std::nullopt;
        }
        auto found = std::ranges::find(aggregated, access.resource, &AggregatedAccess::resource);
        if (found == aggregated.end()) {
            found = aggregated.emplace(aggregated.end(), AggregatedAccess{
                                                             .resource = access.resource,
                                                         });
        }
        // A writable GPU resource may still read or preserve bytes outside the actual write.
        // Snapshot every access before command-wide writer preparation.
        found->is_read = true;
        found->is_written |= access.is_written;
        found->physical_pages.insert(found->physical_pages.end(), access.physical_pages.begin(),
                                     access.physical_pages.end());
    }

    std::vector<u64> physical_pages;
    for (auto& access : aggregated) {
        std::ranges::sort(access.physical_pages);
        access.physical_pages.erase(std::ranges::unique(access.physical_pages).begin(),
                                    access.physical_pages.end());
        physical_pages.insert(physical_pages.end(), access.physical_pages.begin(),
                              access.physical_pages.end());
    }
    std::ranges::sort(physical_pages);
    physical_pages.erase(std::ranges::unique(physical_pages).begin(), physical_pages.end());

    PhysicalBackingCommandAliasPlan plan;
    std::unordered_set<u64> written_pages;
    plan.pages.reserve(physical_pages.size());
    for (const u64 physical_page : physical_pages) {
        PhysicalBackingCommandPagePlan page_plan{.physical_page = physical_page};
        for (const auto& access : aggregated) {
            if (!std::ranges::contains(access.physical_pages, physical_page)) {
                continue;
            }
            if (access.is_read) {
                page_plan.readers.push_back(access.resource);
            }
            if (access.is_written) {
                if (page_plan.writer && *page_plan.writer != access.resource) {
                    return std::nullopt;
                }
                page_plan.writer = access.resource;
            }
        }
        if (page_plan.writer) {
            written_pages.insert(physical_page);
        }
        plan.pages.push_back(std::move(page_plan));
    }

    for (const auto& access : aggregated) {
        if (access.physical_pages.empty()) {
            continue;
        }
        const bool touches_written_page =
            std::ranges::any_of(access.physical_pages, [&](u64 physical_page) {
                return written_pages.contains(physical_page);
            });
        if (access.is_read && touches_written_page) {
            plan.read_snapshot_order.push_back(access.resource);
        }
        if (access.is_written) {
            plan.writer_prepare_order.push_back(access.resource);
            plan.writer_finalize_order.push_back(access.resource);
        }
    }
    return plan;
}

template <typename Buffer, typename Synchronize, typename Publish>
void PublishDmaBufferAfterSynchronization(Buffer& buffer, Synchronize&& synchronize,
                                          Publish&& publish) {
    synchronize(buffer, buffer.CpuAddr(), static_cast<u32>(buffer.SizeBytes()));
    publish();
}

template <typename Touch>
void TouchBufferAfterUploadIfRegistered(bool is_registered, Touch&& touch) {
    if (is_registered) {
        touch();
    }
}

template <typename Request, typename DeviceAddress, typename GuestPage, typename Resolve>
void RefreshPhysicalBackingRegistrationAddresses(std::span<const Request> requests,
                                                 VAddr first_page,
                                                 std::span<DeviceAddress> addresses,
                                                 GuestPage&& guest_page, Resolve&& resolve) {
    for (const auto& request : requests) {
        const VAddr page = guest_page(request);
        if (page < first_page) {
            continue;
        }
        const u64 page_offset = page - first_page;
        if ((page_offset & (16_KB - 1)) != 0) {
            continue;
        }
        const u64 index = page_offset / 16_KB;
        if (index >= addresses.size()) {
            continue;
        }
        if (const auto current = resolve(page)) {
            addresses[index] = *current;
        }
    }
}

} // namespace VideoCore
