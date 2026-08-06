// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <optional>
#include <span>

#include "common/types.h"

namespace VideoCore {

[[nodiscard]] constexpr bool ShouldAcquirePhysicalBackingBufferOwnership(
    bool will_gpu_write) noexcept {
    return will_gpu_write;
}

struct PhysicalBackingAliasMigrationCopy {
    u64 source_offset{};
    u64 destination_offset{};
    u64 size{};
};

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
