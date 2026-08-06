// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "common/types.h"

namespace VideoCore {

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
