// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>

#include "core/address_space_backing_lease.h"
#include "video_core/renderer_vulkan/external_address_space_backing_import.h"
#include "video_core/renderer_vulkan/vk_platform.h"

namespace Vulkan {

class Instance;

/**
 * Imports the canonical AddressSpace backing as one retained Vulkan BDA buffer.
 * This owner intentionally does not publish any guest-page address mappings.
 */
class ExternalAddressSpaceBacking {
public:
    ExternalAddressSpaceBacking(const Instance& instance, Core::AddressSpaceBackingLease lease);
    ~ExternalAddressSpaceBacking() = default;

    ExternalAddressSpaceBacking(const ExternalAddressSpaceBacking&) = delete;
    ExternalAddressSpaceBacking& operator=(const ExternalAddressSpaceBacking&) = delete;

    [[nodiscard]] bool IsAvailable() const noexcept {
        return resources.has_value() && device_address != 0;
    }

    [[nodiscard]] vk::DeviceAddress DeviceAddress() const noexcept {
        return device_address;
    }

    [[nodiscard]] u64 BackingSize() const noexcept {
        return resources ? resources->lease.Size() : 0;
    }

    [[nodiscard]] static constexpr u64 GuestPagePublicationCount() noexcept {
        return 0;
    }

private:
    using Resources = ExternalAddressSpaceImportResources<Core::AddressSpaceBackingLease,
                                                          vk::UniqueDeviceMemory, vk::UniqueBuffer>;

    // Resources destroys its members in the required buffer -> memory -> opaque lease order.
    std::optional<Resources> resources;
    vk::DeviceAddress device_address{};
};

} // namespace Vulkan
