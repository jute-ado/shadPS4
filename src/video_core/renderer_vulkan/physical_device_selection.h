// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>

namespace Vulkan {

enum class PhysicalDeviceClass {
    Other,
    Integrated,
    Discrete,
    Cpu,
};

struct PhysicalDeviceSelectionTraits {
    bool supports_required_api{};
    PhysicalDeviceClass device_class{};
    std::size_t largest_device_local_heap{};
};

[[nodiscard]] constexpr bool PreferPhysicalDevice(const PhysicalDeviceSelectionTraits& left,
                                                  const PhysicalDeviceSelectionTraits& right) {
    if (left.supports_required_api != right.supports_required_api) {
        return left.supports_required_api;
    }

    const bool left_is_discrete = left.device_class == PhysicalDeviceClass::Discrete;
    const bool right_is_discrete = right.device_class == PhysicalDeviceClass::Discrete;
    if (left_is_discrete != right_is_discrete) {
        return left_is_discrete;
    }

    const bool left_is_cpu = left.device_class == PhysicalDeviceClass::Cpu;
    const bool right_is_cpu = right.device_class == PhysicalDeviceClass::Cpu;
    if (left_is_cpu != right_is_cpu) {
        return right_is_cpu;
    }

    return left.largest_device_local_heap > right.largest_device_local_heap;
}

} // namespace Vulkan
