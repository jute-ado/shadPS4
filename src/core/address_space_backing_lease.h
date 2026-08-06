// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <utility>
#include "common/types.h"

namespace Core {

class WindowsAddressSpaceBacking;

/**
 * Keeps the canonical host backing alive without exposing its platform handle.
 */
class AddressSpaceBackingLease {
public:
    AddressSpaceBackingLease() = default;

    [[nodiscard]] u8* Base() const noexcept {
        return base;
    }

    [[nodiscard]] u64 Size() const noexcept {
        return size;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return lifetime != nullptr;
    }

private:
    friend class WindowsAddressSpaceBacking;

    AddressSpaceBackingLease(std::shared_ptr<const void> lifetime_, u8* base_, u64 size_)
        : lifetime{std::move(lifetime_)}, base{base_}, size{size_} {}

    std::shared_ptr<const void> lifetime;
    u8* base{};
    u64 size{};
};

} // namespace Core
