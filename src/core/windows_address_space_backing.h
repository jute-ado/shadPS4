// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifdef _WIN32

#include <memory>
#include "common/types.h"
#include "core/address_space_backing_lease.h"

namespace Core {

class WindowsAddressSpaceBackingApi {
public:
    virtual ~WindowsAddressSpaceBackingApi() = default;

    virtual void* CreateFileMapping(void* file, u32 desired_access, u32 page_protection,
                                    u32 allocation_attributes, u64 maximum_size) = 0;
    virtual u8* ReservePlaceholder(void* process, u64 size, u32 allocation_type,
                                   u32 protection) = 0;
    virtual u8* MapPlaceholder(void* mapping, void* process, u8* placeholder, u64 offset,
                               u64 size, u32 allocation_type, u32 protection) = 0;
    virtual bool UnmapPreservingPlaceholder(void* process, u8* base, u32 flags) = 0;
    virtual bool ReleasePlaceholder(void* process, u8* base, u64 size, u32 flags) = 0;
    virtual bool CloseMapping(void* mapping) = 0;
};

class WindowsAddressSpaceBacking final
    : public std::enable_shared_from_this<WindowsAddressSpaceBacking> {
public:
    [[nodiscard]] static std::shared_ptr<WindowsAddressSpaceBacking> Create(
        std::shared_ptr<WindowsAddressSpaceBackingApi> api, void* process, u64 size);
    ~WindowsAddressSpaceBacking();

    WindowsAddressSpaceBacking(const WindowsAddressSpaceBacking&) = delete;
    WindowsAddressSpaceBacking& operator=(const WindowsAddressSpaceBacking&) = delete;

    [[nodiscard]] AddressSpaceBackingLease AcquireLease() const noexcept;

    [[nodiscard]] u8* Base() const noexcept {
        return base;
    }

    [[nodiscard]] u64 Size() const noexcept {
        return size;
    }

    // Internal Core mapping operations still require the native mapping object.
    [[nodiscard]] void* Mapping() const noexcept {
        return mapping;
    }

private:
    WindowsAddressSpaceBacking(std::shared_ptr<WindowsAddressSpaceBackingApi> api_, void* process_,
                               void* mapping_, u8* base_, u64 size_)
        : api{std::move(api_)}, process{process_}, mapping{mapping_}, base{base_}, size{size_} {}

    std::shared_ptr<WindowsAddressSpaceBackingApi> api;
    void* process{};
    void* mapping{};
    u8* base{};
    u64 size{};
};

} // namespace Core

#endif
