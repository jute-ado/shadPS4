// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <deque>
#include <map>
#include <mutex>
#include <optional>

#include "common/types.h"

namespace VideoCore {

struct DeviceAddressRecord {
    u64 device_address{};
    u64 size{};
    VAddr guest_address{};
    u64 buffer_handle{};
};

struct DeviceAddressMatch {
    DeviceAddressRecord record{};
    u64 offset{};
    bool active{};
};

/// Keeps enough host-address provenance to make VK_EXT_device_fault reports actionable.
/// Queries happen only after device loss; registration follows buffer allocation lifetime.
class DeviceAddressRegistry {
public:
    void Register(u64 device_address, u64 size, VAddr guest_address, u64 buffer_handle) {
        if (device_address == 0 || size == 0) {
            return;
        }
        const std::scoped_lock lock{mutex};
        active.insert_or_assign(device_address,
                                DeviceAddressRecord{device_address, size, guest_address,
                                                    buffer_handle});
    }

    void Retire(u64 device_address) {
        if (device_address == 0) {
            return;
        }
        const std::scoped_lock lock{mutex};
        const auto entry = active.find(device_address);
        if (entry == active.end()) {
            return;
        }
        retired.push_back(entry->second);
        active.erase(entry);
        if (retired.size() > MaxRetiredRecords) {
            retired.pop_front();
        }
    }

    [[nodiscard]] std::optional<DeviceAddressMatch> Find(u64 address) const {
        const std::scoped_lock lock{mutex};
        if (const auto match = FindContainingActive(address)) {
            return match;
        }
        for (auto entry = retired.rbegin(); entry != retired.rend(); ++entry) {
            if (Contains(*entry, address)) {
                return DeviceAddressMatch{*entry, address - entry->device_address, false};
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<DeviceAddressMatch> FindNearestPreceding(u64 address) const {
        const std::scoped_lock lock{mutex};
        std::optional<DeviceAddressMatch> nearest;
        const auto consider = [&](const DeviceAddressRecord& record, bool is_active) {
            if (record.device_address > address) {
                return;
            }
            if (!nearest || record.device_address > nearest->record.device_address ||
                (record.device_address == nearest->record.device_address && is_active)) {
                nearest = DeviceAddressMatch{record, address - record.device_address, is_active};
            }
        };
        if (const auto upper = active.upper_bound(address); upper != active.begin()) {
            consider(std::prev(upper)->second, true);
        }
        for (auto entry = retired.rbegin(); entry != retired.rend(); ++entry) {
            consider(*entry, false);
        }
        return nearest;
    }

    /// Device-fault vendors sometimes report the shader's guest address rather than the resolved
    /// physical buffer address. Keep that interpretation separate so diagnostics cannot confuse
    /// it with a confirmed host-BDA match.
    [[nodiscard]] std::optional<DeviceAddressMatch> FindGuest(VAddr address) const {
        const std::scoped_lock lock{mutex};
        for (const auto& [device_address, record] : active) {
            if (ContainsGuest(record, address)) {
                return DeviceAddressMatch{record, address - record.guest_address, true};
            }
        }
        for (auto entry = retired.rbegin(); entry != retired.rend(); ++entry) {
            if (ContainsGuest(*entry, address)) {
                return DeviceAddressMatch{*entry, address - entry->guest_address, false};
            }
        }
        return std::nullopt;
    }

private:
    static constexpr size_t MaxRetiredRecords = 2048;

    [[nodiscard]] static bool Contains(const DeviceAddressRecord& record, u64 address) {
        return address >= record.device_address && address - record.device_address < record.size;
    }

    [[nodiscard]] static bool ContainsGuest(const DeviceAddressRecord& record, VAddr address) {
        return record.guest_address != 0 && address >= record.guest_address &&
               address - record.guest_address < record.size;
    }

    [[nodiscard]] std::optional<DeviceAddressMatch> FindContainingActive(u64 address) const {
        const auto upper = active.upper_bound(address);
        if (upper == active.begin()) {
            return std::nullopt;
        }
        const auto& record = std::prev(upper)->second;
        if (!Contains(record, address)) {
            return std::nullopt;
        }
        return DeviceAddressMatch{record, address - record.device_address, true};
    }

    mutable std::mutex mutex;
    std::map<u64, DeviceAddressRecord> active;
    std::deque<DeviceAddressRecord> retired;
};

inline DeviceAddressRegistry& GlobalDeviceAddressRegistry() {
    static DeviceAddressRegistry registry;
    return registry;
}

} // namespace VideoCore
