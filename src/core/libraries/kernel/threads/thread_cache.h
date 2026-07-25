// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <list>
#include <mutex>

namespace Libraries::Kernel {

constexpr bool CanAllocateThread(const std::size_t current, const std::size_t maximum) {
    return current < maximum;
}

template <typename T, std::size_t Capacity>
class BoundedThreadCache {
public:
    bool TryPush(T* value) {
        std::scoped_lock lk{mutex};
        if (entries.size() >= Capacity) {
            return false;
        }
        entries.push_back(value);
        return true;
    }

    T* TryPop() {
        std::scoped_lock lk{mutex};
        if (entries.empty()) {
            return nullptr;
        }
        T* value = entries.back();
        entries.pop_back();
        return value;
    }

private:
    std::mutex mutex;
    std::list<T*> entries;
};

} // namespace Libraries::Kernel
