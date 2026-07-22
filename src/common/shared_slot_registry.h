// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <mutex>

#include "common/slot_vector.h"

namespace Common {

template <typename T>
class SharedSlotRegistry {
public:
    SlotId Insert(std::shared_ptr<T> object) {
        std::scoped_lock lk{mutex};
        return objects.insert(std::move(object));
    }

    [[nodiscard]] std::shared_ptr<T> Acquire(SlotId id) {
        std::scoped_lock lk{mutex};
        if (!objects.is_allocated(id)) {
            return {};
        }
        return objects[id];
    }

    [[nodiscard]] std::shared_ptr<T> Remove(SlotId id) {
        std::scoped_lock lk{mutex};
        if (!objects.is_allocated(id)) {
            return {};
        }
        auto object = std::move(objects[id]);
        objects.erase(id);
        return object;
    }

private:
    std::mutex mutex;
    SlotVector<std::shared_ptr<T>> objects;
};

} // namespace Common
