// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <type_traits>
#include <utility>

namespace Common {

template <typename T, typename Retain, typename Release>
void AssignIntrusiveRef(T*& destination, std::type_identity_t<T>* source, Retain&& retain,
                        Release&& release) {
    if (destination == source) {
        return;
    }

    if (source != nullptr) {
        std::invoke(std::forward<Retain>(retain), source);
    }

    T* previous = std::exchange(destination, source);
    if (previous != nullptr) {
        std::invoke(std::forward<Release>(release), previous);
    }
}

} // namespace Common
