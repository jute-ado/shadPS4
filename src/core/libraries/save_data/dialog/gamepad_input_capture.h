// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

namespace Libraries::SaveData::Dialog {

class GamepadInputCapture {
public:
    using Callback = void (*)();

    GamepadInputCapture() = default;

    GamepadInputCapture(Callback acquire, Callback release) : release{release} {
        if (acquire != nullptr) {
            acquire();
        }
    }

    ~GamepadInputCapture() {
        Reset();
    }

    GamepadInputCapture(const GamepadInputCapture&) = delete;
    GamepadInputCapture& operator=(const GamepadInputCapture&) = delete;

    GamepadInputCapture(GamepadInputCapture&& other) noexcept
        : release{std::exchange(other.release, nullptr)} {}

    GamepadInputCapture& operator=(GamepadInputCapture&& other) noexcept {
        if (this != &other) {
            Reset();
            release = std::exchange(other.release, nullptr);
        }
        return *this;
    }

    void Reset() {
        if (auto callback = std::exchange(release, nullptr); callback != nullptr) {
            callback();
        }
    }

private:
    Callback release{};
};

} // namespace Libraries::SaveData::Dialog
