// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Input {

class ControllerConnectionState {
public:
    void ConnectPhysical() {
        physical_connected = true;
    }

    void DisconnectPhysical() {
        physical_connected = false;
    }

    void ConnectVirtual() {
        virtual_connected = true;
    }

    [[nodiscard]] bool IsConnected() const {
        return physical_connected || virtual_connected;
    }

    [[nodiscard]] int ConnectedCount() const {
        return IsConnected() ? 1 : 0;
    }

private:
    bool physical_connected = false;
    bool virtual_connected = false;
};

} // namespace Input
