// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "input/controller_connection.h"

TEST(ControllerConnectionState, StartsDisconnected) {
    const Input::ControllerConnectionState connection;

    EXPECT_FALSE(connection.IsConnected());
    EXPECT_EQ(connection.ConnectedCount(), 0);
}

TEST(ControllerConnectionState, VirtualInputConnectsAHeadlessController) {
    Input::ControllerConnectionState connection;

    connection.ConnectVirtual();

    EXPECT_TRUE(connection.IsConnected());
    EXPECT_EQ(connection.ConnectedCount(), 1);
}

TEST(ControllerConnectionState, PhysicalDisconnectPreservesVirtualConnection) {
    Input::ControllerConnectionState connection;
    connection.ConnectVirtual();
    connection.ConnectPhysical();

    connection.DisconnectPhysical();

    EXPECT_TRUE(connection.IsConnected());
    EXPECT_EQ(connection.ConnectedCount(), 1);
}

TEST(ControllerConnectionState, PhysicalControllerCanDisconnectNormally) {
    Input::ControllerConnectionState connection;
    connection.ConnectPhysical();

    connection.DisconnectPhysical();

    EXPECT_FALSE(connection.IsConnected());
    EXPECT_EQ(connection.ConnectedCount(), 0);
}
