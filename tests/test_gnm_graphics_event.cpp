// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/gnmdriver/graphics_event.h"

TEST(GnmGraphicsEvent, RegistrationUsesClearSemantics) {
    constexpr u64 event_id = 0x40;
    auto* const user_data = reinterpret_cast<void*>(0x1234);

    const auto event = Libraries::GnmDriver::MakeGraphicsEvent(event_id, user_data);

    EXPECT_EQ(event.event.ident, event_id);
    EXPECT_EQ(event.event.filter, Libraries::Kernel::OrbisKernelEvent::Filter::GraphicsCore);
    EXPECT_EQ(event.event.flags, Libraries::Kernel::OrbisKernelEvent::Flags::Add |
                                     Libraries::Kernel::OrbisKernelEvent::Flags::Clear);
    EXPECT_EQ(event.event.udata, user_data);
}

TEST(GnmGraphicsEvent, RepeatedTriggersCoalesceAndCountOccurrencesUntilClear) {
    auto event = Libraries::GnmDriver::MakeGraphicsEvent(0x40, nullptr);

    event.Trigger(reinterpret_cast<void*>(1));
    event.Trigger(reinterpret_cast<void*>(2));
    event.Trigger(reinterpret_cast<void*>(3));

    EXPECT_TRUE(event.IsTriggered());
    EXPECT_EQ(event.event.fflags, 3u);
    EXPECT_EQ(event.event.data, 3u);

    event.Clear();
    EXPECT_FALSE(event.IsTriggered());
    EXPECT_EQ(event.event.fflags, 0u);
    EXPECT_EQ(event.event.data, 0u);

    event.Trigger(reinterpret_cast<void*>(4));
    EXPECT_TRUE(event.IsTriggered());
    EXPECT_EQ(event.event.fflags, 1u);
    EXPECT_EQ(event.event.data, 4u);
}
