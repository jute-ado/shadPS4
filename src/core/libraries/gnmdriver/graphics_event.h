// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/kernel/equeue.h"

namespace Libraries::GnmDriver {

inline Kernel::EqueueEvent MakeGraphicsEvent(u64 id, void* user_data) {
    Kernel::EqueueEvent event{};
    event.event.ident = id;
    event.event.filter = Kernel::OrbisKernelEvent::Filter::GraphicsCore;
    event.event.flags =
        Kernel::OrbisKernelEvent::Flags::Add | Kernel::OrbisKernelEvent::Flags::Clear;
    event.event.fflags = 0;
    event.event.data = id;
    event.event.udata = user_data;
    return event;
}

} // namespace Libraries::GnmDriver
