// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::VideoOut {

enum class OrbisVideoOutEventId : s16 {
    Flip = 0,
    Vblank = 1,
    PreVblankStart = 2,
    SetMode = 8,
    Position = 12,
};

enum class OrbisVideoOutInternalEventId : s16 {
    Flip = 0x6,
    Vblank = 0x7,
    SetMode = 0x51,
    Position = 0x58,
    PreVblankStart = 0x59,
    SysVblank = 0x63,
};

} // namespace Libraries::VideoOut
