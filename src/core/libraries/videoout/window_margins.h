// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/videoout/videoout_error.h"

namespace Libraries::VideoOut {

struct WindowMargins {
    int top = 0;
    int bottom = 0;
};

constexpr int ApplyWindowModeMargins(WindowMargins* margins, int top, int bottom) {
    if (margins == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    margins->top = top;
    margins->bottom = bottom;
    return ORBIS_OK;
}

} // namespace Libraries::VideoOut
