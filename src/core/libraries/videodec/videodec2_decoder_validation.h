// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/error_codes.h"
#include "videodec2.h"
#include "videodec_error.h"

namespace Libraries::Videodec2 {

constexpr s32 ValidateDecoderConfig(const OrbisVideodec2DecoderConfigInfo& config) {
    // The current decoder implementation only supports AVC.
    if (config.codecType != 1) {
        return ORBIS_VIDEODEC2_ERROR_CODEC_TYPE;
    }
    return ORBIS_OK;
}

} // namespace Libraries::Videodec2
