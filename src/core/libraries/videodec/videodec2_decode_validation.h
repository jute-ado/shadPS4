// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/error_codes.h"
#include "videodec2.h"
#include "videodec_error.h"

namespace Libraries::Videodec2 {

constexpr s32 ValidateFrameBufferStorage(const OrbisVideodec2FrameBuffer& frame_buffer) {
    if (frame_buffer.frameBufferSize == 0) {
        return ORBIS_VIDEODEC2_ERROR_FRAME_BUFFER_SIZE;
    }
    if (!frame_buffer.frameBuffer) {
        return ORBIS_VIDEODEC2_ERROR_FRAME_BUFFER_POINTER;
    }
    return ORBIS_OK;
}

constexpr s32 ValidateFrameBufferCapacity(u64 supplied_size, u64 required_size) {
    return supplied_size < required_size ? ORBIS_VIDEODEC2_ERROR_FRAME_BUFFER_SIZE : ORBIS_OK;
}

constexpr s32 ValidateDecodeArguments(const OrbisVideodec2InputData& input_data,
                                      const OrbisVideodec2FrameBuffer& frame_buffer,
                                      const OrbisVideodec2OutputInfo& output_info) {
    if (input_data.thisSize != sizeof(OrbisVideodec2InputData) ||
        frame_buffer.thisSize != sizeof(OrbisVideodec2FrameBuffer) ||
        (output_info.thisSize | 8) != sizeof(OrbisVideodec2OutputInfo)) {
        return ORBIS_VIDEODEC2_ERROR_STRUCT_SIZE;
    }
    if (!input_data.auData) {
        return ORBIS_VIDEODEC2_ERROR_ACCESS_UNIT_POINTER;
    }
    if (input_data.auSize == 0) {
        return ORBIS_VIDEODEC2_ERROR_ACCESS_UNIT_SIZE;
    }
    return ValidateFrameBufferStorage(frame_buffer);
}

} // namespace Libraries::Videodec2
