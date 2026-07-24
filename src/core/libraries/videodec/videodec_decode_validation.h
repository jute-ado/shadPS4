// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/error_codes.h"
#include "videodec.h"
#include "videodec_error.h"

namespace Libraries::Videodec {

constexpr s32 ValidateFrameBufferStorage(const OrbisVideodecFrameBuffer& frame_buffer) {
    if (frame_buffer.frameBufferSize == 0) {
        return ORBIS_VIDEODEC_ERROR_FRAME_BUFFER_SIZE;
    }
    if (!frame_buffer.pFrameBuffer) {
        return ORBIS_VIDEODEC_ERROR_FRAME_BUFFER_POINTER;
    }
    return ORBIS_OK;
}

constexpr s32 ValidateFrameBufferCapacity(u64 supplied_size, u64 required_size) {
    return supplied_size < required_size ? ORBIS_VIDEODEC_ERROR_FRAME_BUFFER_SIZE : ORBIS_OK;
}

constexpr s32 ValidateDecodeArguments(const OrbisVideodecInputData& input_data,
                                      const OrbisVideodecFrameBuffer& frame_buffer,
                                      const OrbisVideodecPictureInfo& picture_info) {
    if (input_data.thisSize != sizeof(OrbisVideodecInputData) ||
        frame_buffer.thisSize != sizeof(OrbisVideodecFrameBuffer) ||
        picture_info.thisSize != sizeof(OrbisVideodecPictureInfo)) {
        return ORBIS_VIDEODEC_ERROR_STRUCT_SIZE;
    }
    if (!input_data.pAuData) {
        return ORBIS_VIDEODEC_ERROR_AU_POINTER;
    }
    if (input_data.auSize == 0) {
        return ORBIS_VIDEODEC_ERROR_AU_SIZE;
    }
    return ValidateFrameBufferStorage(frame_buffer);
}

} // namespace Libraries::Videodec
