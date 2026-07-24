// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>

extern "C" {
#include <libavutil/frame.h>
}

namespace Libraries::Videodec {

struct FrameDeleter {
    void operator()(AVFrame* frame) const {
        av_frame_free(&frame);
    }
};

using FrameHandle = std::unique_ptr<AVFrame, FrameDeleter>;

inline FrameHandle AdoptFrame(AVFrame* frame) {
    return FrameHandle{frame};
}

} // namespace Libraries::Videodec
