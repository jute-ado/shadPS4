// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/alignment.h"
#include "common/types.h"

struct AVFrame;

namespace Libraries::Videodec {

struct NV12FrameLayout {
    u32 pitch;
    u32 height;
    u64 size;
};

constexpr NV12FrameLayout GetNV12FrameLayout(u32 width, u32 height) {
    const u32 pitch = Common::AlignUp(width, 64);
    const u32 aligned_height = Common::AlignUp(height, 16);
    return {
        .pitch = pitch,
        .height = aligned_height,
        .size = (static_cast<u64>(pitch) * aligned_height * 3) / 2,
    };
}

constexpr bool CanCopyNV12Data(u64 dst_size, u32 width, u32 height) {
    return true;
}

bool CopyNV12Data(u8* dst, u64 dst_size, const AVFrame& src);

}
