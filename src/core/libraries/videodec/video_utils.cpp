// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_utils.h"

#include "common/alignment.h"

#include <libavutil/frame.h>

#include <algorithm>
#include <cstring>

namespace Libraries::Videodec {

void CopyNV12Data(u8* dst, const AVFrame& src) {
    if (src.width <= 0 || src.height <= 0) {
        return;
    }

    const auto dst_pitch = Common::AlignUp<u32>(src.width, 64);
    const auto dst_height = Common::AlignUp<u32>(src.height, 16);

    const auto luma_dst = dst;
    const auto chroma_dst = dst + dst_pitch * dst_height;

    for (u32 y = 0; y < dst_height; ++y) {
        const auto source_y = std::min<u32>(y, src.height - 1);
        const auto source = src.data[0] + source_y * src.linesize[0];
        const auto destination = luma_dst + y * dst_pitch;
        std::memcpy(destination, source, src.width);
        std::fill(destination + src.width, destination + dst_pitch, source[src.width - 1]);
    }

    const auto source_chroma_height = std::max(src.height / 2, 1);
    for (u32 y = 0; y < dst_height / 2; ++y) {
        const auto source_y = std::min<u32>(y, source_chroma_height - 1);
        const auto source = src.data[1] + source_y * src.linesize[1];
        const auto destination = chroma_dst + y * dst_pitch;
        std::memcpy(destination, source, src.width);
        for (u32 x = src.width; x < dst_pitch; x += 2) {
            destination[x] = source[src.width - 2];
            destination[x + 1] = source[src.width - 1];
        }
    }
}

} // namespace Libraries::Videodec
