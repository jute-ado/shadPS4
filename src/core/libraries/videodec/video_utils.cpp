// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_utils.h"

#include "common/alignment.h"
#include "core/memory.h"

#include <libavutil/frame.h>

#include <cstring>

namespace Libraries::Videodec {

namespace {

void CopyNV12DataUnchecked(u8* dst, const AVFrame& src, const NV12FrameLayout& layout) {
    const auto dst_pitch = layout.pitch;
    const auto dst_height = layout.height;

    const auto luma_dst = dst;
    const auto chroma_dst = dst + dst_pitch * dst_height;

    if (src.width != dst_pitch) {
        for (u32 y = 0; y < src.height; ++y) {
            std::memcpy(luma_dst + y * dst_pitch, src.data[0] + y * src.linesize[0], src.width);
        }
        for (u32 y = 0; y < src.height / 2; ++y) {
            std::memcpy(chroma_dst + y * dst_pitch, src.data[1] + y * src.linesize[1], src.width);
        }
    } else {
        std::memcpy(luma_dst, src.data[0], src.width * src.height);
        std::memcpy(chroma_dst, src.data[1], (src.width * src.height) / 2);
    }

    if (src.height != dst_height) {
        // Extend the data vertically to the crop space
        const auto ly = src.height - 1;
        for (u32 y = src.height; y < dst_height; ++y) {
            std::memcpy(luma_dst + y * dst_pitch, src.data[0] + ly * src.linesize[0], src.width);
        }
        const auto cy = (src.height / 2) - 1;
        for (u32 y = src.height / 2; y < dst_height / 2; ++y) {
            std::memcpy(chroma_dst + y * dst_pitch, src.data[1] + cy * src.linesize[1], src.width);
        }
    }
}

} // namespace

bool CopyNV12Data(u8* dst, u64 dst_size, const AVFrame& src) {
    if (!dst || src.width <= 0 || src.height <= 0) {
        return false;
    }

    const auto width = static_cast<u32>(src.width);
    const auto height = static_cast<u32>(src.height);
    if (!CanCopyNV12Data(dst_size, width, height)) {
        return false;
    }

    const auto layout = GetNV12FrameLayout(width, height);
    auto* memory = Core::Memory::Instance();
    return memory &&
           memory->WithAccessibleRange(reinterpret_cast<VAddr>(dst), layout.size,
                                       Core::MemoryProt::CpuWrite,
                                       [&] { CopyNV12DataUnchecked(dst, src, layout); });
}

} // namespace Libraries::Videodec
