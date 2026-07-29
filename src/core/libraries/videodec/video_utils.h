// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstring>
#include <span>

#include "common/alignment.h"
#include "common/types.h"

struct AVFrame;

namespace Libraries::Videodec {

struct NV12FrameLayout {
    u32 pitch;
    u32 height;
    u64 size;
};

struct NV12SourceView {
    u32 width;
    u32 height;
    const u8* luma;
    u32 luma_pitch;
    const u8* chroma;
    u32 chroma_pitch;
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
    return dst_size >= GetNV12FrameLayout(width, height).size;
}

inline bool PackNV12Data(std::span<u8> dst, const NV12SourceView& src) {
    if (!src.luma || !src.chroma || src.width == 0 || src.height == 0 ||
        src.luma_pitch < src.width || src.chroma_pitch < src.width) {
        return false;
    }

    const auto layout = GetNV12FrameLayout(src.width, src.height);
    if (dst.size() < layout.size) {
        return false;
    }

    auto* luma_dst = dst.data();
    auto* chroma_dst = dst.data() + static_cast<u64>(layout.pitch) * layout.height;

    if (src.width == layout.pitch && src.luma_pitch == src.width && src.chroma_pitch == src.width) {
        std::memcpy(luma_dst, src.luma, static_cast<u64>(src.width) * src.height);
        std::memcpy(chroma_dst, src.chroma, (static_cast<u64>(src.width) * src.height) / 2);
    } else {
        for (u32 y = 0; y < src.height; ++y) {
            std::memcpy(luma_dst + static_cast<u64>(y) * layout.pitch,
                        src.luma + static_cast<u64>(y) * src.luma_pitch, src.width);
        }
        for (u32 y = 0; y < src.height / 2; ++y) {
            std::memcpy(chroma_dst + static_cast<u64>(y) * layout.pitch,
                        src.chroma + static_cast<u64>(y) * src.chroma_pitch, src.width);
        }
    }

    if (src.height != layout.height) {
        const auto* last_luma = src.luma + static_cast<u64>(src.height - 1) * src.luma_pitch;
        for (u32 y = src.height; y < layout.height; ++y) {
            std::memcpy(luma_dst + static_cast<u64>(y) * layout.pitch, last_luma, src.width);
        }

        const auto chroma_height = src.height / 2;
        if (chroma_height != 0) {
            const auto* last_chroma =
                src.chroma + static_cast<u64>(chroma_height - 1) * src.chroma_pitch;
            for (u32 y = chroma_height; y < layout.height / 2; ++y) {
                std::memcpy(chroma_dst + static_cast<u64>(y) * layout.pitch, last_chroma,
                            src.width);
            }
        }
    }
    return true;
}

bool CopyNV12Data(u8* dst, u64 dst_size, const AVFrame& src);

} // namespace Libraries::Videodec
