// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_utils.h"

#include "core/memory.h"

#include <libavutil/frame.h>

#include <vector>

namespace Libraries::Videodec {

bool CopyNV12Data(u8* dst, u64 dst_size, const AVFrame& src) {
    if (!dst || src.width <= 0 || src.height <= 0 || src.linesize[0] <= 0 || src.linesize[1] <= 0) {
        return false;
    }

    const auto width = static_cast<u32>(src.width);
    const auto height = static_cast<u32>(src.height);
    if (!CanCopyNV12Data(dst_size, width, height)) {
        return false;
    }

    const auto layout = GetNV12FrameLayout(width, height);
    thread_local std::vector<u8> staging;
    staging.resize(layout.size);
    if (!PackNV12Data(staging, NV12SourceView{
                                   .width = width,
                                   .height = height,
                                   .luma = src.data[0],
                                   .luma_pitch = static_cast<u32>(src.linesize[0]),
                                   .chroma = src.data[1],
                                   .chroma_pitch = static_cast<u32>(src.linesize[1]),
                               })) {
        return false;
    }

    auto* memory = Core::Memory::Instance();
    return memory && memory->TryWriteHostMemory(dst, staging.data(), layout.size);
}

} // namespace Libraries::Videodec
