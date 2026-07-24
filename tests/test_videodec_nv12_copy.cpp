// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "common/alignment.h"
#include "core/libraries/videodec/video_utils.h"

extern "C" {
#include <libavutil/frame.h>
}

TEST(VideodecNv12Copy, HonorsSourceStrideWhenWidthIsDestinationAligned) {
    constexpr u32 width = 64;
    constexpr u32 height = 2;
    constexpr u32 source_stride = 80;
    constexpr u32 destination_height = Common::AlignUp(height, 16u);

    std::vector<u8> luma(source_stride * height, 0xee);
    std::ranges::fill_n(luma.begin(), width, 0x11);
    std::ranges::fill_n(luma.begin() + source_stride, width, 0x22);

    std::vector<u8> chroma(source_stride, 0xee);
    std::ranges::fill_n(chroma.begin(), width, 0x33);

    AVFrame frame{};
    frame.width = width;
    frame.height = height;
    frame.data[0] = luma.data();
    frame.data[1] = chroma.data();
    frame.linesize[0] = source_stride;
    frame.linesize[1] = source_stride;

    std::vector<u8> destination(width * destination_height * 3 / 2, 0);
    Libraries::Videodec::CopyNV12Data(destination.data(), frame);

    EXPECT_TRUE(std::ranges::all_of(
        std::span{destination}.subspan(0, width), [](u8 value) { return value == 0x11; }));
    EXPECT_TRUE(std::ranges::all_of(
        std::span{destination}.subspan(width, width), [](u8 value) { return value == 0x22; }));
}
