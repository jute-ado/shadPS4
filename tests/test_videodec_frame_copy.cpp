// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <vector>

#include <gtest/gtest.h>
#include <libavutil/frame.h>

#include "core/libraries/videodec/video_utils.h"

namespace {

using Libraries::Videodec::CopyNV12Data;

TEST(VideodecFrameCopy, InitializesAlignedNv12CropPadding) {
    constexpr int Width = 66;
    constexpr int Height = 18;
    constexpr int SourcePitch = 72;
    constexpr int DestinationPitch = 128;
    constexpr int DestinationHeight = 32;

    std::vector<u8> luma(SourcePitch * Height);
    std::vector<u8> chroma(SourcePitch * (Height / 2));
    for (int y = 0; y < Height; ++y) {
        std::fill_n(luma.data() + y * SourcePitch, Width, static_cast<u8>(0x20 + y));
    }
    for (int y = 0; y < Height / 2; ++y) {
        for (int x = 0; x < Width; x += 2) {
            chroma[y * SourcePitch + x] = static_cast<u8>(0x40 + y);
            chroma[y * SourcePitch + x + 1] = static_cast<u8>(0x80 + y);
        }
    }

    AVFrame frame{};
    frame.width = Width;
    frame.height = Height;
    frame.data[0] = luma.data();
    frame.data[1] = chroma.data();
    frame.linesize[0] = SourcePitch;
    frame.linesize[1] = SourcePitch;

    std::vector<u8> destination(DestinationPitch * DestinationHeight * 3 / 2, 0xcd);
    CopyNV12Data(destination.data(), frame);

    for (int y = 0; y < DestinationHeight; ++y) {
        const auto expected = static_cast<u8>(0x20 + std::min(y, Height - 1));
        EXPECT_TRUE(std::all_of(destination.begin() + y * DestinationPitch,
                                destination.begin() + (y + 1) * DestinationPitch,
                                [expected](u8 value) { return value == expected; }))
            << "luma row " << y << " was not fully initialized";
    }

    const auto chroma_offset = DestinationPitch * DestinationHeight;
    for (int y = 0; y < DestinationHeight / 2; ++y) {
        const auto source_y = std::min(y, Height / 2 - 1);
        const auto expected_u = static_cast<u8>(0x40 + source_y);
        const auto expected_v = static_cast<u8>(0x80 + source_y);
        for (int x = 0; x < DestinationPitch; x += 2) {
            EXPECT_EQ(destination[chroma_offset + y * DestinationPitch + x], expected_u)
                << "chroma U at " << x << ", " << y << " was not initialized";
            EXPECT_EQ(destination[chroma_offset + y * DestinationPitch + x + 1], expected_v)
                << "chroma V at " << x << ", " << y << " was not initialized";
        }
    }
}

} // namespace
