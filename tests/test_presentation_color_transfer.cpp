// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/presentation_color_transfer.h"

using Libraries::VideoOut::PixelFormat;
using VideoCore::PresentationInputTransfer;

TEST(PresentationColorTransfer, DecodesTenBitSrgbBecauseVulkanHasNoSrgbView) {
    EXPECT_EQ(VideoCore::GetPresentationInputTransfer(PixelFormat::A2R10G10B10Srgb),
              PresentationInputTransfer::SrgbEncodedUnorm);
}

TEST(PresentationColorTransfer, LeavesLinearAndAutomaticallyDecodedFormatsAlone) {
    EXPECT_EQ(VideoCore::GetPresentationInputTransfer(PixelFormat::A2R10G10B10),
              PresentationInputTransfer::Linear);
    EXPECT_EQ(VideoCore::GetPresentationInputTransfer(PixelFormat::A8R8G8B8Srgb),
              PresentationInputTransfer::Linear);
    EXPECT_EQ(VideoCore::GetPresentationInputTransfer(PixelFormat::A8B8G8R8Srgb),
              PresentationInputTransfer::Linear);
}

TEST(PresentationColorTransfer, LeavesPqForTheHdrPresentationPath) {
    EXPECT_EQ(VideoCore::GetPresentationInputTransfer(PixelFormat::A2R10G10B10Bt2020Pq),
              PresentationInputTransfer::Linear);
}
