// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/videodec/videodec2_picture_info_store.h"

using Libraries::Videodec2::OrbisVideodec2AvcPictureInfo;
using Libraries::Videodec2::PictureInfoStore;

TEST(Videodec2PictureInfoStore, ReplacesPreviouslyDecodedPicture) {
    PictureInfoStore store;
    OrbisVideodec2AvcPictureInfo first{};
    first.ptsData = 11;
    OrbisVideodec2AvcPictureInfo second{};
    second.ptsData = 22;

    store.Set(first);
    store.Set(second);

    const auto latest = store.Get();
    ASSERT_TRUE(latest.has_value());
    EXPECT_EQ(latest->ptsData, 22);
}

TEST(Videodec2PictureInfoStore, ClearRemovesLatestPicture) {
    PictureInfoStore store;
    OrbisVideodec2AvcPictureInfo picture{};
    store.Set(picture);

    store.Clear();

    EXPECT_FALSE(store.Get().has_value());
}
