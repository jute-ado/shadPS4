// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/videodec/videodec2_picture_info_store.h"

using Libraries::Videodec2::OrbisVideodec2AvcPictureInfo;
using Libraries::Videodec2::PictureInfoStore;

TEST(Videodec2PictureInfoStore, ReplacesPreviouslyDecodedPicture) {
    PictureInfoStore store;
    int decoder;
    int frame_buffer;
    OrbisVideodec2AvcPictureInfo first{};
    first.ptsData = 11;
    OrbisVideodec2AvcPictureInfo second{};
    second.ptsData = 22;

    store.Set(&decoder, &frame_buffer, first);
    store.Set(&decoder, &frame_buffer, second);

    const auto latest = store.Get(&frame_buffer);
    ASSERT_TRUE(latest.has_value());
    EXPECT_EQ(latest->ptsData, 22);
}

TEST(Videodec2PictureInfoStore, KeepsDecoderOutputsIndependent) {
    PictureInfoStore store;
    int first_decoder;
    int second_decoder;
    int first_frame_buffer;
    int second_frame_buffer;
    OrbisVideodec2AvcPictureInfo first{};
    first.ptsData = 11;
    OrbisVideodec2AvcPictureInfo second{};
    second.ptsData = 22;

    store.Set(&first_decoder, &first_frame_buffer, first);
    store.Set(&second_decoder, &second_frame_buffer, second);

    ASSERT_TRUE(store.Get(&first_frame_buffer).has_value());
    EXPECT_EQ(store.Get(&first_frame_buffer)->ptsData, 11);
    ASSERT_TRUE(store.Get(&second_frame_buffer).has_value());
    EXPECT_EQ(store.Get(&second_frame_buffer)->ptsData, 22);
}

TEST(Videodec2PictureInfoStore, ClearRemovesOnlyTheOwningDecoderPictures) {
    PictureInfoStore store;
    int first_decoder;
    int second_decoder;
    int first_frame_buffer;
    int second_frame_buffer;
    OrbisVideodec2AvcPictureInfo first{};
    first.ptsData = 11;
    OrbisVideodec2AvcPictureInfo second{};
    second.ptsData = 22;

    store.Set(&first_decoder, &first_frame_buffer, first);
    store.Set(&second_decoder, &second_frame_buffer, second);

    store.Clear(&first_decoder);

    EXPECT_FALSE(store.Get(&first_frame_buffer).has_value());
    ASSERT_TRUE(store.Get(&second_frame_buffer).has_value());
    EXPECT_EQ(store.Get(&second_frame_buffer)->ptsData, 22);
}
