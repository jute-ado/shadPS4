// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/file_format/playgo_chunk.h"
#include "core/libraries/playgo/playgo.h"

namespace Libraries::PlayGo {
namespace {

TEST(PlayGoLanguageMask, MapsSupportedSystemLanguagesAcrossTheFullMaskWidth) {
    EXPECT_EQ(ConvertLanguageToMask(0), u64{1} << 63);
    EXPECT_EQ(ConvertLanguageToMask(31), u64{1} << 32);
    EXPECT_EQ(ConvertLanguageToMask(32), u64{1} << 31);
    EXPECT_EQ(ConvertLanguageToMask(47), u64{1} << 16);
}

TEST(PlayGoLanguageMask, RejectsSystemLanguagesOutsideTheSupportedRange) {
    EXPECT_EQ(ConvertLanguageToMask(-1), 0);
    EXPECT_EQ(ConvertLanguageToMask(48), 0);
}

TEST(PlayGoMetadataValidation, RejectsChunkTablesTruncatedByTheHostFile) {
    PlaygoHeader header{};
    header.magic = PLAYGO_MAGIC;
    header.file_size = sizeof(PlaygoHeader) + sizeof(playgo_chunk_attr_entry_t);
    header.chunk_count = 2;
    header.chunk_attrs = {
        .offset = sizeof(PlaygoHeader),
        .length = 2 * sizeof(playgo_chunk_attr_entry_t),
    };

    EXPECT_FALSE(ValidatePlaygoHeaderLayout(header, header.file_size));
}

TEST(PlayGoMetadataValidation, AcceptsACompleteChunkTableWithinTheHostFile) {
    PlaygoHeader header{};
    header.magic = PLAYGO_MAGIC;
    header.file_size = sizeof(PlaygoHeader) + 2 * sizeof(playgo_chunk_attr_entry_t);
    header.chunk_count = 2;
    header.chunk_attrs = {
        .offset = sizeof(PlaygoHeader),
        .length = 2 * sizeof(playgo_chunk_attr_entry_t),
    };

    EXPECT_TRUE(ValidatePlaygoHeaderLayout(header, header.file_size));
}

TEST(PlayGoMetadataValidation, RejectsDeclaredMetadataLargerThanTheHostFile) {
    PlaygoHeader header{};
    header.magic = PLAYGO_MAGIC;
    header.file_size = sizeof(PlaygoHeader) + 1;

    EXPECT_FALSE(ValidatePlaygoHeaderLayout(header, sizeof(PlaygoHeader)));
}

TEST(PlayGoMetadataValidation, RejectsTruncatedMicroChunkAttributes) {
    PlaygoHeader header{};
    header.magic = PLAYGO_MAGIC;
    header.file_size = sizeof(PlaygoHeader) + sizeof(playgo_mchunk_attr_entry_t);
    header.mchunk_count = 2;
    header.mchunk_attrs = {
        .offset = sizeof(PlaygoHeader),
        .length = sizeof(playgo_mchunk_attr_entry_t),
    };

    EXPECT_FALSE(ValidatePlaygoHeaderLayout(header, header.file_size));
}

TEST(PlayGoMetadataValidation, RejectsSectionOffsetBeyondTheDeclaredFile) {
    PlaygoHeader header{};
    header.magic = PLAYGO_MAGIC;
    header.file_size = sizeof(PlaygoHeader);
    header.chunk_labels = {
        .offset = header.file_size + 1,
        .length = 0,
    };

    EXPECT_FALSE(ValidatePlaygoHeaderLayout(header, header.file_size));
}

TEST(PlayGoMetadataValidation, RejectsEntrySubrangesBeyondTheirSection) {
    EXPECT_FALSE(IsPlaygoSubrangeWithinSection(4, 2, sizeof(u16), 6));
    EXPECT_FALSE(IsPlaygoSubrangeWithinSection(7, 0, sizeof(u16), 6));
}

TEST(PlayGoMetadataValidation, AcceptsEntrySubrangesAtTheSectionBoundary) {
    EXPECT_TRUE(IsPlaygoSubrangeWithinSection(2, 2, sizeof(u16), 6));
    EXPECT_TRUE(IsPlaygoSubrangeWithinSection(6, 0, sizeof(u16), 6));
}

} // namespace
} // namespace Libraries::PlayGo
