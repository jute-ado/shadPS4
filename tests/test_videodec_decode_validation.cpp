// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/error_codes.h"
#include "core/libraries/videodec/videodec.h"
#include "core/libraries/videodec/videodec_decode_validation.h"
#include "core/libraries/videodec/videodec_error.h"

namespace Libraries::Videodec {
namespace {

struct ValidDecodeArguments {
    u8 access_unit{};
    u8 frame_storage{};
    OrbisVideodecInputData input{
        .thisSize = sizeof(OrbisVideodecInputData),
        .pAuData = &access_unit,
        .auSize = 1,
    };
    OrbisVideodecFrameBuffer frame{
        .thisSize = sizeof(OrbisVideodecFrameBuffer),
        .pFrameBuffer = &frame_storage,
        .frameBufferSize = 1,
    };
    OrbisVideodecPictureInfo picture{
        .thisSize = sizeof(OrbisVideodecPictureInfo),
    };
};

TEST(VideodecDecodeValidation, AcceptsValidArguments) {
    ValidDecodeArguments args;
    EXPECT_EQ(ValidateDecodeArguments(args.input, args.frame, args.picture), ORBIS_OK);
}

TEST(VideodecDecodeValidation, RejectsInvalidGuestStructureSizes) {
    ValidDecodeArguments args;
    args.input.thisSize = 0;
    EXPECT_EQ(ValidateDecodeArguments(args.input, args.frame, args.picture),
              ORBIS_VIDEODEC_ERROR_STRUCT_SIZE);

    args.input.thisSize = sizeof(OrbisVideodecInputData);
    args.picture.thisSize = 0;
    EXPECT_EQ(ValidateDecodeArguments(args.input, args.frame, args.picture),
              ORBIS_VIDEODEC_ERROR_STRUCT_SIZE);
}

TEST(VideodecDecodeValidation, RejectsMissingOrZeroSizedFrameBuffer) {
    ValidDecodeArguments args;
    args.frame.pFrameBuffer = nullptr;
    EXPECT_EQ(ValidateDecodeArguments(args.input, args.frame, args.picture),
              ORBIS_VIDEODEC_ERROR_FRAME_BUFFER_POINTER);

    args.frame.pFrameBuffer = &args.frame_storage;
    args.frame.frameBufferSize = 0;
    EXPECT_EQ(ValidateDecodeArguments(args.input, args.frame, args.picture),
              ORBIS_VIDEODEC_ERROR_FRAME_BUFFER_SIZE);
}

TEST(VideodecDecodeValidation, RejectsFrameBufferSmallerThanDecodedOutput) {
    EXPECT_EQ(ValidateFrameBufferCapacity(0x17ff, 0x1800),
              ORBIS_VIDEODEC_ERROR_FRAME_BUFFER_SIZE);
    EXPECT_EQ(ValidateFrameBufferCapacity(0x1800, 0x1800), ORBIS_OK);
}

} // namespace
} // namespace Libraries::Videodec
