// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/error_codes.h"
#include "core/libraries/videodec/videodec2.h"
#include "core/libraries/videodec/videodec2_decode_validation.h"
#include "core/libraries/videodec/videodec_error.h"

namespace Libraries::Videodec2 {
namespace {

struct ValidDecodeArguments {
    u8 access_unit{};
    u8 frame_storage{};
    OrbisVideodec2InputData input{
        .thisSize = sizeof(OrbisVideodec2InputData),
        .auData = &access_unit,
        .auSize = 1,
    };
    OrbisVideodec2FrameBuffer frame{
        .thisSize = sizeof(OrbisVideodec2FrameBuffer),
        .frameBuffer = &frame_storage,
        .frameBufferSize = 1,
    };
    OrbisVideodec2OutputInfo output{
        .thisSize = sizeof(OrbisVideodec2OutputInfo),
    };
};

TEST(Videodec2DecodeValidation, AcceptsCurrentAndLegacyOutputStructures) {
    ValidDecodeArguments args;
    EXPECT_EQ(ValidateDecodeArguments(args.input, args.frame, args.output), ORBIS_OK);

    args.output.thisSize = sizeof(OrbisVideodec2OutputInfo) - 8;
    EXPECT_EQ(ValidateDecodeArguments(args.input, args.frame, args.output), ORBIS_OK);
}

TEST(Videodec2DecodeValidation, RejectsInvalidOutputStructureSize) {
    ValidDecodeArguments args;
    args.output.thisSize = sizeof(OrbisVideodec2OutputInfo) - 1;

    EXPECT_EQ(ValidateDecodeArguments(args.input, args.frame, args.output),
              ORBIS_VIDEODEC2_ERROR_STRUCT_SIZE);
}

TEST(Videodec2DecodeValidation, RejectsMissingFrameBufferStorage) {
    ValidDecodeArguments args;
    args.frame.frameBuffer = nullptr;

    EXPECT_EQ(ValidateDecodeArguments(args.input, args.frame, args.output),
              ORBIS_VIDEODEC2_ERROR_FRAME_BUFFER_POINTER);
}

TEST(Videodec2DecodeValidation, RejectsZeroFrameBufferSize) {
    ValidDecodeArguments args;
    args.frame.frameBufferSize = 0;

    EXPECT_EQ(ValidateDecodeArguments(args.input, args.frame, args.output),
              ORBIS_VIDEODEC2_ERROR_FRAME_BUFFER_SIZE);
}

TEST(Videodec2DecodeValidation, RejectsFrameBufferSmallerThanDecodedOutput) {
    EXPECT_EQ(ValidateFrameBufferCapacity(0x17ff, 0x1800),
              ORBIS_VIDEODEC2_ERROR_FRAME_BUFFER_SIZE);
    EXPECT_EQ(ValidateFrameBufferCapacity(0x1800, 0x1800), ORBIS_OK);
    EXPECT_EQ(ValidateFrameBufferCapacity(0x1801, 0x1800), ORBIS_OK);
}

} // namespace
} // namespace Libraries::Videodec2
