// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/error_codes.h"
#include "core/libraries/videodec/videodec2.h"
#include "core/libraries/videodec/videodec2_compute_validation.h"
#include "core/libraries/videodec/videodec2_decode_validation.h"
#include "core/libraries/videodec/videodec2_decoder_validation.h"
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

struct ValidComputeArguments {
    OrbisVideodec2ComputeConfigInfo config{
        .thisSize = sizeof(OrbisVideodec2ComputeConfigInfo),
    };
    OrbisVideodec2ComputeMemoryInfo memory{
        .thisSize = sizeof(OrbisVideodec2ComputeMemoryInfo),
        .cpuGpuMemorySize = Videodec2ComputeMemorySize,
        .cpuGpuMemory = reinterpret_cast<void*>(0x10000),
    };
};

OrbisVideodec2DecoderConfigInfo ValidDecoderConfig() {
    return {
        .thisSize = sizeof(OrbisVideodec2DecoderConfigInfo),
        .codecType = 1,
    };
}

TEST(Videodec2DecoderValidation, AcceptsSupportedAvcCodec) {
    EXPECT_EQ(ValidateDecoderConfig(ValidDecoderConfig()), ORBIS_OK);
}

TEST(Videodec2DecoderValidation, RejectsUnsupportedCodecBeforeConstruction) {
    auto config = ValidDecoderConfig();
    config.codecType = 0;
    EXPECT_EQ(ValidateDecoderConfig(config), ORBIS_VIDEODEC2_ERROR_CODEC_TYPE);

    config.codecType = 2;
    EXPECT_EQ(ValidateDecoderConfig(config), ORBIS_VIDEODEC2_ERROR_CODEC_TYPE);
}

TEST(Videodec2ComputeValidation, AcceptsAdvertisedMinimumMemory) {
    ValidComputeArguments args;

    EXPECT_EQ(ValidateComputeQueueArguments(args.config, args.memory), ORBIS_OK);
}

TEST(Videodec2ComputeValidation, RejectsMemoryBelowAdvertisedMinimum) {
    ValidComputeArguments args;
    args.memory.cpuGpuMemorySize = Videodec2ComputeMemorySize - 1;

    EXPECT_EQ(ValidateComputeQueueArguments(args.config, args.memory),
              ORBIS_VIDEODEC2_ERROR_MEMORY_SIZE);
}

TEST(Videodec2ComputeValidation, RejectsInvalidQueueConfiguration) {
    ValidComputeArguments args;
    args.config.computeQueueId = 8;

    EXPECT_EQ(ValidateComputeQueueArguments(args.config, args.memory),
              ORBIS_VIDEODEC2_ERROR_COMPUTE_QUEUE_ID);
}

TEST(Videodec2ComputeValidation, PreservesSpecificConfigurationAndMemoryErrors) {
    ValidComputeArguments args;

    args.config.computePipeId = 5;
    EXPECT_EQ(ValidateComputeQueueArguments(args.config, args.memory),
              ORBIS_VIDEODEC2_ERROR_COMPUTE_PIPE_ID);

    args.config.computePipeId = 0;
    args.config.reserved0 = 1;
    EXPECT_EQ(ValidateComputeQueueArguments(args.config, args.memory),
              ORBIS_VIDEODEC2_ERROR_CONFIG_INFO);

    args.config.reserved0 = 0;
    args.memory.cpuGpuMemory = nullptr;
    EXPECT_EQ(ValidateComputeQueueArguments(args.config, args.memory),
              ORBIS_VIDEODEC2_ERROR_MEMORY_POINTER);
}

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
    EXPECT_EQ(ValidateFrameBufferCapacity(0x17ff, 0x1800), ORBIS_VIDEODEC2_ERROR_FRAME_BUFFER_SIZE);
    EXPECT_EQ(ValidateFrameBufferCapacity(0x1800, 0x1800), ORBIS_OK);
    EXPECT_EQ(ValidateFrameBufferCapacity(0x1801, 0x1800), ORBIS_OK);
}

} // namespace
} // namespace Libraries::Videodec2
