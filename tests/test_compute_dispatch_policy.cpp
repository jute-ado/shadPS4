// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "common/signal_context.h"
#include "shader_recompiler/backend/spirv/loop_iteration_policy.h"
#include "shader_recompiler/ir/passes/srt_walker_code_range.h"
#include "video_core/renderer_vulkan/vk_compute_dispatch_policy.h"

using Vulkan::ComputeDispatchChunk;
using Vulkan::ComputeDispatchChunkAt;
using Vulkan::ComputeDispatchChunkCount;
using Vulkan::ComputeDispatchChunk2DAt;
using Vulkan::ComputeDispatchChunk2DCount;
using Vulkan::ComputeDispatchChunkNeedsBase;
using Vulkan::ComputeDispatchMaxChunkX;
using Vulkan::ComputeDispatchMaxChunkY;
using Vulkan::ComputeDispatchNeedsBasePipeline;
using Vulkan::ComputeDispatchRequiresChunkSubmissions;
using Vulkan::ComputeDispatchRequiresIdleBeforeChunks;
using Vulkan::IsComputeDispatchWithinHostLimits;
using Shader::Backend::SPIRV::LoopIterationLimit;
using Shader::Backend::SPIRV::LoopPartialCountForDynamicScalarMask;
using Shader::Backend::SPIRV::ShouldDisableDiagnosticLoopUnrolling;
using Shader::Backend::SPIRV::ShouldDisableLoopUnrollingForDynamicScalarMask;
using Shader::Backend::SPIRV::ShouldGuardDiagnosticLoop;

TEST(SrtWalkerRegistrationPolicy, CoversCachedWalkerRegisteredBeforeGeneratedWalkers) {
    Shader::SrtWalkerCodeRange range{};
    range.Include(0x1200, 0x40);

    EXPECT_TRUE(range.Contains(0x1200));
    EXPECT_TRUE(range.Contains(0x123f));
    EXPECT_FALSE(range.Contains(0x1240));

    range.Include(0x1240, 0x20);
    EXPECT_TRUE(range.Contains(0x125f));
}

#ifdef _WIN32
TEST(SrtWalkerFaultRecovery, SkipsOnlyTheFaultingLoadAndZerosItsDestination) {
    CONTEXT registers{};
    EXCEPTION_RECORD exception{};
    EXCEPTION_POINTERS context{&exception, &registers};
    registers.Rip = 0x1000;
    registers.Rdi = 0x1234;
    registers.R10 = 0x5678;

    Common::SetX64GprToZero(&context, Common::X64Gpr::Rdi);
    Common::IncrementRip(&context, 7);

    EXPECT_EQ(registers.Rdi, 0U);
    EXPECT_EQ(registers.R10, 0x5678U);
    EXPECT_EQ(registers.Rip, 0x1007U);
}
#endif

TEST(ComputeDispatchPolicy, PreservesOneSmallDispatch) {
    EXPECT_EQ(ComputeDispatchChunkCount(4, 4), 1);
    EXPECT_EQ(ComputeDispatchChunkAt(4, 4, 0),
              (ComputeDispatchChunk{.base_x = 0, .group_count_x = 4}));
}

TEST(ComputeDispatchPolicy, CoversEveryOriginalWorkgroupIdExactlyOnce) {
    EXPECT_EQ(ComputeDispatchChunkCount(10, 4), 3);
    EXPECT_EQ(ComputeDispatchChunkAt(10, 4, 0),
              (ComputeDispatchChunk{.base_x = 0, .group_count_x = 4}));
    EXPECT_EQ(ComputeDispatchChunkAt(10, 4, 1),
              (ComputeDispatchChunk{.base_x = 4, .group_count_x = 4}));
    EXPECT_EQ(ComputeDispatchChunkAt(10, 4, 2),
              (ComputeDispatchChunk{.base_x = 8, .group_count_x = 2}));
}

TEST(ComputeDispatchPolicy, ZeroXDimensionProducesNoWork) {
    EXPECT_EQ(ComputeDispatchChunkCount(0, 4), 0);
}

TEST(ComputeDispatchPolicy, RejectsEmptyAndOutOfRangeDirectDispatches) {
    constexpr std::array<u32, 3> limits{2147483647U, 65535U, 65535U};

    EXPECT_FALSE(IsComputeDispatchWithinHostLimits(0, 1, 1, limits));
    EXPECT_FALSE(IsComputeDispatchWithinHostLimits(1, 0, 1, limits));
    EXPECT_FALSE(IsComputeDispatchWithinHostLimits(1, 1, 0, limits));
    EXPECT_FALSE(IsComputeDispatchWithinHostLimits(0, 0, 65537, limits));
    EXPECT_FALSE(IsComputeDispatchWithinHostLimits(1, 1, 65536, limits));
    EXPECT_TRUE(IsComputeDispatchWithinHostLimits(12004, 16, 1, limits));
}

TEST(ComputeDispatchPolicy, UsesOrdinaryDispatchForOriginChunk) {
    EXPECT_FALSE(ComputeDispatchChunkNeedsBase(
        ComputeDispatchChunk{.base_x = 0, .base_y = 0, .group_count_x = 1,
                             .group_count_y = 1}));
    EXPECT_TRUE(ComputeDispatchChunkNeedsBase(
        ComputeDispatchChunk{.base_x = 1, .base_y = 0, .group_count_x = 1,
                             .group_count_y = 1}));
}

TEST(ComputeDispatchPolicy, BoundsOnlyConfirmedPathologicalShaderLoops) {
    EXPECT_EQ(LoopIterationLimit(0x87cf9b8a), 0);
    EXPECT_EQ(LoopIterationLimit(0x4defcf7f), 32);
    EXPECT_EQ(LoopIterationLimit(0xb33e9db6), 0);
    EXPECT_EQ(LoopIterationLimit(0xbf4600a0), 0);
    EXPECT_EQ(LoopIterationLimit(0x12345678), 0);
}

TEST(ComputeDispatchPolicy, SelectsIndividualDiagnosticLoopWithoutChangingProductionPolicy) {
    EXPECT_FALSE(ShouldGuardDiagnosticLoop(0x87cf9b8a, 0, 0));
    EXPECT_TRUE(ShouldGuardDiagnosticLoop(0x87cf9b8a, 0, 0b0000001));
    EXPECT_FALSE(ShouldGuardDiagnosticLoop(0x87cf9b8a, 1, 0b0000001));
    EXPECT_TRUE(ShouldGuardDiagnosticLoop(0x87cf9b8a, 6, 0b1000000));
    EXPECT_FALSE(ShouldGuardDiagnosticLoop(0x12345678, 0, 0b1111111));
    EXPECT_FALSE(ShouldGuardDiagnosticLoop(0x87cf9b8a, 7, 0xffffffff));
}

TEST(ComputeDispatchPolicy, SelectsIndividualLoopUnrollDiagnosticIndependently) {
    EXPECT_FALSE(ShouldDisableDiagnosticLoopUnrolling(0x87cf9b8a, 2, 0));
    EXPECT_TRUE(ShouldDisableDiagnosticLoopUnrolling(0x87cf9b8a, 2, 0b0000100));
    EXPECT_FALSE(ShouldDisableDiagnosticLoopUnrolling(0x87cf9b8a, 3, 0b0000100));
    EXPECT_FALSE(ShouldDisableDiagnosticLoopUnrolling(0x12345678, 2, 0b0000100));
}

TEST(ComputeDispatchPolicy, DisablesUnrollingOnlyForLoopsWithDynamicScalarMasks) {
    EXPECT_TRUE(ShouldDisableLoopUnrollingForDynamicScalarMask(true));
    EXPECT_FALSE(ShouldDisableLoopUnrollingForDynamicScalarMask(false));
}

TEST(ComputeDispatchPolicy, PartiallyUnrollsDynamicScalarMaskLoopsForRuntimeThroughput) {
    EXPECT_EQ(LoopPartialCountForDynamicScalarMask(true), 16);
    EXPECT_EQ(LoopPartialCountForDynamicScalarMask(false), 0);
}

TEST(ComputeDispatchPolicy, LimitsOnlyConfirmedLongRunningShaders) {
    EXPECT_EQ(ComputeDispatchMaxChunkX(0x87cf9b8a, 7), 1);
    EXPECT_EQ(ComputeDispatchMaxChunkX(0x4defcf7f, 180), 1);
    EXPECT_EQ(ComputeDispatchMaxChunkX(0xbf4600a0, 1024), ~u32{0});
    EXPECT_EQ(ComputeDispatchMaxChunkX(0xbf4600a0, 256), ~u32{0});
    EXPECT_EQ(ComputeDispatchMaxChunkX(0xbf4600a0, 80), 1);
    EXPECT_EQ(ComputeDispatchMaxChunkX(0xb33e9db6, 20), ~u32{0});
    EXPECT_EQ(ComputeDispatchMaxChunkX(0x50784801, 11), ~u32{0});
    EXPECT_EQ(ComputeDispatchMaxChunkX(0x50784801, 12004), 128);
    EXPECT_EQ(ComputeDispatchMaxChunkX(0x19de6f57, 546), 128);
    EXPECT_EQ(ComputeDispatchMaxChunkX(0x19de6f57, 64), ~u32{0});
    EXPECT_EQ(ComputeDispatchMaxChunkX(0x12345678, 12004), ~u32{0});
    EXPECT_EQ(ComputeDispatchMaxChunkY(0x950e34f8, 540), 16);
    EXPECT_EQ(ComputeDispatchMaxChunkY(0x50784801, 2199), 16);
    EXPECT_EQ(ComputeDispatchMaxChunkY(0x12345678, 24694), ~u32{0});
}

TEST(ComputeDispatchPolicy, SubmitsLongDispatchChunksSeparately) {
    EXPECT_TRUE(ComputeDispatchRequiresChunkSubmissions(0x4defcf7f));
    EXPECT_TRUE(ComputeDispatchRequiresChunkSubmissions(0x50784801));
    EXPECT_TRUE(ComputeDispatchRequiresChunkSubmissions(0x87cf9b8a));
    EXPECT_TRUE(ComputeDispatchRequiresChunkSubmissions(0x950e34f8));
    EXPECT_TRUE(ComputeDispatchRequiresChunkSubmissions(0xbf4600a0));
    EXPECT_TRUE(ComputeDispatchRequiresChunkSubmissions(0x19de6f57));
    EXPECT_TRUE(ComputeDispatchRequiresChunkSubmissions(0x57bd9f27));
    EXPECT_FALSE(ComputeDispatchRequiresChunkSubmissions(0xb33e9db6));
    EXPECT_FALSE(ComputeDispatchRequiresChunkSubmissions(0x12345678));
}

TEST(ComputeDispatchPolicy, IsolatesProtectedChunksFromEarlierGpuWork) {
    EXPECT_FALSE(ComputeDispatchRequiresIdleBeforeChunks(0x4defcf7f));
    EXPECT_FALSE(ComputeDispatchRequiresIdleBeforeChunks(0x50784801));
    EXPECT_TRUE(ComputeDispatchRequiresIdleBeforeChunks(0x87cf9b8a));
    EXPECT_FALSE(ComputeDispatchRequiresIdleBeforeChunks(0x12345678));
}

TEST(ComputeDispatchPolicy, RequestsDispatchBasePipelineForChunkedShaders) {
    EXPECT_TRUE(ComputeDispatchNeedsBasePipeline(0x87cf9b8a));
    EXPECT_TRUE(ComputeDispatchNeedsBasePipeline(0x50784801));
    EXPECT_TRUE(ComputeDispatchNeedsBasePipeline(0x4defcf7f));
    EXPECT_TRUE(ComputeDispatchNeedsBasePipeline(0xbf4600a0));
    EXPECT_TRUE(ComputeDispatchNeedsBasePipeline(0x950e34f8));
    EXPECT_TRUE(ComputeDispatchNeedsBasePipeline(0x19de6f57));
    EXPECT_TRUE(ComputeDispatchNeedsBasePipeline(0x57bd9f27));
    EXPECT_FALSE(ComputeDispatchNeedsBasePipeline(0xb33e9db6));
    EXPECT_FALSE(ComputeDispatchNeedsBasePipeline(0x12345678));
}

TEST(ComputeDispatchPolicy, BoundsConfirmedLongRunningTwoDimensionalSceneDispatch) {
    EXPECT_EQ(ComputeDispatchMaxChunkX(0x57bd9f27, 120), 16);
    EXPECT_EQ(ComputeDispatchMaxChunkY(0x57bd9f27, 68), 8);
    EXPECT_TRUE(ComputeDispatchRequiresChunkSubmissions(0x57bd9f27));
    EXPECT_EQ(ComputeDispatchChunk2DCount(120, 68, 16, 8), 72);
    EXPECT_EQ(ComputeDispatchChunk2DAt(120, 68, 16, 8, 71),
              (ComputeDispatchChunk{.base_x = 112, .base_y = 64, .group_count_x = 8,
                                    .group_count_y = 4}));
}

TEST(ComputeDispatchPolicy, SplitsProtectedTwoDimensionalDispatchIntoSafeWorkgroups) {
    ASSERT_EQ(LoopIterationLimit(0x87cf9b8a), 0);
    EXPECT_EQ(ComputeDispatchMaxChunkX(0x87cf9b8a, 7), 1);
    EXPECT_EQ(ComputeDispatchMaxChunkY(0x87cf9b8a, 7), 1);
    EXPECT_TRUE(ComputeDispatchRequiresChunkSubmissions(0x87cf9b8a));
    EXPECT_EQ(ComputeDispatchChunk2DCount(7, 7, 1, 1), 49);
    EXPECT_EQ(ComputeDispatchChunk2DAt(7, 7, 1, 1, 48),
              (ComputeDispatchChunk{.base_x = 6, .base_y = 6, .group_count_x = 1,
                                    .group_count_y = 1}));
}
