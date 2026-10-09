// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/vk_pipeline_cache_serialization.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <ranges>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <spirv/unified1/spirv.hpp11>

#include "common/serdes.h"
#include "shader_recompiler/profile.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {
namespace {

struct SpirvOpcodeRule {
    u16 opcode;
    u16 minimum_words;
    bool has_result;
    bool has_result_type;
};

constexpr std::array<SpirvOpcodeRule, 218> CachedSpirvOpcodeRules{{
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpUndef), 3, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpName), 3, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpMemberName), 4, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpString), 3, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpLine), 4, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpExtension), 2, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpExtInstImport), 3, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpExtInst), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpMemoryModel), 3, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpEntryPoint), 4, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpExecutionMode), 3, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpCapability), 2, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeVoid), 2, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeBool), 2, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeInt), 4, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeFloat), 3, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeVector), 4, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeMatrix), 4, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeImage), 9, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeSampler), 2, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeSampledImage), 3, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeArray), 4, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeRuntimeArray), 3, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeStruct), 2, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeOpaque), 3, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypePointer), 4, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeFunction), 3, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeEvent), 2, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeDeviceEvent), 2, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeReserveId), 2, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypeQueue), 2, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpTypePipe), 3, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpConstantTrue), 3, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpConstantFalse), 3, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpConstant), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpConstantComposite), 3, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpConstantSampler), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpConstantNull), 3, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFunction), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFunctionParameter), 3, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFunctionEnd), 1, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFunctionCall), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpVariable), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageTexelPointer), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpLoad), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpStore), 3, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAccessChain), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpDecorate), 3, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpMemberDecorate), 4, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpVectorExtractDynamic), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpVectorInsertDynamic), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpVectorShuffle), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpCompositeConstruct), 3, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpCompositeExtract), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpCompositeInsert), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSampledImage), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSampleImplicitLod), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSampleExplicitLod), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSampleDrefImplicitLod), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSampleDrefExplicitLod), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSampleProjImplicitLod), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSampleProjExplicitLod), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSampleProjDrefImplicitLod), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSampleProjDrefExplicitLod), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageFetch), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageGather), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageDrefGather), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageRead), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageWrite), 4, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImage), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageQuerySizeLod), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageQuerySize), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageQueryLod), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageQueryLevels), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageQuerySamples), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpConvertFToU), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpConvertFToS), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpConvertSToF), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpConvertUToF), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpUConvert), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSConvert), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFConvert), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpQuantizeToF16), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpConvertPtrToU), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpConvertUToPtr), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpBitcast), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSNegate), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFNegate), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpIAdd), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFAdd), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpISub), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFSub), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpIMul), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFMul), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpUDiv), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSDiv), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFDiv), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpUMod), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSRem), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSMod), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFRem), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFMod), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpVectorTimesScalar), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpDot), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpIAddCarry), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpUMulExtended), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSMulExtended), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAny), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAll), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpIsNan), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpIsInf), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpLogicalEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpLogicalNotEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpLogicalOr), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpLogicalAnd), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpLogicalNot), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSelect), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpIEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpINotEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpUGreaterThan), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSGreaterThan), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpUGreaterThanEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSGreaterThanEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpULessThan), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSLessThan), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpULessThanEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSLessThanEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFOrdEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFUnordEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFOrdNotEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFUnordNotEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFOrdLessThan), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFUnordLessThan), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFOrdGreaterThan), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFUnordGreaterThan), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFOrdLessThanEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFUnordLessThanEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFOrdGreaterThanEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFUnordGreaterThanEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpShiftRightLogical), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpShiftRightArithmetic), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpShiftLeftLogical), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpBitwiseOr), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpBitwiseXor), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpBitwiseAnd), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpNot), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpBitFieldInsert), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpBitFieldSExtract), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpBitFieldUExtract), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpBitReverse), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpBitCount), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpDPdx), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpDPdy), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFwidth), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpDPdxFine), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpDPdyFine), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFwidthFine), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpDPdxCoarse), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpDPdyCoarse), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpFwidthCoarse), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpEmitVertex), 1, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpEndPrimitive), 1, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpEmitStreamVertex), 2, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpEndStreamPrimitive), 2, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpControlBarrier), 4, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpMemoryBarrier), 3, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicLoad), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicStore), 5, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicExchange), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicCompareExchange), 9, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicIIncrement), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicIDecrement), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicIAdd), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicISub), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicSMin), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicUMin), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicSMax), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicUMax), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicAnd), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicOr), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicXor), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpPhi), 3, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpLoopMerge), 4, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSelectionMerge), 3, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpLabel), 2, true, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpBranch), 2, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpBranchConditional), 4, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSwitch), 3, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpKill), 1, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpReturn), 1, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpReturnValue), 2, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpUnreachable), 1, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSparseSampleImplicitLod), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSparseSampleExplicitLod), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSparseSampleDrefImplicitLod), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSparseSampleDrefExplicitLod), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSparseFetch), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSparseGather), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSparseDrefGather), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSparseTexelsResident), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpImageSparseRead), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpGroupNonUniformAll), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpGroupNonUniformAny), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpGroupNonUniformAllEqual), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpGroupNonUniformBroadcast), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpGroupNonUniformBroadcastFirst), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpGroupNonUniformBallot), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpGroupNonUniformBallotFindLSB), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpGroupNonUniformShuffleXor), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpGroupNonUniformQuadBroadcast), 6, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSubgroupBallotKHR), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSubgroupAllKHR), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSubgroupAnyKHR), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSubgroupAllEqualKHR), 4, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpSubgroupReadInvocationKHR), 5, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpDemoteToHelperInvocationEXT), 1, false, false},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicFMinEXT), 7, true, true},
    SpirvOpcodeRule{static_cast<u16>(spv::Op::OpAtomicFMaxEXT), 7, true, true},
}};
static_assert(std::ranges::is_sorted(CachedSpirvOpcodeRules, {}, &SpirvOpcodeRule::opcode));

[[nodiscard]] const SpirvOpcodeRule* FindSpirvOpcodeRule(u16 opcode) noexcept {
    const auto it =
        std::ranges::lower_bound(CachedSpirvOpcodeRules, opcode, {}, &SpirvOpcodeRule::opcode);
    return it != CachedSpirvOpcodeRules.end() && it->opcode == opcode ? &*it : nullptr;
}

[[nodiscard]] bool IsSpirvTypeOpcode(spv::Op opcode) noexcept {
    switch (opcode) {
    case spv::Op::OpTypeVoid:
    case spv::Op::OpTypeBool:
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeVector:
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeImage:
    case spv::Op::OpTypeSampler:
    case spv::Op::OpTypeSampledImage:
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
    case spv::Op::OpTypeStruct:
    case spv::Op::OpTypeOpaque:
    case spv::Op::OpTypePointer:
    case spv::Op::OpTypeFunction:
    case spv::Op::OpTypeEvent:
    case spv::Op::OpTypeDeviceEvent:
    case spv::Op::OpTypeReserveId:
    case spv::Op::OpTypeQueue:
    case spv::Op::OpTypePipe:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool IsSpirvConstantOpcode(spv::Op opcode) noexcept {
    switch (opcode) {
    case spv::Op::OpConstantTrue:
    case spv::Op::OpConstantFalse:
    case spv::Op::OpConstant:
    case spv::Op::OpConstantComposite:
    case spv::Op::OpConstantSampler:
    case spv::Op::OpConstantNull:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool IsSpirvDebugOpcode(spv::Op opcode) noexcept {
    switch (opcode) {
    case spv::Op::OpString:
    case spv::Op::OpLine:
    case spv::Op::OpName:
    case spv::Op::OpMemberName:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool IsSpirvAnnotationOpcode(spv::Op opcode) noexcept {
    return opcode == spv::Op::OpDecorate || opcode == spv::Op::OpMemberDecorate;
}

[[nodiscard]] bool IsSpirvTerminatorOpcode(spv::Op opcode) noexcept {
    switch (opcode) {
    case spv::Op::OpBranch:
    case spv::Op::OpBranchConditional:
    case spv::Op::OpSwitch:
    case spv::Op::OpKill:
    case spv::Op::OpReturn:
    case spv::Op::OpReturnValue:
    case spv::Op::OpUnreachable:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool IsAllowedSpirvCapability(u32 capability) noexcept {
    switch (static_cast<spv::Capability>(capability)) {
    case spv::Capability::Shader:
    case spv::Capability::Geometry:
    case spv::Capability::Tessellation:
    case spv::Capability::Float16:
    case spv::Capability::Float64:
    case spv::Capability::Int64:
    case spv::Capability::Int64Atomics:
    case spv::Capability::Int16:
    case spv::Capability::ImageGatherExtended:
    case spv::Capability::SampleRateShading:
    case spv::Capability::Int8:
    case spv::Capability::Sampled1D:
    case spv::Capability::Image1D:
    case spv::Capability::StorageImageExtendedFormats:
    case spv::Capability::ImageQuery:
    case spv::Capability::InterpolationFunction:
    case spv::Capability::StorageImageReadWithoutFormat:
    case spv::Capability::StorageImageWriteWithoutFormat:
    case spv::Capability::MultiViewport:
    case spv::Capability::DrawParameters:
    case spv::Capability::UniformAndStorageBuffer16BitAccess:
    case spv::Capability::UniformAndStorageBuffer8BitAccess:
    case spv::Capability::StencilExportEXT:
    case spv::Capability::ImageReadWriteLodAMD:
    case spv::Capability::FragmentBarycentricKHR:
    case spv::Capability::ShaderViewportIndex:
    case spv::Capability::ShaderLayer:
    case spv::Capability::PhysicalStorageBufferAddresses:
    case spv::Capability::GroupNonUniform:
    case spv::Capability::GroupNonUniformBallot:
    case spv::Capability::GroupNonUniformQuad:
    case spv::Capability::DemoteToHelperInvocation:
    case spv::Capability::AtomicFloat32MinMaxEXT:
    case spv::Capability::WorkgroupMemoryExplicitLayoutKHR:
    case spv::Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR:
    case spv::Capability::DenormPreserve:
    case spv::Capability::DenormFlushToZero:
    case spv::Capability::SignedZeroInfNanPreserve:
    case spv::Capability::RoundingModeRTZ:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool IsAllowedSpirvExecutionModel(u32 model) noexcept {
    switch (static_cast<spv::ExecutionModel>(model)) {
    case spv::ExecutionModel::Vertex:
    case spv::ExecutionModel::TessellationControl:
    case spv::ExecutionModel::TessellationEvaluation:
    case spv::ExecutionModel::Geometry:
    case spv::ExecutionModel::Fragment:
    case spv::ExecutionModel::GLCompute:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] std::optional<u16> ExpectedExecutionModeWordCount(u32 mode) noexcept {
    switch (static_cast<spv::ExecutionMode>(mode)) {
    case spv::ExecutionMode::Invocations:
    case spv::ExecutionMode::OutputVertices:
    case spv::ExecutionMode::DenormPreserve:
    case spv::ExecutionMode::DenormFlushToZero:
    case spv::ExecutionMode::SignedZeroInfNanPreserve:
    case spv::ExecutionMode::RoundingModeRTZ:
        return 4;
    case spv::ExecutionMode::LocalSize:
        return 6;
    case spv::ExecutionMode::SpacingEqual:
    case spv::ExecutionMode::SpacingFractionalEven:
    case spv::ExecutionMode::SpacingFractionalOdd:
    case spv::ExecutionMode::VertexOrderCw:
    case spv::ExecutionMode::VertexOrderCcw:
    case spv::ExecutionMode::OriginUpperLeft:
    case spv::ExecutionMode::OriginLowerLeft:
    case spv::ExecutionMode::DepthReplacing:
    case spv::ExecutionMode::InputPoints:
    case spv::ExecutionMode::InputLines:
    case spv::ExecutionMode::InputLinesAdjacency:
    case spv::ExecutionMode::Triangles:
    case spv::ExecutionMode::InputTrianglesAdjacency:
    case spv::ExecutionMode::Quads:
    case spv::ExecutionMode::Isolines:
    case spv::ExecutionMode::OutputLineStrip:
    case spv::ExecutionMode::OutputTriangleStrip:
        return 3;
    default:
        return std::nullopt;
    }
}

struct SpirvLiteralString {
    std::string_view value;
    size_t next_word;
};

[[nodiscard]] std::optional<SpirvLiteralString> ReadSpirvLiteralString(
    std::span<const u32> instruction, size_t first_word) noexcept {
    if (first_word >= instruction.size()) {
        return std::nullopt;
    }
    const auto* bytes = reinterpret_cast<const char*>(instruction.data() + first_word);
    const size_t byte_count = (instruction.size() - first_word) * sizeof(u32);
    for (size_t index = 0; index < byte_count; ++index) {
        if (bytes[index] != '\0') {
            continue;
        }
        const size_t next_word = first_word + index / sizeof(u32) + 1;
        const size_t padding_end = (index / sizeof(u32) + 1) * sizeof(u32);
        for (size_t padding = index + 1; padding < padding_end; ++padding) {
            if (bytes[padding] != '\0') {
                return std::nullopt;
            }
        }
        return SpirvLiteralString{std::string_view{bytes, index}, next_word};
    }
    return std::nullopt;
}

[[nodiscard]] bool IsAllowedSpirvExtension(std::string_view extension) noexcept {
    constexpr std::array AllowedExtensions{
        std::string_view{"SPV_AMD_shader_image_load_store_lod"},
        std::string_view{"SPV_EXT_demote_to_helper_invocation"},
        std::string_view{"SPV_EXT_shader_atomic_float_min_max"},
        std::string_view{"SPV_EXT_shader_stencil_export"},
        std::string_view{"SPV_KHR_float_controls"},
        std::string_view{"SPV_KHR_fragment_shader_barycentric"},
        std::string_view{"SPV_KHR_physical_storage_buffer"},
        std::string_view{"SPV_KHR_shader_draw_parameters"},
        std::string_view{"SPV_KHR_workgroup_memory_explicit_layout"},
        std::string_view{"SPV_AMD_shader_explicit_vertex_parameter"},
    };
    return std::ranges::contains(AllowedExtensions, extension);
}

[[nodiscard]] bool IsAllowedSpirvExtInstImport(std::string_view import) noexcept {
    constexpr std::array AllowedImports{
        std::string_view{"GLSL.std.450"},
        std::string_view{"NonSemantic.DebugPrintf"},
        std::string_view{"SPV_AMD_gcn_shader"},
        std::string_view{"SPV_AMD_shader_explicit_vertex_parameter"},
        std::string_view{"SPV_AMD_shader_trinary_minmax"},
    };
    return std::ranges::contains(AllowedImports, import);
}

enum class SpirvModuleSection : u8 {
    Capabilities,
    Extensions,
    Imports,
    MemoryModel,
    EntryPoints,
    ExecutionModes,
    Debug,
    Annotations,
    Declarations,
    Functions,
};

[[nodiscard]] bool AdvanceSpirvSection(SpirvModuleSection& current,
                                       SpirvModuleSection next) noexcept {
    if (next < current) {
        return false;
    }
    current = next;
    return true;
}

void WriteBool(Serialization::Writer& writer, bool value) {
    writer.Write(static_cast<u8>(value ? 1 : 0));
}

[[nodiscard]] bool ReadBool(Serialization::Reader& reader, bool& value) {
    u8 encoded{};
    if (!reader.Read(encoded) || encoded > 1) {
        return false;
    }
    value = encoded != 0;
    return true;
}

void WriteCount(Serialization::Writer& writer, size_t count) {
    ASSERT(count <= std::numeric_limits<u32>::max());
    writer.Write(static_cast<u32>(count));
}

[[nodiscard]] bool ReadCount(Serialization::Reader& reader, u32& count, u32 maximum) {
    return reader.Read(count) && count <= maximum;
}

template <typename Enum, typename Validator>
    requires std::is_enum_v<Enum>
[[nodiscard]] bool ReadEnum(Serialization::Reader& reader, Enum& value, Validator&& validator) {
    u32 encoded{};
    if (!reader.Read(encoded) || !validator(encoded)) {
        return false;
    }
    value = static_cast<Enum>(encoded);
    return true;
}

template <typename Enum>
    requires std::is_enum_v<Enum>
void WriteEnum(Serialization::Writer& writer, Enum value) {
    writer.Write(static_cast<u32>(value));
}

void WriteProfileFields(Serialization::Writer& writer, const Shader::Profile& profile) {
    writer.Write(profile.max_ubo_size);
    writer.Write(profile.max_viewport_width);
    writer.Write(profile.max_viewport_height);
    writer.Write(profile.max_shared_memory_size);
    writer.Write(profile.supported_spirv);
    writer.Write(profile.subgroup_size);
    WriteBool(writer, profile.support_int8);
    WriteBool(writer, profile.support_int16);
    WriteBool(writer, profile.support_int64);
    WriteBool(writer, profile.support_float16);
    WriteBool(writer, profile.support_float64);
    WriteBool(writer, profile.supports_denorm_behavior_independence);
    WriteBool(writer, profile.supports_rounding_mode_independence);
    WriteBool(writer, profile.support_fp16_denorm_preserve);
    WriteBool(writer, profile.support_fp16_denorm_flush);
    WriteBool(writer, profile.support_fp16_round_to_zero);
    WriteBool(writer, profile.support_fp32_denorm_preserve);
    WriteBool(writer, profile.support_fp32_denorm_flush);
    WriteBool(writer, profile.support_fp32_round_to_zero);
    WriteBool(writer, profile.support_fp64_denorm_preserve);
    WriteBool(writer, profile.support_fp64_denorm_flush);
    WriteBool(writer, profile.support_fp64_round_to_zero);
    WriteBool(writer, profile.support_fp16_signed_zero_inf_nan_preserve);
    WriteBool(writer, profile.support_fp32_signed_zero_inf_nan_preserve);
    WriteBool(writer, profile.support_fp64_signed_zero_inf_nan_preserve);
    WriteBool(writer, profile.supports_image_load_store_lod);
    WriteBool(writer, profile.supports_native_cube_calc);
    WriteBool(writer, profile.supports_trinary_minmax);
    WriteBool(writer, profile.supports_buffer_fp32_atomic_min_max);
    WriteBool(writer, profile.supports_image_fp32_atomic_min_max);
    WriteBool(writer, profile.supports_buffer_int64_atomics);
    WriteBool(writer, profile.supports_shared_int64_atomics);
    WriteBool(writer, profile.supports_workgroup_explicit_memory_layout);
    WriteBool(writer, profile.supports_amd_shader_explicit_vertex_parameter);
    WriteBool(writer, profile.supports_fragment_shader_barycentric);
    WriteBool(writer, profile.has_broken_spirv_clamp);
    WriteBool(writer, profile.lower_left_origin_mode);
    WriteBool(writer, profile.needs_manual_interpolation);
    WriteBool(writer, profile.needs_lds_barriers);
    WriteBool(writer, profile.needs_buffer_offsets);
    WriteBool(writer, profile.needs_unorm_fixup);
    WriteBool(writer, profile.needs_clip_distance_emulation);
    WriteBool(writer, profile.supports_shader_stencil_export);
}

[[nodiscard]] bool ReadProfileFields(Serialization::Reader& reader, Shader::Profile& profile) {
    return reader.Read(profile.max_ubo_size) && reader.Read(profile.max_viewport_width) &&
           reader.Read(profile.max_viewport_height) &&
           reader.Read(profile.max_shared_memory_size) && reader.Read(profile.supported_spirv) &&
           reader.Read(profile.subgroup_size) && ReadBool(reader, profile.support_int8) &&
           ReadBool(reader, profile.support_int16) && ReadBool(reader, profile.support_int64) &&
           ReadBool(reader, profile.support_float16) && ReadBool(reader, profile.support_float64) &&
           ReadBool(reader, profile.supports_denorm_behavior_independence) &&
           ReadBool(reader, profile.supports_rounding_mode_independence) &&
           ReadBool(reader, profile.support_fp16_denorm_preserve) &&
           ReadBool(reader, profile.support_fp16_denorm_flush) &&
           ReadBool(reader, profile.support_fp16_round_to_zero) &&
           ReadBool(reader, profile.support_fp32_denorm_preserve) &&
           ReadBool(reader, profile.support_fp32_denorm_flush) &&
           ReadBool(reader, profile.support_fp32_round_to_zero) &&
           ReadBool(reader, profile.support_fp64_denorm_preserve) &&
           ReadBool(reader, profile.support_fp64_denorm_flush) &&
           ReadBool(reader, profile.support_fp64_round_to_zero) &&
           ReadBool(reader, profile.support_fp16_signed_zero_inf_nan_preserve) &&
           ReadBool(reader, profile.support_fp32_signed_zero_inf_nan_preserve) &&
           ReadBool(reader, profile.support_fp64_signed_zero_inf_nan_preserve) &&
           ReadBool(reader, profile.supports_image_load_store_lod) &&
           ReadBool(reader, profile.supports_native_cube_calc) &&
           ReadBool(reader, profile.supports_trinary_minmax) &&
           ReadBool(reader, profile.supports_buffer_fp32_atomic_min_max) &&
           ReadBool(reader, profile.supports_image_fp32_atomic_min_max) &&
           ReadBool(reader, profile.supports_buffer_int64_atomics) &&
           ReadBool(reader, profile.supports_shared_int64_atomics) &&
           ReadBool(reader, profile.supports_workgroup_explicit_memory_layout) &&
           ReadBool(reader, profile.supports_amd_shader_explicit_vertex_parameter) &&
           ReadBool(reader, profile.supports_fragment_shader_barycentric) &&
           ReadBool(reader, profile.has_broken_spirv_clamp) &&
           ReadBool(reader, profile.lower_left_origin_mode) &&
           ReadBool(reader, profile.needs_manual_interpolation) &&
           ReadBool(reader, profile.needs_lds_barriers) &&
           ReadBool(reader, profile.needs_buffer_offsets) &&
           ReadBool(reader, profile.needs_unorm_fixup) &&
           ReadBool(reader, profile.needs_clip_distance_emulation) &&
           ReadBool(reader, profile.supports_shader_stencil_export);
}

[[nodiscard]] constexpr bool IsValidPrimitiveType(AmdGpu::PrimitiveType type) noexcept {
    using enum AmdGpu::PrimitiveType;
    switch (type) {
    case PointList:
    case LineList:
    case LineStrip:
    case TriangleList:
    case TriangleFan:
    case TriangleStrip:
    case PatchPrimitive:
    case AdjLineList:
    case AdjLineStrip:
    case AdjTriangleList:
    case AdjTriangleStrip:
    case RectList:
    case QuadList:
    case Polygon:
        return true;
    case None:
    case LineLoop:
    case QuadStrip:
        return false;
    }
    return false;
}

[[nodiscard]] constexpr bool IsValidBlendFactor(AmdGpu::BlendControl::BlendFactor factor) noexcept {
    using enum AmdGpu::BlendControl::BlendFactor;
    switch (factor) {
    case Zero:
    case One:
    case SrcColor:
    case OneMinusSrcColor:
    case SrcAlpha:
    case OneMinusSrcAlpha:
    case DstAlpha:
    case OneMinusDstAlpha:
    case DstColor:
    case OneMinusDstColor:
    case SrcAlphaSaturate:
    case ConstantColor:
    case OneMinusConstantColor:
    case Src1Color:
    case InvSrc1Color:
    case Src1Alpha:
    case InvSrc1Alpha:
    case ConstantAlpha:
    case OneMinusConstantAlpha:
        return true;
    }
    return false;
}

[[nodiscard]] constexpr bool IsValidBlendControl(const AmdGpu::BlendControl& control) noexcept {
    return IsValidBlendFactor(control.color_src_factor) &&
           IsValidBlendFactor(control.color_dst_factor) &&
           IsValidBlendFactor(control.alpha_src_factor) &&
           IsValidBlendFactor(control.alpha_dst_factor) &&
           static_cast<u32>(control.color_func) <=
               static_cast<u32>(AmdGpu::BlendControl::BlendFunc::ReverseSubtract) &&
           static_cast<u32>(control.alpha_func) <=
               static_cast<u32>(AmdGpu::BlendControl::BlendFunc::ReverseSubtract);
}

[[nodiscard]] constexpr bool IsValidLogicOp(AmdGpu::ColorControl::LogicOp op) noexcept {
    const u32 value = static_cast<u32>(op);
    return value <= 0xFF && (value & 0xFU) == (value >> 4);
}

[[nodiscard]] constexpr bool IsValidNumberFormat(AmdGpu::NumberFormat format) noexcept {
    using enum AmdGpu::NumberFormat;
    switch (format) {
    case Unorm:
    case Snorm:
    case Uscaled:
    case Sscaled:
    case Uint:
    case Sint:
    case SnormNz:
    case Float:
    case Srgb:
    case Ubnorm:
    case UbnormNz:
    case Ubint:
    case Ubscaled:
        return true;
    }
    return false;
}

[[nodiscard]] constexpr bool IsSupportedColorFormat(AmdGpu::DataFormat data,
                                                    AmdGpu::NumberFormat number) noexcept {
    using enum AmdGpu::DataFormat;
    using enum AmdGpu::NumberFormat;
    if (!IsValidNumberFormat(number)) {
        return false;
    }
    switch (data) {
    case FormatInvalid:
        return true;
    case Format8:
    case Format8_8:
    case Format8_8_8_8:
        return number == Unorm || number == Snorm || number == Uint || number == Sint ||
               number == Srgb;
    case Format16:
    case Format16_16:
        return number == Unorm || number == Snorm || number == Uint || number == Sint ||
               number == Float;
    case Format32:
    case Format32_32:
    case Format32_32_32:
    case Format32_32_32_32:
        return number == Uint || number == Sint || number == Float;
    case Format10_11_11:
    case Format5_9_9_9:
        return number == Float;
    case Format2_10_10_10:
        return number == Unorm || number == Snorm || number == Uint || number == Sint;
    case Format16_16_16_16:
        return number == Unorm || number == Snorm || number == SnormNz || number == Uint ||
               number == Sint || number == Float;
    case Format5_6_5:
    case Format1_5_5_5:
    case Format5_5_5_1:
    case Format4_4_4_4:
        return number == Unorm;
    case FormatBc1:
    case FormatBc2:
    case FormatBc3:
    case FormatBc7:
        return number == Unorm || number == Srgb;
    case FormatBc4:
    case FormatBc5:
    case FormatBc6:
        return number == Unorm || number == Snorm;
    default:
        return false;
    }
}

[[nodiscard]] constexpr bool IsValidCompMapping(AmdGpu::CompMapping mapping) noexcept {
    return std::ranges::all_of(mapping.array, [](AmdGpu::CompSwizzle component) {
        const u32 value = static_cast<u32>(component);
        return value <= 1 || (value >= 4 && value <= 7);
    });
}

[[nodiscard]] constexpr bool IsValidSampleCount(u32 samples, bool allow_zero = false) noexcept {
    return (allow_zero && samples == 0) ||
           (samples >= 1 && samples <= 16 && std::has_single_bit(samples));
}

[[nodiscard]] constexpr bool IsCoreFormat(vk::Format format, bool allow_undefined) noexcept {
    const auto value = static_cast<std::underlying_type_t<vk::Format>>(format);
    const auto last =
        static_cast<std::underlying_type_t<vk::Format>>(vk::Format::eAstc12x12SrgbBlock);
    return value >= (allow_undefined ? 0 : 1) && value <= last;
}

[[nodiscard]] bool ValidateVertexInputs(
    const GraphicsPipeline::SerializationSupport& support) noexcept {
    if (support.vertex_attributes.size() != support.vertex_bindings.size() ||
        support.divisors.size() > support.vertex_bindings.size()) {
        return false;
    }

    std::array<bool, MaxVertexBufferCount> bindings{};
    std::array<bool, MaxVertexBufferCount> locations{};
    for (const auto& binding : support.vertex_bindings) {
        if (binding.binding >= MaxVertexBufferCount || bindings[binding.binding] ||
            (binding.inputRate != vk::VertexInputRate::eVertex &&
             binding.inputRate != vk::VertexInputRate::eInstance)) {
            return false;
        }
        bindings[binding.binding] = true;
    }
    for (const auto& attribute : support.vertex_attributes) {
        if (attribute.location >= MaxVertexBufferCount ||
            attribute.binding >= MaxVertexBufferCount || locations[attribute.location] ||
            !bindings[attribute.binding] || !IsCoreFormat(attribute.format, false)) {
            return false;
        }
        locations[attribute.location] = true;
    }
    std::array<bool, MaxVertexBufferCount> divisor_bindings{};
    for (const auto& divisor : support.divisors) {
        if (divisor.binding >= MaxVertexBufferCount || !bindings[divisor.binding] ||
            divisor_bindings[divisor.binding]) {
            return false;
        }
        divisor_bindings[divisor.binding] = true;
    }
    return true;
}

void WriteCompMapping(Serialization::Writer& writer, AmdGpu::CompMapping mapping) {
    WriteEnum(writer, mapping.r);
    WriteEnum(writer, mapping.g);
    WriteEnum(writer, mapping.b);
    WriteEnum(writer, mapping.a);
}

[[nodiscard]] bool ReadCompMapping(Serialization::Reader& reader, AmdGpu::CompMapping& mapping) {
    const auto valid_component = [](u32 value) { return value <= 1 || (value >= 4 && value <= 7); };
    AmdGpu::CompSwizzle r{};
    AmdGpu::CompSwizzle g{};
    AmdGpu::CompSwizzle b{};
    AmdGpu::CompSwizzle a{};
    if (!ReadEnum(reader, r, valid_component) || !ReadEnum(reader, g, valid_component) ||
        !ReadEnum(reader, b, valid_component) || !ReadEnum(reader, a, valid_component)) {
        return false;
    }
    mapping = {.r = r, .g = g, .b = b, .a = a};
    return true;
}

void WriteColorBuffer(Serialization::Writer& writer, const Shader::PsColorBuffer& color) {
    WriteEnum(writer, color.data_format);
    WriteEnum(writer, color.num_format);
    WriteEnum(writer, color.num_conversion);
    WriteEnum(writer, color.export_format);
    WriteCompMapping(writer, color.swizzle);
}

[[nodiscard]] bool ReadColorBuffer(Serialization::Reader& reader, Shader::PsColorBuffer& color) {
    AmdGpu::DataFormat data_format{};
    AmdGpu::NumberFormat num_format{};
    AmdGpu::NumberConversion num_conversion{};
    AmdGpu::ShaderExportFormat export_format{};
    AmdGpu::CompMapping swizzle{};
    const auto valid_data_format = [](u32 value) {
        return value <= static_cast<u32>(AmdGpu::DataFormat::FormatBc7);
    };
    const auto valid_number_format = [](u32 value) {
        return IsValidNumberFormat(static_cast<AmdGpu::NumberFormat>(value));
    };
    if (!ReadEnum(reader, data_format, valid_data_format) ||
        !ReadEnum(reader, num_format, valid_number_format) ||
        !ReadEnum(reader, num_conversion,
                  [](u32 value) {
                      return value <= static_cast<u32>(AmdGpu::NumberConversion::SrgbToNorm);
                  }) ||
        !ReadEnum(reader, export_format,
                  [](u32 value) {
                      return value <= static_cast<u32>(AmdGpu::ShaderExportFormat::ABGR_32);
                  }) ||
        !ReadCompMapping(reader, swizzle) || !IsSupportedColorFormat(data_format, num_format)) {
        return false;
    }
    Shader::PsColorBuffer decoded{};
    decoded.data_format = data_format;
    decoded.num_format = num_format;
    decoded.num_conversion = num_conversion;
    decoded.export_format = export_format;
    decoded.swizzle = swizzle;
    color = decoded;
    return true;
}

void WriteBlendControl(Serialization::Writer& writer, const AmdGpu::BlendControl& blend) {
    WriteEnum(writer, blend.color_src_factor);
    WriteEnum(writer, blend.color_func);
    WriteEnum(writer, blend.color_dst_factor);
    WriteEnum(writer, blend.alpha_src_factor);
    WriteEnum(writer, blend.alpha_func);
    WriteEnum(writer, blend.alpha_dst_factor);
    WriteBool(writer, blend.separate_alpha_blend != 0);
    WriteBool(writer, blend.enable != 0);
    WriteBool(writer, blend.disable_rop3 != 0);
}

[[nodiscard]] bool ReadBlendControl(Serialization::Reader& reader, AmdGpu::BlendControl& blend) {
    AmdGpu::BlendControl::BlendFactor color_src{};
    AmdGpu::BlendControl::BlendFunc color_func{};
    AmdGpu::BlendControl::BlendFactor color_dst{};
    AmdGpu::BlendControl::BlendFactor alpha_src{};
    AmdGpu::BlendControl::BlendFunc alpha_func{};
    AmdGpu::BlendControl::BlendFactor alpha_dst{};
    bool separate_alpha{};
    bool enable{};
    bool disable_rop3{};
    const auto valid_factor = [](u32 value) {
        return IsValidBlendFactor(static_cast<AmdGpu::BlendControl::BlendFactor>(value));
    };
    const auto valid_func = [](u32 value) {
        return value <= static_cast<u32>(AmdGpu::BlendControl::BlendFunc::ReverseSubtract);
    };
    if (!ReadEnum(reader, color_src, valid_factor) || !ReadEnum(reader, color_func, valid_func) ||
        !ReadEnum(reader, color_dst, valid_factor) || !ReadEnum(reader, alpha_src, valid_factor) ||
        !ReadEnum(reader, alpha_func, valid_func) || !ReadEnum(reader, alpha_dst, valid_factor) ||
        !ReadBool(reader, separate_alpha) || !ReadBool(reader, enable) ||
        !ReadBool(reader, disable_rop3)) {
        return false;
    }
    AmdGpu::BlendControl decoded{};
    decoded.color_src_factor = color_src;
    decoded.color_func = color_func;
    decoded.color_dst_factor = color_dst;
    decoded.alpha_src_factor = alpha_src;
    decoded.alpha_func = alpha_func;
    decoded.alpha_dst_factor = alpha_dst;
    decoded.separate_alpha_blend = separate_alpha;
    decoded.enable = enable;
    decoded.disable_rop3 = disable_rop3;
    blend = decoded;
    return true;
}

void WriteVertexAttribute(Serialization::Writer& writer,
                          const vk::VertexInputAttributeDescription& attribute) {
    writer.Write(attribute.location);
    writer.Write(attribute.binding);
    WriteEnum(writer, attribute.format);
    writer.Write(attribute.offset);
}

[[nodiscard]] bool ReadVertexAttribute(Serialization::Reader& reader,
                                       vk::VertexInputAttributeDescription& attribute) {
    vk::VertexInputAttributeDescription decoded{};
    vk::Format format{};
    if (!reader.Read(decoded.location) || !reader.Read(decoded.binding) ||
        !ReadEnum(reader, format,
                  [](u32 value) { return IsCoreFormat(static_cast<vk::Format>(value), false); }) ||
        !reader.Read(decoded.offset)) {
        return false;
    }
    decoded.format = format;
    attribute = decoded;
    return true;
}

void WriteVertexBinding(Serialization::Writer& writer,
                        const vk::VertexInputBindingDescription& binding) {
    writer.Write(binding.binding);
    writer.Write(binding.stride);
    WriteEnum(writer, binding.inputRate);
}

[[nodiscard]] bool ReadVertexBinding(Serialization::Reader& reader,
                                     vk::VertexInputBindingDescription& binding) {
    vk::VertexInputBindingDescription decoded{};
    vk::VertexInputRate input_rate{};
    if (!reader.Read(decoded.binding) || !reader.Read(decoded.stride) ||
        !ReadEnum(reader, input_rate, [](u32 value) {
            return value <= static_cast<u32>(vk::VertexInputRate::eInstance);
        })) {
        return false;
    }
    decoded.inputRate = input_rate;
    binding = decoded;
    return true;
}

void WriteVertexDivisor(Serialization::Writer& writer,
                        const vk::VertexInputBindingDivisorDescriptionEXT& divisor) {
    writer.Write(divisor.binding);
    writer.Write(divisor.divisor);
}

[[nodiscard]] bool ReadVertexDivisor(Serialization::Reader& reader,
                                     vk::VertexInputBindingDivisorDescriptionEXT& divisor) {
    vk::VertexInputBindingDivisorDescriptionEXT decoded{};
    if (!reader.Read(decoded.binding) || !reader.Read(decoded.divisor)) {
        return false;
    }
    divisor = decoded;
    return true;
}

template <typename Container, typename WriteElement>
void WriteRecords(Serialization::Writer& writer, const Container& records,
                  WriteElement&& write_element) {
    WriteCount(writer, records.size());
    for (const auto& record : records) {
        write_element(writer, record);
    }
}

template <typename Container, typename ReadElement>
[[nodiscard]] bool ReadRecords(Serialization::Reader& reader, Container& records, u32 maximum,
                               ReadElement&& read_element) {
    u32 count{};
    if (!ReadCount(reader, count, maximum) || count > records.max_size()) {
        return false;
    }
    Container decoded{};
    try {
        decoded.reserve(count);
        for (u32 index = 0; index < count; ++index) {
            typename Container::value_type record{};
            if (!read_element(reader, record)) {
                return false;
            }
            decoded.emplace_back(record);
        }
    } catch (const std::exception&) {
        return false;
    }
    records = std::move(decoded);
    return true;
}

void WriteSpirv(Serialization::Writer& writer, std::span<const u32> code) {
    WriteCount(writer, code.size());
    for (const u32 word : code) {
        writer.Write(word);
    }
}

[[nodiscard]] bool ReadSpirv(Serialization::Reader& reader, std::vector<u32>& code) {
    u32 count{};
    if (!ReadCount(reader, count, MaxCachedShaderDwords) ||
        count > reader.ar.RemainingBytes() / sizeof(u32)) {
        return false;
    }
    std::vector<u32> decoded;
    try {
        decoded.resize(count);
    } catch (const std::exception&) {
        return false;
    }
    for (u32& word : decoded) {
        if (!reader.Read(word)) {
            return false;
        }
    }
    code = std::move(decoded);
    return true;
}

} // namespace

bool IsValidCachedSpirv(std::span<const u32> code) noexcept {
    constexpr u32 SpirvMagic = 0x07230203U;
    constexpr u32 MinimumVersion = 0x00010000U;
    constexpr u32 MaximumVersion = 0x00010600U;
    if (code.size() <= 5 || code.size() > MaxCachedShaderDwords || code[0] != SpirvMagic ||
        code[1] < MinimumVersion || code[1] > MaximumVersion || (code[1] & 0xFFU) != 0 ||
        code[2] != 0 || code[3] == 0 || code[3] > MaxCachedShaderDwords || code[4] != 0) {
        return false;
    }

    try {
        const u32 id_bound = code[3];
        const auto valid_id = [id_bound](u32 id) { return id != 0 && id < id_bound; };
        std::vector<u8> defined_ids(id_bound);
        std::vector<u16> type_opcodes(id_bound);
        std::vector<u8> function_ids(id_bound);
        std::vector<u8> global_variable_ids(id_bound);
        std::vector<u8> ext_inst_import_ids(id_bound);
        std::vector<u32> label_owners(id_bound);
        std::vector<u32> entry_point_ids;
        std::vector<u32> entry_point_interfaces;
        std::vector<u32> execution_mode_targets;
        std::vector<u32> function_call_targets;
        std::vector<u32> ext_inst_targets;
        std::vector<u32> deferred_id_targets;
        std::vector<std::pair<u32, u32>> branch_targets;

        SpirvModuleSection section = SpirvModuleSection::Capabilities;
        bool has_shader_capability{};
        bool has_memory_model{};
        bool has_function{};
        bool in_function{};
        bool has_label{};
        bool block_terminated{};
        bool is_first_block{};
        bool allow_local_variables{};
        u32 current_function{};

        for (size_t offset = 5; offset < code.size();) {
            const u16 opcode_value = static_cast<u16>(code[offset] & 0xFFFFU);
            const auto opcode = static_cast<spv::Op>(opcode_value);
            const size_t word_count = code[offset] >> 16;
            const SpirvOpcodeRule* rule = FindSpirvOpcodeRule(opcode_value);
            if (!rule || word_count < rule->minimum_words || word_count > code.size() - offset) {
                return false;
            }
            const std::span<const u32> instruction = code.subspan(offset, word_count);

            if (rule->has_result_type) {
                const u32 result_type = instruction[1];
                if (!valid_id(result_type) || type_opcodes[result_type] == 0) {
                    return false;
                }
            }
            u32 result_id{};
            if (rule->has_result) {
                result_id = instruction[rule->has_result_type ? 2 : 1];
                if (!valid_id(result_id) || defined_ids[result_id]) {
                    return false;
                }
                defined_ids[result_id] = 1;
                if (IsSpirvTypeOpcode(opcode)) {
                    type_opcodes[result_id] = opcode_value;
                }
            }

            if (!in_function) {
                if (section == SpirvModuleSection::Functions && opcode != spv::Op::OpFunction) {
                    return false;
                }
                switch (opcode) {
                case spv::Op::OpCapability:
                    if (word_count != 2 || !IsAllowedSpirvCapability(instruction[1]) ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::Capabilities)) {
                        return false;
                    }
                    has_shader_capability |=
                        instruction[1] == static_cast<u32>(spv::Capability::Shader);
                    break;
                case spv::Op::OpExtension: {
                    const auto extension = ReadSpirvLiteralString(instruction, 1);
                    if (!extension || extension->next_word != instruction.size() ||
                        !IsAllowedSpirvExtension(extension->value) ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::Extensions)) {
                        return false;
                    }
                    break;
                }
                case spv::Op::OpExtInstImport: {
                    const auto import = ReadSpirvLiteralString(instruction, 2);
                    if (!import || import->next_word != instruction.size() ||
                        !IsAllowedSpirvExtInstImport(import->value) ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::Imports)) {
                        return false;
                    }
                    ext_inst_import_ids[result_id] = 1;
                    break;
                }
                case spv::Op::OpMemoryModel:
                    if (word_count != 3 || has_memory_model ||
                        (instruction[1] != static_cast<u32>(spv::AddressingModel::Logical) &&
                         instruction[1] !=
                             static_cast<u32>(spv::AddressingModel::PhysicalStorageBuffer64)) ||
                        instruction[2] != static_cast<u32>(spv::MemoryModel::GLSL450) ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::MemoryModel)) {
                        return false;
                    }
                    has_memory_model = true;
                    break;
                case spv::Op::OpEntryPoint: {
                    const auto name = ReadSpirvLiteralString(instruction, 3);
                    if (!has_memory_model || !entry_point_ids.empty() || !name ||
                        name->value != "main" || !IsAllowedSpirvExecutionModel(instruction[1]) ||
                        !valid_id(instruction[2]) ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::EntryPoints)) {
                        return false;
                    }
                    entry_point_ids.push_back(instruction[2]);
                    for (size_t index = name->next_word; index < instruction.size(); ++index) {
                        if (!valid_id(instruction[index])) {
                            return false;
                        }
                        entry_point_interfaces.push_back(instruction[index]);
                    }
                    break;
                }
                case spv::Op::OpExecutionMode: {
                    const auto expected_words = ExpectedExecutionModeWordCount(instruction[2]);
                    if (entry_point_ids.empty() || !expected_words ||
                        word_count != *expected_words || !valid_id(instruction[1]) ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::ExecutionModes)) {
                        return false;
                    }
                    execution_mode_targets.push_back(instruction[1]);
                    break;
                }
                case spv::Op::OpString: {
                    const auto value = ReadSpirvLiteralString(instruction, 2);
                    if (!value || value->next_word != instruction.size() ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::Debug)) {
                        return false;
                    }
                    break;
                }
                case spv::Op::OpLine:
                    if (word_count != 4 || !valid_id(instruction[1]) ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::Debug)) {
                        return false;
                    }
                    deferred_id_targets.push_back(instruction[1]);
                    break;
                case spv::Op::OpName: {
                    const auto name = ReadSpirvLiteralString(instruction, 2);
                    if (!valid_id(instruction[1]) || !name ||
                        name->next_word != instruction.size() ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::Debug)) {
                        return false;
                    }
                    deferred_id_targets.push_back(instruction[1]);
                    break;
                }
                case spv::Op::OpMemberName: {
                    const auto name = ReadSpirvLiteralString(instruction, 3);
                    if (!valid_id(instruction[1]) || !name ||
                        name->next_word != instruction.size() ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::Debug)) {
                        return false;
                    }
                    deferred_id_targets.push_back(instruction[1]);
                    break;
                }
                case spv::Op::OpDecorate:
                case spv::Op::OpMemberDecorate:
                    if (!valid_id(instruction[1]) ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::Annotations)) {
                        return false;
                    }
                    deferred_id_targets.push_back(instruction[1]);
                    break;
                case spv::Op::OpFunction:
                    if (word_count != 5 || instruction[3] != 0 || !valid_id(instruction[4]) ||
                        type_opcodes[instruction[4]] != static_cast<u16>(spv::Op::OpTypeFunction) ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::Functions)) {
                        return false;
                    }
                    function_ids[result_id] = 1;
                    current_function = result_id;
                    has_function = true;
                    in_function = true;
                    has_label = false;
                    block_terminated = false;
                    is_first_block = true;
                    allow_local_variables = false;
                    break;
                default:
                    if ((!IsSpirvTypeOpcode(opcode) && !IsSpirvConstantOpcode(opcode) &&
                         opcode != spv::Op::OpVariable && opcode != spv::Op::OpUndef) ||
                        !AdvanceSpirvSection(section, SpirvModuleSection::Declarations)) {
                        return false;
                    }
                    if (opcode == spv::Op::OpVariable) {
                        global_variable_ids[result_id] = 1;
                    }
                    break;
                }
            } else {
                switch (opcode) {
                case spv::Op::OpFunction:
                    return false;
                case spv::Op::OpFunctionParameter:
                    if (word_count != 3 || has_label) {
                        return false;
                    }
                    break;
                case spv::Op::OpLabel:
                    if (word_count != 2 || (has_label && !block_terminated)) {
                        return false;
                    }
                    label_owners[result_id] = current_function;
                    has_label = true;
                    block_terminated = false;
                    allow_local_variables = is_first_block;
                    break;
                case spv::Op::OpFunctionEnd:
                    if (word_count != 1 || !has_label || !block_terminated) {
                        return false;
                    }
                    in_function = false;
                    current_function = 0;
                    break;
                default:
                    if (!has_label || block_terminated || IsSpirvTypeOpcode(opcode) ||
                        IsSpirvConstantOpcode(opcode) || IsSpirvDebugOpcode(opcode) ||
                        IsSpirvAnnotationOpcode(opcode) || opcode == spv::Op::OpCapability ||
                        opcode == spv::Op::OpExtension || opcode == spv::Op::OpExtInstImport ||
                        opcode == spv::Op::OpMemoryModel || opcode == spv::Op::OpEntryPoint ||
                        opcode == spv::Op::OpExecutionMode) {
                        return false;
                    }
                    if (opcode == spv::Op::OpVariable) {
                        if (!allow_local_variables) {
                            return false;
                        }
                    } else {
                        allow_local_variables = false;
                    }
                    if (opcode == spv::Op::OpFunctionCall) {
                        function_call_targets.push_back(instruction[3]);
                    } else if (opcode == spv::Op::OpExtInst) {
                        ext_inst_targets.push_back(instruction[3]);
                    } else if (opcode == spv::Op::OpBranch) {
                        branch_targets.emplace_back(instruction[1], current_function);
                    } else if (opcode == spv::Op::OpBranchConditional) {
                        if (word_count != 4 && word_count != 6) {
                            return false;
                        }
                        branch_targets.emplace_back(instruction[2], current_function);
                        branch_targets.emplace_back(instruction[3], current_function);
                    } else if (opcode == spv::Op::OpSwitch) {
                        if ((word_count - 3) % 2 != 0) {
                            return false;
                        }
                        branch_targets.emplace_back(instruction[2], current_function);
                        for (size_t index = 4; index < instruction.size(); index += 2) {
                            branch_targets.emplace_back(instruction[index], current_function);
                        }
                    } else if (opcode == spv::Op::OpPhi) {
                        if (word_count < 5 || (word_count - 3) % 2 != 0) {
                            return false;
                        }
                        for (size_t index = 4; index < instruction.size(); index += 2) {
                            branch_targets.emplace_back(instruction[index], current_function);
                        }
                    } else if (opcode == spv::Op::OpLoopMerge) {
                        branch_targets.emplace_back(instruction[1], current_function);
                        branch_targets.emplace_back(instruction[2], current_function);
                    } else if (opcode == spv::Op::OpSelectionMerge) {
                        branch_targets.emplace_back(instruction[1], current_function);
                    }
                    if (IsSpirvTerminatorOpcode(opcode)) {
                        block_terminated = true;
                        is_first_block = false;
                    }
                    break;
                }
            }
            offset += word_count;
        }

        if (!has_shader_capability || !has_memory_model || entry_point_ids.size() != 1 ||
            !has_function || in_function || !function_ids[entry_point_ids.front()]) {
            return false;
        }
        if (std::ranges::any_of(entry_point_interfaces,
                                [&](u32 id) { return !global_variable_ids[id]; }) ||
            std::ranges::any_of(execution_mode_targets,
                                [&](u32 id) { return id != entry_point_ids.front(); }) ||
            std::ranges::any_of(function_call_targets,
                                [&](u32 id) { return !valid_id(id) || !function_ids[id]; }) ||
            std::ranges::any_of(
                ext_inst_targets,
                [&](u32 id) { return !valid_id(id) || !ext_inst_import_ids[id]; }) ||
            std::ranges::any_of(deferred_id_targets, [&](u32 id) { return !defined_ids[id]; }) ||
            std::ranges::any_of(branch_targets, [&](const auto& target) {
                return !valid_id(target.first) || label_owners[target.first] != target.second;
            })) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

void SerializeShaderProfile(Serialization::Archive& ar, const Shader::Profile& profile) {
    Serialization::Writer writer{ar};
    writer.Write(ShaderProfileVersion);
    WriteProfileFields(writer, profile);
}

bool DeserializeShaderProfile(Serialization::Archive& ar, Shader::Profile& profile) {
    Serialization::Reader reader{ar};
    u32 version{};
    Shader::Profile decoded{};
    if (!reader.Read(version) || version != ShaderProfileVersion ||
        !ReadProfileFields(reader, decoded) || !reader.IsValid() || !ar.IsEoS()) {
        return false;
    }
    profile = decoded;
    return true;
}

bool ValidateGraphicsPipelineKey(const GraphicsPipelineKey& key) noexcept {
    const auto compute_slot = static_cast<u32>(Shader::LogicalStage::Compute);
    const auto vertex_slot = static_cast<u32>(Shader::LogicalStage::Vertex);
    const auto tcs_slot = static_cast<u32>(Shader::LogicalStage::TessellationControl);
    const auto tes_slot = static_cast<u32>(Shader::LogicalStage::TessellationEval);
    const bool has_tcs = key.stage_hashes[tcs_slot] != 0;
    const bool has_tes = key.stage_hashes[tes_slot] != 0;
    if (key.stage_hashes[compute_slot] != 0 || key.stage_hashes[vertex_slot] == 0 ||
        has_tcs != has_tes || !IsValidPrimitiveType(key.prim_type) ||
        static_cast<u32>(key.polygon_mode) > static_cast<u32>(AmdGpu::PolygonMode::Fill) ||
        static_cast<u32>(key.clip_space) > static_cast<u32>(AmdGpu::ClipSpace::ZeroToW) ||
        static_cast<u32>(key.provoking_vtx_last) >
            static_cast<u32>(AmdGpu::ProvokingVtxLast::Last) ||
        !IsValidLogicOp(key.logic_op) || !IsValidSampleCount(key.num_samples) ||
        !IsValidSampleCount(key.depth_samples) ||
        (key.z_format != AmdGpu::DepthBuffer::ZFormat::Invalid &&
         key.z_format != AmdGpu::DepthBuffer::ZFormat::Z16 &&
         key.z_format != AmdGpu::DepthBuffer::ZFormat::Z32Float) ||
        static_cast<u32>(key.stencil_format) >
            static_cast<u32>(AmdGpu::DepthBuffer::StencilFormat::Stencil8) ||
        key.num_color_attachments > AmdGpu::NUM_COLOR_BUFFERS ||
        (key.mrt_mask >> AmdGpu::NUM_COLOR_BUFFERS) != 0 ||
        key.num_color_attachments != std::bit_width(key.mrt_mask) ||
        key.patch_control_points > 32) {
        return false;
    }

    const bool is_patch = key.prim_type == AmdGpu::PrimitiveType::PatchPrimitive;
    if ((is_patch && (!has_tcs || key.patch_control_points == 0)) ||
        (!is_patch && key.patch_control_points != 0)) {
        return false;
    }

    u32 maximum_samples = key.depth_samples;
    for (size_t index = 0; index < AmdGpu::NUM_COLOR_BUFFERS; ++index) {
        const auto& color = key.color_buffers[index];
        if (!IsSupportedColorFormat(color.data_format, color.num_format) ||
            static_cast<u32>(color.num_conversion) >
                static_cast<u32>(AmdGpu::NumberConversion::SrgbToNorm) ||
            static_cast<u32>(color.export_format) >
                static_cast<u32>(AmdGpu::ShaderExportFormat::ABGR_32) ||
            !IsValidCompMapping(color.swizzle) || !IsValidBlendControl(key.blend_controls[index]) ||
            (static_cast<u32>(key.write_masks[index]) & ~0xFU) != 0 ||
            !IsValidSampleCount(key.color_samples[index], true)) {
            return false;
        }
        maximum_samples = std::max(maximum_samples, static_cast<u32>(key.color_samples[index]));
    }
    if (key.num_samples != maximum_samples) {
        return false;
    }
    return std::ranges::all_of(key.vertex_buffer_formats,
                               [](vk::Format format) { return IsCoreFormat(format, true); });
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer writer{ar};
    writer.Write(static_cast<u64>(value));
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader reader{ar};
    u64 decoded{};
    if (!reader.Read(decoded) || decoded > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(decoded);
    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer writer{ar};
    for (const size_t hash : stage_hashes) {
        writer.Write(static_cast<u64>(hash));
    }
    for (const vk::Format format : vertex_buffer_formats) {
        WriteEnum(writer, format);
    }
    writer.Write(patch_control_points);
    writer.Write(num_color_attachments);
    for (const auto& color : color_buffers) {
        WriteColorBuffer(writer, color);
    }
    for (const auto& blend : blend_controls) {
        WriteBlendControl(writer, blend);
    }
    for (const auto mask : write_masks) {
        writer.Write(static_cast<u32>(mask));
    }
    writer.Write(cb_shader_mask.raw);
    WriteEnum(writer, logic_op);
    writer.Write(num_samples);
    writer.Write(depth_samples);
    for (const u8 samples : color_samples) {
        writer.Write(samples);
    }
    writer.Write(mrt_mask);
    WriteEnum(writer, z_format);
    WriteEnum(writer, stencil_format);
    WriteBool(writer, depth_clamp_enable != 0);
    WriteEnum(writer, prim_type);
    WriteEnum(writer, polygon_mode);
    WriteEnum(writer, clip_space);
    WriteEnum(writer, provoking_vtx_last);
    WriteBool(writer, depth_clip_enable != 0);
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader reader{ar};
    GraphicsPipelineKey decoded{};
    for (size_t& hash : decoded.stage_hashes) {
        u64 encoded{};
        if (!reader.Read(encoded) || encoded > std::numeric_limits<size_t>::max()) {
            return false;
        }
        hash = static_cast<size_t>(encoded);
    }
    for (vk::Format& format : decoded.vertex_buffer_formats) {
        if (!ReadEnum(reader, format, [](u32 value) {
                return IsCoreFormat(static_cast<vk::Format>(value), true);
            })) {
            return false;
        }
    }
    if (!reader.Read(decoded.patch_control_points) || !reader.Read(decoded.num_color_attachments)) {
        return false;
    }
    for (auto& color : decoded.color_buffers) {
        if (!ReadColorBuffer(reader, color)) {
            return false;
        }
    }
    for (auto& blend : decoded.blend_controls) {
        if (!ReadBlendControl(reader, blend)) {
            return false;
        }
    }
    for (auto& mask : decoded.write_masks) {
        u32 encoded{};
        if (!reader.Read(encoded) || (encoded & ~0xFU) != 0) {
            return false;
        }
        mask = vk::ColorComponentFlags{encoded};
    }

    AmdGpu::ColorControl::LogicOp logic_op{};
    if (!reader.Read(decoded.cb_shader_mask.raw) ||
        !ReadEnum(reader, logic_op,
                  [](u32 value) {
                      return IsValidLogicOp(static_cast<AmdGpu::ColorControl::LogicOp>(value));
                  }) ||
        !reader.Read(decoded.num_samples) || !reader.Read(decoded.depth_samples)) {
        return false;
    }
    decoded.logic_op = logic_op;
    for (u8& samples : decoded.color_samples) {
        if (!reader.Read(samples)) {
            return false;
        }
    }

    AmdGpu::DepthBuffer::ZFormat z_format{};
    AmdGpu::DepthBuffer::StencilFormat stencil_format{};
    bool depth_clamp_enable{};
    AmdGpu::PrimitiveType primitive_type{};
    AmdGpu::PolygonMode polygon_mode{};
    AmdGpu::ClipSpace clip_space{};
    AmdGpu::ProvokingVtxLast provoking_vtx_last{};
    bool depth_clip_enable{};
    if (!reader.Read(decoded.mrt_mask) ||
        !ReadEnum(reader, z_format,
                  [](u32 value) {
                      const auto format = static_cast<AmdGpu::DepthBuffer::ZFormat>(value);
                      return format == AmdGpu::DepthBuffer::ZFormat::Invalid ||
                             format == AmdGpu::DepthBuffer::ZFormat::Z16 ||
                             format == AmdGpu::DepthBuffer::ZFormat::Z32Float;
                  }) ||
        !ReadEnum(reader, stencil_format,
                  [](u32 value) {
                      return value <=
                             static_cast<u32>(AmdGpu::DepthBuffer::StencilFormat::Stencil8);
                  }) ||
        !ReadBool(reader, depth_clamp_enable) ||
        !ReadEnum(reader, primitive_type,
                  [](u32 value) {
                      return IsValidPrimitiveType(static_cast<AmdGpu::PrimitiveType>(value));
                  }) ||
        !ReadEnum(reader, polygon_mode,
                  [](u32 value) { return value <= static_cast<u32>(AmdGpu::PolygonMode::Fill); }) ||
        !ReadEnum(
            reader, clip_space,
            [](u32 value) { return value <= static_cast<u32>(AmdGpu::ClipSpace::ZeroToW); }) ||
        !ReadEnum(
            reader, provoking_vtx_last,
            [](u32 value) { return value <= static_cast<u32>(AmdGpu::ProvokingVtxLast::Last); }) ||
        !ReadBool(reader, depth_clip_enable)) {
        return false;
    }
    decoded.z_format = z_format;
    decoded.stencil_format = stencil_format;
    decoded.depth_clamp_enable = depth_clamp_enable;
    decoded.prim_type = primitive_type;
    decoded.polygon_mode = polygon_mode;
    decoded.clip_space = clip_space;
    decoded.provoking_vtx_last = provoking_vtx_last;
    decoded.depth_clip_enable = depth_clip_enable;

    if (!reader.IsValid() || !ValidateGraphicsPipelineKey(decoded)) {
        return false;
    }
    *this = decoded;
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer writer{ar};
    WriteRecords(writer, vertex_attributes, WriteVertexAttribute);
    WriteRecords(writer, vertex_bindings, WriteVertexBinding);
    WriteRecords(writer, divisors, WriteVertexDivisor);
    writer.Write(static_cast<u32>(multisampling.flags));
    writer.Write(static_cast<u32>(multisampling.rasterizationSamples));
    writer.Write(static_cast<u32>(multisampling.sampleShadingEnable));
    writer.Write(std::bit_cast<u32>(multisampling.minSampleShading));
    writer.Write(static_cast<u32>(multisampling.alphaToCoverageEnable));
    writer.Write(static_cast<u32>(multisampling.alphaToOneEnable));
    WriteSpirv(writer, tcs);
    WriteSpirv(writer, tes);
    WriteSpirv(writer, fragment);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader reader{ar};
    SerializationSupport decoded{};
    u32 flags{};
    u32 rasterization_samples{};
    u32 sample_shading_enable{};
    u32 min_sample_shading_bits{};
    u32 alpha_to_coverage_enable{};
    u32 alpha_to_one_enable{};
    if (!ReadRecords(reader, decoded.vertex_attributes, MaxVertexBufferCount,
                     ReadVertexAttribute) ||
        !ReadRecords(reader, decoded.vertex_bindings, MaxVertexBufferCount, ReadVertexBinding) ||
        !ReadRecords(reader, decoded.divisors, MaxVertexBufferCount, ReadVertexDivisor) ||
        !reader.Read(flags) || !reader.Read(rasterization_samples) ||
        !reader.Read(sample_shading_enable) || !reader.Read(min_sample_shading_bits) ||
        !reader.Read(alpha_to_coverage_enable) || !reader.Read(alpha_to_one_enable) ||
        !ReadSpirv(reader, decoded.tcs) || !ReadSpirv(reader, decoded.tes) ||
        !ReadSpirv(reader, decoded.fragment)) {
        return false;
    }
    const float min_sample_shading = std::bit_cast<float>(min_sample_shading_bits);
    if (!reader.IsValid() || flags || rasterization_samples == 0 || rasterization_samples > 16 ||
        !std::has_single_bit(rasterization_samples) || sample_shading_enable > 1 ||
        alpha_to_coverage_enable > 1 || alpha_to_one_enable > 1 ||
        !std::isfinite(min_sample_shading) || min_sample_shading < 0.0f ||
        min_sample_shading > 1.0f || (!decoded.tcs.empty() && !IsValidCachedSpirv(decoded.tcs)) ||
        (!decoded.tes.empty() && !IsValidCachedSpirv(decoded.tes)) ||
        (!decoded.fragment.empty() && !IsValidCachedSpirv(decoded.fragment))) {
        return false;
    }
    decoded.multisampling = {
        .rasterizationSamples = static_cast<vk::SampleCountFlagBits>(rasterization_samples),
        .sampleShadingEnable = sample_shading_enable,
        .minSampleShading = min_sample_shading,
        .alphaToCoverageEnable = alpha_to_coverage_enable,
        .alphaToOneEnable = alpha_to_one_enable,
    };
    if (!ValidateVertexInputs(decoded)) {
        return false;
    }
    *this = std::move(decoded);
    return true;
}

bool DeserializeGraphicsPipelineCacheRecord(Serialization::Archive& ar, GraphicsPipelineKey& key,
                                            GraphicsPipeline::SerializationSupport& support) {
    GraphicsPipelineKey decoded_key{};
    GraphicsPipeline::SerializationSupport decoded_support{};
    if (!decoded_key.Deserialize(ar) || !decoded_support.Deserialize(ar) || !ar.IsEoS()) {
        return false;
    }
    key = decoded_key;
    support = std::move(decoded_support);
    return true;
}

} // namespace Vulkan
