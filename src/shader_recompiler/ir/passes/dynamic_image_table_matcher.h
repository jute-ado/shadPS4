// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <queue>

#include <boost/container/small_vector.hpp>

#include "shader_recompiler/ir/passes/dynamic_image_table_policy.h"
#include "shader_recompiler/ir/value.h"

namespace Shader::Optimization {

struct DynamicImageTableMatch {
    IR::Inst* image_handle{};
    IR::Inst* pointer_lo{};
    IR::Inst* pointer_hi{};
    IR::Value index{};
    u32 byte_offset{};
    u32 byte_stride{};
    bool is_r128{};
    DynamicImageAccess access{};
    std::optional<u32> index_upper_bound{};

    [[nodiscard]] DynamicImageTablePattern Pattern(u32 base_pointer_location,
                                                   u32 supported_descriptor_window,
                                                   bool supports_nonuniform_indexing,
                                                   bool supports_oob_guard) const noexcept {
        return {
            .base_pointer_location = base_pointer_location,
            .byte_offset = byte_offset,
            .byte_stride = byte_stride,
            .is_r128 = is_r128,
            .access = access,
            .index_upper_bound = index_upper_bound,
            .supported_descriptor_window = supported_descriptor_window,
            .supports_nonuniform_indexing = supports_nonuniform_indexing,
            .supports_oob_guard = supports_oob_guard,
        };
    }
};

namespace Detail {

// Provenance is intentionally fail-closed when a malformed shader expands beyond this bounded
// graph budget. Keep both pointer-source and floating-use traversals bounded; neither may turn
// untrusted IR into an unbounded host-side allocation.
constexpr size_t MaxProvenanceWorklistNodes = 256U;

[[nodiscard]] inline std::optional<DynamicImageAccess> ClassifyDynamicImageAccess(
    const IR::Inst& access) noexcept {
    switch (access.GetOpcode()) {
    case IR::Opcode::ImageSampleRaw: {
        const auto info = access.Flags<IR::TextureInstInfo>();
        // Other raw forms lower to depth, gather, gradient, clamp, or dimension-query paths with
        // different backend semantics. Keep this admission contract deliberately narrow.
        if (info.is_depth || info.is_gather || info.has_derivatives || info.has_lod_clamp ||
            info.is_unnormalized) {
            return std::nullopt;
        }
        return (info.has_lod || info.force_level0) ? DynamicImageAccess::SampledExplicitLod
                                                   : DynamicImageAccess::SampledImplicitLod;
    }
    case IR::Opcode::ImageRead:
        return DynamicImageAccess::StorageRead;
    case IR::Opcode::ImageWrite:
        return DynamicImageAccess::StorageWrite;
    case IR::Opcode::ImageAtomicIAdd32:
    case IR::Opcode::ImageAtomicSMin32:
    case IR::Opcode::ImageAtomicUMin32:
    case IR::Opcode::ImageAtomicSMax32:
    case IR::Opcode::ImageAtomicUMax32:
    case IR::Opcode::ImageAtomicFMax32:
    case IR::Opcode::ImageAtomicFMin32:
    case IR::Opcode::ImageAtomicInc32:
    case IR::Opcode::ImageAtomicDec32:
    case IR::Opcode::ImageAtomicAnd32:
    case IR::Opcode::ImageAtomicOr32:
    case IR::Opcode::ImageAtomicXor32:
    case IR::Opcode::ImageAtomicExchange32:
    case IR::Opcode::ImageAtomicCmpSwap32:
        return DynamicImageAccess::Atomic;
    default:
        return std::nullopt;
    }
}

[[nodiscard]] inline bool IsHostReadableSrtAddress(IR::Value value, u32 depth = 0) noexcept;

// A ReadConst word can be shared by several consumers after IR construction. If one consumer
// interprets it as a floating-point value, the word's role is ambiguous and must not be used as
// a host descriptor-pointer half. Keep this conservative: an ambiguous source falls back to the
// ordinary runtime path, while a pointer-only source remains eligible for flattening.
[[nodiscard]] inline bool HasFloatingPointUse(const IR::Inst& value, u32 depth = 0) noexcept;

[[nodiscard]] inline bool IsAdjacentPointerPair(const IR::Inst& low,
                                                const IR::Inst& high) noexcept {
    if (low.GetOpcode() == IR::Opcode::GetUserData && high.GetOpcode() == IR::Opcode::GetUserData) {
        const u32 low_register = static_cast<u32>(low.Arg(0).ScalarReg());
        return low_register != std::numeric_limits<u32>::max() &&
               static_cast<u32>(high.Arg(0).ScalarReg()) == low_register + 1U &&
               !HasFloatingPointUse(low) && !HasFloatingPointUse(high);
    }
    if (low.GetOpcode() != IR::Opcode::ReadConst || high.GetOpcode() != IR::Opcode::ReadConst ||
        !low.Arg(1).IsImmediate() || !high.Arg(1).IsImmediate() ||
        low.Arg(1).U32() == std::numeric_limits<u32>::max() ||
        high.Arg(1).U32() != low.Arg(1).U32() + 1U) {
        return false;
    }
    // A ReadConst is only a host-readable SRT source when its address is itself built from
    // host-materializable scalar sources. In particular, do not mistake a pair read through a
    // GPU-derived pointer for the table base that the host SRT walker may snapshot.
    return low.Arg(0).Resolve() == high.Arg(0).Resolve() &&
           IsHostReadableSrtAddress(low.Arg(0)) && !HasFloatingPointUse(low) &&
           !HasFloatingPointUse(high);
}

[[nodiscard]] inline IR::Inst* FindUniquePointerSource(IR::Inst& value) noexcept {
    const auto is_pointer_source = [](const IR::Inst& inst) {
        return inst.GetOpcode() == IR::Opcode::GetUserData ||
               inst.GetOpcode() == IR::Opcode::ReadConst;
    };
    if (is_pointer_source(value)) {
        return &value;
    }

    IR::Inst* unique_source{};
    boost::container::small_vector<IR::Inst*, MaxProvenanceWorklistNodes> visited{&value};
    std::queue<IR::Inst*> queue;
    queue.push(&value);
    while (!queue.empty()) {
        IR::Inst* const inst = queue.front();
        queue.pop();
        if (is_pointer_source(*inst)) {
            if (unique_source && unique_source != inst) {
                return nullptr;
            }
            unique_source = inst;
            continue;
        }

        // An undefined U32 may take the sole defined source's value. Structurization can
        // introduce such leaves even when the original paths all initialize the pointer.
        // This still requires a concrete anchor below: all-undef/anchorless cycles reject,
        // as do distinct defined sources. Do not treat zero or other immediates as undef.
        if (inst->GetOpcode() == IR::Opcode::UndefU32) {
            continue;
        }

        // Only traverse Phi merges, never guess through arithmetic, masks, or selects.
        if (inst->GetOpcode() != IR::Opcode::Phi) {
            return nullptr;
        }
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            const IR::Value operand = inst->Arg(arg);
            if (operand.IsImmediate()) {
                return nullptr;
            }
            IR::Inst* const producer = operand.TryInstRecursive();
            if (!producer) {
                return nullptr;
            }
            if (std::ranges::find(visited, producer) == visited.end()) {
                if (visited.size() >= MaxProvenanceWorklistNodes) {
                    return nullptr;
                }
                visited.push_back(producer);
                queue.push(producer);
            }
        }
    }
    return unique_source;
}

[[nodiscard]] inline bool IsHostReadableSrtScalar(IR::Value value, u32 depth) noexcept;

[[nodiscard]] inline bool IsHostReadableSrtAddress(IR::Value value, u32 depth) noexcept {
    if (depth > 64U) {
        return false;
    }
    const auto* const pointer = value.TryInstRecursive();
    if (!pointer || pointer->GetOpcode() != IR::Opcode::CompositeConstructU32x2) {
        return false;
    }
    return IsHostReadableSrtScalar(pointer->Arg(0), depth + 1U) &&
           IsHostReadableSrtScalar(pointer->Arg(1), depth + 1U);
}

constexpr IR::Type FloatingPointTypes =
    IR::Type::F16 | IR::Type::F32 | IR::Type::F64 | IR::Type::F16x2 | IR::Type::F16x3 |
    IR::Type::F16x4 | IR::Type::F32x2 | IR::Type::F32x3 | IR::Type::F32x4 | IR::Type::F64x2 |
    IR::Type::F64x3 | IR::Type::F64x4;

[[nodiscard]] inline bool IsFloatingPointType(const IR::Type type) noexcept {
    return (type & FloatingPointTypes) != IR::Type::Void;
}

// Some integer-typed instructions explicitly reinterpret or numerically convert their U32
// operand to floating point. Their argument type alone is therefore not sufficient evidence.
[[nodiscard]] inline bool IsFloatingPointInterpretation(const IR::Inst& user,
                                                         size_t operand) noexcept {
    if (user.GetOpcode() != IR::Opcode::Phi && operand < user.NumArgs() &&
        IsFloatingPointType(IR::ArgTypeOf(user.GetOpcode(), operand))) {
        return true;
    }
    switch (user.GetOpcode()) {
    case IR::Opcode::BitCastF16U16:
    case IR::Opcode::BitCastF32U32:
    case IR::Opcode::ConvertF32U16:
    case IR::Opcode::ConvertF32S32:
    case IR::Opcode::ConvertF32U32:
    case IR::Opcode::ConvertF64S32:
    case IR::Opcode::ConvertF64U32:
    case IR::Opcode::UnpackUnorm2x16:
    case IR::Opcode::UnpackSnorm2x16:
    case IR::Opcode::UnpackUint2x16:
    case IR::Opcode::UnpackSint2x16:
    case IR::Opcode::UnpackHalf2x16:
    case IR::Opcode::UnpackUnorm4x8:
    case IR::Opcode::UnpackSnorm4x8:
    case IR::Opcode::UnpackUint4x8:
    case IR::Opcode::UnpackSint4x8:
    case IR::Opcode::UnpackUfloat10_11_11:
    case IR::Opcode::UnpackUnorm2_10_10_10:
    case IR::Opcode::UnpackSnorm2_10_10_10:
    case IR::Opcode::UnpackUint2_10_10_10:
    case IR::Opcode::UnpackSint2_10_10_10:
        return operand == 0U;
    case IR::Opcode::PackDouble2x32:
        return operand == 0U;
    default:
        return false;
    }
}

// Return true only for operands whose value is carried into the user's integer/composite result.
// Do not follow predicates, array/composite indices, memory addresses, or other control inputs:
// those can select a floating result without interpreting the pointer half as floating data.
[[nodiscard]] inline bool IsIntegerValueForwardingUse(const IR::Inst& user,
                                                      size_t operand) noexcept {
    switch (user.GetOpcode()) {
    case IR::Opcode::Identity:
        return operand == 0U;
    case IR::Opcode::Phi:
        return true;
    case IR::Opcode::SelectU32:
        return operand == 1U || operand == 2U;
    case IR::Opcode::CompositeConstructU32x2:
    case IR::Opcode::CompositeConstructU32x3:
    case IR::Opcode::CompositeConstructU32x4:
        return true;
    case IR::Opcode::CompositeConstructU32x2x2:
        return true;
    case IR::Opcode::PackUint2x32:
    case IR::Opcode::UnpackUint2x32:
        return operand == 0U;
    case IR::Opcode::CompositeExtractU32x2:
    case IR::Opcode::CompositeExtractU32x3:
    case IR::Opcode::CompositeExtractU32x4:
        return operand == 0U;
    case IR::Opcode::CompositeInsertU32x2:
    case IR::Opcode::CompositeInsertU32x3:
    case IR::Opcode::CompositeInsertU32x4:
        return operand == 0U || operand == 1U;
    case IR::Opcode::CompositeShuffleU32x2:
    case IR::Opcode::CompositeShuffleU32x3:
    case IR::Opcode::CompositeShuffleU32x4:
        return operand < 2U;
    case IR::Opcode::IAddCarry32:
        return operand < 2U;
    case IR::Opcode::IAdd64:
    case IR::Opcode::ISub64:
    case IR::Opcode::IMul64:
    case IR::Opcode::BitwiseAnd64:
    case IR::Opcode::BitwiseOr64:
        return operand < 2U;
    case IR::Opcode::INeg64:
    case IR::Opcode::BitCount64:
    case IR::Opcode::FindUMsb64:
    case IR::Opcode::FindILsb64:
        return operand == 0U;
    case IR::Opcode::ShiftLeftLogical64:
    case IR::Opcode::ShiftRightLogical64:
    case IR::Opcode::ShiftRightArithmetic64:
        return operand == 0U;
    case IR::Opcode::IAdd32:
    case IR::Opcode::ISub32:
    case IR::Opcode::IMul32:
    case IR::Opcode::SMulHi:
    case IR::Opcode::UMulHi:
    case IR::Opcode::SDiv32:
    case IR::Opcode::UDiv32:
    case IR::Opcode::SMod32:
    case IR::Opcode::UMod32:
    case IR::Opcode::INeg32:
    case IR::Opcode::IAbs32:
    case IR::Opcode::ShiftLeftLogical32:
    case IR::Opcode::ShiftRightLogical32:
    case IR::Opcode::ShiftRightArithmetic32:
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::BitwiseOr32:
    case IR::Opcode::BitwiseXor32:
    case IR::Opcode::BitwiseNot32:
    case IR::Opcode::BitFieldInsert:
    case IR::Opcode::BitFieldSExtract:
    case IR::Opcode::BitFieldUExtract:
    case IR::Opcode::BitReverse32:
    case IR::Opcode::BitCount32:
    case IR::Opcode::FindSMsb32:
    case IR::Opcode::FindUMsb32:
    case IR::Opcode::FindILsb32:
    case IR::Opcode::SMin32:
    case IR::Opcode::UMin32:
    case IR::Opcode::SMax32:
    case IR::Opcode::UMax32:
    case IR::Opcode::SMinTri32:
    case IR::Opcode::UMinTri32:
    case IR::Opcode::SMaxTri32:
    case IR::Opcode::UMaxTri32:
    case IR::Opcode::SMedTri32:
    case IR::Opcode::UMedTri32:
    case IR::Opcode::SClamp32:
    case IR::Opcode::UClamp32:
        return operand < user.NumArgs() && IR::ArgTypeOf(user.GetOpcode(), operand) == IR::Type::U32;
    case IR::Opcode::ConvertU16U32:
    case IR::Opcode::ConvertU32U16:
    case IR::Opcode::ConvertU64U32:
    case IR::Opcode::ConvertU8U32:
    case IR::Opcode::ConvertU32U8:
    case IR::Opcode::ConvertS32S8:
    case IR::Opcode::ConvertS32S16:
        return operand == 0U;
    default:
        return false;
    }
}

[[nodiscard]] inline bool HasFloatingPointUse(const IR::Inst& value, u32 depth) noexcept {
    if (depth > 64U) {
        return true;
    }

    // A worklist keeps cyclic Phi/Identity graphs bounded while allowing all value-carrying paths
    // to be inspected. The cap is deliberately conservative: an incomplete provenance proof must
    // decline host flattening rather than risk dereferencing guest payload as an address.
    boost::container::small_vector<const IR::Inst*, MaxProvenanceWorklistNodes> pending{&value};
    boost::container::small_vector<const IR::Inst*, MaxProvenanceWorklistNodes> visited;
    visited.reserve(MaxProvenanceWorklistNodes);
    while (!pending.empty()) {
        const IR::Inst* const current = pending.back();
        pending.pop_back();
        if (std::ranges::find(visited, current) != visited.end()) {
            continue;
        }
        if (visited.size() >= MaxProvenanceWorklistNodes) {
            return true;
        }
        visited.push_back(current);
        for (const auto& use : current->Uses()) {
            const IR::Inst* const user = use.user;
            if (!user) {
                return true;
            }
            // Frontend scalar/vector moves can leave an unused bitcast before DCE. Such a
            // pure leaf does not interpret any runtime value and cannot change pointer-role
            // admission. This is deliberately not whole-subgraph liveness analysis.
            if (!user->HasUses() && !user->MayHaveSideEffects()) {
                continue;
            }
            if (IsFloatingPointInterpretation(*user, use.operand)) {
                return true;
            }
            if (IsIntegerValueForwardingUse(*user, use.operand) &&
                std::ranges::find(visited, user) == visited.end()) {
                // A high-fanout Phi can expose the same user through hundreds of operands before
                // any queued node is visited. Bound the pending queue as well as the visited set;
                // an incomplete provenance proof must conservatively decline host flattening.
                if (pending.size() >= MaxProvenanceWorklistNodes) {
                    return true;
                }
                pending.push_back(user);
            }
        }
    }
    return false;
}

[[nodiscard]] inline bool IsHostReadableSrtScalar(IR::Value value, u32 depth) noexcept {
    if (depth > 64U) {
        return false;
    }
    value = value.Resolve();
    if (value.IsImmediate()) {
        return value.Type() == IR::Type::U32 || value.Type() == IR::Type::U1;
    }
    auto* const inst = value.TryInstRecursive();
    if (!inst) {
        return false;
    }
    switch (inst->GetOpcode()) {
    case IR::Opcode::GetUserData:
        return static_cast<u32>(inst->Arg(0).ScalarReg()) != std::numeric_limits<u32>::max();
    case IR::Opcode::ReadConst:
        return inst->Arg(1).IsImmediate() && IsHostReadableSrtAddress(inst->Arg(0), depth + 1U);
    case IR::Opcode::IAdd32:
    case IR::Opcode::ISub32:
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::ULessThan32:
        return IsHostReadableSrtScalar(inst->Arg(0), depth + 1U) &&
               IsHostReadableSrtScalar(inst->Arg(1), depth + 1U);
    case IR::Opcode::SelectU32:
        return IsHostReadableSrtScalar(inst->Arg(0), depth + 1U) &&
               IsHostReadableSrtScalar(inst->Arg(1), depth + 1U) &&
               IsHostReadableSrtScalar(inst->Arg(2), depth + 1U);
    case IR::Opcode::Phi: {
        auto* const source = FindUniquePointerSource(*inst);
        return source && IsHostReadableSrtScalar(IR::Value{source}, depth + 1U);
    }
    default:
        // ReadFirstLane, LDS/global loads, lane IDs, and opaque arithmetic all remain GPU-side.
        return false;
    }
}

struct AffineImageOffset {
    IR::Value index{};
    u32 byte_offset{};
    u32 byte_stride{};
};

[[nodiscard]] inline std::optional<AffineImageOffset> MatchAffineImageOffset(
    IR::Value value) noexcept {
    const auto* const add = value.TryInstRecursive();
    if (!add || add->GetOpcode() != IR::Opcode::IAdd32) {
        return std::nullopt;
    }
    const IR::Value add_lhs = add->Arg(0);
    const IR::Value add_rhs = add->Arg(1);
    const IR::Value byte_offset = add_lhs.IsImmediate() ? add_lhs : add_rhs;
    const IR::Value scaled_index = add_lhs.IsImmediate() ? add_rhs : add_lhs;
    if (!byte_offset.IsImmediate() || scaled_index.IsImmediate()) {
        return std::nullopt;
    }
    const auto* const multiply = scaled_index.TryInstRecursive();
    if (!multiply || multiply->GetOpcode() != IR::Opcode::IMul32) {
        return std::nullopt;
    }
    const IR::Value mul_lhs = multiply->Arg(0);
    const IR::Value mul_rhs = multiply->Arg(1);
    const IR::Value byte_stride = mul_lhs.IsImmediate() ? mul_lhs : mul_rhs;
    const IR::Value index = mul_lhs.IsImmediate() ? mul_rhs : mul_lhs;
    if (!byte_stride.IsImmediate()) {
        return std::nullopt;
    }
    return AffineImageOffset{index, byte_offset.U32(), byte_stride.U32()};
}

struct AffineImagePointer {
    IR::Inst* base_low{};
    IR::Inst* base_high{};
    AffineImageOffset offset{};
};

[[nodiscard]] inline std::optional<AffineImagePointer> MatchAffineImagePointer(
    const IR::Inst& low, const IR::Inst& high) noexcept {
    if (low.GetOpcode() != IR::Opcode::IAdd32 || high.GetOpcode() != IR::Opcode::IAdd32) {
        return std::nullopt;
    }
    for (u32 base_arg = 0; base_arg < 2U; ++base_arg) {
        auto* const base_low_value = low.Arg(base_arg).TryInstRecursive();
        const auto offset = MatchAffineImageOffset(low.Arg(1U - base_arg));
        if (!base_low_value || !offset) {
            continue;
        }
        auto* const base_low = FindUniquePointerSource(*base_low_value);
        if (!base_low) {
            continue;
        }
        for (u32 high_arg = 0; high_arg < 2U; ++high_arg) {
            auto* const base_high_value = high.Arg(high_arg).TryInstRecursive();
            const auto* const carry = high.Arg(1U - high_arg).TryInstRecursive();
            if (!base_high_value || !carry || carry->GetOpcode() != IR::Opcode::SelectU32 ||
                !carry->Arg(1).IsImmediate() || carry->Arg(1).U32() != 1U ||
                !carry->Arg(2).IsImmediate() || carry->Arg(2).U32() != 0U) {
                continue;
            }
            auto* const base_high = FindUniquePointerSource(*base_high_value);
            const auto* const comparison = carry->Arg(0).TryInstRecursive();
            // Unsigned sum < either original addend is the carry of this exact low add. A
            // nearby source, unrelated sum or reversed comparison is not equivalent.
            if (!base_high || !IsAdjacentPointerPair(*base_low, *base_high) || !comparison ||
                comparison->GetOpcode() != IR::Opcode::ULessThan32 ||
                comparison->Arg(0).TryInstRecursive() != &low ||
                (comparison->Arg(1).Resolve() != low.Arg(0).Resolve() &&
                 comparison->Arg(1).Resolve() != low.Arg(1).Resolve())) {
                continue;
            }
            return AffineImagePointer{base_low, base_high, *offset};
        }
    }
    return std::nullopt;
}

} // namespace Detail

[[nodiscard]] inline std::optional<u32> DynamicImageIndexUpperBound(
    const IR::Value& index) noexcept {
    if (index.IsImmediate()) {
        return index.U32();
    }
    IR::Inst* const producer = index.TryInstRecursive();
    if (!producer) {
        return std::nullopt;
    }
    switch (producer->GetOpcode()) {
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::UMin32:
        if (producer->Arg(0).IsImmediate()) {
            return producer->Arg(0).U32();
        }
        if (producer->Arg(1).IsImmediate()) {
            return producer->Arg(1).U32();
        }
        return std::nullopt;
    default:
        return std::nullopt;
    }
}

[[nodiscard]] inline std::optional<DynamicImageTableMatch> MatchDynamicImageTable(
    IR::Inst& access) noexcept {
    const auto access_kind = Detail::ClassifyDynamicImageAccess(access);
    if (!access_kind || access.Arg(0).IsImmediate()) {
        return std::nullopt;
    }

    IR::Inst* const image_handle = access.Arg(0).TryInstRecursive();
    if (!image_handle || image_handle->GetOpcode() != IR::Opcode::ReadConst ||
        image_handle->Arg(0).IsImmediate()) {
        return std::nullopt;
    }
    IR::Inst* const pointer = image_handle->Arg(0).TryInstRecursive();
    if (!pointer || pointer->GetOpcode() != IR::Opcode::CompositeConstructU32x2 ||
        pointer->Arg(0).IsImmediate() || pointer->Arg(1).IsImmediate()) {
        return std::nullopt;
    }
    IR::Inst* const pointer_lo_value = pointer->Arg(0).TryInstRecursive();
    IR::Inst* const pointer_hi_value = pointer->Arg(1).TryInstRecursive();
    if (!pointer_lo_value || !pointer_hi_value) {
        return std::nullopt;
    }
    IR::Inst* pointer_lo{};
    IR::Inst* pointer_hi{};
    std::optional<Detail::AffineImageOffset> offset;
    if (image_handle->Arg(1).IsImmediate()) {
        const auto affine_pointer =
            Detail::MatchAffineImagePointer(*pointer_lo_value, *pointer_hi_value);
        if (!affine_pointer) {
            return std::nullopt;
        }
        pointer_lo = affine_pointer->base_low;
        pointer_hi = affine_pointer->base_high;
        offset = affine_pointer->offset;
        const u64 combined_offset =
            u64{offset->byte_offset} + u64{image_handle->Arg(1).U32()} * sizeof(u32);
        if (combined_offset > std::numeric_limits<u32>::max()) {
            return std::nullopt;
        }
        offset->byte_offset = static_cast<u32>(combined_offset);
        // The multiply/constant add wraps in U32 before the explicit carry into the base. The
        // table policy must still prove no such offset wrap inside its admitted window; entries
        // outside that host window use the existing GPU guard, never a guessed host address.
    } else {
        pointer_lo = Detail::FindUniquePointerSource(*pointer_lo_value);
        pointer_hi = Detail::FindUniquePointerSource(*pointer_hi_value);
        if (!pointer_lo || !pointer_hi ||
            !Detail::IsAdjacentPointerPair(*pointer_lo, *pointer_hi)) {
            return std::nullopt;
        }
        const auto* const shift = image_handle->Arg(1).TryInstRecursive();
        if (!shift || shift->GetOpcode() != IR::Opcode::ShiftRightLogical32 ||
            !shift->Arg(1).IsImmediate() || shift->Arg(1).U32() != 2U ||
            shift->Arg(0).IsImmediate()) {
            return std::nullopt;
        }
        offset = Detail::MatchAffineImageOffset(shift->Arg(0));
        if (!offset) {
            return std::nullopt;
        }
    }

    const auto texture_info = access.Flags<IR::TextureInstInfo>();
    return DynamicImageTableMatch{
        .image_handle = image_handle,
        .pointer_lo = pointer_lo,
        .pointer_hi = pointer_hi,
        .index = offset->index,
        .byte_offset = offset->byte_offset,
        .byte_stride = offset->byte_stride,
        .is_r128 = bool(texture_info.is_r128),
        .access = *access_kind,
        .index_upper_bound = DynamicImageIndexUpperBound(offset->index),
    };
}

} // namespace Shader::Optimization
