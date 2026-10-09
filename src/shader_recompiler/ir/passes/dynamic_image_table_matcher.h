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

[[nodiscard]] inline bool IsAdjacentPointerPair(const IR::Inst& low,
                                                const IR::Inst& high) noexcept {
    if (low.GetOpcode() == IR::Opcode::GetUserData && high.GetOpcode() == IR::Opcode::GetUserData) {
        const u32 low_register = static_cast<u32>(low.Arg(0).ScalarReg());
        return low_register != std::numeric_limits<u32>::max() &&
               static_cast<u32>(high.Arg(0).ScalarReg()) == low_register + 1U;
    }
    if (low.GetOpcode() != IR::Opcode::ReadConst || high.GetOpcode() != IR::Opcode::ReadConst ||
        !low.Arg(1).IsImmediate() || !high.Arg(1).IsImmediate() ||
        low.Arg(1).U32() == std::numeric_limits<u32>::max() ||
        high.Arg(1).U32() != low.Arg(1).U32() + 1U) {
        return false;
    }
    return low.Arg(0).Resolve() == high.Arg(0).Resolve();
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
    boost::container::small_vector<IR::Inst*, 8> visited{&value};
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

        // Phi is the only generic merge whose operands all represent the same SSA value. Do not
        // guess through arithmetic, masks, selects, or composites here.
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
                visited.push_back(producer);
                queue.push(producer);
            }
        }
    }
    return unique_source;
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
