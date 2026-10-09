// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <optional>
#include <queue>

#include <boost/container/small_vector.hpp>

#include "shader_recompiler/ir/passes/dynamic_image_table_policy.h"
#include "shader_recompiler/ir/passes/sharp_source_traversal.h"
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
        // Dynamic table lowering currently implements only ordinary sampling. Fail closed for
        // raw forms that resource tracking would lower to depth, gather, gradient, clamp, or
        // dimension-query paths with different backend opcodes.
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
        return static_cast<u32>(high.Arg(0).ScalarReg()) ==
               static_cast<u32>(low.Arg(0).ScalarReg()) + 1U;
    }
    if (low.GetOpcode() != IR::Opcode::ReadConst || high.GetOpcode() != IR::Opcode::ReadConst ||
        !low.Arg(1).IsImmediate() || !high.Arg(1).IsImmediate() ||
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

        if (inst->GetOpcode() == IR::Opcode::MaterializeScalarMaskWord &&
            (!inst->Arg(0).IsImmediate() || inst->Arg(0).U1())) {
            return nullptr;
        }

        const auto range = SharpSourceTraversalRange(inst->GetOpcode(), inst->NumArgs());
        if (range.begin == range.end) {
            return nullptr;
        }
        for (size_t arg = range.begin; arg < range.end; ++arg) {
            const IR::Value operand = inst->Arg(arg);
            if (operand.IsImmediate()) {
                return nullptr;
            }
            IR::Inst* const producer = operand.TryInstRecursive();
            if (!producer) {
                return nullptr;
            }
            if (!std::ranges::contains(visited, producer)) {
                visited.push_back(producer);
                queue.push(producer);
            }
        }
    }
    return unique_source;
}

[[nodiscard]] inline std::optional<u32> IndexUpperBound(const IR::Value& index) noexcept {
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

} // namespace Detail

[[nodiscard]] inline std::optional<DynamicImageTableMatch> MatchDynamicImageTable(
    IR::Inst& access) noexcept {
    const auto access_kind = Detail::ClassifyDynamicImageAccess(access);
    if (!access_kind || access.Arg(0).IsImmediate()) {
        return std::nullopt;
    }

    IR::Inst* const image_handle = access.Arg(0).TryInstRecursive();
    if (!image_handle || image_handle->GetOpcode() != IR::Opcode::ReadConst ||
        image_handle->Arg(0).IsImmediate() || image_handle->Arg(1).IsImmediate()) {
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
    IR::Inst* const pointer_lo = Detail::FindUniquePointerSource(*pointer_lo_value);
    IR::Inst* const pointer_hi = Detail::FindUniquePointerSource(*pointer_hi_value);
    if (!pointer_lo || !pointer_hi || !Detail::IsAdjacentPointerPair(*pointer_lo, *pointer_hi)) {
        return std::nullopt;
    }

    IR::Inst* const shift = image_handle->Arg(1).TryInstRecursive();
    if (!shift || shift->GetOpcode() != IR::Opcode::ShiftRightLogical32 ||
        !shift->Arg(1).IsImmediate() || shift->Arg(1).U32() != 2U || shift->Arg(0).IsImmediate()) {
        return std::nullopt;
    }
    IR::Inst* const add = shift->Arg(0).TryInstRecursive();
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
    IR::Inst* const multiply = scaled_index.TryInstRecursive();
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

    const auto texture_info = access.Flags<IR::TextureInstInfo>();
    return DynamicImageTableMatch{
        .image_handle = image_handle,
        .pointer_lo = pointer_lo,
        .pointer_hi = pointer_hi,
        .index = index,
        .byte_offset = byte_offset.U32(),
        .byte_stride = byte_stride.U32(),
        .is_r128 = bool(texture_info.is_r128),
        .access = *access_kind,
        .index_upper_bound = Detail::IndexUpperBound(index),
    };
}

} // namespace Shader::Optimization
