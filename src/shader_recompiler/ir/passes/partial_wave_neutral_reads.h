// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Optimization::PartialWave {

// This does not emulate missing invocations. It recovers a scalar lane read only when
// the guest explicitly initialized that inactive lane and every contributing operation
// preserves its known value. Arbitrary inactive VGPR contents remain unknown.
class NeutralReadEvaluator {
public:
    NeutralReadEvaluator(IR::Value initial_mask_, std::array<u32, 3> shape_, u32 count_)
        : initial_mask{initial_mask_}, shape{shape_}, count{count_} {}

    bool HasInitialParticipationPredicate() {
        if (initial_mask.IsImmediate()) {
            return false;
        }
        for (u32 lane = 0; lane < 64U; ++lane) {
            const auto result = EvaluateInitial(initial_mask, lane, 0);
            if (!result || *result != u32(lane < count)) {
                return false;
            }
        }
        return true;
    }

    bool IsFullExec(const IR::Value& value) {
        for (u32 lane = 0; lane < 64U; ++lane) {
            const auto result = EvaluateMask(value, lane, 0);
            if (!result || !result->value) {
                return false;
            }
        }
        return true;
    }

    std::optional<u32> Recover(IR::Inst& read, const IR::Block& block,
                               const std::unordered_set<const IR::Inst*>& full_exec) {
        if (read.GetOpcode() != IR::Opcode::ReadLane || !read.HasUses() ||
            !read.Arg(1).IsImmediate()) {
            return std::nullopt;
        }
        const u32 lane = read.Arg(1).U32();
        if (lane < count || lane >= 64U) {
            return std::nullopt;
        }
        payload_block = &block;
        full_exec_instructions = &full_exec;
        payload_nodes = 0;
        for (auto& cache : payload_cache) {
            cache.clear();
        }
        for (auto& visiting : payload_visiting) {
            visiting.clear();
        }
        const auto result = EvaluatePayload(read.Arg(0), lane, 0);
        return result && result->initialized ? std::optional{result->bits} : std::nullopt;
    }

private:
    static constexpr u32 MaxDepth = 96;
    static constexpr u32 MaxNodes = 4096;
    static constexpr u32 MaxMaskNodes = 16384;

    struct MaskValue {
        bool value;
        bool from_initial_mask;
    };
    struct PayloadValue {
        u32 bits;
        bool initialized;
    };

    std::optional<u32> EvaluateInitial(const IR::Value& value, u32 lane, u32 depth) {
        if (depth > 24U) {
            return std::nullopt;
        }
        if (value.IsImmediate()) {
            if (value.Type() == IR::Type::U1) {
                return u32(value.U1());
            }
            return value.Type() == IR::Type::U32 ? std::optional{value.U32()} : std::nullopt;
        }
        const auto* const inst = value.TryInstRecursive();
        if (inst == nullptr) {
            return std::nullopt;
        }
        if (const auto found = initial_cache[lane].find(inst); found != initial_cache[lane].end()) {
            return found->second;
        }
        if (initial_nodes >= MaxNodes) {
            return std::nullopt;
        }
        ++initial_nodes;
        const auto result = [&]() -> std::optional<u32> {
            if (inst->GetOpcode() == IR::Opcode::GetAttributeU32) {
                if (inst->Arg(0).Attribute() != IR::Attribute::LocalInvocationId ||
                    inst->Arg(1).U32() >= 3U) {
                    return std::nullopt;
                }
                const std::array local{lane % shape[0], (lane / shape[0]) % shape[1],
                                       lane / (shape[0] * shape[1])};
                return local[inst->Arg(1).U32()];
            }
            if (inst->NumArgs() != 2U) {
                return std::nullopt;
            }
            const auto lhs = EvaluateInitial(inst->Arg(0), lane, depth + 1U);
            const auto rhs = EvaluateInitial(inst->Arg(1), lane, depth + 1U);
            if (!lhs || !rhs) {
                return std::nullopt;
            }
            switch (inst->GetOpcode()) {
            case IR::Opcode::IAdd32:
                return *lhs + *rhs;
            case IR::Opcode::IMul32:
                return *lhs * *rhs;
            case IR::Opcode::ULessThan32:
                return u32(*lhs < *rhs);
            default:
                return std::nullopt;
            }
        }();
        initial_cache[lane].emplace(inst, result);
        return result;
    }

    std::optional<MaskValue> EvaluateMask(const IR::Value& value, u32 lane, u32 depth) {
        if (depth > MaxDepth) {
            return std::nullopt;
        }
        if (value.IsImmediate()) {
            return value.Type() == IR::Type::U1 ? std::optional{MaskValue{value.U1(), false}}
                                                : std::nullopt;
        }
        const auto* const inst = value.TryInstRecursive();
        if (inst == nullptr) {
            return std::nullopt;
        }
        if (inst == initial_mask.TryInstRecursive()) {
            return MaskValue{lane < count, true};
        }
        if (const auto found = mask_cache[lane].find(inst); found != mask_cache[lane].end()) {
            return found->second;
        }
        if (mask_nodes >= MaxMaskNodes || !mask_visiting[lane].insert(inst).second) {
            return std::nullopt;
        }
        ++mask_nodes;
        const auto result = [&]() -> std::optional<MaskValue> {
            if (inst->GetOpcode() == IR::Opcode::Phi) {
                std::optional<MaskValue> known;
                for (u32 index = 0; index < inst->NumArgs(); ++index) {
                    const auto incoming = inst->Arg(index);
                    // Only a direct self-backedge is admitted. Mutual or seedless cycles are
                    // not evidence; every other edge must prove the same defined value.
                    if (!incoming.IsImmediate() && incoming.TryInstRecursive() == inst) {
                        continue;
                    }
                    const auto evaluated = EvaluateMask(incoming, lane, depth + 1U);
                    if (!evaluated || (known && known->value != evaluated->value)) {
                        return std::nullopt;
                    }
                    known = MaskValue{evaluated->value, evaluated->from_initial_mask ||
                                                            (known && known->from_initial_mask)};
                }
                return known;
            }
            if (inst->GetOpcode() != IR::Opcode::LogicalNot &&
                inst->GetOpcode() != IR::Opcode::LogicalAnd &&
                inst->GetOpcode() != IR::Opcode::LogicalOr &&
                inst->GetOpcode() != IR::Opcode::LogicalXor) {
                return std::nullopt;
            }
            const auto lhs = EvaluateMask(inst->Arg(0), lane, depth + 1U);
            if (!lhs) {
                return std::nullopt;
            }
            if (inst->GetOpcode() == IR::Opcode::LogicalNot) {
                return MaskValue{!lhs->value, lhs->from_initial_mask};
            }
            const auto rhs = EvaluateMask(inst->Arg(1), lane, depth + 1U);
            if (!rhs) {
                return std::nullopt;
            }
            const bool result =
                inst->GetOpcode() == IR::Opcode::LogicalAnd  ? lhs->value && rhs->value
                : inst->GetOpcode() == IR::Opcode::LogicalOr ? lhs->value || rhs->value
                                                             : lhs->value != rhs->value;
            return MaskValue{result, lhs->from_initial_mask || rhs->from_initial_mask};
        }();
        mask_visiting[lane].erase(inst);
        mask_cache[lane].emplace(inst, result);
        return result;
    }

    std::optional<PayloadValue> EvaluatePayload(const IR::Value& value, u32 lane, u32 depth) {
        if (depth > MaxDepth || lane < count || lane >= 64U) {
            return std::nullopt;
        }
        if (value.IsImmediate()) {
            if (value.Type() == IR::Type::U32) {
                return PayloadValue{value.U32(), false};
            }
            return value.Type() == IR::Type::F32
                       ? std::optional{PayloadValue{std::bit_cast<u32>(value.F32()), false}}
                       : std::nullopt;
        }
        const auto* const inst = value.TryInstRecursive();
        if (inst == nullptr || inst->GetParent() != payload_block) {
            return std::nullopt;
        }
        const bool representation_only = inst->GetOpcode() == IR::Opcode::BitCastF32U32 ||
                                         inst->GetOpcode() == IR::Opcode::BitCastU32F32;
        // These IR casts reinterpret an existing register; they do not execute a new guest
        // VGPR write. Their underlying value still needs a proven full-EXEC initializer.
        if (!representation_only && !full_exec_instructions->contains(inst)) {
            return std::nullopt;
        }
        if (const auto found = payload_cache[lane].find(inst); found != payload_cache[lane].end()) {
            return found->second;
        }
        if (payload_nodes >= MaxNodes || !payload_visiting[lane].insert(inst).second) {
            return std::nullopt;
        }
        ++payload_nodes;
        const auto result = [&]() -> std::optional<PayloadValue> {
            switch (inst->GetOpcode()) {
            case IR::Opcode::LaneId:
                return PayloadValue{lane, false};
            case IR::Opcode::BitCastF32U32:
            case IR::Opcode::BitCastU32F32:
                return EvaluatePayload(inst->Arg(0), lane, depth + 1U);
            case IR::Opcode::SelectF32:
            case IR::Opcode::SelectU32: {
                const auto mask = EvaluateMask(inst->Arg(0), lane, 0);
                if (!mask || mask->value || !mask->from_initial_mask) {
                    return std::nullopt;
                }
                const auto selected = EvaluatePayload(inst->Arg(2), lane, depth + 1U);
                return selected ? std::optional{PayloadValue{selected->bits, true}} : std::nullopt;
            }
            case IR::Opcode::ReadLane: {
                const auto target = EvaluatePayload(inst->Arg(1), lane, depth + 1U);
                return target ? EvaluatePayload(inst->Arg(0), target->bits, depth + 1U)
                              : std::nullopt;
            }
            case IR::Opcode::BitwiseAnd32:
            case IR::Opcode::BitwiseOr32:
            case IR::Opcode::BitwiseXor32:
            case IR::Opcode::FPMin32:
            case IR::Opcode::FPAdd32:
                break;
            default:
                return std::nullopt;
            }
            const auto lhs = EvaluatePayload(inst->Arg(0), lane, depth + 1U);
            const auto rhs = EvaluatePayload(inst->Arg(1), lane, depth + 1U);
            if (!lhs || !rhs) {
                return std::nullopt;
            }
            const bool initialized = lhs->initialized || rhs->initialized;
            switch (inst->GetOpcode()) {
            case IR::Opcode::BitwiseAnd32:
                return PayloadValue{lhs->bits & rhs->bits, initialized};
            case IR::Opcode::BitwiseOr32:
                return PayloadValue{lhs->bits | rhs->bits, initialized};
            case IR::Opcode::BitwiseXor32:
                return PayloadValue{lhs->bits ^ rhs->bits, initialized};
            case IR::Opcode::FPMin32:
                // Only the minimum identity is modeled. Equal subnormals may be flushed
                // by the runtime float mode, so equality alone is not a safe proof.
                if (lhs->bits == 0x7f800000U && rhs->bits == 0x7f800000U) {
                    return PayloadValue{lhs->bits, initialized};
                }
                return std::nullopt;
            case IR::Opcode::FPAdd32:
                // Avoid inventing a host floating-point/rounding model: +0 + +0 is exact.
                return lhs->bits == 0U && rhs->bits == 0U
                           ? std::optional{PayloadValue{0U, initialized}}
                           : std::nullopt;
            default:
                return std::nullopt;
            }
        }();
        payload_visiting[lane].erase(inst);
        payload_cache[lane].emplace(inst, result);
        return result;
    }

    IR::Value initial_mask;
    std::array<u32, 3> shape;
    u32 count;
    u32 initial_nodes{};
    u32 mask_nodes{};
    u32 payload_nodes{};
    const IR::Block* payload_block{};
    const std::unordered_set<const IR::Inst*>* full_exec_instructions{};
    std::array<std::unordered_map<const IR::Inst*, std::optional<u32>>, 64> initial_cache;
    std::array<std::unordered_map<const IR::Inst*, std::optional<MaskValue>>, 64> mask_cache;
    std::array<std::unordered_set<const IR::Inst*>, 64> mask_visiting;
    std::array<std::unordered_map<const IR::Inst*, std::optional<PayloadValue>>, 64> payload_cache;
    std::array<std::unordered_set<const IR::Inst*>, 64> payload_visiting;
};

inline void RecoverNeutralReads(IR::Program& program, const RuntimeInfo& runtime) {
    if (program.info.l_stage != LogicalStage::Compute || runtime.stage != Stage::Compute ||
        program.blocks.empty()) {
        return;
    }
    const auto& shape = runtime.cs_info.workgroup_size;
    u32 count = 1;
    for (const u32 dimension : shape) {
        if (dimension == 0 || count > 63U / dimension) {
            return;
        }
        count *= dimension;
    }
    const IR::Block* const entry = program.blocks.front();
    if (!entry->ImmPredecessors().empty()) {
        return;
    }
    const auto initial_set = std::ranges::find_if(
        *entry, [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::SetExec; });
    if (initial_set == entry->end()) {
        return;
    }
    NeutralReadEvaluator evaluator{initial_set->Arg(0), shape, count};
    if (!evaluator.HasInitialParticipationPredicate()) {
        return;
    }
    for (IR::Block* const block : program.blocks) {
        bool full_exec = false;
        std::unordered_set<const IR::Inst*> full_exec_instructions;
        std::vector<std::pair<IR::Inst*, u32>> replacements;
        for (IR::Inst& inst : *block) {
            if (inst.GetOpcode() == IR::Opcode::SetExec) {
                full_exec = evaluator.IsFullExec(inst.Arg(0));
            } else if (full_exec) {
                full_exec_instructions.insert(&inst);
            }
            // READLANE ignores EXEC. Its source definitions, not the mask at this scalar
            // read, must establish that the inactive lane was explicitly initialized.
            if (const auto recovered = evaluator.Recover(inst, *block, full_exec_instructions)) {
                replacements.emplace_back(&inst, *recovered);
            }
        }
        for (const auto& [inst, bits] : replacements) {
            inst->ReplaceUsesWithAndRemove(IR::Value{bits});
        }
    }
}

} // namespace Shader::Optimization::PartialWave
