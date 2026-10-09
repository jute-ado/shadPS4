// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// This file implements the SSA rewriting algorithm proposed in
//
//      Simple and Efficient Construction of Static Single Assignment Form.
//      Braun M., Buchwald S., Hack S., Leiba R., Mallon C., Zwinkau A. (2013)
//      In: Jhala R., De Bosschere K. (eds)
//      Compiler Construction. CC 2013.
//      Lecture Notes in Computer Science, vol 7791.
//      Springer, Berlin, Heidelberg
//
//      https://link.springer.com/chapter/10.1007/978-3-642-37051-9_6
//

#include <array>
#include <bitset>
#include <map>
#include <span>
#include <unordered_map>
#include <variant>

#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/opcodes.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/reg.h"
#include "shader_recompiler/ir/value.h"

namespace Shader::Optimization {
namespace {
struct FlagTag {
    auto operator<=>(const FlagTag&) const noexcept = default;
};
struct SccFlagTag : FlagTag {};
struct ExecFlagTag : FlagTag {};
struct VccFlagTag : FlagTag {};
struct VccLoTag : FlagTag {};
struct VccHiTag : FlagTag {};
struct M0Tag : FlagTag {};

struct GotoVariable : FlagTag {
    GotoVariable() = default;
    explicit GotoVariable(u32 index_) : index{index_} {}

    auto operator<=>(const GotoVariable&) const noexcept = default;

    u32 index;
};

// Thread-mask bool spilled to a VGPR lane by V_WRITELANE_B32 and restored by V_READLANE_B32.
struct MaskLaneVariable : FlagTag {
    MaskLaneVariable() = default;
    explicit MaskLaneVariable(IR::VectorReg vgpr_, u32 lane_) : vgpr{vgpr_}, lane{lane_} {}

    auto operator<=>(const MaskLaneVariable&) const noexcept = default;

    IR::VectorReg vgpr{};
    u32 lane{};
};

// A mask-lane value is usable only when a matching V_WRITELANE definition reaches every path.
// Keep this separate from the value: a real spill may itself carry an undefined bit, while an
// absent optional shadow must not reclassify an ordinary V_READLANE result as a thread mask.
struct MaskLaneProvenance : FlagTag {
    MaskLaneProvenance() = default;
    explicit MaskLaneProvenance(IR::VectorReg vgpr_, u32 lane_) : vgpr{vgpr_}, lane{lane_} {}

    auto operator<=>(const MaskLaneProvenance&) const noexcept = default;

    IR::VectorReg vgpr{};
    u32 lane{};
};

struct ThreadBitScalar : FlagTag {
    ThreadBitScalar() = default;
    explicit ThreadBitScalar(IR::ScalarReg sgpr_) : sgpr{sgpr_} {}

    auto operator<=>(const ThreadBitScalar&) const noexcept = default;

    IR::ScalarReg sgpr;
};

// An SGPR dword can be held either as an ordinary scalar word or as one per-invocation bit of a
// flattened 64-lane mask. Track that representation independently from both possible values.
struct ScalarIsThreadBit : FlagTag {
    ScalarIsThreadBit() = default;
    explicit ScalarIsThreadBit(IR::ScalarReg sgpr_) : sgpr{sgpr_} {}

    auto operator<=>(const ScalarIsThreadBit&) const noexcept = default;

    IR::ScalarReg sgpr;
};

// The per-invocation bit may represent either dword of a 64-bit mask. This is a property of the
// defining operation, not of the SGPR's absolute parity: an odd-based pair still starts at word 0,
// and a single S_MOV_B32 from EXEC_LO always writes word 0.
struct ThreadBitIsHighWord : FlagTag {
    ThreadBitIsHighWord() = default;
    explicit ThreadBitIsHighWord(IR::ScalarReg sgpr_) : sgpr{sgpr_} {}

    auto operator<=>(const ThreadBitIsHighWord&) const noexcept = default;

    IR::ScalarReg sgpr;
};

using Variant =
    std::variant<IR::ScalarReg, IR::VectorReg, IR::VirtualReg, GotoVariable, MaskLaneVariable,
                 MaskLaneProvenance, ThreadBitScalar, ScalarIsThreadBit, ThreadBitIsHighWord,
                 SccFlagTag, ExecFlagTag, VccFlagTag, VccLoTag, VccHiTag, M0Tag>;
using ValueMap = std::unordered_map<IR::Block*, IR::Value>;

struct DefTable {
    const IR::Value& Def(IR::Block* block, IR::ScalarReg variable) {
        return block->ssa_sreg_values[RegIndex(variable)];
    }
    void SetDef(IR::Block* block, IR::ScalarReg variable, const IR::Value& value) {
        block->ssa_sreg_values[RegIndex(variable)] = value;
    }

    const IR::Value& Def(IR::Block* block, IR::VectorReg variable) {
        return block->ssa_vreg_values[RegIndex(variable)];
    }
    void SetDef(IR::Block* block, IR::VectorReg variable, const IR::Value& value) {
        block->ssa_vreg_values[RegIndex(variable)] = value;
    }

    const IR::Value& Def(IR::Block* block, IR::VirtualReg variable) {
        return reg_vars[variable.Key()][block];
    }
    void SetDef(IR::Block* block, IR::VirtualReg variable, const IR::Value& value) {
        reg_vars[variable.Key()].insert_or_assign(block, value);
    }

    const IR::Value& Def(IR::Block* block, GotoVariable variable) {
        return goto_vars[variable.index][block];
    }
    void SetDef(IR::Block* block, GotoVariable variable, const IR::Value& value) {
        goto_vars[variable.index].insert_or_assign(block, value);
    }

    const IR::Value& Def(IR::Block* block, MaskLaneVariable variable) {
        return mask_lane_vars[MaskLaneKey(variable)][block];
    }
    void SetDef(IR::Block* block, MaskLaneVariable variable, const IR::Value& value) {
        mask_lane_vars[MaskLaneKey(variable)].insert_or_assign(block, value);
    }

    static u32 MaskLaneKey(MaskLaneVariable variable) {
        return (u32(RegIndex(variable.vgpr)) << 6) | variable.lane;
    }

    const IR::Value& Def(IR::Block* block, MaskLaneProvenance variable) {
        return mask_lane_provenance[MaskLaneKey(variable)][block];
    }
    void SetDef(IR::Block* block, MaskLaneProvenance variable, const IR::Value& value) {
        mask_lane_provenance[MaskLaneKey(variable)].insert_or_assign(block, value);
    }

    static u32 MaskLaneKey(MaskLaneProvenance variable) {
        return (u32(RegIndex(variable.vgpr)) << 6) | variable.lane;
    }

    const IR::Value& Def(IR::Block* block, ThreadBitScalar variable) {
        return block->ssa_sbit_values[RegIndex(variable.sgpr)];
    }
    void SetDef(IR::Block* block, ThreadBitScalar variable, const IR::Value& value) {
        block->ssa_sbit_values[RegIndex(variable.sgpr)] = value;
    }

    const IR::Value& Def(IR::Block* block, ScalarIsThreadBit variable) {
        return scalar_is_thread_bit[RegIndex(variable.sgpr)][block];
    }
    void SetDef(IR::Block* block, ScalarIsThreadBit variable, const IR::Value& value) {
        scalar_is_thread_bit[RegIndex(variable.sgpr)].insert_or_assign(block, value);
    }

    const IR::Value& Def(IR::Block* block, ThreadBitIsHighWord variable) {
        return thread_bit_is_high_word[RegIndex(variable.sgpr)][block];
    }
    void SetDef(IR::Block* block, ThreadBitIsHighWord variable, const IR::Value& value) {
        thread_bit_is_high_word[RegIndex(variable.sgpr)].insert_or_assign(block, value);
    }

    const IR::Value& Def(IR::Block* block, SccFlagTag) {
        return scc_flag[block];
    }
    void SetDef(IR::Block* block, SccFlagTag, const IR::Value& value) {
        scc_flag.insert_or_assign(block, value);
    }

    const IR::Value& Def(IR::Block* block, ExecFlagTag) {
        return exec_flag[block];
    }
    void SetDef(IR::Block* block, ExecFlagTag, const IR::Value& value) {
        exec_flag.insert_or_assign(block, value);
    }

    const IR::Value& Def(IR::Block* block, VccLoTag) {
        return vcc_lo_flag[block];
    }
    void SetDef(IR::Block* block, VccLoTag, const IR::Value& value) {
        vcc_lo_flag.insert_or_assign(block, value);
    }

    const IR::Value& Def(IR::Block* block, VccHiTag) {
        return vcc_hi_flag[block];
    }
    void SetDef(IR::Block* block, VccHiTag, const IR::Value& value) {
        vcc_hi_flag.insert_or_assign(block, value);
    }

    const IR::Value& Def(IR::Block* block, VccFlagTag) {
        return vcc_flag[block];
    }
    void SetDef(IR::Block* block, VccFlagTag, const IR::Value& value) {
        vcc_flag.insert_or_assign(block, value);
    }
    const IR::Value& Def(IR::Block* block, M0Tag) {
        return m0_flag[block];
    }
    void SetDef(IR::Block* block, M0Tag, const IR::Value& value) {
        m0_flag.insert_or_assign(block, value);
    }

    std::unordered_map<u32, ValueMap> goto_vars;
    std::unordered_map<u32, ValueMap> mask_lane_vars;
    std::unordered_map<u32, ValueMap> mask_lane_provenance;
    std::unordered_map<u64, ValueMap> reg_vars;
    std::unordered_map<u32, ValueMap> scalar_is_thread_bit;
    std::unordered_map<u32, ValueMap> thread_bit_is_high_word;
    ValueMap scc_flag;
    ValueMap exec_flag;
    ValueMap vcc_flag;
    ValueMap scc_lo_flag;
    ValueMap vcc_lo_flag;
    ValueMap vcc_hi_flag;
    ValueMap m0_flag;
};

constexpr IR::Opcode UndefOpcode(IR::ScalarReg) noexcept {
    return IR::Opcode::UndefU32;
}

constexpr IR::Opcode UndefOpcode(IR::VectorReg) noexcept {
    return IR::Opcode::UndefU32;
}

constexpr IR::Opcode UndefOpcode(const VccLoTag) noexcept {
    return IR::Opcode::UndefU32;
}

constexpr IR::Opcode UndefOpcode(const VccHiTag) noexcept {
    return IR::Opcode::UndefU32;
}

constexpr IR::Opcode UndefOpcode(const M0Tag) noexcept {
    return IR::Opcode::UndefU32;
}

constexpr IR::Opcode UndefOpcode(const FlagTag) noexcept {
    return IR::Opcode::UndefU1;
}

constexpr IR::Opcode UndefOpcode(const IR::VirtualReg reg) noexcept {
    switch (reg.type) {
    case IR::Type::U32:
        return IR::Opcode::UndefU32;
    case IR::Type::U1:
        return IR::Opcode::UndefU1;
    default:
        UNREACHABLE();
    }
}

class Pass {
public:
    explicit Pass(IR::Block* first_block_) : first_block{first_block_} {}

    template <typename Type>
    void WriteVariable(Type variable, IR::Block* block, const IR::Value& value) {
        current_def.SetDef(block, variable, value);
    }

    template <typename Type>
    IR::Value ReadVariable(Type variable, IR::Block* block) {
        // If variable has a definition in block, return it.
        IR::Value def = current_def.Def(block, variable);
        if (!def.IsEmpty()) {
            return def;
        }

        // Otherwise, look up the value in block predecessors.
        const std::span imm_preds = block->ImmPredecessors();
        if (imm_preds.size() == 1) {
            // Optimize the common case of one predecessor: no phi needed
            def = ReadVariable(variable, imm_preds.front());
        } else if (imm_preds.size() > 1) {
            // This is a join block which may require a phi.
            // That will act as the variables current definition to break potential cycles.
            IR::Inst* const phi{&*block->PrependNewInst(block->begin(), IR::Opcode::Phi)};
            phi->SetFlags(IR::TypeOf(UndefOpcode(variable)));

            // Set the value for this block to avoid an infinite recursion.
            WriteVariable(variable, block, IR::Value{phi});
            def = AddPhiOperands(variable, phi, block);
        }

        // If we could not find a store for this variable, use undef.
        if (def.IsEmpty()) {
            def = IR::Value{GetUndefInst(UndefOpcode(variable))};
        }

        WriteVariable(variable, block, def);
        return def;
    }

    void RegisterMaskLane(IR::VectorReg vgpr, u32 lane) {
        ASSERT(lane < 64);
        known_mask_lanes[RegIndex(vgpr)].set(lane);
    }

    void InvalidateMaskLane(IR::VectorReg vgpr, u32 lane, IR::Block* block) {
        if (lane < 64 && known_mask_lanes[RegIndex(vgpr)].test(lane)) {
            WriteVariable(MaskLaneProvenance{vgpr, lane}, block, IR::Value{false});
        }
    }

    void InvalidateMaskLanes(IR::VectorReg vgpr, IR::Block* block) {
        const auto& lanes = known_mask_lanes[RegIndex(vgpr)];
        for (u32 lane = 0; lane < 64; ++lane) {
            if (lanes.test(lane)) {
                WriteVariable(MaskLaneProvenance{vgpr, lane}, block, IR::Value{false});
            }
        }
    }

    void SeedMaskLaneProvenance(IR::Block* root) {
        for (u32 vgpr_index = 0; vgpr_index < IR::NumVectorRegs; ++vgpr_index) {
            const auto vgpr = static_cast<IR::VectorReg>(vgpr_index);
            const auto& lanes = known_mask_lanes[vgpr_index];
            for (u32 lane = 0; lane < 64; ++lane) {
                if (lanes.test(lane)) {
                    WriteVariable(MaskLaneProvenance{vgpr, lane}, root, IR::Value{false});
                }
            }
        }
    }

    template <typename Type>
    IR::Value AddPhiOperands(Type variable, IR::Inst* phi, IR::Block* block) {
        ASSERT_MSG(phi->NumArgs() == 0, "Phi candidate already has arguments");

        bool found_empty_arg = false;
        for (IR::Block* const pred : block->ImmPredecessors()) {
            // If pred is not sealed, use empty value to indicate that phi
            // needs to be completed after the whole CFG has been processed.
            auto arg = pred->IsSsaSealed() ? ReadVariable(variable, pred) : IR::Value{};
            phi->AddPhiOperand(pred, arg);
            found_empty_arg |= arg.IsEmpty();
        }
        if (found_empty_arg) {
            incomplete_phis[block].insert_or_assign(variable, phi);
            return IR::Value{phi};
        }
        return TryRemoveTrivialPhi(*phi, block, UndefOpcode(variable));
    }

    void FinalizePhiCandidates() {
        for (auto it = incomplete_phis.begin(); it != incomplete_phis.end(); ++it) {
            for (auto& [variant, phi] : it->second) {
                std::visit([&](auto& variable) { FinalizePhiCandidate(variable, *phi, it->first); },
                           variant);
            }
        }
    }

private:
    IR::Inst* GetUndefInst(IR::Opcode undef_opcode) {
        auto [it, is_new] = undef_map.try_emplace(undef_opcode, nullptr);
        if (is_new) {
            IR::Inst* const undef{
                &*first_block->PrependNewInst(first_block->begin(), undef_opcode)};
            it->second = undef;
        }
        return it->second;
    }

    template <typename Type>
    IR::Value FinalizePhiCandidate(Type variable, IR::Inst& phi, IR::Block* block) {
        size_t arg_index{};
        for (IR::Block* const pred : block->ImmPredecessors()) {
            ASSERT(phi.PhiBlock(arg_index) == pred);
            if (!phi.Arg(arg_index).IsEmpty()) {
                arg_index++;
                continue;
            }
            if (pred->IsSsaSealed()) {
                phi.SetArg(arg_index++, ReadVariable(variable, pred));
            } else {
                phi.SetArg(arg_index++, IR::Value{GetUndefInst(UndefOpcode(variable))});
            }
        }
        return TryRemoveTrivialPhi(phi, block, UndefOpcode(variable));
    }

    IR::Value TryRemoveTrivialPhi(IR::Inst& phi, IR::Block* block, IR::Opcode undef_opcode) {
        IR::Value same;
        const size_t num_args{phi.NumArgs()};
        for (size_t arg_index = 0; arg_index < num_args; ++arg_index) {
            const IR::Value& op{phi.Arg(arg_index)};
            if (op.Resolve() == same.Resolve() || op.Resolve() == IR::Value{&phi}) {
                // Unique value or self-reference
                continue;
            }
            if (!same.IsEmpty()) {
                // The phi merges at least two values: not trivial
                return IR::Value{&phi};
            }
            same = op;
        }
        if (same.IsEmpty()) {
            // The phi is unreachable or in the start block
            // Insert an undefined instruction and make it the phi node replacement
            same = IR::Value{GetUndefInst(undef_opcode)};
        }
        // Reroute all uses of the phi to the "same" value
        const auto users = phi.Uses();
        phi.ReplaceUsesWith(same);

        // Try to recursively remove all phi users, which might have become trivial
        for (const auto& [user, arg_index] : users) {
            if (user->GetOpcode() == IR::Opcode::Phi) {
                TryRemoveTrivialPhi(*user, user->GetParent(), undef_opcode);
            }
        }
        return same;
    }

    IR::Block* first_block;
    std::unordered_map<IR::Opcode, IR::Inst*> undef_map;
    std::unordered_map<IR::Block*, std::map<Variant, IR::Inst*>> incomplete_phis;
    std::array<std::bitset<64>, IR::NumVectorRegs> known_mask_lanes{};
    DefTable current_def;
};

IR::ScalarReg NextScalarReg(IR::ScalarReg reg) {
    ASSERT(RegIndex(reg) + 1 < IR::NumScalarRegs);
    return static_cast<IR::ScalarReg>(RegIndex(reg) + 1);
}

IR::U1 ScalarWordBit(IR::IREmitter& ir, const IR::Value& scalar, const IR::U32& lane) {
    const IR::U32 shift{ir.BitwiseAnd(lane, ir.Imm32(31U))};
    const IR::U32 shifted{ir.ShiftRightLogical(IR::U32{scalar}, shift)};
    return ir.INotEqual(ir.BitwiseAnd(shifted, ir.Imm32(1U)), ir.Imm32(0U));
}

IR::Value ReadScalarWord(Pass& pass, IR::Block* block, IR::Inst& inst, IR::ScalarReg reg) {
    const IR::Value is_thread_bit = pass.ReadVariable(ScalarIsThreadBit{reg}, block);
    if (is_thread_bit.IsImmediate() && !is_thread_bit.U1()) {
        return pass.ReadVariable(reg, block);
    }

    IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::U1 bit{pass.ReadVariable(ThreadBitScalar{reg}, block)};
    const IR::Value is_high_word = pass.ReadVariable(ThreadBitIsHighWord{reg}, block);
    const IR::Value scalar = pass.ReadVariable(reg, block);
    const IR::Inst* const scalar_inst = scalar.Resolve().TryInstRecursive();
    const bool has_scalar_lineage =
        scalar.IsImmediate() ||
        (scalar_inst != nullptr && scalar_inst->GetOpcode() != IR::Opcode::UndefU32);

    // Keep the subgroup operation explicit in IR. Wave64 lowering must see and lower this ballot,
    // or reject the shader, rather than allowing a backend-only collective to bypass its guards.
    IR::Value value;
    if (is_high_word.IsImmediate()) {
        IR::U1 materialized_bit = bit;
        if (has_scalar_lineage && !is_thread_bit.IsImmediate()) {
            const IR::U32 lane = ir.LaneId();
            const IR::U1 scalar_bit = ScalarWordBit(ir, scalar, lane);
            materialized_bit = IR::U1{ir.Select(IR::U1{is_thread_bit}, bit, scalar_bit)};
        }
        const IR::Value ballot = ir.Ballot(materialized_bit);
        value = ir.CompositeExtract(ballot, is_high_word.U1() ? 1U : 0U);
    } else {
        // Different predecessors may carry the same SGPR as different words of different mask
        // pairs. Gate each contribution to its defining half, ballot once, then OR the two halves.
        // The result is identical for every invocation; no non-uniform value controls a collective
        // or selects between the reconstructed scalar words.
        const IR::U32 lane = ir.LaneId();
        const IR::U1 lane_is_high = ir.IGreaterThanEqual(lane, ir.Imm32(32U), false);
        const IR::U1 word_matches_lane =
            ir.LogicalNot(ir.LogicalXor(IR::U1{is_high_word}, lane_is_high));
        const IR::U1 thread_word_bit = ir.LogicalAnd(bit, word_matches_lane);
        IR::U1 materialized_bit = thread_word_bit;
        if (has_scalar_lineage && !is_thread_bit.IsImmediate()) {
            const IR::U1 scalar_bit = ScalarWordBit(ir, scalar, lane);
            materialized_bit =
                IR::U1{ir.Select(IR::U1{is_thread_bit}, thread_word_bit, scalar_bit)};
        }
        const IR::Value ballot = ir.Ballot(materialized_bit);
        const IR::U32 low{ir.CompositeExtract(ballot, 0U)};
        const IR::U32 high{ir.CompositeExtract(ballot, 1U)};
        value = ir.BitwiseOr(low, high);
    }

    // Materialization converts this SSA definition back to a scalar word. Reuse it for later
    // scalar consumers instead of emitting another subgroup operation.
    pass.WriteVariable(reg, block, value);
    pass.WriteVariable(ScalarIsThreadBit{reg}, block, ir.Imm1(false));
    pass.WriteVariable(ThreadBitIsHighWord{reg}, block, ir.Imm1(false));
    return value;
}

IR::U1 ReadThreadBitScalar(Pass& pass, IR::Block* block, IR::Inst& inst,
                           IR::ScalarReg requested_reg) {
    const IR::ScalarReg lo_reg = requested_reg;
    const IR::ScalarReg hi_reg = NextScalarReg(lo_reg);
    const IR::Value lo_is_thread_bit = pass.ReadVariable(ScalarIsThreadBit{lo_reg}, block);
    const IR::Value hi_is_thread_bit = pass.ReadVariable(ScalarIsThreadBit{hi_reg}, block);
    const IR::Value lo_is_high_word = pass.ReadVariable(ThreadBitIsHighWord{lo_reg}, block);
    const IR::Value hi_is_high_word = pass.ReadVariable(ThreadBitIsHighWord{hi_reg}, block);
    if (lo_is_thread_bit.IsImmediate() && hi_is_thread_bit.IsImmediate() && lo_is_thread_bit.U1() &&
        hi_is_thread_bit.U1() && lo_is_high_word.IsImmediate() && hi_is_high_word.IsImmediate() &&
        !lo_is_high_word.U1() && hi_is_high_word.U1()) {
        const IR::Value lo_thread_bit = pass.ReadVariable(ThreadBitScalar{lo_reg}, block);
        const IR::Value hi_thread_bit = pass.ReadVariable(ThreadBitScalar{hi_reg}, block);
        if (lo_thread_bit.Resolve() == hi_thread_bit.Resolve()) {
            return IR::U1{lo_thread_bit};
        }
    }

    IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::U32 lane = ir.LaneId();
    const auto read_word = [&](IR::ScalarReg reg, const IR::Value& is_thread_bit,
                               const IR::Value& is_high_word, bool expected_high_word) -> IR::U1 {
        if (is_thread_bit.IsImmediate() && !is_thread_bit.U1()) {
            return ScalarWordBit(ir, pass.ReadVariable(reg, block), lane);
        }
        const IR::U1 thread_bit{pass.ReadVariable(ThreadBitScalar{reg}, block)};
        IR::U1 converted_thread_bit = thread_bit;
        const bool word_matches =
            is_high_word.IsImmediate() && is_high_word.U1() == expected_high_word;
        if (!word_matches) {
            // This SGPR is being consumed at a different pair position from the operation that
            // defined its thread-bit representation. Reconstruct its defining scalar dword first,
            // then extract the bit required by the new pair. Keep the ballot explicit for Wave64.
            IR::U1 ballot_bit = thread_bit;
            if (is_high_word.IsImmediate()) {
                if (!is_thread_bit.IsImmediate()) {
                    ballot_bit = ir.LogicalAnd(IR::U1{is_thread_bit}, thread_bit);
                }
                const IR::Value ballot = ir.Ballot(ballot_bit);
                const IR::U32 scalar_word{ir.CompositeExtract(ballot, is_high_word.U1() ? 1U : 0U)};
                converted_thread_bit = ScalarWordBit(ir, scalar_word, lane);
            } else {
                const IR::U1 lane_is_high = ir.IGreaterThanEqual(lane, ir.Imm32(32U), false);
                const IR::U1 word_matches_lane =
                    ir.LogicalNot(ir.LogicalXor(IR::U1{is_high_word}, lane_is_high));
                ballot_bit = ir.LogicalAnd(thread_bit, word_matches_lane);
                if (!is_thread_bit.IsImmediate()) {
                    ballot_bit = ir.LogicalAnd(IR::U1{is_thread_bit}, ballot_bit);
                }
                const IR::Value ballot = ir.Ballot(ballot_bit);
                const IR::U32 low{ir.CompositeExtract(ballot, 0U)};
                const IR::U32 high{ir.CompositeExtract(ballot, 1U)};
                converted_thread_bit = ScalarWordBit(ir, ir.BitwiseOr(low, high), lane);
            }
        }
        if (is_thread_bit.IsImmediate()) {
            return converted_thread_bit;
        }
        const IR::U1 scalar_bit = ScalarWordBit(ir, pass.ReadVariable(reg, block), lane);
        return IR::U1{ir.Select(IR::U1{is_thread_bit}, converted_thread_bit, scalar_bit)};
    };

    const IR::U1 lo = read_word(lo_reg, lo_is_thread_bit, lo_is_high_word, false);
    const IR::U1 hi = read_word(hi_reg, hi_is_thread_bit, hi_is_high_word, true);
    return IR::U1{ir.Select(ir.ILessThan(lane, ir.Imm32(32U), false), lo, hi)};
}

void VisitInst(Pass& pass, IR::Block* block, IR::Inst& inst) {
    const IR::Opcode opcode{inst.GetOpcode()};
    switch (opcode) {
    case IR::Opcode::SetThreadBitScalarReg: {
        const IR::ScalarReg reg{inst.Arg(0).ScalarReg()};
        pass.WriteVariable(ThreadBitScalar{reg}, block, inst.Arg(1));
        pass.WriteVariable(ScalarIsThreadBit{reg}, block, IR::Value{true});
        pass.WriteVariable(ThreadBitIsHighWord{reg}, block, IR::Value{false});
        break;
    }
    case IR::Opcode::SetThreadBitScalarRegPair: {
        const IR::ScalarReg reg{inst.Arg(0).ScalarReg()};
        const IR::ScalarReg hi_reg = NextScalarReg(reg);
        pass.WriteVariable(ThreadBitScalar{reg}, block, inst.Arg(1));
        pass.WriteVariable(ThreadBitScalar{hi_reg}, block, inst.Arg(1));
        pass.WriteVariable(ScalarIsThreadBit{reg}, block, IR::Value{true});
        pass.WriteVariable(ScalarIsThreadBit{hi_reg}, block, IR::Value{true});
        pass.WriteVariable(ThreadBitIsHighWord{reg}, block, IR::Value{false});
        pass.WriteVariable(ThreadBitIsHighWord{hi_reg}, block, IR::Value{true});
        break;
    }
    case IR::Opcode::SetScalarRegister: {
        const IR::ScalarReg reg{inst.Arg(0).ScalarReg()};
        pass.WriteVariable(reg, block, inst.Arg(1));
        pass.WriteVariable(ScalarIsThreadBit{reg}, block, IR::Value{false});
        pass.WriteVariable(ThreadBitIsHighWord{reg}, block, IR::Value{false});
        break;
    }
    case IR::Opcode::SetVectorRegister: {
        const IR::VectorReg reg{inst.Arg(0).VectorReg()};
        pass.WriteVariable(reg, block, inst.Arg(1));
        const IR::Inst* const value_inst = inst.Arg(1).Resolve().TryInstRecursive();
        if (value_inst != nullptr && value_inst->GetOpcode() == IR::Opcode::WriteLane &&
            value_inst->Arg(2).IsImmediate()) {
            pass.InvalidateMaskLane(reg, value_inst->Arg(2).U32(), block);
        } else {
            pass.InvalidateMaskLanes(reg, block);
        }
        break;
    }
    case IR::Opcode::SetVirtualRegister:
        pass.WriteVariable(inst.Arg(0).VirtualReg(), block, inst.Arg(1));
        break;
    case IR::Opcode::SetGotoVariable:
        pass.WriteVariable(GotoVariable{inst.Arg(0).U32()}, block, inst.Arg(1));
        break;
    case IR::Opcode::SetMaskLaneVariable:
        pass.WriteVariable(MaskLaneVariable{inst.Arg(0).VectorReg(), inst.Arg(1).U32()}, block,
                           inst.Arg(2));
        pass.WriteVariable(MaskLaneProvenance{inst.Arg(0).VectorReg(), inst.Arg(1).U32()}, block,
                           inst.Arg(3));
        break;
    case IR::Opcode::SetExec:
        pass.WriteVariable(ExecFlagTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetScc:
        pass.WriteVariable(SccFlagTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetVcc:
        pass.WriteVariable(VccFlagTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetVccLo:
        pass.WriteVariable(VccLoTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetVccHi:
        pass.WriteVariable(VccHiTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetM0:
        pass.WriteVariable(M0Tag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::GetThreadBitScalarReg: {
        const IR::ScalarReg reg{inst.Arg(0).ScalarReg()};
        inst.ReplaceUsesWith(ReadThreadBitScalar(pass, block, inst, reg));
        break;
    }
    case IR::Opcode::GetStoredThreadBitScalarReg: {
        const IR::ScalarReg reg{inst.Arg(0).ScalarReg()};
        inst.ReplaceUsesWith(pass.ReadVariable(ThreadBitScalar{reg}, block));
        break;
    }
    case IR::Opcode::IsThreadBitScalarRegLow: {
        const IR::ScalarReg reg{inst.Arg(0).ScalarReg()};
        const IR::Value is_thread_bit = pass.ReadVariable(ScalarIsThreadBit{reg}, block);
        const IR::Value is_high_word = pass.ReadVariable(ThreadBitIsHighWord{reg}, block);
        const bool is_proven_low = is_thread_bit.IsImmediate() && is_thread_bit.U1() &&
                                   is_high_word.IsImmediate() && !is_high_word.U1();
        inst.ReplaceUsesWith(IR::Value{is_proven_low});
        break;
    }
    case IR::Opcode::GetScalarRegister: {
        const IR::ScalarReg reg{inst.Arg(0).ScalarReg()};
        inst.ReplaceUsesWith(ReadScalarWord(pass, block, inst, reg));
        break;
    }
    case IR::Opcode::GetVectorRegister: {
        const IR::VectorReg reg{inst.Arg(0).VectorReg()};
        const IR::Value value = pass.ReadVariable(reg, block);
        inst.ReplaceUsesWith(value);
        break;
    }
    case IR::Opcode::GetVirtualRegister:
        inst.ReplaceUsesWith(pass.ReadVariable(inst.Arg(0).VirtualReg(), block));
        break;
    case IR::Opcode::GetGotoVariable:
        inst.ReplaceUsesWith(pass.ReadVariable(GotoVariable{inst.Arg(0).U32()}, block));
        break;
    case IR::Opcode::GetMaskLaneVariable: {
        const IR::VectorReg vgpr = inst.Arg(0).VectorReg();
        const u32 lane = inst.Arg(1).U32();
        const IR::Value provenance = pass.ReadVariable(MaskLaneProvenance{vgpr, lane}, block);
        if (!provenance.IsImmediate() || !provenance.U1()) {
            // The shadow is optional. If a matching mask spill cannot be proven on every path,
            // discard only the speculative restore and retain the ordinary scalar/flag write.
            for (const auto& [user, _] : inst.Uses()) {
                switch (user->GetOpcode()) {
                case IR::Opcode::SetThreadBitScalarReg:
                case IR::Opcode::SetThreadBitScalarRegPair:
                case IR::Opcode::SetExec:
                case IR::Opcode::SetVcc:
                    user->Invalidate();
                    break;
                default:
                    break;
                }
            }
        }
        inst.ReplaceUsesWith(pass.ReadVariable(MaskLaneVariable{vgpr, lane}, block));
        break;
    }
    case IR::Opcode::GetExec:
        inst.ReplaceUsesWith(pass.ReadVariable(ExecFlagTag{}, block));
        break;
    case IR::Opcode::GetScc:
        inst.ReplaceUsesWith(pass.ReadVariable(SccFlagTag{}, block));
        break;
    case IR::Opcode::GetVcc:
        inst.ReplaceUsesWith(pass.ReadVariable(VccFlagTag{}, block));
        break;
    case IR::Opcode::GetVccLo:
        inst.ReplaceUsesWith(pass.ReadVariable(VccLoTag{}, block));
        break;
    case IR::Opcode::GetVccHi:
        inst.ReplaceUsesWith(pass.ReadVariable(VccHiTag{}, block));
        break;
    case IR::Opcode::GetM0:
        inst.ReplaceUsesWith(pass.ReadVariable(M0Tag{}, block));
        break;
    default:
        break;
    }
}

void VisitBlock(Pass& pass, IR::Block* block) {
    for (IR::Inst& inst : block->Instructions()) {
        VisitInst(pass, block, inst);
    }
    block->SsaSeal();
}

} // Anonymous namespace

void SsaRewritePass(IR::Program& program) {
    Pass pass{program.blocks.front()};
    for (IR::Block* const block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::GetMaskLaneVariable) {
                pass.RegisterMaskLane(inst.Arg(0).VectorReg(), inst.Arg(1).U32());
            }
        }
    }
    for (IR::Block* const root : program.blocks) {
        if (!root->ImmPredecessors().empty()) {
            continue;
        }
        for (u32 index = 0; index < IR::NumScalarRegs; ++index) {
            const auto reg = static_cast<IR::ScalarReg>(index);
            pass.WriteVariable(ScalarIsThreadBit{reg}, root, IR::Value{false});
            pass.WriteVariable(ThreadBitIsHighWord{reg}, root, IR::Value{false});
        }
        pass.SeedMaskLaneProvenance(root);
    }
    const auto end = program.post_order_blocks.rend();
    for (auto it = program.post_order_blocks.rbegin(); it != end; ++it) {
        VisitBlock(pass, *it);
    }
    pass.FinalizePhiCandidates();
}

void SsaDestroyPass(IR::Program& program) {
    // This implements the following mesa function with place_writes_in_imm_preds = true
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/55f62fa8/src/compiler/nir/nir_from_ssa.c#L1075
    u32 reg_index{};
    const auto end = program.post_order_blocks.rend();
    for (auto it1 = program.post_order_blocks.rbegin(); it1 != end; ++it1) {
        IR::Block* block = *it1;
        for (auto it = block->begin(); it != block->end(); ++it) {
            IR::Inst& inst = *it;
            if (inst.GetOpcode() != IR::Opcode::Phi) {
                continue;
            }
            const IR::VirtualReg reg{reg_index++, inst.Flags<IR::Type>()};
            for (size_t arg_index = 0; arg_index < inst.NumArgs(); arg_index++) {
                IR::Value arg = inst.Arg(arg_index);
                IR::Block* arg_block = inst.PhiBlock(arg_index);
                arg_block->PrependNewInst(arg_block->end(), IR::Opcode::SetVirtualRegister,
                                          {IR::Value{reg}, arg});
            }
            IR::Inst* const new_inst{
                &*block->PrependNewInst(it, IR::Opcode::GetVirtualRegister, {IR::Value{reg}})};
            inst.ReplaceUsesWith(IR::Value{new_inst});
        }
    }
}

} // namespace Shader::Optimization
