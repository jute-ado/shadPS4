// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#endif

#include <gtest/gtest.h>

#include "common/object_pool.h"
#include "gcn_test_runner.hpp"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv/unified1/spirv.hpp"

namespace {

using namespace Shader;

constexpr u32 GuestWaveSize = 64;
constexpr u32 RecordWords = 5;

TEST(ReadLaneElimination, DynamicWriteLaneSelectorRemainsNative) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    Pools pools{};
    IR::Program program{info};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    IR::IREmitter ir{*block};

    const auto value = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const auto dynamic_lane = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1);
    const auto written = ir.WriteLane(value, ir.Imm32(0x1234U), dynamic_lane);
    const auto selected = ir.ReadLane(written, ir.Imm32(0U));
    ir.Reference(selected);

    Optimization::ReadLaneEliminationPass(program);

    // A dynamic write selector cannot prove that the immediate read lane observes the write.
    // The pass must leave the cross-lane read intact rather than treating Arg(2) as an immediate.
    EXPECT_EQ(selected.Inst()->GetOpcode(), IR::Opcode::ReadLane);
}

TEST(ReadLaneElimination, DynamicWriteLaneGuardPrecedesImmediateInspection) {
    std::ifstream source{SHADPS4_READLANE_ELIMINATION_SOURCE_PATH};
    ASSERT_TRUE(source.is_open());
    const std::string body{std::istreambuf_iterator<char>{source}, {}};
    const auto search_start = body.find("while (inst->GetOpcode() == IR::Opcode::WriteLane)");
    ASSERT_NE(search_start, std::string::npos);
    const auto guard = body.find("if (!inst->Arg(2).IsImmediate())", search_start);
    const auto read = body.find("inst->Arg(2).U32()", search_start);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(read, std::string::npos);
    EXPECT_LT(guard, read);
}

struct WaveRecord {
    u32 lane_id;
    u32 read_lane_zero;
    u32 read_first_lane;
    u32 ballot_low;
    u32 ballot_high;
};

static_assert(sizeof(WaveRecord) == RecordWords * sizeof(u32));

using WaveOutput = std::array<WaveRecord, GuestWaveSize>;

struct ReductionRecord {
    u32 first;
    u32 second;
};

using ReductionOutput = std::array<ReductionRecord, GuestWaveSize>;

struct EntryReductionRecord {
    u32 first;
    u32 second;
    u32 later_native_lane;
    u32 later_native_peer;
};

using EntryReductionOutput = std::array<EntryReductionRecord, GuestWaveSize>;
struct NestedReductionRecord {
    u32 first;
    u32 second;
    u32 interior_native_lane;
    u32 merged_payload;
};

using NestedReductionOutput = std::array<NestedReductionRecord, GuestWaveSize>;
struct TopLevelReductionRecord {
    u32 first;
    u32 second;
    u32 native_lane;
    u32 payload;
    u32 high_word;
};

using TopLevelReductionOutput = std::array<TopLevelReductionRecord, GuestWaveSize>;
constexpr u32 UnselectedLaneSentinel = 0xdeadbeefU;

void AppendBlockNode(IR::Program& program, IR::Block* block) {
    auto& node = program.syntax_list.emplace_back();
    node.type = IR::AbstractSyntaxNode::Type::Block;
    node.data.block = block;
}

void AppendIfNode(IR::Program& program, const IR::U1& condition, IR::Block* body,
                  IR::Block* merge) {
    auto& node = program.syntax_list.emplace_back();
    node.type = IR::AbstractSyntaxNode::Type::If;
    node.data.if_node.cond = condition;
    node.data.if_node.body = body;
    node.data.if_node.merge = merge;
}

void AppendEndIfNode(IR::Program& program, IR::Block* merge) {
    auto& node = program.syntax_list.emplace_back();
    node.type = IR::AbstractSyntaxNode::Type::EndIf;
    node.data.end_if.merge = merge;
}

bool HasBuiltIn(const std::vector<u32>& spirv, spv::BuiltIn builtin) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (words == 0 || offset + words > spirv.size()) {
            return false;
        }
        if (static_cast<spv::Op>(word & 0xffff) == spv::Op::OpDecorate && words >= 4 &&
            spirv[offset + 2] == static_cast<u32>(spv::DecorationBuiltIn) &&
            spirv[offset + 3] == static_cast<u32>(builtin)) {
            return true;
        }
        offset += words;
    }
    return false;
}

bool HasSpirvOpcode(const std::vector<u32>& spirv, spv::Op opcode) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word = spirv[offset];
        const u32 words = word >> 16;
        if (words == 0 || offset + words > spirv.size()) {
            return false;
        }
        if (static_cast<spv::Op>(word & 0xffff) == opcode) {
            return true;
        }
        offset += words;
    }
    return false;
}

#ifdef SHADPS4_SPIRV_VAL_PATH
int RunSpirvValidator(const std::filesystem::path& module) {
#ifdef _WIN32
    const std::string module_path = module.string();
    return static_cast<int>(_spawnl(_P_WAIT, SHADPS4_SPIRV_VAL_PATH, SHADPS4_SPIRV_VAL_PATH,
                                    "--target-env", "vulkan1.3", module_path.c_str(), nullptr));
#else
    const std::string command = std::string{"\""} + SHADPS4_SPIRV_VAL_PATH +
                                "\" --target-env vulkan1.3 \"" + module.string() + "\"";
    return std::system(command.c_str());
#endif
}
#endif

std::vector<u32> EmitWaveProbe(bool apply_guest_wave_lowering) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.flattened_ud_buf.resize(4);
    const AmdGpu::Buffer output_buffer = AmdGpu::Buffer::Null();
    std::memcpy(info.flattened_ud_buf.data(), &output_buffer, sizeof(output_buffer));
    info.buffers.push_back({
        .sharp_idx = 0,
        .used_types = IR::Type::U32,
        .buffer_type = BufferType::Guest,
        .is_written = true,
    });

    IR::Program program{info};
    Pools pools{};
    IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {GuestWaveSize, 1, 1};

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 32 * 1024;

    Gcn::Translator translator{program.info, runtime_info, profile};
    translator.EmitPrologue(block);
    IR::IREmitter ir{*block};
    const IR::U32 invocation = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U32 lane_id = ir.LaneId();
    // Lane 0 exists in both native subgroup32 halves. This exposes the guest-wave mismatch
    // without making an out-of-range cross-subgroup read undefined on the native control.
    const IR::U32 read_lane_zero = ir.ReadLane(invocation, ir.Imm32(0));
    const IR::U32 read_first_lane = ir.ReadFirstLane(invocation);
    const IR::U1 ballot_bit = ir.LogicalOr(
        ir.LogicalOr(ir.IEqual(invocation, ir.Imm32(0)), ir.IEqual(invocation, ir.Imm32(31))),
        ir.LogicalOr(ir.IEqual(invocation, ir.Imm32(32)), ir.IEqual(invocation, ir.Imm32(63))));
    const IR::Value ballot = ir.Ballot(ballot_bit);
    const IR::U32 ballot_low{ir.CompositeExtract(ballot, 0)};
    const IR::U32 ballot_high{ir.CompositeExtract(ballot, 1)};

    const IR::U32 output_handle = ir.Imm32(0);
    const IR::U32 record_base = ir.IMul(invocation, ir.Imm32(RecordWords));
    const auto store_word = [&](u32 word, const IR::U32& value) {
        ir.StoreBufferU32(1, output_handle, ir.IAdd(record_base, ir.Imm32(word)), value, {});
    };
    store_word(0, lane_id);
    store_word(1, read_lane_zero);
    store_word(2, read_first_lane);
    store_word(3, ballot_low);
    store_word(4, ballot_high);
    ir.Epilogue();

    Optimization::SsaRewritePass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::ReadLaneEliminationPass(program);
    if (apply_guest_wave_lowering) {
        Optimization::LowerLaneIdToLocalInvocationIndexPass(program, runtime_info, profile);
    }
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::DeadCodeEliminationPass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::CollectShaderInfoPass(program, profile);

    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> EmitBracketedTwoReductionProbe() {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.flattened_ud_buf.resize(4);
    const AmdGpu::Buffer output_buffer = AmdGpu::Buffer::Null();
    std::memcpy(info.flattened_ud_buf.data(), &output_buffer, sizeof(output_buffer));
    info.buffers.push_back({
        .sharp_idx = 0,
        .used_types = IR::Type::U32,
        .buffer_type = BufferType::Guest,
        .is_written = true,
    });

    IR::Program program{info};
    Pools pools{};
    IR::Block* const entry = pools.block_pool.Create(pools.inst_pool);
    IR::Block* const first_body = pools.block_pool.Create(pools.inst_pool);
    IR::Block* const merge = pools.block_pool.Create(pools.inst_pool);
    IR::Block* const second_body = pools.block_pool.Create(pools.inst_pool);
    IR::Block* const exit = pools.block_pool.Create(pools.inst_pool);
    entry->AddBranch(first_body);
    entry->AddBranch(merge);
    first_body->AddBranch(merge);
    merge->AddBranch(second_body);
    merge->AddBranch(exit);
    second_body->AddBranch(exit);
    program.blocks = {entry, first_body, merge, second_body, exit};

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    runtime_info.cs_info.shared_memory_size = 1024;

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 32 * 1024;

    Gcn::Translator translator{program.info, runtime_info, profile};
    translator.EmitPrologue(entry);
    IR::IREmitter entry_ir{*entry};
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U32 y = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1);
    const IR::U32 invocation = entry_ir.IAdd(x, entry_ir.IMul(y, entry_ir.Imm32(8)));
    const IR::U1 first_condition =
        entry_ir.ConditionRef(entry_ir.IEqual(invocation, entry_ir.Imm32(0)));

    IR::IREmitter merge_ir{*merge};
    merge_ir.Barrier();
    const IR::U32 lane = merge_ir.LaneId();
    IR::U32 first = invocation;
    IR::U32 second = merge_ir.IAdd(merge_ir.IMul(invocation, merge_ir.Imm32(3)), merge_ir.Imm32(5));
    for (const u32 xor_offset : {16U, 8U, 4U, 2U, 1U}) {
        const IR::U32 peer_lane = merge_ir.BitwiseXor(lane, merge_ir.Imm32(xor_offset));
        first = merge_ir.UMax(first, merge_ir.ReadLane(first, peer_lane));
        second = merge_ir.UMax(second, merge_ir.ReadLane(second, peer_lane));
    }
    first = merge_ir.UMax(merge_ir.ReadLane(first, merge_ir.Imm32(0)),
                          merge_ir.ReadLane(first, merge_ir.Imm32(32)));
    second = merge_ir.UMax(merge_ir.ReadLane(second, merge_ir.Imm32(0)),
                           merge_ir.ReadLane(second, merge_ir.Imm32(32)));
    merge_ir.Barrier();

    const IR::U32 output_handle = merge_ir.Imm32(0);
    const IR::U32 record_base = merge_ir.IMul(invocation, merge_ir.Imm32(2));
    merge_ir.StoreBufferU32(1, output_handle, record_base, first, {});
    merge_ir.StoreBufferU32(1, output_handle, merge_ir.IAdd(record_base, merge_ir.Imm32(1)), second,
                            {});
    const IR::U1 second_condition =
        merge_ir.ConditionRef(merge_ir.IEqual(invocation, merge_ir.Imm32(63)));
    IR::IREmitter{*exit}.Epilogue();

    AppendBlockNode(program, entry);
    AppendIfNode(program, first_condition, first_body, merge);
    AppendBlockNode(program, first_body);
    AppendEndIfNode(program, merge);
    AppendBlockNode(program, merge);
    AppendIfNode(program, second_condition, second_body, exit);
    AppendBlockNode(program, second_body);
    AppendEndIfNode(program, exit);
    AppendBlockNode(program, exit);
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    Optimization::SsaRewritePass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::ReadLaneEliminationPass(program);
    Optimization::LowerLaneIdToLocalInvocationIndexPass(program, runtime_info, profile);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::DeadCodeEliminationPass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::CollectShaderInfoPass(program, profile);

    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> EmitEntryReductionWithLaterNativeShuffleProbe() {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.flattened_ud_buf.resize(4);
    const AmdGpu::Buffer output_buffer = AmdGpu::Buffer::Null();
    std::memcpy(info.flattened_ud_buf.data(), &output_buffer, sizeof(output_buffer));
    info.buffers.push_back({
        .sharp_idx = 0,
        .used_types = IR::Type::U32,
        .buffer_type = BufferType::Guest,
        .is_written = true,
    });

    IR::Program program{info};
    Pools pools{};
    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const native_body = pools.block_pool.Create(pools.inst_pool);
    auto* const body_merge = pools.block_pool.Create(pools.inst_pool);
    auto* const exit = pools.block_pool.Create(pools.inst_pool);
    entry->AddBranch(body);
    entry->AddBranch(exit);
    body->AddBranch(native_body);
    body->AddBranch(body_merge);
    native_body->AddBranch(body_merge);
    body_merge->AddBranch(exit);
    program.blocks = {entry, body, native_body, body_merge, exit};

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    runtime_info.cs_info.shared_memory_size = 1024;
    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 32 * 1024;

    Gcn::Translator translator{program.info, runtime_info, profile};
    translator.EmitPrologue(entry);
    IR::IREmitter entry_ir{*entry};
    entry_ir.SetExec(entry_ir.Imm1(true));
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U32 y = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1);
    const IR::U32 invocation = entry_ir.IAdd(x, entry_ir.IMul(y, entry_ir.Imm32(8)));
    const IR::U32 lane = entry_ir.LaneId();
    IR::U32 first = invocation;
    IR::U32 second = entry_ir.IAdd(entry_ir.IMul(invocation, entry_ir.Imm32(3)), entry_ir.Imm32(5));
    for (const u32 xor_offset : {16U, 8U, 4U, 2U, 1U}) {
        const IR::U32 peer_lane = entry_ir.BitwiseXor(lane, entry_ir.Imm32(xor_offset));
        first = entry_ir.UMax(first, entry_ir.ReadLane(first, peer_lane));
        second = entry_ir.UMax(second, entry_ir.ReadLane(second, peer_lane));
    }
    first = entry_ir.UMax(entry_ir.ReadLane(first, entry_ir.Imm32(0)),
                          entry_ir.ReadLane(first, entry_ir.Imm32(32)));
    second = entry_ir.UMax(entry_ir.ReadLane(second, entry_ir.Imm32(0)),
                           entry_ir.ReadLane(second, entry_ir.Imm32(32)));
    const IR::U32 output_handle = entry_ir.Imm32(0);
    const IR::U32 record_base = entry_ir.IMul(invocation, entry_ir.Imm32(4));
    entry_ir.StoreBufferU32(1, output_handle, record_base, first, {});
    entry_ir.StoreBufferU32(1, output_handle, entry_ir.IAdd(record_base, entry_ir.Imm32(1)), second,
                            {});
    for (const u32 word : {2U, 3U}) {
        entry_ir.StoreBufferU32(1, output_handle, entry_ir.IAdd(record_base, entry_ir.Imm32(word)),
                                entry_ir.Imm32(UnselectedLaneSentinel), {});
    }
    const IR::U1 condition = entry_ir.ConditionRef(entry_ir.IEqual(first, entry_ir.Imm32(63)));

    IR::IREmitter body_ir{*body};
    const IR::U32 native_lane = body_ir.LaneId();
    body_ir.StoreBufferU32(1, output_handle, body_ir.IAdd(record_base, body_ir.Imm32(2)),
                           native_lane, {});
    const IR::U1 native_condition = body_ir.ConditionRef(
        body_ir.IEqual(body_ir.BitwiseAnd(native_lane, body_ir.Imm32(1)), body_ir.Imm32(0)));
    IR::IREmitter native_ir{*native_body};
    const IR::U32 native_peer =
        native_ir.ReadLane(native_ir.IAdd(native_lane, native_ir.Imm32(1000)),
                           native_ir.BitwiseXor(native_lane, native_ir.Imm32(2)));
    native_ir.StoreBufferU32(1, output_handle, native_ir.IAdd(record_base, native_ir.Imm32(3)),
                             native_peer, {});
    IR::IREmitter{*exit}.Epilogue();

    AppendBlockNode(program, entry);
    AppendIfNode(program, condition, body, exit);
    AppendBlockNode(program, body);
    AppendIfNode(program, native_condition, native_body, body_merge);
    AppendBlockNode(program, native_body);
    AppendEndIfNode(program, body_merge);
    AppendBlockNode(program, body_merge);
    AppendEndIfNode(program, exit);
    AppendBlockNode(program, exit);
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    Optimization::SsaRewritePass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::ReadLaneEliminationPass(program);
    Optimization::LowerLaneIdToLocalInvocationIndexPass(program, runtime_info, profile);
    EXPECT_EQ(
        std::ranges::count_if(
            *entry, [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::Barrier; }),
        28);
    for (const auto* block : {body, native_body, body_merge, exit}) {
        EXPECT_FALSE(std::ranges::any_of(
            *block, [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::Barrier; }));
    }
    EXPECT_EQ(native_lane.InstRecursive()->GetOpcode(), IR::Opcode::LaneId);
    EXPECT_EQ(native_peer.InstRecursive()->GetOpcode(), IR::Opcode::ReadLane);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::DeadCodeEliminationPass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::CollectShaderInfoPass(program, profile);

    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> EmitFirstUniformBodyProbe(bool enter_body, bool nested_merge = false) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.flattened_ud_buf.resize(4);
    const AmdGpu::Buffer output_buffer = AmdGpu::Buffer::Null();
    std::memcpy(info.flattened_ud_buf.data(), &output_buffer, sizeof(output_buffer));
    info.buffers.push_back({
        .sharp_idx = 0,
        .used_types = IR::Type::U32,
        .buffer_type = BufferType::Guest,
        .is_written = true,
    });
    IR::Program program{info};
    Pools pools{};
    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const exit = pools.block_pool.Create(pools.inst_pool);
    entry->AddBranch(body);
    entry->AddBranch(exit);
    program.blocks = {entry, body, exit};
    IR::Block* first_body{};
    IR::Block* nested_body{};
    IR::Block* first_merge{};
    IR::Block* second_body{};
    IR::Block* second_merge{};
    IR::Block* candidate{};
    if (nested_merge) {
        const auto create = [&] { return pools.block_pool.Create(pools.inst_pool); };
        first_body = create();
        nested_body = create();
        first_merge = create();
        second_body = create();
        second_merge = create();
        candidate = create();
        program.blocks = {entry,       body,         first_body, nested_body, first_merge,
                          second_body, second_merge, candidate,  exit};
        body->AddBranch(first_body);
        body->AddBranch(candidate);
        first_body->AddBranch(nested_body);
        first_body->AddBranch(first_merge);
        nested_body->AddBranch(first_merge);
        first_merge->AddBranch(second_body);
        first_merge->AddBranch(second_merge);
        second_body->AddBranch(second_merge);
        second_merge->AddBranch(candidate);
        candidate->AddBranch(exit);
    } else {
        body->AddBranch(exit);
    }
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    runtime_info.cs_info.shared_memory_size = 1024;
    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 32 * 1024;
    Gcn::Translator translator{program.info, runtime_info, profile};
    translator.EmitPrologue(entry);
    IR::IREmitter entry_ir{*entry};
    entry_ir.SetExec(entry_ir.Imm1(true));
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U32 y = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1);
    const IR::U32 invocation = entry_ir.IAdd(x, entry_ir.IMul(y, entry_ir.Imm32(8)));
    const auto low = entry_ir.ReadLane(invocation, entry_ir.Imm32(0));
    const auto high = entry_ir.ReadLane(invocation, entry_ir.Imm32(32));
    const IR::U32 combined{entry_ir.BitwiseOr(low, high)};
    const IR::U1 condition =
        entry_ir.ConditionRef(entry_ir.IEqual(combined, entry_ir.Imm32(enter_body ? 32U : 0U)));
    const IR::U32 output_handle = entry_ir.Imm32(0);
    const u32 record_words = nested_merge ? 4U : 2U;
    const IR::U32 record_base = entry_ir.IMul(invocation, entry_ir.Imm32(record_words));
    for (u32 word = 0; word < record_words; ++word) {
        entry_ir.StoreBufferU32(1, output_handle, entry_ir.IAdd(record_base, entry_ir.Imm32(word)),
                                entry_ir.Imm32(UnselectedLaneSentinel), {});
    }

    IR::IREmitter body_ir{*body};
    // Inherit the proven-full mask from entry; no redundant SetExec is needed in this body.
    const IR::U32 lane = body_ir.LaneId();
    const IR::U32 peer = body_ir.ReadLane(body_ir.IAdd(invocation, body_ir.Imm32(100)),
                                          body_ir.BitwiseXor(lane, body_ir.Imm32(16)));
    const IR::U32 body_low = body_ir.ReadLane(peer, body_ir.Imm32(0));
    const IR::U32 body_high = body_ir.ReadLane(peer, body_ir.Imm32(32));
    body_ir.StoreBufferU32(1, output_handle, record_base, body_ir.UMax(body_low, body_high), {});
    body_ir.StoreBufferU32(1, output_handle, body_ir.IAdd(record_base, body_ir.Imm32(1)),
                           body_ir.IAdd(body_low, body_high), {});
    IR::IREmitter{*exit}.Epilogue();
    AppendBlockNode(program, entry);
    AppendIfNode(program, condition, body, exit);
    AppendBlockNode(program, body);
    if (nested_merge) {
        const auto merge_u32 = [](IR::Block& block, IR::Block* first, const IR::U32& first_value,
                                  IR::Block* second, const IR::U32& second_value) {
            auto* const phi = &*block.PrependNewInst(block.begin(), IR::Opcode::Phi);
            phi->SetFlags(IR::Type::U32);
            phi->AddPhiOperand(first, first_value);
            phi->AddPhiOperand(second, second_value);
            return IR::U32{phi};
        };
        const IR::U1 inner_condition =
            body_ir.IEqual(body_ir.BitwiseAnd(invocation, body_ir.Imm32(1)), body_ir.Imm32(0));
        body_ir.SetExec(inner_condition);
        const auto inner_ref = body_ir.ConditionRef(inner_condition);
        IR::IREmitter first_ir{*first_body};
        // This observable native ID belongs to a divergent interior, not to a bridged region.
        // The oracle checks its range without assuming any 8x8 subgroup-to-invocation mapping.
        first_ir.StoreBufferU32(1, output_handle, first_ir.IAdd(record_base, first_ir.Imm32(2)),
                                first_ir.LaneId(), {});
        const auto nested_condition = first_ir.ConditionRef(
            first_ir.IEqual(first_ir.BitwiseAnd(invocation, first_ir.Imm32(2)), first_ir.Imm32(0)));
        IR::IREmitter nested_ir{*nested_body};
        const IR::U32 first_payload = nested_ir.IAdd(invocation, nested_ir.Imm32(100));
        const auto first_result =
            merge_u32(*first_merge, nested_body, first_payload, first_body, invocation);
        IR::IREmitter first_merge_ir{*first_merge};
        const auto second_condition = first_merge_ir.ConditionRef(
            first_merge_ir.IEqual(first_merge_ir.BitwiseAnd(invocation, first_merge_ir.Imm32(4)),
                                  first_merge_ir.Imm32(0)));
        IR::IREmitter second_ir{*second_body};
        const IR::U32 second_payload = second_ir.IAdd(first_result, second_ir.Imm32(200));
        const auto second_result =
            merge_u32(*second_merge, second_body, second_payload, first_merge, first_result);
        const auto payload = merge_u32(*candidate, second_merge, second_result, body, invocation);
        IR::IREmitter candidate_ir{*candidate};
        candidate_ir.SetExec(candidate_ir.Imm1(true));
        const auto candidate_lane = candidate_ir.LaneId();
        const auto candidate_peer = candidate_ir.ReadLane(
            payload, candidate_ir.BitwiseXor(candidate_lane, candidate_ir.Imm32(16)));
        const auto candidate_low = candidate_ir.ReadLane(candidate_peer, candidate_ir.Imm32(0));
        const auto candidate_high = candidate_ir.ReadLane(candidate_peer, candidate_ir.Imm32(32));
        candidate_ir.StoreBufferU32(1, output_handle, record_base,
                                    candidate_ir.UMax(candidate_low, candidate_high), {});
        candidate_ir.StoreBufferU32(1, output_handle,
                                    candidate_ir.IAdd(record_base, candidate_ir.Imm32(1)),
                                    candidate_ir.IAdd(candidate_low, candidate_high), {});
        candidate_ir.StoreBufferU32(
            1, output_handle, candidate_ir.IAdd(record_base, candidate_ir.Imm32(3)), payload, {});
        AppendIfNode(program, inner_ref, first_body, candidate);
        AppendBlockNode(program, first_body);
        AppendIfNode(program, nested_condition, nested_body, first_merge);
        AppendBlockNode(program, nested_body);
        AppendEndIfNode(program, first_merge);
        AppendBlockNode(program, first_merge);
        AppendIfNode(program, second_condition, second_body, second_merge);
        AppendBlockNode(program, second_body);
        AppendEndIfNode(program, second_merge);
        AppendBlockNode(program, second_merge);
        AppendEndIfNode(program, candidate);
        AppendBlockNode(program, candidate);
    }
    AppendEndIfNode(program, exit);
    AppendBlockNode(program, exit);
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

    Optimization::SsaRewritePass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::ReadLaneEliminationPass(program);
    Optimization::LowerLaneIdToLocalInvocationIndexPass(program, runtime_info, profile);
    EXPECT_EQ(
        std::ranges::count_if(
            *entry, [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::Barrier; }),
        4);
    EXPECT_EQ(
        std::ranges::count_if(
            *body, [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::Barrier; }),
        6);
    EXPECT_EQ(info.workgroup_scratch_memory_size, 256U);
    if (nested_merge) {
        EXPECT_EQ(std::ranges::count_if(
                      *candidate,
                      [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::Barrier; }),
                  6);
        EXPECT_EQ(std::ranges::count_if(
                      *first_body,
                      [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::LaneId; }),
                  1);
        for (const auto* const block :
             {first_body, nested_body, first_merge, second_body, second_merge}) {
            EXPECT_EQ(std::ranges::count_if(*block,
                                            [](const IR::Inst& inst) {
                                                return inst.GetOpcode() == IR::Opcode::Barrier;
                                            }),
                      0);
        }
    }
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::DeadCodeEliminationPass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::CollectShaderInfoPass(program, profile);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> EmitTopLevelMergeProbe(bool select_even, bool explicit_layout_enabled) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.flattened_ud_buf.resize(4);
    const AmdGpu::Buffer output_buffer = AmdGpu::Buffer::Null();
    std::memcpy(info.flattened_ud_buf.data(), &output_buffer, sizeof(output_buffer));
    info.buffers.push_back({
        .sharp_idx = 0,
        .used_types = IR::Type::U32,
        .buffer_type = BufferType::Guest,
        .is_written = true,
    });
    IR::Program program{info};
    Pools pools{};
    auto* const entry = pools.block_pool.Create(pools.inst_pool);
    auto* const body = pools.block_pool.Create(pools.inst_pool);
    auto* const merge = pools.block_pool.Create(pools.inst_pool);
    entry->AddBranch(body);
    entry->AddBranch(merge);
    body->AddBranch(merge);
    program.blocks = {entry, body, merge};
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {8, 8, 1};
    runtime_info.cs_info.shared_memory_size = 1024;
    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.max_shared_memory_size = 32 * 1024;
    profile.supports_workgroup_explicit_memory_layout = explicit_layout_enabled;
    Gcn::Translator translator{program.info, runtime_info, profile};
    translator.EmitPrologue(entry);
    IR::IREmitter entry_ir{*entry};
    entry_ir.SetExec(entry_ir.Imm1(true));
    const IR::U32 x = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U32 y = entry_ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1);
    const IR::U32 invocation = entry_ir.IAdd(x, entry_ir.IMul(y, entry_ir.Imm32(8)));
    const auto native_lane = entry_ir.LaneId();
    const auto condition =
        entry_ir.ConditionRef(entry_ir.IEqual(entry_ir.BitwiseAnd(native_lane, entry_ir.Imm32(1)),
                                              entry_ir.Imm32(select_even ? 0U : 1U)));
    const auto shared_offset = entry_ir.IMul(invocation, entry_ir.Imm32(8));
    const auto high_word = entry_ir.IAdd(invocation, entry_ir.Imm32(1000));
    entry_ir.WriteShared(64,
                         entry_ir.PackUint2x32(entry_ir.CompositeConstruct(invocation, high_word)),
                         shared_offset);
    const auto output_handle = entry_ir.Imm32(0);
    const auto record_base = entry_ir.IMul(invocation, entry_ir.Imm32(5));
    entry_ir.StoreBufferU32(1, output_handle, entry_ir.IAdd(record_base, entry_ir.Imm32(2)),
                            native_lane, {});
    entry_ir.SetExec(condition);
    IR::IREmitter body_ir{*body};
    body_ir.WriteShared(64,
                        body_ir.PackUint2x32(body_ir.CompositeConstruct(
                            body_ir.IAdd(invocation, body_ir.Imm32(100)), high_word)),
                        shared_offset);
    IR::IREmitter merge_ir{*merge};
    merge_ir.SetExec(merge_ir.Imm1(true));
    const auto payload_words =
        merge_ir.UnpackUint2x32(IR::U64{merge_ir.LoadShared(64, false, shared_offset)});
    const IR::U32 payload{merge_ir.CompositeExtract(payload_words, 0)};
    const auto lane = merge_ir.LaneId();
    const auto peer = merge_ir.ReadLane(payload, merge_ir.BitwiseXor(lane, merge_ir.Imm32(16)));
    const auto low = merge_ir.ReadLane(peer, merge_ir.Imm32(0));
    const auto high = merge_ir.ReadLane(peer, merge_ir.Imm32(32));
    merge_ir.StoreBufferU32(1, output_handle, record_base, merge_ir.UMax(low, high), {});
    merge_ir.StoreBufferU32(1, output_handle, merge_ir.IAdd(record_base, merge_ir.Imm32(1)),
                            merge_ir.IAdd(low, high), {});
    // Read the original allocation again through both U32 and U64 views after all scratch
    // stores/barriers. The low-word cross-type read proves actual aliasing, not two disjoint
    // variables that merely happen to preserve the U64 data independently.
    const auto after_words =
        merge_ir.UnpackUint2x32(IR::U64{merge_ir.LoadShared(64, false, shared_offset)});
    merge_ir.StoreBufferU32(1, output_handle, merge_ir.IAdd(record_base, merge_ir.Imm32(3)),
                            merge_ir.LoadShared(32, false, shared_offset), {});
    merge_ir.StoreBufferU32(1, output_handle, merge_ir.IAdd(record_base, merge_ir.Imm32(4)),
                            merge_ir.CompositeExtract(after_words, 1), {});
    merge_ir.Epilogue();
    AppendBlockNode(program, entry);
    AppendIfNode(program, condition, body, merge);
    AppendBlockNode(program, body);
    AppendEndIfNode(program, merge);
    AppendBlockNode(program, merge);
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());
    Optimization::SsaRewritePass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::ReadLaneEliminationPass(program);
    Optimization::LowerLaneIdToLocalInvocationIndexPass(program, runtime_info, profile);
    EXPECT_EQ(
        std::ranges::count_if(
            *merge, [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::Barrier; }),
        6);
    EXPECT_EQ(
        std::ranges::count_if(
            *entry, [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::LaneId; }),
        1);
    EXPECT_EQ(info.workgroup_scratch_memory_size, 256U);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::DeadCodeEliminationPass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::CollectShaderInfoPass(program, profile);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

struct PartialReductionRecord {
    u32 minimum;
    u32 sum;
    u32 upper_minimum;
    u32 upper_sum;
    u32 invocation;
};

using PartialReductionOutput = std::array<PartialReductionRecord, 32>;

std::vector<u32> EmitPartialWaveNeutralReductionProbe(const std::array<u32, 3>& shape) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.flattened_ud_buf.resize(4);
    const AmdGpu::Buffer output_buffer = AmdGpu::Buffer::Null();
    std::memcpy(info.flattened_ud_buf.data(), &output_buffer, sizeof(output_buffer));
    info.buffers.push_back({
        .sharp_idx = 0,
        .used_types = IR::Type::U32,
        .buffer_type = BufferType::Guest,
        .is_written = true,
    });
    IR::Program program{info};
    Pools pools{};
    auto* const block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    AppendBlockNode(program, block);
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front());
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Compute);
    runtime.cs_info.workgroup_size = shape;
    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    Gcn::Translator translator{info, runtime, profile};
    translator.EmitPrologue(block);
    IR::IREmitter ir{*block};
    const auto initial_mask = ir.GetExec();
    const IR::U32 x = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
    const IR::U32 y = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1);
    const IR::U32 z = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 2);
    const IR::U32 invocation = ir.IAdd(ir.IAdd(x, ir.IMul(y, ir.Imm32(shape[0]))),
                                       ir.IMul(z, ir.Imm32(shape[0] * shape[1])));
    const IR::F32 active_value{ir.ConvertUToF(32, 32, ir.IAdd(invocation, ir.Imm32(1)))};
    ir.SetExec(ir.LogicalOr(initial_mask, ir.LogicalNot(initial_mask)));
    IR::F32 minimum{
        ir.Select(initial_mask, active_value, ir.Imm32(std::numeric_limits<float>::infinity()))};
    IR::F32 sum{ir.Select(initial_mask, active_value, ir.Imm32(0.0f))};
    for (const u32 distance : {16U, 8U, 4U, 2U, 1U}) {
        const IR::U32 peer = ir.BitwiseXor(ir.LaneId(), ir.Imm32(distance));
        minimum = IR::F32{ir.FPMin(
            minimum, ir.BitCast<IR::F32>(ir.ReadLane(ir.BitCast<IR::U32>(minimum), peer)))};
        sum = IR::F32{
            ir.FPAdd(sum, ir.BitCast<IR::F32>(ir.ReadLane(ir.BitCast<IR::U32>(sum), peer)))};
    }
    // These scalar transfers ignore EXEC. All contributing inactive VGPR definitions
    // above were nevertheless made under the explicit full guest mask.
    ir.SetExec(initial_mask);
    const auto lower_minimum = ir.ReadLane(ir.BitCast<IR::U32>(minimum), ir.Imm32(0));
    const auto upper_minimum = ir.ReadLane(ir.BitCast<IR::U32>(minimum), ir.Imm32(32));
    const auto lower_sum = ir.ReadLane(ir.BitCast<IR::U32>(sum), ir.Imm32(0));
    const auto upper_sum = ir.ReadLane(ir.BitCast<IR::U32>(sum), ir.Imm32(32));
    const IR::F32 full_minimum{
        ir.FPMin(ir.BitCast<IR::F32>(lower_minimum), ir.BitCast<IR::F32>(upper_minimum))};
    const IR::F32 full_sum{
        ir.FPAdd(ir.BitCast<IR::F32>(lower_sum), ir.BitCast<IR::F32>(upper_sum))};
    const IR::U32 base = ir.IMul(invocation, ir.Imm32(5));
    const std::array words{ir.BitCast<IR::U32>(full_minimum), ir.BitCast<IR::U32>(full_sum),
                           upper_minimum, upper_sum, invocation};
    for (u32 word = 0; word < words.size(); ++word) {
        ir.StoreBufferU32(1, ir.Imm32(0), ir.IAdd(base, ir.Imm32(word)), words[word], {});
    }
    ir.Epilogue();
    Optimization::SsaRewritePass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::ReadLaneEliminationPass(program);
    Optimization::DeadCodeEliminationPass(program, true);
    Optimization::LowerLaneIdToLocalInvocationIndexPass(program, runtime, profile);
    EXPECT_EQ(info.workgroup_scratch_memory_size, 0U);
    EXPECT_EQ(runtime.cs_info.workgroup_size, shape);
    EXPECT_EQ(std::ranges::count_if(*block,
                                    [](const IR::Inst& inst) {
                                        return inst.GetOpcode() == IR::Opcode::ReadLane &&
                                               inst.Arg(1).IsImmediate() &&
                                               inst.Arg(1).U32() == 32U;
                                    }),
              0);
    EXPECT_EQ(
        std::ranges::count_if(
            *block, [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::Barrier; }),
        0);
    Optimization::IdentityRemovalPass(program.blocks);
    Optimization::DeadCodeEliminationPass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::CollectShaderInfoPass(program, profile);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
}

struct MismatchSummary {
    u32 lane_id{};
    u32 read_lane_zero{};
    u32 read_first_lane{};
    u32 ballot_low{};
    u32 ballot_high{};
};

class Wave64Conformance : public testing::Test {
protected:
    static void TearDownTestSuite() {
        gcn_test::Runner::DestroyInstance();
    }
};

MismatchSummary CompareGuestWave64(const WaveOutput& output) {
    MismatchSummary mismatches{};
    for (u32 invocation = 0; invocation < GuestWaveSize; ++invocation) {
        const auto& record = output[invocation];
        mismatches.lane_id += record.lane_id != invocation;
        mismatches.read_lane_zero += record.read_lane_zero != 0;
        mismatches.read_first_lane += record.read_first_lane != 0;
        mismatches.ballot_low += record.ballot_low != 0x80000001U;
        mismatches.ballot_high += record.ballot_high != 0x80000001U;
    }
    return mismatches;
}

TEST_F(Wave64Conformance, NativeControlProvesTwoSubgroup32Halves) {
    auto runner_result = gcn_test::Runner::subgroup32_instance();
    if (!runner_result && runner_result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << runner_result.error().message;
    }
    ASSERT_TRUE(runner_result.has_value()) << runner_result.error().message;
    auto* const runner = *runner_result;
    ASSERT_EQ(runner->effective_compute_subgroup_size(), 32U)
        << "Selected Vulkan device: " << runner->device_name();

    const auto output = runner->run<WaveOutput>(EmitWaveProbe(false));
    ASSERT_TRUE(output.has_value()) << output.error().message;
    for (u32 invocation = 0; invocation < GuestWaveSize; ++invocation) {
        SCOPED_TRACE(invocation);
        const auto& record = (*output)[invocation];
        EXPECT_EQ(record.lane_id, invocation & 31U);
        EXPECT_EQ(record.read_lane_zero, invocation < 32U ? 0U : 32U);
        EXPECT_EQ(record.read_first_lane, invocation < 32U ? 0U : 32U);
        EXPECT_EQ(record.ballot_low, 0x80000001U);
        EXPECT_EQ(record.ballot_high, 0U);
    }
}

TEST_F(Wave64Conformance, GuestWave64PrimitivesSpanBothNativeSubgroups) {
    auto runner_result = gcn_test::Runner::subgroup32_instance();
    if (!runner_result && runner_result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << runner_result.error().message;
    }
    ASSERT_TRUE(runner_result.has_value()) << runner_result.error().message;
    auto* const runner = *runner_result;
    ASSERT_EQ(runner->effective_compute_subgroup_size(), 32U)
        << "Selected Vulkan device: " << runner->device_name();

    const auto output = runner->run<WaveOutput>(EmitWaveProbe(true));
    ASSERT_TRUE(output.has_value()) << output.error().message;
    const auto mismatches = CompareGuestWave64(*output);

    EXPECT_EQ(mismatches.lane_id, 0U);
    EXPECT_EQ(mismatches.read_lane_zero, 0U);
    EXPECT_EQ(mismatches.read_first_lane, 0U);
    EXPECT_EQ(mismatches.ballot_low, 0U);
    EXPECT_EQ(mismatches.ballot_high, 0U);
}

TEST_F(Wave64Conformance,
       BarrierBracketedMergeReductionsSpanBothSubgroup32HalvesForOneThousandRuns) {
    const auto spirv = EmitBracketedTwoReductionProbe();
    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInLocalInvocationId));
    EXPECT_FALSE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
    EXPECT_FALSE(HasSpirvOpcode(spirv, spv::Op::OpGroupNonUniformBroadcast));

#ifdef SHADPS4_SPIRV_VAL_PATH
    const auto module =
        std::filesystem::temp_directory_path() / "shadps4-wave64-bracketed-reduction.spv";
    {
        std::ofstream file{module, std::ios::binary | std::ios::trunc};
        ASSERT_TRUE(file.good());
        file.write(reinterpret_cast<const char*>(spirv.data()),
                   static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
    }
    EXPECT_EQ(RunSpirvValidator(module), 0);
    std::error_code remove_error;
    std::filesystem::remove(module, remove_error);
#endif

    auto runner_result = gcn_test::Runner::subgroup32_instance();
    if (!runner_result && runner_result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << runner_result.error().message;
    }
    ASSERT_TRUE(runner_result.has_value()) << runner_result.error().message;
    auto* const runner = *runner_result;
    ASSERT_EQ(runner->effective_compute_subgroup_size(), 32U)
        << "Selected Vulkan device: " << runner->device_name();
    std::cout << "Selected Vulkan device: " << runner->device_name() << '\n';

    u64 mismatches{};
    for (u32 repeat = 0; repeat < 1000; ++repeat) {
        const auto output = runner->run<ReductionOutput>(spirv);
        ASSERT_TRUE(output.has_value()) << "repeat " << repeat << ": " << output.error().message;
        for (const auto& record : *output) {
            mismatches += record.first != 63U;
            mismatches += record.second != 194U;
        }
    }
    EXPECT_EQ(mismatches, 0U);
}

TEST_F(Wave64Conformance, PartialWaveNeutralReadsPreserveThirtyTwoInvocationsForTwoHundredRuns) {
    auto runner_result = gcn_test::Runner::subgroup32_instance();
    if (!runner_result && runner_result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << runner_result.error().message;
    }
    ASSERT_TRUE(runner_result.has_value()) << runner_result.error().message;
    auto* const runner = *runner_result;
    ASSERT_EQ(runner->effective_compute_subgroup_size(), 32U)
        << "Selected Vulkan device: " << runner->device_name();
    for (const auto shape : {std::array{32U, 1U, 1U}, std::array{4U, 4U, 2U}}) {
        const auto spirv = EmitPartialWaveNeutralReductionProbe(shape);
        EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
        EXPECT_FALSE(HasSpirvOpcode(spirv, spv::Op::OpControlBarrier));
        for (u32 repeat = 0; repeat < 100U; ++repeat) {
            const auto output = runner->run<PartialReductionOutput>(spirv);
            ASSERT_TRUE(output.has_value()) << output.error().message;
            for (u32 invocation = 0; invocation < output->size(); ++invocation) {
                SCOPED_TRACE(invocation);
                const auto& record = (*output)[invocation];
                EXPECT_EQ(record.minimum, std::bit_cast<u32>(1.0f));
                EXPECT_EQ(record.sum, std::bit_cast<u32>(528.0f));
                EXPECT_EQ(record.upper_minimum, 0x7f800000U);
                EXPECT_EQ(record.upper_sum, 0U);
                EXPECT_EQ(record.invocation, invocation);
            }
        }
    }
}

TEST_F(Wave64Conformance, ConfinedEntryReductionsPreserveLaterNativeShufflesForOneHundredRuns) {
    const auto spirv = EmitEntryReductionWithLaterNativeShuffleProbe();
    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInLocalInvocationId));
    EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
    EXPECT_TRUE(HasSpirvOpcode(spirv, spv::Op::OpGroupNonUniformShuffle));
    EXPECT_FALSE(HasSpirvOpcode(spirv, spv::Op::OpGroupNonUniformBroadcast));

#ifdef SHADPS4_SPIRV_VAL_PATH
    const auto module =
        std::filesystem::temp_directory_path() / "shadps4-wave64-confined-entry-reduction.spv";
    {
        std::ofstream file{module, std::ios::binary | std::ios::trunc};
        ASSERT_TRUE(file.good());
        file.write(reinterpret_cast<const char*>(spirv.data()),
                   static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
    }
    ASSERT_EQ(RunSpirvValidator(module), 0);
    std::error_code remove_error;
    std::filesystem::remove(module, remove_error);
#endif

    auto runner_result = gcn_test::Runner::subgroup32_instance();
    if (!runner_result && runner_result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << runner_result.error().message;
    }
    ASSERT_TRUE(runner_result.has_value()) << runner_result.error().message;
    auto* const runner = *runner_result;
    ASSERT_EQ(runner->effective_compute_subgroup_size(), 32U)
        << "Selected Vulkan device: " << runner->device_name();

    u64 mismatches{};
    for (u32 repeat = 0; repeat < 100; ++repeat) {
        const auto output = runner->run<EntryReductionOutput>(spirv);
        ASSERT_TRUE(output.has_value()) << "repeat " << repeat << ": " << output.error().message;
        std::array<u32, 32> native_lane_counts{};
        for (u32 invocation = 0; invocation < GuestWaveSize; ++invocation) {
            const auto& record = (*output)[invocation];
            mismatches += record.first != 63U;
            mismatches += record.second != 194U;
            if (record.later_native_lane >= native_lane_counts.size()) {
                ++mismatches;
                continue;
            }
            ++native_lane_counts[record.later_native_lane];
            const bool selected = (record.later_native_lane & 1U) == 0;
            mismatches +=
                record.later_native_peer !=
                (selected ? (record.later_native_lane ^ 2U) + 1000U : UnselectedLaneSentinel);
        }
        for (const u32 count : native_lane_counts) {
            mismatches += count != 2U;
        }
    }
    EXPECT_EQ(mismatches, 0U);
}

TEST_F(Wave64Conformance, ProvenFirstUniformBodyRunsOrSkipsTheWholeWaveForOneHundredRunsEach) {
    auto runner_result = gcn_test::Runner::subgroup32_instance();
    if (!runner_result && runner_result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << runner_result.error().message;
    }
    ASSERT_TRUE(runner_result.has_value()) << runner_result.error().message;
    auto* const runner = *runner_result;
    ASSERT_EQ(runner->effective_compute_subgroup_size(), 32U)
        << "Selected Vulkan device: " << runner->device_name();
    u64 mismatches{};
    for (const bool enter_body : {false, true}) {
        SCOPED_TRACE(enter_body);
        const auto spirv = EmitFirstUniformBodyProbe(enter_body);
        EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInLocalInvocationId));
        EXPECT_FALSE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
        EXPECT_FALSE(HasSpirvOpcode(spirv, spv::Op::OpGroupNonUniformBroadcast));
        EXPECT_FALSE(HasSpirvOpcode(spirv, spv::Op::OpGroupNonUniformShuffle));
#ifdef SHADPS4_SPIRV_VAL_PATH
        const auto module = std::filesystem::temp_directory_path() /
                            (enter_body ? "shadps4-wave64-first-uniform-body-true.spv"
                                        : "shadps4-wave64-first-uniform-body-false.spv");
        {
            std::ofstream file{module, std::ios::binary | std::ios::trunc};
            ASSERT_TRUE(file.good());
            file.write(reinterpret_cast<const char*>(spirv.data()),
                       static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
        }
        ASSERT_EQ(RunSpirvValidator(module), 0);
        std::error_code remove_error;
        std::filesystem::remove(module, remove_error);
#endif
        for (u32 repeat = 0; repeat < 100; ++repeat) {
            const auto output = runner->run<ReductionOutput>(spirv);
            ASSERT_TRUE(output.has_value())
                << "repeat " << repeat << ": " << output.error().message;
            for (const auto& record : *output) {
                mismatches += record.first != (enter_body ? 148U : UnselectedLaneSentinel);
                mismatches += record.second != (enter_body ? 264U : UnselectedLaneSentinel);
            }
        }
    }
    EXPECT_EQ(mismatches, 0U);
}

TEST_F(Wave64Conformance, DivergentDiamondsRejoinBeforeWholeWaveReadsForOneHundredRunsEach) {
    auto runner_result = gcn_test::Runner::subgroup32_instance();
    if (!runner_result && runner_result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << runner_result.error().message;
    }
    ASSERT_TRUE(runner_result.has_value()) << runner_result.error().message;
    auto* const runner = *runner_result;
    ASSERT_EQ(runner->effective_compute_subgroup_size(), 32U)
        << "Selected Vulkan device: " << runner->device_name();
    u64 mismatches{};
    for (const bool enter_body : {false, true}) {
        SCOPED_TRACE(enter_body);
        const auto spirv = EmitFirstUniformBodyProbe(enter_body, true);
        EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInLocalInvocationId));
        EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
        EXPECT_FALSE(HasSpirvOpcode(spirv, spv::Op::OpGroupNonUniformBroadcast));
        EXPECT_FALSE(HasSpirvOpcode(spirv, spv::Op::OpGroupNonUniformShuffle));
#ifdef SHADPS4_SPIRV_VAL_PATH
        const auto module = std::filesystem::temp_directory_path() /
                            (enter_body ? "shadps4-wave64-nested-reconvergence-true.spv"
                                        : "shadps4-wave64-nested-reconvergence-false.spv");
        {
            std::ofstream file{module, std::ios::binary | std::ios::trunc};
            ASSERT_TRUE(file.good());
            file.write(reinterpret_cast<const char*>(spirv.data()),
                       static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
        }
        ASSERT_EQ(RunSpirvValidator(module), 0);
        std::error_code remove_error;
        std::filesystem::remove(module, remove_error);
#endif
        for (u32 repeat = 0; repeat < 100; ++repeat) {
            const auto output = runner->run<NestedReductionOutput>(spirv);
            ASSERT_TRUE(output.has_value())
                << "repeat " << repeat << ": " << output.error().message;
            for (u32 invocation = 0; invocation < GuestWaveSize; ++invocation) {
                const auto& record = (*output)[invocation];
                mismatches += record.first != (enter_body ? 348U : UnselectedLaneSentinel);
                mismatches += record.second != (enter_body ? 664U : UnselectedLaneSentinel);
                const bool enters_interior = enter_body && (invocation & 1U) == 0U;
                mismatches += enters_interior
                                  ? record.interior_native_lane >= 32U
                                  : record.interior_native_lane != UnselectedLaneSentinel;
                u32 payload = invocation;
                if ((invocation & 1U) == 0U) {
                    payload += (invocation & 2U) == 0U ? 100U : 0U;
                    payload += (invocation & 4U) == 0U ? 200U : 0U;
                }
                mismatches +=
                    record.merged_payload != (enter_body ? payload : UnselectedLaneSentinel);
            }
        }
    }
    EXPECT_EQ(mismatches, 0U);
}

TEST_F(Wave64Conformance, TopLevelMergePreservesMixedU64AndU32WorkgroupMemoryForTwoHundredRuns) {
    auto runner_result = gcn_test::Runner::subgroup32_instance(true);
    if (!runner_result && runner_result.error().code == gcn_test::Error::NoSuitableDevice) {
        GTEST_SKIP() << "Mixed-layout GPU proof unsupported, not executed: "
                     << runner_result.error().message;
    }
    ASSERT_TRUE(runner_result.has_value()) << runner_result.error().message;
    auto* const runner = *runner_result;
    ASSERT_EQ(runner->effective_compute_subgroup_size(), 32U);
    ASSERT_TRUE(runner->explicit_workgroup_layout_enabled());
    u64 mismatches{};
    for (const bool select_even : {false, true}) {
        SCOPED_TRACE(select_even);
        const auto spirv =
            EmitTopLevelMergeProbe(select_even, runner->explicit_workgroup_layout_enabled());
        EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInLocalInvocationId));
        EXPECT_TRUE(HasBuiltIn(spirv, spv::BuiltInSubgroupLocalInvocationId));
        EXPECT_FALSE(HasSpirvOpcode(spirv, spv::Op::OpGroupNonUniformBroadcast));
        EXPECT_FALSE(HasSpirvOpcode(spirv, spv::Op::OpGroupNonUniformShuffle));
#ifdef SHADPS4_SPIRV_VAL_PATH
        const auto module = std::filesystem::temp_directory_path() /
                            (select_even ? "shadps4-wave64-top-merge-even.spv"
                                         : "shadps4-wave64-top-merge-odd.spv");
        {
            std::ofstream file{module, std::ios::binary | std::ios::trunc};
            ASSERT_TRUE(file.good());
            file.write(reinterpret_cast<const char*>(spirv.data()),
                       static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
        }
        ASSERT_EQ(RunSpirvValidator(module), 0);
        std::error_code remove_error;
        std::filesystem::remove(module, remove_error);
#endif
        for (u32 repeat = 0; repeat < 100; ++repeat) {
            const auto output = runner->run<TopLevelReductionOutput>(spirv);
            ASSERT_TRUE(output.has_value())
                << "repeat " << repeat << ": " << output.error().message;
            const auto expected_payload = [&](u32 invocation) {
                const bool even_native_lane = ((*output)[invocation].native_lane & 1U) == 0U;
                return invocation + (even_native_lane == select_even ? 100U : 0U);
            };
            const auto expected_low = expected_payload(16U);
            const auto expected_high = expected_payload(48U);
            std::array<u32, 32> native_lane_counts{};
            for (u32 invocation = 0; invocation < GuestWaveSize; ++invocation) {
                const auto& record = (*output)[invocation];
                mismatches += record.first != std::max(expected_low, expected_high);
                mismatches += record.second != expected_low + expected_high;
                mismatches += record.payload != expected_payload(invocation);
                mismatches += record.high_word != invocation + 1000U;
                if (record.native_lane < native_lane_counts.size()) {
                    ++native_lane_counts[record.native_lane];
                } else {
                    ++mismatches;
                }
            }
            for (const auto count : native_lane_counts) {
                mismatches += count != 2U;
            }
        }
    }
    EXPECT_EQ(mismatches, 0U);
    // The opt-in feature request must not silently change later default test devices.
    auto default_runner = gcn_test::Runner::subgroup32_instance();
    ASSERT_TRUE(default_runner.has_value()) << default_runner.error().message;
    EXPECT_FALSE((*default_runner)->explicit_workgroup_layout_enabled());
}

} // namespace
