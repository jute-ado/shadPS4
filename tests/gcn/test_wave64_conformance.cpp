// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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

} // namespace

