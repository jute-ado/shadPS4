// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <cstdlib>
#endif

#include <gtest/gtest.h>

#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/opcodes.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv/unified1/spirv.hpp"

namespace Shader {
namespace {

// Synthetic GCN, assembled here rather than copied from a commercial shader. Unlike the
// single-instruction helper, this fixture must call the production decode/CFG/structurizer/SSA/
// SRT/resource pipeline; it does not construct IR or call ResourceTrackingPassStub.
struct ScalarPointerCode {
    std::vector<u32> words{0xBEEB03FFU, 0U}; // s_mov_b32 vcc_hi, literal(0)

    void MovePointer(u32 source) {
        words.push_back(0xBE800000U | (6U << 16U) |
                        (static_cast<u32>(Gcn::OpcodeSOP1::S_MOV_B64) << 8U) | source);
    }

    void Compare(u32 sgpr, Gcn::OpcodeSOPC opcode = Gcn::OpcodeSOPC::S_CMP_EQ_U32) {
        words.push_back(0xBF000000U | (static_cast<u32>(opcode) << 16U) | (128U << 8U) |
                        sgpr); // compare register with inline integer zero
    }

    size_t Branch(Gcn::OpcodeSOPP opcode) {
        const size_t location = words.size();
        words.push_back(0xBF800000U | (static_cast<u32>(opcode) << 16U));
        return location;
    }

    void Target(size_t branch, size_t target) {
        const auto offset = static_cast<s64>(target) - static_cast<s64>(branch) - 1;
        ASSERT_GE(offset, std::numeric_limits<s16>::min());
        ASSERT_LE(offset, std::numeric_limits<s16>::max());
        words[branch] |= static_cast<u16>(static_cast<s16>(offset));
    }

    void StoreThroughPointer(u32 value) {
        // s_load_dwordx4 s[8:11], s[6:7], 0; s_waitcnt lgkmcnt(0)
        words.push_back(0xC0000000U | (static_cast<u32>(Gcn::OpcodeSMRD::S_LOAD_DWORDX4) << 22U) |
                        (8U << 15U) | (3U << 9U) | (1U << 8U));
        words.push_back(0xBF8C007FU);
        words.push_back(0x7E000200U | (128U + value)); // v_mov_b32 v0, inline integer
        words.push_back(0xE0700000U);                  // buffer_store_dword v0, v0, s[8:11], 0
        words.push_back(0x80020000U);
    }

    void End() {
        words.push_back(0xBF810000U); // s_endpgm
    }
};

ScalarPointerCode ConditionalPointers(bool branch_local_use, bool distinct_sources,
                                      bool initialize_else = true) {
    ScalarPointerCode code;
    code.Compare(4U);
    const size_t otherwise = code.Branch(Gcn::OpcodeSOPP::S_CBRANCH_SCC0);
    code.MovePointer(0U);
    if (branch_local_use) {
        code.StoreThroughPointer(1U);
    }
    const size_t merge = code.Branch(Gcn::OpcodeSOPP::S_BRANCH);
    code.Target(otherwise, code.words.size());
    if (initialize_else) {
        code.MovePointer(distinct_sources ? 2U : 0U);
    }
    if (branch_local_use) {
        code.StoreThroughPointer(2U);
    }
    code.Target(merge, code.words.size());
    if (!branch_local_use) {
        code.StoreThroughPointer(3U);
    }
    code.End();
    return code;
}

ScalarPointerCode LoopInvariantPointer() {
    ScalarPointerCode code;
    code.MovePointer(0U);
    code.words.push_back(0xBE8C0304U); // s_mov_b32 s12, s4 (runtime loop count)
    const size_t loop = code.words.size();
    code.StoreThroughPointer(1U);
    code.MovePointer(6U);              // loop-carried self-edge, same exact pointer
    code.words.push_back(0x808C810CU); // s_sub_u32 s12, s12, 1
    code.Compare(12U, Gcn::OpcodeSOPC::S_CMP_LG_U32);
    code.Target(code.Branch(Gcn::OpcodeSOPP::S_CBRANCH_SCC1), loop);
    code.End();
    return code;
}

struct ProductionTranslationFixture {
    static constexpr std::array<VAddr, 2> TableAddresses{0x100000U, 0x200000U};
    static constexpr std::array<VAddr, 2> BufferAddresses{0x300000U, 0x400000U};

    Info info{};
    Pools pools{};
    RuntimeInfo runtime{};
    Profile profile{};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    std::array<AmdGpu::Buffer, 2> descriptors{};
    std::vector<std::pair<VAddr, u64>> reads;
    bool failed{};
    const bool previous_dma = EmulatorSettings.IsDirectMemoryAccessEnabled();
    const bool previous_dump = EmulatorSettings.IsDumpShaders();

    ProductionTranslationFixture() {
        EmulatorSettings.SetDirectMemoryAccessEnabled(false);
        EmulatorSettings.SetDumpShaders(false);
        info.stage = Stage::Compute;
        info.l_stage = LogicalStage::Compute;
        info.user_data = user_data;
        runtime.Initialize(Stage::Compute);
        runtime.num_user_data = 5U;
        runtime.cs_info.workgroup_size = {1U, 1U, 1U};
        profile.supported_spirv = 0x00010600U;
        profile.subgroup_size = 32U;
        for (size_t i = 0; i < TableAddresses.size(); ++i) {
            user_data[2U * i] = static_cast<u32>(TableAddresses[i]);
            user_data[2U * i + 1U] = static_cast<u32>(TableAddresses[i] >> 32U);
            auto& descriptor = descriptors[i];
            descriptor.base_address = BufferAddresses[i];
            descriptor.stride = 4U;
            descriptor.num_records = 64U;
            descriptor.data_format = static_cast<u32>(AmdGpu::DataFormat::Format32);
            descriptor.num_format = static_cast<u32>(AmdGpu::NumberFormat::Uint);
        }
        user_data[4] = 3U;
    }

    ~ProductionTranslationFixture() {
        EmulatorSettings.SetDirectMemoryAccessEnabled(previous_dma);
        EmulatorSettings.SetDumpShaders(previous_dump);
    }

    static bool PS4_SYSV_ABI Read(void* opaque, VAddr address, void* output, u64 size) {
        auto& self = *static_cast<ProductionTranslationFixture*>(opaque);
        self.reads.emplace_back(address, size);
        for (size_t i = 0; i < TableAddresses.size(); ++i) {
            if (address < TableAddresses[i]) {
                continue;
            }
            const u64 offset = address - TableAddresses[i];
            if (offset <= sizeof(AmdGpu::Buffer) && size <= sizeof(AmdGpu::Buffer) - offset) {
                std::memcpy(output, reinterpret_cast<const u8*>(&self.descriptors[i]) + offset,
                            size);
                return true;
            }
        }
        self.failed = true;
        return false;
    }

    IR::Program Translate(const ScalarPointerCode& code) {
        return TranslateProgram(code.words, pools, info, runtime, profile,
                                SrtMemoryReader{Read, this, &failed});
    }

    std::vector<u32> Emit(const IR::Program& program) {
        Backend::Bindings bindings{};
        return Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
    }

    void ExpectBuffer(VAddr address) const {
        const auto found = std::ranges::find_if(info.buffers, [&](const BufferResource& resource) {
            return resource.buffer_type == BufferType::Guest &&
                   info.ReadUdSharp<AmdGpu::Buffer>(resource.sharp_idx).base_address == address;
        });
        ASSERT_NE(found, info.buffers.end());
        EXPECT_TRUE(found->is_written);
        EXPECT_FALSE(found->is_formatted);
        EXPECT_FALSE(static_cast<bool>(found->inline_cbuf));
    }
};

size_t CountSpirv(const std::vector<u32>& words, spv::Op opcode) {
    size_t count{};
    for (size_t i = 5U; i < words.size();) {
        const u32 length = words[i] >> 16U;
        if (length == 0U || length > words.size() - i) {
            ADD_FAILURE() << "Invalid SPIR-V instruction length";
            return 0U;
        }
        count += static_cast<spv::Op>(words[i] & 0xFFFFU) == opcode;
        i += length;
    }
    return count;
}

std::string DescribeProgram(const IR::Program& program) {
    std::map<const IR::Inst*, size_t> instructions;
    std::map<const IR::Block*, size_t> blocks;
    size_t index{};
    for (const auto* block : program.blocks) {
        blocks.emplace(block, index++);
    }
    std::string description;
    for (const auto* block : program.blocks) {
        description += IR::DumpBlock(*block, blocks, instructions, index) + '\n';
    }
    return description;
}

void ExpectValidSpirv(const std::vector<u32>& words) {
    ASSERT_GE(words.size(), 5U);
    EXPECT_EQ(words[0], spv::MagicNumber);
#ifdef SHADPS4_SPIRV_VAL_PATH
    static std::atomic<u32> serial{};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
                      ("shadps4_translate_program_" + std::to_string(stamp) + "_" +
                       std::to_string(serial++) + ".spv");
    ASSERT_FALSE(std::filesystem::exists(path));
    {
        std::ofstream file{path, std::ios::binary};
        ASSERT_TRUE(file.good());
        file.write(reinterpret_cast<const char*>(words.data()),
                   static_cast<std::streamsize>(words.size() * sizeof(u32)));
        ASSERT_TRUE(file.good());
    }
#ifdef _WIN32
    const int result =
        static_cast<int>(_spawnl(_P_WAIT, SHADPS4_SPIRV_VAL_PATH, SHADPS4_SPIRV_VAL_PATH,
                                 "--target-env", "vulkan1.3", path.string().c_str(), nullptr));
#else
    const std::string command = std::string{"\""} + SHADPS4_SPIRV_VAL_PATH +
                                "\" --target-env vulkan1.3 \"" + path.string() + "\"";
    const int result = std::system(command.c_str());
#endif
    EXPECT_EQ(result, 0);
    std::error_code error;
    std::filesystem::remove(path, error);
#else
    GTEST_SKIP() << "spirv-val is not installed; production translation assertions still executed";
#endif
}

TEST(TranslateProgram, BranchLocalDistinctPointersUseRealDescriptorsAndStructuredSpirv) {
    ProductionTranslationFixture fixture;
    const auto code = ConditionalPointers(true, true);
    const auto program = fixture.Translate(code);

    ASSERT_FALSE(fixture.info.translation_failed);
    ASSERT_FALSE(fixture.failed);
    EXPECT_GT(program.blocks.size(), 1U);
    ASSERT_EQ(fixture.info.buffers.size(), 2U);
    fixture.ExpectBuffer(ProductionTranslationFixture::BufferAddresses[0]);
    fixture.ExpectBuffer(ProductionTranslationFixture::BufferAddresses[1]);
    EXPECT_EQ(fixture.reads,
              (std::vector<std::pair<VAddr, u64>>{
                  {ProductionTranslationFixture::TableAddresses[0], sizeof(AmdGpu::Buffer)},
                  {ProductionTranslationFixture::TableAddresses[1], sizeof(AmdGpu::Buffer)}}));
    const auto spirv = fixture.Emit(program);
    EXPECT_GE(CountSpirv(spirv, spv::Op::OpBranchConditional), 1U);
    EXPECT_GE(CountSpirv(spirv, spv::Op::OpSelectionMerge), 1U);
    EXPECT_EQ(CountSpirv(spirv, spv::Op::OpStore), 2U);
    ExpectValidSpirv(spirv);
}

TEST(TranslateProgram, SamePointerDefinedInBothBranchesRemainsOneCapturedResource) {
    ProductionTranslationFixture fixture;
    const auto program = fixture.Translate(ConditionalPointers(false, false));

    ASSERT_FALSE(fixture.info.translation_failed) << DescribeProgram(program);
    ASSERT_FALSE(fixture.failed);
    ASSERT_EQ(fixture.info.buffers.size(), 1U);
    fixture.ExpectBuffer(ProductionTranslationFixture::BufferAddresses[0]);
    EXPECT_EQ(fixture.reads,
              (std::vector<std::pair<VAddr, u64>>{
                  {ProductionTranslationFixture::TableAddresses[0], sizeof(AmdGpu::Buffer)}}));
    ExpectValidSpirv(fixture.Emit(program));
}

TEST(TranslateProgram, LoopCarriedSelfEdgePreservesUniquePointerAndValidPhiDominance) {
    ProductionTranslationFixture fixture;
    const auto program = fixture.Translate(LoopInvariantPointer());

    ASSERT_FALSE(fixture.info.translation_failed);
    ASSERT_FALSE(fixture.failed);
    ASSERT_EQ(fixture.info.buffers.size(), 1U);
    fixture.ExpectBuffer(ProductionTranslationFixture::BufferAddresses[0]);
    const auto spirv = fixture.Emit(program);
    EXPECT_GE(CountSpirv(spirv, spv::Op::OpLoopMerge), 1U);
    EXPECT_GE(CountSpirv(spirv, spv::Op::OpPhi), 1U);
    ExpectValidSpirv(spirv);
}

TEST(TranslateProgram, DefinedPointerCanRefineAnActuallyUndefinedAlternative) {
    ProductionTranslationFixture fixture;
    const auto program = fixture.Translate(ConditionalPointers(false, false, false));

    // Unlike the exhaustive if/else case, one original path has no guest definition at all.
    // IR UndefU32 may take the sole defined source's value; a second real source may not.
    ASSERT_FALSE(fixture.info.translation_failed) << DescribeProgram(program);
    ASSERT_FALSE(fixture.failed);
    ASSERT_EQ(fixture.info.buffers.size(), 1U);
    fixture.ExpectBuffer(ProductionTranslationFixture::BufferAddresses[0]);
    ExpectValidSpirv(fixture.Emit(program));
}

TEST(TranslateProgram, GenuinelyAmbiguousPointerJoinFailsBeforeHostDescriptorUse) {
    ProductionTranslationFixture fixture;
    const auto program = fixture.Translate(ConditionalPointers(false, true));

    EXPECT_TRUE(fixture.info.translation_failed);
    EXPECT_TRUE(fixture.info.buffers.empty());
    EXPECT_TRUE(fixture.reads.empty());
    EXPECT_GT(program.blocks.size(), 1U);
    // This representation has no runtime buffer-descriptor alternative. Do not emit SPIR-V
    // or replace an ambiguous pointer with whichever ancestor happened to be visited first.
}

} // namespace
} // namespace Shader
