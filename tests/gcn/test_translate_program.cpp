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
#include <optional>
#include <string>
#include <string_view>
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
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/dynamic_image_table_matcher.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv/unified1/spirv.hpp"

namespace Shader {
namespace {

TEST(TranslateProgram, RejectsEmptyShaderWithoutReadingPastInput) {
    Pools pools{};
    Info info{};
    info.stage = Stage::Vertex;
    info.l_stage = LogicalStage::Vertex;
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Vertex);
    Profile profile{};

    const auto program = TranslateProgram(std::span<const u32>{}, pools, info, runtime, profile);

    EXPECT_TRUE(info.translation_failed);
    EXPECT_TRUE(program.ins_list.empty());
}

TEST(TranslateProgram, RejectsTruncatedOrUnexpectedPrologue) {
    Pools pools{};
    Info info{};
    info.stage = Stage::Vertex;
    info.l_stage = LogicalStage::Vertex;
    RuntimeInfo runtime{};
    runtime.Initialize(Stage::Vertex);
    Profile profile{};
    const std::array<u32, 2> malformed{0xDEADBEEFU, 0U};

    const auto program = TranslateProgram(malformed, pools, info, runtime, profile);

    EXPECT_TRUE(info.translation_failed);
    EXPECT_TRUE(program.ins_list.empty());
}

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

constexpr std::array<u32, 2> VertexImageTableOffsets{0x100U, 0x1000U};

ScalarPointerCode VertexImageTables(bool dynamic_tables) {
    ScalarPointerCode code;
    const auto load = [&](Gcn::OpcodeSMRD opcode, u32 destination, u32 base, u32 offset,
                          bool immediate) {
        code.words.push_back(0xC0000000U | (static_cast<u32>(opcode) << 22U) |
                             (destination << 15U) | ((base / 2U) << 9U) |
                             (static_cast<u32>(immediate) << 8U) | offset);
    };
    const auto scalar = [&](Gcn::OpcodeSOP2 opcode, u32 destination, u32 lhs, u32 rhs) {
        code.words.push_back(0x80000000U | (static_cast<u32>(opcode) << 23U) |
                             (destination << 16U) | (rhs << 8U) | lhs);
    };
    const auto sample = [&] {
        code.words.push_back(0xBF8C007FU); // s_waitcnt lgkmcnt(0)
        code.words.push_back(0xF0000100U |
                             (static_cast<u32>(Gcn::OpcodeMIMG::IMAGE_SAMPLE_LZ) << 18U));
        code.words.push_back((6U << 21U) | (4U << 16U) | (4U << 8U));
        // image_sample_lz v4, v[0:1], s[16:23], s[24:27]; retain each result in v7.
        code.words.push_back(0xBF8C0F70U);
        code.words.push_back((static_cast<u32>(Gcn::OpcodeVOP2::V_ADD_F32) << 25U) | (7U << 17U) |
                             (7U << 9U) | (256U + 4U));
    };
    // One root contains two nested pointers: image table and static sampler.
    load(Gcn::OpcodeSMRD::S_LOAD_DWORDX4, 8U, 0U, 0U, true);
    code.words.push_back(0xBF8C007FU);
    load(Gcn::OpcodeSMRD::S_LOAD_DWORDX4, 24U, 10U, 0U, true);
    load(Gcn::OpcodeSMRD::S_LOAD_DWORDX8, 16U, 8U, 0U, true);
    code.words.insert(code.words.end(), {0x7E000280U, 0x7E020280U, 0x7E0E0280U});
    sample(); // A static image must coexist with both dynamic arrays.
    if (dynamic_tables) {
        code.words.push_back(0xBE9E0380U); // s_mov_b32 s30, 0
        const size_t loop = code.words.size();
        for (const u32 offset : VertexImageTableOffsets) {
            scalar(Gcn::OpcodeSOP2::S_MUL_I32, 28U, 30U, 128U + 32U);
            scalar(Gcn::OpcodeSOP2::S_ADD_I32, 28U, 28U, 255U);
            code.words.push_back(offset);
            scalar(Gcn::OpcodeSOP2::S_ADD_I32, 29U, 28U, 128U + 16U);
            // Two four-dword loads form one eight-dword image descriptor. The scalar
            // register offsets are byte offsets; both nested pointer halves stay invariant.
            load(Gcn::OpcodeSMRD::S_LOAD_DWORDX4, 20U, 8U, 29U, false);
            load(Gcn::OpcodeSMRD::S_LOAD_DWORDX4, 16U, 8U, 28U, false);
            sample();
        }
        scalar(Gcn::OpcodeSOP2::S_ADD_I32, 30U, 30U, 129U);
        code.words.push_back(0xBF000000U |
                             (static_cast<u32>(Gcn::OpcodeSOPC::S_CMP_LT_U32) << 16U) | (2U << 8U) |
                             30U); // runtime bound in user-data s2
        code.Target(code.Branch(Gcn::OpcodeSOPP::S_CBRANCH_SCC1), loop);
    }
    code.words.insert(code.words.end(), {0xF80008CFU, 0x07070707U}); // export position
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

struct VertexImageTranslationFixture : ProductionTranslationFixture {
    static constexpr VAddr Root = 0x100000U;
    static constexpr VAddr Images = 0x200000U;
    static constexpr VAddr Sampler = 0x300000U;
    static constexpr VAddr StaticImage = 0x400000U;
    static constexpr std::array<VAddr, 2> TableImageAddresses{0x500000U, 0xA00000U};
    std::array<u64, 2> roots{Images, Sampler};
    std::array<u8, 0x2000> image_bytes{};
    AmdGpu::Sampler sampler{};
    bool second_table_unreadable{};

    VertexImageTranslationFixture() {
        info.stage = Stage::Vertex;
        info.l_stage = LogicalStage::Vertex;
        runtime.Initialize(Stage::Vertex);
        runtime.num_user_data = 3U;
        runtime.vs_info.num_outputs = 1U;
        user_data = {};
        user_data[0] = static_cast<u32>(Root);
        user_data[2] = 3U;
        profile.supports_sampled_image_array_nonuniform_indexing = true;
        profile.max_sampled_images_per_stage = 256U;
        profile.max_sampled_images_per_set = 256U;
        profile.max_resources_per_stage = 256U;
        profile.max_sampled_images_per_descriptor_heap_set = 256U;
        const auto write_image = [&](u32 offset, VAddr address) {
            auto image = AmdGpu::Image::Null(false);
            image.base_address = address >> 8U;
            image.width = 15U;
            image.height = 15U;
            std::memcpy(image_bytes.data() + offset, &image, sizeof(image));
        };
        write_image(0U, StaticImage);
        for (size_t table = 0; table < VertexImageTableOffsets.size(); ++table) {
            for (u32 entry = 0; entry < NUM_IMAGES; ++entry) {
                write_image(VertexImageTableOffsets[table] + entry * sizeof(AmdGpu::Image),
                            TableImageAddresses[table] + entry * 0x1000U);
            }
        }
    }

    static bool PS4_SYSV_ABI ReadImages(void* opaque, VAddr address, void* output, u64 size) {
        auto& self = *static_cast<VertexImageTranslationFixture*>(opaque);
        self.reads.emplace_back(address, size);
        const VAddr second_table = Images + VertexImageTableOffsets[1];
        if (self.second_table_unreadable && address <= second_table &&
            size > second_table - address) {
            self.failed = true;
            return false;
        }
        const auto copy = [&](VAddr base, const void* source, u64 length) {
            if (address < base || address - base > length || size > length - (address - base)) {
                return false;
            }
            std::memcpy(output, static_cast<const u8*>(source) + (address - base), size);
            return true;
        };
        if (copy(Root, self.roots.data(), sizeof(self.roots)) ||
            copy(Images, self.image_bytes.data(), self.image_bytes.size()) ||
            copy(Sampler, &self.sampler, sizeof(self.sampler))) {
            return true;
        }
        self.failed = true;
        return false;
    }

    IR::Program TranslateImages(bool dynamic_tables) {
        return TranslateProgram(VertexImageTables(dynamic_tables).words, pools, info, runtime,
                                profile, SrtMemoryReader{ReadImages, this, &failed});
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

struct SpirvInstructionView {
    spv::Op opcode{};
    size_t offset{};
    u32 word_count{};
};

std::vector<SpirvInstructionView> ParseSpirvInstructions(const std::vector<u32>& words) {
    std::vector<SpirvInstructionView> instructions;
    for (size_t offset = 5U; offset < words.size();) {
        const u32 word_count = words[offset] >> 16U;
        if (word_count == 0U || word_count > words.size() - offset) {
            ADD_FAILURE() << "Malformed SPIR-V instruction at word " << offset;
            return {};
        }
        instructions.push_back({static_cast<spv::Op>(words[offset] & 0xFFFFU), offset, word_count});
        offset += word_count;
    }
    return instructions;
}

std::string SpirvName(const std::vector<u32>& words, const SpirvInstructionView& instruction) {
    if (instruction.opcode != spv::Op::OpName || instruction.word_count < 3U) {
        return {};
    }
    const auto* const text = reinterpret_cast<const char*>(words.data() + instruction.offset + 2U);
    return std::string{text};
}

std::optional<u32> FindNamedSpirvId(const std::vector<u32>& words,
                                    const std::vector<SpirvInstructionView>& instructions,
                                    std::string_view name) {
    for (const auto& instruction : instructions) {
        if (SpirvName(words, instruction) == name) {
            return words[instruction.offset + 1U];
        }
    }
    return std::nullopt;
}

std::vector<SpirvInstructionView> FindSpirvFunction(
    const std::vector<u32>& words, const std::vector<SpirvInstructionView>& instructions,
    u32 function_id) {
    std::vector<SpirvInstructionView> body;
    bool in_function = false;
    for (const auto& instruction : instructions) {
        if (!in_function) {
            if (instruction.opcode == spv::Op::OpFunction && instruction.word_count >= 3U &&
                words[instruction.offset + 2U] == function_id) {
                in_function = true;
            }
            continue;
        }
        if (instruction.opcode == spv::Op::OpFunctionEnd) {
            break;
        }
        body.push_back(instruction);
    }
    return body;
}

struct NestedReadConstSpirvFixture {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    IR::Inst* nested_read{};

    NestedReadConstSpirvFixture() {
        info.stage = Stage::Compute;
        info.l_stage = LogicalStage::Compute;
        info.user_data = user_data;

        auto* const block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        program.post_order_blocks.push_back(block);

        IR::IREmitter ir{*block};
        ir.Prologue();
        const auto root = ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0),
                                                ir.GetUserData(IR::ScalarReg::S1));
        const auto low = ir.ReadConst(root, ir.Imm32(0U));
        const auto high = ir.ReadConst(root, ir.Imm32(1U));
        low.InstRecursive()->SetFlags(16U);
        high.InstRecursive()->SetFlags(17U);
        const auto nested_pointer = ir.CompositeConstruct(low, high);
        const auto nested = ir.ReadConst(nested_pointer, ir.Imm32(0U));
        nested_read = nested.InstRecursive();
        nested_read->SetFlags(18U);
        ir.Reference(nested);
        ir.Epilogue();

        program.syntax_list.emplace_back();
        program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
        program.syntax_list.back().data.block = block;
        program.syntax_list.emplace_back();
        program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    }
};

std::vector<u32> EmitNestedReadConstSpirv(NestedReadConstSpirvFixture& fixture) {
    const bool previous_dma = EmulatorSettings.IsDirectMemoryAccessEnabled();
    EmulatorSettings.SetDirectMemoryAccessEnabled(true);
    Profile profile{};
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {1U, 1U, 1U};
    Optimization::CollectShaderInfoPass(fixture.program, profile);
    Backend::Bindings bindings{};
    auto result = Backend::SPIRV::EmitSPIRV(profile, runtime_info, fixture.program, bindings);
    EmulatorSettings.SetDirectMemoryAccessEnabled(previous_dma);
    return result;
}

TEST(TranslateProgram, NestedReadConstPreservesPointerHalvesAndDelegatesFullAddressToBda) {
    NestedReadConstSpirvFixture fixture;
    const auto spirv = EmitNestedReadConstSpirv(fixture);

    const auto instructions = ParseSpirvInstructions(spirv);
    const auto read_const = FindNamedSpirvId(spirv, instructions, "read_const");
    const auto get_bda_pointer = FindNamedSpirvId(spirv, instructions, "get_bda_pointer");
    ASSERT_TRUE(read_const.has_value());
    ASSERT_TRUE(get_bda_pointer.has_value());
    const auto body = FindSpirvFunction(spirv, instructions, *read_const);
    ASSERT_FALSE(body.empty());

    u32 low_u32{};
    u32 high_u32{};
    u32 low_u64{};
    u32 high_u64{};
    u32 shift_result{};
    u32 combined{};
    u32 address{};
    bool saw_offset_shift = false;
    std::map<u32, u32> constants;
    for (const auto& instruction : instructions) {
        if (instruction.opcode == spv::Op::OpConstant && instruction.word_count == 4U) {
            constants.emplace(spirv[instruction.offset + 2U], spirv[instruction.offset + 3U]);
        }
    }
    for (const auto& instruction : body) {
        const auto at = instruction.offset;
        switch (instruction.opcode) {
        case spv::Op::OpCompositeExtract:
            ASSERT_GE(instruction.word_count, 5U);
            if (spirv[at + 4U] == 0U) {
                low_u32 = spirv[at + 2U];
            } else if (spirv[at + 4U] == 1U) {
                high_u32 = spirv[at + 2U];
            }
            break;
        case spv::Op::OpUConvert:
            ASSERT_EQ(instruction.word_count, 4U);
            if (spirv[at + 3U] == low_u32 && low_u32 != 0U) {
                low_u64 = spirv[at + 2U];
            } else if (spirv[at + 3U] == high_u32 && high_u32 != 0U) {
                high_u64 = spirv[at + 2U];
            }
            break;
        case spv::Op::OpShiftLeftLogical:
            ASSERT_EQ(instruction.word_count, 5U);
            if (spirv[at + 3U] == high_u64 && high_u64 != 0U &&
                constants.contains(spirv[at + 4U]) && constants.at(spirv[at + 4U]) == 32U) {
                shift_result = spirv[at + 2U];
            } else if (constants.contains(spirv[at + 4U]) && constants.at(spirv[at + 4U]) == 2U) {
                saw_offset_shift = true;
            }
            break;
        case spv::Op::OpBitwiseOr:
            ASSERT_EQ(instruction.word_count, 5U);
            if (spirv[at + 3U] == low_u64 && spirv[at + 4U] == shift_result && low_u64 != 0U &&
                shift_result != 0U) {
                combined = spirv[at + 2U];
            }
            break;
        case spv::Op::OpIAdd:
            ASSERT_EQ(instruction.word_count, 5U);
            if (spirv[at + 3U] == combined && combined != 0U) {
                address = spirv[at + 2U];
            }
            break;
        case spv::Op::OpFunctionCall:
            ASSERT_GE(instruction.word_count, 5U);
            if (spirv[at + 3U] == *get_bda_pointer) {
                ASSERT_NE(address, 0U);
                EXPECT_EQ(spirv[at + 4U], address);
            }
            break;
        default:
            break;
        }
    }
    EXPECT_NE(low_u32, 0U);
    EXPECT_NE(high_u32, 0U);
    EXPECT_NE(low_u64, 0U);
    EXPECT_NE(high_u64, 0U);
    EXPECT_NE(shift_result, 0U);
    EXPECT_NE(combined, 0U);
    EXPECT_NE(address, 0U);
    EXPECT_TRUE(saw_offset_shift);

    // No truncating mask is allowed between the reconstructed 64-bit pointer and the BDA helper.
    // Invalid high halves (such as 0xbf800000 from the Shadow diagnostic) must reach the helper,
    // where the full-width page-range check reports an invalid address instead of aliasing a low
    // resident page.
    for (const auto& instruction : body) {
        if (instruction.opcode != spv::Op::OpBitwiseAnd || instruction.word_count < 5U) {
            continue;
        }
        EXPECT_NE(spirv[instruction.offset + 3U], combined);
        EXPECT_NE(spirv[instruction.offset + 4U], combined);
        EXPECT_NE(spirv[instruction.offset + 3U], address);
        EXPECT_NE(spirv[instruction.offset + 4U], address);
    }

    const auto bda_body = FindSpirvFunction(spirv, instructions, *get_bda_pointer);
    ASSERT_FALSE(bda_body.empty());
    u32 bda_parameter{};
    u32 page{};
    size_t full_width_range_check{};
    size_t page_narrowing{};
    for (const auto& instruction : bda_body) {
        const auto at = instruction.offset;
        if (instruction.opcode == spv::Op::OpFunctionParameter && instruction.word_count >= 3U) {
            bda_parameter = spirv[at + 2U];
        } else if (instruction.opcode == spv::Op::OpShiftRightLogical &&
                   instruction.word_count == 5U && spirv[at + 3U] == bda_parameter &&
                   bda_parameter != 0U) {
            page = spirv[at + 2U];
        } else if (instruction.opcode == spv::Op::OpULessThan && instruction.word_count == 5U &&
                   (spirv[at + 3U] == page || spirv[at + 4U] == page) && page != 0U) {
            full_width_range_check = at;
        } else if (instruction.opcode == spv::Op::OpUConvert && instruction.word_count == 4U &&
                   spirv[at + 3U] == page && page != 0U) {
            page_narrowing = at;
        }
    }
    EXPECT_NE(bda_parameter, 0U);
    EXPECT_NE(page, 0U);
    EXPECT_NE(full_width_range_check, 0U);
    EXPECT_NE(page_narrowing, 0U);
    EXPECT_LT(full_width_range_check, page_narrowing);

    ExpectValidSpirv(spirv);
}

TEST(TranslateProgram, FourIndependentStaticRootsKeepDescriptorAndDataIdentityOnRefresh) {
    for (const bool matrix_first : {false, true}) {
        ProductionTranslationFixture fixture;
        fixture.runtime.num_user_data = 8U;
        fixture.user_data = {};
        struct Roots {
            std::array<std::array<std::array<u8, 128>, 4>, 2> bytes{};
            bool failed{};
            static VAddr Address(size_t generation, size_t root) {
                return 0x100000U + generation * 0x100000U + root * 0x10000U;
            }
            static bool PS4_SYSV_ABI Read(void* context, VAddr address, void* output, u64 size) {
                auto& self = *static_cast<Roots*>(context);
                for (size_t generation = 0; generation < 2; ++generation) {
                    for (size_t root = 0; root < 4; ++root) {
                        const auto base = Address(generation, root);
                        const auto& source = self.bytes[generation][root];
                        if (address >= base && address - base <= source.size() &&
                            size <= source.size() - (address - base)) {
                            std::memcpy(output, source.data() + (address - base), size);
                            return true;
                        }
                    }
                }
                self.failed = true;
                return false;
            }
        } roots;
        std::array<AmdGpu::Image, 2> images{};
        std::array<AmdGpu::Sampler, 2> samplers{};
        std::array<AmdGpu::Buffer, 2> outputs{};
        std::array<std::array<u32, 4>, 2> matrices{{
            {0x3E800000U, 0x3F000000U, 0x3F400000U, 0x3F800000U},
            {0x3F100000U, 0x3F200000U, 0x3F300000U, 0x3F400000U},
        }};
        for (size_t generation = 0; generation < 2; ++generation) {
            images[generation] = AmdGpu::Image::Null(false);
            images[generation].base_address = (0x900000U + generation * 0x10000U) >> 8U;
            images[generation].width = 15U;
            images[generation].height = 15U;
            samplers[generation].raw0 = (generation + 1U) << 9U;
            outputs[generation] = fixture.descriptors[generation];
            std::memcpy(roots.bytes[generation][0].data(), &images[generation],
                        sizeof(AmdGpu::Image));
            std::memcpy(roots.bytes[generation][1].data() + 20U, matrices[generation].data(), 16U);
            std::memcpy(roots.bytes[generation][2].data(), &samplers[generation],
                        sizeof(AmdGpu::Sampler));
            std::memcpy(roots.bytes[generation][3].data(), &outputs[generation],
                        sizeof(AmdGpu::Buffer));
        }
        const auto select_roots = [&](size_t generation) {
            for (size_t root = 0; root < 4; ++root) {
                const VAddr address = Roots::Address(generation, root);
                fixture.user_data[root * 2U] = static_cast<u32>(address);
                fixture.user_data[root * 2U + 1U] = static_cast<u32>(address >> 32U);
            }
        };
        select_roots(0U);
        ScalarPointerCode code;
        const auto load = [&](Gcn::OpcodeSMRD opcode, u32 destination, u32 base, u32 offset) {
            code.words.push_back(0xC0000000U | (static_cast<u32>(opcode) << 22U) |
                                 (destination << 15U) | ((base / 2U) << 9U) | (1U << 8U) | offset);
        };
        if (matrix_first) {
            load(Gcn::OpcodeSMRD::S_LOAD_DWORDX4, 32U, 2U, 5U);
        }
        load(Gcn::OpcodeSMRD::S_LOAD_DWORDX8, 16U, 0U, 0U);
        if (!matrix_first) {
            load(Gcn::OpcodeSMRD::S_LOAD_DWORDX4, 32U, 2U, 5U);
        }
        load(Gcn::OpcodeSMRD::S_LOAD_DWORDX4, 24U, 4U, 0U);
        load(Gcn::OpcodeSMRD::S_LOAD_DWORDX4, 8U, 6U, 0U);
        code.words.insert(code.words.end(), {0xBF8C007FU, 0x7E000220U, 0x7E020221U});
        code.words.push_back(0xF0000100U |
                             (static_cast<u32>(Gcn::OpcodeMIMG::IMAGE_SAMPLE_LZ) << 18U));
        code.words.push_back((6U << 21U) | (4U << 16U) | (4U << 8U));
        code.words.insert(code.words.end(), {0xBF8C0F70U, 0xE0700000U, 0x80020400U});
        code.End();
        const SrtMemoryReader reader{Roots::Read, &roots, &roots.failed};
        const auto program = TranslateProgram(code.words, fixture.pools, fixture.info,
                                              fixture.runtime, fixture.profile, reader);
        ASSERT_FALSE(fixture.info.translation_failed) << DescribeProgram(program);
        ASSERT_FALSE(roots.failed);
        const auto expect = [&](const Info& info, size_t generation) {
            ASSERT_EQ(info.images.size(), 1U);
            ASSERT_EQ(info.samplers.size(), 1U);
            EXPECT_EQ(info.images[0].GetSharp(info).Address(), images[generation].Address());
            EXPECT_EQ(info.samplers[0].GetSharp(info), samplers[generation]);
            const auto output = std::ranges::find_if(info.buffers, [](const auto& resource) {
                return resource.buffer_type == BufferType::Guest;
            });
            ASSERT_NE(output, info.buffers.end());
            EXPECT_EQ(static_cast<u64>(output->GetSharp(info).base_address),
                      static_cast<u64>(outputs[generation].base_address));
            EXPECT_TRUE(output->is_written);
            for (size_t component = 0; component < 2; ++component) {
                EXPECT_NE(std::ranges::find(info.flattened_ud_buf, matrices[generation][component]),
                          info.flattened_ud_buf.end());
            }
        };
        expect(fixture.info, 0U);
        const auto original = fixture.info;
        auto selected = fixture.info;
        select_roots(1U);
        ASSERT_TRUE(selected.RefreshFlatBuf(reader));
        expect(selected, 1U);
        expect(original, 0U); // A copied cached module retains its already captured bytes.
        select_roots(0U);
        ASSERT_TRUE(selected.RefreshFlatBuf(reader));
        expect(selected, 0U);
        EXPECT_FALSE(roots.failed);
    }
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

TEST(TranslateProgram, VertexStaticNestedImageAndSamplerRemainSupported) {
    VertexImageTranslationFixture fixture;
    const auto program = fixture.TranslateImages(false);

    ASSERT_FALSE(fixture.info.translation_failed) << DescribeProgram(program);
    ASSERT_FALSE(fixture.failed);
    ASSERT_EQ(fixture.info.images.size(), 1U);
    EXPECT_EQ(fixture.info.images.front().GetSharp(fixture.info).Address(), fixture.StaticImage);
    EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_count, 0U);
    ASSERT_EQ(fixture.info.samplers.size(), 1U);
    ExpectValidSpirv(fixture.Emit(program));
}

TEST(TranslateProgram, VertexLoopSelectsTwoDistinctReservedImageTables) {
    VertexImageTranslationFixture fixture;
    const auto program = fixture.TranslateImages(true);

    // On rejection, expose the real production IR matcher result independently of stage
    // admission. A fixture that fails because its assembly did not match is not this RED.
    if (fixture.info.translation_failed) {
        std::vector<u32> matched_offsets;
        for (auto* block : program.blocks) {
            for (auto& inst : *block) {
                if (const auto match = Optimization::MatchDynamicImageTable(inst)) {
                    EXPECT_EQ(match->byte_stride, sizeof(AmdGpu::Image));
                    matched_offsets.push_back(match->byte_offset);
                }
            }
        }
        std::ranges::sort(matched_offsets);
        EXPECT_EQ(matched_offsets,
                  (std::vector<u32>{VertexImageTableOffsets[0], VertexImageTableOffsets[1]}))
            << DescribeProgram(program);
    }
    ASSERT_FALSE(fixture.info.translation_failed) << DescribeProgram(program);
    ASSERT_FALSE(fixture.failed);
    ASSERT_EQ(fixture.info.images.size(), 3U);
    EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_count, 0U);
    for (size_t table = 0; table < VertexImageTableOffsets.size(); ++table) {
        const auto& resource = fixture.info.images[table + 1U];
        ASSERT_EQ(resource.dynamic_descriptor_count, NUM_IMAGES);
        EXPECT_TRUE(resource.requires_dynamic_oob_guard);
        EXPECT_TRUE(resource.HasValidDynamicRange(fixture.info));
        EXPECT_EQ(resource.ReadSharpUnchecked(fixture.info, 0U).Address(),
                  VertexImageTranslationFixture::TableImageAddresses[table]);
        EXPECT_EQ(resource.ReadSharpUnchecked(fixture.info, 2U).Address(),
                  VertexImageTranslationFixture::TableImageAddresses[table] + 0x2000U);
        EXPECT_TRUE(resource.IsTableEntryCompatible(fixture.info, 2U));
        EXPECT_FALSE(resource.IsTableEntryCompatible(fixture.info, NUM_IMAGES));
    }
    const auto spirv = fixture.Emit(program);
    EXPECT_GE(CountSpirv(spirv, spv::Op::OpLoopMerge), 1U);
    EXPECT_EQ(CountSpirv(spirv, spv::Op::OpImageSampleExplicitLod), 3U);
    ExpectValidSpirv(spirv);
}

TEST(TranslateProgram, VertexDynamicTablesRefuseUnreadableReservationsAndMissingFeature) {
    {
        VertexImageTranslationFixture fixture;
        fixture.second_table_unreadable = true;
        fixture.TranslateImages(true);
        EXPECT_TRUE(fixture.failed);
        EXPECT_TRUE(fixture.info.translation_failed);
    }
    {
        VertexImageTranslationFixture fixture;
        fixture.profile.supports_sampled_image_array_nonuniform_indexing = false;
        fixture.TranslateImages(true);
        EXPECT_TRUE(fixture.info.translation_failed);
        ASSERT_EQ(fixture.info.images.size(), 1U); // Only the separately valid static image.
        EXPECT_EQ(fixture.info.images.front().GetSharp(fixture.info).Address(),
                  fixture.StaticImage);
    }
}

TEST(TranslateProgram, VertexTableWindowIsNotProofThatEveryGuestEntryHasTheModelType) {
    VertexImageTranslationFixture fixture;
    auto different = AmdGpu::Image::Null(false);
    different.type = static_cast<u64>(AmdGpu::ImageType::Color3D);
    std::memcpy(fixture.image_bytes.data() + VertexImageTableOffsets[1] + 2U * sizeof(different),
                &different, sizeof(different));
    fixture.TranslateImages(true);
    ASSERT_FALSE(fixture.info.translation_failed);
    ASSERT_EQ(fixture.info.images.size(), 3U);
    EXPECT_TRUE(fixture.info.images[1].IsTableEntryCompatible(fixture.info, 2U));
    EXPECT_FALSE(fixture.info.images[2].IsTableEntryCompatible(fixture.info, 2U));
    // Existing binding supplies a null descriptor for this incompatible slot. Successful
    // translation is not a claim that the selected guest entry renders correctly.
}

TEST(TranslateProgram, BackendRejectsImageCoordinateArityMismatch) {
    VertexImageTranslationFixture fixture;
    auto image = AmdGpu::Image::Null(false);
    std::memcpy(&image, fixture.image_bytes.data(), sizeof(image));
    image.type = static_cast<u64>(AmdGpu::ImageType::Color1DArray);
    std::memcpy(fixture.image_bytes.data(), &image, sizeof(image));

    const auto program = fixture.TranslateImages(false);
    ASSERT_FALSE(fixture.info.translation_failed) << DescribeProgram(program);
    ASSERT_FALSE(fixture.failed);
    ASSERT_FALSE(fixture.info.images.empty());

    // The guest instruction was translated against a non-array 1D view, so it correctly carries
    // one coordinate component. Change only the authoritative descriptor metadata after the IR
    // pass to model the real failure mode: a backend image declaration that now needs two.
    auto& image_resource = fixture.info.images.front();
    ASSERT_EQ(image_resource.GetSharp(fixture.info).GetType(), AmdGpu::ImageType::Color1DArray);
    ASSERT_FALSE(image_resource.is_array);
    image_resource.is_array = true;

    const auto spirv = fixture.Emit(program);
    EXPECT_TRUE(fixture.info.translation_failed);
    EXPECT_EQ(CountSpirv(spirv, spv::Op::OpImageSampleImplicitLod), 0U);
    EXPECT_EQ(CountSpirv(spirv, spv::Op::OpImageSampleExplicitLod), 0U);
    ExpectValidSpirv(spirv);
}

TEST(TranslateProgram, BackendRejectsImageFetchCoordinateArityMismatch) {
    VertexImageTranslationFixture fixture;
    auto image = AmdGpu::Image::Null(false);
    std::memcpy(&image, fixture.image_bytes.data(), sizeof(image));
    image.type = static_cast<u64>(AmdGpu::ImageType::Color1DArray);
    std::memcpy(fixture.image_bytes.data(), &image, sizeof(image));

    auto program = fixture.TranslateImages(false);
    ASSERT_FALSE(fixture.info.translation_failed) << DescribeProgram(program);
    ASSERT_FALSE(fixture.failed);
    ASSERT_FALSE(fixture.info.images.empty());
    auto& image_resource = fixture.info.images.front();
    ASSERT_EQ(image_resource.GetSharp(fixture.info).GetType(), AmdGpu::ImageType::Color1DArray);
    image_resource.is_array = true;

    bool changed_sample = false;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::ImageSampleExplicitLod) {
                continue;
            }
            // ImageRead and ImageSampleExplicitLod have the same handle/coordinate/third/fourth
            // shape. Replacing this already translated sample isolates the OpImageFetch path.
            inst.ReplaceOpcode(IR::Opcode::ImageRead);
            inst.SetArg(3, IR::Value{0U});
            changed_sample = true;
            break;
        }
        if (changed_sample) {
            break;
        }
    }
    ASSERT_TRUE(changed_sample) << DescribeProgram(program);

    const auto spirv = fixture.Emit(program);
    EXPECT_TRUE(fixture.info.translation_failed);
    EXPECT_EQ(CountSpirv(spirv, spv::Op::OpImageFetch), 0U);
    ExpectValidSpirv(spirv);
}

struct IntegerFragmentOutputFixture {
    Info info{};
    Pools pools{};
    RuntimeInfo runtime{};
    Profile profile{};
    IR::Program program{info};
    IR::Block* block{};

    IntegerFragmentOutputFixture() {
        info.stage = Stage::Fragment;
        info.l_stage = LogicalStage::Fragment;
        runtime.Initialize(Stage::Fragment);
        runtime.fs_info.color_buffers[0].data_format = AmdGpu::DataFormat::Format32;
        runtime.fs_info.color_buffers[0].num_format = AmdGpu::NumberFormat::Uint;
        runtime.fs_info.color_buffers[0].num_conversion = AmdGpu::NumberConversion::None;
        runtime.fs_info.color_buffers[0].export_format = AmdGpu::ShaderExportFormat::R_32;
        runtime.fs_info.color_buffers[0].swizzle = AmdGpu::IdentityMapping;
        profile.supported_spirv = 0x00010600U;
        profile.subgroup_size = 32U;

        block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        program.syntax_list.emplace_back();
        program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
        program.syntax_list.back().data.block = block;
        program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
        program.post_order_blocks = IR::PostOrder(program.syntax_list.front());

        IR::IREmitter ir{*block};
        ir.Prologue();
        // GCN export values are raw VGPR words.  Keep the source as a fragment input so the
        // optimizer cannot fold the value away; the SPIR-V backend must preserve those bits when
        // storing them to the typed integer attachment.
        ir.SetAttribute(IR::Attribute::RenderTarget0,
                        ir.GetAttribute(IR::Attribute::FragCoord, 0), 0);
        ir.Epilogue();
    }

    std::vector<u32> Emit() {
        Optimization::SsaRewritePass(program);
        Optimization::ConstantPropagationPass(program.blocks);
        Optimization::DeadCodeEliminationPass(program);
        Optimization::IdentityRemovalPass(program.blocks);
        Optimization::CollectShaderInfoPass(program, profile);
        Backend::Bindings bindings{};
        return Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
    }
};

TEST(TranslateProgram, R32UintFragmentExportUsesUnsigned32OutputVariable) {
    IntegerFragmentOutputFixture fixture;
    const auto spirv = fixture.Emit();
    const auto instructions = ParseSpirvInstructions(spirv);
    const auto output = FindNamedSpirvId(spirv, instructions, "frag_color0");
    ASSERT_TRUE(output.has_value());

    std::optional<u32> output_type;
    for (const auto& instruction : instructions) {
        const auto at = instruction.offset;
        if (instruction.opcode == spv::Op::OpVariable && instruction.word_count >= 4U &&
            spirv[at + 2U] == *output) {
            output_type = spirv[at + 1U];
        }
    }
    ASSERT_TRUE(output_type.has_value());

    std::optional<u32> component_type;
    for (const auto& instruction : instructions) {
        const auto at = instruction.offset;
        if (instruction.opcode == spv::Op::OpTypePointer && instruction.word_count >= 4U &&
            spirv[at + 1U] == *output_type) {
            component_type = spirv[at + 3U];
        }
    }
    ASSERT_TRUE(component_type.has_value());

    std::optional<u32> scalar_type;
    bool found_vector = false;
    for (const auto& instruction : instructions) {
        const auto at = instruction.offset;
        if (instruction.opcode == spv::Op::OpTypeVector && instruction.word_count >= 4U &&
            spirv[at + 1U] == *component_type) {
            ASSERT_EQ(spirv[at + 3U], 4U);
            scalar_type = spirv[at + 2U];
            found_vector = true;
        }
    }
    ASSERT_TRUE(found_vector);
    ASSERT_TRUE(scalar_type.has_value());

    bool found_unsigned_32 = false;
    for (const auto& instruction : instructions) {
        const auto at = instruction.offset;
        if (instruction.opcode == spv::Op::OpTypeInt && instruction.word_count >= 4U &&
            spirv[at + 1U] == *scalar_type) {
            ASSERT_EQ(spirv[at + 2U], 32U);
            ASSERT_EQ(spirv[at + 3U], 0U);
            found_unsigned_32 = true;
        }
    }
    EXPECT_TRUE(found_unsigned_32);
}

TEST(TranslateProgram, R32UintFragmentExportBitcastsRawFloatBitsBeforeStore) {
    IntegerFragmentOutputFixture fixture;
    const auto spirv = fixture.Emit();
    const auto instructions = ParseSpirvInstructions(spirv);
    const auto output = FindNamedSpirvId(spirv, instructions, "frag_color0");
    ASSERT_TRUE(output.has_value());

    std::optional<u32> output_type;
    for (const auto& instruction : instructions) {
        const auto at = instruction.offset;
        if (instruction.opcode == spv::Op::OpVariable && instruction.word_count >= 4U &&
            spirv[at + 2U] == *output) {
            output_type = spirv[at + 1U];
        }
    }
    ASSERT_TRUE(output_type.has_value());

    std::optional<u32> component_type;
    for (const auto& instruction : instructions) {
        const auto at = instruction.offset;
        if (instruction.opcode == spv::Op::OpTypePointer && instruction.word_count >= 4U &&
            spirv[at + 1U] == *output_type) {
            component_type = spirv[at + 3U];
        }
    }
    ASSERT_TRUE(component_type.has_value());

    std::vector<u32> output_component_pointers;
    for (const auto& instruction : instructions) {
        const auto at = instruction.offset;
        if (instruction.opcode == spv::Op::OpAccessChain && instruction.word_count >= 4U &&
            spirv[at + 3U] == *output) {
            output_component_pointers.push_back(spirv[at + 2U]);
        }
    }

    std::optional<u32> stored_value;
    for (const auto& instruction : instructions) {
        const auto at = instruction.offset;
        if (instruction.opcode == spv::Op::OpStore && instruction.word_count >= 3U &&
            std::ranges::find(output_component_pointers, spirv[at + 1U]) !=
                output_component_pointers.end()) {
            stored_value = spirv[at + 2U];
            break;
        }
    }
    ASSERT_TRUE(stored_value.has_value());

    bool stored_raw_bits = false;
    for (const auto& instruction : instructions) {
        const auto at = instruction.offset;
        if (instruction.opcode == spv::Op::OpBitcast && instruction.word_count >= 4U &&
            spirv[at + 2U] == *stored_value) {
            stored_raw_bits = true;
            break;
        }
    }
    EXPECT_TRUE(stored_raw_bits);
}

} // namespace
} // namespace Shader
