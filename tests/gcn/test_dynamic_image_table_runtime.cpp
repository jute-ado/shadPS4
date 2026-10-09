// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <process.h>
#endif

#include <gtest/gtest.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/dynamic_image_table_matcher.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv/unified1/spirv.hpp"

namespace Shader::Optimization {
namespace {

constexpr u32 TableDestination = 20U;
constexpr u32 TablePointer = 18U;
constexpr u32 TableOffsetDw = 14480U / sizeof(u32);
constexpr u32 TableStrideDw = 32U / sizeof(u32);
constexpr u32 DescriptorDwords = sizeof(AmdGpu::Image) / sizeof(u32);
constexpr u32 FullTableSpan = (NUM_IMAGES - 1U) * TableStrideDw + DescriptorDwords;

struct DynamicImageRuntimeFixture {
    Info info{};
    Pools pools{};
    IR::Program program{info};
    Gcn::Block cfg_block{};
    IR::Block* block{};
    IR::IREmitter emitter;

    DynamicImageRuntimeFixture()
        : block{pools.block_pool.Create(pools.inst_pool)}, emitter{*block} {
        info.stage = Stage::Compute;
        info.l_stage = LogicalStage::Compute;
        block->cfg_block = &cfg_block;
        program.blocks.push_back(block);
        program.post_order_blocks.push_back(block);
        emitter.Prologue();
    }

    IR::Value MakeTablePointer() {
        const auto root = emitter.CompositeConstruct(emitter.GetUserData(IR::ScalarReg::S0),
                                                     emitter.GetUserData(IR::ScalarReg::S1));
        return emitter.CompositeConstruct(emitter.ReadConst(root, emitter.Imm32(2U)),
                                          emitter.ReadConst(root, emitter.Imm32(3U)));
    }

    IR::Inst* MakeAccess(IR::U32 descriptor_index, bool pointer_form = false) {
        const auto scaled_index = emitter.IMul(descriptor_index, emitter.Imm32(32U));
        const IR::U32 byte_address{emitter.IAdd(scaled_index, emitter.Imm32(14480U))};
        const auto dword_offset = emitter.ShiftRightLogical(byte_address, emitter.Imm32(2U));
        const auto table_pointer = MakeTablePointer();
        IR::U32 image = emitter.ReadConst(table_pointer, dword_offset);
        if (pointer_form) {
            const IR::U32 base_low{table_pointer.InstRecursive()->Arg(0)};
            const IR::U32 base_high{table_pointer.InstRecursive()->Arg(1)};
            const IR::U32 low{emitter.IAdd(byte_address, base_low)};
            const IR::U32 carry{emitter.Select(emitter.ILessThan(low, byte_address, false),
                                               emitter.Imm32(1U), emitter.Imm32(0U))};
            const auto high = emitter.IAdd(base_high, carry);
            image = emitter.ReadConst(emitter.CompositeConstruct(low, high), emitter.Imm32(0U));
        }
        const auto sampler = emitter.CompositeConstruct(emitter.Imm32(0U), emitter.Imm32(0U),
                                                        emitter.Imm32(0U), emitter.Imm32(0U));
        const auto coords_lod = emitter.CompositeConstruct(
            emitter.Imm32(0.25f), emitter.Imm32(0.5f), emitter.Imm32(0.0f), emitter.Imm32(0.0f));
        const auto unused = emitter.CompositeConstruct(emitter.Imm32(0.0f), emitter.Imm32(0.0f),
                                                       emitter.Imm32(0.0f), emitter.Imm32(0.0f));
        IR::TextureInstInfo texture_info{};
        texture_info.has_lod.Assign(true);
        texture_info.is_r128.Assign(false);
        const auto access = emitter.ImageSampleRaw(image, sampler, coords_lod, unused, unused,
                                                   emitter.Imm32(0.0f), texture_info);
        emitter.Reference(access);
        return access.InstRecursive();
    }
};

Profile DynamicImageProfile() {
    Profile profile{};
    profile.max_sampled_images_per_stage = 256U;
    profile.max_sampled_images_per_set = 256U;
    profile.max_resources_per_stage = 256U;
    profile.max_sampled_images_per_descriptor_heap_set = 256U;
    profile.supports_sampled_image_array_nonuniform_indexing = true;
    return profile;
}

AmdGpu::Image MakeImage(VAddr address) {
    auto image = AmdGpu::Image::Null(false);
    image.base_address = address >> 8U;
    image.width = 63U;
    image.height = 63U;
    return image;
}

struct SpirvInstruction {
    spv::Op opcode{};
    size_t offset{};
    u32 word_count{};
    u32 label{};
};

std::vector<SpirvInstruction> ParseSpirv(const std::vector<u32>& spirv) {
    std::vector<SpirvInstruction> instructions;
    u32 current_label{};
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word_count = spirv[offset] >> 16U;
        if (word_count == 0U || word_count > spirv.size() - offset) {
            ADD_FAILURE() << "Malformed SPIR-V instruction at word " << offset;
            break;
        }
        const auto opcode = static_cast<spv::Op>(spirv[offset] & 0xFFFFU);
        if (opcode == spv::Op::OpLabel && word_count >= 2U) {
            current_label = spirv[offset + 1U];
        }
        instructions.push_back({opcode, offset, word_count, current_label});
        offset += word_count;
    }
    return instructions;
}

std::unordered_map<u32, u32> ScalarConstantLiterals(
    const std::vector<u32>& spirv, const std::vector<SpirvInstruction>& instructions) {
    std::unordered_map<u32, u32> constants;
    for (const auto& inst : instructions) {
        if (inst.opcode == spv::Op::OpConstant && inst.word_count == 4U) {
            constants.emplace(spirv[inst.offset + 2U], spirv[inst.offset + 3U]);
        }
    }
    return constants;
}

std::unordered_set<u32> DecoratedIds(const std::vector<u32>& spirv,
                                     const std::vector<SpirvInstruction>& instructions,
                                     spv::Decoration decoration) {
    std::unordered_set<u32> decorated;
    for (const auto& inst : instructions) {
        if (inst.opcode == spv::Op::OpDecorate && inst.word_count >= 3U &&
            spirv[inst.offset + 2U] == static_cast<u32>(decoration)) {
            decorated.insert(spirv[inst.offset + 1U]);
        }
    }
    return decorated;
}

bool HasCapability(const std::vector<u32>& spirv, const std::vector<SpirvInstruction>& instructions,
                   spv::Capability capability) {
    return std::ranges::any_of(instructions, [&](const auto& inst) {
        return inst.opcode == spv::Op::OpCapability && inst.word_count == 2U &&
               spirv[inst.offset + 1U] == static_cast<u32>(capability);
    });
}

std::vector<u32> EmitTrackedProgram(DynamicImageRuntimeFixture& fixture, Profile profile) {
    fixture.emitter.Epilogue();
    fixture.program.syntax_list.emplace_back();
    fixture.program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    fixture.program.syntax_list.back().data.block = fixture.block;
    fixture.program.syntax_list.emplace_back();
    fixture.program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    fixture.program.post_order_blocks = IR::PostOrder(fixture.program.syntax_list.front());

    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;
    Optimization::IdentityRemovalPass(fixture.program.blocks);
    Optimization::DeadCodeEliminationPass(fixture.program);
    Optimization::CollectShaderInfoPass(fixture.program, profile);

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {32U, 1U, 1U};
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, fixture.program, bindings);
}

struct DynamicSpirvFlow {
    u32 selector{};
    u32 nonuniform_index{};
    u32 in_bounds_label{};
    u32 out_of_bounds_label{};
    u32 image_pointer{};
    u32 image{};
    u32 oob_phi_value{};
};

std::optional<DynamicSpirvFlow> FindDynamicSpirvFlow(
    const std::vector<u32>& spirv, const std::vector<SpirvInstruction>& instructions,
    u32 descriptor_count) {
    const auto constants = ScalarConstantLiterals(spirv, instructions);
    const auto nonuniform = DecoratedIds(spirv, instructions, spv::DecorationNonUniform);

    u32 guard{};
    DynamicSpirvFlow flow{};
    for (const auto& inst : instructions) {
        if (inst.opcode != spv::Op::OpULessThan || inst.word_count != 5U) {
            continue;
        }
        const auto count = constants.find(spirv[inst.offset + 4U]);
        if (count != constants.end() && count->second == descriptor_count) {
            guard = spirv[inst.offset + 2U];
            flow.selector = spirv[inst.offset + 3U];
            break;
        }
    }
    if (guard == 0U) {
        return std::nullopt;
    }
    for (const auto& inst : instructions) {
        if (inst.opcode == spv::Op::OpBranchConditional && inst.word_count >= 4U &&
            spirv[inst.offset + 1U] == guard) {
            flow.in_bounds_label = spirv[inst.offset + 2U];
            flow.out_of_bounds_label = spirv[inst.offset + 3U];
            break;
        }
    }
    if (flow.in_bounds_label == 0U || flow.out_of_bounds_label == 0U) {
        return std::nullopt;
    }

    for (const auto& inst : instructions) {
        if (inst.label != flow.in_bounds_label || inst.opcode != spv::Op::OpAccessChain ||
            inst.word_count < 5U) {
            continue;
        }
        const u32 pointer = spirv[inst.offset + 2U];
        const u32 index = spirv[inst.offset + inst.word_count - 1U];
        if (nonuniform.contains(pointer) && nonuniform.contains(index)) {
            flow.image_pointer = pointer;
            flow.nonuniform_index = index;
            break;
        }
    }
    if (flow.image_pointer == 0U) {
        return std::nullopt;
    }
    for (const auto& inst : instructions) {
        if (inst.label == flow.in_bounds_label && inst.opcode == spv::Op::OpLoad &&
            inst.word_count == 4U && spirv[inst.offset + 3U] == flow.image_pointer &&
            nonuniform.contains(spirv[inst.offset + 2U])) {
            flow.image = spirv[inst.offset + 2U];
            break;
        }
    }
    if (flow.image == 0U) {
        return std::nullopt;
    }

    for (const auto& inst : instructions) {
        if (inst.opcode != spv::Op::OpPhi || inst.word_count < 7U) {
            continue;
        }
        for (u32 operand = 3U; operand + 1U < inst.word_count; operand += 2U) {
            if (spirv[inst.offset + operand + 1U] == flow.out_of_bounds_label) {
                flow.oob_phi_value = spirv[inst.offset + operand];
                break;
            }
        }
        if (flow.oob_phi_value != 0U) {
            break;
        }
    }
    return flow.oob_phi_value != 0U ? std::optional{flow} : std::nullopt;
}

void ExpectOobValueIsZero(const std::vector<u32>& spirv,
                          const std::vector<SpirvInstruction>& instructions, u32 value) {
    const auto constants = ScalarConstantLiterals(spirv, instructions);
    for (const auto& inst : instructions) {
        if (inst.opcode == spv::Op::OpConstantComposite && inst.word_count >= 4U &&
            spirv[inst.offset + 2U] == value) {
            for (u32 operand = 3U; operand < inst.word_count; ++operand) {
                const auto scalar = constants.find(spirv[inst.offset + operand]);
                ASSERT_NE(scalar, constants.end());
                EXPECT_EQ(scalar->second, 0U);
            }
            return;
        }
    }
    ADD_FAILURE() << "OOB phi input is not a constant zero vector";
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

void ExpectValidSpirv(const std::vector<u32>& spirv, std::string_view name) {
    const auto output = std::filesystem::temp_directory_path() / name;
    {
        std::ofstream file{output, std::ios::binary | std::ios::trunc};
        ASSERT_TRUE(file.good());
        file.write(reinterpret_cast<const char*>(spirv.data()),
                   static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
    }
    EXPECT_EQ(RunSpirvValidator(output), 0);
    std::error_code remove_error;
    std::filesystem::remove(output, remove_error);
}
#endif

DynamicImageTableMatch PrepareReservation(DynamicImageRuntimeFixture& fixture, IR::Inst& access,
                                          u32 span_dw = FullTableSpan) {
    const auto match = MatchDynamicImageTable(access);
    EXPECT_TRUE(match.has_value());
    if (!match) {
        return {};
    }
    match->pointer_lo->SetFlags<u32>(TablePointer);
    match->pointer_hi->SetFlags<u32>(TablePointer + 1U);
    fixture.info.srt_info.flattened_bufsize_dw = TableDestination + span_dw;
    fixture.info.srt_info.range_reservations.push_back({
        .pointer_lo_flat_index = TablePointer,
        .source_offset_dw = TableOffsetDw,
        .destination_flat_offset_dw = TableDestination,
        .span_dw = span_dw,
    });
    fixture.info.flattened_ud_buf.resize(TableDestination + span_dw);
    const auto first = MakeImage(0x10000U);
    std::memcpy(fixture.info.flattened_ud_buf.data() + TableDestination, &first, sizeof(first));
    if (span_dw >= TableStrideDw + DescriptorDwords) {
        const auto second = MakeImage(0x20000U);
        std::memcpy(fixture.info.flattened_ud_buf.data() + TableDestination + TableStrideDw,
                    &second, sizeof(second));
    }
    return *match;
}

void ExpectUnmaterializedHandleRefused(const DynamicImageRuntimeFixture& fixture) {
    EXPECT_TRUE(fixture.info.translation_failed);
    EXPECT_TRUE(fixture.info.images.empty());
    EXPECT_TRUE(fixture.info.samplers.empty());
    EXPECT_TRUE(std::ranges::none_of(fixture.block->Instructions(), [](const auto& inst) {
        return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic ||
               inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLod;
    }));
}

TEST(DynamicImageTableRuntime, ActivatesOnlyTheCoherentlyFilledExactReservation) {
    DynamicImageRuntimeFixture fixture;
    IR::Inst* const access = fixture.MakeAccess(fixture.emitter.LaneId());
    const auto match = PrepareReservation(fixture, *access);
    ASSERT_EQ(match.image_handle->Flags<u32>(), 0U);
    ASSERT_TRUE(fixture.info.srt_info.ValidateRangeReservations());

    ResourceTrackingPass(fixture.program, DynamicImageProfile(), true);

    ASSERT_EQ(fixture.info.images.size(), 1U);
    const auto& image = fixture.info.images.front();
    EXPECT_EQ(image.sharp_idx, TableDestination);
    EXPECT_EQ(image.dynamic_descriptor_count, NUM_IMAGES);
    EXPECT_EQ(image.dynamic_descriptor_stride_dw, TableStrideDw);
    EXPECT_TRUE(image.requires_dynamic_oob_guard);
    EXPECT_TRUE(image.HasValidDynamicRange(fixture.info));
    EXPECT_EQ(image.ReadSharpUnchecked(fixture.info, 1U).Address(), 0x20000U);
    EXPECT_TRUE(image.IsTableEntryCompatible(fixture.info, 1U));
    EXPECT_EQ(match.image_handle->Flags<u32>(), TableDestination);

    const auto dynamic_sample = std::ranges::find_if(fixture.block->Instructions(), [](auto& inst) {
        return inst.GetOpcode() == IR::Opcode::ImageSampleExplicitLodDynamic;
    });
    ASSERT_NE(dynamic_sample, fixture.block->Instructions().end());
    EXPECT_EQ(dynamic_sample->Arg(1).InstRecursive()->GetOpcode(), IR::Opcode::LaneId);
}

TEST(DynamicImageTableRuntime, PointerFormUsesTheSameGuardedCubemapArrayBackend) {
    DynamicImageRuntimeFixture fixture;
    auto* const access = fixture.MakeAccess(fixture.emitter.LaneId(), true);
    ASSERT_TRUE(MatchDynamicImageTable(*access));
    PrepareReservation(fixture, *access);
    for (u32 i = 0; i < 2U; ++i) {
        auto cube = MakeImage(0x10000U + i * 0x10000U);
        cube.type = static_cast<u64>(AmdGpu::ImageType::Cube);
        std::memcpy(fixture.info.flattened_ud_buf.data() + TableDestination + i * TableStrideDw,
                    &cube, sizeof(cube));
    }

    ResourceTrackingPass(fixture.program, DynamicImageProfile(), true);

    ASSERT_FALSE(fixture.info.translation_failed);
    ASSERT_EQ(fixture.info.images.size(), 1U);
    EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_count, NUM_IMAGES);
    EXPECT_TRUE(fixture.info.images.front().IsTableEntryCompatible(fixture.info, 1U));
    EXPECT_TRUE(fixture.info.images.front().ReadSharpUnchecked(fixture.info).IsCube());
    const auto spirv = EmitTrackedProgram(fixture, DynamicImageProfile());
    const auto instructions = ParseSpirv(spirv);
    const auto flow = FindDynamicSpirvFlow(spirv, instructions, NUM_IMAGES);
    ASSERT_TRUE(flow);
    ExpectOobValueIsZero(spirv, instructions, flow->oob_phi_value);
#ifdef SHADPS4_SPIRV_VAL_PATH
    ExpectValidSpirv(spirv, "shadps4-dynamic-image-pointer-cube.spv");
#endif
}

TEST(DynamicImageTableRuntime, PointerFormStillRequiresProofFeatureBudgetAndSupportedModel) {
    enum class Missing { Proof, Reservation, Feature, Budget, Model };
    for (const auto missing : {Missing::Proof, Missing::Reservation, Missing::Feature,
                               Missing::Budget, Missing::Model}) {
        DynamicImageRuntimeFixture fixture;
        auto* const access = fixture.MakeAccess(fixture.emitter.LaneId(), true);
        ASSERT_TRUE(MatchDynamicImageTable(*access));
        PrepareReservation(fixture, *access);
        auto profile = DynamicImageProfile();
        if (missing == Missing::Reservation) {
            fixture.info.srt_info.range_reservations.clear();
        } else if (missing == Missing::Feature) {
            profile.supports_sampled_image_array_nonuniform_indexing = false;
        } else if (missing == Missing::Budget) {
            profile.max_resources_per_stage = 0U;
        } else if (missing == Missing::Model) {
            auto unsupported = MakeImage(0x10000U);
            unsupported.type = static_cast<u64>(AmdGpu::ImageType::Color2DMsaa);
            std::memcpy(fixture.info.flattened_ud_buf.data() + TableDestination, &unsupported,
                        sizeof(unsupported));
        }
        ResourceTrackingPass(fixture.program, profile, missing != Missing::Proof);
        ExpectUnmaterializedHandleRefused(fixture);
    }
}

TEST(DynamicImageTableRuntime, MissingCoherentProofCannotActivateAReservedRange) {
    DynamicImageRuntimeFixture fixture;
    IR::Inst* const access = fixture.MakeAccess(fixture.emitter.LaneId());
    const auto match = PrepareReservation(fixture, *access);

    ResourceTrackingPass(fixture.program, DynamicImageProfile(), false);

    ExpectUnmaterializedHandleRefused(fixture);
    EXPECT_EQ(match.image_handle->Flags<u32>(), 0U);
}

TEST(DynamicImageTableRuntime, MissingProofNeverBindsUserDataZeroBesideOrdinaryResources) {
    DynamicImageRuntimeFixture fixture;
    IR::Inst* const dynamic_access = fixture.MakeAccess(fixture.emitter.LaneId());
    const auto match = PrepareReservation(fixture, *dynamic_access);
    IR::Inst* const static_access = fixture.MakeAccess(fixture.emitter.Imm32(0U));
    IR::Inst* const static_handle = static_access->Arg(0).InstRecursive();
    constexpr u32 StaticDescriptor = 8U;
    static_handle->SetArg(1, fixture.emitter.Imm32(StaticDescriptor));
    static_handle->SetFlags<u32>(StaticDescriptor);
    const auto static_image = MakeImage(0x30000U);
    std::memcpy(fixture.info.flattened_ud_buf.data() + StaticDescriptor, &static_image,
                sizeof(static_image));

    ResourceTrackingPass(fixture.program, DynamicImageProfile(), false);

    ExpectUnmaterializedHandleRefused(fixture);
    EXPECT_EQ(match.image_handle->Flags<u32>(), 0U);
}

TEST(DynamicImageTableRuntime, PartialReservationCannotActivateAFullTablePlan) {
    DynamicImageRuntimeFixture fixture;
    IR::Inst* const access = fixture.MakeAccess(fixture.emitter.LaneId());
    const auto match = PrepareReservation(fixture, *access, DescriptorDwords);

    ResourceTrackingPass(fixture.program, DynamicImageProfile(), true);

    ExpectUnmaterializedHandleRefused(fixture);
    EXPECT_EQ(match.image_handle->Flags<u32>(), 0U);
}

TEST(DynamicImageTableRuntime, RealProfileCapacityBoundsTheActivatedWindow) {
    DynamicImageRuntimeFixture fixture;
    IR::Inst* const access = fixture.MakeAccess(fixture.emitter.LaneId());
    PrepareReservation(fixture, *access);
    auto profile = DynamicImageProfile();
    profile.max_sampled_images_per_stage = 7U;
    profile.max_sampled_images_per_set = 5U;
    profile.max_resources_per_stage = 4U;
    profile.max_sampled_images_per_descriptor_heap_set = 3U;

    ResourceTrackingPass(fixture.program, profile, true);

    ASSERT_EQ(fixture.info.images.size(), 1U);
    // The inline sampler consumes one per-stage resource, leaving the aggregate resource limit as
    // the tightest real device capacity.
    EXPECT_EQ(fixture.info.images.front().dynamic_descriptor_count, 3U);
    EXPECT_EQ(fixture.info.images.front().NumBindings(fixture.info), 3U);
}

TEST(DynamicImageTableRuntime, FeatureOffCannotActivateTheReservedTable) {
    DynamicImageRuntimeFixture fixture;
    IR::Inst* const access = fixture.MakeAccess(fixture.emitter.LaneId());
    const auto match = PrepareReservation(fixture, *access);
    auto profile = DynamicImageProfile();
    profile.supports_sampled_image_array_nonuniform_indexing = false;

    ResourceTrackingPass(fixture.program, profile, true);

    ExpectUnmaterializedHandleRefused(fixture);
    EXPECT_EQ(match.image_handle->Flags<u32>(), 0U);
}

TEST(DynamicImageTableRuntime, ZeroResourceBudgetRefusesAnUnmaterializedDescriptor) {
    DynamicImageRuntimeFixture fixture;
    IR::Inst* const access = fixture.MakeAccess(fixture.emitter.LaneId());
    PrepareReservation(fixture, *access);
    auto profile = DynamicImageProfile();
    profile.max_resources_per_stage = 0U;

    ResourceTrackingPass(fixture.program, profile, true);

    ExpectUnmaterializedHandleRefused(fixture);
}

TEST(DynamicImageTableRuntime, RuntimeBufferHandleCannotAliasUserDataZero) {
    DynamicImageRuntimeFixture fixture;
    auto& ir = fixture.emitter;
    const auto pointer =
        ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0), ir.GetUserData(IR::ScalarReg::S1));
    const auto runtime_read = ir.ReadConst(pointer, ir.LaneId());
    const auto handle =
        ir.CompositeConstruct(runtime_read, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U));
    const auto address = ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U));
    ir.Reference(ir.LoadBufferU32(1, handle, address, {}));
    fixture.info.flattened_ud_buf.resize(16U);

    ResourceTrackingPass(fixture.program, DynamicImageProfile(), false);

    EXPECT_TRUE(fixture.info.translation_failed);
    EXPECT_TRUE(fixture.info.buffers.empty());
}

TEST(DynamicImageTableRuntime, RuntimeSamplerHandleCannotAliasUserDataZero) {
    DynamicImageRuntimeFixture fixture;
    auto& ir = fixture.emitter;
    IR::Inst* const access = fixture.MakeAccess(ir.LaneId());
    const auto match = PrepareReservation(fixture, *access);
    // A valid static image must not make an unmaterialized sampler look valid too.
    match.image_handle->SetFlags<u32>(TableDestination);
    const auto pointer =
        ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg::S0), ir.GetUserData(IR::ScalarReg::S1));
    const auto runtime_read = ir.ReadConst(pointer, ir.LaneId());
    access->SetArg(1,
                   ir.CompositeConstruct(runtime_read, ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U)));

    ResourceTrackingPass(fixture.program, DynamicImageProfile(), false);

    EXPECT_TRUE(fixture.info.translation_failed);
    EXPECT_TRUE(fixture.info.samplers.empty());
}

TEST(DynamicImageTableRuntime, UserDataZeroRemainsAValidStaticDescriptorLocation) {
    DynamicImageRuntimeFixture fixture;
    IR::Inst* const access = fixture.MakeAccess(fixture.emitter.LaneId());
    access->SetArg(0, fixture.emitter.GetUserData(IR::ScalarReg::S0));
    fixture.info.flattened_ud_buf.resize(16U);
    const auto image = MakeImage(0x30000U);
    std::memcpy(fixture.info.flattened_ud_buf.data(), &image, sizeof(image));

    ResourceTrackingPass(fixture.program, DynamicImageProfile(), false);

    EXPECT_FALSE(fixture.info.translation_failed);
    ASSERT_EQ(fixture.info.images.size(), 1U);
    EXPECT_EQ(fixture.info.images.front().sharp_idx, 0U);
}

TEST(DynamicImageTableRuntime, NonComputeStageCannotActivateTheReservedTable) {
    DynamicImageRuntimeFixture fixture;
    IR::Inst* const access = fixture.MakeAccess(fixture.emitter.LaneId());
    const auto match = PrepareReservation(fixture, *access);
    fixture.info.stage = Stage::Fragment;
    fixture.info.l_stage = LogicalStage::Fragment;

    ResourceTrackingPass(fixture.program, DynamicImageProfile(), true);

    ExpectUnmaterializedHandleRefused(fixture);
    EXPECT_EQ(match.image_handle->Flags<u32>(), 0U);
}

TEST(DynamicImageTableRuntime, SnapshotStorageMustFitTheAdmittedReservation) {
    DynamicImageRuntimeFixture fixture;
    IR::Inst* const access = fixture.MakeAccess(fixture.emitter.LaneId());
    const auto match = PrepareReservation(fixture, *access);
    fixture.info.flattened_ud_buf.pop_back();

    ResourceTrackingPass(fixture.program, DynamicImageProfile(), true);

    ExpectUnmaterializedHandleRefused(fixture);
    EXPECT_EQ(match.image_handle->Flags<u32>(), 0U);
}

TEST(DynamicImageTableRuntime, LaterEntryUsesItsOwnFullyDecoratedNonUniformIdentity) {
    DynamicImageRuntimeFixture fixture;
    const auto selector = fixture.emitter.IAdd(fixture.emitter.LaneId(), fixture.emitter.Imm32(1U));
    IR::Inst* const access = fixture.MakeAccess(IR::U32{selector});
    PrepareReservation(fixture, *access);

    ResourceTrackingPass(fixture.program, DynamicImageProfile(), true);

    ASSERT_EQ(fixture.info.images.size(), 1U);
    const auto& image = fixture.info.images.front();
    ASSERT_TRUE(image.IsTableEntryCompatible(fixture.info, 1U));
    EXPECT_EQ(image.ReadSharpUnchecked(fixture.info, 1U).Address(), 0x20000U);
    const auto spirv = EmitTrackedProgram(fixture, DynamicImageProfile());
    const auto instructions = ParseSpirv(spirv);
    const auto flow = FindDynamicSpirvFlow(spirv, instructions, NUM_IMAGES);
    ASSERT_TRUE(flow.has_value());
    EXPECT_NE(flow->nonuniform_index, flow->selector);
    EXPECT_TRUE(HasCapability(spirv, instructions, spv::CapabilityShaderNonUniform));
    EXPECT_TRUE(
        HasCapability(spirv, instructions, spv::CapabilitySampledImageArrayNonUniformIndexing));

    const auto constants = ScalarConstantLiterals(spirv, instructions);
    const bool copies_selector_with_zero = std::ranges::any_of(instructions, [&](const auto& inst) {
        if (inst.opcode != spv::Op::OpIAdd || inst.word_count != 5U ||
            spirv[inst.offset + 2U] != flow->nonuniform_index ||
            spirv[inst.offset + 3U] != flow->selector) {
            return false;
        }
        const auto zero = constants.find(spirv[inst.offset + 4U]);
        return zero != constants.end() && zero->second == 0U;
    });
    EXPECT_TRUE(copies_selector_with_zero);

    const bool selector_advances_to_later_entry =
        std::ranges::any_of(instructions, [&](const auto& inst) {
            if (inst.opcode != spv::Op::OpIAdd || inst.word_count != 5U ||
                spirv[inst.offset + 2U] != flow->selector) {
                return false;
            }
            const auto one = constants.find(spirv[inst.offset + 4U]);
            return one != constants.end() && one->second == 1U;
        });
    EXPECT_TRUE(selector_advances_to_later_entry);

#ifdef SHADPS4_SPIRV_VAL_PATH
    ExpectValidSpirv(spirv, "shadps4-dynamic-image-later-entry.spv");
#endif
}

TEST(DynamicImageTableRuntime, OnePastWindowReturnsZeroWithoutDescriptorAccess) {
    DynamicImageRuntimeFixture fixture;
    const auto selector =
        fixture.emitter.IAdd(fixture.emitter.LaneId(), fixture.emitter.Imm32(NUM_IMAGES));
    IR::Inst* const access = fixture.MakeAccess(IR::U32{selector});
    PrepareReservation(fixture, *access);

    ResourceTrackingPass(fixture.program, DynamicImageProfile(), true);

    const auto spirv = EmitTrackedProgram(fixture, DynamicImageProfile());
    const auto instructions = ParseSpirv(spirv);
    const auto flow = FindDynamicSpirvFlow(spirv, instructions, NUM_IMAGES);
    ASSERT_TRUE(flow.has_value());
    EXPECT_TRUE(std::ranges::none_of(instructions, [&](const auto& inst) {
        return inst.label == flow->out_of_bounds_label &&
               (inst.opcode == spv::Op::OpAccessChain || inst.opcode == spv::Op::OpLoad);
    }));
    EXPECT_TRUE(std::ranges::any_of(instructions, [&](const auto& inst) {
        return inst.label == flow->in_bounds_label && inst.opcode == spv::Op::OpAccessChain &&
               spirv[inst.offset + 2U] == flow->image_pointer;
    }));
    ExpectOobValueIsZero(spirv, instructions, flow->oob_phi_value);

    const auto constants = ScalarConstantLiterals(spirv, instructions);
    const bool selector_is_one_past = std::ranges::any_of(instructions, [&](const auto& inst) {
        if (inst.opcode != spv::Op::OpIAdd || inst.word_count != 5U ||
            spirv[inst.offset + 2U] != flow->selector) {
            return false;
        }
        const auto count = constants.find(spirv[inst.offset + 4U]);
        return count != constants.end() && count->second == NUM_IMAGES;
    });
    EXPECT_TRUE(selector_is_one_past);

#ifdef SHADPS4_SPIRV_VAL_PATH
    ExpectValidSpirv(spirv, "shadps4-dynamic-image-one-past.spv");
#endif
}

} // namespace
} // namespace Shader::Optimization
