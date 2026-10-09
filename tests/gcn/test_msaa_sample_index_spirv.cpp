// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv/unified1/spirv.hpp11"
#include "video_core/amdgpu/resource.h"

namespace {

using namespace Shader;

struct SpirvInstruction {
    spv::Op opcode{};
    size_t offset{};
    u32 word_count{};
};

std::vector<SpirvInstruction> ParseSpirv(const std::vector<u32>& spirv) {
    std::vector<SpirvInstruction> result;
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word_count = spirv[offset] >> 16U;
        if (word_count == 0U || word_count > spirv.size() - offset) {
            ADD_FAILURE() << "Malformed SPIR-V instruction at word " << offset;
            break;
        }
        result.push_back({static_cast<spv::Op>(spirv[offset] & 0xFFFFU), offset, word_count});
        offset += word_count;
    }
    return result;
}

bool HasSampleOperand(const std::vector<u32>& spirv, const SpirvInstruction& instruction) {
    // OpImageFetch/OpImageRead carry a result type and result id, while OpImageWrite has no
    // result operands.  The image-operands mask therefore starts at a different word offset.
    const u32 mask_offset = instruction.opcode == spv::Op::OpImageWrite ? 4U : 5U;
    if (instruction.word_count <= mask_offset) {
        return false;
    }
    const auto mask = static_cast<spv::ImageOperandsMask>(spirv[instruction.offset + mask_offset]);
    return (static_cast<u32>(mask) & static_cast<u32>(spv::ImageOperandsMask::Sample)) != 0U;
}

std::vector<u32> EmitMsaaImageRead(bool array_view = false, bool storage_view = false,
                                   bool collect_shader_info = false,
                                   bool* collected_has_storage_images = nullptr,
                                   bool* collected_uses_integer_coordinates = nullptr,
                                   bool* translation_failed = nullptr,
                                   bool storage_feature_supported = true) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;

    // ImageResource resolves this descriptor through the same flattened sharp path used by
    // production shader translation. Two samples make the modulo operation observable while the
    // IR intentionally requests sample index three.
    AmdGpu::Image image = AmdGpu::Image::Null(false);
    image.type = static_cast<u64>(array_view ? AmdGpu::ImageType::Color2DMsaaArray
                                             : AmdGpu::ImageType::Color2DMsaa);
    image.last_level = 1;
    image.width = 64;
    image.height = 64;
    info.flattened_ud_buf.resize(sizeof(AmdGpu::Image) / sizeof(u32));
    std::memcpy(info.flattened_ud_buf.data(), &image, sizeof(image));
    info.images.push_back({.sharp_idx = 0, .is_array = array_view, .is_written = storage_view});
    info.has_storage_images = storage_view;

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

    IR::IREmitter ir{*block};
    ir.Prologue();
    const auto coords = array_view ? ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U))
                                   : ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U));
    const auto texel = ir.ImageRead(ir.Imm32(0U), coords, IR::U32{}, ir.Imm32(3U), {});
    ir.Reference(texel);
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;
    profile.supports_storage_image_multisample = storage_view && storage_feature_supported;

    if (collect_shader_info) {
        Optimization::CollectShaderInfoPass(program, profile);
        if (collected_has_storage_images != nullptr) {
            *collected_has_storage_images = info.has_storage_images;
        }
        if (collected_uses_integer_coordinates != nullptr) {
            *collected_uses_integer_coordinates =
                info.images.empty() ? false : info.images.front().uses_integer_coordinates;
        }
    }

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {1U, 1U, 1U};

    Backend::Bindings bindings{};
    auto spirv = Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    if (translation_failed != nullptr) {
        *translation_failed = info.translation_failed;
    }
    return spirv;
}

std::vector<u32> EmitNonMsImageOperation(bool write) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;

    AmdGpu::Image image = AmdGpu::Image::Null(false);
    image.width = 64;
    image.height = 64;
    info.flattened_ud_buf.resize(sizeof(AmdGpu::Image) / sizeof(u32));
    std::memcpy(info.flattened_ud_buf.data(), &image, sizeof(image));
    info.images.push_back({.sharp_idx = 0, .is_written = write});

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

    IR::IREmitter ir{*block};
    ir.Prologue();
    const auto coords = ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U));
    if (write) {
        const auto color = ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f),
                                                 ir.Imm32(0.0f), ir.Imm32(0.0f));
        // A valid sample ID is intentionally supplied to a non-MS image.  It must not become a
        // SPIR-V Sample image operand merely because the direct IR caller populated the slot.
        ir.ImageWrite(ir.Imm32(0U), coords, IR::U32{}, ir.Imm32(3U), color, {});
    } else {
        const auto texel = ir.ImageRead(ir.Imm32(0U), coords, IR::U32{}, ir.Imm32(3U), {});
        ir.Reference(texel);
    }
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {1U, 1U, 1U};

    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> EmitNonMsImageRead() {
    return EmitNonMsImageOperation(false);
}

std::vector<u32> EmitNonMsImageWrite() {
    return EmitNonMsImageOperation(true);
}

std::vector<u32> EmitMsaaImageQueryDimensions(bool array_view) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;

    AmdGpu::Image image = AmdGpu::Image::Null(false);
    image.type = static_cast<u64>(array_view ? AmdGpu::ImageType::Color2DMsaaArray
                                             : AmdGpu::ImageType::Color2DMsaa);
    image.last_level = 0;
    image.width = 64;
    image.height = 64;
    image.depth = 3;
    info.flattened_ud_buf.resize(sizeof(AmdGpu::Image) / sizeof(u32));
    std::memcpy(info.flattened_ud_buf.data(), &image, sizeof(image));
    info.images.push_back({.sharp_idx = 0, .is_array = array_view});

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

    IR::IREmitter ir{*block};
    ir.Prologue();
    const auto dimensions = ir.ImageQueryDimension(ir.Imm32(0U), ir.Imm32(0U), ir.Imm1(false), {});
    ir.Reference(dimensions);
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {1U, 1U, 1U};

    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> EmitImageQueryLodProgram() {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;

    AmdGpu::Image image = AmdGpu::Image::Null(false);
    image.type = static_cast<u64>(AmdGpu::ImageType::Color2D);
    image.width = 64;
    image.height = 64;
    info.flattened_ud_buf.resize(sizeof(AmdGpu::Image) / sizeof(u32));
    std::memcpy(info.flattened_ud_buf.data(), &image, sizeof(image));
    info.images.push_back({.sharp_idx = 0});
    // ImageQueryLod combines the image with a sampler in the SPIR-V backend.  Keep the
    // fixture's descriptor table complete so this contract test reaches the emitter instead
    // of indexing an absent sampler and aborting in debug builds.
    SamplerResource sampler{};
    sampler.is_inline_sampler = true;
    info.samplers.push_back(sampler);

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

    IR::IREmitter ir{*block};
    ir.Prologue();
    const auto coords = ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f));
    // ImageQueryLod is an F32x4 IR value. The backend must adapt SPIR-V's native vec2 query
    // result to that contract so later component extraction cannot address a two-component ID.
    const auto lod = ir.ImageQueryLod(ir.Imm32(0U), coords, {});
    ir.Reference(IR::F32{ir.CompositeExtract(lod, 2U)});
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {1U, 1U, 1U};

    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

u32 FindVectorComponentCount(const std::vector<u32>& spirv, u32 vector_type_id) {
    const auto instructions = ParseSpirv(spirv);
    for (const auto& instruction : instructions) {
        if (instruction.opcode == spv::Op::OpTypeVector && instruction.word_count >= 4U &&
            spirv[instruction.offset + 1U] == vector_type_id) {
            return spirv[instruction.offset + 3U];
        }
    }
    return 0U;
}

std::vector<u32> EmitMsaaImageAtomic(bool array_view = false, bool collect_shader_info = true) {
    Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;

    AmdGpu::Image image = AmdGpu::Image::Null(false);
    image.type = static_cast<u64>(array_view ? AmdGpu::ImageType::Color2DMsaaArray
                                             : AmdGpu::ImageType::Color2DMsaa);
    image.last_level = 1;
    image.width = 64;
    image.height = 64;
    info.flattened_ud_buf.resize(sizeof(AmdGpu::Image) / sizeof(u32));
    std::memcpy(info.flattened_ud_buf.data(), &image, sizeof(image));

    // Image atomics are storage-image operations, but shader-info collection historically only
    // marked ImageWrite. Keep this fixture's flag false so the capability test exercises that
    // production path rather than masking it with a manually-populated summary bit.
    info.images.push_back({.sharp_idx = 0,
                           .is_atomic = true,
                           .is_array = array_view,
                           .is_written = true});

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

    IR::IREmitter ir{*block};
    ir.Prologue();
    const auto coords = array_view ? ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U),
                                                            ir.Imm32(0U))
                                   : ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U));
    const auto result = ir.ImageAtomicIAdd(ir.Imm32(0U), coords, ir.Imm32(1U), {});
    ir.Reference(result);
    ir.Epilogue();

    // Exercise the same post-translation summary pass used by production. The atomic opcode must
    // set has_storage_images before SPIR-V capability setup runs. The direct-backend test below
    // intentionally omits this pass to verify that EmitSPIRV remains self-consistent when a
    // caller provides an already-populated ImageResource list.
    Profile profile{};
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;
    profile.supports_storage_image_multisample = true;
    if (collect_shader_info) {
        Optimization::CollectShaderInfoPass(program, profile);
    }

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.cs_info.workgroup_size = {1U, 1U, 1U};

    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

bool HasCapability(const std::vector<u32>& spirv, spv::Capability capability) {
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 word_count = spirv[offset] >> 16U;
        if (word_count == 0U || word_count > spirv.size() - offset) {
            return false;
        }
        if (static_cast<spv::Op>(spirv[offset] & 0xFFFFU) == spv::Op::OpCapability &&
            word_count == 2U && spirv[offset + 1U] == static_cast<u32>(capability)) {
            return true;
        }
        offset += word_count;
    }
    return false;
}

TEST(ImageReadSpirv, MsaaSampleIndexIsWrappedByImageSampleCount) {
    const auto spirv = EmitMsaaImageRead();
    const auto instructions = ParseSpirv(spirv);

    u32 sample_count_id{};
    u32 requested_sample_id{};
    u32 wrapped_sample_id{};
    for (const auto& instruction : instructions) {
        if (instruction.opcode == spv::Op::OpImageQuerySamples && instruction.word_count >= 4U) {
            ASSERT_EQ(sample_count_id, 0U);
            sample_count_id = spirv[instruction.offset + 2U];
        }
        if (instruction.opcode == spv::Op::OpUMod && instruction.word_count == 5U) {
            ASSERT_EQ(wrapped_sample_id, 0U);
            wrapped_sample_id = spirv[instruction.offset + 2U];
            requested_sample_id = spirv[instruction.offset + 3U];
            ASSERT_EQ(spirv[instruction.offset + 4U], sample_count_id);
        }
    }

    ASSERT_NE(sample_count_id, 0U);
    ASSERT_NE(requested_sample_id, 0U);
    ASSERT_NE(wrapped_sample_id, 0U);

    const auto requested_sample = std::ranges::find_if(instructions, [&](const auto& instruction) {
        return instruction.opcode == spv::Op::OpConstant && instruction.word_count == 4U &&
               spirv[instruction.offset + 2U] == requested_sample_id;
    });
    ASSERT_NE(requested_sample, instructions.end());
    EXPECT_EQ(spirv[requested_sample->offset + 3U], 3U);

    const auto image_fetch = std::ranges::find_if(instructions, [&](const auto& instruction) {
        if (instruction.opcode != spv::Op::OpImageFetch || instruction.word_count < 7U) {
            return false;
        }
        const auto mask = static_cast<spv::ImageOperandsMask>(spirv[instruction.offset + 5U]);
        return (static_cast<u32>(mask) & static_cast<u32>(spv::ImageOperandsMask::Sample)) != 0U;
    });
    ASSERT_NE(image_fetch, instructions.end());

    const auto mask = static_cast<spv::ImageOperandsMask>(spirv[image_fetch->offset + 5U]);
    EXPECT_EQ(static_cast<u32>(mask) & static_cast<u32>(spv::ImageOperandsMask::Lod), 0U);
    EXPECT_EQ(spirv[image_fetch->offset + 6U], wrapped_sample_id);
}

TEST(ImageReadSpirv, NonMsImageReadOmitsProvidedSampleOperand) {
    const auto spirv = EmitNonMsImageRead();
    const auto instructions = ParseSpirv(spirv);
    const auto image_fetch = std::ranges::find_if(instructions, [](const auto& instruction) {
        return instruction.opcode == spv::Op::OpImageFetch;
    });
    ASSERT_NE(image_fetch, instructions.end());
    EXPECT_FALSE(HasSampleOperand(spirv, *image_fetch));
}

TEST(ImageWriteSpirv, NonMsImageWriteOmitsProvidedSampleOperand) {
    const auto spirv = EmitNonMsImageWrite();
    const auto instructions = ParseSpirv(spirv);
    const auto image_write = std::ranges::find_if(instructions, [](const auto& instruction) {
        return instruction.opcode == spv::Op::OpImageWrite;
    });
    ASSERT_NE(image_write, instructions.end());
    EXPECT_FALSE(HasSampleOperand(spirv, *image_write));
}

TEST(ImageReadSpirv, MsaaArrayUsesAnArrayedSpirvImageType) {
    // This exercises ImageResource::GetViewType(true) all the way through the SPIR-V image type
    // declaration. A Color2DMsaaArray must set the OpTypeImage Arrayed operand; it must not fall
    // through the scalar-MSAA declaration or hit an unsupported image-type branch.
    const auto spirv = EmitMsaaImageRead(true);
    const auto instructions = ParseSpirv(spirv);

    bool found_arrayed_msaa_image = false;
    for (const auto& instruction : instructions) {
        if (instruction.opcode != spv::Op::OpTypeImage || instruction.word_count < 9U) {
            continue;
        }
        // OpTypeImage operands are sampled type, Dim, Depth, Arrayed, MS, Sampled, ImageFormat.
        const bool arrayed = spirv[instruction.offset + 5U] != 0U;
        const bool multisampled = spirv[instruction.offset + 6U] != 0U;
        if (arrayed && multisampled) {
            found_arrayed_msaa_image = true;
            break;
        }
    }
    EXPECT_TRUE(found_arrayed_msaa_image);

    u32 query_samples_count = 0;
    u32 modulo_count = 0;
    bool fetch_has_sample_operand = false;
    for (const auto& instruction : instructions) {
        if (instruction.opcode == spv::Op::OpImageQuerySamples) {
            ++query_samples_count;
        } else if (instruction.opcode == spv::Op::OpUMod) {
            ++modulo_count;
        } else if (instruction.opcode == spv::Op::OpImageFetch && instruction.word_count >= 7U) {
            const auto mask = static_cast<spv::ImageOperandsMask>(spirv[instruction.offset + 5U]);
            fetch_has_sample_operand =
                (static_cast<u32>(mask) & static_cast<u32>(spv::ImageOperandsMask::Sample)) != 0U;
        }
    }
    EXPECT_EQ(query_samples_count, 1U);
    EXPECT_EQ(modulo_count, 1U);
    EXPECT_TRUE(fetch_has_sample_operand);
}

TEST(ImageReadSpirv, MsaaArrayStorageReadCarriesSampleOperand) {
    const auto spirv = EmitMsaaImageRead(true, true);
    const auto instructions = ParseSpirv(spirv);

    const auto image_read = std::ranges::find_if(instructions, [&](const auto& instruction) {
        return instruction.opcode == spv::Op::OpImageRead;
    });
    ASSERT_NE(image_read, instructions.end());
    ASSERT_GE(image_read->word_count, 7U);
    const auto mask = static_cast<spv::ImageOperandsMask>(spirv[image_read->offset + 5U]);
    EXPECT_NE(static_cast<u32>(mask) & static_cast<u32>(spv::ImageOperandsMask::Sample), 0U);
}

TEST(ImageReadSpirv, ReadOnlyImageReadRemainsSampledDuringInfoCollection) {
    bool has_storage_images = true;
    bool uses_integer_coordinates = true;
    const auto spirv = EmitMsaaImageRead(false, false, true, &has_storage_images,
                                         &uses_integer_coordinates);
    EXPECT_FALSE(has_storage_images);
    EXPECT_FALSE(uses_integer_coordinates);
    EXPECT_FALSE(HasCapability(spirv, spv::Capability::StorageImageMultisample));
}

TEST(ImageReadSpirv, UnsupportedMultisampledStorageIsRejectedAtBackendBoundary) {
    for (const bool array_view : {false, true}) {
        SCOPED_TRACE(array_view);
        bool translation_failed = false;
        const auto spirv = EmitMsaaImageRead(array_view, true, false, nullptr, nullptr,
                                             &translation_failed, false);

        // ResourceTrackingPass rejects this combination in production. Keep the SPIR-V backend
        // boundary fail-closed too because tests, cache replays, and future callers may provide an
        // already-populated IR::Program without running that admission pass first.
        EXPECT_TRUE(translation_failed);
        EXPECT_TRUE(spirv.empty());
    }
}

TEST(ImageAtomicSpirv, MultisampledStorageAtomicDeclaresRequiredCapability) {
    for (const bool array_view : {false, true}) {
        SCOPED_TRACE(array_view);
        const auto spirv = EmitMsaaImageAtomic(array_view);
        EXPECT_TRUE(HasCapability(spirv, spv::Capability::StorageImageMultisample));
    }
}

TEST(ImageAtomicSpirv, BackendDerivesStorageCapabilityWithoutSummaryPass) {
    for (const bool array_view : {false, true}) {
        SCOPED_TRACE(array_view);
        // A backend caller can carry the complete image-resource description without having run
        // the summary pass that sets Info::has_storage_images. StorageImageMultisample must still
        // be declared from the resource itself; otherwise Vulkan rejects OpImageTexelPointer
        // even though the selected device supports the feature.
        const auto spirv = EmitMsaaImageAtomic(array_view, false);
        EXPECT_TRUE(HasCapability(spirv, spv::Capability::StorageImageMultisample));
    }
}

TEST(ImageQuerySpirv, MsaaArrayQueriesAllDimensionsWithoutLod) {
    const auto spirv = EmitMsaaImageQueryDimensions(true);
    const auto instructions = ParseSpirv(spirv);

    u32 query_size_count = 0;
    u32 query_size_lod_count = 0;
    for (const auto& instruction : instructions) {
        query_size_count += instruction.opcode == spv::Op::OpImageQuerySize;
        query_size_lod_count += instruction.opcode == spv::Op::OpImageQuerySizeLod;
    }
    EXPECT_EQ(query_size_count, 1U);
    EXPECT_EQ(query_size_lod_count, 0U);
}

TEST(ImageQuerySpirv, QueryLodAdaptsNativeVec2ToTheF32x4IrContract) {
    const auto spirv = EmitImageQueryLodProgram();
    const auto instructions = ParseSpirv(spirv);
    const auto query = std::ranges::find_if(instructions, [&](const auto& instruction) {
        return instruction.opcode == spv::Op::OpImageQueryLod && instruction.word_count >= 4U;
    });
    ASSERT_NE(query, instructions.end());
    const u32 native_result_type = spirv[query->offset + 1U];
    EXPECT_EQ(FindVectorComponentCount(spirv, native_result_type), 2U);

    // SPIR-V's native query is vec2.  The IR opcode is F32x4, so the backend must materialize
    // a vec4 adapter before later IR component extraction can address lanes 2 and 3.
    const u32 native_result_id = spirv[query->offset + 2U];
    const auto adapted = std::ranges::find_if(instructions, [&](const auto& instruction) {
        return instruction.opcode == spv::Op::OpCompositeConstruct && instruction.word_count >= 6U &&
               FindVectorComponentCount(spirv, spirv[instruction.offset + 1U]) == 4U &&
               spirv[instruction.offset + 3U] == native_result_id;
    });
    ASSERT_NE(adapted, instructions.end());
    const u32 adapted_type = spirv[adapted->offset + 1U];
    const u32 adapted_id = spirv[adapted->offset + 2U];

    // The tail extraction is legal under the IR contract and must not create an out-of-range
    // SPIR-V composite extract from the native vec2 query result.
    const auto extract = std::ranges::find_if(instructions, [&](const auto& instruction) {
        return instruction.opcode == spv::Op::OpCompositeExtract && instruction.word_count >= 5U &&
               spirv[instruction.offset + 1U] != native_result_type &&
               spirv[instruction.offset + 3U] == adapted_id &&
               spirv[instruction.offset + 4U] == 2U;
    });
    ASSERT_NE(extract, instructions.end());
    EXPECT_EQ(FindVectorComponentCount(spirv, adapted_type), 4U);
}

} // namespace
