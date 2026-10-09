// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "shader_recompiler/buffer_access_range.h"

TEST(BufferAccessRange, NarrowsDeclaredRangeToStaticAccessPrefix) {
    Shader::BufferAccessRange range;
    range.Add(12, 16);
    range.Add(3, 4);
    range.Add(8, 16);

    EXPECT_TRUE(range.IsBounded());
    EXPECT_EQ(range.UpperBound(), 28);
    EXPECT_EQ(range.Fit(0xfffffffcULL), 28);
}

TEST(BufferAccessRange, PreservesSmallerDescriptorBound) {
    Shader::BufferAccessRange range;
    range.Add(12, 16);

    EXPECT_EQ(range.Fit(16), 16);
}

TEST(BufferAccessRange, DynamicAccessDisablesNarrowing) {
    Shader::BufferAccessRange range;
    range.Add(0, 16);
    range.MarkDynamic();
    range.Add(64, 4);

    EXPECT_FALSE(range.IsBounded());
    EXPECT_EQ(range.Fit(0xfffffffcULL), 0xfffffffcULL);
}

TEST(BufferAccessRange, OverflowDisablesNarrowing) {
    Shader::BufferAccessRange range;
    range.Add(std::numeric_limits<u64>::max() - 3, 8);

    EXPECT_FALSE(range.IsBounded());
    EXPECT_EQ(range.Fit(4_GB), 4_GB);
}

TEST(BufferAccessRange, EmptyAnalysisPreservesDeclaredRange) {
    const Shader::BufferAccessRange range;

    EXPECT_TRUE(range.IsBounded());
    EXPECT_EQ(range.UpperBound(), 0);
    EXPECT_EQ(range.Fit(4_GB), 4_GB);
}

TEST(BufferAddressing, AddTidUsesTheWaveLaneInEveryShaderStage) {
    std::ifstream input{SHADPS4_RESOURCE_TRACKING_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto source = contents.str();

    const auto add_tid = source.find("if (buffer.add_tid_enable)");
    const auto offset = source.find("// offset =", add_tid);
    ASSERT_NE(add_tid, std::string::npos);
    ASSERT_NE(offset, std::string::npos);
    const auto block = source.substr(add_tid, offset - add_tid);
    EXPECT_NE(block.find("ir.LaneId()"), std::string::npos);
    EXPECT_EQ(block.find("LogicalStage::Compute"), std::string::npos);
    EXPECT_EQ(block.find("ASSERT"), std::string::npos);
}

TEST(PersistentShaderCache, PersistedSrtWalkerBytesAreSkippedWithoutRegistration) {
    std::ifstream serialization_input{SHADPS4_PIPELINE_SERIALIZATION_SOURCE_PATH, std::ios::binary};
    std::ostringstream serialization_contents;
    serialization_contents << serialization_input.rdbuf();
    const auto serialization_source = serialization_contents.str();

    const auto deserialize_begin = serialization_source.find("bool PersistentSrtInfo::Deserialize");
    const auto deserialize_end =
        serialization_source.find("void StageSpecialization::Serialize", deserialize_begin);
    ASSERT_NE(deserialize_begin, std::string::npos);
    ASSERT_NE(deserialize_end, std::string::npos);
    const auto deserialize =
        serialization_source.substr(deserialize_begin, deserialize_end - deserialize_begin);
    const auto clear = deserialize.find("walker_func = nullptr");
    const auto skip = deserialize.find("ar.Advance(walker_func_size)");
    ASSERT_NE(clear, std::string::npos);
    ASSERT_NE(skip, std::string::npos);
    EXPECT_LT(clear, skip);
    EXPECT_EQ(deserialize.find("RegisterWalkerCode("), std::string::npos);

    std::ifstream input{SHADPS4_FLATTEN_USERDATA_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto source = contents.str();

    const auto generate_begin = source.find("static void GenerateSrtProgram");
    const auto generate_end = source.find("info.srt_info.walker_func =", generate_begin);
    ASSERT_NE(generate_begin, std::string::npos);
    ASSERT_NE(generate_end, std::string::npos);
    const auto generate = source.substr(generate_begin, generate_end - generate_begin);
    EXPECT_NE(generate.find("EnsureSrtWalkerSignalHandlerRegistered()"), std::string::npos);
}

TEST(PersistentShaderCache, PersistedSrtWalkerCodeIsRecompiledInsteadOfExecuted) {
    std::ifstream input{SHADPS4_PIPELINE_SERIALIZATION_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto source = contents.str();

    const auto load_begin = source.find("bool LoadShaderMeta");
    const auto load_end = source.find("void ComputePipelineKey::Serialize", load_begin);
    ASSERT_NE(load_begin, std::string::npos);
    ASSERT_NE(load_end, std::string::npos);
    const auto load = source.substr(load_begin, load_end - load_begin);
    const auto deserialize = load.find("info.Deserialize(ar)");
    const auto reject = load.find("info.srt_info.walker_func_size");
    const auto success = load.rfind("return true");
    ASSERT_NE(deserialize, std::string::npos);
    ASSERT_NE(reject, std::string::npos);
    ASSERT_NE(success, std::string::npos);
    EXPECT_LT(deserialize, reject);
    EXPECT_LT(reject, success);
}

TEST(PersistentShaderCache, SrtWalkerCodeGenerationAndFaultPatchingAreSerialized) {
    std::ifstream input{SHADPS4_FLATTEN_USERDATA_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto source = contents.str();

    EXPECT_NE(source.find("std::mutex g_srt_codegen_mutex"), std::string::npos);

    const auto register_begin = source.find("PFN_SrtWalker RegisterWalkerCode");
    const auto register_end = source.find("} // namespace Shader", register_begin);
    ASSERT_NE(register_begin, std::string::npos);
    ASSERT_NE(register_end, std::string::npos);
    const auto register_code = source.substr(register_begin, register_end - register_begin);
    EXPECT_LT(register_code.find("std::lock_guard lock{g_srt_codegen_mutex}"),
              register_code.find("g_srt_codegen.getCurr()"));

    const auto handler_begin = source.find("static bool SrtWalkerSignalHandler");
    const auto handler_end =
        source.find("void EnsureSrtWalkerSignalHandlerRegistered", handler_begin);
    ASSERT_NE(handler_begin, std::string::npos);
    ASSERT_NE(handler_end, std::string::npos);
    const auto handler = source.substr(handler_begin, handler_end - handler_begin);
    EXPECT_LT(handler.find("std::lock_guard lock{g_srt_codegen_mutex}"),
              handler.find("g_srt_codegen.getSize()"));
    EXPECT_LT(handler.find("SrtWalkerPatchAlreadyApplied"), handler.find("decodeInstruction"));
    EXPECT_LT(handler.find("memset(code_patch + patch_size"),
              handler.find("FlushSrtWalkerInstructionCache"));

    const auto generate_begin = source.find("static void GenerateSrtProgram");
    const auto generate_end = source.find("info.srt_info.walker_func =", generate_begin);
    ASSERT_NE(generate_begin, std::string::npos);
    ASSERT_NE(generate_end, std::string::npos);
    const auto generate = source.substr(generate_begin, generate_end - generate_begin);
    EXPECT_NE(generate.find("std::lock_guard lock{g_srt_codegen_mutex}"), std::string::npos);
}

TEST(BufferFormatLowering, InvalidFormatPairsFailClosedBeforePackOrUnpack) {
    std::ifstream input{SHADPS4_LOWER_BUFFER_FORMAT_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto source = contents.str();
    const auto lower_begin = source.find("static void LowerBufferFormatInst");
    const auto lower_end = source.find("void LowerBufferFormatToRaw", lower_begin);
    ASSERT_NE(lower_begin, std::string::npos);
    ASSERT_NE(lower_end, std::string::npos);
    const auto lower = source.substr(lower_begin, lower_end - lower_begin);
    const auto validate = lower.find("IsBufferFormatNumberSupported");
    const auto load = lower.find("LoadBufferFormat(");
    ASSERT_NE(validate, std::string::npos);
    ASSERT_NE(load, std::string::npos);
    EXPECT_LT(validate, load);
    EXPECT_NE(lower.substr(validate, load - validate).find("ReplaceUsesWithAndRemove"),
              std::string::npos);
    EXPECT_NE(lower.substr(validate, load - validate).find("inst.Invalidate()"), std::string::npos);
}

TEST(BufferFormatLowering, IntegerTenElevenElevenUsesExactBitfieldWidths) {
    std::ifstream input{SHADPS4_IR_EMITTER_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto source = contents.str();
    const auto pack_begin = source.find("U32 IREmitter::Pack10_11_11");
    const auto unpack_end = source.find("F32F64 IREmitter::FPMul", pack_begin);
    ASSERT_NE(pack_begin, std::string::npos);
    ASSERT_NE(unpack_end, std::string::npos);
    const auto formats = source.substr(pack_begin, unpack_end - pack_begin);
    EXPECT_NE(formats.find("case AmdGpu::NumberFormat::Uint"), std::string::npos);
    EXPECT_NE(formats.find("case AmdGpu::NumberFormat::Sint"), std::string::npos);
    EXPECT_NE(formats.find("BitFieldInsert"), std::string::npos);
    EXPECT_NE(formats.find("BitFieldExtract(value, Imm32(0), Imm32(11),"), std::string::npos);
    EXPECT_NE(formats.find("BitFieldExtract(value, Imm32(11), Imm32(11),"), std::string::npos);
    EXPECT_NE(formats.find("BitFieldExtract(value, Imm32(22), Imm32(10),"), std::string::npos);
}

TEST(BufferBinding, PreservesByteAlignedDescriptorAdjustment) {
    std::ifstream rasterizer_input{SHADPS4_RASTERIZER_SOURCE_PATH, std::ios::binary};
    std::ostringstream rasterizer_contents;
    rasterizer_contents << rasterizer_input.rdbuf();
    const auto rasterizer = rasterizer_contents.str();
    const auto bind_begin = rasterizer.find("void Rasterizer::BindBuffers");
    const auto bind_end = rasterizer.find("void Rasterizer::BindTextures", bind_begin);
    ASSERT_NE(bind_begin, std::string::npos);
    ASSERT_NE(bind_end, std::string::npos);
    const auto bind = rasterizer.substr(bind_begin, bind_end - bind_begin);
    EXPECT_NE(bind.find("push_data.AddOffset(binding.buffer, adjust)"), std::string::npos);
    EXPECT_EQ(bind.find("adjust % 4"), std::string::npos);

    std::ifstream spirv_input{SHADPS4_SPIRV_CONTEXT_SOURCE_PATH, std::ios::binary};
    std::ostringstream spirv_contents;
    spirv_contents << spirv_input.rdbuf();
    const auto spirv = spirv_contents.str();
    EXPECT_NE(spirv.find("buffer.Offset(PointerSize::B8) = buf_offset"), std::string::npos);
    EXPECT_NE(spirv.find("OpShiftRightLogical(U32[1], buf_offset, ConstU32(2U))"),
              std::string::npos);
}

TEST(TextureBinding, ReservedImageFormatsAreRejectedBeforeImageConstruction) {
    std::ifstream rasterizer_input{SHADPS4_RASTERIZER_SOURCE_PATH, std::ios::binary};
    std::ostringstream rasterizer_contents;
    rasterizer_contents << rasterizer_input.rdbuf();
    const auto rasterizer = rasterizer_contents.str();
    const auto bind_begin = rasterizer.find("void Rasterizer::BindTextures");
    const auto bind_end = rasterizer.find("void Rasterizer::Resolve", bind_begin);
    ASSERT_NE(bind_begin, std::string::npos);
    ASSERT_NE(bind_end, std::string::npos);
    const auto bind = rasterizer.substr(bind_begin, bind_end - bind_begin);
    const auto validate = bind.find("IsSurfaceFormatSupported(tsharp.GetDataFmt(), number_format)");
    const auto construct = bind.find("std::tuple{tsharp, image_desc}");
    ASSERT_NE(validate, std::string::npos);
    ASSERT_NE(construct, std::string::npos);
    EXPECT_LT(validate, construct);
    const auto reject = bind.substr(validate, construct - validate);
    EXPECT_NE(reject.find("image_bindings.emplace_back"), std::string::npos);
    EXPECT_NE(reject.find("continue"), std::string::npos);
}

TEST(BufferAccessRange, NarrowsOnlyMatchingAddressLayouts) {
    Shader::BufferAccessRange range;
    const Shader::BufferAddressLayout compiled_layout{
        .stride = 4,
        .data_format = 4,
        .element_size = 1,
        .index_stride = 2,
        .swizzle_enable = false,
        .add_tid_enable = false,
    };
    range.RecordAddressLayout(compiled_layout);
    range.Add(48, 16);

    constexpr u64 declared_size = 0xfffffffcULL;
    EXPECT_EQ(range.Fit(declared_size, compiled_layout), 64);

    auto changed_stride = compiled_layout;
    changed_stride.stride = 8;
    EXPECT_EQ(range.Fit(declared_size, changed_stride), declared_size);

    auto changed_swizzle = compiled_layout;
    changed_swizzle.swizzle_enable = true;
    EXPECT_EQ(range.Fit(declared_size, changed_swizzle), declared_size);
}
