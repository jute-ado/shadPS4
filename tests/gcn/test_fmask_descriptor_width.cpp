// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <span>

#include <gtest/gtest.h>

#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace Shader {
namespace {

AmdGpu::Image MakeFmaskDescriptor() {
    auto image = AmdGpu::Image::Null(false);
    image.base_address = 0x1000U;
    image.data_format = static_cast<u32>(AmdGpu::DataFormat::FormatFmask8_S2_F1);
    image.num_format = static_cast<u32>(AmdGpu::NumberFormat::Uint);
    image.type = static_cast<u64>(AmdGpu::ImageType::Color2DMsaa);
    image.width = 63U;
    image.height = 31U;
    image.pitch = 127U;
    image.last_array = 3U;
    return image;
}

void TrackFmaskQueries(Info& info, std::initializer_list<bool> descriptor_widths) {
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    const auto image = MakeFmaskDescriptor();
    info.flattened_ud_buf.resize(sizeof(image) / sizeof(u32));
    std::memcpy(info.flattened_ud_buf.data(), &image, sizeof(image));

    IR::Program program{info};
    Pools pools{};
    Gcn::Block cfg_block{};
    IR::Block* const block = pools.block_pool.Create(pools.inst_pool);
    block->cfg_block = &cfg_block;
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);

    IR::IREmitter ir{*block};
    ir.Prologue();
    const auto handle = ir.GetUserData(IR::ScalarReg::S0);
    for (const bool is_r128 : descriptor_widths) {
        IR::TextureInstInfo texture_info{};
        texture_info.is_r128.Assign(is_r128);
        ir.Reference(ir.ImageQueryDimension(handle, ir.Imm32(0U), ir.Imm1(false), texture_info));
    }
    ir.Epilogue();

    Profile profile{};
    profile.supported_spirv = 0x00010600U;
    profile.subgroup_size = 32U;
    Optimization::ResourceTrackingPass(program, profile, false);
}

// Record the real production read width without permitting an out-of-bounds test read. An r128
// descriptor can end exactly at the snapshot boundary; the missing upper half is not padding.
struct RecordingSharpSource {
    std::span<const u32> words;
    mutable size_t requested_bytes{};
    mutable u32 requested_index{};

    template <typename T>
    T ReadUdSharp(u32 index) const {
        requested_bytes = sizeof(T);
        requested_index = index;
        T result{};
        if (index < words.size()) {
            const size_t available = (words.size() - index) * sizeof(u32);
            std::memcpy(&result, words.data() + index, std::min(sizeof(T), available));
        }
        return result;
    }
};

TEST(FMaskDescriptorWidth, R128QueryPreservesBoundedReadAndCanonicalUpperHalf) {
    Info info{};
    TrackFmaskQueries(info, {true});
    ASSERT_FALSE(info.translation_failed);
    ASSERT_EQ(info.fmasks.size(), 1U);

    const RecordingSharpSource source{std::span{info.flattened_ud_buf}.first(4U)};
    const auto descriptor = info.fmasks[0].GetSharp(source);
    EXPECT_EQ(source.requested_bytes, sizeof(u128));
    EXPECT_EQ(source.requested_index, 0U);
    EXPECT_EQ(descriptor.width, 63U);
    EXPECT_EQ(descriptor.height, 31U);
    EXPECT_EQ(descriptor.pitch, descriptor.width);
    EXPECT_EQ(descriptor.last_array, 0U);
}

TEST(FMaskDescriptorWidth, FullQueryKeepsAllEightDescriptorWords) {
    Info info{};
    TrackFmaskQueries(info, {false});
    ASSERT_FALSE(info.translation_failed);
    ASSERT_EQ(info.fmasks.size(), 1U);

    const RecordingSharpSource source{info.flattened_ud_buf};
    const auto descriptor = info.fmasks[0].GetSharp(source);
    EXPECT_EQ(source.requested_bytes, sizeof(AmdGpu::Image));
    EXPECT_EQ(descriptor.pitch, 127U);
    EXPECT_EQ(descriptor.last_array, 3U);
}

TEST(FMaskDescriptorWidth, SameIndexWithDifferentDescriptorWidthsDoesNotAlias) {
    Info info{};
    TrackFmaskQueries(info, {true, false, true, false});
    ASSERT_FALSE(info.translation_failed);
    ASSERT_EQ(info.fmasks.size(), 2U);

    const RecordingSharpSource source{info.flattened_ud_buf};
    info.fmasks[0].GetSharp(source);
    EXPECT_EQ(source.requested_bytes, sizeof(u128));
    info.fmasks[1].GetSharp(source);
    EXPECT_EQ(source.requested_bytes, sizeof(AmdGpu::Image));
}

} // namespace
} // namespace Shader
