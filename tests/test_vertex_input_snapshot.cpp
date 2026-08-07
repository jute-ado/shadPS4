// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "shader_recompiler/frontend/fetch_shader.h"

namespace Shader::Gcn {
namespace {

AmdGpu::Buffer MakeBuffer(u64 address, u32 stride, u32 records, u32 format) {
    AmdGpu::Buffer buffer{};
    buffer.base_address = address;
    buffer.stride = stride;
    buffer.num_records = records;
    buffer.data_format = format;
    buffer.num_format = 1;
    buffer.dst_sel_x = 0;
    buffer.dst_sel_y = 1;
    buffer.dst_sel_z = 2;
    buffer.dst_sel_w = 3;
    return buffer;
}

TEST(VertexInputSnapshot, CapturesEachDescriptorOnceForAllDrawPhases) {
    FetchShaderData fetch{};
    fetch.attributes.push_back(VertexAttribute{.semantic = 7});
    fetch.attributes.push_back(VertexAttribute{.semantic = 2});

    const std::array generation_a{
        MakeBuffer(0x1000, 16, 64, 4),
        MakeBuffer(0x2000, 32, 128, 11),
    };
    const std::array generation_b{
        MakeBuffer(0x9000, 4, 8, 1),
        MakeBuffer(0xa000, 8, 16, 2),
    };
    auto live = generation_a;
    size_t reads = 0;

    const auto snapshot = VertexInputSnapshot::Capture(fetch, [&](const VertexAttribute&) {
        const auto index = reads++;
        const auto captured = live[index];
        live[index] = generation_b[index];
        return captured;
    });

    ASSERT_EQ(reads, generation_a.size());
    ASSERT_EQ(snapshot.buffers.size(), generation_a.size());
    for (size_t phase = 0; phase < 3; ++phase) {
        EXPECT_EQ(snapshot.buffers[0], generation_a[0]);
        EXPECT_EQ(snapshot.buffers[1], generation_a[1]);
    }
    EXPECT_EQ(live, generation_b);
}

} // namespace
} // namespace Shader::Gcn
