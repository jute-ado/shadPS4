// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// CPU-only source contract for malformed guest buffer descriptors.  A descriptor can carry a
// small, non-zero address (for example 0x20011) together with a huge binding range.  Passing that
// pair to ClampRangeSize asserts before the rasterizer can reject the draw.  BindBuffers must
// reject the stage before the clamp and make BindResources return false.

#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

namespace {

#ifndef SHADPS4_RASTERIZER_SOURCE_PATH
#define SHADPS4_RASTERIZER_SOURCE_PATH "src/video_core/renderer_vulkan/vk_rasterizer.cpp"
#endif

#ifndef SHADPS4_BUFFER_CACHE_SOURCE_PATH
#define SHADPS4_BUFFER_CACHE_SOURCE_PATH "src/video_core/buffer_cache/buffer_cache.cpp"
#endif

[[nodiscard]] std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input.is_open()) << path;
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

TEST(BufferBindingMappingGuard, RejectsInvalidBaseBeforeClampRangeSize) {
    const auto source = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto begin = source.find("void Rasterizer::BindBuffers(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = source.find("void Rasterizer::BindTextures(", begin);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    const auto binding_size = body.find("const u64 binding_size = desc.GetBindingSize(vsharp);");
    const auto clamp = body.find("memory->ClampRangeSize(vsharp.base_address, binding_size);");
    ASSERT_NE(binding_size, std::string::npos);
    ASSERT_NE(clamp, std::string::npos);
    ASSERT_LT(binding_size, clamp);

    const auto guard =
        body.rfind("if (!memory->IsValidMapping(vsharp.base_address, binding_size))", clamp);
    ASSERT_NE(guard, std::string::npos);
    EXPECT_LT(binding_size, guard);
    EXPECT_LT(guard, clamp);

    const auto failure = body.find("resource_bindings_failed = true", guard);
    const auto return_statement = body.find("return;", guard);
    ASSERT_NE(failure, std::string::npos);
    ASSERT_NE(return_statement, std::string::npos);
    EXPECT_LT(guard, failure);
    EXPECT_LT(failure, return_statement);
    EXPECT_LT(return_statement, clamp);
}

TEST(BufferBindingMappingGuard, BindResourcesChecksResourceBindingFailure) {
    const auto source = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto begin = source.find("bool Rasterizer::BindResources(const Pipeline* pipeline)");
    ASSERT_NE(begin, std::string::npos);
    const auto end = source.find("void Rasterizer::PrepareAddr64PointerResidency(", begin);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    const auto reset = body.find("resource_bindings_failed = false;");
    const auto bind = body.find("BindBuffers(*stage, binding, push_data);");
    const auto check = body.find("resource_bindings_failed", bind);
    ASSERT_NE(reset, std::string::npos);
    ASSERT_NE(bind, std::string::npos);
    ASSERT_NE(check, std::string::npos);
    EXPECT_LT(reset, bind);
    EXPECT_LT(bind, check);
    EXPECT_NE(body.find("return false;", check), std::string::npos);
}

TEST(BufferBindingMappingGuard, RejectsInvalidVertexBaseBeforeClampRangeSize) {
    const auto source = ReadSource(SHADPS4_BUFFER_CACHE_SOURCE_PATH);
    const auto begin = source.find("BufferCache::PreparedVertexBuffers BufferCache::PrepareVertexBuffers(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = source.find("bool BufferCache::FinalizeVertexBuffers(", begin);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    const auto inputs = body.find("pipeline.GetVertexInputs(");
    const auto clamp = body.find("memory->ClampRangeSize(range.base_address, range.GetSize());");
    const auto guard = body.find(
        "memory->IsValidMapping(buffer.base_address, buffer.GetSize())", inputs);
    const auto failure = body.find("prepared.valid = false", guard);
    const auto return_statement = body.find("return prepared;", guard);
    ASSERT_NE(inputs, std::string::npos);
    ASSERT_NE(clamp, std::string::npos);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(failure, std::string::npos);
    ASSERT_NE(return_statement, std::string::npos);
    EXPECT_LT(inputs, guard);
    EXPECT_LT(guard, failure);
    EXPECT_LT(failure, return_statement);
    EXPECT_LT(return_statement, clamp);
}

TEST(BufferBindingMappingGuard, RejectsInvalidVertexRangeBeforeClampRangeSize) {
    const auto source = ReadSource(SHADPS4_BUFFER_CACHE_SOURCE_PATH);
    const auto begin = source.find("BufferCache::PreparedVertexBuffers BufferCache::PrepareVertexBuffers(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = source.find("bool BufferCache::FinalizeVertexBuffers(", begin);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    const auto range_guard = body.find(
        "memory->IsValidMapping(buffer.base_address, buffer.GetSize())");
    const auto clamp = body.find("memory->ClampRangeSize(range.base_address, range.GetSize());");
    ASSERT_NE(range_guard, std::string::npos);
    ASSERT_NE(clamp, std::string::npos);
    EXPECT_LT(range_guard, clamp);
}

TEST(BufferBindingMappingGuard, CentralPreparationRejectsInvalidRangeBeforeObtain) {
    const auto source = ReadSource(SHADPS4_BUFFER_CACHE_SOURCE_PATH);
    const auto begin = source.find("BufferCache::PreparedBufferBinding BufferCache::PrepareBufferBinding(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = source.find("std::optional<BufferCache::BufferBindingEndpoint>", begin);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    const auto guard = body.find("memory->IsValidMapping(device_addr, size)");
    const auto obtain = body.find("ObtainBuffer(device_addr, size", guard);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(obtain, std::string::npos);
    EXPECT_LT(guard, obtain);
}

TEST(BufferBindingMappingGuard, BindBuffersRejectsInvalidRangeBeforeClampRangeSize) {
    const auto source = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto begin = source.find("void Rasterizer::BindBuffers(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = source.find("void Rasterizer::BindTextures(", begin);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);

    const auto guard = body.find("memory->IsValidMapping(vsharp.base_address, binding_size)");
    const auto clamp = body.find("memory->ClampRangeSize(vsharp.base_address, binding_size);");
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(clamp, std::string::npos);
    EXPECT_LT(guard, clamp);
}

TEST(BufferBindingMappingGuard, FinalizeRejectsInvalidPreparedVertexBuffers) {
    const auto source = ReadSource(SHADPS4_BUFFER_CACHE_SOURCE_PATH);
    const auto begin = source.find("bool BufferCache::FinalizeVertexBuffers(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = source.find("void BufferCache::BindVertexBuffers(", begin);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto guard = body.find("if (!prepared.valid)");
    const auto failure = body.find("return false;", guard);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(failure, std::string::npos);
    EXPECT_LT(guard, failure);
}

} // namespace
