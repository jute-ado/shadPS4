// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/async_graphics_optimization_diagnostic.h"

namespace {

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input.is_open());
    std::string source{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    std::erase(source, '\r');
    return source;
}

TEST(AsyncGraphicsOptimizationDiagnostic, DefaultAndNonExactValuesStayDisabled) {
    for (const auto value : {"", "0", "true", "yes", "01", "+1", " 1", "1 ", "1\n", "-1"}) {
        EXPECT_FALSE(Vulkan::ParseAsyncGraphicsDisableOptimization(value)) << value;
    }
}

TEST(AsyncGraphicsOptimizationDiagnostic, OnlyExactOneEnablesTheHint) {
    EXPECT_TRUE(Vulkan::ParseAsyncGraphicsDisableOptimization("1"));
}

TEST(AsyncGraphicsOptimizationDiagnostic, NonAsyncStaysUnchangedEvenWhenOptedIn) {
    EXPECT_EQ(Vulkan::AsyncGraphicsDiagnosticFlags(false, false), 0U);
    EXPECT_EQ(Vulkan::AsyncGraphicsDiagnosticFlags(false, true), 0U);
}

TEST(AsyncGraphicsOptimizationDiagnostic, AsyncDefaultStaysUnchanged) {
    EXPECT_EQ(Vulkan::AsyncGraphicsDiagnosticFlags(true, false), 0U);
}

TEST(AsyncGraphicsOptimizationDiagnostic, AsyncOptInSetsOnlyDisableOptimizationBit) {
    EXPECT_EQ(Vulkan::AsyncGraphicsDiagnosticFlags(true, true),
              VkPipelineCreateFlags{VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT});
}

TEST(AsyncGraphicsOptimizationDiagnostic, ActualCreateInfoFlagsFollowModeAndOptIn) {
    for (const bool asynchronous : {false, true}) {
        for (const auto value : {"", "0", "1", "true", "01"}) {
            const VkGraphicsPipelineCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .flags = Vulkan::AsyncGraphicsDiagnosticFlags(
                    asynchronous, Vulkan::ParseAsyncGraphicsDisableOptimization(value)),
                .stageCount = 3,
            };
            EXPECT_EQ(create_info.flags,
                      asynchronous && std::string_view{value} == "1"
                          ? VkPipelineCreateFlags{VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT}
                          : VkPipelineCreateFlags{});
            EXPECT_EQ(create_info.stageCount, 3U);
            EXPECT_EQ(create_info.pNext, nullptr);
        }
    }
}

TEST(AsyncGraphicsOptimizationDiagnosticWiring, AsyncWorkerPassesPolicyWithoutChangingWaits) {
    const auto source = ReadSource(SHADPS4_OPTIMIZATION_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin = source.find("PipelineCache::CompileGraphicsPipelineAsync(");
    const auto end = source.find("void PipelineCache::PublishAsyncGraphicsPipeline(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto worker = source.substr(begin, end - begin);
    const auto constructor = worker.find("std::make_unique<GraphicsPipeline>(");
    const auto options = worker.find("serialization, false,", constructor);
    const auto hint = worker.find("AsyncGraphicsPipelineCreationFlags()", options);
    ASSERT_NE(constructor, std::string::npos);
    ASSERT_NE(options, std::string::npos);
    EXPECT_LT(hint, worker.find(");", options));
    EXPECT_NE(worker.find("profile, job.key, *pipeline_cache, job.compile_infos"),
              std::string::npos);
    EXPECT_EQ(worker.find("vk::PipelineCache{}"), std::string::npos);
    for (const auto forbidden : {"ScopedHostCompilationGuestPause", "PauseGuest", "wait_for(",
                                 "sleep_for(", "Skip", "GetProgram(", "EmitSPIRV("}) {
        EXPECT_EQ(worker.find(forbidden), std::string::npos) << forbidden;
    }
}

TEST(AsyncGraphicsOptimizationDiagnosticWiring, DriverCacheIsReadyBeforeWorkerAndOutlivesItsJoin) {
    const auto source = ReadSource(SHADPS4_OPTIMIZATION_PIPELINE_CACHE_SOURCE_PATH);
    const auto constructor = source.find("PipelineCache::PipelineCache(");
    const auto destructor = source.find("PipelineCache::~PipelineCache()", constructor);
    ASSERT_NE(constructor, std::string::npos);
    ASSERT_NE(destructor, std::string::npos);
    const auto creation = source.substr(constructor, destructor - constructor);
    const auto default_cache = creation.find("createPipelineCacheUnique({})");
    const auto own_cache = creation.find("pipeline_cache = std::move(cache)");
    const auto start_worker = creation.find("std::make_unique<AsyncGraphicsPipelineQueue>");
    ASSERT_NE(default_cache, std::string::npos);
    ASSERT_NE(own_cache, std::string::npos);
    ASSERT_NE(start_worker, std::string::npos);
    EXPECT_LT(default_cache, own_cache);
    EXPECT_LT(own_cache, start_worker);
    // Flags zero retain Vulkan's internal synchronization for concurrent creation. The
    // destructor body joins the queue before C++ destroys any renderer/cache members.
    const auto destructor_end = source.find("\n}", destructor);
    ASSERT_NE(destructor_end, std::string::npos);
    EXPECT_NE(source.substr(destructor, destructor_end - destructor)
                  .find("async_graphics_compiler.reset()"),
              std::string::npos);
    EXPECT_EQ(source.find("eExternallySynchronized"), std::string::npos);
}

TEST(AsyncGraphicsOptimizationDiagnosticWiring, ImmutablePolicyReadsAndLogsInsideOnceInitializer) {
    const auto source = ReadSource(SHADPS4_OPTIMIZATION_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin =
        source.find("static vk::PipelineCreateFlags AsyncGraphicsPipelineCreationFlags()");
    ASSERT_NE(begin, std::string::npos);
    const auto end = source.find(
        "AsyncGraphicsPipelineResult PipelineCache::CompileGraphicsPipelineAsync", begin);
    ASSERT_NE(end, std::string::npos);
    const auto policy = source.substr(begin, end - begin);
    const auto once = policy.find("static const auto flags = []");
    const auto getenv = policy.find("std::getenv(\"SHADPS4_ASYNC_GRAPHICS_DISABLE_OPTIMIZATION\")");
    const auto parse = policy.find("ParseAsyncGraphicsDisableOptimization(");
    const auto select = policy.find("AsyncGraphicsDiagnosticFlags(true,");
    const auto log = policy.find("LOG_WARNING(");
    const auto initializer_end = policy.find("}();");
    ASSERT_NE(initializer_end, std::string::npos);
    EXPECT_LT(once, getenv);
    EXPECT_LT(getenv, parse);
    EXPECT_LT(parse, select);
    EXPECT_LT(select, log);
    EXPECT_LT(log, initializer_end);
    EXPECT_NE(policy.find("if (creation_flags != 0)"), std::string::npos);
}

TEST(AsyncGraphicsOptimizationDiagnosticWiring, ActualDriverCreateInfoReceivesExplicitFlags) {
    const auto source = ReadSource(SHADPS4_OPTIMIZATION_GRAPHICS_SOURCE_PATH);
    const auto begin = source.find("const vk::GraphicsPipelineCreateInfo pipeline_info = {");
    ASSERT_NE(begin, std::string::npos);
    const auto driver =
        source.find("device.createGraphicsPipelineUnique(pipeline_cache, pipeline_info)", begin);
    ASSERT_NE(driver, std::string::npos);
    EXPECT_NE(source.substr(begin, driver - begin).find(".flags = creation_flags"),
              std::string::npos);
    const auto header = ReadSource(SHADPS4_OPTIMIZATION_GRAPHICS_HEADER_PATH);
    EXPECT_NE(header.find("vk::PipelineCreateFlags creation_flags = {}"), std::string::npos);
}

TEST(AsyncGraphicsOptimizationDiagnosticWiring, SynchronousPreloadAndComputeKeepExistingPolicy) {
    const auto source = ReadSource(SHADPS4_OPTIMIZATION_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin = source.find("const GraphicsPipeline* PipelineCache::GetGraphicsPipeline()");
    const auto end = source.find("bool PipelineCache::RefreshGraphicsKey()", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto synchronous_and_compute = source.substr(begin, end - begin);
    EXPECT_EQ(synchronous_and_compute.find("AsyncGraphicsPipelineCreationFlags"),
              std::string::npos);
    EXPECT_NE(synchronous_and_compute.find("modules, sdata, false)"), std::string::npos);
    const auto serialized = ReadSource(SHADPS4_OPTIMIZATION_SERIALIZATION_SOURCE_PATH);
    EXPECT_NE(serialized.find("modules, sdata, true)"), std::string::npos);
    EXPECT_EQ(serialized.find("AsyncGraphicsPipelineCreationFlags"), std::string::npos);
    EXPECT_EQ(serialized.find("AsyncGraphicsDiagnosticFlags"), std::string::npos);
    EXPECT_EQ(serialized.find("disable_optimization"), std::string::npos);
}

} // namespace
