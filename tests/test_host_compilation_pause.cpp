// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "common/types.h"
#include "video_core/renderer_vulkan/host_compilation_pause.h"

namespace Vulkan {
namespace {

struct FakeGuestPauseController {
    bool paused{};
    u32 pause_calls{};
    u32 resume_calls{};

    [[nodiscard]] bool IsGuestThreadsPaused() const noexcept {
        return paused;
    }

    void PauseGuestThreads() noexcept {
        paused = true;
        ++pause_calls;
    }

    void ResumeGuestThreads() noexcept {
        paused = false;
        ++resume_calls;
    }
};

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

bool ContainsBetween(const std::string& source, std::string_view begin, std::string_view end,
                     std::string_view expected) {
    const auto first = source.find(begin);
    if (first == std::string::npos) {
        return false;
    }
    const auto last = source.find(end, first + begin.size());
    return last != std::string::npos &&
           source.substr(first, last - first).find(expected) != std::string::npos;
}

TEST(HostCompilationPause, PausesGuestOnlyForTheHostCompilationLifetime) {
    FakeGuestPauseController controller;
    {
        ScopedHostCompilationGuestPause pause{controller};
        EXPECT_TRUE(controller.paused);
        EXPECT_EQ(controller.pause_calls, 1u);
        EXPECT_EQ(controller.resume_calls, 0u);
    }
    EXPECT_FALSE(controller.paused);
    EXPECT_EQ(controller.resume_calls, 1u);
}

TEST(HostCompilationPause, PreservesAnExistingGuestPause) {
    FakeGuestPauseController controller{.paused = true};
    {
        ScopedHostCompilationGuestPause pause{controller};
        EXPECT_EQ(controller.pause_calls, 0u);
    }
    EXPECT_TRUE(controller.paused);
    EXPECT_EQ(controller.resume_calls, 0u);
}

TEST(HostCompilationPause, AsyncShaderCompilationDoesNotPauseGuestThreads) {
    FakeGuestPauseController controller;
    {
        ScopedHostCompilationGuestPause pause{controller, false};
        EXPECT_FALSE(controller.paused);
        EXPECT_EQ(controller.pause_calls, 0u);
    }
    EXPECT_FALSE(controller.paused);
    EXPECT_EQ(controller.resume_calls, 0u);
}

TEST(HostCompilationPause, SynchronousHostWorkStopsGuestFrameLifecycleProgress) {
    const auto source = ReadSource(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(source.empty());
    EXPECT_NE(source.find("host_compilation_pause.h"), std::string::npos);

    EXPECT_TRUE(ContainsBetween(source, "Compiling graphics pipeline", "RegisterPipelineData",
                                "ScopedHostCompilationGuestPause"));
    EXPECT_TRUE(ContainsBetween(source, "Compiling compute pipeline", "RegisterPipelineData",
                                "ScopedHostCompilationGuestPause"));
    EXPECT_TRUE(ContainsBetween(source, "vk::ShaderModule PipelineCache::CompileModule",
                                "RegisterShaderBinary", "ScopedHostCompilationGuestPause"));

    const auto compile_module = source.find("vk::ShaderModule PipelineCache::CompileModule");
    const auto register_shader = source.find("RegisterShaderBinary", compile_module);
    ASSERT_NE(compile_module, std::string::npos);
    ASSERT_NE(register_shader, std::string::npos);
    const auto module_compile = source.substr(compile_module, register_shader - compile_module);
    EXPECT_NE(module_compile.find("IsAsyncGraphicsPipelineCompilation"), std::string::npos);
    EXPECT_NE(module_compile.find("!EmulatorSettings.IsAsyncGraphicsPipelineCompilation()"),
              std::string::npos);

    const auto async_compile = source.find("PipelineCache::CompileGraphicsPipelineAsync");
    const auto async_publish = source.find("PipelineCache::PublishAsyncGraphicsPipeline");
    ASSERT_NE(async_compile, std::string::npos);
    ASSERT_NE(async_publish, std::string::npos);
    EXPECT_EQ(source.substr(async_compile, async_publish - async_compile)
                  .find("ScopedHostCompilationGuestPause"),
              std::string::npos);
}

TEST(PersistentShaderCache, BackendSemanticChangesInvalidateStoredSpirv) {
    const auto source = ReadSource(SHADPS4_PIPELINE_SERIALIZATION_SOURCE_PATH);
    const auto graphics_serialization =
        ReadSource(SHADPS4_GRAPHICS_PIPELINE_SERIALIZATION_SOURCE_PATH);
    const auto specialization = ReadSource(SHADPS4_SPECIALIZATION_SERIALIZATION_SOURCE_PATH);

    ASSERT_FALSE(source.empty());
    EXPECT_NE(source.find("static constexpr u32 ShaderBinaryVersion = 7u;"),
               std::string::npos);
    EXPECT_NE(source.find("static constexpr u32 ShaderMetaVersion = 11u;"),
               std::string::npos);
    EXPECT_NE(source.find("static constexpr u32 PipelineKeyVersion = 5u;"),
              std::string::npos);
    EXPECT_NE(graphics_serialization.find("sdata.Write(fragment)"), std::string::npos);
    EXPECT_NE(graphics_serialization.find("sdata.Read(fragment)"), std::string::npos);
    const auto serialize = specialization.find("void StageSpecialization::Serialize");
    const auto deserialize = specialization.find("bool StageSpecialization::Deserialize", serialize);
    ASSERT_NE(serialize, std::string::npos);
    ASSERT_NE(deserialize, std::string::npos);
    EXPECT_NE(specialization.substr(serialize, deserialize - serialize).find("spec.Write(user_data_mask)"),
              std::string::npos);
    EXPECT_NE(specialization.substr(serialize, deserialize - serialize).find("spec.Write(images)"),
              std::string::npos);
    EXPECT_NE(specialization.substr(deserialize).find("spec.Read(user_data_mask)"), std::string::npos);
    EXPECT_NE(specialization.substr(deserialize).find("spec.Read(images)"), std::string::npos);
}

TEST(GraphicsPipelineLifetime, OwnsAuxiliaryShaderModulesThroughPipelineCreation) {
    const auto source = ReadSource(SHADPS4_GRAPHICS_PIPELINE_SOURCE_PATH);

    const auto owned_modules = source.find("auxiliary_shader_modules");
    const auto compile_owned = source.find("CompileSPVUnique(code");
    const auto compile_tcs = source.find("compile_auxiliary_spv(sdata.tcs");
    const auto compile_tes = source.find("compile_auxiliary_spv(sdata.tes");
    const auto compile_fragment = source.find("compile_auxiliary_spv(sdata.fragment");
    const auto create_pipeline = source.find("createGraphicsPipelineUnique");

    ASSERT_NE(owned_modules, std::string::npos);
    ASSERT_NE(compile_owned, std::string::npos);
    ASSERT_NE(compile_tcs, std::string::npos);
    ASSERT_NE(compile_tes, std::string::npos);
    ASSERT_NE(compile_fragment, std::string::npos);
    ASSERT_NE(create_pipeline, std::string::npos);
    EXPECT_LT(owned_modules, compile_tcs);
    EXPECT_LT(owned_modules, compile_owned);
    EXPECT_LT(compile_tcs, create_pipeline);
    EXPECT_LT(compile_tes, create_pipeline);
    EXPECT_LT(compile_fragment, create_pipeline);
    EXPECT_EQ(source.find(".module = CompileSPV(sdata."), std::string::npos);
}

} // namespace
} // namespace Vulkan
