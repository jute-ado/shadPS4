// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <variant>
#include <tsl/robin_map.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/async_compilation_queue.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
}

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace VideoCore {
class BufferCache;
class PageManager;
} // namespace VideoCore

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        std::unique_ptr<Shader::Info> info;
        Shader::StageSpecialization spec;

        Module() = default;
        Module(vk::ShaderModule module_, Shader::StageSpecialization&& spec_,
               Shader::Info&& info_)
            : module{module_}, info{std::make_unique<Shader::Info>(std::move(info_))},
              spec{std::move(spec_)} {
            spec.info = info.get();
        }

        Module(Module&&) noexcept = default;
        Module& operator=(Module&&) noexcept = default;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};

    Program() = default;
    Program(Shader::Stage stage, Shader::LogicalStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                   Shader::Info&& info) {
        modules.emplace_back(module, std::move(spec), std::move(info));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      Shader::Info&& info, size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        modules[perm_idx] = Module{module, std::move(spec), std::move(info)};
    }

    const Shader::Info& GetPermutInfo(size_t perm_idx) const {
        ASSERT(perm_idx < modules.size() && modules[perm_idx].info);
        return *modules[perm_idx].info;
    }

    Shader::Info& GetPermutInfo(size_t perm_idx) {
        return const_cast<Shader::Info&>(std::as_const(*this).GetPermutInfo(perm_idx));
    }
};

struct AsyncGraphicsPipelineJob {
    GraphicsPipelineKey key{};
    std::array<Shader::Info, MaxShaderStages> info_copies{};
    std::array<std::array<u32, Shader::NUM_USER_DATA_REGS>, MaxShaderStages> user_data_copies{};
    std::array<u32, MaxShaderStages> user_data_sizes{};
    std::array<const Shader::Info*, MaxShaderStages> retained_infos{};
    std::array<const Shader::Info*, MaxShaderStages> compile_infos{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};

    void RebindCompileInfo() noexcept;
};

struct AsyncGraphicsPipelineResult {
    std::unique_ptr<GraphicsPipeline> pipeline;
    GraphicsPipeline::SerializationSupport serialization;
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::array<const Shader::Info*, MaxShaderStages> retained_infos{};
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool, VideoCore::PageManager& page_manager,
                           VideoCore::BufferCache& buffer_cache);
    ~PipelineCache();

    void WarmUp();
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    const GraphicsPipeline* GetGraphicsPipeline();

    const ComputePipeline* GetComputePipeline();

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule,
                              std::optional<Shader::Gcn::FetchShaderData>, u64>;
    Result GetProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    using AsyncGraphicsPipelineQueue =
        AsyncCompilationQueue<GraphicsPipelineKey, AsyncGraphicsPipelineResult,
                              AsyncGraphicsPipelineJob>;

    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();
    AsyncGraphicsPipelineResult CompileGraphicsPipelineAsync(AsyncGraphicsPipelineJob job);
    AsyncGraphicsPipelineJob MakeAsyncGraphicsPipelineJob() const;
    void PublishAsyncGraphicsPipeline(AsyncGraphicsPipelineQueue::Completion completion);
    void PublishAsyncGraphicsPipelines();
    const GraphicsPipeline* WaitForAsyncGraphicsPipeline(const GraphicsPipelineKey& key);

    void DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::Stage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding,
                                   const Shader::CompileResourceSnapshot* compile_resources = nullptr,
                                   const Shader::Gcn::FetchShaderData* compile_fetch_shader = nullptr);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::Stage stage, Shader::LogicalStage l_stage);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    VideoCore::PageManager& page_manager;
    VideoCore::BufferCache& buffer_cache;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    std::unique_ptr<AsyncGraphicsPipelineQueue> async_graphics_compiler;

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

} // namespace Vulkan
