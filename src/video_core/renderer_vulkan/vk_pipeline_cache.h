// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <unordered_set>
#include <variant>
#include <vector>
#include <tsl/robin_map.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "shader_recompiler/specialization_identity.h"
#include "video_core/renderer_vulkan/async_compilation_queue.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/shader_selection_trace.h"

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
template <typename PageManager, typename BufferCache>
class SrtMemoryReadSession;
} // namespace VideoCore

namespace Vulkan {

using SrtReadSession =
    VideoCore::SrtMemoryReadSession<VideoCore::PageManager, VideoCore::BufferCache>;

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        std::unique_ptr<Shader::Info> info;
        Shader::StageSpecialization spec;
        std::vector<u32> runtime_vs_copy;
        u64 permutation_id{};

        Module() = default;
        Module(vk::ShaderModule module_, Shader::StageSpecialization&& spec_,
               Shader::Info&& info_, u64 permutation_id_ = {})
            : module{module_}, info{std::make_unique<Shader::Info>(std::move(info_))},
              spec{std::move(spec_)}, permutation_id{permutation_id_} {
            spec.info = info.get();
            if (spec.runtime_info.stage == Shader::Stage::Geometry) {
                runtime_vs_copy.assign(spec.runtime_info.gs_info.vs_copy.begin(),
                                       spec.runtime_info.gs_info.vs_copy.end());
                spec.runtime_info.gs_info.vs_copy = runtime_vs_copy;
            }
        }

        Module(Module&&) noexcept = default;
        Module& operator=(Module&&) noexcept = default;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    enum class PreloadAction {
        Insert,
        Reuse,
        Conflict,
    };

    struct PreloadDecision {
        PreloadAction action{PreloadAction::Conflict};
        const Module* module{};
    };

    Shader::Info info;
    ModuleList modules{};
    u64 generation{};

    Program() = default;
    Program(Shader::Stage stage, Shader::LogicalStage l_stage, Shader::ShaderParams params,
            u64 generation_ = {})
        : info{stage, l_stage, params}, generation{generation_} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                   Shader::Info&& info, u64 permutation_id = {}) {
        modules.emplace_back(module, std::move(spec), std::move(info), permutation_id);
    }

    [[nodiscard]] PreloadDecision InspectPreloadPermutation(
        const Shader::StageSpecialization& spec, u64 permutation_id) const {
        const auto identity_match =
            std::ranges::find_if(modules, [&](const Module& candidate) {
                return candidate.info && candidate.permutation_id == permutation_id;
            });
        if (identity_match != modules.end()) {
            return identity_match->spec == spec
                       ? PreloadDecision{.action = PreloadAction::Reuse,
                                         .module = &*identity_match}
                       : PreloadDecision{.action = PreloadAction::Conflict};
        }

        const auto specialization_match =
            std::ranges::find_if(modules, [&](const Module& candidate) {
                return candidate.info && candidate.spec == spec;
            });
        if (specialization_match != modules.end()) {
            return {.action = PreloadAction::Conflict};
        }
        return {.action = PreloadAction::Insert};
    }

    [[nodiscard]] bool InsertPreloadPermutation(vk::ShaderModule module,
                                                Shader::StageSpecialization&& spec,
                                                Shader::Info&& info, u64 permutation_id) {
        if (InspectPreloadPermutation(spec, permutation_id).action != PreloadAction::Insert) {
            return false;
        }
        modules.emplace_back(module, std::move(spec), std::move(info), permutation_id);
        return true;
    }

    const Shader::Info& GetPermutInfo(size_t perm_idx) const {
        ASSERT(perm_idx < modules.size() && modules[perm_idx].info);
        return *modules[perm_idx].info;
    }

    Shader::Info& GetPermutInfo(size_t perm_idx) {
        return const_cast<Shader::Info&>(std::as_const(*this).GetPermutInfo(perm_idx));
    }

    auto FindPermut(const Shader::StageSpecialization& spec) {
        return std::ranges::find(modules, spec, &Module::spec);
    }

    auto FindPermut(const Shader::StageSpecialization& spec) const {
        return std::ranges::find(modules, spec, &Module::spec);
    }
};

struct AsyncShaderModuleKey {
    Shader::ProgramCacheKey program{};
    u64 program_generation{};
    u64 permutation_id{};

    bool operator==(const AsyncShaderModuleKey&) const noexcept = default;
};

struct AsyncShaderModuleKeyHash {
    [[nodiscard]] size_t operator()(const AsyncShaderModuleKey& key) const noexcept {
        const u64 program_hash = static_cast<u64>(Shader::ProgramCacheKeyHash{}(key.program));
        return static_cast<size_t>(
            HashCombine(HashCombine(program_hash, key.program_generation), key.permutation_id));
    }
};

struct AsyncShaderModuleJob {
    static constexpr u32 InitialInstPoolSize = 1024;
    static constexpr u32 InitialBlockPoolSize = 16;

    AsyncShaderModuleKey key{};
    std::unique_ptr<Shader::Pools> pools{
        std::make_unique<Shader::Pools>(InitialInstPoolSize, InitialBlockPoolSize)};
    Shader::Info info{};
    Shader::RuntimeInfo runtime_info{};
    Shader::Backend::Bindings binding{};
    Shader::StageSpecialization spec{};
    std::vector<u32> code;
    std::array<u32, Shader::NUM_USER_DATA_REGS> user_data_copy{};
    u32 user_data_size{};
    std::vector<u32> runtime_vs_copy;
    std::optional<std::vector<u32>> patch;
    bool is_patched{};
    VideoCore::SrtCaptureProvenance provenance{};
    std::optional<Shader::IR::Program> ir_program;

    void CaptureOwnedInputs(Shader::Stage stage, Shader::LogicalStage logical_stage,
                            const Shader::ShaderParams& params,
                            const Shader::RuntimeInfo& runtime,
                            Shader::Backend::Bindings initial_binding) {
        code.assign(params.code.begin(), params.code.end());
        user_data_size = static_cast<u32>(params.user_data.size());
        std::ranges::copy(params.user_data, user_data_copy.begin());
        // Constructing Info from the original params preserves the guest program base. Rebinding
        // user_data below must not derive that base from the host-owned code copy.
        info = Shader::Info(stage, logical_stage, params);
        runtime_info = runtime;
        binding = initial_binding;
        if (runtime.stage == Shader::Stage::Geometry) {
            runtime_vs_copy.assign(runtime.gs_info.vs_copy.begin(), runtime.gs_info.vs_copy.end());
        }
        RebindOwnedViews();
    }

    void RebindOwnedViews() noexcept {
        info.user_data = std::span{user_data_copy}.first(user_data_size);
        info.ClearCompileCaptureReader();
        info.ClearCompileFetchShaderSnapshot();
        if (runtime_info.stage == Shader::Stage::Geometry) {
            runtime_info.gs_info.vs_copy = runtime_vs_copy;
        }
        spec.info = &info;
        if (spec.runtime_info.stage == Shader::Stage::Geometry) {
            spec.runtime_info.gs_info.vs_copy = runtime_vs_copy;
        }
    }
};

struct AsyncShaderModuleResult {
    std::unique_ptr<AsyncShaderModuleJob> job;
    std::vector<u32> spv;
    vk::UniqueShaderModule module;
};

struct AsyncGraphicsPipelineKey {
    GraphicsPipelineKey pipeline{};
    u64 module_generation{};

    bool operator==(const AsyncGraphicsPipelineKey&) const noexcept = default;
};

struct AsyncGraphicsPipelineKeyHash {
    [[nodiscard]] size_t operator()(const AsyncGraphicsPipelineKey& key) const noexcept {
        return static_cast<size_t>(
            HashCombine(std::hash<GraphicsPipelineKey>{}(key.pipeline), key.module_generation));
    }
};

struct AsyncGraphicsPipelineJob {
    GraphicsPipelineKey key{};
    // Each Info copy owns its CompileResourceSnapshot. RebindCompileInfo only has to repair the
    // user-data span after this job has reached its final worker-thread address.
    std::array<Shader::Info, MaxShaderStages> info_copies{};
    std::array<std::array<u32, Shader::NUM_USER_DATA_REGS>, MaxShaderStages> user_data_copies{};
    std::array<u32, MaxShaderStages> user_data_sizes{};
    std::array<const Shader::Info*, MaxShaderStages> retained_infos{};
    std::array<const Shader::Info*, MaxShaderStages> compile_infos{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<std::vector<u32>, MaxShaderStages> runtime_vs_copies{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    u64 module_generation{};

    void CaptureRuntimeInfos(
        const std::array<Shader::RuntimeInfo, MaxShaderStages>& source_runtime_infos) {
        runtime_infos = source_runtime_infos;
        for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
            if (runtime_infos[stage].stage != Shader::Stage::Geometry) {
                runtime_vs_copies[stage].clear();
                continue;
            }
            const auto source_vs_copy = runtime_infos[stage].gs_info.vs_copy;
            runtime_vs_copies[stage].assign(source_vs_copy.begin(), source_vs_copy.end());
        }
        RebindCompileInfo();
    }

    void RebindCompileInfo() noexcept {
        for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
            if (runtime_infos[stage].stage == Shader::Stage::Geometry) {
                runtime_infos[stage].gs_info.vs_copy = runtime_vs_copies[stage];
            }
            if (!retained_infos[stage]) {
                compile_infos[stage] = nullptr;
                continue;
            }
            info_copies[stage].user_data =
                std::span{user_data_copies[stage]}.first(user_data_sizes[stage]);
            info_copies[stage].ClearCompileCaptureReader();
            info_copies[stage].ClearCompileFetchShaderSnapshot();
            compile_infos[stage] = &info_copies[stage];
        }
    }
};

struct AsyncGraphicsPipelineResult {
    std::unique_ptr<GraphicsPipeline> pipeline;
    GraphicsPipeline::SerializationSupport serialization;
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::array<const Shader::Info*, MaxShaderStages> retained_infos{};
    u64 module_generation{};
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
    bool LoadPipelineStage(Serialization::Archive& ar, size_t output_slot,
                           u64 expected_pipeline_permutation_id,
                           Shader::LogicalStage expected_logical_stage);
    void ResetPipelineLoadScratch();

    const GraphicsPipeline* GetGraphicsPipeline();

    const ComputePipeline* GetComputePipeline();

    /** Marks the generic PM4 compute-dispatch boundary for optional selection diagnostics. */
    void BeginComputeDispatchTrace() noexcept;

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule,
                              std::optional<Shader::Gcn::FetchShaderData>, u64>;
    Result GetProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<u64> permutation_id = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    using AsyncGraphicsPipelineQueue =
        AsyncCompilationQueue<AsyncGraphicsPipelineKey, AsyncGraphicsPipelineResult,
                              AsyncGraphicsPipelineJob, AsyncGraphicsPipelineKeyHash>;
    using AsyncShaderModuleQueue =
        AsyncCompilationQueue<AsyncShaderModuleKey, AsyncShaderModuleResult,
                              std::unique_ptr<AsyncShaderModuleJob>, AsyncShaderModuleKeyHash>;

    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();
    AsyncGraphicsPipelineResult CompileGraphicsPipelineAsync(AsyncGraphicsPipelineJob job);
    AsyncGraphicsPipelineJob MakeAsyncGraphicsPipelineJob() const;
    void PublishAsyncGraphicsPipeline(AsyncGraphicsPipelineQueue::Completion completion);
    void PublishAsyncGraphicsPipelines();
    AsyncShaderModuleResult CompileShaderModuleAsync(std::unique_ptr<AsyncShaderModuleJob> job);
    void PublishAsyncShaderModule(AsyncShaderModuleQueue::Completion completion);
    void PublishAsyncShaderModules();
    bool SubmitAsyncShaderModule(std::unique_ptr<AsyncShaderModuleJob> job);
    std::optional<vk::ShaderModule> ReplaceShaderOnGpuOwner(vk::ShaderModule module,
                                                            std::span<const u32> spv_code);

    void DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage, std::string_view ext,
                    std::optional<u64> permutation_id = {});
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::Stage stage,
                                                   u64 permutation_id, std::string_view ext);
    std::optional<Shader::IR::Program> TranslateModule(
        Shader::Pools& translation_pools, Shader::Info& info,
        Shader::RuntimeInfo& runtime_info, const std::span<const u32>& code, size_t perm_idx,
        const Shader::CompileResourceSnapshot* compile_resources = nullptr,
        const Shader::Gcn::FetchShaderData* compile_fetch_shader = nullptr,
        Shader::SrtMemoryReader compile_reader = {});
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code,
                                   const Shader::IR::Program& ir_program,
                                   Shader::Backend::Bindings& binding, u64 permutation_id);
    Result GetProgramAttempt(Shader::Stage stage, Shader::LogicalStage l_stage,
                             const Shader::ShaderParams& params,
                             Shader::Backend::Bindings& binding, SrtReadSession& srt_read_session,
                             bool& pending);
    void RecordShaderSelection(Shader::Stage stage, Shader::LogicalStage l_stage,
                               u64 program_hash,
                               const VideoCore::SrtCaptureProvenance& provenance,
                               u64 semantic_identity,
                               VideoCore::ShaderSelectionCacheAction cache_action,
                               size_t module_count, u64 selected_permutation_id) noexcept;
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
    tsl::robin_map<Shader::ProgramCacheKey, std::unique_ptr<Program>,
                   Shader::ProgramCacheKeyHash>
        program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    VideoCore::ShaderSelectionTrace shader_selection_trace;
    std::atomic<u64> compute_dispatch_sequence{};
    std::optional<VideoCore::ShaderSelectionTraceRecord> pending_compute_selection_trace;

    std::unique_ptr<AsyncGraphicsPipelineQueue> async_graphics_compiler;
    std::unordered_set<GraphicsPipelineKey> failed_graphics_pipelines;
    std::unique_ptr<AsyncShaderModuleQueue> async_shader_compiler;
    std::unordered_set<AsyncShaderModuleKey, AsyncShaderModuleKeyHash> pending_shader_modules;
    u64 next_program_generation{1};
    u64 module_generation{1};
    std::vector<vk::ShaderModule> retired_shader_modules;

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

} // namespace Vulkan
