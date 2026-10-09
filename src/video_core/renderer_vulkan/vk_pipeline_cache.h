// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <memory>
#include <variant>
#include <tsl/robin_map.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "shader_recompiler/specialization_identity.h"
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
        u64 permutation_id{};

        Module() = default;
        Module(vk::ShaderModule module_, Shader::StageSpecialization&& spec_, Shader::Info&& info_,
               u64 permutation_id_)
            : module{module_}, info{std::make_unique<Shader::Info>(std::move(info_))},
              spec{std::move(spec_)}, permutation_id{permutation_id_} {
            spec.info = info.get();
        }

        Module(Module&&) = default;
        Module& operator=(Module&&) = default;
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

    Program() = default;
    Program(Shader::Stage stage, Shader::LogicalStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec, Shader::Info&& info,
                   u64 permutation_id) {
        modules.emplace_back(module, std::move(spec), std::move(info), permutation_id);
    }

    [[nodiscard]] PreloadDecision InspectPreloadPermutation(const Shader::StageSpecialization& spec,
                                                            u64 permutation_id) const {
        const auto identity_match = std::ranges::find_if(modules, [&](const Module& candidate) {
            return candidate.info && candidate.permutation_id == permutation_id;
        });
        if (identity_match != modules.end()) {
            return identity_match->spec == spec
                       ? PreloadDecision{.action = PreloadAction::Reuse, .module = &*identity_match}
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
        AddPermut(module, std::move(spec), std::move(info), permutation_id);
        return true;
    }
};

using ProgramCacheMap =
    tsl::robin_map<Shader::ProgramCacheKey, std::unique_ptr<Program>, Shader::ProgramCacheKeyHash>;

/**
 * One shader permutation decoded and compiled during cache preload, but not yet made visible
 *
 * in the program cache. The module handle remains owned by the caller until the whole pipeline
 *
 * record has been accepted and this entry is published.
 */
struct PendingProgramPermutation {
    Shader::ProgramCacheKey cache_key{};
    Program* existing_program{};
    std::unique_ptr<Program> new_program{};
    Program::Module module{};
};

[[nodiscard]] inline bool PublishPreloadedPrograms(ProgramCacheMap& programs,
                                                   std::span<PendingProgramPermutation> pending) {
    const auto info_matches_key = [](const Shader::Info& info, const Shader::ProgramCacheKey& key) {
        return info.pgm_hash == key.program_hash && info.stage == key.stage &&
               info.l_stage == key.logical_stage;
    };

    size_t new_program_count{};
    for (size_t index = 0; index < pending.size(); ++index) {
        auto& entry = pending[index];
        if (!entry.module.info || entry.module.spec.info != entry.module.info.get() ||
            !info_matches_key(*entry.module.info, entry.cache_key) ||
            entry.module.spec.runtime_info.stage != entry.cache_key.stage ||
            entry.module.permutation_id == 0) {
            return false;
        }
        if (std::ranges::any_of(pending.first(index), [&](const auto& prior) {
                return prior.cache_key == entry.cache_key;
            })) {
            return false;
        }

        const auto found = programs.find(entry.cache_key);
        if (entry.new_program) {
            if (entry.existing_program || found != programs.end() ||
                !entry.new_program->modules.empty() ||
                !info_matches_key(entry.new_program->info, entry.cache_key)) {
                return false;
            }
            ++new_program_count;
        } else if (!entry.existing_program || found == programs.end() ||
                   found.value().get() != entry.existing_program ||
                   entry.existing_program
                           ->InspectPreloadPermutation(entry.module.spec,
                                                       entry.module.permutation_id)
                           .action != Program::PreloadAction::Insert) {
            return false;
        }
    }

    // Complete every potentially-throwing allocation before changing either cache or Program.
    programs.reserve(programs.size() + new_program_count);
    for (auto& entry : pending) {
        if (entry.existing_program) {
            entry.existing_program->modules.reserve(entry.existing_program->modules.size() + 1);
        } else {
            entry.new_program->modules.reserve(1);
        }
    }

    std::vector<Shader::ProgramCacheKey> inserted_keys;
    std::vector<Program*> extended_programs;
    inserted_keys.reserve(new_program_count);
    extended_programs.reserve(pending.size() - new_program_count);
    try {
        for (auto& entry : pending) {
            if (entry.existing_program) {
                continue;
            }
            entry.new_program->modules.emplace_back(std::move(entry.module));
            auto [inserted, is_new] =
                programs.try_emplace(entry.cache_key, std::move(entry.new_program));
            if (!is_new) {
                for (const auto& key : inserted_keys) {
                    programs.erase(key);
                }
                return false;
            }
            entry.existing_program = inserted.value().get();
            inserted_keys.push_back(entry.cache_key);
        }
        for (auto& entry : pending) {
            if (entry.module.info) {
                entry.existing_program->modules.emplace_back(std::move(entry.module));
                extended_programs.push_back(entry.existing_program);
            }
        }
    } catch (...) {
        for (auto* program : extended_programs) {
            program->modules.pop_back();
        }
        for (const auto& key : inserted_keys) {
            programs.erase(key);
        }
        return false;
    }
    return true;
}

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
    bool LoadPipelineStage(Serialization::Archive& ar, size_t output_slot,
                           u64 expected_pipeline_permutation_id,
                           Shader::LogicalStage expected_logical_stage,
                           std::vector<PendingProgramPermutation>& pending_programs,
                           std::vector<vk::ShaderModule>& created_modules);

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
    void ResetPipelineLoadScratch();
    AsyncGraphicsPipelineResult CompileGraphicsPipelineAsync(AsyncGraphicsPipelineJob job);
    AsyncGraphicsPipelineJob MakeAsyncGraphicsPipelineJob() const;
    void PublishAsyncGraphicsPipeline(AsyncGraphicsPipelineQueue::Completion completion);
    bool PublishAsyncGraphicsPipelines(const GraphicsPipelineKey* requested_key = nullptr);
    const GraphicsPipeline* WaitForAsyncGraphicsPipeline(const GraphicsPipelineKey& key);

    void DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::Stage stage, size_t perm_idx,
                                                   std::string_view ext);
    std::optional<Shader::IR::Program> TranslateModule(Shader::Info& info,
                                                       Shader::RuntimeInfo& runtime_info,
                                                       const std::span<const u32>& code,
                                                       Shader::SrtMemoryReader reader);
    vk::ShaderModule CompileModule(Shader::Info& info, const std::span<const u32>& code,
                                   size_t perm_idx, const std::vector<u32>& spv);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::Stage stage, Shader::LogicalStage l_stage,
                                                Shader::SrtMemoryReader reader = {});

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
    ProgramCacheMap program_cache;
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
