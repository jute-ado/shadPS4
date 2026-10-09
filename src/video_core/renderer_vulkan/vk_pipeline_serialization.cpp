// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/scope_exit.h"
#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/specialization_identity.h"
#include "shader_recompiler/specialization_serialization.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache_serialization.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(PipelineKeyVersion);
    pldata.Write(u32{1}); // compute

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("c_{:#018x}", key.value), ar.TakeOff());
}

void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(PipelineKeyVersion);
    pldata.Write(u32{0}); // graphics

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("g_{:#018x}", hash), ar.TakeOff());
}

bool RegisterShaderMeta(const Shader::Info& info, const Shader::StageSpecialization& spec,
                        u64 permutation_id) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return false;
    }

    Serialization::Archive ar;
    if (!Shader::SerializeShaderCacheMetadata(ar, info, spec, permutation_id)) {
        return false;
    }

    return Storage::DataBase::Instance().Save(
        Storage::BlobType::ShaderMeta, fmt::format("{:#018x}", permutation_id), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 permutation_id) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}", permutation_id), std::move(spv));
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    return key.Read(value) && key.IsValid();
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    // Nothing here yet
    return;
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    // Nothing here yet
    return true;
}

void PipelineCache::ResetPipelineLoadScratch() {
    infos.fill(nullptr);
    runtime_infos.fill({});
    modules.fill(nullptr);
    fetch_shader.reset();
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    ResetPipelineLoadScratch();
    std::vector<PendingProgramPermutation> pending_programs;
    std::vector<vk::ShaderModule> created_modules;
    bool remove_pipeline_on_failure{};
    SCOPE_EXIT {
        if (remove_pipeline_on_failure) {
            compute_pipelines.erase(compute_key);
        }
        ResetPipelineLoadScratch();
        for (const auto module : created_modules) {
            instance.GetDevice().destroyShaderModule(module);
        }
    };
    if (!compute_key.Deserialize(ar)) {
        return false;
    }

    ComputePipeline::SerializationSupport sdata{};
    if (!sdata.Deserialize(ar) || !ar.IsEoS() || compute_key.value == 0 ||
        compute_pipelines.contains(compute_key)) {
        return false;
    }

    std::vector<u8> meta_blob;
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", compute_key.value), meta_blob);
    if (meta_blob.empty()) {
        return false;
    }

    Serialization::Archive meta_ar{std::move(meta_blob)};

    if (!LoadPipelineStage(meta_ar, 0, compute_key.value, Shader::LogicalStage::Compute,
                           pending_programs, created_modules)) {
        return false;
    }

    auto pipeline =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *infos[0], modules[0], sdata, true);
    if (!pipeline->IsValid()) {
        return false;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (!is_new) {
        return false;
    }
    it.value() = std::move(pipeline);
    remove_pipeline_on_failure = true;
    if (!PublishPreloadedPrograms(program_cache, pending_programs)) {
        return false;
    }
    created_modules.clear();
    remove_pipeline_on_failure = false;

    return true;
}

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    ResetPipelineLoadScratch();
    std::vector<PendingProgramPermutation> pending_programs;
    std::vector<vk::ShaderModule> created_modules;
    bool remove_pipeline_on_failure{};
    SCOPE_EXIT {
        if (remove_pipeline_on_failure) {
            graphics_pipelines.erase(graphics_key);
        }
        ResetPipelineLoadScratch();
        for (const auto module : created_modules) {
            instance.GetDevice().destroyShaderModule(module);
        }
    };
    GraphicsPipeline::SerializationSupport sdata{};
    if (!DeserializeGraphicsPipelineCacheRecord(ar, graphics_key, sdata)) {
        return false;
    }
    if (graphics_pipelines.contains(graphics_key)) {
        return false;
    }

    for (int stage_idx = 0; stage_idx < MaxShaderStages; ++stage_idx) {
        const auto& hash = graphics_key.stage_hashes[stage_idx];
        if (!hash) {
            continue;
        }

        std::vector<u8> meta_blob;
        Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                           fmt::format("{:#018x}", hash), meta_blob);
        if (meta_blob.empty()) {
            return false;
        }

        Serialization::Archive meta_ar{std::move(meta_blob)};

        if (!LoadPipelineStage(meta_ar, stage_idx, hash,
                               static_cast<Shader::LogicalStage>(stage_idx), pending_programs,
                               created_modules)) {
            return false;
        }
    }

    const auto tcs_slot = static_cast<u32>(Shader::LogicalStage::TessellationControl);
    const auto tes_slot = static_cast<u32>(Shader::LogicalStage::TessellationEval);
    const auto fragment_slot = static_cast<u32>(Shader::LogicalStage::Fragment);
    const bool is_aux_tess_primitive = graphics_key.prim_type == AmdGpu::PrimitiveType::RectList ||
                                       graphics_key.prim_type == AmdGpu::PrimitiveType::QuadList;
    const bool needs_aux_tess = is_aux_tess_primitive && !graphics_key.stage_hashes[tcs_slot] &&
                                !graphics_key.stage_hashes[tes_slot];
    if (needs_aux_tess != (!sdata.tcs.empty() && !sdata.tes.empty()) ||
        (!needs_aux_tess && (!sdata.tcs.empty() || !sdata.tes.empty())) ||
        (graphics_key.stage_hashes[fragment_slot] && !sdata.fragment.empty())) {
        return false;
    }

    auto pipeline = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, fetch_shader, modules, sdata, true);
    if (!pipeline->IsValid()) {
        return false;
    }
    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (!is_new) {
        return false;
    }
    it.value() = std::move(pipeline);
    remove_pipeline_on_failure = true;
    if (!PublishPreloadedPrograms(program_cache, pending_programs)) {
        return false;
    }
    created_modules.clear();
    remove_pipeline_on_failure = false;

    return true;
}

bool PipelineCache::LoadPipelineStage(Serialization::Archive& ar, size_t output_slot,
                                      u64 expected_pipeline_permutation_id,
                                      Shader::LogicalStage expected_logical_stage,
                                      std::vector<PendingProgramPermutation>& pending_programs,
                                      std::vector<vk::ShaderModule>& created_modules) {
    if (output_slot >= infos.size() ||
        static_cast<u32>(expected_logical_stage) >=
            static_cast<u32>(Shader::LogicalStage::NumLogicalStages)) {
        return false;
    }
    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    spec.info = &program->info;
    std::optional<Shader::Gcn::FetchShaderData> stage_fetch_shader;
    u64 permutation_id{};
    if (!Shader::DeserializeShaderCacheMetadata(ar, program->info, stage_fetch_shader, spec,
                                                permutation_id)) {
        return false;
    }

    const u64 expected_permutation_id = Shader::ComputeShaderPermutationIdentity(
        program->info.pgm_hash, program->info.l_stage, spec);
    if (permutation_id != expected_permutation_id) {
        LOG_WARNING(Render_Vulkan,
                    "Cached permutation identity {:#x} of {}_{:x} does not match its metadata "
                    "({:#x}), skipping preload",
                    permutation_id, program->info.stage, program->info.pgm_hash,
                    expected_permutation_id);
        return false;
    }
    if (permutation_id != expected_pipeline_permutation_id) {
        LOG_WARNING(Render_Vulkan,
                    "Cached permutation identity {:#x} does not match pipeline stage key {:#x}, "
                    "skipping preload",
                    permutation_id, expected_pipeline_permutation_id);
        return false;
    }
    if (program->info.l_stage != expected_logical_stage) {
        LOG_WARNING(Render_Vulkan,
                    "Cached permutation identity {:#x} belongs to logical stage {} instead of "
                    "pipeline stage {}, skipping preload",
                    permutation_id, static_cast<u32>(program->info.l_stage),
                    static_cast<u32>(expected_logical_stage));
        return false;
    }
    if (spec.runtime_info.stage != program->info.stage) {
        LOG_WARNING(Render_Vulkan,
                    "Cached permutation identity {:#x} has runtime physical stage {} instead of "
                    "metadata stage {}, skipping preload",
                    permutation_id, static_cast<u32>(spec.runtime_info.stage),
                    static_cast<u32>(program->info.stage));
        return false;
    }
    if (!Shader::IsPhysicalStageCompatible(program->info.stage, program->info.l_stage)) {
        LOG_WARNING(Render_Vulkan,
                    "Cached permutation identity {:#x} has incompatible physical stage {} and "
                    "logical stage {}, skipping preload",
                    permutation_id, static_cast<u32>(program->info.stage),
                    static_cast<u32>(program->info.l_stage));
        return false;
    }
    auto module_info = program->info;

    std::vector<u32> spv{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}", permutation_id), spv);
    if (!IsValidCachedSpirv(spv)) {
        return false;
    }

    const Shader::ProgramCacheKey cache_key{
        .program_hash = program->info.pgm_hash,
        .stage = program->info.stage,
        .logical_stage = program->info.l_stage,
    };
    auto it_pgm = program_cache.find(cache_key);
    const bool new_program = it_pgm == program_cache.end();
    Program* cached_program = new_program ? program.get() : it_pgm.value().get();
    const auto preload = cached_program->InspectPreloadPermutation(spec, permutation_id);
    if (preload.action == Program::PreloadAction::Conflict) {
        LOG_WARNING(Render_Vulkan,
                    "Cached permutation identity {:#x} of {}_{:x} conflicts with an already "
                    "loaded permutation, skipping preload",
                    permutation_id, cached_program->info.stage, cached_program->info.pgm_hash);
        return false;
    }

    const Program::Module* selected{};
    if (preload.action == Program::PreloadAction::Reuse) {
        if (!preload.module) {
            return false;
        }
        selected = preload.module;
    } else {
        const auto compiled_module = TryCompileSPV(spv, instance.GetDevice());
        if (!compiled_module) {
            return false;
        }
        const vk::ShaderModule module = *compiled_module;
        bool tracked{};
        SCOPE_EXIT {
            if (!tracked) {
                instance.GetDevice().destroyShaderModule(module);
            }
        };
        created_modules.push_back(module);
        tracked = true;
        pending_programs.push_back(PendingProgramPermutation{
            .cache_key = cache_key,
            .existing_program = new_program ? nullptr : cached_program,
            .new_program = new_program ? std::move(program) : nullptr,
            .module =
                Program::Module{module, std::move(spec), std::move(module_info), permutation_id},
        });
        selected = &pending_programs.back().module;
        if (!selected->info) {
            return false;
        }
    }

    if (!selected || !selected->info || selected->permutation_id != permutation_id) {
        return false;
    }
    infos[output_slot] = selected->info.get();
    runtime_infos[output_slot] = selected->spec.runtime_info;
    modules[output_slot] = selected->module;
    if (selected->spec.fetch_shader_data) {
        fetch_shader = selected->spec.fetch_shader_data;
    }

    return true;
}

void PipelineCache::WarmUp() {
    if (!EmulatorSettings.IsPipelineCacheEnabled()) {
        return;
    }

    Storage::DataBase::Instance().Open();

    // Check if cache is compatible
    std::vector<u8> profile_data{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderProfile, "profile", profile_data);
    if (profile_data.empty()) {
        Storage::DataBase::Instance().FinishPreload();

        profile_data.resize(sizeof(profile));
        std::memcpy(profile_data.data(), &profile, sizeof(profile));
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(profile_data));
        return;
    }
    if (profile_data.size() != sizeof(Shader::Profile)) {
        LOG_WARNING(Render,
                    "Pipeline cache profile has unexpected size ({} != {}). Ignoring the cache",
                    profile_data.size(), sizeof(Shader::Profile));
        Storage::DataBase::Instance().Close();
        return;
    }

    Shader::Profile cached_profile{};
    std::memcpy(&cached_profile, profile_data.data(), sizeof(cached_profile));
    if (cached_profile != profile) {
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current system. Ignoring the cache");
        Storage::DataBase::Instance().Close();
        return;
    }

    u32 num_pipelines{};
    u32 num_total_pipelines{};

    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::PipelineKey, [&](std::vector<u8>&& data) {
            ++num_total_pipelines;

            try {
                Serialization::Archive ar{std::move(data)};
                Serialization::Reader pldata{ar};

                u32 version{};
                if (!pldata.Read(version) || version != PipelineKeyVersion) {
                    return;
                }

                u32 is_compute{};
                if (!pldata.Read(is_compute) || is_compute > 1) {
                    return;
                }

                bool result{};
                if (is_compute) {
                    result = LoadComputePipeline(ar);
                } else {
                    result = LoadGraphicsPipeline(ar);
                }

                if (result) {
                    ++num_pipelines;
                }
            } catch (const std::exception& exception) {
                LOG_WARNING(Render, "Ignoring malformed cached pipeline: {}", exception.what());
            } catch (...) {
                LOG_WARNING(Render, "Ignoring malformed cached pipeline with unknown error");
            }
        });

    LOG_INFO(Render, "Preloaded {} pipelines", num_pipelines);
    if (num_total_pipelines > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total_pipelines - num_pipelines);
    }

    Storage::DataBase::Instance().FinishPreload();
}

void PipelineCache::Sync() {
    Storage::DataBase::Instance().Close();
}

} // namespace Vulkan
