// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/serdes.h"
#include "common/scope_exit.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/specialization_identity.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
static constexpr u32 ShaderBinaryVersion = 7u;
static constexpr u32 ShaderMetaVersion = 11u;
static constexpr u32 PipelineKeyVersion = 5u;
} // namespace Serialization

namespace Vulkan {

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
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

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{0}); // graphics

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("g_{:#018x}", hash), ar.TakeOff());
}

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, u64 permutation_id) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion);
    meta.Write(Serialization::ShaderBinaryVersion);

    meta.Write(permutation_id);

    spec.Serialize(ar);
    info.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", permutation_id), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 permutation_id) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}", permutation_id), std::move(spv));
}

bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                    Shader::StageSpecialization& spec, u64& permutation_id) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if (meta_version != Serialization::ShaderMetaVersion) {
        return false;
    }

    u32 binary_version{};
    meta.Read(binary_version);
    if (binary_version != Serialization::ShaderBinaryVersion) {
        return false;
    }

    meta.Read(permutation_id);

    spec.Deserialize(ar);
    info.Deserialize(ar);

    fetch_shader_data = spec.fetch_shader_data;
    return true;
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    key.Read(value);
    return true;
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
    SCOPE_EXIT {
        ResetPipelineLoadScratch();
    };
    compute_key.Deserialize(ar);

    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    std::vector<u8> meta_blob;
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", compute_key.value), meta_blob);
    if (meta_blob.empty()) {
        return false;
    }

    Serialization::Archive meta_ar{std::move(meta_blob)};

    if (!LoadPipelineStage(meta_ar, 0, compute_key.value, Shader::LogicalStage::Compute)) {
        return false;
    }

    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    ASSERT(is_new);

    it.value() =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *infos[0], modules[0], sdata, true);

    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};

    key.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};

    key.Read(this, sizeof(*this));
    return true;
}

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    ResetPipelineLoadScratch();
    SCOPE_EXIT {
        ResetPipelineLoadScratch();
    };
    graphics_key.Deserialize(ar);
    if (graphics_key.stage_hashes[static_cast<u32>(Shader::LogicalStage::Compute)] != 0) {
        LOG_WARNING(Render_Vulkan,
                    "Cached graphics pipeline contains a compute-stage permutation, skipping "
                    "preload");
        return false;
    }

    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

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
                               static_cast<Shader::LogicalStage>(stage_idx))) {
            return false;
        }
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    ASSERT(is_new);

    it.value() = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, fetch_shader, modules, sdata, true);

    return true;
}

bool PipelineCache::LoadPipelineStage(Serialization::Archive& ar, size_t output_slot,
                                      u64 expected_pipeline_permutation_id,
                                      Shader::LogicalStage expected_logical_stage) {
    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    spec.info = &program->info;
    std::optional<Shader::Gcn::FetchShaderData> stage_fetch_shader;
    u64 permutation_id{};
    if (!LoadShaderMeta(ar, program->info, stage_fetch_shader, spec, permutation_id)) {
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
    if (spv.empty()) {
        return false;
    }

    const Shader::ProgramCacheKey cache_key{
        .program_hash = program->info.pgm_hash,
        .stage = program->info.stage,
        .logical_stage = program->info.l_stage,
    };
    auto [it_pgm, new_program] = program_cache.try_emplace(cache_key);
    if (new_program) {
        program->generation = next_program_generation++;
        it_pgm.value() = std::move(program);
    }

    auto& cached_program = *it_pgm.value();
    const auto preload = cached_program.InspectPreloadPermutation(spec, permutation_id);
    if (preload.action == Program::PreloadAction::Conflict) {
        LOG_WARNING(Render_Vulkan,
                    "Cached permutation identity {:#x} of {}_{:x} conflicts with an already "
                    "loaded permutation, skipping preload",
                    permutation_id, cached_program.info.stage, cached_program.info.pgm_hash);
        return false;
    }

    vk::ShaderModule module{};
    if (preload.action == Program::PreloadAction::Reuse) {
        ASSERT(preload.module);
        module = preload.module->module;
    } else {
        module = CompileSPV(spv, instance.GetDevice());
        if (!cached_program.InsertPreloadPermutation(module, std::move(spec),
                                                     std::move(module_info), permutation_id)) {
            instance.GetDevice().destroyShaderModule(module);
            return false;
        }
    }

    const auto selected = std::ranges::find_if(cached_program.modules, [&](const auto& candidate) {
        return candidate.info && candidate.permutation_id == permutation_id;
    });
    ASSERT(selected != cached_program.modules.end());
    infos[output_slot] = selected->info.get();
    runtime_infos[output_slot] = selected->spec.runtime_info;
    modules[output_slot] = module;
    if (stage_fetch_shader) {
        fetch_shader = std::move(stage_fetch_shader);
    }

    return true;
}

void PipelineCache::WarmUp() {
    if (!EmulatorSettings.IsPipelineCacheEnabled() || EmulatorSettings.IsPatchShaders()) {
        if (EmulatorSettings.IsPatchShaders()) {
            LOG_WARNING(Render,
                        "Persistent pipeline caching is disabled while shader patches are enabled");
        }
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

            Serialization::Archive ar{std::move(data)};
            Serialization::Reader pldata{ar};

            u32 version{};
            pldata.Read(version);
            if (version != Serialization::PipelineKeyVersion) {
                return;
            }

            u32 is_compute{};
            pldata.Read(is_compute);

            bool result{};
            if (is_compute) {
                result = LoadComputePipeline(ar);
            } else {
                result = LoadGraphicsPipeline(ar);
            }

            if (result) {
                ++num_pipelines;
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

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);

    return srt_info.Deserialize(ar);
}

} // namespace Shader
