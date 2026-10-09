// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bitset>
#include <utility>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/profile.h"

namespace Shader {

struct VsAttribSpecialization {
    u32 divisor{};
    AmdGpu::NumberClass num_class{};
    AmdGpu::CompMapping dst_select{};

    bool operator==(const VsAttribSpecialization&) const = default;
};

struct BufferSpecialization {
    u32 stride : 14;
    u32 is_storage : 1;
    u32 is_formatted : 1;
    u32 swizzle_enable : 1;
    u32 data_format : 6;
    u32 num_format : 4;
    u32 index_stride : 2;
    u32 element_size : 2;
    u32 add_tid_enable : 1;
    AmdGpu::CompMapping dst_select{};
    AmdGpu::NumberConversion num_conversion{};

    bool operator==(const BufferSpecialization& other) const {
        return stride == other.stride && is_storage == other.is_storage &&
               is_formatted == other.is_formatted && swizzle_enable == other.swizzle_enable &&
               add_tid_enable == other.add_tid_enable &&
               (!is_formatted ||
                (data_format == other.data_format && num_format == other.num_format &&
                 dst_select == other.dst_select && num_conversion == other.num_conversion)) &&
               (!swizzle_enable ||
                (index_stride == other.index_stride && element_size == other.element_size));
    }
};

struct ImageSpecialization {
    AmdGpu::ImageType type = AmdGpu::ImageType::Color2D;
    AmdGpu::NumberClass number_class = AmdGpu::NumberClass::Float;
    bool is_storage = false;
    bool is_cube = false;
    bool is_srgb = false;
    AmdGpu::CompMapping dst_select{};
    AmdGpu::NumberConversion num_conversion{};
    // Atomic storage images encode an exact SPIR-V image format. NumberClass alone cannot
    // distinguish formats such as R16f and R32f, which compile to different module types.
    AmdGpu::DataFormat atomic_data_format = AmdGpu::DataFormat::FormatInvalid;
    AmdGpu::NumberFormat atomic_number_format = AmdGpu::NumberFormat::Unorm;
    u32 num_bindings = 0;

    bool operator==(const ImageSpecialization&) const = default;
};

struct FMaskSpecialization {
    u32 width;
    u32 height;

    bool operator==(const FMaskSpecialization&) const = default;
};

struct SamplerSpecialization {
    u8 force_unnormalized : 1;
    u8 force_degamma : 1;

    bool operator==(const SamplerSpecialization&) const = default;
};

/**
 * Alongside runtime information, this structure also checks bound resources
 * for compatibility. Can be used as a key for storing shader permutations.
 * Is separate from runtime information, because resource layout can only be deduced
 * after the first compilation of a module.
 */
struct StageSpecialization {
    static constexpr size_t MaxStageResources = 128;

    const Info* info{};
    RuntimeInfo runtime_info{};
    // RuntimeInfo contains a view of the geometry copy shader. Keep the pointed-to words owned by
    // the specialization so copies, moves, module retention and disk preloading cannot leave a
    // dangling span.
    std::vector<u32> owned_runtime_vs_copy;
    std::bitset<MaxStageResources> bitset{};
    std::optional<Gcn::FetchShaderData> fetch_shader_data{};
    boost::container::small_vector<VsAttribSpecialization, 32> vs_attribs;
    boost::container::small_vector<BufferSpecialization, 16> buffers;
    boost::container::small_vector<ImageSpecialization, 16> images;
    boost::container::small_vector<FMaskSpecialization, 8> fmasks;
    boost::container::small_vector<SamplerSpecialization, 16> samplers;
    Backend::Bindings start{};
    u32 user_data_mask{};

    StageSpecialization() = default;
    StageSpecialization(const StageSpecialization& other) {
        *this = other;
    }
    StageSpecialization& operator=(const StageSpecialization& other) {
        if (this == &other) {
            return *this;
        }
        info = other.info;
        runtime_info = other.runtime_info;
        owned_runtime_vs_copy = other.owned_runtime_vs_copy;
        if (runtime_info.stage == Stage::Geometry && owned_runtime_vs_copy.empty() &&
            !other.runtime_info.gs_info.vs_copy.empty()) {
            owned_runtime_vs_copy.assign(other.runtime_info.gs_info.vs_copy.begin(),
                                         other.runtime_info.gs_info.vs_copy.end());
        }
        bitset = other.bitset;
        fetch_shader_data = other.fetch_shader_data;
        vs_attribs = other.vs_attribs;
        buffers = other.buffers;
        images = other.images;
        fmasks = other.fmasks;
        samplers = other.samplers;
        start = other.start;
        user_data_mask = other.user_data_mask;
        RebindOwnedRuntimeInfo();
        return *this;
    }
    StageSpecialization(StageSpecialization&& other) {
        *this = std::move(other);
    }
    StageSpecialization& operator=(StageSpecialization&& other) {
        if (this == &other) {
            return *this;
        }
        info = other.info;
        runtime_info = other.runtime_info;
        if (runtime_info.stage == Stage::Geometry && other.owned_runtime_vs_copy.empty() &&
            !other.runtime_info.gs_info.vs_copy.empty()) {
            owned_runtime_vs_copy.assign(other.runtime_info.gs_info.vs_copy.begin(),
                                         other.runtime_info.gs_info.vs_copy.end());
        } else {
            owned_runtime_vs_copy = std::move(other.owned_runtime_vs_copy);
        }
        bitset = other.bitset;
        fetch_shader_data = std::move(other.fetch_shader_data);
        vs_attribs = std::move(other.vs_attribs);
        buffers = std::move(other.buffers);
        images = std::move(other.images);
        fmasks = std::move(other.fmasks);
        samplers = std::move(other.samplers);
        start = other.start;
        user_data_mask = other.user_data_mask;
        RebindOwnedRuntimeInfo();
        return *this;
    }
    StageSpecialization(const Info& info_, RuntimeInfo runtime_info_, const Profile& profile_,
                        Backend::Bindings start_)
        : info{&info_}, runtime_info{runtime_info_}, start{start_},
          user_data_mask{info_.ud_mask.mask} {
        OwnRuntimeInfoViews();
        fetch_shader_data = Gcn::ParseFetchShader(info_);
        if (info_.stage == Stage::Vertex && fetch_shader_data) {
            // Specialize shader on VS input number types to follow spec.
            ForEachSharp(vs_attribs, fetch_shader_data->attributes,
                         [this](auto& spec, const auto& desc, AmdGpu::Buffer sharp) {
                             using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
                             if (const auto step_rate = desc.GetStepRate();
                                 step_rate != InstanceIdType::None) {
                                 spec.divisor = step_rate == InstanceIdType::OverStepRate0
                                                    ? runtime_info.vs_info.step_rate_0
                                                    : (step_rate == InstanceIdType::OverStepRate1
                                                           ? runtime_info.vs_info.step_rate_1
                                                           : 1);
                             }
                             spec.num_class = AmdGpu::GetNumberClass(sharp.GetNumberFmt());
                             spec.dst_select = sharp.DstSelect();
                         });
        }
        u32 binding{};
        ForEachSharp(binding, buffers, info->buffers,
                     [](auto& spec, const auto& desc, AmdGpu::Buffer sharp) {
                         spec.stride = sharp.GetStride();
                         spec.is_storage = desc.IsStorage(sharp);
                         spec.is_formatted = desc.is_formatted;
                         spec.swizzle_enable = sharp.swizzle_enable;
                         spec.add_tid_enable = sharp.add_tid_enable;
                         if (spec.is_formatted) {
                             spec.data_format = static_cast<u32>(sharp.GetDataFmt());
                             spec.num_format = static_cast<u32>(sharp.GetNumberFmt());
                             spec.dst_select = sharp.DstSelect();
                             spec.num_conversion = sharp.GetNumberConversion();
                         }
                         if (spec.swizzle_enable) {
                             spec.index_stride = sharp.index_stride;
                             spec.element_size = sharp.element_size;
                         }
                     });
        ForEachSharp(binding, images, info->images,
                     [&](auto& spec, const auto& desc, AmdGpu::Image sharp) {
                         spec.type = sharp.GetViewType(desc.is_array);
                         spec.number_class = AmdGpu::GetNumberClass(sharp.GetNumberFmt());
                         spec.is_storage = desc.is_written;
                         spec.is_cube = sharp.IsCube();
                         if (spec.is_storage) {
                             spec.dst_select = sharp.DstSelect();
                         } else {
                             spec.is_srgb = sharp.GetNumberFmt() == AmdGpu::NumberFormat::Srgb;
                         }
                         spec.num_conversion = sharp.GetNumberConversion();
                         if (desc.is_atomic) {
                             spec.atomic_data_format = sharp.GetDataFmt();
                             spec.atomic_number_format = sharp.GetNumberFmt();
                         }
                         spec.num_bindings = desc.NumBindings(*info);
                     });
        ForEachSharp(binding, fmasks, info->fmasks,
                     [](auto& spec, const auto& desc, AmdGpu::Image sharp) {
                         spec.width = sharp.width;
                         spec.height = sharp.height;
                     });
        ForEachSharp(samplers, info->samplers,
                     [](auto& spec, const auto& desc, AmdGpu::Sampler sharp) {
                         spec.force_unnormalized = sharp.force_unnormalized;
                         spec.force_degamma = sharp.force_degamma;
                     });

        // Initialize runtime_info fields that rely on analysis in tessellation passes
        if (info->l_stage == LogicalStage::TessellationControl ||
            info->l_stage == LogicalStage::TessellationEval) {
            TessellationDataConstantBuffer tess_constants{};
            info->ReadTessConstantBuffer(tess_constants);
            runtime_info.InitFromTessConstants(tess_constants);
        }
    }

    void ForEachSharp(auto& spec_list, auto& desc_list, auto&& func) {
        for (const auto& desc : desc_list) {
            auto& spec = spec_list.emplace_back();
            const auto sharp = desc.GetSharp(*info);
            if (!sharp) {
                continue;
            }
            func(spec, desc, sharp);
        }
    }

    void ForEachSharp(u32& binding, auto& spec_list, auto& desc_list, auto&& func) {
        for (const auto& desc : desc_list) {
            auto& spec = spec_list.emplace_back();
            const auto sharp = desc.GetSharp(*info);
            if (sharp) {
                bitset.set(binding);
            }
            ++binding;
            // Descriptor layout and format fields can affect emitted SPIR-V even when the current
            // guest binding is null. Normalize them from the descriptor/default SHARP while the
            // bitset retains the observed runtime boundness for diagnostics and persistence only.
            func(spec, desc, sharp);
        }
    }

    [[nodiscard]] bool Valid() const {
        return info != nullptr;
    }

    bool operator==(const StageSpecialization& other) const {
        if (!Valid() || !other.Valid()) {
            return false;
        }
        if (vs_attribs != other.vs_attribs || runtime_info != other.runtime_info ||
            fetch_shader_data != other.fetch_shader_data || fmasks != other.fmasks ||
            user_data_mask != other.user_data_mask || buffers != other.buffers ||
            images != other.images || samplers != other.samplers) {
            return false;
        }
        if ((!buffers.empty() || !images.empty() || !samplers.empty()) &&
            start.unified != other.start.unified) {
            return false;
        }
        if (!buffers.empty() && start.buffer != other.start.buffer) {
            return false;
        }
        return user_data_mask == 0 || start.user_data == other.start.user_data;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);

    void RebindOwnedRuntimeInfo() noexcept {
        if (runtime_info.stage == Stage::Geometry) {
            runtime_info.gs_info.vs_copy = owned_runtime_vs_copy;
        }
    }

private:
    void OwnRuntimeInfoViews() {
        if (runtime_info.stage != Stage::Geometry || runtime_info.gs_info.vs_copy.empty()) {
            return;
        }
        owned_runtime_vs_copy.assign(runtime_info.gs_info.vs_copy.begin(),
                                     runtime_info.gs_info.vs_copy.end());
        RebindOwnedRuntimeInfo();
    }
};

} // namespace Shader
