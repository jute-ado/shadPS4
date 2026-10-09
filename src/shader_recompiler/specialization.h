// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bitset>

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
    AmdGpu::CompMapping dst_select{};
    AmdGpu::NumberConversion num_conversion{};

    bool operator==(const BufferSpecialization& other) const {
        return stride == other.stride && is_storage == other.is_storage &&
               is_formatted == other.is_formatted && swizzle_enable == other.swizzle_enable &&
               (!is_formatted ||
                (data_format == other.data_format && num_format == other.num_format &&
                 dst_select == other.dst_select && num_conversion == other.num_conversion)) &&
               (!swizzle_enable ||
                (index_stride == other.index_stride && element_size == other.element_size));
    }
};

struct ImageSpecialization {
    AmdGpu::ImageType type = AmdGpu::ImageType::Color2D;
    bool is_integer = false;
    bool is_storage = false;
    bool is_cube = false;
    bool is_srgb = false;
    AmdGpu::CompMapping dst_select{};
    AmdGpu::NumberConversion num_conversion{};
    // FIXME any pipeline cache changes needed?
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

// Immutable draw-local values of every shader resource descriptor. Pipeline specialization and
// descriptor binding must consume the same snapshot; reading guest descriptor memory independently
// in those phases can combine two guest generations in one draw.
struct StageResourceSnapshot {
    static constexpr size_t MaxBuffers = NUM_BUFFERS;
    static constexpr size_t MaxImages = NUM_IMAGES;
    static constexpr size_t MaxSamplers = NUM_SAMPLERS;
    static constexpr size_t MaxFMasks = NUM_FMASKS;

    boost::container::static_vector<AmdGpu::Buffer, MaxBuffers> buffers;
    boost::container::static_vector<AmdGpu::Image, MaxImages> images;
    boost::container::static_vector<AmdGpu::Sampler, MaxSamplers> samplers;
    boost::container::static_vector<AmdGpu::Image, MaxFMasks> fmasks;
    bool valid{true};

    template <typename BufferReader, typename ImageReader, typename SamplerReader,
              typename FMaskReader>
    static StageResourceSnapshot Capture(size_t buffer_count, size_t image_count,
                                         size_t sampler_count, size_t fmask_count,
                                         BufferReader&& buffer_reader,
                                         ImageReader&& image_reader,
                                         SamplerReader&& sampler_reader,
                                         FMaskReader&& fmask_reader) {
        StageResourceSnapshot snapshot;
        if (buffer_count > snapshot.buffers.capacity() ||
            image_count > snapshot.images.capacity() ||
            sampler_count > snapshot.samplers.capacity() ||
            fmask_count > snapshot.fmasks.capacity()) {
            snapshot.valid = false;
            return snapshot;
        }
        for (size_t index = 0; index < buffer_count; ++index) {
            snapshot.buffers.emplace_back(buffer_reader(index));
        }
        for (size_t index = 0; index < image_count; ++index) {
            snapshot.images.emplace_back(image_reader(index));
        }
        for (size_t index = 0; index < sampler_count; ++index) {
            snapshot.samplers.emplace_back(sampler_reader(index));
        }
        for (size_t index = 0; index < fmask_count; ++index) {
            snapshot.fmasks.emplace_back(fmask_reader(index));
        }
        return snapshot;
    }

    static StageResourceSnapshot Capture(const Info& info) {
        return Capture(
            info.buffers.size(), info.images.size(), info.samplers.size(), info.fmasks.size(),
            [&](size_t index) { return info.buffers[index].GetSharp(info); },
            [&](size_t index) { return info.images[index].GetSharp(info); },
            [&](size_t index) { return info.samplers[index].GetSharp(info); },
            [&](size_t index) { return info.fmasks[index].GetSharp(info); });
    }
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
    std::bitset<MaxStageResources> bitset{};
    std::optional<Gcn::FetchShaderData> fetch_shader_data{};
    boost::container::small_vector<VsAttribSpecialization, 32> vs_attribs;
    boost::container::small_vector<BufferSpecialization, 16> buffers;
    boost::container::small_vector<ImageSpecialization, 16> images;
    boost::container::small_vector<FMaskSpecialization, 8> fmasks;
    boost::container::small_vector<SamplerSpecialization, 16> samplers;
    Backend::Bindings start{};

    StageSpecialization() = default;
    StageSpecialization(const Info& info_, RuntimeInfo runtime_info_, const Profile& profile_,
                        Backend::Bindings start_, Gcn::VertexInputSnapshot* vertex_inputs = nullptr,
                        const StageResourceSnapshot* resource_snapshot = nullptr)
        : info{&info_}, runtime_info{runtime_info_}, start{start_} {
        const StageResourceSnapshot captured_resources =
            resource_snapshot ? StageResourceSnapshot{} : StageResourceSnapshot::Capture(info_);
        const StageResourceSnapshot& resources =
            resource_snapshot ? *resource_snapshot : captured_resources;
        if (!resources.valid || resources.buffers.size() != info_.buffers.size() ||
            resources.images.size() != info_.images.size() ||
            resources.samplers.size() != info_.samplers.size() ||
            resources.fmasks.size() != info_.fmasks.size()) {
            info = nullptr;
            return;
        }
        fetch_shader_data = Gcn::ParseFetchShader(info_);
        Gcn::VertexInputSnapshot captured_vertex_inputs;
        if (fetch_shader_data) {
            captured_vertex_inputs = Gcn::VertexInputSnapshot::Capture(*fetch_shader_data, info_);
        }
        if (vertex_inputs) {
            *vertex_inputs = captured_vertex_inputs;
        }
        if (!captured_vertex_inputs.valid) {
            return;
        }
        if (info_.stage == Stage::Vertex && fetch_shader_data) {
            // Specialize shader on VS input number types to follow spec.
            for (size_t index = 0; index < fetch_shader_data->attributes.size(); ++index) {
                const auto& desc = fetch_shader_data->attributes[index];
                auto& spec = vs_attribs.emplace_back();
                const auto sharp = captured_vertex_inputs.buffers[index];
                if (!sharp) {
                    continue;
                }
                using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
                if (const auto step_rate = desc.GetStepRate(); step_rate != InstanceIdType::None) {
                    spec.divisor = step_rate == InstanceIdType::OverStepRate0
                                       ? runtime_info.vs_info.step_rate_0
                                       : (step_rate == InstanceIdType::OverStepRate1
                                              ? runtime_info.vs_info.step_rate_1
                                              : 1);
                }
                spec.num_class = AmdGpu::GetNumberClass(sharp.GetNumberFmt());
                spec.dst_select = sharp.DstSelect();
            }
        }
        u32 binding{};
        ForEachSharp(binding, buffers, info->buffers, resources.buffers,
                     [](auto& spec, const auto& desc, AmdGpu::Buffer sharp) {
                         spec.stride = sharp.GetStride();
                         spec.is_storage = desc.IsStorage(sharp);
                         spec.is_formatted = desc.is_formatted;
                         spec.swizzle_enable = sharp.swizzle_enable;
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
        ForEachSharp(binding, images, info->images, resources.images,
                     [&](auto& spec, const auto& desc, AmdGpu::Image sharp) {
                         spec.type = sharp.GetViewType(desc.is_array);
                         spec.is_integer = AmdGpu::IsInteger(sharp.GetNumberFmt());
                         spec.is_storage = desc.is_written;
                         spec.is_cube = sharp.IsCube();
                         if (spec.is_storage) {
                             spec.dst_select = sharp.DstSelect();
                         } else {
                             spec.is_srgb = sharp.GetNumberFmt() == AmdGpu::NumberFormat::Srgb;
                         }
                         spec.num_conversion = sharp.GetNumberConversion();
                         spec.num_bindings = desc.NumBindings(*info);
                     });
        ForEachSharp(binding, fmasks, info->fmasks, resources.fmasks,
                     [](auto& spec, const auto& desc, AmdGpu::Image sharp) {
                         spec.width = sharp.width;
                         spec.height = sharp.height;
                     });
        ForEachSharp(samplers, info->samplers, resources.samplers,
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

    void ForEachSharp(auto& spec_list, auto& desc_list, const auto& sharp_list, auto&& func) {
        for (size_t index = 0; index < desc_list.size(); ++index) {
            const auto& desc = desc_list[index];
            auto& spec = spec_list.emplace_back();
            const auto sharp = sharp_list[index];
            if (!sharp) {
                continue;
            }
            func(spec, desc, sharp);
        }
    }

    void ForEachSharp(u32& binding, auto& spec_list, auto& desc_list, const auto& sharp_list,
                      auto&& func) {
        for (size_t index = 0; index < desc_list.size(); ++index) {
            const auto& desc = desc_list[index];
            auto& spec = spec_list.emplace_back();
            const auto sharp = sharp_list[index];
            if (!sharp) {
                binding++;
                continue;
            }
            bitset.set(binding++);
            func(spec, desc, sharp);
        }
    }

    [[nodiscard]] bool Valid() const {
        return info != nullptr;
    }

    bool operator==(const StageSpecialization& other) const {
        if (!Valid()) {
            return false;
        }

        if (vs_attribs != other.vs_attribs) {
            return false;
        }

        if (runtime_info != other.runtime_info) {
            return false;
        }

        // Descriptor presence is part of the compiled permutation. The scalar
        // specialization entries are value-initialized for missing descriptors,
        // so comparing those entries alone cannot distinguish a missing binding
        // from a present descriptor whose specialization fields are all default.
        if (bitset != other.bitset) {
            return false;
        }

        // The fetch program defines vertex input routing even when the shader has no ordinary
        // buffer/image/sampler resources. Compare it before the empty-resource shortcut so a
        // cached module cannot be reused for a different fetch layout with identical formats.
        if (fetch_shader_data != other.fetch_shader_data) {
            return false;
        }

        // For VS which only generates geometry and doesn't have any inputs, its start
        // bindings still may change as they depend on previously processed FS. The check below
        // handles this case and prevents generation of redundant permutations. This is also safe
        // for other types of shaders with no bindings.
        if (bitset.none() && other.bitset.none()) {
            return true;
        }

        if (fmasks != other.fmasks) {
            return false;
        }

        if (start != other.start) {
            return false;
        }

        u32 binding{};
        for (u32 i = 0; i < buffers.size(); i++) {
            if (other.bitset[binding++] && buffers[i] != other.buffers[i]) {
                return false;
            }
        }
        for (u32 i = 0; i < images.size(); i++) {
            if (other.bitset[binding++] && images[i] != other.images[i]) {
                return false;
            }
        }

        for (u32 i = 0; i < samplers.size(); i++) {
            if (samplers[i] != other.samplers[i]) {
                return false;
            }
        }
        return true;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
