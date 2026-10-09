// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/specialization_serialization.h"

#include <algorithm>
#include <array>
#include <bit>
#include <exception>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>

#include "common/serdes.h"
#include "shader_recompiler/specialization.h"
#include "shader_recompiler/specialization_identity.h"

namespace Shader {
namespace {

constexpr u32 MaxFetchAttributes = 32;
constexpr u32 MaxGeometryCopyDwords = 1U << 20;
constexpr u32 MaxFlattenedUserDataDwords = 1U << 20;
constexpr u32 MaxImageDescriptorBindings = NUM_IMAGES;
constexpr u32 UserDataPointerDwords = sizeof(uintptr_t) / sizeof(u32);

void WriteBool(Serialization::Writer& writer, bool value) {
    writer.Write(static_cast<u8>(value ? 1 : 0));
}

[[nodiscard]] bool ReadBool(Serialization::Reader& reader, bool& value) {
    u8 encoded{};
    if (!reader.Read(encoded) || encoded > 1) {
        return false;
    }
    value = encoded != 0;
    return true;
}

template <typename Enum>
    requires std::is_enum_v<Enum>
void WriteEnum(Serialization::Writer& writer, Enum value) {
    writer.Write(static_cast<u32>(value));
}

template <typename Enum, typename Validator>
    requires std::is_enum_v<Enum>
[[nodiscard]] bool ReadEnum(Serialization::Reader& reader, Enum& value, Validator&& is_valid) {
    u32 encoded{};
    if (!reader.Read(encoded) || !is_valid(encoded)) {
        return false;
    }
    value = static_cast<Enum>(encoded);
    return true;
}

void WriteCount(Serialization::Writer& writer, size_t count) {
    ASSERT(count <= std::numeric_limits<u32>::max());
    writer.Write(static_cast<u32>(count));
}

[[nodiscard]] bool ReadCount(Serialization::Reader& reader, u32& count, u32 maximum) {
    return reader.Read(count) && count <= maximum;
}

template <typename... Enums>
[[nodiscard]] constexpr bool IsOneOf(u32 value, Enums... candidates) noexcept {
    return ((value == static_cast<u32>(candidates)) || ...);
}

[[nodiscard]] constexpr bool IsValidDataFormat(u32 value) noexcept {
    using E = AmdGpu::DataFormat;
    return IsOneOf(
        value, E::FormatInvalid, E::Format8, E::Format16, E::Format8_8, E::Format32, E::Format16_16,
        E::Format10_11_11, E::Format11_11_10, E::Format10_10_10_2, E::Format2_10_10_10,
        E::Format8_8_8_8, E::Format32_32, E::Format16_16_16_16, E::Format32_32_32,
        E::Format32_32_32_32, E::Format5_6_5, E::Format1_5_5_5, E::Format5_5_5_1, E::Format4_4_4_4,
        E::Format8_24, E::Format24_8, E::FormatX24_8_32, E::FormatGB_GR, E::FormatBG_RG,
        E::Format5_9_9_9, E::FormatBc1, E::FormatBc2, E::FormatBc3, E::FormatBc4, E::FormatBc5,
        E::FormatBc6, E::FormatBc7, E::FormatFmask8_S2_F1, E::FormatFmask8_S4_F1,
        E::FormatFmask8_S8_F1, E::FormatFmask8_S2_F2, E::FormatFmask8_S4_F2, E::FormatFmask8_S4_F4,
        E::FormatFmask16_S16_F1, E::FormatFmask16_S8_F2, E::FormatFmask32_S16_F2,
        E::FormatFmask32_S8_F4, E::FormatFmask32_S8_F8, E::FormatFmask64_S16_F4,
        E::FormatFmask64_S16_F8, E::Format4_4, E::Format6_5_5, E::Format1, E::Format1_Reversed,
        E::Format32_As_8, E::Format32_As_8_8, E::Format32_As_32_32_32_32);
}

[[nodiscard]] constexpr bool IsValidNumberFormat(u32 value) noexcept {
    using E = AmdGpu::NumberFormat;
    return IsOneOf(value, E::Unorm, E::Snorm, E::Uscaled, E::Sscaled, E::Uint, E::Sint, E::SnormNz,
                   E::Float, E::Srgb, E::Ubnorm, E::UbnormNz, E::Ubint, E::Ubscaled);
}

[[nodiscard]] constexpr bool IsValidCompSwizzle(u32 value) noexcept {
    using E = AmdGpu::CompSwizzle;
    return IsOneOf(value, E::Zero, E::One, E::Red, E::Green, E::Blue, E::Alpha);
}

[[nodiscard]] constexpr bool IsValidNumberClass(u32 value) noexcept {
    return value <= static_cast<u32>(AmdGpu::NumberClass::Uint);
}

[[nodiscard]] constexpr bool IsValidNumberConversion(u32 value) noexcept {
    return value <= static_cast<u32>(AmdGpu::NumberConversion::Sint32ToSnormNz);
}

[[nodiscard]] constexpr bool IsValidColorNumberConversion(u32 value) noexcept {
    return value <= static_cast<u32>(AmdGpu::NumberConversion::SrgbToNorm);
}

[[nodiscard]] constexpr bool IsValidImageType(u32 value) noexcept {
    using E = AmdGpu::ImageType;
    return IsOneOf(value, E::Invalid, E::Color1D, E::Color2D, E::Color3D, E::Cube, E::Color1DArray,
                   E::Color2DArray, E::Color2DMsaa, E::Color2DMsaaArray);
}

[[nodiscard]] constexpr bool IsValidPrimitiveType(u32 value) noexcept {
    using E = AmdGpu::PrimitiveType;
    return IsOneOf(value, E::None, E::PointList, E::LineList, E::LineStrip, E::TriangleList,
                   E::TriangleFan, E::TriangleStrip, E::PatchPrimitive, E::AdjLineList,
                   E::AdjLineStrip, E::AdjTriangleList, E::AdjTriangleStrip, E::RectList,
                   E::LineLoop, E::QuadList, E::QuadStrip, E::Polygon);
}

[[nodiscard]] constexpr bool IsValidOutput(u32 value) noexcept {
    return value <= static_cast<u32>(Output::ClipDist7);
}

[[nodiscard]] constexpr bool IsValidShaderExportFormat(u32 value) noexcept {
    return value <= static_cast<u32>(AmdGpu::ShaderExportFormat::ABGR_32);
}

[[nodiscard]] constexpr bool IsValidIrType(u32 value) noexcept {
    constexpr u32 LastTypeBit = static_cast<u32>(IR::Type::StringLiteral);
    constexpr u32 ValidMask = LastTypeBit | (LastTypeBit - 1U);
    return (value & ~ValidMask) == 0;
}

[[nodiscard]] constexpr bool IsValidSamplerDescriptor(u64 raw0, u64 raw1) noexcept {
    constexpr u64 MaxAnisoMask = 0x7ULL;
    constexpr u64 FilterModeMask = 0x3ULL;
    constexpr u64 MipFilterMask = 0x3ULL;
    const u64 max_aniso = (raw0 >> 9U) & MaxAnisoMask;
    const u64 filter_mode = (raw0 >> 29U) & FilterModeMask;
    const u64 mip_filter = (raw1 >> 26U) & MipFilterMask;
    return max_aniso <= static_cast<u64>(AmdGpu::AnisoRatio::Sixteen) &&
           filter_mode <= static_cast<u64>(AmdGpu::FilterMode::Max) &&
           mip_filter <= static_cast<u64>(AmdGpu::MipFilter::Linear);
}

void WriteCompMapping(Serialization::Writer& writer, AmdGpu::CompMapping mapping) {
    WriteEnum(writer, mapping.r);
    WriteEnum(writer, mapping.g);
    WriteEnum(writer, mapping.b);
    WriteEnum(writer, mapping.a);
}

[[nodiscard]] bool ReadCompMapping(Serialization::Reader& reader, AmdGpu::CompMapping& mapping) {
    AmdGpu::CompSwizzle r{};
    AmdGpu::CompSwizzle g{};
    AmdGpu::CompSwizzle b{};
    AmdGpu::CompSwizzle a{};
    if (!ReadEnum(reader, r, IsValidCompSwizzle) || !ReadEnum(reader, g, IsValidCompSwizzle) ||
        !ReadEnum(reader, b, IsValidCompSwizzle) || !ReadEnum(reader, a, IsValidCompSwizzle)) {
        return false;
    }
    mapping = {.r = r, .g = g, .b = b, .a = a};
    return true;
}

void WriteOutputMap(Serialization::Writer& writer, const OutputMap& outputs) {
    for (const auto output : outputs) {
        WriteEnum(writer, output);
    }
}

[[nodiscard]] bool ReadOutputMap(Serialization::Reader& reader, OutputMap& outputs) {
    OutputMap decoded{};
    for (auto& output : decoded) {
        if (!ReadEnum(reader, output, IsValidOutput)) {
            return false;
        }
    }
    outputs = decoded;
    return true;
}

template <size_t NumWords, typename Descriptor>
void WriteDescriptorWords(Serialization::Writer& writer, const Descriptor& descriptor) {
    static_assert(sizeof(Descriptor) == NumWords * sizeof(u32));
    static_assert(std::is_trivially_copyable_v<Descriptor>);
    static_assert(std::has_unique_object_representations_v<Descriptor>);
    const auto words = std::bit_cast<std::array<u32, NumWords>>(descriptor);
    for (const u32 word : words) {
        writer.Write(word);
    }
}

template <size_t NumWords, typename Descriptor>
[[nodiscard]] bool ReadDescriptorWords(Serialization::Reader& reader, Descriptor& descriptor) {
    static_assert(sizeof(Descriptor) == NumWords * sizeof(u32));
    static_assert(std::is_trivially_copyable_v<Descriptor>);
    static_assert(std::has_unique_object_representations_v<Descriptor>);
    std::array<u32, NumWords> words{};
    for (u32& word : words) {
        if (!reader.Read(word)) {
            return false;
        }
    }
    descriptor = std::bit_cast<Descriptor>(words);
    return true;
}

template <typename Container, typename WriteElement>
void WriteRecords(Serialization::Writer& writer, const Container& records,
                  WriteElement&& write_element) {
    WriteCount(writer, records.size());
    for (const auto& record : records) {
        write_element(writer, record);
    }
}

template <typename Container, typename ReadElement>
[[nodiscard]] bool ReadRecords(Serialization::Reader& reader, Container& records, u32 maximum,
                               ReadElement&& read_element) {
    u32 count{};
    if (!ReadCount(reader, count, maximum) || count > records.max_size()) {
        return false;
    }

    Container decoded{};
    try {
        decoded.reserve(count);
        for (u32 index = 0; index < count; ++index) {
            typename Container::value_type record{};
            if (!read_element(reader, record)) {
                return false;
            }
            decoded.emplace_back(std::move(record));
        }
    } catch (const std::exception&) {
        return false;
    }
    records = std::move(decoded);
    return true;
}

template <typename Sharp>
[[nodiscard]] bool SharpFits(const Info& info, u32 sharp_idx) noexcept {
    static_assert(sizeof(Sharp) % sizeof(u32) == 0);
    constexpr size_t SharpDwords = sizeof(Sharp) / sizeof(u32);
    return sharp_idx <= info.flattened_ud_buf.size() &&
           SharpDwords <= info.flattened_ud_buf.size() - sharp_idx;
}

[[nodiscard]] constexpr bool UserDataPointerFits(u32 sgpr_base) noexcept {
    return sgpr_base <= NUM_USER_DATA_REGS &&
           UserDataPointerDwords <= NUM_USER_DATA_REGS - sgpr_base;
}

[[nodiscard]] bool ValidateFetchShaderData(const Gcn::FetchShaderData& fetch) noexcept {
    const auto is_valid_offset_sgpr = [](s8 sgpr) {
        return sgpr == -1 || (sgpr >= 0 && static_cast<u32>(sgpr) < NUM_USER_DATA_REGS);
    };
    if (!is_valid_offset_sgpr(fetch.vertex_offset_sgpr) ||
        !is_valid_offset_sgpr(fetch.instance_offset_sgpr)) {
        return false;
    }

    std::array<bool, IR::NumParams> semantics{};
    for (const auto& attribute : fetch.attributes) {
        if (attribute.semantic >= semantics.size() || semantics[attribute.semantic] ||
            attribute.num_elements == 0 || attribute.num_elements > 4 ||
            !UserDataPointerFits(attribute.sgpr_base) ||
            attribute.instance_data > Gcn::VertexAttribute::Plain || attribute.data_format > 14 ||
            attribute.num_format > 7 || (attribute.data_format == 0 && attribute.num_format != 0)) {
            return false;
        }
        semantics[attribute.semantic] = true;
    }
    return true;
}

[[nodiscard]] bool ValidateTessConstantLocation(const Info& info) noexcept {
    const bool is_tessellation = info.l_stage == LogicalStage::TessellationControl ||
                                 info.l_stage == LogicalStage::TessellationEval;
    if (!is_tessellation) {
        return info.tess_consts_dword_offset == -1 &&
               info.tess_consts_ptr_base == IR::ScalarReg::Max;
    }
    if (info.tess_consts_dword_offset < 0) {
        return false;
    }

    const u32 ptr_base = static_cast<u32>(info.tess_consts_ptr_base);
    if (ptr_base == IR::NumScalarRegs) {
        constexpr u32 BufferDwords = sizeof(AmdGpu::Buffer) / sizeof(u32);
        const u32 offset = static_cast<u32>(info.tess_consts_dword_offset);
        return offset <= NUM_USER_DATA_REGS && BufferDwords <= NUM_USER_DATA_REGS - offset;
    }
    return UserDataPointerFits(ptr_base);
}

[[nodiscard]] constexpr bool IsValidStage(Stage stage) noexcept {
    return static_cast<u32>(stage) <= static_cast<u32>(Stage::Compute);
}

[[nodiscard]] constexpr bool IsValidLogicalStage(LogicalStage stage) noexcept {
    return static_cast<u32>(stage) < static_cast<u32>(LogicalStage::NumLogicalStages);
}

[[nodiscard]] bool ValidateRuntimeInfo(const RuntimeInfo& info) noexcept {
    if (!IsValidStage(info.stage) || info.num_user_data > NUM_USER_DATA_REGS) {
        return false;
    }
    if (info.stage == Stage::Fragment && info.fs_info.num_inputs > info.fs_info.inputs.size()) {
        return false;
    }
    if (info.stage == Stage::Geometry && info.gs_info.vs_copy.size() > MaxGeometryCopyDwords) {
        return false;
    }
    return true;
}

[[nodiscard]] constexpr bool IsValidAtomicImageFormat(const ImageSpecialization& image) noexcept {
    using AmdGpu::DataFormat;
    using AmdGpu::NumberFormat;
    const auto data = image.atomic_data_format;
    const auto number = image.atomic_number_format;
    if (data == DataFormat::FormatInvalid) {
        return number == NumberFormat::Unorm;
    }
    switch (data) {
    case DataFormat::Format32:
        return number == NumberFormat::Uint || number == NumberFormat::Sint ||
               number == NumberFormat::Float;
    case DataFormat::Format32_32:
        return number == NumberFormat::Float || number == NumberFormat::Uint;
    case DataFormat::Format32_32_32_32:
        return number == NumberFormat::Uint || number == NumberFormat::Float;
    case DataFormat::Format16:
        return number == NumberFormat::Float || number == NumberFormat::Uint;
    case DataFormat::Format16_16:
        return number == NumberFormat::Float || number == NumberFormat::Snorm;
    case DataFormat::Format8_8:
        return number == NumberFormat::Unorm;
    case DataFormat::Format16_16_16_16:
        return number == NumberFormat::Float || number == NumberFormat::Unorm;
    case DataFormat::Format8:
        return number == NumberFormat::Unorm;
    case DataFormat::Format8_8_8_8:
        return number == NumberFormat::Unorm || number == NumberFormat::Uint;
    case DataFormat::Format10_11_11:
        return number == NumberFormat::Float;
    default:
        return false;
    }
}

[[nodiscard]] bool ValidateInfo(const Info& info) noexcept {
    if (!IsValidStage(info.stage) || !IsValidLogicalStage(info.l_stage) ||
        info.buffers.size() > NUM_BUFFERS || info.images.size() > NUM_IMAGES ||
        info.samplers.size() > NUM_SAMPLERS || info.fmasks.size() > NUM_FMASKS ||
        info.flattened_ud_buf.size() != info.srt_info.flattened_bufsize_dw ||
        !info.srt_info.ValidateRangeReservations() ||
        info.num_addr64_pointer_roots > MaxAddr64PointerRoots ||
        (info.ud_mask.mask >> NUM_USER_DATA_REGS) != 0 ||
        (info.has_fetch_shader && !UserDataPointerFits(info.fetch_shader_sgpr_base)) ||
        !ValidateTessConstantLocation(info)) {
        return false;
    }
    if (std::ranges::any_of(info.buffers, [&](const BufferResource& resource) {
            return static_cast<u32>(resource.buffer_type) >
                       static_cast<u32>(BufferType::ClipPlanes) ||
                   (!resource.inline_cbuf && !SharpFits<AmdGpu::Buffer>(info, resource.sharp_idx));
        })) {
        return false;
    }
    if (std::ranges::any_of(
            info.images,
            [&](const ImageResource& resource) {
                if (static_cast<u32>(resource.mip_fallback_mode) >
                    static_cast<u32>(MipStorageFallbackMode::ConstantIndex)) {
                    return true;
                }
                if (resource.dynamic_descriptor_count == 0) {
                    return resource.dynamic_descriptor_stride_dw != 0 ||
                           resource.requires_dynamic_oob_guard ||
                           (resource.is_r128 ? !SharpFits<u128>(info, resource.sharp_idx)
                                             : !SharpFits<AmdGpu::Image>(info, resource.sharp_idx));
                }
                if (!resource.HasValidDynamicRange(info)) {
                    return true;
                }
                const u32 descriptor_dw = resource.is_r128 ? 4U : 8U;
                const u64 resource_end = u64{resource.sharp_idx} +
                                         u64{resource.dynamic_descriptor_count - 1U} *
                                             resource.dynamic_descriptor_stride_dw +
                                         descriptor_dw;
                return std::ranges::none_of(
                    info.srt_info.range_reservations, [&](const auto& reservation) {
                        const u64 reservation_end =
                            u64{reservation.destination_flat_offset_dw} + reservation.span_dw;
                        return reservation.destination_flat_offset_dw <= resource.sharp_idx &&
                               resource_end <= reservation_end;
                    });
            }) ||
        std::ranges::any_of(info.samplers,
                            [&](const SamplerResource& resource) {
                                return (!resource.is_inline_sampler &&
                                        !SharpFits<AmdGpu::Sampler>(info, resource.sharp_idx)) ||
                                       (resource.disable_aniso &&
                                        resource.associated_image >= info.images.size());
                            }) ||
        std::ranges::any_of(info.fmasks, [&](const FMaskResource& resource) {
            return !SharpFits<AmdGpu::Image>(info, resource.sharp_idx);
        })) {
        return false;
    }
    return true;
}

void WriteBufferAccessRange(Serialization::Writer& writer, const BufferAccessRange& range) {
    writer.Write(range.UpperBound());
    WriteBool(writer, range.IsBounded());
    WriteBool(writer, range.HasAddressLayout());

    const auto layout = range.AddressLayout();
    writer.Write(layout.stride);
    writer.Write(layout.data_format);
    writer.Write(layout.element_size);
    writer.Write(layout.index_stride);
    WriteBool(writer, layout.swizzle_enable);
    WriteBool(writer, layout.add_tid_enable);
}

[[nodiscard]] bool ReadBufferAccessRange(Serialization::Reader& reader, BufferAccessRange& range) {
    u64 upper_bound{};
    bool is_bounded{};
    bool has_address_layout{};
    BufferAddressLayout layout{};
    if (!reader.Read(upper_bound) || !ReadBool(reader, is_bounded) ||
        !ReadBool(reader, has_address_layout) || !reader.Read(layout.stride) ||
        !reader.Read(layout.data_format) || !reader.Read(layout.element_size) ||
        !reader.Read(layout.index_stride) || !ReadBool(reader, layout.swizzle_enable) ||
        !ReadBool(reader, layout.add_tid_enable) || layout.stride > 0x3FFFU ||
        layout.data_format > 0xFU || layout.element_size > 3 || layout.index_stride > 3) {
        return false;
    }
    if (!has_address_layout && layout != BufferAddressLayout{}) {
        return false;
    }
    range = BufferAccessRange::FromSerialized(upper_bound, layout, is_bounded, has_address_layout);
    return true;
}

void WriteBufferResource(Serialization::Writer& writer, const BufferResource& resource) {
    writer.Write(resource.sharp_idx);
    writer.Write(static_cast<u32>(resource.used_types));
    WriteDescriptorWords<4>(writer, resource.inline_cbuf);
    WriteEnum(writer, resource.buffer_type);
    writer.Write(resource.instance_attrib);
    WriteBool(writer, resource.is_written);
    WriteBool(writer, resource.is_formatted);
    WriteBufferAccessRange(writer, resource.access_range);
}

[[nodiscard]] bool ReadBufferResource(Serialization::Reader& reader, BufferResource& resource) {
    u32 used_types{};
    BufferType buffer_type{};
    bool is_written{};
    bool is_formatted{};
    BufferResource decoded{};
    if (!reader.Read(decoded.sharp_idx) || !reader.Read(used_types) || !IsValidIrType(used_types) ||
        !ReadDescriptorWords<4>(reader, decoded.inline_cbuf) ||
        !ReadEnum(reader, buffer_type,
                  [](u32 value) { return value <= static_cast<u32>(BufferType::ClipPlanes); }) ||
        !reader.Read(decoded.instance_attrib) || !ReadBool(reader, is_written) ||
        !ReadBool(reader, is_formatted) || !ReadBufferAccessRange(reader, decoded.access_range)) {
        return false;
    }
    decoded.used_types = static_cast<IR::Type>(used_types);
    decoded.buffer_type = buffer_type;
    decoded.is_written = is_written;
    decoded.is_formatted = is_formatted;
    resource = decoded;
    return true;
}

void WriteImageResource(Serialization::Writer& writer, const ImageResource& resource) {
    writer.Write(resource.sharp_idx);
    writer.Write(resource.dynamic_descriptor_count);
    writer.Write(resource.dynamic_descriptor_stride_dw);
    WriteBool(writer, resource.is_depth);
    WriteBool(writer, resource.is_atomic);
    WriteBool(writer, resource.is_array);
    WriteBool(writer, resource.is_written);
    WriteBool(writer, resource.uses_integer_coordinates);
    WriteBool(writer, resource.queries_dimensions);
    WriteBool(writer, resource.is_r128);
    WriteBool(writer, resource.requires_dynamic_oob_guard);
    WriteEnum(writer, resource.mip_fallback_mode);
    writer.Write(resource.constant_mip_index);
}

[[nodiscard]] bool ReadImageResource(Serialization::Reader& reader, ImageResource& resource) {
    ImageResource decoded{};
    MipStorageFallbackMode mip_fallback_mode{};
    if (!reader.Read(decoded.sharp_idx) || !reader.Read(decoded.dynamic_descriptor_count) ||
        !reader.Read(decoded.dynamic_descriptor_stride_dw) || !ReadBool(reader, decoded.is_depth) ||
        !ReadBool(reader, decoded.is_atomic) || !ReadBool(reader, decoded.is_array) ||
        !ReadBool(reader, decoded.is_written) ||
        !ReadBool(reader, decoded.uses_integer_coordinates) ||
        !ReadBool(reader, decoded.queries_dimensions) || !ReadBool(reader, decoded.is_r128) ||
        !ReadBool(reader, decoded.requires_dynamic_oob_guard) ||
        !ReadEnum(reader, mip_fallback_mode,
                  [](u32 value) {
                      return value <= static_cast<u32>(MipStorageFallbackMode::ConstantIndex);
                  }) ||
        !reader.Read(decoded.constant_mip_index)) {
        return false;
    }
    decoded.mip_fallback_mode = mip_fallback_mode;
    resource = decoded;
    return true;
}

void WriteSamplerResource(Serialization::Writer& writer, const SamplerResource& resource) {
    writer.Write(resource.sharp_idx);
    writer.Write(resource.inline_sampler.raw0);
    writer.Write(resource.inline_sampler.raw1);
    WriteBool(writer, resource.is_inline_sampler);
    writer.Write(static_cast<u32>(resource.associated_image));
    WriteBool(writer, resource.disable_aniso);
}

[[nodiscard]] bool ReadSamplerResource(Serialization::Reader& reader, SamplerResource& resource) {
    SamplerResource decoded{};
    u64 inline_raw0{};
    u64 inline_raw1{};
    bool is_inline_sampler{};
    bool disable_aniso{};
    u32 associated_image{};
    if (!reader.Read(decoded.sharp_idx) || !reader.Read(inline_raw0) || !reader.Read(inline_raw1) ||
        !ReadBool(reader, is_inline_sampler) || !reader.Read(associated_image) ||
        associated_image >= NUM_IMAGES || !ReadBool(reader, disable_aniso) ||
        (is_inline_sampler && !IsValidSamplerDescriptor(inline_raw0, inline_raw1))) {
        return false;
    }
    decoded.inline_sampler.raw0 = inline_raw0;
    decoded.inline_sampler.raw1 = inline_raw1;
    decoded.is_inline_sampler = is_inline_sampler;
    decoded.associated_image = associated_image;
    decoded.disable_aniso = disable_aniso;
    resource = decoded;
    return true;
}

void WriteFMaskResource(Serialization::Writer& writer, const FMaskResource& resource) {
    writer.Write(resource.sharp_idx);
}

[[nodiscard]] bool ReadFMaskResource(Serialization::Reader& reader, FMaskResource& resource) {
    FMaskResource decoded{};
    if (!reader.Read(decoded.sharp_idx)) {
        return false;
    }
    resource = decoded;
    return true;
}

void WriteInfoPersistent(Serialization::Writer& writer, const Info& info) {
    WriteRecords(writer, info.buffers, WriteBufferResource);
    WriteRecords(writer, info.images, WriteImageResource);
    WriteRecords(writer, info.samplers, WriteSamplerResource);
    WriteRecords(writer, info.fmasks, WriteFMaskResource);
    writer.Write(info.ud_mask.mask);
    writer.Write(info.fetch_shader_sgpr_base);
    writer.Write(info.pgm_hash);
    writer.Write(info.tess_consts_dword_offset);
    WriteEnum(writer, info.tess_consts_ptr_base);
    WriteEnum(writer, info.stage);
    WriteEnum(writer, info.l_stage);
    writer.Write(info.mrt_mask);
    WriteBool(writer, info.has_fetch_shader);
    WriteBool(writer, info.has_bitwise_xor);
    WriteBool(writer, info.uses_dma);
    WriteCount(writer, info.num_addr64_pointer_roots);
    const u32 encoded_root_count =
        std::min<u32>(info.num_addr64_pointer_roots, info.addr64_pointer_roots.size());
    for (u32 index = 0; index < encoded_root_count; ++index) {
        const auto& root = info.addr64_pointer_roots[index];
        writer.Write(root.buffer_resource_index);
        writer.Write(root.pointer_lo_flat_index);
    }
    WriteBool(writer, info.addr64_pointer_roots_overflow);
}

[[nodiscard]] bool ReadInfoPersistent(Serialization::Reader& reader, Info& info) {
    if (!ReadRecords(reader, info.buffers, NUM_BUFFERS, ReadBufferResource) ||
        !ReadRecords(reader, info.images, NUM_IMAGES, ReadImageResource) ||
        !ReadRecords(reader, info.samplers, NUM_SAMPLERS, ReadSamplerResource) ||
        !ReadRecords(reader, info.fmasks, NUM_FMASKS, ReadFMaskResource) ||
        !reader.Read(info.ud_mask.mask) || !reader.Read(info.fetch_shader_sgpr_base) ||
        !reader.Read(info.pgm_hash) || !reader.Read(info.tess_consts_dword_offset) ||
        !ReadEnum(reader, info.tess_consts_ptr_base,
                  [](u32 value) { return value <= static_cast<u32>(IR::ScalarReg::Max); }) ||
        !ReadEnum(reader, info.stage,
                  [](u32 value) { return value <= static_cast<u32>(Stage::Compute); }) ||
        !ReadEnum(
            reader, info.l_stage,
            [](u32 value) { return value < static_cast<u32>(LogicalStage::NumLogicalStages); }) ||
        !reader.Read(info.mrt_mask) || !ReadBool(reader, info.has_fetch_shader) ||
        !ReadBool(reader, info.has_bitwise_xor) || !ReadBool(reader, info.uses_dma)) {
        return false;
    }

    u32 root_count{};
    if (!ReadCount(reader, root_count, MaxAddr64PointerRoots)) {
        return false;
    }
    std::array<Addr64PointerRoot, MaxAddr64PointerRoots> roots{};
    for (u32 index = 0; index < root_count; ++index) {
        if (!reader.Read(roots[index].buffer_resource_index) ||
            !reader.Read(roots[index].pointer_lo_flat_index)) {
            return false;
        }
    }
    bool roots_overflow{};
    if (!ReadBool(reader, roots_overflow)) {
        return false;
    }
    info.addr64_pointer_roots = roots;
    info.num_addr64_pointer_roots = root_count;
    info.addr64_pointer_roots_overflow = roots_overflow;
    return true;
}

void WritePsInputFlags(Serialization::Writer& writer, const AmdGpu::PsInput& input) {
    WriteBool(writer, input.persp_sample_ena);
    WriteBool(writer, input.persp_center_ena);
    WriteBool(writer, input.persp_centroid_ena);
    WriteBool(writer, input.persp_pull_model_ena);
    WriteBool(writer, input.linear_sample_ena);
    WriteBool(writer, input.linear_center_ena);
    WriteBool(writer, input.linear_centroid_ena);
    WriteBool(writer, input.line_stipple_tex_ena);
    WriteBool(writer, input.pos_x_float_ena);
    WriteBool(writer, input.pos_y_float_ena);
    WriteBool(writer, input.pos_z_float_ena);
    WriteBool(writer, input.pos_w_float_ena);
    WriteBool(writer, input.front_face_ena);
    WriteBool(writer, input.ancillary_ena);
    WriteBool(writer, input.sample_coverage_ena);
    WriteBool(writer, input.pos_fixed_pt_ena);
}

[[nodiscard]] bool ReadPsInputFlags(Serialization::Reader& reader, AmdGpu::PsInput& input) {
    std::array<bool, 16> flags{};
    for (bool& flag : flags) {
        if (!ReadBool(reader, flag)) {
            return false;
        }
    }
    AmdGpu::PsInput decoded{};
    decoded.persp_sample_ena = flags[0];
    decoded.persp_center_ena = flags[1];
    decoded.persp_centroid_ena = flags[2];
    decoded.persp_pull_model_ena = flags[3];
    decoded.linear_sample_ena = flags[4];
    decoded.linear_center_ena = flags[5];
    decoded.linear_centroid_ena = flags[6];
    decoded.line_stipple_tex_ena = flags[7];
    decoded.pos_x_float_ena = flags[8];
    decoded.pos_y_float_ena = flags[9];
    decoded.pos_z_float_ena = flags[10];
    decoded.pos_w_float_ena = flags[11];
    decoded.front_face_ena = flags[12];
    decoded.ancillary_ena = flags[13];
    decoded.sample_coverage_ena = flags[14];
    decoded.pos_fixed_pt_ena = flags[15];
    input = decoded;
    return true;
}

void WritePsColorBuffer(Serialization::Writer& writer, const PsColorBuffer& color) {
    WriteEnum(writer, color.data_format);
    WriteEnum(writer, color.num_format);
    WriteEnum(writer, color.num_conversion);
    WriteEnum(writer, color.export_format);
    WriteCompMapping(writer, color.swizzle);
}

[[nodiscard]] bool ReadPsColorBuffer(Serialization::Reader& reader, PsColorBuffer& color) {
    AmdGpu::DataFormat data_format{};
    AmdGpu::NumberFormat num_format{};
    AmdGpu::NumberConversion num_conversion{};
    AmdGpu::ShaderExportFormat export_format{};
    AmdGpu::CompMapping swizzle{};
    if (!ReadEnum(reader, data_format, IsValidDataFormat) ||
        !ReadEnum(reader, num_format, IsValidNumberFormat) ||
        !ReadEnum(reader, num_conversion, IsValidColorNumberConversion) ||
        !ReadEnum(reader, export_format, IsValidShaderExportFormat) ||
        !ReadCompMapping(reader, swizzle)) {
        return false;
    }
    PsColorBuffer decoded{};
    decoded.data_format = data_format;
    decoded.num_format = num_format;
    decoded.num_conversion = num_conversion;
    decoded.export_format = export_format;
    decoded.swizzle = swizzle;
    color = decoded;
    return true;
}

void WriteCommonRuntimeInfo(Serialization::Writer& writer, const RuntimeInfo& info) {
    WriteEnum(writer, info.stage);
    writer.Write(info.num_user_data);
    writer.Write(info.num_input_vgprs);
    writer.Write(info.num_allocated_vgprs);
    WriteEnum(writer, info.fp_denorm_mode32);
    WriteEnum(writer, info.fp_denorm_mode16_64);
    WriteEnum(writer, info.fp_round_mode32);
    WriteEnum(writer, info.fp_round_mode16_64);
}

[[nodiscard]] bool ReadCommonRuntimeInfo(Serialization::Reader& reader, RuntimeInfo& info) {
    Stage stage{};
    u32 num_user_data{};
    u32 num_input_vgprs{};
    u32 num_allocated_vgprs{};
    AmdGpu::FpDenormMode fp_denorm_mode32{};
    AmdGpu::FpDenormMode fp_denorm_mode16_64{};
    AmdGpu::FpRoundMode fp_round_mode32{};
    AmdGpu::FpRoundMode fp_round_mode16_64{};
    const auto valid_fp_mode = [](u32 value) { return value <= 3; };
    if (!ReadEnum(reader, stage,
                  [](u32 value) { return value <= static_cast<u32>(Stage::Compute); }) ||
        !reader.Read(num_user_data) || !reader.Read(num_input_vgprs) ||
        !reader.Read(num_allocated_vgprs) || !ReadEnum(reader, fp_denorm_mode32, valid_fp_mode) ||
        !ReadEnum(reader, fp_denorm_mode16_64, valid_fp_mode) ||
        !ReadEnum(reader, fp_round_mode32, valid_fp_mode) ||
        !ReadEnum(reader, fp_round_mode16_64, valid_fp_mode)) {
        return false;
    }
    info.Initialize(stage);
    info.num_user_data = num_user_data;
    info.num_input_vgprs = num_input_vgprs;
    info.num_allocated_vgprs = num_allocated_vgprs;
    info.fp_denorm_mode32 = fp_denorm_mode32;
    info.fp_denorm_mode16_64 = fp_denorm_mode16_64;
    info.fp_round_mode32 = fp_round_mode32;
    info.fp_round_mode16_64 = fp_round_mode16_64;
    return true;
}

void WriteCommonHsEsVsInfo(Serialization::Writer& writer, const RuntimeInfo& info) {
    writer.Write(info.hs_es_vs_info.hs_output_cp_stride);
}

[[nodiscard]] bool ReadCommonHsEsVsInfo(Serialization::Reader& reader, RuntimeInfo& info) {
    return reader.Read(info.hs_es_vs_info.hs_output_cp_stride);
}

void WriteCommonEsVsInfo(Serialization::Writer& writer, const RuntimeInfo& info) {
    WriteCommonHsEsVsInfo(writer, info);
    WriteEnum(writer, info.es_vs_info.tess_type);
    WriteEnum(writer, info.es_vs_info.tess_topology);
    WriteEnum(writer, info.es_vs_info.tess_partitioning);
}

[[nodiscard]] bool ReadCommonEsVsInfo(Serialization::Reader& reader, RuntimeInfo& info) {
    AmdGpu::TessellationType tess_type{};
    AmdGpu::TessellationTopology tess_topology{};
    AmdGpu::TessellationPartitioning tess_partitioning{};
    if (!ReadCommonHsEsVsInfo(reader, info) ||
        !ReadEnum(
            reader, tess_type,
            [](u32 value) { return value <= static_cast<u32>(AmdGpu::TessellationType::Quad); }) ||
        !ReadEnum(reader, tess_topology,
                  [](u32 value) {
                      return value <= static_cast<u32>(AmdGpu::TessellationTopology::TriangleCcw);
                  }) ||
        !ReadEnum(reader, tess_partitioning, [](u32 value) {
            return value <= static_cast<u32>(AmdGpu::TessellationPartitioning::FracEven);
        })) {
        return false;
    }
    info.es_vs_info.tess_type = tess_type;
    info.es_vs_info.tess_topology = tess_topology;
    info.es_vs_info.tess_partitioning = tess_partitioning;
    return true;
}

void WriteRuntimeInfo(Serialization::Writer& writer, const RuntimeInfo& info) {
    WriteCommonRuntimeInfo(writer, info);
    switch (info.stage) {
    case Stage::Local:
        writer.Write(info.ls_info.ls_stride);
        break;
    case Stage::Export:
        WriteCommonEsVsInfo(writer, info);
        writer.Write(info.es_info.vertex_data_size);
        break;
    case Stage::Vertex:
        WriteCommonEsVsInfo(writer, info);
        writer.Write(info.vs_info.num_outputs);
        for (const auto& outputs : info.vs_info.outputs) {
            WriteOutputMap(writer, outputs);
        }
        WriteBool(writer, info.vs_info.tess_emulated_primitive);
        WriteBool(writer, info.vs_info.emulate_depth_negative_one_to_one);
        WriteBool(writer, info.vs_info.clip_disable);
        writer.Write(info.vs_info.step_rate_0);
        writer.Write(info.vs_info.step_rate_1);
        writer.Write(info.vs_info.user_clip_plane_mask);
        break;
    case Stage::Hull:
        WriteCommonHsEsVsInfo(writer, info);
        writer.Write(info.hs_info.num_input_control_points);
        writer.Write(info.hs_info.num_threads);
        WriteEnum(writer, info.hs_info.tess_type);
        WriteBool(writer, info.hs_info.offchip_lds_enable);
        writer.Write(info.hs_info.ls_stride);
        writer.Write(info.hs_info.hs_output_base);
        break;
    case Stage::Geometry:
        writer.Write(info.gs_info.num_outputs);
        for (const auto& outputs : info.gs_info.outputs) {
            WriteOutputMap(writer, outputs);
        }
        writer.Write(info.gs_info.num_invocations);
        writer.Write(info.gs_info.output_vertices);
        writer.Write(info.gs_info.in_vertex_data_size);
        writer.Write(info.gs_info.out_vertex_data_size);
        WriteEnum(writer, info.gs_info.in_primitive);
        for (const auto primitive : info.gs_info.out_primitive) {
            WriteEnum(writer, primitive);
        }
        WriteEnum(writer, info.gs_info.mode);
        WriteCount(writer, info.gs_info.vs_copy.size());
        for (const u32 instruction : info.gs_info.vs_copy) {
            writer.Write(instruction);
        }
        writer.Write(info.gs_info.vs_copy_hash);
        break;
    case Stage::Fragment:
        WritePsInputFlags(writer, info.fs_info.en_flags);
        WritePsInputFlags(writer, info.fs_info.addr_flags);
        writer.Write(info.fs_info.num_inputs);
        for (u32 index = 0;
             index < std::min<u32>(info.fs_info.num_inputs, info.fs_info.inputs.size()); ++index) {
            const auto& input = info.fs_info.inputs[index];
            writer.Write(input.param_index);
            WriteBool(writer, input.is_default);
            WriteBool(writer, input.is_flat);
            writer.Write(input.default_value);
        }
        for (const auto& color : info.fs_info.color_buffers) {
            WritePsColorBuffer(writer, color);
        }
        WriteEnum(writer, info.fs_info.z_export_format);
        writer.Write(info.fs_info.mrtz_mask);
        WriteBool(writer, info.fs_info.dual_source_blending);
        WriteBool(writer, info.fs_info.clip_distance_emulation);
        break;
    case Stage::Compute:
        writer.Write(info.cs_info.shared_memory_size);
        for (const u32 size : info.cs_info.workgroup_size) {
            writer.Write(size);
        }
        for (const bool enabled : info.cs_info.tgid_enable) {
            WriteBool(writer, enabled);
        }
        break;
    }
}

[[nodiscard]] bool ReadRuntimeInfo(Serialization::Reader& reader, RuntimeInfo& info,
                                   std::vector<u32>& owned_vs_copy) {
    RuntimeInfo decoded{};
    std::vector<u32> decoded_vs_copy;
    if (!ReadCommonRuntimeInfo(reader, decoded)) {
        return false;
    }

    switch (decoded.stage) {
    case Stage::Local:
        if (!reader.Read(decoded.ls_info.ls_stride)) {
            return false;
        }
        break;
    case Stage::Export:
        if (!ReadCommonEsVsInfo(reader, decoded) ||
            !reader.Read(decoded.es_info.vertex_data_size)) {
            return false;
        }
        break;
    case Stage::Vertex:
        if (!ReadCommonEsVsInfo(reader, decoded) || !reader.Read(decoded.vs_info.num_outputs) ||
            decoded.vs_info.num_outputs > decoded.vs_info.outputs.size()) {
            return false;
        }
        for (auto& outputs : decoded.vs_info.outputs) {
            if (!ReadOutputMap(reader, outputs)) {
                return false;
            }
        }
        if (!ReadBool(reader, decoded.vs_info.tess_emulated_primitive) ||
            !ReadBool(reader, decoded.vs_info.emulate_depth_negative_one_to_one) ||
            !ReadBool(reader, decoded.vs_info.clip_disable) ||
            !reader.Read(decoded.vs_info.step_rate_0) ||
            !reader.Read(decoded.vs_info.step_rate_1) ||
            !reader.Read(decoded.vs_info.user_clip_plane_mask)) {
            return false;
        }
        break;
    case Stage::Hull: {
        AmdGpu::TessellationType tess_type{};
        bool offchip_lds_enable{};
        if (!ReadCommonHsEsVsInfo(reader, decoded) ||
            !reader.Read(decoded.hs_info.num_input_control_points) ||
            !reader.Read(decoded.hs_info.num_threads) ||
            !ReadEnum(reader, tess_type,
                      [](u32 value) {
                          return value <= static_cast<u32>(AmdGpu::TessellationType::Quad);
                      }) ||
            !ReadBool(reader, offchip_lds_enable) || !reader.Read(decoded.hs_info.ls_stride) ||
            !reader.Read(decoded.hs_info.hs_output_base)) {
            return false;
        }
        decoded.hs_info.tess_type = tess_type;
        decoded.hs_info.offchip_lds_enable = offchip_lds_enable;
        break;
    }
    case Stage::Geometry: {
        if (!reader.Read(decoded.gs_info.num_outputs) ||
            decoded.gs_info.num_outputs > decoded.gs_info.outputs.size()) {
            return false;
        }
        for (auto& outputs : decoded.gs_info.outputs) {
            if (!ReadOutputMap(reader, outputs)) {
                return false;
            }
        }
        AmdGpu::PrimitiveType in_primitive{};
        std::array<AmdGpu::GsOutputPrimitiveType, GsMaxOutputStreams> out_primitive{};
        AmdGpu::GsScenario mode{};
        if (!reader.Read(decoded.gs_info.num_invocations) ||
            !reader.Read(decoded.gs_info.output_vertices) ||
            !reader.Read(decoded.gs_info.in_vertex_data_size) ||
            !reader.Read(decoded.gs_info.out_vertex_data_size) ||
            !ReadEnum(reader, in_primitive, IsValidPrimitiveType)) {
            return false;
        }
        for (auto& primitive : out_primitive) {
            if (!ReadEnum(reader, primitive, [](u32 value) {
                    return value <= static_cast<u32>(AmdGpu::GsOutputPrimitiveType::TriangleStrip);
                })) {
                return false;
            }
        }
        if (!ReadEnum(reader, mode, [](u32 value) {
                return value <= static_cast<u32>(AmdGpu::GsScenario::ScenarioC);
            })) {
            return false;
        }
        u32 copy_size{};
        if (!ReadCount(reader, copy_size, MaxGeometryCopyDwords)) {
            return false;
        }
        try {
            decoded_vs_copy.resize(copy_size);
        } catch (const std::exception&) {
            return false;
        }
        for (u32& instruction : decoded_vs_copy) {
            if (!reader.Read(instruction)) {
                return false;
            }
        }
        if (!reader.Read(decoded.gs_info.vs_copy_hash)) {
            return false;
        }
        decoded.gs_info.in_primitive = in_primitive;
        decoded.gs_info.out_primitive = out_primitive;
        decoded.gs_info.mode = mode;
        break;
    }
    case Stage::Fragment: {
        if (!ReadPsInputFlags(reader, decoded.fs_info.en_flags) ||
            !ReadPsInputFlags(reader, decoded.fs_info.addr_flags) ||
            !reader.Read(decoded.fs_info.num_inputs) ||
            decoded.fs_info.num_inputs > decoded.fs_info.inputs.size()) {
            return false;
        }
        for (u32 index = 0; index < decoded.fs_info.num_inputs; ++index) {
            auto& input = decoded.fs_info.inputs[index];
            if (!reader.Read(input.param_index) || input.param_index >= IR::NumParams ||
                !ReadBool(reader, input.is_default) || !ReadBool(reader, input.is_flat) ||
                !reader.Read(input.default_value) || input.default_value > 3) {
                return false;
            }
        }
        for (auto& color : decoded.fs_info.color_buffers) {
            if (!ReadPsColorBuffer(reader, color)) {
                return false;
            }
        }
        AmdGpu::ShaderExportFormat z_export_format{};
        bool dual_source_blending{};
        bool clip_distance_emulation{};
        if (!ReadEnum(reader, z_export_format, IsValidShaderExportFormat) ||
            !reader.Read(decoded.fs_info.mrtz_mask) || !ReadBool(reader, dual_source_blending) ||
            !ReadBool(reader, clip_distance_emulation)) {
            return false;
        }
        decoded.fs_info.z_export_format = z_export_format;
        decoded.fs_info.dual_source_blending = dual_source_blending;
        decoded.fs_info.clip_distance_emulation = clip_distance_emulation;
        break;
    }
    case Stage::Compute:
        if (!reader.Read(decoded.cs_info.shared_memory_size)) {
            return false;
        }
        for (u32& size : decoded.cs_info.workgroup_size) {
            if (!reader.Read(size)) {
                return false;
            }
        }
        for (bool& enabled : decoded.cs_info.tgid_enable) {
            if (!ReadBool(reader, enabled)) {
                return false;
            }
        }
        break;
    }

    if (!ValidateRuntimeInfo(decoded)) {
        return false;
    }
    owned_vs_copy = std::move(decoded_vs_copy);
    if (decoded.stage == Stage::Geometry) {
        decoded.gs_info.vs_copy = owned_vs_copy;
    }
    info = decoded;
    return true;
}

void WriteVertexAttribute(Serialization::Writer& writer, const Gcn::VertexAttribute& attribute) {
    writer.Write(attribute.semantic);
    writer.Write(attribute.dest_vgpr);
    writer.Write(attribute.num_elements);
    writer.Write(attribute.sgpr_base);
    writer.Write(attribute.dword_offset);
    writer.Write(attribute.instance_data);
    writer.Write(attribute.inst_offset);
    writer.Write(attribute.data_format);
    writer.Write(attribute.num_format);
}

[[nodiscard]] bool ReadVertexAttribute(Serialization::Reader& reader,
                                       Gcn::VertexAttribute& attribute) {
    Gcn::VertexAttribute decoded{};
    if (!reader.Read(decoded.semantic) || !reader.Read(decoded.dest_vgpr) ||
        !reader.Read(decoded.num_elements) || !reader.Read(decoded.sgpr_base) ||
        !reader.Read(decoded.dword_offset) || !reader.Read(decoded.instance_data) ||
        !reader.Read(decoded.inst_offset) || !reader.Read(decoded.data_format) ||
        !reader.Read(decoded.num_format)) {
        return false;
    }
    attribute = decoded;
    return true;
}

void WriteVsAttribSpecialization(Serialization::Writer& writer,
                                 const VsAttribSpecialization& attribute) {
    writer.Write(attribute.divisor);
    WriteEnum(writer, attribute.num_class);
    WriteCompMapping(writer, attribute.dst_select);
}

[[nodiscard]] bool ReadVsAttribSpecialization(Serialization::Reader& reader,
                                              VsAttribSpecialization& attribute) {
    VsAttribSpecialization decoded{};
    if (!reader.Read(decoded.divisor) || !ReadEnum(reader, decoded.num_class, IsValidNumberClass) ||
        !ReadCompMapping(reader, decoded.dst_select)) {
        return false;
    }
    attribute = decoded;
    return true;
}

void WriteBufferSpecialization(Serialization::Writer& writer, const BufferSpecialization& buffer) {
    writer.Write(static_cast<u32>(buffer.stride));
    WriteBool(writer, buffer.is_storage);
    WriteBool(writer, buffer.is_formatted);
    WriteBool(writer, buffer.swizzle_enable);
    WriteBool(writer, buffer.add_tid_enable);
    if (buffer.is_formatted) {
        writer.Write(static_cast<u32>(buffer.data_format));
        writer.Write(static_cast<u32>(buffer.num_format));
        WriteCompMapping(writer, buffer.dst_select);
        WriteEnum(writer, buffer.num_conversion);
    }
    if (buffer.swizzle_enable) {
        writer.Write(static_cast<u32>(buffer.index_stride));
        writer.Write(static_cast<u32>(buffer.element_size));
    }
}

[[nodiscard]] bool ReadBufferSpecialization(Serialization::Reader& reader,
                                            BufferSpecialization& buffer) {
    u32 stride{};
    bool is_storage{};
    bool is_formatted{};
    bool swizzle_enable{};
    bool add_tid_enable{};
    if (!reader.Read(stride) || stride > 0x3FFFU || !ReadBool(reader, is_storage) ||
        !ReadBool(reader, is_formatted) || !ReadBool(reader, swizzle_enable) ||
        !ReadBool(reader, add_tid_enable)) {
        return false;
    }

    u32 data_format{};
    u32 num_format{};
    AmdGpu::CompMapping dst_select{};
    AmdGpu::NumberConversion num_conversion{};
    if (is_formatted &&
        (!reader.Read(data_format) || !IsValidDataFormat(data_format) || !reader.Read(num_format) ||
         !IsValidNumberFormat(num_format) || !ReadCompMapping(reader, dst_select) ||
         !ReadEnum(reader, num_conversion, IsValidNumberConversion))) {
        return false;
    }

    u32 index_stride{};
    u32 element_size{};
    if (swizzle_enable && (!reader.Read(index_stride) || index_stride > 3 ||
                           !reader.Read(element_size) || element_size > 3)) {
        return false;
    }

    BufferSpecialization decoded{};
    decoded.stride = stride;
    decoded.is_storage = is_storage;
    decoded.is_formatted = is_formatted;
    decoded.swizzle_enable = swizzle_enable;
    decoded.add_tid_enable = add_tid_enable;
    decoded.data_format = data_format;
    decoded.num_format = num_format;
    decoded.dst_select = dst_select;
    decoded.num_conversion = num_conversion;
    decoded.index_stride = index_stride;
    decoded.element_size = element_size;
    buffer = decoded;
    return true;
}

void WriteImageSpecialization(Serialization::Writer& writer, const ImageSpecialization& image) {
    WriteEnum(writer, image.type);
    WriteEnum(writer, image.number_class);
    WriteBool(writer, image.is_storage);
    WriteBool(writer, image.is_cube);
    WriteBool(writer, image.is_srgb);
    WriteCompMapping(writer, image.dst_select);
    WriteEnum(writer, image.num_conversion);
    WriteEnum(writer, image.atomic_data_format);
    WriteEnum(writer, image.atomic_number_format);
    writer.Write(image.num_bindings);
    writer.Write(image.dynamic_descriptor_count);
    writer.Write(image.dynamic_descriptor_stride_dw);
    WriteBool(writer, image.requires_dynamic_oob_guard);
}

[[nodiscard]] bool ReadImageSpecialization(Serialization::Reader& reader,
                                           ImageSpecialization& image) {
    ImageSpecialization decoded{};
    if (!ReadEnum(reader, decoded.type, IsValidImageType) ||
        !ReadEnum(reader, decoded.number_class, IsValidNumberClass) ||
        !ReadBool(reader, decoded.is_storage) || !ReadBool(reader, decoded.is_cube) ||
        !ReadBool(reader, decoded.is_srgb) || !ReadCompMapping(reader, decoded.dst_select) ||
        !ReadEnum(reader, decoded.num_conversion, IsValidNumberConversion) ||
        !ReadEnum(reader, decoded.atomic_data_format, IsValidDataFormat) ||
        !ReadEnum(reader, decoded.atomic_number_format, IsValidNumberFormat) ||
        !reader.Read(decoded.num_bindings) || decoded.num_bindings < 1 ||
        decoded.num_bindings > MaxImageDescriptorBindings ||
        !reader.Read(decoded.dynamic_descriptor_count) ||
        !reader.Read(decoded.dynamic_descriptor_stride_dw) ||
        !ReadBool(reader, decoded.requires_dynamic_oob_guard) ||
        decoded.dynamic_descriptor_count > NUM_IMAGES ||
        (decoded.dynamic_descriptor_count == 0 &&
         (decoded.dynamic_descriptor_stride_dw != 0 || decoded.requires_dynamic_oob_guard)) ||
        (decoded.dynamic_descriptor_count != 0 &&
         (decoded.dynamic_descriptor_stride_dw == 0 ||
          decoded.num_bindings != decoded.dynamic_descriptor_count)) ||
        !IsValidAtomicImageFormat(decoded)) {
        return false;
    }
    image = decoded;
    return true;
}

void WriteFMaskSpecialization(Serialization::Writer& writer, const FMaskSpecialization& fmask) {
    writer.Write(fmask.width);
    writer.Write(fmask.height);
}

[[nodiscard]] bool ReadFMaskSpecialization(Serialization::Reader& reader,
                                           FMaskSpecialization& fmask) {
    FMaskSpecialization decoded{};
    if (!reader.Read(decoded.width) || !reader.Read(decoded.height)) {
        return false;
    }
    fmask = decoded;
    return true;
}

void WriteSamplerSpecialization(Serialization::Writer& writer,
                                const SamplerSpecialization& sampler) {
    WriteBool(writer, sampler.force_unnormalized);
    WriteBool(writer, sampler.force_degamma);
}

[[nodiscard]] bool ReadSamplerSpecialization(Serialization::Reader& reader,
                                             SamplerSpecialization& sampler) {
    bool force_unnormalized{};
    bool force_degamma{};
    if (!ReadBool(reader, force_unnormalized) || !ReadBool(reader, force_degamma)) {
        return false;
    }
    SamplerSpecialization decoded{};
    decoded.force_unnormalized = force_unnormalized;
    decoded.force_degamma = force_degamma;
    sampler = decoded;
    return true;
}

} // namespace

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer fetch{ar};
    fetch.Write(size);
    fetch.Write(vertex_offset_sgpr);
    fetch.Write(instance_offset_sgpr);
    WriteRecords(fetch, attributes, WriteVertexAttribute);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader fetch{ar};
    FetchShaderData decoded{};
    if (!fetch.Read(decoded.size) || !fetch.Read(decoded.vertex_offset_sgpr) ||
        !fetch.Read(decoded.instance_offset_sgpr) ||
        !ReadRecords(fetch, decoded.attributes, MaxFetchAttributes, ReadVertexAttribute) ||
        !fetch.IsValid() || !ValidateFetchShaderData(decoded)) {
        return false;
    }
    *this = std::move(decoded);
    return true;
}

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer spec{ar};
    spec.Write(start.unified);
    spec.Write(start.buffer);
    spec.Write(start.user_data);
    WriteRuntimeInfo(spec, runtime_info);
    spec.Write(user_data_mask);

    for (size_t word_index = 0; word_index < MaxStageResources / 64; ++word_index) {
        u64 word{};
        for (size_t bit_index = 0; bit_index < 64; ++bit_index) {
            word |= static_cast<u64>(bitset.test(word_index * 64 + bit_index)) << bit_index;
        }
        spec.Write(word);
    }

    if (fetch_shader_data) {
        WriteBool(spec, true);
        fetch_shader_data->Serialize(ar);
    } else {
        WriteBool(spec, false);
    }

    WriteRecords(spec, vs_attribs, WriteVsAttribSpecialization);
    WriteRecords(spec, buffers, WriteBufferSpecialization);
    WriteRecords(spec, images, WriteImageSpecialization);
    WriteRecords(spec, fmasks, WriteFMaskSpecialization);
    WriteRecords(spec, samplers, WriteSamplerSpecialization);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader spec{ar};
    StageSpecialization decoded{};
    decoded.info = info;
    if (!spec.Read(decoded.start.unified) || !spec.Read(decoded.start.buffer) ||
        !spec.Read(decoded.start.user_data) ||
        !ReadRuntimeInfo(spec, decoded.runtime_info, decoded.owned_runtime_vs_copy) ||
        !spec.Read(decoded.user_data_mask)) {
        return false;
    }

    for (size_t word_index = 0; word_index < MaxStageResources / 64; ++word_index) {
        u64 word{};
        if (!spec.Read(word)) {
            return false;
        }
        for (size_t bit_index = 0; bit_index < 64; ++bit_index) {
            decoded.bitset.set(word_index * 64 + bit_index, (word >> bit_index) & 1U);
        }
    }

    bool has_fetch_shader{};
    if (!ReadBool(spec, has_fetch_shader)) {
        return false;
    }
    if (has_fetch_shader) {
        Gcn::FetchShaderData fetch_data;
        if (!fetch_data.Deserialize(ar)) {
            return false;
        }
        decoded.fetch_shader_data = std::move(fetch_data);
    }

    if (!ReadRecords(spec, decoded.vs_attribs, MaxFetchAttributes, ReadVsAttribSpecialization) ||
        !ReadRecords(spec, decoded.buffers, NUM_BUFFERS, ReadBufferSpecialization) ||
        !ReadRecords(spec, decoded.images, NUM_IMAGES, ReadImageSpecialization) ||
        !ReadRecords(spec, decoded.fmasks, NUM_FMASKS, ReadFMaskSpecialization) ||
        !ReadRecords(spec, decoded.samplers, NUM_SAMPLERS, ReadSamplerSpecialization) ||
        !spec.IsValid()) {
        return false;
    }

    *this = std::move(decoded);
    return true;
}

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};
    WriteInfoPersistent(info, *this);
    WriteCount(info, flattened_ud_buf.size());
    for (const u32 value : flattened_ud_buf) {
        info.Write(value);
    }
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader info{ar};
    Info decoded{};
    if (!ReadInfoPersistent(info, decoded)) {
        return false;
    }
    u32 flattened_size{};
    if (!ReadCount(info, flattened_size, MaxFlattenedUserDataDwords)) {
        return false;
    }
    try {
        decoded.flattened_ud_buf.resize(flattened_size);
    } catch (const std::exception&) {
        return false;
    }
    for (u32& value : decoded.flattened_ud_buf) {
        if (!info.Read(value)) {
            return false;
        }
    }
    if (!info.IsValid() || !decoded.srt_info.Deserialize(ar) || !ValidateInfo(decoded)) {
        return false;
    }
    *this = std::move(decoded);
    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};
    // Native walker code is process- and build-specific executable state. Persist only the
    // declarative snapshot layout; cached metadata always recompiles a walker when needed.
    srt.Write(u32{0});
    srt.Write(flattened_bufsize_dw);
    WriteCount(srt, range_reservations.size());
    for (const auto& reservation : range_reservations) {
        srt.Write(reservation.pointer_lo_flat_index);
        srt.Write(reservation.source_offset_dw);
        srt.Write(reservation.destination_flat_offset_dw);
        srt.Write(reservation.span_dw);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader srt{ar};
    u32 code_size{};
    u32 flattened_size{};
    u32 reservation_count{};
    if (!srt.Read(code_size) || !srt.Read(flattened_size) || code_size != 0 ||
        !ReadCount(srt, reservation_count, MaxRangeReservations)) {
        return false;
    }

    PersistentSrtInfo decoded{};
    decoded.flattened_bufsize_dw = flattened_size;
    try {
        decoded.range_reservations.reserve(reservation_count);
        for (u32 index = 0; index < reservation_count; ++index) {
            auto& reservation = decoded.range_reservations.emplace_back();
            if (!srt.Read(reservation.pointer_lo_flat_index) ||
                !srt.Read(reservation.source_offset_dw) ||
                !srt.Read(reservation.destination_flat_offset_dw) ||
                !srt.Read(reservation.span_dw)) {
                return false;
            }
        }
    } catch (const std::exception&) {
        return false;
    }
    if (!srt.IsValid() || !decoded.ValidateRangeReservations()) {
        return false;
    }

    *this = std::move(decoded);
    return true;
}

bool SerializeShaderCacheMetadata(Serialization::Archive& ar, const Info& info,
                                  const StageSpecialization& spec, u64 permutation_id) {
    if (info.srt_info.walker_func || info.srt_info.walker_func_size != 0 ||
        !info.srt_info.ValidateRangeReservations() ||
        info.srt_info.flattened_bufsize_dw != info.flattened_ud_buf.size()) {
        return false;
    }

    Serialization::Archive encoded;
    Serialization::Writer meta{encoded};
    meta.Write(ShaderMetaVersion);
    meta.Write(ShaderBinaryVersion);
    meta.Write(permutation_id);
    spec.Serialize(encoded);
    info.Serialize(encoded);
    ar.Merge(encoded);
    return true;
}

bool DeserializeShaderCacheMetadata(Serialization::Archive& ar, Info& info,
                                    std::optional<Gcn::FetchShaderData>& fetch_shader_data,
                                    StageSpecialization& spec, u64& permutation_id) {
    Serialization::Reader meta{ar};
    u32 meta_version{};
    u32 binary_version{};
    u64 decoded_permutation_id{};
    if (!meta.Read(meta_version) || !meta.Read(binary_version) ||
        meta_version != ShaderMetaVersion || binary_version != ShaderBinaryVersion ||
        !meta.Read(decoded_permutation_id)) {
        return false;
    }

    Info decoded_info{};
    StageSpecialization decoded_spec{};
    decoded_spec.info = &decoded_info;
    if (!decoded_spec.Deserialize(ar) || !decoded_info.Deserialize(ar) ||
        !ValidateShaderCacheMetadata(decoded_info, decoded_spec) || !ar.IsEoS()) {
        return false;
    }

    info = std::move(decoded_info);
    spec = std::move(decoded_spec);
    spec.info = &info;
    fetch_shader_data = spec.fetch_shader_data;
    permutation_id = decoded_permutation_id;
    return true;
}

bool ValidateShaderCacheMetadata(const Info& info, const StageSpecialization& spec) noexcept {
    if (!ValidateInfo(info) || !ValidateRuntimeInfo(spec.runtime_info)) {
        return false;
    }
    const bool has_fetch_shader = spec.fetch_shader_data.has_value();
    const bool resource_shapes_match =
        info.buffers.size() == spec.buffers.size() && info.images.size() == spec.images.size() &&
        info.fmasks.size() == spec.fmasks.size() && info.samplers.size() == spec.samplers.size();
    const bool atomic_formats_match =
        resource_shapes_match &&
        std::ranges::equal(
            info.images, spec.images,
            [](const ImageResource& resource, const ImageSpecialization& image) {
                const bool has_exact_format =
                    image.atomic_data_format != AmdGpu::DataFormat::FormatInvalid;
                return resource.is_atomic == has_exact_format && IsValidAtomicImageFormat(image) &&
                       (!resource.is_atomic ||
                        image.number_class == AmdGpu::GetNumberClass(image.atomic_number_format));
            });
    const bool image_binding_counts_match =
        resource_shapes_match &&
        std::ranges::equal(
            info.images, spec.images,
            [&](const ImageResource& resource, const ImageSpecialization& image) {
                return image.num_bindings >= 1 &&
                       image.num_bindings <= MaxImageDescriptorBindings &&
                       resource.NumBindings(info) == image.num_bindings &&
                       resource.dynamic_descriptor_count == image.dynamic_descriptor_count &&
                       resource.dynamic_descriptor_stride_dw ==
                           image.dynamic_descriptor_stride_dw &&
                       resource.requires_dynamic_oob_guard == image.requires_dynamic_oob_guard;
            });
    return resource_shapes_match && atomic_formats_match && image_binding_counts_match &&
           info.stage == spec.runtime_info.stage &&
           IsPhysicalStageCompatible(info.stage, info.l_stage) &&
           info.ud_mask.mask == spec.user_data_mask && info.has_fetch_shader == has_fetch_shader &&
           (!has_fetch_shader || info.l_stage == LogicalStage::Vertex) &&
           (!has_fetch_shader ||
            (ValidateFetchShaderData(*spec.fetch_shader_data) &&
             spec.fetch_shader_data->attributes.size() <= MaxFetchAttributes &&
             spec.fetch_shader_data->attributes.size() == spec.vs_attribs.size()));
}

} // namespace Shader
