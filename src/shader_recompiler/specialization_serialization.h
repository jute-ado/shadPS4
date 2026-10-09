// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>

#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

struct Info;
struct StageSpecialization;
namespace Gcn {
struct FetchShaderData;
}

inline constexpr u32 ShaderBinaryVersion = 8u;
inline constexpr u32 ShaderMetaVersion = 8u;

[[nodiscard]] bool SerializeShaderCacheMetadata(Serialization::Archive& ar, const Info& info,
                                                const StageSpecialization& spec,
                                                u64 permutation_id);

[[nodiscard]] bool DeserializeShaderCacheMetadata(
    Serialization::Archive& ar, Info& info, std::optional<Gcn::FetchShaderData>& fetch_shader_data,
    StageSpecialization& spec, u64& permutation_id);

[[nodiscard]] bool ValidateShaderCacheMetadata(const Info& info,
                                               const StageSpecialization& spec) noexcept;

} // namespace Shader
