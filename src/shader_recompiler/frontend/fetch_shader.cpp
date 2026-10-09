// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/fetch_shader.h"

namespace Shader::Gcn {

/**
 * s_load_dwordx4 s[8:11], s[2:3], 0x00
 * s_load_dwordx4 s[12:15], s[2:3], 0x04
 * s_load_dwordx4 s[16:19], s[2:3], 0x08
 * s_waitcnt     lgkmcnt(0)
 * buffer_load_format_xyzw v[4:7], v0, s[8:11], 0 idxen
 * buffer_load_format_xyz v[8:10], v0, s[12:15], 0 idxen
 * buffer_load_format_xy v[12:13], v0, s[16:19], 0 idxen
 * s_waitcnt     0
 * s_setpc_b64   s[0:1]

 * s_load_dwordx4  s[4:7], s[2:3], 0x0
 * s_waitcnt       lgkmcnt(0)
 * buffer_load_format_xyzw v[4:7], v0, s[4:7], 0 idxen
 * s_load_dwordx4  s[4:7], s[2:3], 0x8
 * s_waitcnt       lgkmcnt(0)
 * buffer_load_format_xyzw v[8:11], v0, s[4:7], 0 idxen
 * s_waitcnt       vmcnt(0) & expcnt(0) & lgkmcnt(0)
 * s_setpc_b64     s[0:1]

 * A normal fetch shader looks like the above, the instructions are generated
 * using input semantics on cpu side. Load instructions can either be separate or interleaved
 * We take the reverse way, extract the original input semantics from these instructions.
 **/

static bool IsTypedBufferLoad(const Gcn::GcnInst& inst) {
    return inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_X ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XY ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XYZ ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XYZW;
}

const u32* GetFetchShaderCode(const Info& info, u32 sgpr_base) {
    if (sgpr_base > info.user_data.size() || info.user_data.size() - sgpr_base < 2) {
        return nullptr;
    }
    const u32* code;
    std::memcpy(&code, &info.user_data[sgpr_base], sizeof(code));
    return reinterpret_cast<const u32*>(VAddr(code) & 0xFFFFFFFFFFFFULL);
}

std::optional<FetchShaderData> ParseFetchShader(const Shader::Info& info,
                                                std::span<const u32> code) {
    if (!info.has_fetch_shader || code.empty()) {
        return std::nullopt;
    }

    FetchShaderData data{};
    GcnCodeSlice code_slice{code.data(), code.data() + code.size()};
    GcnDecodeContext decoder;

    struct VsharpLoad {
        u32 dword_offset{};
        u32 base_sgpr{};
    };
    std::array<VsharpLoad, 104> loads{};

    u32 semantic_index = 0;
    while (!code_slice.atEnd()) {
        if (!code_slice.canReadWords(1)) {
            return std::nullopt;
        }
        const auto encoding = GetInstructionEncoding(code_slice.at(0));
        if (encoding == InstEncoding::ILLEGAL) {
            return std::nullopt;
        }
        const auto encoding_length = GetEncodingLength(encoding);
        if (encoding_length == 0 || encoding_length % sizeof(u32) != 0 ||
            !code_slice.canReadWords(encoding_length / sizeof(u32))) {
            return std::nullopt;
        }
        const auto inst = decoder.decodeInstruction(code_slice);
        if (!code_slice.valid() || inst.length < encoding_length ||
            inst.length % sizeof(u32) != 0) {
            return std::nullopt;
        }
        data.size += inst.length;

        if (inst.opcode == Opcode::S_SETPC_B64) {
            return data;
        }

        if (inst.inst_class == InstClass::ScalarMemRd) {
            if (inst.dst[0].code >= loads.size() || inst.src[0].code >= loads.size()) {
                return std::nullopt;
            }
            loads[inst.dst[0].code] = VsharpLoad{inst.control.smrd.offset, inst.src[0].code * 2};
            continue;
        }

        if (inst.opcode == Opcode::V_ADD_I32) {
            const auto vgpr = inst.dst[0].code;
            const auto sgpr = s8(inst.src[0].code);
            switch (vgpr) {
            case 0: // V0 is always the vertex offset
                data.vertex_offset_sgpr = sgpr;
                break;
            case 3: // V3 is always the instance offset
                data.instance_offset_sgpr = sgpr;
                break;
            default:
                return std::nullopt;
            }
        }

        if (inst.inst_class == InstClass::VectorMemBufFmt) {
            // SRSRC is in units of 4 SPGRs while SBASE is in pairs of SGPRs
            const u32 base_sgpr = inst.src[2].code * 4;
            if (base_sgpr >= loads.size() || semantic_index >= IR::NumParams) {
                return std::nullopt;
            }
            auto& attrib = data.attributes.emplace_back();
            attrib.semantic = semantic_index++;
            attrib.dest_vgpr = inst.src[1].code;
            attrib.num_elements = inst.control.mubuf.count;
            attrib.sgpr_base = loads[base_sgpr].base_sgpr;
            attrib.dword_offset = loads[base_sgpr].dword_offset;
            attrib.inst_offset = inst.control.mtbuf.offset;
            attrib.instance_data = inst.src[0].code;
            if (IsTypedBufferLoad(inst)) {
                attrib.data_format = inst.control.mtbuf.dfmt;
                attrib.num_format = inst.control.mtbuf.nfmt;
            }
        }
    }

    return std::nullopt;
}

std::optional<FetchShaderData> ParseFetchShader(const Shader::Info& info) {
    if (!info.has_fetch_shader || info.fetch_shader_sgpr_base >= info.user_data.size()) {
        return std::nullopt;
    }
    const auto* code = GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
    if (!code) {
        return std::nullopt;
    }
    return ParseFetchShader(info, std::span<const u32>{code, MaxFetchShaderWords});
}

} // namespace Shader::Gcn
