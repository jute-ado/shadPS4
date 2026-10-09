// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <map>
#include <string>

#include <fmt/format.h>

#include "common/io_file.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/value.h"

namespace Shader::IR {

void DumpProgram(const Program& program, const Info& info, const std::string& type) {
    using namespace Common::FS;

    // TEMP Shadow diagnostic: retain IR only for the opaque-scene and list-building shaders.
    if (!EmulatorSettings.IsDumpShaders() && info.pgm_hash != 0x675450f7 &&
        info.pgm_hash != 0x87cf9b8a && info.pgm_hash != 0x79704cba &&
        info.pgm_hash != 0x729c541e && info.pgm_hash != 0x291dbcd0 &&
        info.pgm_hash != 0xa402ddac && info.pgm_hash != 0x0654a5a1 &&
        info.pgm_hash != 0x617e09a3 && info.pgm_hash != 0xc61ae8f9 &&
        info.pgm_hash != 0xef6bb4ef && info.pgm_hash != 0xa9027daa &&
        info.pgm_hash != 0x0f76366e && info.pgm_hash != 0x944260e7 &&
        info.pgm_hash != 0x93440267 && info.pgm_hash != 0x75aad501 &&
        info.pgm_hash != 0xb33e9db6 && info.pgm_hash != 0x48c4e8ab &&
        info.pgm_hash != 0xa5a21e78 && info.pgm_hash != 0x4db42a1a &&
        info.pgm_hash != 0x862e4da8) {
        return;
    }

    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto ir_filename =
        fmt::format("{}_{:#018x}.{}irprogram.txt", info.stage, info.pgm_hash, type);
    const auto ir_file = IOFile{dump_dir / ir_filename, FileAccessMode::Create, FileType::TextFile};

    size_t index{0};
    std::map<const IR::Inst*, size_t> inst_to_index;
    std::map<const IR::Block*, size_t> block_to_index;

    for (const IR::Block* const block : program.blocks) {
        block_to_index.emplace(block, index);
        ++index;
    }

    for (const auto& block : program.blocks) {
        std::string s = IR::DumpBlock(*block, block_to_index, inst_to_index, index) + '\n';
        ir_file.WriteString(s);
    }

    const auto asl_filename = fmt::format("{}_{:#018x}.{}asl.txt", info.stage, info.pgm_hash, type);
    const auto asl_file =
        IOFile{dump_dir / asl_filename, FileAccessMode::Create, FileType::TextFile};

    for (const auto& node : program.syntax_list) {
        std::string s = IR::DumpASLNode(node, block_to_index, inst_to_index) + '\n';
        asl_file.WriteString(s);
    }
}

} // namespace Shader::IR
