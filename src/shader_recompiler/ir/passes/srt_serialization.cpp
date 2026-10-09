// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/serdes.h"
#include "shader_recompiler/ir/passes/srt.h"

namespace Shader {

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};
    srt.Write(walker_func_size);
    srt.Write(flattened_bufsize_dw);
    srt.Write(range_reservations);
    if (walker_func_size) {
        srt.Write(reinterpret_cast<const void*>(walker_func), walker_func_size);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader srt{ar};
    walker_func = nullptr;
    range_reservations.clear();
    srt.Read(walker_func_size);
    srt.Read(flattened_bufsize_dw);
    srt.Read(range_reservations);
    if (walker_func_size) {
        walker_func = RegisterWalkerCode(ar.CurrPtr(), walker_func_size);
        ar.Advance(walker_func_size);
    }
    return true;
}

} // namespace Shader
