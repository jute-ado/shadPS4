// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Shader::Optimization {

template <typename Map, typename T>
void AddSrtPointerUse(Map& uses, u32 source_offset_dwords, T user) {
    uses[source_offset_dwords].push_back(user);
}

} // namespace Shader::Optimization
