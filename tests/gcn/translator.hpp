// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <span>
#include <vector>

#include "common/types.h"

struct TranslationEnvironment {
    u32 subgroup_size{32};
    std::array<u32, 3> workgroup_size{1, 1, 1};
    bool force_dma_helpers{};
};

std::vector<u32> TranslateToSpirv(u64 raw_gcn_inst);
std::vector<u32> TranslateToSpirv(std::span<const u64> raw_gcn_insts);
std::vector<u32> TranslateToSpirv(u64 raw_gcn_inst, const TranslationEnvironment& environment);
std::vector<u32> TranslateToSpirv(std::span<const u64> raw_gcn_insts,
                                  const TranslationEnvironment& environment);
