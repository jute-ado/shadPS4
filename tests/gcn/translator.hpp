// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>

#include "common/types.h"

struct TranslationResult {
    std::vector<u32> spirv;
    std::size_t guest_buffer_count;
    u32 addr64_pointer_root_count;
    bool addr64_pointer_roots_overflow;
    bool uses_dma;
};

std::vector<u32> TranslateToSpirv(u64 raw_gcn_inst);
std::vector<u32> TranslateToSpirv(std::span<const u64> raw_gcn_insts);
TranslationResult TranslateToSpirvWithInfo(u64 raw_gcn_inst);
TranslationResult TranslateToSpirvWithInfo(u64 raw_gcn_inst, bool direct_memory_access);
TranslationResult TranslateToSpirvWithInfo(u64 raw_gcn_inst, bool direct_memory_access,
                                           u64 source_buffer_base);
TranslationResult TranslateToSpirvWithInfo(std::span<const u64> raw_gcn_insts);
