// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/memory.h"

TEST(GpuMappingAdmission, OnlyGpuVisibleMappingsRequireACompletedSubmissionBoundary) {
    using Core::MemoryProt;

    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::NoAccess));
    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::CpuReadWrite));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::GpuRead));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::GpuWrite));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::CpuReadWrite |
                                                              MemoryProt::GpuReadWrite));
}
