// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/kernel/memory.h"

namespace Libraries::Kernel {
namespace {

TEST(KernelMemoryValidation, RejectsMissingNamedDirectMemoryPointers) {
    void* address{};

    EXPECT_FALSE(IsMemoryAddressStorageValid(nullptr));
    EXPECT_FALSE(AreNamedMemoryPointersValid(nullptr, "anon"));
    EXPECT_FALSE(AreNamedMemoryPointersValid(&address, nullptr));
}

TEST(KernelMemoryValidation, AcceptsAddressStorageAndName) {
    void* address{};

    EXPECT_TRUE(IsMemoryAddressStorageValid(&address));
    EXPECT_TRUE(AreNamedMemoryPointersValid(&address, "anon"));
}

} // namespace
} // namespace Libraries::Kernel
