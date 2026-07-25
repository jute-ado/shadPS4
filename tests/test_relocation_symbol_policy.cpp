// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/loader/elf.h"
#include "core/loader/relocation_symbol_policy.h"

using namespace Core::Loader;

TEST(RelocationSymbolPolicy, DefinedGlobalAndWeakSymbolsResolveInsideTheirModule) {
    EXPECT_EQ(ClassifyRelocationSymbol(STB_GLOBAL, 0x1234), RelocationSymbolSource::Module);
    EXPECT_EQ(ClassifyRelocationSymbol(STB_WEAK, 0x5678), RelocationSymbolSource::Module);
}

TEST(RelocationSymbolPolicy, UndefinedGlobalSymbolsUseExternalResolution) {
    EXPECT_EQ(ClassifyRelocationSymbol(STB_GLOBAL, 0), RelocationSymbolSource::External);
}

TEST(RelocationSymbolPolicy, UndefinedWeakSymbolsHaveAZeroFallback) {
    EXPECT_EQ(ClassifyRelocationSymbol(STB_WEAK, 0), RelocationSymbolSource::UndefinedWeak);
}

TEST(RelocationSymbolPolicy, LocalSymbolsAlwaysResolveInsideTheirModule) {
    EXPECT_EQ(ClassifyRelocationSymbol(STB_LOCAL, 0), RelocationSymbolSource::Module);
    EXPECT_EQ(ClassifyRelocationSymbol(STB_LOCAL, 0x9abc), RelocationSymbolSource::Module);
}

TEST(RelocationSymbolPolicy, RejectsUnknownBindings) {
    EXPECT_EQ(ClassifyRelocationSymbol(15, 0), RelocationSymbolSource::Unsupported);
}

TEST(RelocationSymbolPolicy, TreatsSceLifecycleSymbolsAsFunctions) {
    EXPECT_TRUE(IsRelocationFunctionType(STT_FUN));
    EXPECT_TRUE(IsRelocationFunctionType(STT_SCE));
    EXPECT_FALSE(IsRelocationFunctionType(STT_NOTYPE));
    EXPECT_FALSE(IsRelocationFunctionType(STT_OBJECT));
}
