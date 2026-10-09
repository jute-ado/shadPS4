// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <vector>
#include <gtest/gtest.h>
#include "core/libraries/libc_internal/libc_internal_memory.h"
#include "core/loader/symbols_resolver.h"

namespace {
struct Export {
    Core::Loader::SymbolResolver symbol;
    u64 address;
};
std::vector<Export> exports;
using Move = void*(PS4_SYSV_ABI*)(void*, const void*, size_t);

Move FindMove() {
    exports.clear();
    Core::Loader::SymbolsResolver resolver;
    Libraries::LibcInternal::RegisterlibSceLibcInternalMemory(&resolver);
    // Public AeroLib table: STUB("+P6FRGH4LfA", memmove).
    for (const auto& entry : exports) {
        if (entry.symbol.name == "+P6FRGH4LfA" && entry.symbol.library == "libSceLibcInternal" &&
            entry.symbol.module == "libSceLibcInternal" && entry.symbol.library_version == 1 &&
            entry.symbol.type == Core::Loader::SymbolType::Function) {
            return reinterpret_cast<Move>(entry.address);
        }
    }
    return nullptr;
}
} // namespace

// Registration sink only: production registration and its ABI wrapper remain under test.
void Core::Loader::SymbolsResolver::AddSymbol(const SymbolResolver& symbol, u64 address) {
    exports.push_back({symbol, address});
}

TEST(LibcInternalMemory, RegistersMemmove) {
    EXPECT_NE(FindMove(), nullptr);
}

TEST(LibcInternalMemory, OverlapAndBoundariesThroughRegisteredSysvExport) {
    const auto move = FindMove();
    ASSERT_NE(move, nullptr);
    // Includes both overlapping directions, unaligned spans, zero size, self copy,
    // adjacent/nonoverlapping spans, and every small size/offset combination.
    for (size_t source = 0; source < 24; ++source) {
        for (size_t destination = 0; destination < 24; ++destination) {
            for (size_t count = 0; count <= 32 - std::max(source, destination); ++count) {
                std::array<unsigned char, 32> bytes{};
                for (size_t i = 0; i < bytes.size(); ++i) {
                    bytes[i] = static_cast<unsigned char>(i * 7 + 3);
                }
                const auto before = bytes;
                auto expected = before;
                for (size_t i = 0; i < count; ++i) {
                    expected[destination + i] = before[source + i];
                }
                SCOPED_TRACE(::testing::Message() << source << '/' << destination << '/' << count);
                EXPECT_EQ(move(bytes.data() + destination, bytes.data() + source, count),
                          bytes.data() + destination);
                EXPECT_EQ(bytes, expected); // Every byte outside the destination is untouched.
            }
        }
    }
}

TEST(LibcInternalMemory, SeparateSourceRemainsUnchanged) {
    const auto move = FindMove();
    ASSERT_NE(move, nullptr);
    const std::array<unsigned char, 9> source{9, 8, 7, 6, 5, 4, 3, 2, 1};
    const auto before = source;
    std::array<unsigned char, 11> destination{};
    EXPECT_EQ(move(destination.data() + 1, source.data() + 1, 7), destination.data() + 1);
    EXPECT_EQ(source, before);
    EXPECT_EQ(destination, (std::array<unsigned char, 11>{0, 8, 7, 6, 5, 4, 3, 2, 0, 0, 0}));
}
