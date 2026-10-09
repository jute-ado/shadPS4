// CPU-only RED specification for indirect command byte-range admission.
//
// This file intentionally lives in a scratch worktree.  It does not include Vulkan or construct
// an emulator, and the source checks below never execute production code.  The model captures the
// range contract that the indirect draw, indexed draw, and dispatch paths need before they call
// BufferCache::ObtainBuffer.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Address = std::uintptr_t;
using U32 = std::uint32_t;
using U64 = std::uint64_t;

static_assert(std::numeric_limits<Address>::digits >= 32);

struct ByteRange {
    Address start;
    U64 size;
};

// ObtainBuffer currently accepts a u32 byte count.  Keep that limitation explicit in the model:
// widening that API is an alternative implementation, but silently narrowing a checked u64 is
// not admissible.
constexpr U64 kObtainBufferMaxSize = std::numeric_limits<U32>::max();

[[nodiscard]] std::optional<U64> CheckedMultiply(U64 lhs, U64 rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<U64>::max() / lhs) {
        return std::nullopt;
    }
    return lhs * rhs;
}

[[nodiscard]] std::optional<ByteRange> CheckedRange(Address base, U64 offset, U64 size) {
    const U64 max_address = std::numeric_limits<Address>::max();
    if (offset > max_address - base) {
        return std::nullopt;
    }

    const Address start = static_cast<Address>(base + offset);
    if (size > max_address - start) {
        return std::nullopt;
    }
    return ByteRange{start, size};
}

[[nodiscard]] std::optional<ByteRange> CheckedBufferRange(Address base, U64 offset, U64 size) {
    if (size > kObtainBufferMaxSize) {
        return std::nullopt;
    }
    return CheckedRange(base, offset, size);
}

[[nodiscard]] std::optional<ByteRange> AdmitIndirectDraw(Address base, U32 offset, U32 stride,
                                                          U32 max_count) {
    const auto byte_count = CheckedMultiply(stride, max_count);
    if (!byte_count) {
        return std::nullopt;
    }
    return CheckedBufferRange(base, offset, *byte_count);
}

[[nodiscard]] std::optional<ByteRange> AdmitIndexedDraw(Address base, U32 index_offset,
                                                         U32 index_size, U32 index_count) {
    const auto byte_offset = CheckedMultiply(index_offset, index_size);
    const auto byte_count = CheckedMultiply(index_count, index_size);
    if (!byte_offset || !byte_count) {
        return std::nullopt;
    }
    // The byte offset is a guest address displacement and may exceed u32.  Only the requested
    // byte count is constrained by the current ObtainBuffer API.
    return CheckedBufferRange(base, *byte_offset, *byte_count);
}

constexpr U32 kDispatchDimensionsBytes = sizeof(U32) * 3;
constexpr U32 kIndirectCountBytes = sizeof(U32);

[[nodiscard]] std::optional<ByteRange> AdmitDispatch(Address base, U32 offset) {
    return CheckedBufferRange(base, offset, kDispatchDimensionsBytes);
}

[[nodiscard]] std::optional<ByteRange> AdmitCountRead(Address base) {
    return CheckedBufferRange(base, 0, kIndirectCountBytes);
}

[[nodiscard]] bool SameRange(const std::optional<ByteRange>& actual, Address start, U64 size) {
    return actual && actual->start == start && actual->size == size;
}

void Expect(bool condition, std::string_view name, int& failures) {
    if (!condition) {
        std::cerr << "MODEL FAIL: " << name << '\n';
        ++failures;
    }
}

[[nodiscard]] std::string ReadFile(const char* path) {
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        return {};
    }
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] std::vector<std::size_t> LinesContaining(std::string_view source,
                                                       std::string_view token,
                                                       bool identifier_boundary = false) {
    std::vector<std::size_t> lines;
    std::size_t search_from = 0;
    for (;;) {
        const auto position = source.find(token, search_from);
        if (position == std::string_view::npos) {
            break;
        }
        if (identifier_boundary && position != 0) {
            const char preceding = source[position - 1];
            const bool is_identifier =
                (preceding >= 'a' && preceding <= 'z') ||
                (preceding >= 'A' && preceding <= 'Z') ||
                (preceding >= '0' && preceding <= '9') || preceding == '_';
            if (is_identifier) {
                search_from = position + token.size();
                continue;
            }
        }
        std::size_t line = 1;
        for (std::size_t i = 0; i < position; ++i) {
            line += source[i] == '\n';
        }
        lines.push_back(line);
        search_from = position + token.size();
    }
    return lines;
}

struct SourceCheck {
    const char* label;
    const char* path;
    std::string_view forbidden_token;
    const char* requirement;
    bool identifier_boundary = false;
};

int CheckSource(const SourceCheck& check) {
    const auto source = ReadFile(check.path);
    if (source.empty()) {
        std::cerr << "SOURCE ERROR: could not read " << check.path << '\n';
        return 2;
    }

    const auto lines =
        LinesContaining(source, check.forbidden_token, check.identifier_boundary);
    if (lines.empty()) {
        std::cout << "SOURCE PASS: " << check.label << '\n';
        return 0;
    }

    std::cerr << "RED: " << check.label << " still contains '" << check.forbidden_token
              << "' at line(s)";
    for (const auto line : lines) {
        std::cerr << ' ' << line;
    }
    std::cerr << " (" << check.requirement << ")\n";
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: test_indirect_range_admission_red.exe <vk_rasterizer.cpp> "
                     "<buffer_cache.cpp>\n";
        return 2;
    }

    int model_failures = 0;
    constexpr Address kMaxAddress = std::numeric_limits<Address>::max();

    // Ordinary ranges retain the exact address and byte count that a future production helper
    // should pass to ObtainBuffer.
    Expect(SameRange(AdmitIndirectDraw(0x1000, 0x20, 0x10, 2), 0x1020, 0x20),
           "ordinary indirect draw", model_failures);
    Expect(SameRange(AdmitIndexedDraw(0x2000, 3, 2, 4), 0x2006, 8), "ordinary indexed draw",
           model_failures);
    Expect(SameRange(AdmitDispatch(0x4000, 0x80), 0x4080, kDispatchDimensionsBytes),
           "ordinary indirect dispatch", model_failures);
    Expect(SameRange(AdmitCountRead(0x5000), 0x5000, kIndirectCountBytes),
           "ordinary indirect count read", model_failures);

    // The current draw path evaluates stride * max_count as u32.  The model must reject the
    // mathematically valid 8,589,934,590-byte request instead of accepting its 32-bit residue.
    Expect(!AdmitIndirectDraw(0x1000, 0, std::numeric_limits<U32>::max(), 2),
           "indirect draw byte-count does not narrow after multiplication", model_failures);

    // The current draw and dispatch paths add a guest address and offset without checking either
    // the intermediate sum or the final end address.
    Expect(!AdmitIndirectDraw(kMaxAddress - 3, 4, 0x10, 1),
           "indirect draw base plus offset overflow", model_failures);
    Expect(!AdmitIndirectDraw(kMaxAddress - 0x10, 0, 0x20, 1),
           "indirect draw end address overflow", model_failures);
    Expect(!AdmitDispatch(kMaxAddress - 8, 1), "indirect dispatch end address overflow",
           model_failures);

    // Index offset arithmetic must be widened before multiplication: this valid 64-bit guest
    // displacement is 0x3fffffffc bytes, not the u32 residue 0xfffffffc.
    const auto wide_index_offset = AdmitIndexedDraw(0x1000, std::numeric_limits<U32>::max(), 4, 1);
    Expect(SameRange(wide_index_offset, static_cast<Address>(0x400000ffcULL), 4),
           "indexed draw widens byte offset before multiply", model_failures);
    Expect(!AdmitIndexedDraw(0x1000, 0, 4, std::numeric_limits<U32>::max()),
           "indexed draw byte-count does not narrow after multiplication", model_failures);
    Expect(!AdmitIndexedDraw(kMaxAddress - 2, 1, 4, 1),
           "indexed draw base plus byte offset overflow", model_failures);

    // A count-indirect draw performs a second four-byte read and needs the same end-address check.
    Expect(!AdmitCountRead(kMaxAddress - 3), "indirect count read end address overflow",
           model_failures);

    // Zero draw count is representable as an empty range; a caller may short-circuit it, but must
    // not turn a wrapped non-zero request into an apparently empty request.
    Expect(SameRange(AdmitIndirectDraw(0x7000, 0x20, 0x100, 0), 0x7020, 0),
           "zero indirect draw count remains an empty range", model_failures);

    if (model_failures != 0) {
        std::cerr << "MODEL RED: " << model_failures << " CPU contract assertion(s) failed\n";
        return 3;
    }
    std::cout << "MODEL GREEN: overflow and narrowing cases are covered\n";

    const std::vector<SourceCheck> source_checks{
        {
            "indirect draw byte-count product",
            argv[1],
            "stride * max_count",
            "multiply in a checked wide type before the ObtainBuffer byte-count boundary",
        },
        {
            "indirect draw guest-address addition",
            argv[1],
            "arg_address + offset",
            "admit the complete guest range before using its start in ObtainBuffer and diagnostics",
        },
        {
            "indirect dispatch guest-address addition",
            argv[1],
            "address + offset",
            "admit the complete 12-byte dispatch range before using its start",
            true,
        },
        {
            "indexed draw byte-offset product",
            argv[2],
            "index_offset * index_size",
            "widen index offset arithmetic before forming the guest index address",
        },
        {
            "indexed draw byte-count product",
            argv[2],
            "regs.num_indices * index_size",
            "multiply index count and element size in a checked wide type",
        },
    };

    int source_failures = 0;
    for (const auto& check : source_checks) {
        const int result = CheckSource(check);
        if (result == 2) {
            return 2;
        }
        source_failures += result;
    }

    if (source_failures != 0) {
        std::cerr << "RED SUMMARY: " << source_failures
                  << " targeted production-source assertion(s) failed; no GPU was launched\n";
        return 1;
    }

    std::cout << "GREEN: production source has no targeted unchecked range arithmetic\n";
    return 0;
}
