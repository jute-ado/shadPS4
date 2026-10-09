// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#ifdef _WIN32
#include <process.h>
#endif

#include "spirv/unified1/spirv.hpp"
#include "translator.hpp"
#include "video_core/buffer_cache/buffer_cache.h"

namespace {

constexpr u64 PageCount = VideoCore::BufferCache::CACHING_NUMPAGES;
constexpr u64 PageSize = VideoCore::BufferCache::CACHING_PAGESIZE;
constexpr u64 AddressLimit = PageCount * PageSize;

struct LookupResult {
    u64 address{};
    std::vector<u64> page_reads;
    std::vector<u64> fault_reads;
    std::vector<u64> fault_writes;
    std::unordered_map<u64, u32> faults;
};

// Execute the actual emitted helper on sparse synthetic memory, without a GPU or
// allocating its 512 MiB page table. Unsupported instructions fail the test, rather
// than silently approximating them. Out-of-range accesses remain visible in the
// trace even when the simulated storage-buffer load returns zero.
LookupResult RunLookup(u64 address, const std::unordered_map<u64, u64>& resident_pages) {
    constexpr u64 addr64_load = 0x80010000e030800cULL;
    const auto code = TranslateToSpirvWithInfo(addr64_load, true).spirv;
    std::unordered_map<u32, u64> values;
    std::unordered_map<u32, u32> widths;
    std::unordered_map<u32, size_t> labels;
    std::unordered_map<std::string, u32> names;
    for (size_t pos = 5; pos < code.size();) {
        const u32 count = code[pos] >> 16;
        if (!count || count > code.size() - pos) {
            throw std::runtime_error("Malformed emitted SPIR-V");
        }
        const auto op = static_cast<spv::Op>(code[pos] & 0xffffU);
        if (op == spv::Op::OpName) {
            names.emplace(reinterpret_cast<const char*>(&code[pos + 2]), code[pos + 1]);
        } else if (op == spv::Op::OpTypeInt) {
            widths[code[pos + 1]] = code[pos + 2];
        } else if (op == spv::Op::OpTypeBool) {
            widths[code[pos + 1]] = 1;
        } else if (op == spv::Op::OpConstant) {
            values[code[pos + 2]] = code[pos + 3];
            if (count == 5) {
                values[code[pos + 2]] |= u64{code[pos + 4]} << 32;
            }
        } else if (op == spv::Op::OpLabel) {
            labels[code[pos + 1]] = pos;
        }
        pos += count;
    }

    const u32 function = names.at("get_bda_pointer");
    const u32 page_table = names.at("bda_pagetable");
    const u32 fault_buffer = names.at("fault_buffer");
    struct Pointer {
        u32 buffer;
        u64 index;
    };
    std::unordered_map<u32, Pointer> pointers;
    LookupResult result;
    u32 current_label{}, predecessor{};
    bool in_function = false;
    for (size_t pos = 5, steps = 0; pos < code.size() && steps < code.size(); ++steps) {
        const u32 count = code[pos] >> 16;
        const auto op = static_cast<spv::Op>(code[pos] & 0xffffU);
        const u32* inst = code.data() + pos;
        if (!in_function) {
            in_function = op == spv::Op::OpFunction && inst[2] == function;
            pos += count;
            continue;
        }
        const auto value = [&](u32 operand) { return values.at(inst[operand]); };
        const auto assign = [&](u64 value) {
            const u32 width = widths.at(inst[1]);
            values[inst[2]] = width == 64 ? value : value & ((u64{1} << width) - 1);
        };
        switch (op) {
        case spv::Op::OpFunctionParameter:
            values[inst[2]] = address;
            break;
        case spv::Op::OpLabel:
            current_label = inst[1];
            break;
        case spv::Op::OpSelectionMerge:
            break;
        case spv::Op::OpShiftRightLogical:
            assign(value(3) >> value(4));
            break;
        case spv::Op::OpShiftLeftLogical:
            assign(value(3) << value(4));
            break;
        case spv::Op::OpUConvert:
            assign(value(3));
            break;
        case spv::Op::OpIAdd:
            assign(value(3) + value(4));
            break;
        case spv::Op::OpBitwiseAnd:
            assign(value(3) & value(4));
            break;
        case spv::Op::OpBitwiseOr:
            assign(value(3) | value(4));
            break;
        case spv::Op::OpIEqual:
            assign(value(3) == value(4));
            break;
        case spv::Op::OpULessThan:
            assign(value(3) < value(4));
            break;
        case spv::Op::OpUGreaterThanEqual:
            assign(value(3) >= value(4));
            break;
        case spv::Op::OpAccessChain:
            if (count != 6 || value(4) != 0) {
                throw std::runtime_error("Unexpected helper buffer layout");
            }
            pointers[inst[2]] = {inst[3], value(5)};
            break;
        case spv::Op::OpLoad: {
            const auto ptr = pointers.at(inst[3]);
            if (ptr.buffer == page_table) {
                result.page_reads.push_back(ptr.index);
                const auto it = resident_pages.find(ptr.index);
                assign(it == resident_pages.end() ? 0 : it->second);
            } else if (ptr.buffer == fault_buffer) {
                result.fault_reads.push_back(ptr.index);
                assign(result.faults[ptr.index]);
            } else {
                throw std::runtime_error("Unknown helper buffer");
            }
            break;
        }
        case spv::Op::OpStore: {
            const auto ptr = pointers.at(inst[1]);
            if (ptr.buffer != fault_buffer) {
                throw std::runtime_error("Unexpected helper write");
            }
            result.fault_writes.push_back(ptr.index);
            result.faults[ptr.index] = static_cast<u32>(value(2));
            break;
        }
        case spv::Op::OpBranch:
        case spv::Op::OpBranchConditional:
            predecessor = current_label;
            pos = labels.at(op == spv::Op::OpBranch ? inst[1] : value(1) ? inst[2] : inst[3]);
            continue;
        case spv::Op::OpPhi: {
            bool found = false;
            for (u32 operand = 3; operand + 1 < count; operand += 2) {
                if (inst[operand + 1] == predecessor) {
                    assign(value(operand));
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::runtime_error("Missing helper Phi predecessor");
            }
            break;
        }
        case spv::Op::OpReturnValue:
            result.address = value(1);
            return result;
        default:
            throw std::runtime_error("Unsupported helper instruction: " + std::to_string(u32(op)));
        }
        pos += count;
    }
    throw std::runtime_error("Helper did not return within the instruction budget");
}

TEST(BdaPointerBounds, ResidentPagePreservesOffsetAndDoesNotFault) {
    const auto result = RunLookup(7 * PageSize + 12, {{7, 0x100000U}});
    EXPECT_EQ(result.address, 0x10000cU);
    EXPECT_EQ(result.page_reads, (std::vector<u64>{7}));
    EXPECT_TRUE(result.fault_reads.empty());
    EXPECT_TRUE(result.fault_writes.empty());
}

TEST(BdaPointerBounds, UnmappedPageKeepsExistingFaultAndNullFallback) {
    const auto result = RunLookup(33 * PageSize, {});
    EXPECT_EQ(result.address, 0U);
    EXPECT_EQ(result.page_reads, (std::vector<u64>{33}));
    EXPECT_EQ(result.fault_reads, (std::vector<u64>{1}));
    EXPECT_EQ(result.fault_writes, (std::vector<u64>{1}));
    EXPECT_EQ(result.faults.at(1), 2U);
}

TEST(BdaPointerBounds, LastPageAndLastAlignedDwordRemainAddressable) {
    const auto result = RunLookup(AddressLimit - sizeof(u32), {{PageCount - 1, 0x200000U}});
    EXPECT_EQ(result.address, 0x200000U + PageSize - sizeof(u32));
    EXPECT_EQ(result.page_reads, (std::vector<u64>{PageCount - 1}));
    EXPECT_TRUE(result.fault_writes.empty());

    const auto missing = RunLookup(AddressLimit - sizeof(u32), {});
    EXPECT_EQ(missing.fault_writes, (std::vector<u64>{PageCount / 32 - 1}));
    EXPECT_EQ(missing.faults.at(PageCount / 32 - 1), 0x80000000U);
}

TEST(BdaPointerBounds, OutOfRangeAddressDoesNotTouchEitherGpuTable) {
    for (const u64 address :
         {AddressLimit, AddressLimit + PageSize, u64{1} << 45, std::numeric_limits<u64>::max()}) {
        SCOPED_TRACE(address);
        const auto result = RunLookup(address, {});
        EXPECT_EQ(result.address, 0U);
        EXPECT_TRUE(result.page_reads.empty());
        EXPECT_TRUE(result.fault_reads.empty());
        EXPECT_TRUE(result.fault_writes.empty());
    }
}

TEST(BdaPointerBounds, HighAddressCannotAliasAResidentLowPageAfterNarrowing) {
    const auto result = RunLookup((u64{1} << 46) + 7 * PageSize + 12, {{7, 0x100000U}});
    EXPECT_EQ(result.address, 0U);
    EXPECT_TRUE(result.page_reads.empty());
    EXPECT_TRUE(result.fault_reads.empty());
    EXPECT_TRUE(result.fault_writes.empty());
}

TEST(BdaPointerBounds, AdjacentPagesAreTranslatedIndependently) {
    const std::unordered_map<u64, u64> pages{{7, 0x100000U}, {8, 0x300000U}};
    const auto last_dword = RunLookup(8 * PageSize - sizeof(u32), pages);
    const auto next_dword = RunLookup(8 * PageSize, pages);
    EXPECT_EQ(last_dword.address, 0x100000U + PageSize - sizeof(u32));
    EXPECT_EQ(next_dword.address, 0x300000U);
}

#ifdef SHADPS4_SPIRV_VAL_PATH
TEST(BdaPointerBounds, GeneratedControlFlowPassesSpirvValidation) {
    constexpr u64 addr64_load = 0x80010000e030800cULL;
    const auto code = TranslateToSpirvWithInfo(addr64_load, true).spirv;
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    // Keep this small synthetic artifact inside the test's build working directory.
    const auto path =
        std::filesystem::current_path() / ("bda-pointer-bounds-" + std::to_string(suffix) + ".spv");
    {
        std::ofstream output{path, std::ios::binary | std::ios::trunc};
        ASSERT_TRUE(output.good());
        output.write(reinterpret_cast<const char*>(code.data()), code.size() * sizeof(u32));
    }
#ifdef _WIN32
    // CRT spawn joins arguments; the generated relative name has no spaces,
    // independently of spaces in the build directory.
    const auto file = path.filename().string();
    const auto result = _spawnl(_P_WAIT, SHADPS4_SPIRV_VAL_PATH, SHADPS4_SPIRV_VAL_PATH,
                                "--target-env", "vulkan1.3", file.c_str(), nullptr);
#else
    const auto command = std::string{"\""} + SHADPS4_SPIRV_VAL_PATH +
                         "\" --target-env vulkan1.3 \"" + path.string() + "\"";
    const auto result = std::system(command.c_str());
#endif
    EXPECT_EQ(result, 0);
    std::error_code error;
    std::filesystem::remove(path, error);
}
#endif

} // namespace
