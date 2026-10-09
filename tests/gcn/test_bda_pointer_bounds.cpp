// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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
    struct AtomicFaultAccess {
        u64 index;
        u32 scope;
        u32 semantics;
        u32 mask;
        u32 original;
    };
    u64 address{};
    std::vector<u64> page_reads;
    std::vector<u64> fault_reads;
    std::vector<u64> fault_writes;
    std::unordered_map<u64, u32> faults;
    std::vector<AtomicFaultAccess> atomic_fault_accesses;
    u32 plain_fault_reads{};
    u32 plain_fault_writes{};
};

// Execute the actual emitted helper on sparse synthetic memory, without a GPU or
// allocating its 512 MiB page table. Unsupported instructions fail the test, rather
// than silently approximating them. Out-of-range accesses remain visible in the
// trace even when the simulated storage-buffer load returns zero.
LookupResult RunLookup(u64 address, const std::unordered_map<u64, u64>& resident_pages,
                       std::unordered_map<u64, u32>& faults,
                       std::function<void()> after_fault_access = {}) {
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
                ++result.plain_fault_reads;
                assign(faults[ptr.index]);
                if (auto yield = std::exchange(after_fault_access, {})) {
                    yield();
                }
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
            ++result.plain_fault_writes;
            faults[ptr.index] = static_cast<u32>(value(2));
            break;
        }
        case spv::Op::OpAtomicOr: {
            if (count != 7) {
                throw std::runtime_error("Malformed helper atomic OR");
            }
            const auto ptr = pointers.at(inst[3]);
            if (ptr.buffer != fault_buffer) {
                throw std::runtime_error("Unexpected helper atomic destination");
            }
            const u32 original = faults[ptr.index];
            const u32 mask = static_cast<u32>(value(6));
            result.atomic_fault_accesses.push_back({ptr.index, static_cast<u32>(value(4)),
                                                    static_cast<u32>(value(5)), mask, original});
            result.fault_reads.push_back(ptr.index);
            result.fault_writes.push_back(ptr.index);
            faults[ptr.index] = original | mask;
            assign(original);
            // The atomic instruction is indivisible. Another invocation may run after this
            // RMW, but never between its read and write as it can for separate load/store.
            if (auto yield = std::exchange(after_fault_access, {})) {
                yield();
            }
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
            result.faults = faults;
            return result;
        default:
            throw std::runtime_error("Unsupported helper instruction: " + std::to_string(u32(op)));
        }
        pos += count;
    }
    throw std::runtime_error("Helper did not return within the instruction budget");
}

LookupResult RunLookup(u64 address, const std::unordered_map<u64, u64>& resident_pages) {
    std::unordered_map<u64, u32> faults;
    return RunLookup(address, resident_pages, faults);
}

TEST(BdaPointerFaults, InterleavedMissingPagesPreserveEveryBitInTheSameWord) {
    for (const bool reversed : {false, true}) {
        std::unordered_map<u64, u32> faults{{1, 0x80000000U}};
        LookupResult second;
        bool interleaved = false;
        const auto first_page = reversed ? 34U : 33U;
        const auto second_page = reversed ? 33U : 34U;
        const auto first = RunLookup(first_page * PageSize, {}, faults, [&] {
            interleaved = true;
            second = RunLookup(second_page * PageSize, {}, faults);
        });
        ASSERT_TRUE(interleaved);
        EXPECT_EQ(first.address, 0U);
        EXPECT_EQ(second.address, 0U);
        EXPECT_EQ(first.page_reads, (std::vector<u64>{first_page}));
        EXPECT_EQ(second.page_reads, (std::vector<u64>{second_page}));
        EXPECT_EQ(faults.at(1), 0x80000006U);
    }
}

TEST(BdaPointerFaults, SamePageReportsAreIdempotentAndPreserveExistingBits) {
    std::unordered_map<u64, u32> faults{{1, 0x80000000U}};
    (void)RunLookup(33 * PageSize, {}, faults, [&] { (void)RunLookup(33 * PageSize, {}, faults); });
    EXPECT_EQ(faults.at(1), 0x80000002U);
    EXPECT_EQ(faults.size(), 1U);
}

TEST(BdaPointerFaults, DifferentBitmapWordsRemainIndependent) {
    std::unordered_map<u64, u32> faults{{1, 0x80000000U}, {2, 1U}};
    (void)RunLookup(33 * PageSize, {}, faults, [&] { (void)RunLookup(66 * PageSize, {}, faults); });
    EXPECT_EQ(faults.at(1), 0x80000002U);
    EXPECT_EQ(faults.at(2), 5U);
}

TEST(BdaPointerFaults, ResidentAndInvalidAddressesDoNotAccessSharedFaultWords) {
    std::unordered_map<u64, u32> faults{{1, 0x80000000U}};
    const auto original = faults;
    bool yielded = false;
    const auto resident =
        RunLookup(33 * PageSize, {{33, 0x100000U}}, faults, [&] { yielded = true; });
    const auto invalid = RunLookup(AddressLimit, {}, faults, [&] { yielded = true; });
    EXPECT_EQ(resident.address, 0x100000U);
    EXPECT_EQ(invalid.address, 0U);
    EXPECT_FALSE(yielded);
    EXPECT_TRUE(resident.fault_reads.empty());
    EXPECT_TRUE(resident.fault_writes.empty());
    EXPECT_TRUE(invalid.page_reads.empty());
    EXPECT_TRUE(invalid.fault_reads.empty());
    EXPECT_TRUE(invalid.fault_writes.empty());
    EXPECT_EQ(faults, original);
}

TEST(BdaPointerFaults, FaultRecordingUsesOneDeviceScopedRelaxedAtomicOr) {
    const auto result = RunLookup(33 * PageSize, {});
    ASSERT_EQ(result.atomic_fault_accesses.size(), 1U);
    const auto& atomic = result.atomic_fault_accesses.front();
    EXPECT_EQ(atomic.index, 1U);
    EXPECT_EQ(atomic.scope, static_cast<u32>(spv::ScopeDevice));
    EXPECT_EQ(atomic.semantics, static_cast<u32>(spv::MemorySemanticsMaskNone));
    EXPECT_EQ(atomic.mask, 2U);
    EXPECT_EQ(atomic.original, 0U);
    EXPECT_EQ(result.plain_fault_reads, 0U);
    EXPECT_EQ(result.plain_fault_writes, 0U);
}

std::string FaultManagerSource() {
    std::ifstream stream{SHADPS4_FAULT_MANAGER_SOURCE_PATH};
    if (!stream.is_open()) {
        throw std::runtime_error("Cannot inspect actual fault-manager barrier wiring");
    }
    return {std::istreambuf_iterator<char>{stream}, {}};
}

std::string BarrierField(const std::string& source, const std::string& barrier,
                         const std::string& field) {
    const auto begin = source.find("const vk::BufferMemoryBarrier2 " + barrier + " = {");
    if (begin == std::string::npos) {
        throw std::runtime_error("Missing production fault barrier " + barrier);
    }
    const auto end = source.find("};", begin);
    const auto field_begin = source.find("." + field + " =", begin);
    if (end == std::string::npos || field_begin == std::string::npos || field_begin >= end) {
        throw std::runtime_error("Missing production fault barrier field " + field);
    }
    return source.substr(field_begin, source.find(',', field_begin) - field_begin);
}

TEST(BdaPointerFaultBarriers, ParserReadAndClearAndNextAtomicHaveReadWriteDestinations) {
    const auto source = FaultManagerSource();
    for (const std::string barrier : {"pre_barrier", "post_barrier"}) {
        const auto access = BarrierField(source, barrier, "dstAccessMask");
        EXPECT_NE(access.find("vk::AccessFlagBits2::eShaderRead"), std::string::npos) << barrier;
        EXPECT_NE(access.find("vk::AccessFlagBits2::eShaderWrite"), std::string::npos) << barrier;
        EXPECT_NE(BarrierField(source, barrier, "srcAccessMask")
                      .find("vk::AccessFlagBits2::eShaderWrite"),
                  std::string::npos);
        EXPECT_NE(BarrierField(source, barrier, "buffer").find("fault_buffer.Handle()"),
                  std::string::npos);
        EXPECT_NE(BarrierField(source, barrier, "size").find("fault_buffer_size"),
                  std::string::npos);
    }
}

TEST(BdaPointerFaultBarriers, ActualParserDispatchRemainsBetweenMatchingStageBarriers) {
    const auto source = FaultManagerSource();
    const auto pre = source.find(".pBufferMemoryBarriers = &pre_barrier");
    const auto dispatch = source.find("cmdbuf.dispatch(num_workgroups, 1, 1)");
    const auto post = source.find(".pBufferMemoryBarriers = &post_barrier");
    ASSERT_NE(pre, std::string::npos);
    ASSERT_NE(dispatch, std::string::npos);
    ASSERT_NE(post, std::string::npos);
    EXPECT_LT(pre, dispatch);
    EXPECT_LT(dispatch, post);
    EXPECT_NE(BarrierField(source, "pre_barrier", "srcStageMask").find("eAllCommands"),
              std::string::npos);
    EXPECT_NE(BarrierField(source, "pre_barrier", "dstStageMask").find("eComputeShader"),
              std::string::npos);
    EXPECT_NE(BarrierField(source, "post_barrier", "srcStageMask").find("eComputeShader"),
              std::string::npos);
    EXPECT_NE(BarrierField(source, "post_barrier", "dstStageMask").find("eAllCommands"),
              std::string::npos);
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
