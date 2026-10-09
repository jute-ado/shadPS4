// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <boost/container/flat_map.hpp>
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>
#include "common/alignment.h"
#include "common/arch.h"
#include "common/decoder.h"
#include "common/io_file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/signal_context.h"
#include "core/emulator_settings.h"
#include "core/signals.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/opcodes.h"
#include "shader_recompiler/ir/passes/dynamic_image_table_matcher.h"
#include "shader_recompiler/ir/passes/dynamic_image_table_policy.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/passes/srt.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/reg.h"
#include "shader_recompiler/ir/srt_gvn_table.h"
#include "shader_recompiler/ir/value.h"

#ifdef ARCH_X86_64

using namespace Xbyak::util;

// SRT walkers are retained for the lifetime of the process because compiled shader metadata
// stores their function pointers. A single growable Xbyak buffer cannot be used here: growing it
// relocates already-published function pointers. Keep immutable, non-moving executable segments
// instead. Code is emitted into a per-walker temporary buffer and copied into one of these
// segments only after the complete walker has been finalized.
constexpr size_t SrtWalkerSegmentSize = 32_MB;
constexpr size_t SrtWalkerTemporarySize = 1_MB;
constexpr size_t SrtWalkerArenaBudget = 1_GB;
constexpr size_t MaxSrtWalkerSegments = SrtWalkerArenaBudget / SrtWalkerSegmentSize;

struct SrtCodeSegment {
    explicit SrtCodeSegment()
        : generator(std::make_unique<Xbyak::CodeGenerator>(SrtWalkerSegmentSize)) {}

    std::unique_ptr<Xbyak::CodeGenerator> generator;
    // Published after bytes are finalized so the signal handler can inspect the segment without
    // racing the generator's mutable size field.
    std::atomic<size_t> published_size{};
};

static std::mutex g_srt_codegen_mutex;
static std::vector<std::unique_ptr<SrtCodeSegment>> g_srt_code_segments;
static std::array<std::atomic<SrtCodeSegment*>, MaxSrtWalkerSegments> g_srt_code_segment_slots{};
static std::atomic<SrtCodeSegment*> g_srt_active_segment{};
static size_t g_srt_reserved_bytes{};

namespace Shader {

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    if (ptr == nullptr || size == 0 || size > SrtWalkerSegmentSize) {
        return nullptr;
    }

    try {
        std::scoped_lock lock{g_srt_codegen_mutex};
        SrtCodeSegment* segment = g_srt_active_segment.load(std::memory_order_acquire);
        if (segment == nullptr || segment->generator->getSize() > SrtWalkerSegmentSize - size) {
            if (g_srt_code_segments.size() >= MaxSrtWalkerSegments ||
                g_srt_reserved_bytes > SrtWalkerArenaBudget - SrtWalkerSegmentSize) {
                return nullptr;
            }
            auto owned_segment = std::make_unique<SrtCodeSegment>();
            segment = owned_segment.get();
            g_srt_code_segments.push_back(std::move(owned_segment));
            g_srt_reserved_bytes += SrtWalkerSegmentSize;
            const size_t slot = g_srt_code_segments.size() - 1;
            g_srt_code_segment_slots[slot].store(segment, std::memory_order_release);
            g_srt_active_segment.store(segment, std::memory_order_release);
        }

        const auto func_addr = segment->generator->getCurr<PFN_SrtWalker>();
        segment->generator->db(ptr, size);
        segment->generator->ready();
        segment->published_size.store(segment->generator->getSize(), std::memory_order_release);
        return func_addr;
    } catch (const Xbyak::Error&) {
        // Do not expose a partially emitted walker. The segment remains valid up to its last
        // published size, while the next request starts a fresh segment.
        g_srt_active_segment.store(nullptr, std::memory_order_release);
        return nullptr;
    } catch (const std::bad_alloc&) {
        g_srt_active_segment.store(nullptr, std::memory_order_release);
        return nullptr;
    }
}

} // namespace Shader

namespace {

static void DumpSrtProgram(const Shader::Info& info, const u8* code, size_t codesize) {
    using namespace Common::FS;

    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}_{:#018x}.srtprogram.txt", info.stage, info.pgm_hash);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create, FileType::TextFile};

    u64 address = reinterpret_cast<u64>(code);
    u64 code_end = address + codesize;
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    ZyanStatus status = ZYAN_STATUS_SUCCESS;
    while (address < code_end && ZYAN_SUCCESS(Common::Decoder::Instance()->decodeInstruction(
                                     instruction, operands, reinterpret_cast<void*>(address)))) {
        std::string s =
            Common::Decoder::Instance()->disassembleInst(instruction, operands, address);
        s += "\n";
        file.WriteString(s);
        address += instruction.length;
    }
}

static bool SrtWalkerSignalHandler(void* context, void* fault_address) {
    const void* code = Common::GetRip(context);
    SrtCodeSegment* segment = nullptr;
    for (size_t i = 0; i < MaxSrtWalkerSegments; ++i) {
        SrtCodeSegment* const candidate =
            g_srt_code_segment_slots[i].load(std::memory_order_acquire);
        if (candidate == nullptr) {
            break;
        }
        const auto* const code_start = candidate->generator->getCode();
        const auto* const code_end = code_start +
                                     candidate->published_size.load(std::memory_order_acquire);
        if (code >= code_start && code < code_end) {
            segment = candidate;
            break;
        }
    }
    if (segment == nullptr) {
        return false; // Not in SRT code range
    }

    // Patch instruction to zero register
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    ZyanStatus status = Common::Decoder::Instance()->decodeInstruction(instruction, operands,
                                                                       const_cast<void*>(code), 15);

    ASSERT(ZYAN_SUCCESS(status) && instruction.mnemonic == ZYDIS_MNEMONIC_MOV &&
           operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
           operands[1].type == ZYDIS_OPERAND_TYPE_MEMORY);

    size_t len = instruction.length;
    const size_t patch_size = 3;
    u8* code_patch = const_cast<u8*>(reinterpret_cast<const u8*>(code));

    // We can only encounter rdi or r10d as the first operand in a
    // fault memory access for SRT walker.
    switch (operands[0].reg.value) {
    case ZYDIS_REGISTER_RDI:
        // mov rdi, [rdi + (off_dw << 2)] -> xor rdi, rdi
        code_patch[0] = 0x48;
        code_patch[1] = 0x31;
        code_patch[2] = 0xFF;
        break;
    case ZYDIS_REGISTER_R10D:
        // mov r10d, [rdi + (off_dw << 2)] -> xor r10d, r10d
        code_patch[0] = 0x45;
        code_patch[1] = 0x31;
        code_patch[2] = 0xD2;
        break;
    default:
        UNREACHABLE_MSG("Unsupported register for SRT walker patch");
        return false;
    }

    // Fill nops
    memset(code_patch + patch_size, 0x90, len - patch_size);

    LOG_WARNING(Render_Recompiler, "Patched SRT walker at {}, fault address {}", code,
                fault_address);

    return true;
}

using namespace Shader;

struct PassInfo {
    // map offset to inst
    using PtrUserList = boost::container::flat_map<u32, Shader::IR::Inst*>;

    Optimization::SrtGvnTable gvn_table;
    // Key by the complete pointer expression. Two adjustments of one source are not one table.
    std::unordered_map<IR::Inst*, PtrUserList> pointer_uses;
    boost::container::small_vector<IR::Inst*, 8> pointer_order;
    boost::container::small_vector<IR::Inst*, 8> emitted_pointers;
    std::unordered_set<IR::Inst*> visited_dependencies;
    enum class ScalarState { Visiting, Supported, Unsupported };
    std::unordered_map<IR::Inst*, ScalarState> scalar_validation;
    std::unordered_set<IR::Inst*> host_descriptor_dependencies;
    // Resource handles often name only their first dword. The complete descriptor must remain
    // contiguous in flat memory even when its other words have no explicit IR consumers.
    std::unordered_map<IR::Inst*, u32> descriptor_dword_counts;
    bool invalid_descriptor_span{};
    u32 dynamic_user_data_mask{};
    bool dynamic_descriptor_dependency{};
    IR::Inst* rejected_scalar{};
    std::string_view rejection_reason;
    // Arithmetic results are written once per walker invocation into private flattened storage.
    // Shared SSA diamonds must not expand exponentially into native instructions.
    std::unordered_map<IR::Inst*, u32> scalar_offsets;

    // pick a single inst for a given value number
    std::unordered_map<u32, IR::Inst*> vn_to_inst;

    // Bumped during codegen to assign offsets to readconsts
    u32 dst_off_dw;

    boost::container::small_vector<Optimization::DynamicImageTableMatch, 2> dynamic_image_tables;

    PtrUserList* GetUsesAsPointer(IR::Inst* inst) {
        auto it = pointer_uses.find(inst);
        if (it != pointer_uses.end()) {
            return &it->second;
        }
        return nullptr;
    }

    const PtrUserList* GetUsesAsPointer(const IR::Inst* inst) const {
        const auto it = pointer_uses.find(const_cast<IR::Inst*>(inst));
        if (it != pointer_uses.end()) {
            return &it->second;
        }
        return nullptr;
    }

    // Return a single instruction that this instruction is identical to, according
    // to value number
    // The "original" is arbitrary. Here it's the first instruction found for a given value number
    IR::Inst* DeduplicateInstruction(IR::Inst* inst) {
        auto it = vn_to_inst.try_emplace(gvn_table.GetValueNumber(inst), inst);
        return it.first->second;
    }
};
} // namespace

namespace Shader::Optimization {

namespace {

static bool HasSrtSource(IR::Value value) {
    return IR::BreadthFirstSearch(value,
                                  [](IR::Inst* inst) -> std::optional<IR::Inst*> {
                                      if (inst->GetOpcode() == IR::Opcode::GetUserData ||
                                          inst->GetOpcode() == IR::Opcode::ReadConst) {
                                          return inst;
                                      }
                                      return std::nullopt;
                                  })
        .has_value();
}

static void CollectDescriptorSourceSpans(IR::Value handle, u32 dword_count, PassInfo& pass_info) {
    auto* source = handle.TryInstRecursive();
    if (source && source->GetOpcode() == IR::Opcode::CompositeConstructU32x4) {
        // Buffer and sampler handles expose four register words; resource tracking follows the
        // first register and reads their contiguous source descriptor, not each operand's source.
        source = source->Arg(0).TryInstRecursive();
    }
    if (!source) {
        return;
    }
    boost::container::small_vector<IR::Inst*, 8> pending{source};
    std::unordered_set<IR::Inst*> visited;
    while (!pending.empty()) {
        auto* const current = pending.back();
        pending.pop_back();
        if (!visited.insert(current).second) {
            continue;
        }
        if (current->GetOpcode() == IR::Opcode::GetUserData) {
            const u32 first = static_cast<u32>(current->Arg(0).ScalarReg());
            for (u32 location = 0U; location < NUM_USER_DATA_REGS; ++location) {
                if ((pass_info.dynamic_user_data_mask & (1U << location)) != 0U &&
                    location >= first && location - first < dword_count) {
                    pass_info.dynamic_descriptor_dependency = true;
                }
            }
            continue;
        }
        if (current->GetOpcode() == IR::Opcode::ReadConst) {
            // Stop at the actual SHARP source. Its pointer ancestors are ordinary scalar
            // dependencies, not additional descriptors of this width.
            if (current->Arg(1).IsImmediate()) {
                if (current->Arg(1).U32() > std::numeric_limits<u32>::max() - (dword_count - 1U)) {
                    pass_info.invalid_descriptor_span = true;
                    continue;
                }
                auto& width = pass_info.descriptor_dword_counts[current];
                width = std::max(width, dword_count);
            }
            continue;
        }
        const auto enqueue = [&](IR::Value value) {
            if (auto* const inst = value.TryInstRecursive()) {
                pending.push_back(inst);
            }
        };
        switch (current->GetOpcode()) {
        case IR::Opcode::Phi:
            for (size_t index = 0; index < current->NumArgs(); ++index) {
                enqueue(current->Arg(index));
            }
            break;
        case IR::Opcode::SelectU32:
            // The anisotropy sampler pattern selects between the original sampler and a
            // masked copy. Its condition reads unrelated image metadata and is not a sampler.
            enqueue(current->Arg(1));
            enqueue(current->Arg(2));
            break;
        case IR::Opcode::BitwiseAnd32:
            enqueue(current->Arg(0));
            enqueue(current->Arg(1));
            break;
        default:
            break;
        }
    }
}

static void CollectHostDescriptorDependencies(IR::Program& program, PassInfo& pass_info) {
    boost::container::small_vector<IR::Inst*, 32> pending;
    const auto enqueue = [&](IR::Value value) {
        if (auto* const producer = value.TryInstRecursive();
            producer && pass_info.host_descriptor_dependencies.insert(producer).second) {
            pending.push_back(producer);
        }
    };
    for (auto* const block : program.blocks) {
        for (IR::Inst& inst : *block) {
            const u32 mask = GetResourceHandleOperandMask(inst);
            if (mask == 0) {
                continue;
            }
            const auto match = MatchDynamicImageTable(inst);
            const bool known_table =
                match && PlanDynamicImageTable(match->Pattern(0U, NUM_IMAGES, true, true));
            for (u32 arg = 0; arg < inst.NumArgs(); ++arg) {
                if ((mask & (1U << arg)) == 0) {
                    continue;
                }
                if (arg == 0 && known_table) {
                    // Table bytes/base must be captured; its selecting index executes on GPU.
                    // This is not activation proof: ResourceTrackingPass must still admit the
                    // exact reservation/profile or refuse its unmaterialized static fallback.
                    enqueue(IR::Value{match->pointer_lo});
                    enqueue(IR::Value{match->pointer_hi});
                } else {
                    // Include indirect arithmetic and nested pointer-table dependencies.
                    CollectDescriptorSourceSpans(inst.Arg(arg),
                                                GetResourceHandleDwordCount(inst, arg), pass_info);
                    enqueue(inst.Arg(arg));
                }
            }
        }
    }
    while (!pending.empty()) {
        auto* const inst = pending.back();
        pending.pop_back();
        if (inst->GetOpcode() == IR::Opcode::GetAttributeU32) {
            const auto attribute = inst->Arg(0).Attribute();
            if (attribute == IR::Attribute::BaseVertex || attribute == IR::Attribute::BaseInstance ||
                attribute == IR::Attribute::DrawIndex) {
                pass_info.dynamic_descriptor_dependency = true;
            }
        }
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            enqueue(inst->Arg(arg));
        }
    }
}

// Only evaluate explicit, side-effect-free scalar operations. In particular, looking through an
// unknown operation to its first user-data ancestor changes the guest address rather than merely
// discovering its provenance. The supported select/comparison pattern covers S_ADD/S_ADDC carry.
static bool IsSupportedPointerScalar(IR::Value value, PassInfo& pass_info, u32 depth = 0,
                                     bool allow_shared_user_data_float_use = false) {
    const auto reject = [&](IR::Inst* inst, std::string_view reason) {
        if (pass_info.rejection_reason.empty()) {
            pass_info.rejected_scalar = inst;
            pass_info.rejection_reason = reason;
        }
        return false;
    };
    if (depth > 64) {
        return reject(value.TryInstRecursive(), "scalar recursion depth exceeded");
    }
    value = value.Resolve();
    if (value.IsImmediate()) {
        return value.Type() == IR::Type::U32 || value.Type() == IR::Type::U1 ||
               reject(nullptr, "scalar immediate is not U32 or U1");
    }
    auto* const inst = value.TryInstRecursive();
    if (!inst) {
        return reject(nullptr, "scalar has no producer");
    }
    const auto [state, inserted] =
        pass_info.scalar_validation.try_emplace(inst, PassInfo::ScalarState::Visiting);
    if (!inserted) {
        return state->second == PassInfo::ScalarState::Supported ||
               reject(inst, state->second == PassInfo::ScalarState::Visiting
                                ? "cyclic scalar dependency"
                                : "previously rejected scalar dependency");
    }
    const auto result = [&](bool supported) {
        // Recursive insertion may rehash the map; do not retain an iterator across recursion.
        pass_info.scalar_validation.at(inst) =
            supported ? PassInfo::ScalarState::Supported : PassInfo::ScalarState::Unsupported;
        if (!supported) {
            reject(inst, "scalar operand or source is unsupported");
        }
        return supported;
    };
    const auto supported = [&](IR::Value argument) {
        return IsSupportedPointerScalar(argument, pass_info, depth + 1,
                                       allow_shared_user_data_float_use);
    };
    switch (inst->GetOpcode()) {
    case IR::Opcode::GetUserData:
        return result(static_cast<u32>(inst->Arg(0).ScalarReg()) < NUM_USER_DATA_REGS &&
                      (allow_shared_user_data_float_use || !Detail::HasFloatingPointUse(*inst)));
    case IR::Opcode::ReadConst: {
        auto* const pointer = inst->Arg(0).TryInstRecursive();
        return result(inst->Arg(1).IsImmediate() && pointer && HasSrtSource(inst->Arg(0)) &&
                      (allow_shared_user_data_float_use || !Detail::HasFloatingPointUse(*inst)) &&
                      pointer->GetOpcode() == IR::Opcode::CompositeConstructU32x2 &&
                      supported(pointer->Arg(0)) && supported(pointer->Arg(1)));
    }
    case IR::Opcode::Phi: {
        auto* const source = Detail::FindUniquePointerSource(*inst);
        if (!source) {
            reject(inst, "phi has no unique scalar source");
        }
        return result(source && supported(IR::Value{source}));
    }
    case IR::Opcode::IAdd32:
    case IR::Opcode::ISub32:
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::ULessThan32:
        return result(supported(inst->Arg(0)) && supported(inst->Arg(1)));
    case IR::Opcode::SelectU32:
        return result(supported(inst->Arg(0)) && supported(inst->Arg(1)) &&
                      supported(inst->Arg(2)));
    default:
        reject(inst, "scalar opcode is unsupported");
        return result(false);
    }
}

static void LogUnsupportedPointer(const Info& info, IR::Inst& read, const PassInfo& pass_info) {
    // Rejected shaders can be retried every dispatch. Keep diagnostics useful and bounded without
    // dumping shader contents or user-data values. This records graph structure, not guest bytes.
    static std::mutex diagnostic_mutex;
    static std::unordered_set<u64> logged_shaders;
    {
        const std::scoped_lock lock{diagnostic_mutex};
        if (logged_shaders.size() >= 64 || !logged_shaders.insert(info.pgm_hash).second) {
            return;
        }
    }
    constexpr size_t MaxNodes = 32;
    boost::container::small_vector<IR::Inst*, MaxNodes> nodes;
    if (auto* const pointer = read.Arg(0).TryInstRecursive()) {
        nodes.push_back(pointer);
    }
    std::string graph;
    for (size_t index = 0; index < nodes.size(); ++index) {
        auto* const node = nodes[index];
        graph += fmt::format("{}:{}(", index, node->GetOpcode());
        for (size_t arg = 0; arg < std::min(node->NumArgs(), size_t{8}); ++arg) {
            const auto operand = node->Arg(arg).Resolve();
            if (arg != 0) {
                graph += ',';
            }
            if (operand.IsImmediate()) {
                graph += fmt::format("immediate:{}", operand.Type());
                continue;
            }
            auto* const producer = operand.TryInstRecursive();
            if (!producer) {
                graph += "no-producer";
                continue;
            }
            auto found = std::ranges::find(nodes, producer);
            if (found == nodes.end()) {
                if (nodes.size() >= MaxNodes) {
                    graph += "...";
                    continue;
                }
                nodes.push_back(producer);
                found = std::prev(nodes.end());
            }
            graph += fmt::format("{}", std::distance(nodes.begin(), found));
        }
        graph += ");";
    }
    LOG_WARNING(Render_Recompiler,
                "Unsupported scalar SRT pointer: shader={:#x} stage={} reason={} rejected={} "
                "host_descriptor_dependency={} read_offset_dw={} graph={}",
                info.pgm_hash, info.stage,
                pass_info.rejection_reason.empty() ? "pointer is not a scalar pair"
                                                   : pass_info.rejection_reason,
                pass_info.rejected_scalar ? IR::NameOf(pass_info.rejected_scalar->GetOpcode())
                                          : std::string_view{"none"},
                pass_info.host_descriptor_dependencies.contains(&read), read.Arg(1).U32(), graph);
}

// Leaves the exact 32-bit scalar result (including wrapping arithmetic) in r10d.
static void EmitPointerScalar(IR::Value value, PassInfo& pass_info, Xbyak::CodeGenerator& c) {
    value = value.Resolve();
    if (value.IsImmediate()) {
        c.mov(r10d, value.Type() == IR::Type::U1 ? u32{value.U1()} : value.U32());
        return;
    }
    auto* const inst = value.InstRecursive();
    if (const auto existing = pass_info.scalar_offsets.find(inst);
        existing != pass_info.scalar_offsets.end()) {
        c.mov(r10d, ptr[r13 + (existing->second << 2)]);
        return;
    }
    const auto cache_result = [&] {
        const u32 offset = pass_info.dst_off_dw++;
        pass_info.scalar_offsets.emplace(inst, offset);
        c.mov(ptr[r13 + (offset << 2)], r10d);
    };
    switch (inst->GetOpcode()) {
    case IR::Opcode::GetUserData:
        c.mov(r10d, ptr[r13 + (static_cast<u32>(inst->Arg(0).ScalarReg()) << 2)]);
        return;
    case IR::Opcode::ReadConst:
        c.mov(r10d, ptr[r13 + (pass_info.DeduplicateInstruction(inst)->Flags<u32>() << 2)]);
        return;
    case IR::Opcode::Phi:
        EmitPointerScalar(IR::Value{Detail::FindUniquePointerSource(*inst)}, pass_info, c);
        return;
    case IR::Opcode::SelectU32: {
        // Operands are pure scalar arithmetic over captured values. Emit both before selecting,
        // so a cached subexpression is initialized on every path that reuses it.
        EmitPointerScalar(inst->Arg(0), pass_info, c);
        c.sub(rsp, 16);
        c.mov(ptr[rsp], r10d);
        EmitPointerScalar(inst->Arg(1), pass_info, c);
        c.sub(rsp, 16);
        c.mov(ptr[rsp], r10d);
        EmitPointerScalar(inst->Arg(2), pass_info, c);
        c.mov(r11d, r10d);
        c.mov(r10d, ptr[rsp]);
        c.add(rsp, 16);
        c.mov(eax, ptr[rsp]);
        c.add(rsp, 16);
        c.test(eax, eax);
        c.cmovz(r10d, r11d);
        cache_result();
        return;
    }
    default:
        break;
    }

    EmitPointerScalar(inst->Arg(0), pass_info, c);
    c.sub(rsp, 16);
    c.mov(ptr[rsp], r10d);
    EmitPointerScalar(inst->Arg(1), pass_info, c);
    c.mov(r11d, r10d);
    c.mov(r10d, ptr[rsp]);
    c.add(rsp, 16);
    switch (inst->GetOpcode()) {
    case IR::Opcode::IAdd32:
        c.add(r10d, r11d);
        break;
    case IR::Opcode::ISub32:
        c.sub(r10d, r11d);
        break;
    case IR::Opcode::BitwiseAnd32:
        c.and_(r10d, r11d);
        break;
    case IR::Opcode::ULessThan32:
        c.cmp(r10d, r11d);
        c.setb(r10b);
        c.movzx(r10d, r10b);
        break;
    default:
        UNREACHABLE_MSG("Unvalidated SRT pointer expression");
    }
    cache_result();
}

static void EmitPointer(IR::Inst* pointer, PassInfo& pass_info, Xbyak::CodeGenerator& c) {
    EmitPointerScalar(pointer->Arg(0), pass_info, c);
    c.sub(rsp, 16);
    c.mov(ptr[rsp], r10d);
    EmitPointerScalar(pointer->Arg(1), pass_info, c);
    c.mov(edi, r10d);
    c.shl(rdi, 32);
    c.mov(r11d, ptr[rsp]);
    c.add(rsp, 16);
    c.or_(rdi, r11);
    // Keep the raw 64-bit address until the runtime canonicality gate in VisitNode.  Masking
    // here would alias malformed high halves (for example float payload bits) into a different
    // low-only address before the SRT reader can reject them.
}

struct SrtSourceRange {
    u64 begin_dw;
    u64 end_dw;
};

static boost::container::small_vector<SrtSourceRange, 4> GetSourceRanges(
    const PassInfo::PtrUserList& use_list) {
    boost::container::small_vector<SrtSourceRange, 4> ranges;
    for (const auto [src_off_dw, use] : use_list) {
        const u64 begin_dw = src_off_dw;
        // Each actual pointer half has its own ReadConst dependency. Adjacent halves naturally
        // coalesce; nonadjacent halves must not read the intervening dword as their high half.
        const u64 end_dw = begin_dw + 1U;
        if (!ranges.empty() && begin_dw <= ranges.back().end_dw) {
            ranges.back().end_dw = std::max(ranges.back().end_dw, end_dw);
        } else {
            ranges.push_back({begin_dw, end_dw});
        }
    }
    return ranges;
}

// In direct-memory mode, a pointer whose address is assembled from ReadConst data is not a safe
// host snapshot candidate unless a resource consumer explicitly requires it. Those words may be
// ordinary runtime payload (the GPU read_const path can resolve them through BDA/fault handling),
// and treating them as a CPU address can turn payload bits into an invalid host read. Root pointers
// assembled directly from user-data or arithmetic over it remain eligible; descriptor consumers
// opt the complete dependency chain back into host materialization through the dependency set.
static bool ShouldMaterializePointer(const IR::Inst& pointer, const PassInfo& pass_info) {
    if (!EmulatorSettings.IsDirectMemoryAccessEnabled()) {
        return true;
    }
    if (const auto* const uses = pass_info.GetUsesAsPointer(&pointer); uses) {
        for (const auto [unused_offset, use] : *uses) {
            if (pass_info.host_descriptor_dependencies.contains(use)) {
                return true;
            }
        }
    }
    const auto has_read_const_dependency = [](IR::Value value) {
        return IR::BreadthFirstSearch(
                   value, [](IR::Inst* inst) -> std::optional<IR::Inst*> {
                       if (inst->GetOpcode() == IR::Opcode::ReadConst) {
                           return inst;
                       }
                       return std::nullopt;
                   })
            .has_value();
    };
    return !has_read_const_dependency(pointer.Arg(0)) &&
           !has_read_const_dependency(pointer.Arg(1));
}

static void VisitNode(IR::Inst* subtree, PassInfo& pass_info, Xbyak::CodeGenerator& c) {
    PassInfo::PtrUserList* use_list = pass_info.GetUsesAsPointer(subtree);
    ASSERT(use_list);

    if (!ShouldMaterializePointer(*subtree, pass_info)) {
        return;
    }

    // A single canonical pointer expression can feed both ordinary shader data and a resource
    // descriptor (the compiler's value numbering intentionally merges those equivalent
    // expressions).  Materializing the whole use list in that case turns the data reads into
    // host-address candidates as well.  Keep only the uses explicitly required by a descriptor;
    // the remaining ReadConst instructions retain flags=0 and execute through the GPU/BDA path.
    const auto has_read_const_dependency = [](IR::Value value) {
        return IR::BreadthFirstSearch(
                   value, [](IR::Inst* inst) -> std::optional<IR::Inst*> {
                       if (inst->GetOpcode() == IR::Opcode::ReadConst) {
                           return inst;
                       }
                       return std::nullopt;
                   })
            .has_value();
    };
    const bool descriptor_only_uses =
        EmulatorSettings.IsDirectMemoryAccessEnabled() &&
        (has_read_const_dependency(subtree->Arg(0)) ||
         has_read_const_dependency(subtree->Arg(1)));
    PassInfo::PtrUserList materialized_uses;
    materialized_uses.reserve(use_list->size());

    struct SrtCopy {
        u32 src_off_dw;
        u32 dst_off_dw;
    };
    boost::container::small_vector<SrtCopy, 8> copies;
    for (const auto [src_off_dw, use] : *use_list) {
        if (descriptor_only_uses && !pass_info.host_descriptor_dependencies.contains(use)) {
            // Explicitly clear a previous value-number assignment.  This pointer may be emitted
            // again through a shared SSA path after a descriptor use assigned a flat offset.
            use->SetFlags<u32>(0U);
            continue;
        }
        materialized_uses.emplace(src_off_dw, use);
        if (const auto span = pass_info.descriptor_dword_counts.find(use);
            span != pass_info.descriptor_dword_counts.end()) {
            for (u32 word = 1; word < span->second; ++word) {
                materialized_uses.try_emplace(src_off_dw + word, nullptr);
            }
        }
    }
    // Allocate the sorted source union, including descriptor words that are absent from the IR.
    // Overlapping descriptors share their exact source offsets and retain contiguous layouts.
    for (const auto [src_off_dw, unused_use] : materialized_uses) {
        const u32 dst_off_dw = pass_info.dst_off_dw++;
        if (const auto existing = use_list->find(src_off_dw); existing != use_list->end()) {
            existing->second->SetFlags<u32>(dst_off_dw);
        }
        copies.push_back({src_off_dw, dst_off_dw});
    }

    if (materialized_uses.empty()) {
        return;
    }

    // Read every source range into private stack storage. A reader may obtain authoritative GPU
    // bytes without publishing them back into mutable guest memory. Without a reader, preserve
    // the original direct-memory behavior for callers that only consume CPU-owned SRTs.
    EmitPointer(subtree, pass_info, c);
    c.mov(rsi, r13);
    for (const auto& range : GetSourceRanges(materialized_uses)) {
        const u64 range_size = (range.end_dw - range.begin_dw) * sizeof(u32);
        const u64 scratch_size = Common::AlignUp(range_size, 16ULL);
        const u64 stack_size = scratch_size + 16;
        Xbyak::Label direct_read;
        Xbyak::Label refused;
        Xbyak::Label null_source;
        Xbyak::Label invalid_pointer;
        Xbyak::Label valid_pointer;
        Xbyak::Label finished;
        const auto zero_range = [&] {
            for (const auto [src_off_dw, dst_off_dw] : copies) {
                if (src_off_dw >= range.begin_dw && src_off_dw < range.end_dw) {
                    c.mov(dword[r13 + (dst_off_dw << 2)], 0);
                }
            }
        };

        c.test(rdi, rdi);
        c.jz(null_source, Xbyak::CodeGenerator::T_NEAR);
        // Guest SRT pointers must be canonical 48-bit values.  Reject non-zero upper halves
        // before either calling a reader or taking the direct-memory path; otherwise a later
        // 48-bit mask would make malformed payload words look like valid low addresses.
        c.mov(r10, rdi);
        c.shr(r10, 48);
        c.test(r10, r10);
        c.jz(valid_pointer, Xbyak::CodeGenerator::T_NEAR);
        c.cmp(r10, 0xffff);
        c.jnz(invalid_pointer, Xbyak::CodeGenerator::T_NEAR);
        c.L(valid_pointer);
        c.test(r14, r14);
        c.jz(direct_read, Xbyak::CodeGenerator::T_NEAR);
        c.sub(rsp, stack_size);
        c.mov(ptr[rsp + scratch_size], rdi);
        c.lea(rsi, ptr[rdi + range.begin_dw * sizeof(u32)]);
        c.mov(rdx, rsp);
        c.mov(rcx, range_size);
        c.mov(rdi, r15);
        c.call(r14);
        c.mov(rdi, ptr[rsp + scratch_size]);
        c.mov(rsi, r13);
        c.test(al, al);
        c.jz(refused, Xbyak::CodeGenerator::T_NEAR);
        for (const auto [src_off_dw, dst_off_dw] : copies) {
            if (src_off_dw >= range.begin_dw && src_off_dw < range.end_dw) {
                c.mov(r10d, ptr[rsp + (src_off_dw - range.begin_dw) * sizeof(u32)]);
                c.mov(ptr[rsi + (dst_off_dw << 2)], r10d);
            }
        }
        c.add(rsp, stack_size);
        c.jmp(finished, Xbyak::CodeGenerator::T_NEAR);

        c.L(refused);
        zero_range();
        c.add(rsp, stack_size);
        c.jmp(finished, Xbyak::CodeGenerator::T_NEAR);

        c.L(null_source);
        zero_range();
        c.jmp(finished, Xbyak::CodeGenerator::T_NEAR);

        c.L(invalid_pointer);
        zero_range();
        c.jmp(finished, Xbyak::CodeGenerator::T_NEAR);

        c.L(direct_read);
        // Direct-memory mode still uses the normalized 48-bit host address after the raw value
        // has passed the canonicality check above.  This preserves sign-extended canonical
        // pointers without ever aliasing a non-canonical high half.
        c.mov(r10, 0xFFFFFFFFFFFFULL);
        c.and_(rdi, r10);
        for (const auto [src_off_dw, dst_off_dw] : copies) {
            if (src_off_dw >= range.begin_dw && src_off_dw < range.end_dw) {
                c.mov(r10d, ptr[rdi + (src_off_dw << 2)]);
                c.mov(ptr[rsi + (dst_off_dw << 2)], r10d);
            }
        }
        c.L(finished);
    }
}

static void VisitPointer(IR::Inst* pointer, PassInfo& pass_info, Xbyak::CodeGenerator& c);

static void VisitPointerDependencies(IR::Value value, PassInfo& pass_info,
                                     Xbyak::CodeGenerator& c) {
    value = value.Resolve();
    if (value.IsImmediate()) {
        return;
    }
    auto* const inst = value.InstRecursive();
    if (!pass_info.visited_dependencies.insert(inst).second) {
        return;
    }
    if (inst->GetOpcode() == IR::Opcode::GetUserData) {
        return;
    }
    if (inst->GetOpcode() == IR::Opcode::ReadConst) {
        auto* const dependency = pass_info.DeduplicateInstruction(inst->Arg(0).InstRecursive());
        VisitPointer(dependency, pass_info, c);
        return;
    }
    if (inst->GetOpcode() == IR::Opcode::Phi) {
        VisitPointerDependencies(IR::Value{Detail::FindUniquePointerSource(*inst)}, pass_info, c);
        return;
    }
    for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
        VisitPointerDependencies(inst->Arg(arg), pass_info, c);
    }
}

static void VisitPointer(IR::Inst* pointer, PassInfo& pass_info, Xbyak::CodeGenerator& c) {
    if (std::ranges::find(pass_info.emitted_pointers, pointer) !=
        pass_info.emitted_pointers.end()) {
        return;
    }
    VisitPointerDependencies(pointer->Arg(0), pass_info, c);
    VisitPointerDependencies(pointer->Arg(1), pass_info, c);
    VisitNode(pointer, pass_info, c);
    pass_info.emitted_pointers.push_back(pointer);
}

static bool GenerateSrtProgramImpl(Info& info, PassInfo& pass_info) {
    if (pass_info.pointer_order.empty()) {
        return true;
    }

    // Generate each walker into a private, fixed buffer. Publishing directly into the process
    // arena while emitting makes a failed/oversized walker leave partial bytes behind, and a
    // single growable arena would relocate already-published function pointers.
    Xbyak::CodeGenerator c(SrtWalkerTemporarySize);
    pass_info.dst_off_dw = NUM_USER_DATA_REGS;
    ASSERT(pass_info.dst_off_dw == info.srt_info.flattened_bufsize_dw);

    // SysV arguments arrive in rdi/rsi/rdx/rcx. Retain them in callee-saved registers and align
    // the stack once because generated walkers can now call the optional synchronizer.
    c.push(r12);
    c.push(r13);
    c.push(r14);
    c.push(r15);
    c.sub(rsp, 8);
    c.mov(r12, rdi);
    c.mov(r13, rsi);
    c.mov(r14, rdx);
    c.mov(r15, rcx);

    for (auto* const pointer : pass_info.pointer_order) {
        VisitPointer(pointer, pass_info, c);
    }

    c.add(rsp, 8);
    c.pop(r15);
    c.pop(r14);
    c.pop(r13);
    c.pop(r12);
    c.ret();
    c.ready();

    const size_t walker_size = c.getSize();
    const PFN_SrtWalker walker = Shader::RegisterWalkerCode(c.getCode(), walker_size);
    if (walker == nullptr) {
        LOG_ERROR(Render_Recompiler,
                  "SRT walker generation exhausted its bounded code arena ({} bytes)",
                  walker_size);
        info.translation_failed = true;
        info.srt_info.walker_func = nullptr;
        info.srt_info.walker_func_size = 0;
        return false;
    }
    info.srt_info.walker_func = walker;
    info.srt_info.walker_func_size = walker_size;

    // Register the signal handler for SRT walkers once. Segments are process-lifetime and are
    // published atomically, so the handler can identify the segment containing a faulting RIP.
    static std::once_flag signal_handler_once;
    std::call_once(signal_handler_once, [] {
        auto* signals = Core::Signals::Instance();
        // Call after the memory invalidation handler.
        constexpr u32 priority = 1;
        signals->RegisterAccessViolationHandler(SrtWalkerSignalHandler, priority);
    });

    if (EmulatorSettings.IsDumpShaders()) {
        DumpSrtProgram(info, reinterpret_cast<const u8*>(info.srt_info.walker_func),
                       info.srt_info.walker_func_size);
    }

    info.srt_info.flattened_bufsize_dw = pass_info.dst_off_dw;
    return true;
}

static bool GenerateSrtProgram(Info& info, PassInfo& pass_info) {
    try {
        return GenerateSrtProgramImpl(info, pass_info);
    } catch (const Xbyak::Error& error) {
        LOG_ERROR(Render_Recompiler, "SRT walker generation failed: {}", error.what());
    } catch (const std::bad_alloc&) {
        LOG_ERROR(Render_Recompiler, "SRT walker generation failed: out of memory");
    }
    info.translation_failed = true;
    info.srt_info.walker_func = nullptr;
    info.srt_info.walker_func_size = 0;
    return false;
}

}; // namespace

bool FlattenExtendedUserdataPass(IR::Program& program, SrtMemoryReader reader,
                                const RuntimeInfo* runtime_info) {
    Shader::Info& info = program.info;
    PassInfo pass_info;
    if (runtime_info && runtime_info->stage == Stage::Vertex && info.stage == Stage::Vertex &&
        info.l_stage == LogicalStage::Vertex) {
        const auto& mapping = runtime_info->vs_info.draw_user_data;
        if (!mapping.IsValid()) {
            info.translation_failed = true;
            return false;
        }
        pass_info.dynamic_user_data_mask = mapping.RegisterMask();
    }
    CollectHostDescriptorDependencies(program, pass_info);
    if (pass_info.dynamic_descriptor_dependency) {
        LOG_ERROR(Render_Recompiler,
                  "Rejecting host descriptor capture from dynamic draw user data in shader {:#x}",
                  info.pgm_hash);
        info.translation_failed = true;
        return false;
    }
    if (pass_info.invalid_descriptor_span) {
        info.translation_failed = true;
        return false;
    }

    // traverse at end and assign offsets to duplicate readconsts, using
    // vn_to_inst as the source
    boost::container::small_vector<IR::Inst*, 32> all_readconsts;

    for (auto r_it = program.post_order_blocks.rbegin(); r_it != program.post_order_blocks.rend();
         r_it++) {
        IR::Block* block = *r_it;
        for (IR::Inst& inst : *block) {
            if (const auto match = MatchDynamicImageTable(inst)) {
                // This pass has no device profile. Reserve only a policy-safe host window here;
                // ResourceTrackingPass separately gates stage, feature, and Vulkan capacities.
                if (PlanDynamicImageTable(match->Pattern(0U, NUM_IMAGES, true, true))) {
                    pass_info.dynamic_image_tables.push_back(*match);
                }
            }
            if (inst.GetOpcode() == IR::Opcode::ReadConst) {
                if (!inst.Arg(1).IsImmediate()) {
                    LOG_WARNING(Render_Recompiler, "ReadConst has non-immediate offset");
                    continue;
                }

                pass_info.rejected_scalar = nullptr;
                pass_info.rejection_reason = {};
                // Validation depends on whether this particular read is a host descriptor. Do
                // not reuse a cached rejection from a data-only read that shares its user-data
                // source with the descriptor pointer.
                pass_info.scalar_validation.clear();
                IR::Inst* ptr_composite = inst.Arg(0).TryInstRecursive();
                const bool allow_shared_user_data_float_use =
                    pass_info.host_descriptor_dependencies.contains(&inst);
                if (!HasSrtSource(inst.Arg(0)) || !ptr_composite ||
                    ptr_composite->GetOpcode() != IR::Opcode::CompositeConstructU32x2 ||
                    !IsSupportedPointerScalar(ptr_composite->Arg(0), pass_info, 0,
                                              allow_shared_user_data_float_use) ||
                    !IsSupportedPointerScalar(ptr_composite->Arg(1), pass_info, 0,
                                              allow_shared_user_data_float_use)) {
                    if (EmulatorSettings.IsDirectMemoryAccessEnabled() &&
                        !pass_info.host_descriptor_dependencies.contains(&inst)) {
                        // SMRD data pointers may come from ReadFirstLane/shared memory. Keep the
                        // entire address expression on the GPU; do not capture a nearby ancestor.
                        // Only data reads can use flags=0: a host descriptor consumer would treat
                        // it as user-data register zero, and the no-DMA backend has no direct read.
                        inst.SetFlags<u32>(0U);
                        continue;
                    }
                    // Descriptor consumers need a correct host snapshot too; leaving a zero flat
                    // offset here could make resource tracking read user-data register zero.
                    LogUnsupportedPointer(info, inst, pass_info);
                    info.translation_failed = true;
                    return false;
                }

                all_readconsts.push_back(&inst);
                auto* const canonical = pass_info.DeduplicateInstruction(&inst);
                if (pass_info.host_descriptor_dependencies.contains(&inst)) {
                    pass_info.host_descriptor_dependencies.insert(canonical);
                }
                if (const auto span = pass_info.descriptor_dword_counts.find(&inst);
                    span != pass_info.descriptor_dword_counts.end()) {
                    const u32 width = span->second;
                    auto& canonical_width = pass_info.descriptor_dword_counts[canonical];
                    canonical_width = std::max(canonical_width, width);
                }
                if (canonical != &inst) {
                    // This is a duplicate of a readconst we've already visited
                    continue;
                }

                ptr_composite = pass_info.DeduplicateInstruction(ptr_composite);
                auto ptr_uses_kv =
                    pass_info.pointer_uses.try_emplace(ptr_composite, PassInfo::PtrUserList{});
                PassInfo::PtrUserList& user_list = ptr_uses_kv.first->second;
                if (ptr_uses_kv.second) {
                    pass_info.pointer_order.push_back(ptr_composite);
                }
                user_list[inst.Arg(1).U32()] = &inst;
            }
        }
    }

    if (!GenerateSrtProgram(info, pass_info)) {
        return false;
    }

    // Assign offsets to duplicate readconsts
    for (IR::Inst* readconst : all_readconsts) {
        ASSERT(pass_info.vn_to_inst.contains(pass_info.gvn_table.GetValueNumber(readconst)));
        IR::Inst* original = pass_info.DeduplicateInstruction(readconst);
        readconst->SetFlags<u32>(original->Flags<u32>());
    }

    const auto flat_pointer_location = [](IR::Inst* pointer_dword) -> std::optional<u32> {
        switch (pointer_dword->GetOpcode()) {
        case IR::Opcode::GetUserData:
            return static_cast<u32>(pointer_dword->Arg(0).ScalarReg());
        case IR::Opcode::ReadConst: {
            const u32 location = pointer_dword->Flags<u32>();
            return location >= NUM_USER_DATA_REGS ? std::optional{location} : std::nullopt;
        }
        default:
            return std::nullopt;
        }
    };

    struct ReservationRequest {
        DynamicImageTablePlan plan{};
    };
    boost::container::small_vector<ReservationRequest, 2> requests;
    for (auto& match : pass_info.dynamic_image_tables) {
        const auto pointer_lo = flat_pointer_location(match.pointer_lo);
        const auto pointer_hi = flat_pointer_location(match.pointer_hi);
        if (!pointer_lo || !pointer_hi || *pointer_lo == std::numeric_limits<u32>::max() ||
            *pointer_hi != *pointer_lo + 1U) {
            continue;
        }
        const auto plan = PlanDynamicImageTable(match.Pattern(*pointer_lo, NUM_IMAGES, true, true));
        if (plan) {
            requests.push_back({.plan = *plan});
        }
    }
    std::ranges::sort(requests, {}, [](const ReservationRequest& request) {
        return std::tuple{request.plan.base_pointer_location, request.plan.offset_dw,
                          request.plan.snapshot_span_dw};
    });
    for (auto& request : requests) {
        [[maybe_unused]] const auto destination =
            info.srt_info.TryReserveRange(request.plan.base_pointer_location,
                                          request.plan.offset_dw, request.plan.snapshot_span_dw);
    }

    return info.RefreshFlatBuf(reader);
}

} // namespace Shader::Optimization

#else

namespace Shader {

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    UNREACHABLE_MSG("RegisterWalkerCode unimplemented for target architecture.");
}

namespace Optimization {

bool FlattenExtendedUserdataPass(IR::Program& program, SrtMemoryReader, const RuntimeInfo*) {
    UNREACHABLE_MSG("FlattenExtendedUserdataPass unimplemented for target architecture.");
}

} // namespace Optimization

} // namespace Shader

#endif
