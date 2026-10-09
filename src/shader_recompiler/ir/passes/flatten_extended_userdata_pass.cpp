// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
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

static Xbyak::CodeGenerator g_srt_codegen(32_MB);
static const u8* g_srt_codegen_start = nullptr;

namespace Shader {

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    const auto func_addr = (PFN_SrtWalker)g_srt_codegen.getCurr();
    g_srt_codegen.db(ptr, size);
    g_srt_codegen.ready();
    return func_addr;
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
    // Only handle if the fault address is within the SRT code range
    const u8* code_start = g_srt_codegen_start;
    const u8* code_end = code_start + g_srt_codegen.getSize();
    const void* code = Common::GetRip(context);
    if (code < code_start || code >= code_end) {
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
                    enqueue(inst.Arg(arg));
                }
            }
        }
    }
    while (!pending.empty()) {
        auto* const inst = pending.back();
        pending.pop_back();
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            enqueue(inst->Arg(arg));
        }
    }
}

// Only evaluate explicit, side-effect-free scalar operations. In particular, looking through an
// unknown operation to its first user-data ancestor changes the guest address rather than merely
// discovering its provenance. The supported select/comparison pattern covers S_ADD/S_ADDC carry.
static bool IsSupportedPointerScalar(IR::Value value, PassInfo& pass_info, u32 depth = 0) {
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
        return IsSupportedPointerScalar(argument, pass_info, depth + 1);
    };
    switch (inst->GetOpcode()) {
    case IR::Opcode::GetUserData:
        return result(static_cast<u32>(inst->Arg(0).ScalarReg()) < NUM_USER_DATA_REGS);
    case IR::Opcode::ReadConst: {
        auto* const pointer = inst->Arg(0).TryInstRecursive();
        return result(inst->Arg(1).IsImmediate() && pointer && HasSrtSource(inst->Arg(0)) &&
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
                "read_offset_dw={} graph={}",
                info.pgm_hash, info.stage,
                pass_info.rejection_reason.empty() ? "pointer is not a scalar pair"
                                                   : pass_info.rejection_reason,
                pass_info.rejected_scalar ? IR::NameOf(pass_info.rejected_scalar->GetOpcode())
                                          : std::string_view{"none"},
                read.Arg(1).U32(), graph);
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
    c.mov(r10, 0xFFFFFFFFFFFFULL);
    c.and_(rdi, r10);
}

struct SrtSourceRange {
    u64 begin_dw;
    u64 end_dw;
};

static boost::container::small_vector<SrtSourceRange, 4> GetSourceRanges(
    PassInfo::PtrUserList& use_list) {
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

static void VisitNode(IR::Inst* subtree, PassInfo& pass_info, Xbyak::CodeGenerator& c) {
    PassInfo::PtrUserList* use_list = pass_info.GetUsesAsPointer(subtree);
    ASSERT(use_list);

    struct SrtCopy {
        u32 src_off_dw;
        u32 dst_off_dw;
    };
    boost::container::small_vector<SrtCopy, 8> copies;
    for (const auto [src_off_dw, use] : *use_list) {
        const u32 dst_off_dw = pass_info.dst_off_dw++;
        use->SetFlags<u32>(dst_off_dw);
        copies.push_back({src_off_dw, dst_off_dw});
    }

    // Read every source range into private stack storage. A reader may obtain authoritative GPU
    // bytes without publishing them back into mutable guest memory. Without a reader, preserve
    // the original direct-memory behavior for callers that only consume CPU-owned SRTs.
    EmitPointer(subtree, pass_info, c);
    c.mov(rsi, r13);
    for (const auto& range : GetSourceRanges(*use_list)) {
        const u64 range_size = (range.end_dw - range.begin_dw) * sizeof(u32);
        const u64 scratch_size = Common::AlignUp(range_size, 16ULL);
        const u64 stack_size = scratch_size + 16;
        Xbyak::Label direct_read;
        Xbyak::Label refused;
        Xbyak::Label null_source;
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

        c.L(direct_read);
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

static void GenerateSrtProgram(Info& info, PassInfo& pass_info) {
    Xbyak::CodeGenerator& c = g_srt_codegen;

    if (pass_info.pointer_order.empty()) {
        return;
    }

    // Register the signal handler for SRT walker, if not already registered
    if (g_srt_codegen_start == nullptr) {
        g_srt_codegen_start = c.getCurr();
        auto* signals = Core::Signals::Instance();
        // Call after the memory invalidation handler
        constexpr u32 priority = 1;
        signals->RegisterAccessViolationHandler(SrtWalkerSignalHandler, priority);
    }

    info.srt_info.walker_func = c.getCurr<PFN_SrtWalker>();
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

    info.srt_info.walker_func_size =
        c.getCurr() - reinterpret_cast<const u8*>(info.srt_info.walker_func);

    if (EmulatorSettings.IsDumpShaders()) {
        DumpSrtProgram(info, reinterpret_cast<const u8*>(info.srt_info.walker_func),
                       info.srt_info.walker_func_size);
    }

    info.srt_info.flattened_bufsize_dw = pass_info.dst_off_dw;
}

}; // namespace

bool FlattenExtendedUserdataPass(IR::Program& program, SrtMemoryReader reader) {
    Shader::Info& info = program.info;
    PassInfo pass_info;
    CollectHostDescriptorDependencies(program, pass_info);

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
                IR::Inst* ptr_composite = inst.Arg(0).TryInstRecursive();
                if (!HasSrtSource(inst.Arg(0)) || !ptr_composite ||
                    ptr_composite->GetOpcode() != IR::Opcode::CompositeConstructU32x2 ||
                    !IsSupportedPointerScalar(ptr_composite->Arg(0), pass_info) ||
                    !IsSupportedPointerScalar(ptr_composite->Arg(1), pass_info)) {
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
                if (pass_info.DeduplicateInstruction(&inst) != &inst) {
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

    GenerateSrtProgram(info, pass_info);

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

bool FlattenExtendedUserdataPass(IR::Program& program, SrtMemoryReader) {
    UNREACHABLE_MSG("FlattenExtendedUserdataPass unimplemented for target architecture.");
}

} // namespace Optimization

} // namespace Shader

#endif
