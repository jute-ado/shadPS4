// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>
#include <tuple>
#include <unordered_map>
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
    // keys are GetUserData or ReadConst instructions that are used as pointers
    std::unordered_map<IR::Inst*, PtrUserList> pointer_uses;
    // GetUserData instructions corresponding to sgpr_base of SRT roots
    boost::container::small_flat_map<IR::ScalarReg, IR::Inst*, 1> srt_roots;

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

static inline void PushPtr(Xbyak::CodeGenerator& c, u32 off_dw) {
    // Keep the stack aligned for the optional range-sync callback at every tree depth.
    c.sub(rsp, 16);
    c.mov(ptr[rsp], rdi);
    c.mov(rdi, ptr[rdi + (off_dw << 2)]);
    c.mov(r10, 0xFFFFFFFFFFFFULL);
    c.and_(rdi, r10);
}

static inline void PopPtr(Xbyak::CodeGenerator& c) {
    c.mov(rdi, ptr[rsp]);
    c.add(rsp, 16);
};

struct SrtSourceRange {
    u64 begin_dw;
    u64 end_dw;
};

static boost::container::small_vector<SrtSourceRange, 4> GetSourceRanges(
    PassInfo::PtrUserList& use_list, PassInfo& pass_info) {
    boost::container::small_vector<SrtSourceRange, 4> ranges;
    for (const auto [src_off_dw, use] : use_list) {
        const u64 begin_dw = src_off_dw;
        // A nested SRT pointer consumes both address dwords. Synchronizing both prevents a torn
        // low/high pointer load across a tracking-page boundary.
        const u64 end_dw = begin_dw + (pass_info.GetUsesAsPointer(use) ? 2U : 1U);
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
    for (const auto& range : GetSourceRanges(*use_list, pass_info)) {
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

    // Follow the validated flattened pointer. If its range was refused, it is zero and the child
    // synchronizer receives derived addresses without a raw guest-memory dereference.
    for (const auto [src_off_dw, use] : *use_list) {
        if (pass_info.GetUsesAsPointer(use)) {
            c.sub(rsp, 16);
            c.mov(ptr[rsp], rdi);
            c.mov(rdi, ptr[r13 + (use->Flags<u32>() << 2)]);
            c.mov(r10, 0xFFFFFFFFFFFFULL);
            c.and_(rdi, r10);
            VisitNode(use, pass_info, c);
            c.mov(rdi, ptr[rsp]);
            c.add(rsp, 16);
        }
    }
}

static void VisitPointer(u32 off_dw, IR::Inst* subtree, PassInfo& pass_info,
                         Xbyak::CodeGenerator& c) {
    PushPtr(c, off_dw);
    VisitNode(subtree, pass_info, c);
    PopPtr(c);
}

static void GenerateSrtProgram(Info& info, PassInfo& pass_info) {
    Xbyak::CodeGenerator& c = g_srt_codegen;

    if (pass_info.srt_roots.empty()) {
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

    for (const auto& [sgpr_base, root] : pass_info.srt_roots) {
        c.mov(rdi, r12);
        c.mov(rsi, r13);
        VisitPointer(static_cast<u32>(sgpr_base), root, pass_info, c);
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

                IR::Inst* ptr_composite = inst.Arg(0).InstRecursive();

                const auto pred = [](IR::Inst* inst) -> std::optional<IR::Inst*> {
                    if (inst->GetOpcode() == IR::Opcode::GetUserData ||
                        inst->GetOpcode() == IR::Opcode::ReadConst) {
                        return inst;
                    }
                    return std::nullopt;
                };
                auto base0 = IR::BreadthFirstSearch(ptr_composite->Arg(0), pred);
                auto base1 = IR::BreadthFirstSearch(ptr_composite->Arg(1), pred);
                if (!base0 || !base1) {
                    // Direct guest-memory reads, such as MUBUF ADDR64, may use a VGPR-derived
                    // address. They are not part of an extended user-data/SRT tree and must
                    // remain dynamic for the backend's direct-memory path.
                    continue;
                }

                all_readconsts.push_back(&inst);
                if (pass_info.DeduplicateInstruction(&inst) != &inst) {
                    // This is a duplicate of a readconst we've already visited
                    continue;
                }

                IR::Inst* ptr_lo = base0.value();
                ptr_lo = pass_info.DeduplicateInstruction(ptr_lo);

                auto ptr_uses_kv =
                    pass_info.pointer_uses.try_emplace(ptr_lo, PassInfo::PtrUserList{});
                PassInfo::PtrUserList& user_list = ptr_uses_kv.first->second;

                user_list[inst.Arg(1).U32()] = &inst;

                if (ptr_lo->GetOpcode() == IR::Opcode::GetUserData) {
                    IR::ScalarReg ud_reg = ptr_lo->Arg(0).ScalarReg();
                    pass_info.srt_roots[ud_reg] = ptr_lo;
                }
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
