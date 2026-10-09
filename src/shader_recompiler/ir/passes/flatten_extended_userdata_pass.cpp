// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <boost/container/flat_map.hpp>
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>
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
#include "shader_recompiler/ir/passes/srt.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/reg.h"
#include "shader_recompiler/ir/srt_gvn_table.h"
#include "shader_recompiler/ir/value.h"

#ifdef ARCH_X86_64

using namespace Xbyak::util;

static Xbyak::CodeGenerator g_srt_codegen(32_MB);
static std::atomic<size_t> g_srt_codegen_published_size{};
static std::mutex g_srt_codegen_mutex;
static std::once_flag g_srt_handler_once;

namespace {
static bool SrtWalkerSignalHandler(void* context, void* fault_address);

static void EnsureSrtHandlerRegistered() {
    std::call_once(g_srt_handler_once, [] {
        // Decoder construction may allocate and is not safe inside a POSIX/VEH fault path.
        static_cast<void>(Common::Decoder::Instance());
        constexpr u32 priority = 1;
        Core::Signals::Instance()->RegisterAccessViolationHandler(SrtWalkerSignalHandler, priority);
    });
}
} // namespace

namespace Shader {

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    std::scoped_lock lock{g_srt_codegen_mutex};
    EnsureSrtHandlerRegistered();
    const auto func_addr = (PFN_SrtWalker)g_srt_codegen.getCurr();
    g_srt_codegen.db(ptr, size);
    g_srt_codegen.ready();
    g_srt_codegen_published_size.store(g_srt_codegen.getSize(), std::memory_order_release);
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
    if (Common::IsWriteError(context)) {
        return false;
    }

    const auto code_start = reinterpret_cast<uintptr_t>(g_srt_codegen.getCode());
    const auto code_size = g_srt_codegen_published_size.load(std::memory_order_acquire);
    const auto code = reinterpret_cast<uintptr_t>(Common::GetRip(context));
    if (code < code_start || code - code_start >= code_size) {
        return false;
    }

    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    const ZyanStatus status = Common::Decoder::Instance()->decodeInstruction(
        instruction, operands, reinterpret_cast<void*>(code), 15);
    if (!ZYAN_SUCCESS(status) || instruction.mnemonic != ZYDIS_MNEMONIC_MOV ||
        operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        operands[1].type != ZYDIS_OPERAND_TYPE_MEMORY ||
        operands[1].mem.base != ZYDIS_REGISTER_RDI) {
        return false;
    }

    switch (operands[0].reg.value) {
    case ZYDIS_REGISTER_RDI:
        Common::SetRdi(context, 0);
        break;
    case ZYDIS_REGISTER_R10D:
        Common::SetR10(context, 0);
        break;
    default:
        return false;
    }

    Common::IncrementRip(context, instruction.length);
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
        // A ReadConst used as another SRT pointer consumes both address dwords. Synchronizing
        // the entire pointer prevents a torn low/high load across tracking-page boundaries.
        const u64 end_dw = begin_dw + (pass_info.GetUsesAsPointer(use) ? 2U : 1U);
        if (!ranges.empty() && begin_dw <= ranges.back().end_dw) {
            ranges.back().end_dw = std::max(ranges.back().end_dw, end_dw);
        } else {
            ranges.push_back({begin_dw, end_dw});
        }
    }
    return ranges;
}

static void EmitRangeSync(Xbyak::CodeGenerator& c, const SrtSourceRange& range) {
    Xbyak::Label skip;
    c.mov(eax, 1);
    c.test(r14, r14);
    c.jz(skip);

    // The callback follows the PS4 SysV ABI: context, guest address, byte size. Preserve the
    // current tree pointer in a 16-byte slot so nested walkers retain call-site alignment.
    c.sub(rsp, 16);
    c.mov(ptr[rsp], rdi);
    c.lea(rsi, ptr[rdi + range.begin_dw * sizeof(u32)]);
    c.mov(rdx, (range.end_dw - range.begin_dw) * sizeof(u32));
    c.mov(rdi, r15);
    c.call(r14);
    c.mov(rdi, ptr[rsp]);
    c.add(rsp, 16);
    c.mov(rsi, r13);

    c.L(skip);
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

    // A synchronizer refusal means the guest range is not safe to dereference. Publish zeros for
    // this invocation and keep the generated walker immutable so later valid invocations can
    // recover. Contiguous guest words remain contiguous in the flattened buffer.
    for (const auto& range : GetSourceRanges(*use_list, pass_info)) {
        Xbyak::Label readable;
        Xbyak::Label finished;
        EmitRangeSync(c, range);
        c.test(al, al);
        c.jnz(readable, Xbyak::CodeGenerator::T_NEAR);
        for (const auto [src_off_dw, dst_off_dw] : copies) {
            if (src_off_dw >= range.begin_dw && src_off_dw < range.end_dw) {
                c.mov(dword[rsi + (dst_off_dw << 2)], 0);
            }
        }
        c.jmp(finished, Xbyak::CodeGenerator::T_NEAR);
        c.L(readable);
        for (const auto [src_off_dw, dst_off_dw] : copies) {
            if (src_off_dw >= range.begin_dw && src_off_dw < range.end_dw) {
                c.mov(r10d, ptr[rdi + (src_off_dw << 2)]);
                c.mov(ptr[rsi + (dst_off_dw << 2)], r10d);
            }
        }
        c.L(finished);
    }

    // Then visit any children used as pointers
    for (const auto [src_off_dw, use] : *use_list) {
        if (pass_info.GetUsesAsPointer(use)) {
            // Follow the just-validated flattened pointer. If its source range was refused, this
            // is zero and the child synchronizer will reject its derived ranges without a raw
            // guest-memory access.
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
    std::scoped_lock lock{g_srt_codegen_mutex};
    Xbyak::CodeGenerator& c = g_srt_codegen;

    if (pass_info.srt_roots.empty()) {
        return;
    }

    EnsureSrtHandlerRegistered();

    info.srt_info.walker_func = c.getCurr<PFN_SrtWalker>();
    pass_info.dst_off_dw = NUM_USER_DATA_REGS;
    ASSERT(pass_info.dst_off_dw == info.srt_info.flattened_bufsize_dw);

    // PS4_SYSV_ABI arguments arrive in rdi/rsi/rdx/rcx. The generated walker makes optional
    // callbacks, so retain all four values in callee-saved registers and align the stack once.
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
    g_srt_codegen_published_size.store(c.getSize(), std::memory_order_release);

    info.srt_info.walker_func_size =
        c.getCurr() - reinterpret_cast<const u8*>(info.srt_info.walker_func);

    if (EmulatorSettings.IsDumpShaders()) {
        DumpSrtProgram(info, reinterpret_cast<const u8*>(info.srt_info.walker_func),
                       info.srt_info.walker_func_size);
    }

    info.srt_info.flattened_bufsize_dw = pass_info.dst_off_dw;
}

}; // namespace

void FlattenExtendedUserdataPass(IR::Program& program, SrtMemorySynchronizer synchronizer) {
    Shader::Info& info = program.info;
    PassInfo pass_info;

    // traverse at end and assign offsets to duplicate readconsts, using
    // vn_to_inst as the source
    boost::container::small_vector<IR::Inst*, 32> all_readconsts;

    for (auto r_it = program.post_order_blocks.rbegin(); r_it != program.post_order_blocks.rend();
         r_it++) {
        IR::Block* block = *r_it;
        for (IR::Inst& inst : *block) {
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

    info.RefreshFlatBuf(synchronizer);
}

} // namespace Shader::Optimization

#else

namespace Shader {

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    UNREACHABLE_MSG("RegisterWalkerCode unimplemented for target architecture.");
}

namespace Optimization {

void FlattenExtendedUserdataPass(IR::Program& program, SrtMemorySynchronizer) {
    UNREACHABLE_MSG("FlattenExtendedUserdataPass unimplemented for target architecture.");
}

} // namespace Optimization

} // namespace Shader

#endif
