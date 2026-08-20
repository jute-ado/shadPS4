// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <queue>

#include <boost/container/small_vector.hpp>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/value.h"

namespace Shader::Optimization {

inline void RecordAddr64PointerRoots(const IR::Inst& inst, u32 buffer_binding, Info& info) {
    if (inst.GetOpcode() != IR::Opcode::ReadConstBufferAddr64 ||
        info.addr64_pointer_roots_overflow || inst.Arg(1).IsImmediate()) {
        return;
    }

    boost::container::small_vector<u32, MaxAddr64ReadConstDependencies> dependencies;
    Addr64PointerTraversalSet traversal;
    std::queue<const IR::Inst*> pending;
    const auto fail_closed = [&] {
        info.num_addr64_pointer_roots = 0;
        info.addr64_pointer_roots_overflow = true;
    };
    const IR::Inst* root = inst.Arg(1).InstRecursive();
    if (traversal.TryInsert(reinterpret_cast<std::uintptr_t>(root)) !=
        Addr64PointerTraversalInsert::Inserted) {
        fail_closed();
        return;
    }
    pending.push(root);
    while (!pending.empty()) {
        const IR::Inst* current = pending.front();
        pending.pop();
        if (current->GetOpcode() == IR::Opcode::ReadConst) {
            if (dependencies.size() == MaxAddr64ReadConstDependencies) {
                fail_closed();
                return;
            }
            dependencies.push_back(current->Flags<u32>());
        }
        for (u32 arg = 0; arg < current->NumArgs(); ++arg) {
            const IR::Value value = current->Arg(arg);
            if (!value.IsImmediate()) {
                const IR::Inst* dependency = value.InstRecursive();
                switch (traversal.TryInsert(reinterpret_cast<std::uintptr_t>(dependency))) {
                case Addr64PointerTraversalInsert::Inserted:
                    pending.push(dependency);
                    break;
                case Addr64PointerTraversalInsert::Duplicate:
                    break;
                case Addr64PointerTraversalInsert::Overflow:
                    fail_closed();
                    return;
                }
            }
        }
    }

    const auto selection = SelectAddr64PointerRoots(buffer_binding, dependencies);
    if (selection.overflow) {
        fail_closed();
        return;
    }
    for (u32 i = 0; i < selection.count; ++i) {
        const auto& root = selection.values[i];
        const auto begin = info.addr64_pointer_roots.begin();
        const auto end = begin + info.num_addr64_pointer_roots;
        if (std::ranges::find(begin, end, root) != end) {
            continue;
        }
        if (info.num_addr64_pointer_roots == info.addr64_pointer_roots.size()) {
            info.num_addr64_pointer_roots = 0;
            info.addr64_pointer_roots_overflow = true;
            return;
        }
        info.addr64_pointer_roots[info.num_addr64_pointer_roots++] = root;
    }
}

} // namespace Shader::Optimization
