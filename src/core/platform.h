// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/singleton.h"
#include "common/types.h"

#include <magic_enum/magic_enum.hpp>

#include <array>
#include <functional>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

namespace Platform {

enum class InterruptId : u32 {
    Compute0RelMem = 0x00,
    Compute1RelMem = 0x01,
    Compute2RelMem = 0x02,
    Compute3RelMem = 0x03,
    Compute4RelMem = 0x04,
    Compute5RelMem = 0x05,
    Compute6RelMem = 0x06,
    GfxEop = 0x40,
    GfxFlip = 0x08,
    GpuIdle = 0x09,

    InterruptIdMax = 0x40, ///< Max possible value (GfxEop)
};

using IrqHandler = std::function<void(InterruptId)>;

struct IrqController {
    void RegisterOnce(InterruptId irq, IrqHandler handler) {
        ASSERT_MSG(static_cast<u32>(irq) <= static_cast<u32>(InterruptId::InterruptIdMax),
                   "Invalid IRQ number");
        auto& ctx = irq_contexts[static_cast<size_t>(irq)];
        std::unique_lock lock{ctx.m_lock};
        ctx.one_time_subscribers.emplace(std::move(handler));
    }

    void Register(InterruptId irq, IrqHandler handler, void* uid) {
        ASSERT_MSG(static_cast<u32>(irq) <= static_cast<u32>(InterruptId::InterruptIdMax),
                   "Invalid IRQ number");
        auto& ctx = irq_contexts[static_cast<size_t>(irq)];

        std::unique_lock lock{ctx.m_lock};
        ASSERT_MSG(ctx.persistent_handlers.find(uid) == ctx.persistent_handlers.cend(),
                   "The handler is already registered!");
        ctx.persistent_handlers.emplace(uid, std::move(handler));
    }

    void Unregister(InterruptId irq, void* uid) {
        ASSERT_MSG(static_cast<u32>(irq) <= static_cast<u32>(InterruptId::InterruptIdMax),
                   "Invalid IRQ number");
        auto& ctx = irq_contexts[static_cast<size_t>(irq)];
        std::unique_lock lock{ctx.m_lock};
        ctx.persistent_handlers.erase(uid);
    }

    void Signal(InterruptId irq) {
        ASSERT_MSG(static_cast<u32>(irq) <= static_cast<u32>(InterruptId::InterruptIdMax),
                   "Unexpected IRQ signaled");
        auto& ctx = irq_contexts[static_cast<size_t>(irq)];
        std::unique_lock signal_lock{ctx.m_signal_lock};
        std::vector<IrqHandler> persistent_handlers;
        IrqHandler one_time_handler;
        {
            std::unique_lock lock{ctx.m_lock};
            persistent_handlers.reserve(ctx.persistent_handlers.size());
            for (const auto& [uid, handler] : ctx.persistent_handlers) {
                persistent_handlers.emplace_back(handler);
            }
            if (!ctx.one_time_subscribers.empty()) {
                one_time_handler = std::move(ctx.one_time_subscribers.front());
                ctx.one_time_subscribers.pop();
            }
        }

        LOG_TRACE(Core, "IRQ signaled: {}", magic_enum::enum_name(irq));

        for (auto& handler : persistent_handlers) {
            handler(irq);
        }
        if (one_time_handler) {
            one_time_handler(irq);
        }
    }

private:
    struct IrqContext {
        std::unordered_map<void*, IrqHandler> persistent_handlers{};
        std::queue<IrqHandler> one_time_subscribers{};
        std::mutex m_lock{};
        std::mutex m_signal_lock{};
    };
    static constexpr size_t NumInterrupts = static_cast<size_t>(InterruptId::InterruptIdMax) + 1;
    std::array<IrqContext, NumInterrupts> irq_contexts{};
};

using IrqC = Common::Singleton<IrqController>;

} // namespace Platform
