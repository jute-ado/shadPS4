// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "common/compilation_timing.h"
#include "common/logging/log.h"

namespace Common::Log {

inline bool IsCompilationTimingEnabled() {
    const auto logger = ALL_LOGGERS.find(Class::Render_Compilation);
    return logger != ALL_LOGGERS.end() && logger->second &&
           logger->second->should_log(spdlog::level::debug);
}

struct CompilationTraceSink {
    void operator()(const CompilationTraceRecord& record) const {
        static constexpr std::array<std::string_view, 6> phases{"graphics_worker", "graphics_wait",
                                                                "owner_translate", "owner_spirv",
                                                                "owner_module",    "owner_compute"};
        static constexpr std::array<std::string_view, 6> outcomes{
            "exception", "complete", "ready", "failed", "stopped", "rejected"};
        const auto& context = record.context;
        switch (record.kind) {
        case CompilationRecordKind::Start:
            LOG_DEBUG(Render_Compilation,
                      "compilation_span_start id={} phase={} key={:#x} stage={} logical={} perm={} "
                      "atNs={}",
                      record.id, phases[std::to_underlying(context.phase)], context.key,
                      context.stage, context.logical, context.perm, record.at_ns);
            break;
        case CompilationRecordKind::Finish:
            LOG_DEBUG(Render_Compilation,
                      "compilation_span_finish id={} phase={} key={:#x} stage={} logical={} "
                      "perm={} atNs={} elapsedNs={} outcome={}",
                      record.id, phases[std::to_underlying(context.phase)], context.key,
                      context.stage, context.logical, context.perm, record.at_ns, record.elapsed_ns,
                      outcomes[std::to_underlying(record.outcome)]);
            break;
        case CompilationRecordKind::Overflow:
            LOG_WARNING(Render_Compilation,
                        "compilation_trace_overflow atNs={} limit={} admitted={} dropped={}",
                        record.at_ns, record.limit, record.admitted, record.dropped);
            break;
        case CompilationRecordKind::Terminal:
            LOG_CRITICAL(Render_Compilation,
                         "compilation_trace_terminal atNs={} admitted={} dropped={} code={:#x}",
                         record.at_ns, record.admitted, record.dropped, record.code);
            break;
        }
    }
};

template <typename ContextFactory>
auto TraceCompilation(ContextFactory&& context_factory) {
    return ScopedCompilationSpan{CompilationTraceSink{}, IsCompilationTimingEnabled(),
                                 ProcessCompilationTraceBudget(),
                                 std::forward<ContextFactory>(context_factory)};
}

inline void ReportCompilationTraceTerminal(std::uint32_t code) {
    EmitCompilationTerminal(IsCompilationTimingEnabled(), ProcessCompilationTraceBudget(),
                            CompilationTraceSink{}, code);
}

} // namespace Common::Log
