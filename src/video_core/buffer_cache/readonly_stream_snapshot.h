// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

#include "common/scope_exit.h"

namespace VideoCore {

template <typename Acquire, typename Snapshot, typename Restore>
void WithReadonlyStreamPageTransaction(Acquire&& acquire, Snapshot&& snapshot, Restore&& restore) {
    SCOPE_EXIT {
        std::invoke(std::forward<Restore>(restore));
    };
    std::invoke(std::forward<Acquire>(acquire));
    std::invoke(std::forward<Snapshot>(snapshot));
}

template <typename Reserve, typename Transaction, typename Snapshot, typename Commit>
auto StageReadonlyStreamSnapshot(Reserve&& reserve, Transaction&& transaction, Snapshot&& snapshot,
                                 Commit&& commit) {
    auto reservation = std::invoke(std::forward<Reserve>(reserve));
    std::invoke(std::forward<Transaction>(transaction),
                [&] { std::invoke(std::forward<Snapshot>(snapshot), reservation); });
    std::invoke(std::forward<Commit>(commit));
    return reservation;
}

} // namespace VideoCore
