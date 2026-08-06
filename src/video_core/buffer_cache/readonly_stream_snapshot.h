// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

namespace VideoCore {

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
