// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace VideoCore {

constexpr bool NeedsBufferBarrier(bool same_access_and_stage, bool source_writes,
                                  bool destination_writes) noexcept {
    return !same_access_and_stage || source_writes || destination_writes;
}

} // namespace VideoCore
