// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::Kernel {

constexpr int ORBIS_KERNEL_PRIO_FIFO_DEFAULT = 700;
constexpr int ORBIS_KERNEL_PRIO_FIFO_LOWEST = 256;
constexpr int ORBIS_KERNEL_PRIO_FIFO_HIGHEST = 767;
constexpr int ORBIS_KERNEL_PRIO_OTHER_DEFAULT = 900;
constexpr int ORBIS_KERNEL_PRIO_OTHER_LOWEST = 768;
constexpr int ORBIS_KERNEL_PRIO_OTHER_HIGHEST = 959;
constexpr int ORBIS_KERNEL_PRIO_RR_DEFAULT = 700;
constexpr int ORBIS_KERNEL_PRIO_RR_LOWEST = 256;
constexpr int ORBIS_KERNEL_PRIO_RR_HIGHEST = 767;

enum class SchedPolicy : u32 {
    Fifo = 1,
    Other = 2,
    RoundRobin = 3,
};

} // namespace Libraries::Kernel
