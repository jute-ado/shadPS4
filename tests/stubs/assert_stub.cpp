// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>

void assert_fail_impl() {
    std::abort();
}

[[noreturn]] void unreachable_impl() {
    std::abort();
}
