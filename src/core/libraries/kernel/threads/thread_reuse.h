// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <utility>

namespace Libraries::Kernel {

template <typename Thread>
void DestroyThreadForReuse(Thread* thread) {
    delete std::exchange(thread->sleepqueue, nullptr);
    std::destroy_at(thread);
}

} // namespace Libraries::Kernel
