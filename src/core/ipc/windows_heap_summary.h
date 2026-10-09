// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <thread>

namespace Core::Ipc {

void MaybeStartWindowsHeapSummary(std::jthread& thread);

} // namespace Core::Ipc
