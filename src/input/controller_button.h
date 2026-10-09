// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string_view>

#include "core/libraries/pad/pad.h"

namespace Input {

std::optional<Libraries::Pad::OrbisPadButtonDataOffset> ParseControllerButton(
    std::string_view name);

} // namespace Input
