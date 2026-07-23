// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>

namespace Core {

/// Returns the side-effect-free Emulator Test Lab protocol and capabilities
/// exposed by this build.
std::string_view TestLabProbeJson();

} // namespace Core
