// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/ir/basic_block.h"

// Forced into the test target's microinstruction.cpp only. Exercise real use-list assertions
// in release tests without defining _DEBUG or changing the standard-library ABI.
// Parse the normal inline IR headers first so their definitions remain identical in every TU.
#undef DEBUG_ASSERT
#define DEBUG_ASSERT(condition) ASSERT(condition)
