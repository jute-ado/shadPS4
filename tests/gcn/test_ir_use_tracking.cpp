// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/ir/value.h"

namespace Shader::IR {

TEST(IrUseTracking, UpdatesUsersWhenOperandsAreReplacedOrMadeImmediate) {
    Inst source{Opcode::IAdd32, 0};
    Inst replacement{Opcode::IAdd32, 0};
    Inst first_user{Opcode::IAdd32, 0};
    Inst second_user{Opcode::IAdd32, 0};

    first_user.SetArg(0, Value{&source});
    second_user.SetArg(1, Value{&source});
    ASSERT_EQ(source.UseCount(), 2);

    first_user.SetArg(0, Value{u32{7}});
    EXPECT_EQ(source.UseCount(), 1);

    source.ReplaceUsesWith(Value{&replacement});
    EXPECT_EQ(source.UseCount(), 0);
    // The remaining user and source's preserved Identity both reference the replacement.
    ASSERT_EQ(replacement.UseCount(), 2);
    EXPECT_EQ(second_user.Arg(1).Inst(), &replacement);
}

} // namespace Shader::IR
