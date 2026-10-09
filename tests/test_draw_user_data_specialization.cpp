// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <optional>
#include <vector>
#include <gtest/gtest.h>

#include "common/serdes.h"
#include "shader_recompiler/specialization.h"
#include "shader_recompiler/specialization_identity.h"
#include "shader_recompiler/specialization_serialization.h"

namespace {
using namespace Shader;

StageSpecialization MakeDrawSpec(DrawUserDataLocations mapping) {
    StageSpecialization spec{};
    spec.runtime_info.Initialize(Stage::Vertex);
    spec.runtime_info.num_user_data = NUM_USER_DATA_REGS;
    spec.runtime_info.vs_info.draw_user_data = mapping;
    return spec;
}

std::vector<u8> EncodeDrawSpec(const StageSpecialization& spec) {
    Serialization::Archive archive;
    spec.Serialize(archive);
    return archive.TakeOff();
}

TEST(DrawUserDataSpecialization, EveryDestinationChangesSerializedPermutationIdentity) {
    const auto plain = MakeDrawSpec({});
    const auto plain_identity =
        ComputeShaderPermutationIdentity(0x1234, LogicalStage::Vertex, plain);
    for (const auto mapping :
         {DrawUserDataLocations{.base_vertex = 2}, DrawUserDataLocations{.base_instance = 2},
          DrawUserDataLocations{.draw_index = 2}}) {
        const auto mapped = MakeDrawSpec(mapping);
        EXPECT_NE(mapped.runtime_info, plain.runtime_info);
        EXPECT_NE(EncodeDrawSpec(mapped), EncodeDrawSpec(plain));
        EXPECT_NE(ComputeShaderPermutationIdentity(0x1234, LogicalStage::Vertex, mapped),
                  plain_identity);
    }
}

TEST(DrawUserDataSpecialization, AllDestinationsRoundTripWithoutBecomingRegisterZero) {
    const auto original = MakeDrawSpec({.base_vertex = 0, .base_instance = 2, .draw_index = 15});
    const auto bytes = EncodeDrawSpec(original);
    Serialization::Archive archive{std::vector<u8>{bytes}};
    StageSpecialization decoded{};
    ASSERT_TRUE(decoded.Deserialize(archive));
    EXPECT_EQ(decoded.runtime_info.vs_info.draw_user_data,
              original.runtime_info.vs_info.draw_user_data);
    const auto disabled = MakeDrawSpec({});
    Serialization::Archive disabled_archive{EncodeDrawSpec(disabled)};
    ASSERT_TRUE(decoded.Deserialize(disabled_archive));
    EXPECT_EQ(decoded.runtime_info.vs_info.draw_user_data, DrawUserDataLocations{});
}

TEST(DrawUserDataSpecialization, InvalidAndCollidingSerializedDestinationsAreRejected) {
    for (const auto mapping :
         {DrawUserDataLocations{.base_instance = 16U}, DrawUserDataLocations{.base_vertex = 254U},
          DrawUserDataLocations{.base_instance = 2U, .draw_index = 2U}}) {
        const auto invalid = MakeDrawSpec(mapping);
        Serialization::Archive archive{EncodeDrawSpec(invalid)};
        StageSpecialization decoded = MakeDrawSpec({.base_instance = 4U});
        EXPECT_FALSE(decoded.Deserialize(archive));
        EXPECT_EQ(decoded.runtime_info.vs_info.draw_user_data.base_instance, 4U);
    }
}

TEST(DrawUserDataSpecialization, DrawMappingInvalidatesPriorCompilerAndMetadataSchemas) {
    EXPECT_GE(ShaderBinaryVersion, 13U);
    EXPECT_GE(ShaderMetaVersion, 13U);
}
} // namespace
