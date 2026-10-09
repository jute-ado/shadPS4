// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#if __has_include("video_core/indirect_draw_user_data.h")
#include "video_core/indirect_draw_user_data.h"
#define HAS_INDIRECT_DRAW_RESOLVER 1
#endif

namespace {

#ifndef HAS_INDIRECT_DRAW_RESOLVER
TEST(IndirectDrawUserDataResolver, MissingCheckedResolverIsRed) {
    FAIL() << "The inert indirect draw-user-data resolver has not been implemented";
}
#else
using VideoCore::IndirectDrawRegisterLocations;
using VideoCore::ResolveVertexDrawUserData;

TEST(IndirectDrawUserDataResolver, AbsentMetadataAlwaysReturnsDisabledFreshContext) {
    const auto populated = ResolveVertexDrawUserData({.present = true,
                                                      .base_vtx_loc = 0x4d,
                                                      .start_inst_loc = 0x4e,
                                                      .draw_index_present = true,
                                                      .draw_index_enabled = true,
                                                      .draw_index_loc = 0x4f});
    ASSERT_TRUE(populated);
    EXPECT_EQ(populated->base_vertex, 1U);
    EXPECT_EQ(populated->base_instance, 2U);
    EXPECT_EQ(populated->draw_index, 3U);
    const auto direct = ResolveVertexDrawUserData({.base_vtx_loc = 0x4d,
                                                   .start_inst_loc = 0x4e,
                                                   .draw_index_present = true,
                                                   .draw_index_enabled = true,
                                                   .draw_index_loc = 0x4f});
    ASSERT_TRUE(direct);
    EXPECT_EQ(*direct, Shader::DrawUserDataLocations{});
    EXPECT_EQ(ResolveVertexDrawUserData({}), direct);
}

TEST(IndirectDrawUserDataResolver, ZeroLocationsAndDisabledDrawIndexHaveNoAuthority) {
    const auto disabled = ResolveVertexDrawUserData(
        {.present = true, .draw_index_present = true, .draw_index_loc = 0xffff});
    ASSERT_TRUE(disabled);
    EXPECT_EQ(*disabled, Shader::DrawUserDataLocations{});
    const auto absent_field = ResolveVertexDrawUserData(
        {.present = true, .draw_index_enabled = true, .draw_index_loc = 0xffff});
    ASSERT_TRUE(absent_field);
    EXPECT_EQ(*absent_field, Shader::DrawUserDataLocations{});
    EXPECT_FALSE(ResolveVertexDrawUserData({.present = true,
                                            .draw_index_present = true,
                                            .draw_index_enabled = true,
                                            .draw_index_loc = 0}));
}

TEST(IndirectDrawUserDataResolver, VertexBankEdgesAreExactAndNeverMaskOversizedOffsets) {
    const auto edges =
        ResolveVertexDrawUserData({.present = true, .base_vtx_loc = 0x4c, .start_inst_loc = 0x5b});
    ASSERT_TRUE(edges);
    EXPECT_EQ(edges->base_vertex, 0U);
    EXPECT_EQ(edges->base_instance, 15U);
    for (const auto offset : {0x4bU, 0x5cU, 0xffffU, 0x1004eU, 0xffffffffU}) {
        EXPECT_FALSE(ResolveVertexDrawUserData({.present = true, .start_inst_loc = offset}));
    }
}

TEST(IndirectDrawUserDataResolver, ActiveNonVertexDestinationsAreExplicitlyUnsupported) {
    // This first implementation is conventional VS only. It must not silently discard a
    // real write to another physical stage when accepting the packet context.
    for (const auto offset : {0x0dU, 0x8dU, 0xcdU, 0x10dU, 0x14dU}) {
        EXPECT_FALSE(ResolveVertexDrawUserData({.present = true, .start_inst_loc = offset}));
    }
}

TEST(IndirectDrawUserDataResolver, CollidingEnabledDestinationsAreRejected) {
    EXPECT_FALSE(
        ResolveVertexDrawUserData({.present = true, .base_vtx_loc = 0x4e, .start_inst_loc = 0x4e}));
    EXPECT_FALSE(ResolveVertexDrawUserData({.present = true,
                                            .start_inst_loc = 0x4e,
                                            .draw_index_present = true,
                                            .draw_index_enabled = true,
                                            .draw_index_loc = 0x4e}));
    EXPECT_FALSE(ResolveVertexDrawUserData({.present = true,
                                            .base_vtx_loc = 0x4e,
                                            .draw_index_present = true,
                                            .draw_index_enabled = true,
                                            .draw_index_loc = 0x4e}));
    EXPECT_TRUE(ResolveVertexDrawUserData({.present = true,
                                           .start_inst_loc = 0x4e,
                                           .draw_index_present = true,
                                           .draw_index_loc = 0x4e}));
}

TEST(IndirectDrawUserDataResolver, LaterPacketCannotMutateEarlierResolvedValue) {
    IndirectDrawRegisterLocations packet{.present = true, .start_inst_loc = 0x4e};
    const auto first = ResolveVertexDrawUserData(packet);
    ASSERT_TRUE(first);
    packet.start_inst_loc = 0x4f;
    const auto second = ResolveVertexDrawUserData(packet);
    ASSERT_TRUE(second);
    EXPECT_EQ(first->base_instance, 2U);
    EXPECT_EQ(second->base_instance, 3U);
    EXPECT_NE(*first, *second);
}
#endif

} // namespace
