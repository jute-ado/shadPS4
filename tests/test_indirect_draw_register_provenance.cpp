// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#if __has_include("video_core/indirect_draw_register_provenance.h")
#include "video_core/indirect_draw_register_provenance.h"
#define HAS_INDIRECT_REGISTER_PROVENANCE 1
#endif

namespace {

std::string ReadSource(const char* path) {
    std::ifstream file{path, std::ios::binary};
    EXPECT_TRUE(file.is_open());
    std::string text{std::istreambuf_iterator<char>{file}, {}};
    std::erase(text, '\r');
    return text;
}

TEST(IndirectDrawRegisterProvenanceSource, AllFivePacketsForwardMetadataInBothMarkerBranches) {
    const auto text = ReadSource(SHADPS4_INDEX_PROVENANCE_LIVERPOOL_SOURCE);
    for (const auto opcode : {"DrawIndirect", "DrawIndirectMulti", "DrawIndexIndirect",
                              "DrawIndexIndirectMulti", "DrawIndexIndirectCountMulti"}) {
        const auto marker = std::string{"case PM4ItOpcode::"} + opcode + ": {";
        const auto begin = text.find(marker);
        ASSERT_NE(begin, std::string::npos) << opcode;
        const auto end = text.find("case PM4ItOpcode::", begin + marker.size());
        ASSERT_NE(end, std::string::npos) << opcode;
        const auto body = text.substr(begin, end - begin);
        EXPECT_NE(body.find("CaptureIndirectDrawRegisterLocations("), std::string::npos) << opcode;
        std::size_t count{};
        for (std::size_t position{};
             (position = body.find("register_locations);", position)) != std::string::npos;
             ++position) {
            ++count;
        }
        EXPECT_EQ(count, 2U) << opcode;
    }
}

TEST(IndirectDrawRegisterProvenanceSource, DrawCarriesOptionalMetadataIntoExistingReceipt) {
    const auto header = ReadSource(SHADPS4_INDEX_PROVENANCE_RASTERIZER_HEADER);
    EXPECT_NE(header.find("VideoCore::IndirectDrawRegisterLocations register_locations = {}"),
              std::string::npos);
    const auto text = ReadSource(SHADPS4_INDEX_PROVENANCE_RASTERIZER_SOURCE);
    const auto begin = text.find("void Rasterizer::DrawIndirect(");
    const auto end = text.find("void Rasterizer::DispatchDirect(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    EXPECT_NE(text.substr(begin, end - begin)
                  .find("RecordIndexBindingProvenance(*pipeline, true, 0, register_locations)"),
              std::string::npos);
}

TEST(IndirectDrawRegisterProvenanceSource, GatedCpuRegisterObservationNeverReadsGpuArguments) {
    const auto text = ReadSource(SHADPS4_INDEX_PROVENANCE_RASTERIZER_SOURCE);
    const auto begin = text.find("void Rasterizer::RecordIndexBindingProvenance(");
    const auto end = text.find("\nvoid Rasterizer::", begin + 1);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = text.substr(begin, end - begin);
    const auto observe = body.find("ObserveIndirectUserDataRegister(");
    ASSERT_NE(observe, std::string::npos);
    EXPECT_LT(body.find("if (!index_binding_provenance.Enabled())"), observe);
    EXPECT_LT(body.find("if (!row)"), observe);
    for (const auto field :
         {"indirect_userdata_applicable={}", "base_vtx_loc={:#x}", "start_inst_loc={:#x}",
          "draw_index_loc={:#x}", "cpu_value_is_gpu_indirect_argument=false"}) {
        EXPECT_NE(body.find(field), std::string::npos) << field;
    }
    for (const auto forbidden :
         {"ReadMemory(", "ReadMemorySnapshot(", "Wait(", "Finish(", "& 0xffff", "& 0xFFFF"}) {
        EXPECT_EQ(body.find(forbidden), std::string::npos) << forbidden;
    }
}

#ifndef HAS_INDIRECT_REGISTER_PROVENANCE
TEST(IndirectDrawRegisterProvenance, MissingBoundedRegisterMapperIsRed) {
    FAIL() << "The diagnostic-only checked register mapper has not been implemented";
}
#else
TEST(IndirectDrawRegisterProvenance, MapsEveryUserDataBankAndBothEdges) {
    constexpr std::array<unsigned, 6> bases{0x0c, 0x4c, 0x8c, 0xcc, 0x10c, 0x14c};
    for (unsigned stage = 0; stage < bases.size(); ++stage) {
        for (unsigned index : {0U, 15U}) {
            const auto location = VideoCore::MapIndirectUserDataLocation(bases[stage] + index);
            ASSERT_TRUE(location);
            EXPECT_EQ(location->stage, stage);
            EXPECT_EQ(location->sgpr, index);
        }
    }
}

TEST(IndirectDrawRegisterProvenance, RejectsDisabledHolesAndOversizedLocationsWithoutMasking) {
    for (const auto location :
         {0U, 0x0bU, 0x1cU, 0x4bU, 0x5cU, 0x15cU, 0xffffU, 0x1004cU, 0x2004cU, 0xffffffffU}) {
        EXPECT_FALSE(VideoCore::MapIndirectUserDataLocation(location)) << location;
    }
}

TEST(IndirectDrawRegisterProvenance, ReturnsWholeCpuValueAndDoesNotInventGpuArgument) {
    unsigned reads{};
    const auto reader = [&](unsigned stage, unsigned sgpr) {
        ++reads;
        EXPECT_EQ(stage, 1U);
        EXPECT_EQ(sgpr, 2U);
        return 0xfedc1234U;
    };
    const auto valid = VideoCore::ObserveIndirectUserDataRegister(0x4e, true, reader);
    EXPECT_TRUE(valid.value_known);
    EXPECT_EQ(valid.value, 0xfedc1234U);
    EXPECT_EQ(valid.stage, 1U);
    EXPECT_EQ(valid.sgpr, 2U);
    EXPECT_EQ(reads, 1U);
    for (const auto location : {0U, 0x1004eU, 0x4eU}) {
        const auto disabled = VideoCore::ObserveIndirectUserDataRegister(location, false, reader);
        EXPECT_FALSE(disabled.value_known);
    }
    EXPECT_FALSE(VideoCore::ObserveIndirectUserDataRegister(0x1004e, true, reader).value_known);
    EXPECT_EQ(reads, 1U);
}

struct Field {
    unsigned value;
    unsigned Value() const {
        return value;
    }
};
struct SimplePacket {
    Field base_vtx_loc;
    Field start_inst_loc;
};
struct CountPacket : SimplePacket {
    Field draw_index_loc;
    Field draw_index_enable;
};

TEST(IndirectDrawRegisterProvenance, CopiesPacketFieldsWithoutGivingMissingFieldsAuthority) {
    const auto simple =
        VideoCore::CaptureIndirectDrawRegisterLocations(SimplePacket{{0x4d}, {0x4e}});
    EXPECT_TRUE(simple.present);
    EXPECT_EQ(simple.base_vtx_loc, 0x4dU);
    EXPECT_EQ(simple.start_inst_loc, 0x4eU);
    EXPECT_FALSE(simple.draw_index_present);
    EXPECT_FALSE(simple.draw_index_enabled);
    const auto count =
        VideoCore::CaptureIndirectDrawRegisterLocations(CountPacket{{{0x4d}, {0x4e}}, {0x4f}, {1}});
    EXPECT_TRUE(count.draw_index_present);
    EXPECT_TRUE(count.draw_index_enabled);
    EXPECT_EQ(count.draw_index_loc, 0x4fU);
    const auto disabled =
        VideoCore::CaptureIndirectDrawRegisterLocations(CountPacket{{{0}, {0}}, {0x4f}, {0}});
    EXPECT_TRUE(disabled.draw_index_present);
    EXPECT_FALSE(disabled.draw_index_enabled);
    EXPECT_FALSE(VideoCore::IndirectDrawRegisterLocations{}.present);
}
#endif

} // namespace
