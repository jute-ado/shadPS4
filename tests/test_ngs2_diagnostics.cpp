// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "core/libraries/ngs2/ngs2_diagnostics.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libraries::Ngs2::Diagnostic {
namespace {

struct Fixture : testing::Test {
    Observer observer;
    std::map<u64, std::vector<u8>> memory;
    std::vector<Record> records;
    size_t reads{};
    size_t read_bytes{};
    Reader reader = [this](u64 address, std::span<u8> output) {
        ++reads;
        read_bytes += output.size();
        for (auto& [base, bytes] : memory) {
            if (address >= base && address - base <= bytes.size() &&
                output.size() <= bytes.size() - (address - base)) {
                std::memcpy(output.data(), bytes.data() + (address - base), output.size());
                return true;
            }
        }
        return false;
    };
    Sink sink = [this](const Record& record) { records.push_back(record); };

    template <typename T>
    void Put(u64 base, size_t offset, T value) {
        auto& bytes = memory[base];
        bytes.resize(std::max(bytes.size(), offset + sizeof(T)));
        std::memcpy(bytes.data() + offset, &value, sizeof(T));
    }

    void Header(u64 base, u16 size, s16 next, u32 id) {
        Put(base, 0, size);
        Put(base, 2, next);
        Put(base, 4, id);
    }

    static u64 Value(const Record& record, std::string_view name) {
        for (u32 i = 0; i < record.count; ++i) {
            if (record.fields[i].name == name) {
                return record.fields[i].value;
            }
        }
        ADD_FAILURE() << "Missing field " << name;
        return 0;
    }
};

TEST_F(Fixture, DisabledReadsAndEmitsNothingAndDoesNotConsumeBudgets) {
    for (int i = 0; i < 1000; ++i) {
        observer.System(false, 1, true, reader, sink);
        observer.Rack(false, 0x1000, 1, false, reader, sink);
        observer.Control(false, 1, reader, sink);
        observer.Render(false, 1, 99, reader, sink);
        observer.State(false, 48, sink);
    }
    EXPECT_EQ(reads, 0);
    EXPECT_TRUE(records.empty());
    observer.State(true, 48, sink);
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(Value(records.front(), "state_size"), 48);
}

TEST_F(Fixture, SystemAndRackKeepOnlyDeclaredScalarOptions) {
    Put<u64>(0x1000, 0, 64);
    Put<u32>(0x1000, 24, 7);
    Put<u32>(0x1000, 28, 512);
    Put<u32>(0x1000, 32, 256);
    Put<u32>(0x1000, 36, 48000);
    Put<u64>(0x2000, 0, 128);
    for (size_t i = 0; i < 6; ++i) {
        Put<u32>(0x2000, 24 + 4 * i, static_cast<u32>(i + 10));
    }
    observer.System(true, 0x1000, true, reader, sink);
    observer.Rack(true, 0x4002, 0x2000, false, reader, sink);
    ASSERT_EQ(records.size(), 2);
    EXPECT_EQ(Value(records[0], "sample_rate"), 48000);
    EXPECT_EQ(Value(records[0], "grain_samples"), 256);
    EXPECT_EQ(Value(records[1], "rack_id"), 0x4002);
    EXPECT_EQ(Value(records[1], "max_voices"), 12);
    EXPECT_EQ(Value(records[1], "max_ports"), 15);
    EXPECT_EQ(read_bytes, 56); // size + selected fields; not names/reserved/pointers
}

TEST_F(Fixture, NullUnreadableAndShortOptionsAreExplicitWithoutPayloadReads) {
    observer.System(true, 0, false, reader, sink);
    observer.System(true, 7, false, reader, sink);
    Put<u64>(0x1000, 0, 8);
    observer.Rack(true, 0x1000, 0x1000, false, reader, sink);
    ASSERT_EQ(records.size(), 3);
    EXPECT_EQ(records[0].status, Status::Null);
    EXPECT_EQ(records[1].status, Status::Unreadable);
    EXPECT_EQ(records[2].status, Status::ShortAbi);
    EXPECT_EQ(read_bytes, 16);
}

TEST_F(Fixture, ControlUsesSignedRelativeLinksNotContiguousSizes) {
    Header(0x2000, 8, -32, 0xfedcba98);
    Header(0x1fe0, 12, 0, 6);
    Put<u32>(0x1fe0, 8, 1);
    observer.Control(true, 0x2000, reader, sink);
    ASSERT_EQ(records.size(), 2);
    EXPECT_EQ(Value(records[0], "id"), 0xfedcba98);
    EXPECT_EQ(static_cast<s64>(Value(records[0], "next")), -32);
    EXPECT_EQ(records[0].status, Status::UnknownId);
    EXPECT_EQ(Value(records[1], "event_id"), 1);
}

TEST_F(Fixture, SetupRecordsFormatButNeverReadsWaveformOrPointerPayloads) {
    Header(0x1000, 40, 64, 0x10000000);
    for (u32 i = 0; i < 7; ++i) {
        Put<u32>(0x1000, 8 + 4 * i, i + 0x18);
    }
    Header(0x1040, 32, 0, 0x10000001);
    Put<u64>(0x1040, 8, 0xdeadbeefdeadbeef);
    Put<u32>(0x1040, 16, 3);
    Put<u32>(0x1040, 20, 2);
    Put<u64>(0x1040, 24, 0xbaadf00dbaadf00d);
    observer.Control(true, 0x1000, reader, sink);
    ASSERT_EQ(records.size(), 2);
    EXPECT_EQ(Value(records[0], "waveform_type"), 0x18);
    EXPECT_EQ(Value(records[0], "sample_rate"), 0x1a);
    EXPECT_EQ(Value(records[1], "blocks"), 2);
    EXPECT_EQ(read_bytes, 52); // two headers + 28 setup scalars + 8 block scalars
    for (const auto& record : records) {
        for (u32 i = 0; i < record.count; ++i) {
            EXPECT_NE(record.fields[i].value, 0xdeadbeefdeadbeef);
            EXPECT_NE(record.fields[i].value, 0xbaadf00dbaadf00d);
        }
    }
}

TEST_F(Fixture, PatchReportsPortAndInputButNotDestinationHandle) {
    Header(0x1000, 24, 0, 5);
    Put<u32>(0x1000, 8, 3);
    Put<u32>(0x1000, 12, 7);
    Put<u64>(0x1000, 16, 0x123456789abcdef0);
    observer.Control(true, 0x1000, reader, sink);
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(Value(records[0], "port"), 3);
    EXPECT_EQ(Value(records[0], "dest_input"), 7);
    EXPECT_EQ(read_bytes, 16);
}

TEST_F(Fixture, UnknownPayloadIsNeverInspectedEvenWithLargeDeclaredSize) {
    Header(0x1000, 65535, 0, 0x4002abcd);
    observer.Control(true, 0x1000, reader, sink);
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records[0].status, Status::UnknownId);
    EXPECT_EQ(read_bytes, 8);
}

TEST_F(Fixture, ShortKnownPayloadStopsBeforePayloadRead) {
    Header(0x1000, 8, 16, 0x10000000);
    observer.Control(true, 0x1000, reader, sink);
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records[0].status, Status::ShortAbi);
    EXPECT_EQ(read_bytes, 8);
}

TEST_F(Fixture, UnreadablePayloadStopsWithoutFollowingNextLink) {
    Header(0x1000, 40, 16, 0x10000000);
    observer.Control(true, 0x1000, reader, sink);
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records[0].status, Status::Unreadable);
    EXPECT_EQ(reads, 2);
}

TEST_F(Fixture, CycleAndAddressOverflowTerminateWithoutInvalidRead) {
    Header(0x1000, 8, 8, 0xff);
    Header(0x1008, 8, -8, 0xff);
    observer.Control(true, 0x1000, reader, sink);
    ASSERT_EQ(records.size(), 2);
    EXPECT_EQ(records.back().status, Status::Cycle);
    EXPECT_EQ(reads, 2);
    records.clear();
    const u64 high = std::numeric_limits<u64>::max() - 7;
    Header(high, 8, 16, 0xff);
    observer.Control(true, high, reader, sink);
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records.back().status, Status::Overflow);
    Header(4, 8, -8, 0xff);
    observer.Control(true, 4, reader, sink);
    EXPECT_EQ(records.back().status, Status::Overflow);
}

TEST_F(Fixture, ControlAndRequestCapsBoundReadsAndMessages) {
    for (u32 i = 0; i < 12; ++i) {
        Header(0x1000 + 8 * i, 8, 8, 0xff);
    }
    for (u32 i = 0; i < 100; ++i) {
        observer.Control(true, 0x1000, reader, sink);
    }
    ASSERT_EQ(reads, Observer::RequestCaps[2] * Observer::MaxControlHeaders);
    ASSERT_EQ(records.size(), reads);
    EXPECT_EQ(records.back().status, Status::Limit);
}

TEST_F(Fixture, RenderBoundsDescriptorsAndDoesNotReadTheirBufferPointers) {
    for (u32 i = 0; i < 5; ++i) {
        Put<u64>(0x1000, i * 24, 0xdeadbeefdeadbeef);
        Put<u64>(0x1000, i * 24 + 8, 8192);
        Put<u32>(0x1000, i * 24 + 16, 0x18);
        Put<u32>(0x1000, i * 24 + 20, 8);
    }
    observer.Render(true, 0x1000, 1000, reader, sink);
    ASSERT_EQ(records.size(), 5);
    EXPECT_EQ(Value(records[0], "buffer_count"), 1000);
    EXPECT_EQ(records[0].status, Status::Limit);
    EXPECT_EQ(Value(records[1], "buffer_size"), 8192);
    EXPECT_EQ(Value(records[1], "channels"), 8);
    EXPECT_EQ(reads, Observer::MaxRenderBuffers);
    EXPECT_EQ(read_bytes, 64);
}

TEST_F(Fixture, RenderNullZeroAndOverflowAreSafe) {
    observer.Render(true, 0, 0, reader, sink);
    observer.Render(true, 0, 1, reader, sink);
    observer.Render(true, std::numeric_limits<u64>::max() - 4, 2, reader, sink);
    EXPECT_EQ(reads, 0);
    ASSERT_EQ(records.size(), 4);
    EXPECT_EQ(records[0].status, Status::Observed);
    EXPECT_EQ(records[1].status, Status::Null);
    EXPECT_EQ(records[3].status, Status::Overflow);
}

TEST_F(Fixture, ReaderAndSinkExceptionsNeverEscapeDiagnostics) {
    Reader throwing_reader = [](u64, std::span<u8>) -> bool { throw std::runtime_error("read"); };
    Sink throwing_sink = [](const Record&) { throw std::runtime_error("sink"); };
    EXPECT_NO_THROW(observer.Control(true, 1, throwing_reader, sink));
    EXPECT_NO_THROW(observer.System(true, 0, false, reader, throwing_sink));
    EXPECT_NO_THROW(observer.State(true, 48, throwing_sink));
}

TEST(Ngs2DiagnosticBudget, ConcurrentCallersCannotExceedPerKindCap) {
    Observer observer;
    std::atomic<u32> emitted{};
    const Sink sink = [&](const Record&) { ++emitted; };
    std::vector<std::jthread> threads;
    for (int i = 0; i < 12; ++i) {
        threads.emplace_back([&] {
            for (int call = 0; call < 100; ++call) {
                observer.State(true, 48, sink);
            }
        });
    }
    threads.clear();
    EXPECT_EQ(emitted, Observer::RequestCaps[4]);
}

TEST(Ngs2DiagnosticRead, CheckedSelfReadRequiresReservationAndExactAccessibleBytes) {
    const std::array<u8, 8> input{1, 2, 3, 4, 5, 6, 7, 8};
    std::array<u8, 8> output{};
    const auto address = reinterpret_cast<u64>(input.data());
    const std::array ranges{AddressRange{address, input.size()}};
    EXPECT_FALSE(ReadMetadata(address, output, {}));
    EXPECT_FALSE(ReadMetadata(address + 1, output, ranges));
    const std::array invalid_ranges{AddressRange{1, 100}};
    EXPECT_FALSE(ReadMetadata(1, output, invalid_ranges));
#ifdef _WIN32
    ASSERT_TRUE(ReadMetadata(address, output, ranges));
    EXPECT_EQ(output, input);
#else
    EXPECT_FALSE(ReadMetadata(address, output, ranges));
#endif
}

#ifdef _WIN32
TEST(Ngs2DiagnosticRead, InaccessibleCrossPageAndGuardMemoryRemainUntouched) {
    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const auto page_size = static_cast<size_t>(system_info.dwPageSize);
    auto* memory = static_cast<u8*>(
        VirtualAlloc(nullptr, page_size * 2, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    ASSERT_NE(memory, nullptr);
    struct Free {
        void* memory;
        ~Free() {
            VirtualFree(memory, 0, MEM_RELEASE);
        }
    } free{memory};
    const auto address = reinterpret_cast<u64>(memory);
    const std::array ranges{AddressRange{address, page_size * 2}};
    std::array<u8, 8> output{};
    output.fill(0xa5);
    DWORD previous{};
    ASSERT_TRUE(VirtualProtect(memory + page_size, page_size, PAGE_NOACCESS, &previous));
    EXPECT_FALSE(ReadMetadata(address + page_size - 4, output, ranges));
    EXPECT_TRUE(std::ranges::all_of(output, [](u8 byte) { return byte == 0xa5; }));
    ASSERT_TRUE(VirtualProtect(memory, page_size, PAGE_READWRITE | PAGE_GUARD, &previous));
    EXPECT_FALSE(ReadMetadata(address, output, ranges));
    MEMORY_BASIC_INFORMATION info{};
    ASSERT_NE(VirtualQuery(memory, &info, sizeof(info)), 0);
    EXPECT_NE(info.Protect & PAGE_GUARD, 0);
}
#endif

TEST_F(Fixture, AllKindsHaveIndependentSaturatingMessageBudgets) {
    for (int i = 0; i < 100; ++i) {
        observer.System(true, 0, false, reader, sink);
        observer.Rack(true, 0x1000, 0, false, reader, sink);
        observer.Render(true, 0, 0, reader, sink);
        observer.State(true, 48, sink);
    }
    for (const auto kind : {Kind::System, Kind::Rack, Kind::Render, Kind::State}) {
        EXPECT_EQ(std::ranges::count(records, kind, &Record::kind),
                  Observer::RequestCaps[static_cast<size_t>(kind)]);
    }
    EXPECT_EQ(reads, 0);
}

std::string ReadSource(const char* path) {
    std::ifstream file{path};
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

TEST(Ngs2DiagnosticWiring, PublicEntryPointsUseDedicatedDefaultOffObserver) {
    const auto source = ReadSource(SHADPS4_NGS2_SOURCE_PATH);
    ASSERT_FALSE(source.empty());
    EXPECT_NE(source.find("Class::Lib_Ngs2_Diagnostics"), std::string::npos);
    EXPECT_NE(source.find("should_log(spdlog::level::debug)"), std::string::npos);
    for (const auto name :
         {"TraceNgs2System(option, false)", "TraceNgs2System(option, true)",
          "TraceNgs2Rack(rackId, option, false)", "TraceNgs2Rack(rackId, option, true)",
          "TraceNgs2Control(paramList)", "TraceNgs2Render(aBufferInfo, numBufferInfo)",
          "TraceNgs2State(stateSize)"}) {
        EXPECT_NE(source.find(name), std::string::npos) << name;
    }
    const auto classes = ReadSource(SHADPS4_LOG_CLASSES_PATH);
    const auto logging = ReadSource(SHADPS4_LOG_SOURCE_PATH);
    EXPECT_NE(classes.find("Lib.Ngs2.Diagnostics"), std::string::npos);
    EXPECT_NE(logging.find("{Class::Lib_Ngs2_Diagnostics, nullptr}"), std::string::npos);
}

} // namespace
} // namespace Libraries::Ngs2::Diagnostic
