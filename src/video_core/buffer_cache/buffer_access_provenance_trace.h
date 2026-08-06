// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bit>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "video_core/buffer_cache/buffer_access_interval_provenance.h"

namespace VideoCore {

struct BufferAccessProvenanceTraceRequest {
    std::filesystem::path path;
    std::uint64_t record_limit{};
    std::uint64_t observation_limit{};

    static std::optional<BufferAccessProvenanceTraceRequest> Parse(
        std::string_view path_value, std::string_view record_limit_value,
        std::string_view observation_limit_value) {
        constexpr std::uint64_t MaxRecordLimit = 1'000'000;
        constexpr std::uint64_t MaxObservationLimit = 50'000'000;
        std::uint64_t record_limit{};
        std::uint64_t observation_limit{};
        const auto record_result =
            std::from_chars(record_limit_value.data(),
                            record_limit_value.data() + record_limit_value.size(), record_limit);
        const auto observation_result = std::from_chars(
            observation_limit_value.data(),
            observation_limit_value.data() + observation_limit_value.size(), observation_limit);
        if (path_value.empty() || record_result.ec != std::errc{} ||
            record_result.ptr != record_limit_value.data() + record_limit_value.size() ||
            observation_result.ec != std::errc{} ||
            observation_result.ptr !=
                observation_limit_value.data() + observation_limit_value.size() ||
            record_limit == 0 || record_limit > MaxRecordLimit ||
            observation_limit < record_limit || observation_limit > MaxObservationLimit) {
            return std::nullopt;
        }
        return BufferAccessProvenanceTraceRequest{
            .path = std::filesystem::path{path_value},
            .record_limit = record_limit,
            .observation_limit = observation_limit,
        };
    }
};

template <typename Key>
class BasicBufferAccessProvenanceTrace {
public:
    explicit BasicBufferAccessProvenanceTrace(const BufferAccessProvenanceTraceRequest& request)
        : record_limit{request.record_limit}, observation_limit{request.observation_limit} {
        output.open(request.path, std::ios::out | std::ios::app | std::ios::binary);
        if (!output) {
            throw std::runtime_error{"Cannot create buffer access provenance trace"};
        }
        output << nlohmann::json{
                      {"kind", "header"},
                      {"protocolVersion", 1},
                      {"source", "buffer_access_interval_provenance"},
                      {"recordLimit", record_limit},
                      {"observationLimit", observation_limit},
                  }
                      .dump()
               << '\n';
        output.flush();
    }

    BasicBufferAccessProvenanceTrace(const BasicBufferAccessProvenanceTrace&) = delete;
    BasicBufferAccessProvenanceTrace& operator=(const BasicBufferAccessProvenanceTrace&) = delete;

    void BeginCommand(std::uint64_t command_id, std::uint64_t tick) {
        if (complete) {
            return;
        }
        tracker.BeginCommand(command_id, tick);
        current_command_id = command_id;
        current_tick = tick;
        command_active = true;
    }

    bool Observe(Key key, std::uint64_t offset, std::uint64_t size, BufferAccessRole roles,
                 std::uint64_t access, std::uint64_t stages, bool writes,
                 BufferBarrierObservation barrier = {}) {
        if (complete || !command_active || observation_count >= observation_limit) {
            complete = observation_count >= observation_limit;
            return false;
        }
        if (!tracker.Observe(key, offset, size, roles, access, stages, writes, barrier)) {
            return false;
        }
        ++observation_count;
        return true;
    }

    void CommitCommand() {
        if (!command_active) {
            return;
        }
        auto transitions = tracker.CommitCommand();
        command_active = false;
        if (complete || transitions.empty()) {
            return;
        }

        struct Group {
            Key key{};
            BufferAccessRole roles{};
        };
        std::vector<Group> groups;
        for (const auto& transition : transitions) {
            auto group = std::ranges::find(groups, transition.key, &Group::key);
            if (group == groups.end()) {
                group = groups.emplace(groups.end(), Group{.key = transition.key});
            }
            group->roles |= transition.current_roles;
        }

        for (const auto& transition : transitions) {
            const auto group = std::ranges::find(groups, transition.key, &Group::key);
            const bool multi_role = std::popcount(static_cast<std::uint32_t>(group->roles)) > 1;
            const bool gpu_write_to_geometry =
                transition.prior.has_value() && transition.prior->writes &&
                (HasBufferAccessRole(transition.current_roles, BufferAccessRole::VertexRead) ||
                 HasBufferAccessRole(transition.current_roles, BufferAccessRole::IndexRead));
            if (!multi_role && !gpu_write_to_geometry) {
                continue;
            }
            WriteTransition(transition, multi_role, gpu_write_to_geometry);
            if (record_count >= record_limit) {
                complete = true;
                break;
            }
        }
    }

    void CancelCommand() noexcept {
        if (command_active) {
            tracker.CancelCommand();
            command_active = false;
        }
    }

    [[nodiscard]] bool IsComplete() const noexcept {
        return complete;
    }

private:
    using Tracker = BasicBufferAccessIntervalProvenance<Key>;
    using Transition = typename Tracker::Transition;
    using Producer = typename Tracker::Producer;
    using Observation = typename Tracker::Observation;

    static nlohmann::json RoleNames(BufferAccessRole roles) {
        nlohmann::json names = nlohmann::json::array();
        constexpr std::pair<BufferAccessRole, std::string_view> KnownRoles[] = {
            {BufferAccessRole::ShaderRead, "shader_read"},
            {BufferAccessRole::ShaderReadWrite, "shader_read_write"},
            {BufferAccessRole::VertexRead, "vertex_read"},
            {BufferAccessRole::IndexRead, "index_read"},
            {BufferAccessRole::IndirectRead, "indirect_read"},
            {BufferAccessRole::TransferRead, "transfer_read"},
            {BufferAccessRole::TransferWrite, "transfer_write"},
            {BufferAccessRole::UploadWrite, "upload_write"},
            {BufferAccessRole::FillWrite, "fill_write"},
            {BufferAccessRole::JoinRead, "join_read"},
            {BufferAccessRole::JoinWrite, "join_write"},
        };
        for (const auto& [role, name] : KnownRoles) {
            if (HasBufferAccessRole(roles, role)) {
                names.push_back(name);
            }
        }
        return names;
    }

    static nlohmann::json ProducerJson(const Producer& producer) {
        return nlohmann::json{
            {"commandId", producer.command_id},   {"tick", producer.tick},
            {"roles", RoleNames(producer.roles)}, {"access", producer.access},
            {"stages", producer.stages},          {"writes", producer.writes},
        };
    }

    static nlohmann::json ObservationJson(const Observation& observation) {
        nlohmann::json barrier;
        if (observation.barrier.emitted) {
            barrier = nlohmann::json{
                {"offset", observation.barrier.offset},
                {"size", observation.barrier.size},
                {"sourceAccess", observation.barrier.source_access},
                {"sourceStages", observation.barrier.source_stages},
                {"destinationAccess", observation.barrier.destination_access},
                {"destinationStages", observation.barrier.destination_stages},
            };
        }
        return nlohmann::json{
            {"offset", observation.offset},           {"size", observation.size},
            {"roles", RoleNames(observation.roles)},  {"access", observation.access},
            {"stages", observation.stages},           {"writes", observation.writes},
            {"resultingBarrier", std::move(barrier)},
        };
    }

    std::uint64_t BufferId(const Key& key) {
        const auto entry =
            std::ranges::find(buffer_ids, key, &std::pair<Key, std::uint64_t>::first);
        if (entry != buffer_ids.end()) {
            return entry->second;
        }
        const auto id = static_cast<std::uint64_t>(buffer_ids.size() + 1);
        buffer_ids.emplace_back(key, id);
        return id;
    }

    void WriteTransition(const Transition& transition, bool multi_role,
                         bool gpu_write_to_geometry) {
        nlohmann::json observations = nlohmann::json::array();
        for (const auto& observation : transition.observations) {
            observations.push_back(ObservationJson(observation));
        }
        nlohmann::json prior;
        if (transition.prior.has_value()) {
            prior = ProducerJson(*transition.prior);
        }
        output << nlohmann::json{
                      {"kind", "transition"},
                      {"protocolVersion", 1},
                      {"commandId", current_command_id},
                      {"tick", current_tick},
                      {"bufferId", BufferId(transition.key)},
                      {"offset", transition.offset},
                      {"size", transition.size},
                      {"prior", std::move(prior)},
                      {"currentRoles", RoleNames(transition.current_roles)},
                      {"currentAccess", transition.current_access},
                      {"currentStages", transition.current_stages},
                      {"currentWrites", transition.current_writes},
                      {"observations", std::move(observations)},
                      {"interesting",
                       {{"multiRole", multi_role},
                        {"gpuWriteToVertexOrIndex", gpu_write_to_geometry}}},
                  }
                      .dump()
               << '\n';
        output.flush();
        ++record_count;
    }

    Tracker tracker;
    std::ofstream output;
    std::vector<std::pair<Key, std::uint64_t>> buffer_ids;
    std::uint64_t record_limit{};
    std::uint64_t observation_limit{};
    std::uint64_t record_count{};
    std::uint64_t observation_count{};
    std::uint64_t current_command_id{};
    std::uint64_t current_tick{};
    bool command_active{};
    bool complete{};
};

class Buffer;

class BufferAccessProvenanceTrace final : public BasicBufferAccessProvenanceTrace<Buffer*> {
public:
    using BasicBufferAccessProvenanceTrace::BasicBufferAccessProvenanceTrace;

    static std::unique_ptr<BufferAccessProvenanceTrace> CreateFromEnvironment() {
        const auto* path = std::getenv("SHADPS4_BUFFER_ACCESS_PROVENANCE_PATH");
        const auto* record_limit = std::getenv("SHADPS4_BUFFER_ACCESS_PROVENANCE_RECORD_LIMIT");
        const auto* observation_limit =
            std::getenv("SHADPS4_BUFFER_ACCESS_PROVENANCE_OBSERVATION_LIMIT");
        if (path == nullptr && record_limit == nullptr && observation_limit == nullptr) {
            return {};
        }
        if (path == nullptr || record_limit == nullptr || observation_limit == nullptr) {
            std::cerr << "[Diagnostic] Invalid buffer access provenance trace request.\n";
            return {};
        }
        const auto request =
            BufferAccessProvenanceTraceRequest::Parse(path, record_limit, observation_limit);
        if (!request.has_value()) {
            std::cerr << "[Diagnostic] Invalid buffer access provenance trace request.\n";
            return {};
        }
        try {
            return std::make_unique<BufferAccessProvenanceTrace>(*request);
        } catch (const std::exception&) {
            std::cerr << "[Diagnostic] Cannot create buffer access provenance trace.\n";
            return {};
        }
    }
};

} // namespace VideoCore
