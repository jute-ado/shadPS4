// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <utility>

namespace VideoCore {

enum class DmaAttemptResult {
    DiscoverWithoutPublication,
    RetryWithoutPublication,
    PublishExactlyOnce,
    AbortWithoutPublication,
    Complete,
};

enum class DmaExecutionResult {
    Published,
    Aborted,
};

class DmaFaultEpoch {
public:
    enum class Status {
        Clean,
        Faulted,
        Invalid,
        Overflow,
    };

    constexpr DmaFaultEpoch() = default;

    static constexpr DmaFaultEpoch Clean() {
        return DmaFaultEpoch{Status::Clean, 0};
    }

    static constexpr DmaFaultEpoch Faults(std::initializer_list<std::uint64_t> addresses) {
        return DmaFaultEpoch{Status::Faulted, addresses.size()};
    }

    static constexpr DmaFaultEpoch FaultCount(std::size_t count) {
        return DmaFaultEpoch{Status::Faulted, count};
    }

    static constexpr DmaFaultEpoch Invalid() {
        return DmaFaultEpoch{Status::Invalid, 0};
    }

    static constexpr DmaFaultEpoch Overflow() {
        return DmaFaultEpoch{Status::Overflow, 0};
    }

    constexpr Status GetStatus() const {
        return status;
    }

    constexpr std::size_t FaultCount() const {
        return fault_count;
    }

private:
    constexpr DmaFaultEpoch(Status status, std::size_t fault_count)
        : status{status}, fault_count{fault_count} {}

    Status status{Status::Clean};
    std::size_t fault_count{};

    friend class DmaPublicationGate;
};

class DmaPublicationGate {
public:
    struct Config {
        std::uint32_t maximum_attempts;
        std::uint64_t binding_generation;
    };

    explicit constexpr DmaPublicationGate(Config config) : config{config} {}

    constexpr DmaAttemptResult BeginAttempt(std::uint64_t binding_generation) {
        if (state == State::Complete) {
            return DmaAttemptResult::Complete;
        }
        if (state != State::Ready || binding_generation != config.binding_generation ||
            attempt_count >= config.maximum_attempts) {
            state = State::Aborted;
            return DmaAttemptResult::AbortWithoutPublication;
        }

        ++attempt_count;
        state = State::AttemptActive;
        return DmaAttemptResult::DiscoverWithoutPublication;
    }

    constexpr DmaAttemptResult CompleteAttempt(DmaFaultEpoch epoch) {
        if (state != State::AttemptActive) {
            state = State::Aborted;
            return DmaAttemptResult::AbortWithoutPublication;
        }

        if (epoch.status == DmaFaultEpoch::Status::Clean) {
            ++publication_count;
            state = State::Complete;
            return DmaAttemptResult::PublishExactlyOnce;
        }

        if (epoch.status != DmaFaultEpoch::Status::Faulted || epoch.fault_count == 0 ||
            attempt_count >= config.maximum_attempts) {
            state = State::Aborted;
            return DmaAttemptResult::AbortWithoutPublication;
        }

        state = State::Ready;
        return DmaAttemptResult::RetryWithoutPublication;
    }

    constexpr std::uint32_t AttemptCount() const {
        return attempt_count;
    }

    constexpr std::uint32_t PublicationCount() const {
        return publication_count;
    }

private:
    enum class State {
        Ready,
        AttemptActive,
        Complete,
        Aborted,
    };

    Config config;
    State state{State::Ready};
    std::uint32_t attempt_count{};
    std::uint32_t publication_count{};
};

template <typename DiscoverAttempt, typename Publish>
DmaExecutionResult ExecuteDmaPublicationGate(DmaPublicationGate& gate,
                                             std::uint64_t binding_generation,
                                             DiscoverAttempt&& discover_attempt,
                                             Publish&& publish) {
    for (;;) {
        if (gate.BeginAttempt(binding_generation) != DmaAttemptResult::DiscoverWithoutPublication) {
            return DmaExecutionResult::Aborted;
        }

        const auto epoch =
            std::invoke(std::forward<DiscoverAttempt>(discover_attempt), gate.AttemptCount());
        switch (gate.CompleteAttempt(epoch)) {
        case DmaAttemptResult::RetryWithoutPublication:
            continue;
        case DmaAttemptResult::PublishExactlyOnce:
            return std::invoke(std::forward<Publish>(publish)) ? DmaExecutionResult::Published
                                                               : DmaExecutionResult::Aborted;
        default:
            return DmaExecutionResult::Aborted;
        }
    }
}

struct DmaWorkTraits {
    bool vertex_dma_reads{};
    bool vertex_dynamic_dma_reads{};
    bool fragment_dma_reads{};
    bool fragment_dynamic_dma_reads{};
    bool unsupported_stage_dma_reads{};
    bool guest_or_gds_writes{};
    bool atomics{};
    bool storage_image_writes{};
};

enum class DmaDiscoveryMode {
    None,
    RasterizerDiscard,
    SuppressAttachmentWrites,
};

struct DmaDiscoveryBindingKey {
    std::uint64_t pipeline{};
    std::uint64_t bindings{};

    constexpr bool operator==(const DmaDiscoveryBindingKey&) const = default;
};

class DmaDiscoveryBindingCache {
public:
    static constexpr std::size_t Capacity = 256;

    [[nodiscard]] constexpr bool NeedsDiscovery(DmaDiscoveryBindingKey key) const {
        for (std::size_t index = 0; index < size; ++index) {
            if (entries[index] == key) {
                return false;
            }
        }
        return true;
    }

    constexpr void MarkClean(DmaDiscoveryBindingKey key) {
        if (!NeedsDiscovery(key) || size == Capacity) {
            return;
        }
        entries[size++] = key;
    }

    constexpr void Clear() {
        size = 0;
    }

private:
    std::array<DmaDiscoveryBindingKey, Capacity> entries{};
    std::size_t size{};
};

constexpr DmaDiscoveryMode ClassifyDmaDiscovery(const DmaWorkTraits& traits,
                                                bool dynamic_color_write_mask_supported) {
    if (traits.unsupported_stage_dma_reads || traits.guest_or_gds_writes || traits.atomics ||
        traits.storage_image_writes) {
        return DmaDiscoveryMode::None;
    }
    if (traits.fragment_dma_reads) {
        return traits.fragment_dynamic_dma_reads && dynamic_color_write_mask_supported
                   ? DmaDiscoveryMode::SuppressAttachmentWrites
                   : DmaDiscoveryMode::None;
    }
    if (traits.vertex_dma_reads && traits.vertex_dynamic_dma_reads) {
        return DmaDiscoveryMode::RasterizerDiscard;
    }
    return DmaDiscoveryMode::None;
}

constexpr bool IsDmaDiscoveryEligible(const DmaWorkTraits& traits) {
    return ClassifyDmaDiscovery(traits, true) != DmaDiscoveryMode::None;
}

} // namespace VideoCore
