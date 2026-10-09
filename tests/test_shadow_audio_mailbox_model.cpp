// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "core/libraries/audio/sdl_audio_queue_policy.h"

namespace {

using Libraries::AudioOut::CalculateAudioQueueThreshold;
using Tick = std::uint64_t;

constexpr std::size_t kReceiptCapacity = 8;

struct FakeClock {
    [[nodiscard]] Tick Now() const {
        return now_us;
    }

    void Advance(Tick delta_us) {
        now_us += delta_us;
    }

    Tick now_us = 0;
};

template <typename T, std::size_t Capacity>
class BoundedReceiptLog {
public:
    void Push(T value) {
        if (entries_.size() == Capacity) {
            ++dropped_entries_;
            return;
        }
        entries_.push_back(std::move(value));
    }

    [[nodiscard]] const std::vector<T>& Entries() const {
        return entries_;
    }

    [[nodiscard]] std::size_t DroppedEntries() const {
        return dropped_entries_;
    }

private:
    std::vector<T> entries_;
    std::size_t dropped_entries_ = 0;
};

struct AudioBlock {
    std::uint32_t port_id;
    std::uint64_t sequence;
    Tick submitted_at_us;
    std::size_t bytes;
};

struct SubmissionReceipt {
    std::uint32_t port_id;
    std::uint64_t sequence;
    Tick submitted_at_us;
    std::size_t bytes;
    std::size_t pending_depth;
};

struct ConsumptionReceipt {
    std::uint32_t port_id;
    std::uint64_t sequence;
    Tick submitted_at_us;
    Tick consumed_at_us;
    std::size_t bytes;
    std::size_t pending_depth;
    std::size_t queued_bytes;
    std::size_t queue_clear_count;
};

class FakeAudioBackend {
public:
    explicit FakeAudioBackend(std::size_t queue_threshold_bytes)
        : queue_threshold_bytes_(queue_threshold_bytes) {}

    void Output(const AudioBlock& block, Tick consumed_at_us, std::size_t pending_depth) {
        if (queued_bytes_ >= queue_threshold_bytes_) {
            queued_bytes_ = 0;
            ++queue_clear_count_;
        }
        queued_bytes_ += block.bytes;
        receipts_.Push({
            .port_id = block.port_id,
            .sequence = block.sequence,
            .submitted_at_us = block.submitted_at_us,
            .consumed_at_us = consumed_at_us,
            .bytes = block.bytes,
            .pending_depth = pending_depth,
            .queued_bytes = queued_bytes_,
            .queue_clear_count = queue_clear_count_,
        });
    }

    void SetQueuedBytesForTest(std::size_t queued_bytes) {
        queued_bytes_ = queued_bytes;
    }

    [[nodiscard]] std::size_t QueuedBytes() const {
        return queued_bytes_;
    }

    [[nodiscard]] std::size_t QueueClearCount() const {
        return queue_clear_count_;
    }

    [[nodiscard]] const auto& Receipts() const {
        return receipts_.Entries();
    }

    [[nodiscard]] std::size_t DroppedReceipts() const {
        return receipts_.DroppedEntries();
    }

private:
    std::size_t queue_threshold_bytes_;
    std::size_t queued_bytes_ = 0;
    std::size_t queue_clear_count_ = 0;
    BoundedReceiptLog<ConsumptionReceipt, kReceiptCapacity> receipts_;
};

enum class SubmitResult {
    Accepted,
    Pending,
    Closed,
};

enum class WaitResult {
    Ready,
    Blocked,
    Closed,
};

class AudioMailbox {
public:
    explicit AudioMailbox(std::uint32_t port_id) : port_id_(port_id) {}

    SubmitResult TrySubmit(std::size_t bytes, const FakeClock& clock) {
        switch (WaitForCapacity()) {
        case WaitResult::Closed:
            return SubmitResult::Closed;
        case WaitResult::Blocked:
            return SubmitResult::Pending;
        case WaitResult::Ready:
            break;
        }
        Accept(bytes, clock);
        return SubmitResult::Accepted;
    }

    bool Consume(const FakeClock& clock, FakeAudioBackend& backend) {
        if (closing_ || !pending_.has_value()) {
            if (!closing_) {
                ++empty_consume_polls_;
            }
            return false;
        }

        const AudioBlock block = *pending_;
        pending_.reset();
        backend.Output(block, clock.Now(), PendingDepth());
        return true;
    }

    void Close() {
        closing_ = true;
        pending_.reset();
    }

    [[nodiscard]] bool CanSubmit() const {
        return WaitForCapacity() == WaitResult::Ready;
    }

    [[nodiscard]] WaitResult WaitForCapacity() const {
        if (closing_) {
            return WaitResult::Closed;
        }
        return pending_.has_value() ? WaitResult::Blocked : WaitResult::Ready;
    }

    [[nodiscard]] std::size_t PendingDepth() const {
        return pending_.has_value() ? 1 : 0;
    }

    [[nodiscard]] std::size_t EmptyConsumePolls() const {
        return empty_consume_polls_;
    }

    [[nodiscard]] const auto& Submissions() const {
        return submissions_.Entries();
    }

private:
    friend class AudioMailboxSet;

    void Accept(std::size_t bytes, const FakeClock& clock) {
        const AudioBlock block{
            .port_id = port_id_,
            .sequence = next_sequence_++,
            .submitted_at_us = clock.Now(),
            .bytes = bytes,
        };
        pending_ = block;
        submissions_.Push({
            .port_id = block.port_id,
            .sequence = block.sequence,
            .submitted_at_us = block.submitted_at_us,
            .bytes = block.bytes,
            .pending_depth = PendingDepth(),
        });
    }

    std::uint32_t port_id_;
    std::uint64_t next_sequence_ = 1;
    std::optional<AudioBlock> pending_;
    bool closing_ = false;
    std::size_t empty_consume_polls_ = 0;
    BoundedReceiptLog<SubmissionReceipt, kReceiptCapacity> submissions_;
};

struct BatchItem {
    std::uint32_t port_id;
    std::size_t bytes;
};

// This is deliberately a lock-acquisition model, not a replacement for the
// production mutexes. It lets the test exercise the interleaving that would
// otherwise require two threads to block forever. `sceAudioOutOutputs`
// must acquire its port mutexes in a stable slot order, even when callers
// name the same ports in opposite guest parameter orders.
class BatchLockModel {
public:
    explicit BatchLockModel(std::size_t port_count) : owners_(port_count, kUnlocked) {}

    [[nodiscard]] bool TryAcquire(std::uint32_t owner, std::uint32_t port_id) {
        if (port_id >= owners_.size() ||
            (owners_[port_id] != kUnlocked && owners_[port_id] != owner)) {
            return false;
        }
        owners_[port_id] = owner;
        return true;
    }

    void Release(std::uint32_t owner) {
        for (auto& port_owner : owners_) {
            if (port_owner == owner) {
                port_owner = kUnlocked;
            }
        }
    }

private:
    static constexpr std::uint32_t kUnlocked = 0;
    std::vector<std::uint32_t> owners_;
};

[[nodiscard]] std::vector<std::uint32_t> CanonicalBatchLockOrder(
    std::span<const BatchItem> items) {
    std::vector<std::uint32_t> order;
    order.reserve(items.size());
    for (const auto& item : items) {
        order.push_back(item.port_id);
    }
    std::ranges::sort(order);
    return order;
}

class AudioMailboxSet {
public:
    explicit AudioMailboxSet(std::size_t port_count) {
        for (std::size_t port_id = 0; port_id < port_count; ++port_id) {
            ports_.emplace_back(static_cast<std::uint32_t>(port_id));
        }
    }

    [[nodiscard]] AudioMailbox& Port(std::uint32_t port_id) {
        return ports_.at(port_id);
    }

    bool TrySubmitBatch(std::span<const BatchItem> items, const FakeClock& clock) {
        if (items.empty()) {
            return false;
        }

        for (std::size_t index = 0; index < items.size(); ++index) {
            if (items[index].port_id >= ports_.size() || !Port(items[index].port_id).CanSubmit()) {
                return false;
            }
            for (std::size_t previous = 0; previous < index; ++previous) {
                if (items[previous].port_id == items[index].port_id) {
                    return false;
                }
            }
        }

        for (const auto& item : items) {
            Port(item.port_id).Accept(item.bytes, clock);
        }
        return true;
    }

private:
    std::vector<AudioMailbox> ports_;
};

enum class WorkerFailureStage {
    CaptureSetup,
    Processing,
    Backend,
};

struct WorkerFailureReceipt {
    WorkerFailureStage stage;
};

class InjectedWorkerFailure final : public std::exception {
public:
    explicit InjectedWorkerFailure(WorkerFailureStage stage) : stage_(stage) {}

    [[nodiscard]] WorkerFailureStage Stage() const {
        return stage_;
    }

    [[nodiscard]] const char* what() const noexcept override {
        return "injected audio worker failure";
    }

private:
    WorkerFailureStage stage_;
};

// Device-free lifecycle model for the AudioOut worker boundary. The worker
// keeps a failed bit separate from the explicit close bit so a producer can
// be released and observe why its mailbox stopped making progress. `joinable_`
// intentionally remains true after the worker exits: Kernel::Thread::Stop()
// still has to join the completed native thread exactly once.
class AudioOutputFailureModel {
public:
    explicit AudioOutputFailureModel(WorkerFailureStage injected_stage)
        : injected_stage_(injected_stage) {}

    SubmitResult Submit() {
        if (failed_ || closing_) {
            return SubmitResult::Closed;
        }
        if (pending_) {
            producer_waiting_ = true;
            return SubmitResult::Pending;
        }
        pending_ = true;
        return SubmitResult::Accepted;
    }

    void Run() {
        if (!joinable_ || failed_ || closing_) {
            return;
        }

        try {
            RunStage(WorkerFailureStage::CaptureSetup);
            pending_ = false;
            RunStage(WorkerFailureStage::Processing);
            RunStage(WorkerFailureStage::Backend);
        } catch (const InjectedWorkerFailure& failure) {
            RecordFailure(failure.Stage());
        }
    }

    void Close() {
        closing_ = true;
        pending_ = false;
        WakeProducer();
    }

    void Stop() {
        stop_requested_ = true;
        if (joinable_) {
            joinable_ = false;
            ++join_count_;
        }
    }

    [[nodiscard]] bool Failed() const {
        return failed_;
    }

    [[nodiscard]] bool Closing() const {
        return closing_;
    }

    [[nodiscard]] bool ProducerWoke() const {
        return producer_woke_;
    }

    [[nodiscard]] bool Pending() const {
        return pending_;
    }

    [[nodiscard]] bool Joinable() const {
        return joinable_;
    }

    [[nodiscard]] bool StopRequested() const {
        return stop_requested_;
    }

    [[nodiscard]] std::size_t JoinCount() const {
        return join_count_;
    }

    [[nodiscard]] const std::optional<WorkerFailureReceipt>& FailureReceipt() const {
        return failure_receipt_;
    }

    [[nodiscard]] std::size_t FailureReportCount() const {
        return failure_report_count_;
    }

private:
    void RunStage(WorkerFailureStage stage) {
        if (stage == injected_stage_) {
            throw InjectedWorkerFailure{stage};
        }
    }

    void RecordFailure(WorkerFailureStage stage) {
        if (failure_receipt_) {
            return;
        }
        failure_receipt_ = WorkerFailureReceipt{.stage = stage};
        ++failure_report_count_;
        failed_ = true;
        pending_ = false;
        WakeProducer();
    }

    void WakeProducer() {
        if (producer_waiting_) {
            producer_waiting_ = false;
            producer_woke_ = true;
        }
    }

    WorkerFailureStage injected_stage_;
    std::optional<WorkerFailureReceipt> failure_receipt_;
    bool pending_ = false;
    bool producer_waiting_ = false;
    bool producer_woke_ = false;
    bool failed_ = false;
    bool closing_ = false;
    bool joinable_ = true;
    bool stop_requested_ = false;
    std::size_t failure_report_count_ = 0;
    std::size_t join_count_ = 0;
};

TEST(ShadowAudioWorkerFailureModel, InjectedStageFailsOnceAndWakesProducer) {
    constexpr std::array stages{
        WorkerFailureStage::CaptureSetup,
        WorkerFailureStage::Processing,
        WorkerFailureStage::Backend,
    };

    for (const auto stage : stages) {
        AudioOutputFailureModel worker{stage};

        ASSERT_EQ(worker.Submit(), SubmitResult::Accepted);
        ASSERT_EQ(worker.Submit(), SubmitResult::Pending);
        EXPECT_FALSE(worker.ProducerWoke());

        worker.Run();

        ASSERT_TRUE(worker.Failed());
        EXPECT_FALSE(worker.Closing());
        EXPECT_TRUE(worker.ProducerWoke());
        EXPECT_FALSE(worker.Pending());
        ASSERT_TRUE(worker.FailureReceipt().has_value());
        EXPECT_EQ(worker.FailureReceipt()->stage, stage);
        EXPECT_EQ(worker.FailureReportCount(), 1);

        // A completed worker may still own a joinable native thread. A
        // repeated run/report path must not create another receipt.
        worker.Run();
        EXPECT_EQ(worker.FailureReportCount(), 1);
        EXPECT_EQ(worker.FailureReceipt()->stage, stage);
    }
}

TEST(ShadowAudioWorkerFailureModel, FailureAndClosingRemainDistinctAndStopIsIdempotent) {
    AudioOutputFailureModel worker{WorkerFailureStage::Processing};

    ASSERT_EQ(worker.Submit(), SubmitResult::Accepted);
    ASSERT_EQ(worker.Submit(), SubmitResult::Pending);
    worker.Run();

    ASSERT_TRUE(worker.Failed());
    EXPECT_FALSE(worker.Closing());
    EXPECT_EQ(worker.Submit(), SubmitResult::Closed);

    worker.Close();
    EXPECT_TRUE(worker.Failed());
    EXPECT_TRUE(worker.Closing());
    EXPECT_FALSE(worker.Pending());

    EXPECT_TRUE(worker.Joinable());
    worker.Stop();
    worker.Stop();
    EXPECT_TRUE(worker.StopRequested());
    EXPECT_FALSE(worker.Joinable());
    EXPECT_EQ(worker.JoinCount(), 1);
    EXPECT_EQ(worker.FailureReportCount(), 1);
    ASSERT_TRUE(worker.FailureReceipt().has_value());
    EXPECT_EQ(worker.FailureReceipt()->stage, WorkerFailureStage::Processing);
}

TEST(ShadowAudioMailboxModel, AcceptedBlocksAreConsumedExactlyOnceWithPairedTiming) {
    FakeClock clock{.now_us = 100};
    AudioMailbox mailbox{0};
    FakeAudioBackend backend{1024};

    EXPECT_EQ(mailbox.TrySubmit(256, clock), SubmitResult::Accepted);
    EXPECT_EQ(mailbox.PendingDepth(), 1);
    EXPECT_EQ(mailbox.TrySubmit(128, clock), SubmitResult::Pending);
    EXPECT_TRUE(backend.Receipts().empty());

    clock.Advance(50);
    EXPECT_TRUE(mailbox.Consume(clock, backend));
    ASSERT_EQ(backend.Receipts().size(), 1);
    EXPECT_EQ(backend.Receipts()[0].sequence, 1);
    EXPECT_EQ(backend.Receipts()[0].submitted_at_us, 100);
    EXPECT_EQ(backend.Receipts()[0].consumed_at_us, 150);
    EXPECT_EQ(backend.Receipts()[0].pending_depth, 0);

    clock.Advance(25);
    EXPECT_EQ(mailbox.TrySubmit(128, clock), SubmitResult::Accepted);
    clock.Advance(25);
    EXPECT_TRUE(mailbox.Consume(clock, backend));
    ASSERT_EQ(backend.Receipts().size(), 2);
    EXPECT_EQ(backend.Receipts()[1].sequence, 2);
    EXPECT_EQ(backend.Receipts()[1].submitted_at_us, 175);
    EXPECT_EQ(backend.Receipts()[1].consumed_at_us, 200);
    EXPECT_LE(backend.Receipts()[0].sequence, backend.Receipts()[1].sequence);
    EXPECT_EQ(mailbox.PendingDepth(), 0);
}

TEST(ShadowAudioMailboxModel, ProducerPauseIsAnObservableGapWithoutSynthesizedSilence) {
    FakeClock clock{.now_us = 1000};
    AudioMailbox mailbox{0};
    FakeAudioBackend backend{1024};

    ASSERT_EQ(mailbox.TrySubmit(256, clock), SubmitResult::Accepted);
    clock.Advance(20);
    ASSERT_TRUE(mailbox.Consume(clock, backend));

    clock.Advance(500);
    EXPECT_FALSE(mailbox.Consume(clock, backend));
    EXPECT_EQ(mailbox.EmptyConsumePolls(), 1);

    clock.Advance(10);
    ASSERT_EQ(mailbox.TrySubmit(256, clock), SubmitResult::Accepted);
    clock.Advance(20);
    ASSERT_TRUE(mailbox.Consume(clock, backend));

    ASSERT_EQ(backend.Receipts().size(), 2);
    EXPECT_EQ(backend.Receipts()[1].sequence, 2);
    EXPECT_EQ(backend.Receipts()[1].submitted_at_us - backend.Receipts()[0].consumed_at_us, 510);
    EXPECT_EQ(backend.Receipts()[1].bytes, backend.Receipts()[0].bytes);
    EXPECT_EQ(backend.Receipts()[1].queued_bytes, 512);
}

TEST(ShadowAudioMailboxModel, BatchSubmissionIsAtomicAcrossIndependentMailboxes) {
    FakeClock clock{.now_us = 200};
    AudioMailboxSet mailboxes{2};
    FakeAudioBackend backend{1024};

    ASSERT_EQ(mailboxes.Port(0).TrySubmit(64, clock), SubmitResult::Accepted);
    const std::array blocked_batch{BatchItem{.port_id = 0, .bytes = 128},
                                   BatchItem{.port_id = 1, .bytes = 128}};
    EXPECT_FALSE(mailboxes.TrySubmitBatch(blocked_batch, clock));
    EXPECT_EQ(mailboxes.Port(1).PendingDepth(), 0);
    EXPECT_TRUE(mailboxes.Port(1).Submissions().empty());

    clock.Advance(10);
    ASSERT_TRUE(mailboxes.Port(0).Consume(clock, backend));
    ASSERT_TRUE(mailboxes.TrySubmitBatch(blocked_batch, clock));
    EXPECT_EQ(mailboxes.Port(0).PendingDepth(), 1);
    EXPECT_EQ(mailboxes.Port(1).PendingDepth(), 1);
    ASSERT_EQ(mailboxes.Port(0).Submissions().size(), 2);
    ASSERT_EQ(mailboxes.Port(1).Submissions().size(), 1);
    EXPECT_EQ(mailboxes.Port(0).Submissions()[1].submitted_at_us,
              mailboxes.Port(1).Submissions()[0].submitted_at_us);

    clock.Advance(10);
    EXPECT_TRUE(mailboxes.Port(1).Consume(clock, backend));
    EXPECT_TRUE(mailboxes.Port(0).Consume(clock, backend));
    ASSERT_EQ(backend.Receipts().size(), 3);
    EXPECT_EQ(backend.Receipts()[1].sequence, 1);
    EXPECT_EQ(backend.Receipts()[2].sequence, 2);
}

TEST(ShadowAudioMailboxModel, QueueThresholdReceiptRecordsBelowExactAndAboveDecisions) {
    FakeClock clock{.now_us = 300};
    AudioMailbox mailbox{0};
    FakeAudioBackend backend{100};

    ASSERT_EQ(mailbox.TrySubmit(40, clock), SubmitResult::Accepted);
    ASSERT_TRUE(mailbox.Consume(clock, backend));
    ASSERT_EQ(backend.Receipts().size(), 1);
    EXPECT_EQ(backend.Receipts()[0].queue_clear_count, 0);
    EXPECT_EQ(backend.Receipts()[0].queued_bytes, 40);

    backend.SetQueuedBytesForTest(100);
    clock.Advance(1);
    ASSERT_EQ(mailbox.TrySubmit(10, clock), SubmitResult::Accepted);
    ASSERT_TRUE(mailbox.Consume(clock, backend));
    ASSERT_EQ(backend.Receipts().size(), 2);
    EXPECT_EQ(backend.Receipts()[1].queue_clear_count, 1);
    EXPECT_EQ(backend.Receipts()[1].queued_bytes, 10);

    backend.SetQueuedBytesForTest(101);
    clock.Advance(1);
    ASSERT_EQ(mailbox.TrySubmit(20, clock), SubmitResult::Accepted);
    ASSERT_TRUE(mailbox.Consume(clock, backend));
    ASSERT_EQ(backend.Receipts().size(), 3);
    EXPECT_EQ(backend.Receipts()[2].queue_clear_count, 2);
    EXPECT_EQ(backend.Receipts()[2].queued_bytes, 20);
}

TEST(ShadowAudioMailboxModel, CloseRejectsLateWorkAndDropsPendingMailboxData) {
    FakeClock clock{.now_us = 400};
    AudioMailbox mailbox{0};
    FakeAudioBackend backend{1024};

    ASSERT_EQ(mailbox.TrySubmit(64, clock), SubmitResult::Accepted);
    EXPECT_EQ(mailbox.WaitForCapacity(), WaitResult::Blocked);
    mailbox.Close();
    EXPECT_EQ(mailbox.WaitForCapacity(), WaitResult::Closed);
    EXPECT_EQ(mailbox.PendingDepth(), 0);
    EXPECT_EQ(mailbox.TrySubmit(64, clock), SubmitResult::Closed);
    EXPECT_FALSE(mailbox.Consume(clock, backend));
    EXPECT_TRUE(backend.Receipts().empty());
}

TEST(ShadowAudioMailboxModel, SubmissionAndConsumptionReceiptsStayBounded) {
    FakeClock clock{.now_us = 500};
    AudioMailbox mailbox{0};
    FakeAudioBackend backend{1024};

    for (std::size_t index = 0; index < kReceiptCapacity + 4; ++index) {
        ASSERT_EQ(mailbox.TrySubmit(16, clock), SubmitResult::Accepted);
        ASSERT_TRUE(mailbox.Consume(clock, backend));
        clock.Advance(1);
    }

    EXPECT_EQ(mailbox.Submissions().size(), kReceiptCapacity);
    EXPECT_EQ(mailbox.Submissions()[0].sequence, 1);
    EXPECT_EQ(mailbox.Submissions().back().sequence, kReceiptCapacity);
    EXPECT_EQ(backend.Receipts().size(), kReceiptCapacity);
    EXPECT_EQ(backend.Receipts()[0].sequence, 1);
    EXPECT_EQ(backend.Receipts().back().sequence, kReceiptCapacity);
    EXPECT_EQ(backend.DroppedReceipts(), 4);
}

TEST(ShadowAudioBatchLockModel, OppositeGuestOrdersCanCrossHoldPortLocks) {
    // Owner one models [port 0, port 1], owner two models [port 1, port 0].
    // Each has acquired its first mutex before trying the second one.
    BatchLockModel locks{2};
    ASSERT_TRUE(locks.TryAcquire(1, 0));
    ASSERT_TRUE(locks.TryAcquire(2, 1));

    EXPECT_FALSE(locks.TryAcquire(1, 1));
    EXPECT_FALSE(locks.TryAcquire(2, 0));
}

TEST(ShadowAudioBatchLockModel, CanonicalOrderPreventsCrossHoldWithoutChangingCopyOrder) {
    const std::array<BatchItem, 2> first{{BatchItem{.port_id = 1, .bytes = 20},
                                          BatchItem{.port_id = 0, .bytes = 10}}};
    const std::array<BatchItem, 2> second{{BatchItem{.port_id = 0, .bytes = 30},
                                           BatchItem{.port_id = 1, .bytes = 40}}};

    const auto first_lock_order = CanonicalBatchLockOrder(first);
    const auto second_lock_order = CanonicalBatchLockOrder(second);
    ASSERT_EQ(first_lock_order, second_lock_order);
    EXPECT_EQ(first_lock_order, (std::vector<std::uint32_t>{0, 1}));

    // Locking in canonical order serializes the callers at the first shared
    // mutex. The guest's original order remains available for data copies.
    BatchLockModel locks{2};
    ASSERT_TRUE(locks.TryAcquire(1, first_lock_order[0]));
    ASSERT_TRUE(locks.TryAcquire(1, first_lock_order[1]));
    EXPECT_FALSE(locks.TryAcquire(2, second_lock_order[0]));
    locks.Release(1);
    ASSERT_TRUE(locks.TryAcquire(2, second_lock_order[0]));
    ASSERT_TRUE(locks.TryAcquire(2, second_lock_order[1]));

    EXPECT_EQ(first[0].port_id, 1);
    EXPECT_EQ(first[1].port_id, 0);
    EXPECT_EQ(second[0].port_id, 0);
    EXPECT_EQ(second[1].port_id, 1);
}

TEST(ShadowAudioBatchLockModel, ReleasingAllLocksLetsLaterWorkersDrainWhileWaiting) {
    // The batch is waiting for port 0, while port 1 already has a block that
    // its independent output worker must consume. Holding both mutexes while
    // waiting on port 0 would prevent that worker from making progress.
    BatchLockModel locks{2};
    std::array<bool, 2> output_ready{false, true};

    ASSERT_TRUE(locks.TryAcquire(1, 0));
    ASSERT_TRUE(locks.TryAcquire(1, 1));
    EXPECT_FALSE(locks.TryAcquire(2, 1));

    // The production wait path releases the whole batch, waits on the
    // selected port, and then retries the canonical acquisition.
    locks.Release(1);
    ASSERT_TRUE(locks.TryAcquire(2, 1));
    output_ready[1] = false;
    locks.Release(2);

    ASSERT_TRUE(locks.TryAcquire(1, 0));
    ASSERT_TRUE(locks.TryAcquire(1, 1));
    EXPECT_FALSE(output_ready[0]);
    EXPECT_FALSE(output_ready[1]);
}

TEST(ShadowAudioBatchLockSourceContract, ValidatesSlotsBeforeCanonicalLocking) {
#ifdef SHADPS4_AUDIOOUT_SOURCE_PATH
    std::ifstream input{SHADPS4_AUDIOOUT_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input.is_open());
    const std::string source{std::istreambuf_iterator<char>{input},
                             std::istreambuf_iterator<char>{}};

    const auto batch = source.find("sceAudioOutOutputs(");
    ASSERT_NE(batch, std::string::npos);

    // A malformed pair of distinct handles can still address one slot. The
    // implementation must reject that before taking the same mutex twice.
    const auto duplicate_slot = source.find("previous_port_id == port_id", batch);
    ASSERT_NE(duplicate_slot, std::string::npos);

    const auto lock_order = source.find("std::vector<u32> lock_order(num)", duplicate_slot);
    const auto sort_order = source.find("std::sort(lock_order.begin()", lock_order);
    const auto acquire = source.find("locks[index] = std::unique_lock", sort_order);
    const auto guest_wait = source.find("ports[wait_index]->output_cv.wait(wait_lock", acquire);
    ASSERT_NE(lock_order, std::string::npos);
    ASSERT_NE(sort_order, std::string::npos);
    ASSERT_NE(acquire, std::string::npos);
    ASSERT_NE(guest_wait, std::string::npos);
    EXPECT_LT(duplicate_slot, lock_order);
    EXPECT_LT(lock_order, sort_order);
    EXPECT_LT(sort_order, acquire);
    EXPECT_LT(acquire, guest_wait);

    // The loop encloses the canonical acquisition, so it begins before the
    // first lock assignment rather than after it.
    const auto wait_loop = source.find("while (true)", sort_order);
    ASSERT_NE(wait_loop, std::string::npos);
    const auto release_all = source.find("locks.clear()", wait_loop);
    const auto wait_lock = source.find("std::unique_lock<std::mutex> wait_lock", release_all);
    const auto wait_call = source.find("output_cv.wait(", wait_lock);
    const auto copy = source.find("std::memcpy(ports[i]->output_buffer", wait_call);
    ASSERT_NE(release_all, std::string::npos);
    ASSERT_NE(wait_lock, std::string::npos);
    ASSERT_NE(wait_call, std::string::npos);
    ASSERT_NE(copy, std::string::npos);
    EXPECT_EQ(source.find("output_cv.wait(locks[i]", wait_loop), std::string::npos);
    EXPECT_LT(wait_loop, release_all);
    EXPECT_LT(release_all, wait_lock);
    EXPECT_LT(wait_lock, wait_call);
    EXPECT_LT(wait_call, copy);
#else
    GTEST_SKIP() << "source-contract path is not configured";
#endif
}

TEST(ShadowAudioPortRoutingModel, ForgedCrossTypeHandleMustNotReuseLiveSlot) {
    struct PortIdentity {
        std::uint32_t slot;
        std::uint32_t type;
    };

    const PortIdentity live_main{.slot = 0, .type = 0};
    const PortIdentity forged_bgm{.slot = 0, .type = 1};
    const auto addresses_same_slot = live_main.slot == forged_bgm.slot;
    const auto types_match = live_main.type == forged_bgm.type;

    EXPECT_TRUE(addresses_same_slot);
    EXPECT_FALSE(types_match);
}

TEST(ShadowAudioPortRoutingSourceContract, ComparesHandleTypeWithLivePortBeforeUse) {
#if defined(SHADPS4_AUDIOOUT_SOURCE_PATH)
    std::ifstream input{SHADPS4_AUDIOOUT_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input.is_open());
    const std::string source{std::istreambuf_iterator<char>{input},
                             std::istreambuf_iterator<char>{}};

    const auto guard = source.find("PortMatchesHandleType");
    ASSERT_NE(guard, std::string::npos);

    // Every operation that dereferences a resolved port must apply the same
    // slot/type identity check. A slot-only lookup is insufficient because a
    // forged handle can retain a different encoded port type.
    constexpr std::array<const char*, 7> guarded_operations{
        "sceAudioOutClose(",       "sceAudioOutGetLastOutputTime(",
        "sceAudioOutGetPortState(", "sceAudioOutOutput(",
        "sceAudioOutOutputs(",     "sceAudioOutSetVolume(",
        "sceAudioOutSetMixLevelPadSpk(",
    };
    for (std::size_t index = 0; index < std::size(guarded_operations); ++index) {
        const auto* operation = guarded_operations[index];
        const auto operation_offset = source.find(operation);
        ASSERT_NE(operation_offset, std::string::npos) << operation;
        const auto next_operation_offset =
            index + 1 < std::size(guarded_operations)
                ? source.find(guarded_operations[index + 1], operation_offset + 1)
                : source.size();
        ASSERT_NE(next_operation_offset, std::string::npos) << operation;
        const auto type_guard_offset = source.find("PortMatchesHandleType", operation_offset);
        ASSERT_NE(type_guard_offset, std::string::npos) << operation;
        EXPECT_LT(type_guard_offset, next_operation_offset) << operation;
    }
#else
    GTEST_SKIP() << "source-contract path is not configured";
#endif
}

TEST(ShadowAudioOutputSourceContract, CopiesReadyBlockBeforeReleasingPortMutex) {
#ifdef SHADPS4_AUDIOOUT_SOURCE_PATH
    std::ifstream input{SHADPS4_AUDIOOUT_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input.is_open());
    const std::string source{std::istreambuf_iterator<char>{input},
                             std::istreambuf_iterator<char>{}};

    const auto worker = source.find("static void AudioOutputThread(");
    const auto worker_end = source.find("/*\n * sceAudioOut implementation", worker);
    ASSERT_NE(worker, std::string::npos);
    ASSERT_NE(worker_end, std::string::npos);

    const auto body = source.substr(worker, worker_end - worker);
    const auto snapshot = body.find("output_snapshot");
    const auto copy = body.find("std::memcpy(output_snapshot");
    const auto clear = body.find("port->output_ready = false", copy);
    const auto lock_end = body.find("// Wake producers", clear);
    const auto notify = body.find("port->output_cv.notify_all();", lock_end);
    const auto capture = body.find("CaptureTestLabMainPortAudio(", lock_end);
    const auto capture_snapshot = body.find("output_snapshot.data()", capture);
    const auto backend = body.find("port->impl->Output(output_snapshot", capture);

    ASSERT_NE(snapshot, std::string::npos);
    ASSERT_NE(copy, std::string::npos);
    ASSERT_NE(clear, std::string::npos);
    ASSERT_NE(lock_end, std::string::npos);
    ASSERT_NE(notify, std::string::npos);
    ASSERT_NE(capture, std::string::npos);
    ASSERT_NE(capture_snapshot, std::string::npos);
    ASSERT_NE(backend, std::string::npos);
    EXPECT_LT(copy, clear);
    EXPECT_LT(clear, lock_end);
    EXPECT_LT(lock_end, notify);
    EXPECT_LT(notify, capture_snapshot);
    EXPECT_LT(capture, backend);

    // The producer may refill output_buffer immediately after output_ready is
    // cleared, so all slow diagnostics/backend work must consume the snapshot.
    EXPECT_EQ(body.find("CaptureTestLabMainPortAudio(port->output_buffer"), std::string::npos);
    EXPECT_EQ(body.find("port->impl->Output(port->output_buffer"), std::string::npos);
#else
    GTEST_SKIP() << "source-contract path is not configured";
#endif
}

TEST(ShadowAudioOutputSourceContract, BackendCallsUseDedicatedMutex) {
#if defined(SHADPS4_AUDIOOUT_SOURCE_PATH) && defined(SHADPS4_AUDIOOUT_HEADER_PATH)
    std::ifstream source_input{SHADPS4_AUDIOOUT_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(source_input.is_open());
    const std::string source{std::istreambuf_iterator<char>{source_input},
                             std::istreambuf_iterator<char>{}};

    std::ifstream header_input{SHADPS4_AUDIOOUT_HEADER_PATH, std::ios::binary};
    ASSERT_TRUE(header_input.is_open());
    const std::string header{std::istreambuf_iterator<char>{header_input},
                             std::istreambuf_iterator<char>{}};
    EXPECT_NE(header.find("std::mutex backend_mutex"), std::string::npos);

    const auto worker = source.find("static void AudioOutputThread(");
    const auto worker_end = source.find("/*\n * sceAudioOut implementation", worker);
    ASSERT_NE(worker, std::string::npos);
    ASSERT_NE(worker_end, std::string::npos);
    const auto worker_body = source.substr(worker, worker_end - worker);
    EXPECT_NE(worker_body.find("backend_mutex"), std::string::npos);

    const auto adjust = source.find("void AdjustVol()");
    ASSERT_NE(adjust, std::string::npos);
    const auto adjust_end = source.find("static void AudioOutputThread(", adjust);
    ASSERT_NE(adjust_end, std::string::npos);
    EXPECT_NE(source.substr(adjust, adjust_end - adjust).find("backend_mutex"),
              std::string::npos);

    const auto set_volume = source.find("sceAudioOutSetVolume(");
    ASSERT_NE(set_volume, std::string::npos);
    const auto set_volume_end = source.find("sceAudioOutSetMixLevelPadSpk(", set_volume);
    ASSERT_NE(set_volume_end, std::string::npos);
    EXPECT_NE(source.substr(set_volume, set_volume_end - set_volume).find("backend_mutex"),
              std::string::npos);
#else
    GTEST_SKIP() << "source-contract paths are not configured";
#endif
}

TEST(ShadowAudioOutputSourceContract, FailureBoundaryIsLocalAndWakesProducers) {
#if defined(SHADPS4_AUDIOOUT_SOURCE_PATH) && defined(SHADPS4_AUDIOOUT_HEADER_PATH)
    std::ifstream source_input{SHADPS4_AUDIOOUT_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(source_input.is_open());
    const std::string source{std::istreambuf_iterator<char>{source_input},
                             std::istreambuf_iterator<char>{}};

    std::ifstream header_input{SHADPS4_AUDIOOUT_HEADER_PATH, std::ios::binary};
    ASSERT_TRUE(header_input.is_open());
    const std::string header{std::istreambuf_iterator<char>{header_input},
                             std::istreambuf_iterator<char>{}};

    const auto worker = source.find("static void AudioOutputThread(");
    const auto worker_end = source.find("/*\n * sceAudioOut implementation", worker);
    ASSERT_NE(worker, std::string::npos);
    ASSERT_NE(worker_end, std::string::npos);
    const auto worker_body = source.substr(worker, worker_end - worker);

    // The failure state is AudioOut-owned. It must not be introduced in the
    // shared Kernel::Thread wrapper, whose other users have no common failure
    // receipt or restart contract.
    EXPECT_NE(header.find("enum class AudioWorkerFailureStage"), std::string::npos);
    EXPECT_NE(header.find("std::optional<AudioWorkerFailureReceipt> worker_failure"),
              std::string::npos);
    EXPECT_NE(header.find("bool worker_failed"), std::string::npos);
    EXPECT_NE(worker_body.find("RecordAudioOutputWorkerFailure"), std::string::npos);
    EXPECT_NE(worker_body.find("catch (const std::exception&)"), std::string::npos);
    EXPECT_NE(worker_body.find("catch (...)") , std::string::npos);

    const auto failure = worker_body.find("RecordAudioOutputWorkerFailure");
    const auto failure_notify = worker_body.find("port->output_cv.notify_all()", failure);
    ASSERT_NE(failure, std::string::npos);
    ASSERT_NE(failure_notify, std::string::npos);
    EXPECT_LT(failure, failure_notify);

    const auto output = source.find("s32 PS4_SYSV_ABI sceAudioOutOutput(");
    const auto output_end = source.find("s32 PS4_SYSV_ABI sceAudioOutOutputs(", output);
    ASSERT_NE(output, std::string::npos);
    ASSERT_NE(output_end, std::string::npos);
    const auto output_body = source.substr(output, output_end - output);
    const auto output_wait = output_body.find("output_cv.wait");
    const auto output_failed = output_body.find("worker_failed", output_wait);
    ASSERT_NE(output_wait, std::string::npos);
    ASSERT_NE(output_failed, std::string::npos);
    EXPECT_LT(output_wait, output_failed);

    const auto batch = source.find("s32 PS4_SYSV_ABI sceAudioOutOutputs(");
    ASSERT_NE(batch, std::string::npos);
    const auto batch_failed = source.find("worker_failed", batch);
    ASSERT_NE(batch_failed, std::string::npos);

    // This test deliberately checks that the generic thread wrapper remains
    // untouched; adding a catch there would change unrelated worker behavior.
#if defined(SHADPS4_KERNEL_THREADS_HEADER_PATH)
    std::ifstream thread_input{SHADPS4_KERNEL_THREADS_HEADER_PATH, std::ios::binary};
    ASSERT_TRUE(thread_input.is_open());
    const std::string thread_header{std::istreambuf_iterator<char>{thread_input},
                                    std::istreambuf_iterator<char>{}};
    EXPECT_EQ(thread_header.find("catch ("), std::string::npos);
#endif
#else
    GTEST_SKIP() << "source-contract paths are not configured";
#endif
}

TEST(ShadowAudioQueueSourceContract, ThresholdUsesTheStreamInputByteUnits) {
#ifdef SHADPS4_SDLAUDIO_SOURCE_PATH
    std::ifstream input{SHADPS4_SDLAUDIO_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input.is_open());
    const std::string source{std::istreambuf_iterator<char>{input},
                             std::istreambuf_iterator<char>{}};

    const auto function = source.find("void CalculateQueueThreshold()");
    ASSERT_NE(function, std::string::npos);
    const auto end = source.find("\n    }", function);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(function, end - function);

    // SDL reports queued bytes in the stream's input format. AudioOut always
    // feeds the stream converted float data, and multichannel ports may be
    // folded to stereo before the stream is recreated. Comparing that queue
    // against the original guest byte count or guest channel count mixes
    // units and changes when the clear threshold is reached.
    EXPECT_NE(body.find("internal_buffer_size"), std::string::npos);
    EXPECT_NE(body.find("ps4_downmix"), std::string::npos);
    EXPECT_NE(body.find("stream_channels"), std::string::npos);
    EXPECT_EQ(body.find("std::max(guest_buffer_size"), std::string::npos);
#else
    GTEST_SKIP() << "source-contract path is not configured";
#endif
}

TEST(ShadowAudioQueuePolicy, UsesConvertedStreamBytesForGuestFormatAndDownmix) {
    // A 1024-frame S16 stereo guest block becomes 8192 bytes of float input
    // for SDL. The queue threshold must be four converted blocks, not four
    // 4096-byte guest blocks.
    EXPECT_EQ(CalculateAudioQueueThreshold(8192, 8, 256), 32768u);

    // An eight-channel S16 port folded to stereo still feeds the stream as
    // float stereo. Its original 16384-byte guest block must not inflate the
    // threshold fourfold after downmix.
    EXPECT_EQ(CalculateAudioQueueThreshold(8192, 8, 1024), 32768u);

    // Non-downmixed eight-channel float input retains its full stream frame
    // width and therefore scales from the converted block size accordingly.
    EXPECT_EQ(CalculateAudioQueueThreshold(32768, 32, 1024), 131072u);
}

TEST(ShadowAudioQueuePolicy, HandlesMissingDeviceFramesAndSaturates) {
    EXPECT_EQ(CalculateAudioQueueThreshold(8192, 8, 0), 32768u);
    EXPECT_EQ(CalculateAudioQueueThreshold(8192, 8, -1), 32768u);
    EXPECT_EQ(CalculateAudioQueueThreshold(std::numeric_limits<std::uint32_t>::max(), 32, 1),
              std::numeric_limits<std::uint32_t>::max());
}

} // namespace
