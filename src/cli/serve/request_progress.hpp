#ifndef GUFO_SERVER_REQUEST_PROGRESS_HPP_
#define GUFO_SERVER_REQUEST_PROGRESS_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/text_generation_backend.hpp"

namespace gufo::server {
namespace detail {

inline std::atomic<std::uint64_t>& ActiveGenerations() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}

inline std::atomic<std::uint64_t>& LastTtftMs() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}

inline std::atomic<std::uint64_t>& LastQueueMs() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}

inline std::atomic<std::uint64_t>& LastPromptTokens() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}

inline std::atomic<std::uint64_t>& LastCompletionTokens() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}

inline std::atomic<std::uint64_t>& OccupancySessionsActive() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}
inline std::atomic<std::uint64_t>& OccupancySessionsCapacity() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}
inline std::atomic<std::uint64_t>& OccupancyUsedTokens() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}
inline std::atomic<std::uint64_t>& OccupancyCapacityTokens() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}
inline std::atomic<std::uint64_t>& OccupancyMaxUsedTokens() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}
inline std::atomic<std::uint64_t>& OccupancyQueued() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}
inline std::atomic<std::uint32_t>& OccupancyMaxContext() {
  static std::atomic<std::uint32_t> value{0};
  return value;
}

inline void PublishOccupancyGauges(const TextServingSnapshot& snap) noexcept {
  OccupancySessionsActive().store(snap.active_sessions,
                                  std::memory_order_relaxed);
  OccupancySessionsCapacity().store(snap.session_capacity,
                                    std::memory_order_relaxed);
  OccupancyUsedTokens().store(snap.used_tokens, std::memory_order_relaxed);
  OccupancyCapacityTokens().store(snap.capacity_tokens,
                                  std::memory_order_relaxed);
  OccupancyMaxUsedTokens().store(snap.max_used_tokens,
                                 std::memory_order_relaxed);
  OccupancyQueued().store(snap.queued, std::memory_order_relaxed);
  OccupancyMaxContext().store(snap.max_context, std::memory_order_relaxed);
}

inline constexpr std::array<double, 11> kTtftBucketsMs = {
    5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000};

inline std::array<std::atomic<std::uint64_t>, 12>& TtftBucketCounts() {
  static std::array<std::atomic<std::uint64_t>, 12> counts{};
  return counts;
}
inline std::atomic<std::uint64_t>& TtftObservationCount() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TtftSumMicros() {
  static std::atomic<std::uint64_t> sum{0};
  return sum;
}

inline void ObserveTtftMs(double ttft_ms) noexcept {
  if (!std::isfinite(ttft_ms) || ttft_ms < 0.0) {
    return;
  }
  const auto micros = static_cast<std::uint64_t>(ttft_ms * 1000.0 + 0.5);
  TtftSumMicros().fetch_add(micros, std::memory_order_relaxed);
  TtftObservationCount().fetch_add(1, std::memory_order_relaxed);
  auto& buckets = TtftBucketCounts();
  for (std::size_t i = 0; i < kTtftBucketsMs.size(); ++i) {
    if (ttft_ms <= kTtftBucketsMs[i]) {
      buckets[i].fetch_add(1, std::memory_order_relaxed);
    }
  }
  buckets.back().fetch_add(1, std::memory_order_relaxed);
}

}  // namespace detail

/// Throttled mid-request logs while a generation is in flight.
///
/// Until the first token piece arrives the poller reports phase=prefill
/// (queue + prompt eval). Decode updates come from OnPiece.
class RequestProgressLogger {
public:
  using Clock = std::chrono::steady_clock;

  explicit RequestProgressLogger(std::string request_id)
      : request_id_(std::move(request_id)),
        start_(Clock::now()),
        last_log_(start_) {
    detail::ActiveGenerations().fetch_add(1, std::memory_order_relaxed);
    Log("started");
    poller_ = std::thread([this] { PrefillPoll(); });
  }

  RequestProgressLogger(const RequestProgressLogger&) = delete;
  RequestProgressLogger& operator=(const RequestProgressLogger&) = delete;

  ~RequestProgressLogger() {
    StopPoller();
    detail::ActiveGenerations().fetch_sub(1, std::memory_order_relaxed);
  }

  bool OnPiece(std::string_view piece) {
    if (piece.empty()) {
      return true;
    }
    pieces_.fetch_add(1, std::memory_order_relaxed);
    bytes_.fetch_add(piece.size(), std::memory_order_relaxed);

    bool expected = false;
    if (decode_started_.compare_exchange_strong(expected, true,
                                                std::memory_order_acq_rel)) {
      Log("decode");
      last_log_ = Clock::now();
      return true;
    }

    const auto now = Clock::now();
    if (now - last_log_ >= kInterval) {
      Log("decode");
      last_log_ = now;
    }
    return true;
  }

private:
  static constexpr auto kInterval = std::chrono::seconds(2);

  void PrefillPoll() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
      if (cv_.wait_for(lock, kInterval, [this] { return stop_; })) {
        return;
      }
      if (!decode_started_.load(std::memory_order_acquire)) {
        lock.unlock();
        Log("prefill");
        lock.lock();
      }
    }
  }

  void StopPoller() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (poller_.joinable()) {
      poller_.join();
    }
  }

  void Log(std::string_view phase) const {
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                              start_)
            .count();
    std::ostringstream message;
    message << "request=" << request_id_ << " event=progress phase=" << phase
            << " elapsed_ms=" << elapsed_ms
            << " pieces=" << pieces_.load(std::memory_order_relaxed)
            << " bytes=" << bytes_.load(std::memory_order_relaxed)
            << " sessions_active="
            << detail::OccupancySessionsActive().load(std::memory_order_relaxed)
            << '/'
            << detail::OccupancySessionsCapacity().load(
                   std::memory_order_relaxed)
            << " context_used="
            << detail::OccupancyUsedTokens().load(std::memory_order_relaxed)
            << '/'
            << detail::OccupancyCapacityTokens().load(std::memory_order_relaxed)
            << " context_hot="
            << detail::OccupancyMaxUsedTokens().load(std::memory_order_relaxed)
            << '/'
            << detail::OccupancyMaxContext().load(std::memory_order_relaxed)
            << " queued="
            << detail::OccupancyQueued().load(std::memory_order_relaxed) << ' '
            << Logger::MemoryStatus();
    Logger::Info("http", message.str());
  }

  std::string request_id_;
  Clock::time_point start_;
  Clock::time_point last_log_;
  std::atomic<bool> decode_started_{false};
  std::atomic<std::uint64_t> pieces_{0};
  std::atomic<std::uint64_t> bytes_{0};
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_{false};
  std::thread poller_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_REQUEST_PROGRESS_HPP_
