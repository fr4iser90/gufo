#ifndef GUFO_SERVER_REQUEST_PROGRESS_HPP_
#define GUFO_SERVER_REQUEST_PROGRESS_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>

#include "src/cli/serve/logging.hpp"

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
            << " bytes=" << bytes_.load(std::memory_order_relaxed) << ' '
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
