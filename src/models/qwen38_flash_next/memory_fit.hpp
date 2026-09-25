#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MEMORY_FIT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MEMORY_FIT_HPP_

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::models::qwen38_flash_next {

inline constexpr std::uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;
/// Host headroom kept free for the n-gram page cache and OS. Override with
/// `--host-reserve-gib` when fitting a larger shared KV pool.
inline constexpr std::uint64_t kDefaultHostReserveBytes = 16ULL * kGiB;

[[nodiscard]] inline std::size_t EstimateAttentionPoolBytes(
    const Config& c, std::uint32_t positions) noexcept {
  if (positions == 0 || c.full_attention_interval == 0 ||
      c.compress_ratio == 0) {
    return 0;
  }
  const std::size_t attention_layers =
      static_cast<std::size_t>(c.num_layers / c.full_attention_interval);
  const std::size_t kv_row = c.AttentionKvDim();
  const std::size_t block_rows =
      static_cast<std::size_t>(positions) / c.compress_ratio + 1;
  constexpr std::size_t kHalf = 2;  // __half
  const std::size_t per_layer =
      static_cast<std::size_t>(positions) * kv_row * kHalf * 2 +
      block_rows * c.indexer_head_dim * kHalf;
  return attention_layers * per_layer;
}

/// Private per-session state excluding shared attention pool rows. Speculative
/// MTP still keeps a private draft KV sized to max_context.
[[nodiscard]] inline std::size_t EstimatePrivateSessionBytes(
    const Config& c, std::uint32_t max_context, bool speculative,
    std::uint32_t max_batch = 1) noexcept {
  if (max_context == 0 || c.full_attention_interval == 0) {
    return 0;
  }
  const std::size_t attention =
      static_cast<std::size_t>(c.num_layers / c.full_attention_interval);
  const std::size_t linear = static_cast<std::size_t>(c.num_layers) - attention;
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t state =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  const std::uint32_t indexer_cap =
      static_cast<std::uint32_t>(std::bit_ceil(std::min<std::uint64_t>(
          max_context, std::uint64_t{c.indexer_top_k} + max_batch)));
  const std::size_t private_index =
      std::size_t{indexer_cap} * c.indexer_head_dim * sizeof(float);
  std::size_t bytes = (linear * conv + ple + linear * state) * sizeof(float) +
                      attention * private_index;
  if (speculative) {
    constexpr std::size_t kHalf = 2;
    const std::size_t kv =
        2 * std::size_t{max_context} * c.AttentionKvDim() * kHalf;
    const std::size_t index =
        std::size_t{indexer_cap} * c.indexer_head_dim * sizeof(float) +
        std::size_t{max_context / c.compress_ratio + 1} * c.indexer_head_dim *
            kHalf;
    bytes += kv + index + std::size_t{8} * c.HcDim() * sizeof(float);
  }
  return bytes;
}

struct SharedKvFitRequest {
  std::uint32_t requested_positions{0};
  std::uint32_t max_context{0};
  std::size_t sessions{1};
  std::uint64_t weight_bytes{0};
  std::uint64_t mem_total_bytes{0};
  std::uint64_t host_reserve_bytes{kDefaultHostReserveBytes};
  bool speculative{false};
  std::uint32_t max_batch{1};
};

struct SharedKvFitResult {
  bool ok{false};
  std::uint32_t positions{0};
  std::uint32_t requested_positions{0};
  std::uint64_t pool_bytes{0};
  std::uint64_t sessions_bytes{0};
  std::uint64_t need_bytes{0};
  std::uint64_t budget_bytes{0};
  std::uint64_t host_left_bytes{0};
  std::string message;
};

[[nodiscard]] inline SharedKvFitResult FitSharedKvPool(
    const Config& c, const SharedKvFitRequest& request) noexcept {
  SharedKvFitResult result;
  result.requested_positions = request.requested_positions;
  result.positions = request.requested_positions;
  if (request.requested_positions == 0) {
    result.ok = true;
    result.message = "shared kv pool disabled";
    return result;
  }
  if (request.requested_positions < request.max_context) {
    result.message = "kv-pool-positions must be at least --context (" +
                     std::to_string(request.max_context) + ")";
    return result;
  }
  if (request.mem_total_bytes <=
      request.host_reserve_bytes + request.weight_bytes) {
    result.message =
        "host memory is smaller than weights plus reserve; levers: lighter "
        "GGUF (lower bpw), --host-reserve-gib, fewer --sessions";
    return result;
  }
  result.budget_bytes = request.mem_total_bytes - request.host_reserve_bytes -
                        request.weight_bytes;

  auto eval = [&](std::uint32_t positions) {
    result.positions = positions;
    result.pool_bytes = EstimateAttentionPoolBytes(c, positions);
    result.sessions_bytes =
        request.sessions * EstimatePrivateSessionBytes(c, request.max_context,
                                                       request.speculative,
                                                       request.max_batch);
    result.need_bytes = result.pool_bytes + result.sessions_bytes;
    result.host_left_bytes = result.budget_bytes > result.need_bytes
                                 ? result.budget_bytes - result.need_bytes
                                 : 0;
  };

  eval(request.requested_positions);
  while (result.need_bytes > result.budget_bytes &&
         result.positions > request.max_context) {
    const auto next = std::max(request.max_context, result.positions / 2);
    if (next == result.positions) {
      break;
    }
    eval(next);
  }

  if (result.need_bytes > result.budget_bytes) {
    std::ostringstream out;
    out << "kv pool: " << result.requested_positions << " positions need ~"
        << (static_cast<double>(result.need_bytes) / static_cast<double>(kGiB))
        << " GiB (pool "
        << (static_cast<double>(result.pool_bytes) / static_cast<double>(kGiB))
        << " GiB + sessions "
        << (static_cast<double>(result.sessions_bytes) /
            static_cast<double>(kGiB))
        << " GiB) but only ~"
        << (static_cast<double>(result.budget_bytes) /
            static_cast<double>(kGiB))
        << " GiB remain after weights/reserve; levers: lower "
           "--kv-pool-positions, --context, --sessions, --host-reserve-gib, "
           "or a lower-bpw GGUF";
    if (request.speculative) {
      out << " (MTP keeps a private draft KV sized to --context)";
    }
    result.message = out.str();
    result.ok = false;
    return result;
  }

  result.ok = true;
  std::ostringstream out;
  out << std::fixed;
  out.precision(1);
  if (result.positions != result.requested_positions) {
    out << "kv pool: " << result.requested_positions << " positions need ~"
        << (static_cast<double>(
                EstimateAttentionPoolBytes(c, result.requested_positions) +
                request.sessions *
                    EstimatePrivateSessionBytes(c, request.max_context,
                                                request.speculative,
                                                request.max_batch)) /
            static_cast<double>(kGiB))
        << " GiB after weights/reserve; LOWERING THE POOL TO "
        << result.positions;
  } else {
    out << "kv pool: " << result.positions << " positions (~"
        << (static_cast<double>(result.pool_bytes) / static_cast<double>(kGiB))
        << " GiB) fits; host memory left ~"
        << (static_cast<double>(result.host_left_bytes) /
            static_cast<double>(kGiB))
        << " GiB";
  }
  result.message = out.str();
  return result;
}

[[nodiscard]] inline std::uint64_t ReadMemTotalBytes() noexcept {
  std::ifstream meminfo("/proc/meminfo");
  for (std::string line; std::getline(meminfo, line);) {
    if (!line.starts_with("MemTotal:")) {
      continue;
    }
    std::istringstream fields(line.substr(9));
    std::uint64_t kib = 0;
    if (fields >> kib) {
      return kib * 1024ULL;
    }
  }
  return 0;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_MEMORY_FIT_HPP_
