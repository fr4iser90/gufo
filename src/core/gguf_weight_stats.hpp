#ifndef GUFO_CORE_GGUF_WEIGHT_STATS_HPP_
#define GUFO_CORE_GGUF_WEIGHT_STATS_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::core {

/// Aggregate weight accounting from a GGUF tensor table.
///
/// Bits-per-weight is derived from shapes and encoded payload sizes (padding
/// and scale planes included), matching Halogen's QUANT.md arithmetic rather
/// than a format name.
struct GgufWeightStats {
  struct TypeBucket {
    GgmlType type{GgmlType::kF16};
    std::uint64_t tensors{0};
    std::uint64_t parameters{0};
    std::uint64_t bytes{0};
  };

  std::uint64_t tensors{0};
  std::uint64_t parameters{0};
  std::uint64_t bytes{0};
  double bits_per_weight{0.0};
  std::string quant_label;
  std::vector<TypeBucket> by_type;
};

[[nodiscard]] GgufWeightStats ComputeGgufWeightStats(const GgufReader& reader);

[[nodiscard]] std::string FormatGgufWeightStats(
    const GgufWeightStats& stats) noexcept;

}  // namespace gufo::core

#endif  // GUFO_CORE_GGUF_WEIGHT_STATS_HPP_
