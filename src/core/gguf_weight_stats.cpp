#include "src/core/gguf_weight_stats.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace gufo::core {

GgufWeightStats ComputeGgufWeightStats(const GgufReader& reader) {
  GgufWeightStats stats;
  stats.quant_label = reader.GetQuantizationLabel();
  std::unordered_map<std::uint16_t, GgufWeightStats::TypeBucket> buckets;
  for (const auto& tensor : reader.GetTensors()) {
    const auto params = tensor.ElementCount();
    if (params == 0 || tensor.size_bytes == 0) {
      continue;
    }
    ++stats.tensors;
    stats.parameters += params;
    stats.bytes += tensor.size_bytes;
    auto& bucket = buckets[static_cast<std::uint16_t>(tensor.type)];
    bucket.type = tensor.type;
    ++bucket.tensors;
    bucket.parameters += params;
    bucket.bytes += tensor.size_bytes;
  }
  stats.by_type.reserve(buckets.size());
  for (auto& [_, bucket] : buckets) {
    stats.by_type.push_back(std::move(bucket));
  }
  std::sort(stats.by_type.begin(), stats.by_type.end(),
            [](const auto& left, const auto& right) {
              return left.bytes > right.bytes;
            });
  if (stats.parameters != 0) {
    stats.bits_per_weight = (static_cast<double>(stats.bytes) * 8.0) /
                            static_cast<double>(stats.parameters);
  }
  return stats;
}

std::string FormatGgufWeightStats(const GgufWeightStats& stats) noexcept {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2);
  out << "quant=" << stats.quant_label << " tensors=" << stats.tensors
      << " params=" << stats.parameters << " weight_gib="
      << (static_cast<double>(stats.bytes) / (1024.0 * 1024.0 * 1024.0))
      << " bpw=" << stats.bits_per_weight;
  return out.str();
}

}  // namespace gufo::core
