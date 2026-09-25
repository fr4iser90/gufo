#include "src/core/gguf_weight_stats.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {

void Expect(bool ok, std::string_view message) {
  if (!ok) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

}  // namespace

int main() {
  gufo::core::GgufWeightStats stats;
  stats.quant_label = "Q4_K";
  stats.tensors = 2;
  stats.parameters = 1'000'000;
  stats.bytes = 500'000;  // 4 bpw
  stats.bits_per_weight = (static_cast<double>(stats.bytes) * 8.0) /
                          static_cast<double>(stats.parameters);
  stats.by_type.push_back({.type = gufo::core::GgmlType::kQ4_K,
                           .tensors = 2,
                           .parameters = stats.parameters,
                           .bytes = stats.bytes});

  Expect(std::fabs(stats.bits_per_weight - 4.0) < 1e-9, "4.0 bpw");
  const auto line = gufo::core::FormatGgufWeightStats(stats);
  Expect(line.find("bpw=4.00") != std::string::npos, "formatted bpw");
  Expect(line.find("quant=Q4_K") != std::string::npos, "formatted label");
  Expect(line.find("weight_gib=") != std::string::npos, "formatted gib");
  return 0;
}
