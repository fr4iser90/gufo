#include "src/models/qwen38_flash_next/yarn.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace {

using gufo::models::qwen38_flash_next::EffectiveContextLimit;
using gufo::models::qwen38_flash_next::MakeYarnScales;
using gufo::models::qwen38_flash_next::YarnFactorSupported;

void Expect(bool ok, const char* message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

void TestFactorGate() {
  Expect(YarnFactorSupported(0), "off");
  Expect(YarnFactorSupported(1), "identity");
  Expect(YarnFactorSupported(4), "halogen 1M");
  Expect(!YarnFactorSupported(2), "unsupported factor");
  Expect(!YarnFactorSupported(8), "unsupported factor");
}

void TestContextLimit() {
  Expect(EffectiveContextLimit(262144, 0) == 262144, "native off");
  Expect(EffectiveContextLimit(262144, 1) == 262144, "native identity");
  Expect(EffectiveContextLimit(262144, 4) == 1048576, "1M with yarn 4");
}

void TestScales() {
  const auto off = MakeYarnScales(0);
  Expect(off.freq_scale == 1.0F && off.attn_factor == 1.0F, "off scales");
  const auto yarn = MakeYarnScales(4);
  Expect(std::abs(yarn.freq_scale - 0.25F) < 1e-6F, "freq scale");
  Expect(std::abs(yarn.attn_factor - (1.0F + 0.1F * std::log(4.0F))) < 1e-5F,
         "attn factor");
}

}  // namespace

int main() {
  try {
    TestFactorGate();
    TestContextLimit();
    TestScales();
  } catch (const std::exception&) {
    return 1;
  }
  return 0;
}
