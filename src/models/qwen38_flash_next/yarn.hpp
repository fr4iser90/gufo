#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_YARN_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_YARN_HPP_

#include <cmath>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

/// Static YaRN scales for long-context RoPE: frequency compression by
/// `1/factor` and a mild attention rescale (factor 4 → ~1M on a 262k native).
struct YarnScales {
  float freq_scale{1.0F};
  float attn_factor{1.0F};
};

[[nodiscard]] inline constexpr bool YarnFactorSupported(
    std::uint32_t factor) noexcept {
  return factor == 0 || factor == 1 || factor == 4;
}

[[nodiscard]] inline constexpr std::uint32_t EffectiveContextLimit(
    std::uint32_t native_context, std::uint32_t yarn_factor) noexcept {
  const std::uint32_t factor = yarn_factor == 0 ? 1 : yarn_factor;
  return native_context * factor;
}

[[nodiscard]] inline YarnScales MakeYarnScales(std::uint32_t factor) noexcept {
  if (factor <= 1) {
    return {};
  }
  const float f = static_cast<float>(factor);
  return {.freq_scale = 1.0F / f, .attn_factor = 1.0F + 0.1F * std::log(f)};
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_YARN_HPP_
