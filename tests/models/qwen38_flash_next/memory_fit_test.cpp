#include "src/models/qwen38_flash_next/memory_fit.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using gufo::models::qwen38_flash_next::Config;
using gufo::models::qwen38_flash_next::EstimateAttentionPoolBytes;
using gufo::models::qwen38_flash_next::FitSharedKvPool;
using gufo::models::qwen38_flash_next::kGiB;
using gufo::models::qwen38_flash_next::SharedKvFitRequest;

void Expect(bool ok, std::string_view message) {
  if (!ok) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

Config FlashNextLikeConfig() {
  Config c;
  c.num_layers = 48;
  c.hidden_size = 2560;
  c.full_attention_interval = 4;
  c.num_kv_heads = 2;
  c.head_dim = 256;
  c.indexer_heads = 4;
  c.indexer_head_dim = 128;
  c.indexer_top_k = 2048;
  c.compress_ratio = 4;
  c.ssm_conv_kernel = 4;
  c.ssm_head_dim = 128;
  c.ssm_num_k_heads = 16;
  c.ssm_num_v_heads = 48;
  c.hc_count = 4;
  return c;
}

}  // namespace

int main() {
  const auto c = FlashNextLikeConfig();
  Expect(EstimateAttentionPoolBytes(c, 262144) >
             EstimateAttentionPoolBytes(c, 131072),
         "larger pool needs more bytes");

  // Tiny machine: requested 512k must shrink toward --context.
  {
    const auto fit = FitSharedKvPool(c, SharedKvFitRequest{
                                            .requested_positions = 524288,
                                            .max_context = 262144,
                                            .sessions = 2,
                                            .weight_bytes = 68ULL * kGiB,
                                            .mem_total_bytes = 96ULL * kGiB,
                                            .host_reserve_bytes = 20ULL * kGiB,
                                            .speculative = false,
                                        });
    // Budget ≈ 8 GiB; 512k pool alone is ~13 GiB, so shrink to 262k (~6.5 GiB).
    Expect(fit.ok, "tight box still fits after shrink: " + fit.message);
    Expect(fit.positions < fit.requested_positions, "pool was lowered");
    Expect(fit.positions >= 262144, "never below --context");
    Expect(fit.message.find("LOWERING THE POOL") != std::string::npos,
           "log names the shrink");
  }

  {
    const auto fit = FitSharedKvPool(c, SharedKvFitRequest{
                                            .requested_positions = 1048576,
                                            .max_context = 1048576,
                                            .sessions = 1,
                                            .weight_bytes = 80ULL * kGiB,
                                            .mem_total_bytes = 96ULL * kGiB,
                                            .host_reserve_bytes = 16ULL * kGiB,
                                            .speculative = true,
                                        });
    Expect(!fit.ok, "1M speculative claim that cannot shrink must refuse");
    Expect(fit.message.find("lighter GGUF") != std::string::npos ||
               fit.message.find("lower-bpw") != std::string::npos,
           "refusal names GGUF lever");
  }

  {
    const auto fit = FitSharedKvPool(c, SharedKvFitRequest{
                                            .requested_positions = 262144,
                                            .max_context = 262144,
                                            .sessions = 2,
                                            .weight_bytes = 68ULL * kGiB,
                                            .mem_total_bytes = 128ULL * kGiB,
                                            .host_reserve_bytes = 16ULL * kGiB,
                                            .speculative = false,
                                        });
    Expect(fit.ok, "comfortable box keeps requested pool");
    Expect(fit.positions == 262144, "no unnecessary shrink");
  }
  return 0;
}
