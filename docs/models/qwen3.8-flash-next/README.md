# Qwen3.8 Flash-Next

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards).
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
nix build
MODEL=/path/to/first-target-shard.gguf
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

The loader discovers the remaining shards. Omit the speculative options for AR;
AR sessions allocate no predictor state even if a shared model has MTP loaded.
Adaptive MTP is default, with `--draft-tokens` capping 1–7 proposals. Sampled
requests use deterministic acceptance/cost control for seeded replay; all-greedy
C>1 batches may use measured cycle costs. Each request keeps private caches,
rollback and RNG. See [MTP qualification](QUALITY.md).

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
for explicit effort/thinking overrides. Memory grows with used context and selected rollback depth;
admission reserves the configured capacity before creating sessions.

`--kv-pool-positions N` (Flash-Next HTTP only) enables a shared attention KV
pool. Sessions keep private SSM/indexer/MTP state and reserve contiguous spans
from the pool as context grows. `N` must be at least `--context`. At load, Gufo
fits the pool to host RAM (weights + `--host-reserve-gib`, default 16) and may
lower `N`, logging `event=kv_pool_fit`. Example for two concurrent requests
sharing one native context budget:

```sh
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 262144 \
  --kv-pool-positions 262144
```

Unset (`0`, the default) keeps private per-session arenas
(`sessions × context`).

### Long context (~1M)

Native training length is 262144. Opt in to static YaRN ×4 for a 1_048_576
ceiling, then set `--context` and `--kv-pool-positions` up to that limit. The
shared pool still auto-fits; MTP draft KV grows with need instead of claiming
the full context at session create:

```sh
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 1 --context 1048576 \
  --kv-pool-positions 1048576 --rope-yarn 4
```

If fit refuses, levers are: lower `--kv-pool-positions` / `--context`, lower
`--host-reserve-gib`, fewer `--sessions`, disable MTP, or a lower-bpw GGUF.
Load logs `event=weight_stats` with bits-per-weight from the tensor table.

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG CLI and HTTP requests use the
[same image interface](../qwen3.8-27b/README.md#images). Image state participates
in prefill, decoding, verification, multi-turn reuse and disk cache identity.
The predictor embeds shifted text IDs; visual information comes from target
hidden states and mRoPE. Image snapshots require matching prompt attachment.

## Tools and artifacts

Model tests are in `tests/models/qwen38_flash_next`, focused microbenchmarks in
`tools/qwen-flash`. [Quality](QUALITY.md) summarizes qualification and focused checks.
Build a microbenchmark with
`nix develop -c tools/bench/build.sh tools/qwen-flash/projection_plans.hip`;
`dense_blaslt_sweep.hip` times hipBLASLt on the dense prefill shapes.
No historical logit dump is required. New retained result summaries belong in
`artifacts/`; generated traces stay in the ignored top-level `artifacts/` tree.
