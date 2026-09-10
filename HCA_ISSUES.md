# HCA Native Issues

## Fixed

- HCA layer detection was hardcoded to `il == 11` (qwen35.cpp:27). Now any
  layer with `blk.N.compress_k.weight` is detected.
- Partial checkpoint restore (speculative decode rollback) did not clear
  stale compressed K/V cells at the boundary the draft may have crossed.
  Now removes from `pos_max/ratio` onward in partial `state_read`.
- Full save/load did not serialize HCA compressed V. Now `dsv4_state_write/read_k_cache`
  accepts `write_v`/`read_v` flag, passed `true` for `kv_hca` only.
- `seq_add` did not shift compressed caches. Now forwards to `kv_csa`,
  `kv_hca`, `kv_lid` with position-scaled `p0`/`p1`.
- `llama_memory_hybrid_hca` recurrent allocation used `n_seq_max` for
  `rs_size`. Confirmed correct: matches existing Qwen hybrid path.

## Remaining

- `seq_div` does not forward to compressed caches. Compressed positions
  are derived from token positions via integer division, so dividing
  token positions by `d` does not produce a clean compressed-position
  mapping. Only raw KV is shifted. Low impact: `seq_div` is only used
  for RoPE position base scaling, not common in server workflows.

- HCA graph (`build_layer_attn_hca` in `qwen35.cpp`) does not consume
  `state_restore_*` / `state_snapshot_*` entries from the DSV4 plan.
  With `n_rs_seq=0` these vectors are empty, so current single-sequence
  and checkpoint-based speculative decode work. If `n_rs_seq > 0` were
  enabled for bounded rollback, the graph would need to wire these.
  Not needed for DFlash2 (uses checkpoint restore, not bounded rollback).

- `plan.n_kv` is padded to 256 via `GGML_PAD(plan.n_kv, 256u)` in the
  DSV4 comp plan. For HCA with ratio 128 and small context, this
  overallocates compressed cache cells (4 logical cells padded to 256).
  Correctness is fine (masked cells are zero-initialized); memory
  overhead is ~1 MiB for f16 K+V on one layer.

- Cache instrumentation logs to `stderr` under `LLAMA_DSV4_COMPRESS_DEBUG=1`.
  Remove or gate behind a more granular flag before production use.

- `llama-debug` example has `LLAMA_DEBUG_DECODE_CHUNKS` env var for
  incremental decode testing. Not part of upstream, remove before PR.

- `common/debug.cpp` has `LLAMA_DEBUG_TENSOR_DUMP_DIR` env var for
  dumping tensors as f32 files. Not part of upstream, remove before PR.
