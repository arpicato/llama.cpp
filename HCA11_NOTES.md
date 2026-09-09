# Qwen HCA11 Native Runtime - Implementation Notes

## Goal
Patch llama.cpp commit 91f6a6cf to load and run Qwen3.8-27B with HCA11
compressed attention instead of normal full KV attention.

## Python HCA Semantics (attention/hca.py)

### Cache Layout (QwenHCACache)
- `local_k`: [batch, num_kv_heads, local_window-1, head_dim]  (127 tokens)
- `local_v`: same shape
- `compressed_k`: [batch, num_kv_heads, n_blocks, head_dim]  (1 per 128 tokens)
- `compressed_v`: same shape
- `buffer_k/v/gate`: [batch, remainder, num_kv_heads, head_dim]  (< 128 tokens)
- `entry_count`: number of completed compressed blocks
- `tokens_seen`: total tokens processed

### Compression (_compress method)
1. Project hidden through compress_k, compress_v, compress_gate (all Linear)
2. Concatenate with buffer (leftover from previous calls)
3. Split into complete 128-token blocks
4. For each block: weights = softmax(gate + position_bias, dim=2)
5. compressed_k = sum(k * weights, dim=2)  -> [batch, num_kv_heads, n_blocks, head_dim]
6. compressed_v = sum(v * weights, dim=2)
7. Apply k_norm to compressed_k
8. Apply RoPE to compressed_k at block positions (block_idx * 128)
9. Store remainder as new buffer

### Attention (forward method)
1. Standard Q/K/V projection + q_norm + k_norm + RoPE (on raw tokens)
2. Concatenate local K/V with past local K/V, keep last 127
3. Concatenate compressed K/V with past compressed K/V
4. Build mask: local (window=128, causal) + compressed (block_idx < query_pos // 128)
5. Repeat K/V for GQA
6. SDPA with combined K/V and mask
7. Output gate (sigmoid) * o_proj

### Key Differences from DSV4 HCA
- DSV4 stores K-only; Qwen stores both K and V compressed
- DSV4 uses score-based softmax; Qwen uses gate+position_bias
- DSV4 has separate raw SWA cache; Qwen uses local window of 128 (127 past + 1 current)
- DSV4 applies RoPE with offset; Qwen applies standard RoPE at block positions
- DSV4 uses attn_sinks; Qwen does not

## Implementation Plan

### 1. Tensor Registration (llama-arch.h/cpp)
Add new LLM_TENSOR enum values:
- LLM_TENSOR_ATTN_HCA_COMP_K    -> "blk.%d.hca_compress_k.weight"
- LLM_TENSOR_ATTN_HCA_COMP_V    -> "blk.%d.hca_compress_v.weight"
- LLM_TENSOR_ATTN_HCA_COMP_GATE -> "blk.%d.hca_compress_gate.weight"
- LLM_TENSOR_ATTN_HCA_POS_BIAS  -> "blk.%d.hca_position_bias"

### 2. Layer Struct (llama-model.h)
Add to llama_model_layer:
- struct ggml_tensor * hca_compress_k = nullptr;
- struct ggml_tensor * hca_compress_v = nullptr;
- struct ggml_tensor * hca_compress_gate = nullptr;
- struct ggml_tensor * hca_position_bias = nullptr;

### 3. Tensor Loading (qwen35.cpp load_arch_tensors)
In the non-recurrent (full attention) branch, after existing tensors:
- Load HCA tensors with TENSOR_NOT_REQUIRED flag
- If hca_compress_k != nullptr, mark layer as HCA

### 4. Graph Dispatch (qwen35.cpp graph::operator())
In the full-attention branch, check if HCA tensors exist:
- If yes: call build_layer_attn_hca()
- If no: call build_layer_attn() as before

### 5. HCA Attention Graph (new method in qwen35 graph)
Implement build_layer_attn_hca() that:
- Computes standard Q/K/V from q_proj/k_proj/v_proj
- Applies q_norm, k_norm, RoPE to Q/K
- Computes compressed K/V from compress_k/v/gate + position_bias
- Uses a custom cache context for compressed + local storage
- Concatenates local K/V with compressed K/V
- Runs SDPA with combined mask
- Applies output gate (sigmoid) and o_proj

### 6. Cache (new or adapted)
For initial parity test: use a simple approach that stores
compressed K/V in a separate buffer and local K/V in the standard KV cache
with a small window. This avoids implementing a full custom memory type.

### 7. Parity Test
- Load splice GGUF with HCA11 tensors
- Forward a known input through layer 11
- Compare output h12 against Python reference
- Verify cache bytes match expected compressed size
