# Memory Analysis: 4k → 132k Sequence Length

**Model:** GptOss 24-layer MoE, hidden=2880, 64 heads, 8 KV heads, 32 experts (4 active), vocab=201088. H200 80GB.

---

## Model Parameters (fixed, always in VRAM)

| Component | Params | Dtype | GB |
|---|---|---|---|
| embed_tokens [201088, 2880] | 579M | BF16 | 1.16 GB |
| lm_head [201088, 2880] | 579M | BF16 | 1.16 GB |
| Attention × 24 layers (q,k,v,o + biases) | ~36M/layer | BF16 | 1.85 GB |
| MoE experts × 24 (gate_up_blocks + down_blocks) | ~420M/layer | MXFP4 (U8) | ~10 GB |
| Router weights × 24 [32, 2880] | tiny | BF16 | ~0.1 GB |
| LayerNorms, misc | — | BF16 | ~0.1 GB |
| **Total weights** | | | **~14.4 GB** |

---

## Old: 4k Tokens — Why It Barely Fit

| Component | Memory |
|---|---|
| Model weights | 14.4 GB |
| LoRA grads (attn only) | ~1 GB |
| Graph: 12 suffix layers × [1, 4k, 2880] | ~0.3 GB |
| Graph: attn Q/K/V/O × 12 layers | ~0.8 GB |
| Frozen prefix hidden (CPU offload) | ~0 GB |
| Logit spike [4k, 201088] fp32 | ~3.2 GB |
| 8-bit AdamW states | ~1 GB |
| **Total peak** | **~21 GB ✓** |

Fits in 80GB because the sequence is short. Every component scales linearly with T so even small increases hurt badly.

---

## Naive 132k — Why It OOMs

| Component | Memory |
|---|---|
| Model weights | 14.4 GB |
| LoRA grads | ~1 GB |
| Graph: 12 suffix layers × [1, 132k, 2880] | **9.6 GB** |
| Graph: Q [132k, 4096] × 12 layers | **12.8 GB** |
| Graph: K/V/O × 12 layers | **5.1 GB** |
| Logit spike [132k, 201088] fp32 | **~105 GB** |
| 8-bit AdamW states | ~1 GB |
| **Total peak** | **~149 GB 💀** |

Logit spike alone exceeds the H200. Graph is 32× larger than at 4k. Would need ~2× H200s just for naive scaling.

---

## Optimizations

### 1. Chunked lm_head — already in your code ✓

Never materializes `[T × 201088]`. Computes `[chunk × 201088]` at a time (chunk=100), accumulates gradients, frees immediately.

**Saving: 105 GB → 0.16 GB peak logits. Biggest single win.**

---

### 2. Gradient Checkpointing on Suffix Layers

Wrap each of the 12 suffix layers in `checkpoint()`. Hidden states are discarded after forward, recomputed on demand during backward. Only one layer's hidden state in the graph at a time instead of all 12 simultaneously.

**Saving: 9.6 GB → 0.8 GB. 12× reduction.**

---

### 3. Chunked Q Attention — Full Attention Layers Only

During checkpointed reforward: compute K/V once for the full sequence under `no_grad`, then loop Q in 4096-token chunks. Each chunk attends to `K[:end]` causally. 

- Sliding window layers (window=128) left completely vanilla — they're already tiny
- No wrapping the whole model, minimal code change
- K/V cached, no extra graph nodes

**Saving: 2.1 GB spike × 12 → ~0.2 GB peak per layer. 10× per full-attention layer.**

---

### 4. MoE Expert LoRA via `target_parameters`

Experts are `nn.Parameter` not `nn.Linear` — need `target_parameters` not `target_modules` in LoraConfig. Target top 12 layers only with explicit layer indices:

```python
target_params = []
for i in range(split_layer, num_layers):
    target_params.append(f"model.layers.{i}.mlp.experts.gate_up_proj")
    target_params.append(f"model.layers.{i}.mlp.experts.down_proj")
```

MoE sparsity means only ~16.5k tokens per expert on average at 132k tokens, so peak per-expert intermediate is ~190 MB not 1.5 GB. Checkpoint wrapper handles gradient storage.

**Effect: Small grad overhead, large training quality gain. Expert intermediates safe due to MoE sparsity.**

---

### 5. Immediate Backward Per Sequence

Call `.backward()` after each generation `g`, accumulate gradients, then optimizer step after all G. Prevents G × graph sitting in memory simultaneously.

**Saving: G=8 graph copies → 1 copy at a time. 8× reduction for multi-generation batches.**

---

## Final Memory Budget at 132k Tokens

| Component | Before | After |
|---|---|---|
| Model weights | 14.4 GB | 14.4 GB |
| LoRA grads (attn + MoE top layers) | ~1 GB | ~2 GB |
| Graph hidden states (12 layers) | 9.6 GB | 0.8 GB |
| Attention Q/K/V/O in graph | 17.9 GB | 0.2 GB |
| MoE expert intermediates | 1.5 GB | ~0.2 GB |
| Logit spike | 105 GB | 0.16 GB |
| 8-bit AdamW states | ~1 GB | ~1 GB |
| Frozen prefix (CPU offload) | 0 GB | 0 GB |
| **Total peak** | **~149 GB 💀** | **~19 GB ✓** |

Fits in **~19 GB** — well within H200 80GB. Leaves ~60 GB headroom for larger batch accumulation or longer sequences.
