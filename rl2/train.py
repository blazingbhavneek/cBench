# =============================================================================
# train.py — training model, GRPO loss, optimizer
# No TRL. Raw PyTorch + PEFT.
# =============================================================================

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    CLIP_RATIO,
    GRAD_ACCUM_STEPS,
    KL_COEFF,
    LORA_ALPHA,
    LORA_LAYERS_FRAC,
    LORA_RANK,
    LORA_TARGET,
    LR,
    LR_SCHEDULER,
    MODEL_PATH,
    OPTIMIZER,
    SAVE_STEPS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from logits import chunked_token_entropy, selective_log_softmax

log = logging.getLogger("grpo.train")


# =============================================================================
# Model loading
# =============================================================================

def _get_lora_cutoff_index(model, frac: float) -> int:
    """Return the integer layer index where LoRA begins."""
    if frac <= 0.0:
        return 0
    layer_indices = set()
    for name, _ in model.named_modules():
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i+1].isdigit():
                layer_indices.add(int(parts[i+1]))
    if not layer_indices:
        return 0
    max_layer = max(layer_indices)
    return int(math.floor(max_layer * (1.0 - frac)))


def _get_lora_target_layers(model, frac: float) -> List[str]:
    """
    Return target module names for only the top `frac` of transformer layers.
    frac=0.5 means LoRA on the top 50% of layers only.
    frac=0.0 means all layers.
    """
    if frac <= 0.0:
        return LORA_TARGET

    layer_names = []
    for name, _ in model.named_modules():
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i+1].isdigit():
                layer_names.append(int(parts[i+1]))

    if not layer_names:
        log.warning("Could not detect layer indices — applying LoRA to all layers")
        return LORA_TARGET

    max_layer = max(layer_names)
    cutoff    = int(math.floor(max_layer * (1.0 - frac)))

    targets = []
    for name, _ in model.named_modules():
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i+1].isdigit():
                layer_idx = int(parts[i+1])
                if layer_idx >= cutoff:
                    leaf = parts[-1]
                    if leaf in LORA_TARGET:
                        targets.append(name)
    log.info(f"LoRA target layers: {cutoff}..{max_layer} ({len(targets)} modules)")
    return targets if targets else LORA_TARGET

def _patch_rms_norms_bf16(model: torch.nn.Module):
    """
    Replace Qwen3RMSNorm.forward to stay in bf16 instead of upcasting to float32.
    
    The stock implementation does:
        hidden_states = hidden_states.to(torch.float32)   # <── OOM here
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        ...
        return self.weight * hidden_states.to(input_dtype)
    
    For T=3000, H=5120: that float32 copy costs 61 MB per norm call.
    With q_norm + k_norm × N_layers this blows the budget on a 16GB card.
    
    bf16 RMSNorm is numerically fine — this is how vLLM/SGLang run it.
    """
    import torch.nn as nn
    patched = 0
    for module in model.modules():
        # Target any RMSNorm-shaped module (weight + variance_epsilon, no bias)
        if (
            hasattr(module, "weight")
            and hasattr(module, "variance_epsilon")
            and not hasattr(module, "bias")
            and isinstance(module.weight, torch.Tensor)
            and module.weight.dim() == 1
        ):
            def _make_bf16_forward(mod):
                def _forward(hidden_states: torch.Tensor) -> torch.Tensor:
                    # Compute variance in the input dtype — bf16 is fine here
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(
                        variance + mod.variance_epsilon
                    )
                    return mod.weight.to(hidden_states.dtype) * hidden_states
                return _forward

            module.forward = _make_bf16_forward(module)
            patched += 1

    log.info(f"Patched {patched} RMSNorm modules to bf16 (eliminates float32 upcast)")

def load_training_model() -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load 4-bit frozen base + bf16 LoRA adapters on MLP only (top N layers).
    Returns (peft_model, tokenizer).
    """
    log.info(f"Loading training model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": "cuda:0"},
        dtype=torch.bfloat16,
        use_cache=False,
    )
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base.enable_input_require_grads()

    # Determine which layers to target
    target_modules = _get_lora_target_layers(base, LORA_LAYERS_FRAC)

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(base, lora_cfg)
    _patch_rms_norms_bf16(model)

    # Record where LoRA starts — used for split forward
    split_layer = _get_lora_cutoff_index(base, LORA_LAYERS_FRAC)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(
        f"LoRA trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.3f}%)  "
        f"split_layer={split_layer}  "
        f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB"
    )
    return model, tokenizer, split_layer


# =============================================================================
# Reference model logprobs (frozen, computed once per prompt)
# =============================================================================

@torch.inference_mode()
def compute_ref_logprobs(
    model:       torch.nn.Module,
    input_ids:   torch.Tensor,    # (1, T)
    completion_ids: torch.Tensor, # (G, T_c)
) -> torch.Tensor:
    """
    Compute reference model log P(completion tokens) using the frozen base.
    Called with torch.inference_mode() — no gradients stored.

    Uses selective_log_softmax (chunked, memory efficient).
    Returns ref_logprobs: (G, T_c)
    """
    G, T_c = completion_ids.shape
    # Expand prompt for each completion
    prompt_len = input_ids.shape[1]
    full_ids = torch.cat([
        input_ids.expand(G, -1),
        completion_ids,
    ], dim=1)  # (G, T_prompt + T_c)

    outputs = model(
        input_ids=full_ids,
        use_cache=False,
    )
    # Logits for completion positions only
    logits = outputs.logits[:, prompt_len - 1 : prompt_len - 1 + T_c, :]  # (G, T_c, V)
    return selective_log_softmax(logits, completion_ids)  # (G, T_c)


# =============================================================================
# GRPO advantage computation
# =============================================================================

def compute_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Group-normalize rewards within a problem's completion group.
    rewards: (G,) — one per completion
    Returns advantages: (G,) — zero-mean, unit-variance within group
    """
    mean = rewards.mean()
    std  = rewards.std() + eps
    return (rewards - mean) / std


# =============================================================================
# Split forward — frozen prefix (no grad) + LoRA suffix (with grad)
# =============================================================================

def _get_transformer_layers(model: torch.nn.Module):
    """
    Return the list of decoder layers from a PEFT-wrapped Qwen3/LLaMA model.
    Handles both PeftModel wrapper and raw model.
    """
    # Unwrap PEFT if needed
    base = model.base_model.model if hasattr(model, 'base_model') else model
    # Qwen3 / LLaMA structure: model.model.layers
    if hasattr(base, 'model') and hasattr(base.model, 'layers'):
        return base.model.layers
    # Fallback
    for name, mod in base.named_modules():
        if name == 'model.layers' or name == 'layers':
            return mod
    raise RuntimeError("Cannot find transformer layers in model")


def _get_rope(inner, hidden: torch.Tensor, position_ids: torch.Tensor):
    """
    Compute RoPE by calling the full model forward on a single dummy token
    and extracting the cos/sin from the rotary_emb module directly using
    the correct head_dim from the model config.
    """
    cfg      = inner.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    T        = position_ids.shape[1]
    dummy    = hidden.new_zeros(1, T, head_dim)
    cos, sin = inner.rotary_emb(dummy, position_ids)
    return cos, sin


def _run_frozen_prefix(
    model:       torch.nn.Module,
    input_ids:   torch.Tensor,      # (1, T)
    split_layer: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run frozen prefix up to split_layer.
    Uses a forward hook to capture only the one hidden state we need,
    instead of output_hidden_states=True which stores all N_layers tensors.
    Saves ~(split_layer × T × H × 2 bytes) GPU memory per call.
    """
    base  = model.base_model.model if hasattr(model, 'base_model') else model
    inner = base.model

    with torch.inference_mode():
        T            = input_ids.shape[1]
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)

        if split_layer == 0:
            # No frozen layers — just embed and hand off
            hidden = inner.embed_tokens(input_ids).cpu()
            return hidden, position_ids.cpu()

        captured: Dict[str, torch.Tensor] = {}

        def _hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["h"] = hs.cpu()   # move to CPU immediately, free GPU copy

        # Hook on last frozen layer — its output = hidden_states[split_layer]
        # in the old output_hidden_states indexing
        handle = inner.layers[split_layer - 1].register_forward_hook(_hook)
        try:
            inner(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
                # No output_hidden_states — only ONE tensor ever lives on GPU
            )
        finally:
            handle.remove()

    return captured["h"], position_ids.cpu()


def _run_lora_suffix(
    model:            torch.nn.Module,
    hidden_cpu:       torch.Tensor,
    position_ids_cpu: torch.Tensor,
    split_layer:      int,
    completion_ids:   torch.Tensor,
    T_c:              int,
    device:           torch.device,
    gen_idx:          int = 0,          # ← new: for tqdm label
    total_gens:       int = 1,          # ← new
) -> torch.Tensor:
    base   = model.base_model.model if hasattr(model, 'base_model') else model
    inner  = base.model

    hidden       = hidden_cpu.to(device)
    position_ids = position_ids_cpu.to(device)
    T            = hidden.shape[1]

    with torch.no_grad():
        head_dim = getattr(inner.config, "head_dim",
                           inner.config.hidden_size // inner.config.num_attention_heads)
        dummy_cos_input     = hidden.new_zeros(1, T, head_dim)
        position_embeddings = inner.rotary_emb(dummy_cos_input, position_ids)

    suffix_layers = list(inner.layers[split_layer:])
    N             = len(suffix_layers)

    # ── Per-generation layer progress bar ────────────────────────────────
    with tqdm(
        total=N,
        desc=f"    backward gen {gen_idx+1}/{total_gens}",
        unit="layer",
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} layers [{elapsed}]",
    ) as lbar:

        for layer_i, layer in enumerate(suffix_layers):
            lbar.set_postfix_str(f"attn", refresh=True)

            with torch.no_grad():
                residual_attn = hidden
                normed_attn   = layer.input_layernorm(hidden)
                attn_out, _   = layer.self_attn(
                    normed_attn,
                    attention_mask=None,                   # ← fix
                    position_embeddings=position_embeddings,
                )
                hidden = residual_attn + attn_out.detach()

            lbar.set_postfix_str(f"mlp", refresh=True)

            residual_mlp = hidden
            normed_mlp   = layer.post_attention_layernorm(hidden)
            mlp_out      = layer.mlp(normed_mlp)
            hidden       = residual_mlp + mlp_out

            del normed_mlp, mlp_out, residual_mlp
            del normed_attn, attn_out, residual_attn

            lbar.update(1)

    hidden_last    = inner.norm(hidden)
    hidden_comp    = hidden_last[0, -T_c:]
    lm_head        = base.lm_head
    logits         = lm_head(hidden_comp)
    lse            = _chunked_lse(logits)
    token_logprobs = logits[torch.arange(T_c, device=device), completion_ids] - lse

    del logits, lse, hidden, hidden_last, hidden_comp
    return token_logprobs

# =============================================================================
# GRPO loss
# =============================================================================

def grpo_loss(
    model:           torch.nn.Module,
    split_layer:     int,
    prompt_ids:      torch.Tensor,           # (1, T_p)
    completion_ids:  torch.Tensor,           # (G, T_c)
    advantages:      torch.Tensor,           # (G,)
    ref_logprobs:    Optional[torch.Tensor], # (G, T_c) or None
    completion_mask: torch.Tensor,           # (G, T_c)
    device:          torch.device,
) -> Tuple[torch.Tensor, Dict]:
    """
    GRPO loss using split forward:
      - Frozen prefix (embed + attention layers[0..split]) runs in inference_mode
        Hidden states stored on CPU — no GPU activation memory for this part
      - LoRA suffix (MLP layers[split..N] + lm_head) runs with grad
        Only these activations stored for backward — no O(T²) attention storage

    Peak GPU memory: (1, T, H) hidden + LoRA layer activations + (T_c, V) logits
    No T² attention matrix ever lives on GPU during backward.
    """
    G, T_c = completion_ids.shape
    T_p    = prompt_ids.shape[1]

    seq_logprobs = []

    for g in range(G):
        full_ids = torch.cat([prompt_ids, completion_ids[g:g+1]], dim=1)

        hidden_cpu, pos_cpu = _run_frozen_prefix(model, full_ids, split_layer)
        torch.cuda.empty_cache()

        token_logprobs = _run_lora_suffix(
            model, hidden_cpu, pos_cpu,
            split_layer, completion_ids[g], T_c, device,
            gen_idx=g, total_gens=G,           # ← add
        )

        masked_lp = token_logprobs * completion_mask[g]
        seq_logprobs.append(masked_lp.sum())

        del hidden_cpu, pos_cpu, token_logprobs, masked_lp
        torch.cuda.empty_cache()

    seq_logprobs_t = torch.stack(seq_logprobs)              # (G,) all with grad_fn

    pg_loss  = -(advantages * seq_logprobs_t).mean()

    kl_loss = torch.tensor(0.0, device=device)
    if KL_COEFF > 0.0 and ref_logprobs is not None:
        kl_per_seq = (seq_logprobs_t.detach() - ref_logprobs.sum(dim=-1)) * completion_mask.sum(dim=-1)
        kl_loss    = kl_per_seq.mean()

    total_loss = pg_loss + KL_COEFF * kl_loss

    stats = {
        "pg_loss":        pg_loss.item(),
        "kl_loss":        kl_loss.item(),
        "mean_advantage": advantages.mean().item(),
        "mean_seq_logp":  seq_logprobs_t.detach().mean().item(),
    }
    return total_loss, stats


def _chunked_lse(logits: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """
    Differentiable logsumexp over vocab dimension in chunks.
    logits: (T, V)  → returns (T,)
    Avoids materializing (T, V) float32 — peak: (T, chunk) float32.
    Uses the log-sum-exp identity: LSE(x) = m + log(sum(exp(x - m)))
    where m = max(x), computed in a first pass over chunks.
    """
    T, V = logits.shape
    device = logits.device

    # Pass 1: find running max per row (no grad needed for stability anchor)
    with torch.no_grad():
        m = torch.full((T,), float("-inf"), device=device, dtype=torch.float32)
        for s in range(0, V, chunk):
            zc = logits[:, s:s+chunk].float()
            m  = torch.maximum(m, zc.max(dim=-1).values)

    # Pass 2: accumulate sum(exp(z - m)) WITH grad
    sum_exp = torch.zeros(T, device=device, dtype=torch.float32)
    for s in range(0, V, chunk):
        zc       = logits[:, s:s+chunk].float()
        sum_exp  = sum_exp + torch.exp(zc - m.unsqueeze(-1)).sum(dim=-1)

    lse = (m + torch.log(sum_exp)).to(logits.dtype)
    return lse


# =============================================================================
# Trainer class
# =============================================================================

class GRPOTrainer:
    def __init__(self, total_steps: int):
        self.model, self.tokenizer, self.split_layer = load_training_model()
        self.device = torch.device("cuda:0")
        self.step   = 0
        self.total_steps = total_steps

        # Optimizer — 8-bit Adam if available, else standard AdamW
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=LR, weight_decay=WEIGHT_DECAY,
            )
            log.info("Using 8-bit AdamW (bitsandbytes)")
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=LR, weight_decay=WEIGHT_DECAY,
            )
            log.warning("bitsandbytes not found — using fp32 AdamW")

        # LR scheduler
        warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=LR * 0.1,
        )
        self._warmup_steps = warmup_steps

        self._accum_loss  = torch.tensor(0.0, device=self.device)
        self._accum_stats: List[Dict] = []
        self._accum_count = 0

        log.info(f"Trainer ready  steps={total_steps}  warmup={warmup_steps}")

    @torch.inference_mode()
    def compute_ref_logprobs(
        self,
        prompt_ids:     torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Reference model forward (no grad). Uses base model weights directly."""
        return compute_ref_logprobs(self.model, prompt_ids, completion_ids)

    def accumulate(
        self,
        prompt_ids:      torch.Tensor,          # (1, T_p)
        completion_ids:  torch.Tensor,          # (G, T_c)
        advantages:      torch.Tensor,          # (G,)
        ref_logprobs:    Optional[torch.Tensor],# (G, T_c) or None
        completion_mask: torch.Tensor,          # (G, T_c)
    ) -> Dict:
        """
        Compute GRPO loss and accumulate gradients.
        Runs a real differentiable forward pass — one completion at a time.
        Does NOT step the optimizer — call optimizer_step() after BATCH_SIZE problems.
        """
        loss, stats = grpo_loss(
            model=self.model,
            split_layer=self.split_layer,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            advantages=advantages,
            ref_logprobs=ref_logprobs,
            completion_mask=completion_mask,
            device=self.device,
        )

        scaled_loss = loss / GRAD_ACCUM_STEPS
        scaled_loss.backward()

        self._accum_loss  = self._accum_loss + loss.detach()
        self._accum_stats.append(stats)
        self._accum_count += 1
        return stats

    def optimizer_step(self) -> Dict:
        """
        Clip gradients, step optimizer, update LR, zero grads.
        Call after accumulating BATCH_SIZE problems.
        Returns aggregated stats for logging.
        """
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()

        # Warmup: linear LR ramp
        if self.step < self._warmup_steps:
            lr_scale = (self.step + 1) / self._warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = LR * lr_scale
        else:
            self.scheduler.step()

        self.optimizer.zero_grad()
        self.step += 1

        # Aggregate stats
        agg = {
            "step":       self.step,
            "lr":         self.optimizer.param_groups[0]["lr"],
            "mean_loss":  self._accum_loss.item() / max(self._accum_count, 1),
            "pg_loss":    sum(s["pg_loss"]  for s in self._accum_stats) / max(len(self._accum_stats), 1),
            "kl_loss":    sum(s["kl_loss"]  for s in self._accum_stats) / max(len(self._accum_stats), 1),
            "mean_adv":   sum(s["mean_advantage"] for s in self._accum_stats) / max(len(self._accum_stats), 1),
        }
        self._accum_loss  = torch.tensor(0.0, device=self.device)
        self._accum_stats = []
        self._accum_count = 0
        return agg

    def save_lora(self, path: str) -> str:
        """Save LoRA adapter to disk. Returns path for SGLang hot-swap."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        log.info(f"Saved LoRA adapter → {path}")
        return path

    def save_checkpoint(self, tag: str):
        """Save a named checkpoint."""
        from config import OUTPUT_DIR
        path = str(Path(OUTPUT_DIR) / tag)
        self.save_lora(path)
        self.tokenizer.save_pretrained(path)
        return path
