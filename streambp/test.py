"""
StreamBP correctness + max-sequence-length test for:
  - Qwen3.5-4B  (~9 GB VRAM)
  - GPT-OSS-20B (~13 GB VRAM with MXFP4 kernels)

Prerequisites for GPT-OSS-20B MXFP4 (no dequant):
    pip install kernels
    # triton >= 3.4 is already satisfied by your torch 2.9 install

Run:
    python test_streambp.py
"""

import gc
import math
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Paths — edit if needed
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Qwen3.5-4B": {
        "path": "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3.5-4B",
        "model_type": "qwen3_5",
        "lora_target_modules": ["q_proj", "v_proj"],
        "vocab_size": None,
        # device_map="auto" for clean load; no CPU staging needed at 9GB
        "force_device_map": False,
    },
    "GPT-OSS-20B": {
        "path": "/media/blazingbhavneek/Common/Code/sglangServer/Infer/openai/gpt-oss-20b",
        # GptOssForCausalLM is a MoE model — NOT the old OpenAI GPT architecture.
        # Use generic StreamModel path; the actual class is auto-detected at runtime.
        "model_type": "auto",
        # Target the attention projections in dense layers (avoid MoE experts for LoRA)
        "lora_target_modules": ["q_proj", "v_proj", "k_proj"],
        "vocab_size": None,
        # Must load on CPU first: MXFP4 swizzle for MoE experts OOMs on GPU
        # when the first 14 dense layers already fill ~13 GB VRAM.
        "force_device_map": True,
    },
}

LORA_RANK = 16
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Gradient correctness test — keep small so model+grads fit in 15.5 GB
CORRECTNESS_SEQ_LEN = 256
CORRECTNESS_BATCH    = 1

# Max-seqlen sweep (doubles until OOM)
SWEEP_START    = 1024
SWEEP_MAX      = 131072
SWEEP_BATCH    = 1

# Tolerance for gradient comparison
REL_TOL = 0.10   # 10% — generous for bf16
ABS_TOL = 1e-5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30


def make_batch(seq_len: int, vocab_size: int, batch: int = 1):
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=DEVICE)
    attention_mask = torch.ones_like(input_ids)
    # mask last 10% as padding
    pad = max(1, int(seq_len * 0.1))
    attention_mask[:, -pad:] = 0
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_base(path: str, force_device_map: bool = False) -> AutoModelForCausalLM:
    """
    Load model cleanly.

    GPT-OSS-20B (MXFP4):
      - Requires `pip install kernels` to run in native MXFP4 (~13 GB VRAM).
      - Without `kernels`, transformers dequantizes to bf16 (~48 GB) — OOM on 16 GB.
      - On 16 GB cards without kernels: use Mxfp4Config(dequantize=True) which
        dequantizes lazily on CPU shard-by-shard, then we move to GPU.
        This avoids the GPU OOM during the swizzle conversion.

    Qwen3.5-4B: plain load to CPU then move to GPU.
    """
    import os
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    if force_device_map:
        # Try native MXFP4 first (needs `pip install kernels`)
        # If kernels are missing, fall back to CPU-side dequant via Mxfp4Config(dequantize=True)
        try:
            from transformers import Mxfp4Config
            print("  Loading GPT-OSS with Mxfp4Config(dequantize=True) on CPU to avoid GPU OOM during swizzle...")
            quant_cfg = Mxfp4Config(dequantize=True)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    quantization_config=quant_cfg,
                    dtype=torch.bfloat16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    quantization_config=quant_cfg,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )
            print("  Moving dequantized model to GPU...")
            model = model.to(DEVICE)
        except ImportError:
            # Old transformers without Mxfp4Config — try plain auto load
            print("  Mxfp4Config not available, trying plain auto load...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    path, dtype="auto", device_map="auto", low_cpu_mem_usage=True
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True
                )
    else:
        # Qwen: load to CPU, move to GPU (avoid accelerate meta-device offloading)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path, dtype="auto", low_cpu_mem_usage=True
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype="auto", low_cpu_mem_usage=True
            )
        model = model.to(DEVICE)

    return model.train()


def apply_lora(base_model, target_modules):
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(base_model, cfg)


def make_stream(peft_model, model_type: str):
    """Wrap a PEFT model in the appropriate StreamModel."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    # Auto-detect from the underlying model class name if not specified
    if model_type == "auto":
        # Unwrap PEFT to get the base model class name
        m = peft_model
        while hasattr(m, "base_model") or (hasattr(m, "model") and m.model is not m):
            if hasattr(m, "base_model"):
                m = m.base_model
            else:
                m = m.model
        cls_name = type(m).__name__.lower()
        if "qwen3_5" in cls_name or "qwen3.5" in cls_name:
            model_type = "qwen3_5"
        elif "openai" in cls_name or "gptmodel" in cls_name:
            model_type = "openai_gpt"
        else:
            # Generic: use base StreamModel (Llama/Qwen2 style .model.layers)
            model_type = "generic"
        print(f"  [auto] detected model class: {type(m).__name__} -> model_type={model_type}")

    if model_type == "qwen3_5":
        from stream_model import StreamModelForQwen3_5
        stream = StreamModelForQwen3_5(
            peft_model,
            gradient_accumulation_steps=1,
            logits_chunk_size=100,
            checkpoint_chunk_size=500,
            stream_checkpoint=True,
        )
    elif model_type == "openai_gpt":
        from stream_model import StreamModelForGPT
        stream = StreamModelForGPT(
            peft_model,
            gradient_accumulation_steps=1,
            logits_chunk_size=100,
            checkpoint_chunk_size=100,
            stream_checkpoint=True,
        )
    else:
        # Generic fallback: standard StreamModel expecting .model.model.layers
        from stream_model import StreamModel
        stream = StreamModel(
            peft_model,
            gradient_accumulation_steps=1,
            logits_chunk_size=100,
            checkpoint_chunk_size=500,
            stream_checkpoint=True,
        )
    stream.gradient_checkpointing_enable()
    return stream


def extract_lora_grads(peft_model):
    """Return a dict of {name: grad_tensor} for LoRA A/B matrices."""
    grads = {}
    for name, param in peft_model.named_parameters():
        if "lora_" in name and param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def rel_error(ref: torch.Tensor, cmp: torch.Tensor, eps: float = 1e-10) -> float:
    diff = torch.abs(ref.float() - cmp.float())
    return (diff / (torch.abs(ref.float()) + eps)).mean().item()


def abs_error(ref: torch.Tensor, cmp: torch.Tensor) -> float:
    return torch.abs(ref.float() - cmp.float()).mean().item()


# ---------------------------------------------------------------------------
# Test 1 — Gradient correctness
# ---------------------------------------------------------------------------

def test_correctness(name: str, cfg: dict):
    print(f"\n{'='*60}")
    print(f"  GRADIENT CORRECTNESS: {name}")
    print(f"  seq_len={CORRECTNESS_SEQ_LEN}, batch={CORRECTNESS_BATCH}")
    print(f"{'='*60}")

    torch.manual_seed(42)

    use_device_map = cfg.get("force_device_map", False)

    # ---------- Baseline (plain PEFT, grad-checkpoint, bf16) ----------
    base = load_base(cfg["path"], force_device_map=use_device_map)
    vocab_size = base.config.vocab_size
    cfg["vocab_size"] = vocab_size
    print(f"  Model class   : {type(base).__name__}")

    peft_base = apply_lora(base, cfg["lora_target_modules"])
    peft_base.gradient_checkpointing_enable()

    batch = make_batch(CORRECTNESS_SEQ_LEN, vocab_size, CORRECTNESS_BATCH)

    peft_base.zero_grad()
    out = peft_base(**batch, use_cache=False, return_dict=True)
    out.loss.backward()
    # Move grads to CPU immediately so GPU is free for the StreamBP model
    ref_grads = {k: v.cpu() for k, v in extract_lora_grads(peft_base).items()}

    print(f"  Baseline loss : {out.loss.item():.6f}")
    del peft_base, base, out
    free()
    # Hard reset allocator to reclaim fragmented pages before loading stream model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # ---------- StreamBP ----------
    torch.manual_seed(42)
    base2 = load_base(cfg["path"], force_device_map=use_device_map)
    peft_base2 = apply_lora(base2, cfg["lora_target_modules"])
    stream = make_stream(peft_base2, cfg["model_type"])

    stream.zero_grad()
    out2 = stream(**batch, use_cache=False, return_dict=True)
    # StreamBP accumulates grads during forward; loss.backward() is a no-op
    # but we still call it for compatibility (it will be near-zero)
    if out2.loss.requires_grad:
        out2.loss.backward()
    stream_grads = {k: v.cpu() for k, v in extract_lora_grads(peft_base2).items()}
    print(f"  StreamBP loss : {out2.loss.item():.6f}  (LoRA grads found: {len(stream_grads)})")
    del stream, peft_base2, base2, out2
    free()

    # ---------- Compare (both grad dicts are on CPU) ----------
    all_pass = True
    common_keys = set(ref_grads.keys()) & set(stream_grads.keys())
    missing_in_stream = set(ref_grads.keys()) - set(stream_grads.keys())
    extra_in_stream = set(stream_grads.keys()) - set(ref_grads.keys())

    if missing_in_stream:
        print(f"\n  ⚠ {len(missing_in_stream)} LoRA grads in baseline but NOT in stream:")
        for k in sorted(missing_in_stream)[:5]:
            print(f"    [MISSING] {k}")
        if len(missing_in_stream) > 5:
            print(f"    ... and {len(missing_in_stream)-5} more")
        all_pass = False

    print(f"\n  Comparing {len(common_keys)} common LoRA grad tensors:")
    for key in sorted(common_keys):
        ae = abs_error(ref_grads[key], stream_grads[key])
        re = rel_error(ref_grads[key], stream_grads[key])
        status = "✓" if re <= REL_TOL else "✗"
        print(f"  {status} {key:60s}  abs={ae:.2e}  rel={re:.4%}")
        if re > REL_TOL:
            all_pass = False

    print(f"\n  Result: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    return all_pass


# ---------------------------------------------------------------------------
# Test 2 — Maximum sequence length sweep
# ---------------------------------------------------------------------------

def test_max_seqlen(name: str, cfg: dict):
    print(f"\n{'='*60}")
    print(f"  MAX SEQ LEN SWEEP (LoRA + StreamBP): {name}")
    print(f"  VRAM budget: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    print(f"{'='*60}")

    use_device_map = cfg.get("force_device_map", False)

    vocab_size = cfg.get("vocab_size")
    if vocab_size is None:
        base_tmp = load_base(cfg["path"], force_device_map=use_device_map)
        vocab_size = base_tmp.config.vocab_size
        cfg["vocab_size"] = vocab_size
        del base_tmp
        free()

    last_ok = 0
    seq_len = SWEEP_START

    while seq_len <= SWEEP_MAX:
        torch.manual_seed(42)
        try:
            torch.cuda.reset_peak_memory_stats()

            base = load_base(cfg["path"], force_device_map=use_device_map)
            peft_m = apply_lora(base, cfg["lora_target_modules"])
            stream = make_stream(peft_m, cfg["model_type"])

            batch = make_batch(seq_len, vocab_size, SWEEP_BATCH)

            stream.zero_grad()
            out = stream(**batch, use_cache=False, return_dict=True)
            if out.loss.requires_grad:
                out.loss.backward()

            torch.cuda.synchronize()
            mem = peak_gb()
            print(f"  seq={seq_len:>7,}  peak_mem={mem:.2f} GB  loss={out.loss.item():.4f}  ✓")
            last_ok = seq_len

            del stream, peft_m, base, batch, out
            free()

        except torch.cuda.OutOfMemoryError:
            print(f"  seq={seq_len:>7,}  OOM ✗")
            try:
                del stream, peft_m, base, batch
            except Exception:
                pass
            free()
            break

        seq_len *= 2

    print(f"\n  Max seq length with LoRA + StreamBP on {name}: {last_ok:,} tokens")
    return last_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")

    results = {}
    for model_name, model_cfg in MODEL_CONFIGS.items():
        print(f"\n{'#'*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'#'*60}")

        try:
            passed = test_correctness(model_name, model_cfg)
        except Exception as e:
            print(f"  Correctness test ERROR: {e}")
            passed = False

        free()

        try:
            max_len = test_max_seqlen(model_name, model_cfg)
        except Exception as e:
            print(f"  Max seqlen test ERROR: {e}")
            max_len = 0

        free()
        results[model_name] = {"correctness": passed, "max_seqlen": max_len}

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for model_name, r in results.items():
        print(f"  {model_name:20s}  correctness={'PASS' if r['correctness'] else 'FAIL'}  max_seqlen={r['max_seqlen']:,}")
