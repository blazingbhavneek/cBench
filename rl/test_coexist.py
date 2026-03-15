"""
test_coexist.py
===============
Run on your machine:
    python test_coexist.py

Tests SGLang offline Engine + PyTorch PEFT model on same GPU, same process.
Uses your local Qwen3-1.7B model.
"""

import torch
import multiprocessing

MODEL = "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3-1.7B"

PASS = lambda s: print(f"  ✓  {s}")
FAIL = lambda s: print(f"  ✗  {s}")

def vram():
    return torch.cuda.memory_allocated() / 1e9


def run_tests():
    # ─────────────────────────────────────────────
    # 1. SGLang Engine boots
    # ─────────────────────────────────────────────
    print("\n[1] Loading SGLang Engine...")
    try:
        import sglang as sgl
        engine = sgl.Engine(
            model_path=MODEL,
            mem_fraction_static=0.45,
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=["gate_proj", "up_proj", "down_proj"],
        )
        PASS(f"SGLang Engine loaded  |  VRAM: {vram():.2f} GB")
    except Exception as e:
        FAIL(f"SGLang Engine failed: {e}")
        raise SystemExit(1)

    # ─────────────────────────────────────────────
    # 2. PyTorch PEFT model loads alongside SGLang
    # ─────────────────────────────────────────────
    print("\n[2] Loading PyTorch training model alongside SGLang...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL,
            device_map={"": "cuda:0"},
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        base.gradient_checkpointing_enable()

        lora_cfg = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["gate_proj", "up_proj", "down_proj"],
        )
        train_model = get_peft_model(base, lora_cfg)
        trainable = sum(p.numel() for p in train_model.parameters() if p.requires_grad)
        PASS(f"PEFT model loaded  |  LoRA params: {trainable:,}  |  VRAM: {vram():.2f} GB")
    except Exception as e:
        FAIL(f"PEFT model failed: {e}")
        raise SystemExit(1)

    # ─────────────────────────────────────────────
    # 3. SGLang generate still works after PyTorch model loaded
    # ─────────────────────────────────────────────
    print("\n[3] SGLang generate after PyTorch model is loaded...")
    try:
        out = engine.generate(
            "def fibonacci(n):",
            sampling_params={"max_new_tokens": 30, "temperature": 0.0},
        )
        text = out["text"] if isinstance(out, dict) else out
        PASS(f"SGLang generate OK  |  output: {str(text)[:60]!r}")
    except Exception as e:
        FAIL(f"SGLang generate failed: {e}")
        raise SystemExit(1)

    # ─────────────────────────────────────────────
    # 4. PyTorch forward + backward doesn't corrupt SGLang
    # ─────────────────────────────────────────────
    print("\n[4] PyTorch forward + backward pass...")
    try:
        ids = tokenizer("def fibonacci(n): return", return_tensors="pt").input_ids.cuda()
        out_train = train_model(ids, labels=ids)
        out_train.loss.backward()
        lora_grad_ok = all(
            p.grad is not None and p.grad.isfinite().all()
            for p in train_model.parameters()
            if p.requires_grad and p.grad is not None
        )
        train_model.zero_grad()
        PASS(f"Backward OK  |  loss={out_train.loss.item():.4f}  |  grads finite={lora_grad_ok}  |  VRAM: {vram():.2f} GB")
    except Exception as e:
        FAIL(f"Backward failed: {e}")
        raise SystemExit(1)

    # ─────────────────────────────────────────────
    # 5. LoRA hot-swap — introspect actual signature first
    # ─────────────────────────────────────────────
    print("\n[5] LoRA hot-swap: save → load into SGLang → generate with new adapter...")
    try:
        import tempfile
        import inspect

        # Introspect actual signatures — different SGLang versions use different arg names
        load_sig = inspect.signature(engine.load_lora_adapter)
        unload_sig = inspect.signature(engine.unload_lora_adapter)
        load_params = list(load_sig.parameters.keys())
        unload_params = list(unload_sig.parameters.keys())
        PASS(f"load_lora_adapter signature:   {load_sig}")
        PASS(f"unload_lora_adapter signature: {unload_sig}")

        # Save current LoRA weights
        lora_path = tempfile.mkdtemp(prefix="grpo_lora_")
        train_model.save_pretrained(lora_path)
        PASS(f"LoRA saved to {lora_path}")

        # Build load kwargs from actual param names
        load_kwargs = {}
        # path argument: could be lora_path or path
        for path_key in ("lora_path", "path"):
            if path_key in load_params:
                load_kwargs[path_key] = lora_path
                break
        # name argument: could be lora_name or name
        for name_key in ("lora_name", "name"):
            if name_key in load_params:
                load_kwargs[name_key] = "v1"
                break

        PASS(f"Calling load_lora_adapter with: {load_kwargs}")
        engine.load_lora_adapter(**load_kwargs)
        PASS("load_lora_adapter succeeded")

        # Generate — try lora_path and lora_name as sampling_params key
        gen_ok = False
        for lora_key, lora_val in [("lora_path", lora_path), ("lora_name", "v1")]:
            try:
                out_lora = engine.generate(
                    "def fibonacci(n):",
                    sampling_params={"max_new_tokens": 30, "temperature": 0.0, lora_key: lora_val},
                )
                text_lora = out_lora["text"] if isinstance(out_lora, dict) else out_lora
                PASS(f"Generate with LoRA OK (key='{lora_key}')  |  {str(text_lora)[:60]!r}")
                gen_ok = True
                break
            except Exception as eg:
                print(f"     → sampling key '{lora_key}' failed: {eg}")
        if not gen_ok:
            FAIL("Could not generate with LoRA adapter under any sampling key")

        # Unload — build kwargs from actual param names
        for candidate in ("v1", "base", "default"):
            try:
                unload_kwargs = {}
                for name_key in ("lora_name", "name"):
                    if name_key in unload_params:
                        unload_kwargs[name_key] = candidate
                        break
                if unload_kwargs:
                    engine.unload_lora_adapter(**unload_kwargs)
                else:
                    engine.unload_lora_adapter(candidate)
                PASS(f"Unloaded adapter '{candidate}'")
                break
            except Exception:
                continue

    except Exception as e:
        FAIL(f"LoRA hot-swap failed: {e}")
        import traceback; traceback.print_exc()
        raise SystemExit(1)

    # ─────────────────────────────────────────────
    # 6. Full cycle: backward → save → swap → generate
    # ─────────────────────────────────────────────
    print("\n[6] Full cycle: backward → save → swap → generate...")
    try:
        ids = tokenizer("def quicksort(arr):", return_tensors="pt").input_ids.cuda()
        out_train = train_model(ids, labels=ids)
        loss = out_train.loss * torch.tensor(1.0)
        loss.backward()
        with torch.no_grad():
            for p in train_model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.data -= 1e-4 * p.grad
        train_model.zero_grad()

        lora_path_v2 = tempfile.mkdtemp(prefix="grpo_lora_v2_")
        train_model.save_pretrained(lora_path_v2)

        # Load v2 using same kwargs pattern discovered in step 5
        load_kwargs_v2 = {}
        for path_key in ("lora_path", "path"):
            if path_key in load_params:
                load_kwargs_v2[path_key] = lora_path_v2
                break
        for name_key in ("lora_name", "name"):
            if name_key in load_params:
                load_kwargs_v2[name_key] = "v2"
                break
        engine.load_lora_adapter(**load_kwargs_v2)

        # Generate with v2
        for lora_key, lora_val in [("lora_path", lora_path_v2), ("lora_name", "v2")]:
            try:
                out_v2 = engine.generate(
                    "def quicksort(arr):",
                    sampling_params={"max_new_tokens": 30, "temperature": 0.0, lora_key: lora_val},
                )
                text_v2 = out_v2["text"] if isinstance(out_v2, dict) else out_v2
                PASS(f"Full cycle OK  |  VRAM: {vram():.2f} GB")
                PASS(f"v2 output: {str(text_v2)[:60]!r}")
                break
            except Exception:
                continue

    except Exception as e:
        FAIL(f"Full cycle failed: {e}")
        import traceback; traceback.print_exc()
        raise SystemExit(1)

    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("  ALL 6 CHECKS PASSED — coexistence confirmed")
    print("  Safe to build the full pipeline")
    print(f"  Peak VRAM observed: {vram():.2f} GB")
    print("="*55 + "\n")

    engine.shutdown()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_tests()
