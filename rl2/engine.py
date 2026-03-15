# =============================================================================
# engine.py — SGLang offline Engine wrapper
# =============================================================================

import inspect
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
import torch

from config import (
    LORA_RANK,
    LORA_TARGET,
    MAX_COMPLETION_TOKENS,
    MAX_SEQ_LEN,
    MODEL_PATH,
    NUM_GENERATIONS,
    REASONING_EFFORT,
    SGLANG_MEM_FRAC,
    TEMPERATURE,
)
from logits import SparseLogitBuffer, build_sparse_buffer, buffer_to_cpu

log = logging.getLogger("grpo.engine")


class SGLangEngine:

    def __init__(self):
        import sglang as sgl
        from transformers import AutoTokenizer

        log.info(f"Booting SGLang Engine  model={MODEL_PATH}  mem_frac={SGLANG_MEM_FRAC}")
        self.engine = sgl.Engine(
            model_path=MODEL_PATH,
            mem_fraction_static=SGLANG_MEM_FRAC,
            enable_lora=True,
            max_lora_rank=LORA_RANK,
            lora_target_modules=LORA_TARGET,
            context_length=MAX_SEQ_LEN,
            log_level="error",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self._active_lora_path: Optional[str] = None  # set by swap_lora() after probe succeeds

        # Introspect load/unload signatures once — arg names vary across SGLang versions
        load_sig   = inspect.signature(self.engine.load_lora_adapter)
        unload_sig = inspect.signature(self.engine.unload_lora_adapter)
        self._load_params   = list(load_sig.parameters.keys())
        self._unload_params = list(unload_sig.parameters.keys())
        log.info(f"load_lora_adapter params:   {self._load_params}")
        log.info(f"unload_lora_adapter params: {self._unload_params}")

        log.info("SGLang Engine ready")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _sampling_params(self, base_params: dict) -> "SamplingParams":
        """Build SamplingParams, stripping any keys this SGLang version rejects."""
        from sglang.srt.sampling.sampling_params import SamplingParams
        clean = {k: v for k, v in base_params.items() if k != "lora_name"}
        return SamplingParams(**clean)

    def _apply_chat_template(self, messages: List[Dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_generate_kwargs(self, sampling_params) -> dict:
        """
        Build kwargs for engine.generate().
        Includes lora_path only if swap_lora() previously verified it works.
        This is the single place lora_path is injected — no manual uncommenting needed.
        """
        kwargs = dict(sampling_params=sampling_params)
        if self._active_lora_path is not None:
            kwargs["lora_path"] = self._active_lora_path
        return kwargs

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_with_logits(self, messages, return_logits=True):
        prompt_text     = self._apply_chat_template(messages)
        sampling_params = {"max_new_tokens": MAX_COMPLETION_TOKENS, "temperature": TEMPERATURE, "n": 1}
        generate_kwargs = self._build_generate_kwargs(sampling_params)
        out = self.engine.generate(prompt_text, **generate_kwargs)

        if isinstance(out, list): out = out[0]
        text      = out.get("text", "") if isinstance(out, dict) else str(out)
        token_ids = out.get("token_ids", []) if isinstance(out, dict) else []
        if not token_ids and text:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return token_ids, text

    def _build_generate_kwargs(self, sampling_params) -> dict:
        """Build kwargs for engine.generate(). sampling_params must be a plain dict."""
        kwargs = dict(sampling_params=sampling_params)
        if self._active_lora_path is not None:
            kwargs["lora_path"] = self._active_lora_path
        return kwargs

    def sample_n(
        self,
        messages: List[Dict],
        n:        int = NUM_GENERATIONS,
    ) -> List[Dict]:
        prompt_text     = self._apply_chat_template(messages)
        sampling_params = {                         # plain dict — SGLang requires this
            "max_new_tokens": MAX_COMPLETION_TOKENS,
            "temperature":    TEMPERATURE,
            "n":              n,
        }
        generate_kwargs = self._build_generate_kwargs(sampling_params)

        done = threading.Event()

        with tqdm(
            total=n,
            desc="    sglang generate",
            unit="gen",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
        ) as gbar:
            gbar.set_postfix_str("waiting for SGLang…", refresh=True)

            def _spinner():
                elapsed = 0
                while not done.wait(timeout=0.5):
                    elapsed += 0.5
                    gbar.set_postfix_str(f"SGLang running… {elapsed:.0f}s", refresh=True)
                gbar.update(n)
                gbar.set_postfix_str("done", refresh=True)

            t = threading.Thread(target=_spinner, daemon=True)
            t.start()

            try:
                outs = self.engine.generate(prompt_text, **generate_kwargs)
            finally:
                done.set()
                t.join()

        if not isinstance(outs, list):
            outs = [outs]

        results = []
        for item in outs:
            if isinstance(item, dict):
                text      = item.get("text", "")
                token_ids = item.get("token_ids", [])
            else:
                text      = str(item)
                token_ids = []
            if not token_ids and text:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            results.append({"text": text, "token_ids": token_ids})

        return results

    def swap_lora(self, lora_path: str):
        probe_params = {"max_new_tokens": 1, "temperature": 0.0}
        try:
            self.engine.generate("test", sampling_params=probe_params, lora_path=lora_path)
            self._active_lora_path = lora_path
            log.info(f"LoRA hot-swap verified and active → {lora_path}")
        except TypeError as e:
            if "lora_path" in str(e):
                self._active_lora_path = None
                log.warning(f"SGLang does not support lora_path in generate() ({e}) — base weights only.")
            else:
                raise
        except Exception as e:
            self._active_lora_path = None
            log.warning(f"LoRA hot-swap probe failed ({type(e).__name__}: {e}) — staying on previous adapter")


    def shutdown(self):
        try:
            self.engine.shutdown()
        except Exception:
            pass
