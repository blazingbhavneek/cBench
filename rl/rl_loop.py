"""
RL Loop — GPT-OSS-20B + Unsloth QLoRA + SGLang + Custom PPO
=============================================================
Two processes, one GPU, clean split:

  SGLang server (separate process)      This process
  ────────────────────────────────      ─────────────────────────────────
  gpt-oss-20b full precision            gpt-oss-20b 4-bit QLoRA (Unsloth)
  ~13 GB VRAM                           ~14 GB VRAM
  Fast generation, reasoning=high       Reward + PPO loss + backward
  KV cache fills remaining VRAM         Gradient checkpointing (Unsloth Flex)
  async HTTP (aiohttp)                  ProcessPoolExecutor (gcc verifier)

Total model VRAM: ~27 GB
Remaining ~114 GB: SGLang KV cache (large) + Unsloth activation memory (small,
gradient checkpointing keeps it bounded)

Why Unsloth for backprop only:
  - FA3 breaks gpt-oss backward pass silently → Unsloth Flex Attention fixes it
  - 4-bit QLoRA: base stays NF4, only LoRA adapters (~0.5 GB) are BF16 trainable
  - No GRPOTrainer needed — we own the full loop: generate → verify → refine → backward

Rewards (tiered):
  +1.0        all test cases pass
  +0.1–+0.8   partial (fraction of tests passed)
  +0.05       compiled and ran, zero tests passed
  -1.0        compile error / runtime error / truncated (masked from loss)

Usage:
  # 1. Start SGLang server
  python -m sglang.launch_server \
      --model openai/gpt-oss-20b \
      --port 8000 --reasoning-effort high

  # 2. Train (this script)
  python rl_loop.py \
      --model unsloth/gpt-oss-20b \
      --server-url http://localhost:8000 \
      --dataset-path ./ag_extended/train.jsonl \
      --group-size 8 --batch-size 16 \
      --llm-concurrency 16 --validator-concurrency 64 \
      --refinement-rounds 3 --num-steps 2000
"""

import argparse
import asyncio
import json
import logging
import random
import re
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import aiohttp
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel

log = logging.getLogger("rl_loop")

# Force our handler onto the root logger AFTER Unsloth has imported and patched.
# Unsloth resets the root logger during import; we re-apply after.
def _setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(h)
    # Also pin our own logger explicitly
    log.setLevel(logging.INFO)

_setup_logging()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    id: str
    statement: str
    test_cases: List[Dict]   # [{"input": str, "output": str}]
    source: str = ""


@dataclass
class Attempt:
    problem:    Problem
    messages:   List[Dict]   # the chat messages sent to the server
    completion: str
    truncated:  bool
    round:      int          # 0 = initial, 1+ = refinement
    reward:     float = 0.0
    advantage:  float = 0.0
    # log_prob not available from server; IS ratio treated as 1.0 (clipped by PPO anyway)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset:
    def __init__(self, path: str):
        import base64, pickle, zlib
        self.problems: List[Problem] = []

        with open(path) as f:
            for line in f:
                row = json.loads(line)
                tcs = self._decode(row["private_test_cases"], base64, pickle, zlib)
                if not tcs:
                    continue
                self.problems.append(Problem(
                    id=row["question_id"],
                    statement=row["question_content"],
                    test_cases=tcs,
                    source=row.get("source", ""),
                ))
        log.info(f"Loaded {len(self.problems)} problems from {path}")

    def sample(self, n: int) -> List[Problem]:
        return random.choices(self.problems, k=n)

    @staticmethod
    def _decode(raw, base64, pickle, zlib) -> List[Dict]:
        if not raw:
            return []
        try:
            obj = pickle.loads(zlib.decompress(base64.b64decode(raw.encode())))
            if isinstance(obj, (str, bytes)):
                obj = json.loads(obj)
            out = []
            for item in obj:
                if isinstance(item, dict) and "input" in item:
                    out.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append({"input": str(item[0]), "output": str(item[1])})
            return out
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM = """\
You are an expert competitive programmer. Solve problems in C using stdin/stdout. Always wrap your solution in a ```c ... ``` block.

CRITICAL REQUIREMENTS:

1. INCLUDES - You MUST include ALL necessary headers:
   - #include <stdio.h>      // printf, scanf
   - #include <stdlib.h>     // malloc, free, qsort
   - #include <string.h>     // strlen, strcmp, memset
   - #include <stdbool.h>    // bool, true, false
   - #include <math.h>       // sqrt, pow, floor, ceil
   - #include <limits.h>     // INT_MAX, INT_MIN
   - #include <ctype.h>      // isdigit, isalpha, tolower
   - #include <stdint.h>     // uint64_t, int64_t
   - #include <gmp.h>        // arbitrary precision arithmetic (mpz_t)
   - #include "uthash.h"     // hash tables

2. DATA STRUCTURES:

   Hash Table (uthash):
     struct hash_entry { int key; int value; UT_hash_handle hh; };
     struct hash_entry *hash = NULL;
     // Add:  HASH_ADD_INT(hash, key, entry);
     // Find: HASH_FIND_INT(hash, &key, found);
     // Free: HASH_ITER(hh, hash, cur, tmp) { HASH_DEL(hash, cur); free(cur); }

   Dynamic Array: malloc/realloc/free

   Big Integers (GMP):
     mpz_t a; mpz_init(a); mpz_set_str(a, "123", 10);
     mpz_add/mul/mod(result, a, b); gmp_printf("%Zd\n", result); mpz_clear(a);

3. I/O: stdin via scanf(), stdout via printf(). Match output format EXACTLY (spaces, newlines).
   - int: scanf("%d",&n)  long: scanf("%lld",&n)  string: scanf("%s",str)

4. COMPILATION: gcc -std=c11 -O2 -o program code.c -lm -lgmp
   uthash.h is header-only, no -l flag needed.

5. Always free malloc'd memory, mpz_clear() all GMP vars, return 0 from main().
"""

def solve_messages(problem: Problem) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"Solve this problem in C:\n\n{problem.statement}"},
    ]


def refine_messages(problem: Problem, prev_code: str) -> List[Dict]:
    return [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": f"Solve this problem in C:\n\n{problem.statement}"},
        {"role": "assistant", "content": f"```c\n{prev_code}\n```"},
        {"role": "user",      "content": (
            "Your solution failed the test cases. "
            "Analyse what went wrong and write a corrected C solution."
        )},
    ]

def extract_code(text: str) -> Optional[str]:
    for lang in ["c", "cpp", ""]:
        m = re.search(rf"```{lang}\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Inference client — SGLang / vLLM OpenAI-compatible server
# ---------------------------------------------------------------------------

async def chat_complete(
    session:      aiohttp.ClientSession,
    server_url:   str,
    server_model: str,
    messages:     List[Dict],
    temperature:  float,
    sem:          asyncio.Semaphore,
    max_tokens:   int = 16384,
    max_retries:  int = 3,
) -> Tuple[str, bool]:
    """
    Single chat completion. Returns (text, truncated).
    Retries up to max_retries times on timeout or connection error.
    The semaphore is acquired once for the whole attempt including retries
    so we don't release a slot until we actually have a result.
    """
    payload = {
        "model":       server_model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    # Per-request timeout: 10 min connect + 20 min total.
    # Model can think freely; vLLM long reasoning responses can take several minutes.
    req_timeout = aiohttp.ClientTimeout(connect=30, total=1200)

    async with sem:
        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{server_url}/v1/chat/completions",
                    json=payload,
                    timeout=req_timeout,
                ) as resp:
                    data = await resp.json()

                if "error" in data:
                    raise RuntimeError(f"Server error: {data['error']}")

                choice    = data["choices"][0]
                text      = choice["message"]["content"] or ""
                truncated = choice.get("finish_reason") == "length"
                return text, truncated

            except (asyncio.TimeoutError, aiohttp.ServerDisconnectedError,
                    aiohttp.ClientConnectorError) as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt   # 1s, 2s, 4s backoff
                    log.warning(f"  [gen] request failed ({type(e).__name__}), "
                                f"retry {attempt+1}/{max_retries} in {wait}s ...")
                    await asyncio.sleep(wait)
                else:
                    log.error(f"  [gen] request failed after {max_retries} retries: {e}")
                    return "", True   # treat as truncated, gets reward=-1 and masked from loss


# ---------------------------------------------------------------------------
# Verifier — runs in a ProcessPoolExecutor worker (no Docker)
# ---------------------------------------------------------------------------

def _verify_worker(code: str, test_cases: List[Dict], timeout_s: int) -> Dict:
    """
    Compile with gcc and run each test case. Returns:
      {"result": "success"|"compile_error"|"runtime_error"|"wrong_output"|"timeout",
       "passed": int, "total": int}
    """
    import subprocess, tempfile, os
    from pathlib import Path

    total = len(test_cases)

    with tempfile.TemporaryDirectory(prefix="rl_verify_") as tmp:
        src = Path(tmp) / "sol.c"
        exe = Path(tmp) / "sol"
        src.write_text(code)

        # Compile
        try:
            cp = subprocess.run(
                ["gcc", "-std=c11", "-O2", str(src), "-o", str(exe), "-lm", "-lgmp"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return {"result": "compile_error", "passed": 0, "total": total,
                    "stderr": "compile timeout"}

        if cp.returncode != 0:
            return {"result": "compile_error", "passed": 0, "total": total,
                    "stderr": cp.stderr[:400]}

        # Run test cases
        passed = 0
        for tc in test_cases:
            try:
                rp = subprocess.run(
                    [str(exe)], input=tc["input"],
                    capture_output=True, text=True, timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                return {"result": "timeout", "passed": passed, "total": total, "stderr": ""}

            if rp.returncode != 0:
                return {"result": "runtime_error", "passed": passed, "total": total,
                        "stderr": rp.stderr[:200]}

            actual   = "\n".join(l.rstrip() for l in rp.stdout.rstrip("\n").split("\n"))
            expected = "\n".join(l.rstrip() for l in tc["output"].rstrip("\n").split("\n"))
            if actual == expected:
                passed += 1
            else:
                return {"result": "wrong_output", "passed": passed, "total": total, "stderr": ""}

        return {"result": "success", "passed": passed, "total": total, "stderr": ""}


def reward_from_result(v: Dict) -> float:
    """Tiered reward from verifier output."""
    result = v["result"]
    passed, total = v["passed"], max(v["total"], 1)

    if result == "success":
        return 1.0
    if result in ("compile_error",):
        return -1.0
    if result in ("runtime_error", "timeout"):
        return -1.0

    # wrong_output — partial credit
    frac = passed / total
    if frac == 0.0:
        return 0.05   # at least compiled and ran
    return 0.1 + 0.7 * frac   # 0.1 → 0.8


# ---------------------------------------------------------------------------
# Advantages — group-normalised per problem
# ---------------------------------------------------------------------------

def assign_advantages(attempts: List[Attempt]) -> None:
    """Normalise rewards within each problem group. Mutates attempts in-place."""
    groups: Dict[str, List[int]] = {}
    for i, a in enumerate(attempts):
        groups.setdefault(a.problem.id, []).append(i)

    for indices in groups.values():
        g   = torch.tensor([attempts[i].reward for i in indices], dtype=torch.float32)
        std = g.std()
        if std >= 1e-8:
            norm = ((g - g.mean()) / std).tolist()
        else:
            norm = [0.0] * len(g)
        for idx, adv in zip(indices, norm):
            attempts[idx].advantage = adv


# ---------------------------------------------------------------------------
# PPO loss — clipped surrogate, no KL, no entropy (veRL / DeepCoder recipe)
# log_prob not available from inference server → old_lp = 0, ratio = exp(new_lp)
# This is equivalent to treating the behaviour policy as having lp=0,
# which is fine when clipped: the clip bounds still limit the update.
# ---------------------------------------------------------------------------

def ppo_loss(
    model,
    tokenizer,
    attempts:   List[Attempt],
    epsilon:    float,
    clip_high:  float,
    max_length: int,
    device:     torch.device,
) -> Tuple[torch.Tensor, Dict]:
    model.train()
    terms, masked = [], 0

    for a in attempts:
        if a.truncated:
            masked += 1
            continue

        try:
            prompt_str = tokenizer.apply_chat_template(
                a.messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            prompt_str = " ".join(m["content"] for m in a.messages)

        full_str   = prompt_str + a.completion
        full       = tokenizer(full_str, return_tensors="pt",
                               truncation=True, max_length=max_length).to(device)
        prompt_len = tokenizer(prompt_str, return_tensors="pt",
                               truncation=True, max_length=max_length
                               )["input_ids"].shape[1]

        tgt    = full["input_ids"][0]
        start  = prompt_len
        end    = tgt.shape[0]
        if end <= start:
            continue

        # ── KEY CHANGE: gather only the target token logits, never materialise
        # the full [seq_len, vocab_size] log_softmax tensor.
        # out.logits shape: [1, seq_len, vocab_size]
        # We slice out only the completion positions before softmax.
        out         = model(**full)
        # logits for positions [start-1 .. end-2] predict tokens [start .. end-1]
        comp_logits = out.logits[0, start-1:end-1, :]          # [comp_len, vocab_size]
        comp_tgt    = tgt[start:end]                           # [comp_len]

        # gather the logit for each target token, then compute log_softmax only
        # on those gathered values — avoids the huge intermediate tensor
        token_logits = comp_logits.gather(
            1, comp_tgt.unsqueeze(1)
        ).squeeze(1)                                           # [comp_len]

        # log_softmax denominator: logsumexp over vocab (no large alloc needed
        # because we do it row-wise and PyTorch fuses it)
        log_z        = torch.logsumexp(comp_logits, dim=-1)    # [comp_len]
        token_lps    = token_logits - log_z                    # [comp_len]

        # free the big logits tensor immediately before backward
        del out, comp_logits
        torch.cuda.empty_cache()

        new_lp = token_lps.mean()
        ratio  = torch.exp(new_lp)
        adv    = torch.tensor(a.advantage, device=device, dtype=torch.float32)
        surr1  = ratio * adv
        surr2  = torch.clamp(ratio, 1 - epsilon, clip_high) * adv
        terms.append(-torch.min(surr1, surr2))

    if not terms:
        return torch.tensor(0.0, device=device, requires_grad=True), \
               {"masked": masked, "active": 0}

    loss = torch.stack(terms).mean()
    return loss, {"masked": masked, "active": len(terms), "loss": loss.item()}

# ---------------------------------------------------------------------------
# Async pipeline step
# ---------------------------------------------------------------------------

async def run_step(
    problems: List[Problem],
    session:  aiohttp.ClientSession,
    executor: ProcessPoolExecutor,
    args,
    llm_sem:  asyncio.Semaphore,
    val_sem:  asyncio.Semaphore,
    loop:     asyncio.AbstractEventLoop,
) -> List[Attempt]:
    """
    Runs one full RL step:
      generate G completions per problem  (concurrent, capped by llm_sem)
      verify each concurrently            (capped by val_sem)
      refine failures up to X rounds      (same llm_sem, then verify again)
      assign group-normalised advantages
    """

    async def gen_one(messages: List[Dict]) -> Tuple[str, bool]:
        return await chat_complete(
            session, args.server_url, args.server_model,
            messages, args.temperature, llm_sem, args.max_tokens,
        )

    async def verify_one(code: str, test_cases: List[Dict]) -> Dict:
        async with val_sem:
            return await loop.run_in_executor(
                executor, _verify_worker, code, test_cases, args.verify_timeout,
            )

    # ---- Round 0: generate G completions per problem ----
    all_attempts: List[Attempt] = []
    n_gen = len(problems) * args.group_size
    log.info(f"  [gen] round=0  firing {n_gen} requests "
             f"({len(problems)} problems × {args.group_size} completions) ...")
    t_gen = time.time()

    gen_tasks = [
        gen_one(solve_messages(p))
        for p in problems
        for _ in range(args.group_size)
    ]
    gen_results = await asyncio.gather(*gen_tasks)
    log.info(f"  [gen] round=0  done  {n_gen} completions in {time.time()-t_gen:.1f}s")

    initial: List[Attempt] = []
    idx = 0
    for p in problems:
        msgs = solve_messages(p)
        for _ in range(args.group_size):
            comp, trunc = gen_results[idx]; idx += 1
            initial.append(Attempt(problem=p, messages=msgs,
                                   completion=comp, truncated=trunc, round=0))

    # ---- Verify initial ----
    log.info(f"  [verify] round=0  verifying {len(initial)} completions ...")
    t_ver = time.time()
    verify_tasks = [
        verify_one(extract_code(a.completion) or a.completion, a.problem.test_cases)
        for a in initial
    ]
    verify_results = await asyncio.gather(*verify_tasks)
    for a, v in zip(initial, verify_results):
        a.reward = -1.0 if a.truncated else reward_from_result(v)

    r0_pass    = sum(1 for a in initial if a.reward == 1.0)
    r0_partial = sum(1 for a in initial if 0.0 < a.reward < 1.0)
    r0_fail    = sum(1 for a in initial if a.reward <= 0.0)
    log.info(f"  [verify] round=0  done in {time.time()-t_ver:.1f}s  "
             f"pass={r0_pass}  partial={r0_partial}  fail/error={r0_fail}")

    all_attempts.extend(initial)

    # ---- Refinement rounds ----
    best: Dict[str, Tuple[str, float]] = {}
    for a in initial:
        code = extract_code(a.completion) or a.completion
        pid  = a.problem.id
        if pid not in best or a.reward > best[pid][1]:
            best[pid] = (code, a.reward)

    unsolved_problems = {p.id for p in problems if best.get(p.id, (None, -1))[1] < 1.0}

    for round_idx in range(1, args.refinement_rounds + 1):
        if not unsolved_problems:
            log.info(f"  [refine] all problems solved — skipping rounds {round_idx}+")
            break

        failed = [a for a in all_attempts
                  if a.problem.id in unsolved_problems
                  and a.round == round_idx - 1
                  and not a.truncated]
        if not failed:
            break

        log.info(f"  [gen] round={round_idx}  refining {len(failed)} failed attempts "
                 f"({len(unsolved_problems)} unsolved problems) ...")
        t_ref = time.time()
        refine_tasks = [
            gen_one(refine_messages(a.problem, best[a.problem.id][0]))
            for a in failed
        ]
        refine_results = await asyncio.gather(*refine_tasks)
        log.info(f"  [gen] round={round_idx}  done in {time.time()-t_ref:.1f}s")

        refined: List[Attempt] = []
        for orig, (comp, trunc) in zip(failed, refine_results):
            msgs = refine_messages(orig.problem, best[orig.problem.id][0])
            refined.append(Attempt(problem=orig.problem, messages=msgs,
                                   completion=comp, truncated=trunc, round=round_idx))

        log.info(f"  [verify] round={round_idx}  verifying {len(refined)} refined completions ...")
        t_ver2 = time.time()
        verify_tasks = [
            verify_one(extract_code(a.completion) or a.completion, a.problem.test_cases)
            for a in refined
        ]
        for a, v in zip(refined, await asyncio.gather(*verify_tasks)):
            a.reward = -1.0 if a.truncated else reward_from_result(v)
            code = extract_code(a.completion) or a.completion
            if a.reward > best.get(a.problem.id, (None, -1.0))[1]:
                best[a.problem.id] = (code, a.reward)
            if a.reward == 1.0:
                unsolved_problems.discard(a.problem.id)

        rn_pass = sum(1 for a in refined if a.reward == 1.0)
        rn_fail = sum(1 for a in refined if a.reward <= 0.0)
        log.info(f"  [verify] round={round_idx}  done in {time.time()-t_ver2:.1f}s  "
                 f"newly solved={rn_pass}  still failing={rn_fail}  "
                 f"unsolved remaining={len(unsolved_problems)}")

        all_attempts.extend(refined)

    log.info(f"  [advantages] computing over {len(all_attempts)} total attempts ...")
    assign_advantages(all_attempts)
    reward_dist = {
        "pass":    sum(1 for a in all_attempts if a.reward == 1.0),
        "partial": sum(1 for a in all_attempts if 0.0 < a.reward < 1.0),
        "error":   sum(1 for a in all_attempts if a.reward < 0.0),
        "zero":    sum(1 for a in all_attempts if a.reward == 0.0),
    }
    log.info(f"  [rewards] {reward_dist}")
    return all_attempts


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model loading — Unsloth 4-bit QLoRA
# ---------------------------------------------------------------------------

def load_model(args):
    """
    Load gpt-oss-20b via Unsloth for backprop only (SGLang handles generation).

    Key choices:
      load_in_4bit=True              base stays NF4, ~14 GB vs ~40 GB full BF16
      fast_inference=False           we don't generate here; SGLang does that
      use_gradient_checkpointing="unsloth"
                                     Unsloth's Flex Attention checkpointing —
                                     the only correct backward pass for gpt-oss.
                                     FA2/FA3 silently produce wrong gradients.
      lora_dropout=0.0               Unsloth optimises for dropout=0; use 0.05
                                     only if you see overfitting.
      target_modules excludes router GptOssTopKRouter is not nn.Linear;
                                     PEFT will crash if you include it.
    """
    log.info(f"Loading {args.model} via Unsloth (4-bit QLoRA, backprop only)...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name             = args.model,
        max_seq_length         = args.max_seq_length,
        load_in_4bit           = True,
        dtype                  = None,           # auto → bfloat16 on H200
        fast_inference         = False,          # no inference here; SGLang handles it
        offload_embedding      = True,           # saves ~1 GB VRAM
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r              = args.lora_rank,
        lora_alpha     = args.lora_rank * 2,
        lora_dropout   = 0.0,
        bias           = "none",
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            # router / gate excluded — GptOssTopKRouter is not nn.Linear
        ],
        use_gradient_checkpointing = "unsloth",  # Flex Attention, correct for gpt-oss
        random_state               = 42,
    )

    # ── Patch: neutralise torch.compile on GptOssTopKRouter in compiled cache ──
    # The compiled cache wraps GptOssTopKRouter_forward with @torch.compile,
    # which hits a StopIteration in dict_keys_getitem (dynamo bug with MoE routers).
    # We monkey-patch the cached module's global to replace the compiled function
    # with a dynamo-disabled version before any forward pass runs.
    import importlib, sys
    
    cache_mod_name = "unsloth_compiled_module_gpt_oss"
    if cache_mod_name in sys.modules:
        cache_mod = sys.modules[cache_mod_name]
        if hasattr(cache_mod, "GptOssTopKRouter_forward"):
            original_fn = cache_mod.GptOssTopKRouter_forward
            # Strip any existing compile/dynamo wrapper, then hard-disable
            unwrapped = getattr(original_fn, "__wrapped__", original_fn)
            cache_mod.GptOssTopKRouter_forward = torch._dynamo.disable(unwrapped, recursive=True)
            log.info("  [patch] GptOssTopKRouter_forward: torch.compile disabled in compiled cache")
        else:
            log.warning("  [patch] GptOssTopKRouter_forward not found in compiled cache — check module name")
    else:
        log.warning(f"  [patch] {cache_mod_name} not in sys.modules yet — patch may not apply")
    # ────────────────────────────────────────────────────────────────────────────

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"LoRA trainable: {trainable:,}/{total:,} ({100*trainable/total:.3f}%)")
    log.info(f"VRAM after load: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

async def train(args):
    import torch
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Dataset(args.dataset_path)
    model, tokenizer = load_model(args)
    _setup_logging()   # re-apply after Unsloth model load resets root logger
    
    # ── Patch: disable torch.compile on GptOssTopKRouter ──────────────────────
    # torch.compile's dynamo hits a StopIteration in dict_keys_getitem inside the
    # router's compiled forward, which PEP 479 re-raises as RuntimeError inside
    # the async coroutine. Wrapping with torch._dynamo.disable fixes it cleanly.
    import torch._dynamo
    for module in model.modules():
        if type(module).__name__ == "GptOssTopKRouter":
            module.forward = torch._dynamo.disable(module.forward)
            # log.info(f"  [patch] disabled torch.compile on {type(module).__name__}")
    # ──────────────────────────────────────────────────────────────────────────

    # Paged AdamW 8-bit: optimizer states stay 8-bit, paged to CPU if needed.
    # Saves ~2 GB vs full AdamW on H200, frees more VRAM for SGLang KV cache.
    from bitsandbytes.optim import PagedAdamW8bit
    optimizer = PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=args.lr * 0.1,
    )

    llm_sem  = asyncio.Semaphore(args.llm_concurrency)
    val_sem  = asyncio.Semaphore(args.validator_concurrency)
    loop     = asyncio.get_event_loop()
    executor = ProcessPoolExecutor(max_workers=args.validator_concurrency)

    # Persistent session: large pool size, no internal timeout (we set per-request)
    connector = aiohttp.TCPConnector(limit=args.llm_concurrency * 2, keepalive_timeout=300)
    session   = aiohttp.ClientSession(connector=connector)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_pass = 0.0

    log.info("=" * 60)
    log.info(f"model={args.model}  server={args.server_url}")
    log.info(f"group={args.group_size}  batch={args.batch_size}  temp={args.temperature}")
    log.info(f"llm_concurrency={args.llm_concurrency}  val_concurrency={args.validator_concurrency}")
    log.info(f"refinement_rounds={args.refinement_rounds}  steps={args.num_steps}")
    log.info(f"max_seq_length={args.max_seq_length}")
    log.info("=" * 60)

    try:
        for step in range(args.num_steps):
            t0  = time.time()
            problems = dataset.sample(args.batch_size)
    
            # ── async: SGLang generates, verifier runs, refinement loops ──
            attempts = await run_step(problems, session, executor, args, llm_sem, val_sem, loop)
    
            # ── sync: PPO backward on Unsloth 4-bit model ──
            log.info(f"  [backprop] computing PPO loss over {len(attempts)} attempts ...")
            t_bp = time.time()
            optimizer.zero_grad()
            loss, metrics = ppo_loss(
                model, tokenizer, attempts,
                args.epsilon, args.clip_high,
                args.max_seq_length,
                device,
            )
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                optimizer.step()
                scheduler.step()
            log.info(f"  [backprop] done in {time.time()-t_bp:.1f}s  "
                     f"loss={loss.item():.4f}  active={metrics['active']}  masked={metrics['masked']}")
    
            rewards   = [a.reward for a in attempts]
            pass_rate = sum(1 for r in rewards if r == 1.0) / max(len(rewards), 1)
            trunc_r   = sum(1 for a in attempts if a.truncated) / max(len(attempts), 1)
            log.info(
                f"step={step:4d}  loss={loss.item():.4f}  pass@1={pass_rate:.2%}  "
                f"trunc={trunc_r:.2%}  active={metrics['active']}  masked={metrics['masked']}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  t={time.time()-t0:.1f}s"
            )
    
            if (step + 1) % args.save_every == 0:
                ckpt = save_dir / f"step_{step+1}"
                ckpt.mkdir(exist_ok=True)
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                log.info(f"Saved → {ckpt}")
                if pass_rate > best_pass:
                    best_pass = pass_rate
                    model.save_pretrained(save_dir / "best")
                    log.info(f"New best: {best_pass:.2%}")

    finally:
        await session.close()
        executor.shutdown(wait=False)
        log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",                 default="unsloth/gpt-oss-20b",
                   help="Unsloth model id for backprop (4-bit QLoRA).")
    p.add_argument("--server-model",          default="openai/gpt-oss-20b",
                   help="Model name as registered in vLLM (must match --served-model-name "
                        "or the HF id vLLM was launched with).")
    p.add_argument("--server-url",            default="http://localhost:8000",
                   help="SGLang OpenAI-compatible server base URL")
    p.add_argument("--dataset-path",          required=True)
    p.add_argument("--max-seq-length",        type=int,   default=16384,
                   help="Total context window for backprop tokenisation. "
                        "Unsloth supports up to 380K for gpt-oss.")
    p.add_argument("--group-size",            type=int,   default=8,
                   help="Completions per problem (exploration breadth)")
    p.add_argument("--batch-size",            type=int,   default=16,
                   help="Problems per training step")
    p.add_argument("--llm-concurrency",       type=int,   default=16,
                   help="Max concurrent requests to SGLang server (semaphore Y)")
    p.add_argument("--validator-concurrency", type=int,   default=64,
                   help="Max concurrent gcc verifier workers (semaphore Z)")
    p.add_argument("--refinement-rounds",     type=int,   default=3,
                   help="Max refinement rounds per failed attempt (X)")
    p.add_argument("--verify-timeout",        type=int,   default=10,
                   help="Per-test-case execution timeout in seconds")
    p.add_argument("--max-tokens",            type=int,   default=16384,
                   help="Max tokens per completion sent to vLLM. "
                        "4096 is fast; raise to 8192 for harder problems.")
    p.add_argument("--temperature",           type=float, default=0.7)
    p.add_argument("--lora-rank",             type=int,   default=128)
    p.add_argument("--lr",                    type=float, default=1e-6)
    p.add_argument("--epsilon",               type=float, default=0.2)
    p.add_argument("--clip-high",             type=float, default=5.0)
    p.add_argument("--num-steps",             type=int,   default=2000)
    p.add_argument("--save-dir",              default="./checkpoints")
    p.add_argument("--save-every",            type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(train(parse_args()))
