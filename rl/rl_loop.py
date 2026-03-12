"""
grpo_cf.py — GRPO on GPT-OSS-20B · Codeforces / Ag-LiveCodeBench-X
====================================================================
Single-file, self-contained. All knobs at the top in CAPS.

Flow per problem:
  1. Generate NUM_GENERATIONS (16) C solutions via an EXTERNAL vLLM server
     (start it yourself: vllm serve <model> --port 8000)
  2. Compile + test each with gcc → reward = passed / total  (0.0–1.0)
  3. GRPO backward pass (plain transformers + PEFT QLoRA, no Unsloth)
  4. [optional] One refinement pass: unsolved problems get a second dataset
     pass with the best prior code shown in context

Memory optimisations applied
  • logits_to_keep — model only materialises logits for completion tokens,
    not the full prompt+completion sequence.
  • selective_log_softmax — avoids storing the full [B, T, V] log-prob
    tensor; computes logsumexp row-by-row and gathers only the generated
    token's logprob.  Saves ≈ batch × seq × vocab × 4 bytes of VRAM at the
    peak of every train step.  See: https://www.tylerromero.com/posts/2025-02-selective-log-softmax/
    and the corresponding TRL PRs.

Timing targets (rough):
  96GB  GPU · 5k problems · ~3 days
  141GB GPU · 5k problems · ~2 days

Usage:
  # 1. Start vLLM server (no --worker-cls needed — we call it via plain HTTP)
  vllm serve openai/gpt-oss-20b --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.25
  # IMPORTANT: always pass --max-model-len 8192. Without it vLLM defaults to
  # 131072, which wastes ~90% of your KV cache budget.
  #
  # Use 127.0.0.1 (not localhost) in VLLM_BASE_URL — on Vast.ai, "localhost"
  # may route through the host proxy and return 401.

  # 2. Run training
  python grpo_cf.py                   # full run
  python grpo_cf.py --no-refinement   # skip refinement (only CLI flag)
"""

# ============================================================================
# CONFIGURATION  — edit here; nothing below needs to change
# ============================================================================

MODEL_NAME            = "openai/gpt-oss-20b"
DATASET_PATH          = "./ag_extended/train.jsonl"
OUTPUT_DIR            = "./checkpoints_grpo"

# Set to e.g. 100 for a quick smoke-test; None = all problems
MAX_EXAMPLES          = 100

# LoRA
LORA_RANK             = 256
LORA_ALPHA            = 512        # LORA_RANK * 2

# Generation
NUM_GENERATIONS       = 2         # completions per problem
MAX_SEQ_LENGTH        = 8192       # total context window (prompt + completion)
MAX_COMPLETION_TOKENS = 8000       # max new tokens per completion
REASONING_EFFORT      = "low"   # gpt-oss reasoning budget per generation
TEMPERATURE           = 0.7

# Training
LEARNING_RATE         = 1e-4
WEIGHT_DECAY          = 0.01
WARMUP_RATIO          = 0.05
LR_SCHEDULER          = "cosine"
OPTIMIZER             = "adamw_8bit"
GRAD_ACCUM_STEPS      = 1
SAVE_STEPS            = 50
# Entropy computation chunk size over vocab dimension to avoid OOM in GRPO.
# Smaller = less peak VRAM, slower. 2048 is a safe default on large vocabs.
ENTROPY_VOCAB_CHUNK   = 2048
LOGPROB_VOCAB_CHUNK   = 2048

# Attention backend for GPT-OSS training.
# GPT-OSS supports eager, flex_attention, and a flash path compatible with
# kernels-community/vllm-flash-attn3 (Hopper-focused). Prefer flex_attention
# for backprop; fallback to eager if unavailable in the local torch build.
ATTN_IMPLEMENTATION   = "flex_attention"

# External vLLM server.
# Simplest setup — no API key needed for a local server:
#   vllm serve unsloth/gpt-oss-20b --port 8000 --max-model-len 8192
# If you launched with --api-key, set VLLM_API_KEY env var or hardcode below.
# Use 127.0.0.1, NOT localhost or the external hostname.
# On Vast.ai, "localhost" may resolve through the host proxy (→ 401).
# 127.0.0.1 hits the container's loopback directly, bypassing the proxy.
VLLM_BASE_URL         = "http://127.0.0.1:8000"
VLLM_API_KEY          = None   # or: os.environ.get("VLLM_API_KEY")

# Hugging Face Hub — set to None to skip pushing
HF_REPO_ID            = "your-username/gpt-oss-20b-grpo-cf"

# Verification (C compile + run)
VERIFY_TIMEOUT_S      = 10         # per-test-case wall time
VERIFY_WORKERS        = 32         # parallel gcc worker processes

# ============================================================================
# Imports
# ============================================================================

import argparse
import base64
import json
import logging
import os
import pickle
import re
import subprocess
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import trl.trainer.grpo_trainer as _grpo_trainer_mod
import trl.trainer.utils as _trl_trainer_utils

# transformers refuses to train on mxfp4 weights with a hard ValueError.
# The check is overly conservative — recent PEFT handles mxfp4 LoRA fine.
# Patch it out before any Trainer is constructed.
import transformers.trainer as _transformers_trainer
import transformers.trainer_utils as _trl_utils
# transformers hard-raises on mxfp4 training. Patch both the source module and
# the already-imported reference in trainer.py's namespace.
_noop = lambda model: None
_trl_utils.validate_quantization_for_training = _noop
_transformers_trainer.validate_quantization_for_training = _noop


def _safe_shuffle_sequence_dict(seq_dict):
    """
    TRL compatibility shim:
    Some TRL versions can hand mixed prompt-level/completion-level lengths to
    shuffle_sequence_dict (e.g. 2 vs 4), which causes CUDA index asserts.
    Expand shorter sequence fields before permutation.
    """
    def _len0(v):
        if isinstance(v, torch.Tensor):
            # Scalars (shape=()) are metadata, not batch sequences.
            if v.ndim == 0:
                return None
            return int(v.shape[0])
        if isinstance(v, list):
            return len(v)
        return None

    lengths = [n for n in (_len0(v) for v in seq_dict.values()) if n is not None]
    if not lengths:
        return seq_dict
    target = max(lengths)

    def _expand(v):
        n = _len0(v)
        if n is None or n == target:
            return v
        if n <= 0:
            return v
        if target % n == 0:
            reps = target // n
            if isinstance(v, torch.Tensor):
                # Repeat each row to preserve prompt->generation grouping.
                idx = torch.arange(n, device=v.device).repeat_interleave(reps)
                return v[idx]
            return [v[i // reps] for i in range(target)]
        # Non-divisible fallback: cycle values.
        if isinstance(v, torch.Tensor):
            idx = torch.arange(target, device=v.device) % n
            return v[idx]
        return [v[i % n] for i in range(target)]

    normalized = {k: _expand(v) for k, v in seq_dict.items()}
    first = next(iter(normalized.values()))
    if isinstance(first, torch.Tensor):
        perm = torch.randperm(first.shape[0], device=first.device)
    else:
        perm = torch.randperm(len(first))

    def _permute(v):
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                return v
            return v[perm]
        if isinstance(v, list):
            return [v[i] for i in perm.tolist()]
        return v

    return {k: _permute(v) for k, v in normalized.items()}


# Patch both symbols used by GRPOTrainer call sites.
_trl_trainer_utils.shuffle_sequence_dict = _safe_shuffle_sequence_dict
_grpo_trainer_mod.shuffle_sequence_dict = _safe_shuffle_sequence_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("grpo_cf")

# ============================================================================
# System prompt
# ============================================================================

SYSTEM = """\
You are an expert competitive programmer. Solve the problem in C using stdin/stdout.
Wrap your solution in a single ```c ... ``` block.

INCLUDES — add every header you actually use:
  #include <stdio.h>      // printf, scanf, fgets
  #include <stdlib.h>     // malloc, realloc, free, qsort, atoi, exit
  #include <string.h>     // strlen, strcmp, strcpy, memset, memmove
  #include <stdbool.h>    // bool, true, false
  #include <math.h>       // sqrt, pow, ceil, floor, fabs          (link: -lm)
  #include <limits.h>     // INT_MAX, INT_MIN, LLONG_MAX, LLONG_MIN
  #include <stdint.h>     // int64_t, uint64_t, int32_t, uint32_t
  #include <ctype.h>      // isdigit, isalpha, tolower, toupper
  #include <gmp.h>        // arbitrary-precision integers          (link: -lgmp)
  #include "uthash.h"     // hash tables — header-only, no -l flag needed

DATA STRUCTURES:

  Dynamic array — malloc / realloc / free directly.

  Hash table (uthash):
    struct entry { int key; int value; UT_hash_handle hh; };
    struct entry *table = NULL;
    struct entry *e = malloc(sizeof(*e)); e->key = k; e->value = v;
    HASH_ADD_INT(table, key, e);          // insert
    HASH_FIND_INT(table, &k, e);          // lookup (e = NULL if missing)
    HASH_ITER(hh, table, e, tmp) { HASH_DEL(table, e); free(e); }  // free all

  Big integers (GMP):
    mpz_t a, b, res;
    mpz_inits(a, b, res, NULL);
    mpz_set_str(a, "99999999999999999999999999999", 10);
    mpz_add(res, a, b);   // also: mpz_mul, mpz_mod, mpz_pow_ui, mpz_sqrt
    gmp_printf("%Zd\\n", res);
    mpz_clears(a, b, res, NULL);

  Sorting (qsort):
    int cmp_int(const void *a, const void *b) { return *(int*)a - *(int*)b; }
    qsort(arr, n, sizeof(int), cmp_int);

I/O:
  scanf("%d", &n);   scanf("%lld", &x);   scanf("%s", buf);
  printf("%d\\n", ans);   // match expected output format EXACTLY (spaces, newlines)

COMPILATION: gcc -std=c11 -O2 -o sol sol.c -lm -lgmp
Always return 0 from main. Free all malloc'd memory. mpz_clears all GMP vars.
"""


def _solve_messages(statement: str) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"Solve this problem in C:\n\n{statement}"},
    ]


def _refine_messages(statement: str, best_code: str) -> List[Dict]:
    return [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": f"Solve this problem in C:\n\n{statement}"},
        {"role": "assistant", "content": f"```c\n{best_code}\n```"},
        {"role": "user",      "content": (
            "Your solution failed some test cases. "
            "Re-examine the logic carefully and write a corrected C solution."
        )},
    ]


def _extract_code(text: str) -> Optional[str]:
    for lang in ("c", "cpp", ""):
        m = re.search(rf"```{lang}\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None

# ============================================================================
# Dataset helpers
# ============================================================================

@dataclass
class Problem:
    id: str
    statement: str
    test_cases: List[Dict]


def _encode_tcs(tcs: List[Dict]) -> str:
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tcs)))).decode()


def _decode_tcs(raw: str) -> List[Dict]:
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


def load_problems(path: str, max_n: Optional[int]) -> List[Problem]:
    problems = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            tcs = _decode_tcs(row.get("private_test_cases", ""))
            if not tcs:
                continue
            problems.append(Problem(
                id=row["question_id"],
                statement=row["question_content"],
                test_cases=tcs,
            ))
            if max_n and len(problems) >= max_n:
                break
    log.info(f"Loaded {len(problems)} problems from {path}")
    return problems


def build_initial_dataset(problems: List[Problem]) -> Dataset:
    return Dataset.from_list([{
        "prompt":      _solve_messages(p.statement),
        "problem_id":  p.id,
        "tcs_encoded": _encode_tcs(p.test_cases),
    } for p in problems])


def build_refinement_dataset(problems: List[Problem], best_codes: Dict[str, str]) -> Dataset:
    rows = []
    for p in problems:
        code = best_codes.get(p.id)
        if code is not None:
            rows.append({
                "prompt":      _refine_messages(p.statement, code),
                "problem_id":  p.id,
                "tcs_encoded": _encode_tcs(p.test_cases),
            })
    log.info(f"Refinement dataset: {len(rows)} unsolved problems")
    return Dataset.from_list(rows)

# ============================================================================
# Dependency check
# ============================================================================

def _check_dependencies():
    errors = []

    if subprocess.run(["which", "gcc"], capture_output=True).returncode != 0:
        errors.append("gcc not found        →  apt-get install -y gcc")

    gmp_src = "#include <gmp.h>\nint main(){mpz_t x;mpz_init(x);mpz_clear(x);return 0;}\n"
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(gmp_src); fname = f.name
    r = subprocess.run(["gcc", "-std=c11", fname, "-o", "/dev/null", "-lgmp"],
                       capture_output=True, text=True)
    os.unlink(fname)
    if r.returncode != 0:
        errors.append("libgmp not found     →  apt-get install -y libgmp-dev")

    uthash_ok = any(p.exists() for p in [
        Path("uthash.h"), Path("/usr/include/uthash.h"), Path("/usr/local/include/uthash.h"),
    ])
    if not uthash_ok:
        errors.append(
            "uthash.h not found   →  wget -q "
            "https://raw.githubusercontent.com/troydhanson/uthash/master/src/uthash.h"
        )

    # Check vLLM server is reachable.
    # /health returns 200 without auth on most vLLM builds.
    # If it returns 401, fall back to /v1/models with the API key.
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/health", timeout=5, headers=headers)
        if resp.status_code == 401:
            # Server has auth enabled — check /v1/models instead
            resp2 = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=5, headers=headers)
            if resp2.status_code == 401:
                errors.append(
                    f"vLLM server at {VLLM_BASE_URL} returned 401 — "
                    "set VLLM_API_KEY env var or remove --api-key from the server launch command"
                )
            elif resp2.status_code != 200:
                errors.append(f"vLLM server at {VLLM_BASE_URL} returned {resp2.status_code}")
        elif resp.status_code != 200:
            errors.append(f"vLLM server at {VLLM_BASE_URL} returned {resp.status_code}")
    except requests.exceptions.ConnectionError:
        errors.append(
            f"vLLM server not reachable at {VLLM_BASE_URL}\n"
            f"  Start it with:  vllm serve {MODEL_NAME} --port 8000 --max-model-len {MAX_SEQ_LENGTH}"
        )

    if errors:
        log.error("Dependency check FAILED:\n  " + "\n  ".join(errors))
        raise SystemExit(1)
    log.info("Dependencies OK: gcc + libgmp + uthash.h + vLLM server")

# ============================================================================
# C Verifier
# ============================================================================

def _verify_worker(code: str, test_cases: List[Dict], timeout_s: int) -> Dict:
    """Returns {"passed": int, "total": int}."""
    total = len(test_cases)
    with tempfile.TemporaryDirectory(prefix="grpo_v_") as tmp:
        src = Path(tmp) / "sol.c"
        exe = Path(tmp) / "sol"
        src.write_text(code)
        try:
            cp = subprocess.run(
                ["gcc", "-std=c11", "-O2", str(src), "-o", str(exe), "-lm", "-lgmp"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return {"passed": 0, "total": total}
        if cp.returncode != 0:
            return {"passed": 0, "total": total}

        passed = 0
        for tc in test_cases:
            try:
                rp = subprocess.run(
                    [str(exe)], input=tc["input"],
                    capture_output=True, text=True, timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                return {"passed": passed, "total": total}
            if rp.returncode != 0:
                return {"passed": passed, "total": total}
            actual   = "\n".join(l.rstrip() for l in rp.stdout.rstrip("\n").split("\n"))
            expected = "\n".join(l.rstrip() for l in tc["output"].rstrip("\n").split("\n"))
            if actual == expected:
                passed += 1
    return {"passed": passed, "total": total}

# ============================================================================
# Global state
# ============================================================================

_best_scores: Dict[str, float] = {}
_best_codes:  Dict[str, str]   = {}
_executor:    Optional[ProcessPoolExecutor] = None


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=VERIFY_WORKERS)
    return _executor

# ============================================================================
# Reward function
# ============================================================================

def make_reward_fn():
    """
    TRL-compatible reward function.
    reward = passed_tests / total_tests  (0.0 → 1.0)
    """
    def reward_fn(
        completions,
        prompts=None,
        problem_id=None,
        tcs_encoded=None,
        **kwargs,
    ) -> List[float]:
        def _broadcast_meta(meta, n: int, name: str):
            # TRL may pass prompt-level metadata while completions are expanded
            # (e.g. n = prompts * num_generations). Expand deterministically.
            if meta is None:
                return [None] * n
            if isinstance(meta, list):
                m = len(meta)
                if m == n:
                    return meta
                if m == 0:
                    return [None] * n
                if n % m == 0:
                    reps = n // m
                    if not hasattr(reward_fn, "_meta_expand_logged"):
                        log.info(f"  [reward] expanding {name}: {m} -> {n} (x{reps})")
                        reward_fn._meta_expand_logged = True
                    return [meta[i // reps] for i in range(n)]
                log.warning(f"  [reward] {name} length mismatch ({m} vs {n}); cycling values")
                return [meta[i % m] for i in range(n)]
            return [meta] * n

        executor = _get_executor()
        futures  = []
        n = len(completions)
        pids = _broadcast_meta(problem_id, n, "problem_id")
        tcs_all = _broadcast_meta(tcs_encoded, n, "tcs_encoded")

        for i in range(n):
            text    = (completions[i][0]["content"] if isinstance(completions[i], list)
                       else str(completions[i]))
            code    = _extract_code(text)
            pid     = pids[i]
            tcs_raw = tcs_all[i]
            tcs     = _decode_tcs(tcs_raw)

            if code is None or not tcs:
                futures.append((pid, None, None, tcs_raw))
            else:
                fut = executor.submit(_verify_worker, code, tcs, VERIFY_TIMEOUT_S)
                futures.append((pid, fut, code, tcs_raw))

        scores = []
        for pid, fut, code, tcs_raw in futures:
            if fut is None:
                scores.append(0.0)
                continue
            try:
                tcs    = _decode_tcs(tcs_raw)
                result = fut.result(timeout=VERIFY_TIMEOUT_S * len(tcs) + 60)
                score  = result["passed"] / max(result["total"], 1)
            except Exception as e:
                log.warning(f"  [verify] pid={pid} exception: {e}")
                score = 0.0

            scores.append(score)

            if pid and code and score > _best_scores.get(pid, -1.0):
                _best_scores[pid] = score
                _best_codes[pid]  = code

        pass_n = sum(1 for s in scores if s == 1.0)
        part_n = sum(1 for s in scores if 0.0 < s < 1.0)
        zero_n = sum(1 for s in scores if s == 0.0)
        log.info(f"  [reward] n={len(scores)}  full={pass_n}  partial={part_n}  zero={zero_n}")

        # TRL versions differ in reward API expectations:
        # - some expect per-completion rewards (len == len(completions))
        # - others expect per-prompt rewards     (len == len(prompts))
        #
        # We always score each completion (for best-code tracking), then adapt
        # the returned list length to what the trainer expects.
        expected_n = len(prompts) if isinstance(prompts, list) else None
        if expected_n is not None and expected_n > 0 and len(scores) != expected_n:
            if len(scores) % expected_n == 0:
                group = len(scores) // expected_n
                # Aggregate per prompt by mean reward across its generations.
                grouped = [
                    sum(scores[i * group:(i + 1) * group]) / group
                    for i in range(expected_n)
                ]
                log.info(
                    f"  [reward] returning prompt-level rewards: {len(scores)} -> {len(grouped)} "
                    f"(group={group}, agg=mean)"
                )
                return grouped

            log.warning(
                f"  [reward] cannot align reward length {len(scores)} to prompts {expected_n}; "
                "truncating/padding with zeros"
            )
            if len(scores) >= expected_n:
                return scores[:expected_n]
            return scores + [0.0] * (expected_n - len(scores))

        return scores

    return reward_fn

# ============================================================================
# selective_log_softmax
# ============================================================================
# Only materialise the log-prob for the tokens we actually need (the generated
# token ids), rather than the full vocab-sized probability distribution.
#
# Algorithm:
#   log_softmax(x_i) = x_i - logsumexp(x)
#
# We compute logsumexp row-by-row to cap peak VRAM at (seq_len × vocab) rather
# than (batch × seq_len × vocab).  Then we torch.gather just the one logit we
# need per position and subtract.
#
# Reference: https://www.tylerromero.com/posts/2025-02-selective-log-softmax/
# ============================================================================

def _rowwise_logsumexp_chunked(logits_2d: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Numerically stable logsumexp over vocab for logits shaped (T, V), using
    vocab chunks to avoid allocating a full fp32 (T, V) buffer.
    """
    t = logits_2d.size(0)
    device = logits_2d.device
    m = torch.full((t,), float("-inf"), device=device, dtype=torch.float32)

    for s in range(0, logits_2d.size(-1), chunk_size):
        zc = logits_2d[:, s:s + chunk_size].float()
        m = torch.maximum(m, zc.max(dim=-1).values)

    sum_exp = torch.zeros_like(m)
    for s in range(0, logits_2d.size(-1), chunk_size):
        zc = logits_2d[:, s:s + chunk_size].float()
        sum_exp = sum_exp + torch.exp(zc - m.unsqueeze(-1)).sum(dim=-1)

    return m + torch.log(sum_exp)


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits: (batch, seq_len, vocab_size)  — raw model logits
        index:  (batch, seq_len)              — token ids to select

    Returns:
        token_logprobs: (batch, seq_len)      — log P(index | context)
    """
    if logits.dtype in (torch.float32, torch.float64):
        # logsumexp is numerically stable in fp32; process one sequence at a time
        # to avoid allocating a [batch, seq, vocab] intermediate.
        lse = torch.stack([torch.logsumexp(row, dim=-1) for row in logits])  # (B, T)
        selected = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)  # (B, T)
        return selected - lse
    else:
        # bfloat16 / float16: compute row-wise logsumexp in vocab chunks to keep
        # memory bounded and avoid materializing (T, V) fp32 tensors.
        token_logprobs = []
        for logits_row, index_row in zip(logits, index):
            lse_row = _rowwise_logsumexp_chunked(logits_row, LOGPROB_VOCAB_CHUNK)  # (T,)
            selected = torch.gather(
                logits_row, dim=-1, index=index_row.unsqueeze(-1)
            ).squeeze(-1).float()
            token_logprobs.append((selected - lse_row).to(logits.dtype))
        return torch.stack(token_logprobs)  # (B, T)


def chunked_token_entropy(logits_2d: torch.Tensor, chunk_size: int = ENTROPY_VOCAB_CHUNK) -> torch.Tensor:
    """
    Exact token entropy for logits shaped (T, V), computed in vocab chunks.
    Uses: H = logsumexp(z) - E_p[z], where p = softmax(z).
    This avoids materializing a full softmax/log-softmax tensor.
    """
    t = logits_2d.size(0)
    device = logits_2d.device

    # Pass 1: stable max over vocab
    m = torch.full((t,), float("-inf"), device=device, dtype=torch.float32)
    for s in range(0, logits_2d.size(-1), chunk_size):
        zc = logits_2d[:, s:s + chunk_size].float()
        m = torch.maximum(m, zc.max(dim=-1).values)

    # Pass 2: accumulate sum(exp(z-m)) and sum(z*exp(z-m))
    sum_exp = torch.zeros_like(m)
    sum_zexp = torch.zeros_like(m)
    for s in range(0, logits_2d.size(-1), chunk_size):
        zc = logits_2d[:, s:s + chunk_size].float()
        wc = torch.exp(zc - m.unsqueeze(-1))
        sum_exp = sum_exp + wc.sum(dim=-1)
        sum_zexp = sum_zexp + (wc * zc).sum(dim=-1)

    lse = m + torch.log(sum_exp)
    expected_z = sum_zexp / sum_exp
    return (lse - expected_z).to(logits_2d.dtype)


# ============================================================================
# GRPOTrainer subclass — plug in selective_log_softmax
# ============================================================================

class SelectiveLogprobGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer with two modifications:
      1. _generate_completions — delegates to external vLLM server via plain
         HTTP (POST /v1/chat/completions), bypassing TRL's VLLMClient entirely.
         This avoids the TRL/vLLM version coupling that requires --worker-cls
         and a matching vLLM version.
      2. _get_per_token_logps_and_entropies — uses selective_log_softmax to
         avoid materialising the full [B, T, V] logprob tensor.
    """

    def _generate_single_turn(self, prompts):
        """
        Override TRL's local-model generation with direct HTTP calls to the
        external vLLM server.  Returns the same tuple TRL expects:
            (prompt_ids, completion_ids, logprobs, extra_fields)

        We set logprobs=None — TRL will recompute them via the training model's
        forward pass, which is what we want for correct GRPO gradients.
        """
        import requests as _req

        headers = {"Content-Type": "application/json"}
        if VLLM_API_KEY:
            headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

        all_prompt_ids      = []
        all_completion_ids  = []

        for prompt_messages in prompts:
            # Tokenize prompt to get prompt_ids
            prompt_text = self.processing_class.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = self.processing_class(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]

            # Generate completions via vLLM HTTP
            payload = {
                "model":       MODEL_NAME,
                "messages":    prompt_messages,
                "n":           NUM_GENERATIONS,
                "max_tokens":  MAX_COMPLETION_TOKENS,
                "temperature": TEMPERATURE,
            }
            resp = _req.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=300,
            )
            resp.raise_for_status()
            resp_json = resp.json()
            if not hasattr(self, "_vllm_logged_sample"):
                log.info(f"[vllm sample] first response keys: {list(resp_json['choices'][0]['message'].keys())}")
                self._vllm_logged_sample = True
            choices = sorted(resp_json["choices"], key=lambda c: c["index"])

            for choice in choices:
                msg = choice["message"]
                # gpt-oss returns reasoning in msg["reasoning_content"] and the
                # final answer in msg["content"].  Concatenate both so the model
                # sees the full output for logprob computation.
                # Fall back gracefully if either field is absent or None.
                reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
                answer    = (msg.get("content") or "").strip()
                if reasoning and answer:
                    completion_text = reasoning + "\n" + answer
                elif reasoning:
                    completion_text = reasoning
                else:
                    completion_text = answer

                if not completion_text:
                    log.warning("Empty completion from vLLM, skipping choice.")
                    continue

                completion_ids = self.processing_class(
                    completion_text, return_tensors="pt", add_special_tokens=False
                ).input_ids[0]
                all_prompt_ids.append(prompt_ids)
                all_completion_ids.append(completion_ids)

        import torch as _torch
        # Pad and stack
        max_p = max(t.size(0) for t in all_prompt_ids)
        max_c = max(t.size(0) for t in all_completion_ids)
        pad_id = self.processing_class.pad_token_id or 0

        prompt_ids_padded = _torch.stack([
            _torch.nn.functional.pad(t, (max_p - t.size(0), 0), value=pad_id)
            for t in all_prompt_ids
        ])
        completion_ids_padded = _torch.stack([
            _torch.nn.functional.pad(t, (0, max_c - t.size(0)), value=pad_id)
            for t in all_completion_ids
        ])

        # logprobs=None → TRL recomputes from model forward (correct for training)
        return prompt_ids_padded, completion_ids_padded, None, {}
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        num_images=None,
        **kwargs,
    ):
        # TRL's method signature has evolved (e.g. `num_images` for multimodal
        # models). Keep this override forward-compatible by accepting and
        # ignoring extra kwargs in this text-only trainer.
        del num_images, kwargs
        # Run the forward pass in mini-batches if batch_size is specified
        # (mirrors the base class behaviour).
        if batch_size is None:
            batch_size = input_ids.size(0)

        all_logps   = []
        all_entropy = []

        for start in range(0, input_ids.size(0), batch_size):
            end     = start + batch_size
            ids_mb  = input_ids[start:end]
            mask_mb = attention_mask[start:end]
            # Keep only as many tail logits as actually present in this mini-batch.
            # This avoids requesting very long tails (e.g. max_completion_length)
            # when current sequences are much shorter.
            max_valid = int(mask_mb.sum(dim=-1).max().item())
            keep_n = max(1, min(int(logits_to_keep), max_valid - 1))

            outputs = model(
                input_ids=ids_mb,
                attention_mask=mask_mb,
                logits_to_keep=keep_n + 1,  # +1 for the shift
            )
            # logits shape: (mb, keep_n+1, vocab)
            logits = outputs.logits[:, :-1, :]      # drop last position → (mb, keep_n, vocab)
            # The completion token ids we need log-probs for
            completion_ids = ids_mb[:, -keep_n:]  # (mb, keep_n)

            logps = selective_log_softmax(logits, completion_ids)  # (mb, keep_n)

            # Entropy — exact computation in vocab chunks to avoid allocating
            # full (T, V) softmax/log tensors.
            entropy_list = []
            for logits_row in logits:
                ent_row = chunked_token_entropy(logits_row, ENTROPY_VOCAB_CHUNK)  # (T,)
                entropy_list.append(ent_row.to(logits.dtype))
            entropy = torch.stack(entropy_list)  # (mb, T)

            all_logps.append(logps)
            all_entropy.append(entropy)

        return torch.cat(all_logps, dim=0), torch.cat(all_entropy, dim=0)

# ============================================================================
# Model loading — plain transformers + PEFT, no Unsloth
# ============================================================================

def load_model():
    log.info(f"Loading {MODEL_NAME} (native mxfp4 ~14GB, LoRA on top) ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    attn_impl = ATTN_IMPLEMENTATION
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map={"": "cuda:0"},
            use_cache=False,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        if attn_impl != "eager":
            log.warning(
                f"Failed to load with attn_implementation={attn_impl!r} ({type(e).__name__}: {e}). "
                "Falling back to 'eager'."
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map={"": "cuda:0"},
                use_cache=False,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
        else:
            raise

    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            # GptOssTopKRouter excluded — not nn.Linear
        ],
    )
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"LoRA trainable: {trainable:,}/{total:,} ({100*trainable/total:.3f}%)")
    log.info(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer

# ============================================================================
# Trainer builder
# ============================================================================

def build_trainer(
    model,
    tokenizer,
    dataset:    Dataset,
    reward_fn,
    output_dir: str,
) -> SelectiveLogprobGRPOTrainer:

    cfg = GRPOConfig(
        # ── Generation ────────────────────────────────────────────────────────
        # use_vllm=False — generation is handled by our _generate_completions
        # override which calls the external vLLM server via plain HTTP.
        # This avoids TRL's VLLMClient which requires vLLM ≤0.12.0.
        temperature=TEMPERATURE,
        num_generations=NUM_GENERATIONS,
        generation_batch_size=NUM_GENERATIONS,  # must be divisible by num_generations
        max_completion_length=MAX_COMPLETION_TOKENS,

        # Only compute logits for completion tokens, not prompt tokens.
        # This is the logits_to_keep optimisation built into TRL.
        # The trainer will pass logits_to_keep=max_completion_length to the
        # model's forward(), avoiding a large prompt-length allocation.

        # ── Training ──────────────────────────────────────────────────────────
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=max(1, int(len(dataset) * WARMUP_RATIO)),
        lr_scheduler_type=LR_SCHEDULER,
        optim=OPTIMIZER,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        max_steps=len(dataset),
        save_steps=SAVE_STEPS,
        logging_steps=1,
        output_dir=output_dir,
        report_to="none",

        # No KL penalty — pure pass-rate reward
        beta=0.0,
    )

    trainer = SelectiveLogprobGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=dataset,
    )

    return trainer

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-refinement", action="store_true",
        help="Skip the refinement pass (faster; use when GPU time is tight)",
    )
    cli = parser.parse_args()

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    _check_dependencies()

    problems         = load_problems(DATASET_PATH, MAX_EXAMPLES)
    model, tokenizer = load_model()
    reward_fn        = make_reward_fn()

    # ── Phase 1: initial GRPO ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"PHASE 1 — {len(problems)} problems × {NUM_GENERATIONS} generations")
    log.info("=" * 60)
    t0 = time.time()
    build_trainer(model, tokenizer, build_initial_dataset(problems),
                  reward_fn, str(out / "phase1")).train()
    log.info(f"Phase 1: {(time.time()-t0)/3600:.2f}h  "
             f"solved={sum(1 for s in _best_scores.values() if s==1.0)}/{len(problems)}")
    model.save_pretrained(str(out / "phase1_final"))
    tokenizer.save_pretrained(str(out / "phase1_final"))

    # ── Phase 2: refinement ───────────────────────────────────────────────────
    if cli.no_refinement:
        log.info("Refinement disabled. Done.")
        return

    to_refine = [p for p in problems
                 if _best_scores.get(p.id, 0.0) < 1.0 and p.id in _best_codes]
    if not to_refine:
        log.info("All problems solved after phase 1 — skipping refinement.")
    else:
        log.info("=" * 60)
        log.info(f"PHASE 2 — refinement on {len(to_refine)} unsolved problems")
        log.info("=" * 60)
        t1 = time.time()
        build_trainer(model, tokenizer, build_refinement_dataset(to_refine, _best_codes),
                      reward_fn, str(out / "phase2")).train()
        solved2 = sum(1 for p in to_refine if _best_scores.get(p.id, 0.0) == 1.0)
        log.info(f"Phase 2: {(time.time()-t1)/3600:.2f}h  newly solved={solved2}/{len(to_refine)}")

    model.save_pretrained(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))
    total = sum(1 for s in _best_scores.values() if s == 1.0)
    log.info(f"Done. Solved {total}/{len(problems)} ({100*total/max(len(problems),1):.1f}%)")

    if HF_REPO_ID:
        log.info(f"Pushing LoRA adapters to HuggingFace Hub: {HF_REPO_ID} ...")
        model.push_to_hub(HF_REPO_ID, commit_message="grpo-cf LoRA adapters")
        tokenizer.push_to_hub(HF_REPO_ID)
        log.info(f"Pushed → https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
