"""
Production RL Loop: vLLM + ValidatorPool + GRPO+ for Ag-LiveCodeBench-X
========================================================================
Goal:     Improve code generation on Ag-LiveCodeBench-X
Train on: Codeforces + LeetCode (from build_dataset.py)
Eval on:  Ag-LiveCodeBench-X held-out split
Hardware: Single H200 (80GB), 1TB RAM, everything inside one container

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │  vLLM server           ValidatorPool        GRPO+ trainer    │
  │  (port 8000)           (port 8001)          (this process)   │
  │                                                              │
  │  batch prompts  →  N completions  →  ProcessPoolExecutor     │
  │  ONE batched call   per problem      256 concurrent workers  │
  │  PagedAttention     asyncio.gather   binary rewards          │
  │       ↓                  ↓                  ↓               │
  │  5-10x faster than  ~85 tasks/sec    GRPO+ loss              │
  │  HF model.generate  vs 2/sec Docker  single backward        │
  └──────────────────────────────────────────────────────────────┘

Model: GPT OSS 20B (openai/gpt-oss-20b)
  - Loads in native MXFP4 — base weights NEVER dequantized
  - BF16 LoRA adapters attached on top (~0.5GB trainable)
  - Harmony chat template applied to all prompts
  - MoE-aware target modules: attention + FFN experts (NO router — GptOssTopKRouter
    is a custom module PEFT cannot wrap)

GRPO+ recipe (DeepCoder):
  ✓ No KL penalty       (allows broader exploration)
  ✓ No entropy loss     (standard entropy causes collapse)
  ✓ Overlong filter     (truncated → reward=-1, MASKED in loss)
  ✓ Clip-High=5.0       (raises upper surrogate bound)

Single-GPU mode (--single-gpu-mode):
  NO vLLM. NO second model instance.
  The HF model (with LoRA) does BOTH generation (model.generate) AND the
  backward pass. One model, one process, one GPU.
  Slower generation than vLLM but zero duplicate VRAM cost.
  Use on 96GB H100 where you can't afford two model copies.

  In this mode: do NOT start a vLLM server. --vllm-url is ignored.

Dual-instance mode (default, large VRAM):
  vLLM server handles generation (fast, PagedAttention).
  HF model in this process handles backward pass.
  Two copies of the model in VRAM.
  Use on H200 140GB or multi-GPU.

VRAM budget (H200 140GB, dual-instance):
  GPT OSS 20B MXFP4 base (vLLM)   : ~13GB
  vLLM KV cache at 0.75            : ~90GB
  Trainer copy + LoRA r=64         : ~18GB
  Activations + gradients          : ~10GB
  Total                            : ~131GB ✓

VRAM budget (H100 96GB, --single-gpu-mode):
  GPT OSS 20B MXFP4 + LoRA r=32   : ~18GB
  KV cache (model.generate)        : ~20GB
  Activations + gradients          : ~15GB
  Total                            : ~53GB ✓

Usage:
  # 1. Start validator server
  python validator_server.py --port 8001 --workers 256 --container-name cbench

  # 2. Start vLLM
  vllm serve openai/gpt-oss-20b \
      --api-key sk-dummy \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.75 \
      --enable-prefix-caching \
      --port 8000

  # 3a. Train — dual instance (multi-GPU / large VRAM)
  python rl_loop.py \
      --model openai/gpt-oss-20b \
      --api-key sk-dummy \
      --dataset-path ./ag_extended/train.jsonl \
      --vllm-url http://localhost:8000 \
      --validator-url http://localhost:8001 \
      --group-size 8 --batch-size 16 \
      --max-new-tokens 4096 --lora-rank 64 \
      --num-steps 2000 --max-refinement-rounds 3

  # 3b. Train — single GPU (96GB H100, no second model copy)
  python rl_loop.py \
      --model openai/gpt-oss-20b \
      --api-key sk-dummy \
      --single-gpu-mode \
      --dataset-path ./ag_extended/train.jsonl \
      --vllm-url http://localhost:8000 \
      --validator-url http://localhost:8001 \
      --group-size 8 --batch-size 8 \
      --max-new-tokens 2048 --lora-rank 32 \
      --num-steps 2000 --max-refinement-rounds 3
"""

import argparse
import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer

from problem.solve import SolveProblemWrapper
from problem.refine import RefineProblemWrapper
from prompt.c import C_CRITICAL_CODING_REQUIREMENTS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Problem:
    problem_id: str
    statement:  str
    test_cases: List[Dict]
    source:     str          # "codeforces" | "leetcode" | "ag_lcbx"
    difficulty: str = ""


@dataclass
class Rollout:
    problem:    Problem
    prompt:     str
    completion: str
    truncated:  bool


@dataclass
class ScoredRollout:
    rollout:   Rollout
    reward:    float         # 1.0=pass | 0.0=fail | -1.0=truncated
    log_prob:  float         # sequence log prob from vLLM (used as old_lp in IS ratio)
    advantage: float = 0.0


# ============================================================================
# Prompt — Harmony chat template for GPT OSS
# ============================================================================

SYSTEM_PROMPT = (
    "You are an expert competitive programmer. "
    "Solve problems in C using stdin/stdout. "
    "Always wrap your solution in a ```c ... ``` code block."
)

USER_TEMPLATE = """\
Solve the following competitive programming problem in C.
Write a complete C program that reads from stdin and writes to stdout.
Include all necessary headers. Handle all edge cases.
Put your solution in a ```c ... ``` code block.

Problem:
{statement}
"""

def build_prompt(problem: Problem, tokenizer: AutoTokenizer) -> str:
    """
    Apply chat template. enable_thinking=False prevents Qwen3 5000-token think blocks.
    Falls back to plain string if template unavailable (e.g. gpt-oss tokenizer quirks).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(statement=problem.statement)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        return f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(statement=problem.statement)}"


# ============================================================================
# Dataset loader
# ============================================================================

class DatasetLoader:
    """
    Loads JSONL dataset from build_dataset.py.
    Decodes base64 -> zlib -> pickle -> json test case chain.
    Supports weighted sampling across CF / LeetCode / Ag-LCBX.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        mix: Optional[Dict[str, float]] = None,
    ):
        import base64, pickle, zlib
        from datasets import load_dataset

        ds = load_dataset("json", data_files=dataset_path, split="train")
        log.info(f"Loaded JSON dataset: {dataset_path}")

        self.problems: List[Problem] = []
        self._by_source: Dict[str, List[Problem]] = {}

        for row in ds:
            tcs = self._decode(row["private_test_cases"], base64, pickle, zlib)
            if not tcs:
                continue
            p = Problem(
                problem_id=row["question_id"],
                statement=row["question_content"],
                test_cases=tcs,
                source=row.get("source", "unknown"),
                difficulty=row.get("difficulty", ""),
            )
            self.problems.append(p)
            self._by_source.setdefault(p.source, []).append(p)

        log.info(
            f"Loaded {len(self.problems)} problems: "
            + ", ".join(f"{s}={len(v)}" for s, v in self._by_source.items())
        )
        self.mix = mix or {"codeforces": 0.5, "ag_lcbx": 0.3, "leetcode": 0.2}

    def sample_batch(self, n: int) -> List[Problem]:
        batch = []
        for source, ratio in self.mix.items():
            k    = max(1, round(n * ratio))
            pool = self._by_source.get(source, [])
            if pool:
                batch.extend(random.choices(pool, k=min(k, len(pool))))
        random.shuffle(batch)
        return batch[:n]

    def sample_batch_curriculum(self, n: int, problem_stats: dict) -> List[Problem]:
        """
        Curriculum-based sampling: prioritize unsolved / hard problems.
        never solved  → weight 4.0
        solved in 1   → weight 0.2  (deprioritize — model already handles it)
        solved in 2-3 → weight 2-3  (keep practicing)
        """
        weights = []
        for p in self.problems:
            stats = problem_stats.get(p.problem_id, {})
            if not stats.get("solved"):
                weights.append(4.0)
            else:
                rounds = stats.get("rounds_needed", 1)
                weights.append(max(0.2, float(rounds)))
        return random.choices(self.problems, weights=weights, k=n)

    @staticmethod
    def _decode(raw, base64, pickle, zlib) -> List[Dict]:
        if not raw:
            return []
        try:
            obj = pickle.loads(zlib.decompress(base64.b64decode(raw.encode())))
            if isinstance(obj, (str, bytes)):
                obj = json.loads(obj)
            if not isinstance(obj, list):
                return []
            out = []
            for item in obj:
                if isinstance(item, dict) and "input" in item:
                    out.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append({"input": str(item[0]), "output": str(item[1])})
            return out
        except Exception:
            return []


# ============================================================================
# vLLM client
# ============================================================================

class VLLMClient:
    """
    Wraps vLLM /v1/completions.
    n=group_size in ONE request — server samples all completions internally.
    Returns per-sequence mean logprobs for the IS ratio in GRPO+ loss.
    Auth header is always sent — set --api-key to match vLLM --api-key.
    """

    def __init__(
        self,
        base_url:       str,
        model:          str,
        max_new_tokens: int,
        api_key:        str = "sk-dummy",
    ):
        self.url            = base_url.rstrip("/") + "/v1/completions"
        self.model          = model
        self.max_new_tokens = max_new_tokens
        self.headers        = {"Authorization": f"Bearer {api_key}"}

    async def generate_group(
        self,
        prompt:      str,
        group_size:  int,
        temperature: float = 0.8,
        top_p:       float = 0.95,
    ) -> Tuple[List[str], List[float], List[bool]]:
        payload = {
            "model":       self.model,
            "prompt":      prompt,
            "n":           group_size,
            "max_tokens":  self.max_new_tokens,
            "temperature": temperature,
            "top_p":       top_p,
            "logprobs":    1,   # per-token logprobs for IS ratio
        }
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                self.url, json=payload, headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                data = await resp.json()

        if "error" in data:
            raise RuntimeError(f"vLLM error: {data['error']}")

        completions, log_probs, truncated = [], [], []
        for choice in data["choices"]:
            completions.append(choice["text"])
            lps = [
                x for x in
                (choice.get("logprobs") or {}).get("token_logprobs", [])
                if x is not None
            ]
            log_probs.append(sum(lps) / max(len(lps), 1))
            truncated.append(choice.get("finish_reason") == "length")
        return completions, log_probs, truncated

    async def generate_batch(
        self,
        prompts:     List[str],
        group_size:  int,
        temperature: float = 0.8,
    ) -> List[Tuple[List[str], List[float], List[bool]]]:
        return await asyncio.gather(
            *[self.generate_group(p, group_size, temperature) for p in prompts]
        )


# ============================================================================
# HF generation client — single-GPU mode only
# ============================================================================

# ============================================================================
# ValidatorPool — single HTTP port, ProcessPoolExecutor backend
# ============================================================================

class ValidatorPool:
    """
    Sends rollouts to validator_server.py over one port.
    asyncio.gather fires all requests simultaneously.
    Server-side worker pool (--workers N) handles concurrency.
    """

    def __init__(self, url: str = "http://localhost:8001", timeout: int = 30):
        self.url     = url.rstrip("/") + "/validate"
        self.timeout = timeout
        self._pass   = 0
        self._total  = 0

    @property
    def pass_rate(self) -> float:
        return self._pass / max(self._total, 1)

    def reset_stats(self):
        self._pass = self._total = 0

    async def evaluate_batch(self, rollouts: List[Rollout]) -> List[float]:
        rewards = await asyncio.gather(*[self._one(r) for r in rollouts])
        self._pass  += sum(1 for r in rewards if r == 1.0)
        self._total += len(rewards)
        return list(rewards)

    async def _one(self, rollout: Rollout) -> float:
        if rollout.truncated:
            return -1.0   # overlong penalty; also masked in GRPO+ loss

        payload = {
            "code":       self._extract_code(rollout.completion),
            "test_cases": rollout.problem.test_cases,
            "timeout_s":  self.timeout,
        }
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    self.url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout + 70),
                ) as resp:
                    data = await resp.json()
                    return 1.0 if data.get("result") == "success" else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _extract_code(text: str) -> str:
        for lang in ["c", "cpp", ""]:
            m = re.search(rf"```{lang}\s*\n(.*?)```", text, re.DOTALL)
            if m:
                return m.group(1).strip()
        return text.strip()


# ============================================================================
# Advantage computation
# ============================================================================

def compute_advantages(rewards: List[float], group_size: int) -> List[float]:
    """
    Group-normalised advantages.
    All same reward in group → zero advantage → zero gradient (correct).
    """
    advantages = []
    for i in range(0, len(rewards), group_size):
        g   = torch.tensor(rewards[i : i + group_size], dtype=torch.float32)
        std = g.std()
        if std < 1e-8:
            advantages.extend([0.0] * len(g))
        else:
            advantages.extend(((g - g.mean()) / std).tolist())
    return advantages


# ============================================================================
# GRPO+ loss
# ============================================================================

def compute_grpo_plus_loss(
    model:           torch.nn.Module,
    tokenizer:       AutoTokenizer,
    scored:          List[ScoredRollout],
    epsilon:         float        = 0.2,
    clip_high:       float        = 5.0,
    max_length:      int          = 6144,
    device:          torch.device = torch.device("cuda"),
    single_gpu_mode: bool         = False,
) -> Tuple[torch.Tensor, Dict]:
    """
    L = -mean[ min(r*A, clip(r, 1-eps, clip_high)*A) ]

    single_gpu_mode=True:
      model is None. ratio=exp(new_lp - old_lp) where old_lp came from vLLM.
      We still need a forward pass to get new_lp, so model must be passed even
      in single-GPU mode — it's just the LoRA-only wrapper, not a full base copy.

    vs standard GRPO:
      - no KL term
      - no entropy term
      - clip_high=5.0     → more exploration
      - truncated masked  → reward=-1 already penalises, loss skips them
    """
    loss_terms = []
    n_masked   = 0

    for sr in scored:
        if sr.rollout.truncated:
            n_masked += 1
            continue

        tokens = tokenizer(
            sr.rollout.prompt + sr.rollout.completion,
            return_tensors="pt", truncation=True, max_length=max_length,
        ).to(device)

        prompt_len = tokenizer(
            sr.rollout.prompt,
            return_tensors="pt", truncation=True, max_length=max_length,
        )["input_ids"].shape[1]

        out    = model(**tokens)
        logits = out.logits[0]
        lp     = F.log_softmax(logits, dim=-1)
        tgt    = tokens["input_ids"][0]

        gen_start = prompt_len
        gen_end   = tgt.shape[0]
        if gen_end <= gen_start:
            continue

        lp_slice  = lp[gen_start - 1 : gen_end - 1]
        tgt_slice = tgt[gen_start : gen_end]
        token_lps = lp_slice[torch.arange(lp_slice.shape[0], device=device), tgt_slice]
        new_lp    = token_lps.mean()

        # old_lp from vLLM (0.0 for refinement rounds via wrappers — ratio=1, no IS correction)
        ratio = torch.exp(new_lp - sr.log_prob)
        adv   = torch.tensor(sr.advantage, device=device)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - epsilon, clip_high) * adv
        loss_terms.append(-torch.min(surr1, surr2))

    if not loss_terms:
        return (
            torch.tensor(0.0, device=device, requires_grad=True),
            {"masked": n_masked, "active": 0},
        )

    loss = torch.stack(loss_terms).mean()
    return loss, {"masked": n_masked, "active": len(loss_terms), "loss": loss.item()}


# ============================================================================
# HF generation (single-GPU mode — no vLLM)
# ============================================================================

def hf_generate_group(
    model:       torch.nn.Module,
    tokenizer:   AutoTokenizer,
    prompt:      str,
    group_size:  int,
    max_new_tokens: int,
    temperature: float,
    device:      torch.device,
) -> Tuple[List[str], List[float], List[bool]]:
    """
    Generate group_size completions using HF model.generate().
    Returns (completions, log_probs, truncated) same shape as VLLMClient.
    Model must be in eval() mode before calling.
    log_probs are mean per-token log probs over generated tokens.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=4096).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=group_size,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    completions, log_probs, truncated = [], [], []
    for i in range(group_size):
        gen_ids = outputs.sequences[i][prompt_len:]
        text    = tokenizer.decode(gen_ids, skip_special_tokens=True)
        completions.append(text)

        # compute mean log prob over generated tokens from scores
        # outputs.scores is a tuple of (group_size, vocab) tensors per step
        lps = []
        for step, score in enumerate(outputs.scores):
            if step >= len(gen_ids):
                break
            tok_id = gen_ids[step].item()
            lp     = F.log_softmax(score[i], dim=-1)[tok_id].item()
            lps.append(lp)
        log_probs.append(sum(lps) / max(len(lps), 1))
        truncated.append(len(gen_ids) >= max_new_tokens)

    return completions, log_probs, truncated


# ============================================================================
# Async pipeline — generation + reward
# ============================================================================

async def pipeline_step(
    problems:       List[Problem],
    vllm:           Optional["VLLMClient"],
    validator:      ValidatorPool,
    tokenizer:      AutoTokenizer,
    group_size:     int,
    temperature:    float,
    # single-GPU args — only used when vllm is None
    model:          Optional[torch.nn.Module] = None,
    max_new_tokens: int                       = 2048,
    device:         torch.device              = torch.device("cuda"),
) -> List[ScoredRollout]:
    """Single-round pipeline. vllm=None → HF model.generate (single-GPU mode)."""
    all_rollouts:   List[Rollout] = []
    flat_log_probs: List[float]   = []

    if vllm is not None:
        prompts     = [build_prompt(p, tokenizer) for p in problems]
        gen_results = await vllm.generate_batch(prompts, group_size, temperature)
        for problem, (completions, log_probs, truncated_flags) in zip(problems, gen_results):
            prompt = build_prompt(problem, tokenizer)
            for comp, lp, trunc in zip(completions, log_probs, truncated_flags):
                all_rollouts.append(Rollout(problem=problem, prompt=prompt,
                                            completion=comp, truncated=trunc))
                flat_log_probs.append(lp)
    else:
        # Single-GPU: HF generate, no vLLM
        assert model is not None, "model required in single-GPU mode"
        model.eval()
        for problem in problems:
            prompt = build_prompt(problem, tokenizer)
            completions, log_probs, truncated_flags = hf_generate_group(
                model, tokenizer, prompt, group_size, max_new_tokens, temperature, device
            )
            for comp, lp, trunc in zip(completions, log_probs, truncated_flags):
                all_rollouts.append(Rollout(problem=problem, prompt=prompt,
                                            completion=comp, truncated=trunc))
                flat_log_probs.append(lp)

    rewards    = await validator.evaluate_batch(all_rollouts)
    advantages = compute_advantages(rewards, group_size)
    return [
        ScoredRollout(rollout=r, reward=rew, log_prob=lp, advantage=adv)
        for r, rew, lp, adv in zip(all_rollouts, rewards, flat_log_probs, advantages)
    ]


async def pipeline_step_with_refinement(
    problems:       List[Problem],
    solver:         Optional[SolveProblemWrapper],
    refiner:        Optional[RefineProblemWrapper],
    validator:      ValidatorPool,
    tokenizer:      AutoTokenizer,
    group_size:     int,
    temperature:    float,
    max_rounds:     int              = 3,
    problem_stats:  dict             = None,
    # single-GPU args — set when solver/refiner are None
    model:          Optional[torch.nn.Module] = None,
    max_new_tokens: int              = 2048,
    device:         torch.device    = torch.device("cuda"),
) -> List[ScoredRollout]:
    """
    Iterative refinement pipeline.

    Dual-instance mode (solver/refiner set, model=None):
      Round 0 : SolveProblemWrapper  → vLLM HTTP call
      Round 1+: RefineProblemWrapper → vLLM HTTP call
      log_prob=0.0 (wrappers don't expose logprobs; IS ratio≈1, clipped)

    Single-GPU mode (solver=refiner=None, model set):
      All rounds use hf_generate_group() on the same HF model instance.
      log_probs are exact — computed from generation scores.
      Refinement prompt is built manually (problem + failed attempt).
      No vLLM. No second model copy. One GPU.
    """
    assert (solver is not None) or (model is not None), \
        "Either solver (dual-instance) or model (single-GPU) must be provided"

    single_gpu = model is not None

    all_rollouts:   List[Rollout] = []
    flat_log_probs: List[float]   = []
    all_rewards:    List[float]   = []

    for problem in problems:
        base_prompt  = build_prompt(problem, tokenizer)
        conversation: List[dict] = []

        for round_idx in range(max_rounds):

            # ---- DUAL-INSTANCE: use solver/refiner wrappers ----
            if not single_gpu:
                if round_idx == 0:
                    tasks = [
                        solver.aforward(
                            language="c",
                            question_content=problem.statement,
                            question_id=problem.problem_id,
                        )
                        for _ in range(group_size)
                    ]
                else:
                    best_code = conversation[-1]["code"]
                    tasks = [
                        refiner.aforward(
                            language="c",
                            problem_statement=problem.statement,
                            original_code=best_code,
                            error_feedback={"result": "fail", "stderr": ""},
                            question_id=problem.problem_id,
                        )
                        for _ in range(group_size)
                    ]

                results = await asyncio.gather(*tasks)
                round_rollouts = []
                for res in results:
                    code = res.get("solution") or res.get("refined_code") or ""
                    round_rollouts.append(Rollout(
                        problem=problem,
                        prompt=base_prompt,
                        completion=f"```c\n{code}\n```" if code else "",
                        truncated=(not code),
                    ))
                    flat_log_probs.append(0.0)

            # ---- SINGLE-GPU: HF model.generate, same instance ----
            else:
                if round_idx == 0:
                    prompt = base_prompt
                else:
                    # Build refinement prompt manually
                    best_code = conversation[-1]["code"]
                    refine_messages = [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": USER_TEMPLATE.format(
                            statement=problem.statement)},
                        {"role": "assistant", "content": best_code},
                        {"role": "user",      "content": (
                            "Your solution failed the test cases. "
                            "Analyse what went wrong and write a corrected C solution."
                        )},
                    ]
                    try:
                        prompt = tokenizer.apply_chat_template(
                            refine_messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False,
                        )
                    except Exception:
                        prompt = base_prompt  # fallback

                model.eval()
                completions, log_probs, truncated_flags = hf_generate_group(
                    model, tokenizer, prompt, group_size,
                    max_new_tokens, temperature, device,
                )
                round_rollouts = []
                for comp, lp, trunc in zip(completions, log_probs, truncated_flags):
                    round_rollouts.append(Rollout(
                        problem=problem,
                        prompt=prompt,
                        completion=comp,
                        truncated=trunc,
                    ))
                    flat_log_probs.append(lp)

            rewards = await validator.evaluate_batch(round_rollouts)
            all_rollouts.extend(round_rollouts)
            all_rewards.extend(rewards)

            solved = any(r == 1.0 for r in rewards)

            if problem_stats is not None:
                prev = problem_stats.get(problem.problem_id, {})
                problem_stats[problem.problem_id] = {
                    "solved":        solved or prev.get("solved", False),
                    "rounds_needed": round_idx + 1,
                }

            if solved:
                break

            best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
            conversation.append({
                "code":  round_rollouts[best_idx].completion,
                "error": {"result": "fail", "stderr": ""},
            })

    advantages = compute_advantages(
        all_rewards,
        len(all_rewards) // max(len(problems), 1),
    )

    return [
        ScoredRollout(rollout=r, reward=rew, log_prob=lp, advantage=adv)
        for r, rew, lp, adv in zip(all_rollouts, all_rewards, flat_log_probs, advantages)
    ]


# ============================================================================
# Model loading
# ============================================================================

def _detect_target_modules(model) -> List[str]:
    """
    Finds all PEFT-compatible linear layers.

    IMPORTANT: 'router' and 'gate' are intentionally excluded.
    GptOssTopKRouter is a custom module that PEFT cannot wrap
    (not a subclass of torch.nn.Linear). Including it causes:
      ValueError: Target module GptOssTopKRouter() is not supported.
    """
    candidates = {
        # Attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        "qkv_proj", "out_proj",
        "query_key_value", "Wqkv",
        # MLP / Dense FFN
        "gate_proj", "up_proj", "down_proj",
        "fc1", "fc2",
        "dense_h_to_4h", "dense_4h_to_h", "dense",
        # MoE expert weights
        "w1", "w2", "w3",
        "shared_expert",
        # NOTE: "router" and "gate" deliberately omitted —
        # GptOssTopKRouter is not a Linear and PEFT will crash.
    }
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    found   = list(candidates & present)
    if not found:
        found = ["q_proj", "v_proj"]
        log.warning("No known layers found — defaulting to q_proj + v_proj")

    attn    = [m for m in found if any(x in m for x in ["proj","qkv","Wqkv","query","dense"])]
    ffn     = [m for m in found if any(x in m for x in ["gate","up","down","fc","h_to","4h"])]
    experts = [m for m in found if any(x in m for x in ["w1","w2","w3","expert"])]
    log.info(f"LoRA targets -- attn:{attn}  ffn:{ffn}  experts:{experts}")
    return found


def load_model(args) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loading paths:

    --quantize auto (default, production):
      Calls from_pretrained with no quantization_config.
      The kernels package handles MXFP4 transparently when installed.
      DO NOT pass quantization_config — it will be None for gpt-oss and
      transformers will crash trying to call .get() on None.

    --quantize 4bit / 8bit (dev / small GPU):
      BitsAndBytes. Works on any CUDA GPU.

    --single-gpu-mode:
      Sets gpu-memory-utilization=0.55 recommendation (vLLM side).
      Trainer loads same model but PEFT LoRA only — base is frozen.
      See docstring on LoRAOnlyModel above.
    """
    from peft import LoraConfig, get_peft_model

    log.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "eager"
    if torch.cuda.is_available():
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            pass
    log.info(f"Attention: {attn_impl}")

    quant_cfg  = None
    load_dtype: Any = "auto"

    if args.quantize == "4bit":
        from transformers import BitsAndBytesConfig
        quant_cfg  = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_dtype = None
        log.info("Quantization: 4-bit NF4")
    elif args.quantize == "8bit":
        from transformers import BitsAndBytesConfig
        quant_cfg  = BitsAndBytesConfig(load_in_8bit=True)
        load_dtype = None
        log.info("Quantization: 8-bit")
    elif args.quantize == "none":
        load_dtype = None
        log.info("No quantization")
    else:
        # "auto" — native MXFP4 for gpt-oss-20b via kernels package
        # Do NOT pass quantization_config — gpt-oss config.quantization_config
        # is handled internally by transformers + kernels, passing it explicitly
        # causes AttributeError: 'NoneType'.get()
        log.info("Quantization: auto (native MXFP4 via kernels package)")

    log.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_cfg,   # None for auto/none — intentional
        torch_dtype=load_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    if torch.cuda.is_available():
        log.info(f"Base loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    else:
        log.info("Base loaded on CPU.")

    if args.use_fsdp:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            use_orig_params=True,
        )
        log.info("FSDP enabled (multi-GPU full fine-tuning)")
        return model, tokenizer

    for param in model.parameters():
        param.requires_grad = False

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.enable_input_require_grads()

    targets = _detect_target_modules(model)
    lora_rank = args.lora_rank
    if args.single_gpu_mode:
        # Halve rank in single-GPU mode to save activation memory
        lora_rank = max(8, lora_rank // 2)
        log.info(f"Single-GPU mode: LoRA rank reduced to {lora_rank}")

    model = get_peft_model(model, LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    ))

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"LoRA trainable: {trainable:,}/{total:,} ({100*trainable/total:.3f}%)")
    if torch.cuda.is_available():
        log.info(f"VRAM after LoRA: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    return model, tokenizer


# ============================================================================
# Trainer
# ============================================================================

class ProductionTrainer:

    def __init__(self, model, tokenizer, vllm, validator, dataset, args, device):
        self.model     = model
        self.tokenizer = tokenizer
        self.validator = validator
        self.dataset   = dataset
        self.args      = args
        self.device    = device
        self.problem_stats: dict = {}

        self.solver = SolveProblemWrapper(
            base_url=args.vllm_url + "/v1",
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            critical_coding_requirements=C_CRITICAL_CODING_REQUIREMENTS,
        )
        self.refiner = RefineProblemWrapper(
            base_url=args.vllm_url + "/v1",
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            critical_coding_requirements=C_CRITICAL_CODING_REQUIREMENTS,
        )

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.num_steps, eta_min=args.learning_rate * 0.1,
        )

    async def train(self):
        args     = self.args
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        best_pass = 0.0

        log.info("=" * 60)
        log.info(f"Model:       {args.model}")
        log.info(f"Group:       {args.group_size}  Batch: {args.batch_size}")
        log.info(f"Steps:       {args.num_steps}")
        log.info(f"Refinements: {args.max_refinement_rounds}")
        log.info(f"vLLM:        {args.vllm_url}")
        log.info(f"Validator:   {args.validator_url}")
        log.info(f"Single-GPU:  {args.single_gpu_mode}")
        log.info("=" * 60)

        for step in range(args.num_steps):
            t0       = time.time()
            problems = self.dataset.sample_batch_curriculum(
                args.batch_size, self.problem_stats
            )

            self.model.eval()
            if self.args.single_gpu_mode:
                # Single-GPU: ONE model — HF generate + backprop, no vLLM
                scored = await pipeline_step_with_refinement(
                    problems=problems,
                    solver=None,
                    refiner=None,
                    validator=self.validator,
                    tokenizer=self.tokenizer,
                    group_size=args.group_size,
                    temperature=args.temperature,
                    max_rounds=args.max_refinement_rounds,
                    problem_stats=self.problem_stats,
                    model=self.model,
                    max_new_tokens=args.max_new_tokens,
                    device=self.device,
                )
            else:
                # Dual-instance: vLLM generates, HF model does backward
                scored = await pipeline_step_with_refinement(
                    problems=problems,
                    solver=self.solver,
                    refiner=self.refiner,
                    validator=self.validator,
                    tokenizer=self.tokenizer,
                    group_size=args.group_size,
                    temperature=args.temperature,
                    max_rounds=args.max_refinement_rounds,
                    problem_stats=self.problem_stats,
                )

            self.model.train()
            self.optimizer.zero_grad()
            loss, metrics = compute_grpo_plus_loss(
                model=self.model,
                tokenizer=self.tokenizer,
                scored=scored,
                epsilon=args.epsilon,
                clip_high=args.clip_high,
                max_length=args.max_prompt_tokens + args.max_new_tokens,
                device=self.device,
                single_gpu_mode=args.single_gpu_mode,
            )
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                self.optimizer.step()
                self.scheduler.step()

            rewards   = [sr.reward for sr in scored]
            pass_rate = sum(1 for r in rewards if r == 1.0) / max(len(rewards), 1)
            trunc_r   = sum(1 for r in rewards if r == -1.0) / max(len(rewards), 1)
            log.info(
                f"Step {step:4d} | loss={loss.item():.4f} | "
                f"pass@1={pass_rate:.2%} | trunc={trunc_r:.2%} | "
                f"active={metrics['active']} masked={metrics['masked']} | "
                f"lr={self.scheduler.get_last_lr()[0]:.2e} | "
                f"t={time.time()-t0:.1f}s"
            )

            if (step + 1) % args.save_every == 0:
                ckpt = save_dir / f"step_{step+1}"
                ckpt.mkdir(exist_ok=True)
                self.model.save_pretrained(ckpt)
                self.tokenizer.save_pretrained(ckpt)
                log.info(f"Saved -> {ckpt}")
                if pass_rate > best_pass:
                    best_pass = pass_rate
                    self.model.save_pretrained(save_dir / "best")
                    log.info(f"New best: {best_pass:.2%}")

            if (step + 1) % args.eval_every == 0:
                await self._eval_ag_lcbx()

        log.info("Training complete.")

    async def _eval_ag_lcbx(self):
        """Greedy eval on held-out Ag-LiveCodeBench-X problems."""
        eval_problems = self.dataset._by_source.get("ag_lcbx", [])
        if not eval_problems:
            return
        log.info(f"Evaluating {len(eval_problems)} Ag-LCBX problems (greedy)...")
        self.model.eval()
        prompts = [build_prompt(p, self.tokenizer) for p in eval_problems]

        if self.args.single_gpu_mode:
            # Single-GPU: HF generate directly
            all_completions, all_lps, all_truncs = [], [], []
            for prompt in prompts:
                comps, lps, truncs = hf_generate_group(
                    self.model, self.tokenizer, prompt,
                    group_size=1, max_new_tokens=self.args.max_new_tokens,
                    temperature=0.0, device=self.device,
                )
                all_completions.append(comps[0])
                all_truncs.append(truncs[0])
            gen_results = [(([c], [], [t])) for c, t in zip(all_completions, all_truncs)]
        else:
            gen_results = await self.vllm.generate_batch(prompts, group_size=1, temperature=0.0)

        rollouts = [
            Rollout(
                problem=p,
                prompt=prompt,
                completion=comps[0],
                truncated=trunc[0],
            )
            for p, prompt, (comps, _, trunc) in zip(eval_problems, prompts, gen_results)
        ]
        rewards   = await self.validator.evaluate_batch(rollouts)
        passed    = sum(1 for r in rewards if r == 1.0)
        pass_rate = passed / max(len(rewards), 1)
        log.info(f"Ag-LCBX Pass@1: {pass_rate:.2%} ({passed}/{len(rewards)})")
        self.model.train()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Production RL: vLLM + ValidatorPool + GRPO+ for Ag-LiveCodeBench-X"
    )
    p.add_argument("--model",             default="openai/gpt-oss-20b")
    p.add_argument("--api-key",           default="sk-dummy",
                   help="API key sent to vLLM (must match --api-key on vllm serve)")
    p.add_argument("--vllm-url",          default="http://localhost:8000")
    p.add_argument("--validator-url",     default="http://localhost:8001")
    p.add_argument("--max-prompt-tokens", type=int,   default=4096)
    p.add_argument("--max-new-tokens",    type=int,   default=2048)
    p.add_argument("--dataset-path",      required=True)
    p.add_argument("--cf-ratio",          type=float, default=0.5)
    p.add_argument("--lc-ratio",          type=float, default=0.2)
    p.add_argument("--ag-ratio",          type=float, default=0.3)
    p.add_argument("--validator-timeout", type=int,   default=30)
    p.add_argument("--group-size",        type=int,   default=8)
    p.add_argument("--batch-size",        type=int,   default=16)
    p.add_argument("--learning-rate",     type=float, default=1e-6)
    p.add_argument("--epsilon",           type=float, default=0.2)
    p.add_argument("--clip-high",         type=float, default=5.0)
    p.add_argument("--temperature",       type=float, default=0.8)
    p.add_argument("--top-p",             type=float, default=0.95)
    p.add_argument("--num-steps",         type=int,   default=2000)
    p.add_argument("--lora-rank",         type=int,   default=64)
    p.add_argument("--quantize",          default="auto",
                   choices=["auto", "4bit", "8bit", "none"],
                   help=(
                       "auto = native MXFP4 via kernels package (gpt-oss-20b, H100+). "
                       "4bit = BitsAndBytes NF4. "
                       "8bit = BitsAndBytes 8-bit. "
                       "none = full BF16."
                   ))
    p.add_argument("--use-fsdp",          action="store_true",
                   help="Multi-GPU full fine-tuning via FSDP (torchrun)")
    p.add_argument("--single-gpu-mode",   action="store_true",
                   help=(
                       "Single GPU: NO vLLM server needed. "
                       "HF model does both generation (model.generate) AND backward pass. "
                       "One model instance, one process. Slower generation but half the VRAM."
                   ))
    p.add_argument("--save-dir",          default="./checkpoints")
    p.add_argument("--save-every",        type=int,   default=100)
    p.add_argument("--eval-every",        type=int,   default=50)
    p.add_argument("--max-refinement-rounds", type=int, default=3,
                   help="Max refinement rounds per problem per RL step")
    p.add_argument("--dev-mode",          action="store_true",
                   help="Local PC testing: small model, group=2, batch=2, 20 steps")
    args = p.parse_args()

    if args.dev_mode:
        args.group_size = args.group_size if args.group_size != 8    else 2
        args.batch_size = args.batch_size if args.batch_size != 16   else 2
        args.num_steps  = args.num_steps  if args.num_steps  != 2000 else 20
        args.save_every = args.save_every if args.save_every != 100  else 10
        args.eval_every = args.eval_every if args.eval_every != 50   else 10
        log.info("DEV MODE: small model + minimal batch")

    if args.use_fsdp and args.single_gpu_mode:
        raise ValueError("--use-fsdp and --single-gpu-mode are incompatible")

    if args.single_gpu_mode:
        log.info(
            "SINGLE-GPU MODE: no vLLM needed. "
            "HF model handles generation + backward. --vllm-url is ignored."
        )

    return args


async def main():
    args      = parse_args()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset   = DatasetLoader(
        args.dataset_path, split="train",
        mix={"codeforces": args.cf_ratio, "leetcode": args.lc_ratio, "ag_lcbx": args.ag_ratio},
    )
    vllm      = None if args.single_gpu_mode else \
                VLLMClient(args.vllm_url, args.model, args.max_new_tokens, args.api_key)
    validator = ValidatorPool(args.validator_url, args.validator_timeout)
    model, tokenizer = load_model(args)
    trainer   = ProductionTrainer(model, tokenizer, vllm, validator, dataset, args, device)
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
