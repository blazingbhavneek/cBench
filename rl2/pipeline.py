# =============================================================================
# pipeline.py — async orchestration: generation ↔ backward ↔ scoring
# =============================================================================

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    KL_COEFF,
    MAX_COMPLETION_TOKENS,
    MAX_SEQ_LEN,
    NUM_GENERATIONS,
    OUTPUT_DIR,
    SAVE_STEPS,
)
from data import ProblemBatch, Problem
from engine import SGLangEngine
from logits import selective_log_softmax
from train import GRPOTrainer, compute_advantages
from verify import compute_reward, verify_batch

log = logging.getLogger("grpo.pipeline")

# tqdm bar format — compact, shows all key stats
_BAR_FMT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"

# ── add near top of pipeline.py ───────────────────────────────────────────────

@torch.inference_mode()
def _ref_logprobs_chunked(
    model:          torch.nn.Module,
    full_ids:       torch.Tensor,   # (1, T_p + T_c)
    token_ids:      torch.Tensor,   # (T_c,)  — the completion token ids
    T_c:            int,
    device:         torch.device,
    vocab_chunk:    int = 4096,
) -> torch.Tensor:
    """
    Full frozen forward → last_hidden_state → chunked lm_head logprob.
    Never materialises (T_c, V) on GPU; peak is (T_c, vocab_chunk) fp32.
    Returns (T_c,) log-probs on CPU.
    """
    base    = model.base_model.model if hasattr(model, 'base_model') else model
    inner   = base.model
    lm_head = base.lm_head   # nn.Linear (V, H), no bias usually

    T   = full_ids.shape[1]
    pos = torch.arange(T, device=device).unsqueeze(0)
    out = inner(input_ids=full_ids, position_ids=pos, use_cache=False)

    # Grab only completion positions, upcast to fp32 immediately
    hidden = out.last_hidden_state[0, -T_c:].float()   # (T_c, H)
    del out
    torch.cuda.empty_cache()

    V          = lm_head.weight.shape[0]
    lse        = torch.full((T_c,), float("-inf"), device=device, dtype=torch.float32)
    tok_logits = torch.zeros(T_c, device=device, dtype=torch.float32)

    for v0 in range(0, V, vocab_chunk):
        v1  = min(v0 + vocab_chunk, V)
        w   = lm_head.weight[v0:v1].float()        # (chunk, H)
        lc  = hidden @ w.T                          # (T_c, chunk)
        if lm_head.bias is not None:
            lc = lc + lm_head.bias[v0:v1]

        # Stable running LSE update
        cmax    = lc.max(dim=-1).values
        new_max = torch.maximum(lse, cmax)
        lse     = new_max + torch.log(
            torch.exp(lse - new_max)
            + torch.exp(lc - new_max.unsqueeze(-1)).sum(dim=-1)
        )

        # Collect logit for the actual token in this vocab shard
        in_chunk = (token_ids >= v0) & (token_ids < v1)
        if in_chunk.any():
            local_ids          = (token_ids[in_chunk] - v0).long()
            t_idx              = in_chunk.nonzero(as_tuple=True)[0]
            tok_logits[t_idx]  = lc[t_idx, local_ids]

        del lc, w

    return (tok_logits - lse).cpu()

class AsyncGRPOPipeline:

    def __init__(self, trainer: GRPOTrainer, engine: SGLangEngine):
        self.trainer = trainer
        self.engine  = engine
        self.device  = torch.device("cuda:0")

        self.best_scores: Dict[str, float] = {}
        self.best_codes:  Dict[str, str]   = {}
        self.best_errors: Dict[str, str]   = {}

        self._optimizer_step_count = 0
        self._total_solved = 0
        self._total_seen   = 0

    # ── Reference logprobs (no grad, cheap) ───────────────────────────────────

    def _compute_ref_logprobs(
        self,
        prompt_ids:     torch.Tensor,   # (1, T_p)
        completion_ids: torch.Tensor,   # (G, T_c)
        pbar:           Optional[tqdm] = None,
    ) -> torch.Tensor:
        G, T_c = completion_ids.shape
        ref_lps = []

        for g in range(G):
            if pbar:
                pbar.set_postfix_str(f"ref logprobs {g+1}/{G}", refresh=True)

            full_ids = torch.cat([prompt_ids, completion_ids[g:g+1]], dim=1)
            lp = _ref_logprobs_chunked(
                self.trainer.model,
                full_ids,
                completion_ids[g],
                T_c,
                self.device,
            )
            ref_lps.append(lp)
            torch.cuda.empty_cache()

        return torch.stack(ref_lps)   # (G, T_c) on CPU

    # ── Per-problem processing ─────────────────────────────────────────────────

    def process_problem(
        self,
        problem:     Problem,
        messages:    List[Dict],
        pass_number: int,
        pbar:        Optional[tqdm] = None,
    ) -> Dict:
        pid = problem.id
        t0  = time.time()

        # ── Phase A: SGLang generates N completions ─────────────────────────
        if pbar:
            pbar.set_postfix_str(f"[{pid[:12]}] generating {NUM_GENERATIONS} completions…", refresh=True)

        completions = self.engine.sample_n(messages, n=NUM_GENERATIONS)
        t_gen = time.time() - t0

        # ── Phase B: Training model forward + sparse buffer ──────────────────
        if pbar:
            pbar.set_postfix_str(f"[{pid[:12]}] training forward…", refresh=True)

        prompt_text   = self.engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tensor = self.engine.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        completion_texts = [c["text"] for c in completions]
        completion_enc   = [
            self.engine.tokenizer.encode(t, add_special_tokens=False)
            for t in completion_texts
        ]
        max_len = min(
            max(len(c) for c in completion_enc),
            MAX_COMPLETION_TOKENS,
            MAX_SEQ_LEN - prompt_tensor.shape[1] - 1,  # leave room for prompt
        )
        max_len = max(max_len, 1)

        pad_id          = self.engine.tokenizer.pad_token_id or 0
        completion_ids  = torch.full((NUM_GENERATIONS, max_len), pad_id,  dtype=torch.long,  device=self.device)
        completion_mask = torch.zeros((NUM_GENERATIONS, max_len),          dtype=torch.float, device=self.device)
        for g, enc in enumerate(completion_enc):
            length = min(len(enc), max_len)
            completion_ids[g,  :length] = torch.tensor(enc[:length], device=self.device)
            completion_mask[g, :length] = 1.0

        # ── Phase B: ref logprobs (no grad, inference_mode) ─────────────────
        if pbar:
            pbar.set_postfix_str(f"[{pid[:12]}] ref logprobs…", refresh=True)

        ref_logprobs = None
        if KL_COEFF > 0.0:
            ref_logprobs = self._compute_ref_logprobs(
                prompt_tensor, completion_ids, pbar=pbar
            )  # (G, T_c) on CPU

        t_fwd = time.time() - t0 - t_gen

        # ── Phase C: Score completions (CPU) ─────────────────────────────────
        if pbar:
            pbar.set_postfix_str(f"[{pid[:12]}] scoring (gcc)…", refresh=True)

        verify_results = verify_batch(completion_texts, problem.test_cases)
        t_score = time.time() - t0 - t_gen - t_fwd

        error_context = self.best_errors.get(pid) if pass_number == 2 else None
        rewards = []
        for g, (vr, text) in enumerate(zip(verify_results, completion_texts)):
            r = compute_reward(
                verify_result=vr,
                completion_text=text,
                pass_number=pass_number,
                error_context=error_context,
            )
            rewards.append(r)

            score = vr["passed"] / max(vr["total"], 1)
            if score > self.best_scores.get(pid, -1.0):
                self.best_scores[pid] = score
                from verify import extract_code
                code = extract_code(text)
                if code:
                    self.best_codes[pid] = code
            if vr["error"] and pid not in self.best_errors:
                self.best_errors[pid] = vr["error"]

        rewards_t  = torch.tensor(rewards, dtype=torch.float, device=self.device)
        advantages = compute_advantages(rewards_t)

        full_pass_n = sum(1 for vr in verify_results if vr["passed"] == vr["total"] and vr["total"] > 0)
        best_r      = max(rewards)
        mean_r      = sum(rewards) / len(rewards)

        log.debug(
            f"[{pid}] pass={pass_number}  "
            f"rewards=[{min(rewards):.2f}..{max(rewards):.2f}]  "
            f"full={full_pass_n}/{NUM_GENERATIONS}  "
            f"gen={t_gen:.1f}s ref={t_fwd:.1f}s score={t_score:.1f}s"
        )

        return {
            "problem_id":      pid,
            "rewards":         rewards_t,
            "advantages":      advantages,
            "prompt_ids":      prompt_tensor.cpu(),   # for backward
            "completion_ids":  completion_ids.cpu(),
            "completion_mask": completion_mask.cpu(),
            "ref_logprobs":    ref_logprobs,           # (G, T_c) CPU or None
            "verify_results":  verify_results,
            "full_pass_n":     full_pass_n,
            "mean_reward":     mean_r,
            "best_reward":     best_r,
            "t_gen":           t_gen,
            "t_fwd":           t_fwd,
            "t_score":         t_score,
        }

    # ── Batch runner ───────────────────────────────────────────────────────────

    def run_batch(self, batch: ProblemBatch, global_pbar: Optional[tqdm] = None) -> Dict:
        problems    = batch.problems
        messages    = batch.messages
        pass_number = batch.pass_number

        batch_rewards  = []
        batch_solved   = 0
        batch_pg_loss  = 0.0
        batch_kl_loss  = 0.0

        # Per-batch tqdm — shows each problem inside the batch
        with tqdm(
            total=len(problems),
            desc=f"  Pass {pass_number} batch",
            unit="problem",
            bar_format=_BAR_FMT,
            leave=False,
        ) as bpbar:

            for i, (problem, msgs) in enumerate(zip(problems, messages)):
                bpbar.set_postfix_str(f"problem {i+1}/{len(problems)}", refresh=True)

                result = self.process_problem(problem, msgs, pass_number, pbar=bpbar)

                # Accumulate gradients — real backward through model
                bpbar.set_postfix_str(f"[{problem.id[:12]}] backward…", refresh=True)
                stats = self.trainer.accumulate(
                    prompt_ids=result["prompt_ids"].to(self.device),
                    completion_ids=result["completion_ids"].to(self.device),
                    advantages=result["advantages"],
                    ref_logprobs=(
                        result["ref_logprobs"].to(self.device)
                        if KL_COEFF > 0.0 and result["ref_logprobs"] is not None else None
                    ),
                    completion_mask=result["completion_mask"].to(self.device),
                )

                batch_rewards.append(result["mean_reward"])
                batch_pg_loss += stats["pg_loss"]
                batch_kl_loss += stats["kl_loss"]
                solved_flag    = self.best_scores.get(problem.id, 0.0) == 1.0
                if solved_flag:
                    batch_solved += 1

                bpbar.set_postfix({
                    "id":      problem.id[:10],
                    "r_mean":  f"{result['mean_reward']:.2f}",
                    "r_best":  f"{result['best_reward']:.2f}",
                    "full":    f"{result['full_pass_n']}/{NUM_GENERATIONS}",
                    "pg":      f"{stats['pg_loss']:.3f}",
                    "t_gen":   f"{result['t_gen']:.0f}s",
                    "t_fwd":   f"{result['t_fwd']:.0f}s",
                })
                bpbar.update(1)

                if global_pbar:
                    self._total_seen += 1
                    self._total_solved = sum(
                        1 for s in self.best_scores.values() if s == 1.0
                    )
                    global_pbar.set_postfix({
                        "solved":   f"{self._total_solved}/{self._total_seen}",
                        "r_mean":   f"{sum(batch_rewards)/len(batch_rewards):.2f}",
                        "pg_loss":  f"{batch_pg_loss/(i+1):.3f}",
                        "step":     self.trainer.step,
                    })
                    global_pbar.update(1)

        # ── Optimizer step ─────────────────────────────────────────────────
        tqdm.write(f"  → optimizer step…")
        opt_stats = self.trainer.optimizer_step()
        self._optimizer_step_count += 1

        tqdm.write(
            f"  ✓ step={opt_stats['step']}  "
            f"loss={opt_stats['mean_loss']:.4f}  "
            f"pg={opt_stats['pg_loss']:.4f}  "
            f"kl={opt_stats['kl_loss']:.4f}  "
            f"lr={opt_stats['lr']:.2e}  "
            f"solved={batch_solved}/{len(problems)}"
        )

        # ── LoRA hot-swap ──────────────────────────────────────────────────
        tmp_path = f"/tmp/grpo_lora_step_{opt_stats['step']}"
        tqdm.write(f"  → saving & hot-swapping LoRA…")
        self.trainer.save_lora(tmp_path)
        self.engine.swap_lora(tmp_path)
        tqdm.write(f"  ✓ LoRA hot-swapped  (step {opt_stats['step']})")

        # ── Checkpoint ────────────────────────────────────────────────────
        if opt_stats["step"] % SAVE_STEPS == 0:
            ckpt = self.trainer.save_checkpoint(f"step_{opt_stats['step']:05d}")
            tqdm.write(f"  ✓ checkpoint → {ckpt}")

        return {
            **opt_stats,
            "n_problems":  len(problems),
            "pass_number": pass_number,
            "solved":      batch_solved,
            "mean_reward": sum(batch_rewards) / max(len(batch_rewards), 1),
        }

    # ── Full phase runner ──────────────────────────────────────────────────────

    def run(self, batch: ProblemBatch) -> List[Dict]:
        problems    = batch.problems
        messages    = batch.messages
        pass_number = batch.pass_number
        n_chunks    = (len(problems) + BATCH_SIZE - 1) // BATCH_SIZE

        all_stats = []

        # Outer tqdm — one tick per problem across all batches
        with tqdm(
            total=len(problems),
            desc=f"Phase {pass_number}",
            unit="problem",
            bar_format=_BAR_FMT,
        ) as gpbar:

            for chunk_idx, start in enumerate(range(0, len(problems), BATCH_SIZE)):
                end   = start + BATCH_SIZE
                chunk = ProblemBatch(
                    problems=problems[start:end],
                    messages=messages[start:end],
                    pass_number=pass_number,
                )

                tqdm.write(
                    f"\n── Batch {chunk_idx+1}/{n_chunks}  "
                    f"problems {start+1}–{min(end, len(problems))}  "
                    f"pass={pass_number} ──"
                )

                stats = self.run_batch(chunk, global_pbar=gpbar)
                all_stats.append(stats)

                tqdm.write(
                    f"── Batch {chunk_idx+1}/{n_chunks} done  "
                    f"solved={stats['solved']}/{stats['n_problems']}  "
                    f"mean_r={stats['mean_reward']:.3f}\n"
                )

        return all_stats
