"""
main.py — GRPO training entry point
====================================
Usage:
    python main.py                    # full run (Phase 1 + Phase 2)
    python main.py --no-refinement    # Phase 1 only
    python main.py --smoke-test       # MAX_EXAMPLES=10, NUM_GENERATIONS=2

Must be run as:
    if __name__ == "__main__":
        main()
because SGLang spawns internal subprocesses and requires the __main__ guard.
"""
import warnings
warnings.filterwarnings("ignore", message="resource_tracker")

import logging
logging.getLogger("multiprocessing.resource_tracker").setLevel(logging.CRITICAL)

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import multiprocessing
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("grpo.main")


def main():
    parser = argparse.ArgumentParser(description="GRPO coding trainer")
    parser.add_argument(
        "--no-refinement", action="store_true",
        help="Skip Phase 2 refinement pass"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Override MAX_EXAMPLES=10, NUM_GENERATIONS=2 for quick testing"
    )
    args = parser.parse_args()

    # Apply smoke-test overrides BEFORE importing config-dependent modules
    if args.smoke_test:
        import config
        config.MAX_EXAMPLES    = 10
        config.NUM_GENERATIONS = 2
        config.BATCH_SIZE      = 5
        log.info("Smoke test mode: MAX_EXAMPLES=10, NUM_GENERATIONS=2, BATCH_SIZE=5")

    # Now import everything (config values are read at import time in some modules)
    from config import (
        DATASET_PATH,
        HF_REPO_ID,
        MAX_EXAMPLES,
        NUM_GENERATIONS,
        OUTPUT_DIR,
    )
    from data import load_problems, build_pass1_batch, build_pass2_batch
    from engine import SGLangEngine
    from pipeline import AsyncGRPOPipeline
    from train import GRPOTrainer
    from verify import check_dependencies

    # ── Preflight ──────────────────────────────────────────────────────────
    log.info("Checking dependencies...")
    check_dependencies()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Load dataset ───────────────────────────────────────────────────────
    log.info(f"Loading problems from {DATASET_PATH}")
    problems = load_problems()
    log.info(f"Loaded {len(problems)} problems")

    # ── Estimate total optimizer steps for LR scheduler ────────────────────
    import config as cfg
    import math
    total_steps = math.ceil(len(problems) / cfg.BATCH_SIZE)
    if not args.no_refinement:
        total_steps *= 2   # rough upper bound including refinement pass

    # ── Boot models ────────────────────────────────────────────────────────
    log.info("Booting training model...")
    trainer = GRPOTrainer(total_steps=total_steps)

    log.info("Booting SGLang engine...")
    engine = SGLangEngine()

    pipeline = AsyncGRPOPipeline(trainer=trainer, engine=engine)

    # ── Phase 1: first attempt ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"PHASE 1 — {len(problems)} problems × {NUM_GENERATIONS} generations")
    log.info("=" * 60)

    t0 = time.time()
    phase1_batch = build_pass1_batch(problems)
    phase1_stats = pipeline.run(phase1_batch)

    solved_p1 = sum(
        1 for p in problems if pipeline.best_scores.get(p.id, 0.0) == 1.0
    )
    log.info(
        f"Phase 1 complete  "
        f"time={( time.time()-t0)/3600:.2f}h  "
        f"solved={solved_p1}/{len(problems)}"
    )

    ckpt_p1 = trainer.save_checkpoint("phase1_final")
    log.info(f"Phase 1 checkpoint → {ckpt_p1}")

    # ── Phase 2: refinement on unsolved problems ───────────────────────────
    if not args.no_refinement:
        to_refine = [
            p for p in problems
            if pipeline.best_scores.get(p.id, 0.0) < 1.0
            and p.id in pipeline.best_codes
            and p.id in pipeline.best_errors
        ]

        if not to_refine:
            log.info("All problems solved after Phase 1 — skipping refinement")
        else:
            log.info("=" * 60)
            log.info(f"PHASE 2 — refinement on {len(to_refine)} unsolved problems")
            log.info("=" * 60)

            t1 = time.time()
            phase2_batch = build_pass2_batch(
                to_refine,
                pipeline.best_codes,
                pipeline.best_errors,
            )
            phase2_stats = pipeline.run(phase2_batch)

            solved_p2 = sum(
                1 for p in to_refine if pipeline.best_scores.get(p.id, 0.0) == 1.0
            )
            log.info(
                f"Phase 2 complete  "
                f"time={(time.time()-t1)/3600:.2f}h  "
                f"newly solved={solved_p2}/{len(to_refine)}"
            )

        ckpt_final = trainer.save_checkpoint("final")
        log.info(f"Final checkpoint → {ckpt_final}")

    # ── Summary ────────────────────────────────────────────────────────────
    total_solved = sum(
        1 for p in problems if pipeline.best_scores.get(p.id, 0.0) == 1.0
    )
    log.info(
        f"Done. Solved {total_solved}/{len(problems)} "
        f"({100*total_solved/max(len(problems),1):.1f}%)  "
        f"total time={(time.time()-t0)/3600:.2f}h"
    )

    # ── HuggingFace Hub push ───────────────────────────────────────────────
    if HF_REPO_ID:
        log.info(f"Pushing to HuggingFace Hub: {HF_REPO_ID}")
        trainer.model.push_to_hub(HF_REPO_ID, commit_message="grpo LoRA adapters")
        trainer.tokenizer.push_to_hub(HF_REPO_ID)
        log.info(f"Pushed → https://huggingface.co/{HF_REPO_ID}")

    engine.shutdown()


if __name__ == "__main__":
    # SGLang spawns internal subprocesses — this guard is mandatory.
    # Without it, each worker re-imports main.py and tries to boot another
    # Engine, causing a multiprocessing bootstrap deadlock.
    multiprocessing.set_start_method("spawn", force=True)
    main()
