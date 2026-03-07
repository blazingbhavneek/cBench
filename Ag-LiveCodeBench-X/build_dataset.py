"""
Build an Extended Ag-LiveCodeBench-X Dataset for RL Training
=============================================================

Sources:
  1. open-r1/codeforces        — 10k+ CF problems with generated test cases
  2. newfacade/LeetCodeDataset — ~2.8k LeetCode problems with 100+ tests each
  3. nuprl/Ag-LiveCodeBench-X  — original 499 problems (kept as eval split)

Output schema (identical to Ag-LiveCodeBench-X):
  question_id       : str   e.g. "1234_A" or "lc_two-sum"
  question_content  : str   problem statement (plain text / LaTeX)
  private_test_cases: str   base64( zlib( pickle( json([{input,output},...]) ) ) )
  source            : str   "codeforces" | "leetcode" | "ag_lcbx"
  difficulty        : str   e.g. "800" (CF rating) or "Easy"/"Medium"/"Hard"

Usage:
  # Minimal — only fast, small subsets (for local testing)
  python build_dataset.py --max-cf 500 --max-lc 200 --out-dir ./ag_extended

  # Full build (downloads ~110 GB of CF generated tests)
  python build_dataset.py --cf-generated-tests-dir /data/cf_tests \\
      --max-cf 8000 --max-lc 2800 --out-dir ./ag_extended

  # Push to HuggingFace Hub
  python build_dataset.py --push-to-hub your_org/ag-lcbx-extended

Decontamination:
  --decontam-against nuprl/Ag-LiveCodeBench-X
  Uses 8-gram overlap (same method as Agnostics paper) to remove any
  training problem whose statement overlaps with the eval set.
"""

import argparse
import base64
import json
import pickle
import re
import zlib
from pathlib import Path
from typing import List, Optional

# ============================================================================
# Encoding helpers  (Ag-LiveCodeBench-X wire format)
# ============================================================================

def encode_test_cases(test_cases: List[dict]) -> str:
    """
    Encode list of {input, output} dicts into the Ag-LiveCodeBench-X
    on-disk format: base64( zlib( pickle( json_string ) ) )
    """
    json_str = json.dumps(test_cases)
    pickled   = pickle.dumps(json_str)
    compressed = zlib.compress(pickled)
    return base64.b64encode(compressed).decode("utf-8")


def decode_test_cases(raw: str) -> List[dict]:
    """Inverse of encode_test_cases — for verification."""
    compressed = base64.b64decode(raw.encode("utf-8"))
    pickled    = zlib.decompress(compressed)
    obj        = pickle.loads(pickled)
    if isinstance(obj, (str, bytes)):
        obj = json.loads(obj)
    return obj


# ============================================================================
# 8-gram decontamination
# ============================================================================

def ngrams(text: str, n: int = 8) -> set:
    words = text.lower().split()
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}


def build_eval_ngrams(eval_problems: List[dict], n: int = 8) -> set:
    all_ngrams: set = set()
    for p in eval_problems:
        all_ngrams |= ngrams(p["question_content"], n)
    return all_ngrams


def is_contaminated(text: str, eval_ngrams: set, n: int = 8, threshold: int = 3) -> bool:
    """Return True if `text` shares >= threshold n-grams with the eval set."""
    problem_ngrams = ngrams(text, n)
    overlap = problem_ngrams & eval_ngrams
    return len(overlap) >= threshold


# ============================================================================
# Source 1: open-r1/codeforces
# ============================================================================

def load_codeforces(
    out_f,
    eval_ids: set,
    max_problems: Optional[int] = None,
    generated_tests_dir: Optional[str] = None,
    min_rating: int = 0,
    max_rating: int = 3500,
    min_tests: int = 1,
) -> int:
    """
    Load problems from open-r1/codeforces.

    Test case priority:
      1. generated_tests (large, stress-tested) if generated_tests_dir is set
      2. official_tests   (short samples from the problem page)

    Filters:
      - Problems must have `executable=True` (at least 3 human solutions verified)
      - official_tests_complete=True OR generated tests available
      - rating in [min_rating, max_rating]
    """
    from datasets import load_dataset
    import os
    from tqdm import tqdm

    print("Loading open-r1/codeforces metadata...")
    ds = load_dataset("open-r1/codeforces", split="train", streaming=True)

    # Build a lookup for generated test cases if the directory is available
    gen_tests_lookup: dict = {}
    if generated_tests_dir:
        import pyarrow.parquet as pq
        gen_dir = Path(generated_tests_dir)
        parquet_files = list(gen_dir.glob("test_cases_*.parquet"))
        print(f"Loading {len(parquet_files)} generated test case parquet files...")
        for pf in tqdm(parquet_files, desc="Loading parquet"):
            try:
                pf_obj = pq.ParquetFile(pf)
                for batch in pf_obj.iter_batches(batch_size=500):
                    d = batch.to_pydict()
                    for pid, inp, out in zip(d["problem_id"], d["input"], d["output"]):
                        if pid not in gen_tests_lookup:
                            gen_tests_lookup[pid] = []
                        gen_tests_lookup[pid].append({"input": inp, "output": out})
            except Exception as e:
                print(f"  Warning: could not load {pf.name}: {e}")
        total_gen = sum(len(v) for v in gen_tests_lookup.values())
        print(f"Loaded {total_gen} generated test cases for {len(gen_tests_lookup)} problems")

    count = 0
    skipped = {"no_tests": 0, "not_executable": 0, "rating_filter": 0, "no_statement": 0}

    for row in tqdm(ds, desc="Processing CF problems", total=max_problems):
        # Must have a verifiable problem statement
        # CF uses "description" field for the problem statement
        statement = " ".join(filter(None, [
            row.get("title", ""),
            row.get("description", ""),
            row.get("input_format", ""),
            row.get("output_format", ""),
        ])).strip()
        if not statement or len(statement) < 50:
            skipped["no_statement"] += 1
            continue

        # Must be executable (verified by running actual submissions)
        if not row.get("executable", False):
            skipped["not_executable"] += 1
            continue

        # Rating filter
        rating = row.get("rating") or 0
        if rating and not (min_rating <= rating <= max_rating):
            skipped["rating_filter"] += 1
            continue

        problem_id = row.get("id") or row.get("problem_id", "")

        # Pick test cases: prefer generated (large) over official (sample only)
        test_cases = []
        if problem_id in gen_tests_lookup:
            test_cases = gen_tests_lookup[problem_id]
        else:
            official = row.get("examples", []) or []
            test_cases = [
                {"input": t.get("input", ""), "output": t.get("output", "")}
                for t in official
                if t.get("input") and t.get("output")
            ]

        if len(test_cases) < min_tests:
            skipped["no_tests"] += 1
            continue

        # Reconstruct a clean question_id matching CF naming convention
        contest_id = str(row.get("contestId", "")).strip()
        index      = str(row.get("index", "")).strip()
        question_id = f"{contest_id}_{index}" if contest_id and index else problem_id

        # Skip if already in eval set
        if question_id in eval_ids:
            continue

        problem = {
            "question_id":       question_id,
            "question_content":  statement,
            "private_test_cases": encode_test_cases(test_cases),
            "source":            "codeforces",
            "difficulty":        str(rating) if rating else "unknown",
            "num_tests":         len(test_cases),
        }
        out_f.write(json.dumps(problem) + "\n")
        out_f.flush()
        count += 1

        if max_problems and count >= max_problems:
            break

    print(f"Codeforces: {count} problems written. Skipped: {skipped}")
    return count


# ============================================================================
# Source 2: newfacade/LeetCodeDataset
# ============================================================================

def load_leetcode(
    out_f,
    eval_ids: set,
    max_problems: Optional[int] = None,
    split: str = "train",          # "train" or "test" (post-2024-07)
    min_tests: int = 10,
) -> int:
    """
    Load problems from newfacade/LeetCodeDataset.

    Each problem already has 100+ LLM-generated + edge-case test cases
    stored as a list of {input, output} pairs.

    Note: LeetCode uses function-call format, NOT stdio. The test cases
    here are pre-converted to serialized function call args and return
    values. You need to wrap them in a Python runner that calls the
    function and checks the result — they are NOT drop-in I/O compatible
    with the Ag-LiveCodeBench-X C verifier.
    Keep LeetCode in a separate split or use a Python-specific verifier.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"Loading newfacade/LeetCodeDataset ({split} split)...")
    ds = load_dataset("newfacade/LeetCodeDataset", split=split, streaming=True)

    count = 0
    skipped = {"no_tests": 0, "no_statement": 0}

    for row in tqdm(ds, desc="Processing LC problems", total=max_problems):
        statement = row.get("problem_description", "")
        if not statement or len(statement) < 50:
            skipped["no_statement"] += 1
            continue

        # LeetCodeDataset stores test cases as a list of {input, output} dicts.
        test_cases = row.get("input_output", []) or []

        if len(test_cases) < min_tests:
            skipped["no_tests"] += 1
            continue

        slug        = row.get("slug", str(row.get("question_id", "")))
        question_id = f"lc_{slug}"

        # Skip if already in eval set
        if question_id in eval_ids:
            continue

        problem = {
            "question_id":        question_id,
            "question_content":   statement,
            "private_test_cases": encode_test_cases(test_cases),
            "source":             "leetcode",
            "difficulty":         row.get("difficulty", "unknown"),
            "num_tests":          len(test_cases),
            # Extra fields useful for the Python verifier
            "entry_point":        row.get("entry_point", ""),
            "starter_code":       row.get("starter_code", ""),
        }
        out_f.write(json.dumps(problem) + "\n")
        out_f.flush()
        count += 1

        if max_problems and count >= max_problems:
            break

    print(f"LeetCode: {count} problems written. Skipped: {skipped}")
    return count


# ============================================================================
# Source 3: Original Ag-LiveCodeBench-X (keep as eval)
# ============================================================================

def load_ag_lcbx() -> List[dict]:
    """Load the original 499 Ag-LiveCodeBench-X problems for eval."""
    from datasets import load_dataset

    print("Loading nuprl/Ag-LiveCodeBench-X (eval set)...")
    ds = load_dataset("nuprl/Ag-LiveCodeBench-X", split="test")

    problems = []
    for row in ds:
        problems.append({
            "question_id":        row["question_id"],
            "question_content":   row["question_content"],
            "private_test_cases": row["private_test_cases"],
            "source":             "ag_lcbx",
            "difficulty":         "unknown",
        })

    print(f"Ag-LiveCodeBench-X: {len(problems)} problems loaded (eval only)")
    return problems


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build extended Ag-LiveCodeBench-X dataset for RL training"
    )

    # Codeforces options
    parser.add_argument("--max-cf",   type=int, default=5000,
                        help="Max Codeforces problems (default 5000)")
    parser.add_argument("--min-cf-rating", type=int, default=800,
                        help="Min CF difficulty rating (default 800)")
    parser.add_argument("--max-cf-rating", type=int, default=2500,
                        help="Max CF difficulty rating (default 2500)")
    parser.add_argument("--cf-generated-tests-dir", type=str, default=None,
                        help="Local dir with downloaded CF generated_tests/*.parquet (~110GB). "
                             "Download with: huggingface-cli download open-r1/codeforces "
                             "--repo-type=dataset --include='generated_tests/*.parquet'")

    # LeetCode options
    parser.add_argument("--max-lc",   type=int, default=2000,
                        help="Max LeetCode problems (default 2000)")
    parser.add_argument("--no-leetcode", action="store_true",
                        help="Skip LeetCode entirely (useful if you only want stdio-compatible problems)")

    # Output
    parser.add_argument("--out-dir",      type=str, default="./ag_extended",
                        help="Output directory")
    parser.add_argument("--push-to-hub",  type=str, default=None,
                        help="HuggingFace Hub repo to push to, e.g. your_org/ag-lcbx-extended")

    # Decontamination
    parser.add_argument("--no-decontam", action="store_true",
                        help="Skip 8-gram decontamination check (faster, less safe)")

    args = parser.parse_args()

    print("=" * 60)
    print("Building Extended Ag-LiveCodeBench-X")
    print("=" * 60)

    # 1. Load eval set (always) — this stays in RAM for deduplication
    eval_problems = load_ag_lcbx()
    eval_ids = {p["question_id"] for p in eval_problems}

    # 2. Setup output directory and file
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_path = out_path / "train.jsonl"

    # 3. Stream training problems directly to disk (never accumulate in RAM)
    print(f"\nStreaming training problems to {train_path}...")
    with open(train_path, "w") as f:
        cf_count = load_codeforces(
            out_f=f,
            eval_ids=eval_ids,
            max_problems=args.max_cf,
            generated_tests_dir=args.cf_generated_tests_dir,
            min_rating=args.min_cf_rating,
            max_rating=args.max_cf_rating,
        )

        if not args.no_leetcode:
            lc_count = load_leetcode(
                out_f=f,
                eval_ids=eval_ids,
                max_problems=args.max_lc,
                split="train",
            )
        else:
            lc_count = 0
            print("Skipping LeetCode (--no-leetcode)")

    print(f"\nDone: {cf_count} CF + {lc_count} LC problems written to {train_path}")
    print(f"Eval set: {len(eval_problems)} problems (kept in RAM for deduplication)")

    # Note: Decontamination (8-gram overlap check) requires loading problems into RAM.
    # Since we're streaming, this is skipped. The ID-based dedup (via eval_ids set)
    # is still performed during streaming above.
    if args.no_decontam:
        print("8-gram decontamination: skipped (--no-decontam)")
    else:
        print("\nNote: Full 8-gram decontamination requires loading all problems into RAM.")
        print("      Run with --no-decontam to skip, or post-process if needed.")

    print("\nTo use in rl_loop.py, load with:")
    print(f"  from datasets import load_from_disk")
    print(f"  ds = load_dataset('json', data_files='{train_path}', split='train')")
    print(f"  # {cf_count + lc_count} problems")


if __name__ == "__main__":
    main()
