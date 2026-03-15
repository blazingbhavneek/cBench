# =============================================================================
# verify.py — C compilation + test harness, shaped reward function
# Ported from old grpo_cf.py; no TRL dependency.
# =============================================================================

import logging
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    MIN_COMPLETION_TOKENS,
    REWARD_COMPILE,
    REWARD_ERROR_ENGAGE,
    REWARD_LENGTH_PENALTY,
    REWARD_PER_TEST,
    VERIFY_TIMEOUT_S,
    VERIFY_WORKERS,
)

log = logging.getLogger("grpo.verify")

# ── Code extraction ───────────────────────────────────────────────────────────

def extract_code(text: str) -> Optional[str]:
    """Pull the first ```c ... ``` (or ```cpp```) block from model output."""
    for lang in ("c", "cpp", ""):
        m = re.search(rf"```{lang}\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


# ── Single-problem verifier (runs in subprocess) ──────────────────────────────

def _verify_worker(
    code:       str,
    test_cases: List[Dict],
    timeout_s:  int = VERIFY_TIMEOUT_S,
) -> Dict:
    """
    Compile and run against all test cases.
    Returns {"passed": int, "total": int, "error": str | None}
    where error is the gcc stderr if compilation failed, or the first
    test failure description, or None on full pass.
    Runs in a subprocess via ProcessPoolExecutor — must be picklable.
    """
    total = len(test_cases)
    with tempfile.TemporaryDirectory(prefix="grpo_v_") as tmp:
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
            return {"passed": 0, "total": total, "error": "compilation timed out"}

        if cp.returncode != 0:
            # Return actual gcc error — used as Pass 2 context
            return {"passed": 0, "total": total, "error": cp.stderr.strip()}

        # Run test cases
        passed = 0
        first_failure = None
        for i, tc in enumerate(test_cases):
            try:
                rp = subprocess.run(
                    [str(exe)],
                    input=tc["input"],
                    capture_output=True, text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                first_failure = first_failure or f"test {i+1}: timed out after {timeout_s}s"
                # Don't break — count remaining as failed
                continue

            if rp.returncode != 0:
                first_failure = first_failure or f"test {i+1}: runtime error (exit {rp.returncode})"
                continue

            actual   = "\n".join(l.rstrip() for l in rp.stdout.rstrip("\n").split("\n"))
            expected = "\n".join(l.rstrip() for l in tc["output"].rstrip("\n").split("\n"))
            if actual == expected:
                passed += 1
            else:
                first_failure = first_failure or (
                    f"test {i+1}: expected:\n{expected[:200]}\ngot:\n{actual[:200]}"
                )

    return {"passed": passed, "total": total, "error": first_failure}


# ── Batch verifier ────────────────────────────────────────────────────────────

_executor: Optional[ProcessPoolExecutor] = None

def get_executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=VERIFY_WORKERS)
    return _executor


def verify_batch(
    completions: List[str],     # raw model output text, one per completion
    test_cases:  List[Dict],    # same test cases for all completions in this problem
) -> List[Dict]:
    """
    Verify all completions for a single problem in parallel.
    Returns list of {"passed", "total", "error"} dicts, one per completion.
    """
    executor = get_executor()
    futures = []
    for text in completions:
        code = extract_code(text)
        if code is None:
            futures.append(None)
        else:
            futures.append(executor.submit(_verify_worker, code, test_cases))

    results = []
    for i, fut in enumerate(futures):
        if fut is None:
            results.append({"passed": 0, "total": len(test_cases), "error": "no code block found"})
        else:
            try:
                results.append(fut.result(timeout=VERIFY_TIMEOUT_S * len(test_cases) + 60))
            except Exception as e:
                results.append({"passed": 0, "total": len(test_cases), "error": str(e)})
    return results


# ── Shaped reward ─────────────────────────────────────────────────────────────

def compute_reward(
    verify_result:   Dict,
    completion_text: str,
    pass_number:     int = 1,
    error_context:   Optional[str] = None,  # the error shown to model in Pass 2
) -> float:
    """
    Shaped reward. Not flat binary.

    Pass 1 reward:
        compile_bonus + (passed / total) * per_test_reward - length_penalty

    Pass 2 adds:
        error_engage_bonus if thinking tokens reference the specific error

    All values configurable in config.py.
    """
    passed = verify_result["passed"]
    total  = verify_result["total"]
    error  = verify_result["error"]

    # Base: proportion of tests passed
    if total == 0:
        score = 0.0
    else:
        score = (passed / total) * REWARD_PER_TEST

    # Compile bonus: even partial test pass implies compilation succeeded
    if error is None or "compilation" not in str(error):
        if passed > 0 or (error and "test" in str(error)):
            score += REWARD_COMPILE

    # Length penalty for suspiciously short completions
    token_count = len(completion_text.split())
    if token_count < MIN_COMPLETION_TOKENS:
        score -= REWARD_LENGTH_PENALTY * (MIN_COMPLETION_TOKENS - token_count)

    # Pass 2: bonus for engaging with the actual error in the thinking
    if pass_number == 2 and error_context:
        # Extract distinctive tokens from error (function names, line numbers, keywords)
        error_tokens = set(re.findall(r'\b[a-zA-Z_]\w+\b|\b\d+\b', error_context))
        thinking_text = completion_text[:8000]  # check thinking section only
        engaged = sum(1 for tok in error_tokens if tok in thinking_text and len(tok) > 3)
        if engaged >= 2:
            score += REWARD_ERROR_ENGAGE

    return max(0.0, score)


# ── Dependency check ──────────────────────────────────────────────────────────

def check_dependencies():
    """Called at startup. Raises SystemExit if required tools are missing."""
    errors = []

    if subprocess.run(["which", "gcc"], capture_output=True).returncode != 0:
        errors.append("gcc not found  →  apt-get install -y gcc")

    gmp_src = "#include <gmp.h>\nint main(){mpz_t x;mpz_init(x);mpz_clear(x);return 0;}\n"
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(gmp_src)
        fname = f.name
    r = subprocess.run(
        ["gcc", "-std=c11", fname, "-o", "/dev/null", "-lgmp"],
        capture_output=True, text=True,
    )
    os.unlink(fname)
    if r.returncode != 0:
        errors.append("libgmp not found  →  apt-get install -y libgmp-dev")

    uthash_ok = any(p.exists() for p in [
        Path("uthash.h"),
        Path("/usr/include/uthash.h"),
        Path("/usr/local/include/uthash.h"),
    ])
    if not uthash_ok:
        errors.append(
            "uthash.h not found  →  wget -q "
            "https://raw.githubusercontent.com/troydhanson/uthash/master/src/uthash.h"
        )

    if errors:
        log.error("Dependency check FAILED:\n  " + "\n  ".join(errors))
        raise SystemExit(1)

    log.info("Dependencies OK: gcc + libgmp + uthash.h")
