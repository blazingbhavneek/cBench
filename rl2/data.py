# =============================================================================
# data.py — dataset loading, problem dataclass, message formatters
# Ported from old grpo_cf.py; no TRL dependency.
# =============================================================================

import base64
import json
import logging
import pickle
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import DATASET_PATH, MAX_EXAMPLES, REASONING_EFFORT

log = logging.getLogger("grpo.data")

# ── System prompt ─────────────────────────────────────────────────────────────

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
    HASH_ADD_INT(table, key, e);
    HASH_FIND_INT(table, &k, e);
    HASH_ITER(hh, table, e, tmp) { HASH_DEL(table, e); free(e); }

  Big integers (GMP):
    mpz_t a, b, res;
    mpz_inits(a, b, res, NULL);
    mpz_set_str(a, "99999999999999999999999999999", 10);
    mpz_add(res, a, b);
    gmp_printf("%Zd\\n", res);
    mpz_clears(a, b, res, NULL);

  Sorting (qsort):
    int cmp_int(const void *a, const void *b) { return *(int*)a - *(int*)b; }
    qsort(arr, n, sizeof(int), cmp_int);

I/O:
  scanf("%d", &n);   scanf("%lld", &x);   scanf("%s", buf);
  printf("%d\\n", ans);

COMPILATION: gcc -std=c11 -O2 -o sol sol.c -lm -lgmp
Always return 0 from main. Free all malloc'd memory. mpz_clears all GMP vars.
"""


# ── Problem dataclass ─────────────────────────────────────────────────────────

@dataclass
class Problem:
    id:         str
    statement:  str
    test_cases: List[Dict]


# ── Test case encoding (compact storage in dataset rows) ──────────────────────

def encode_tcs(tcs: List[Dict]) -> str:
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tcs)))).decode()


def decode_tcs(raw: str) -> List[Dict]:
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


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_problems(
    path: str = DATASET_PATH,
    max_n: Optional[int] = MAX_EXAMPLES,
) -> List[Problem]:
    problems = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            tcs = decode_tcs(row.get("private_test_cases", ""))
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


# ── Message builders ──────────────────────────────────────────────────────────

def solve_messages(statement: str) -> List[Dict]:
    """Pass 1: first attempt."""
    return [
        {"role": "system",  "content": SYSTEM},
        {"role": "user",    "content": f"Solve this problem in C:\n\n{statement}"},
    ]


def refine_messages(
    statement:  str,
    best_code:  str,
    error_text: str,
) -> List[Dict]:
    """
    Pass 2: refinement with real compiler/test error injected.
    The error_text is the actual gcc output or test failure message —
    not hallucinated. This forces the thinking tokens to engage with
    the real failure rather than imagining one.
    """
    return [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": f"Solve this problem in C:\n\n{statement}"},
        {"role": "assistant", "content": f"```c\n{best_code}\n```"},
        {"role": "user",      "content": (
            f"Your solution failed with the following error:\n\n"
            f"```\n{error_text.strip()}\n```\n\n"
            f"Re-examine the logic carefully and write a corrected C solution."
        )},
    ]


# ── Batch builders ────────────────────────────────────────────────────────────

@dataclass
class ProblemBatch:
    """A batch of problems ready for generation + training."""
    problems:    List[Problem]
    messages:    List[List[Dict]]   # one per problem
    pass_number: int                # 1 or 2


def build_pass1_batch(problems: List[Problem]) -> ProblemBatch:
    return ProblemBatch(
        problems=problems,
        messages=[solve_messages(p.statement) for p in problems],
        pass_number=1,
    )


def build_pass2_batch(
    problems:    List[Problem],
    best_codes:  Dict[str, str],
    best_errors: Dict[str, str],
) -> ProblemBatch:
    """
    Only includes problems that have a prior failed attempt with a real
    error message. Problems without both are skipped.
    """
    filtered_problems = []
    filtered_messages = []
    for p in problems:
        code  = best_codes.get(p.id)
        error = best_errors.get(p.id)
        if code is None or error is None:
            continue
        filtered_problems.append(p)
        filtered_messages.append(refine_messages(p.statement, code, error))

    log.info(
        f"Pass 2 batch: {len(filtered_problems)}/{len(problems)} problems "
        f"have prior attempts with real errors"
    )
    return ProblemBatch(
        problems=filtered_problems,
        messages=filtered_messages,
        pass_number=2,
    )
