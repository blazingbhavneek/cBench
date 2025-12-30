#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dspy==3.0.0b2",
#     "datasets==4.0.*",
#     "bounded-subprocess==2.3.1",
#     "abstractions>=0.4.0",
# ]
# [tool.uv]
# exclude-newer = "2025-08-05T00:00:00Z"
# ///
"""
This script runs LiveCodeBench-X. To understand how this relates to LiveCodeBench,
you can read prepare_lcbx.py. But you will not need to run that script:
a prepared dataset is already available on the Hugging Face Hub.

You can use this script to evaluate LiveCodeBench-X on any programming language.
You can use one of the Agnostics verifier containers, or you can write your own.
The container must follow the Agnostics protocol, which communicates over JSON on
the standard input and output.

An input *line* has the shape:

```
{ "code": str, "timeout_s": int, "test_cases": [ { "input": str, "output": str }, ... ] }
```

An output *line* has the shape:

```
{ result: 'success'; stderr: string } |
{ result: 'fail:wrong-output'; expected: string; got: string; stderr: string } |
{ result: 'fail:error'; exit_code: number; stdout: string; stderr: string } |
{ result: 'fail:timeout'; stdout: string; stderr: string } |
{ result: 'fail:other'; stdout: string; stderr: string }
```

The code will only check for the "success" value and record the entire line of output. So, it is okay to have other
error codes and other fields.

Here is a simple usage example:

```
uv run livecodebench_x.py \
    completions \
    --model-name openai/qwen3_8b_awq \
    --completions-path completions.jsonl \
    --temperature 0.2 \
    --num-concurrent 50 \
    --max-tokens 2048 \
    --language "Lua /nothink"

uv run livecodebench_x.py executions \
    --container-name ghcr.io/nuprl/agnostics:lua \
    --timeout-seconds 15 \
    --generations-path completions.jsonl \
    --executions-path executions.jsonl \
    --num-concurrent 50

uv run livecodebench_x.py pass1 executions.jsonl
```

Notice that the model name is in LiteLLM format. It is up to you to ensure that the container supports the specified
language, which is only fed to the prompt. You can add more directions to the language if needed, e.g.,
a version number.

The settings above are the recommended settings for Qwen 3 models, based on an experiment with Julia. If you get many
"LM response was truncated" warnings, you may have forgotten to add "/nothink" to the --language argument to
disable thinking. (A few warnings are inevitable due to repetitive generations.)
"""

import argparse
import asyncio
import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import AsyncIterator, Awaitable, List, Optional, Tuple, TypedDict

import datasets
import dspy
from abstractions.async_abstractions import run_bounded
from abstractions.storage import (
    map_by_key_jsonl_file,
    run_bounded_create_or_resume_jsonl_file,
)
from bounded_subprocess.bounded_subprocess_async import run
from tqdm.auto import tqdm


def decompress_lcb_private_tests(text: str):
    """
    LiveCodeBench compresses its private tests because they are enormous (8GB
    when we write our 499 problem subset to disk).
    """
    return json.loads(
        pickle.loads(zlib.decompress(base64.b64decode(text.encode("utf-8"))))
    )


class Candidate(TypedDict):
    question_id: str
    solution: str
    reasoning: str


class Result(TypedDict):
    result: str
    question_id: str
    solution: str
    raw_exit_code: int
    raw_stdout: str
    raw_stderr: str


class SolveProblem(dspy.Signature):
    """
    Solve the following programming problem using C with GLib support.

    CRITICAL REQUIREMENTS:

    1. INCLUDES - You MUST include ALL necessary headers:
       Standard C headers:
       - #include <stdio.h>      // for printf, scanf, FILE operations
       - #include <stdlib.h>     // for malloc, free, atoi, qsort
       - #include <string.h>     // for strlen, strcmp, strcpy, memset
       - #include <stdbool.h>    // for bool, true, false
       - #include <math.h>       // for sqrt, pow, floor, ceil
       - #include <limits.h>     // for INT_MAX, INT_MIN
       - #include <ctype.h>      // for isdigit, isalpha, tolower

       GLib headers for data structures:
       - #include <glib.h>       // for GHashTable, GArray, GQueue, GList, etc.

    2. DATA STRUCTURES - Use GLib for complex data structures:

       Hash Table (Dictionary/Map):
       - GHashTable *hash = g_hash_table_new(g_direct_hash, g_direct_equal);  // for integers as keys
       - GHashTable *hash = g_hash_table_new(g_str_hash, g_str_equal);        // for strings as keys
       - g_hash_table_insert(hash, GINT_TO_POINTER(key), GINT_TO_POINTER(value));
       - gpointer value = g_hash_table_lookup(hash, GINT_TO_POINTER(key));
       - int val = GPOINTER_TO_INT(value);
       - g_hash_table_destroy(hash);

       Dynamic Array:
       - GArray *arr = g_array_new(FALSE, FALSE, sizeof(int));
       - g_array_append_val(arr, value);
       - int val = g_array_index(arr, int, index);
       - g_array_free(arr, TRUE);

       Queue:
       - GQueue *queue = g_queue_new();
       - g_queue_push_tail(queue, GINT_TO_POINTER(value));
       - gpointer val = g_queue_pop_head(queue);
       - g_queue_free(queue);

       List (Linked List):
       - GList *list = NULL;
       - list = g_list_append(list, GINT_TO_POINTER(value));
       - GList *node = g_list_first(list);
       - g_list_free(list);

    3. INPUT/OUTPUT FORMAT:
       - Input comes from STDIN using scanf()
       - Output goes to STDOUT using printf()
       - Read integers: scanf("%d", &n);
       - Read strings: char str[1000]; scanf("%s", str);
       - Read line: fgets(str, sizeof(str), stdin);
       - Print integer: printf("%d\n", result);
       - Print string: printf("%s\n", str);
       - Always add newline at the end of output
       - Match output format EXACTLY as specified in the problem

    4. MEMORY MANAGEMENT:
       - Always free dynamically allocated memory
       - GLib structures have their own free functions (g_hash_table_destroy, g_array_free, etc.)
       - Regular malloc() requires free()
       - Avoid memory leaks

    5. CODE STRUCTURE:
       - Write a complete, runnable C program
       - Always include a main() function that returns int
       - Return 0 from main() on success
       - Handle edge cases (empty input, boundary values, etc.)
       - Initialize all variables before use

    6. COMMON PATTERNS:

       Reading multiple integers:
       int n;
       scanf("%d", &n);
       int arr[n];  // or use GArray for dynamic sizing
       for (int i = 0; i < n; i++) {
           scanf("%d", &arr[i]);
       }

       String processing:
       char str[1000];
       scanf("%s", str);
       int len = strlen(str);

       Using hash map for counting:
       GHashTable *count = g_hash_table_new(g_direct_hash, g_direct_equal);
       int val = GPOINTER_TO_INT(g_hash_table_lookup(count, GINT_TO_POINTER(key)));
       g_hash_table_insert(count, GINT_TO_POINTER(key), GINT_TO_POINTER(val + 1));

       Sorting:
       int compare(const void *a, const void *b) {
           return (*(int*)a - *(int*)b);
       }
       qsort(arr, n, sizeof(int), compare);

    7. ALGORITHM TYPES YOU MAY ENCOUNTER:
       - Array manipulation (search, sort, reverse, rotate)
       - String processing (parsing, pattern matching, transformations)
       - Hash tables (frequency counting, two-sum, anagrams)
       - Dynamic programming (memoization using hash tables)
       - Graph algorithms (BFS/DFS using GQueue/GList)
       - Tree traversal (using recursion or queues)
       - Sliding window problems
       - Two pointers technique
       - Greedy algorithms
       - Mathematical computations

    8. TESTING YOUR SOLUTION:
       - Your program will be compiled with: gcc -std=c11 -O2 [glib flags] -o program code.c
       - It will be run with test inputs via STDIN
       - Output will be compared character-by-character with expected output
       - Ensure output format matches exactly (spaces, newlines, etc.)

    9. EXAMPLE STRUCTURE:

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <glib.h>

    int main() {
        // Read input
        int n;
        scanf("%d", &n);

        // Process using appropriate data structure
        GHashTable *map = g_hash_table_new(g_direct_hash, g_direct_equal);

        // Your algorithm here
        for (int i = 0; i < n; i++) {
            // process
        }

        // Output result
        printf("%d\n", result);

        // Clean up
        g_hash_table_destroy(map);

        return 0;
    }

    Remember: Write clean, efficient, and correct C code that solves the problem completely.
    """

    programming_language: str = dspy.InputField()
    problem_statement: str = dspy.InputField()
    solution: str = dspy.OutputField()


def extract_code_from_markdown(markdown: Optional[str]) -> Optional[str]:
    """
    Extracts the first markdown block of code from markdown.

    Strips away the language tag on the first line if present. Supports markdown
    that has several code blocks (just returns the first).
    """
    if markdown is None:
        return None
    # Find the first code block
    code_block_start = markdown.find("```")
    if code_block_start == -1:
        # Assume that the whole string is code.
        return markdown

    # Skip past the opening ```
    code_start = code_block_start + 3

    # Find the end of this code block
    code_block_end = markdown.find("```", code_start)
    if code_block_end == -1:
        return None

    # Extract the code between the markers
    code = markdown[code_start:code_block_end].strip()

    if "# Example usage:" in code:
        code = code.split("# Example usage:")[0]

    # Remove language tag if present on first line
    first_newline = code.find("\n")
    if first_newline > 0:
        # Consider the case where the block begins with "```python\n...". In this
        # case, code would already be "python\n..." and first_newline would be 7.
        # Thus first_newline + 1 is the index of "...".
        code = code[first_newline + 1 :]

    return code.strip()


class SolveProblemWrapper(dspy.Module):
    """
    Wrapper around SolveProblem that:
    1. Maps field names from LiveCodeBench to the field names that SolveProblem expects.
    2. Suppresses DSPy errors, which are rare.
    3. Extracts the code from the DSPy output.
    """

    def __init__(self):
        self.solve_problem = dspy.ChainOfThought(SolveProblem)

    async def aforward(
        self, language: str, question_content: str, question_id: str, private_test_cases
    ) -> dict:
        try:
            result = await self.solve_problem.aforward(
                programming_language=language,
                problem_statement=question_content,
            )
            solution = extract_code_from_markdown(result.solution)
            reasoning = result.reasoning
        except Exception as e:
            solution = None
            reasoning = f"DSPy error\n\n{str(e)}"
        return {
            "solution": solution,
            "reasoning": reasoning,
            "question_id": question_id,
        }


async def do_completions(
    *,
    model_name: str,
    completions_path: Path,
    temperature: float,
    num_concurrent: int,
    max_tokens: int,
    top_p: float,
    language: str,
    enable_dspy_cache: bool,
    num_completions: int,
) -> None:
    dspy.configure_cache(
        enable_disk_cache=enable_dspy_cache,
        enable_memory_cache=enable_dspy_cache,
    )

    if enable_dspy_cache:
        assert num_concurrent == 1, "caching requires num_concurrent == 1"

    lm = dspy.LM(
        model_name,
        model_type="chat",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    dspy.configure(lm=lm)

    # Test the model. If this crashes, no point trying to run the benchmark.
    lm("Say this is a test!", temperature=1.0)

    # problems = datasets.load_dataset("nuprl/Ag-LiveCodeBench-X", split="test")
    from datasets import load_dataset

    cache_dir = "/run/media/blazingbhavneek/Common/Code/cBench/Ag-LiveCodeBench-X/data"

    problems = load_dataset(
        "nuprl/Ag-LiveCodeBench-X",
        split="test[:100]",
        cache_dir=cache_dir,
    )

    print(problems)

    solve_problem = SolveProblemWrapper()

    metadata = {
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "language": language,
    }

    async def do_generate(
        new_keys: List[Tuple[str, int]],
    ) -> AsyncIterator[Awaitable[Candidate]]:
        pbar = tqdm(total=sum(n for _, n in new_keys), desc="Generating")

        counts = {k: n for k, n in new_keys}
        for problem in problems:
            if problem["question_id"] not in counts:
                continue
            this_problem_count = counts[problem["question_id"]]
            for _ in range(this_problem_count):
                yield solve_problem.aforward(language=language, **problem)
                pbar.update(1)

    completions_path.parent.mkdir(parents=True, exist_ok=True)

    await run_bounded_create_or_resume_jsonl_file(
        file_name=completions_path,
        key_name="question_id",
        key_count=num_completions,
        key_generator=problems["question_id"],
        value_generator=do_generate,
        limit=num_concurrent,
        on_error="print",
    )


def container_command_from_name(container_name: str) -> List[str]:
    if container_name.endswith(".sif"):
        return ["apptainer", "run", "--contain", "--writable-tmpfs", container_name]
    else:
        return [
            "docker",
            "run",
            "--rm",
            "-i",
            "--tmpfs",
            "/ramdisk:size=512m,exec",
            container_name,
        ]


async def do_execute(
    *,
    container_name: str,
    timeout_seconds: int,
    generations_path: Path,
    executions_path: Path,
    num_concurrent: int,
):
    # This will consume a few GB of memory.
    problems = datasets.load_dataset("nuprl/Ag-LiveCodeBench-X", split="test")
    tests_by_id = {p["question_id"]: p["private_test_cases"] for p in problems}
    problems = None

    pbar = tqdm(desc="Executing")

    async def execute(row):
        question_id = row["question_id"]
        solution = row["solution"]
        private_test_cases = tests_by_id.get(question_id)
        result = await run(
            container_command_from_name(container_name),
            timeout_seconds=timeout_seconds,
            stdin_data=json.dumps(
                {
                    "code": solution,
                    "timeout_s": timeout_seconds,
                    "test_cases": decompress_lcb_private_tests(private_test_cases),
                }
            ),
            stdin_write_timeout=300,
        )
        result_dict = {
            "raw_exit_code": result.exit_code,
            "raw_stdout": result.stdout,
            "raw_stderr": result.stderr,
        }
        if result.exit_code != 0:
            return {**result_dict, "result": "fail"}
        try:
            # result.stdout has a JSON dictionary with fields result, stdout,
            # etc. That result is the real result.
            return {**result_dict, **json.loads(result.stdout)}
        except json.JSONDecodeError:
            return {**result_dict, "result": "fail"}

    executions_path.parent.mkdir(parents=True, exist_ok=True)

    await map_by_key_jsonl_file(
        generations_path,
        executions_path,
        f=execute,
        key="solution",
        keep_columns=["question_id"],
        on_error="raise",
        num_concurrent=num_concurrent,
        progress=lambda: pbar.update(1),
    )


def do_pass1(*, paths: List[str]) -> None:
    """
    This function summarizes results by question_id, counting total and successful
    completions.
    """
    print("Path,Success Rate,Error Rate")
    for p in paths:
        # We assume that every question has exactly the same number of
        # completions. That's what allows us to naively add and divide to
        # compute pass@1. If not, we would have to group by question_id to do
        # this right.
        num_rows = 0
        num_successes = 0
        num_run_errors = 0
        with Path(p).open("rt") as f:
            for line in f:
                row = json.loads(line)
                num_rows = num_rows + 1
                if row["result"] == "success":
                    num_successes = num_successes + 1
                elif "stderr" in row and row["stderr"].endswith(
                    "failed to write to stdin"
                ):
                    num_run_errors = num_run_errors + 1
        success_rate = num_successes / num_rows
        run_error_rate = num_run_errors / num_rows
        print(f"{p},{success_rate:.2f},{run_error_rate:.2f}")


async def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate subcommand
    completions_parser = subparsers.add_parser(
        "completions", help="Generate solutions for LiveCodeBench problems"
    )
    completions_parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="The model name in LiteLLM format.",
    )
    completions_parser.add_argument("--completions-path", type=Path, required=True)
    completions_parser.add_argument("--temperature", type=float, default=0.6)
    completions_parser.add_argument("--num-concurrent", type=int, default=20)
    completions_parser.add_argument("--max-tokens", type=int, default=5000)
    completions_parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="The programming language to use.",
    )
    completions_parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p value for sampling",
    )
    completions_parser.add_argument(
        "--enable-dspy-cache",
        type=bool,
        default=False,
        help="Enable DSPy cache. Do not set this unless you understand DSPy.",
    )
    completions_parser.add_argument("--num-completions", type=int, default=1)

    pass1_parser = subparsers.add_parser("pass1", help="Summarize results by task_id")
    pass1_parser.add_argument(
        "paths",
        type=str,
        nargs="+",
        help="Paths to results JSONL files from the 'bench' command",
    )

    execute_parser = subparsers.add_parser(
        "executions", help="Execute existing generations"
    )
    execute_parser.add_argument("--container-name", type=str, required=True)
    execute_parser.add_argument("--timeout-seconds", type=int, required=True)
    execute_parser.add_argument("--generations-path", type=Path, required=True)
    execute_parser.add_argument("--executions-path", type=Path, required=True)
    execute_parser.add_argument("--num-concurrent", type=int, required=True)

    args = parser.parse_args()

    args_dict = {k: v for k, v in vars(args).items() if k != "command"}

    if args.command == "completions":
        await do_completions(**args_dict)
    elif args.command == "pass1":
        do_pass1(**args_dict)
    elif args.command == "executions":
        await do_execute(**args_dict)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
