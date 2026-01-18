import argparse
import asyncio
import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import datasets
from abstractions.async_abstractions import run_bounded
from abstractions.storage import map_by_key_jsonl_file
from bounded_subprocess.bounded_subprocess_async import run
from datasets import load_dataset
from tqdm import tqdm
from tqdm.auto import tqdm

from problem.solve import SolveProblemWrapper
from problem.refine import RefineProblemWrapper

from prompt.c import C_CRITICAL_CODING_REQUIREMENTS
from prompt.py import Python_CRITICAL_CODING_REQUIREMENTS

from models.data import Candidate, Result, RefinementTrainingExample
from models.llm import SolutionResponse, RefinementResponse

CRITICAL_CODING_REQUIREMENTS = C_CRITICAL_CODING_REQUIREMENTS


def decompress_lcb_private_tests(text: str):
    """
    LiveCodeBench compresses its private tests because they are enormous (8GB
    when we write our 499 problem subset to disk).
    """
    return json.loads(
        pickle.loads(zlib.decompress(base64.b64decode(text.encode("utf-8"))))
    )


async def do_completions(
    *,
    model_name: str,
    completions_path: Path,
    temperature: float,
    num_concurrent: int,
    max_tokens: int,
    top_p: float,
    language: str,
    num_completions: int,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = None,
    use_thinking_budget: bool = False,
    tokenizer_name_or_path: str = None,
    max_thinking_budget: int = 512,
    max_agent_iterations: int = 0,
    summarize_context: bool = False,
    cache_dir: str = None,
    num_problems: int = None,
) -> None:

    problems = load_dataset(
        "nuprl/Ag-LiveCodeBench-X",
        split="test",
        cache_dir=cache_dir,
    )

    # Limit number of problems if specified
    if num_problems is not None:
        problems = problems.select(range(min(num_problems, len(problems))))

    print(f"Loaded {len(problems)} problems")

    solve_problem = SolveProblemWrapper(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        use_thinking_budget=use_thinking_budget,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_thinking_budget=max_thinking_budget,
        max_tokens=max_tokens,
        critical_coding_requirements=CRITICAL_CODING_REQUIREMENTS
    )

    completions_path.parent.mkdir(parents=True, exist_ok=True)

    # Create all tasks
    tasks = []
    for problem in problems:
        for _ in range(num_completions):
            tasks.append(
                solve_problem.aforward(
                    language=language,
                    question_content=problem["question_content"],
                    question_id=problem["question_id"],
                    private_test_cases=problem.get("private_test_cases"),
                    max_agent_iterations=max_agent_iterations,
                    summarize_context=summarize_context,
                )
            )

    # Execute concurrently
    semaphore = asyncio.Semaphore(num_concurrent)
    results = []
    pbar = tqdm(total=len(tasks), desc="Generating")

    async def execute_with_semaphore(task):
        async with semaphore:
            result = await task
            pbar.update(1)
            return result

    results = await asyncio.gather(*[execute_with_semaphore(task) for task in tasks])
    pbar.close()

    # Save results
    with open(completions_path, "wt") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


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
    cache_dir: str = None,
):
    # This will consume a few GB of memory.
    problems = datasets.load_dataset(
        "nuprl/Ag-LiveCodeBench-X", split="test", cache_dir=cache_dir
    )
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


def do_pass1(*, paths: List[str]) -> dict:
    """
    This function summarizes results by question_id, counting total and successful
    completions. Returns statistics dictionary.
    """
    results = {}
    print("\n" + "=" * 60)
    print("Path,Success Rate,Error Rate,Solved Problems")
    print("=" * 60)

    for p in paths:
        num_rows = 0
        num_successes = 0
        num_run_errors = 0
        solved_problems = set()

        with Path(p).open("rt") as f:
            for line in f:
                row = json.loads(line)
                num_rows = num_rows + 1
                if row["result"] == "success":
                    num_successes = num_successes + 1
                    solved_problems.add(row["question_id"])
                elif "stderr" in row and row["stderr"].endswith(
                    "failed to write to stdin"
                ):
                    num_run_errors = num_run_errors + 1

        success_rate = num_successes / num_rows if num_rows > 0 else 0
        run_error_rate = num_run_errors / num_rows if num_rows > 0 else 0

        results[p] = {
            "success_rate": success_rate,
            "run_error_rate": run_error_rate,
            "num_solved": len(solved_problems),
            "total_executions": num_rows,
        }

        print(f"{p},{success_rate:.2%},{run_error_rate:.2%},{len(solved_problems)}")

    print("=" * 60 + "\n")
    return results


async def do_refinements(
    *,
    model_name: str,
    executions_path: Path,
    refinements_path: Path,
    completions_path: Path,
    temperature: float,
    num_concurrent: int,
    max_tokens: int,
    top_p: float,
    language: str,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = None,
    use_thinking_budget: bool = False,
    tokenizer_name_or_path: str = None,
    max_thinking_budget: int = 512,
    max_agent_iterations: int = 0,
    summarize_context: bool = False,
    cache_dir: str = None,
) -> int:
    """
    Generate refined solutions for failed executions and store training data.
    Returns the number of problems that were successfully refined.
    """

    # Load problems to get problem statements
    print("Loading problems dataset...")
    problems = datasets.load_dataset(
        "nuprl/Ag-LiveCodeBench-X", split="test", cache_dir=cache_dir
    )
    problem_statements = {p["question_id"]: p["question_content"] for p in problems}

    # Load executions and separate successful from failed
    print(f"Loading executions from {executions_path}...")
    successful_executions = []
    failed_executions = []

    with open(executions_path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row.get("result") == "success":
                successful_executions.append(row)
            else:
                failed_executions.append(row)

    print(f"Found {len(successful_executions)} successful executions (will be kept)")
    print(f"Found {len(failed_executions)} failed executions to refine")

    if not failed_executions:
        print("No failed executions to refine. Exiting.")
        return 0

    refine_wrapper = RefineProblemWrapper(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        use_thinking_budget=use_thinking_budget,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_thinking_budget=max_thinking_budget,
        max_tokens=max_tokens,
        critical_coding_requirements=CRITICAL_CODING_REQUIREMENTS
    )

    # Prepare output directories
    refinements_path.parent.mkdir(parents=True, exist_ok=True)
    completions_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate refinements with concurrency control
    training_examples: List[RefinementTrainingExample] = []
    refined_completions: List[Candidate] = []

    pbar = tqdm(total=len(failed_executions), desc="Generating refinements")

    async def process_execution(
        execution: dict,
    ) -> Optional[Tuple[RefinementTrainingExample, Candidate]]:
        """Process a single failed execution and return training example if successful."""
        question_id = execution["question_id"]
        original_code = execution.get("solution")

        if not original_code:
            print(f"Warning: No solution code found for {question_id}")
            return None

        problem_statement = problem_statements.get(question_id)

        if not problem_statement:
            print(f"Warning: Problem statement not found for {question_id}")
            return None

        # Extract error feedback - handle the nested JSON structure
        error_feedback = {
            "result": execution.get("result", "unknown"),
            "exit_code": execution.get("exit_code", execution.get("raw_exit_code", -1)),
            "stdout": execution.get("stdout", execution.get("raw_stdout", "")),
            "stderr": execution.get("stderr", execution.get("raw_stderr", "")),
        }

        # Generate refined solution
        result = await refine_wrapper.aforward(
            language=language,
            problem_statement=problem_statement,
            original_code=original_code,
            error_feedback=error_feedback,
            question_id=question_id,
            max_agent_iterations=max_agent_iterations,
            summarize_context=summarize_context,
        )

        pbar.update(1)

        if result["refined_code"]:
            # Create training example
            training_example: RefinementTrainingExample = {
                "question_id": result["question_id"],
                "language": result["language"],
                "problem_statement": result["problem_statement"],
                "original_code": result["original_code"],
                "error_feedback": result["error_feedback"],
                "refined_code": result["refined_code"],
                "reasoning": result["reasoning"],
            }

            # Create completion record for re-execution
            completion_record: Candidate = {
                "question_id": result["question_id"],
                "solution": result["refined_code"],
                "reasoning": result["reasoning"],
            }

            return (training_example, completion_record)
        else:
            print(f"Warning: Failed to generate refined code for {question_id}")
            return None

    # Process all failed executions with concurrency limit
    async def iter_async(seq):
        for item in seq:
            yield item

    tasks = [process_execution(exec) for exec in failed_executions]
    async for coro in run_bounded(iter_async(tasks), limit=num_concurrent):
        result = await coro
        if result is not None:
            training_example, completion_record = result
            training_examples.append(training_example)
            refined_completions.append(completion_record)

    pbar.close()

    # Save training data
    print(f"Saving {len(training_examples)} training examples to {refinements_path}...")
    with open(refinements_path, "wt") as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")

    # Carry forward successful solutions + refined failures
    with open(completions_path, "wt") as f:
        # 1) keep already-successful solutions unchanged
        for row in successful_executions:
            f.write(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "solution": row["solution"],
                    }
                )
                + "\n"
            )

        # 2) add refined solutions for failures
        for record in refined_completions:
            f.write(json.dumps(record) + "\n")

    print(f"Done! Generated {len(training_examples)} refined solutions.")
    print(f"Successfully refined: {len(training_examples)}/{len(failed_executions)}")

    return len(refined_completions)


async def do_iterative_refinement(
    *,
    model_name: str,
    container_name: str,
    timeout_seconds: int,
    output_dir: Path,
    temperature: float,
    num_concurrent: int,
    max_tokens: int,
    top_p: float,
    language: str,
    num_completions: int,
    max_refinement_iterations: int = 3,
    num_problems: int = 20,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    api_key: str = "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",
    use_thinking_budget: bool = False,
    tokenizer_name_or_path: str = None,
    max_thinking_budget: int = 512,
    max_agent_iterations: int = 0,
    summarize_context: bool = False,
    cache_dir: str = None,
) -> None:
    """
    Run iterative refinement loop:
    completions -> execution -> pass1 -> refinement -> execution -> pass1 -> refinement

    Keeps successful solutions and only refines failures in each iteration.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics across iterations
    iteration_stats = []

    print("\n" + "=" * 80)
    print(f"STARTING ITERATIVE REFINEMENT PIPELINE")
    print(f"Model: {model_name}")
    print(f"Language: {language}")
    print(f"Number of problems: {num_problems}")
    print(f"Max refinement iterations: {max_refinement_iterations}")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")

    # Initial completions
    print(f"\n{'='*80}")
    print(f"ITERATION 0: Initial Completions")
    print(f"{'='*80}\n")

    completions_path = output_dir / "completions_iter0.jsonl"
    executions_path = output_dir / "executions_iter0.jsonl"

    await do_completions(
        model_name=model_name,
        completions_path=completions_path,
        temperature=temperature,
        num_concurrent=num_concurrent,
        max_tokens=max_tokens,
        top_p=top_p,
        language=language,
        num_completions=num_completions,
        base_url=base_url,
        api_key=api_key,
        use_thinking_budget=use_thinking_budget,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_thinking_budget=max_thinking_budget,
        max_agent_iterations=max_agent_iterations,
        summarize_context=summarize_context,
        cache_dir=cache_dir,
        num_problems=num_problems,
    )

    # Execute initial completions
    print(f"\n{'='*80}")
    print(f"ITERATION 0: Executing Initial Completions")
    print(f"{'='*80}\n")

    await do_execute(
        container_name=container_name,
        timeout_seconds=timeout_seconds,
        generations_path=completions_path,
        executions_path=executions_path,
        num_concurrent=num_concurrent,
        cache_dir=cache_dir,
    )

    # Get pass@1 for initial completions
    print(f"\n{'='*80}")
    print(f"ITERATION 0: Results")
    print(f"{'='*80}\n")

    stats = do_pass1(paths=[str(executions_path)])
    iteration_stats.append(
        {"iteration": 0, "type": "initial", **stats[str(executions_path)]}
    )

    # Refinement iterations
    for iteration in range(1, max_refinement_iterations + 1):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}: Refinement")
        print(f"{'='*80}\n")

        prev_executions_path = output_dir / f"executions_iter{iteration-1}.jsonl"
        refinements_path = output_dir / f"refinements_iter{iteration}.jsonl"
        refined_completions_path = output_dir / f"completions_iter{iteration}.jsonl"
        refined_executions_path = output_dir / f"executions_iter{iteration}.jsonl"

        # Generate refinements
        num_refined = await do_refinements(
            model_name=model_name,
            executions_path=prev_executions_path,
            refinements_path=refinements_path,
            completions_path=refined_completions_path,
            temperature=temperature,
            num_concurrent=num_concurrent,
            max_tokens=max_tokens,
            top_p=top_p,
            language=language,
            base_url=base_url,
            api_key=api_key,
            use_thinking_budget=use_thinking_budget,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_thinking_budget=max_thinking_budget,
            max_agent_iterations=max_agent_iterations,
            summarize_context=summarize_context,
            cache_dir=cache_dir,
        )

        if num_refined == 0:
            print(f"\nNo more problems to refine. Stopping at iteration {iteration}.")
            break

        # Execute refined completions
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}: Executing Refined Completions")
        print(f"{'='*80}\n")

        await do_execute(
            container_name=container_name,
            timeout_seconds=timeout_seconds,
            generations_path=refined_completions_path,
            executions_path=refined_executions_path,
            num_concurrent=num_concurrent,
            cache_dir=cache_dir,
        )

        # Get pass@1 for refined completions
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}: Results")
        print(f"{'='*80}\n")

        stats = do_pass1(paths=[str(refined_executions_path)])
        iteration_stats.append(
            {
                "iteration": iteration,
                "type": "refinement",
                **stats[str(refined_executions_path)],
            }
        )

    # Save summary statistics
    summary_path = output_dir / "summary_stats.json"
    with open(summary_path, "w") as f:
        json.dump(iteration_stats, f, indent=2)

    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*80}\n")
    print(f"Summary of all iterations:")
    print(f"{'Iter':<6} {'Type':<12} {'Success Rate':<15} {'Solved Problems':<20}")
    print("-" * 80)
    for stat in iteration_stats:
        print(
            f"{stat['iteration']:<6} {stat['type']:<12} "
            f"{stat['success_rate']:<15.2%} {stat['num_solved']:<20}"
        )
    print(f"\nSummary saved to: {summary_path}")


async def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Shared arguments
    def add_common_llm_args(p):
        p.add_argument("--model-name", type=str, required=True)
        p.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
        p.add_argument("--api-key", type=str, default=None)
        p.add_argument("--temperature", type=float, default=0.6)
        p.add_argument("--num-concurrent", type=int, default=20)
        p.add_argument("--max-tokens", type=int, default=5000)
        p.add_argument("--top-p", type=float, default=0.95)
        p.add_argument("--language", type=str, required=True)
        p.add_argument("--use-thinking-budget", action="store_true")
        p.add_argument("--tokenizer-name-or-path", type=str, default=None)
        p.add_argument("--max-thinking-budget", type=int, default=512)
        p.add_argument("--max-agent-iterations", type=int, default=0)
        p.add_argument("--summarize-context", action="store_true")
        p.add_argument("--cache-dir", type=str, default=None)

    # Generate subcommand
    completions_parser = subparsers.add_parser(
        "completions", help="Generate solutions for LiveCodeBench problems"
    )
    add_common_llm_args(completions_parser)
    completions_parser.add_argument("--completions-path", type=Path, required=True)
    completions_parser.add_argument("--num-completions", type=int, default=1)
    completions_parser.add_argument("--num-problems", type=int, default=None)

    # Pass1 subcommand
    pass1_parser = subparsers.add_parser("pass1", help="Summarize results by task_id")
    pass1_parser.add_argument(
        "paths",
        type=str,
        nargs="+",
        help="Paths to results JSONL files from the 'bench' command",
    )

    # Execute subcommand
    execute_parser = subparsers.add_parser(
        "executions", help="Execute existing generations"
    )
    execute_parser.add_argument("--container-name", type=str, required=True)
    execute_parser.add_argument("--timeout-seconds", type=int, required=True)
    execute_parser.add_argument("--generations-path", type=Path, required=True)
    execute_parser.add_argument("--executions-path", type=Path, required=True)
    execute_parser.add_argument("--num-concurrent", type=int, required=True)
    execute_parser.add_argument("--cache-dir", type=str, default=None)

    # Refinements subcommand
    refinements_parser = subparsers.add_parser(
        "refinements",
        help="Generate refined solutions from failed executions and create training data",
    )
    add_common_llm_args(refinements_parser)
    refinements_parser.add_argument("--executions-path", type=Path, required=True)
    refinements_parser.add_argument("--refinements-path", type=Path, required=True)
    refinements_parser.add_argument("--completions-path", type=Path, required=True)

    # NEW: Iterative refinement subcommand
    iterative_parser = subparsers.add_parser(
        "iterative",
        help="Run iterative refinement pipeline: completions -> execute -> refine -> execute (loop)",
    )
    add_common_llm_args(iterative_parser)
    iterative_parser.add_argument("--container-name", type=str, required=True)
    iterative_parser.add_argument("--timeout-seconds", type=int, required=True)
    iterative_parser.add_argument("--output-dir", type=Path, required=True)
    iterative_parser.add_argument("--num-completions", type=int, default=1)
    iterative_parser.add_argument(
        "--max-refinement-iterations",
        type=int,
        default=3,
        help="Maximum number of refinement iterations",
    )
    iterative_parser.add_argument(
        "--num-problems",
        type=int,
        default=20,
        help="Number of problems to evaluate",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args_dict = {k: v for k, v in vars(args).items() if k != "command"}

    if args.command == "completions":
        await do_completions(**args_dict)
    elif args.command == "pass1":
        do_pass1(**args_dict)
    elif args.command == "executions":
        await do_execute(**args_dict)
    elif args.command == "refinements":
        await do_refinements(**args_dict)
    elif args.command == "iterative":
        await do_iterative_refinement(**args_dict)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
