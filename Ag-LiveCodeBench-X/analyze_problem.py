import asyncio
import json
from pathlib import Path
from collections import Counter, defaultdict
import re
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


async def analyze_problem(client, problem, semaphore, model="gpt-4"):
    """Analyze a single problem to extract metadata about libraries and problem types."""
    async with semaphore:
        try:
            prompt = f"""Analyze this coding problem and respond with ONLY a JSON object (no thinking, no explanation):

{{
    "standard_libs": ["list of commonly needed libraries like math, collections, itertools, heapq, bisect, etc."],
    "third_party_libs": ["list of third-party libs like numpy, pandas, scipy, networkx, sympy, etc."],
    "categories": ["problem types like dynamic programming, graph theory, greedy, number theory, etc."],
    "difficulty": "easy or medium or hard",
    "algorithms": ["key algorithms used"],
    "data_structures": ["key data structures used"]
}}

Problem (first 1500 chars):
{problem['question_content'][:1500]}"""

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            content = response.choices[0].message.content
            
            if not content:
                raise ValueError("Empty response from API")
            
            # Handle thinking tokens - extract everything after </think>
            if '<think>' in content:
                parts = content.split('</think>')
                if len(parts) > 1:
                    content = parts[-1].strip()
                else:
                    # Incomplete think tag, try to find JSON anyway
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(0)
            
            # Try to extract JSON
            try:
                analysis = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON in markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                    analysis = json.loads(content)
                else:
                    # Try to find any JSON object
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group(0))
                    else:
                        raise ValueError(f"No JSON found in response")
            
            # Validate and normalize
            analysis['question_id'] = problem['question_id']
            analysis['standard_libs'] = [lib.strip() for lib in analysis.get('standard_libs', []) if lib]
            analysis['third_party_libs'] = [lib.strip() for lib in analysis.get('third_party_libs', []) if lib]
            analysis['categories'] = [cat.strip() for cat in analysis.get('categories', []) if cat]
            analysis['difficulty'] = str(analysis.get('difficulty', 'unknown')).lower().strip()
            analysis['algorithms'] = [algo.strip() for algo in analysis.get('algorithms', []) if algo]
            analysis['data_structures'] = [ds.strip() for ds in analysis.get('data_structures', []) if ds]
            
            return analysis
            
        except Exception as e:
            return {
                "question_id": problem.get('question_id', 'unknown'),
                "error": str(e)[:200],
                "standard_libs": [],
                "third_party_libs": [],
                "categories": [],
                "difficulty": "unknown",
                "algorithms": [],
                "data_structures": []
            }


async def analyze_dataset(
    *,
    output_path: Path,
    model_name: str = "gpt-4",
    num_concurrent: int = 10,
    base_url: str = None,
    api_key: str = None,
    cache_dir: str = None,
    num_problems: int = None,
    retry_failed: bool = True,
) -> None:
    """Analyze all problems in the dataset to determine library requirements."""
    
    print("Loading dataset...")
    problems = load_dataset(
        "nuprl/Ag-LiveCodeBench-X",
        split="test",
        cache_dir=cache_dir,
    )
    
    if num_problems is not None:
        problems = problems.select(range(min(num_problems, len(problems))))
    
    print(f"Loaded {len(problems)} problems")
    
    client_kwargs = {"api_key": api_key} if api_key else {}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)
    
    semaphore = asyncio.Semaphore(num_concurrent)
    
    print("Analyzing problems...")
    tasks = [
        analyze_problem(client, problem, semaphore, model_name)
        for problem in problems
    ]
    
    results = await tqdm_asyncio.gather(*tasks, desc="Analyzing")
    
    if retry_failed:
        failed_indices = [i for i, r in enumerate(results) if "error" in r and r.get("error")]
        if failed_indices:
            print(f"\nRetrying {len(failed_indices)} failed analyses...")
            retry_semaphore = asyncio.Semaphore(max(1, num_concurrent // 2))
            retry_tasks = [
                analyze_problem(client, problems[i], retry_semaphore, model_name)
                for i in failed_indices
            ]
            retry_results = await tqdm_asyncio.gather(*retry_tasks, desc="Retrying")
            
            for idx, retry_result in zip(failed_indices, retry_results):
                results[idx] = retry_result
    
    print("\nAggregating results...")
    standard_libs = Counter()
    third_party_libs = Counter()
    categories = Counter()
    difficulties = Counter()
    algorithms = Counter()
    data_structures = Counter()
    
    successful_count = 0
    for result in results:
        if "error" not in result or not result.get("error"):
            successful_count += 1
            for lib in result.get("standard_libs", []):
                if lib:
                    standard_libs[lib.lower().strip()] += 1
            for lib in result.get("third_party_libs", []):
                if lib:
                    third_party_libs[lib.lower().strip()] += 1
            for cat in result.get("categories", []):
                if cat:
                    categories[cat.lower().strip()] += 1
            diff = result.get("difficulty", "unknown").lower().strip()
            difficulties[diff] += 1
            for algo in result.get("algorithms", []):
                if algo:
                    algorithms[algo.lower().strip()] += 1
            for ds in result.get("data_structures", []):
                if ds:
                    data_structures[ds.lower().strip()] += 1
    
    report = {
        "total_problems": len(problems),
        "analyzed_problems": successful_count,
        "failed_analyses": len(problems) - successful_count,
        "summary": {
            "standard_libraries": dict(standard_libs.most_common(30)),
            "third_party_libraries": dict(third_party_libs.most_common(20)),
            "categories": dict(categories.most_common(20)),
            "difficulty_distribution": dict(difficulties),
            "common_algorithms": dict(algorithms.most_common(20)),
            "common_data_structures": dict(data_structures.most_common(20))
        },
        "detailed_results": results
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nTotal Problems: {report['total_problems']}")
    print(f"Successfully Analyzed: {report['analyzed_problems']}")
    print(f"Failed: {report['failed_analyses']}")
    
    print("\n--- TOP STANDARD LIBRARIES ---")
    for lib, count in standard_libs.most_common(15):
        print(f"  {lib}: {count} problems ({count/successful_count*100:.1f}%)")
    
    print("\n--- TOP THIRD-PARTY LIBRARIES ---")
    if third_party_libs:
        for lib, count in third_party_libs.most_common(15):
            print(f"  {lib}: {count} problems ({count/successful_count*100:.1f}%)")
    else:
        print("  None identified")
    
    print("\n--- PROBLEM CATEGORIES ---")
    for cat, count in categories.most_common(12):
        print(f"  {cat}: {count} problems ({count/successful_count*100:.1f}%)")
    
    print("\n--- DIFFICULTY DISTRIBUTION ---")
    for diff, count in difficulties.most_common():
        print(f"  {diff}: {count} problems ({count/successful_count*100:.1f}%)")
    
    print(f"\nFull report saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze LiveCodeBench problems")
    parser.add_argument("--output", type=Path, default=Path("problem_analysis.json"),
                        help="Output path for analysis report")
    parser.add_argument("--model", default="gpt-4",
                        help="Model to use for analysis")
    parser.add_argument("--concurrent", type=int, default=10,
                        help="Number of concurrent requests")
    parser.add_argument("--base-url", default=None,
                        help="Base URL for OpenAI API")
    parser.add_argument("--api-key", default=None,
                        help="API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache directory for dataset")
    parser.add_argument("--num-problems", type=int, default=None,
                        help="Limit number of problems to analyze")
    parser.add_argument("--no-retry", action="store_true",
                        help="Don't retry failed analyses")
    
    args = parser.parse_args()
    
    asyncio.run(analyze_dataset(
        output_path=args.output,
        model_name=args.model,
        num_concurrent=args.concurrent,
        base_url=args.base_url,
        api_key=args.api_key,
        cache_dir=args.cache_dir,
        num_problems=args.num_problems,
        retry_failed=not args.no_retry,
    ))
