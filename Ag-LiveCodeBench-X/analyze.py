import json
from pathlib import Path
import re
from collections import defaultdict

def analyze_execution_folder(folder: Path):
    exec_files = sorted(
        folder.glob("executions_iter*.jsonl"),
        key=lambda p: int(re.search(r"iter(\d+)", p.name).group(1))
    )

    solved_first_iter = {}   # question_id -> iteration
    all_questions = set()
    cumulative_stats = []

    for exec_file in exec_files:
        iteration = int(re.search(r"iter(\d+)", exec_file.name).group(1))
        solved_this_iter = set()

        with exec_file.open() as f:
            for line in f:
                row = json.loads(line)
                qid = row["question_id"]
                all_questions.add(qid)

                if row.get("result") == "success":
                    solved_this_iter.add(qid)
                    if qid not in solved_first_iter:
                        solved_first_iter[qid] = iteration

        cumulative_stats.append({
            "iteration": iteration,
            "solved_this_iteration": len(solved_this_iter),
            "cumulative_solved": len(solved_first_iter),
        })

    summary = {
        "total_unique_problems": len(all_questions),
        "total_solved_at_least_once": len(solved_first_iter),
        "solve_rate": (
            len(solved_first_iter) / len(all_questions)
            if all_questions else 0.0
        ),
        "first_solved_iteration": solved_first_iter,
        "per_iteration_cumulative": cumulative_stats,
    }

    out_path = folder / "cumulative_solution_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to: {out_path}")
    print(f"Solved at least once: {len(solved_first_iter)}/{len(all_questions)} "
          f"({summary['solve_rate']:.2%})")


if __name__ == "__main__":
    analyze_execution_folder(Path("./output"))
