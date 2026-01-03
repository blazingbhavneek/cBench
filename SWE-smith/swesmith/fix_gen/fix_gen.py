"""
Purpose: Generate fixes for buggy functions and validate them

Usage: 
python swesmith/fix_gen/fix_generator.py \
    logs/task_insts/DaveGamble__cJSON.c859b25d.json \
    --model openai/nemotron-nano \
    --n_workers 1 \
    --max_fixes 1 \
    --max_retries 5
"""

import argparse
import json
import litellm
import logging
import os
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from litellm import completion
from pathlib import Path
from tqdm import tqdm
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    FAIL_TO_PASS,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
)
from swebench.harness.docker_build import close_logger

from swesmith.bug_gen.llm.utils import extract_code_block
from swesmith.bug_gen.utils import get_patch
from swesmith.constants import (
    KEY_PATCH,
    KEY_TIMED_OUT,
    LOG_DIR_RUN_VALIDATION,
    REF_SUFFIX,
)
from swesmith.harness.grading import get_valid_report
from swesmith.harness.utils import run_patch_in_container
from swesmith.profiles import registry

load_dotenv(dotenv_path=os.getenv("SWEFT_DOTENV_PATH"))
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
litellm.suppress_debug_info = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_fix_instruction(patch_metadata: dict, model: str, full_function: str) -> str:
    print(f"\n[DEBUG] Generating instruction...")
    print(f"[DEBUG] Full function length: {len(full_function)} chars")
    print(f"[DEBUG] First 200 chars: {full_function[:200]}")
    
    messages = [
        {
            "role": "system",
            "content": "Create a fix instruction. Describe WHAT is wrong (behavior), not WHERE. Be concise."
        },
        {
            "role": "user",
            "content": (
                f"**Full Function:**\n```c\n{full_function}\n```\n\n"
                f"**Bug Diff:**\n```diff\n{patch_metadata['patch']}\n```\n\n"
                f"**Failing Tests:** {patch_metadata.get('explanation', '')}\n\n"
                f"Instruction:"
            )
        }
    ]
    
    print(f"[DEBUG] Calling LLM for instruction...")
    response = completion(model=model, messages=messages, temperature=0.3)
    instruction = response.choices[0].message.content.strip()
    print(f"[DEBUG] Generated instruction: {instruction}")
    return instruction


def generate_fix(buggy_function: str, instruction: str, model: str) -> str:
    print(f"\n[DEBUG] Generating fix...")
    print(f"[DEBUG] Instruction: {instruction}")
    print(f"[DEBUG] Buggy function length: {len(buggy_function)} chars")
    
    messages = [
        {
            "role": "system",
            "content": "You are a C programmer. Fix the buggy function. Output ONLY the fixed function code in a code block."
        },
        {
            "role": "user",
            "content": (
                f"**Instruction:** {instruction}\n\n"
                f"**Buggy Function:**\n```c\n{buggy_function}\n```\n\n"
                f"**Fixed Function:**"
            )
        }
    ]
    
    print(f"[DEBUG] Calling LLM for fix...")
    response = completion(model=model, messages=messages, temperature=0.7)
    fixed_code = extract_code_block(response.choices[0].message.content)
    print(f"[DEBUG] Fixed code length: {len(fixed_code)} chars")
    print(f"[DEBUG] First 200 chars: {fixed_code[:200]}")
    return fixed_code


def create_fix_patch_from_existing(patch_meta, fixed_code, repo_path, original_full_function):
    print(f"\n[DEBUG] Creating fix patch...")
    print(f"[DEBUG] Repo path: {repo_path}")
    print(f"[DEBUG] Original function length: {len(original_full_function)}")
    print(f"[DEBUG] Fixed code length: {len(fixed_code)}")
    
    import subprocess
    DEVNULL = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    patch_file = Path("temp_patch.diff")
    
    try:
        # Reset
        print(f"[DEBUG] Resetting repo...")
        subprocess.run(["git", "-C", repo_path, "reset", "--hard"], check=True, **DEVNULL)
        subprocess.run(["git", "-C", repo_path, "clean", "-fdx"], check=True, **DEVNULL)
        
        # Apply bug patch
        print(f"[DEBUG] Applying bug patch...")
        patch_file.write_text(patch_meta['patch'])
        subprocess.run(["git", "-C", repo_path, "apply", str(patch_file.absolute())], check=True, **DEVNULL)
        
        # Replace function
        file_path = next((l.replace('--- a/', '').strip() for l in patch_meta['patch'].split('\n') if l.startswith('--- a/')), None)
        print(f"[DEBUG] Target file: {file_path}")
        
        full_path = Path(repo_path) / file_path
        # After applying bug patch, extract the NOW-BUGGY function
        lines = full_path.read_text().split('\n')
        # Extract function name from original_full_function
        import re
        match = re.search(r'(static\s+\w+\s+\**)?(\w+)\s*\(', original_full_function)
        if not match:
            return None
        func_name = match.group(2)

        print(f"[DEBUG] Function name: {func_name}")

        # Find function by name in file
        content = full_path.read_text()
        lines = content.split('\n')

        # Find start of function
        start = None
        for i, line in enumerate(lines):
            if func_name in line and '(' in line:
                start = i
                break

        if start is None:
            print(f"[DEBUG] Function {func_name} not found!")
            return None

        # Find end (matching braces)
        end = start
        brace_count = 0
        for i in range(start, len(lines)):
            brace_count += lines[i].count('{') - lines[i].count('}')
            if brace_count == 0 and i > start:
                end = i
                break

        buggy_in_file = '\n'.join(lines[start:end+1])
        content = content.replace(buggy_in_file, fixed_code, 1)
        full_path.write_text(content)
        
        # Generate patch
        print(f"[DEBUG] Generating patch...")
        patch = get_patch(repo_path, reset_changes=True)
        print(f"[DEBUG] Generated patch length: {len(patch) if patch else 0}")
        return patch
    finally:
        if patch_file.exists():
            patch_file.unlink()

def validate_fix(instance: dict, repo: str, timeout: int) -> dict:
    instance_id = instance[KEY_INSTANCE_ID]
    rp = registry.get(repo)
    valid_folder = LOG_DIR_RUN_VALIDATION / repo
    
    # Apply fix patch and run tests
    logger, timed_out = run_patch_in_container(
        instance, repo, LOG_DIR_RUN_VALIDATION, timeout,
        patch=instance[KEY_PATCH],
    )
    
    if timed_out:
        close_logger(logger)
        return {"reward": -1.0, "tests_passed": False}
    
    # Parse test output
    test_output_path = valid_folder / instance_id / LOG_TEST_OUTPUT
    test_output = test_output_path.read_text()
    test_status_map = rp.log_parser(test_output)
    
    close_logger(logger)
    
    # Check if ALL tests pass
    all_pass = all(status == "PASSED" for status in test_status_map.values())
    
    return {
        "reward": 1.0 if all_pass else -1.0,
        "tests_passed": all_pass,
        "test_results": test_status_map
    }

def generate_and_validate_with_retry(patch, repo, model, rp, max_retries=5):
    print(f"\n{'='*60}")
    print(f"[DEBUG] Processing {patch['instance_id']}")
    print(f"{'='*60}")
    
    attempts = []
    
    # Clone and extract
    print(f"[DEBUG] Cloning repo...")
    repo_path, _ = rp.clone()
    print(f"[DEBUG] Repo cloned to: {repo_path}")
    
    print(f"[DEBUG] Extracting entities...")
    candidates = rp.extract_entities()
    print(f"[DEBUG] Found {len(candidates)} candidates")
    
    # Find matching function
    full_function = None
    for i, candidate in enumerate(candidates):
        print(f"[DEBUG] Candidate {i}: {candidate.name} in {candidate.file_path}")
        if candidate.file_path in patch['patch']:
            # Match by checking if some lines appear in patch
            code_lines = [l.strip() for l in candidate.src_code.split('\n') if l.strip()][:5]
            if any(line in patch['patch'] for line in code_lines):
                full_function = candidate.src_code
                print(f"[DEBUG] MATCHED! Using function: {candidate.name}")
                break
    
    if not full_function:
        print(f"[DEBUG] NO MATCH - Extracting from file using patch context...")
        # Get file path
        file_path = next((l.replace('--- a/', '').strip() for l in patch['patch'].split('\n') if l.startswith('--- a/')), None)
        
        # Get line number from patch
        for line in patch['patch'].split('\n'):
            if line.startswith('@@'):
                # Extract line number
                import re
                match = re.search(r'@@ -(\d+),', line)
                if match:
                    line_num = int(match.group(1))
                    break
        
        # Read file and extract function
        full_path = Path(repo_path) / file_path
        lines = full_path.read_text().split('\n')
        
        # Find function start (go backwards from line_num)
        start = line_num - 1
        while start > 0 and not lines[start].startswith('static ') and not lines[start].startswith('CJSON_PUBLIC'):
            start -= 1
        
        # Find function end (find matching brace)
        end = line_num
        brace_count = 0
        for i in range(start, len(lines)):
            brace_count += lines[i].count('{') - lines[i].count('}')
            if brace_count == 0 and i > start:
                end = i
                break
        
        full_function = '\n'.join(lines[start:end+1])
        print(f"[DEBUG] Extracted function: {len(full_function)} chars")
    
    if not full_function:
        print(f"[DEBUG] NO MATCH FOUND! Using fallback")
        full_function = patch['rewrite']
    
    print(f"[DEBUG] Full function length: {len(full_function)}")
    
    # Generate instruction
    instruction = generate_fix_instruction(patch, model, full_function)
    
    try:
        for attempt in range(max_retries):
            print(f"\n[DEBUG] ====== ATTEMPT {attempt + 1}/{max_retries} ======")
            
            try:
                # Generate fix
                fixed_code = generate_fix(full_function, instruction, model)
                
                # Create patch
                fix_patch = create_fix_patch_from_existing(patch, fixed_code, repo_path, full_function)
                
                if not fix_patch:
                    print(f"[DEBUG] Patch creation FAILED")
                    attempts.append({"attempt": attempt + 1, "status": "patch_failed", "reward": -1.0})
                    continue
                
                # Validate
                instance = {
                    KEY_INSTANCE_ID: f"{patch['instance_id']}_attempt_{attempt}",
                    KEY_PATCH: fix_patch,
                    "repo": repo,
                    'FAIL_TO_PASS': patch['FAIL_TO_PASS'],
                    'original_bug_patch': patch['patch'],  # Add this
                }
                result = validate_fix(instance, repo, rp.timeout)
                attempts.append({"attempt": attempt + 1, "fixed_code": fixed_code, "patch": fix_patch, **result})
                
                if result['tests_passed']:
                    print(f"[DEBUG] SUCCESS! Breaking retry loop")
                    break
                else:
                    print(f"[DEBUG] Tests FAILED, retrying...")
                    
            except Exception as e:
                print(f"[DEBUG] EXCEPTION: {e}")
                import traceback
                traceback.print_exc()
                attempts.append({"attempt": attempt + 1, "status": "error", "reward": -1.0, "error": str(e)})
    finally:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
    
    best = max(attempts, key=lambda x: (x.get('tests_passed', False), x.get('f2p_actual', 0)))
    
    return {
        "instance_id": patch['instance_id'],
        "fix_instruction": instruction,
        "full_function_used": full_function,
        "attempts": attempts,
        "best_attempt": best,
        "final_status": "success" if best.get('tests_passed', False) else "failed",
        "final_reward": best.get('reward', -1.0),
    }


def main(task_instances_path: str, model: str, n_workers: int = 4, max_fixes: int = -1, max_retries: int = 5):
    print(f"Loading from {task_instances_path}...")
    with open(task_instances_path, 'r') as f:
        instances = json.load(f)
    
    if max_fixes > 0:
        instances = instances[:max_fixes]
    
    print(f"Loaded {len(instances)} instances")
    repo = instances[0]['repo'].split('/')[-1]
    
    # Convert to patch format
    patches = []
    for inst in instances:
        patches.append({
            'instance_id': inst['instance_id'],
            'patch': inst['patch'],
            'repo': repo,
            'rewrite': inst['patch'],  # Store full patch
            'explanation': f"Tests fail: {', '.join(inst['FAIL_TO_PASS'][:3])}",
            'FAIL_TO_PASS': inst['FAIL_TO_PASS'],
        })
    
    output_dir = Path("logs/fix_gen") / repo
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Starting fix generation...")
    print("="*60)
    
    fix_results = []
    stats = {"success": 0, "failed": 0}
    rp = registry.get(repo)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(generate_and_validate_with_retry, patch, repo, model, rp, max_retries) for patch in patches]
        
        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                result = future.result()
                stats["success" if result["final_status"] == "success" else "failed"] += 1
                fix_results.append(result)
                pbar.set_postfix(stats)
                pbar.update(1)
    
    results_path = output_dir / "fix_results.json"
    with open(results_path, 'w') as f:
        json.dump(fix_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Success: {stats['success']}, Failed: {stats['failed']}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_instances_path", type=str)
    parser.add_argument("--model", type=str, default="openai/nemotron-nano")
    parser.add_argument("-w", "--n_workers", type=int, default=1)
    parser.add_argument("-m", "--max_fixes", type=int, default=-1)
    parser.add_argument("--max_retries", type=int, default=5)
    args = parser.parse_args()
    main(**vars(args))