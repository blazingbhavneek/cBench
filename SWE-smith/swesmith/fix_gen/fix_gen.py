"""
Purpose: Generate fixes for buggy functions and validate them

Task 1: Generate chat-style instructions + LLM fixes for buggy functions
Task 2: Validate the fixes using Docker containers and extract RL rewards

Usage: 
python swesmith/fix_gen/fix_generator.py \
    --buggy_patches logs/bug_gen/swesmith/DaveGamble__cJSON.c859b25d_all_patches.json \
    --model openai/nemotron-nano \
    --n_workers 4 \
    --max_fixes 10

This will:
1. Load buggy patches
2. Generate fix instructions (what to fix, not where)
3. Generate fixed code using LLM
4. Apply fixes and create patches
5. Validate in Docker containers
6. Extract RL rewards (+1 if tests pass, -1 if fail)
7. Save everything to logs/fix_gen/<repo_name>/
"""

import argparse
import json
import litellm
import logging
import os
import shutil
import threading

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

# Import from existing swesmith modules
from swesmith.bug_gen.llm.utils import extract_code_block
from swesmith.bug_gen.utils import get_patch
from swesmith.constants import (
    KEY_PATCH,
    KEY_TIMED_OUT,
    LOG_DIR_RUN_VALIDATION,
    REF_SUFFIX,
    TEMP_PATCH,
)
from swesmith.harness.grading import get_valid_report
from swesmith.harness.utils import run_patch_in_container
from swesmith.profiles import registry

load_dotenv(dotenv_path=os.getenv("SWEFT_DOTENV_PATH"))

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
litellm.suppress_debug_info = True

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_function_signature(code: str) -> str:
    """Extract just the function signature from C code"""
    lines = code.strip().split('\n')
    signature_lines = []
    
    for line in lines:
        signature_lines.append(line)
        if '{' in line:
            break
    
    return '\n'.join(signature_lines).replace('{', '').strip()


def generate_fix_instruction(patch_metadata: dict, model: str, full_function: str) -> str:
    """Generate instruction with full context"""
    
    messages = [
        {
            "role": "system",
            "content": (
                "Create a fix instruction for buggy code. "
                "Describe WHAT is wrong (behavior), not WHERE. "
                "Be concise, human-like. "
                "Example: 'Function fails to handle null inputs correctly'"
            )
        },
        {
            "role": "user",
            "content": (
                f"**Full Function:**\n```c\n{full_function}\n```\n\n"
                f"**Bug Diff:**\n```diff\n{patch_metadata['patch']}\n```\n\n"
                f"**Failing Tests:** {patch_metadata.get('explanation', '')}\n\n"
                f"Create a brief instruction describing what needs to be fixed:"
            )
        }
    ]
    
    try:
        response = completion(model=model, messages=messages, temperature=0.3)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Instruction generation failed: {e}")
        return f"Fix the bug causing test failures: {patch_metadata.get('explanation', '')}"
    

def generate_fix(buggy_function: str, instruction: str, model: str) -> str:
    """
    Generate fixed code using LLM
    
    Args:
        buggy_function: The buggy C function code
        instruction: What to fix (behavioral description)
        model: LLM model to use
    
    Returns:
        Fixed function code
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert C programmer. "
                "You will receive a buggy function and an instruction describing what's wrong. "
                "Fix the function so it behaves correctly. "
                "Output ONLY the complete fixed function code in a code block, no explanation or comments about the fix."
            )
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
    
    response = completion(model=model, messages=messages, temperature=0.7)
    fixed_code = extract_code_block(response.choices[0].message.content)
    return fixed_code


def create_fix_patch(original_patch_meta: dict, fixed_code: str, repo: str) -> str:
    """
    Apply the fixed code and generate a patch
    
    Args:
        original_patch_meta: Original bug patch metadata
        fixed_code: The fixed function code
        repo: Repository name
    
    Returns:
        Git diff patch string
    """
    import subprocess
    from swesmith.constants import TEMP_PATCH
    
    DEVNULL = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    
    # Clone repo to swesmith/ directory
    rp = registry.get(repo)
    rp.clone()
    repo_path = f"swesmith/{repo}"
    
    try:
        # Apply the original bug patch first (to get to buggy state)
        original_patch = original_patch_meta['patch']
        patch_file = Path(TEMP_PATCH)
        patch_file.write_text(original_patch)
        
        subprocess.run(
            ["git", "-C", repo_path, "apply", TEMP_PATCH],
            check=True,
            **DEVNULL
        )
        
        # Extract file path from the patch
        file_path = None
        for line in original_patch.split('\n'):
            if line.startswith('--- a/'):
                file_path = line.replace('--- a/', '').strip()
                break
        
        if not file_path:
            raise ValueError("Could not extract file path from patch")
        
        # Read the current (buggy) file
        full_path = Path(repo_path) / file_path
        file_content = full_path.read_text()
        
        # Replace the buggy function with fixed function
        buggy_func = original_patch_meta['rewrite']
        
        if buggy_func in file_content:
            file_content = file_content.replace(buggy_func, fixed_code, 1)
            full_path.write_text(file_content)
        else:
            raise ValueError("Buggy function not found in file")
        
        # Generate patch from this fix
        patch = get_patch(repo_path, reset_changes=True)
        return patch if patch else ""
        
    except Exception as e:
        logger.error(f"Failed to create fix patch: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ""
    finally:
        # Cleanup
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        if os.path.exists(TEMP_PATCH):
            os.remove(TEMP_PATCH)


def validate_fix(instance: dict, repo: str, timeout: int) -> dict:
    instance_id = instance[KEY_INSTANCE_ID]
    rp = registry.get(repo)
    valid_folder = LOG_DIR_RUN_VALIDATION / repo
    report_path = valid_folder / instance_id / LOG_REPORT
    
    # Run fix patch in container
    logger, timed_out = run_patch_in_container(
        instance, repo, LOG_DIR_RUN_VALIDATION, timeout,
        patch=instance[KEY_PATCH],
    )
    
    if timed_out or not (valid_folder / instance_id / LOG_TEST_OUTPUT).exists():
        close_logger(logger)
        return {"status": "timeout", "reward": -1.0, "compiled": False, "tests_passed": False}
    
    # Get test output
    val_pregold_path = valid_folder / instance_id / LOG_TEST_OUTPUT
    ref_inst_id = f"{repo}{REF_SUFFIX}"
    val_postgold_path = valid_folder / ref_inst_id / LOG_TEST_OUTPUT
    
    report = get_valid_report(
        val_pregold_path=str(val_pregold_path),
        val_postgold_path=str(val_postgold_path),
        instance=instance,
    )
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    close_logger(logger)
    
    # Success = ALL originally failing tests now pass AND no regressions
    expected_f2p = set(instance.get('FAIL_TO_PASS', []))
    actual_f2p = set(report.get(FAIL_TO_PASS, []))
    regressions = len(report.get('PASS_TO_FAIL', []))
    
    all_fixed = expected_f2p == actual_f2p
    no_breaks = regressions == 0
    tests_passed = all_fixed and no_breaks
    
    return {
        "status": "success" if tests_passed else "failed",
        "reward": 1.0 if tests_passed else -1.0,
        "compiled": True,
        "tests_passed": tests_passed,
        "f2p_expected": len(expected_f2p),
        "f2p_actual": len(actual_f2p),
        "regressions": regressions,
        "report": report
    }


def create_fix_patch_from_existing(patch_meta, fixed_code, repo_path, original_full_function):
    """Create patch from already-cloned repo"""
    import subprocess
    
    DEVNULL = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    patch_file = Path("temp_patch.diff")
    
    try:
        # Reset to clean state
        subprocess.run(["git", "-C", repo_path, "reset", "--hard"], check=True, **DEVNULL)
        subprocess.run(["git", "-C", repo_path, "clean", "-fdx"], check=True, **DEVNULL)
        
        # Apply bug patch
        patch_file.write_text(patch_meta['patch'])
        subprocess.run(["git", "-C", repo_path, "apply", str(patch_file.absolute())], check=True, **DEVNULL)
        
        # Replace function
        file_path = next((l.replace('--- a/', '').strip() for l in patch_meta['patch'].split('\n') if l.startswith('--- a/')), None)
        full_path = Path(repo_path) / file_path
        content = full_path.read_text().replace(original_full_function, fixed_code, 1)
        full_path.write_text(content)
        
        return get_patch(repo_path, reset_changes=True)
    finally:
        if patch_file.exists():
            patch_file.unlink()


def generate_and_validate_with_retry(patch, repo, model, rp, max_retries=5):
    attempts = []
    
    # Clone and extract full function
    repo_path, _ = rp.clone()
    candidates = rp.extract_entities()
    
    # Find the function that matches this patch
    full_function = None
    for candidate in candidates:
        if candidate.file_path in patch['patch']:
            # Check if this is the right function by matching some code
            if any(line.strip() in patch['patch'] for line in candidate.src_code.split('\n')[:3]):
                full_function = candidate.src_code
                break
    
    instruction = generate_fix_instruction(patch, model, full_function)
    
    if not full_function:
        # Fallback: use buggy_code
        full_function = patch['rewrite']
    
    try:
        for attempt in range(max_retries):
            try:
                # Send FULL function to LLM
                fixed_code = generate_fix(full_function, instruction, model)
                
                # Create patch using full function replacement
                fix_patch = create_fix_patch_from_existing(patch, fixed_code, repo_path, full_function)
                
                if not fix_patch:
                    attempts.append({"attempt": attempt + 1, "status": "patch_failed", "reward": -1.0})
                    continue
                
                instance = {
                    KEY_INSTANCE_ID: f"{patch['instance_id']}_attempt_{attempt}",
                    KEY_PATCH: fix_patch,
                    "repo": repo,
                    'FAIL_TO_PASS': patch['FAIL_TO_PASS'],
                }
                result = validate_fix(instance, repo, rp.timeout)
                attempts.append({"attempt": attempt + 1, "fixed_code": fixed_code, "patch": fix_patch, **result})
                
                if result['tests_passed']:
                    break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                attempts.append({"attempt": attempt + 1, "status": "error", "reward": -1.0, "error": str(e)})
    finally:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
    
    best = max(attempts, key=lambda x: (x.get('tests_passed', False), x.get('f2p_count', 0)))
    return {
        "instance_id": patch['instance_id'],
        "fix_instruction": instruction,
        "buggy_code": patch['rewrite'],
        "attempts": attempts,
        "best_attempt": best,
        "final_status": "success" if best.get('tests_passed', False) else "failed",
        "final_reward": best.get('reward', -1.0),
    }


def extract_buggy_code_from_patch(patch: str) -> str:
    """Extract modified lines from diff"""
    lines = []
    for line in patch.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            lines.append(line[1:])
        elif line.startswith(' '):  # Context lines
            lines.append(line[1:])
    
    result = '\n'.join(lines).strip()
    return result if result else "Code from patch"



def main(
    task_instances_path: str,
    model: str,
    n_workers: int = 4,
    max_fixes: int = -1,
    max_retries: int = 5,
):
    """
    Main pipeline: Load task instances, generate fixes with retry, validate
    """
    print(f"Loading task instances from {task_instances_path}...")
    with open(task_instances_path, 'r') as f:
        instances = json.load(f)
    
    if max_fixes > 0 and len(instances) > max_fixes:
        instances = instances[:max_fixes]
        print(f"Limited to {len(instances)} instances (max_fixes={max_fixes})")
    
    print(f"Loaded {len(instances)} task instances")
    
    # Extract repo name
    repo = instances[0]['repo'].split('/')[-1] if instances else "unknown"
    
    # Convert task instances to patch format
    patches = []
    for inst in instances:
        patches.append({
            'instance_id': inst['instance_id'],
            'patch': inst['patch'],
            'repo': repo,
            'rewrite': extract_buggy_code_from_patch(inst['patch']),
            'explanation': f"Tests fail: {', '.join(inst['FAIL_TO_PASS'][:3])}",
            'FAIL_TO_PASS': inst['FAIL_TO_PASS'],  # UPPERCASE
        })

    
    # Setup output directory
    output_dir = Path("logs/fix_gen") / repo
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Generate and validate fixes with retry
    print("\n" + "="*60)
    print("Generating & validating fixes with retry...")
    print("="*60)
    
    fix_results = []
    stats = {"success": 0, "failed": 0}
    rp = registry.get(repo)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(generate_and_validate_with_retry, patch, repo, model, rp, max_retries)
            for patch in patches
        ]
        
        with tqdm(total=len(futures), desc="Generating & validating fixes") as pbar:
            for future in as_completed(futures):
                result = future.result()
                stats["success" if result["final_status"] == "success" else "failed"] += 1
                fix_results.append(result)
                pbar.set_postfix(stats)
                pbar.update(1)
    
    # Save results
    results_path = output_dir / "fix_results_with_retries.json"
    with open(results_path, 'w') as f:
        json.dump(fix_results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total instances: {len(instances)}")
    print(f"Successful fixes (reward +1): {stats['success']}")
    print(f"Failed fixes (reward -1): {stats['failed']}")
    
    total_attempts = sum(len(r['attempts']) for r in fix_results)
    avg_attempts = total_attempts / len(fix_results) if fix_results else 0
    print(f"\nRetry statistics:")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Average per instance: {avg_attempts:.1f}")
    print(f"  Max retries: {max_retries}")
    
    if fix_results:
        success_rate = (stats['success'] / len(fix_results)) * 100
        print(f"\nSuccess rate: {success_rate:.1f}%")
    
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task_instances_path",
        type=str,
        help="Path to task_instances JSON file (e.g., logs/task_insts/DaveGamble__cJSON.c859b25d.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/nemotron-nano",
        help="LLM model to use for fix generation",
    )
    parser.add_argument(
        "-w",
        "--n_workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "-m",
        "--max_fixes",
        type=int,
        default=-1,
        help="Maximum number of fixes to generate (-1 for all)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum retry attempts per patch (default: 5)",
    )
    args = parser.parse_args()
    main(**vars(args))
