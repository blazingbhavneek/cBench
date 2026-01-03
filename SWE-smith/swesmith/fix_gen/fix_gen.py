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


def generate_fix_instruction(patch_metadata: dict, model: str) -> str:
    """
    Generate a chat-style instruction telling WHAT to fix (not WHERE)
    
    Args:
        patch_metadata: Dict containing bug explanation and code
        model: LLM model to use
    
    Returns:
        Instruction string describing what behavior to fix
    """
    explanation = patch_metadata.get('explanation', 'Fix this buggy function')
    buggy_code = patch_metadata.get('rewrite', '')
    
    # Extract signature to give context without being too specific
    signature = extract_function_signature(buggy_code)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are creating fix instructions for a C function. "
                "Describe WHAT is wrong (the incorrect behavior), not WHERE in the code. "
                "Be concise and focus on the symptom/bug behavior. "
                "Do not mention line numbers, variable names, or specific code locations. "
                "Example: 'The function returns incorrect results when...' not 'Change line 5'"
            )
        },
        {
            "role": "user",
            "content": (
                f"Bug explanation from chaos testing:\n{explanation}\n\n"
                f"Function signature:\n{signature}\n\n"
                f"Create a brief instruction (1-3 sentences) describing WHAT needs to be fixed:"
            )
        }
    ]
    
    try:
        response = completion(model=model, messages=messages, temperature=0.3)
        instruction = response.choices[0].message.content.strip()
        return instruction
    except Exception as e:
        logger.warning(f"Failed to generate instruction, using default: {e}")
        return f"Fix the logical bug in this function. {explanation[:100]}"


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
    import subprocess
    
    DEVNULL = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    
    try:
        rp = registry.get(repo)
        repo_path, _ = rp.clone()  # Returns "swesmith/DaveGamble__cJSON.c859b25d"
        
        # Apply bug patch
        patch_file = Path("patch_temp.diff")
        patch_file.write_text(original_patch_meta['patch'])
        
        for git_cmd in ["git apply", "git apply --ignore-whitespace"]:
            result = subprocess.run(
                f"{git_cmd} ../../{patch_file}",
                cwd=repo_path,
                shell=True,
                **DEVNULL
            )
            if result.returncode == 0:
                break
        
        # Find file and replace
        file_path = next((l.replace('--- a/', '').strip() for l in original_patch_meta['patch'].split('\n') if l.startswith('--- a/')), None)
        full_path = Path(repo_path) / file_path
        content = full_path.read_text()
        content = content.replace(original_patch_meta['rewrite'], fixed_code, 1)
        full_path.write_text(content)
        
        # Generate patch
        patch = get_patch(repo_path, reset_changes=True)
        return patch if patch else ""
        
    finally:
        if patch_file.exists():
            patch_file.unlink()
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)


def validate_fix(instance: dict, repo: str, timeout: int) -> dict:
    """
    Validate a fix attempt in Docker container
    
    Args:
        instance: Instance dict with patch to validate
        repo: Repository name
        timeout: Timeout for validation
    
    Returns:
        Dict with validation results and RL reward
    """
    instance_id = instance[KEY_INSTANCE_ID]
    rp = registry.get(repo)
    valid_folder = LOG_DIR_RUN_VALIDATION / repo
    report_path = valid_folder / instance_id / LOG_REPORT
    
    # Run the fix patch in container
    logger, timed_out = run_patch_in_container(
        instance,
        repo,
        LOG_DIR_RUN_VALIDATION,
        timeout,
        patch=instance[KEY_PATCH],
    )
    
    if timed_out:
        logger.info(f"Timed out for {instance_id}")
        with open(report_path, "w") as f:
            json.dump({KEY_TIMED_OUT: True, "timeout": timeout}, f, indent=4)
        close_logger(logger)
        return {
            "status": "timeout",
            "reward": -1.0,
            "compiled": False,
            "tests_passed": False,
            "f2p_count": 0
        }
    
    val_pregold_path = valid_folder / instance_id / LOG_TEST_OUTPUT
    if not val_pregold_path.exists():
        logger.info(f"Validation failed for {instance_id}")
        close_logger(logger)
        return {
            "status": "failed",
            "reward": -1.0,
            "compiled": False,
            "tests_passed": False,
            "f2p_count": 0
        }
    
    # Get the reference pre-gold test output
    ref_inst_id = f"{repo}{REF_SUFFIX}"
    val_postgold_path = valid_folder / ref_inst_id / LOG_TEST_OUTPUT
    
    # Get grading report
    report = get_valid_report(
        val_pregold_path=str(val_pregold_path),
        val_postgold_path=str(val_postgold_path),
        instance=instance,
    )
    
    # Write report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    
    close_logger(logger)
    
    # Calculate RL reward
    f2p_count = len(report.get(FAIL_TO_PASS, []))
    tests_passed = f2p_count > 0
    
    return {
        "status": "success" if tests_passed else "failed",
        "reward": 1.0 if tests_passed else -1.0,
        "compiled": True,
        "tests_passed": tests_passed,
        "f2p_count": f2p_count,
        "report": report
    }


def generate_and_validate_with_retry(patch, repo, model, rp, max_retries=5):
    instruction = generate_fix_instruction(patch, model)
    attempts = []
    
    # Clone repo ONCE for all retries
    repo_path, _ = rp.clone()
    
    try:
        for attempt in range(max_retries):
            try:
                # Generate fix
                fixed_code = generate_fix(patch['rewrite'], instruction, model)
                
                # Create patch (reuses existing clone)
                fix_patch = create_fix_patch_from_existing(
                    patch, fixed_code, repo_path
                )
                
                if not fix_patch:
                    attempts.append({"attempt": attempt + 1, "status": "patch_failed", "reward": -1.0})
                    continue
                
                # Validate
                instance = {
                    KEY_INSTANCE_ID: f"{patch['instance_id']}_attempt_{attempt}",
                    KEY_PATCH: fix_patch,
                    "repo": repo,
                }
                result = validate_fix(instance, repo, rp.timeout)
                
                attempts.append({"attempt": attempt + 1, "fixed_code": fixed_code, "patch": fix_patch, **result})
                
                if result['tests_passed']:
                    break
                    
            except Exception as e:
                attempts.append({"attempt": attempt + 1, "status": "error", "reward": -1.0, "error": str(e)})
    finally:
        # Cleanup once
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
    
    best = max(attempts, key=lambda x: (x.get('tests_passed', False), x.get('f2p_count', 0)))
    return {
        "instance_id": patch['instance_id'],
        "fix_instruction": instruction,
        "attempts": attempts,
        "best_attempt": best,
        "final_status": "success" if best.get('tests_passed', False) else "failed",
        "final_reward": best.get('reward', -1.0),
    }

def create_fix_patch_from_existing(patch_meta, fixed_code, repo_path):
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
        content = full_path.read_text().replace(patch_meta['rewrite'], fixed_code, 1)
        full_path.write_text(content)
        
        return get_patch(repo_path, reset_changes=True)
    finally:
        if patch_file.exists():
            patch_file.unlink()

def main(
    buggy_patches: str,
    model: str,
    n_workers: int = 4,
    max_fixes: int = -1,
    max_retries: int = 5,
):
    """
    Main pipeline: Generate fix instructions, generate fixes with retry, validate
    """
    print(f"Loading buggy patches from {buggy_patches}...")
    with open(buggy_patches, 'r') as f:
        patches = json.load(f)
    
    # Limit number of patches if specified
    if max_fixes > 0 and len(patches) > max_fixes:
        patches = patches[:max_fixes]
        print(f"Limited to {len(patches)} patches (max_fixes={max_fixes})")
    
    print(f"Loaded {len(patches)} buggy patches")
    
    # Extract repo name for logging
    repo = patches[0]['repo'] if patches else "unknown"
    
    # Setup output directory
    output_dir = Path("logs/fix_gen") / repo
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # TASK 1: Generate instructions and fixes with retry logic
    print("\n" + "="*60)
    print("TASK 1: Generating fix instructions and fixes with retry...")
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
                if result["final_status"] == "success":
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                fix_results.append(result)
                pbar.set_postfix(stats)
                pbar.update(1)
    
    print(f"\nTask 1 Complete: {stats['success']} successful fixes, {stats['failed']} failed")
    
    # Save results
    results_path = output_dir / "fix_results_with_retries.json"
    with open(results_path, 'w') as f:
        json.dump(fix_results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    # Print detailed summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total patches processed: {len(patches)}")
    print(f"Successful fixes (reward +1): {stats['success']}")
    print(f"Failed fixes (reward -1): {stats['failed']}")
    
    # Calculate retry statistics
    total_attempts = sum(len(r['attempts']) for r in fix_results)
    avg_attempts = total_attempts / len(fix_results) if fix_results else 0
    print(f"\nRetry statistics:")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Average attempts per patch: {avg_attempts:.1f}")
    print(f"  Max retries allowed: {max_retries}")
    
    # Calculate success rate
    if len(fix_results) > 0:
        success_rate = (stats['success'] / len(fix_results)) * 100
        print(f"\nOverall success rate: {success_rate:.1f}%")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "buggy_patches",
        type=str,
        help="Path to JSON file with buggy patches (output from collect_patches.py)",
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
