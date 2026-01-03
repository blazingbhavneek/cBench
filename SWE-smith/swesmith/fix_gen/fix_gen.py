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

# ===== DEBUG FLAG - SET TO True TO ENABLE DEBUG PRINTS =====
DEBUG = False
# ===========================================================

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

def debug_print(message):
    """Helper function to print debug messages only when DEBUG flag is True"""
    if DEBUG:
        print(message)

def error_print(message):
    """Helper function to print error messages regardless of DEBUG flag"""
    logger.error(message)
    print(f"\033[91m{message}\033[0m")  # Red color for errors

def warning_print(message):
    """Helper function to print warning messages regardless of DEBUG flag"""
    logger.warning(message)
    print(f"\033[93m{message}\033[0m")  # Yellow color for warnings


def create_fix_patch_from_existing(patch_meta, fixed_code, repo_path, original_full_function):
    debug_print(f"\n[DEBUG] Creating fix patch...")
    debug_print(f"[DEBUG] Repo path: {repo_path}")
    debug_print(f"[DEBUG] Original function length: {len(original_full_function)}")
    debug_print(f"[DEBUG] Fixed code length: {len(fixed_code)}")
    
    import subprocess
    DEVNULL = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    patch_file = Path("temp_patch.diff")
    
    try:
        # Reset
        debug_print(f"[DEBUG] Resetting repo...")
        subprocess.run(["git", "-C", repo_path, "reset", "--hard"], check=True, **DEVNULL)
        subprocess.run(["git", "-C", repo_path, "clean", "-fdx"], check=True, **DEVNULL)
        
        # Apply bug patch and commit it
        debug_print(f"[DEBUG] Applying bug patch...")
        patch_file.write_text(patch_meta['patch'])
        subprocess.run(["git", "-C", repo_path, "apply", str(patch_file.absolute())], check=True, **DEVNULL)
        subprocess.run(["git", "-C", repo_path, "add", "-A"], check=True, **DEVNULL)
        subprocess.run(["git", "-C", repo_path, "commit", "-m", "Apply bug"], check=True, **DEVNULL)
        
        # NOW extract the buggy function from the modified file
        file_path = next((l.replace('--- a/', '').strip() for l in patch_meta['patch'].split('\n') if l.startswith('--- a/')), None)
        debug_print(f"[DEBUG] Target file: {file_path}")
        
        full_path = Path(repo_path) / file_path
        content = full_path.read_text()
        debug_print(f"[DEBUG] File content length (after bug patch): {len(content)}")
        
        # Extract function name from original_full_function
        import re
        match = re.search(r'(static\s+\w+\s+\**)?(\w+)\s*\(', original_full_function)
        if not match:
            error_print(f"[ERROR] Could not extract function name from original function")
            return None
        func_name = match.group(2)

        debug_print(f"[DEBUG] Function name: {func_name}")
        
        lines = content.split('\n')

        # Find start of function
        start = None
        for i, line in enumerate(lines):
            if func_name in line and '(' in line:
                start = i
                break

        if start is None:
            error_print(f"[ERROR] Function {func_name} not found in file!")
            return None

        # Find end (matching braces)
        end = start
        brace_count = 0
        for i in range(start, len(lines)):
            brace_count += lines[i].count('{') - lines[i].count('}')
            if brace_count == 0 and i > start:
                end = i
                break

        debug_print(f"[DEBUG] Function found: start={start}, end={end}")
        buggy_in_file = '\n'.join(lines[start:end+1])
        debug_print(f"[DEBUG] Extracted buggy function length: {len(buggy_in_file)}")
        debug_print(f"[DEBUG] Buggy function:\n{buggy_in_file}\n")
        
        debug_print(f"[DEBUG] Fixed code:\n{fixed_code}\n")
        
        if buggy_in_file == fixed_code:
            warning_print(f"[WARNING] Buggy and fixed code are IDENTICAL! No changes made.")
            return None
        
        content = content.replace(buggy_in_file, fixed_code, 1)
        debug_print(f"[DEBUG] Replacement done. New content length: {len(content)}")
        
        # Verify replacement
        if fixed_code in content:
            debug_print(f"[DEBUG] Fixed code found in content")
            idx = content.find(fixed_code)
            debug_print(f"[DEBUG] Fixed function at position {idx}:\n{content[idx:idx+200]}")
        else:
            warning_print(f"[WARNING] Fixed code NOT found in content after replacement!")
        
        full_path.write_text(content)
        debug_print(f"[DEBUG] File written. Checking git diff...")
        
        result = subprocess.run(["git", "-C", repo_path, "diff", "--stat"], capture_output=True, text=True)
        debug_print(f"[DEBUG] Git diff stat:\n{result.stdout}")
        
        # Capture patch BEFORE resetting
        debug_print(f"[DEBUG] Capturing git diff...")
        result = subprocess.run(["git", "-C", repo_path, "diff", "HEAD"], capture_output=True, text=True)
        patch = result.stdout if result.stdout else None
        
        debug_print(f"[DEBUG] Generated patch length: {len(patch) if patch else 0}")
        if patch:
            debug_print(f"[DEBUG] Patch (first 300 chars):\n{patch[:300]}")
        
        # Now reset
        debug_print(f"[DEBUG] Resetting repo...")
        subprocess.run(["git", "-C", repo_path, "reset", "--hard"], check=True, **DEVNULL)
        
        return patch
    except Exception as e:
        error_print(f"[ERROR] Failed to create fix patch: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
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
        error_print(f"[ERROR] Validation timed out for instance {instance_id}")
        return {"reward": -1.0, "tests_passed": False}
    
    # Parse test output
    test_output_path = valid_folder / instance_id / LOG_TEST_OUTPUT
    if not test_output_path.exists():
        error_print(f"[ERROR] Test output file not found: {test_output_path}")
        close_logger(logger)
        return {"reward": -1.0, "tests_passed": False}
    
    test_output = test_output_path.read_text()
    test_status_map = rp.log_parser(test_output)
    
    close_logger(logger)
    
    # Check if ALL tests pass
    all_pass = all(status == "PASSED" for status in test_status_map.values())
    
    if not all_pass:
        debug_print(f"[DEBUG] Test failures for {instance_id}:")
        for test_name, status in test_status_map.items():
            if status != "PASSED":
                debug_print(f"[DEBUG]   {test_name}: {status}")
    
    return {
        "reward": 1.0 if all_pass else -1.0,
        "tests_passed": all_pass,
        "test_results": test_status_map
    }

def generate_fix_instruction(patch_metadata: dict, model: str, buggy_function: str) -> str:
    debug_print(f"\n[DEBUG] Generating instruction...")
    debug_print(f"[DEBUG] Buggy function:\n{buggy_function}\n")
    
    messages = [
        {
            "role": "system",
            "content": "You are analyzing a buggy C function. Describe WHAT is wrong with the behavior (not WHERE in the code). Focus on what the function does incorrectly."
        },
        {
            "role": "user",
            "content": (
                f"This function has a bug:\n\n"
                f"```c\n{buggy_function}\n```\n\n"
                f"Tests that fail: {patch_metadata.get('explanation', 'whitespace parsing tests')}\n\n"
                f"Describe what behavior is wrong (e.g., 'function doesn't handle X correctly', 'function skips Y when it should include it'):"
            )
        }
    ]
    
    debug_print(f"[DEBUG] Calling LLM for instruction...")
    try:
        response = completion(model=model, messages=messages, temperature=0.3)
        instruction = response.choices[0].message.content.strip()
        debug_print(f"[DEBUG] Generated instruction: {instruction}")
        return instruction
    except Exception as e:
        error_print(f"[ERROR] Failed to generate instruction: {str(e)}")
        raise

def generate_fix(buggy_function: str, instruction: str, model: str) -> str:
    debug_print(f"\n[DEBUG] Generating fix...")
    debug_print(f"[DEBUG] Instruction: {instruction}")
    debug_print(f"[DEBUG] Buggy function length: {len(buggy_function)} chars")
    
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
    
    debug_print(f"[DEBUG] Calling LLM for fix...")
    try:
        response = completion(model=model, messages=messages, temperature=0.7)
        fixed_code = extract_code_block(response.choices[0].message.content)
        debug_print(f"[DEBUG] Fixed code length: {len(fixed_code)} chars")
        return fixed_code
    except Exception as e:
        error_print(f"[ERROR] Failed to generate fix: {str(e)}")
        raise

def create_fix_patch(buggy_function, fixed_code, repo_path, file_path):
    """Create a patch by replacing buggy function with fixed code"""
    debug_print(f"[DEBUG] create_fix_patch called")
    
    import subprocess
    DEVNULL = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    
    # Read file
    full_path = Path(repo_path) / file_path
    if not full_path.exists():
        error_print(f"[ERROR] File not found: {full_path}")
        return None
    
    content = full_path.read_text()
    debug_print(f"[DEBUG] File content length: {len(content)}")
    
    # Replace function
    if buggy_function not in content:
        warning_print(f"[WARNING] Buggy function not found in file!")
        debug_print(f"[DEBUG] Looking for:\n{buggy_function[:200]}")
        return None
    
    new_content = content.replace(buggy_function, fixed_code, 1)
    debug_print(f"[DEBUG] Replacement done. New length: {len(new_content)}")
    
    # Write file
    full_path.write_text(new_content)
    
    # Capture diff
    result = subprocess.run(["git", "-C", repo_path, "diff", "HEAD"], capture_output=True, text=True)
    patch = result.stdout if result.stdout else None
    
    if not patch or len(patch.strip()) == 0:
        warning_print(f"[WARNING] Generated patch is empty!")
    
    # Reset
    subprocess.run(["git", "-C", repo_path, "reset", "--hard"], check=True, **DEVNULL)
    
    return patch


def generate_and_validate_with_retry(patch, repo, model, rp, max_retries=5):
    debug_print(f"\n{'='*60}")
    debug_print(f"[DEBUG] Processing {patch['instance_id']}")
    debug_print(f"{'='*60}")
    
    attempts = []
    
    # Clone repo
    debug_print(f"[DEBUG] Cloning repo...")
    try:
        repo_path, _ = rp.clone()
        debug_print(f"[DEBUG] Repo cloned to: {repo_path}")
    except Exception as e:
        error_print(f"[ERROR] Failed to clone repo: {str(e)}")
        return {"instance_id": patch['instance_id'], "final_status": "failed", "final_reward": -1.0, "attempts": []}
    
    import subprocess
    import re
    DEVNULL = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    
    try:
        # Apply bug patch and commit
        debug_print(f"[DEBUG] Applying bug patch and committing...")
        patch_file = Path("temp_bug_patch.diff")
        patch_file.write_text(patch['patch'])
        subprocess.run(["git", "-C", repo_path, "apply", str(patch_file.absolute())], check=True, **DEVNULL)
        subprocess.run(["git", "-C", repo_path, "add", "-A"], check=True, **DEVNULL)
        subprocess.run(["git", "-C", repo_path, "commit", "-m", "Apply bug"], check=True, **DEVNULL)
        patch_file.unlink()
        
        # Extract buggy function from file AFTER bug patch
        file_path = next((l.replace('--- a/', '').strip() for l in patch['patch'].split('\n') if l.startswith('--- a/')), None)
        debug_print(f"[DEBUG] Target file: {file_path}")
        
        if not file_path:
            error_print(f"[ERROR] Could not find target file in patch!")
            return {"instance_id": patch['instance_id'], "final_status": "failed", "final_reward": -1.0, "attempts": []}
        
        full_path = Path(repo_path) / file_path
        if not full_path.exists():
            error_print(f"[ERROR] File not found after applying patch: {full_path}")
            return {"instance_id": patch['instance_id'], "final_status": "failed", "final_reward": -1.0, "attempts": []}
        
        content = full_path.read_text()
        
        # Get function name from @@ hunk header
        func_name = None
        for line in patch['patch'].split('\n'):
            if line.startswith('@@'):
                # Extract function name from hunk header like "@@ -1094,7 +1094,7 @@ static parse_buffer *buffer_skip_whitespace(parse_buffer * const buffer)"
                match = re.search(r'@@[^@]+@@\s+(?:static\s+)?(?:\w+\s+\**)?\s*(\w+)\s*\(', line)
                if match:
                    func_name = match.group(1)
                    break
        
        if not func_name:
            warning_print(f"[WARNING] Could not find function name in patch!")
            debug_print(f"[DEBUG] Patch preview:\n{patch['patch'][:500]}")
            return {"instance_id": patch['instance_id'], "final_status": "failed", "final_reward": -1.0, "attempts": []}
        
        debug_print(f"[DEBUG] Function name: {func_name}")
        
        # Find function in file
        lines = content.split('\n')
        start = None
        for i, line in enumerate(lines):
            if func_name in line and '(' in line:
                start = i
                break
        
        if start is None:
            error_print(f"[ERROR] Function {func_name} not found in file!")
            return {"instance_id": patch['instance_id'], "final_status": "failed", "final_reward": -1.0, "attempts": []}
        
        # Find end (matching braces)
        end = start
        brace_count = 0
        for i in range(start, len(lines)):
            brace_count += lines[i].count('{') - lines[i].count('}')
            if brace_count == 0 and i > start:
                end = i
                break
        
        buggy_function = '\n'.join(lines[start:end+1])
        debug_print(f"[DEBUG] Extracted buggy function ({len(buggy_function)} chars):")
        debug_print(f"[DEBUG] {buggy_function}\n")
        
        if not buggy_function or len(buggy_function.strip()) == 0:
            error_print(f"[ERROR] Extracted buggy function is empty!")
            return {"instance_id": patch['instance_id'], "final_status": "failed", "final_reward": -1.0, "attempts": []}
        
        # Generate instruction using BUGGY function
        debug_print(f"[DEBUG] Generating fix instruction from BUGGY function...")
        try:
            instruction = generate_fix_instruction(patch, model, buggy_function)
            debug_print(f"[DEBUG] Instruction: {instruction}\n")
        except Exception as e:
            error_print(f"[ERROR] Failed to generate instruction: {str(e)}")
            return {"instance_id": patch['instance_id'], "final_status": "failed", "final_reward": -1.0, "attempts": []}
        
        # Retry loop
        success = False
        for attempt in range(max_retries):
            debug_print(f"\n[DEBUG] ====== ATTEMPT {attempt + 1}/{max_retries} ======")
            
            try:
                # Generate fix from BUGGY function
                debug_print(f"[DEBUG] Generating fix from BUGGY function...")
                fixed_code = generate_fix(buggy_function, instruction, model)
                debug_print(f"[DEBUG] Fixed code ({len(fixed_code)} chars):")
                debug_print(f"[DEBUG] {fixed_code}\n")
                
                if not fixed_code or len(fixed_code.strip()) == 0:
                    warning_print(f"[WARNING] Generated fixed code is empty!")
                    attempts.append({"attempt": attempt + 1, "status": "empty_fix", "reward": -1.0})
                    continue
                
                # Check if identical
                if buggy_function.strip() == fixed_code.strip():
                    warning_print(f"[WARNING] Fixed code is IDENTICAL to buggy code (after stripping whitespace)!")
                    attempts.append({"attempt": attempt + 1, "status": "identical", "reward": -1.0})
                    continue
                
                # Create patch
                debug_print(f"[DEBUG] Creating patch...")
                fix_patch = create_fix_patch(buggy_function, fixed_code, repo_path, file_path)
                
                if not fix_patch:
                    warning_print(f"[WARNING] Patch creation FAILED")
                    attempts.append({"attempt": attempt + 1, "status": "patch_failed", "reward": -1.0})
                    continue
                
                debug_print(f"[DEBUG] Patch created ({len(fix_patch)} chars)")
                debug_print(f"[DEBUG] Patch preview:\n{fix_patch[:300]}\n")
                
                # Validate
                instance = {
                    KEY_INSTANCE_ID: f"{patch['instance_id']}_attempt_{attempt}",
                    KEY_PATCH: fix_patch,
                    "repo": repo,
                    FAIL_TO_PASS: patch['FAIL_TO_PASS'],
                    'original_bug_patch': patch['patch'],
                }
                result = validate_fix(instance, repo, rp.timeout)
                attempts.append({"attempt": attempt + 1, "fixed_code": fixed_code, "patch": fix_patch, **result})
                
                if result['tests_passed']:
                    debug_print(f"[DEBUG] SUCCESS! All tests passed!")
                    success = True
                    break
                else:
                    debug_print(f"[DEBUG] Tests FAILED")
                    
            except Exception as e:
                error_print(f"[ERROR] Exception during attempt {attempt + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                attempts.append({"attempt": attempt + 1, "status": "error", "reward": -1.0, "error": str(e)})
        
        if not success and attempts:
            debug_print(f"[DEBUG] All attempts failed for {patch['instance_id']}")
    
    except Exception as e:
        error_print(f"[ERROR] Critical error processing {patch['instance_id']}: {str(e)}")
        import traceback
        traceback.print_exc()
        attempts.append({"attempt": "critical_error", "status": "critical_error", "reward": -1.0, "error": str(e)})
    
    finally:
        if os.path.exists(repo_path):
            try:
                shutil.rmtree(repo_path)
                debug_print(f"[DEBUG] Cleaned up repo directory: {repo_path}")
            except Exception as e:
                error_print(f"[ERROR] Failed to clean up repo directory: {str(e)}")
    
    best = max(attempts, key=lambda x: (x.get('tests_passed', False), x.get('reward', -1.0))) if attempts else {"reward": -1.0}
    
    return {
        "instance_id": patch['instance_id'],
        "fix_instruction": instruction if 'instruction' in locals() else None,
        "buggy_function_used": buggy_function if 'buggy_function' in locals() else None,
        "attempts": attempts,
        "best_attempt": best,
        "final_status": "success" if best.get('tests_passed', False) else "failed",
        "final_reward": best.get('reward', -1.0),
    }



def main(task_instances_path: str, model: str, n_workers: int = 4, max_fixes: int = -1, max_retries: int = 5):
    if DEBUG:
        print(f"[DEBUG] Loading from {task_instances_path}...")
    
    try:
        with open(task_instances_path, 'r') as f:
            instances = json.load(f)
    except Exception as e:
        error_print(f"[ERROR] Failed to load task instances: {str(e)}")
        raise
    
    if max_fixes > 0:
        instances = instances[:max_fixes]
        debug_print(f"[DEBUG] Limited to {max_fixes} instances")
    
    debug_print(f"Loaded {len(instances)} instances")
    if not instances:
        error_print("[ERROR] No instances loaded!")
        return
    
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
    
    debug_print("\n" + "="*60)
    debug_print("Starting fix generation...")
    debug_print("="*60)
    
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
    try:
        with open(results_path, 'w') as f:
            json.dump(fix_results, f, indent=2)
        debug_print(f"[DEBUG] Results saved to {results_path}")
    except Exception as e:
        error_print(f"[ERROR] Failed to save results: {str(e)}")
    
    debug_print(f"\n{'='*60}")
    debug_print("SUMMARY")
    debug_print(f"{'='*60}")
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
    
    debug_print(f"[DEBUG] Starting with args: {args}")
    main(**vars(args))
