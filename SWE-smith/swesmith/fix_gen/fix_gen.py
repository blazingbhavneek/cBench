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
    
    # Apply bug patch first, then fix patch
    bug_patch = instance.get('bug_patch', '')
    fix_patch = instance[KEY_PATCH]
    
    # Combine patches: bug first, then fix
    combined_patch = bug_patch + '\n' + fix_patch if bug_patch else fix_patch
    
    print("combined_patch:", combined_patch)
    
    # Apply combined patch and run tests
    logger, timed_out = run_patch_in_container(
        instance, repo, LOG_DIR_RUN_VALIDATION, timeout,
        patch=combined_patch,
    )
    
    if timed_out:
        close_logger(logger)
        error_print(f"[ERROR] Validation timed out for instance {instance_id}")
        return {"reward": -1.0, "tests_passed": False, "error": "timeout"}
    
    # Check build logs for compilation errors
    report_path = valid_folder / instance_id / LOG_REPORT
    if report_path.exists():
        report = report_path.read_text()
        # Check for compilation failure indicators
        if "error:" in report.lower() or ("gmake[" in report and "Error" in report):
            close_logger(logger)
            error_print(f"[ERROR] Compilation failed for {instance_id}")
            debug_print(f"[DEBUG] Build report excerpt:\n{report[:500]}")
            return {"reward": -1.0, "tests_passed": False, "error": "compilation_failed"}
    
    # Parse test output
    test_output_path = valid_folder / instance_id / LOG_TEST_OUTPUT
    if not test_output_path.exists():
        error_print(f"[ERROR] Test output file not found: {test_output_path}")
        close_logger(logger)
        return {"reward": -1.0, "tests_passed": False, "error": "no_test_output"}
    
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
    """Create a proper git patch by modifying files in-repo and using git diff HEAD"""
    import subprocess
    import tempfile
    
    DEVNULL = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    full_path = Path(repo_path) / file_path
    
    if not full_path.exists():
        error_print(f"[ERROR] File not found: {full_path}")
        return None
    
    try:
        # Read original file content
        with open(full_path, 'r', newline='', encoding='utf-8') as f:
            original_content = f.read()
        
        debug_print(f"[DEBUG] Original file content length: {len(original_content)}")
        
        # Extract the exact buggy function from the file
        # This ensures we get the right whitespace and line endings
        extracted_buggy = extract_exact_function(original_content, buggy_function)
        if not extracted_buggy:
            error_print(f"[ERROR] Could not locate buggy function in file")
            return None
        
        start_line, end_line, actual_buggy = extracted_buggy
        debug_print(f"[DEBUG] Found function at lines {start_line}-{end_line}")
        debug_print(f"[DEBUG] Actual buggy function ({len(actual_buggy)} chars)")
        
        # Verify the extracted function matches what we expect
        if normalize_whitespace(actual_buggy) != normalize_whitespace(buggy_function):
            warning_print(f"[WARNING] Extracted function doesn't match expected buggy function")
            debug_print(f"[DEBUG] Expected: {buggy_function[:100]}...")
            debug_print(f"[DEBUG] Actual: {actual_buggy[:100]}...")
        
        # Create modified content
        lines = original_content.splitlines(keepends=True)
        fixed_lines = fixed_code.splitlines(keepends=True)
        
        # Ensure fixed code ends with newline if original did
        if lines and lines[-1].endswith('\n') and (not fixed_lines or not fixed_lines[-1].endswith('\n')):
            if fixed_lines:
                fixed_lines[-1] = fixed_lines[-1] + '\n'
        
        modified_lines = lines[:start_line-1] + fixed_lines + lines[end_line:]
        modified_content = ''.join(modified_lines)
        
        # Write modified content to file
        with open(full_path, 'w', newline='', encoding='utf-8') as f:
            f.write(modified_content)
        
        debug_print(f"[DEBUG] Modified content length: {len(modified_content)}")
        
        # Generate patch using git diff HEAD (this gives proper paths)
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", "HEAD", "--", file_path],
            capture_output=True, text=True, encoding='utf-8'
        )
        
        if result.returncode != 0 and not result.stdout:
            error_print(f"[ERROR] git diff failed: {result.stderr}")
            return None
        
        patch = result.stdout
        
        # Validate patch
        if not patch or len(patch.strip()) < 50:
            warning_print(f"[WARNING] Generated patch seems too small")
            return None
        
        # Clean up: revert changes
        subprocess.run(["git", "-C", repo_path, "checkout", "--", file_path], 
                      check=True, **DEVNULL)
        
        # Additional validation: try to apply the patch to verify it works
        validation_result = validate_patch(patch, repo_path, file_path)
        if not validation_result['valid']:
            error_print(f"[ERROR] Generated patch is invalid: {validation_result['error']}")
            debug_print(f"[DEBUG] Patch content:\n{patch}")
            return None
        
        debug_print(f"[DEBUG] Patch generated successfully ({len(patch)} chars)")
        debug_print(f"[DEBUG] Patch preview:\n{patch[:500]}")
        
        return patch
        
    except Exception as e:
        error_print(f"[ERROR] Failed to create patch: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def extract_exact_function(content, function_template):
    """Extract the exact function from content based on a template"""
    import re
    
    # Get function signature (first few lines)
    template_lines = [line for line in function_template.splitlines() if line.strip()]
    if not template_lines:
        return None
    
    # Build pattern from first 2-3 significant lines
    pattern_lines = []
    for i, line in enumerate(template_lines[:3]):
        # Escape special chars but preserve whitespace structure
        escaped = re.escape(line.rstrip())
        pattern_lines.append(escaped)
    
    # Create regex with flexible whitespace matching
    pattern = r'\n'.join(pattern_lines)
    regex = re.compile(pattern, re.MULTILINE)
    
    match = regex.search(content)
    if not match:
        # Try without leading/trailing whitespace on each line
        pattern = r'\s*\n\s*'.join([line.strip() for line in template_lines[:3]])
        regex = re.compile(pattern, re.MULTILINE)
        match = regex.search(content)
    
    if not match:
        return None
    
    # Find start position
    start_pos = match.start()
    start_line = content[:start_pos].count('\n') + 1
    
    # Find function end by matching braces
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)
    
    # Find which line we're on
    line_idx = 0
    char_count = 0
    for i, line in enumerate(lines):
        char_count += len(line)
        if char_count > start_pos:
            line_idx = i
            break
    
    # Now find matching braces
    brace_count = 0
    found_start = False
    end_line = line_idx
    
    for i in range(line_idx, total_lines):
        line = lines[i]
        
        if not found_start:
            if '{' in line:
                found_start = True
                brace_count = line.count('{') - line.count('}')
                if brace_count == 0:
                    # Might have multiple braces on one line
                    continue
                else:
                    end_line = i
        else:
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                end_line = i
                break
    
    # Extract the full function
    function_lines = lines[line_idx:end_line+1]
    full_function = ''.join(function_lines)
    
    return (line_idx + 1, end_line + 1, full_function)

def normalize_whitespace(text):
    """Normalize whitespace for comparison"""
    import re
    return re.sub(r'\s+', ' ', text.strip())

def validate_patch(patch_content, repo_path, file_path):
    """Validate that a patch can be applied"""
    import subprocess
    import tempfile
    
    result = {
        'valid': False,
        'error': None
    }
    
    # Basic format checks
    if not patch_content:
        result['error'] = "Empty patch"
        return result
    
    if '--- a/' not in patch_content or '+++ b/' not in patch_content:
        result['error'] = "Missing git diff headers"
        return result
    
    if '@@' not in patch_content:
        result['error'] = "Missing hunk headers"
        return result
    
    # Try to apply the patch in a temp repo to verify
    try:
        # Create a temporary copy of the repo
        with tempfile.TemporaryDirectory() as temp_dir:
            import shutil
            temp_repo = Path(temp_dir) / "repo"
            shutil.copytree(repo_path, temp_repo, ignore=shutil.ignore_patterns('.git'))
            
            # Initialize git and apply original state
            subprocess.run(["git", "init"], cwd=temp_repo, capture_output=True, check=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=temp_repo, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=temp_repo, capture_output=True)
            
            # Copy the actual file from the real repo (in bug-state)
            src_file = Path(repo_path) / file_path
            dest_file = temp_repo / file_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)
            
            subprocess.run(["git", "add", "."], cwd=temp_repo, capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "bug state"], cwd=temp_repo, capture_output=True, check=True)
            
            # Try to apply the patch
            apply_result = subprocess.run(
                ["git", "apply", "--check", "-"],
                cwd=temp_repo,
                input=patch_content,
                text=True,
                capture_output=True
            )
            
            if apply_result.returncode == 0:
                result['valid'] = True
            else:
                result['error'] = f"Patch apply check failed: {apply_result.stderr}"
    except Exception as e:
        result['error'] = f"Validation error: {str(e)}"
    
    return result

def merge_patches(bug_patch, fix_patch):
    """Merge two patches that modify the same file"""
    if not bug_patch:
        return fix_patch
    if not fix_patch:
        return bug_patch
    
    # If they modify different files, simple concatenation works
    bug_files = get_modified_files(bug_patch)
    fix_files = get_modified_files(fix_patch)
    
    if bug_files != fix_files:
        return bug_patch + '\n' + fix_patch
    
    # For same file, we need to apply them sequentially and regenerate
    # This is a simplified approach - a full solution would parse both patches
    # and merge the hunks intelligently
    warning_print("Warning: Merging patches for same file - using sequential application")
    return bug_patch + '\n' + fix_patch

def get_modified_files(patch):
    """Get set of files modified in a patch"""
    import re
    files = set()
    for line in patch.split('\n'):
        if line.startswith('--- a/'):
            files.add(line[6:])  # Remove '--- a/' prefix
        elif line.startswith('+++ b/'):
            files.add(line[6:])  # Remove '+++ b/' prefix
    return files

def extract_function_with_context(file_content, function_content, context_lines=5):
    """Extract function with surrounding context lines to preserve git patch context"""
    import re
    
    # Normalize whitespace for matching but keep original content
    normalized_function = re.sub(r'\s+', ' ', function_content.strip())
    normalized_file = re.sub(r'\s+', ' ', file_content)
    
    # Find the function in the file
    start_pos = normalized_file.find(normalized_function)
    if start_pos == -1:
        # Try to find with more flexible matching
        function_lines = [line.strip() for line in function_content.splitlines() if line.strip()]
        if not function_lines:
            return None
        
        # Try to match the first few significant lines
        first_lines_pattern = '.*'.join(re.escape(line) for line in function_lines[:2])
        match = re.search(first_lines_pattern, normalized_file, re.DOTALL)
        if not match:
            return None
        
        start_pos = match.start()
    
    # Find the exact position in the original content
    approx_line = file_content.count('\n', 0, start_pos) + 1
    
    # Get context around the function
    lines = file_content.splitlines(keepends=True)
    func_start_line = max(0, approx_line - context_lines - 1)
    func_end_line = min(len(lines), approx_line + context_lines)
    
    context_block = ''.join(lines[func_start_line:func_end_line])
    
    return {
        'context': context_block,
        'start_line': func_start_line + 1,  # 1-based line numbers
        'end_line': func_end_line,
        'approx_function_start': approx_line
    }

def validate_patch_format(patch_content, file_path):
    """Validate that patch has proper format for git apply"""
    import re
    
    if not patch_content or len(patch_content.strip()) == 0:
        error_print(f"[ERROR] Patch content is empty")
        return False
    
    # Check for required git diff headers
    has_src_header = bool(re.search(r'^--- a/', patch_content, re.MULTILINE))
    has_dst_header = bool(re.search(r'^\+\+\+ b/', patch_content, re.MULTILINE))
    
    if not has_src_header or not has_dst_header:
        error_print(f"[ERROR] Patch missing required headers (--- a/ or +++ b/)")
        debug_print(f"[DEBUG] Patch content:\n{patch_content[:500]}")
        return False
    
    # Check if patch mentions the target file
    if file_path not in patch_content:
        warning_print(f"[WARNING] Patch doesn't reference target file: {file_path}")
        debug_print(f"[DEBUG] Patch content:\n{patch_content[:500]}")
    
    # Check for hunk headers (@@ ... @@)
    has_hunks = bool(re.search(r'^@@', patch_content, re.MULTILINE))
    if not has_hunks:
        error_print(f"[ERROR] Patch has no hunks (@@ markers)")
        debug_print(f"[DEBUG] Patch content:\n{patch_content[:500]}")
        return False
    
    # Check for actual changes (+ or - lines)
    has_changes = bool(re.search(r'^[+-]', patch_content, re.MULTILINE))
    if not has_changes:
        error_print(f"[ERROR] Patch has no actual changes (no + or - lines)")
        debug_print(f"[DEBUG] Patch content:\n{patch_content[:500]}")
        return False
    
    # Check for proper line ending consistency
    if '\r\n' in patch_content and '\n' in patch_content and not patch_content.endswith('\n'):
        warning_print(f"[WARNING] Patch has inconsistent line endings")
    
    debug_print(f"[DEBUG] Patch format validation passed")
    return True

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
                
                if not fix_patch or not validate_patch_format(fix_patch, file_path):
                    warning_print(f"[WARNING] Patch format validation failed for attempt {attempt + 1}")
                    attempts.append({"attempt": attempt + 1, "status": "invalid_patch_format", "reward": -1.0})
                    continue
                
                if not fix_patch:
                    warning_print(f"[WARNING] Patch creation FAILED")
                    attempts.append({"attempt": attempt + 1, "status": "patch_failed", "reward": -1.0})
                    continue
                
                debug_print(f"[DEBUG] Patch created ({len(fix_patch)} chars)")
                debug_print(f"[DEBUG] Patch preview:\n{fix_patch[:300]}\n")
                
                # Validate - include BOTH bug patch and fix patch
                instance = {
                    KEY_INSTANCE_ID: f"{patch['instance_id']}_attempt_{attempt}",
                    KEY_PATCH: fix_patch,  # LLM's fix
                    "bug_patch": patch['patch'],  # Original bug patch
                    "repo": repo,
                    FAIL_TO_PASS: patch['FAIL_TO_PASS'],
                }
                result = validate_fix(instance, repo, rp.timeout)
                attempts.append({"attempt": attempt + 1, "fixed_code": fixed_code, "patch": fix_patch, **result})
                
                if result['tests_passed']:
                    debug_print(f"[DEBUG] SUCCESS! All tests passed!")
                    success = True
                    break
                else:
                    debug_print(f"[DEBUG] Tests FAILED - Error: {result.get('error', 'unknown')}")
                    
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
