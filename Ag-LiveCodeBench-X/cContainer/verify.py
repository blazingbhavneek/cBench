#!/usr/bin/env python3
"""
Agnostics verifier script for C language with GLib support.

This script implements the Agnostics protocol:
- Reads JSON from stdin: {"code": str, "timeout_s": int, "test_cases": [{"input": str, "output": str}, ...]}
- Writes JSON to stdout: {"result": "success", ...} or {"result": "fail:...", ...}

The script:
1. Receives C code and test cases via stdin
2. Compiles the C code using gcc with GLib support
3. Runs the compiled program with each test case
4. Compares output with expected results
5. Returns results as JSON to stdout
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


class TimeoutException(Exception):
    """Exception raised when execution times out."""

    pass


@contextmanager
def timeout_context(seconds):
    """Context manager for timeout handling."""

    def timeout_handler(signum, frame):
        raise TimeoutException("Execution timed out")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the old handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def compile_c_code(code: str, output_path: Path) -> dict:
    """
    Compile C code using gcc.

    Returns:
        dict with 'success' (bool), 'stderr' (str), and optionally 'exit_code' (int)
    """
    # Create a temporary C source file
    source_file = output_path.parent / f"{output_path.stem}.c"

    try:
        with open(source_file, "w") as f:
            f.write(code)

        # Compile with gcc
        result = subprocess.run(
            ["gcc", "-std=c11", "-O2", "-o", str(output_path), str(source_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return {
                "success": False,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }

        return {"success": True, "stderr": result.stderr}

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stderr": "Compilation timed out after 30 seconds",
            "exit_code": -1,
        }
    finally:
        # Clean up source file
        if source_file.exists():
            source_file.unlink()


def run_test_case(executable: Path, test_input: str, timeout_s: int) -> dict:
    """
    Run the compiled executable with a test case.

    Returns:
        dict with 'success' (bool), 'stdout' (str), 'stderr' (str),
        'exit_code' (int), and optionally 'timeout' (bool)
    """
    try:
        result = subprocess.run(
            [str(executable)],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "timeout": False,
        }

    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "stdout": e.stdout.decode() if e.stdout else "",
            "stderr": e.stderr.decode() if e.stderr else "",
            "exit_code": -1,
            "timeout": True,
        }


def normalize_output(output: str) -> str:
    """
    Normalize output by stripping trailing whitespace from each line
    and removing the final newline if present.
    """
    lines = output.rstrip("\n").split("\n")
    return "\n".join(line.rstrip() for line in lines)


def verify_code(code: str, test_cases: list, timeout_s: int) -> dict:
    """
    Main verification function.

    Returns:
        dict conforming to Agnostics protocol output format
    """
    # Create a temporary directory for compilation and execution
    with tempfile.TemporaryDirectory(prefix="agnostics_c_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        executable_path = tmpdir_path / "program"

        # Step 1: Compile the code
        compile_result = compile_c_code(code, executable_path)

        if not compile_result["success"]:
            return {
                "result": "fail:error",
                "exit_code": compile_result.get("exit_code", 1),
                "stdout": "",
                "stderr": f"Compilation failed:\n{compile_result['stderr']}",
            }

        # Step 2: Run test cases
        for i, test_case in enumerate(test_cases):
            test_input = test_case.get("input", "")
            expected_output = test_case.get("output", "")

            # Run the test
            run_result = run_test_case(executable_path, test_input, timeout_s)

            # Handle timeout
            if run_result["timeout"]:
                return {
                    "result": "fail:timeout",
                    "stdout": run_result["stdout"],
                    "stderr": f"Test case {i} timed out after {timeout_s} seconds\n{run_result['stderr']}",
                }

            # Handle runtime error
            if not run_result["success"]:
                return {
                    "result": "fail:error",
                    "exit_code": run_result["exit_code"],
                    "stdout": run_result["stdout"],
                    "stderr": f"Test case {i} failed with exit code {run_result['exit_code']}:\n{run_result['stderr']}",
                }

            # Compare output
            actual_output = normalize_output(run_result["stdout"])
            expected_output = normalize_output(expected_output)

            if actual_output != expected_output:
                return {
                    "result": "fail:wrong-output",
                    "expected": expected_output,
                    "got": actual_output,
                    "stderr": f"Test case {i} produced wrong output\n{run_result['stderr']}",
                }

        # All tests passed!
        return {"result": "success", "stderr": compile_result.get("stderr", "")}


def main():
    """
    Main entry point - reads from stdin and writes to stdout.
    """
    try:
        # Read input from stdin
        input_line = sys.stdin.readline()
        if not input_line:
            print(
                json.dumps(
                    {
                        "result": "fail:other",
                        "stdout": "",
                        "stderr": "No input received",
                    }
                )
            )
            return

        # Parse JSON input
        try:
            input_data = json.loads(input_line)
        except json.JSONDecodeError as e:
            print(
                json.dumps(
                    {
                        "result": "fail:other",
                        "stdout": "",
                        "stderr": f"Invalid JSON input: {str(e)}",
                    }
                )
            )
            return

        # Extract required fields
        code = input_data.get("code")
        timeout_s = input_data.get("timeout_s", 15)
        test_cases = input_data.get("test_cases", [])

        if not code:
            print(
                json.dumps(
                    {"result": "fail:other", "stdout": "", "stderr": "No code provided"}
                )
            )
            return

        if not test_cases:
            print(
                json.dumps(
                    {
                        "result": "fail:other",
                        "stdout": "",
                        "stderr": "No test cases provided",
                    }
                )
            )
            return

        # Verify the code
        result = verify_code(code, test_cases, timeout_s)

        # Output result as JSON
        print(json.dumps(result))

    except Exception as e:
        # Catch-all for unexpected errors
        print(
            json.dumps(
                {
                    "result": "fail:other",
                    "stdout": "",
                    "stderr": f"Unexpected error: {str(e)}",
                }
            )
        )


if __name__ == "__main__":
    main()
