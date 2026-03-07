"""
Validator Server — wraps verify.py via Docker container
=========================================================
Exposes POST /validate — same interface as rl_loop_verl.py expects.
Workers call `docker run agnostics-c` with verify.py as entrypoint,
piping JSON in via stdin and reading JSON result from stdout.

Why wrap Docker instead of reimplementing:
  - verify.py already handles compile + run + normalize + compare
  - Docker provides the same isolation you already tested and trust
  - ProcessPoolExecutor pre-forks workers to amortize Docker startup

Throughput:
  - Local PC:     --workers 1 or 2  (match your CPU cores)
  - Large server: --workers 256     (1TB RAM, many cores)

Usage:
  pip install fastapi uvicorn

  # Build the container first (one time)
  docker build -f Docker/Dockerfile -t agnostics-c .

  # Start server (local PC)
  python validator_server.py --port 8001 --workers 2

  # Start server (large server)
  python validator_server.py --port 8001 --workers 256 --max-memory-mb 512

  # Test
  curl -X POST http://localhost:8001/validate \\
    -H "Content-Type: application/json" \\
    -d '{"code": "#include<stdio.h>\\nint main(){printf(\\"1\\n\\");}", 
         "test_cases": [{"input": "", "output": "1"}], "timeout_s": 10}'
"""

import argparse
import asyncio
import json
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app  = FastAPI(title="C Validator Server")
pool: ProcessPoolExecutor = None   # set at startup


# ============================================================================
# Request / Response schemas
# ============================================================================

class TestCase(BaseModel):
    input:  str
    output: str

class ValidateRequest(BaseModel):
    code:       str
    test_cases: List[TestCase]
    timeout_s:  int = 10

class ValidateResponse(BaseModel):
    result:    str    # "success" | "wrong-output" | "compile-error" | "runtime-error" | "timeout"
    passed:    int
    total:     int
    error_msg: str = ""


# ============================================================================
# Worker — runs in subprocess pool, calls Docker verify.py
# ============================================================================

def _worker_validate(
    code:           str,
    test_cases:     List[Dict],
    timeout_s:      int,
    container_name: str,
) -> Dict:
    """
    Runs in a worker process (not the server process).
    Calls `docker run agnostics-c`, pipes JSON to stdin, reads JSON from stdout.

    The verify.py inside the container handles:
      - gcc compilation
      - per-test-case execution
      - output normalization + comparison
    """
    payload = json.dumps({
        "code":       code,
        "timeout_s":  timeout_s,
        "test_cases": test_cases,
    })

    docker_cmd = [
        "docker", "run", "--rm", "-i",
        "--tmpfs", "/ramdisk:size=512m,exec",
        container_name,
    ]

    try:
        proc = subprocess.run(
            docker_cmd,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s + 60,   # compile timeout (30s) + exec timeout + buffer
        )
    except subprocess.TimeoutExpired:
        return {"result": "timeout",        "passed": 0, "total": len(test_cases), "error_msg": "docker timed out"}
    except FileNotFoundError:
        return {"result": "runtime-error",  "passed": 0, "total": len(test_cases), "error_msg": "docker not found — is Docker installed?"}
    except Exception as e:
        return {"result": "runtime-error",  "passed": 0, "total": len(test_cases), "error_msg": str(e)}

    if proc.returncode != 0:
        return {
            "result":    "runtime-error",
            "passed":    0,
            "total":     len(test_cases),
            "error_msg": (proc.stderr or proc.stdout)[:300],
        }

    # Parse verify.py JSON output
    try:
        raw = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return {
            "result":    "runtime-error",
            "passed":    0,
            "total":     len(test_cases),
            "error_msg": f"bad JSON from verify.py: {proc.stdout[:200]}",
        }

    # verify.py result field: "success" | "fail:error" | "fail:timeout" | "fail:wrong-output"
    vpy_result = raw.get("result", "fail:other")

    if vpy_result == "success":
        return {"result": "success",       "passed": len(test_cases), "total": len(test_cases), "error_msg": ""}
    elif "timeout"      in vpy_result:
        return {"result": "timeout",       "passed": 0, "total": len(test_cases), "error_msg": raw.get("stderr", "")[:200]}
    elif "wrong-output" in vpy_result:
        return {"result": "wrong-output",  "passed": 0, "total": len(test_cases), "error_msg": raw.get("stderr", "")[:200]}
    elif "error"        in vpy_result:
        return {"result": "compile-error" if "Compilation" in raw.get("stderr", "") else "runtime-error",
                "passed": 0, "total": len(test_cases), "error_msg": raw.get("stderr", "")[:200]}
    else:
        return {"result": "runtime-error", "passed": 0, "total": len(test_cases), "error_msg": vpy_result}


# ============================================================================
# FastAPI endpoint
# ============================================================================

@app.post("/validate", response_model=ValidateResponse)
async def validate(req: ValidateRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            pool,
            _worker_validate,
            req.code,
            [tc.model_dump() for tc in req.test_cases],
            req.timeout_s,
            app.state.container_name,
        )
        return ValidateResponse(**result)
    except Exception as e:
        log.exception("Worker error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "workers": app.state.num_workers, "container": app.state.container_name}


# ============================================================================
# Startup / shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    global pool
    pool = ProcessPoolExecutor(max_workers=app.state.num_workers)
    log.info(f"Started ProcessPoolExecutor: {app.state.num_workers} workers → container={app.state.container_name}")

@app.on_event("shutdown")
async def shutdown():
    pool.shutdown(wait=False, cancel_futures=True)
    log.info("ProcessPoolExecutor shut down")


# ============================================================================
# Entry point
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="C Validator Server (wraps verify.py via Docker)")
    p.add_argument("--port",           type=int, default=8001)
    p.add_argument("--workers",        type=int, default=2,
                   help="Concurrent Docker executions (default 2 for local, 256 for server)")
    p.add_argument("--container-name", type=str, default="agnostics-c",
                   help="Docker image/container name built from your Dockerfile")
    p.add_argument("--max-memory-mb",  type=int, default=512,
                   help="(informational only — memory limits enforced inside Docker)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app.state.num_workers    = args.workers
    app.state.container_name = args.container_name
    app.state.max_memory_mb  = args.max_memory_mb
    log.info(f"Validator server: port={args.port} workers={args.workers} container={args.container_name}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
