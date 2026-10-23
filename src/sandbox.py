"""Restricted Docker execution; never execute dataset programs in this process."""

import concurrent.futures
import json
import math
import subprocess
import uuid

IMAGE = "python@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d"

CHILD = r"""
import sys, json, resource, math, numbers, decimal

resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (16384, 16384))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_NPROC, (8, 8))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
code = json.loads(sys.stdin.read())
try:
    ns = {}
    exec(compile(code, "<program>", "exec"), ns)
    fn = ns.get("solve")
    if not callable(fn):
        result = {"status": "missing_callable"}
    else:
        value = fn()
        if value is None:
            result = {"status": "no_result"}
        elif isinstance(value, bool) or not isinstance(
            value, (numbers.Real, decimal.Decimal)
        ):
            result = {"status": "nonnumeric", "type": type(value).__name__}
        else:
            try:
                number = float(value)
                result = (
                    {"status": "ok", "value": number}
                    if math.isfinite(number)
                    else {"status": "nonfinite"}
                )
            except (OverflowError, ValueError):
                result = {"status": "nonfinite"}
except BaseException as e:
    result = {
        "status": "runtime_error",
        "error": type(e).__name__,
        "message": str(e)[:160],
    }
print("\n__HW2_RESULT__" + json.dumps(result, allow_nan=False), flush=True)
"""

PARENT = r"""
import subprocess, json, sys, tempfile, os, signal

payload = json.load(sys.stdin)
for code in payload["codes"]:
    with tempfile.TemporaryFile() as out:
        p = subprocess.Popen(
            [sys.executable, "-I", "-c", payload["child"]],
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=out,
            start_new_session=True,
        )
        try:
            p.communicate(json.dumps(code).encode(), timeout=2)
            out.seek(0)
            text = out.read(16384).decode("utf-8", errors="replace")
            marker = "__HW2_RESULT__"
            try:
                result = json.loads(text.rsplit(marker, 1)[1].splitlines()[0])
            except Exception:
                result = {
                    "status": "process_error",
                    "returncode": p.returncode,
                    "output_tail": text[-160:],
                }
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            result = {"status": "timeout"}
        # Kill any surviving descendant in this task's process group.
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print(json.dumps(result), flush=True)
"""


def one_batch(codes):
    name = "hw2-audit-" + uuid.uuid4().hex[:12]
    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--user",
        "65534:65534",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "16",
        "--memory",
        "256m",
        "--memory-swap",
        "256m",
        "--cpus",
        "1",
        "--ulimit",
        "nofile=64:64",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        IMAGE,
        "python",
        "-I",
        "-c",
        PARENT,
    ]
    try:
        r = subprocess.run(
            cmd,
            input=json.dumps({"codes": codes, "child": CHILD}),
            text=True,
            capture_output=True,
            timeout=len(codes) * 3 + 20,
        )
        if r.returncode:
            raise RuntimeError(r.stderr[-1000:])
        result = [json.loads(line) for line in r.stdout.splitlines()]
        if len(result) != len(codes):
            raise RuntimeError("sandbox result count mismatch")
        return result
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )


def execute(codes, workers=4):
    batches = [codes[i : i + 32] for i in range(0, len(codes), 32)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for batch in pool.map(one_batch, batches) for r in batch]


def compare(pred, ref):
    valid = [i for (i, r) in enumerate(ref) if r["status"] == "ok"]
    matched = sum(
        (
            pred[i]["status"] == "ok"
            and math.isclose(
                pred[i]["value"], ref[i]["value"], rel_tol=1e-06, abs_tol=1e-06
            )
            for i in valid
        )
    )
    from collections import Counter

    return {
        "execution_total": len(pred),
        "prediction_status_counts": dict(Counter((r["status"] for r in pred))),
        "reference_status_counts": dict(Counter((r["status"] for r in ref))),
        "proxy_matches": matched,
        "proxy_denominator": len(valid),
        "proxy_accuracy": matched / len(valid) if valid else None,
        "proxy_all_examples_rate": matched / len(pred),
        "comparison": "math.isclose(rel_tol=1e-6, abs_tol=1e-6); noisy reference agreement, NOT official pass@1",
    }
