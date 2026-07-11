"""Run model-generated Python safely-enough: isolated interpreter subprocess,
hard timeout, no stdin. This is our own container; the risk model is
accidental hangs/crashes, not adversarial code."""

import subprocess
import sys

RUN_TIMEOUT_S = 8


def run_python(code: str, timeout_s: float = RUN_TIMEOUT_S):
    """Execute code, return (ok: bool, stdout: str, stderr: str)."""
    if "input(" in code:
        code = "def input(*a, **k):\n    return ''\n" + code
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL,
        )
        return proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        return False, "", f"SANDBOX_ERROR: {e}"


def run_with_tests(func_code: str, test_code: str):
    """Run function + assert-based tests. Returns (ok, err_detail)."""
    sentinel = "ALL_TESTS_PASSED_7731"
    program = f"{func_code}\n\n{test_code}\nprint({sentinel!r})\n"
    ok, out, err = run_python(program)
    if ok and sentinel in out:
        return True, ""
    return False, (err or out or "tests failed with no output")[-500:]
