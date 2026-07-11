#!/usr/bin/env python3
"""Run one pipeline handler inside the container with verbose tracing.
Usage (inside container): python /app/eval/debug_one.py code_debug "prompt..."
"""

import sys

sys.path.insert(0, "/app")

from agent import llm, pipelines  # noqa: E402
from agent.util import extract_code, log  # noqa: E402


def main():
    cat, prompt = sys.argv[1], sys.argv[2]
    if not llm.start_all():
        sys.exit("servers failed")
    llm.probe_tps()

    if cat == "code_debug":
        out = llm.CODER.chat(
            [{"role": "system", "content": pipelines.DEBUG_SYS},
             {"role": "user", "content": prompt}],
            max_tokens=380, temperature=0.3)
        log(f"RAW DEBUG REPLY:\n{out}\n---")
        code = extract_code(out)
        log(f"EXTRACTED CODE:\n{code}\n---")
        sig = pipelines._signature_of(code)
        tests = pipelines._gen_tests(prompt, sig)
        log(f"GENERATED TESTS:\n{tests}\n---")
        from agent.sandbox import run_with_tests
        ok, err = run_with_tests(code, tests)
        log(f"TESTS OK={ok} ERR={err}")

    res = pipelines.run_task(cat, prompt, "full")
    log(f"FINAL: conf={res.confidence}\n{res.answer}")
    llm.stop_all()


if __name__ == "__main__":
    main()
